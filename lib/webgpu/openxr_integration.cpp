#include "openxr_integration.hpp"
#include "gpu.hpp"

#define XR_USE_GRAPHICS_API_VULKAN
#include <vulkan/vulkan.h>
#include <openxr/openxr.h>
#include <openxr/openxr_platform.h>

#include <dawn/native/VulkanBackend.h>

#include <iostream>
#include <vector>
#include <stdexcept>
#include <cstring>
#include <unistd.h>

#define XR_CHECK(cmd) \
    do { \
        XrResult res = (cmd); \
        if (XR_FAILED(res)) { \
            std::cerr << "OpenXR Error: " << res << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            throw std::runtime_error("OpenXR error"); \
        } \
    } while (0)

#define VK_CHECK(cmd) \
    do { \
        VkResult res = (cmd); \
        if (res != VK_SUCCESS) { \
            std::cerr << "Vulkan Error: " << res << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            throw std::runtime_error("Vulkan error"); \
        } \
    } while (0)

namespace aurora::openxr {

namespace {
    XrInstance g_xrInstance = XR_NULL_HANDLE;
    XrSession g_xrSession = XR_NULL_HANDLE;
    XrSystemId g_xrSystemId = XR_NULL_SYSTEM_ID;
    XrSpace g_xrSpace = XR_NULL_HANDLE;

    // Vulkan objects (XR-owned device)
    VkInstance g_vkInstance = VK_NULL_HANDLE;
    VkPhysicalDevice g_vkPhysicalDevice = VK_NULL_HANDLE;
    VkDevice g_vkDevice = VK_NULL_HANDLE;
    VkQueue g_vkQueue = VK_NULL_HANDLE;
    uint32_t g_vkQueueFamilyIndex = 0;
    VkCommandPool g_vkCommandPool = VK_NULL_HANDLE;
    VkCommandBuffer g_vkCommandBuffer = VK_NULL_HANDLE;

    // Swapchain
    XrSwapchain g_xrSwapchain = XR_NULL_HANDLE;
    std::vector<XrSwapchainImageVulkanKHR> g_swapchainImages;
    uint32_t g_swapchainWidth = 0;
    uint32_t g_swapchainHeight = 0;

    // CPU readback bridge: Dawn -> wgpu readback buffer -> CPU -> VkBuffer staging -> XR swapchain
    wgpu::Texture g_dawnRenderTexture = nullptr;
    wgpu::Buffer  g_dawnReadbackBuffer = nullptr;

    VkBuffer       g_stagingBuffer = VK_NULL_HANDLE;
    VkDeviceMemory g_stagingMemory = VK_NULL_HANDLE;
    void*          g_stagingMapped = nullptr;

    bool g_sessionRunning = false;

    // Most-recent HMD head pose (orientation only; from views[0].pose)
    XrQuaternionf g_headOrientation{0.0f, 0.0f, 0.0f, 1.0f};
    bool          g_headPoseValid = false;

    bool create_vulkan_instance() {
        PFN_xrGetVulkanGraphicsRequirements2KHR pfnGetReqs = nullptr;
        xrGetInstanceProcAddr(g_xrInstance, "xrGetVulkanGraphicsRequirements2KHR",
                              (PFN_xrVoidFunction*)&pfnGetReqs);
        XrGraphicsRequirementsVulkanKHR reqs{XR_TYPE_GRAPHICS_REQUIREMENTS_VULKAN_KHR};
        XR_CHECK(pfnGetReqs(g_xrInstance, g_xrSystemId, &reqs));

        VkApplicationInfo appInfo{VK_STRUCTURE_TYPE_APPLICATION_INFO};
        appInfo.pApplicationName = "Aurora OpenXR";
        appInfo.apiVersion = VK_API_VERSION_1_1;

        VkInstanceCreateInfo ci{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
        ci.pApplicationInfo = &appInfo;

        PFN_xrCreateVulkanInstanceKHR pfnCreateInst = nullptr;
        xrGetInstanceProcAddr(g_xrInstance, "xrCreateVulkanInstanceKHR",
                              (PFN_xrVoidFunction*)&pfnCreateInst);

        XrVulkanInstanceCreateInfoKHR xrCI{XR_TYPE_VULKAN_INSTANCE_CREATE_INFO_KHR};
        xrCI.systemId = g_xrSystemId;
        xrCI.pfnGetInstanceProcAddr = &vkGetInstanceProcAddr;
        xrCI.vulkanCreateInfo = &ci;

        VkResult vkResult;
        XR_CHECK(pfnCreateInst(g_xrInstance, &xrCI, &g_vkInstance, &vkResult));
        VK_CHECK(vkResult);
        return true;
    }

    bool create_vulkan_device() {
        PFN_xrGetVulkanGraphicsDevice2KHR pfnGetDev = nullptr;
        xrGetInstanceProcAddr(g_xrInstance, "xrGetVulkanGraphicsDevice2KHR",
                              (PFN_xrVoidFunction*)&pfnGetDev);

        XrVulkanGraphicsDeviceGetInfoKHR devInfo{XR_TYPE_VULKAN_GRAPHICS_DEVICE_GET_INFO_KHR};
        devInfo.systemId = g_xrSystemId;
        devInfo.vulkanInstance = g_vkInstance;
        XR_CHECK(pfnGetDev(g_xrInstance, &devInfo, &g_vkPhysicalDevice));

        uint32_t qfc = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(g_vkPhysicalDevice, &qfc, nullptr);
        std::vector<VkQueueFamilyProperties> qfp(qfc);
        vkGetPhysicalDeviceQueueFamilyProperties(g_vkPhysicalDevice, &qfc, qfp.data());
        for (uint32_t i = 0; i < qfc; i++) {
            if (qfp[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) {
                g_vkQueueFamilyIndex = i;
                break;
            }
        }

        float prio = 1.0f;
        VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
        qci.queueFamilyIndex = g_vkQueueFamilyIndex;
        qci.queueCount = 1;
        qci.pQueuePriorities = &prio;

        VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
        dci.queueCreateInfoCount = 1;
        dci.pQueueCreateInfos = &qci;

        PFN_xrCreateVulkanDeviceKHR pfnCreateDev = nullptr;
        xrGetInstanceProcAddr(g_xrInstance, "xrCreateVulkanDeviceKHR",
                              (PFN_xrVoidFunction*)&pfnCreateDev);

        XrVulkanDeviceCreateInfoKHR xrDCI{XR_TYPE_VULKAN_DEVICE_CREATE_INFO_KHR};
        xrDCI.systemId = g_xrSystemId;
        xrDCI.pfnGetInstanceProcAddr = &vkGetInstanceProcAddr;
        xrDCI.vulkanPhysicalDevice = g_vkPhysicalDevice;
        xrDCI.vulkanCreateInfo = &dci;

        VkResult vkResult;
        XR_CHECK(pfnCreateDev(g_xrInstance, &xrDCI, &g_vkDevice, &vkResult));
        VK_CHECK(vkResult);

        vkGetDeviceQueue(g_vkDevice, g_vkQueueFamilyIndex, 0, &g_vkQueue);

        VkCommandPoolCreateInfo poolCI{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        poolCI.queueFamilyIndex = g_vkQueueFamilyIndex;
        poolCI.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        VK_CHECK(vkCreateCommandPool(g_vkDevice, &poolCI, nullptr, &g_vkCommandPool));

        VkCommandBufferAllocateInfo cbAI{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        cbAI.commandPool = g_vkCommandPool;
        cbAI.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cbAI.commandBufferCount = 1;
        VK_CHECK(vkAllocateCommandBuffers(g_vkDevice, &cbAI, &g_vkCommandBuffer));
        return true;
    }

    uint32_t find_memory_type(uint32_t filter, VkMemoryPropertyFlags props) {
        VkPhysicalDeviceMemoryProperties mp;
        vkGetPhysicalDeviceMemoryProperties(g_vkPhysicalDevice, &mp);
        for (uint32_t i = 0; i < mp.memoryTypeCount; i++) {
            if ((filter & (1 << i)) && (mp.memoryTypes[i].propertyFlags & props) == props)
                return i;
        }
        throw std::runtime_error("No suitable memory type");
    }

    void create_staging_buffer(uint32_t width, uint32_t height) {
        VkDeviceSize size = (VkDeviceSize)width * height * 4;

        VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        bci.size = size;
        bci.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        VK_CHECK(vkCreateBuffer(g_vkDevice, &bci, nullptr, &g_stagingBuffer));

        VkMemoryRequirements mr;
        vkGetBufferMemoryRequirements(g_vkDevice, g_stagingBuffer, &mr);

        VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        mai.allocationSize = mr.size;
        mai.memoryTypeIndex = find_memory_type(mr.memoryTypeBits,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        VK_CHECK(vkAllocateMemory(g_vkDevice, &mai, nullptr, &g_stagingMemory));
        VK_CHECK(vkBindBufferMemory(g_vkDevice, g_stagingBuffer, g_stagingMemory, 0));
        VK_CHECK(vkMapMemory(g_vkDevice, g_stagingMemory, 0, size, 0, &g_stagingMapped));
    }

    void create_dawn_textures() {
        if (!webgpu::g_device) return;

        wgpu::TextureDescriptor td{};
        td.size = {g_swapchainWidth, g_swapchainHeight, 1};
        td.format = wgpu::TextureFormat::RGBA8Unorm;
        td.usage = wgpu::TextureUsage::RenderAttachment | wgpu::TextureUsage::CopySrc;
        g_dawnRenderTexture = webgpu::g_device.CreateTexture(&td);

        wgpu::BufferDescriptor bd{};
        bd.size = (uint64_t)g_swapchainWidth * g_swapchainHeight * 4;
        bd.usage = wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::MapRead;
        g_dawnReadbackBuffer = webgpu::g_device.CreateBuffer(&bd);
    }
}

bool initialize() {
    try {
        const char* exts[] = { XR_KHR_VULKAN_ENABLE2_EXTENSION_NAME };

        XrInstanceCreateInfo ci{XR_TYPE_INSTANCE_CREATE_INFO};
        ci.applicationInfo.applicationName[0] = 'A';
        ci.applicationInfo.applicationName[1] = '\0';
        ci.applicationInfo.apiVersion = XR_CURRENT_API_VERSION;
        ci.enabledExtensionCount = 1;
        ci.enabledExtensionNames = exts;
        XR_CHECK(xrCreateInstance(&ci, &g_xrInstance));

        XrSystemGetInfo sysInfo{XR_TYPE_SYSTEM_GET_INFO};
        sysInfo.formFactor = XR_FORM_FACTOR_HEAD_MOUNTED_DISPLAY;
        XR_CHECK(xrGetSystem(g_xrInstance, &sysInfo, &g_xrSystemId));

        create_vulkan_instance();
        create_vulkan_device();

        XrGraphicsBindingVulkanKHR binding{XR_TYPE_GRAPHICS_BINDING_VULKAN_KHR};
        binding.instance = g_vkInstance;
        binding.physicalDevice = g_vkPhysicalDevice;
        binding.device = g_vkDevice;
        binding.queueFamilyIndex = g_vkQueueFamilyIndex;
        binding.queueIndex = 0;

        XrSessionCreateInfo sessCI{XR_TYPE_SESSION_CREATE_INFO};
        sessCI.next = &binding;
        sessCI.systemId = g_xrSystemId;
        XR_CHECK(xrCreateSession(g_xrInstance, &sessCI, &g_xrSession));

        XrReferenceSpaceCreateInfo spCI{XR_TYPE_REFERENCE_SPACE_CREATE_INFO};
        spCI.referenceSpaceType = XR_REFERENCE_SPACE_TYPE_LOCAL;
        spCI.poseInReferenceSpace.orientation.w = 1.0f;
        XR_CHECK(xrCreateReferenceSpace(g_xrSession, &spCI, &g_xrSpace));

        uint32_t vcc = 0;
        xrEnumerateViewConfigurationViews(g_xrInstance, g_xrSystemId,
            XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO, 0, &vcc, nullptr);
        std::vector<XrViewConfigurationView> views(vcc, {XR_TYPE_VIEW_CONFIGURATION_VIEW});
        xrEnumerateViewConfigurationViews(g_xrInstance, g_xrSystemId,
            XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO, vcc, &vcc, views.data());

        g_swapchainWidth  = views[0].recommendedImageRectWidth;
        g_swapchainHeight = views[0].recommendedImageRectHeight;

        uint32_t fmtCount = 0;
        xrEnumerateSwapchainFormats(g_xrSession, 0, &fmtCount, nullptr);
        std::vector<int64_t> fmts(fmtCount);
        xrEnumerateSwapchainFormats(g_xrSession, fmtCount, &fmtCount, fmts.data());

        int64_t selectedFmt = fmts[0];
        for (int64_t f : fmts) {
            if (f == VK_FORMAT_R8G8B8A8_SRGB || f == VK_FORMAT_B8G8R8A8_SRGB) {
                selectedFmt = f;
                break;
            }
        }

        XrSwapchainCreateInfo scCI{XR_TYPE_SWAPCHAIN_CREATE_INFO};
        scCI.usageFlags  = XR_SWAPCHAIN_USAGE_COLOR_ATTACHMENT_BIT | XR_SWAPCHAIN_USAGE_TRANSFER_DST_BIT;
        scCI.format      = selectedFmt;
        scCI.sampleCount = views[0].recommendedSwapchainSampleCount;
        scCI.width       = g_swapchainWidth;
        scCI.height      = g_swapchainHeight;
        scCI.faceCount   = 1;
        scCI.arraySize   = 1;
        scCI.mipCount    = 1;
        XR_CHECK(xrCreateSwapchain(g_xrSession, &scCI, &g_xrSwapchain));

        uint32_t imgCount = 0;
        xrEnumerateSwapchainImages(g_xrSwapchain, 0, &imgCount, nullptr);
        g_swapchainImages.resize(imgCount, {XR_TYPE_SWAPCHAIN_IMAGE_VULKAN_KHR});
        xrEnumerateSwapchainImages(g_xrSwapchain, imgCount, &imgCount,
                                   (XrSwapchainImageBaseHeader*)g_swapchainImages.data());

        create_staging_buffer(g_swapchainWidth, g_swapchainHeight);
        return true;
    } catch (const std::exception& e) {
        std::cerr << "OpenXR initialization failed: " << e.what() << std::endl;
        shutdown();
        return false;
    }
}

void shutdown() {
    g_dawnRenderTexture  = nullptr;
    g_dawnReadbackBuffer = nullptr;
    if (g_stagingMapped)  { vkUnmapMemory(g_vkDevice, g_stagingMemory); g_stagingMapped = nullptr; }
    if (g_stagingBuffer)  { vkDestroyBuffer(g_vkDevice, g_stagingBuffer, nullptr); g_stagingBuffer = VK_NULL_HANDLE; }
    if (g_stagingMemory)  { vkFreeMemory(g_vkDevice, g_stagingMemory, nullptr); g_stagingMemory = VK_NULL_HANDLE; }
    if (g_xrSwapchain)    { xrDestroySwapchain(g_xrSwapchain); g_xrSwapchain = XR_NULL_HANDLE; }
    if (g_xrSpace)        { xrDestroySpace(g_xrSpace); g_xrSpace = XR_NULL_HANDLE; }
    if (g_vkCommandPool)  { vkDestroyCommandPool(g_vkDevice, g_vkCommandPool, nullptr); g_vkCommandPool = VK_NULL_HANDLE; }
    if (g_vkDevice)       { vkDestroyDevice(g_vkDevice, nullptr); g_vkDevice = VK_NULL_HANDLE; }
    if (g_vkInstance)     { vkDestroyInstance(g_vkInstance, nullptr); g_vkInstance = VK_NULL_HANDLE; }
    if (g_xrSession)      { xrDestroySession(g_xrSession); g_xrSession = XR_NULL_HANDLE; }
    if (g_xrInstance)     { xrDestroyInstance(g_xrInstance); g_xrInstance = XR_NULL_HANDLE; }
    g_sessionRunning = false;
}

bool begin_frame() {
    if (!g_xrSession) return false;

    XrEventDataBuffer ev{XR_TYPE_EVENT_DATA_BUFFER};
    while (xrPollEvent(g_xrInstance, &ev) == XR_SUCCESS) {
        if (ev.type == XR_TYPE_EVENT_DATA_SESSION_STATE_CHANGED) {
            auto* sc = (XrEventDataSessionStateChanged*)&ev;
            if (sc->state == XR_SESSION_STATE_READY) {
                XrSessionBeginInfo bi{XR_TYPE_SESSION_BEGIN_INFO};
                bi.primaryViewConfigurationType = XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO;
                xrBeginSession(g_xrSession, &bi);
                g_sessionRunning = true;
            } else if (sc->state == XR_SESSION_STATE_STOPPING) {
                g_sessionRunning = false;
                xrEndSession(g_xrSession);
            }
        }
        ev.type = XR_TYPE_EVENT_DATA_BUFFER;
    }

    if (!g_sessionRunning) return false;

    if (!g_dawnRenderTexture && webgpu::g_device) {
        create_dawn_textures();
    }
    return true;
}

void copy_to_shared(wgpu::CommandEncoder encoder) {
    if (!g_dawnRenderTexture || !g_dawnReadbackBuffer) return;

    const wgpu::TexelCopyTextureInfo src{.texture = g_dawnRenderTexture};
    const wgpu::TexelCopyBufferInfo dst{
        .layout = {
            .bytesPerRow  = g_swapchainWidth * 4,
            .rowsPerImage = g_swapchainHeight,
        },
        .buffer = g_dawnReadbackBuffer,
    };
    const wgpu::Extent3D extent{g_swapchainWidth, g_swapchainHeight, 1};
    encoder.CopyTextureToBuffer(&src, &dst, &extent);
}

void end_frame() {
    if (!g_sessionRunning || !g_dawnReadbackBuffer || !g_stagingMapped) return;

    // --- Map the Dawn readback buffer (synchronous via device tick) ---
    bool mapped = false;
    g_dawnReadbackBuffer.MapAsync(wgpu::MapMode::Read, 0, (uint64_t)g_swapchainWidth * g_swapchainHeight * 4,
        wgpu::CallbackMode::AllowSpontaneous,
        [&](wgpu::MapAsyncStatus status, wgpu::StringView) {
            mapped = (status == wgpu::MapAsyncStatus::Success);
        });

    // Tick until mapped
    while (!mapped) {
        webgpu::g_device.Tick();
    }

    if (!mapped) return; // safety

    const void* data = g_dawnReadbackBuffer.GetConstMappedRange(0, (uint64_t)g_swapchainWidth * g_swapchainHeight * 4);
    memcpy(g_stagingMapped, data, (size_t)g_swapchainWidth * g_swapchainHeight * 4);
    g_dawnReadbackBuffer.Unmap();

    // --- XR frame ---
    XrFrameWaitInfo waitInfo{XR_TYPE_FRAME_WAIT_INFO};
    XrFrameState frameState{XR_TYPE_FRAME_STATE};
    if (XR_FAILED(xrWaitFrame(g_xrSession, &waitInfo, &frameState))) return;

    XrFrameBeginInfo beginInfo{XR_TYPE_FRAME_BEGIN_INFO};
    xrBeginFrame(g_xrSession, &beginInfo);

    if (frameState.shouldRender) {
        uint32_t imageIndex;
        XrSwapchainImageAcquireInfo acqInfo{XR_TYPE_SWAPCHAIN_IMAGE_ACQUIRE_INFO};
        xrAcquireSwapchainImage(g_xrSwapchain, &acqInfo, &imageIndex);

        XrSwapchainImageWaitInfo swWait{XR_TYPE_SWAPCHAIN_IMAGE_WAIT_INFO};
        swWait.timeout = XR_INFINITE_DURATION;
        xrWaitSwapchainImage(g_xrSwapchain, &swWait);

        // Upload staging -> swapchain image
        VkCommandBufferBeginInfo cbBI{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        vkBeginCommandBuffer(g_vkCommandBuffer, &cbBI);

        // Transition swapchain image: UNDEFINED -> TRANSFER_DST_OPTIMAL
        VkImageMemoryBarrier toTransfer{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        toTransfer.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        toTransfer.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        toTransfer.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        toTransfer.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        toTransfer.image = g_swapchainImages[imageIndex].image;
        toTransfer.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        toTransfer.srcAccessMask = 0;
        toTransfer.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        vkCmdPipelineBarrier(g_vkCommandBuffer,
            VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
            0, 0, nullptr, 0, nullptr, 1, &toTransfer);

        VkBufferImageCopy region{};
        region.bufferOffset      = 0;
        region.bufferRowLength   = 0;
        region.bufferImageHeight = 0;
        region.imageSubresource  = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        region.imageOffset       = {0, 0, 0};
        region.imageExtent       = {g_swapchainWidth, g_swapchainHeight, 1};
        vkCmdCopyBufferToImage(g_vkCommandBuffer, g_stagingBuffer,
            g_swapchainImages[imageIndex].image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            1, &region);

        // Transition: TRANSFER_DST_OPTIMAL -> COLOR_ATTACHMENT_OPTIMAL (for XR compositor)
        VkImageMemoryBarrier toColor{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        toColor.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        toColor.newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        toColor.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        toColor.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        toColor.image = g_swapchainImages[imageIndex].image;
        toColor.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        toColor.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        toColor.dstAccessMask = 0;
        vkCmdPipelineBarrier(g_vkCommandBuffer,
            VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
            0, 0, nullptr, 0, nullptr, 1, &toColor);

        vkEndCommandBuffer(g_vkCommandBuffer);

        VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        submit.commandBufferCount = 1;
        submit.pCommandBuffers = &g_vkCommandBuffer;
        vkQueueSubmit(g_vkQueue, 1, &submit, VK_NULL_HANDLE);
        vkQueueWaitIdle(g_vkQueue);

        XrSwapchainImageReleaseInfo relInfo{XR_TYPE_SWAPCHAIN_IMAGE_RELEASE_INFO};
        xrReleaseSwapchainImage(g_xrSwapchain, &relInfo);
    }

    // Locate views
    XrViewState viewState{XR_TYPE_VIEW_STATE};
    uint32_t viewCount;
    XrViewLocateInfo vli{XR_TYPE_VIEW_LOCATE_INFO};
    vli.viewConfigurationType = XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO;
    vli.displayTime = frameState.predictedDisplayTime;
    vli.space = g_xrSpace;
    std::vector<XrView> views(2, {XR_TYPE_VIEW});
    xrLocateViews(g_xrSession, &vli, &viewState, 2, &viewCount, views.data());

    // Store the head orientation from the first view's pose for the game camera.
    // We use the orientation validity flag from viewState; if both position and
    // orientation bits are set the pose is fully tracked.
    if ((viewState.viewStateFlags & XR_VIEW_STATE_ORIENTATION_VALID_BIT) && viewCount >= 1) {
        g_headOrientation = views[0].pose.orientation;
        g_headPoseValid = true;
    } else {
        g_headPoseValid = false;
    }

    XrCompositionLayerProjection layer{XR_TYPE_COMPOSITION_LAYER_PROJECTION};
    layer.space = g_xrSpace;
    std::vector<XrCompositionLayerProjectionView> projViews(2, {XR_TYPE_COMPOSITION_LAYER_PROJECTION_VIEW});
    for (int i = 0; i < 2; i++) {
        projViews[i].pose = views[i].pose;
        projViews[i].fov  = views[i].fov;
        projViews[i].subImage.swapchain = g_xrSwapchain;
        projViews[i].subImage.imageRect.extent = {(int32_t)g_swapchainWidth, (int32_t)g_swapchainHeight};
    }
    layer.viewCount = 2;
    layer.views = projViews.data();

    std::vector<XrCompositionLayerBaseHeader*> layers;
    if (frameState.shouldRender) {
        layers.push_back((XrCompositionLayerBaseHeader*)&layer);
    }

    XrFrameEndInfo endInfo{XR_TYPE_FRAME_END_INFO};
    endInfo.displayTime         = frameState.predictedDisplayTime;
    endInfo.environmentBlendMode = XR_ENVIRONMENT_BLEND_MODE_OPAQUE;
    endInfo.layerCount          = layers.size();
    endInfo.layers              = layers.data();
    xrEndFrame(g_xrSession, &endInfo);
}

bool is_initialized() { return g_xrInstance != XR_NULL_HANDLE; }

bool get_head_pose(float& ox, float& oy, float& oz, float& ow) {
    if (!g_headPoseValid) return false;
    ox = g_headOrientation.x;
    oy = g_headOrientation.y;
    oz = g_headOrientation.z;
    ow = g_headOrientation.w;
    return true;
}

wgpu::TextureView get_texture_view() {
    if (g_dawnRenderTexture) return g_dawnRenderTexture.CreateView();
    return nullptr;
}

uint32_t get_width()  { return g_swapchainWidth; }
uint32_t get_height() { return g_swapchainHeight; }

} // namespace aurora::openxr

#include "xr.hpp"
#include "logging.hpp"
#include "webgpu/gpu.hpp"
#include <vector>
#include <sstream>
#include <cstring>
#include <cmath>
#include <iostream>
#include <dawn/native/VulkanBackend.h>
#ifdef __ANDROID__
#include <SDL3/SDL_system.h>
#include <android/hardware_buffer.h>
#include <vulkan/vulkan_android.h>
#endif

namespace aurora::xr {
static Module Log("aurora::xr");

static XrInstance g_instance = XR_NULL_HANDLE;
static XrSystemId g_systemId = XR_NULL_SYSTEM_ID;
static XrSession g_session = XR_NULL_HANDLE;
static XrSpace g_xrSpace = XR_NULL_HANDLE;
static XrSwapchain g_xrSwapchain = XR_NULL_HANDLE;

static VkInstance g_vkInstance = VK_NULL_HANDLE;
static VkPhysicalDevice g_vkPhysicalDevice = VK_NULL_HANDLE;
static VkDevice g_vkDevice = VK_NULL_HANDLE;
static VkQueue g_vkQueue = VK_NULL_HANDLE;
static uint32_t g_vkQueueFamilyIndex = 0;
static VkCommandPool g_vkCommandPool = VK_NULL_HANDLE;
static VkCommandBuffer g_vkCommandBuffer = VK_NULL_HANDLE;

static std::vector<XrSwapchainImageVulkanKHR> g_swapchainImages;
static int64_t g_swapchainFormat = 0;
static uint32_t g_swapchainWidth = 0;
static uint32_t g_swapchainHeight = 0;
static uint32_t g_eyeWidth = 0;
static bool g_sessionRunning = false;

static XrQuaternionf g_headOrientation{0.f, 0.f, 0.f, 1.f};
static bool g_headPoseValid = false;
static XrView g_eyeViews[2] = { {XR_TYPE_VIEW}, {XR_TYPE_VIEW} };
static bool g_eyeViewsValid = false;

// Persistent zero-copy Vulkan-WebGPU texture interop resources
static VkImage g_sharedVkImage = VK_NULL_HANDLE;
static VkDeviceMemory g_sharedVkMemory = VK_NULL_HANDLE;
static wgpu::Texture g_dawnRenderTexture = nullptr;
#ifdef __ANDROID__
static AHardwareBuffer* g_hardwareBuffer = nullptr;
#endif

bool initialize_instance() {
#ifdef __ANDROID__
    // Initialize the OpenXR loader on Android - must be done before any other xr* calls.
    PFN_xrInitializeLoaderKHR xrInitializeLoaderKHR = nullptr;
    xrGetInstanceProcAddr(XR_NULL_HANDLE, "xrInitializeLoaderKHR",
                          (PFN_xrVoidFunction*)&xrInitializeLoaderKHR);
    if (xrInitializeLoaderKHR) {
        JNIEnv* env = static_cast<JNIEnv*>(SDL_GetAndroidJNIEnv());
        if (env == nullptr) {
            Log.error("Failed to get Android JNI Env!");
            return false;
        }
        JavaVM* vm = nullptr;
        env->GetJavaVM(&vm);
        if (vm == nullptr) {
            Log.error("Failed to get Android JavaVM!");
            return false;
        }
        jobject activity = static_cast<jobject>(SDL_GetAndroidActivity());
        if (activity == nullptr) {
            Log.error("Failed to get Android Activity context!");
            return false;
        }
        XrLoaderInitInfoAndroidKHR loaderInfo{XR_TYPE_LOADER_INIT_INFO_ANDROID_KHR};
        loaderInfo.applicationVM = vm;
        loaderInfo.applicationContext = activity;
        XrResult res = xrInitializeLoaderKHR((const XrLoaderInitInfoBaseHeaderKHR*)&loaderInfo);
        if (XR_FAILED(res)) {
            Log.error("Failed to initialize OpenXR Loader: {}", (int)res);
            return false;
        }
        Log.info("OpenXR Loader initialized successfully on Android");
    } else {
        Log.error("xrInitializeLoaderKHR function not found!");
        return false;
    }
#endif

    uint32_t extCount = 0;
    xrEnumerateInstanceExtensionProperties(nullptr, 0, &extCount, nullptr);
    std::vector<XrExtensionProperties> extProps(extCount, {XR_TYPE_EXTENSION_PROPERTIES});
    xrEnumerateInstanceExtensionProperties(nullptr, extCount, &extCount, extProps.data());

    auto has_ext = [&](const char* name) {
        for (const auto& prop : extProps) {
            if (std::strcmp(prop.extensionName, name) == 0) return true;
        }
        return false;
    };

    std::vector<const char*> extensions;
    bool hasVulkanEnable2 = has_ext(XR_KHR_VULKAN_ENABLE2_EXTENSION_NAME);
    bool hasVulkanEnable = has_ext(XR_KHR_VULKAN_ENABLE_EXTENSION_NAME);

    if (hasVulkanEnable) {
        extensions.push_back(XR_KHR_VULKAN_ENABLE_EXTENSION_NAME);
        Log.info("Using OpenXR Vulkan Enable (Legacy) extension");
    } else if (hasVulkanEnable2) {
        extensions.push_back(XR_KHR_VULKAN_ENABLE2_EXTENSION_NAME);
        Log.info("Using OpenXR Vulkan Enable 2 extension");
    } else {
        Log.error("Neither Vulkan Enable nor Vulkan Enable 2 extensions are supported by OpenXR!");
        return false;
    }

    XrInstanceCreateInfo createInfo{};
    createInfo.type = XR_TYPE_INSTANCE_CREATE_INFO;
    createInfo.next = nullptr;

    snprintf(
        createInfo.applicationInfo.applicationName,
        XR_MAX_APPLICATION_NAME_SIZE,
        "Aurora"
    );

    createInfo.applicationInfo.applicationVersion = 1;

    snprintf(
        createInfo.applicationInfo.engineName,
        XR_MAX_ENGINE_NAME_SIZE,
        "Aurora"
    );

    createInfo.applicationInfo.engineVersion = 1;
    createInfo.applicationInfo.apiVersion = XR_API_VERSION_1_0;

    createInfo.enabledExtensionCount =
        static_cast<uint32_t>(extensions.size());

    createInfo.enabledExtensionNames = extensions.data();

    XrResult result =
        xrCreateInstance(&createInfo, &g_instance);

    if (XR_FAILED(result)) {
        Log.error(
            "Failed to create OpenXR instance: {}",
            (int)result
        );
        return false;
    }

    XrSystemGetInfo systemInfo{};
    systemInfo.type = XR_TYPE_SYSTEM_GET_INFO;
    systemInfo.formFactor =
        XR_FORM_FACTOR_HEAD_MOUNTED_DISPLAY;

    result =
        xrGetSystem(g_instance,
                    &systemInfo,
                    &g_systemId);

    if (XR_FAILED(result)) {
        Log.error(
            "Failed to get OpenXR system: {}",
            (int)result
        );
        return false;
    }

    return true;
}

void shutdown() {
    g_dawnRenderTexture = nullptr;

    if (g_sharedVkImage && g_vkDevice != VK_NULL_HANDLE) {
        vkDestroyImage(g_vkDevice, g_sharedVkImage, nullptr);
        g_sharedVkImage = VK_NULL_HANDLE;
    }
    if (g_sharedVkMemory && g_vkDevice != VK_NULL_HANDLE) {
        vkFreeMemory(g_vkDevice, g_sharedVkMemory, nullptr);
        g_sharedVkMemory = VK_NULL_HANDLE;
    }
    if (g_vkCommandPool && g_vkDevice != VK_NULL_HANDLE) {
        vkDestroyCommandPool(g_vkDevice, g_vkCommandPool, nullptr);
        g_vkCommandPool = VK_NULL_HANDLE;
        g_vkCommandBuffer = VK_NULL_HANDLE;
    }
    if (g_xrSwapchain) {
        xrDestroySwapchain(g_xrSwapchain);
        g_xrSwapchain = XR_NULL_HANDLE;
    }
    if (g_xrSpace) {
        xrDestroySpace(g_xrSpace);
        g_xrSpace = XR_NULL_HANDLE;
    }
    if (g_session) {
        xrDestroySession(g_session);
        g_session = XR_NULL_HANDLE;
    }
    if (g_instance != XR_NULL_HANDLE) {
        xrDestroyInstance(g_instance);
        g_instance = XR_NULL_HANDLE;
    }

    g_sessionRunning = false;
    g_headPoseValid = false;
    g_eyeViewsValid = false;
}

bool is_enabled() {
    return g_instance != XR_NULL_HANDLE;
}

std::vector<std::string> get_vulkan_instance_extensions() {
    if (g_instance == XR_NULL_HANDLE) return {};
    
    PFN_xrGetVulkanInstanceExtensionsKHR xrGetVulkanInstanceExtensionsKHR;
    xrGetInstanceProcAddr(g_instance, "xrGetVulkanInstanceExtensionsKHR", (PFN_xrVoidFunction*)&xrGetVulkanInstanceExtensionsKHR);
    
    uint32_t count = 0;
    xrGetVulkanInstanceExtensionsKHR(g_instance, g_systemId, 0, &count, nullptr);
    std::string extensionsStr(count, ' ');
    xrGetVulkanInstanceExtensionsKHR(g_instance, g_systemId, count, &count, extensionsStr.data());
    
    std::vector<std::string> extensions;
    std::istringstream iss(extensionsStr);
    std::string s;
    while (iss >> s) {
        extensions.push_back(s);
    }
    return extensions;
}

std::vector<std::string> get_vulkan_device_extensions() {
    if (g_instance == XR_NULL_HANDLE) return {};
    
    PFN_xrGetVulkanDeviceExtensionsKHR xrGetVulkanDeviceExtensionsKHR;
    xrGetInstanceProcAddr(g_instance, "xrGetVulkanDeviceExtensionsKHR", (PFN_xrVoidFunction*)&xrGetVulkanDeviceExtensionsKHR);
    
    uint32_t count = 0;
    xrGetVulkanDeviceExtensionsKHR(g_instance, g_systemId, 0, &count, nullptr);
    std::string extensionsStr(count, ' ');
    xrGetVulkanDeviceExtensionsKHR(g_instance, g_systemId, count, &count, extensionsStr.data());
    
    std::vector<std::string> extensions;
    std::istringstream iss(extensionsStr);
    std::string s;
    while (iss >> s) {
        extensions.push_back(s);
    }
    return extensions;
}

VkPhysicalDevice get_vulkan_graphics_device(VkInstance instance) {
    if (g_instance == XR_NULL_HANDLE) return VK_NULL_HANDLE;
    
    // Call graphics requirements first as required by OpenXR spec!
    // Try Vulkan Enable 2 requirements first
    PFN_xrGetVulkanGraphicsRequirements2KHR xrGetVulkanGraphicsRequirements2KHR = nullptr;
    xrGetInstanceProcAddr(g_instance, "xrGetVulkanGraphicsRequirements2KHR", (PFN_xrVoidFunction*)&xrGetVulkanGraphicsRequirements2KHR);
    if (xrGetVulkanGraphicsRequirements2KHR) {
        XrGraphicsRequirementsVulkanKHR requirements{XR_TYPE_GRAPHICS_REQUIREMENTS_VULKAN_KHR};
        xrGetVulkanGraphicsRequirements2KHR(g_instance, g_systemId, &requirements);
    } else {
        PFN_xrGetVulkanGraphicsRequirementsKHR xrGetVulkanGraphicsRequirementsKHR = nullptr;
        xrGetInstanceProcAddr(g_instance, "xrGetVulkanGraphicsRequirementsKHR", (PFN_xrVoidFunction*)&xrGetVulkanGraphicsRequirementsKHR);
        if (xrGetVulkanGraphicsRequirementsKHR) {
            XrGraphicsRequirementsVulkanKHR requirements{XR_TYPE_GRAPHICS_REQUIREMENTS_VULKAN_KHR};
            xrGetVulkanGraphicsRequirementsKHR(g_instance, g_systemId, &requirements);
        }
    }

    // Try Vulkan Enable 2 first
    PFN_xrGetVulkanGraphicsDevice2KHR xrGetVulkanGraphicsDevice2KHR = nullptr;
    xrGetInstanceProcAddr(g_instance, "xrGetVulkanGraphicsDevice2KHR", (PFN_xrVoidFunction*)&xrGetVulkanGraphicsDevice2KHR);
    if (xrGetVulkanGraphicsDevice2KHR) {
        XrVulkanGraphicsDeviceGetInfoKHR getInfo{XR_TYPE_VULKAN_GRAPHICS_DEVICE_GET_INFO_KHR};
        getInfo.systemId = g_systemId;
        getInfo.vulkanInstance = instance;
        
        VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
        XrResult res = xrGetVulkanGraphicsDevice2KHR(g_instance, &getInfo, &physicalDevice);
        if (XR_SUCCEEDED(res) && physicalDevice != VK_NULL_HANDLE) {
            return physicalDevice;
        } else {
            Log.warn("xrGetVulkanGraphicsDevice2KHR returned error: {} or physicalDevice is null", (int)res);
        }
    }

    // Fallback to legacy KHR
    PFN_xrGetVulkanGraphicsDeviceKHR xrGetVulkanGraphicsDeviceKHR = nullptr;
    xrGetInstanceProcAddr(g_instance, "xrGetVulkanGraphicsDeviceKHR", (PFN_xrVoidFunction*)&xrGetVulkanGraphicsDeviceKHR);
    if (xrGetVulkanGraphicsDeviceKHR) {
        VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
        XrResult res = xrGetVulkanGraphicsDeviceKHR(g_instance, g_systemId, instance, &physicalDevice);
        if (XR_SUCCEEDED(res) && physicalDevice != VK_NULL_HANDLE) {
            return physicalDevice;
        } else {
            Log.warn("xrGetVulkanGraphicsDeviceKHR returned error: {} or physicalDevice is null", (int)res);
        }
    }
    
    Log.error("Failed to query physical device from OpenXR");
    return VK_NULL_HANDLE;
}

static uint32_t find_memory_type(uint32_t filter, VkMemoryPropertyFlags props) {
    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(g_vkPhysicalDevice, &mp);
    for (uint32_t i = 0; i < mp.memoryTypeCount; i++) {
        if ((filter & (1 << i)) && (mp.memoryTypes[i].propertyFlags & props) == props)
            return i;
    }
    throw std::runtime_error("No suitable memory type");
}

static void create_dawn_textures() {
    if (!webgpu::g_device) return;

    VkFormat vkFmt = VK_FORMAT_R8G8B8A8_UNORM;
    wgpu::TextureFormat wgpuFmt = wgpu::TextureFormat::RGBA8Unorm;

    if (g_swapchainFormat == VK_FORMAT_B8G8R8A8_UNORM) {
        vkFmt = VK_FORMAT_B8G8R8A8_UNORM;
        wgpuFmt = wgpu::TextureFormat::BGRA8Unorm;
    } else if (g_swapchainFormat == VK_FORMAT_B8G8R8A8_SRGB) {
        vkFmt = VK_FORMAT_B8G8R8A8_SRGB;
        wgpuFmt = wgpu::TextureFormat::BGRA8UnormSrgb;
    } else if (g_swapchainFormat == VK_FORMAT_R8G8B8A8_SRGB) {
        vkFmt = VK_FORMAT_R8G8B8A8_SRGB;
        wgpuFmt = wgpu::TextureFormat::RGBA8UnormSrgb;
    } else if (g_swapchainFormat == VK_FORMAT_R8G8B8A8_UNORM) {
        vkFmt = VK_FORMAT_R8G8B8A8_UNORM;
        wgpuFmt = wgpu::TextureFormat::RGBA8Unorm;
    }

    wgpu::TextureDescriptor td{};
    td.size = {g_swapchainWidth, g_swapchainHeight, 1};
    td.format = wgpuFmt;
    td.usage = wgpu::TextureUsage::RenderAttachment | wgpu::TextureUsage::CopySrc;

    WGPUTexture wgpuTex = nullptr;

#ifdef __ANDROID__
    // --- Android path: use AHardwareBuffer for Dawn image sharing ---
    // Convert VkFormat to AHardwareBuffer_Format
    // AHardwareBuffer only supports RGBA formats natively; BGRA is not available.
    // Use RGBA for all cases - the swapchain format selection will pick RGBA on Android.
    uint32_t ahbFormat = AHARDWAREBUFFER_FORMAT_R8G8B8A8_UNORM;

    // 1. Allocate AHardwareBuffer
    AHardwareBuffer_Desc ahbDesc{};
    ahbDesc.width = g_swapchainWidth;
    ahbDesc.height = g_swapchainHeight;
    ahbDesc.layers = 1;
    ahbDesc.format = ahbFormat;
    ahbDesc.usage = AHARDWAREBUFFER_USAGE_GPU_FRAMEBUFFER |
                    AHARDWAREBUFFER_USAGE_GPU_SAMPLED_IMAGE;
    if (AHardwareBuffer_allocate(&ahbDesc, &g_hardwareBuffer) != 0) {
        Log.error("Failed to allocate AHardwareBuffer");
        return;
    }

    // 2. Query Vulkan properties for the AHardwareBuffer
    auto pfnGetAHBProperties = (PFN_vkGetAndroidHardwareBufferPropertiesANDROID)
        vkGetDeviceProcAddr(g_vkDevice, "vkGetAndroidHardwareBufferPropertiesANDROID");
    if (!pfnGetAHBProperties) {
        Log.error("vkGetAndroidHardwareBufferPropertiesANDROID not found");
        AHardwareBuffer_release(g_hardwareBuffer);
        g_hardwareBuffer = nullptr;
        return;
    }
    VkAndroidHardwareBufferPropertiesANDROID ahbProps{
        VK_STRUCTURE_TYPE_ANDROID_HARDWARE_BUFFER_PROPERTIES_ANDROID};
    if (pfnGetAHBProperties(g_vkDevice, g_hardwareBuffer, &ahbProps) != VK_SUCCESS) {
        Log.error("vkGetAndroidHardwareBufferPropertiesANDROID failed");
        AHardwareBuffer_release(g_hardwareBuffer);
        g_hardwareBuffer = nullptr;
        return;
    }

    // 3. Create VkImage backed by the AHardwareBuffer
    VkExternalMemoryImageCreateInfo extImageCI{VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO};
    extImageCI.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_ANDROID_HARDWARE_BUFFER_BIT_ANDROID;

    VkImageCreateInfo imageCI{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    imageCI.pNext = &extImageCI;
    imageCI.imageType = VK_IMAGE_TYPE_2D;
    imageCI.format = vkFmt;
    imageCI.extent = {g_swapchainWidth, g_swapchainHeight, 1};
    imageCI.mipLevels = 1;
    imageCI.arrayLayers = 1;
    imageCI.samples = VK_SAMPLE_COUNT_1_BIT;
    imageCI.tiling = VK_IMAGE_TILING_OPTIMAL;
    imageCI.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    imageCI.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    imageCI.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    if (vkCreateImage(g_vkDevice, &imageCI, nullptr, &g_sharedVkImage) != VK_SUCCESS) {
        Log.error("Failed to create AHardwareBuffer-backed VkImage");
        AHardwareBuffer_release(g_hardwareBuffer);
        g_hardwareBuffer = nullptr;
        return;
    }

    // 4. Import AHardwareBuffer as VkDeviceMemory
    VkImportAndroidHardwareBufferInfoANDROID importInfo{
        VK_STRUCTURE_TYPE_IMPORT_ANDROID_HARDWARE_BUFFER_INFO_ANDROID};
    importInfo.buffer = g_hardwareBuffer;

    VkMemoryDedicatedAllocateInfo dedicatedAllocInfo{VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO};
    dedicatedAllocInfo.pNext = &importInfo;
    dedicatedAllocInfo.image = g_sharedVkImage;

    uint32_t memTypeIndex = 0;
    uint32_t memTypeBits = ahbProps.memoryTypeBits;
    for (uint32_t i = 0; i < 32; ++i) {
        if (memTypeBits & (1u << i)) { memTypeIndex = i; break; }
    }

    VkMemoryAllocateInfo allocInfo{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    allocInfo.pNext = &dedicatedAllocInfo;
    allocInfo.allocationSize = ahbProps.allocationSize;
    allocInfo.memoryTypeIndex = memTypeIndex;
    if (vkAllocateMemory(g_vkDevice, &allocInfo, nullptr, &g_sharedVkMemory) != VK_SUCCESS) {
        Log.error("Failed to allocate VkDeviceMemory from AHardwareBuffer");
        vkDestroyImage(g_vkDevice, g_sharedVkImage, nullptr);
        g_sharedVkImage = VK_NULL_HANDLE;
        AHardwareBuffer_release(g_hardwareBuffer);
        g_hardwareBuffer = nullptr;
        return;
    }
    vkBindImageMemory(g_vkDevice, g_sharedVkImage, g_sharedVkMemory, 0);

    // 5. Wrap the AHardwareBuffer in Dawn
    dawn::native::vulkan::ExternalImageDescriptorAHardwareBuffer desc;
    desc.cTextureDescriptor = reinterpret_cast<const WGPUTextureDescriptor*>(&td);
    desc.isInitialized = false;
    // Dawn forward-declares 'struct AHardwareBuffer' inside its own namespace, making
    // desc.handle a 'dawn::native::vulkan::AHardwareBuffer*' that is incompatible with
    // our '::AHardwareBuffer*' at the type system level even though they are the same.
    // Use memcpy to assign the pointer safely without a type-system cast.
    { void* _ahb_ptr = g_hardwareBuffer; std::memcpy(&desc.handle, &_ahb_ptr, sizeof(void*)); }
    desc.releasedOldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    desc.releasedNewLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    wgpuTex = dawn::native::vulkan::WrapVulkanImage(webgpu::g_device.Get(), &desc);

#elif defined(_WIN32)
    // --- Windows path: Vulkan external memory wrapping is not supported in Dawn ---
    Log.error("Vulkan external image sharing is not supported on Windows");
    return;
#else // Linux / Desktop: OpaqueFD path
    // 1. Create external VkImage
    VkFormat viewFormats[] = { vkFmt };
    VkImageFormatListCreateInfo formatListCI{VK_STRUCTURE_TYPE_IMAGE_FORMAT_LIST_CREATE_INFO};
    formatListCI.viewFormatCount = 1;
    formatListCI.pViewFormats = viewFormats;

    VkExternalMemoryImageCreateInfo extImageCI{VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO};
    extImageCI.pNext = &formatListCI;
    extImageCI.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;

    VkImageCreateInfo imageCI{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    imageCI.flags = VK_IMAGE_CREATE_ALIAS_BIT;
    imageCI.pNext = &extImageCI;
    imageCI.imageType = VK_IMAGE_TYPE_2D;
    imageCI.format = vkFmt;
    imageCI.extent = { g_swapchainWidth, g_swapchainHeight, 1 };
    imageCI.mipLevels = 1;
    imageCI.arrayLayers = 1;
    imageCI.samples = VK_SAMPLE_COUNT_1_BIT;
    imageCI.tiling = VK_IMAGE_TILING_OPTIMAL;
    imageCI.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    imageCI.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    imageCI.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    if (vkCreateImage(g_vkDevice, &imageCI, nullptr, &g_sharedVkImage) != VK_SUCCESS) {
        Log.error("Failed to create shared VkImage");
        return;
    }

    VkMemoryRequirements memReqs;
    vkGetImageMemoryRequirements(g_vkDevice, g_sharedVkImage, &memReqs);

    VkMemoryDedicatedAllocateInfo dedicatedAllocInfo{VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO};
    dedicatedAllocInfo.image = g_sharedVkImage;
    dedicatedAllocInfo.buffer = VK_NULL_HANDLE;

    VkExportMemoryAllocateInfo exportAllocInfo{VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO};
    exportAllocInfo.pNext = &dedicatedAllocInfo;
    exportAllocInfo.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;

    VkMemoryAllocateInfo allocInfo{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    allocInfo.pNext = &exportAllocInfo;
    allocInfo.allocationSize = memReqs.size;
    try {
        allocInfo.memoryTypeIndex = find_memory_type(memReqs.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    } catch (const std::exception& e) {
        Log.error("Shared memory type check failed: {}", e.what());
        return;
    }

    if (vkAllocateMemory(g_vkDevice, &allocInfo, nullptr, &g_sharedVkMemory) != VK_SUCCESS) {
        Log.error("Failed to allocate shared VkDeviceMemory");
        return;
    }
    vkBindImageMemory(g_vkDevice, g_sharedVkImage, g_sharedVkMemory, 0);

    PFN_vkGetMemoryFdKHR pfnGetMemoryFdKHR = (PFN_vkGetMemoryFdKHR)vkGetDeviceProcAddr(g_vkDevice, "vkGetMemoryFdKHR");
    if (!pfnGetMemoryFdKHR) {
        Log.error("vkGetMemoryFdKHR function pointer not found!");
        return;
    }
    VkMemoryGetFdInfoKHR getFdInfo{VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR};
    getFdInfo.memory = g_sharedVkMemory;
    getFdInfo.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
    int memoryFD = -1;
    if (pfnGetMemoryFdKHR(g_vkDevice, &getFdInfo, &memoryFD) != VK_SUCCESS) {
        Log.error("Failed to get Vulkan memory FD");
        return;
    }

    dawn::native::vulkan::ExternalImageDescriptorOpaqueFD desc;
    desc.cTextureDescriptor = reinterpret_cast<const WGPUTextureDescriptor*>(&td);
    desc.isInitialized = false;
    desc.memoryFD = memoryFD;
    desc.allocationSize = memReqs.size;
    desc.memoryTypeIndex = allocInfo.memoryTypeIndex;
    desc.releasedOldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    desc.releasedNewLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    wgpuTex = dawn::native::vulkan::WrapVulkanImage(webgpu::g_device.Get(), &desc);
#endif // __ANDROID__

    if (!wgpuTex) {
        Log.error("Failed to wrap Vulkan image in Dawn");
        return;
    }
    g_dawnRenderTexture = wgpu::Texture::Acquire(wgpuTex);
    Log.info("Zero-copy Vulkan image wrapped in Dawn successfully. Dimensions: {}x{}", g_swapchainWidth, g_swapchainHeight);
}

bool initialize_session(VkInstance instance, VkPhysicalDevice physicalDevice, VkDevice device, uint32_t queueFamilyIndex, uint32_t queueIndex) {
    g_vkInstance = instance;
    g_vkPhysicalDevice = physicalDevice;
    g_vkDevice = device;
    g_vkQueueFamilyIndex = queueFamilyIndex;
    vkGetDeviceQueue(device, queueFamilyIndex, queueIndex, &g_vkQueue);

    PFN_xrGetVulkanGraphicsRequirements2KHR xrGetVulkanGraphicsRequirements2KHR = nullptr;
    xrGetInstanceProcAddr(g_instance, "xrGetVulkanGraphicsRequirements2KHR", (PFN_xrVoidFunction*)&xrGetVulkanGraphicsRequirements2KHR);
    if (xrGetVulkanGraphicsRequirements2KHR) {
        XrGraphicsRequirementsVulkanKHR requirements{XR_TYPE_GRAPHICS_REQUIREMENTS_VULKAN_KHR};
        XrResult res = xrGetVulkanGraphicsRequirements2KHR(g_instance, g_systemId, &requirements);
        if (XR_FAILED(res)) {
            Log.error("Failed to get Vulkan 2 graphics requirements: {}", (int)res);
            return false;
        }
    } else {
        PFN_xrGetVulkanGraphicsRequirementsKHR xrGetVulkanGraphicsRequirementsKHR = nullptr;
        xrGetInstanceProcAddr(g_instance, "xrGetVulkanGraphicsRequirementsKHR", (PFN_xrVoidFunction*)&xrGetVulkanGraphicsRequirementsKHR);
        if (xrGetVulkanGraphicsRequirementsKHR) {
            XrGraphicsRequirementsVulkanKHR requirements{XR_TYPE_GRAPHICS_REQUIREMENTS_VULKAN_KHR};
            XrResult res = xrGetVulkanGraphicsRequirementsKHR(g_instance, g_systemId, &requirements);
            if (XR_FAILED(res)) {
                Log.error("Failed to get legacy Vulkan graphics requirements: {}", (int)res);
                return false;
            }
        } else {
            Log.error("Failed to load any xrGetVulkanGraphicsRequirements function pointer");
            return false;
        }
    }

    XrGraphicsBindingVulkanKHR binding{XR_TYPE_GRAPHICS_BINDING_VULKAN_KHR};
    binding.instance = instance;
    binding.physicalDevice = physicalDevice;
    binding.device = device;
    binding.queueFamilyIndex = queueFamilyIndex;
    binding.queueIndex = queueIndex;

    XrSessionCreateInfo createInfo{XR_TYPE_SESSION_CREATE_INFO};
    createInfo.next = &binding;
    createInfo.systemId = g_systemId;

    XrResult result = xrCreateSession(g_instance, &createInfo, &g_session);
    if (XR_FAILED(result)) {
        Log.error("Failed to create OpenXR session: {}", (int)result);
        return false;
    }

    XrReferenceSpaceCreateInfo spCI{XR_TYPE_REFERENCE_SPACE_CREATE_INFO};
    spCI.referenceSpaceType = XR_REFERENCE_SPACE_TYPE_LOCAL;
    spCI.poseInReferenceSpace.orientation.w = 1.0f;
    result = xrCreateReferenceSpace(g_session, &spCI, &g_xrSpace);
    if (XR_FAILED(result)) {
        Log.error("Failed to create OpenXR reference space: {}", (int)result);
        return false;
    }

    uint32_t vcc = 0;
    xrEnumerateViewConfigurationViews(g_instance, g_systemId, XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO, 0, &vcc, nullptr);
    std::vector<XrViewConfigurationView> views(vcc, {XR_TYPE_VIEW_CONFIGURATION_VIEW});
    xrEnumerateViewConfigurationViews(g_instance, g_systemId, XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO, vcc, &vcc, views.data());

    g_eyeWidth = views[0].recommendedImageRectWidth;
    g_swapchainWidth = g_eyeWidth * 2;
    g_swapchainHeight = views[0].recommendedImageRectHeight;

    uint32_t fmtCount = 0;
    xrEnumerateSwapchainFormats(g_session, 0, &fmtCount, nullptr);
    std::vector<int64_t> fmts(fmtCount);
    xrEnumerateSwapchainFormats(g_session, fmtCount, &fmtCount, fmts.data());

    int64_t selectedFmt = fmts[0];
    for (int64_t f : fmts) {
        if (f == VK_FORMAT_R8G8B8A8_UNORM || f == VK_FORMAT_R8G8B8A8_SRGB || f == VK_FORMAT_B8G8R8A8_SRGB || f == VK_FORMAT_B8G8R8A8_UNORM) {
            selectedFmt = f;
            break;
        }
    }
    g_swapchainFormat = selectedFmt;

    XrSwapchainCreateInfo scCI{XR_TYPE_SWAPCHAIN_CREATE_INFO};
    scCI.usageFlags = XR_SWAPCHAIN_USAGE_COLOR_ATTACHMENT_BIT | XR_SWAPCHAIN_USAGE_TRANSFER_DST_BIT;
    scCI.format = selectedFmt;
    scCI.sampleCount = 1;
    scCI.width = g_swapchainWidth;
    scCI.height = g_swapchainHeight;
    scCI.faceCount = 1;
    scCI.arraySize = 1;
    scCI.mipCount = 1;
    result = xrCreateSwapchain(g_session, &scCI, &g_xrSwapchain);
    if (XR_FAILED(result)) {
        Log.error("Failed to create OpenXR swapchain: {}", (int)result);
        return false;
    }

    uint32_t imgCount = 0;
    xrEnumerateSwapchainImages(g_xrSwapchain, 0, &imgCount, nullptr);
    g_swapchainImages.resize(imgCount, {XR_TYPE_SWAPCHAIN_IMAGE_VULKAN_KHR});
    xrEnumerateSwapchainImages(g_xrSwapchain, imgCount, &imgCount, (XrSwapchainImageBaseHeader*)g_swapchainImages.data());

    VkCommandPoolCreateInfo poolCI{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    poolCI.queueFamilyIndex = queueFamilyIndex;
    poolCI.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    if (vkCreateCommandPool(device, &poolCI, nullptr, &g_vkCommandPool) != VK_SUCCESS) {
        Log.error("Failed to create Vulkan command pool for OpenXR presenter");
        return false;
    }

    VkCommandBufferAllocateInfo cbAI{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cbAI.commandPool = g_vkCommandPool;
    cbAI.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbAI.commandBufferCount = 1;
    if (vkAllocateCommandBuffers(device, &cbAI, &g_vkCommandBuffer) != VK_SUCCESS) {
        Log.error("Failed to allocate Vulkan command buffer for OpenXR presenter");
        return false;
    }

    Log.info("OpenXR session and Vulkan resources created successfully");
    return true;
}

bool begin_frame() {
    if (!g_session) return false;

    XrEventDataBuffer ev{XR_TYPE_EVENT_DATA_BUFFER};
    while (xrPollEvent(g_instance, &ev) == XR_SUCCESS) {
        if (ev.type == XR_TYPE_EVENT_DATA_SESSION_STATE_CHANGED) {
            auto* sc = (XrEventDataSessionStateChanged*)&ev;
            if (sc->state == XR_SESSION_STATE_READY) {
                XrSessionBeginInfo bi{XR_TYPE_SESSION_BEGIN_INFO};
                bi.primaryViewConfigurationType = XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO;
                xrBeginSession(g_session, &bi);
                g_sessionRunning = true;
                Log.info("OpenXR session started running");
            } else if (sc->state == XR_SESSION_STATE_STOPPING) {
                g_sessionRunning = false;
                xrEndSession(g_session);
                Log.info("OpenXR session stopped");
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

void end_frame() {
    if (!g_sessionRunning || !g_dawnRenderTexture) return;

    if (g_vkDevice != VK_NULL_HANDLE) {
        vkDeviceWaitIdle(g_vkDevice);
    }

    XrFrameWaitInfo waitInfo{XR_TYPE_FRAME_WAIT_INFO};
    XrFrameState frameState{XR_TYPE_FRAME_STATE};
    if (XR_FAILED(xrWaitFrame(g_session, &waitInfo, &frameState))) return;

    xrBeginFrame(g_session, nullptr);

    if (frameState.shouldRender) {
        uint32_t imageIndex;
        XrSwapchainImageAcquireInfo acqInfo{XR_TYPE_SWAPCHAIN_IMAGE_ACQUIRE_INFO};
        xrAcquireSwapchainImage(g_xrSwapchain, &acqInfo, &imageIndex);

        XrSwapchainImageWaitInfo swWait{XR_TYPE_SWAPCHAIN_IMAGE_WAIT_INFO};
        swWait.timeout = XR_INFINITE_DURATION;
        xrWaitSwapchainImage(g_xrSwapchain, &swWait);

        // Record and submit the direct Vulkan copy command buffer
        VkCommandBufferBeginInfo cbBI{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        vkBeginCommandBuffer(g_vkCommandBuffer, &cbBI);

        // 1. Transition our shared image: COLOR_ATTACHMENT_OPTIMAL -> TRANSFER_SRC_OPTIMAL
        VkImageMemoryBarrier barrierSrc{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        barrierSrc.oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        barrierSrc.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        barrierSrc.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        barrierSrc.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        barrierSrc.image = g_sharedVkImage;
        barrierSrc.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        barrierSrc.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrierSrc.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;

        // 2. Transition swapchain image: UNDEFINED -> TRANSFER_DST_OPTIMAL
        VkImageMemoryBarrier barrierDst{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        barrierDst.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        barrierDst.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        barrierDst.srcAccessMask = 0;
        barrierDst.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrierDst.image = g_swapchainImages[imageIndex].image;
        barrierDst.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        barrierDst.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrierDst.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;

        vkCmdPipelineBarrier(g_vkCommandBuffer,
            VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
            0, 0, nullptr, 0, nullptr, 1, &barrierSrc);

        vkCmdPipelineBarrier(g_vkCommandBuffer,
            VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
            0, 0, nullptr, 0, nullptr, 1, &barrierDst);

        // 3. Perform the direct Vulkan image copy on the GPU
        VkImageCopy copyRegion{};
        copyRegion.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        copyRegion.srcOffset = {0, 0, 0};
        copyRegion.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        copyRegion.dstOffset = {0, 0, 0};
        copyRegion.extent = {g_swapchainWidth, g_swapchainHeight, 1};

        vkCmdCopyImage(g_vkCommandBuffer,
            g_sharedVkImage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            g_swapchainImages[imageIndex].image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            1, &copyRegion);

        // 4. Transition swapchain image: TRANSFER_DST_OPTIMAL -> COLOR_ATTACHMENT_OPTIMAL (for XR compositor)
        VkImageMemoryBarrier barrierDstColor{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        barrierDstColor.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        barrierDstColor.newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        barrierDstColor.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrierDstColor.dstAccessMask = 0;
        barrierDstColor.image = g_swapchainImages[imageIndex].image;
        barrierDstColor.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        barrierDstColor.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrierDstColor.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;

        // 5. Transition our shared image back: TRANSFER_SRC_OPTIMAL -> COLOR_ATTACHMENT_OPTIMAL (for WebGPU next frame)
        VkImageMemoryBarrier barrierSrcColor{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        barrierSrcColor.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        barrierSrcColor.newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        barrierSrcColor.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        barrierSrcColor.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        barrierSrcColor.image = g_sharedVkImage;
        barrierSrcColor.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        barrierSrcColor.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrierSrcColor.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;

        vkCmdPipelineBarrier(g_vkCommandBuffer,
            VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
            0, 0, nullptr, 0, nullptr, 1, &barrierDstColor);

        vkCmdPipelineBarrier(g_vkCommandBuffer,
            VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
            0, 0, nullptr, 0, nullptr, 1, &barrierSrcColor);

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
    uint32_t viewCount = 0;
    XrViewLocateInfo vli{XR_TYPE_VIEW_LOCATE_INFO};
    vli.viewConfigurationType = XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO;
    vli.displayTime = frameState.predictedDisplayTime;
    vli.space = g_xrSpace;
    std::vector<XrView> views(2, {XR_TYPE_VIEW});
    xrLocateViews(g_session, &vli, &viewState, 2, &viewCount, views.data());

    if ((viewState.viewStateFlags & XR_VIEW_STATE_ORIENTATION_VALID_BIT) && viewCount >= 1) {
        g_headOrientation = views[0].pose.orientation;
        g_headPoseValid = true;
    } else {
        g_headPoseValid = false;
    }

    if ((viewState.viewStateFlags & (XR_VIEW_STATE_ORIENTATION_VALID_BIT | XR_VIEW_STATE_POSITION_VALID_BIT)) && viewCount == 2) {
        g_eyeViews[0] = views[0];
        g_eyeViews[1] = views[1];
        g_eyeViewsValid = true;
    } else {
        g_eyeViewsValid = false;
    }

    XrCompositionLayerProjection layer{XR_TYPE_COMPOSITION_LAYER_PROJECTION};
    layer.space = g_xrSpace;
    std::vector<XrCompositionLayerProjectionView> projViews(2, {XR_TYPE_COMPOSITION_LAYER_PROJECTION_VIEW});

    for (int i = 0; i < 2; i++) {
        projViews[i].pose = views[i].pose;
        projViews[i].fov = views[i].fov;
        projViews[i].subImage.swapchain = g_xrSwapchain;
        projViews[i].subImage.imageRect.offset = {(int32_t)(i * g_eyeWidth), 0};
        projViews[i].subImage.imageRect.extent = {(int32_t)g_eyeWidth, (int32_t)g_swapchainHeight};
    }
    layer.viewCount = 2;
    layer.views = projViews.data();

    std::vector<XrCompositionLayerBaseHeader*> layers;
    if (frameState.shouldRender) {
        layers.push_back((XrCompositionLayerBaseHeader*)&layer);
    }

    XrFrameEndInfo endInfo{XR_TYPE_FRAME_END_INFO};
    endInfo.displayTime = frameState.predictedDisplayTime;
    endInfo.environmentBlendMode = XR_ENVIRONMENT_BLEND_MODE_OPAQUE;
    endInfo.layerCount = layers.size();
    endInfo.layers = layers.data();
    xrEndFrame(g_session, &endInfo);
}

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

uint32_t get_width() { return g_swapchainWidth; }
uint32_t get_height() { return g_swapchainHeight; }
uint32_t get_eye_width() { return g_eyeWidth; }

bool get_eye_info(int eye, XrEyeInfo& out) {
    if (!g_eyeViewsValid || eye < 0 || eye > 1) return false;

    const auto& p0 = g_eyeViews[0].pose.position;
    const auto& p1 = g_eyeViews[1].pose.position;
    const float dx = p1.x - p0.x;
    const float dy = p1.y - p0.y;
    const float dz = p1.z - p0.z;
    const float ipd = std::sqrt(dx*dx + dy*dy + dz*dz);

    out.poseX = (eye == 0) ? (-ipd * 0.5f) : (ipd * 0.5f);
    out.tanLeft = std::tan(-g_eyeViews[eye].fov.angleLeft);
    out.tanRight = std::tan(g_eyeViews[eye].fov.angleRight);
    return true;
}

} // namespace aurora::xr

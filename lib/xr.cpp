#include "xr.hpp"
#include "logging.hpp"
#include <vector>
#include <sstream>
#include <cstring>

namespace aurora::xr {
static Module Log("aurora::xr");

static XrInstance g_instance = XR_NULL_HANDLE;
static XrSystemId g_systemId = XR_NULL_SYSTEM_ID;
static XrSession g_session = XR_NULL_HANDLE;

bool initialize_instance() {
    std::vector<const char*> extensions = {XR_KHR_VULKAN_ENABLE_EXTENSION_NAME};
    
    XrInstanceCreateInfo createInfo{XR_TYPE_INSTANCE_CREATE_INFO};
    strncpy(createInfo.applicationInfo.applicationName, "Aurora", XR_MAX_APPLICATION_NAME_SIZE);
    createInfo.applicationInfo.applicationVersion = 1;
    strncpy(createInfo.applicationInfo.engineName, "Aurora", XR_MAX_ENGINE_NAME_SIZE);
    createInfo.applicationInfo.engineVersion = 1;
    createInfo.applicationInfo.apiVersion = XR_CURRENT_API_VERSION;
    createInfo.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
    createInfo.enabledExtensionNames = extensions.data();

    XrResult result = xrCreateInstance(&createInfo, &g_instance);
    if (XR_FAILED(result)) {
        Log.error("Failed to create OpenXR instance: {}", (int)result);
        return false;
    }

    XrSystemGetInfo systemInfo{XR_TYPE_SYSTEM_GET_INFO};
    systemInfo.formFactor = XR_FORM_FACTOR_HEAD_MOUNTED_DISPLAY;
    result = xrGetSystem(g_instance, &systemInfo, &g_systemId);
    if (XR_FAILED(result)) {
        Log.error("Failed to get OpenXR system: {}", (int)result);
        return false;
    }

    return true;
}

void shutdown() {
    if (g_session != XR_NULL_HANDLE) {
        xrDestroySession(g_session);
        g_session = XR_NULL_HANDLE;
    }
    if (g_instance != XR_NULL_HANDLE) {
        xrDestroyInstance(g_instance);
        g_instance = XR_NULL_HANDLE;
    }
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
    
    PFN_xrGetVulkanGraphicsDeviceKHR xrGetVulkanGraphicsDeviceKHR;
    xrGetInstanceProcAddr(g_instance, "xrGetVulkanGraphicsDeviceKHR", (PFN_xrVoidFunction*)&xrGetVulkanGraphicsDeviceKHR);
    
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    xrGetVulkanGraphicsDeviceKHR(g_instance, g_systemId, instance, &physicalDevice);
    return physicalDevice;
}

bool initialize_session(VkInstance instance, VkPhysicalDevice physicalDevice, VkDevice device, uint32_t queueFamilyIndex, uint32_t queueIndex) {
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

    Log.info("OpenXR session created");
    return true;
}

} // namespace aurora::xr

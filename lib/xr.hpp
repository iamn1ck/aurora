#pragma once

#include <aurora/aurora.h>
#include <openxr/openxr.h>
#include <vector>
#include <string>

#define XR_USE_GRAPHICS_API_VULKAN
#include <vulkan/vulkan.h>
#include <openxr/openxr_platform.h>

namespace aurora::xr {

bool initialize_instance();
void shutdown();

bool is_enabled();

std::vector<std::string> get_vulkan_instance_extensions();
std::vector<std::string> get_vulkan_device_extensions();

VkPhysicalDevice get_vulkan_graphics_device(VkInstance instance);

bool initialize_session(VkInstance instance, VkPhysicalDevice physicalDevice, VkDevice device, uint32_t queueFamilyIndex, uint32_t queueIndex);

} // namespace aurora::xr

#pragma once

#include <aurora/aurora.h>
#include <openxr/openxr.h>
#include <vector>
#include <string>

#define XR_USE_GRAPHICS_API_VULKAN
#include <vulkan/vulkan.h>
#include <openxr/openxr_platform.h>
#include <webgpu/webgpu_cpp.h>

namespace aurora::xr {

bool initialize_instance();
void shutdown();

bool is_enabled();

std::vector<std::string> get_vulkan_instance_extensions();
std::vector<std::string> get_vulkan_device_extensions();

VkPhysicalDevice get_vulkan_graphics_device(VkInstance instance);

bool initialize_session(VkInstance instance, VkPhysicalDevice physicalDevice, VkDevice device, uint32_t queueFamilyIndex, uint32_t queueIndex);

bool begin_frame();
void end_frame();

wgpu::TextureView get_texture_view();
uint32_t get_width();
uint32_t get_height();
uint32_t get_eye_width();

struct XrEyeInfo {
    float poseX;
    float tanLeft;
    float tanRight;
};

bool get_eye_info(int eye, XrEyeInfo& out);
bool get_head_pose(float& ox, float& oy, float& oz, float& ow);

} // namespace aurora::xr

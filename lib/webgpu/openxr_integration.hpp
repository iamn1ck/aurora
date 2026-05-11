#pragma once

#include <aurora/aurora.h>
#include <webgpu/webgpu_cpp.h>

namespace aurora::openxr {

bool initialize();
void shutdown();

bool begin_frame();
void end_frame();

bool is_initialized();
wgpu::TextureView get_texture_view();
wgpu::TextureView get_menu_texture_view();
void copy_to_shared(wgpu::CommandEncoder encoder);
void copy_menu_to_shared(wgpu::CommandEncoder encoder);
uint32_t get_width();    // full side-by-side swapchain width (2 × per-eye)
uint32_t get_height();
uint32_t get_eye_width(); // per-eye half-width

// Per-eye information captured from the most recent xrLocateViews call.
struct XrEyeInfo {
    float poseX;    // eye X offset from head center in metres (left eye < 0, right > 0)
    float tanLeft;  // tan(|angleLeft|)  — positive, horizontal half-extent left of center
    float tanRight; // tan(angleRight)   — positive, horizontal half-extent right of center
    float tanUp;    // tan(angleUp)      — positive, vertical half-extent above center
    float tanDown;  // tan(|angleDown|)  — positive, vertical half-extent below center
};
// Returns false when OpenXR is not active or the pose has not yet been located.
bool get_eye_info(int eye, XrEyeInfo& out);

// Returns the HMD head orientation as a quaternion (x, y, z, w) from the
// most recent xrLocateViews call. Returns false if no valid pose is available.
bool get_head_pose(float& ox, float& oy, float& oz, float& ow);

} // namespace aurora::openxr

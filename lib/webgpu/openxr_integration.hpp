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
void copy_to_shared(wgpu::CommandEncoder encoder);
uint32_t get_width();
uint32_t get_height();

// Returns the HMD head orientation as a quaternion (x, y, z, w) from the
// most recent xrLocateViews call. Returns false if no valid pose is available.
bool get_head_pose(float& ox, float& oy, float& oz, float& ow);

} // namespace aurora::openxr

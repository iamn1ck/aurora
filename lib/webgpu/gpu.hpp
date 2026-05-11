#pragma once

#include <aurora/aurora.h>
#include <aurora/math.hpp>

#include "wgpu.hpp"

#include <array>
#include <cstdint>

struct SDL_Window;

namespace aurora::webgpu {
struct GraphicsConfig {
  wgpu::SurfaceConfiguration surfaceConfiguration;
  wgpu::TextureFormat depthFormat;
  uint32_t msaaSamples;
  uint16_t textureAnisotropy;
};
struct TextureWithSampler {
  wgpu::Texture texture;
  wgpu::TextureView view;
  wgpu::Extent3D size;
  wgpu::TextureFormat format;
  wgpu::Sampler sampler;
};
struct Viewport {
  float left;
  float top;
  float width;
  float height;
  float znear;
  float zfar;

  bool operator==(const Viewport& rhs) const {
    return left == rhs.left && top == rhs.top && width == rhs.width && height == rhs.height && znear == rhs.znear &&
           zfar == rhs.zfar;
  }
  bool operator!=(const Viewport& rhs) const { return !(*this == rhs); }
};

extern wgpu::Device g_device;
extern wgpu::Queue g_queue;
extern wgpu::Surface g_surface;
extern wgpu::BackendType g_backendType;
extern GraphicsConfig g_graphicsConfig;
extern TextureWithSampler g_frameBuffer;
extern TextureWithSampler g_frameBufferResolved;
extern TextureWithSampler g_depthBuffer;
extern wgpu::RenderPipeline g_CopyPipeline;
extern wgpu::BindGroup g_CopyBindGroup;
extern wgpu::Buffer g_CopyUniformBuffer;
extern wgpu::BindGroup g_xrEyeBindGroups[2]; // per-eye stereo blit bind groups
extern wgpu::Instance g_instance;

// Write per-eye UV horizontal offsets into the two eye-specific uniform buffers.
// Call this BEFORE the XR stereo render pass (each eye has its own buffer so
// writes are independent and do not overwrite each other at submit time).
void prepare_xr_stereo_uniforms(float leftOffsetX, float rightOffsetX);

// Update the default copy uniform (used for the desktop blit). Identity by default.
void update_copy_uniforms(float uvOffsetX, float uvOffsetY, float uvScaleX, float uvScaleY);

bool initialize(AuroraBackend backend);
void shutdown();
void release_surface() noexcept;
bool refresh_surface(bool recreate = true);
void resize_swapchain(uint32_t width, uint32_t height, uint32_t native_width, uint32_t native_height,
                      bool force = false);
TextureWithSampler create_render_texture(uint32_t width, uint32_t height, bool multisampled);
const TextureWithSampler& present_source() noexcept;
wgpu::BindGroup create_copy_bind_group(const TextureWithSampler& source);
Viewport calculate_present_viewport(uint32_t surface_width, uint32_t surface_height, uint32_t content_width,
                                    uint32_t content_height) noexcept;
void draw_clear(const wgpu::RenderPassEncoder& pass, bool clearColor, bool clearAlpha, bool clearDepth,
                const Vec4<float>& clearColorValue, float clearDepthValue);

size_t load_from_cache(void const* key, size_t keySize, void* value, size_t valueSize, void* userdata);
void store_to_cache(void const* key, size_t keySize, void const* value, size_t valueSize, void* userdata);
void cache_shutdown();

} // namespace aurora::webgpu

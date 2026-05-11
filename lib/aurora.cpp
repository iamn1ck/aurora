#include <aurora/aurora.h>

#ifdef AURORA_ENABLE_GX
#include "gfx/common.hpp"
#include "gx/fifo.hpp"
#include "imgui.hpp"
#include "webgpu/gpu.hpp"
#include "webgpu/openxr_integration.hpp"
#include <webgpu/webgpu_cpp.h>
#endif

#ifdef AURORA_ENABLE_RMLUI
#include "rmlui.hpp"
#endif

#include "input.hpp"
#include "internal.hpp"
#include "window.hpp"

#include <SDL3/SDL_filesystem.h>
#include <magic_enum.hpp>

#include "tracy/Tracy.hpp"

namespace aurora {
AuroraConfig g_config;
uint32_t g_sdlCustomEventsStart;
char g_gameName[4];

namespace {
Module Log("aurora");

#ifdef AURORA_ENABLE_GX
// GPU
using webgpu::g_device;
using webgpu::g_queue;
using webgpu::g_surface;
#endif

#ifdef AURORA_ENABLE_GX
constexpr std::array PreferredBackendOrder{
#ifdef ENABLE_BACKEND_WEBGPU
    BACKEND_WEBGPU,
#endif
#ifdef DAWN_ENABLE_BACKEND_D3D12
    BACKEND_D3D12,
#endif
#ifdef DAWN_ENABLE_BACKEND_METAL
    BACKEND_METAL,
#endif
#ifdef DAWN_ENABLE_BACKEND_VULKAN
    BACKEND_VULKAN,
#endif
#ifdef DAWN_ENABLE_BACKEND_D3D11
    BACKEND_D3D11,
#endif
// #ifdef DAWN_ENABLE_BACKEND_DESKTOP_GL
//     BACKEND_OPENGL,
// #endif
// #ifdef DAWN_ENABLE_BACKEND_OPENGLES
//     BACKEND_OPENGLES,
// #endif
#ifdef DAWN_ENABLE_BACKEND_NULL
    BACKEND_NULL,
#endif
};
#else
constexpr std::array<AuroraBackend, 0> PreferredBackendOrder{};
#endif

bool g_initialFrame = false;

AuroraInfo initialize(int argc, char* argv[], const AuroraConfig& config) noexcept {
  g_config = config;
  Log.info("Aurora initializing");
  if (g_config.appName == nullptr) {
    g_config.appName = "Aurora";
  } else {
    g_config.appName = strdup(g_config.appName);
  }
  if (g_config.configPath == nullptr) {
    g_config.configPath = SDL_GetPrefPath(nullptr, g_config.appName);
  } else {
    g_config.configPath = strdup(g_config.configPath);
  }
  if (g_config.msaa == 0) {
    g_config.msaa = 1;
  }
  if (g_config.maxTextureAnisotropy == 0) {
    g_config.maxTextureAnisotropy = 16;
  }
  ASSERT(window::initialize(), "Error initializing window");

  g_sdlCustomEventsStart = SDL_RegisterEvents(2);
  ASSERT(g_sdlCustomEventsStart, "Failed to allocate user events: {}", SDL_GetError());
  ASSERT(window::initialize_event_watch(), "Error initializing SDL event watch");

#ifdef AURORA_ENABLE_GX
  /* Attempt to create a window using the calling application's desired backend */
  AuroraBackend selectedBackend = config.desiredBackend;
  bool windowCreated = false;
  if (selectedBackend != BACKEND_AUTO && window::create_window(selectedBackend)) {
    if (webgpu::initialize(selectedBackend)) {
      windowCreated = true;
    } else {
      window::destroy_window();
    }
  }

  if (!windowCreated) {
    for (const auto backendType : PreferredBackendOrder) {
      selectedBackend = backendType;
      if (!window::create_window(selectedBackend)) {
        continue;
      }
      if (webgpu::initialize(selectedBackend)) {
        windowCreated = true;
        break;
      } else {
        window::destroy_window();
      }
    }
  }

  ASSERT(windowCreated, "Error creating window: {}", SDL_GetError());

  // Initialize SDL_Renderer for ImGui when we can't use a Dawn backend
  if (webgpu::g_backendType == wgpu::BackendType::Null) {
    ASSERT(window::create_renderer(), "Failed to initialize SDL renderer: {}", SDL_GetError());
  }
#else
  AuroraBackend selectedBackend = BACKEND_NULL;
  ASSERT(window::create_window(BACKEND_NULL), "Error creating window: {}", SDL_GetError());
  ASSERT(window::create_renderer(), "Failed to initialize SDL renderer: {}", SDL_GetError());
#endif

  window::show_window();

#ifdef AURORA_ENABLE_GX
  gfx::initialize();

  imgui::create_context();
#endif
  const auto size = window::get_window_size();
  Log.info("Using framebuffer size {}x{} scale {}", size.fb_width, size.fb_height, size.scale);
#ifdef AURORA_ENABLE_GX
  if (g_config.imGuiInitCallback != nullptr) {
    g_config.imGuiInitCallback(&size);
  }
  imgui::initialize();
#endif

#ifdef AURORA_ENABLE_RMLUI
  rmlui::initialize(size);
#endif

  g_initialFrame = true;
  g_config.desiredBackend = selectedBackend;
  return {
      .backend = selectedBackend,
      .configPath = g_config.configPath,
      .window = window::get_sdl_window(),
      .windowSize = size,
  };
}

#ifdef AURORA_ENABLE_GX
wgpu::TextureView g_currentView;
#endif

void shutdown() noexcept {
#ifdef AURORA_ENABLE_RMLUI
  rmlui::shutdown();
#endif
#ifdef AURORA_ENABLE_GX
  g_currentView = {};
  imgui::shutdown();
  gfx::shutdown();
  webgpu::shutdown();
#endif
  input::shutdown();
  window::shutdown();
}

const AuroraEvent* update() noexcept {
  ZoneScoped;
  if (g_initialFrame) {
    g_initialFrame = false;
    input::initialize();
  }
  return window::poll_events();
}

bool begin_frame() noexcept {
  ZoneScoped;
#ifdef AURORA_ENABLE_GX
  {
    window::SurfaceLock surfaceLock;
    if (!window::is_presentable()) {
      webgpu::release_surface();
      return false;
    }
    if (window::is_paused()) {
      return false;
    }
    if (!g_surface) {
      webgpu::refresh_surface(true);
      if (!g_surface) {
        return false;
      }
    }
    wgpu::SurfaceTexture surfaceTexture;
    g_surface.GetCurrentTexture(&surfaceTexture);
    switch (surfaceTexture.status) {
    case wgpu::SurfaceGetCurrentTextureStatus::SuccessOptimal:
      g_currentView = surfaceTexture.texture.CreateView();
      break;
    case wgpu::SurfaceGetCurrentTextureStatus::Timeout:
      Log.warn("Surface texture acquisition timed out");
      return false;
    case wgpu::SurfaceGetCurrentTextureStatus::SuccessSuboptimal:
    case wgpu::SurfaceGetCurrentTextureStatus::Outdated:
      Log.info("Surface texture is {}, reconfiguring swapchain", magic_enum::enum_name(surfaceTexture.status));
      webgpu::refresh_surface(false);
      return false;
    case wgpu::SurfaceGetCurrentTextureStatus::Lost:
      Log.warn("Surface texture is {}, releasing surface", magic_enum::enum_name(surfaceTexture.status));
      webgpu::release_surface();
    case wgpu::SurfaceGetCurrentTextureStatus::Error:
      Log.warn("Surface texture is {}, dropping surface", magic_enum::enum_name(surfaceTexture.status));
      g_surface = {};
      return false;
    default:
      Log.error("Failed to get surface texture: {}", magic_enum::enum_name(surfaceTexture.status));
      return false;
    }
  }

  openxr::begin_frame();

  imgui::new_frame(window::get_window_size());
  if (!gfx::begin_frame()) {
    g_currentView = {};
    return false;
  }
#endif
  return true;
}

void end_frame() noexcept {
  ZoneScoped;
#ifdef AURORA_ENABLE_GX
  gx::fifo::drain();
  const auto encoderDescriptor = wgpu::CommandEncoderDescriptor{
      .label = "Redraw encoder",
  };
  auto encoder = g_device.CreateCommandEncoder(&encoderDescriptor);
  gfx::end_frame(encoder);
  gfx::render(encoder);
    const auto& presentSource = webgpu::present_source();
    auto desktopViewport = webgpu::calculate_present_viewport(webgpu::g_graphicsConfig.surfaceConfiguration.width,
                                                       webgpu::g_graphicsConfig.surfaceConfiguration.height,
                                                       presentSource.size.width, presentSource.size.height);
    wgpu::BindGroup presentBindGroup = webgpu::g_CopyBindGroup;
  #if AURORA_ENABLE_RMLUI
    if (rmlui::is_initialized()) {
      const auto rmlOutput = rmlui::render(encoder, desktopViewport);
      if (rmlOutput.texture != nullptr) {
        presentBindGroup = rmlOutput.copyBindGroup;
      }
    }
  #endif

    if (openxr::is_initialized()) {
      wgpu::TextureView xrView = openxr::get_texture_view();
      if (xrView) {
        // The side-by-side XR Dawn texture is the render target for both passes.
        // We use a single render-pass with Load so the second eye doesn't clear the first.
        const std::array xrAttachments{
            wgpu::RenderPassColorAttachment{
                .view = xrView,
                .loadOp = wgpu::LoadOp::Clear,
                .storeOp = wgpu::StoreOp::Store,
            },
        };
        const wgpu::RenderPassDescriptor xrPassDesc{
            .label = "VR EFB stereo blit",
            .colorAttachmentCount = xrAttachments.size(),
            .colorAttachments = xrAttachments.data(),
        };
        const auto xrPass = encoder.BeginRenderPass(&xrPassDesc);
        xrPass.SetPipeline(webgpu::g_CopyPipeline);
        xrPass.SetBindGroup(0, presentBindGroup, 0, nullptr);

        const float eyeW = static_cast<float>(openxr::get_eye_width());
        const float eyeH = static_cast<float>(openxr::get_height());

        for (int eye = 0; eye < 2; ++eye) {
          // Compute the UV horizontal shift for this eye.
          // poseX is the eye's offset from head centre in metres.
          // The horizontal FOV half-extents in tangent space give the total
          // angular width of the view; dividing poseX by fovWidth gives the
          // fraction of the screen to shift.  Negate so a left eye (negative X)
          // shifts the sampled region to the right (showing more of the left).
          openxr::XrEyeInfo ei{};
          float uvOffsetX = 0.0f;
          if (openxr::get_eye_info(eye, ei)) {
            const float fovWidth = ei.tanLeft + ei.tanRight;
            if (fovWidth > 0.0f) {
              uvOffsetX = -ei.poseX / fovWidth;
            }
          }
          webgpu::update_copy_uniforms(uvOffsetX, 0.0f, 1.0f, 1.0f);

          xrPass.SetViewport(
              static_cast<float>(eye) * eyeW, 0.0f,
              eyeW, eyeH,
              0.0f, 1.0f);
          xrPass.Draw(3);
        }
        xrPass.End();

        // Restore identity uniforms so the desktop blit below is unaffected.
        webgpu::update_copy_uniforms(0.0f, 0.0f, 1.0f, 1.0f);

        openxr::copy_to_shared(encoder);
      }
    }

  {
    window::SurfaceLock surfaceLock;
    if (window::is_presentable() && g_surface && g_currentView) {
      {
        const std::array attachments{
            wgpu::RenderPassColorAttachment{
                .view = g_currentView,
                .loadOp = wgpu::LoadOp::Clear,
                .storeOp = wgpu::StoreOp::Store,
            },
        };
        const wgpu::RenderPassDescriptor renderPassDescriptor{
            .label = "EFB copy render pass",
            .colorAttachmentCount = attachments.size(),
            .colorAttachments = attachments.data(),
        };
        const auto pass = encoder.BeginRenderPass(&renderPassDescriptor);
        // Copy EFB -> XFB (swapchain)
        pass.SetPipeline(webgpu::g_CopyPipeline);
        pass.SetBindGroup(0, presentBindGroup, 0, nullptr);
        pass.SetViewport(desktopViewport.left, desktopViewport.top, desktopViewport.width, desktopViewport.height, desktopViewport.znear, desktopViewport.zfar);

        pass.Draw(3);
        pass.End();
      }
      {
        const std::array attachments{
            wgpu::RenderPassColorAttachment{
                .view = g_currentView,
                .loadOp = wgpu::LoadOp::Load,
                .storeOp = wgpu::StoreOp::Store,
            },
        };
        const wgpu::RenderPassDescriptor renderPassDescriptor{
            .label = "ImGui render pass",
            .colorAttachmentCount = attachments.size(),
            .colorAttachments = attachments.data(),
        };
        const auto pass = encoder.BeginRenderPass(&renderPassDescriptor);
        pass.SetViewport(0.f, 0.f, static_cast<float>(webgpu::g_graphicsConfig.surfaceConfiguration.width),
                         static_cast<float>(webgpu::g_graphicsConfig.surfaceConfiguration.height), 0.f, 1.f);
        imgui::render(pass);
        pass.End();
      }
    } else {
      Log.info("Skipping present; window not presentable");
      webgpu::release_surface();
    }
    const wgpu::CommandBufferDescriptor cmdBufDescriptor{.label = "Redraw command buffer"};
    const auto buffer = encoder.Finish(&cmdBufDescriptor);
    g_queue.Submit(1, &buffer);
    gfx::after_submit();
    if (window::is_presentable() && g_surface) {
      auto presentStatus = g_surface.Present();
      if (presentStatus != wgpu::Status::Success) {
        Log.warn("Surface present failed: {}", static_cast<int>(presentStatus));
        webgpu::release_surface();
      }
    } else if (g_surface) {
      webgpu::release_surface();
    }
    g_currentView = {};
  }
  
  openxr::end_frame();

  TracyPlotConfig("aurora: lastVertSize", tracy::PlotFormatType::Memory, false, true, 0);
  TracyPlotConfig("aurora: lastUniformSize", tracy::PlotFormatType::Memory, false, true, 0);
  TracyPlotConfig("aurora: lastIndexSize", tracy::PlotFormatType::Memory, false, true, 0);
  TracyPlotConfig("aurora: lastStorageSize", tracy::PlotFormatType::Memory, false, true, 0);
  TracyPlotConfig("aurora: lastTextureUploadSize", tracy::PlotFormatType::Memory, false, true, 0);

  TracyPlot("aurora: queuedPipelines", static_cast<int64_t>(gfx::g_stats.queuedPipelines));
  TracyPlot("aurora: createdPipelines", static_cast<int64_t>(gfx::g_stats.createdPipelines));
  TracyPlot("aurora: drawCallCount", static_cast<int64_t>(gfx::g_stats.drawCallCount));
  TracyPlot("aurora: mergedDrawCallCount", static_cast<int64_t>(gfx::g_stats.mergedDrawCallCount));
  TracyPlot("aurora: lastVertSize", static_cast<int64_t>(gfx::g_stats.lastVertSize));
  TracyPlot("aurora: lastUniformSize", static_cast<int64_t>(gfx::g_stats.lastUniformSize));
  TracyPlot("aurora: lastIndexSize", static_cast<int64_t>(gfx::g_stats.lastIndexSize));
  TracyPlot("aurora: lastStorageSize", static_cast<int64_t>(gfx::g_stats.lastStorageSize));
  TracyPlot("aurora: lastTextureUploadSize", static_cast<int64_t>(gfx::g_stats.lastTextureUploadSize));

#endif
}
} // namespace
} // namespace aurora

// C API bindings
AuroraInfo aurora_initialize(int argc, char* argv[], const AuroraConfig* config) {
  return aurora::initialize(argc, argv, *config);
}
void aurora_shutdown() { aurora::shutdown(); }
const AuroraEvent* aurora_update() { return aurora::update(); }
bool aurora_begin_frame() { return aurora::begin_frame(); }
void aurora_end_frame() { aurora::end_frame(); }
AuroraBackend aurora_get_backend() { return aurora::g_config.desiredBackend; }
const AuroraBackend* aurora_get_available_backends(size_t* count) {
  if (count != nullptr) {
    *count = aurora::PreferredBackendOrder.size();
  }
  return aurora::PreferredBackendOrder.data();
}
void aurora_set_log_level(AuroraLogLevel level) { aurora::g_config.logLevel = level; }
void aurora_set_pause_on_focus_lost(bool value) { aurora::g_config.pauseOnFocusLost = value; }
void aurora_set_background_input(bool value) {
  aurora::g_config.allowJoystickBackgroundEvents = value;
  aurora::window::set_background_input(value);
}

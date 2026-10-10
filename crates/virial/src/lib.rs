//! Virial: GPU physics simulation in Rust and WebGPU.
//!
//! A simulation implements [`Scene`] and hands it to [`run`]. Virial opens a
//! window (or a canvas in the browser), sets up the GPU, and calls the scene once per
//! frame with a command encoder and the frame's colour target.

use std::sync::Arc;

use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, EventLoop, EventLoopProxy};
use winit::window::{Window, WindowId};

pub use virial_gpu::{Gpu, GpuError, wgpu};
pub use winit;

/// Settings for [`run`].
pub struct Config {
    pub title: &'static str,
}

impl Default for Config {
    fn default() -> Self {
        Self { title: "Virial" }
    }
}

/// What a frame hands the scene to draw with.
pub struct Frame<'a> {
    pub encoder: &'a mut wgpu::CommandEncoder,
    /// The colour target for this frame, already the window's size.
    pub view: &'a wgpu::TextureView,
}

/// A simulation that Virial runs and draws.
pub trait Scene: 'static {
    /// Called once the GPU is ready.
    fn new(gpu: &Gpu) -> Self
    where
        Self: Sized;

    /// Record this frame's GPU work into `frame.encoder`.
    fn render(&mut self, gpu: &Gpu, frame: Frame<'_>);
}

/// Opens the window, starts the GPU and runs `G` until the window closes.
/// In the browser the canvas is appended to `<body>` and this returns at once.
pub fn run<G: Scene>(config: Config) {
    init_logging();

    let event_loop = EventLoop::<GpuReady>::with_user_event()
        .build()
        .expect("could not create the event loop");
    let app = App::<G> {
        config,
        proxy: event_loop.create_proxy(),
        window: None,
        running: None,
    };

    #[cfg(not(target_arch = "wasm32"))]
    {
        let mut app = app;
        event_loop.run_app(&mut app).expect("event loop failed");
    }
    #[cfg(target_arch = "wasm32")]
    {
        use winit::platform::web::EventLoopExtWebSys;
        event_loop.spawn_app(app);
    }
}

/// Sent to the event loop when GPU setup finishes. Setup is async because the
/// browser cannot block waiting for the GPU.
struct GpuReady(Result<Gpu, GpuError>);

struct Running<G> {
    gpu: Gpu,
    scene: G,
}

struct App<G> {
    config: Config,
    proxy: EventLoopProxy<GpuReady>,
    window: Option<Arc<Window>>,
    running: Option<Running<G>>,
}

impl<G: Scene> ApplicationHandler<GpuReady> for App<G> {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }
        let attributes = Window::default_attributes().with_title(self.config.title);
        #[cfg(target_arch = "wasm32")]
        let attributes = {
            use winit::platform::web::WindowAttributesExtWebSys;
            attributes.with_append(true)
        };
        let window = Arc::new(
            event_loop
                .create_window(attributes)
                .expect("could not create the window"),
        );
        self.window = Some(window.clone());

        let size = window.inner_size();
        let proxy = self.proxy.clone();
        let setup = async move {
            let gpu = Gpu::new(window, size.width, size.height).await;
            // Fails only if the event loop has already exited
            let _ = proxy.send_event(GpuReady(gpu));
        };
        #[cfg(not(target_arch = "wasm32"))]
        pollster::block_on(setup);
        #[cfg(target_arch = "wasm32")]
        wasm_bindgen_futures::spawn_local(setup);
    }

    fn user_event(&mut self, event_loop: &ActiveEventLoop, GpuReady(gpu): GpuReady) {
        match gpu {
            Ok(mut gpu) => {
                // The window may have been resized while the GPU was starting
                if let Some(window) = &self.window {
                    let size = window.inner_size();
                    gpu.resize(size.width, size.height);
                    window.request_redraw();
                }
                let scene = G::new(&gpu);
                self.running = Some(Running { gpu, scene });
            }
            Err(error) => {
                show_error(&error);
                event_loop.exit();
            }
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => {
                if let Some(running) = &mut self.running {
                    running.gpu.resize(size.width, size.height);
                }
            }
            WindowEvent::RedrawRequested => {
                if let Some(running) = &mut self.running {
                    running.draw();
                }
                if let Some(window) = &self.window {
                    window.request_redraw();
                }
            }
            _ => {}
        }
    }
}

impl<G: Scene> Running<G> {
    fn draw(&mut self) {
        let gpu = &self.gpu;
        let (texture, suboptimal) = match gpu.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(texture) => (texture, false),
            wgpu::CurrentSurfaceTexture::Suboptimal(texture) => (texture, true),
            wgpu::CurrentSurfaceTexture::Outdated => {
                gpu.surface.configure(&gpu.device, &gpu.config);
                return;
            }
            wgpu::CurrentSurfaceTexture::Timeout | wgpu::CurrentSurfaceTexture::Occluded => return,
            other => {
                log::warn!("skipped a frame: {other:?}");
                return;
            }
        };
        let view = texture
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("frame"),
            });
        self.scene.render(
            gpu,
            Frame {
                encoder: &mut encoder,
                view: &view,
            },
        );
        gpu.queue.submit([encoder.finish()]);
        gpu.queue.present(texture);
        if suboptimal {
            gpu.surface.configure(&gpu.device, &gpu.config);
        }
    }
}

fn init_logging() {
    #[cfg(not(target_arch = "wasm32"))]
    {
        let _ = env_logger::Builder::from_env(
            env_logger::Env::default().default_filter_or("info,wgpu_core=warn,wgpu_hal=warn"),
        )
        .try_init();
    }
    #[cfg(target_arch = "wasm32")]
    {
        console_error_panic_hook::set_once();
        let _ = console_log::init_with_level(log::Level::Info);
    }
}

/// Tells the player why the scene can't start: in the page in the browser, on
/// stderr natively.
fn show_error(error: &GpuError) {
    log::error!("{error}");
    #[cfg(target_arch = "wasm32")]
    if let Some(document) = web_sys::window().and_then(|w| w.document())
        && let Some(body) = document.body()
    {
        let message = document
            .create_element("p")
            .expect("could not create an element");
        message.set_class_name("engine-error");
        message.set_text_content(Some(&error.to_string()));
        let _ = body.append_child(&message);
    }
}

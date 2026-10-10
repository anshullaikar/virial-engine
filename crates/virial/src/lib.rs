//! Virial: GPU physics simulation in Rust and WebGPU.
//!
//! A simulation implements [`Scene`] and hands it to [`run`]. Virial opens a
//! window (or a canvas in the browser), sets up the GPU, and each frame calls
//! [`Scene::update`] then [`Scene::render`].

use std::sync::Arc;

use web_time::Instant;
use winit::application::ApplicationHandler;
use winit::event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop, EventLoopProxy};
use winit::keyboard::PhysicalKey;
use winit::window::{Window, WindowId};

mod args;
pub mod data;
mod input;
#[cfg(not(target_arch = "wasm32"))]
pub mod record;
mod stats;

pub use args::{arg, args};
pub use data::DataLog;
pub use input::Input;
pub use stats::Stats;
pub use virial_gpu::{Gpu, GpuError, GpuTimer, wgpu};
pub use winit;
pub use winit::keyboard::KeyCode;

/// Settings for [`run`].
pub struct Config {
    pub title: &'static str,
}

impl Default for Config {
    fn default() -> Self {
        Self { title: "Virial" }
    }
}

/// Wall-clock time. Simulations keep their own simulation clock; this is only
/// for pacing and display.
#[derive(Clone, Copy, Debug)]
pub struct Time {
    /// Seconds since the previous frame, capped at 0.1 s.
    pub dt: f32,
    /// Seconds since the scene started.
    pub elapsed: f64,
    /// Frames drawn so far.
    pub frame: u64,
}

/// What a scene gets each frame.
pub struct Context<'a> {
    pub gpu: &'a Gpu,
    pub input: &'a Input,
    /// Wall-clock time in the live preview; the video clock when recording
    pub time: Time,
    pub stats: &'a mut Stats,
    /// Measurements, snapshots and run metadata
    pub data: &'a mut DataLog,
}

/// What a frame hands the scene to draw with.
pub struct Frame<'a> {
    pub encoder: &'a mut wgpu::CommandEncoder,
    /// The colour target for this frame, already the window's size.
    pub view: &'a wgpu::TextureView,
    /// Times GPU passes when the device supports it. Pass
    /// `timer.and_then(|t| t.compute("label"))` as a pass's timestamp writes.
    pub timer: Option<&'a GpuTimer>,
}

/// A simulation that Virial runs and draws.
pub trait Scene: 'static {
    /// Called once the GPU is ready.
    fn new(gpu: &Gpu) -> Self
    where
        Self: Sized;

    /// CPU work for this frame: read input, write uniforms.
    fn update(&mut self, _ctx: &mut Context<'_>) {}

    /// Record this frame's GPU work into `frame.encoder`.
    fn render(&mut self, ctx: &mut Context<'_>, frame: Frame<'_>);

    /// Called once after the last frame of a recording: log final
    /// measurements and snapshots here.
    fn finish(&mut self, _ctx: &mut Context<'_>) {}
}

/// Opens the window, starts the GPU and runs `S` until the window closes.
/// In the browser the canvas is appended to `<body>` and this returns at once.
///
/// Natively, the setting `record=<dir>` renders offline instead (see
/// [`record`]): `width`, `height`, `fps`, `seconds`, `video=0`.
pub fn run<S: Scene>(config: Config) {
    init_logging();

    #[cfg(not(target_arch = "wasm32"))]
    if let Some(dir) = args().get("record").filter(|d| !d.is_empty()) {
        record::record::<S>(config.title, &record::RecordSettings::from_args(dir));
        return;
    }

    let event_loop = EventLoop::<GpuReady>::with_user_event()
        .build()
        .expect("could not create the event loop");
    let app = App::<S> {
        stats: Stats::new(config.title),
        data: DataLog::preview(),
        config,
        proxy: event_loop.create_proxy(),
        window: None,
        running: None,
        input: Input::default(),
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

struct Running<S> {
    gpu: Gpu,
    scene: S,
    timer: Option<GpuTimer>,
    start: Instant,
    last: Instant,
    frame: u64,
}

struct App<S> {
    config: Config,
    proxy: EventLoopProxy<GpuReady>,
    window: Option<Arc<Window>>,
    running: Option<Running<S>>,
    input: Input,
    stats: Stats,
    data: DataLog,
}

impl<S: Scene> ApplicationHandler<GpuReady> for App<S> {
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
                    self.input.set_surface_size(size.width, size.height);
                    window.request_redraw();
                }
                let timer = gpu
                    .has_timestamps()
                    .then(|| GpuTimer::new(&gpu.device, &gpu.queue));
                let scene = S::new(&gpu);
                let now = Instant::now();
                self.running = Some(Running {
                    gpu,
                    scene,
                    timer,
                    start: now,
                    last: now,
                    frame: 0,
                });
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
                self.input.set_surface_size(size.width, size.height);
                if let Some(running) = &mut self.running {
                    running.gpu.resize(size.width, size.height);
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                self.input.cursor_moved(position.x, position.y)
            }
            WindowEvent::CursorLeft { .. } => self.input.cursor_left(),
            WindowEvent::MouseInput { state, button, .. } => {
                let down = state == ElementState::Pressed;
                match button {
                    MouseButton::Left => self.input.left = down,
                    MouseButton::Right => self.input.right = down,
                    _ => {}
                }
            }
            WindowEvent::MouseWheel { delta, .. } => {
                self.input.scroll += match delta {
                    MouseScrollDelta::LineDelta(_, y) => y,
                    MouseScrollDelta::PixelDelta(p) => p.y as f32 / 100.0,
                };
            }
            WindowEvent::KeyboardInput { event, .. } => {
                if let PhysicalKey::Code(code) = event.physical_key {
                    self.input
                        .key(code, event.state == ElementState::Pressed, event.repeat);
                }
            }
            WindowEvent::RedrawRequested => {
                if let Some(running) = &mut self.running {
                    running.draw(&self.input, &mut self.stats, &mut self.data);
                    if let Some(window) = &self.window {
                        self.stats.show(window);
                    }
                }
                self.input.end_frame();
                if let Some(window) = &self.window {
                    window.request_redraw();
                }
            }
            _ => {}
        }
    }
}

impl<S: Scene> Running<S> {
    fn draw(&mut self, input: &Input, stats: &mut Stats, data: &mut DataLog) {
        // Native backends deliver buffer-map callbacks here; on the web the
        // browser delivers them itself
        let _ = self.gpu.device.poll(wgpu::PollType::Poll);
        if let Some(timer) = &mut self.timer {
            timer.begin_frame();
            stats.set_gpu_timings(timer.results());
        }

        let now = Instant::now();
        let time = Time {
            dt: now.duration_since(self.last).as_secs_f32().min(0.1),
            elapsed: now.duration_since(self.start).as_secs_f64(),
            frame: self.frame,
        };
        self.last = now;
        self.frame += 1;
        // The first frame's interval includes start-up, so it isn't a frame time
        if time.frame > 0 {
            stats.frame(time.dt);
        }

        let mut ctx = Context {
            gpu: &self.gpu,
            input,
            time,
            stats,
            data,
        };
        self.scene.update(&mut ctx);

        let gpu = &self.gpu;
        let surface = gpu.surface.as_ref().expect("the live preview has a window");
        let (texture, suboptimal) = match surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(texture) => (texture, false),
            wgpu::CurrentSurfaceTexture::Suboptimal(texture) => (texture, true),
            wgpu::CurrentSurfaceTexture::Outdated => {
                surface.configure(&gpu.device, &gpu.config);
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
            &mut ctx,
            Frame {
                encoder: &mut encoder,
                view: &view,
                timer: self.timer.as_ref(),
            },
        );
        if let Some(timer) = &mut self.timer {
            timer.resolve(&mut encoder);
        }
        gpu.queue.submit([encoder.finish()]);
        if let Some(timer) = &mut self.timer {
            timer.after_submit();
        }
        gpu.queue.present(texture);
        if suboptimal {
            surface.configure(&gpu.device, &gpu.config);
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

/// Tells the user why the simulation can't start: in the page in the browser,
/// on stderr natively.
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

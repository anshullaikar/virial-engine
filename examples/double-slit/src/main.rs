//! The double slit, from the Schrodinger equation (hbar = m = 1). A wave
//! packet with wavelength 1.26 hits a wall with two slits 1 wide and 4
//! apart; what gets through spreads from each slit and interferes. A screen
//! at x = 20 records the probability current through it over time: the
//! bars on the right. Ticks beside them mark where the dark fringes must
//! be, from the path difference between the slits (half-integer numbers of
//! wavelengths). Waves reaching the box edges are absorbed.
//!
//! Settings: `color=phase` (hue shows the phase), `n` (grid side, default
//! 1024 natively and 512 in the browser), `steps` (steps of 0.005
//! per frame, default 2).
//! Controls: Space to pause, R to restart.
//!
//! Render a video natively:
//! `cargo run --release -p double-slit -- record=renders/double-slit seconds=19`

use std::f64::consts::TAU;
use std::sync::OnceLock;

use bytemuck::{Pod, Zeroable};
use virial::{Config, Context, Frame, Gpu, KeyCode, Scene, arg, wgpu};
use virial_quantum::{Absorber, Measurement, Settings, Time, Wave};
use virial_render::{COLORMAPS_WGSL, Colormap, Overlay};
use wgpu::util::DeviceExt;

/// Grid cells per side: the setting `n` (a power of two from 256 to 2048).
/// 1024 natively, 512 in the browser so it keeps a smooth frame rate.
fn grid() -> u32 {
    static N: OnceLock<u32> = OnceLock::new();
    *N.get_or_init(|| {
        let default = if cfg!(target_arch = "wasm32") {
            512
        } else {
            1024
        };
        arg::<u32>("n", default)
            .clamp(256, 2048)
            .next_power_of_two()
    })
}
const LENGTH: f64 = 80.0;
const DT: f64 = 0.005;
const K0: f64 = 5.0;
const WALL: f64 = 0.5;
const BARRIER: f64 = 100.0;
const SEPARATION: f64 = 4.0;
const SLIT: f64 = 1.0;
const SCREEN: f64 = 20.0;
/// Starting packet: centre x and spreads along x and y. Far enough from the
/// wall that its tail inside the wall (amplitude 5e-6) sheds no visible
/// fast waves.
const START: f64 = -14.0;
const SPREAD: [f64; 2] = [2.0, 6.0];
/// Region shown: x from VIEW_X, y from -VIEW_SIDE/2, VIEW_SIDE across
const VIEW_X: f64 = -22.0;
const VIEW_SIDE: f64 = 50.0;

fn settings() -> Settings {
    Settings {
        n: grid(),
        length: LENGTH,
        dt: DT,
        time: Time::Real,
        absorber: Some(Absorber {
            width: 8.0,
            strength: 10.0,
        }),
        detector: Some(SCREEN),
    }
}

fn potential(x: f64, y: f64) -> f64 {
    let in_slit = (y.abs() - SEPARATION / 2.0).abs() < SLIT / 2.0;
    if x.abs() < WALL / 2.0 && !in_slit {
        BARRIER
    } else {
        0.0
    }
}

fn packet(x: f64, y: f64) -> [f64; 2] {
    let (dx, [sx, sy]) = (x - START, SPREAD);
    let envelope = (-dx * dx / (4.0 * sx * sx) - y * y / (4.0 * sy * sy)).exp();
    [envelope * (K0 * x).cos(), envelope * (K0 * x).sin()]
}

/// y > 0 where the path difference from the two slit centres to the screen
/// is `m` wavelengths
fn fringe(m: f64, screen: f64) -> f64 {
    let lambda = TAU / K0;
    let path = |y: f64| {
        let r = |s: f64| ((y + s).powi(2) + screen * screen).sqrt();
        r(SEPARATION / 2.0) - r(-SEPARATION / 2.0)
    };
    let (mut lo, mut hi) = (0.0, 100.0);
    for _ in 0..100 {
        let mid = 0.5 * (lo + hi);
        if path(mid) < m * lambda {
            lo = mid
        } else {
            hi = mid
        }
    }
    0.5 * (lo + hi)
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct View {
    image: [f32; 4],
    graph: [f32; 4],
    domain: [f32; 4],
    misc: [f32; 4],
    n: u32,
    mode: u32,
    _pad: [u32; 2],
}

struct DoubleSlit {
    wave: Wave,
    potential: wgpu::Buffer,
    peak: wgpu::Buffer,
    view: wgpu::Buffer,
    draw: wgpu::RenderPipeline,
    find_peak: wgpu::ComputePipeline,
    draw_group: wgpu::BindGroup,
    peak_group: wgpu::BindGroup,
    overlay: Overlay,
    phase: bool,
    steps_per_frame: u32,
    paused: bool,
    frames: u64,
    latest: Option<Measurement>,
}

impl DoubleSlit {
    fn groups(
        device: &wgpu::Device,
        draw: &wgpu::RenderPipeline,
        find_peak: &wgpu::ComputePipeline,
        wave: &Wave,
        view: &wgpu::Buffer,
        potential: &wgpu::Buffer,
        peak: &wgpu::Buffer,
    ) -> (wgpu::BindGroup, wgpu::BindGroup) {
        let entry = |binding, resource| wgpu::BindGroupEntry { binding, resource };
        let draw_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("field"),
            layout: &draw.get_bind_group_layout(0),
            entries: &[
                entry(0, view.as_entire_binding()),
                entry(1, wave.psi()),
                entry(2, potential.as_entire_binding()),
                entry(3, wave.detector()),
                entry(4, peak.as_entire_binding()),
            ],
        });
        let peak_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("screen peak"),
            layout: &find_peak.get_bind_group_layout(0),
            entries: &[
                entry(0, view.as_entire_binding()),
                entry(3, wave.detector()),
                entry(5, peak.as_entire_binding()),
            ],
        });
        (draw_group, peak_group)
    }

    fn start(&mut self, gpu: &Gpu) {
        self.wave = Wave::new(&gpu.device, &gpu.queue, settings(), packet, potential);
        (self.draw_group, self.peak_group) = Self::groups(
            &gpu.device,
            &self.draw,
            &self.find_peak,
            &self.wave,
            &self.view,
            &self.potential,
            &self.peak,
        );
        self.latest = None;
    }

    fn readouts(&self) -> Vec<String> {
        let mut lines = vec![
            format!(
                "Double slit: wavelength {:.3}, slits {SLIT} wide, {SEPARATION} apart",
                TAU / K0
            ),
            format!(
                "t = {:.2}   (hbar = m = 1, {}^2 grid)",
                self.wave.time(),
                grid()
            ),
        ];
        if let Some(m) = self.latest {
            // Energy per unit probability means nothing once it has all gone
            lines.push(if m.norm > 1e-3 {
                format!(
                    "Probability in the box {:.4}   energy {:.4}",
                    m.norm,
                    m.energy()
                )
            } else {
                format!("Probability in the box {:.4}", m.norm)
            });
        }
        lines
    }

    /// Image and graph rectangles for a frame size
    fn layout(size: (u32, u32)) -> ([f32; 4], [f32; 4], f32) {
        let (w, h) = (size.0 as f32, size.1 as f32);
        let scale = (h / 540.0).round().max(1.0);
        let side = h.min(w * 0.7);
        let left = ((w - side * 1.3) * 0.5).max(0.0);
        let image = [left, (h - side) * 0.5, side, 0.0];
        let graph = [left + side + 8.0 * scale, image[1], side * 0.26, side];
        (image, graph, scale)
    }
}

impl Scene for DoubleSlit {
    fn new(gpu: &Gpu) -> Self {
        let device = &gpu.device;
        let wave = Wave::new(device, &gpu.queue, settings(), packet, potential);
        let dx = LENGTH / grid() as f64;
        let walls: Vec<f32> = (0..grid() * grid())
            .map(|c| {
                let (i, j) = ((c % grid()) as f64, (c / grid()) as f64);
                let half = (grid() / 2) as f64;
                potential((i - half) * dx, (j - half) * dx) as f32
            })
            .collect();
        let potential = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("walls"),
            contents: bytemuck::cast_slice(&walls),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let peak = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("screen peak"),
            size: 16,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let view = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("field view"),
            size: size_of::<View>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("field"),
            source: wgpu::ShaderSource::Wgsl(
                format!("{COLORMAPS_WGSL}\n{}", include_str!("field.wgsl")).into(),
            ),
        });
        let draw = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("field"),
            layout: None,
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: Some("vs"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: Some("fs"),
                targets: &[Some(gpu.surface_format().into())],
                compilation_options: Default::default(),
            }),
            primitive: Default::default(),
            depth_stencil: None,
            multisample: Default::default(),
            multiview_mask: None,
            cache: None,
        });
        let find_peak = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("screen peak"),
            layout: None,
            module: &module,
            entry_point: Some("find_peak"),
            compilation_options: Default::default(),
            cache: None,
        });
        let (draw_group, peak_group) =
            Self::groups(device, &draw, &find_peak, &wave, &view, &potential, &peak);
        Self {
            wave,
            potential,
            peak,
            view,
            draw,
            find_peak,
            draw_group,
            peak_group,
            overlay: Overlay::new(device, gpu.surface_format()),
            phase: arg("color", String::from("density")) == "phase",
            steps_per_frame: arg("steps", 2u32).clamp(1, 100),
            paused: false,
            frames: 0,
            latest: None,
        }
    }

    fn update(&mut self, ctx: &mut Context<'_>) {
        if self.frames == 0 && ctx.data.is_recording() {
            let data = &mut *ctx.data;
            data.meta("grid", grid());
            data.meta("box_length", LENGTH);
            data.meta("dt", DT);
            data.meta("steps_per_frame", self.steps_per_frame);
            data.meta("wavenumber", K0);
            data.meta("slit_width", SLIT);
            data.meta("slit_separation", SEPARATION);
            data.meta("wall_thickness", WALL);
            data.meta("wall_potential", BARRIER);
            data.meta("screen_x", self.wave.detector_x().unwrap_or(SCREEN));
        }
        if ctx.input.pressed(KeyCode::Space) {
            self.paused = !self.paused;
        }
        if ctx.input.pressed(KeyCode::KeyR) {
            self.start(ctx.gpu);
        }
        self.wave.after_submit();
        for m in self.wave.poll() {
            self.latest = Some(m);
            ctx.data.row(&[
                ("step", m.step as f64),
                ("time", m.time),
                ("probability", m.norm),
                ("energy", m.energy()),
                ("mean_x", m.mean[0]),
                ("mean_y", m.mean[1]),
                ("in_absorber", m.in_absorber),
            ]);
        }
        for (label, line) in ["Scene", "Clock", "Measured"].iter().zip(self.readouts()) {
            ctx.stats.set(label, line);
        }
    }

    fn render(&mut self, ctx: &mut Context<'_>, frame: Frame<'_>) {
        if !self.paused {
            self.wave.step(frame.encoder, self.steps_per_frame);
        }
        if self.frames.is_multiple_of(5) {
            self.wave.measure(frame.encoder);
        }
        self.frames += 1;

        let gpu = ctx.gpu;
        let size = (gpu.config.width, gpu.config.height);
        let (image, graph, scale) = Self::layout(size);
        // Full brightness at 6% of the starting peak density, so the weak
        // waves past the slits show
        let reference = 0.06 / (TAU * SPREAD[0] * SPREAD[1]);
        let view = View {
            image,
            graph,
            domain: [
                VIEW_X as f32,
                (VIEW_SIDE / 2.0) as f32,
                VIEW_SIDE as f32,
                (LENGTH / grid() as f64) as f32,
            ],
            misc: [
                self.wave.detector_x().unwrap_or(SCREEN) as f32,
                reference as f32,
                2e-4,
                0.012,
            ],
            n: grid(),
            mode: self.phase as u32,
            _pad: [0; 2],
        };
        gpu.queue
            .write_buffer(&self.view, 0, bytemuck::bytes_of(&view));
        {
            let mut pass = frame.encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.find_peak);
            pass.set_bind_group(0, &self.peak_group, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        {
            let mut pass = frame
                .encoder
                .begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("field"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: frame.view,
                        depth_slice: None,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    ..Default::default()
                });
            pass.set_pipeline(&self.draw);
            pass.set_bind_group(0, &self.draw_group, &[]);
            pass.draw(0..3, 0..1);
        }

        // Ticks at the predicted dark fringes, beside the screen graph
        let pixels_per_unit = image[2] / VIEW_SIDE as f32;
        let centre_row = image[1] + image[2] * 0.5;
        let screen = self.wave.detector_x().unwrap_or(SCREEN);
        for m in [0.5, 1.5, 2.5] {
            let y = fringe(m, screen) as f32;
            if y < VIEW_SIDE as f32 / 2.0 {
                for row in [
                    centre_row - y * pixels_per_unit,
                    centre_row + y * pixels_per_unit,
                ] {
                    self.overlay.panel(
                        graph[0] + graph[2] + 2.0 * scale,
                        row - scale,
                        8.0 * scale,
                        2.0 * scale,
                        [0.55, 0.75, 1.0, 1.0],
                    );
                }
            }
        }
        let text = |s: &str| s.to_string();
        // In the browser the stats panel shows these already
        if !cfg!(target_arch = "wasm32") {
            let lines = self.readouts();
            self.overlay.lines(
                image[0] + 12.0 * scale,
                image[1] + 12.0 * scale,
                scale,
                &lines,
            );
        }
        let note_y = graph[1] + graph[3] - 40.0 * scale;
        for (k, line) in [
            text("Screen: current through x = 20,"),
            text("summed over time. Ticks: dark"),
            text("fringes from path difference"),
        ]
        .iter()
        .enumerate()
        {
            self.overlay.text(
                graph[0],
                note_y + k as f32 * 11.0 * scale,
                scale,
                [0.7, 0.72, 0.78, 1.0],
                line,
            );
        }
        if self.phase {
            self.overlay.text(
                image[0] + 12.0 * scale,
                image[1] + image[2] - 20.0 * scale,
                scale,
                [0.7, 0.72, 0.78, 1.0],
                "Hue: phase   brightness: |psi|",
            );
        } else {
            self.overlay.color_bar(
                image[0] + 12.0 * scale,
                image[1] + image[2] - 36.0 * scale,
                180.0 * scale,
                8.0 * scale,
                scale,
                Colormap::Inferno,
                ("0", "high"),
                "|psi|^2 (square-root scale)",
            );
        }
        self.overlay
            .draw(&gpu.device, &gpu.queue, frame.encoder, frame.view, size);
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn finish(&mut self, ctx: &mut Context<'_>) {
        let counts = self.wave.read_detector();
        ctx.data.snapshot("screen", &[counts.len()], &counts);
        let psi: Vec<f64> = self
            .wave
            .read_psi()
            .iter()
            .flat_map(|p| [p[0] as f64, p[1] as f64])
            .collect();
        ctx.data
            .snapshot("psi", &[grid() as usize, grid() as usize, 2], &psi);
    }
}

fn main() {
    virial::run::<DoubleSlit>(Config {
        title: "Virial: double slit",
    });
}

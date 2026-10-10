//! A gas reaching equilibrium. Every molecule starts at the same speed, in a
//! random direction: as far from equilibrium as a gas gets. Collisions alone
//! (no thermostat) spread the speeds into the Maxwell-Boltzmann distribution,
//! shown live as a histogram against the curve for the measured temperature.
//!
//! Lennard-Jones molecules (shifted-force cutoff at 2.5) in a periodic box,
//! in reduced units (for argon:
//! length 3.405e-10 m, temperature 119.8 K, time 2.15e-12 s).
//!
//! Settings: `n` (molecules, default 32768), `density` (default 0.1),
//! `temperature` (default 2), `steps` (steps per frame, default 8).
//! Controls: drag to orbit, scroll to zoom, Space to pause, R to restart.
//!
//! Render a video natively:
//! `cargo run --release -p gas -- record=renders/gas seconds=40`

use bytemuck::{Pod, Zeroable};
use virial::{Config, Context, Frame, Gpu, KeyCode, Scene, arg, wgpu};
use virial_particles::lennard_jones::{LennardJones, Measurement, Particle, Settings, lattice};
use virial_render::{
    COLORMAPS_WGSL, Camera, CameraPath, Coloring, Colormap, Overlay, Particles, Style,
};

const BINS: u32 = 64;
/// Step size. Energy drift over 40 time units (4096 molecules, shifted force)
/// is 5e-5 at dt 0.004, 1e-5 at 0.002 and 2.6e-6 at 0.001.
const DT: f64 = 0.002;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct HistogramUniforms {
    rect: [f32; 4],
    viewport: [f32; 2],
    speed_max: f32,
    count: u32,
    temperature: f32,
    density_max: f32,
    bins: u32,
    _pad: f32,
}

/// Every molecule at the same speed, in a random direction
fn start(n: usize, box_size: f64, temperature: f64) -> Vec<Particle> {
    let mut particles = lattice(n, box_size, 1.0, 5);
    let speed = (3.0 * temperature).sqrt();
    for p in &mut particles {
        let norm = p.vel.iter().map(|v| v * v).sum::<f64>().sqrt();
        p.vel = p.vel.map(|v| v / norm * speed);
    }
    particles
}

struct Gas {
    n: usize,
    density: f64,
    temperature: f64,
    sim: LennardJones,
    particles: Particles,
    overlay: Overlay,
    group: wgpu::BindGroup,
    histogram: (
        wgpu::ComputePipeline,
        wgpu::ComputePipeline,
        wgpu::RenderPipeline,
        wgpu::RenderPipeline,
    ),
    histogram_layouts: (wgpu::BindGroupLayout, wgpu::BindGroupLayout),
    /// Compute (counting) and draw bind groups: a buffer can't be read-write
    /// and read-only in the same group
    histogram_groups: (wgpu::BindGroup, wgpu::BindGroup),
    histogram_uniforms: wgpu::Buffer,
    counts: wgpu::Buffer,
    camera: Camera,
    path: Option<CameraPath>,
    steps_per_frame: u32,
    paused: bool,
    frames: u64,
    first: Option<Measurement>,
    latest: Option<Measurement>,
}

impl Gas {
    fn simulation(gpu: &Gpu, n: usize, density: f64, temperature: f64) -> LennardJones {
        let box_size = (n as f64 / density).cbrt();
        let settings = Settings {
            box_size,
            cutoff: 2.5,
            dt: DT,
            thermostat: None,
            shifted_force: true,
        };
        LennardJones::new(
            &gpu.device,
            &gpu.queue,
            &start(n, box_size, temperature),
            settings,
        )
    }

    fn bind_all(&mut self, gpu: &Gpu) {
        self.group = self
            .particles
            .bind(&gpu.device, self.sim.positions(), self.sim.velocities());
        self.histogram_groups = Self::histogram_groups(
            gpu,
            &self.histogram_layouts,
            &self.histogram_uniforms,
            &self.counts,
            &self.sim,
        );
    }

    fn histogram_groups(
        gpu: &Gpu,
        layouts: &(wgpu::BindGroupLayout, wgpu::BindGroupLayout),
        uniforms: &wgpu::Buffer,
        counts: &wgpu::Buffer,
        sim: &LennardJones,
    ) -> (wgpu::BindGroup, wgpu::BindGroup) {
        let count = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("histogram counting"),
            layout: &layouts.0,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniforms.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: sim.velocities(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: counts.as_entire_binding(),
                },
            ],
        });
        let draw = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("histogram drawing"),
            layout: &layouts.1,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniforms.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: counts.as_entire_binding(),
                },
            ],
        });
        (count, draw)
    }

    /// Temperature: measured once a measurement exists, else the starting one
    fn kinetic_temperature(&self) -> f64 {
        self.latest.map_or(self.temperature, |m| {
            m.temperature(3.0 * self.n as f64 - 3.0)
        })
    }

    fn readouts(&self) -> Vec<String> {
        let mut lines = vec![
            format!(
                "Gas reaching equilibrium: {} Lennard-Jones molecules",
                self.n
            ),
            format!(
                "t = {:.2}   density {}   no thermostat",
                self.sim.time(),
                self.density
            ),
        ];
        if let (Some(first), Some(m)) = (self.first, self.latest) {
            let t = self.kinetic_temperature();
            let rho = self.n as f64 / m.volume;
            let drift = (m.energy() - first.energy()) / first.energy().abs();
            lines.push(format!(
                "T = {t:.4}   P = {:.5}   PV/NkT = {:.4}",
                m.pressure(),
                m.pressure() / (rho * t)
            ));
            lines.push(format!("Energy drift {drift:+.2e}"));
        }
        lines
    }
}

impl Scene for Gas {
    fn new(gpu: &Gpu) -> Self {
        let n = arg("n", 32_768usize).clamp(512, 1 << 20);
        let density = arg("density", 0.1f64).clamp(0.001, 0.8);
        let temperature = arg("temperature", 2.0f64).clamp(0.1, 20.0);
        let sim = Self::simulation(gpu, n, density, temperature);
        let device = &gpu.device;
        let particles = Particles::new(device, gpu.surface_format());
        let overlay = Overlay::new(device, gpu.surface_format());

        let entry = |binding, visibility, ty| wgpu::BindGroupLayoutEntry {
            binding,
            visibility,
            ty: wgpu::BindingType::Buffer {
                ty,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let compute = wgpu::ShaderStages::COMPUTE;
        let both = wgpu::ShaderStages::COMPUTE | wgpu::ShaderStages::VERTEX;
        let counting_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("histogram counting"),
            entries: &[
                entry(0, both, wgpu::BufferBindingType::Uniform),
                entry(
                    1,
                    compute,
                    wgpu::BufferBindingType::Storage { read_only: true },
                ),
                entry(
                    2,
                    compute,
                    wgpu::BufferBindingType::Storage { read_only: false },
                ),
            ],
        });
        let drawing_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("histogram drawing"),
            entries: &[
                entry(0, both, wgpu::BufferBindingType::Uniform),
                entry(
                    3,
                    wgpu::ShaderStages::VERTEX,
                    wgpu::BufferBindingType::Storage { read_only: true },
                ),
            ],
        });
        let counting = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("histogram counting"),
            bind_group_layouts: &[Some(&counting_layout)],
            immediate_size: 0,
        });
        let drawing = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("histogram drawing"),
            bind_group_layouts: &[Some(&drawing_layout)],
            immediate_size: 0,
        });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("histogram"),
            source: wgpu::ShaderSource::Wgsl(
                format!("{COLORMAPS_WGSL}\n{}", include_str!("histogram.wgsl")).into(),
            ),
        });
        let compute_pipeline = |entry: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&counting),
                module: &module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let render_pipeline = |vs: &str| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(vs),
                layout: Some(&drawing),
                vertex: wgpu::VertexState {
                    module: &module,
                    entry_point: Some(vs),
                    compilation_options: Default::default(),
                    buffers: &[],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &module,
                    entry_point: Some("fs_main"),
                    compilation_options: Default::default(),
                    targets: &[Some(gpu.surface_format().into())],
                }),
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            })
        };
        let histogram = (
            compute_pipeline("clear"),
            compute_pipeline("fill"),
            render_pipeline("vs_bars"),
            render_pipeline("vs_curve"),
        );
        let histogram_uniforms = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("histogram"),
            size: size_of::<HistogramUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let counts = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("histogram counts"),
            size: BINS as u64 * 4,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let placeholder = particles.bind(device, sim.positions(), sim.velocities());
        let histogram_layouts = (counting_layout, drawing_layout);
        let histogram_groups =
            Self::histogram_groups(gpu, &histogram_layouts, &histogram_uniforms, &counts, &sim);
        let box_size = (n as f64 / density).cbrt() as f32;
        let centre = [box_size / 2.0; 3];
        let camera = Camera {
            target: centre,
            yaw: arg("yaw", 0.6),
            pitch: arg("pitch", 1.05),
            distance: arg("distance", box_size * 2.6),
            ..Camera::default()
        };
        Self {
            n,
            density,
            temperature,
            sim,
            particles,
            overlay,
            group: placeholder,
            histogram,
            histogram_layouts,
            histogram_groups,
            histogram_uniforms,
            counts,
            camera,
            path: None,
            steps_per_frame: arg("steps", 8u32).clamp(1, 64),
            paused: false,
            frames: 0,
            first: None,
            latest: None,
        }
    }

    fn update(&mut self, ctx: &mut Context<'_>) {
        if self.frames == 0 && ctx.data.is_recording() {
            if arg("path", 1u32) != 0 {
                let c = self.camera;
                self.path = Some(CameraPath::new().key(0.0, c).key(
                    40.0,
                    Camera {
                        yaw: c.yaw + 1.4,
                        pitch: c.pitch - 0.2,
                        distance: c.distance * 0.85,
                        ..c
                    },
                ));
            }
            let data = &mut *ctx.data;
            data.meta("molecules", self.n);
            data.meta("density", self.density);
            data.meta("initial_temperature", self.temperature);
            data.meta("dt", DT);
            data.meta("cutoff", 2.5);
            data.meta("steps_per_frame", self.steps_per_frame);
        }
        let input = ctx.input;
        if input.pressed(KeyCode::Space) {
            self.paused = !self.paused;
        }
        if input.pressed(KeyCode::KeyR) {
            self.sim = Self::simulation(ctx.gpu, self.n, self.density, self.temperature);
            self.bind_all(ctx.gpu);
            self.first = None;
            self.latest = None;
        }
        match &self.path {
            Some(path) => self.camera = path.at(ctx.time.elapsed as f32),
            None => self.camera.orbit(input.drag(), input.scroll()),
        }
        if self.frames == 0 {
            self.bind_all(ctx.gpu);
        }

        self.sim.after_submit();
        for m in self.sim.poll() {
            let first = *self.first.get_or_insert(m);
            self.latest = Some(m);
            let t = m.temperature(3.0 * self.n as f64 - 3.0);
            ctx.data.row(&[
                ("step", m.step as f64),
                ("time", m.time),
                ("kinetic", m.kinetic),
                ("potential", m.potential),
                ("energy", m.energy()),
                (
                    "relative_drift",
                    (m.energy() - first.energy()) / first.energy().abs(),
                ),
                ("temperature", t),
                ("pressure", m.pressure()),
                ("virial", m.virial),
            ]);
        }
        for (label, line) in ["Scene", "Clock", "Thermodynamics", "Energy"]
            .iter()
            .zip(self.readouts())
        {
            ctx.stats.set(label, line);
        }

        let size = (ctx.gpu.config.width as f32, ctx.gpu.config.height as f32);
        let scale = (size.1 / 540.0).round().max(1.0);
        let (w, h) = (300.0 * scale, 150.0 * scale);
        let t = self.kinetic_temperature();
        let speed_max = 4.0 * t.sqrt() as f32;
        let peak = 4.0
            * std::f64::consts::PI
            * (1.0 / (2.0 * std::f64::consts::PI * t)).powf(1.5)
            * 2.0
            * t
            * (-1.0f64).exp();
        let uniforms = HistogramUniforms {
            rect: [size.0 - w - 24.0 * scale, size.1 - h - 30.0 * scale, w, h],
            viewport: [size.0, size.1],
            speed_max,
            count: self.n as u32,
            temperature: t as f32,
            density_max: (peak * 1.6) as f32,
            bins: BINS,
            _pad: 0.0,
        };
        ctx.gpu
            .queue
            .write_buffer(&self.histogram_uniforms, 0, bytemuck::bytes_of(&uniforms));
    }

    fn render(&mut self, ctx: &mut Context<'_>, frame: Frame<'_>) {
        if !self.paused {
            self.sim.step(
                frame.encoder,
                self.steps_per_frame,
                frame.timer.and_then(|t| t.compute("molecules")),
            );
        }
        if self.frames.is_multiple_of(10) {
            self.sim.measure(frame.encoder);
        }
        self.frames += 1;
        {
            let mut pass = frame.encoder.begin_compute_pass(&Default::default());
            pass.set_bind_group(0, &self.histogram_groups.0, &[]);
            pass.set_pipeline(&self.histogram.0);
            pass.dispatch_workgroups(1, 1, 1);
            pass.set_pipeline(&self.histogram.1);
            pass.dispatch_workgroups((self.n as u32).div_ceil(256), 1, 1);
        }

        let gpu = ctx.gpu;
        let size = (gpu.config.width, gpu.config.height);
        let t = self.kinetic_temperature() as f32;
        let speed_max = 4.0 * t.sqrt();
        self.particles.draw(
            &gpu.device,
            &gpu.queue,
            frame.encoder,
            frame.view,
            size,
            &self.camera,
            &self.group,
            self.n as u32,
            Style::Spheres { radius: 0.5 },
            Coloring::Speed {
                map: Colormap::Inferno,
                min: 0.0,
                max: speed_max * 0.75,
            },
            [0.012, 0.013, 0.02],
            frame.timer.and_then(|t| t.render("draw")),
        );

        // Histogram panel behind the bars, then bars and curve, then labels
        let scale = (size.1 as f32 / 540.0).round().max(1.0);
        let (w, h) = (300.0 * scale, 150.0 * scale);
        let (x, y) = (
            size.0 as f32 - w - 24.0 * scale,
            size.1 as f32 - h - 30.0 * scale,
        );
        self.overlay.panel(
            x - 10.0 * scale,
            y - 26.0 * scale,
            w + 20.0 * scale,
            h + 46.0 * scale,
            [0.03, 0.035, 0.05, 0.82],
        );
        self.overlay
            .draw(&gpu.device, &gpu.queue, frame.encoder, frame.view, size);
        {
            let mut pass = frame
                .encoder
                .begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("histogram"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: frame.view,
                        depth_slice: None,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    ..Default::default()
                });
            pass.set_bind_group(0, &self.histogram_groups.1, &[]);
            pass.set_pipeline(&self.histogram.2);
            pass.draw(0..6, 0..BINS);
            pass.set_pipeline(&self.histogram.3);
            pass.draw(0..6, 0..160);
        }
        let white = [0.92, 0.93, 0.96, 1.0];
        let cyan = [0.35, 0.85, 1.0, 1.0];
        self.overlay
            .text(x, y - 18.0 * scale, scale, white, "Speed distribution");
        let legend = "Maxwell-Boltzmann";
        self.overlay.text(
            x + w - virial_render::text_width(legend, scale),
            y - 18.0 * scale,
            scale,
            cyan,
            legend,
        );
        self.overlay.text(x, y + h + 6.0 * scale, scale, white, "0");
        let top = format!("{speed_max:.1}");
        self.overlay.text(
            x + w - virial_render::text_width(&top, scale),
            y + h + 6.0 * scale,
            scale,
            white,
            &top,
        );
        if !cfg!(target_arch = "wasm32") {
            self.overlay
                .lines(16.0 * scale, 16.0 * scale, scale, &self.readouts());
        }
        self.overlay
            .draw(&gpu.device, &gpu.queue, frame.encoder, frame.view, size);
    }

    fn finish(&mut self, ctx: &mut Context<'_>) {
        #[cfg(not(target_arch = "wasm32"))]
        {
            let speeds: Vec<f64> = self
                .sim
                .read_particles()
                .iter()
                .map(|p| p.vel.iter().map(|v| v * v).sum::<f64>().sqrt())
                .collect();
            ctx.data.snapshot("final_speeds", &[speeds.len()], &speeds);
        }
        #[cfg(target_arch = "wasm32")]
        let _ = ctx;
    }
}

fn main() {
    virial::run::<Gas>(Config {
        title: "Virial: gas",
    });
}

//! Two disk galaxies on a close flyby. Their stars are pulled into tidal tails
//! and bridges, the shapes seen in real interacting galaxies such as the
//! Antennae.
//!
//! Every star feels every other star and both cores (exact all-pairs gravity),
//! with double-float state. The stats panel shows the energy drift since the
//! start, measured on the GPU.
//!
//! Units: G = 1, each core has mass 1, disks have radius about 2.
//! Settings: `n` (bodies, default 8192), `steps` (steps per frame, default 4).
//! Controls: drag to orbit, scroll to zoom, Space to pause, R to restart,
//! [ and ] to slow down or speed up.

use std::f64::consts::TAU;

use bytemuck::{Pod, Zeroable};
use virial::{Config, Context, Frame, Gpu, KeyCode, Scene, arg, wgpu};
use virial_particles::{Body, Gravity, Measurement, NBody};

const GRAVITY: Gravity = Gravity { g: 1.0, softening: 0.05 };
const DT: f64 = 0.005;
const CORE_MASS: f64 = 1.0;
/// Each disk's mass, shared by its stars
const DISK_MASS: f64 = 0.15;
/// Frames between energy measurements. Each costs about ten force passes, so
/// the preview pauses briefly when one runs.
const MEASURE_EVERY: u64 = 300;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct View {
    view_proj: [[f32; 4]; 4],
    viewport: [f32; 2],
    size: f32,
    half: u32,
    intensity: f32,
    heavy_mass: f32,
    _pad: [f32; 2],
}

/// xorshift: deterministic, so every run starts from the same galaxies
struct Random(u64);

impl Random {
    fn next(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Standard normal (Box-Muller)
    fn normal(&mut self) -> f64 {
        let u = self.next().max(1e-12);
        (-2.0 * u.ln()).sqrt() * (TAU * self.next()).cos()
    }
}

/// A core and `stars` stars on near-circular orbits in the x-y plane, then
/// tilted by `tilt` about the x axis and moved to `centre` at `velocity`.
fn galaxy(stars: usize, centre: [f64; 3], velocity: [f64; 3], tilt: f64, spin: f64, random: &mut Random) -> Vec<Body> {
    let (r_min, r_max) = (0.25, 2.0);
    let star_mass = DISK_MASS / stars as f64;
    let mut bodies = Vec::with_capacity(stars + 1);
    bodies.push(Body { pos: [0.0; 3], vel: [0.0; 3], mass: CORE_MASS });
    let eps2 = GRAVITY.softening * GRAVITY.softening;
    for _ in 0..stars {
        // Uniform density across the disk
        let r = (r_min * r_min + (r_max * r_max - r_min * r_min) * random.next()).sqrt();
        let angle = TAU * random.next();
        // Pull of the core (softened) plus the disk mass inside r
        let inside = DISK_MASS * (r * r - r_min * r_min) / (r_max * r_max - r_min * r_min);
        let v2 = GRAVITY.g * (CORE_MASS * r * r / (r * r + eps2).powf(1.5) + inside / r);
        let v = v2.sqrt() * (1.0 + 0.02 * random.normal());
        let (s, c) = angle.sin_cos();
        bodies.push(Body {
            pos: [r * c, r * s, 0.02 * random.normal()],
            vel: [-v * s * spin, v * c * spin, 0.01 * random.normal()],
            mass: star_mass,
        });
    }
    let (st, ct) = tilt.sin_cos();
    let rotate = |p: [f64; 3]| [p[0], p[1] * ct - p[2] * st, p[1] * st + p[2] * ct];
    for b in &mut bodies {
        let p = rotate(b.pos);
        let v = rotate(b.vel);
        b.pos = [0, 1, 2].map(|k| p[k] + centre[k]);
        b.vel = [0, 1, 2].map(|k| v[k] + velocity[k]);
    }
    bodies
}

/// Two galaxies on a near-parabolic encounter, in their centre-of-mass frame.
fn encounter(n: usize) -> Vec<Body> {
    let mut random = Random(0x5eed_1234_abcd_ef01);
    let stars = n / 2 - 1;
    let separation = 8.0f64;
    let total = 2.0 * (CORE_MASS + DISK_MASS);
    // Speed of a parabolic orbit at this separation, shared between the two
    let v = (2.0 * GRAVITY.g * total / separation).sqrt() * 0.5;
    let mut bodies = galaxy(stars, [-4.0, -1.2, 0.0], [v, 0.0, 0.0], 0.0, 1.0, &mut random);
    bodies.extend(galaxy(stars, [4.0, 1.2, 0.0], [-v, 0.0, 0.0], 1.05, 1.0, &mut random));
    bodies
}

fn perspective(fov_y: f32, aspect: f32, near: f32, far: f32) -> [[f32; 4]; 4] {
    let f = 1.0 / (fov_y / 2.0).tan();
    [
        [f / aspect, 0.0, 0.0, 0.0],
        [0.0, f, 0.0, 0.0],
        [0.0, 0.0, far / (near - far), -1.0],
        [0.0, 0.0, near * far / (near - far), 0.0],
    ]
}

fn look_at(eye: [f32; 3], target: [f32; 3], up: [f32; 3]) -> [[f32; 4]; 4] {
    let sub = |a: [f32; 3], b: [f32; 3]| [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
    let dot = |a: [f32; 3], b: [f32; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    let cross = |a: [f32; 3], b: [f32; 3]| [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
    let norm = |a: [f32; 3]| {
        let l = dot(a, a).sqrt();
        [a[0] / l, a[1] / l, a[2] / l]
    };
    let f = norm(sub(target, eye));
    let s = norm(cross(f, up));
    let u = cross(s, f);
    [
        [s[0], u[0], -f[0], 0.0],
        [s[1], u[1], -f[1], 0.0],
        [s[2], u[2], -f[2], 0.0],
        [-dot(s, eye), -dot(u, eye), dot(f, eye), 1.0],
    ]
}

fn multiply(a: [[f32; 4]; 4], b: [[f32; 4]; 4]) -> [[f32; 4]; 4] {
    let mut out = [[0.0; 4]; 4];
    for (c, col) in out.iter_mut().enumerate() {
        for (r, value) in col.iter_mut().enumerate() {
            *value = (0..4).map(|k| a[k][r] * b[c][k]).sum();
        }
    }
    out
}

struct Galaxies {
    n: usize,
    sim: NBody,
    pipeline: wgpu::RenderPipeline,
    view: wgpu::Buffer,
    layout: wgpu::BindGroupLayout,
    group: wgpu::BindGroup,
    yaw: f32,
    pitch: f32,
    distance: f32,
    steps_per_frame: u32,
    paused: bool,
    frames: u64,
    first: Option<Measurement>,
    latest: Option<Measurement>,
    steps_window: (f64, u64),
}

impl Galaxies {
    fn bind(gpu: &Gpu, layout: &wgpu::BindGroupLayout, view: &wgpu::Buffer, sim: &NBody) -> wgpu::BindGroup {
        gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("stars"),
            layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: view.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: sim.positions().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: sim.velocities().as_entire_binding() },
            ],
        })
    }

    fn restart(&mut self, gpu: &Gpu) {
        self.sim = NBody::new(&gpu.device, &gpu.queue, &encounter(self.n), GRAVITY, DT);
        self.group = Self::bind(gpu, &self.layout, &self.view, &self.sim);
        self.first = None;
        self.latest = None;
        self.frames = 0;
    }
}

impl Scene for Galaxies {
    fn new(gpu: &Gpu) -> Self {
        let n = arg("n", 8_192usize).clamp(256, 262_144);
        let steps_per_frame = arg("steps", 4u32).clamp(1, 64);
        let sim = NBody::new(&gpu.device, &gpu.queue, &encounter(n), GRAVITY, DT);
        let device = &gpu.device;

        let entry = |binding, ty| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::VERTEX,
            ty: wgpu::BindingType::Buffer { ty, has_dynamic_offset: false, min_binding_size: None },
            count: None,
        };
        let storage = wgpu::BufferBindingType::Storage { read_only: true };
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("stars"),
            entries: &[entry(0, wgpu::BufferBindingType::Uniform), entry(1, storage), entry(2, storage)],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("stars"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let module = device.create_shader_module(wgpu::include_wgsl!("stars.wgsl"));
        let additive = wgpu::BlendComponent {
            src_factor: wgpu::BlendFactor::One,
            dst_factor: wgpu::BlendFactor::One,
            operation: wgpu::BlendOperation::Add,
        };
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("stars"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: gpu.surface_format(),
                    blend: Some(wgpu::BlendState { color: additive, alpha: additive }),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });
        let view = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("view"),
            size: size_of::<View>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let group = Self::bind(gpu, &layout, &view, &sim);

        Self {
            n,
            sim,
            pipeline,
            view,
            layout,
            group,
            yaw: 0.35,
            pitch: 0.9,
            distance: 14.0,
            steps_per_frame,
            paused: false,
            frames: 0,
            first: None,
            latest: None,
            steps_window: (0.0, 0),
        }
    }

    fn update(&mut self, ctx: &mut Context<'_>) {
        let input = ctx.input;
        if input.pressed(KeyCode::Space) {
            self.paused = !self.paused;
        }
        if input.pressed(KeyCode::KeyR) {
            self.restart(ctx.gpu);
        }
        if input.pressed(KeyCode::BracketRight) {
            self.steps_per_frame = (self.steps_per_frame * 2).min(64);
        }
        if input.pressed(KeyCode::BracketLeft) {
            self.steps_per_frame = (self.steps_per_frame / 2).max(1);
        }
        let [dx, dy] = input.drag();
        self.yaw -= dx * 2.5;
        self.pitch = (self.pitch + dy * 2.5).clamp(0.05, 3.09);
        self.distance = (self.distance * (1.0 - input.scroll() * 0.1)).clamp(2.0, 80.0);

        // Measurements recorded last frame can start reading back now
        self.sim.after_submit();
        for m in self.sim.poll() {
            self.first.get_or_insert(m);
            self.latest = Some(m);
        }

        let aspect = ctx.gpu.aspect();
        let (sp, cp) = self.pitch.sin_cos();
        let (sy, cy) = self.yaw.sin_cos();
        let eye = [self.distance * sp * sy, -self.distance * sp * cy, self.distance * cp];
        let view_proj = multiply(perspective(0.8, aspect, 0.05, 500.0), look_at(eye, [0.0; 3], [0.0, 0.0, 1.0]));
        let config = &ctx.gpu.config;
        let view = View {
            view_proj,
            viewport: [config.width as f32, config.height as f32],
            size: (config.height as f32 / 900.0 * 2.2).max(1.5),
            half: (self.n / 2) as u32,
            intensity: (6000.0 / self.n as f32).sqrt().clamp(0.12, 1.0),
            heavy_mass: 0.5,
            _pad: [0.0; 2],
        };
        ctx.gpu.queue.write_buffer(&self.view, 0, bytemuck::bytes_of(&view));

        // Readouts
        let (elapsed, steps) = &mut self.steps_window;
        *elapsed += ctx.time.dt as f64;
        if *elapsed >= 1.0 {
            let rate = (self.sim.steps() - *steps) as f64 / *elapsed;
            ctx.stats.set("Steps per second", format!("{rate:.0}"));
            *elapsed = 0.0;
            *steps = self.sim.steps();
        }
        ctx.stats.set("Bodies", format!("{}", self.n));
        ctx.stats.set("Simulation time", format!("{:.2} (dt {DT}, {} steps/frame)", self.sim.time(), self.steps_per_frame));
        if let (Some(first), Some(latest)) = (self.first, self.latest) {
            let drift = (latest.energy() - first.energy()) / first.energy().abs();
            ctx.stats.set("Energy", format!("{:.9}", latest.energy()));
            ctx.stats.set("Energy drift", format!("{drift:+.2e}  (at t = {:.1})", latest.time));
            let p = latest.momentum.iter().map(|x| x * x).sum::<f64>().sqrt();
            ctx.stats.set("Total momentum", format!("{p:.1e}"));
        }
        ctx.stats.set("Controls", "drag · scroll · Space · R · [ ]");
        if self.paused {
            ctx.stats.set("State", "paused");
        } else {
            ctx.stats.set("State", "running");
        }
    }

    fn render(&mut self, _ctx: &mut Context<'_>, frame: Frame<'_>) {
        if !self.paused {
            self.sim.step(frame.encoder, self.steps_per_frame, frame.timer.and_then(|t| t.compute("gravity")));
        }
        if self.frames % MEASURE_EVERY == 0 {
            self.sim.measure(frame.encoder);
        }
        self.frames += 1;

        let mut pass = frame.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("stars"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: frame.view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color { r: 0.004, g: 0.005, b: 0.012, a: 1.0 }),
                    store: wgpu::StoreOp::Store,
                },
            })],
            timestamp_writes: frame.timer.and_then(|t| t.render("draw")),
            ..Default::default()
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.group, &[]);
        pass.draw(0..6, 0..self.n as u32);
    }
}

fn main() {
    virial::run::<Galaxies>(Config { title: "Virial: galaxies" });
}

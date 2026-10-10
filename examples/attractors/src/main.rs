//! Step 2: the gpu-playground WebGPU particle demo ported to Virial.
//!
//! Same kernel, same starting layout, same controls, so its frame rate can be
//! compared with the original. Settings: `n` (particle count, up to
//! 16000000) and `work` (`light` = 1 attractor, `heavy` = 16).
//! Browser: `?n=4000000&work=heavy`. Native: `cargo run -p attractors --release -- n=4000000 work=heavy`.

use std::f32::consts::TAU;

use bytemuck::{Pod, Zeroable};
use virial::wgpu::util::DeviceExt;
use virial::{Config, Context, Frame, Gpu, Scene, arg, wgpu};

const MAX_ATTRACTORS: usize = 16;
/// Every particle is simulated, but at most this many are drawn, as in the
/// original: drawing millions of overlapping points would make drawing the
/// bottleneck and hide the simulation's speed.
const MAX_DRAWN: u32 = 1_000_000;
const DT: f32 = 1.0 / 60.0;
const STRENGTH: f32 = 0.8;
const DAMPING: f32 = 0.996;
const SOFTENING: f32 = 0.0025;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    scale: [f32; 2],
    dt: f32,
    strength: f32,
    damping: f32,
    count: u32,
    softening: f32,
    intensity: f32,
    attractor_count: u32,
    _pad: [u32; 3],
}

struct Attractors {
    count: u32,
    simulate: wgpu::ComputePipeline,
    draw: wgpu::RenderPipeline,
    params: wgpu::Buffer,
    attractors: wgpu::Buffer,
    sim_group: wgpu::BindGroup,
    draw_group: wgpu::BindGroup,
    attractor_count: u32,
    pointer: [f32; 2],
    last_move: f64,
}

/// The original's seeded random numbers, so both start from the same layout.
fn mulberry32(mut seed: u32) -> impl FnMut() -> f32 {
    move || {
        seed = seed.wrapping_add(0x6d2b_79f5);
        let mut t = (seed ^ (seed >> 15)).wrapping_mul(1 | seed);
        t = (t.wrapping_add((t ^ (t >> 7)).wrapping_mul(61 | t))) ^ t;
        ((t ^ (t >> 14)) as f64 / 4_294_967_296.0) as f32
    }
}

/// A ring already swirling: position and velocity per particle.
fn seed_particles(n: u32) -> Vec<[f32; 4]> {
    let mut random = mulberry32(42);
    (0..n)
        .map(|_| {
            let angle = random() * TAU;
            let radius = 0.25 + random().sqrt() * 0.65;
            let (s, c) = angle.sin_cos();
            [c * radius, s * radius, -s * 0.35, c * 0.35]
        })
        .collect()
}

impl Scene for Attractors {
    fn new(gpu: &Gpu) -> Self {
        let count = arg("n", 250_000u32).clamp(1, 16_000_000);
        let attractor_count = if arg("work", String::new()) == "heavy" {
            16
        } else {
            1
        };
        let device = &gpu.device;

        let module = device.create_shader_module(wgpu::include_wgsl!("shaders.wgsl"));
        let simulate = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("simulate"),
            layout: None,
            module: &module,
            entry_point: Some("simulate"),
            compilation_options: Default::default(),
            cache: None,
        });
        let additive = wgpu::BlendComponent {
            src_factor: wgpu::BlendFactor::One,
            dst_factor: wgpu::BlendFactor::One,
            operation: wgpu::BlendOperation::Add,
        };
        let draw = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("draw"),
            layout: None,
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
                // Additive blending: overlapping particles glow
                targets: &[Some(wgpu::ColorTargetState {
                    format: gpu.surface_format(),
                    blend: Some(wgpu::BlendState {
                        color: additive,
                        alpha: additive,
                    }),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::PointList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        let particles = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("particles"),
            contents: bytemuck::cast_slice(&seed_particles(count)),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("params"),
            size: size_of::<Params>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let attractors = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("attractors"),
            size: (MAX_ATTRACTORS * 16) as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let sim_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("simulate"),
            layout: &simulate.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: particles.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: attractors.as_entire_binding(),
                },
            ],
        });
        let draw_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("draw"),
            layout: &draw.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: particles.as_entire_binding(),
                },
            ],
        });

        Self {
            count,
            simulate,
            draw,
            params,
            attractors,
            sim_group,
            draw_group,
            attractor_count,
            pointer: [0.0, 0.0],
            last_move: f64::NEG_INFINITY,
        }
    }

    fn update(&mut self, ctx: &mut Context<'_>) {
        let aspect = ctx.gpu.aspect();
        let now = ctx.time.elapsed;

        // Pulled towards the mouse, or a slow figure-eight when idle
        if let Some([x, y]) = ctx.input.cursor() {
            let moved = [x * aspect, y];
            if moved != self.pointer {
                self.pointer = moved;
                self.last_move = now;
            }
        }
        let centre = if now - self.last_move < 2.5 {
            self.pointer
        } else {
            let t = now as f32;
            [(t * 0.53).sin() * 0.55 * aspect, (t * 0.71).sin() * 0.5]
        };

        // One attractor at the centre, or a slowly turning ring of them
        let mut points = [[0.0f32; 4]; MAX_ATTRACTORS];
        let n = self.attractor_count;
        for (a, point) in points.iter_mut().take(n as usize).enumerate() {
            let angle = a as f32 / n as f32 * TAU + now as f32 * 0.4;
            let radius = if n == 1 { 0.0 } else { 0.3 };
            point[0] = centre[0] + angle.cos() * radius;
            point[1] = centre[1] + angle.sin() * radius;
        }
        ctx.gpu
            .queue
            .write_buffer(&self.attractors, 0, bytemuck::cast_slice(&points));

        let drawn = self.count.min(MAX_DRAWN);
        let params = Params {
            scale: [1.0 / aspect, 1.0],
            dt: DT,
            strength: STRENGTH,
            damping: DAMPING,
            count: self.count,
            softening: SOFTENING,
            // Fewer particles glow brighter each, so every count looks similar
            intensity: (40_000.0 / drawn as f32).clamp(0.035, 0.7),
            attractor_count: n,
            _pad: [0; 3],
        };
        ctx.gpu
            .queue
            .write_buffer(&self.params, 0, bytemuck::bytes_of(&params));

        ctx.stats.set("Particles", format_count(self.count));
        ctx.stats.set(
            "Work",
            if n == 1 {
                "light (1 attractor)"
            } else {
                "heavy (16 attractors)"
            },
        );
    }

    fn render(&mut self, _ctx: &mut Context<'_>, frame: Frame<'_>) {
        {
            let mut pass = frame
                .encoder
                .begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("simulate"),
                    timestamp_writes: frame.timer.and_then(|t| t.compute("simulate")),
                });
            pass.set_pipeline(&self.simulate);
            pass.set_bind_group(0, &self.sim_group, &[]);
            pass.dispatch_workgroups(self.count.div_ceil(256), 1, 1);
        }
        let mut pass = frame
            .encoder
            .begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("draw"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: frame.view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.02,
                            g: 0.024,
                            b: 0.035,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                timestamp_writes: frame.timer.and_then(|t| t.render("draw")),
                ..Default::default()
            });
        pass.set_pipeline(&self.draw);
        pass.set_bind_group(0, &self.draw_group, &[]);
        pass.draw(0..self.count.min(MAX_DRAWN), 0..1);
    }
}

fn format_count(n: u32) -> String {
    if n >= 1_000_000 {
        format!("{}M", n as f32 / 1e6)
    } else {
        format!("{}k", n / 1000)
    }
}

fn main() {
    virial::run::<Attractors>(Config {
        title: "Virial: attractors",
    });
}

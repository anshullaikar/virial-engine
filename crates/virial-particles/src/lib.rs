//! GPU particle dynamics.
//!
//! [`NBody`] integrates point masses under Newtonian gravity on the GPU:
//!
//! - **Velocity Verlet** (kick, drift, kick). It's symplectic, so energy error
//!   stays bounded over long runs instead of growing.
//! - **Double-float state.** Positions and velocities carry about 48
//!   significant bits; forces use per-part differences, so precision doesn't
//!   depend on distance from the origin.
//! - **Exact all-pairs forces**, tiled through workgroup memory. O(N²): right
//!   for up to roughly 10⁵ bodies.
//! - **Measurements** of energy and momentum, summed in 64-bit fixed point
//!   and read back without stalling the GPU.
//!
//! Units are whatever the caller uses consistently; SI by convention.

pub mod lennard_jones;

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use bytemuck::{Pod, Zeroable};
use virial_precision::{Df64, WGSL_DF64, WGSL_FIXED, fixed};
use wgpu::util::DeviceExt;

/// A point mass.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Body {
    pub pos: [f64; 3],
    pub vel: [f64; 3],
    pub mass: f64,
}

/// Newtonian gravity with Plummer softening: the pull between two bodies
/// falls off as d / (|d|² + softening²)^(3/2). Softening 0 is exact gravity.
#[derive(Clone, Copy, Debug)]
pub struct Gravity {
    pub g: f64,
    pub softening: f64,
}

/// Energy and momentum at one step.
#[derive(Clone, Copy, Debug)]
pub struct Measurement {
    pub step: u64,
    /// Simulation time: steps × dt.
    pub time: f64,
    pub kinetic: f64,
    pub potential: f64,
    pub momentum: [f64; 3],
}

impl Measurement {
    pub fn energy(&self) -> f64 {
        self.kinetic + self.potential
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    count: u32,
    g: f32,
    eps2: f32,
    dt: f32,
    half_dt: f32,
    one: f32,
    energy_scale: f32,
    momentum_scale: f32,
    eps2_lo: f32,
    _pad: [f32; 3],
}

const WORKGROUP: u32 = 256;
const FIXED_SLOTS: u64 = 8;
const READBACK_RING: usize = 3;

struct Readback {
    buffer: wgpu::Buffer,
    step: u64,
    ready: Arc<AtomicBool>,
    in_flight: bool,
    pending_map: bool,
}

/// N bodies under gravity, on the GPU.
// Some fields are only read by the native-only helpers
#[cfg_attr(target_arch = "wasm32", allow(dead_code))]
pub struct NBody {
    device: wgpu::Device,
    queue: wgpu::Queue,
    count: u32,
    dt: f64,
    steps: u64,
    primed: bool,
    pos_hi: wgpu::Buffer,
    pos_lo: wgpu::Buffer,
    vel_hi: wgpu::Buffer,
    vel_lo: wgpu::Buffer,
    fixed: wgpu::Buffer,
    group: wgpu::BindGroup,
    drift: wgpu::ComputePipeline,
    prime: wgpu::ComputePipeline,
    force_kick: wgpu::ComputePipeline,
    measure: wgpu::ComputePipeline,
    readback: Vec<Readback>,
    energy_scale: f64,
    momentum_scale: f64,
}

/// Largest power of two `s` with `magnitude × s` at most 2^20: totals then sit
/// well inside the fixed-point range (2^31) with 2^-52 relative resolution.
fn fixed_scale(magnitude: f64) -> f64 {
    if magnitude <= 0.0 || !magnitude.is_finite() {
        return 1.0;
    }
    2f64.powi((20.0 - magnitude.log2()).floor() as i32)
}

/// Rough size of the potential energy, for choosing the fixed-point scale.
/// Exact for small systems, sampled for large ones.
fn potential_estimate(bodies: &[Body], gravity: Gravity) -> f64 {
    let n = bodies.len();
    let stride = (n / 2048).max(1);
    let sample: Vec<&Body> = bodies.iter().step_by(stride).collect();
    let eps2 = gravity.softening * gravity.softening;
    let mut sum = 0.0;
    for (i, a) in sample.iter().enumerate() {
        for b in &sample[i + 1..] {
            let r2: f64 = (0..3).map(|k| (a.pos[k] - b.pos[k]).powi(2)).sum();
            sum += gravity.g * a.mass * b.mass / (r2 + eps2).sqrt().max(1e-300);
        }
    }
    let s = sample.len() as f64;
    let pairs = |k: f64| k * (k - 1.0) / 2.0;
    if s < 2.0 {
        0.0
    } else {
        sum * pairs(n as f64) / pairs(s)
    }
}

impl NBody {
    /// Uploads `bodies`. `dt` is the step length.
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        bodies: &[Body],
        gravity: Gravity,
        dt: f64,
    ) -> Self {
        let count = bodies.len() as u32;
        let split = Df64::from_f64;
        let mut pos_hi = Vec::with_capacity(bodies.len());
        let mut pos_lo = Vec::with_capacity(bodies.len());
        let mut vel_hi = Vec::with_capacity(bodies.len());
        let mut vel_lo = Vec::with_capacity(bodies.len());
        for b in bodies {
            let p = b.pos.map(split);
            let v = b.vel.map(split);
            let m = split(b.mass);
            pos_hi.push([p[0].hi, p[1].hi, p[2].hi, m.hi]);
            pos_lo.push([p[0].lo, p[1].lo, p[2].lo, m.lo]);
            vel_hi.push([v[0].hi, v[1].hi, v[2].hi, 0.0]);
            vel_lo.push([v[0].lo, v[1].lo, v[2].lo, 0.0]);
        }

        let kinetic: f64 = bodies
            .iter()
            .map(|b| 0.5 * b.mass * b.vel.iter().map(|v| v * v).sum::<f64>())
            .sum();
        let energy_scale = fixed_scale(kinetic.max(potential_estimate(bodies, gravity)));
        let momentum: f64 = bodies
            .iter()
            .map(|b| b.mass * b.vel.iter().map(|v| v * v).sum::<f64>().sqrt())
            .sum();
        let momentum_scale = fixed_scale(momentum);

        let eps2 = split(gravity.softening * gravity.softening);
        let params = Params {
            count,
            g: gravity.g as f32,
            eps2: eps2.hi,
            dt: dt as f32,
            half_dt: (dt * 0.5) as f32,
            one: 1.0,
            energy_scale: energy_scale as f32,
            momentum_scale: momentum_scale as f32,
            eps2_lo: eps2.lo,
            _pad: [0.0; 3],
        };

        let storage = |label: &str, data: &[[f32; 4]]| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: bytemuck::cast_slice(data),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
            })
        };
        let pos_hi = storage("positions (hi)", &pos_hi);
        let pos_lo = storage("positions (lo)", &pos_lo);
        let vel_hi = storage("velocities (hi)", &vel_hi);
        let vel_lo = storage("velocities (lo)", &vel_lo);
        let acc = storage("accelerations", &vec![[0.0; 4]; bodies.len()]);
        let fixed = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("measurement accumulators"),
            size: FIXED_SLOTS * 8,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("n-body params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let storage_entry = |binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("n-body"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                storage_entry(1),
                storage_entry(2),
                storage_entry(3),
                storage_entry(4),
                storage_entry(5),
                storage_entry(6),
            ],
        });
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("n-body"),
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: pos_hi.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: pos_lo.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: vel_hi.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: vel_lo.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: acc.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: fixed.as_entire_binding(),
                },
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("n-body"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let source = format!(
            "{}\n{WGSL_FIXED}\n{WGSL_DF64}",
            include_str!("gravity.wgsl")
        );
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("n-body"),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
        let pipeline = |entry: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&pipeline_layout),
                module: &module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };

        let readback = (0..READBACK_RING)
            .map(|_| Readback {
                buffer: device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("measurement readback"),
                    size: FIXED_SLOTS * 8,
                    usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }),
                step: 0,
                ready: Arc::new(AtomicBool::new(false)),
                in_flight: false,
                pending_map: false,
            })
            .collect();

        Self {
            device: device.clone(),
            queue: queue.clone(),
            count,
            dt,
            steps: 0,
            primed: false,
            drift: pipeline("drift"),
            prime: pipeline("prime"),
            force_kick: pipeline("force_kick"),
            measure: pipeline("measure"),
            pos_hi,
            pos_lo,
            vel_hi,
            vel_lo,
            fixed,
            group,
            readback,
            energy_scale,
            momentum_scale,
        }
    }

    pub fn count(&self) -> u32 {
        self.count
    }

    pub fn dt(&self) -> f64 {
        self.dt
    }

    /// Steps taken so far.
    pub fn steps(&self) -> u64 {
        self.steps
    }

    /// Simulation time so far.
    pub fn time(&self) -> f64 {
        self.steps as f64 * self.dt
    }

    /// `vec4f` per body: position (hi part) in xyz, mass in w. For drawing.
    pub fn positions(&self) -> &wgpu::Buffer {
        &self.pos_hi
    }

    /// `vec4f` per body: velocity (hi part) in xyz. For drawing.
    pub fn velocities(&self) -> &wgpu::Buffer {
        &self.vel_hi
    }

    /// Workgroups for one-body-per-thread kernels (drift, measure)
    fn groups(&self) -> u32 {
        self.count.div_ceil(WORKGROUP)
    }

    /// Workgroups for the force kernels, which handle two bodies per thread
    fn force_groups(&self) -> u32 {
        self.count.div_ceil(2 * WORKGROUP)
    }

    /// Records `steps` steps into one compute pass.
    pub fn step(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        steps: u32,
        timestamp_writes: Option<wgpu::ComputePassTimestampWrites<'_>>,
    ) {
        let (groups, force_groups) = (self.groups(), self.force_groups());
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("n-body steps"),
            timestamp_writes,
        });
        pass.set_bind_group(0, &self.group, &[]);
        if !self.primed {
            pass.set_pipeline(&self.prime);
            pass.dispatch_workgroups(force_groups, 1, 1);
            self.primed = true;
        }
        for _ in 0..steps {
            pass.set_pipeline(&self.drift);
            pass.dispatch_workgroups(groups, 1, 1);
            pass.set_pipeline(&self.force_kick);
            pass.dispatch_workgroups(force_groups, 1, 1);
        }
        self.steps += steps as u64;
    }

    /// Records a measurement of the current state: every distance and mass in
    /// double-float, so the totals are limited only by fixed-point resolution.
    /// Costs about ten force passes (56 ms at 8k bodies, 213 ms at 16k on an
    /// M1), so measure periodically, not every step. Returns false (and
    /// records nothing) when all read-back buffers are still in use.
    pub fn measure(&mut self, encoder: &mut wgpu::CommandEncoder) -> bool {
        let Some(index) = self.readback.iter().position(|r| !r.in_flight) else {
            return false;
        };
        encoder.clear_buffer(&self.fixed, 0, None);
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("n-body measure"),
                timestamp_writes: None,
            });
            pass.set_bind_group(0, &self.group, &[]);
            if !self.primed {
                pass.set_pipeline(&self.prime);
                pass.dispatch_workgroups(self.force_groups(), 1, 1);
                self.primed = true;
            }
            pass.set_pipeline(&self.measure);
            pass.dispatch_workgroups(self.groups(), 1, 1);
        }
        let slot = &mut self.readback[index];
        encoder.copy_buffer_to_buffer(&self.fixed, 0, &slot.buffer, 0, FIXED_SLOTS * 8);
        slot.step = self.steps;
        slot.in_flight = true;
        slot.pending_map = true;
        true
    }

    /// Call after submitting the commands that recorded [`NBody::measure`].
    pub fn after_submit(&mut self) {
        for slot in self.readback.iter_mut().filter(|s| s.pending_map) {
            slot.pending_map = false;
            let ready = slot.ready.clone();
            slot.buffer
                .slice(..)
                .map_async(wgpu::MapMode::Read, move |result| {
                    if result.is_ok() {
                        ready.store(true, Ordering::Release);
                    }
                });
        }
    }

    fn decode(&self, words: &[u32], step: u64) -> Measurement {
        let v = fixed::decode(words);
        Measurement {
            step,
            time: step as f64 * self.dt,
            kinetic: v[0] / self.energy_scale,
            potential: v[1] / self.energy_scale,
            momentum: [v[2], v[3], v[4]].map(|p| p / self.momentum_scale),
        }
    }

    /// Finished measurements, oldest first. Natively, call
    /// `device.poll(wgpu::PollType::Poll)` regularly so they can finish.
    pub fn poll(&mut self) -> Vec<Measurement> {
        let mut done = Vec::new();
        for i in 0..self.readback.len() {
            let slot = &self.readback[i];
            if !(slot.in_flight && slot.ready.load(Ordering::Acquire)) {
                continue;
            }
            let words: Vec<u32> = {
                let view = slot.buffer.slice(..).get_mapped_range().expect("mapped");
                bytemuck::cast_slice(&view).to_vec()
            };
            slot.buffer.unmap();
            done.push(self.decode(&words, slot.step));
            let slot = &mut self.readback[i];
            slot.in_flight = false;
            slot.ready.store(false, Ordering::Release);
        }
        done.sort_by_key(|m| m.step);
        done
    }

    /// Measures now, waiting for the GPU. Native only.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn measure_now(&mut self) -> Measurement {
        let mut encoder = self.device.create_command_encoder(&Default::default());
        assert!(
            self.measure(&mut encoder),
            "a measurement is already in flight"
        );
        self.queue.submit([encoder.finish()]);
        self.after_submit();
        self.device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            })
            .expect("the GPU did not finish");
        self.poll().pop().expect("the measurement finished")
    }

    /// Advances `steps` steps, waiting for the GPU. Submits in batches so no
    /// single submission runs long. Native only.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn run(&mut self, steps: u64) {
        let mut left = steps;
        while left > 0 {
            let batch = left.min(500) as u32;
            let mut encoder = self.device.create_command_encoder(&Default::default());
            self.step(&mut encoder, batch, None);
            self.queue.submit([encoder.finish()]);
            left -= batch as u64;
        }
        self.device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            })
            .expect("the GPU did not finish");
    }

    /// Reads every body back at full double-float precision. Native only.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn read_bodies(&self) -> Vec<Body> {
        let read = |buffer: &wgpu::Buffer| -> Vec<[f32; 4]> {
            let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("read bodies"),
                size: buffer.size(),
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let mut encoder = self.device.create_command_encoder(&Default::default());
            encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, buffer.size());
            self.queue.submit([encoder.finish()]);
            staging
                .slice(..)
                .map_async(wgpu::MapMode::Read, |r| r.expect("read back"));
            self.device
                .poll(wgpu::PollType::Wait {
                    submission_index: None,
                    timeout: None,
                })
                .expect("the GPU did not finish");
            let out = bytemuck::cast_slice(&staging.slice(..).get_mapped_range().expect("mapped"))
                .to_vec();
            staging.unmap();
            out
        };
        let (ph, pl, vh, vl) = (
            read(&self.pos_hi),
            read(&self.pos_lo),
            read(&self.vel_hi),
            read(&self.vel_lo),
        );
        (0..self.count as usize)
            .map(|i| Body {
                pos: [0, 1, 2].map(|k| ph[i][k] as f64 + pl[i][k] as f64),
                vel: [0, 1, 2].map(|k| vh[i][k] as f64 + vl[i][k] as f64),
                mass: ph[i][3] as f64 + pl[i][3] as f64,
            })
            .collect()
    }
}

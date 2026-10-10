//! Lennard-Jones particles in a periodic box: the model fluid of molecular
//! dynamics, and Virial's bridge from mechanics to thermodynamics.
//!
//! Reduced units: length sigma, energy epsilon, mass m and k_B are 1, so time
//! is sigma * sqrt(m / epsilon) and temperature epsilon / k_B. For argon,
//! sigma = 3.405e-10 m, epsilon / k_B = 119.8 K, m = 6.63e-26 kg, and one time
//! unit is 2.15e-12 s.
//!
//! The potential 4 (r^-12 - r^-6) is cut at `cutoff` and shifted so it's
//! continuous there; optionally the force is shifted too (see
//! [`Settings::shifted_force`]). Temperature comes from kinetic energy; pressure from the
//! virial theorem, P = (2 KE + W) / (3 V) with W the sum of r . F over pairs.

use bytemuck::{Pod, Zeroable};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use virial_precision::{Df64, WGSL_DF64, WGSL_FIXED, fixed};
use wgpu::util::DeviceExt;

/// A particle (mass 1).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Particle {
    pub pos: [f64; 3],
    pub vel: [f64; 3],
}

/// Langevin thermostat: friction and matching random kicks hold the system at
/// `temperature`. Integrated with the BAOAB splitting.
#[derive(Clone, Copy, Debug)]
pub struct Langevin {
    pub temperature: f64,
    pub friction: f64,
    /// Seed for the per-particle random streams; same seed, same run
    pub seed: u32,
}

/// Box, cutoff, time step and thermostat.
#[derive(Clone, Copy, Debug)]
pub struct Settings {
    /// Side of the periodic cube. Rounded to the nearest f32 (see
    /// [`LennardJones::box_size`]).
    pub box_size: f64,
    pub cutoff: f64,
    pub dt: f64,
    /// `None`: constant energy (velocity Verlet)
    pub thermostat: Option<Langevin>,
    /// Shift the force as well as the energy, so both reach zero smoothly at
    /// the cutoff. Conserves energy much better; slightly changes the model,
    /// so compare thermodynamics against results for the same potential.
    pub shifted_force: bool,
    /// Extra distance kept in each neighbour list beyond the cutoff (0.3 is
    /// typical). Lists rebuild only when something has moved more than half
    /// of it.
    pub skin: f64,
}

/// Totals at one step.
#[derive(Clone, Copy, Debug)]
pub struct Measurement {
    pub step: u64,
    pub time: f64,
    pub kinetic: f64,
    pub potential: f64,
    /// Sum over pairs of r . F
    pub virial: f64,
    pub momentum: [f64; 3],
    pub count: u32,
    pub volume: f64,
    /// Neighbour-list rebuilds so far
    pub list_rebuilds: u32,
    /// 0, or the largest neighbour count that didn't fit in a list (the
    /// forces are then wrong; raise [`Settings::skin`]'s capacity margin)
    pub list_overflow: u32,
}

impl Measurement {
    pub fn energy(&self) -> f64 {
        self.kinetic + self.potential
    }

    /// Temperature from kinetic energy with `dof` degrees of freedom: 3N with
    /// a thermostat, 3N - 3 at constant energy with zero total momentum.
    pub fn temperature(&self, dof: f64) -> f64 {
        2.0 * self.kinetic / dof
    }

    /// Pressure from the virial theorem.
    pub fn pressure(&self) -> f64 {
        (2.0 * self.kinetic + self.virial) / (3.0 * self.volume)
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    count: u32,
    cells: u32,
    total_cells: u32,
    thermostat: u32,
    box_size: f32,
    cell_size: f32,
    rc2: f32,
    shift: f32,
    dt: f32,
    half_dt: f32,
    one: f32,
    energy_scale: f32,
    momentum_scale: f32,
    c1: f32,
    c2: f32,
    segment: u32,
    force_shift: f32,
    list_r2: f32,
    half_skin2: f32,
    capacity: u32,
    _pad: [u32; 3],
}

const WORKGROUP: u32 = 256;
const SLOTS: u64 = 8;
/// Words in the accumulator buffer: 8 two-word slots, then list control words
const CONTROL_WORDS: u64 = 20;
const RING: usize = 3;

struct Readback {
    buffer: wgpu::Buffer,
    step: u64,
    ready: Arc<AtomicBool>,
    in_flight: bool,
    pending_map: bool,
}

/// Lennard-Jones particles on the GPU.
#[cfg_attr(target_arch = "wasm32", allow(dead_code))]
pub struct LennardJones {
    device: wgpu::Device,
    queue: wgpu::Queue,
    count: u32,
    box_size: f64,
    dt: f64,
    steps: u64,
    primed: bool,
    segment: u32,
    state: wgpu::Buffer,
    ints: wgpu::Buffer,
    fixed: wgpu::Buffer,
    group: wgpu::BindGroup,
    decide_group: wgpu::BindGroup,
    pipelines: Pipelines,
    dispatch: wgpu::Buffer,
    readback: Vec<Readback>,
    energy_scale: f64,
    momentum_scale: f64,
}

struct Pipelines {
    begin: wgpu::ComputePipeline,
    pack: wgpu::ComputePipeline,
    decide: wgpu::ComputePipeline,
    build_list: wgpu::ComputePipeline,
    reorder_gather: wgpu::ComputePipeline,
    reorder_scatter: wgpu::ComputePipeline,
    clear_cells: wgpu::ComputePipeline,
    bin: wgpu::ComputePipeline,
    scan_blocks: wgpu::ComputePipeline,
    scan_sums: wgpu::ComputePipeline,
    scan_add: wgpu::ComputePipeline,
    scatter: wgpu::ComputePipeline,
    sort_cells: wgpu::ComputePipeline,
    prime: wgpu::ComputePipeline,
    force_kick: wgpu::ComputePipeline,
    measure: wgpu::ComputePipeline,
}

fn scale_for(magnitude: f64) -> f64 {
    if magnitude <= 0.0 || !magnitude.is_finite() {
        return 1.0;
    }
    2f64.powi((20.0 - magnitude.log2()).floor() as i32)
}

impl LennardJones {
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        particles: &[Particle],
        settings: Settings,
    ) -> Self {
        let count = particles.len() as u32;
        let box_size = settings.box_size as f32 as f64;
        let list_radius = settings.cutoff + settings.skin;
        // Cells at least the list radius wide, so the list search spans
        // 3 x 3 x 3. (Half-width cells with a 5 x 5 x 5 search measured slower:
        // the loop over mostly empty cells costs more than it saves.) Capped
        // at 101 per side: the prefix sum handles up to 1,048,576 cells.
        let cells = ((box_size / list_radius).floor() as u32).min(101);
        assert!(
            cells >= 3,
            "the box must be at least three (cutoff + skin) wide"
        );
        let total_cells = cells.pow(3);
        let cell_size = box_size / cells as f64;
        // Sections start on 256-byte boundaries, so a renderer can bind one
        let segment = count.div_ceil(16) * 16;
        // Room for twice the average neighbour count plus a margin
        let density = count as f64 / box_size.powi(3);
        let expected = 4.0 / 3.0 * std::f64::consts::PI * list_radius.powi(3) * density;
        let capacity = (2.0 * expected + 32.0).ceil() as u32;

        let split = Df64::from_f64;
        let wrap = |x: f64| x.rem_euclid(box_size);
        let mut pos_hi = Vec::new();
        let mut pos_lo = Vec::new();
        let mut vel_hi = Vec::new();
        let mut vel_lo = Vec::new();
        for p in particles {
            let x = p.pos.map(|v| split(wrap(v)));
            let v = p.vel.map(split);
            pos_hi.push([x[0].hi, x[1].hi, x[2].hi, 0.0f32]);
            pos_lo.push([x[0].lo, x[1].lo, x[2].lo, 0.0]);
            vel_hi.push([v[0].hi, v[1].hi, v[2].hi, 0.0]);
            vel_lo.push([v[0].lo, v[1].lo, v[2].lo, 0.0]);
        }

        let kinetic: f64 = particles
            .iter()
            .map(|p| 0.5 * p.vel.iter().map(|v| v * v).sum::<f64>())
            .sum();
        let thermal = settings
            .thermostat
            .map_or(0.0, |t| 1.5 * t.temperature * count as f64);
        let energy_scale = scale_for(kinetic.max(thermal).max(count as f64));
        let speed: f64 = particles
            .iter()
            .map(|p| p.vel.iter().map(|v| v * v).sum::<f64>().sqrt())
            .sum();
        let momentum_scale = scale_for(speed.max(count as f64));

        let rc2 = settings.cutoff * settings.cutoff;
        let inv6 = 1.0 / (rc2 * rc2 * rc2);
        let shift = 4.0 * (inv6 * inv6 - inv6);
        // Force magnitude at the cutoff: 24 (2 rc^-12 - rc^-6) / rc
        let force_shift = if settings.shifted_force {
            24.0 * (2.0 * inv6 * inv6 - inv6) / settings.cutoff
        } else {
            0.0
        };
        let (thermostat, c1, c2) = match settings.thermostat {
            Some(t) => {
                let c1 = (-t.friction * settings.dt).exp();
                (1, c1, ((1.0 - c1 * c1) * t.temperature).sqrt())
            }
            None => (0, 1.0, 0.0),
        };
        let params = Params {
            count,
            cells,
            total_cells,
            thermostat,
            box_size: box_size as f32,
            cell_size: cell_size as f32,
            rc2: rc2 as f32,
            shift: shift as f32,
            dt: settings.dt as f32,
            half_dt: (settings.dt * 0.5) as f32,
            one: 1.0,
            energy_scale: energy_scale as f32,
            momentum_scale: momentum_scale as f32,
            c1: c1 as f32,
            c2: c2 as f32,
            segment,
            force_shift: force_shift as f32,
            list_r2: (list_radius * list_radius) as f32,
            half_skin2: (0.25 * settings.skin * settings.skin) as f32,
            capacity,
            _pad: [0; 3],
        };

        let usage = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST;
        let init = |label: &str, contents: &[u8]| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents,
                usage,
            })
        };
        let zeroed = |label: &str, size: u64| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: size.max(16),
                usage,
                mapped_at_creation: false,
            })
        };
        let seed = settings.thermostat.map_or(1, |t| t.seed);
        let rng: Vec<u32> = (0..count)
            .map(|i| {
                // splitmix-style hash so neighbouring particles get unrelated streams
                let mut z = (seed as u64) << 32 | i as u64;
                z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
                z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
                (z ^ (z >> 31)) as u32
            })
            .collect();
        let seg = segment as usize;
        let mut state = vec![[0.0f32; 4]; 7 * seg];
        for i in 0..particles.len() {
            state[i] = pos_hi[i];
            state[seg + i] = pos_lo[i];
            state[2 * seg + i] = vel_hi[i];
            state[3 * seg + i] = vel_lo[i];
        }
        let mut ints = vec![0u32; 4 * seg];
        ints[2 * seg..2 * seg + rng.len()].copy_from_slice(&rng);
        for i in 0..particles.len() {
            ints[3 * seg + i] = i as u32;
        }
        let state = init("lj state", bytemuck::cast_slice(&state));
        let fixed = zeroed("lj accumulators and list control", CONTROL_WORDS * 4);
        let neighbors = zeroed(
            "neighbour lists",
            segment as u64 * (capacity as u64 + 1) * 4,
        );
        let scratch = zeroed("reorder scratch", 8 * segment as u64 * 16);
        let dispatch = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("list rebuild dispatch sizes"),
            size: 12 * 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::INDIRECT,
            mapped_at_creation: false,
        });
        let cell_count = zeroed("cell counts", total_cells as u64 * 4);
        let cells_buffer = zeroed(
            "cell starts and block sums",
            (total_cells as u64 + 1 + 4096) * 4,
        );
        let ints = init(
            "cells of particles, cell order, random state",
            bytemuck::cast_slice(&ints),
        );
        let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("lj params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let all: [(u32, &wgpu::Buffer); 9] = [
            (0, &params_buffer),
            (1, &state),
            (2, &fixed),
            (3, &cell_count),
            (4, &cells_buffer),
            (5, &ints),
            (6, &neighbors),
            (7, &dispatch),
            (8, &scratch),
        ];
        // `decide` writes the dispatch sizes, which the rebuild kernels then use
        // as indirect arguments. A buffer can't be bound for writing while it
        // supplies indirect arguments, so only `decide`'s group includes it.
        let layout_entry = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: if binding == 0 {
                    wgpu::BufferBindingType::Uniform
                } else {
                    wgpu::BufferBindingType::Storage { read_only: false }
                },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let make_group = |label: &str, bindings: &[u32]| {
            let chosen: Vec<_> = all.iter().filter(|(b, _)| bindings.contains(b)).collect();
            let entries: Vec<_> = chosen.iter().map(|(b, _)| layout_entry(*b)).collect();
            let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some(label),
                entries: &entries,
            });
            let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(label),
                layout: &layout,
                entries: &chosen
                    .iter()
                    .map(|(b, buffer)| wgpu::BindGroupEntry {
                        binding: *b,
                        resource: buffer.as_entire_binding(),
                    })
                    .collect::<Vec<_>>(),
            });
            let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(label),
                bind_group_layouts: &[Some(&layout)],
                immediate_size: 0,
            });
            (group, pipeline_layout)
        };
        // Everything but the dispatch sizes (7 storage buffers, within
        // WebGPU's default of 8 per shader)
        let (group, pipeline_layout) = make_group("lennard-jones", &[0, 1, 2, 3, 4, 5, 6, 8]);
        let (decide_group, decide_layout) = make_group("lennard-jones decide", &[0, 2, 7]);
        let source = format!(
            "{}
{}
{WGSL_FIXED}
{WGSL_DF64}",
            include_str!("lennard_jones.wgsl"),
            include_str!("grid.wgsl")
        );
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("lennard-jones"),
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
        let pipelines = Pipelines {
            begin: pipeline("begin"),
            pack: pipeline("pack"),
            decide: device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("decide"),
                layout: Some(&decide_layout),
                module: &module,
                entry_point: Some("decide"),
                compilation_options: Default::default(),
                cache: None,
            }),
            build_list: pipeline("build_list"),
            reorder_gather: pipeline("reorder_gather"),
            reorder_scatter: pipeline("reorder_scatter"),
            clear_cells: pipeline("clear_cells"),
            bin: pipeline("bin"),
            scan_blocks: pipeline("scan_blocks"),
            scan_sums: pipeline("scan_sums"),
            scan_add: pipeline("scan_add"),
            scatter: pipeline("scatter"),
            sort_cells: pipeline("sort_cells"),
            prime: pipeline("prime"),
            force_kick: pipeline("force_kick"),
            measure: pipeline("measure"),
        };
        let readback = (0..RING)
            .map(|_| Readback {
                buffer: device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("lj readback"),
                    size: CONTROL_WORDS * 4,
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
            box_size,
            dt: settings.dt,
            steps: 0,
            primed: false,
            segment,
            state,
            ints,
            fixed,
            group,
            decide_group,
            pipelines,
            dispatch,
            readback,
            energy_scale,
            momentum_scale,
        }
    }

    pub fn count(&self) -> u32 {
        self.count
    }

    /// The box side actually used: the requested size rounded to f32, so the
    /// GPU and any CPU check agree exactly.
    pub fn box_size(&self) -> f64 {
        self.box_size
    }

    pub fn volume(&self) -> f64 {
        self.box_size.powi(3)
    }

    pub fn steps(&self) -> u64 {
        self.steps
    }

    pub fn time(&self) -> f64 {
        self.steps as f64 * self.dt
    }

    fn section(&self, index: u64) -> wgpu::BindingResource<'_> {
        let bytes = self.segment as u64 * 16;
        wgpu::BindingResource::Buffer(wgpu::BufferBinding {
            buffer: &self.state,
            offset: index * bytes,
            size: wgpu::BufferSize::new(bytes),
        })
    }

    /// `vec4f` per particle, position (hi part) in xyz. For drawing.
    pub fn positions(&self) -> wgpu::BindingResource<'_> {
        self.section(0)
    }

    /// `vec4f` per particle, velocity (hi part) in xyz. For drawing.
    pub fn velocities(&self) -> wgpu::BindingResource<'_> {
        self.section(2)
    }

    /// Decides on the GPU whether the neighbour lists need rebuilding, then
    /// runs the rebuild kernels with dispatch sizes that are zero when not.
    fn update_lists(&self, pass: &mut wgpu::ComputePass<'_>) {
        let p = &self.pipelines;
        pass.set_bind_group(0, &self.decide_group, &[]);
        pass.set_pipeline(&p.decide);
        pass.dispatch_workgroups(1, 1, 1);
        pass.set_bind_group(0, &self.group, &[]);
        // Byte offsets of the (cells), (particles), (single) and (reorder) sizes
        let (cells, particles, single, reorder) = (0, 12, 24, 36);
        for (pipeline, offset) in [
            (&p.clear_cells, cells),
            (&p.bin, particles),
            (&p.scan_blocks, cells),
            (&p.scan_sums, single),
            (&p.scan_add, cells),
            (&p.scatter, particles),
            (&p.sort_cells, cells),
            (&p.reorder_gather, reorder),
            (&p.reorder_scatter, reorder),
            (&p.build_list, particles),
        ] {
            pass.set_pipeline(pipeline);
            pass.dispatch_workgroups_indirect(&self.dispatch, offset);
        }
    }

    fn prime_if_needed(&mut self, pass: &mut wgpu::ComputePass<'_>) {
        if !self.primed {
            pass.set_pipeline(&self.pipelines.pack);
            pass.dispatch_workgroups(self.count.div_ceil(WORKGROUP), 1, 1);
            self.update_lists(pass);
            pass.set_pipeline(&self.pipelines.prime);
            pass.dispatch_workgroups(self.count.div_ceil(WORKGROUP), 1, 1);
            self.primed = true;
        }
    }

    /// Records `steps` steps into one compute pass.
    pub fn step(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        steps: u32,
        timestamp_writes: Option<wgpu::ComputePassTimestampWrites<'_>>,
    ) {
        let particles = self.count.div_ceil(WORKGROUP);
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("lennard-jones steps"),
            timestamp_writes,
        });
        pass.set_bind_group(0, &self.group, &[]);
        self.prime_if_needed(&mut pass);
        for _ in 0..steps {
            pass.set_pipeline(&self.pipelines.begin);
            pass.dispatch_workgroups(particles, 1, 1);
            self.update_lists(&mut pass);
            pass.set_pipeline(&self.pipelines.force_kick);
            pass.dispatch_workgroups(particles, 1, 1);
        }
        self.steps += steps as u64;
    }

    /// Records a measurement. Returns false when all read-back buffers are busy.
    pub fn measure(&mut self, encoder: &mut wgpu::CommandEncoder) -> bool {
        let Some(index) = self.readback.iter().position(|r| !r.in_flight) else {
            return false;
        };
        // Only the measurement slots: the words after them control the lists
        encoder.clear_buffer(&self.fixed, 0, Some(SLOTS * 8));
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("lennard-jones measure"),
                timestamp_writes: None,
            });
            pass.set_bind_group(0, &self.group, &[]);
            self.prime_if_needed(&mut pass);
            pass.set_pipeline(&self.pipelines.measure);
            pass.dispatch_workgroups(self.count.div_ceil(WORKGROUP), 1, 1);
        }
        let slot = &mut self.readback[index];
        encoder.copy_buffer_to_buffer(&self.fixed, 0, &slot.buffer, 0, CONTROL_WORDS * 4);
        slot.step = self.steps;
        slot.in_flight = true;
        slot.pending_map = true;
        true
    }

    /// Call after submitting the commands that recorded [`LennardJones::measure`].
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

    /// Finished measurements, oldest first.
    pub fn poll(&mut self) -> Vec<Measurement> {
        let mut done = Vec::new();
        for slot in &mut self.readback {
            if !(slot.in_flight && slot.ready.load(Ordering::Acquire)) {
                continue;
            }
            let words: Vec<u32> = {
                let view = slot.buffer.slice(..).get_mapped_range().expect("mapped");
                bytemuck::cast_slice(&view).to_vec()
            };
            slot.buffer.unmap();
            slot.in_flight = false;
            slot.ready.store(false, Ordering::Release);
            let v = fixed::decode(&words[..SLOTS as usize * 2]);
            done.push(Measurement {
                step: slot.step,
                time: slot.step as f64 * self.dt,
                kinetic: v[0] / self.energy_scale,
                potential: v[1] / self.energy_scale,
                virial: v[2] / self.energy_scale,
                momentum: [v[3], v[4], v[5]].map(|p| p / self.momentum_scale),
                count: self.count,
                volume: self.box_size.powi(3),
                list_rebuilds: words[17],
                list_overflow: words[18],
            });
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
        self.wait();
        self.poll().pop().expect("the measurement finished")
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn wait(&self) {
        self.device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            })
            .expect("the GPU did not finish");
    }

    /// Advances `steps` steps, waiting for the GPU. Native only.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn run(&mut self, steps: u64) {
        let mut left = steps;
        while left > 0 {
            let batch = left.min(200) as u32;
            let mut encoder = self.device.create_command_encoder(&Default::default());
            self.step(&mut encoder, batch, None);
            self.queue.submit([encoder.finish()]);
            left -= batch as u64;
        }
        self.wait();
    }

    /// Reads every particle back at full precision, in the order they were
    /// given (the GPU keeps them in cell order). Native only.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn read_particles(&self) -> Vec<Particle> {
        let read = |buffer: &wgpu::Buffer| -> Vec<u8> {
            let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("read particles"),
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
            self.wait();
            let out = staging
                .slice(..)
                .get_mapped_range()
                .expect("mapped")
                .to_vec();
            staging.unmap();
            out
        };
        let state: Vec<[f32; 4]> = bytemuck::cast_slice(&read(&self.state)).to_vec();
        let ints: Vec<u32> = bytemuck::cast_slice(&read(&self.ints)).to_vec();
        let seg = self.segment as usize;
        let mut out = vec![
            Particle {
                pos: [0.0; 3],
                vel: [0.0; 3]
            };
            self.count as usize
        ];
        for slot in 0..self.count as usize {
            out[ints[3 * seg + slot] as usize] = Particle {
                pos: [0, 1, 2].map(|k| state[slot][k] as f64 + state[seg + slot][k] as f64),
                vel: [0, 1, 2]
                    .map(|k| state[2 * seg + slot][k] as f64 + state[3 * seg + slot][k] as f64),
            };
        }
        out
    }
}

/// `count` particles on a simple cubic lattice filling the box, with random
/// velocities at `temperature` and zero total momentum. Deterministic.
pub fn lattice(count: usize, box_size: f64, temperature: f64, seed: u64) -> Vec<Particle> {
    let per_side = (count as f64).cbrt().ceil() as usize;
    let spacing = box_size / per_side as f64;
    let mut s = seed | 1;
    let mut random = move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        ((s >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    };
    let mut gaussian = move || {
        let r = (-2.0 * random().ln()).sqrt();
        r * (std::f64::consts::TAU * random()).cos()
    };
    let mut particles: Vec<Particle> = (0..count)
        .map(|i| {
            let (x, y, z) = (
                i % per_side,
                (i / per_side) % per_side,
                i / (per_side * per_side),
            );
            Particle {
                pos: [x, y, z].map(|c| (c as f64 + 0.5) * spacing),
                vel: [0; 3].map(|_| gaussian() * temperature.sqrt()),
            }
        })
        .collect();
    let mean: [f64; 3] =
        [0, 1, 2].map(|k| particles.iter().map(|p| p.vel[k]).sum::<f64>() / count as f64);
    for p in &mut particles {
        for (v, m) in p.vel.iter_mut().zip(mean) {
            *v -= m;
        }
    }
    particles
}

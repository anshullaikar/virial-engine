//! Granular material: spheres in contact, in SI units.
//!
//! Normal contacts follow Hertz elasticity with viscoelastic damping
//! (Brilliantov et al. 1996); tangential contacts follow regularised Coulomb
//! friction. Gravity points along -z. Grains sit in a square box of side
//! `box_size` with a floor that can tilt about the y axis, so a pile can be
//! released down a slope.
//!
//! Grains spin: friction acts at the contact point (including the surfaces'
//! spin) and turns them; optional rolling resistance opposes spin. Friction
//! has no memory (no tangential spring), so it resists sliding but can't hold
//! a steep pile still forever.
//!
//! Stiffness: real glass has E* around 4e10 Pa, which needs time steps near
//! 1e-7 s. Granular simulations commonly soften E* (here by default to 2e6
//! Pa): contacts last longer, so steps can be larger, while flow behaviour
//! changes little. Contact time scales as E*^(-2/5).

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use bytemuck::{Pod, Zeroable};
use virial_precision::{Df64, WGSL_DF64, WGSL_FIXED, fixed};
use wgpu::util::DeviceExt;

/// A grain.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Grain {
    pub pos: [f64; 3],
    pub vel: [f64; 3],
    /// Angular velocity, rad/s
    pub spin: [f64; 3],
    pub radius: f64,
}

/// Material, box and time step. All SI.
#[derive(Clone, Copy, Debug)]
pub struct Settings {
    pub box_size: f64,
    pub gravity: f64,
    /// Grain density, kg/m^3
    pub density: f64,
    /// Effective Young's modulus E*, Pa
    pub modulus: f64,
    /// Dissipative constant A, s (0: perfectly elastic)
    pub damping: f64,
    /// Coulomb friction coefficient
    pub friction: f64,
    /// Viscous regularisation of friction at low sliding speed, N s/m
    pub friction_gamma: f64,
    /// Rolling resistance coefficient mu_r: torque up to mu_r r* F_n against
    /// relative spin (0: none)
    pub rolling_friction: f64,
    /// Floor tilt about the y axis, radians; it slopes down towards +x
    pub floor_angle: f64,
    /// Floor height at x = box_size
    pub floor_height: f64,
    pub dt: f64,
}

impl Settings {
    /// Glass beads (density 2500 kg/m^3, friction 0.5) with softened
    /// stiffness, in a box of side `box_size` on a flat floor.
    pub fn glass(box_size: f64, dt: f64) -> Self {
        Self {
            box_size,
            gravity: 9.81,
            density: 2500.0,
            modulus: 2e6,
            damping: 2e-5,
            friction: 0.5,
            friction_gamma: 0.05,
            rolling_friction: 0.0,
            floor_angle: 0.0,
            floor_height: 0.0,
            dt,
        }
    }
}

/// Totals at one step.
#[derive(Clone, Copy, Debug)]
pub struct Measurement {
    pub step: u64,
    pub time: f64,
    /// Kinetic energy of translation
    pub kinetic: f64,
    /// Kinetic energy of spin
    pub rotational: f64,
    /// m g z summed over grains
    pub gravitational: f64,
    /// Energy stored in compressed contacts
    pub elastic: f64,
    pub momentum: [f64; 3],
    /// Upward force from the floor and walls
    pub support: f64,
    pub grains: u32,
}

impl Measurement {
    pub fn energy(&self) -> f64 {
        self.kinetic + self.rotational + self.gravitational + self.elastic
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    count: u32,
    cells: u32,
    total_cells: u32,
    segment: u32,
    cell_size: f32,
    box_size: f32,
    dt: f32,
    half_dt: f32,
    one: f32,
    gravity: f32,
    density: f32,
    modulus: f32,
    damping: f32,
    friction: f32,
    friction_gamma: f32,
    energy_scale: f32,
    momentum_scale: f32,
    force_scale: f32,
    floor_nx: f32,
    floor_nz: f32,
    floor_offset: f32,
    rolling_friction: f32,
    _pad: [f32; 2],
}

const WORKGROUP: u32 = 256;
const SLOTS: u64 = 8;
const RING: usize = 3;

struct Readback {
    buffer: wgpu::Buffer,
    step: u64,
    ready: Arc<AtomicBool>,
    in_flight: bool,
    pending_map: bool,
}

/// Grains on the GPU.
#[cfg_attr(target_arch = "wasm32", allow(dead_code))]
pub struct Granular {
    device: wgpu::Device,
    queue: wgpu::Queue,
    count: u32,
    total_cells: u32,
    segment: u32,
    dt: f64,
    steps: u64,
    primed: bool,
    state: wgpu::Buffer,
    fixed: wgpu::Buffer,
    group: wgpu::BindGroup,
    pipelines: Vec<(&'static str, wgpu::ComputePipeline)>,
    readback: Vec<Readback>,
    energy_scale: f64,
    momentum_scale: f64,
    force_scale: f64,
}

fn scale_for(magnitude: f64) -> f64 {
    if magnitude <= 0.0 || !magnitude.is_finite() {
        return 1.0;
    }
    2f64.powi((20.0 - magnitude.log2()).floor() as i32)
}

impl Granular {
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        grains: &[Grain],
        settings: Settings,
    ) -> Self {
        let count = grains.len() as u32;
        let r_max = grains.iter().map(|g| g.radius).fold(0.0, f64::max);
        let cells = ((settings.box_size / (2.0 * r_max)).floor() as u32).clamp(1, 64);
        let total_cells = cells.pow(3);
        let cell_size = settings.box_size / cells as f64;
        let segment = count.div_ceil(16) * 16;
        let seg = segment as usize;

        let mut state = vec![[0.0f32; 4]; 7 * seg];
        for (i, g) in grains.iter().enumerate() {
            let x = g.pos.map(Df64::from_f64);
            let v = g.vel.map(Df64::from_f64);
            state[i] = [x[0].hi, x[1].hi, x[2].hi, g.radius as f32];
            state[seg + i] = [x[0].lo, x[1].lo, x[2].lo, 0.0];
            state[2 * seg + i] = [v[0].hi, v[1].hi, v[2].hi, 0.0];
            state[3 * seg + i] = [v[0].lo, v[1].lo, v[2].lo, 0.0];
            state[5 * seg + i] = [g.spin[0] as f32, g.spin[1] as f32, g.spin[2] as f32, 0.0];
        }

        let mass = |r: f64| settings.density * 4.0 / 3.0 * std::f64::consts::PI * r.powi(3);
        let total_mass: f64 = grains.iter().map(|g| mass(g.radius)).sum();
        let weight = total_mass * settings.gravity;
        let energy_scale = scale_for(weight * settings.box_size);
        let momentum_scale =
            scale_for(total_mass * (2.0 * settings.gravity * settings.box_size).sqrt());
        let force_scale = scale_for(weight * 4.0);

        let (s, c) = settings.floor_angle.sin_cos();
        let floor_offset = s * settings.box_size + c * settings.floor_height;
        let params = Params {
            count,
            cells,
            total_cells,
            segment,
            cell_size: cell_size as f32,
            box_size: settings.box_size as f32,
            dt: settings.dt as f32,
            half_dt: (settings.dt * 0.5) as f32,
            one: 1.0,
            gravity: settings.gravity as f32,
            density: settings.density as f32,
            modulus: settings.modulus as f32,
            damping: settings.damping as f32,
            friction: settings.friction as f32,
            friction_gamma: settings.friction_gamma as f32,
            energy_scale: energy_scale as f32,
            momentum_scale: momentum_scale as f32,
            force_scale: force_scale as f32,
            floor_nx: s as f32,
            floor_nz: c as f32,
            floor_offset: floor_offset as f32,
            rolling_friction: settings.rolling_friction as f32,
            _pad: [0.0; 2],
        };

        let usage = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST;
        let zeroed = |label: &str, size: u64| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: size.max(16),
                usage,
                mapped_at_creation: false,
            })
        };
        let state = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("grain state"),
            contents: bytemuck::cast_slice(&state),
            usage,
        });
        let fixed = zeroed("grain accumulators", SLOTS * 8);
        let cell_count = zeroed("cell counts", total_cells as u64 * 4);
        let cells_buffer = zeroed(
            "cell starts and block sums",
            (total_cells as u64 + 1 + 1024) * 4,
        );
        let ints = zeroed("cells of grains and cell order", 3 * segment as u64 * 4);
        let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("granular params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let buffers = [
            &params_buffer,
            &state,
            &fixed,
            &cell_count,
            &cells_buffer,
            &ints,
        ];
        let entries: Vec<wgpu::BindGroupLayoutEntry> = (0..buffers.len() as u32)
            .map(|binding| wgpu::BindGroupLayoutEntry {
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
            })
            .collect();
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("granular"),
            entries: &entries,
        });
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("granular"),
            layout: &layout,
            entries: &buffers
                .iter()
                .enumerate()
                .map(|(i, b)| wgpu::BindGroupEntry {
                    binding: i as u32,
                    resource: b.as_entire_binding(),
                })
                .collect::<Vec<_>>(),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("granular"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let source = format!(
            "{}\n{}\n{WGSL_FIXED}\n{WGSL_DF64}",
            include_str!("granular.wgsl"),
            include_str!("grid.wgsl")
        );
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("granular"),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
        let pipelines = [
            "begin",
            "clear_cells",
            "bin",
            "scan_blocks",
            "scan_sums",
            "scan_add",
            "scatter",
            "sort_cells",
            "prime",
            "force_kick",
            "measure",
        ]
        .map(|entry| {
            let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&pipeline_layout),
                module: &module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            });
            (entry, pipeline)
        })
        .to_vec();
        let readback = (0..RING)
            .map(|_| Readback {
                buffer: device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("granular readback"),
                    size: SLOTS * 8,
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
            total_cells,
            segment,
            dt: settings.dt,
            steps: 0,
            primed: false,
            state,
            fixed,
            group,
            pipelines,
            readback,
            energy_scale,
            momentum_scale,
            force_scale,
        }
    }

    fn pipeline(&self, name: &str) -> &wgpu::ComputePipeline {
        &self
            .pipelines
            .iter()
            .find(|(n, _)| *n == name)
            .expect("known pipeline")
            .1
    }

    pub fn count(&self) -> u32 {
        self.count
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

    /// `vec4f` per grain: position (hi part) in xyz, radius in w. For drawing.
    pub fn positions(&self) -> wgpu::BindingResource<'_> {
        self.section(0)
    }

    /// `vec4f` per grain: velocity (hi part) in xyz. For drawing.
    pub fn velocities(&self) -> wgpu::BindingResource<'_> {
        self.section(2)
    }

    fn rebuild_cells(&self, pass: &mut wgpu::ComputePass<'_>) {
        let grains = self.count.div_ceil(WORKGROUP);
        let cells = self.total_cells.div_ceil(WORKGROUP);
        for (name, groups) in [
            ("clear_cells", cells),
            ("bin", grains),
            ("scan_blocks", cells),
            ("scan_sums", 1),
            ("scan_add", cells),
            ("scatter", grains),
            ("sort_cells", cells),
        ] {
            pass.set_pipeline(self.pipeline(name));
            pass.dispatch_workgroups(groups, 1, 1);
        }
    }

    fn prime_if_needed(&mut self, pass: &mut wgpu::ComputePass<'_>) {
        if !self.primed {
            self.rebuild_cells(pass);
            pass.set_pipeline(self.pipeline("prime"));
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
        let grains = self.count.div_ceil(WORKGROUP);
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("granular steps"),
            timestamp_writes,
        });
        pass.set_bind_group(0, &self.group, &[]);
        self.prime_if_needed(&mut pass);
        for _ in 0..steps {
            pass.set_pipeline(self.pipeline("begin"));
            pass.dispatch_workgroups(grains, 1, 1);
            self.rebuild_cells(&mut pass);
            pass.set_pipeline(self.pipeline("force_kick"));
            pass.dispatch_workgroups(grains, 1, 1);
        }
        self.steps += steps as u64;
    }

    /// Records a measurement. Returns false when all read-back buffers are busy.
    pub fn measure(&mut self, encoder: &mut wgpu::CommandEncoder) -> bool {
        let Some(index) = self.readback.iter().position(|r| !r.in_flight) else {
            return false;
        };
        encoder.clear_buffer(&self.fixed, 0, None);
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("granular measure"),
                timestamp_writes: None,
            });
            pass.set_bind_group(0, &self.group, &[]);
            self.prime_if_needed(&mut pass);
            pass.set_pipeline(self.pipeline("measure"));
            pass.dispatch_workgroups(self.count.div_ceil(WORKGROUP), 1, 1);
        }
        let slot = &mut self.readback[index];
        encoder.copy_buffer_to_buffer(&self.fixed, 0, &slot.buffer, 0, SLOTS * 8);
        slot.step = self.steps;
        slot.in_flight = true;
        slot.pending_map = true;
        true
    }

    /// Call after submitting the commands that recorded [`Granular::measure`].
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
            let v = fixed::decode(&words);
            done.push(Measurement {
                step: slot.step,
                time: slot.step as f64 * self.dt,
                kinetic: v[0] / self.energy_scale,
                gravitational: v[1] / self.energy_scale,
                elastic: v[2] / self.energy_scale,
                rotational: v[7] / self.energy_scale,
                momentum: [v[3], v[4], v[5]].map(|p| p / self.momentum_scale),
                support: v[6] / self.force_scale,
                grains: self.count,
            });
        }
        done.sort_by_key(|m| m.step);
        done
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

    /// Reads every grain back. Native only.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn read_grains(&self) -> Vec<Grain> {
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("read grains"),
            size: self.state.size(),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = self.device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(&self.state, 0, &staging, 0, self.state.size());
        self.queue.submit([encoder.finish()]);
        staging
            .slice(..)
            .map_async(wgpu::MapMode::Read, |r| r.expect("read back"));
        self.wait();
        let state: Vec<[f32; 4]> =
            bytemuck::cast_slice(&staging.slice(..).get_mapped_range().expect("mapped")).to_vec();
        staging.unmap();
        let seg = self.segment as usize;
        (0..self.count as usize)
            .map(|i| Grain {
                pos: [0, 1, 2].map(|k| state[i][k] as f64 + state[seg + i][k] as f64),
                vel: [0, 1, 2].map(|k| state[2 * seg + i][k] as f64 + state[3 * seg + i][k] as f64),
                spin: [0, 1, 2].map(|k| state[5 * seg + i][k] as f64),
                radius: state[i][3] as f64,
            })
            .collect()
    }
}

/// Grains of radius `radius` (with a small deterministic spread) packed on a
/// loose cubic lattice inside the box region `min`..`max`.
pub fn block(min: [f64; 3], max: [f64; 3], radius: f64, spread: f64, seed: u64) -> Vec<Grain> {
    let spacing = 2.0 * radius * (1.0 + spread) * 1.02;
    let counts: Vec<usize> = (0..3)
        .map(|k| ((max[k] - min[k]) / spacing).floor().max(1.0) as usize)
        .collect();
    let mut s = seed | 1;
    let mut random = move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s >> 11) as f64 / (1u64 << 53) as f64
    };
    let mut grains = Vec::new();
    for z in 0..counts[2] {
        for y in 0..counts[1] {
            for x in 0..counts[0] {
                let jitter = |r: f64| (r - 0.5) * 0.02 * radius;
                grains.push(Grain {
                    pos: [
                        min[0] + (x as f64 + 0.5) * spacing + jitter(random()),
                        min[1] + (y as f64 + 0.5) * spacing + jitter(random()),
                        min[2] + (z as f64 + 0.5) * spacing + jitter(random()),
                    ],
                    vel: [0.0; 3],
                    spin: [0.0; 3],
                    radius: radius * (1.0 + spread * (2.0 * random() - 1.0)),
                });
            }
        }
    }
    grains
}

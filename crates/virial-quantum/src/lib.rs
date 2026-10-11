//! Quantum motion: the time-dependent Schrödinger equation on the GPU.
//!
//! [`Wave`] evolves a wavefunction on an N × N periodic grid centred on the
//! origin, in natural units (ħ = m = 1), by the split-operator method:
//! half a potential step, a kinetic step in momentum space (GPU FFT), half a
//! potential step. Each factor is a pure phase, so probability is conserved
//! up to rounding; an optional absorbing boundary removes waves reaching the
//! edges instead of letting them wrap around. In imaginary time the same
//! machinery relaxes any starting state to the ground state.
//!
//! Measurements: probability, ⟨x⟩, ⟨y⟩, spreads, ⟨V⟩ and ⟨T⟩ (from the
//! momentum-space wavefunction), summed in fixed point.

use std::f64::consts::{PI, TAU};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use bytemuck::{Pod, Zeroable};
use virial_precision::{WGSL_FIXED, fixed};
use wgpu::util::DeviceExt;

/// Real time evolves the state; imaginary time relaxes it to the ground
/// state (renormalising every step).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Time {
    Real,
    Imaginary,
}

/// A layer near the box edges that absorbs outgoing waves: a negative
/// imaginary potential rising as ((width - d) / width)² over the last
/// `width` of each side, up to `strength`.
#[derive(Clone, Copy, Debug)]
pub struct Absorber {
    pub width: f64,
    pub strength: f64,
}

/// Grid, time step and boundary.
#[derive(Clone, Copy, Debug)]
pub struct Settings {
    /// Cells per side; a power of two
    pub n: u32,
    /// Side of the square box, centred on the origin
    pub length: f64,
    pub dt: f64,
    pub time: Time,
    pub absorber: Option<Absorber>,
    /// A screen along the line x = this value, recording the probability
    /// current through it, integrated over time. See [`Wave::read_detector`].
    pub detector: Option<f64>,
}

/// Expectation values at one step. Spreads are standard deviations.
#[derive(Clone, Copy, Debug)]
pub struct Measurement {
    pub step: u64,
    pub time: f64,
    /// Total probability (1 unless absorbed)
    pub norm: f64,
    pub mean: [f64; 2],
    pub spread: [f64; 2],
    pub potential: f64,
    pub kinetic: f64,
    /// Probability inside the absorbing layer
    pub in_absorber: f64,
}

impl Measurement {
    /// ⟨H⟩ per unit probability
    pub fn energy(&self) -> f64 {
        (self.potential + self.kinetic) / self.norm
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Pass {
    stage: u32,
    axis: u32,
    inverse: u32,
    src: u32,
    dst: u32,
    _pad: [u32; 3],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Grid {
    n: u32,
    log2n: u32,
    cells: u32,
    measure_base: u32,
    dx: f32,
    scale_norm: f32,
    scale_x: f32,
    scale_x2: f32,
    scale_v: f32,
    scale_t: f32,
    dt: f32,
    detector_column: u32,
}

/// Dynamic uniform offsets must be multiples of 256 bytes
const PASS_STRIDE: u64 = 256;
/// Largest grid side whose rows fit in workgroup memory (ROW_MAX in the shader)
const ROW_MAX: u32 = 1024;

/// How the 2D FFTs run.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Fft {
    /// A whole row or column per workgroup, in workgroup memory: two
    /// dispatches per transform. The default up to n = 1024.
    Rows,
    /// One radix-2 Stockham pass over the grid per dispatch, 2 log2 n of
    /// them per transform. Any n; the default above 1024.
    Passes,
}
const WORKGROUP: u32 = 256;
const SLOTS: u64 = 8;
const RING: usize = 3;

/// The f32 pair near (cos a, sin a) whose magnitude is closest to one.
/// Plain rounding lands each factor slightly inside or outside the unit
/// circle; with dozens of FFT passes per step that bias drains (or adds)
/// probability.
fn unit_phase(angle: f64) -> [f32; 2] {
    let (c, s) = (angle.cos(), angle.sin());
    let step = |x: f32, d: i32| -> f32 {
        let bits = x.to_bits() as i64 + if x >= 0.0 { d as i64 } else { -d as i64 };
        f32::from_bits(bits as u32)
    };
    let (c0, s0) = (c as f32, s as f32);
    let mut best = [c0, s0];
    let mut best_err = f64::MAX;
    for dc in -2..=2 {
        for ds in -2..=2 {
            let (cc, ss) = (step(c0, dc), step(s0, ds));
            let magnitude = ((cc as f64).powi(2) + (ss as f64).powi(2) - 1.0).abs();
            // Prefer the unit magnitude, then the nearest angle
            let err = magnitude * 1e3 + (cc as f64 - c).abs() + (ss as f64 - s).abs();
            if err < best_err {
                best_err = err;
                best = [cc, ss];
            }
        }
    }
    best
}

/// The grid column nearest the detector line. Samples sit at
/// x = (i - N/2) dx.
fn detector_column(settings: &Settings) -> Option<u32> {
    let x = settings.detector?;
    let column = (x * settings.n as f64 / settings.length).round() + (settings.n / 2) as f64;
    assert!(
        column >= 3.0 && column < settings.n as f64 - 3.0,
        "the detector must lie inside the box"
    );
    Some(column as u32)
}

fn scale_for(magnitude: f64) -> f64 {
    if magnitude <= 0.0 || !magnitude.is_finite() {
        return 1.0;
    }
    2f64.powi((20.0 - magnitude.log2()).floor() as i32)
}

struct Readback {
    buffer: wgpu::Buffer,
    step: u64,
    ready: Arc<AtomicBool>,
    in_flight: bool,
    pending_map: bool,
}

/// A wavefunction on the GPU.
#[cfg_attr(target_arch = "wasm32", allow(dead_code))]
/// Everything bound alongside the wavefunction, accumulators and detector
struct Buffers {
    layout: wgpu::BindGroupLayout,
    passes: wgpu::Buffer,
    grid: wgpu::Buffer,
    twiddles: wgpu::Buffer,
    factors: wgpu::Buffer,
    fields: wgpu::Buffer,
}

impl Buffers {
    /// The bind group, with `fixed` as the accumulators (binding 6)
    fn group(
        &self,
        device: &wgpu::Device,
        psi: &wgpu::Buffer,
        fixed: &wgpu::Buffer,
        detector: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        fn whole(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
            wgpu::BindGroupEntry {
                binding,
                resource: buffer.as_entire_binding(),
            }
        }
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("wave"),
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &self.passes,
                        offset: 0,
                        size: wgpu::BufferSize::new(size_of::<Pass>() as u64),
                    }),
                },
                whole(1, &self.grid),
                whole(2, psi),
                whole(3, &self.twiddles),
                whole(4, &self.factors),
                whole(5, &self.fields),
                whole(6, fixed),
                whole(7, detector),
            ],
        })
    }
}

/// A recorded autocorrelation series
#[cfg_attr(target_arch = "wasm32", allow(dead_code))]
struct Series {
    capacity: u32,
    buffer: wgpu::Buffer,
    group: wgpu::BindGroup,
    samples: u32,
}

pub struct Wave {
    device: wgpu::Device,
    queue: wgpu::Queue,
    settings: Settings,
    log2n: u32,
    steps: u64,
    psi: wgpu::Buffer,
    fixed: wgpu::Buffer,
    detector: wgpu::Buffer,
    group: wgpu::BindGroup,
    pipelines: Vec<(&'static str, wgpu::ComputePipeline)>,
    readback: Vec<Readback>,
    scales: [f64; 5],
    fft: Fft,
    buffers: Buffers,
    autocorrelation: Option<Series>,
}

impl Wave {
    /// `initial(x, y)` is the starting wavefunction (re, im), `potential(x, y)`
    /// the real potential. Both are sampled at cell centres.
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        settings: Settings,
        initial: impl Fn(f64, f64) -> [f64; 2],
        potential: impl Fn(f64, f64) -> f64,
    ) -> Self {
        let n = settings.n;
        assert!(
            n.is_power_of_two() && n >= 8,
            "the grid side must be a power of two"
        );
        let log2n = n.trailing_zeros();
        let cells = (n * n) as usize;
        let dx = settings.length / n as f64;
        let coord = |i: u32| (i as f64 - (n / 2) as f64) * dx;
        let wavenumber = |i: u32| {
            let m = if i < n / 2 {
                i as f64
            } else {
                i as f64 - n as f64
            };
            TAU * m / settings.length
        };
        let absorb = |x: f64, y: f64| -> f64 {
            let Some(a) = settings.absorber else {
                return 0.0;
            };
            let edge = settings.length / 2.0 - x.abs().max(y.abs());
            if edge < a.width {
                a.strength * ((a.width - edge) / a.width).powi(2)
            } else {
                0.0
            }
        };

        // Regions: wavefunction, FFT scratch, measurement copy, psi(0) for
        // the autocorrelation
        let mut psi0 = vec![[0.0f32; 2]; 4 * cells];
        let mut factors = vec![[0.0f32; 4]; cells];
        let mut fields = vec![[0.0f32; 4]; cells];
        let mut v_max = 0.0f64;
        let norm_fft = 1.0 / (cells as f64);
        for j in 0..n {
            for i in 0..n {
                let c = (j * n + i) as usize;
                let (x, y) = (coord(i), coord(j));
                let p = initial(x, y);
                psi0[c] = [p[0] as f32, p[1] as f32];
                let v = potential(x, y);
                let w = absorb(x, y);
                let k2 = wavenumber(i).powi(2) + wavenumber(j).powi(2);
                v_max = v_max.max(v.abs());
                // 1 / N^2 is a power of two, so scaling by it is exact in f32
                let scale = norm_fft as f32;
                factors[c] = match settings.time {
                    Time::Real => {
                        let decay = (-w * settings.dt / 2.0).exp() as f32;
                        let pv = unit_phase(-v * settings.dt / 2.0);
                        let pk = unit_phase(-k2 / 2.0 * settings.dt);
                        if w > 0.0 {
                            [decay * pv[0], decay * pv[1], scale * pk[0], scale * pk[1]]
                        } else {
                            [pv[0], pv[1], scale * pk[0], scale * pk[1]]
                        }
                    }
                    Time::Imaginary => [
                        (-v * settings.dt / 2.0).exp() as f32,
                        0.0,
                        scale * (-k2 / 2.0 * settings.dt).exp() as f32,
                        0.0,
                    ],
                };
                fields[c] = [v as f32, w as f32, (k2 / 2.0) as f32, 0.0];
            }
        }
        // Normalise the starting state
        let total: f64 = psi0[..cells]
            .iter()
            .map(|p| (p[0] as f64).powi(2) + (p[1] as f64).powi(2))
            .sum::<f64>()
            * dx
            * dx;
        let rescale = 1.0 / total.sqrt();
        for p in &mut psi0[..cells] {
            p[0] = (p[0] as f64 * rescale) as f32;
            p[1] = (p[1] as f64 * rescale) as f32;
        }

        let twiddles: Vec<[f32; 2]> = (0..n / 2)
            .map(|m| unit_phase(-TAU * m as f64 / n as f64))
            .collect();
        let half_length = settings.length / 2.0;
        let detector_column = detector_column(&settings);
        let k_max2 = 2.0 * (PI / dx).powi(2);
        let scales = [
            scale_for(1.0),
            scale_for(half_length),
            scale_for(half_length * half_length),
            scale_for(v_max.max(1e-6)),
            scale_for(k_max2 / 2.0),
        ];
        let grid = Grid {
            n,
            log2n,
            cells: cells as u32,
            measure_base: 2 * cells as u32,
            dx: dx as f32,
            scale_norm: scales[0] as f32,
            scale_x: scales[1] as f32,
            scale_x2: scales[2] as f32,
            scale_v: scales[3] as f32,
            scale_t: scales[4] as f32,
            dt: settings.dt as f32,
            detector_column: detector_column.unwrap_or(0),
        };

        // Every FFT pass configuration: forward from A, inverse from A,
        // forward from C (measurements); each alternates with scratch B
        let mut passes = Vec::new();
        for (inverse, start) in [(0, 0u32), (1, 0), (0, 2 * cells as u32)] {
            let scratch = cells as u32;
            for p in 0..2 * log2n {
                let (src, dst) = if p % 2 == 0 {
                    (start, scratch)
                } else {
                    (scratch, start)
                };
                let pass = Pass {
                    stage: p % log2n,
                    axis: p / log2n,
                    inverse,
                    src,
                    dst,
                    _pad: [0; 3],
                };
                let mut bytes = bytemuck::bytes_of(&pass).to_vec();
                bytes.resize(PASS_STRIDE as usize, 0);
                passes.extend(bytes);
            }
        }
        // Whole-row transforms, in place: the same three sets, x then y
        for (inverse, start) in [(0, 0u32), (1, 0), (0, 2 * cells as u32)] {
            for axis in 0..2 {
                let pass = Pass {
                    stage: 0,
                    axis,
                    inverse,
                    src: start,
                    dst: start,
                    _pad: [0; 3],
                };
                let mut bytes = bytemuck::bytes_of(&pass).to_vec();
                bytes.resize(PASS_STRIDE as usize, 0);
                passes.extend(bytes);
            }
        }

        let init = |label: &str, contents: &[u8], usage| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents,
                usage,
            })
        };
        let storage = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST;
        let pass_buffer = init("fft passes", &passes, wgpu::BufferUsages::UNIFORM);
        let grid_buffer = init(
            "grid",
            bytemuck::bytes_of(&grid),
            wgpu::BufferUsages::UNIFORM,
        );
        let psi = init("wavefunction", bytemuck::cast_slice(&psi0), storage);
        let twiddles = init(
            "twiddles",
            bytemuck::cast_slice(&twiddles),
            wgpu::BufferUsages::STORAGE,
        );
        let factors = init(
            "phase factors",
            bytemuck::cast_slice(&factors),
            wgpu::BufferUsages::STORAGE,
        );
        let fields = init(
            "potential and wavenumbers",
            bytemuck::cast_slice(&fields),
            wgpu::BufferUsages::STORAGE,
        );
        let detector = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("detector"),
            size: n as u64 * 4,
            usage: storage,
            mapped_at_creation: false,
        });
        let fixed = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("wave accumulators"),
            size: 32 * 4,
            usage: storage,
            mapped_at_creation: false,
        });

        let entry = |binding: u32, ty, dynamic| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty,
                has_dynamic_offset: dynamic,
                min_binding_size: None,
            },
            count: None,
        };
        let read = wgpu::BufferBindingType::Storage { read_only: true };
        let rw = wgpu::BufferBindingType::Storage { read_only: false };
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("wave"),
            entries: &[
                entry(0, wgpu::BufferBindingType::Uniform, true),
                entry(1, wgpu::BufferBindingType::Uniform, false),
                entry(2, rw, false),
                entry(3, read, false),
                entry(4, read, false),
                entry(5, read, false),
                entry(6, rw, false),
                entry(7, rw, false),
            ],
        });
        let buffers = Buffers {
            layout,
            passes: pass_buffer,
            grid: grid_buffer,
            twiddles,
            factors,
            fields,
        };
        let group = buffers.group(device, &psi, &fixed, &detector);
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("wave"),
            bind_group_layouts: &[Some(&buffers.layout)],
            immediate_size: 0,
        });
        let source = format!("{}\n{WGSL_FIXED}", include_str!("schrodinger.wgsl"));
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("schrodinger"),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
        let pipelines = [
            "fft_pass",
            "fft_rows",
            "remember_initial",
            "overlap",
            "overlap_advance",
            "potential_half",
            "kinetic",
            "copy_to_measure",
            "measure_position",
            "measure_kinetic",
            "norm_sum",
            "norm_factor",
            "renormalise",
            "detect",
        ]
        .map(|name| {
            let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(name),
                layout: Some(&pipeline_layout),
                module: &module,
                entry_point: Some(name),
                compilation_options: Default::default(),
                cache: None,
            });
            (name, pipeline)
        })
        .to_vec();
        let readback = (0..RING)
            .map(|_| Readback {
                buffer: device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("wave readback"),
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
            settings,
            log2n,
            steps: 0,
            psi,
            fixed,
            detector,
            group,
            pipelines,
            readback,
            scales,
            fft: if settings.n <= ROW_MAX {
                Fft::Rows
            } else {
                Fft::Passes
            },
            buffers,
            autocorrelation: None,
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

    fn cells(&self) -> u32 {
        self.settings.n * self.settings.n
    }

    pub fn settings(&self) -> Settings {
        self.settings
    }

    pub fn steps(&self) -> u64 {
        self.steps
    }

    pub fn time(&self) -> f64 {
        self.steps as f64 * self.settings.dt
    }

    /// The wavefunction: N × N `vec2f` (re, im), row-major from the lowest y.
    /// For drawing.
    pub fn psi(&self) -> wgpu::BindingResource<'_> {
        wgpu::BindingResource::Buffer(wgpu::BufferBinding {
            buffer: &self.psi,
            offset: 0,
            size: wgpu::BufferSize::new(self.cells() as u64 * 8),
        })
    }

    /// The detector: N `f32`, one per row from the lowest y, each the
    /// probability current through that cell of the screen integrated over
    /// time so far (per unit length). For drawing.
    pub fn detector(&self) -> wgpu::BindingResource<'_> {
        self.detector.as_entire_binding()
    }

    /// Where the detector actually is: its x snapped to the grid
    pub fn detector_x(&self) -> Option<f64> {
        let column = detector_column(&self.settings)?;
        Some(
            (column as f64 - (self.settings.n / 2) as f64) * self.settings.length
                / self.settings.n as f64,
        )
    }

    fn run(&self, pass: &mut wgpu::ComputePass<'_>, name: &str, groups: u32) {
        pass.set_bind_group(0, &self.group, &[0]);
        pass.set_pipeline(self.pipeline(name));
        pass.dispatch_workgroups(groups, 1, 1);
    }

    /// Remembers the wavefunction now as psi(0) and makes room to record
    /// `capacity` samples of the autocorrelation C(t) = <psi(0)|psi(t)> (see
    /// [`Wave::sample_autocorrelation`] and [`energy_levels`]).
    pub fn record_autocorrelation(&mut self, capacity: u32) {
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("autocorrelation series"),
            size: 8 * (1 + 2 * capacity as u64),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let group = self
            .buffers
            .group(&self.device, &self.psi, &buffer, &self.detector);
        let mut encoder = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            self.run(
                &mut pass,
                "remember_initial",
                self.cells().div_ceil(WORKGROUP),
            );
        }
        self.queue.submit([encoder.finish()]);
        self.autocorrelation = Some(Series {
            capacity,
            buffer,
            group,
            samples: 0,
        });
    }

    /// Records C(t) = <psi(0)|psi(t)> now, the next sample in the series.
    pub fn sample_autocorrelation(&mut self, encoder: &mut wgpu::CommandEncoder) {
        let groups = self.cells().div_ceil(WORKGROUP);
        let series = self
            .autocorrelation
            .as_mut()
            .expect("record_autocorrelation first");
        assert!(
            series.samples < series.capacity,
            "autocorrelation series full"
        );
        series.samples += 1;
        let series = self.autocorrelation.as_ref().expect("recording");
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_bind_group(0, &series.group, &[0]);
        pass.set_pipeline(self.pipeline("overlap"));
        pass.dispatch_workgroups(groups, 1, 1);
        pass.set_pipeline(self.pipeline("overlap_advance"));
        pass.dispatch_workgroups(1, 1, 1);
    }

    /// The autocorrelation samples so far (re, im). Native only.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn read_autocorrelation(&self) -> Vec<[f64; 2]> {
        let series = self
            .autocorrelation
            .as_ref()
            .expect("record_autocorrelation first");
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("read autocorrelation"),
            size: series.buffer.size(),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = self.device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(&series.buffer, 0, &staging, 0, series.buffer.size());
        self.queue.submit([encoder.finish()]);
        staging
            .slice(..)
            .map_async(wgpu::MapMode::Read, |r| r.expect("read back"));
        self.wait();
        let words: Vec<u32> =
            bytemuck::cast_slice(&staging.slice(..).get_mapped_range().expect("mapped")).to_vec();
        staging.unmap();
        let values = fixed::decode(&words);
        (0..series.samples as usize)
            .map(|k| [1, 2].map(|q| values[2 * k + q] / self.scales[0]))
            .collect()
    }

    /// How 2D FFTs run; both give the same bits.
    pub fn set_fft(&mut self, fft: Fft) {
        assert!(
            fft == Fft::Passes || self.settings.n <= ROW_MAX,
            "whole-row FFTs need n <= {ROW_MAX}"
        );
        self.fft = fft;
    }

    /// One 2D FFT: `set` 0 forward from the wavefunction, 1 inverse, 2
    /// forward from the measurement copy
    fn fft(&self, pass: &mut wgpu::ComputePass<'_>, set: u32) {
        if self.fft == Fft::Rows {
            pass.set_pipeline(self.pipeline("fft_rows"));
            for axis in 0..2 {
                let index = 3 * 2 * self.log2n + 2 * set + axis;
                let offset = (index as u64 * PASS_STRIDE) as u32;
                pass.set_bind_group(0, &self.group, &[offset]);
                pass.dispatch_workgroups(self.settings.n, 1, 1);
            }
            return;
        }
        let groups = (self.cells() / 2).div_ceil(WORKGROUP);
        pass.set_pipeline(self.pipeline("fft_pass"));
        for p in 0..2 * self.log2n {
            let offset = ((set * 2 * self.log2n + p) as u64 * PASS_STRIDE) as u32;
            pass.set_bind_group(0, &self.group, &[offset]);
            pass.dispatch_workgroups(groups, 1, 1);
        }
    }

    /// Records `steps` steps into one compute pass.
    pub fn step(&mut self, encoder: &mut wgpu::CommandEncoder, steps: u32) {
        let groups = self.cells().div_ceil(WORKGROUP);
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("schrodinger steps"),
            timestamp_writes: None,
        });
        for _ in 0..steps {
            self.run(&mut pass, "potential_half", groups);
            self.fft(&mut pass, 0);
            self.run(&mut pass, "kinetic", groups);
            self.fft(&mut pass, 1);
            self.run(&mut pass, "potential_half", groups);
            if self.settings.time == Time::Imaginary {
                self.run(&mut pass, "norm_sum", groups);
                self.run(&mut pass, "norm_factor", 1);
                self.run(&mut pass, "renormalise", groups);
            }
            if self.settings.detector.is_some() {
                self.run(&mut pass, "detect", self.settings.n.div_ceil(WORKGROUP));
            }
        }
        self.steps += steps as u64;
    }

    /// Records a measurement. Returns false when all read-back buffers are busy.
    pub fn measure(&mut self, encoder: &mut wgpu::CommandEncoder) -> bool {
        let Some(index) = self.readback.iter().position(|r| !r.in_flight) else {
            return false;
        };
        let groups = self.cells().div_ceil(WORKGROUP);
        encoder.clear_buffer(&self.fixed, 0, Some(SLOTS * 8));
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            self.run(&mut pass, "measure_position", groups);
            self.run(&mut pass, "copy_to_measure", groups);
            self.fft(&mut pass, 2);
            self.run(&mut pass, "measure_kinetic", groups);
        }
        let slot = &mut self.readback[index];
        encoder.copy_buffer_to_buffer(&self.fixed, 0, &slot.buffer, 0, SLOTS * 8);
        slot.step = self.steps;
        slot.in_flight = true;
        slot.pending_map = true;
        true
    }

    /// Call after submitting the commands that recorded [`Wave::measure`].
    pub fn after_submit(&mut self) {
        for slot in self.readback.iter_mut().filter(|s| s.pending_map) {
            slot.pending_map = false;
            let ready = slot.ready.clone();
            slot.buffer
                .slice(..)
                .map_async(wgpu::MapMode::Read, move |r| {
                    if r.is_ok() {
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
            let [sn, sx, sx2, sv, st] = self.scales;
            let norm = v[0] / sn;
            let mean = [v[1] / sx / norm, v[2] / sx / norm];
            let second = [v[3] / sx2 / norm, v[4] / sx2 / norm];
            done.push(Measurement {
                step: slot.step,
                time: slot.step as f64 * self.settings.dt,
                norm,
                mean,
                spread: [0, 1].map(|k| (second[k] - mean[k] * mean[k]).max(0.0).sqrt()),
                potential: v[5] / sv,
                kinetic: v[7] / st,
                in_absorber: v[6] / sn,
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
    pub fn run_steps(&mut self, steps: u64) {
        let mut left = steps;
        while left > 0 {
            let batch = left.min(100) as u32;
            let mut encoder = self.device.create_command_encoder(&Default::default());
            self.step(&mut encoder, batch);
            self.queue.submit([encoder.finish()]);
            left -= batch as u64;
        }
        self.wait();
    }

    /// Reads the wavefunction back: N × N (re, im), row-major. Native only.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn read_psi(&self) -> Vec<[f32; 2]> {
        let size = self.cells() as u64 * 8;
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("read psi"),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = self.device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(&self.psi, 0, &staging, 0, size);
        self.queue.submit([encoder.finish()]);
        staging
            .slice(..)
            .map_async(wgpu::MapMode::Read, |r| r.expect("read back"));
        self.wait();
        let out =
            bytemuck::cast_slice(&staging.slice(..).get_mapped_range().expect("mapped")).to_vec();
        staging.unmap();
        out
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl Wave {
    /// Reads the detector back: for each row (from the lowest y), the time
    /// integral of the probability current j_x = Im(psi* d psi / dx) through
    /// the screen, per unit length. Summed times dx it is the probability that
    /// has crossed the screen (net, left to right). Native only.
    pub fn read_detector(&self) -> Vec<f64> {
        let size = self.settings.n as u64 * 4;
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("read detector"),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = self.device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(&self.detector, 0, &staging, 0, size);
        self.queue.submit([encoder.finish()]);
        staging
            .slice(..)
            .map_async(wgpu::MapMode::Read, |r| r.expect("read back"));
        self.wait();
        let out: Vec<f32> =
            bytemuck::cast_slice(&staging.slice(..).get_mapped_range().expect("mapped")).to_vec();
        staging.unmap();
        out.into_iter().map(f64::from).collect()
    }
}

/// A Gaussian wave packet centred at `centre` with momentum `momentum` and
/// position spread `spread` (standard deviation of |psi|^2 along each axis).
pub fn gaussian(
    centre: [f64; 2],
    momentum: [f64; 2],
    spread: f64,
) -> impl Fn(f64, f64) -> [f64; 2] {
    move |x, y| {
        let (dx, dy) = (x - centre[0], y - centre[1]);
        let amplitude = (-(dx * dx + dy * dy) / (4.0 * spread * spread)).exp();
        let phase = momentum[0] * x + momentum[1] * y;
        [amplitude * phase.cos(), amplitude * phase.sin()]
    }
}

/// The spectrum of an autocorrelation series `c` (samples `dt` apart from
/// t = 0) at `energy`: S(E) = 2 Re of the integral over t >= 0 of
/// C(t) e^(iEt) e^(-(t/tau)^2), by the trapezoid rule. For
/// C(t) = sum |c_n|^2 e^(-i E_n t) it peaks at each E_n with height
/// |c_n|^2 tau sqrt(pi) and width about 4 / tau. Keep `tau` below a quarter
/// of the recorded time so the window has died away by the end.
pub fn spectrum(c: &[[f64; 2]], dt: f64, tau: f64, energy: f64) -> f64 {
    let mut sum = 0.5 * c[0][0];
    for (k, ck) in c.iter().enumerate().skip(1) {
        let t = k as f64 * dt;
        let (s, co) = (energy * t).sin_cos();
        sum += (ck[0] * co - ck[1] * s) * (-(t / tau).powi(2)).exp();
    }
    2.0 * dt * sum
}

/// Energy levels from an autocorrelation series: the peaks of [`spectrum`]
/// between `from` and `to` whose weight |c_n|^2 is at least `min_weight`,
/// each located to about 1e-10 by golden-section search. Returns (energy,
/// weight) pairs, lowest first.
pub fn energy_levels(
    c: &[[f64; 2]],
    dt: f64,
    tau: f64,
    from: f64,
    to: f64,
    min_weight: f64,
) -> Vec<(f64, f64)> {
    let s = |e: f64| spectrum(c, dt, tau, e);
    let height = tau * PI.sqrt();
    // Eight samples across each peak's width
    let step = 0.5 / tau;
    let grid: Vec<f64> = (0..=((to - from) / step) as usize)
        .map(|k| from + k as f64 * step)
        .collect();
    let values: Vec<f64> = grid.iter().map(|&e| s(e)).collect();
    let mut levels = Vec::new();
    for k in 1..grid.len().saturating_sub(1) {
        if values[k] <= values[k - 1] || values[k] < values[k + 1] {
            continue;
        }
        if values[k] / height < min_weight {
            continue;
        }
        // Golden-section search for the maximum in [e - step, e + step]
        let (mut a, mut b) = (grid[k] - step, grid[k] + step);
        let g = (5f64.sqrt() - 1.0) / 2.0;
        let (mut x1, mut x2) = (b - g * (b - a), a + g * (b - a));
        let (mut f1, mut f2) = (s(x1), s(x2));
        while b - a > 1e-10 {
            if f1 > f2 {
                b = x2;
                x2 = x1;
                f2 = f1;
                x1 = b - g * (b - a);
                f1 = s(x1);
            } else {
                a = x1;
                x1 = x2;
                f1 = f2;
                x2 = a + g * (b - a);
                f2 = s(x2);
            }
        }
        let e = 0.5 * (a + b);
        levels.push((e, s(e) / height));
    }
    levels
}

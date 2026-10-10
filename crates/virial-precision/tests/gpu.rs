//! Runs the WGSL libraries on the real GPU and checks them against exact or
//! 64-bit results. Skipped (with a message) on machines with no GPU.

use virial_gpu::Compute;
use virial_precision::{Df64, WGSL_DF64, WGSL_FIXED, fixed};
use wgpu::util::DeviceExt;

fn gpu() -> Option<Compute> {
    match pollster::block_on(Compute::new()) {
        Ok(gpu) => Some(gpu),
        Err(error) => {
            eprintln!("skipped: no GPU ({error})");
            None
        }
    }
}

/// Deterministic random numbers in [0, 1)
fn random(seed: u64) -> impl FnMut() -> f64 {
    let mut s = seed;
    move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s >> 11) as f64 / (1u64 << 53) as f64
    }
}

fn run(gpu: &Compute, source: &str, buffers: &[&wgpu::Buffer], threads: u32, workgroup: u32) {
    let device = &gpu.device;
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: None,
        module: &module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });
    let entries: Vec<_> = buffers
        .iter()
        .enumerate()
        .map(|(i, b)| wgpu::BindGroupEntry {
            binding: i as u32,
            resource: b.as_entire_binding(),
        })
        .collect();
    let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pipeline.get_bind_group_layout(0),
        entries: &entries,
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &group, &[]);
        pass.dispatch_workgroups(threads.div_ceil(workgroup), 1, 1);
    }
    gpu.queue.submit([encoder.finish()]);
    gpu.wait();
}

fn storage(gpu: &Compute, data: &[u8]) -> wgpu::Buffer {
    gpu.device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: data,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        })
}

/// A million values from 1e-6 to 1e3, both signs, plus a few up to 1e6: the
/// kind of spread that loses small terms in a float sum.
fn spread_values(n: usize) -> Vec<f32> {
    let mut r = random(7);
    (0..n)
        .map(|i| {
            let top = if i % 1024 == 0 { 6.0 } else { 3.0 };
            let magnitude = 10f64.powf(r() * (top + 6.0) - 6.0);
            (if r() < 0.5 { -magnitude } else { magnitude }) as f32
        })
        .collect()
}

#[test]
fn fixed_point_sums_are_exact_and_identical_every_run() {
    let Some(gpu) = gpu() else { return };
    let values = spread_values(1 << 20);
    let source = format!(
        "@group(0) @binding(0) var<storage, read_write> virial_fixed: array<atomic<u32>>;
         @group(0) @binding(1) var<storage, read> values: array<f32>;
         {WGSL_FIXED}
         @compute @workgroup_size(256)
         fn main(@builtin(global_invocation_id) id: vec3u, @builtin(local_invocation_index) li: u32) {{
             fixed_wg_clear(li);
             workgroupBarrier();
             var x = 0.0;
             if id.x < arrayLength(&values) {{
                 x = values[id.x];
                 // Slot 0: every thread adds to the global accumulator directly
                 fixed_add(0u, x);
             }}
             // Slot 1: summed per workgroup first, then one global add each
             fixed_wg_add(0u, x);
             workgroupBarrier();
             fixed_wg_flush(li, 1u);
         }}"
    );

    let expected: i128 = values.iter().map(|&x| fixed::to_raw(x) as i128).sum();
    let expected = i64::try_from(expected).expect("the total fits in 64 bits");

    let mut runs = Vec::new();
    for _ in 0..3 {
        let acc = storage(&gpu, &[0u8; 18 * 4]);
        let input = storage(&gpu, bytemuck::cast_slice(&values));
        run(&gpu, &source, &[&acc, &input], values.len() as u32, 256);
        let words: Vec<u32> = bytemuck::cast_slice(&gpu.read_back(&acc)).to_vec();
        runs.push(words);
    }

    for words in &runs {
        assert_eq!(
            fixed::from_words(words[0], words[1]),
            expected,
            "direct atomics"
        );
        assert_eq!(
            fixed::from_words(words[2], words[3]),
            expected,
            "workgroup atomics"
        );
    }
    assert!(runs.windows(2).all(|w| w[0] == w[1]), "runs differ");

    // For comparison: the same values summed in order in f32 and in f64
    let exact: f64 = values.iter().map(|&x| x as f64).sum();
    let naive = values.iter().fold(0.0f32, |s, &x| s + x) as f64;
    let ours = fixed::to_f64(expected);
    println!(
        "sum of 2^20 values from 1e-6 to 1e6: exact {exact:.9}, fixed-point {ours:.9} (error {:.2e}), sequential f32 error {:.2e}",
        (ours - exact).abs(),
        (naive - exact).abs()
    );
}

#[test]
fn double_floats_on_the_gpu_match_f64_to_13_digits() {
    let Some(gpu) = gpu() else { return };
    let mut r = random(11);
    let n = 1 << 16;
    let pairs: Vec<(Df64, Df64)> = (0..n)
        .map(|_| {
            let a = (r() - 0.5) * 10f64.powf(r() * 8.0 - 4.0);
            let b = (r() - 0.5) * 10f64.powf(r() * 8.0 - 4.0);
            (Df64::from_f64(a), Df64::from_f64(b))
        })
        .collect();
    let input: Vec<f32> = pairs
        .iter()
        .flat_map(|(a, b)| [a.hi, a.lo, b.hi, b.lo])
        .collect();
    let source = format!(
        "@group(0) @binding(0) var<storage, read> inputs: array<vec4f>;
         @group(0) @binding(1) var<storage, read_write> outputs: array<vec2f>;
         @group(0) @binding(2) var<storage, read> one: f32;
         {WGSL_DF64}
         @compute @workgroup_size(64)
         fn main(@builtin(global_invocation_id) id: vec3u) {{
             df_init(one);
             let i = id.x;
             if i >= arrayLength(&inputs) {{ return; }}
             let a = inputs[i].xy;
             let b = inputs[i].zw;
             outputs[5u * i] = df_add(a, b);
             outputs[5u * i + 1u] = df_sub(a, b);
             outputs[5u * i + 2u] = df_mul(a, b);
             outputs[5u * i + 3u] = df_div(a, b);
             outputs[5u * i + 4u] = df_sqrt(select(a, -a, a.x < 0.0));
         }}"
    );
    let inputs = storage(&gpu, bytemuck::cast_slice(&input));
    let outputs = storage(&gpu, &vec![0u8; n * 5 * 8]);
    let one = storage(&gpu, bytemuck::bytes_of(&1.0f32));
    run(&gpu, &source, &[&inputs, &outputs, &one], n as u32, 64);
    let out: Vec<f32> = bytemuck::cast_slice(&gpu.read_back(&outputs)).to_vec();

    let mut worst = [0.0f64; 5];
    let mut same_as_cpu = 0usize;
    for (i, (a, b)) in pairs.iter().enumerate() {
        let (x, y) = (a.to_f64(), b.to_f64());
        let got = |k: usize| Df64 {
            hi: out[10 * i + 2 * k],
            lo: out[10 * i + 2 * k + 1],
        };
        let scale = x.abs().max(y.abs());
        let errors = [
            (got(0).to_f64() - (x + y)).abs() / scale,
            (got(1).to_f64() - (x - y)).abs() / scale,
            ((got(2).to_f64() - x * y) / (x * y)).abs(),
            ((got(3).to_f64() - x / y) / (x / y)).abs(),
            ((got(4).to_f64() - x.abs().sqrt()) / x.abs().sqrt()).abs(),
        ];
        for (w, e) in worst.iter_mut().zip(errors) {
            *w = w.max(e);
        }
        if got(2) == *a * *b {
            same_as_cpu += 1;
        }
    }
    println!(
        "df64 worst relative errors on the GPU: add {:.1e}, sub {:.1e}, mul {:.1e}, div {:.1e}, sqrt {:.1e}; \
         products bit-identical to the CPU mirror: {same_as_cpu}/{n}",
        worst[0], worst[1], worst[2], worst[3], worst[4]
    );
    for (name, err) in ["add", "sub", "mul", "div", "sqrt"].iter().zip(worst) {
        assert!(err < 1e-13, "{name}: worst relative error {err:e}");
    }
}

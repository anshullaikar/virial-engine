//! The two FFT paths (whole rows in workgroup memory, and Stockham passes)
//! give the same bits, and how long a step takes with each. Skipped (with a
//! message) on machines with no hardware GPU.

use virial_gpu::Compute;
use virial_quantum::{Absorber, Fft, Settings, Time, Wave, gaussian};

fn gpu() -> Option<Compute> {
    match pollster::block_on(Compute::hardware()) {
        Ok(gpu) => Some(gpu),
        Err(error) => {
            eprintln!("skipped: no GPU ({error})");
            None
        }
    }
}

fn wave(gpu: &Compute, n: u32, fft: Fft) -> Wave {
    let settings = Settings {
        n,
        length: 80.0,
        dt: 0.01,
        time: Time::Real,
        absorber: Some(Absorber {
            width: 8.0,
            strength: 2.0,
        }),
        detector: None,
    };
    // A packet in a harmonic well: every part of the step does real work
    let mut wave = Wave::new(
        &gpu.device,
        &gpu.queue,
        settings,
        gaussian([-6.0, 3.0], [1.5, -0.5], 1.2),
        |x, y| 0.02 * (x * x + y * y),
    );
    wave.set_fft(fft);
    wave
}

#[test]
fn row_and_pass_ffts_give_the_same_bits() {
    let Some(gpu) = gpu() else { return };
    for n in [64, 256, 1024] {
        let mut rows = wave(&gpu, n, Fft::Rows);
        let mut passes = wave(&gpu, n, Fft::Passes);
        rows.run_steps(50);
        passes.run_steps(50);
        let (a, b) = (rows.read_psi(), passes.read_psi());
        let same = a
            .iter()
            .zip(&b)
            .all(|(p, q)| p[0].to_bits() == q[0].to_bits() && p[1].to_bits() == q[1].to_bits());
        let (ma, mb) = (rows.measure_now(), passes.measure_now());
        println!(
            "n = {n}: identical {same}; norm {:.10} vs {:.10}, kinetic {:.10} vs {:.10}",
            ma.norm, mb.norm, ma.kinetic, mb.kinetic
        );
        assert!(same, "row and pass FFTs differ at n = {n}");
    }
}

/// Step times for both paths (run by hand: cargo test --release -p
/// virial-quantum --test fft -- --ignored --nocapture)
#[test]
#[ignore]
fn step_times() {
    let Some(gpu) = gpu() else { return };
    for n in [512, 1024] {
        for fft in [Fft::Passes, Fft::Rows] {
            let mut w = wave(&gpu, n, fft);
            w.run_steps(50);
            let steps = 1000;
            let start = std::time::Instant::now();
            w.run_steps(steps);
            let ms = start.elapsed().as_secs_f64() * 1000.0 / steps as f64;
            println!("n = {n} {fft:?}: {ms:.3} ms per step");
        }
    }
}

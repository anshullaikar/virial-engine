//! Speed of the gravity step. Run with
//! `cargo test --release -p virial-particles --test bench -- --ignored --nocapture`.

use std::time::Instant;

use virial_gpu::Compute;
use virial_particles::{Body, Gravity, NBody};

fn bodies(n: usize) -> Vec<Body> {
    let mut s = 0x9e37_79b9_7f4a_7c15u64;
    let mut r = move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s >> 11) as f64 / (1u64 << 53) as f64
    };
    (0..n)
        .map(|_| Body {
            pos: [r() - 0.5, r() - 0.5, r() - 0.5],
            vel: [0.0; 3],
            mass: 1.0 / n as f64,
        })
        .collect()
}

#[test]
#[ignore = "benchmark: run explicitly"]
fn steps_per_second() {
    let Ok(gpu) = pollster::block_on(Compute::new()) else {
        eprintln!("skipped: no GPU");
        return;
    };
    for n in [16_384usize, 65_536] {
        let mut sim = NBody::new(
            &gpu.device,
            &gpu.queue,
            &bodies(n),
            Gravity {
                g: 1.0,
                softening: 0.01,
            },
            1e-4,
        );
        sim.run(2);
        let steps = if n > 20_000 { 20 } else { 100 };
        let start = Instant::now();
        sim.run(steps);
        let seconds = start.elapsed().as_secs_f64();
        let rate = steps as f64 / seconds;
        println!(
            "{n} bodies: {rate:.1} steps/s, {:.2} ms/step, {:.2e} interactions/s",
            1e3 / rate,
            rate * (n * n) as f64
        );
    }
}

#[test]
#[ignore = "benchmark: run explicitly"]
fn measurement_cost() {
    let Ok(gpu) = pollster::block_on(Compute::new()) else {
        return;
    };
    for n in [8_192usize, 16_384] {
        let mut sim = NBody::new(
            &gpu.device,
            &gpu.queue,
            &bodies(n),
            Gravity {
                g: 1.0,
                softening: 0.01,
            },
            1e-4,
        );
        sim.measure_now();
        let start = Instant::now();
        for _ in 0..3 {
            sim.measure_now();
        }
        println!(
            "{n} bodies: {:.1} ms per measurement",
            start.elapsed().as_secs_f64() * 1e3 / 3.0
        );
    }
}

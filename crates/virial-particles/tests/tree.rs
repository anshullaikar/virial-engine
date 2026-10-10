//! Barnes-Hut against exact gravity: force errors, reproducibility and speed.
//! Skipped (with a message) on machines with no GPU.

use virial_gpu::Compute;
use virial_particles::{Body, Gravity, NBody, Solver};

fn gpu() -> Option<Compute> {
    match pollster::block_on(Compute::hardware()) {
        Ok(gpu) => Some(gpu),
        Err(error) => {
            eprintln!("skipped: no GPU ({error})");
            None
        }
    }
}

/// A clumpy cloud: a few dense clusters in a sparse halo, which is where
/// tree codes are most likely to go wrong
fn clusters(n: usize) -> Vec<Body> {
    let mut s = 0x1234_5678_9abc_def1u64;
    let mut r = move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s >> 11) as f64 / (1u64 << 53) as f64
    };
    let centres = [
        [0.0, 0.0, 0.0],
        [2.0, 0.5, -0.3],
        [-1.5, 1.2, 0.8],
        [0.4, -1.8, 1.1],
    ];
    (0..n)
        .map(|i| {
            let c = centres[i % centres.len()];
            // Halo for a quarter of the bodies, tight clusters for the rest
            let spread = if i % 4 == 0 { 3.0 } else { 0.3 };
            let g = |r: &mut dyn FnMut() -> f64| {
                (-2.0 * r().max(1e-12).ln()).sqrt() * (std::f64::consts::TAU * r()).cos()
            };
            Body {
                pos: [
                    c[0] + spread * g(&mut r),
                    c[1] + spread * g(&mut r),
                    c[2] + spread * g(&mut r),
                ],
                vel: [0.0; 3],
                mass: 1.0 / n as f64,
            }
        })
        .collect()
}

fn percentile(sorted: &[f64], p: f64) -> f64 {
    sorted[((sorted.len() - 1) as f64 * p).round() as usize]
}

#[test]
fn barnes_hut_forces_match_exact_gravity() {
    let Some(gpu) = gpu() else { return };
    let bodies = clusters(8192);
    let gravity = Gravity {
        g: 1.0,
        softening: 0.01,
    };
    let mut exact = NBody::new(&gpu.device, &gpu.queue, &bodies, gravity, 1e-3);
    let reference = exact.read_accelerations();
    let mut medians = Vec::new();
    for theta in [0.5, 0.3] {
        let mut tree = NBody::new(&gpu.device, &gpu.queue, &bodies, gravity, 1e-3);
        tree.set_solver(Solver::BarnesHut { theta });
        let approx = tree.read_accelerations();
        let mut errors: Vec<f64> = reference
            .iter()
            .zip(&approx)
            .map(|(a, b)| {
                let diff = (0..3).map(|k| (a[k] - b[k]).powi(2)).sum::<f64>().sqrt();
                diff / (0..3).map(|k| a[k] * a[k]).sum::<f64>().sqrt()
            })
            .collect();
        errors.sort_by(|a, b| a.total_cmp(b));
        let (median, p99) = (percentile(&errors, 0.5), percentile(&errors, 0.99));
        println!(
            "theta {theta}: relative force error median {median:.2e}, 99th percentile {p99:.2e}, worst {:.2e}",
            errors[errors.len() - 1]
        );
        medians.push(median);
        assert!(median < 5e-3, "median force error {median:e}");
        assert!(p99 < 5e-2, "99th percentile force error {p99:e}");
    }
    assert!(
        medians[1] < medians[0],
        "smaller theta should be more accurate"
    );
}

#[test]
fn barnes_hut_reruns_are_bit_identical() {
    let Some(gpu) = gpu() else { return };
    let bodies = clusters(5000);
    let gravity = Gravity {
        g: 1.0,
        softening: 0.01,
    };
    let run = || {
        let mut sim = NBody::new(&gpu.device, &gpu.queue, &bodies, gravity, 1e-3);
        sim.set_solver(Solver::BarnesHut { theta: 0.5 });
        sim.run(50);
        sim.read_bodies()
    };
    assert!(run() == run(), "Barnes-Hut runs differ");
}

/// Run with `cargo test --release -p virial-particles --test tree speed -- --ignored --nocapture`.
#[test]
#[ignore = "benchmark: run explicitly"]
fn barnes_hut_speed() {
    let Some(gpu) = gpu() else { return };
    for n in [65_536usize, 262_144] {
        let mut sim = NBody::new(
            &gpu.device,
            &gpu.queue,
            &clusters(n),
            Gravity {
                g: 1.0,
                softening: 0.01,
            },
            1e-4,
        );
        sim.set_solver(Solver::BarnesHut { theta: 0.5 });
        sim.run(2);
        let steps = 20;
        let start = std::time::Instant::now();
        sim.run(steps);
        let ms = start.elapsed().as_secs_f64() * 1e3 / steps as f64;
        println!(
            "{n} bodies, Barnes-Hut theta 0.5: {ms:.1} ms per step, {:.1} steps/s",
            1e3 / ms
        );
    }
}

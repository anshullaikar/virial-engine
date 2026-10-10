//! Energy drift against step size for both cutoff treatments. Run with
//! `cargo test --release -p virial-particles --test lj_dt_study -- --ignored --nocapture`.
//! On an M1 (4096 molecules, density 0.1, T about 2, 40 time units), drift
//! falls about fourfold per halving of dt: step size dominates, not the cutoff.

use virial_gpu::Compute;
use virial_particles::lennard_jones::{LennardJones, Settings, lattice};

#[test]
#[ignore = "study: run explicitly"]
fn lj_timestep_study() {
    let Ok(gpu) = pollster::block_on(Compute::new()) else {
        eprintln!("skipped: no GPU");
        return;
    };
    let n = 4096;
    let box_size = (n as f64 / 0.1).cbrt();
    let particles = lattice(n, box_size, 2.0, 13);
    for shifted_force in [false, true] {
        for dt in [0.004, 0.002, 0.001] {
            let settings = Settings {
                box_size,
                cutoff: 2.5,
                dt,
                thermostat: None,
                shifted_force,
            };
            let mut sim = LennardJones::new(&gpu.device, &gpu.queue, &particles, settings);
            let start = sim.measure_now().energy();
            sim.run((40.0 / dt) as u64);
            let drift = (sim.measure_now().energy() - start) / start;
            println!(
                "shifted_force {shifted_force}, dt {dt}: drift over 40 time units {drift:+.2e}"
            );
        }
    }
}

/// Steps per second of the Lennard-Jones solver as the count grows. Run with
/// `cargo test --release -p virial-particles --test lj_dt_study speed -- --ignored --nocapture`.
#[test]
#[ignore = "benchmark: run explicitly"]
fn lj_speed() {
    let Ok(gpu) = pollster::block_on(Compute::new()) else {
        eprintln!("skipped: no GPU");
        return;
    };
    for n in [32_768usize, 131_072, 262_144, 1_048_576] {
        // Density 0.4 keeps a million molecules within the cell grid's limit
        let box_size = (n as f64 / 0.4).cbrt();
        let settings = Settings {
            box_size,
            cutoff: 2.5,
            dt: 0.002,
            thermostat: None,
            shifted_force: true,
        };
        let mut sim = LennardJones::new(
            &gpu.device,
            &gpu.queue,
            &lattice(n, box_size, 1.5, 3),
            settings,
        );
        sim.run(10);
        let start = std::time::Instant::now();
        sim.run(50);
        let ms = start.elapsed().as_secs_f64() * 1e3 / 50.0;
        println!(
            "{n} molecules (density 0.4): {ms:.2} ms per step, {:.0} steps/s",
            1e3 / ms
        );
    }
}

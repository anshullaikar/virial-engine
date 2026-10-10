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

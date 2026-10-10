//! Validation against known physics and the f64 CPU reference.
//! Skipped (with a message) on machines with no GPU.

use virial_gpu::Compute;
use virial_particles::{Body, Gravity, NBody};
use virial_reference as reference;

fn gpu() -> Option<Compute> {
    match pollster::block_on(Compute::new()) {
        Ok(gpu) => Some(gpu),
        Err(error) => {
            eprintln!("skipped: no GPU ({error})");
            None
        }
    }
}

fn to_gpu(b: &reference::Body<f64>) -> Body {
    Body {
        pos: b.pos,
        vel: b.vel,
        mass: b.mass,
    }
}

/// Energy drift that ignores Verlet's bounded oscillation: mean energy over
/// the last tenth of the run against the first tenth, relative to |E0|.
fn drift(energies: &[f64]) -> f64 {
    let tenth = (energies.len() / 10).max(1);
    let mean = |s: &[f64]| s.iter().sum::<f64>() / s.len() as f64;
    let first = mean(&energies[..tenth]);
    let last = mean(&energies[energies.len() - tenth..]);
    ((last - first) / energies[0]).abs()
}

#[test]
fn kepler_orbit_conserves_energy_over_100_orbits() {
    let Some(gpu) = gpu() else { return };
    let (bodies, period) = reference::kepler(1.0, 1e-3, 1.0, 0.5, 1.0);
    let gravity = Gravity {
        g: 1.0,
        softening: 0.0,
    };
    let dt = period / 1000.0;
    let steps = 100_000u64;
    let every = 100u64;

    let initial: Vec<Body> = bodies.iter().map(to_gpu).collect();
    let mut sim = NBody::new(&gpu.device, &gpu.queue, &initial, gravity, dt);
    let mut gpu_energy = vec![sim.measure_now().energy()];
    for _ in 0..steps / every {
        sim.run(every);
        gpu_energy.push(sim.measure_now().energy());
    }

    let rg = reference::Gravity {
        g: 1.0,
        softening: 0.0,
    };
    let mut cpu64 = reference::NBody::new(bodies.clone(), rg);
    let mut cpu32 = reference::NBody::<f32>::new(bodies.iter().map(|b| b.convert()).collect(), rg);
    let (mut e64, mut e32) = (vec![cpu64.totals().energy()], vec![cpu32.totals().energy()]);
    for _ in 0..steps / every {
        cpu64.step(dt, every as usize);
        cpu32.step(dt, every as usize);
        e64.push(cpu64.totals().energy());
        e32.push(cpu32.totals().energy());
    }

    let exact = -1e-3 / 2.0; // -G m1 m2 / (2a)
    let gpu_bodies = sim.read_bodies();
    let separation = |b: &[[f64; 3]; 2]| {
        (0..3)
            .map(|k| (b[0][k] - b[1][k]).powi(2))
            .sum::<f64>()
            .sqrt()
    };
    let gpu_sep = separation(&[gpu_bodies[0].pos, gpu_bodies[1].pos]);
    let cpu_sep = separation(&[cpu64.bodies[0].pos, cpu64.bodies[1].pos]);
    println!(
        "Kepler, e = 0.5, 1000 steps per orbit, 100 orbits:\n  \
         energy at start: GPU {:.12e}, exact {exact:.12e}\n  \
         energy drift: GPU (double-float) {:.2e}, CPU f64 {:.2e}, CPU f32 {:.2e}\n  \
         separation after 100 orbits: GPU {gpu_sep:.10}, CPU f64 {cpu_sep:.10} (difference {:.2e})",
        gpu_energy[0],
        drift(&gpu_energy),
        drift(&e64),
        drift(&e32),
        (gpu_sep - cpu_sep).abs()
    );
    assert!(
        ((gpu_energy[0] - exact) / exact).abs() < 1e-12,
        "initial energy"
    );
    assert!(
        drift(&gpu_energy) < 1e-6,
        "GPU energy drift {:e}",
        drift(&gpu_energy)
    );
    assert!(
        (gpu_sep - cpu_sep).abs() < 1e-6,
        "GPU and CPU orbits diverged"
    );
}

/// A cold, clumpy cloud of bodies: deterministic, and messy enough that any
/// difference in arithmetic order would show.
fn cloud(n: usize) -> Vec<Body> {
    let mut s = 0x2545_f491_4f6c_dd1du64;
    let mut r = move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s >> 11) as f64 / (1u64 << 53) as f64
    };
    (0..n)
        .map(|_| Body {
            pos: [r() - 0.5, r() - 0.5, r() - 0.5],
            vel: [0.1 * (r() - 0.5), 0.1 * (r() - 0.5), 0.1 * (r() - 0.5)],
            mass: (0.5 + r()) / n as f64,
        })
        .collect()
}

#[test]
fn reruns_are_bit_identical() {
    let Some(gpu) = gpu() else { return };
    let bodies = cloud(4096);
    let gravity = Gravity {
        g: 1.0,
        softening: 0.01,
    };
    let run = || {
        let mut sim = NBody::new(&gpu.device, &gpu.queue, &bodies, gravity, 1e-3);
        sim.run(200);
        let m = sim.measure_now();
        (
            sim.read_bodies(),
            m.kinetic.to_bits(),
            m.potential.to_bits(),
        )
    };
    let a = run();
    let b = run();
    assert!(a.0 == b.0, "states differ between runs");
    assert_eq!((a.1, a.2), (b.1, b.2), "measurements differ between runs");
}

#[test]
fn cluster_matches_the_f64_reference() {
    let Some(gpu) = gpu() else { return };
    let bodies = cloud(512);
    let gravity = Gravity {
        g: 1.0,
        softening: 0.05,
    };
    let mut sim = NBody::new(&gpu.device, &gpu.queue, &bodies, gravity, 1e-3);
    let start = sim.measure_now();
    sim.run(1000);
    let end = sim.measure_now();

    let refs: Vec<reference::Body<f64>> = bodies
        .iter()
        .map(|b| reference::Body {
            pos: b.pos,
            vel: b.vel,
            mass: b.mass,
        })
        .collect();
    let mut cpu = reference::NBody::new(
        refs,
        reference::Gravity {
            g: 1.0,
            softening: 0.05,
        },
    );
    let cpu_start = cpu.totals();
    cpu.step(1e-3, 1000);
    let cpu_end = cpu.totals();

    let gpu_bodies = sim.read_bodies();
    let worst = gpu_bodies
        .iter()
        .zip(&cpu.bodies)
        .map(|(g, c)| {
            (0..3)
                .map(|k| (g.pos[k] - c.pos[k]).powi(2))
                .sum::<f64>()
                .sqrt()
        })
        .fold(0.0, f64::max);
    let momentum = end.momentum.iter().map(|p| p * p).sum::<f64>().sqrt();
    println!(
        "512-body cluster, 1000 steps:\n  energy start: GPU {:.12e}, CPU {:.12e}\n  \
         energy end:   GPU {:.12e}, CPU {:.12e}\n  \
         worst position difference from CPU f64: {worst:.2e}\n  total momentum at end: {momentum:.2e}",
        start.energy(),
        cpu_start.energy(),
        end.energy(),
        cpu_end.energy()
    );
    assert!(((start.energy() - cpu_start.energy()) / cpu_start.energy()).abs() < 1e-9);
    assert!(((end.energy() - cpu_end.energy()) / cpu_end.energy()).abs() < 1e-6);
}

/// The 0.1.0 release target: energy drift below 1e-6 over a million steps
/// (1000 orbits). Takes about a minute. Run with
/// `cargo test --release -p virial-particles --test validation million -- --ignored --nocapture`.
#[test]
#[ignore = "long: run explicitly"]
fn kepler_orbit_over_a_million_steps() {
    let Some(gpu) = gpu() else { return };
    let (bodies, period) = reference::kepler(1.0, 1e-3, 1.0, 0.5, 1.0);
    let initial: Vec<Body> = bodies.iter().map(to_gpu).collect();
    let mut sim = NBody::new(
        &gpu.device,
        &gpu.queue,
        &initial,
        Gravity {
            g: 1.0,
            softening: 0.0,
        },
        period / 1000.0,
    );
    let mut energies = vec![sim.measure_now().energy()];
    for _ in 0..1000 {
        sim.run(1000);
        energies.push(sim.measure_now().energy());
    }
    let worst = energies
        .iter()
        .map(|e| ((e - energies[0]) / energies[0]).abs())
        .fold(0.0, f64::max);
    println!(
        "Kepler, 10^6 steps (1000 orbits): drift {:.2e}, worst excursion {worst:.2e}",
        drift(&energies)
    );
    assert!(drift(&energies) < 1e-6, "drift {:e}", drift(&energies));
}

//! Granular validation: Hertz contact time, restitution, a settled pile
//! carrying its weight, and reproducibility. Skipped on machines with no GPU.

use virial_gpu::Compute;
use virial_particles::granular::{Grain, Granular, Settings, block};

fn gpu() -> Option<Compute> {
    match pollster::block_on(Compute::new()) {
        Ok(gpu) => Some(gpu),
        Err(error) => {
            eprintln!("skipped: no GPU ({error})");
            None
        }
    }
}

const R: f64 = 2.5e-3;

/// Two grains approaching along x, far from the floor and walls, no gravity
fn pair(settings: &Settings, speed: f64) -> Vec<Grain> {
    let mid = settings.box_size / 2.0;
    vec![
        Grain {
            pos: [mid - R * 1.2, mid, mid],
            vel: [speed / 2.0, 0.0, 0.0],
            radius: R,
        },
        Grain {
            pos: [mid + R * 1.2, mid, mid],
            vel: [-speed / 2.0, 0.0, 0.0],
            radius: R,
        },
    ]
}

/// Steps one at a time, returning (contact duration, final velocities)
fn collide(gpu: &Compute, settings: Settings, speed: f64) -> (f64, [f64; 2]) {
    let mut sim = Granular::new(&gpu.device, &gpu.queue, &pair(&settings, speed), settings);
    let (mut start, mut end) = (None, None);
    for step in 0..200_000u64 {
        sim.run(1);
        let g = sim.read_grains();
        let touching = (g[1].pos[0] - g[0].pos[0]) < g[0].radius + g[1].radius;
        if touching && start.is_none() {
            start = Some(step);
        }
        if !touching && start.is_some() {
            end = Some(step);
            break;
        }
    }
    let g = sim.read_grains();
    let steps = end.expect("the grains separate") - start.expect("the grains touch");
    (steps as f64 * settings.dt, [g[0].vel[0], g[1].vel[0]])
}

#[test]
fn elastic_collision_matches_hertz_contact_time() {
    let Some(gpu) = gpu() else { return };
    let speed = 0.5;
    let mut settings = Settings::glass(0.05, 0.0);
    settings.gravity = 0.0;
    settings.damping = 0.0;
    settings.friction = 0.0;
    settings.floor_height = -1.0;

    // Hertz theory: delta_max = (15 m* v^2 / (16 E* sqrt(R*)))^(2/5),
    // t_c = 2 delta_max / v * integral_0^1 (1 - x^(5/2))^(-1/2) dx
    let m = settings.density * 4.0 / 3.0 * std::f64::consts::PI * R.powi(3);
    let (m_eff, r_eff) = (m / 2.0, R / 2.0);
    let delta_max =
        (15.0 * m_eff * speed * speed / (16.0 * settings.modulus * r_eff.sqrt())).powf(0.4);
    let integral = {
        // Substitute x = 1 - u^2 to remove the endpoint singularity
        let n = 100_000;
        (0..n)
            .map(|i| {
                let u = (i as f64 + 0.5) / n as f64;
                let x: f64 = 1.0 - u * u;
                2.0 * u / (1.0 - x.powf(2.5)).sqrt() / n as f64
            })
            .sum::<f64>()
    };
    let theory = 2.0 * delta_max / speed * integral;
    settings.dt = theory / 2000.0;

    let (measured, v) = collide(&gpu, settings, speed);
    let restitution = (v[1] - v[0]) / speed;
    println!(
        "elastic collision at {speed} m/s: contact time {measured:.4e} s, Hertz theory {theory:.4e} s \
         (difference {:.2}%); restitution {restitution:.6}; momentum after {:.2e}",
        (measured / theory - 1.0) * 100.0,
        v[0] + v[1]
    );
    assert!(
        (measured / theory - 1.0).abs() < 0.005,
        "contact time differs from Hertz theory"
    );
    assert!(
        (restitution - 1.0).abs() < 1e-3,
        "an elastic collision should keep its speed"
    );
    assert!((v[0] + v[1]).abs() < 1e-9, "momentum not conserved");
}

#[test]
fn damped_collision_loses_energy() {
    let Some(gpu) = gpu() else { return };
    let mut settings = Settings::glass(0.05, 2e-6);
    settings.gravity = 0.0;
    settings.friction = 0.0;
    settings.floor_height = -1.0;
    let (_, v) = collide(&gpu, settings, 0.5);
    let restitution = (v[1] - v[0]) / 0.5;
    println!(
        "damped collision (A = {:e} s): restitution {restitution:.4}",
        settings.damping
    );
    assert!(restitution > 0.0 && restitution < 0.99);
    assert!((v[0] + v[1]).abs() < 1e-9, "momentum not conserved");
}

#[test]
fn settled_pile_rests_on_the_floor_with_its_weight() {
    let Some(gpu) = gpu() else { return };
    let settings = Settings::glass(0.1, 2e-5);
    let grains = block([0.004, 0.004, 0.004], [0.096, 0.096, 0.06], R, 0.1, 7);
    let n = grains.len();
    let weight: f64 = grains
        .iter()
        .map(|g| {
            settings.density * 4.0 / 3.0
                * std::f64::consts::PI
                * g.radius.powi(3)
                * settings.gravity
        })
        .sum();
    let mut sim = Granular::new(&gpu.device, &gpu.queue, &grains, settings);
    let start = sim.measure_now();
    sim.run(100_000);
    let end = sim.measure_now();
    println!(
        "{n} grains after 2 s: support {:.6} N vs weight {weight:.6} N ({:+.3}%); kinetic energy {:.2e} J \
         (from {:.2e} J potential lost)",
        end.support,
        (end.support / weight - 1.0) * 100.0,
        end.kinetic,
        start.gravitational - end.gravitational
    );
    assert_eq!(end.grains as usize, n);
    assert!(
        (end.support / weight - 1.0).abs() < 0.01,
        "the floor and walls don't carry the pile's weight"
    );
    assert!(
        end.kinetic < 1e-3 * (start.gravitational - end.gravitational),
        "the pile hasn't settled"
    );
}

#[test]
fn reruns_are_bit_identical() {
    let Some(gpu) = gpu() else { return };
    let settings = Settings::glass(0.06, 2e-5);
    let grains = block([0.003, 0.003, 0.01], [0.057, 0.057, 0.05], R, 0.1, 3);
    let run = || {
        let mut sim = Granular::new(&gpu.device, &gpu.queue, &grains, settings);
        sim.run(2000);
        let m = sim.measure_now();
        (sim.read_grains(), m.kinetic.to_bits(), m.support.to_bits())
    };
    let a = run();
    let b = run();
    assert!(a.0 == b.0, "states differ between runs");
    assert_eq!((a.1, a.2), (b.1, b.2), "measurements differ between runs");
}

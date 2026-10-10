//! Lennard-Jones validation: neighbour search against all pairs, relaxation
//! to the Maxwell-Boltzmann distribution, the ideal gas law and the second
//! virial coefficient. Skipped (with a message) on machines with no GPU.

use virial_gpu::Compute;
use virial_particles::lennard_jones::{Langevin, LennardJones, Particle, Settings, lattice};

fn gpu() -> Option<Compute> {
    match pollster::block_on(Compute::new()) {
        Ok(gpu) => Some(gpu),
        Err(error) => {
            eprintln!("skipped: no GPU ({error})");
            None
        }
    }
}

/// Cut-and-shifted Lennard-Jones potential, in f64
fn potential(r2: f64, rc2: f64) -> f64 {
    if r2 >= rc2 {
        return 0.0;
    }
    let lj = |r2: f64| {
        let s6 = 1.0 / (r2 * r2 * r2);
        4.0 * (s6 * s6 - s6)
    };
    lj(r2) - lj(rc2)
}

/// Potential energy and virial sum over all pairs (nearest image), in f64
fn all_pairs(particles: &[Particle], box_size: f64, cutoff: f64) -> (f64, f64) {
    let rc2 = cutoff * cutoff;
    let (mut pe, mut w) = (0.0, 0.0);
    for (i, a) in particles.iter().enumerate() {
        for b in &particles[i + 1..] {
            let d: Vec<f64> = (0..3)
                .map(|k| {
                    let x = b.pos[k] - a.pos[k];
                    x - box_size * (x / box_size).round()
                })
                .collect();
            let r2 = d.iter().map(|x| x * x).sum::<f64>();
            if r2 < rc2 {
                let s6 = 1.0 / (r2 * r2 * r2);
                pe += potential(r2, rc2);
                w += 24.0 * (2.0 * s6 * s6 - s6);
            }
        }
    }
    (pe, w)
}

#[test]
fn cell_grid_finds_every_pair() {
    let Some(gpu) = gpu() else { return };
    // A dense, jittered lattice: many pairs inside the cutoff, many crossing
    // cell and box boundaries
    let mut particles = lattice(1000, 11.0, 1.0, 3);
    let mut s = 99u64;
    for p in &mut particles {
        for k in 0..3 {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            p.pos[k] += ((s >> 11) as f64 / (1u64 << 53) as f64 - 0.5) * 0.3;
        }
    }
    let settings = Settings {
        box_size: 11.0,
        cutoff: 2.5,
        dt: 0.002,
        thermostat: None,
    };
    let mut sim = LennardJones::new(&gpu.device, &gpu.queue, &particles, settings);
    let m = sim.measure_now();
    let wrapped: Vec<Particle> = particles
        .iter()
        .map(|p| Particle {
            pos: p.pos.map(|x| x.rem_euclid(sim.box_size())),
            vel: p.vel,
        })
        .collect();
    let (pe, w) = all_pairs(&wrapped, sim.box_size(), 2.5);
    println!(
        "1000 particles: potential GPU {:.10} CPU {pe:.10}; virial GPU {:.10} CPU {w:.10}",
        m.potential, m.virial
    );
    assert!(
        ((m.potential - pe) / pe).abs() < 1e-5,
        "potential energy differs"
    );
    assert!(((m.virial - w) / w).abs() < 1e-5, "virial differs");
}

/// Kolmogorov-Smirnov distance between the speeds and the Maxwell-Boltzmann
/// distribution at temperature `t` (mass 1)
fn ks_distance(speeds: &mut [f64], t: f64) -> f64 {
    speeds.sort_by(|a, b| a.total_cmp(b));
    let n = speeds.len() as f64;
    // CDF: erf(x) - sqrt(2/pi) * v/a * exp(-v^2 / 2a^2), a = sqrt(t), x = v / (a sqrt 2)
    let cdf = |v: f64| {
        let a = t.sqrt();
        let x = v / (a * std::f64::consts::SQRT_2);
        erf(x) - (2.0 / std::f64::consts::PI).sqrt() * (v / a) * (-x * x).exp()
    };
    speeds
        .iter()
        .enumerate()
        .map(|(i, &v)| {
            let f = cdf(v);
            (f - i as f64 / n).abs().max(((i + 1) as f64 / n - f).abs())
        })
        .fold(0.0, f64::max)
}

/// Abramowitz and Stegun 7.1.26 is too coarse here; use a series/continued
/// fraction erf accurate to about 1e-15
fn erf(x: f64) -> f64 {
    if x < 0.0 {
        return -erf(-x);
    }
    if x < 3.0 {
        // Taylor series
        let mut term = x;
        let mut sum = x;
        let mut k = 0.0;
        loop {
            k += 1.0;
            term *= -x * x / k;
            let add = term / (2.0 * k + 1.0);
            sum += add;
            if add.abs() < 1e-17 * sum.abs() {
                break;
            }
        }
        sum * 2.0 / std::f64::consts::PI.sqrt()
    } else {
        // Continued fraction for erfc
        let mut f = 0.0;
        for k in (1..60).rev() {
            f = (k as f64 / 2.0) / (x + f);
        }
        1.0 - (-x * x).exp() / (std::f64::consts::PI.sqrt() * (x + f))
    }
}

#[test]
fn gas_relaxes_to_maxwell_boltzmann_at_constant_energy() {
    let Some(gpu) = gpu() else { return };
    // Every particle starts at the same speed in a random direction: far from
    // equilibrium. Collisions alone (no thermostat) must produce the
    // Maxwell-Boltzmann distribution while total energy stays constant.
    let n = 4096;
    let density = 0.1;
    let box_size = (n as f64 / density).cbrt();
    let mut particles = lattice(n, box_size, 1.0, 5);
    let speed = (3.0 * 2.0f64).sqrt();
    for p in &mut particles {
        let norm = p.vel.iter().map(|v| v * v).sum::<f64>().sqrt();
        p.vel = p.vel.map(|v| v / norm * speed);
    }
    let settings = Settings {
        box_size,
        cutoff: 2.5,
        dt: 0.004,
        thermostat: None,
    };
    let mut sim = LennardJones::new(&gpu.device, &gpu.queue, &particles, settings);
    let start = sim.measure_now();
    let dof = 3.0 * n as f64 - 3.0;

    let speeds_of = |sim: &LennardJones| -> Vec<f64> {
        sim.read_particles()
            .iter()
            .map(|p| p.vel.iter().map(|v| v * v).sum::<f64>().sqrt())
            .collect()
    };
    let mut first = speeds_of(&sim);
    let d_start = ks_distance(&mut first, start.temperature(dof));

    // Relax for 40 time units, then pool 10 snapshots 2 time units apart
    sim.run(10_000);
    let mut pooled = Vec::new();
    let mut temps = Vec::new();
    for _ in 0..10 {
        sim.run(500);
        pooled.extend(speeds_of(&sim));
        temps.push(sim.measure_now().temperature(dof));
    }
    let end = sim.measure_now();
    let t = temps.iter().sum::<f64>() / temps.len() as f64;
    let d_end = ks_distance(&mut pooled, t);
    let drift = (end.energy() - start.energy()) / start.energy().abs();
    println!(
        "gas, N = {n}, density {density}: KS distance from Maxwell-Boltzmann {d_start:.3} at start -> {d_end:.4} \
         after 60 time units (T = {t:.4}); energy {:.8} -> {:.8} (drift {drift:.2e})",
        start.energy(),
        end.energy()
    );
    assert!(d_start > 0.2, "the start should be far from equilibrium");
    assert!(
        d_end < 0.01,
        "speeds don't follow Maxwell-Boltzmann (KS distance {d_end})"
    );
    assert!(drift.abs() < 1e-4, "energy drift {drift:e}");
}

/// Second virial coefficient of the cut-and-shifted potential at temperature
/// `t`: B2 = -2 pi * integral of (exp(-U/T) - 1) r^2 dr, by Simpson's rule
fn second_virial(t: f64, cutoff: f64) -> f64 {
    let steps = 200_000;
    let h = cutoff / steps as f64;
    let f = |r: f64| {
        if r == 0.0 {
            return 0.0;
        }
        ((-potential(r * r, cutoff * cutoff) / t).exp() - 1.0) * r * r
    };
    let mut sum = f(0.0) + f(cutoff);
    for i in 1..steps {
        sum += f(i as f64 * h) * if i % 2 == 1 { 4.0 } else { 2.0 };
    }
    -2.0 * std::f64::consts::PI * sum * h / 3.0
}

#[test]
fn dilute_gas_obeys_the_ideal_gas_law_and_virial_expansion() {
    let Some(gpu) = gpu() else { return };
    let n = 4096;
    let density = 0.01;
    let temperature = 2.0;
    let box_size = (n as f64 / density).cbrt();
    let particles = lattice(n, box_size, temperature, 9);
    let thermostat = Langevin {
        temperature,
        friction: 1.0,
        seed: 7,
    };
    let settings = Settings {
        box_size,
        cutoff: 2.5,
        dt: 0.005,
        thermostat: Some(thermostat),
    };
    let mut sim = LennardJones::new(&gpu.device, &gpu.queue, &particles, settings);
    sim.run(4_000);

    // Compressibility factor Z = PV / (N T), sampled every 20 steps, with the
    // statistical error from 20 batch means
    let dof = 3.0 * n as f64;
    let mut z = Vec::new();
    for _ in 0..2000 {
        sim.run(20);
        let m = sim.measure_now();
        let rho = n as f64 / m.volume;
        z.push(m.pressure() / (rho * m.temperature(dof)));
    }
    let mean = z.iter().sum::<f64>() / z.len() as f64;
    let batches: Vec<f64> = z
        .chunks(z.len() / 20)
        .map(|c| c.iter().sum::<f64>() / c.len() as f64)
        .collect();
    let spread = (batches.iter().map(|b| (b - mean).powi(2)).sum::<f64>()
        / (batches.len() - 1) as f64)
        .sqrt();
    let error = spread / (batches.len() as f64).sqrt();
    let b2 = second_virial(temperature, 2.5);
    let rho = n as f64 / sim.volume();
    let predicted = 1.0 + b2 * rho;
    println!(
        "dilute gas, N = {n}, density {density}, T = {temperature}: Z = PV/NkT = {mean:.5} +/- {error:.5}; \
         ideal gas 1; virial expansion 1 + B2 rho = {predicted:.5} (B2 = {b2:.4})"
    );
    assert!(
        (mean - 1.0).abs() < 0.01,
        "not within 1% of the ideal gas law"
    );
    assert!(
        (mean - predicted).abs() < 4.0 * error.max(1e-4),
        "differs from the virial expansion"
    );
}

#[test]
fn thermostatted_reruns_are_bit_identical() {
    let Some(gpu) = gpu() else { return };
    let particles = lattice(2000, 16.0, 1.5, 11);
    let settings = Settings {
        box_size: 16.0,
        cutoff: 2.5,
        dt: 0.004,
        thermostat: Some(Langevin {
            temperature: 1.5,
            friction: 0.5,
            seed: 3,
        }),
    };
    let run = || {
        let mut sim = LennardJones::new(&gpu.device, &gpu.queue, &particles, settings);
        sim.run(300);
        let m = sim.measure_now();
        (
            sim.read_particles(),
            m.potential.to_bits(),
            m.virial.to_bits(),
        )
    };
    let a = run();
    let b = run();
    assert!(a.0 == b.0, "states differ between runs");
    assert_eq!((a.1, a.2), (b.1, b.2), "measurements differ between runs");
}

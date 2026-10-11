//! Heat conduction in a Lennard-Jones liquid: heat exchange between a hot
//! and a cold slab (bookkeeping), the microscopic heat current against an f64
//! CPU sum, and thermal conductivity from Fourier's law (non-equilibrium)
//! against Green-Kubo (equilibrium). Skipped (with a message) on machines
//! with no GPU.

use virial_gpu::Compute;
use virial_particles::lennard_jones::{
    HeatExchange, LennardJones, Particle, Settings, Thermostat, lattice_box,
};

fn gpu() -> Option<Compute> {
    match pollster::block_on(Compute::hardware()) {
        Ok(gpu) => Some(gpu),
        Err(error) => {
            eprintln!("skipped: no GPU ({error})");
            None
        }
    }
}

const CUTOFF: f64 = 2.5;
/// Small enough for this dense liquid to conserve energy (at 0.004 it
/// drifts measurably even without heat exchange)
const DT: f64 = 0.002;

fn settings(box_size: [f64; 3], bins: u32) -> Settings {
    Settings {
        box_size,
        cutoff: CUTOFF,
        dt: DT,
        thermostat: Thermostat::None,
        shifted_force: true,
        skin: 0.3,
        profile_bins: bins,
    }
}

/// Shifted-force pair terms (potential, r . F) in f64, as the GPU computes them
fn pair(r2: f64) -> (f64, f64) {
    let lj = |r2: f64| {
        let s6 = 1.0 / (r2 * r2 * r2);
        (4.0 * (s6 * s6 - s6), 24.0 * (2.0 * s6 * s6 - s6))
    };
    let (u, rf) = lj(r2);
    let (uc, rfc) = lj(CUTOFF * CUTOFF);
    let r = r2.sqrt();
    let fc = rfc / CUTOFF;
    (u - uc + (r - CUTOFF) * fc, rf - r * fc)
}

/// The heat current J = sum_i e_i v_i + 1/2 sum_i sum_j (F_ij . v_i) r_ij
/// over all pairs (nearest image), in f64
fn heat_current(particles: &[Particle], size: [f64; 3]) -> [f64; 3] {
    let n = particles.len();
    let mut energy: Vec<f64> = particles
        .iter()
        .map(|p| 0.5 * p.vel.iter().map(|v| v * v).sum::<f64>())
        .collect();
    let mut pairs = vec![[0.0; 3]; n];
    for i in 0..n {
        for j in i + 1..n {
            let (a, b) = (&particles[i], &particles[j]);
            // d = r_j - r_i
            let d: [f64; 3] = std::array::from_fn(|k| {
                let x = b.pos[k] - a.pos[k];
                x - size[k] * (x / size[k]).round()
            });
            let r2: f64 = d.iter().map(|x| x * x).sum();
            if r2 >= CUTOFF * CUTOFF {
                continue;
            }
            let (u, rf) = pair(r2);
            energy[i] += 0.5 * u;
            energy[j] += 0.5 * u;
            // r_ij (F_ij . v_i) = d (d . v_i) rf / r^2, the same for j with v_j
            let dv_i: f64 = (0..3).map(|k| d[k] * a.vel[k]).sum();
            let dv_j: f64 = (0..3).map(|k| d[k] * b.vel[k]).sum();
            for k in 0..3 {
                pairs[i][k] += d[k] * dv_i * rf / r2;
                pairs[j][k] += d[k] * dv_j * rf / r2;
            }
        }
    }
    std::array::from_fn(|k| {
        (0..n)
            .map(|i| energy[i] * particles[i].vel[k] + 0.5 * pairs[i][k])
            .sum()
    })
}

/// A liquid at density 0.8442 filling the box, at temperature `t`
fn liquid(size: [f64; 3], t: f64) -> Vec<Particle> {
    let count = (0.8442 * size.iter().product::<f64>()).round() as usize;
    lattice_box(count, size, t, 11)
}

#[test]
fn heat_current_matches_cpu() {
    let Some(gpu) = gpu() else { return };
    let size = [9.0, 9.0, 12.0];
    let mut sim = LennardJones::new(
        &gpu.device,
        &gpu.queue,
        &liquid(size, 1.0),
        settings(size, 0),
    );
    // Melt the lattice first, so the current isn't a lattice's special case
    sim.run(500);
    sim.record_heat_current(2);
    let mut encoder = gpu.device.create_command_encoder(&Default::default());
    sim.sample_heat_current(&mut encoder);
    gpu.queue.submit([encoder.finish()]);
    let gpu_j = sim.read_heat_current()[0];
    let particles: Vec<Particle> = sim
        .read_particles()
        .into_iter()
        .map(|p| Particle {
            pos: std::array::from_fn(|k| p.pos[k].rem_euclid(sim.box_size()[k])),
            vel: p.vel,
        })
        .collect();
    let cpu_j = heat_current(&particles, sim.box_size());
    // Terms are of order 1 per particle with both signs; compare against the
    // size of the sum of their magnitudes, not of J, which nearly cancels
    let scale = (particles.len() as f64).sqrt() * 10.0;
    println!(
        "{} particles: heat current GPU {gpu_j:?}, CPU {cpu_j:?}",
        particles.len()
    );
    for k in 0..3 {
        assert!(
            (gpu_j[k] - cpu_j[k]).abs() < 1e-4 * scale,
            "heat current differs"
        );
    }
}

/// Energy drift per unit time: least-squares slope of energy against time,
/// and its standard error
fn drift(time: &[f64], energy: &[f64]) -> (f64, f64) {
    let n = time.len() as f64;
    let (mt, me) = (time.iter().sum::<f64>() / n, energy.iter().sum::<f64>() / n);
    let sxx: f64 = time.iter().map(|t| (t - mt).powi(2)).sum();
    let rate = time
        .iter()
        .zip(energy)
        .map(|(t, e)| (t - mt) * (e - me))
        .sum::<f64>()
        / sxx;
    let residual: f64 = time
        .iter()
        .zip(energy)
        .map(|(t, e)| (e - me - rate * (t - mt)).powi(2))
        .sum();
    (rate, (residual / (n - 2.0) / sxx).sqrt())
}

#[test]
fn heat_exchange_moves_energy_and_keeps_momentum() {
    let Some(gpu) = gpu() else { return };
    let size = [9.0, 9.0, 36.0];
    let mut sim = LennardJones::new(
        &gpu.device,
        &gpu.queue,
        &liquid(size, 1.0),
        settings(size, 12),
    );
    sim.run(2000);
    let before = sim.measure_now();
    let energy = 0.01;
    sim.set_heat_exchange(Some(HeatExchange {
        hot: (0.0, 3.0),
        cold: (18.0, 21.0),
        energy,
    }));
    // Energy every 100 steps for 40 time units; Verlet's own fluctuation
    // (about 0.05 here) hides any drift in a single before-and-after pair
    let (mut time, mut total) = (Vec::new(), Vec::new());
    for _ in 0..200 {
        sim.run(100);
        sim.sample_profile_now();
        let m = sim.measure_now();
        time.push(m.time);
        total.push(m.energy());
    }
    let after = sim.measure_now();
    let (rate, error) = drift(&time, &total);
    let duration = time[time.len() - 1] - time[0];
    let moved = energy / DT * duration;
    // Averaged over the 200 samples: the hot slab (0..3) should be about
    // 0.08 warmer than the cold one (18..21) at this flux
    let t = sim.read_profile().temperature();
    println!(
        "moved {moved:.0} from cold to hot over {duration:.0} time units; energy drift \
         {rate:+.2e} +/- {error:.1e} per time unit, {:.1e} of the energy moved; \
         momentum {:?} -> {:?}; slab temperatures {t:.3?}",
        (rate * duration / moved).abs(),
        before.momentum,
        after.momentum
    );
    assert!(
        (rate * duration).abs() < 1e-3 * moved,
        "energy not conserved"
    );
    for k in 0..3 {
        assert!(
            (after.momentum[k] - before.momentum[k]).abs() < 1e-6 * after.count as f64,
            "momentum changed"
        );
    }
    assert!(t[0] > t[6] + 0.04, "hot slab not hotter than cold slab");
}

/// Mean and standard error of block results
fn mean_error(x: &[f64]) -> (f64, f64) {
    let n = x.len() as f64;
    let mean = x.iter().sum::<f64>() / n;
    let var = x.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1.0);
    (mean, (var / n).sqrt())
}

/// Least-squares slope of y against x
fn slope(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len() as f64;
    let (mx, my) = (x.iter().sum::<f64>() / n, y.iter().sum::<f64>() / n);
    let sxy: f64 = x.iter().zip(y).map(|(a, b)| (a - mx) * (b - my)).sum();
    let sxx: f64 = x.iter().map(|a| (a - mx).powi(2)).sum();
    sxy / sxx
}

/// Density 0.8442 at a mean temperature near 1, in a 10 x 10 x 40 box
const LIQUID_BOX: [f64; 3] = [10.0, 10.0, 40.0];

/// Fourier's law: heat exchange between a hot slab at z 0..2 and a cold one
/// at 20..22; the temperature profile between them, away from the slabs, is
/// fitted with a line in each half. Returns, for each block, the
/// conductivity, the mean temperature, and the bow of the profile (how far the
/// middle third of each linear region sits above the line through its outer
/// thirds, relative to the drop across the region; 0 for a straight line).
fn non_equilibrium(gpu: &Compute, blocks: usize) -> Vec<(f64, f64, f64)> {
    let mut sim = LennardJones::new(
        &gpu.device,
        &gpu.queue,
        &liquid(LIQUID_BOX, 1.0),
        settings(LIQUID_BOX, 80),
    );
    sim.run(5_000);
    // Flux 0.1: a drop of about 0.3 between the slabs, well above the noise
    sim.set_heat_exchange(Some(HeatExchange {
        hot: (0.0, 2.0),
        cold: (20.0, 22.0),
        energy: 0.04,
    }));
    let flux = sim.heat_flux().expect("heat exchange on");
    // Reach the steady state (several relaxation times of about 12)
    sim.run(50_000);
    let mut out = Vec::new();
    for _ in 0..blocks {
        let mut encoder = gpu.device.create_command_encoder(&Default::default());
        sim.clear_profile(&mut encoder);
        for sample in 0..500 {
            sim.step(&mut encoder, 20, None);
            sim.sample_profile(&mut encoder);
            if sample % 10 == 9 {
                gpu.queue.submit([encoder.finish()]);
                encoder = gpu.device.create_command_encoder(&Default::default());
            }
        }
        gpu.queue.submit([encoder.finish()]);
        let p = sim.read_profile();
        let t = p.temperature();
        let z: Vec<f64> = p.slabs.iter().map(|s| s.z).collect();
        // Linear regions: 4 clear of each exchange slab. Heat flows from
        // z = 1 up to 21 (falling T) and from 41 = 1 down to 21 (rising T)
        let fit = |lo: f64, hi: f64| {
            let (zs, ts): (Vec<f64>, Vec<f64>) = z
                .iter()
                .zip(&t)
                .filter(|(z, _)| **z > lo && **z < hi)
                .map(|(z, t)| (*z, *t))
                .unzip();
            slope(&zs, &ts)
        };
        let gradient = 0.5 * (-fit(6.0, 16.0) + fit(26.0, 36.0));
        let mean_t = t.iter().sum::<f64>() / t.len() as f64;
        let mean = |lo: f64, hi: f64| {
            let inside: Vec<f64> = z
                .iter()
                .zip(&t)
                .filter(|(z, _)| **z > lo && **z < hi)
                .map(|(_, t)| *t)
                .collect();
            inside.iter().sum::<f64>() / inside.len() as f64
        };
        // Thirds of each region; for a line the middle equals the outer mean
        let bow = |lo: f64| {
            let third = 10.0 / 3.0;
            let (a, b, c) = (
                mean(lo, lo + third),
                mean(lo + third, lo + 2.0 * third),
                mean(lo + 2.0 * third, lo + 10.0),
            );
            (b - 0.5 * (a + c)) / (a - c).abs()
        };
        out.push((flux / gradient, mean_t, 0.5 * (bow(6.0) + bow(26.0))));
    }
    out
}

/// Green-Kubo: lambda = 1 / (3 V T^2) integral of <J(0) . J(t)> dt, in an
/// equilibrium run at constant energy. Returns (conductivity, temperature)
/// for each block.
fn green_kubo(gpu: &Compute, blocks: usize) -> Vec<(f64, f64)> {
    let size = [16.0; 3];
    let mut sim = LennardJones::new(
        &gpu.device,
        &gpu.queue,
        &liquid(size, 1.0),
        settings(size, 0),
    );
    sim.run(5_000);
    let n = sim.count() as f64;
    let volume = sim.volume();
    let every = 2u32;
    let dt = DT * every as f64;
    let samples = 50_000u32;
    // Integrate the correlation to t = 6
    let lags = (6.0 / dt) as usize;
    let mut out = Vec::new();
    for _ in 0..blocks {
        sim.record_heat_current(samples);
        let mut kinetic = 0.0;
        let mut encoder = gpu.device.create_command_encoder(&Default::default());
        for sample in 0..samples {
            sim.step(&mut encoder, every, None);
            sim.sample_heat_current(&mut encoder);
            if sample % 500 == 499 {
                gpu.queue.submit([encoder.finish()]);
                encoder = gpu.device.create_command_encoder(&Default::default());
                kinetic += sim.measure_now().kinetic;
            }
        }
        gpu.queue.submit([encoder.finish()]);
        let t = 2.0 * kinetic / (samples / 500) as f64 / (3.0 * n - 3.0);
        let j = sim.read_heat_current();
        // <J(0) . J(t)> averaged over time origins, trapezoid rule in t
        let correlation: Vec<f64> = (0..=lags)
            .map(|lag| {
                let pairs = j.len() - lag;
                (0..pairs)
                    .map(|s| (0..3).map(|k| j[s][k] * j[s + lag][k]).sum::<f64>())
                    .sum::<f64>()
                    / pairs as f64
            })
            .collect();
        let integral =
            dt * (correlation.iter().sum::<f64>() - 0.5 * (correlation[0] + correlation[lags]));
        out.push((integral / (3.0 * volume * t * t), t));
    }
    out
}

/// Fourier's law and Green-Kubo, two independent routes to the thermal
/// conductivity of the same liquid, must agree
#[test]
fn conductivity_from_fourier_law_agrees_with_green_kubo() {
    let Some(gpu) = gpu() else { return };
    let start = std::time::Instant::now();
    let nemd = non_equilibrium(&gpu, 8);
    let nemd_time = start.elapsed().as_secs_f64();
    let gk = green_kubo(&gpu, 8);
    let (l_nemd, e_nemd) = mean_error(&nemd.iter().map(|b| b.0).collect::<Vec<_>>());
    let (t_nemd, _) = mean_error(&nemd.iter().map(|b| b.1).collect::<Vec<_>>());
    let (bow, e_bow) = mean_error(&nemd.iter().map(|b| b.2).collect::<Vec<_>>());
    let (l_gk, e_gk) = mean_error(&gk.iter().map(|b| b.0).collect::<Vec<_>>());
    let (t_gk, _) = mean_error(&gk.iter().map(|b| b.1).collect::<Vec<_>>());
    let combined = (e_nemd * e_nemd + e_gk * e_gk).sqrt();
    println!(
        "density 0.8442: Fourier's law {l_nemd:.3} +/- {e_nemd:.3} (mean T {t_nemd:.4}, \
         {nemd_time:.0} s); Green-Kubo {l_gk:.3} +/- {e_gk:.3} (T {t_gk:.4}, {:.0} s); \
         difference {:.1} combined standard errors; profile bow {bow:+.3} +/- {e_bow:.3} \
         of the drop across each linear region",
        start.elapsed().as_secs_f64() - nemd_time,
        (l_nemd - l_gk).abs() / combined
    );
    assert!(
        (l_nemd - l_gk).abs() < 3.0 * combined,
        "conductivities disagree"
    );
    // Fourier's law with a nearly constant conductivity: a straight profile
    assert!(
        bow.abs() < 3.0 * e_bow + 0.02,
        "temperature profile not linear"
    );
}

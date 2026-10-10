//! Liquid-vapour coexistence: a slab of Lennard-Jones liquid in a long box
//! evaporates until its vapour balances it. Densities and vapour pressure are
//! checked against NIST's grand-canonical transition-matrix Monte Carlo for
//! the same potential (force-shifted at 2.5 sigma), and the surface tension
//! against the correlation of Vrabec et al. for the truncated and shifted
//! potential at 2.5 sigma. Skipped (with a message) on machines with no GPU.

use virial_gpu::Compute;
use virial_particles::lennard_jones::{
    LennardJones, NoseHoover, Particle, Profile, Settings, Thermostat, lattice_box,
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

fn settings(box_size: [f64; 3], shifted_force: bool, bins: u32) -> Settings {
    Settings {
        box_size,
        cutoff: 2.5,
        dt: 0.005,
        thermostat: Thermostat::NoseHoover(NoseHoover {
            temperature: 0.7,
            period: 0.5,
        }),
        shifted_force,
        skin: 0.3,
        profile_bins: bins,
    }
}

/// Lennard-Jones pair terms (potential, r . F) in f64, shifted as the GPU does
fn pair(r2: f64, cutoff: f64, shifted_force: bool) -> (f64, f64) {
    let lj = |r2: f64| {
        let s6 = 1.0 / (r2 * r2 * r2);
        (4.0 * (s6 * s6 - s6), 24.0 * (2.0 * s6 * s6 - s6))
    };
    let (u, rf) = lj(r2);
    let (uc, rfc) = lj(cutoff * cutoff);
    if shifted_force {
        let r = r2.sqrt();
        let fc = rfc / cutoff;
        (u - uc + (r - cutoff) * fc, rf - r * fc)
    } else {
        (u - uc, rf)
    }
}

/// Potential energy and virial tensor diagonal over all pairs (nearest
/// image in a rectangular box), in f64
fn all_pairs(particles: &[Particle], size: [f64; 3], shifted_force: bool) -> (f64, [f64; 3]) {
    let (mut pe, mut w) = (0.0, [0.0; 3]);
    for (i, a) in particles.iter().enumerate() {
        for b in &particles[i + 1..] {
            let d: [f64; 3] = std::array::from_fn(|k| {
                let x = b.pos[k] - a.pos[k];
                x - size[k] * (x / size[k]).round()
            });
            let r2 = d.iter().map(|x| x * x).sum::<f64>();
            if r2 < 2.5 * 2.5 {
                let (u, rf) = pair(r2, 2.5, shifted_force);
                pe += u;
                for (wk, dk) in w.iter_mut().zip(d) {
                    *wk += dk * dk * rf / r2;
                }
            }
        }
    }
    (pe, w)
}

/// A jittered lattice filling a rectangular box
fn jittered(count: usize, size: [f64; 3]) -> Vec<Particle> {
    let mut particles = lattice_box(count, size, 1.0, 3);
    let mut s = 99u64;
    for p in &mut particles {
        for k in 0..3 {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            p.pos[k] += ((s >> 11) as f64 / (1u64 << 53) as f64 - 0.5) * 0.3;
        }
    }
    particles
}

#[test]
fn rectangular_box_matches_all_pairs() {
    let Some(gpu) = gpu() else { return };
    // Different side on every axis, each a different number of cells
    let size = [9.0, 10.5, 14.25];
    let particles = jittered(1000, size);
    for shifted_force in [false, true] {
        let mut sim = LennardJones::new(
            &gpu.device,
            &gpu.queue,
            &particles,
            settings(size, shifted_force, 57),
        );
        let m = sim.measure_now();
        sim.sample_profile_now();
        let profile = sim.read_profile();
        let wrapped: Vec<Particle> = particles
            .iter()
            .map(|p| Particle {
                pos: std::array::from_fn(|k| p.pos[k].rem_euclid(sim.box_size()[k])),
                vel: p.vel,
            })
            .collect();
        let (pe, w) = all_pairs(&wrapped, sim.box_size(), shifted_force);
        let w_sum: f64 = w.iter().sum();
        println!(
            "box {size:?}, shifted force {shifted_force}: potential GPU {:.8} CPU {pe:.8}; \
             virial GPU {:.8} CPU {w_sum:.8}; tensor GPU {:?} CPU {w:?}",
            m.potential, m.virial, profile.virial
        );
        let close = |a: f64, b: f64| ((a - b) / b).abs() < 1e-5;
        assert!(close(m.potential, pe), "potential energy differs");
        assert!(close(m.virial, w_sum), "virial differs");
        for (gpu, cpu) in profile.virial.into_iter().zip(w) {
            assert!(close(gpu, cpu), "virial tensor differs");
        }
    }
}

#[test]
fn profiles_add_up_to_the_whole_box() {
    let Some(gpu) = gpu() else { return };
    let size = [10.0, 10.0, 30.0];
    let particles = jittered(2400, size);
    let run = || {
        let mut sim = LennardJones::new(
            &gpu.device,
            &gpu.queue,
            &particles,
            settings(size, true, 120),
        );
        sim.run(200);
        let m = sim.measure_now();
        sim.sample_profile_now();
        (m, sim.read_profile())
    };
    let (m, p) = run();
    // Fixed-point sums don't depend on the order threads add in
    let bits = |p: &Profile| -> Vec<u64> {
        p.slabs
            .iter()
            .flat_map(|s| {
                [s.particles, s.potential]
                    .into_iter()
                    .chain(s.kinetic)
                    .chain(s.virial)
            })
            .chain(p.virial)
            .map(f64::to_bits)
            .collect()
    };
    assert!(bits(&p) == bits(&run().1), "profiles differ between runs");
    let count: f64 = p.slabs.iter().map(|s| s.particles).sum();
    let kinetic: f64 = p.slabs.iter().flat_map(|s| s.kinetic).sum();
    let potential: f64 = p.slabs.iter().map(|s| s.potential).sum();
    let slab_virial: [f64; 3] =
        std::array::from_fn(|k| p.slabs.iter().map(|s| s.virial[k]).sum::<f64>());
    let total_virial: f64 = p.virial.iter().sum();
    println!(
        "particles {count} of {}; 2 KE {kinetic:.6} vs {:.6}; potential {potential:.6} vs {:.6}; \
         virial {total_virial:.6} vs {:.6}; slab virial {slab_virial:?} vs whole box {:?}",
        m.count,
        2.0 * m.kinetic,
        m.potential,
        m.virial,
        p.virial
    );
    assert_eq!(count, m.count as f64, "every particle is in one slab");
    let close = |a: f64, b: f64, tol: f64| ((a - b) / b).abs() < tol;
    // Profiles add f32 terms; measurements double-float ones
    assert!(close(kinetic, 2.0 * m.kinetic, 1e-6));
    assert!(close(potential, m.potential, 1e-5));
    assert!(close(total_virial, m.virial, 1e-5));
    // Spreading each pair over slabs keeps all of it
    for (slabs, whole) in slab_virial.into_iter().zip(p.virial) {
        assert!(close(slabs, whole, 1e-6));
    }
}

/// Mean and standard error of block results
fn mean_error(x: &[f64]) -> (f64, f64) {
    let n = x.len() as f64;
    let mean = x.iter().sum::<f64>() / n;
    let var = x.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1.0);
    (mean, (var / n).sqrt())
}

/// What one block of samples says about the slab
struct Block {
    liquid: f64,
    vapour: f64,
    normal_pressure: f64,
    /// Normal pressure averaged over the liquid, the interfaces and the vapour
    normal_by_region: [f64; 3],
    surface_tension: f64,
    temperature: f64,
}

/// Mean over the slabs within `range` of `centre` (periodic), from `values`
fn region(p: &Profile, values: &[f64], centre: f64, range: (f64, f64)) -> f64 {
    let lz = p.box_size[2];
    let (mut sum, mut n) = (0.0, 0.0);
    for (s, v) in p.slabs.iter().zip(values) {
        let mut d = s.z - centre;
        d -= lz * (d / lz).round();
        if d.abs() >= range.0 && d.abs() < range.1 {
            sum += v;
            n += 1.0;
        }
    }
    sum / n
}

fn analyse(p: &Profile) -> Block {
    let density = p.density();
    let lz = p.box_size[2];
    // Centre of the slab: the mean angle of the density around the periodic z
    let (mut c, mut s) = (0.0, 0.0);
    for (slab, rho) in p.slabs.iter().zip(&density) {
        let angle = std::f64::consts::TAU * slab.z / lz;
        c += rho * angle.cos();
        s += rho * angle.sin();
    }
    let centre = s.atan2(c).rem_euclid(std::f64::consts::TAU) * lz / std::f64::consts::TAU;
    let normal: Vec<f64> = p.pressure_tensor().iter().map(|t| t[2]).collect();
    let half = lz / 2.0;
    // The liquid reaches about 12 from the centre in these runs
    let liquid = (0.0, 6.0);
    let interface = (9.0, 15.0);
    let vapour = (half - 8.0, half + 1.0);
    let kinetic: f64 = p.slabs.iter().flat_map(|s| s.kinetic).sum();
    let particles: f64 = p.slabs.iter().map(|s| s.particles).sum();
    Block {
        liquid: region(p, &density, centre, liquid),
        vapour: region(p, &density, centre, vapour),
        normal_pressure: p.total_pressure_tensor()[2],
        normal_by_region: [liquid, interface, vapour].map(|r| region(p, &normal, centre, r)),
        surface_tension: p.surface_tension(),
        temperature: kinetic / (3.0 * particles),
    }
}

/// A slab of liquid at T = 0.7 in a 30 x 30 x 60 box: equilibrates, then
/// samples the profiles every 20 steps in `blocks` blocks of 20,000 steps
fn coexistence(gpu: &Compute, shifted_force: bool, blocks: usize) -> Vec<Block> {
    let size = [30.0, 30.0, 60.0];
    // About 24 thick at liquid density, centred in the box
    let thickness = 24.0;
    let mut particles = lattice_box(15_600, [30.0, 30.0, thickness], 0.7, 7);
    for p in &mut particles {
        p.pos[2] += (size[2] - thickness) / 2.0;
    }
    let mut sim = LennardJones::new(
        &gpu.device,
        &gpu.queue,
        &particles,
        settings(size, shifted_force, 240),
    );
    sim.run(20_000);
    let start = std::time::Instant::now();
    let mut out = Vec::new();
    for _ in 0..blocks {
        let mut encoder = gpu.device.create_command_encoder(&Default::default());
        sim.clear_profile(&mut encoder);
        for sample in 0..1000 {
            sim.step(&mut encoder, 20, None);
            sim.sample_profile(&mut encoder);
            if sample % 10 == 9 {
                gpu.queue.submit([encoder.finish()]);
                encoder = gpu.device.create_command_encoder(&Default::default());
            }
        }
        gpu.queue.submit([encoder.finish()]);
        out.push(analyse(&sim.read_profile()));
    }
    let m = sim.measure_now();
    println!(
        "{} steps of {} particles in {:.1} s after equilibrating; momentum {:?}",
        blocks * 20_000,
        sim.count(),
        start.elapsed().as_secs_f64(),
        m.momentum
    );
    out
}

fn summary(blocks: &[Block]) -> [(f64, f64); 8] {
    let get = |f: &dyn Fn(&Block) -> f64| mean_error(&blocks.iter().map(f).collect::<Vec<_>>());
    [
        get(&|b| b.liquid),
        get(&|b| b.vapour),
        get(&|b| b.normal_pressure),
        get(&|b| b.normal_by_region[0]),
        get(&|b| b.normal_by_region[1]),
        get(&|b| b.normal_by_region[2]),
        get(&|b| b.surface_tension),
        get(&|b| b.temperature),
    ]
}

#[test]
fn force_shifted_slab_coexists_at_nist_densities() {
    let Some(gpu) = gpu() else { return };
    let blocks = coexistence(&gpu, true, 10);
    let [liquid, vapour, p_n, p_liq, p_int, p_vap, gamma, t] = summary(&blocks);
    // NIST SAT-TMMC, linear-force-shifted LJ at 2.5 sigma, T = 0.7, with 95%
    // confidence limits
    let (nist_liquid, nist_liquid_95) = (0.729309, 1.93161e-4);
    let (nist_vapour, nist_vapour_95) = (1.95113e-2, 1.67304e-6);
    let (nist_p, nist_p_95) = (1.18969e-2, 8.67525e-7);
    println!(
        "force-shifted LJ, T = 0.7 (measured {:.5} +/- {:.5}):\n  \
         liquid density {:.5} +/- {:.5} (NIST {nist_liquid})\n  \
         vapour density {:.6} +/- {:.6} (NIST {nist_vapour})\n  \
         normal pressure {:.6} +/- {:.6} (NIST p_sat {nist_p}); \
         liquid {:.5} +/- {:.5}, interfaces {:.5} +/- {:.5}, vapour {:.6} +/- {:.6}\n  \
         surface tension {:.4} +/- {:.4}",
        t.0,
        t.1,
        liquid.0,
        liquid.1,
        vapour.0,
        vapour.1,
        p_n.0,
        p_n.1,
        p_liq.0,
        p_liq.1,
        p_int.0,
        p_int.1,
        p_vap.0,
        p_vap.1,
        gamma.0,
        gamma.1
    );
    // Within three combined standard errors (NIST's 95% limits are about two)
    let agrees = |(x, e): (f64, f64), reference: f64, limit95: f64| {
        (x - reference).abs() < 3.0 * (e * e + (limit95 / 2.0).powi(2)).sqrt()
    };
    assert!((t.0 - 0.7).abs() < 0.002, "temperature held");
    assert!(
        agrees(liquid, nist_liquid, nist_liquid_95),
        "liquid density"
    );
    assert!(
        agrees(vapour, nist_vapour, nist_vapour_95),
        "vapour density"
    );
    assert!(agrees(p_n, nist_p, nist_p_95), "vapour pressure");
    // Mechanical equilibrium: the same normal pressure in every region
    for region in [p_liq, p_int, p_vap] {
        assert!(
            agrees(region, p_n.0, 2.0 * p_n.1),
            "normal pressure not flat"
        );
    }
}

#[test]
fn truncated_shifted_slab_has_published_surface_tension() {
    let Some(gpu) = gpu() else { return };
    let blocks = coexistence(&gpu, false, 10);
    let [liquid, vapour, p_n, _, _, _, gamma, t] = summary(&blocks);
    // Vrabec, Kedia, Fuchs and Hasse, Mol. Phys. 104, 1509 (2006):
    // gamma = 2.08 (1 - T / 1.0779)^1.21 for the truncated and shifted
    // potential at 2.5 sigma
    let published = 2.08 * (1.0 - 0.7 / 1.0779f64).powf(1.21);
    println!(
        "truncated and shifted LJ, T = 0.7 (measured {:.5} +/- {:.5}):\n  \
         liquid density {:.5} +/- {:.5}; vapour density {:.6} +/- {:.6}; \
         normal pressure {:.6} +/- {:.6}\n  \
         surface tension {:.4} +/- {:.4} (published correlation {published:.4}, {:+.1}%)",
        t.0,
        t.1,
        liquid.0,
        liquid.1,
        vapour.0,
        vapour.1,
        p_n.0,
        p_n.1,
        gamma.0,
        gamma.1,
        100.0 * (gamma.0 / published - 1.0)
    );
    assert!((t.0 - 0.7).abs() < 0.002, "temperature held");
    assert!(
        (gamma.0 / published - 1.0).abs() < 0.05,
        "surface tension more than 5% from published"
    );
}

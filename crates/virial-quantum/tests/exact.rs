//! The Schrodinger solver against exact solutions. Skipped (with a message)
//! on machines with no hardware GPU.

use virial_gpu::Compute;
use virial_quantum::{Absorber, Settings, Time, Wave, gaussian};

fn gpu() -> Option<Compute> {
    match pollster::block_on(Compute::hardware()) {
        Ok(gpu) => Some(gpu),
        Err(error) => {
            eprintln!("skipped: no GPU ({error})");
            None
        }
    }
}

#[test]
fn free_packet_moves_and_spreads_exactly() {
    let Some(gpu) = gpu() else { return };
    let (s0, k0) = (1.5f64, [2.0f64, 0.0]);
    let settings = Settings {
        n: 512,
        length: 80.0,
        dt: 0.01,
        time: Time::Real,
        absorber: None,
        detector: None,
    };
    let mut wave = Wave::new(
        &gpu.device,
        &gpu.queue,
        settings,
        gaussian([-10.0, 0.0], k0, s0),
        |_, _| 0.0,
    );
    let start = wave.measure_now();
    wave.run_steps(500);
    let end = wave.measure_now();
    let t = end.time;
    // |psi|^2 spread grows as s0 sqrt(1 + (t / (2 s0^2))^2); the centre moves at k0
    let spread = s0 * (1.0 + (t / (2.0 * s0 * s0)).powi(2)).sqrt();
    let kinetic = (k0[0] * k0[0] + 2.0 / (4.0 * s0 * s0)) / 2.0;
    println!(
        "free packet at t = {t}: centre {:.6} (exact {:.6}), spread {:.6} (exact {spread:.6}), norm {:.8} -> {:.8}, \
         kinetic {:.6} -> {:.6} (exact {kinetic:.6})",
        end.mean[0],
        -10.0 + k0[0] * t,
        end.spread[0],
        start.norm,
        end.norm,
        start.kinetic,
        end.kinetic
    );
    assert!((end.mean[0] - (-10.0 + k0[0] * t)).abs() < 1e-4);
    assert!((end.spread[0] / spread - 1.0).abs() < 1e-4);
    // 32-bit FFTs lose about 5e-8 of probability per step on an M1 (see
    // fft_round_trips_lose_less_than_1e_7_per_step); 500 steps stay well
    // under 1e-4
    assert!((end.norm - 1.0).abs() < 1e-4, "probability not conserved");
    assert!((end.kinetic / kinetic - 1.0).abs() < 1e-4);
}

#[test]
fn fft_round_trips_lose_less_than_1e_7_per_step() {
    let Some(gpu) = gpu() else { return };
    // dt = 0: every factor is exactly 1, so a step is a forward and an
    // inverse FFT and nothing else. What's lost is 32-bit FFT rounding.
    for n in [128u32, 512] {
        let settings = Settings {
            n,
            length: 80.0,
            dt: 0.0,
            time: Time::Real,
            absorber: None,
            detector: None,
        };
        let mut wave = Wave::new(
            &gpu.device,
            &gpu.queue,
            settings,
            gaussian([-10.0, 0.0], [2.0, 0.0], 1.5),
            |_, _| 0.0,
        );
        let start = wave.measure_now().norm;
        wave.run_steps(500);
        let per_step = (wave.measure_now().norm - start) / 500.0;
        println!("{n}^2 grid: probability change per FFT round trip {per_step:+.2e}");
        assert!(per_step.abs() < 1e-7);
    }
}

#[test]
fn coherent_state_oscillates_without_spreading() {
    let Some(gpu) = gpu() else { return };
    let (omega, a) = (1.0f64, 3.0f64);
    let width = (1.0 / (2.0 * omega)).sqrt();
    let settings = Settings {
        n: 256,
        length: 24.0,
        dt: std::f64::consts::PI / 1000.0,
        time: Time::Real,
        absorber: None,
        detector: None,
    };
    let mut wave = Wave::new(
        &gpu.device,
        &gpu.queue,
        settings,
        gaussian([a, 0.0], [0.0, 0.0], width),
        move |x, y| 0.5 * omega * omega * (x * x + y * y),
    );
    let start = wave.measure_now();
    // Half a period: the packet should be at -a, the same shape
    wave.run_steps(1000);
    let half = wave.measure_now();
    wave.run_steps(1000);
    let full = wave.measure_now();
    let energy = omega + 0.5 * omega * omega * a * a;
    println!(
        "coherent state: x(T/2) = {:.6} (exact {:.1}), x(T) = {:.6}; spread {:.6} (exact {width:.6}); energy {:.7} -> {:.7} (exact {energy})",
        half.mean[0],
        -a,
        full.mean[0],
        half.spread[0],
        start.energy(),
        full.energy()
    );
    assert!((half.mean[0] + a).abs() < 1e-4);
    assert!((full.mean[0] - a).abs() < 1e-4);
    assert!((half.spread[0] / width - 1.0).abs() < 1e-4);
    assert!((start.energy() / energy - 1.0).abs() < 1e-4);
    assert!(
        (full.energy() / start.energy() - 1.0).abs() < 1e-5,
        "energy not conserved"
    );
}

#[test]
fn imaginary_time_finds_the_ground_state() {
    let Some(gpu) = gpu() else { return };
    let omega = 1.0f64;
    let settings = Settings {
        n: 256,
        length: 24.0,
        dt: 0.005,
        time: Time::Imaginary,
        absorber: None,
        detector: None,
    };
    // Start off-centre, too wide and moving: relaxation removes all of it
    let mut wave = Wave::new(
        &gpu.device,
        &gpu.queue,
        settings,
        gaussian([1.5, -1.0], [0.5, 0.3], 2.0),
        move |x, y| 0.5 * omega * omega * (x * x + y * y),
    );
    wave.run_steps(4000);
    let m = wave.measure_now();
    println!(
        "harmonic ground state: energy {:.7} (exact {omega}), spread {:.6} (exact {:.6}), centre ({:.1e}, {:.1e})",
        m.energy(),
        m.spread[0],
        (1.0 / (2.0 * omega)).sqrt(),
        m.mean[0],
        m.mean[1]
    );
    assert!((m.energy() / omega - 1.0).abs() < 1e-4);
    assert!((m.spread[0] / (0.5f64).sqrt() - 1.0).abs() < 1e-3);
}

/// Transmission through V0 sech^2(x / a) (Landau & Lifshitz):
/// D = sinh^2(pi k a) / (sinh^2(pi k a) + cosh^2(pi/2 sqrt(8 V0 a^2 - 1)))
fn eckart(k: f64, v0: f64, a: f64) -> f64 {
    let s = (std::f64::consts::PI * k * a).sinh().powi(2);
    let c = (std::f64::consts::FRAC_PI_2 * (8.0 * v0 * a * a - 1.0).sqrt())
        .cosh()
        .powi(2);
    s / (s + c)
}

#[test]
fn packet_tunnels_through_a_barrier_at_the_exact_rate() {
    let Some(gpu) = gpu() else { return };
    let (v0, a, k0, sx) = (1.2f64, 1.0f64, 1.5f64, 6.0f64);
    let settings = Settings {
        n: 1024,
        length: 160.0,
        dt: 0.02,
        time: Time::Real,
        absorber: Some(Absorber {
            width: 12.0,
            strength: 1.0,
        }),
        detector: None,
    };
    let mut wave = Wave::new(
        &gpu.device,
        &gpu.queue,
        settings,
        gaussian([-50.0, 0.0], [k0, 0.0], sx),
        move |x, _| v0 / (x / a).cosh().powi(2),
    );
    wave.run_steps(2750);
    let psi = wave.read_psi();
    let n = settings.n as usize;
    let dx = settings.length / n as f64;
    let mut transmitted = 0.0;
    let mut total = 0.0;
    for j in 0..n {
        for i in 0..n {
            let x = (i as f64 - (n / 2) as f64) * dx;
            let p = psi[j * n + i];
            let w = (p[0] as f64).powi(2) + (p[1] as f64).powi(2);
            total += w;
            if x > 5.0 {
                transmitted += w;
            }
        }
    }
    transmitted *= dx * dx;
    total *= dx * dx;
    // Expected: D(k) averaged over the packet's momentum distribution in x,
    // a Gaussian of standard deviation 1 / (2 sx)
    let sk = 1.0 / (2.0 * sx);
    let (mut num, mut den) = (0.0, 0.0);
    for i in 0..20_001 {
        let k = k0 + (i as f64 - 10_000.0) / 10_000.0 * 8.0 * sk;
        let w = (-(k - k0).powi(2) / (2.0 * sk * sk)).exp();
        num += w * eckart(k, v0, a);
        den += w;
    }
    let expected = num / den;
    println!(
        "tunnelling through V0 sech^2(x/a), V0 = {v0}, E = {:.4}: transmitted {transmitted:.5} (exact {expected:.5}, \
         {:+.3}%), total probability {total:.6}",
        k0 * k0 / 2.0,
        (transmitted / expected - 1.0) * 100.0
    );
    assert!(
        (transmitted / expected - 1.0).abs() < 0.01,
        "transmission differs from the exact rate"
    );
}

#[test]
fn harmonic_levels_from_the_autocorrelation() {
    let Some(gpu) = gpu() else { return };
    // 2D harmonic well, omega = 0.2: levels at omega (n + 1), n = nx + ny
    let omega = 0.2f64;
    let settings = Settings {
        n: 256,
        length: 80.0,
        dt: 0.01,
        time: Time::Real,
        absorber: None,
        detector: None,
    };
    // A coherent state displaced by 3: mean n = omega x0^2 / 2 = 0.9
    let width = (1.0 / (2.0 * omega)).sqrt();
    let mut wave = Wave::new(
        &gpu.device,
        &gpu.queue,
        settings,
        gaussian([3.0, 0.0], [0.0, 0.0], width),
        |x, y| 0.5 * omega * omega * (x * x + y * y),
    );
    let (every, samples) = (5u32, 4000u32);
    wave.record_autocorrelation(samples);
    let mut encoder = gpu.device.create_command_encoder(&Default::default());
    wave.sample_autocorrelation(&mut encoder);
    gpu.queue.submit([encoder.finish()]);
    for _ in 1..samples {
        let mut encoder = gpu.device.create_command_encoder(&Default::default());
        wave.step(&mut encoder, every);
        wave.sample_autocorrelation(&mut encoder);
        gpu.queue.submit([encoder.finish()]);
    }
    let c = wave.read_autocorrelation();
    let dt = 0.01 * every as f64;
    let tau = 0.24 * dt * samples as f64;
    let levels = virial_quantum::energy_levels(&c, dt, tau, 0.05, 1.5, 1e-3);
    // A coherent state occupies level n with Poisson weight
    // e^-m m^n / n!, mean m = 0.9
    let mean = omega * 3.0f64.powi(2) / 2.0;
    let poisson = |n: f64| (-mean).exp() * mean.powf(n) / (1..=n as u32).product::<u32>() as f64;
    let (mut worst, mut worst_weight) = (0.0f64, 0.0f64);
    for (e, w) in &levels {
        let n = (e / omega - 1.0).round();
        let error = e / omega - (n + 1.0);
        worst = worst.max(error.abs());
        worst_weight = worst_weight.max((w - poisson(n)).abs());
        println!(
            "E = {e:.10} = omega ({:.8}), level n = {n}, weight {w:.5} (Poisson {:.5})",
            e / omega,
            poisson(n)
        );
    }
    println!(
        "{} levels from C(t) over t = 0..{:.0}, window tau {tau:.0}; worst error {worst:.2e} \
         in units of omega; weights within {worst_weight:.1e} of Poisson",
        levels.len(),
        dt * samples as f64
    );
    assert!(levels.len() >= 4, "too few levels found");
    assert!(worst < 1e-4, "levels not at omega (n + 1)");
    assert!(worst_weight < 1e-3, "weights not Poisson");
}

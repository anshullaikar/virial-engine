//! The detector screen and the double slit. Skipped (with a message) on
//! machines with no hardware GPU.

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

/// Probability on the grid with x above `x0`, a sample column. The boundary
/// runs through the middle of that column, so it counts half.
fn right_of(psi: &[[f32; 2]], n: usize, length: f64, x0: f64) -> f64 {
    let dx = length / n as f64;
    let mut total = 0.0;
    for j in 0..n {
        for i in 0..n {
            let x = (i as f64 - (n / 2) as f64) * dx;
            let weight = if (x - x0).abs() < dx / 2.0 {
                0.5
            } else if x > x0 {
                1.0
            } else {
                0.0
            };
            let p = psi[j * n + i];
            total += weight * ((p[0] as f64).powi(2) + (p[1] as f64).powi(2));
        }
    }
    total * dx * dx
}

/// Probability is conserved, so the current through the screen, summed over
/// the run, must equal the change in probability on its far side.
#[test]
fn detector_counts_the_probability_that_crosses_it() {
    let Some(gpu) = gpu() else { return };
    let (n, length) = (512usize, 80.0);
    let settings = Settings {
        n: n as u32,
        length,
        dt: 0.005,
        time: Time::Real,
        absorber: None,
        detector: Some(5.0),
    };
    let mut wave = Wave::new(
        &gpu.device,
        &gpu.queue,
        settings,
        gaussian([-5.0, 0.0], [3.0, 0.0], 2.0),
        |_, _| 0.0,
    );
    let x_d = wave.detector_x().expect("a detector");
    let before = right_of(&wave.read_psi(), n, length, x_d);
    wave.run_steps(1000);
    let after = right_of(&wave.read_psi(), n, length, x_d);
    let dx = length / n as f64;
    let crossed: f64 = wave.read_detector().iter().sum::<f64>() * dx;
    let expected = after - before;
    println!(
        "crossed {crossed:.6}, change on the far side {expected:.6} (difference {:.1e})",
        crossed - expected
    );
    assert!(
        (crossed - expected).abs() < 2e-4,
        "crossed {crossed}, expected {expected}"
    );
}

/// Dark fringes where the path difference from the two slits is a
/// half-integer number of wavelengths. The pattern is a smooth envelope
/// (each slit's own diffraction, the beam profile) times cos^2 of half the
/// phase difference: the envelope pulls the bright maxima toward the centre
/// but cannot move the zeros, so only the dark fringes are an exact test.
/// The bright ones are printed for comparison.
#[test]
fn double_slit_fringes_sit_where_the_path_difference_says() {
    let Some(gpu) = gpu() else { return };
    let (n, length) = (1024u32, 80.0);
    let (k0, wall, separation, width, screen) = (5.0f64, 0.5, 4.0, 1.0, 20.0);
    let lambda = std::f64::consts::TAU / k0;
    let settings = Settings {
        n,
        length,
        dt: 0.005,
        time: Time::Real,
        absorber: Some(Absorber {
            width: 8.0,
            strength: 10.0,
        }),
        detector: Some(screen),
    };
    let potential = move |x: f64, y: f64| {
        let in_slit = (y.abs() - separation / 2.0).abs() < width / 2.0;
        if x.abs() < wall / 2.0 && !in_slit {
            100.0
        } else {
            0.0
        }
    };
    let initial = move |x: f64, y: f64| {
        let envelope = (-(x + 14.0).powi(2) / (4.0 * 2.0 * 2.0) - y * y / (4.0 * 6.0 * 6.0)).exp();
        [envelope * (k0 * x).cos(), envelope * (k0 * x).sin()]
    };
    let mut wave = Wave::new(&gpu.device, &gpu.queue, settings, initial, potential);
    let d = wave.detector_x().expect("a detector");
    wave.run_steps(2800);
    let counts = wave.read_detector();
    let dy = length / n as f64;
    let y = |j: usize| (j as f64 - (n / 2) as f64) * dy;
    let crossed: f64 = counts.iter().sum::<f64>() * dy;

    // y where r1 - r2 = m lambda, by bisection on y > 0
    let path = |y: f64| {
        ((y + separation / 2.0).powi(2) + d * d).sqrt()
            - ((y - separation / 2.0).powi(2) + d * d).sqrt()
    };
    let solve = |m: f64| {
        let (mut lo, mut hi) = (0.0, 30.0);
        for _ in 0..100 {
            let mid = 0.5 * (lo + hi);
            if path(mid) < m * lambda {
                lo = mid
            } else {
                hi = mid
            }
        }
        0.5 * (lo + hi)
    };
    // Extremum near a guess, refined by a parabola through three samples
    let extremum = |guess: f64, find_max: bool| {
        let centre = (guess / dy).round() as i64 + (n / 2) as i64;
        let window = (1.5 / dy) as i64;
        let pick = (centre - window..=centre + window)
            .max_by(|&a, &b| {
                let (va, vb) = (counts[a as usize], counts[b as usize]);
                if find_max {
                    va.total_cmp(&vb)
                } else {
                    vb.total_cmp(&va)
                }
            })
            .expect("a window") as usize;
        let (a, b, c) = (counts[pick - 1], counts[pick], counts[pick + 1]);
        y(pick) + 0.5 * dy * (a - c) / (a - 2.0 * b + c)
    };
    println!(
        "screen at x = {d:.4}, lambda = {lambda:.4}, probability that reached it {crossed:.4}"
    );
    let mut worst = 0.0f64;
    for (m, bright) in [(0.5, false), (1.0, true), (1.5, false), (2.0, true)] {
        let exact = solve(m);
        let up = extremum(exact, bright);
        let down = -extremum(-exact, bright);
        let error = (0.5 * (up + down) - exact) / lambda;
        if !bright {
            worst = worst.max(error.abs());
        }
        println!(
            "{} fringe {m}: y = {up:.4} / {down:.4} (path-difference rule {exact:.4}), off by {error:+.4} wavelengths",
            if bright { "bright" } else { "dark" }
        );
    }
    assert!(worst < 0.05, "a dark fringe is off by {worst} wavelengths");
}

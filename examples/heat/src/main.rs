//! Heat flow. A Lennard-Jones liquid (density 0.8442) at constant energy,
//! with heat moved every step from a cold slab (at the bottom of the box,
//! which the periodic box joins to the top) to a hot slab in the middle by
//! eHEX (Wirnsberger,
//! Frenkel and Dellago 2015): velocities in each slab are scaled about its
//! centre of mass, so energy and momentum are conserved. The liquid between
//! settles into Fourier's law: a straight temperature profile whose slope is
//! the heat flux over the thermal conductivity.
//!
//! About 20,000 molecules (shifted-force cutoff at 2.5) in a periodic
//! 20 x 20 x 60 box. Reduced units (for argon: length 3.405e-10 m,
//! temperature 119.8 K, time 2.15e-12 s). Molecules are coloured by their
//! local temperature, m v^2 / 3 averaged over each molecule and its
//! neighbours within the cutoff.
//!
//! Recording ends with the temperature profile averaged from t = 60, and
//! the conductivity from its slope, in run.json (compare docs/VALIDATION.md,
//! where Fourier's law and Green-Kubo agree at lambda about 8).
//!
//! Settings: `steps` (steps per frame, default 21), `flux` (default 0.15).
//! Controls: drag to orbit, scroll to zoom, Space to pause, R to restart.
//!
//! Render a video natively:
//! `cargo run --release -p heat -- record=renders/heat seconds=40`

use virial::{Config, Context, Frame, Gpu, KeyCode, Scene, arg, wgpu};
use virial_particles::lennard_jones::{
    Drawn, HeatExchange, LennardJones, Measurement, Settings, Thermostat, lattice_box,
};
use virial_render::{Camera, CameraPath, Coloring, Colormap, Overlay, Particles, Style};

const SIZE: [f64; 3] = [20.0, 20.0, 60.0];
const DENSITY: f64 = 0.8442;
const DT: f64 = 0.002;
/// Hot slab in the middle; cold slab at the bottom, which the periodic box
/// also joins to the top, so heat flows from the middle to both ends
const HOT: (f64, f64) = (30.0, 33.0);
const COLD: (f64, f64) = (0.0, 3.0);
/// Profiles of the steady state are averaged from this time on
const PROFILE_FROM: f64 = 60.0;
const PROFILE_BINS: u32 = 60;
const T_RANGE: (f32, f32) = (0.8, 1.4);
/// Energy drift is measured from the first measurement after this time: the
/// starting lattice collapses into liquid first, with forces large enough
/// that Verlet's energy error jumps by about 1e-4 while it does
const MELTED: f64 = 5.0;

fn molecules() -> usize {
    (DENSITY * SIZE.iter().product::<f64>()).round() as usize
}

fn simulation(gpu: &Gpu, flux: f64) -> LennardJones {
    let settings = Settings {
        box_size: SIZE,
        cutoff: 2.5,
        dt: DT,
        thermostat: Thermostat::None,
        shifted_force: true,
        skin: 0.3,
        profile_bins: PROFILE_BINS,
    };
    let mut sim = LennardJones::new(
        &gpu.device,
        &gpu.queue,
        &lattice_box(molecules(), SIZE, 1.0, 11),
        settings,
    );
    // Flux J = energy / (2 A dt): heat leaves the hot slab both ways
    sim.set_heat_exchange(Some(HeatExchange {
        hot: HOT,
        cold: COLD,
        energy: flux * 2.0 * SIZE[0] * SIZE[1] * DT,
    }));
    sim.set_drawn(Drawn::Temperature);
    sim
}

struct Heat {
    flux: f64,
    sim: LennardJones,
    particles: Particles,
    overlay: Overlay,
    group: wgpu::BindGroup,
    camera: Camera,
    path: Option<CameraPath>,
    steps_per_frame: u32,
    paused: bool,
    frames: u64,
    profile_samples: u64,
    first: Option<Measurement>,
    latest: Option<Measurement>,
}

impl Heat {
    fn restart(&mut self, gpu: &Gpu) {
        self.sim = simulation(gpu, self.flux);
        self.group = self
            .particles
            .bind(&gpu.device, self.sim.positions(), self.sim.velocities());
        self.profile_samples = 0;
        self.first = None;
        self.latest = None;
    }

    fn readouts(&self) -> Vec<String> {
        let mut lines = vec![
            format!(
                "Heat flow: {} Lennard-Jones molecules, density {DENSITY}",
                molecules()
            ),
            format!(
                "t = {:.1}   heat flux {:.3} (hot slab middle, cold at ends)",
                self.sim.time(),
                self.flux
            ),
        ];
        if let Some(m) = self.latest {
            let drift = match self.first {
                Some(f) => format!(
                    "{:+.2e} since t = {:.1}",
                    (m.energy() - f.energy()) / f.energy().abs(),
                    f.time
                ),
                None => "from t = 5, after the lattice melts".into(),
            };
            lines.push(format!(
                "Mean T = {:.4}   energy drift {drift}",
                m.temperature(3.0 * molecules() as f64 - 3.0)
            ));
        }
        lines
    }
}

impl Scene for Heat {
    fn new(gpu: &Gpu) -> Self {
        let flux = arg("flux", 0.15f64).clamp(0.0, 1.0);
        let sim = simulation(gpu, flux);
        let particles = Particles::new(&gpu.device, gpu.surface_format());
        let overlay = Overlay::new(&gpu.device, gpu.surface_format());
        let group = particles.bind(&gpu.device, sim.positions(), sim.velocities());
        let camera = Camera {
            target: SIZE.map(|l| l as f32 / 2.0),
            yaw: arg("yaw", 0.5),
            pitch: arg("pitch", 1.35),
            distance: arg("distance", 88.0),
            fov: 0.8,
        };
        Self {
            flux,
            sim,
            particles,
            overlay,
            group,
            camera,
            path: None,
            steps_per_frame: arg("steps", 21u32).clamp(1, 2000),
            paused: false,
            frames: 0,
            profile_samples: 0,
            first: None,
            latest: None,
        }
    }

    fn update(&mut self, ctx: &mut Context<'_>) {
        if self.frames == 0 && ctx.data.is_recording() {
            if arg("path", 1u32) != 0 {
                let c = self.camera;
                self.path = Some(CameraPath::new().key(0.0, c).key(
                    40.0,
                    Camera {
                        yaw: c.yaw + 0.9,
                        pitch: c.pitch - 0.1,
                        ..c
                    },
                ));
            }
            let data = &mut *ctx.data;
            data.meta("molecules", molecules());
            data.meta("box", format!("{SIZE:?}"));
            data.meta("density", DENSITY);
            data.meta("dt", DT);
            data.meta("cutoff", "2.5, shifted force");
            data.meta("heat_flux", self.flux);
            data.meta("hot_slab", format!("{HOT:?}"));
            data.meta("cold_slab", format!("{COLD:?}"));
            data.meta("steps_per_frame", self.steps_per_frame);
        }
        let input = ctx.input;
        if input.pressed(KeyCode::Space) {
            self.paused = !self.paused;
        }
        if input.pressed(KeyCode::KeyR) {
            self.restart(ctx.gpu);
        }
        match &self.path {
            Some(path) => self.camera = path.at(ctx.time.elapsed as f32),
            None => self.camera.orbit(input.drag(), input.scroll()),
        }

        self.sim.after_submit();
        for m in self.sim.poll() {
            if self.first.is_none() && m.time >= MELTED {
                self.first = Some(m);
            }
            self.latest = Some(m);
            // NaN until the lattice has melted
            let drift = self
                .first
                .map_or(f64::NAN, |f| (m.energy() - f.energy()) / f.energy().abs());
            ctx.data.row(&[
                ("step", m.step as f64),
                ("time", m.time),
                ("kinetic", m.kinetic),
                ("potential", m.potential),
                ("energy", m.energy()),
                ("relative_drift", drift),
                ("temperature", m.temperature(3.0 * molecules() as f64 - 3.0)),
                ("momentum_z", m.momentum[2]),
            ]);
        }
        for (label, line) in ["Scene", "Clock", "Energy"].iter().zip(self.readouts()) {
            ctx.stats.set(label, line);
        }
    }

    fn render(&mut self, ctx: &mut Context<'_>, frame: Frame<'_>) {
        if !self.paused {
            self.sim.step(
                frame.encoder,
                self.steps_per_frame,
                frame.timer.and_then(|t| t.compute("molecules")),
            );
        }
        // Every frame: the measurement also refreshes the local temperatures
        // the colours come from
        self.sim.measure(frame.encoder);
        if !cfg!(target_arch = "wasm32")
            && !self.paused
            && self.sim.time() >= PROFILE_FROM
            && self.frames.is_multiple_of(4)
        {
            self.sim.sample_profile(frame.encoder);
            self.profile_samples += 1;
        }
        self.frames += 1;

        let gpu = ctx.gpu;
        let size = (gpu.config.width, gpu.config.height);
        self.particles.draw(
            &gpu.device,
            &gpu.queue,
            frame.encoder,
            frame.view,
            size,
            &self.camera,
            &self.group,
            molecules() as u32,
            Style::Spheres { radius: 0.5 },
            Coloring::Value {
                map: Colormap::Inferno,
                min: T_RANGE.0,
                max: T_RANGE.1,
            },
            [0.012, 0.013, 0.02],
            frame.timer.and_then(|t| t.render("draw")),
        );
        let scale = (size.1 as f32 / 540.0).round().max(1.0);
        if !cfg!(target_arch = "wasm32") {
            self.overlay
                .lines(16.0 * scale, 16.0 * scale, scale, &self.readouts());
        }
        let (w, x) = (220.0 * scale, size.0 as f32 - 240.0 * scale);
        let range = (format!("{:.1}", T_RANGE.0), format!("{:.1}", T_RANGE.1));
        self.overlay.color_bar(
            x,
            size.1 as f32 - 60.0 * scale,
            w,
            10.0 * scale,
            scale,
            Colormap::Inferno,
            (&range.0, &range.1),
            "Local temperature",
        );
        self.overlay
            .draw(&gpu.device, &gpu.queue, frame.encoder, frame.view, size);
    }

    fn finish(&mut self, ctx: &mut Context<'_>) {
        #[cfg(not(target_arch = "wasm32"))]
        if self.profile_samples > 0 {
            let p = self.sim.read_profile();
            let t = p.temperature();
            let z: Vec<f64> = p.slabs.iter().map(|s| s.z).collect();
            // Linear regions 4 clear of each slab: heat flows from the hot
            // slab (30..33) down to the cold one at 60 = 0 on both sides
            let slope = |lo: f64, hi: f64| {
                let (zs, ts): (Vec<f64>, Vec<f64>) = z
                    .iter()
                    .zip(&t)
                    .filter(|(z, _)| **z > lo && **z < hi)
                    .map(|(z, t)| (*z, *t))
                    .unzip();
                let n = zs.len() as f64;
                let (mz, mt) = (zs.iter().sum::<f64>() / n, ts.iter().sum::<f64>() / n);
                zs.iter()
                    .zip(&ts)
                    .map(|(a, b)| (a - mz) * (b - mt))
                    .sum::<f64>()
                    / zs.iter().map(|a| (a - mz).powi(2)).sum::<f64>()
            };
            let gradient = 0.5 * (slope(7.0, 26.0) - slope(37.0, 56.0));
            let conductivity = self.flux / gradient;
            println!(
                "steady state, {} samples from t = {PROFILE_FROM}: gradient {gradient:.5}, \
                 conductivity {conductivity:.3} (flux {})",
                p.samples, self.flux
            );
            let data = &mut *ctx.data;
            data.meta("profile_samples", p.samples);
            data.meta("temperature_gradient", gradient);
            data.meta("conductivity", conductivity);
            data.snapshot("temperature_profile", &[t.len()], &t);
        }
        #[cfg(target_arch = "wasm32")]
        let _ = ctx;
    }
}

fn main() {
    virial::run::<Heat>(Config {
        title: "Virial: heat",
    });
}

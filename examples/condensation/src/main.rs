//! Condensation. A Lennard-Jones fluid starts above its critical temperature
//! (T = 1.0; critical point 0.937 for this potential) and is cooled to
//! T = 0.7, well inside the liquid-vapour coexistence region. It separates:
//! dense patches appear everywhere, coarsen and merge into slabs of liquid
//! with vapour between. (The long box makes slabs win over columns, which
//! would have about three times the surface. Flat slabs then stay apart:
//! none has curvature to drive the others to evaporate into it.)
//!
//! 20,250 molecules (shifted-force cutoff at 2.5) in a periodic
//! 30 x 30 x 90 box, overall density 0.25, Nose-Hoover thermostat whose
//! target follows the schedule. Reduced units (for argon: length
//! 3.405e-10 m, temperature 119.8 K, time 2.15e-12 s).
//!
//! Molecules are coloured by their neighbours within 1.5 sigma, a local
//! density (about 14.1 rho neighbours at density rho). A molecule counts as
//! condensed with 6 or more: local density above 0.37, halfway between the
//! vapour (0.0195) and the liquid (0.729) at T = 0.7.
//!
//! Recording ends with the liquid and vapour densities of the final slabs,
//! averaged from t = 350, in run.json, to compare with NIST's coexistence
//! values (see docs/VALIDATION.md).
//!
//! Settings: `steps` (steps per frame, default 25).
//! Controls: drag to orbit, scroll to zoom, Space to pause, R to restart.
//!
//! Render a video natively:
//! `cargo run --release -p condensation -- record=renders/condensation seconds=60`

use virial::{Config, Context, Frame, Gpu, KeyCode, Scene, arg, wgpu};
#[cfg(not(target_arch = "wasm32"))]
use virial_particles::lennard_jones::Phase;
use virial_particles::lennard_jones::{
    LennardJones, Measurement, NoseHoover, Settings, Thermostat, lattice_box,
};
use virial_render::{Camera, CameraPath, Coloring, Colormap, Overlay, Particles, Style};

const SIZE: [f64; 3] = [30.0, 30.0, 90.0];
const MOLECULES: usize = 20_250;
const DT: f64 = 0.005;
const HOT: f64 = 1.0;
const COLD: f64 = 0.7;
/// Held at HOT until this time, then cooled linearly to COLD over COOLING
const HOLD: f64 = 15.0;
const COOLING: f64 = 50.0;
/// Neighbours within 1.5 sigma that make a molecule condensed
const NEIGHBOURS: u32 = 6;
/// Profiles of the final slab are averaged from this time on
const PROFILE_FROM: f64 = 350.0;
const PROFILE_BINS: u32 = 180;

/// The thermostat's target at time `t`
fn schedule(t: f64) -> f64 {
    let f = ((t - HOLD) / COOLING).clamp(0.0, 1.0);
    HOT + (COLD - HOT) * f
}

fn simulation(gpu: &Gpu) -> LennardJones {
    let settings = Settings {
        box_size: SIZE,
        cutoff: 2.5,
        dt: DT,
        thermostat: Thermostat::NoseHoover(NoseHoover {
            temperature: HOT,
            period: 0.5,
        }),
        shifted_force: true,
        skin: 0.3,
        profile_bins: PROFILE_BINS,
    };
    let mut sim = LennardJones::new(
        &gpu.device,
        &gpu.queue,
        &lattice_box(MOLECULES, SIZE, HOT, 7),
        settings,
    );
    sim.set_condensed_neighbours(NEIGHBOURS);
    sim
}

struct Condensation {
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
    latest: Option<Measurement>,
}

impl Condensation {
    fn restart(&mut self, gpu: &Gpu) {
        self.sim = simulation(gpu);
        self.group = self
            .particles
            .bind(&gpu.device, self.sim.positions(), self.sim.velocities());
        self.profile_samples = 0;
        self.latest = None;
    }

    fn readouts(&self) -> Vec<String> {
        let target = self.sim.target_temperature().unwrap_or(HOT);
        let mut lines = vec![
            format!("Condensation: {MOLECULES} Lennard-Jones molecules"),
            format!(
                "t = {:.1}   thermostat {target:.3}   density 0.25",
                self.sim.time()
            ),
        ];
        if let Some(m) = self.latest {
            lines.push(format!(
                "T = {:.4}   P = {:.5}",
                m.temperature(3.0 * MOLECULES as f64 - 3.0),
                m.pressure()
            ));
            lines.push(format!(
                "Condensed {:.1}%   (6+ neighbours within 1.5 sigma)",
                100.0 * m.condensed as f64 / MOLECULES as f64
            ));
        }
        lines
    }
}

impl Scene for Condensation {
    fn new(gpu: &Gpu) -> Self {
        let sim = simulation(gpu);
        let particles = Particles::new(&gpu.device, gpu.surface_format());
        let overlay = Overlay::new(&gpu.device, gpu.surface_format());
        let group = particles.bind(&gpu.device, sim.positions(), sim.velocities());
        let camera = Camera {
            target: SIZE.map(|l| l as f32 / 2.0),
            yaw: arg("yaw", 0.5),
            pitch: arg("pitch", 1.32),
            distance: arg("distance", 125.0),
            fov: 0.8,
        };
        Self {
            sim,
            particles,
            overlay,
            group,
            camera,
            path: None,
            steps_per_frame: arg("steps", 25u32).clamp(1, 2000),
            paused: false,
            frames: 0,
            profile_samples: 0,
            latest: None,
        }
    }

    fn update(&mut self, ctx: &mut Context<'_>) {
        if self.frames == 0 && ctx.data.is_recording() {
            if arg("path", 1u32) != 0 {
                let c = self.camera;
                self.path = Some(CameraPath::new().key(0.0, c).key(
                    60.0,
                    Camera {
                        yaw: c.yaw + 1.2,
                        pitch: c.pitch - 0.15,
                        ..c
                    },
                ));
            }
            let data = &mut *ctx.data;
            data.meta("molecules", MOLECULES);
            data.meta("box", format!("{SIZE:?}"));
            data.meta("dt", DT);
            data.meta("cutoff", "2.5, shifted force");
            data.meta("thermostat", "Nose-Hoover, period 0.5");
            data.meta(
                "schedule",
                format!(
                    "T = {HOT} until t = {HOLD}, linear to {COLD} by t = {}, then held; \
                     target updated once per frame",
                    HOLD + COOLING
                ),
            );
            data.meta("condensed_neighbours_within_1.5", NEIGHBOURS);
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
        // The target for the steps this frame is about to take
        let target = schedule(self.sim.time());
        if self.sim.target_temperature() != Some(target) {
            self.sim.set_temperature(target);
        }

        self.sim.after_submit();
        for m in self.sim.poll() {
            self.latest = Some(m);
            ctx.data.row(&[
                ("step", m.step as f64),
                ("time", m.time),
                ("target_temperature", schedule(m.time)),
                ("temperature", m.temperature(3.0 * MOLECULES as f64 - 3.0)),
                ("kinetic", m.kinetic),
                ("potential", m.potential),
                ("pressure", m.pressure()),
                ("condensed_fraction", m.condensed as f64 / MOLECULES as f64),
            ]);
        }
        for (label, line) in ["Scene", "Clock", "Thermodynamics", "Condensed"]
            .iter()
            .zip(self.readouts())
        {
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
        // Every frame: the measurement also refreshes the neighbour counts
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
            MOLECULES as u32,
            Style::Spheres { radius: 0.5 },
            Coloring::Value {
                map: Colormap::Viridis,
                min: 0.0,
                max: 12.0,
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
        self.overlay.color_bar(
            x,
            size.1 as f32 - 60.0 * scale,
            w,
            10.0 * scale,
            scale,
            Colormap::Viridis,
            ("0", "12"),
            "Neighbours within 1.5 sigma",
        );
        self.overlay
            .draw(&gpu.device, &gpu.queue, frame.encoder, frame.view, size);
    }

    fn finish(&mut self, ctx: &mut Context<'_>) {
        #[cfg(not(target_arch = "wasm32"))]
        if self.profile_samples > 0 {
            let p = self.sim.read_profile();
            let density = p.density();
            // Interfaces where the density crosses 0.37, bulk at least 6 from
            // any, as in the coexistence test; any number of slabs
            let phases = p.phases(0.37, 6.0);
            let slabs = p.interfaces(0.37).len() / 2;
            let mean = |phase| p.mean_in(&density, &phases, phase).unwrap_or(f64::NAN);
            let (liquid, vapour) = (mean(Phase::Liquid), mean(Phase::Vapour));
            let normal: Vec<f64> = p.pressure_tensor().iter().map(|t| t[2]).collect();
            println!(
                "final state, {} samples from t = {PROFILE_FROM}: {slabs} slab(s) of liquid; \
                 liquid density {liquid:.5}, vapour density {vapour:.5}, normal pressure {:.5} \
                 (NIST at T = 0.7: 0.72931, 0.019511, 0.011897)",
                p.samples,
                p.total_pressure_tensor()[2]
            );
            let data = &mut *ctx.data;
            data.meta("profile_samples", p.samples);
            data.meta("final_liquid_slabs", slabs);
            data.meta("final_liquid_density", liquid);
            data.meta("final_vapour_density", vapour);
            data.meta("final_normal_pressure", p.total_pressure_tensor()[2]);
            data.snapshot("density_profile", &[density.len()], &density);
            data.snapshot("normal_pressure_profile", &[normal.len()], &normal);
        }
        #[cfg(target_arch = "wasm32")]
        let _ = ctx;
    }
}

fn main() {
    virial::run::<Condensation>(Config {
        title: "Virial: condensation",
    });
}

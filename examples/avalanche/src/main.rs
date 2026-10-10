//! A granular avalanche, in SI units. About 21,000 glass beads (radius 2.5 mm,
//! 10% size spread, friction 0.5) are released onto a 30 degree slope in a
//! 40 cm box. They pour down, spread and pile up against the end wall, losing
//! energy in collisions and friction.
//!
//! Contact physics: Hertz elasticity with viscoelastic damping, Coulomb
//! friction at the contact point (beads spin), rolling resistance 0.02,
//! stiffness softened to E* = 2 MPa (see virial_particles::granular).
//!
//! Settings: `angle` (degrees, default 30), `steps` (steps of 20 us per
//! frame, default 50, so 1 ms of real time per frame).
//! Controls: drag to orbit, scroll to zoom, Space to pause, R to restart.
//!
//! Render a video natively (1.8 s of real time, slowed 17x):
//! `cargo run --release -p avalanche -- record=renders/avalanche seconds=30`

use virial::{Config, Context, Frame, Gpu, KeyCode, Scene, arg, wgpu};
use virial_particles::granular::{Grain, Granular, Measurement, Settings, block};
use virial_render::{Camera, CameraPath, Coloring, Colormap, Overlay, Particles, Plane, Style};

const BOX: f64 = 0.4;
const RADIUS: f64 = 2.5e-3;
const DT: f64 = 2e-5;
/// Floor height at the low end (x = BOX)
const FLOOR_LOW: f64 = 0.02;

fn settings(angle: f64) -> Settings {
    let mut s = Settings::glass(BOX, DT);
    s.floor_angle = angle;
    s.floor_height = FLOOR_LOW;
    // A little rolling resistance, as real beads have
    s.rolling_friction = 0.02;
    s
}

/// Floor height at position x along the slope
fn floor_at(angle: f64, x: f64) -> f64 {
    FLOOR_LOW + (BOX - x) * angle.tan()
}

/// A block of beads just above the top of the slope
fn beads(angle: f64) -> Vec<Grain> {
    let z0 = floor_at(angle, 0.01) + 0.004;
    block(
        [0.006, 0.006, z0],
        [0.12, BOX - 0.006, z0 + 0.1],
        RADIUS,
        0.1,
        21,
    )
}

struct Avalanche {
    angle: f64,
    /// Total weight of the beads, N
    weight: f64,
    sim: Granular,
    particles: Particles,
    overlay: Overlay,
    group: wgpu::BindGroup,
    camera: Camera,
    path: Option<CameraPath>,
    steps_per_frame: u32,
    paused: bool,
    frames: u64,
    first: Option<Measurement>,
    latest: Option<Measurement>,
}

impl Avalanche {
    fn start(&mut self, gpu: &Gpu) {
        self.sim = Granular::new(
            &gpu.device,
            &gpu.queue,
            &beads(self.angle),
            settings(self.angle),
        );
        self.group = self
            .particles
            .bind(&gpu.device, self.sim.positions(), self.sim.velocities());
        self.first = None;
        self.latest = None;
    }

    fn readouts(&self) -> Vec<String> {
        let mut lines = vec![
            format!(
                "Granular avalanche: {} glass beads, r = 2.5 mm",
                self.sim.count()
            ),
            format!(
                "t = {:.3} s   slope {:.0} deg   friction 0.5",
                self.sim.time(),
                self.angle.to_degrees()
            ),
        ];
        if let (Some(first), Some(m)) = (self.first, self.latest) {
            lines.push(format!(
                "Kinetic {:.4} J   dissipated {:.4} J",
                m.kinetic,
                first.energy() - m.energy()
            ));
            lines.push(format!(
                "Floor and walls carry {:.3} of the weight",
                m.support / self.weight
            ));
        }
        lines
    }
}

impl Scene for Avalanche {
    fn new(gpu: &Gpu) -> Self {
        let angle = arg("angle", 30.0f64).clamp(0.0, 60.0).to_radians();
        let grains = beads(angle);
        let s = settings(angle);
        let weight = grains
            .iter()
            .map(|g| s.density * 4.0 / 3.0 * std::f64::consts::PI * g.radius.powi(3) * s.gravity)
            .sum();
        let sim = Granular::new(&gpu.device, &gpu.queue, &grains, s);
        let mut particles = Particles::new(&gpu.device, gpu.surface_format());
        let overlay = Overlay::new(&gpu.device, gpu.surface_format());
        let group = particles.bind(&gpu.device, sim.positions(), sim.velocities());

        // The tilted floor, back wall and end wall, with a 5 cm grid
        let b = BOX as f32;
        let (z_high, z_low) = (floor_at(angle, 0.0) as f32, FLOOR_LOW as f32);
        particles.set_planes(
            &gpu.queue,
            &[
                Plane {
                    corners: [
                        [0.0, 0.0, z_high],
                        [b, 0.0, z_low],
                        [b, b, z_low],
                        [0.0, b, z_high],
                    ],
                    color: [0.035, 0.038, 0.05],
                    grid: 0.05,
                },
                Plane {
                    corners: [
                        [0.0, b, 0.0],
                        [b, b, 0.0],
                        [b, b, b * 0.75],
                        [0.0, b, b * 0.75],
                    ],
                    color: [0.02, 0.022, 0.03],
                    grid: 0.05,
                },
                Plane {
                    corners: [
                        [b, 0.0, 0.0],
                        [b, b, 0.0],
                        [b, b, b * 0.5],
                        [b, 0.0, b * 0.5],
                    ],
                    color: [0.024, 0.026, 0.035],
                    grid: 0.05,
                },
            ],
        );
        let camera = Camera {
            target: [b * 0.52, b * 0.5, b * 0.2],
            yaw: arg("yaw", 0.3),
            pitch: arg("pitch", 0.98),
            distance: arg("distance", 0.95),
            fov: 0.7,
        };
        Self {
            angle,
            weight,
            sim,
            particles,
            overlay,
            group,
            camera,
            path: None,
            steps_per_frame: arg("steps", 50u32).clamp(1, 500),
            paused: false,
            frames: 0,
            first: None,
            latest: None,
        }
    }

    fn update(&mut self, ctx: &mut Context<'_>) {
        if self.frames == 0 && ctx.data.is_recording() {
            if arg("path", 1u32) != 0 {
                let c = self.camera;
                self.path = Some(CameraPath::new().key(0.0, c).key(
                    30.0,
                    Camera {
                        yaw: c.yaw + 0.5,
                        pitch: c.pitch - 0.12,
                        distance: c.distance * 0.9,
                        ..c
                    },
                ));
            }
            let data = &mut *ctx.data;
            data.meta("beads", self.sim.count());
            data.meta("radius_m", RADIUS);
            data.meta("slope_degrees", self.angle.to_degrees());
            data.meta("dt_s", DT);
            data.meta("steps_per_frame", self.steps_per_frame);
            data.meta("weight_N", self.weight);
        }
        let input = ctx.input;
        if input.pressed(KeyCode::Space) {
            self.paused = !self.paused;
        }
        if input.pressed(KeyCode::KeyR) {
            self.start(ctx.gpu);
        }
        match &self.path {
            Some(path) => self.camera = path.at(ctx.time.elapsed as f32),
            None => self.camera.orbit(input.drag(), input.scroll()),
        }

        self.sim.after_submit();
        for m in self.sim.poll() {
            let first = *self.first.get_or_insert(m);
            self.latest = Some(m);
            ctx.data.row(&[
                ("step", m.step as f64),
                ("time_s", m.time),
                ("kinetic_J", m.kinetic),
                ("gravitational_J", m.gravitational),
                ("elastic_J", m.elastic),
                ("dissipated_J", first.energy() - m.energy()),
                ("support_N", m.support),
            ]);
        }
        for (label, line) in ["Scene", "Clock", "Energy", "Support"]
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
                frame.timer.and_then(|t| t.compute("grains")),
            );
        }
        if self.frames.is_multiple_of(5) {
            self.sim.measure(frame.encoder);
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
            self.sim.count(),
            Style::Spheres {
                radius: RADIUS as f32,
            },
            Coloring::Speed {
                map: Colormap::Viridis,
                min: 0.0,
                max: 1.2,
            },
            [0.015, 0.016, 0.022],
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
            ("0", "1.2 m/s"),
            "Speed",
        );
        self.overlay
            .draw(&gpu.device, &gpu.queue, frame.encoder, frame.view, size);
    }
}

fn main() {
    virial::run::<Avalanche>(Config {
        title: "Virial: avalanche",
    });
}

//! Drawing for Virial.
//!
//! - [`Camera`] and [`CameraPath`]: an orbit camera, and keyframes for videos
//! - [`Particles`]: point masses as glowing sprites or shaded spheres,
//!   coloured by group or by speed through a [`Colormap`]
//! - [`Overlay`]: text, panels and colour bars drawn into the frame, so
//!   readouts appear in recorded videos too

mod font;
mod overlay;
mod particles;

pub use overlay::{Overlay, text_width};
pub use particles::{Coloring, Particles, Plane, Style};

pub type Mat4 = [[f32; 4]; 4];

/// WGSL for the colour maps: `fn colormap(map: u32, x: f32) -> vec3f`, map 0
/// viridis and 1 inferno, for scenes with their own shaders.
pub const COLORMAPS_WGSL: &str = include_str!("colormaps.wgsl");

/// Perceptually uniform colour maps (CC0, by Smith, van der Walt and Firing).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Colormap {
    Viridis = 0,
    Inferno = 1,
}

/// A camera orbiting `target`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Camera {
    pub target: [f32; 3],
    /// Angle around the z axis, radians
    pub yaw: f32,
    /// Angle down from the z axis, radians (0 looks straight down)
    pub pitch: f32,
    pub distance: f32,
    /// Vertical field of view, radians
    pub fov: f32,
}

impl Default for Camera {
    fn default() -> Self {
        Self {
            target: [0.0; 3],
            yaw: 0.35,
            pitch: 0.9,
            distance: 14.0,
            fov: 0.8,
        }
    }
}

impl Camera {
    pub fn eye(&self) -> [f32; 3] {
        let (sp, cp) = self.pitch.sin_cos();
        let (sy, cy) = self.yaw.sin_cos();
        [
            self.target[0] + self.distance * sp * sy,
            self.target[1] - self.distance * sp * cy,
            self.target[2] + self.distance * cp,
        ]
    }

    pub fn view(&self) -> Mat4 {
        look_at(self.eye(), self.target, [0.0, 0.0, 1.0])
    }

    pub fn projection(&self, aspect: f32) -> Mat4 {
        let near = (self.distance * 0.002).max(1e-6);
        perspective(self.fov, aspect, near, self.distance * 50.0)
    }

    /// Drag to orbit, scroll to zoom: the usual mouse controls.
    pub fn orbit(&mut self, drag: [f32; 2], scroll: f32) {
        self.yaw -= drag[0] * 2.5;
        self.pitch = (self.pitch + drag[1] * 2.5).clamp(0.02, 3.12);
        self.distance *= 1.0 - scroll * 0.1;
    }
}

/// Camera keyframes at times in seconds; poses between them are eased.
#[derive(Clone, Debug, Default)]
pub struct CameraPath {
    keys: Vec<(f32, Camera)>,
}

impl CameraPath {
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds a keyframe. Keys can be added in any order.
    pub fn key(mut self, time: f32, camera: Camera) -> Self {
        self.keys.push((time, camera));
        self.keys.sort_by(|a, b| a.0.total_cmp(&b.0));
        self
    }

    /// The camera at `time`, eased in and out between keyframes. Distances are
    /// interpolated logarithmically so zooms feel even.
    pub fn at(&self, time: f32) -> Camera {
        let Some(first) = self.keys.first() else {
            return Camera::default();
        };
        if time <= first.0 {
            return first.1;
        }
        for pair in self.keys.windows(2) {
            let ((t0, a), (t1, b)) = (pair[0], pair[1]);
            if time <= t1 {
                let x = ((time - t0) / (t1 - t0).max(1e-9)).clamp(0.0, 1.0);
                let s = x * x * (3.0 - 2.0 * x);
                let mix = |p: f32, q: f32| p + (q - p) * s;
                return Camera {
                    target: [0, 1, 2].map(|k| mix(a.target[k], b.target[k])),
                    yaw: mix(a.yaw, b.yaw),
                    pitch: mix(a.pitch, b.pitch),
                    distance: mix(a.distance.ln(), b.distance.ln()).exp(),
                    fov: mix(a.fov, b.fov),
                };
            }
        }
        self.keys.last().expect("not empty").1
    }
}

/// Right-handed perspective projection with WebGPU's 0..1 depth.
pub fn perspective(fov_y: f32, aspect: f32, near: f32, far: f32) -> Mat4 {
    let f = 1.0 / (fov_y / 2.0).tan();
    [
        [f / aspect, 0.0, 0.0, 0.0],
        [0.0, f, 0.0, 0.0],
        [0.0, 0.0, far / (near - far), -1.0],
        [0.0, 0.0, near * far / (near - far), 0.0],
    ]
}

pub fn look_at(eye: [f32; 3], target: [f32; 3], up: [f32; 3]) -> Mat4 {
    let sub = |a: [f32; 3], b: [f32; 3]| [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
    let dot = |a: [f32; 3], b: [f32; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    let cross = |a: [f32; 3], b: [f32; 3]| {
        [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ]
    };
    let norm = |a: [f32; 3]| {
        let l = dot(a, a).sqrt();
        [a[0] / l, a[1] / l, a[2] / l]
    };
    let f = norm(sub(target, eye));
    let s = norm(cross(f, up));
    let u = cross(s, f);
    [
        [s[0], u[0], -f[0], 0.0],
        [s[1], u[1], -f[1], 0.0],
        [s[2], u[2], -f[2], 0.0],
        [-dot(s, eye), -dot(u, eye), dot(f, eye), 1.0],
    ]
}

/// Column-major matrix product a × b.
pub fn multiply(a: Mat4, b: Mat4) -> Mat4 {
    let mut out = [[0.0; 4]; 4];
    for (c, col) in out.iter_mut().enumerate() {
        for (r, value) in col.iter_mut().enumerate() {
            *value = (0..4).map(|k| a[k][r] * b[c][k]).sum();
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn camera_path_hits_its_keyframes_and_eases_between() {
        let a = Camera {
            distance: 10.0,
            yaw: 0.0,
            ..Camera::default()
        };
        let b = Camera {
            distance: 40.0,
            yaw: 1.0,
            ..Camera::default()
        };
        let path = CameraPath::new().key(2.0, b).key(0.0, a);
        assert_eq!(path.at(-1.0), a);
        assert_eq!(path.at(0.0), a);
        assert_eq!(path.at(2.0), b);
        assert_eq!(path.at(5.0), b);
        let mid = path.at(1.0);
        assert!((mid.yaw - 0.5).abs() < 1e-6);
        assert!(
            (mid.distance - 20.0).abs() < 1e-3,
            "log-interpolated distance"
        );
    }
}

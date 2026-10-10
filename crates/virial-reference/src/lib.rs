//! CPU reference solvers, generic over f32 and f64.
//!
//! Each matches a GPU solver's maths step for step, so the only difference
//! between them is precision and summation order.

// Index loops over x, y, z mirror the GPU kernels line for line
#![allow(clippy::needless_range_loop)]

use std::ops::{Add, AddAssign, Div, Mul, Neg, Sub};

/// The float operations the solvers need. Implemented for f32 and f64.
pub trait Real:
    Copy
    + PartialEq
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self>
    + AddAssign
{
    const ZERO: Self;
    fn from_f64(x: f64) -> Self;
    fn to_f64(self) -> f64;
    fn sqrt(self) -> Self;
}

impl Real for f32 {
    const ZERO: Self = 0.0;
    fn from_f64(x: f64) -> Self {
        x as f32
    }
    fn to_f64(self) -> f64 {
        self as f64
    }
    fn sqrt(self) -> Self {
        f32::sqrt(self)
    }
}

impl Real for f64 {
    const ZERO: Self = 0.0;
    fn from_f64(x: f64) -> Self {
        x
    }
    fn to_f64(self) -> f64 {
        self
    }
    fn sqrt(self) -> Self {
        f64::sqrt(self)
    }
}

/// A point mass. SI units by convention; any consistent units work.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Body<T> {
    pub pos: [T; 3],
    pub vel: [T; 3],
    pub mass: T,
}

impl<T: Real> Body<T> {
    pub fn convert<U: Real>(&self) -> Body<U> {
        Body {
            pos: self.pos.map(|x| U::from_f64(x.to_f64())),
            vel: self.vel.map(|x| U::from_f64(x.to_f64())),
            mass: U::from_f64(self.mass.to_f64()),
        }
    }
}

/// Newtonian gravity with Plummer softening: the force between two bodies
/// falls off as d / (|d|^2 + softening^2)^(3/2).
#[derive(Clone, Copy, Debug)]
pub struct Gravity {
    pub g: f64,
    pub softening: f64,
}

/// Energy and momentum of a system.
#[derive(Clone, Copy, Debug, Default)]
pub struct Totals {
    pub kinetic: f64,
    pub potential: f64,
    pub momentum: [f64; 3],
}

impl Totals {
    pub fn energy(&self) -> f64 {
        self.kinetic + self.potential
    }
}

/// All-pairs gravity solved with velocity Verlet (kick, drift, kick), the
/// same scheme as the GPU solver.
pub struct NBody<T> {
    pub bodies: Vec<Body<T>>,
    acc: Vec<[T; 3]>,
    gravity: Gravity,
}

impl<T: Real> NBody<T> {
    pub fn new(bodies: Vec<Body<T>>, gravity: Gravity) -> Self {
        let mut system = Self {
            acc: vec![[T::ZERO; 3]; bodies.len()],
            bodies,
            gravity,
        };
        system.accelerations();
        system
    }

    fn accelerations(&mut self) {
        let g = T::from_f64(self.gravity.g);
        let eps2 = T::from_f64(self.gravity.softening * self.gravity.softening);
        for i in 0..self.bodies.len() {
            let mut a = [T::ZERO; 3];
            for (j, other) in self.bodies.iter().enumerate() {
                if i == j {
                    continue;
                }
                let d = [0, 1, 2].map(|k| other.pos[k] - self.bodies[i].pos[k]);
                let r2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2] + eps2;
                let inv = T::from_f64(1.0) / r2.sqrt();
                let s = other.mass * inv * inv * inv;
                for k in 0..3 {
                    a[k] += d[k] * s;
                }
            }
            self.acc[i] = a.map(|x| x * g);
        }
    }

    /// Advances `steps` steps of length `dt`.
    pub fn step(&mut self, dt: f64, steps: usize) {
        let dt = T::from_f64(dt);
        let half = dt * T::from_f64(0.5);
        for _ in 0..steps {
            for (b, a) in self.bodies.iter_mut().zip(&self.acc) {
                for k in 0..3 {
                    b.vel[k] += a[k] * half;
                    b.pos[k] += b.vel[k] * dt;
                }
            }
            self.accelerations();
            for (b, a) in self.bodies.iter_mut().zip(&self.acc) {
                for k in 0..3 {
                    b.vel[k] += a[k] * half;
                }
            }
        }
    }

    /// Kinetic and potential energy and total momentum, summed in f64.
    pub fn totals(&self) -> Totals {
        let g = self.gravity.g;
        let eps2 = self.gravity.softening * self.gravity.softening;
        let mut t = Totals::default();
        let bodies: Vec<Body<f64>> = self.bodies.iter().map(|b| b.convert()).collect();
        for (i, b) in bodies.iter().enumerate() {
            let v2: f64 = b.vel.iter().map(|v| v * v).sum();
            t.kinetic += 0.5 * b.mass * v2;
            for k in 0..3 {
                t.momentum[k] += b.mass * b.vel[k];
            }
            for other in &bodies[i + 1..] {
                let r2: f64 = (0..3).map(|k| (other.pos[k] - b.pos[k]).powi(2)).sum();
                t.potential -= g * b.mass * other.mass / (r2 + eps2).sqrt();
            }
        }
        t
    }
}

/// Two bodies on a Kepler orbit about their centre of mass, starting at
/// apocentre. Returns the bodies and the orbital period.
pub fn kepler(
    m1: f64,
    m2: f64,
    semi_major: f64,
    eccentricity: f64,
    g: f64,
) -> (Vec<Body<f64>>, f64) {
    let total = m1 + m2;
    let r = semi_major * (1.0 + eccentricity);
    let v = (g * total * (1.0 - eccentricity) / r).sqrt();
    let bodies = vec![
        Body {
            pos: [-r * m2 / total, 0.0, 0.0],
            vel: [0.0, -v * m2 / total, 0.0],
            mass: m1,
        },
        Body {
            pos: [r * m1 / total, 0.0, 0.0],
            vel: [0.0, v * m1 / total, 0.0],
            mass: m2,
        },
    ];
    let period = std::f64::consts::TAU * (semi_major.powi(3) / (g * total)).sqrt();
    (bodies, period)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kepler_energy_is_bounded_in_f64() {
        let (bodies, period) = kepler(1.0, 1e-3, 1.0, 0.5, 1.0);
        let mut system = NBody::new(
            bodies,
            Gravity {
                g: 1.0,
                softening: 0.0,
            },
        );
        let e0 = system.totals().energy();
        system.step(period / 1000.0, 10_000);
        let e1 = system.totals().energy();
        assert!(((e1 - e0) / e0).abs() < 1e-4, "{e0} -> {e1}");
    }
}

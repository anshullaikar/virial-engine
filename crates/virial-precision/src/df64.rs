//! The Rust side of the WGSL double-float maths. The same algorithms on f32
//! pairs, used to prepare GPU inputs and to test the algorithms themselves.

use std::ops::{Add, Div, Mul, Neg, Sub};

/// A number stored as the unevaluated sum `hi + lo` of two f32s.
/// Laid out like WGSL's `vec2f(hi, lo)`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Df64 {
    pub hi: f32,
    pub lo: f32,
}

fn two_sum(a: f32, b: f32) -> Df64 {
    let s = a + b;
    let bb = s - a;
    Df64 {
        hi: s,
        lo: (a - (s - bb)) + (b - bb),
    }
}

fn quick_two_sum(a: f32, b: f32) -> Df64 {
    let s = a + b;
    Df64 {
        hi: s,
        lo: b - (s - a),
    }
}

fn split(a: f32) -> (f32, f32) {
    let c = 4097.0 * a;
    let hi = c - (c - a);
    (hi, a - hi)
}

fn two_prod(a: f32, b: f32) -> Df64 {
    let p = a * b;
    let (ah, al) = split(a);
    let (bh, bl) = split(b);
    Df64 {
        hi: p,
        lo: ((ah * bh - p) + ah * bl + al * bh) + al * bl,
    }
}

impl Df64 {
    /// The nearest double-float to `x`.
    pub fn from_f64(x: f64) -> Self {
        let hi = x as f32;
        Self {
            hi,
            lo: (x - hi as f64) as f32,
        }
    }

    pub fn to_f64(self) -> f64 {
        self.hi as f64 + self.lo as f64
    }

    pub fn from_f32(x: f32) -> Self {
        Self { hi: x, lo: 0.0 }
    }

    pub fn add_f32(self, b: f32) -> Self {
        let s = two_sum(self.hi, b);
        quick_two_sum(s.hi, s.lo + self.lo)
    }

    pub fn mul_f32(self, b: f32) -> Self {
        let p = two_prod(self.hi, b);
        quick_two_sum(p.hi, p.lo + self.lo * b)
    }

    pub fn sqrt(self) -> Self {
        if self.hi <= 0.0 {
            return Self::default();
        }
        let s = self.hi.sqrt();
        let e = self - two_prod(s, s);
        Self::from_f32(s).add_f32((e.hi + e.lo) / (2.0 * s))
    }
}

impl Add for Df64 {
    type Output = Self;
    fn add(self, b: Self) -> Self {
        let s = two_sum(self.hi, b.hi);
        let t = two_sum(self.lo, b.lo);
        let u = quick_two_sum(s.hi, s.lo + t.hi);
        quick_two_sum(u.hi, u.lo + t.lo)
    }
}

impl Neg for Df64 {
    type Output = Self;
    fn neg(self) -> Self {
        Self {
            hi: -self.hi,
            lo: -self.lo,
        }
    }
}

impl Sub for Df64 {
    type Output = Self;
    fn sub(self, b: Self) -> Self {
        self + -b
    }
}

impl Mul for Df64 {
    type Output = Self;
    fn mul(self, b: Self) -> Self {
        let p = two_prod(self.hi, b.hi);
        quick_two_sum(p.hi, p.lo + (self.hi * b.lo + self.lo * b.hi))
    }
}

impl Div for Df64 {
    type Output = Self;
    fn div(self, b: Self) -> Self {
        let q1 = self.hi / b.hi;
        let r = self - b.mul_f32(q1);
        let q2 = r.hi / b.hi;
        let r = r - b.mul_f32(q2);
        let q3 = r.hi / b.hi;
        quick_two_sum(q1, q2).add_f32(q3)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rel(a: f64, b: f64) -> f64 {
        ((a - b) / b).abs()
    }

    /// Inputs spread over many magnitudes and both signs
    fn samples() -> Vec<(Df64, Df64)> {
        let mut seed = 0x9e37_79b9_u64;
        let mut next = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed >> 11) as f64 / (1u64 << 53) as f64
        };
        (0..20_000)
            .map(|_| {
                let a = (next() - 0.5) * 10f64.powf(next() * 8.0 - 4.0);
                let b = (next() - 0.5) * 10f64.powf(next() * 8.0 - 4.0);
                (Df64::from_f64(a), Df64::from_f64(b))
            })
            .collect()
    }

    #[test]
    fn arithmetic_is_accurate_to_about_14_digits() {
        let mut worst = [0.0f64; 4];
        for (a, b) in samples() {
            let (x, y) = (a.to_f64(), b.to_f64());
            // Addition is only accurate relative to the inputs when they cancel
            let add_scale = x.abs().max(y.abs());
            worst[0] = worst[0].max(((a + b).to_f64() - (x + y)).abs() / add_scale);
            worst[1] = worst[1].max(rel((a * b).to_f64(), x * y));
            worst[2] = worst[2].max(rel((a / b).to_f64(), x / y));
            worst[3] = worst[3].max(
                rel(a.sqrt().to_f64().abs().max(1e-300), x.abs().sqrt())
                    .min(rel(Df64::from_f64(x.abs()).sqrt().to_f64(), x.abs().sqrt())),
            );
        }
        for (name, err) in ["add", "mul", "div", "sqrt"].iter().zip(worst) {
            assert!(err < 1e-13, "{name}: worst relative error {err:e}");
        }
    }
}

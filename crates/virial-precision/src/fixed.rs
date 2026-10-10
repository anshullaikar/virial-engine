//! The Rust side of the WGSL fixed-point accumulators: split values the same
//! way the GPU does, and decode accumulator words into numbers.

/// One unit in fixed point: the low word holds 32 bits of fraction.
pub const ONE: f64 = 4_294_967_296.0;

/// The (low, high) words `fixed_split` produces on the GPU for `x`: the same
/// integer algorithm, bit for bit. Valid for |x| below 2^31.
pub fn split(x: f32) -> (u32, u32) {
    let bits = x.to_bits();
    let exponent = ((bits >> 23) & 0xff) as i32 - 127;
    if exponent < -34 {
        return (0, 0);
    }
    let mantissa = (bits & 0x7f_ffff) | 0x80_0000;
    let shift = exponent + 9;
    let (low, high) = if shift >= 32 {
        (0, mantissa << (shift - 32))
    } else if shift > 0 {
        (mantissa << shift, mantissa >> (32 - shift))
    } else if shift == 0 {
        (mantissa, 0)
    } else {
        let r = (-shift) as u32;
        let mut low = mantissa >> r;
        let rest = mantissa & ((1 << r) - 1);
        let half = 1 << (r - 1);
        if rest > half || (rest == half && low & 1 == 1) {
            low += 1;
        }
        (low, 0)
    };
    if bits & 0x8000_0000 == 0 {
        (low, high)
    } else {
        (
            (!low).wrapping_add(1),
            (!high).wrapping_add((low == 0) as u32),
        )
    }
}

/// The exact fixed-point value of `x`, as the GPU accumulates it.
pub fn to_raw(x: f32) -> i64 {
    let (low, high) = split(x);
    from_words(low, high)
}

/// Joins a (low, high) word pair into the signed 64-bit total.
pub fn from_words(low: u32, high: u32) -> i64 {
    (((high as u64) << 32) | low as u64) as i64
}

/// A raw fixed-point total as a float, in the units the shader added in.
pub fn to_f64(raw: i64) -> f64 {
    raw as f64 / ONE
}

/// Decodes every slot of a `virial_fixed` buffer read back from the GPU.
pub fn decode(words: &[u32]) -> Vec<f64> {
    words
        .as_chunks::<2>()
        .0
        .iter()
        .map(|&[low, high]| to_f64(from_words(low, high)))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_values_with_no_bits_below_the_resolution() {
        for x in [
            0.0f32, 1.0, -1.0, 0.5, -0.5, 1234.5678, -98765.43, 0.0625, -3.0e5,
        ] {
            assert_eq!(to_f64(to_raw(x)), x as f64, "{x}");
        }
    }

    #[test]
    fn rounds_finer_bits_to_the_nearest_step() {
        for x in [1e-3f32, -1e-3, 1e-12, -7.25e-9, 3.3e-6] {
            let error = (to_f64(to_raw(x)) - x as f64).abs();
            assert!(error <= 0.5 / ONE, "{x}: error {error:e}");
        }
    }

    #[test]
    fn matches_exact_rounding_across_magnitudes() {
        let mut seed = 12345u64;
        for _ in 0..200_000 {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            let magnitude = 2f64.powf((seed >> 11) as f64 / (1u64 << 53) as f64 * 64.0 - 34.0);
            let x = (if seed & 1 == 0 { magnitude } else { -magnitude }) as f32;
            // f64 holds x * 2^32 exactly, so this is the correctly rounded value
            let exact = (x as f64 * ONE).round_ties_even() as i64;
            assert_eq!(to_raw(x), exact, "{x:e}");
        }
    }

    #[test]
    fn sums_are_exact_in_any_order() {
        let values: Vec<f32> = (0..1000)
            .map(|i| ((i * 7919) % 1000) as f32 * 0.37 - 150.0)
            .collect();
        let forward: i64 = values.iter().map(|&x| to_raw(x)).sum();
        let backward: i64 = values.iter().rev().map(|&x| to_raw(x)).sum();
        assert_eq!(forward, backward);
    }
}

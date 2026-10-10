// 64-bit fixed-point accumulators built from 32-bit integer atomics.
//
// Integer addition is associative, so the total never depends on the order the
// GPU threads add in: every run gives the same bits. Small terms are never
// lost the way they are in a float sum.
//
// A value is split into a whole part (high word, two's complement) and a
// fraction (low word, 32 bits). Range: about ±2.1e9 (|x| below 2^31) with a resolution of
// 2.3e-10, in whatever units the caller scales to. Scale by a power of two to
// move that window without losing bits.
//
// The including shader declares the global accumulators:
//   @group(G) @binding(B) var<storage, read_write> virial_fixed: array<atomic<u32>>;
// Slot k uses elements 2k (low word) and 2k + 1 (high word).

// (low, high) words of x, rounded to the nearest 2^-32 (ties to even).
// Done entirely in integer arithmetic on the float's bits: a float version,
// (x - floor(x)) * 2^32, gets rewritten under fast math (Metal's default) and
// loses every fraction bit once |x| is more than about 1000.
fn fixed_split(x: f32) -> vec2u {
    let bits = bitcast<u32>(x);
    let exponent = i32((bits >> 23u) & 0xffu) - 127;
    // Below 2^-34 rounds to zero (this also covers zero and subnormals)
    if exponent < -34 {
        return vec2u(0u, 0u);
    }
    // |x| = mantissa * 2^(exponent - 23), so in units of 2^-32 it's
    // mantissa * 2^shift
    let mantissa = (bits & 0x7fffffu) | 0x800000u;
    let shift = exponent + 9;
    var low = 0u;
    var high = 0u;
    if shift >= 32 {
        high = mantissa << u32(shift - 32);
    } else if shift > 0 {
        low = mantissa << u32(shift);
        high = mantissa >> u32(32 - shift);
    } else if shift == 0 {
        low = mantissa;
    } else {
        // Right shift by 1..25 bits, rounding to nearest, ties to even
        let r = u32(-shift);
        low = mantissa >> r;
        let rest = mantissa & ((1u << r) - 1u);
        let half = 1u << (r - 1u);
        if rest > half || (rest == half && (low & 1u) == 1u) {
            low += 1u;
        }
    }
    if (bits & 0x80000000u) == 0u {
        return vec2u(low, high);
    }
    // Two's complement negation of the 64-bit (high, low)
    return vec2u(~low + 1u, ~high + select(0u, 1u, low == 0u));
}

fn fixed_add_parts(slot: u32, parts: vec2u) {
    let old = atomicAdd(&virial_fixed[2u * slot], parts.x);
    // The low word wrapped: carry one into the high word
    let carry = select(0u, 1u, old + parts.x < old);
    let high = parts.y + carry;
    if high != 0u {
        atomicAdd(&virial_fixed[2u * slot + 1u], high);
    }
}

fn fixed_add(slot: u32, x: f32) {
    fixed_add_parts(slot, fixed_split(x));
}

// Per-workgroup accumulators, so a workgroup makes one global atomic add per
// slot instead of one per thread. Usage, with every thread reaching each line
// (no early returns before the last barrier):
//   fixed_wg_clear(local_index); workgroupBarrier();
//   fixed_wg_add(slot, x);       workgroupBarrier();
//   fixed_wg_flush(local_index, first_global_slot);
// Needs a workgroup of at least 2 * FIXED_WG_SLOTS threads.
const FIXED_WG_SLOTS: u32 = 8u;
var<workgroup> virial_fixed_wg: array<atomic<u32>, 16>;

fn fixed_wg_clear(local_index: u32) {
    if local_index < 2u * FIXED_WG_SLOTS {
        atomicStore(&virial_fixed_wg[local_index], 0u);
    }
}

fn fixed_wg_add(slot: u32, x: f32) {
    let parts = fixed_split(x);
    let old = atomicAdd(&virial_fixed_wg[2u * slot], parts.x);
    let carry = select(0u, 1u, old + parts.x < old);
    let high = parts.y + carry;
    if high != 0u {
        atomicAdd(&virial_fixed_wg[2u * slot + 1u], high);
    }
}

fn fixed_wg_flush(local_index: u32, first_global_slot: u32) {
    if local_index < FIXED_WG_SLOTS {
        let low = atomicLoad(&virial_fixed_wg[2u * local_index]);
        let high = atomicLoad(&virial_fixed_wg[2u * local_index + 1u]);
        if (low | high) != 0u {
            fixed_add_parts(first_global_slot + local_index, vec2u(low, high));
        }
    }
}

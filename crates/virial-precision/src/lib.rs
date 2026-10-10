//! Precision tools for 32-bit GPU maths.
//!
//! WGSL has no 64-bit floats, and many GPUs (all of Apple's) have no 64-bit
//! hardware. GPU science codes such as OpenMM and GROMACS stay accurate in
//! 32 bits with three techniques; this crate holds the first and third:
//!
//! 1. **Fixed-point accumulation** ([`WGSL_FIXED`], [`fixed`]): totals summed
//!    as 64-bit integers through 32-bit atomics. Exact and identical on every
//!    run, whatever order the GPU adds in.
//! 2. **Relative coordinates**: positions stored relative to a nearby origin.
//!    That's a data-layout choice, made in the particle solver.
//! 3. **Double-float maths** ([`WGSL_DF64`], [`Df64`]): two f32s as one number
//!    with about 48 significant bits, for long integrations and totals.
//!
//! The WGSL sources are plain text to prepend to a shader. Each Rust type
//! mirrors its WGSL counterpart so values can be prepared and checked on the CPU.

pub mod df64;
pub mod fixed;

pub use df64::Df64;

/// WGSL fixed-point accumulators. The including shader must declare
/// `var<storage, read_write> virial_fixed: array<atomic<u32>>`.
pub const WGSL_FIXED: &str = include_str!("fixed.wgsl");

/// WGSL double-float arithmetic on `vec2f(hi, lo)`.
pub const WGSL_DF64: &str = include_str!("df64.wgsl");

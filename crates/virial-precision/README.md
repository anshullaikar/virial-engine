# virial-precision

Precision tools for 32-bit GPU maths, part of
[Virial](https://github.com/virial-engine/virial-engine).

- **Fixed-point accumulators** (WGSL + Rust): totals summed as 64-bit integers
  through 32-bit atomics. Exact, and identical on every run whatever order the
  GPU adds in.
- **Double-float maths** (WGSL + Rust): two f32s as one number with about 48
  significant bits. Measured on an Apple M1 against f64: worst relative error
  8e-15 (add), 1.3e-14 (multiply), 8e-15 (divide), 2.1e-14 (square root).

Metal compiles shaders in fast-math mode, which silently turns textbook
double-float code back into plain f32 accuracy. The WGSL here routes every
step through an integer XOR with a run-time zero, which the optimiser can't
reason across. Call `df_init(one)` with a 1.0 read from a buffer at the start
of each kernel.

Licensed under MIT or Apache-2.0, at your option.

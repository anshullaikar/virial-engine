# virial

GPU physics simulation in Rust and WebGPU, in the browser and natively.
Science first: energy-conserving methods, SI units, and results you can
measure, rendered to video.

This crate runs scenes: a window or browser canvas through WebGPU (Metal,
Vulkan or DX12 natively), input, a stats panel, and offline rendering on a
fixed clock (MP4 natively, WebM in the browser through WebCodecs) with
measurements as CSV, `.npy` snapshots and `run.json`. The solvers live in
[`virial-particles`](https://crates.io/crates/virial-particles), drawing in
[`virial-render`](https://crates.io/crates/virial-render). Every accuracy claim
and its test: [docs/VALIDATION.md](https://github.com/virial-engine/virial-engine/blob/main/docs/VALIDATION.md).

Licensed under MIT or Apache-2.0, at your option.

# Changelog

## 0.1.0 (2026-10-10)

First feature-complete version: GPU physics in Rust and WebGPU, validated
against known physics, rendered to video with its data.

### Solvers (`virial-particles`)
- **Gravity**: exact all-pairs forces with velocity Verlet, double-float
  positions and velocities, per-tile f32 force sums with double-float totals.
- **Lennard-Jones fluid**: periodic box, GPU cell grid sorted per cell for
  reproducibility, pressure from the virial theorem, Langevin thermostat with
  reproducible random streams, optional shifted-force cutoff.
- **Granular materials** (SI units): Hertz contact with viscoelastic damping,
  Coulomb friction, gravity, tiltable floor and walls.
- Energy, momentum, temperature, pressure and support-force measurements,
  summed in 64-bit fixed point and read back without stalling.

### Precision (`virial-precision`)
- Fixed-point accumulation through 32-bit atomics: exact, order-independent.
- Double-float arithmetic within 2.1e-14 of f64 on the GPU, made to survive
  Metal's fast-math mode with integer barriers.

### Output (`virial`)
- `record=<dir>` renders any scene offline on a simulation clock: MP4 via
  ffmpeg, a hash of every frame, measurements as exact CSV, `.npy` snapshots
  and `run.json` (settings, GPU, version, commit, uncommitted-changes flag).
- Live preview in the browser (WebGPU) and natively (Metal, Vulkan, DX12).

### Drawing (`virial-render`)
- Glowing sprites or shaded spheres, viridis and inferno colour maps, floors
  and walls with grids, orbit cameras with keyframe paths, text readouts and
  colour bars drawn into the frame.

### Examples
- `galaxies`, `gas`, `avalanche`: the three showcases.
- `attractors`, `triangle`.

### Also in 0.1.0
- Barnes–Hut gravity (GPU tree with quadrupoles) as an explicit choice for
  large N; measurements stay exact.
- Verlet neighbour lists for Lennard-Jones, rebuilt only when needed, decided
  on the GPU; Nosé–Hoover thermostat with a conserved extended energy.
- Granular grains spin, with Cundall–Strack friction springs and rolling
  resistance.
- Gravity measurements spread over frames, bit-identical to complete ones.

### Release targets (M1 MacBook Air)
- Met: energy drift below 1e-6 over 10^6 Kepler steps (2.8e-7); ideal gas law
  within 1% (0.76%); identical reruns; wasm under 3 MB (under 200 KB).
- Met with Barnes–Hut, not with exact gravity: 65,536 bodies at 10 steps/s
  (10.4; exact all pairs reaches 6.1).
- Not met: 1M short-range particles at 60 fps (one step takes about 19 ms).

See [docs/VALIDATION.md](docs/VALIDATION.md) for every test and result, and
[GOALS.md](GOALS.md#known-gaps) for known gaps.

## 0.0.1
- Names reserved on crates.io (`virial`, `virial-gpu`) and npm (`@virial/engine`).

# Changelog

## Unreleased

### Quantum motion (`virial-quantum`, new)
- The 2D time-dependent Schrödinger equation (ħ = m = 1) by Strang
  split-operator steps, with radix-2 Stockham FFTs on the GPU and phase
  factors computed in f64 and rounded to unit magnitude.
- Absorbing edges, imaginary-time ground states, a detector screen that
  integrates the probability current through a line (6th-order derivative).
- Measurements in fixed point: probability, position, spread, ⟨V⟩, ⟨T⟩.
- Tested against exact solutions: free packet, coherent state, ground state,
  tunnelling, double-slit fringes (see docs/VALIDATION.md).

### Liquid–vapour coexistence (`virial-particles`, `LennardJones`)
- Rectangular periodic boxes. **Breaking:** `Settings::box_size` is now
  `[f64; 3]` and `LennardJones::box_size()` returns the three sides.
- Profiles along z (`Settings::profile_bins`, `sample_profile`,
  `read_profile`): density, kinetic temperature, potential energy and the
  diagonal pressure tensor per slab, with the virial spread along each pair's
  line (Irving–Kirkwood) so the normal pressure is flat through interfaces.
  Summed in fixed point over samples: reruns give identical bits.
- `Profile::surface_tension` from the whole box's pressure anisotropy.
- `lattice_box`: a starting lattice for a box of any shape.
- Tested against NIST coexistence data and published surface tension (see
  docs/VALIDATION.md).

### Examples
- `double-slit`: a wave packet through two slits, drawn as density or phase,
  with the screen building up the interference pattern.

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

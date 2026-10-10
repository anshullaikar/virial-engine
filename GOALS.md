# Virial: goals and decisions

The source of truth for what Virial is and how we decide things. Read this
before planning or changing direction. Update it when a decision changes, in
the same commit as the change.

## What Virial is

A GPU physics simulator written in Rust, running on WebGPU in the browser and
on Metal, Vulkan or DX12 natively. **Science first:** results must be
accurate, reproducible and measurable. **The first product is simulations
rendered to video**, with their measurements as data. Games come later, as a
separate branch.

## Priorities, in order

When two of these conflict, the higher one wins.

1. **Correctness.** Physically right, with the error measured and shown.
2. **Reproducibility.** The same input on the same machine gives identical
   results, bit for bit.
3. **Video and data output.** Renders on a simulation clock, plus CSV of every
   measurement.
4. **Speed.** As fast as possible *without* giving up 1–3.
5. **Live preview and games.** Nice to have. Never a reason to lower 1–3.

## Rules that keep us honest

- **Frame rate only matters for the live preview.** Offline video rendering
  doesn't care how long a frame takes. Never trade precision for preview
  speed. (We added and then removed a "fast measurement" mode for exactly this
  reason.)
- **One trustworthy number per quantity.** No second, approximate reading of
  the same thing shown alongside the precise one.
- **Every accuracy claim has a test** against exact physics (Kepler orbits,
  ideal gas law, ...) or the f64 CPU reference in `virial-reference`. Report
  the measured error, never the hoped-for one.
- **Numbers in docs come from real runs**, with the hardware named. Targets
  and estimates are labelled as such.
- **Measure the error, don't hide it.** Energy drift and momentum are shown on
  screen and written to data files.
- **Test precision tricks on the GPU, not just the CPU.** Metal compiles in
  fast-math mode and has silently undone two of them (double-float maths and
  the fixed-point split). WGSL that depends on exact float behaviour goes
  through integer barriers (see `virial-precision`).

## Decided

| Topic | Decision |
|---|---|
| Name | **Virial**, after the virial theorem |
| Rust crates | `virial` and `virial-*` on crates.io (versions 0.0.x until 0.1.0) |
| npm | `@virial/engine` (the unscoped `virial` was refused as too similar to `vorpal`) |
| Repository | github.com/virial-engine/virial-engine (public) |
| Licence | MIT OR Apache-2.0 |
| Users | One developer for now, so the API is Rust-only; a TypeScript API waits until others build on it |
| Platform | WebGPU only. No WebGL or CPU fallback: the physics needs compute shaders |
| Units | SI by convention; any consistent units work |
| Precision | f32 on the GPU (WGSL has no f64), made trustworthy with fixed-point accumulation, relative coordinates and double-float maths, all checked against a 64-bit CPU reference |
| Integrators | Energy-conserving (symplectic) methods. Velocity Verlet for particles |
| Physics scope, in order | Classical mechanics (particles: gravity, molecular dynamics, granular) → thermodynamics → quantum motion |
| Games | Later, separate branch. AVBD rigid bodies live there because they trade energy for stability |
| Showcase videos | Gas reaching equilibrium, galaxy collision, granular avalanche |

## 0.1.0 milestones

| Step | What | Status |
|---|---|---|
| 1 | Workspace, wgpu on web and native | Done (Safari untested) |
| 2 | Port the particle playground | Done: on par with the original in Chrome |
| 3 | Precision core | Done: double-float within 2e-14 of f64 on GPU; fixed-point sums exact |
| 4 | Particle dynamics (gravity, Verlet, measurements) | Done: Kepler energy exact to 13 digits, drift 3.4e-7 over 100 orbits, reruns bit-identical |
| 5 | Renderer for science (sprites, colour maps, cameras, readouts) | Started: galaxies example has glowing sprites and an orbit camera |
| 6 | Video and data export on a simulation clock | Not started (the galaxies page has a real-time record button as a stopgap) |
| 7 | Bridge to thermodynamics: temperature, pressure, thermostats, granular contact | Not started |
| 8 | Validation report and release | Not started |

Release targets (estimates until measured): energy drift below 1e-6 over 10⁶
Kepler steps; ideal gas law within 1%; identical reruns; 1M short-range
particles in live preview at 60 fps; 65,536 bodies of exact gravity at 10 or
more steps per second; wasm at most 3 MB gzipped.

## Known gaps

- **All-pairs gravity speed:** 65,536 bodies run at 6.1 steps/s on an M1
  (2.6e10 interactions/s), against the 10 steps/s target. Next levers:
  Barnes–Hut or FMM for large N; making the low-part correction optional.
- **Chrome preview is much slower than native:** the galaxies scene at 8,192
  bodies runs about 6 fps in Chrome against 55 fps natively on the same M1.
  Cause not yet found.
- **Precise measurements are expensive:** about ten force passes (213 ms at
  16k bodies). Plan: snapshot the state, then measure the snapshot spread over
  several frames.
- **Safari** is untested. **CI** has no GPU, so GPU tests skip there.

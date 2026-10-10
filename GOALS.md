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
- **Measure browser speed in a foreground tab.** Chrome throttles background
  windows and tabs to a few frames per second, which once made the galaxies
  look 10× slower than they are. Bring the window to the front (and keep it
  there) before reading fps, GPU timings or steps per second, and say in the
  result that it was measured in the foreground.
- **Benchmark a cool machine.** The development machine is a fanless
  MacBook Air M1: sustained GPU load throttles it, and other GPU work (an
  open demo tab) competes. Close other GPU work, let it cool for a few
  minutes, and run the benchmark more than once.
- **Never store a number close to 1 in f32 when its distance from 1 is what
  matters.** The Nosé–Hoover velocity scale exp(-xi dt / 2) sits within 1e-4
  of 1, and f32 rounding of it drifted the conserved energy by 1e-3 per 10^4
  steps; storing s - 1 instead cut the drift 165-fold.
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
| Gravity solvers | Exact all pairs by default; Barnes–Hut as an explicit choice for large N, never silently |
| Physics scope, in order | Classical mechanics (particles: gravity, molecular dynamics, granular) → thermodynamics → quantum motion |
| Games | Later, separate branch. AVBD rigid bodies live there because they trade energy for stability |
| Showcase videos | Gas reaching equilibrium, galaxy collision, granular avalanche |
| Data output | Per-frame measurements as **CSV** with 17 significant digits; per-particle snapshots as NumPy **`.npy`** (exact binary, no dependencies, loads with `np.load`); a **`run.json`** per run with scene settings, time step, GPU, Virial version and git commit. Parquet or HDF5 only if a real need appears |

## 0.1.0 milestones

| Step | What | Status |
|---|---|---|
| 1 | Workspace, wgpu on web and native | Done (Safari untested) |
| 2 | Port the particle playground | Done: on par with the original in Chrome |
| 3 | Precision core | Done: double-float within 2e-14 of f64 on GPU; fixed-point sums exact |
| 4 | Particle dynamics (gravity, Verlet, measurements) | Done: Kepler energy exact to 13 digits, drift 3.4e-7 over 100 orbits, reruns bit-identical |
| 5 | Renderer for science (sprites, colour maps, cameras, readouts) | Done: `virial-render` draws glowing sprites or depth-correct shaded spheres, viridis and inferno colour maps (CC0), floors and walls with grids, orbit cameras with keyframed paths, and text, panels and colour bars burned into frames (public-domain font8x8). Not yet measured: the 1M-particle preview target |
| 6 | Video and data export on a simulation clock | Done natively: `record=<dir>` renders MP4 (ffmpeg) plus frames.csv, measurements.csv, .npy snapshots and run.json; reruns give identical frame hashes. 60 s 1080p60 galaxy collision rendered in 7.5 min. Browser offline export (WebCodecs) still to do |
| 7 | Bridge to thermodynamics | Done. Lennard-Jones gas: ideal gas law within 0.76%, virial expansion within 1.3 standard errors, relaxes to Maxwell-Boltzmann (KS 0.003) at constant energy. Granular contact (SI units): Hertz contact time exact to 5 digits, a settled pile carries its weight to 0.009%. Nosé–Hoover thermostat added later: canonical fluctuations, extended energy conserved to 5e-5. Grain rotation added later: a bead rolls down an incline at 5/7 g sin(theta) within 0.11%. Not done: friction memory |
| 8 | Validation report and release | Report (docs/VALIDATION.md), changelog and the three showcase videos done. Release (crates 0.1.0, tag, GitHub release with videos) waits for approval |

Release targets (estimates until measured): energy drift below 1e-6 over 10⁶
Kepler steps; ideal gas law within 1%; identical reruns; 1M short-range
particles in live preview at 60 fps; 65,536 bodies of exact gravity at 10 or
more steps per second; wasm at most 3 MB gzipped.

## Known gaps

- **1M-particle live preview at 60 fps isn't reached on an M1 Air**: with
  Verlet neighbour lists, a 1M-molecule Lennard-Jones step takes about 19 ms
  (53 steps/s, density 0.4), down from 35 ms with per-step cell search.
- **Exact all-pairs gravity tops out around 2.5e10 pairs/s on an M1 Air**:
  65,536 bodies at 6.1 steps/s. Tile size and bodies per thread were tuned;
  it's hardware-bound. For large N, `Solver::BarnesHut` (GPU tree, monopole +
  quadrupole) gives 10.4 steps/s at 65,536 with median force error 1.2e-3
  (θ 0.5). Measurements stay exact either way, so energy readouts show the
  approximation's cost.
- **Galaxy energy drift depends on the step:** worst drift to t = 90 (4096
  bodies) is 1.0e-3 at dt 0.005, 1.5e-4 at 0.0025 and 4.3e-5 at 0.00125,
  after switching force sums to per-tile f32 plus double-float totals (before
  that, drift stalled at 1.7e-4 however small the step). The scene now
  defaults to dt 0.0025. Next lever: adaptive (block) time steps.
- **Granular friction has no memory** (no tangential spring), so it resists
  sliding but can't hold a steep slope still indefinitely. Grains do spin,
  with rolling resistance. Needed for angle of repose studies.
- **Lennard-Jones energy drift is set by the step size**, not the cutoff:
  over 40 time units (4096 molecules, T about 2) it falls from 5e-5 at dt
  0.004 to 1e-5 at 0.002 and 2.6e-6 at 0.001 (shifted force). The gas
  showcase uses dt 0.002.
- **Precise gravity measurements cost about ten force passes** (213 ms at 16k
  bodies). `NBody::measure_gradually` snapshots the state and spreads the work
  over frames in chunks of 1024 bodies, with a result bit-identical to a
  complete measurement, so the preview no longer pauses.
- **Safari** is untested.
- **CI runs only on release tags** (and by hand). Day-to-day, `scripts/check.sh`
  is the gate; it runs the GPU tests that CI can't, since CI has no GPU.

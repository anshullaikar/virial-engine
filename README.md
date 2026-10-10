# Virial

GPU physics simulation in Rust and WebGPU, in the browser and natively.
Science first: energy-conserving methods, SI units, and results you can
measure, rendered to video. The scope grows from classical mechanics to
thermodynamics and, later, quantum motion. Games come later.

Named after the virial theorem, which links a system's kinetic and potential
energy. The same theorem describes galaxies settling into equilibrium and gives
a gas its pressure.

Status: 0.1.0, the first feature-complete release (see [CHANGELOG.md](CHANGELOG.md)). Virial runs scenes in a window or a browser
canvas through WebGPU (browser) or Metal / Vulkan / DX12 (native), with mouse
and keyboard input, a stats panel and GPU pass timings.

## Run the examples

Natively:

```sh
cargo run --release -p attractors -- n=4000000 work=heavy
```

In the browser (needs `rustup target add wasm32-unknown-unknown` and
`cargo install --locked wasm-bindgen-cli --version 0.2.128`):

```sh
scripts/build-web.sh attractors
python3 scripts/serve.py   # then open http://127.0.0.1:5173/examples/attractors/
```

| Example | What it shows |
|---|---|
| `galaxies` | Two disk galaxies colliding under exact gravity, with energy readouts |
| `gas` | A gas reaching equilibrium: speeds relax onto the Maxwell–Boltzmann curve |
| `avalanche` | 23,000 glass beads pouring down a slope, in SI units |
| `attractors` | Up to 16M particles pulled towards the mouse; a speed baseline |
| `triangle` | One binary drawing in the browser and natively |

Browser settings go in the URL (`?n=4000000&work=heavy`), native settings as
arguments (`n=4000000 work=heavy`).

## Speed baseline

`attractors` is the WebGPU demo from the earlier particle playground, ported to
Virial with the same kernel. Chrome on an Apple M1, Heavy mode (16 attractors):

| Particles | Original (JavaScript + WebGPU) | Virial (Rust + wgpu) |
|---|---|---|
| 4M | 32 fps | 36 fps |
| 16M | 19 fps | 17 fps |

The difference is within run-to-run noise. GPU timings show drawing 1M
overlapping points costs about 14 ms a frame, more than simulating 4M.

WebGPU only: Chrome, Edge, or Safari 26+. Browsers without WebGPU get a
message instead of a blank page.

Every accuracy claim, its test and the measured result: [docs/VALIDATION.md](docs/VALIDATION.md).

## Precision

WGSL has no 64-bit floats and Apple GPUs have no 64-bit hardware, so Virial
uses the techniques GPU molecular-dynamics codes rely on. Measured on an M1:

| Technique | Result |
|---|---|
| Fixed-point accumulation (64-bit integers via atomics) | A sum of 2^20 values from 1e-6 to 1e3 is exact to the bit and identical across runs; error vs the true sum 6e-8, against 0.28 for a sequential f32 sum |
| Double-float maths (two f32s per number) | Worst relative error vs f64: 8e-15 add, 1.3e-14 multiply, 8e-15 divide, 2.1e-14 square root |

Metal compiles shaders in fast-math mode, which quietly reduces textbook
double-float code to plain f32 accuracy (1e-7). `virial-precision` routes each
step through an integer barrier the optimiser can't see through.

## Particle dynamics

`virial-particles` integrates point masses under Newtonian gravity on the GPU:
velocity Verlet, positions and velocities in double-float precision, exact
all-pairs forces, and energy and momentum measured in double-float and summed
in fixed point. Validation on an M1 against known physics and the f64 CPU
reference in `virial-reference`:

| Check | Result |
|---|---|
| Kepler orbit (e = 0.5), energy at start vs exact | −5.000000000000e-4, all 13 digits |
| Energy drift over 100 orbits (10⁵ steps) | 3.4e-7 (plain f32: 2.9e-5; CPU f64: 3.2e-10) |
| Orbit after 100 orbits vs CPU f64 | separation agrees to 7.5e-7 |
| 512-body cluster energy vs CPU f64 | identical to 13 digits |
| Rerun of a 4096-body simulation | bit-identical state and measurements |

## Thermodynamics

`virial_particles::lennard_jones` simulates a Lennard-Jones fluid in a
periodic box: neighbour search on the GPU with a cell grid (sorted per cell so
reruns are bit-identical), temperature from kinetic energy, pressure from the
virial theorem, and an optional Langevin thermostat with reproducible random
numbers. Validation on an M1:

| Check | Result |
|---|---|
| Cell grid vs all pairs (1000 particles) | Potential energy within 4e-7, virial within 2.6e-7 |
| Dilute gas (density 0.01, T = 2) | PV/NkT = 0.99244 ± 0.00012: within 0.76% of the ideal gas law, and within 1.3 standard errors of the virial expansion 1 + B₂ρ = 0.99228 |
| Gas starting with every particle at the same speed, no thermostat | Kolmogorov–Smirnov distance from Maxwell–Boltzmann 0.61 → 0.003 after 60 time units, energy drift 4e-5 |
| Thermostatted reruns | Bit-identical |

## Granular materials

`virial_particles::granular` simulates grains in SI units: Hertz contact with
viscoelastic damping, Coulomb friction, gravity, a tiltable floor and walls.
Validation on an M1:

| Check | Result |
|---|---|
| Elastic head-on collision (0.5 m/s, 2.5 mm glass beads) | Contact time 8.7732e-4 s, Hertz theory 8.7732e-4 s; restitution 1.000000; momentum exactly conserved |
| Pile of 2304 grains after settling | Floor and walls carry 3.732933 N against a weight of 3.732605 N (+0.009%) |
| Reruns | Bit-identical |

Run all GPU tests with `cargo test --release` (they skip on machines without
a GPU), or everything with `scripts/check.sh`.

## Videos and data

Any example renders offline natively with `record=<dir>`:

```sh
cargo run --release -p galaxies -- record=renders/galaxies seconds=60 n=16384 spin=0.04
```

Each frame covers a fixed slice of simulated time however long it takes. The
directory gets `video.mp4` (needs ffmpeg), `frames.csv` (a hash of every frame:
reruns match exactly), `measurements.csv` (exact values), `.npy` snapshots of
the full state, and `run.json` (settings, GPU, version, git commit). Options:
`width`, `height`, `fps`, `seconds`, `video=0` to skip encoding.

## Layout

| Path | What it is |
|---|---|
| `crates/virial` | Public API: `Scene` trait, `run`, input, stats panel, settings |
| `crates/virial-gpu` | wgpu device (windowed or headless), limit checks, GPU pass timer |
| `crates/virial-precision` | Fixed-point accumulators and double-float maths for WGSL, tested on the GPU |
| `crates/virial-particles` | GPU particle dynamics: gravity, velocity Verlet, energy and momentum measurements |
| `crates/virial-reference` | CPU reference solvers in f64 (and f32 for comparison), used by tests |
| `crates/virial-render` | Particles (sprites or spheres), colour maps, floors and walls, cameras and paths, text readouts |
| `examples/` | Example scenes, each with an `index.html` for the browser |
| `npm/` | The `@virial/engine` npm package (name reserved; browser build to come) |
| `scripts/build-web.sh` | Builds an example for the browser and reports its gzipped size |
| `scripts/serve.py` | Local server that disables caching, so rebuilt wasm always loads |

## Licence

Dual-licensed under [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE), at your option.

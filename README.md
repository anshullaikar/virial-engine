# Virial

GPU physics simulation in Rust and WebGPU, in the browser and natively.
Science first: energy-conserving methods, SI units, and results you can
measure, rendered to video. The scope grows from classical mechanics to
thermodynamics and, later, quantum motion. Games come later.

Named after the virial theorem, which links a system's kinetic and potential
energy. The same theorem describes galaxies settling into equilibrium and gives
a gas its pressure.

Status: step 2 of 8 towards 0.1.0. Virial runs scenes in a window or a browser
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
| `triangle` | Step 1: one binary drawing in the browser and natively |
| `attractors` | Step 2: up to 16M particles pulled towards the mouse; a speed baseline |

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

## Layout

| Path | What it is |
|---|---|
| `crates/virial` | Public API: `Scene` trait, `run`, input, stats panel, settings |
| `crates/virial-gpu` | wgpu device, surface, limit checks, GPU pass timer |
| `examples/` | Example scenes, each with an `index.html` for the browser |
| `npm/` | The `virial` npm package (name reserved; browser build to come) |
| `scripts/build-web.sh` | Builds an example for the browser and reports its gzipped size |
| `scripts/serve.py` | Local server that disables caching, so rebuilt wasm always loads |

## Licence

Dual-licensed under [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE), at your option.

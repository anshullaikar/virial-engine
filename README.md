# Virial

GPU physics simulation in Rust and WebGPU, in the browser and natively.
Science first: energy-conserving methods, SI units, and results you can
measure, rendered to video. The scope grows from classical mechanics to
thermodynamics and, later, quantum motion. Games come later.

Named after the virial theorem, which links a system's kinetic and potential
energy. The same theorem describes galaxies settling into equilibrium and gives
a gas its pressure.

Status: milestone 1 of 8 towards 0.1.0.
Virial opens a window or browser canvas and draws through WebGPU (browser)
or Metal / Vulkan / DX12 (native).

## Run the triangle example

Natively:

```sh
cargo run -p triangle
```

In the browser (needs `rustup target add wasm32-unknown-unknown` and
`cargo install --locked wasm-bindgen-cli --version 0.2.128`):

```sh
scripts/build-web.sh triangle
python3 scripts/serve.py   # then open http://127.0.0.1:5173/examples/triangle/
```

WebGPU only: Chrome, Edge, or Safari 26+. Browsers without WebGPU get a
message instead of a blank page.

## Layout

| Path | What it is |
|---|---|
| `crates/virial` | Public API: `Scene` trait, `run`, window and frame loop |
| `crates/virial-gpu` | wgpu device, surface and WebGPU limit checks |
| `examples/triangle` | Milestone 1: one binary, browser and native |
| `scripts/build-web.sh` | Builds an example for the browser and reports its gzipped size |
| `scripts/serve.py` | Local server that disables caching, so rebuilt wasm always loads |

## Licence

Dual-licensed under [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE), at your option.

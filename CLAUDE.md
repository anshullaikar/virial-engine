# Working on Virial

@GOALS.md

## How to work in this repo

- Check a change against GOALS.md before starting it. If it conflicts with
  the priorities or a decision there, say so instead of quietly proceeding.
- When a decision changes, update GOALS.md in the same commit.
- Verify claims by running them: `cargo test --release` (GPU tests run on
  this machine's GPU), the examples natively, and the web build in a browser.
  Put measured numbers in docs, with the hardware.
- Before committing: `cargo fmt --all`, then `cargo clippy --workspace
  --all-targets -- -D warnings` for native and again with
  `--target wasm32-unknown-unknown`.
- Browser speed only counts from a foreground tab in a front window: Chrome
  throttles anything else. Before measuring, check `document.visibilityState`
  is `visible` and `requestAnimationFrame` runs near the display rate (about
  60 per second); if not, ask for the window to be brought to the front.
- Web builds: `scripts/build-web.sh <example>`, served by
  `python3 scripts/serve.py` at http://127.0.0.1:5173/examples/<example>/.

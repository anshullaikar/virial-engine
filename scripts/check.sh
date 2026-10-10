#!/bin/sh
# Everything CI checks, run locally. Unlike CI, this machine has a GPU, so the
# GPU tests run for real. Run before pushing: scripts/check.sh
set -eu
cd "$(dirname "$0")/.."

echo "== format"
cargo fmt --all --check
echo "== clippy (native)"
cargo clippy --quiet --workspace --all-targets -- -D warnings
echo "== clippy (wasm)"
cargo clippy --quiet --workspace --target wasm32-unknown-unknown -- -D warnings
echo "== tests (release, GPU tests included)"
cargo test --quiet --release --workspace
echo "== web builds"
for dir in examples/*/; do
  [ -f "$dir/Cargo.toml" ] || continue
  scripts/build-web.sh "$(basename "$dir")" | tail -1
done
echo "== all checks passed"

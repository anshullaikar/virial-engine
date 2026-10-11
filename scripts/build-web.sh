#!/bin/sh
# Builds an example for the browser: scripts/build-web.sh triangle
# Output goes to examples/<name>/pkg. Needs wasm-bindgen-cli 0.2.128:
#   cargo install --locked wasm-bindgen-cli --version 0.2.128
set -eu
name="${1:?usage: scripts/build-web.sh <example>}"
cd "$(dirname "$0")/.."

# The commit, for run.json when the page records video
VIRIAL_GIT_COMMIT=$(git rev-parse HEAD 2>/dev/null || echo unknown)
VIRIAL_GIT_DIRTY=$(test -z "$(git status --porcelain 2>/dev/null)" && echo 0 || echo 1)
export VIRIAL_GIT_COMMIT VIRIAL_GIT_DIRTY
cargo build --release --target wasm32-unknown-unknown -p "$name"
wasm-bindgen --target web --no-typescript \
  --out-dir "examples/$name/pkg" \
  "target/wasm32-unknown-unknown/release/$name.wasm"

if command -v wasm-opt >/dev/null 2>&1; then
  wasm-opt -O3 --enable-simd --enable-bulk-memory --enable-nontrapping-float-to-int \
    "examples/$name/pkg/${name}_bg.wasm" -o "examples/$name/pkg/${name}_bg.wasm"
fi

# The 0.1.0 budget is 3 MB gzipped
bytes=$(gzip -9 -c "examples/$name/pkg/${name}_bg.wasm" | wc -c | tr -d ' ')
echo "$name: $((bytes / 1024)) KB gzipped (budget 3072 KB)"

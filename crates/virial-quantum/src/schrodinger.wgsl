// The time-dependent Schrodinger equation on an N x N periodic grid
// (hbar = m = 1), by Strang splitting:
//   psi <- V/2 . F^-1 . K . F . V/2 psi
// with V/2 = exp(-i V dt / 2) (times exp(-W dt / 2) where an absorbing
// boundary W is on) and K = exp(-i k^2 dt / 2) / N^2, both precomputed in
// f64 on the CPU. In imaginary time the factors are real decays instead and
// the wavefunction is renormalised after each step.
//
// The FFT is radix-2 Stockham (out of place, natural order out), one
// dispatch per pass, along x then y. Twiddles come from a table computed in
// f64, not from GPU sin/cos. `psi` holds three N^2 regions: the wavefunction
// (A), FFT scratch (B) and a copy for measurements (C). A 2D transform has
// 2 log2 N passes, an even number, so it always ends where it started.
//
// Prepended at build time: WGSL_FIXED.

struct Pass {
    // log2 of the half-length of the sub-transforms merged in this pass
    stage: u32,
    // 0 along x (contiguous), 1 along y
    axis: u32,
    // 0 forward, 1 inverse (conjugated twiddles)
    inverse: u32,
    // Element offsets of the source and destination regions
    src: u32,
    dst: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

struct Grid {
    n: u32,
    log2n: u32,
    cells: u32,
    // Measurement region offset (region C)
    measure_base: u32,
    dx: f32,
    // Fixed-point scales for the measured sums
    scale_norm: f32,
    scale_x: f32,
    scale_x2: f32,
    scale_v: f32,
    scale_t: f32,
    dt: f32,
    // Grid column of the detector screen
    detector_column: u32,
}

@group(0) @binding(0) var<uniform> pass_info: Pass;
@group(0) @binding(1) var<uniform> grid: Grid;
@group(0) @binding(2) var<storage, read_write> psi: array<vec2f>;
// exp(-2 pi i m / N) for m in 0 .. N/2
@group(0) @binding(3) var<storage, read> twiddles: array<vec2f>;
// Per-cell factors: half-step potential (x,y), kinetic (z,w)
@group(0) @binding(4) var<storage, read> factors: array<vec4f>;
// Per-cell potential V (x), absorbing W (y), k^2 / 2 (z)
@group(0) @binding(5) var<storage, read> fields: array<vec4f>;
@group(0) @binding(6) var<storage, read_write> virial_fixed: array<atomic<u32>>;
// Per row: time-integrated probability current through the detector column
@group(0) @binding(7) var<storage, read_write> detector: array<f32>;

fn cmul(a: vec2f, b: vec2f) -> vec2f {
    return vec2f(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

@compute @workgroup_size(256)
fn fft_pass(@builtin(global_invocation_id) id: vec3u) {
    let n = grid.n;
    let halfn = n / 2u;
    let t = id.x;
    if t >= halfn * n {
        return;
    }
    let row = t / halfn;
    let i = t % halfn;
    let p = 1u << pass_info.stage;
    let k = i & (p - 1u);
    // Element addresses along the chosen axis
    var base = row * n;
    var stride = 1u;
    if pass_info.axis == 1u {
        base = row;
        stride = n;
    }
    let u0 = psi[pass_info.src + base + i * stride];
    var w = twiddles[k * (halfn / p)];
    if pass_info.inverse == 1u {
        w.y = -w.y;
    }
    let u1 = cmul(psi[pass_info.src + base + (i + halfn) * stride], w);
    let j = 2u * i - k;
    psi[pass_info.dst + base + j * stride] = u0 + u1;
    psi[pass_info.dst + base + (j + p) * stride] = u0 - u1;
}

// psi *= half-step potential factor (region A)
@compute @workgroup_size(256)
fn potential_half(@builtin(global_invocation_id) id: vec3u) {
    if id.x < grid.cells {
        psi[id.x] = cmul(psi[id.x], factors[id.x].xy);
    }
}

// psi~ *= kinetic factor, including the 1 / N^2 of the inverse transform
@compute @workgroup_size(256)
fn kinetic(@builtin(global_invocation_id) id: vec3u) {
    if id.x < grid.cells {
        psi[id.x] = cmul(psi[id.x], factors[id.x].zw);
    }
}

// Copies the wavefunction into region C for measurements
@compute @workgroup_size(256)
fn copy_to_measure(@builtin(global_invocation_id) id: vec3u) {
    if id.x < grid.cells {
        psi[grid.measure_base + id.x] = psi[id.x];
    }
}

// Slots: 0 norm, 1 <x>, 2 <y>, 3 <x^2>, 4 <y^2>, 5 <V>, 6 absorbed-region
// weight (W > 0), all as sums of |psi|^2 dA times the quantity
@compute @workgroup_size(256)
fn measure_position(@builtin(global_invocation_id) id: vec3u, @builtin(local_invocation_index) li: u32) {
    fixed_wg_clear(li);
    workgroupBarrier();
    if id.x < grid.cells {
        let c = psi[id.x];
        let w = dot(c, c) * grid.dx * grid.dx;
        let n = grid.n;
        let centre = f32(n / 2u);
        let x = (f32(id.x % n) - centre) * grid.dx;
        let y = (f32(id.x / n) - centre) * grid.dx;
        let f = fields[id.x];
        fixed_wg_add(0u, w * grid.scale_norm);
        fixed_wg_add(1u, w * x * grid.scale_x);
        fixed_wg_add(2u, w * y * grid.scale_x);
        fixed_wg_add(3u, w * x * x * grid.scale_x2);
        fixed_wg_add(4u, w * y * y * grid.scale_x2);
        fixed_wg_add(5u, w * f.x * grid.scale_v);
        fixed_wg_add(6u, select(0.0, w, f.y > 0.0) * grid.scale_norm);
    }
    workgroupBarrier();
    fixed_wg_flush(li, 0u);
}

// After a forward FFT of region C: slot 7 gets sum (k^2 / 2) |psi~|^2 dA / N^2,
// which with Parseval's theorem gives the kinetic energy
@compute @workgroup_size(256)
fn measure_kinetic(@builtin(global_invocation_id) id: vec3u, @builtin(local_invocation_index) li: u32) {
    fixed_wg_clear(li);
    workgroupBarrier();
    if id.x < grid.cells {
        let c = psi[grid.measure_base + id.x];
        let n2 = f32(grid.cells);
        let w = dot(c, c) / n2 * grid.dx * grid.dx;
        fixed_wg_add(7u, w * fields[id.x].z * grid.scale_t);
    }
    workgroupBarrier();
    fixed_wg_flush(li, 0u);
}

// Imaginary time: sum |psi|^2 dA into slot 8, then one thread turns it into
// a rescale factor, then every cell is rescaled to norm 1
@compute @workgroup_size(256)
fn norm_sum(@builtin(global_invocation_id) id: vec3u, @builtin(local_invocation_index) li: u32) {
    fixed_wg_clear(li);
    workgroupBarrier();
    if id.x < grid.cells {
        let c = psi[id.x];
        fixed_wg_add(0u, dot(c, c) * grid.dx * grid.dx * grid.scale_norm);
    }
    workgroupBarrier();
    fixed_wg_flush(li, 8u);
}

@compute @workgroup_size(1)
fn norm_factor() {
    let low = atomicLoad(&virial_fixed[16]);
    let high = atomicLoad(&virial_fixed[17]);
    atomicStore(&virial_fixed[16], 0u);
    atomicStore(&virial_fixed[17], 0u);
    let norm = (f32(bitcast<i32>(high)) + f32(low >> 8u) / 16777216.0) / grid.scale_norm;
    atomicStore(&virial_fixed[18], bitcast<u32>(inverseSqrt(norm)));
}

@compute @workgroup_size(256)
fn renormalise(@builtin(global_invocation_id) id: vec3u) {
    if id.x < grid.cells {
        psi[id.x] *= bitcast<f32>(atomicLoad(&virial_fixed[18]));
    }
}

// One thread per row: adds j_x dt at the detector column, with
// j_x = Im(conj(psi) dpsi/dx). The derivative is a 6th-order central
// difference: relative error (k dx)^6 / 140, against (k dx)^2 / 6 for the
// plain 2-point one.
@compute @workgroup_size(256)
fn detect(@builtin(global_invocation_id) id: vec3u) {
    if id.x >= grid.n {
        return;
    }
    let row = id.x * grid.n + grid.detector_column;
    let c = psi[row];
    let d = (45.0 * (psi[row + 1u] - psi[row - 1u]) - 9.0 * (psi[row + 2u] - psi[row - 2u])
        + (psi[row + 3u] - psi[row - 3u])) / (60.0 * grid.dx);
    detector[id.x] += (c.x * d.y - c.y * d.x) * grid.dt;
}

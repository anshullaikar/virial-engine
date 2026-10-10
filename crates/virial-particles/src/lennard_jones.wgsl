// Lennard-Jones particles in a periodic cubic box, in reduced units
// (sigma = epsilon = mass = k_B = 1). Short-range forces through a cell grid:
// each step bins particles into cells of side >= cutoff, prefix-sums the cell
// counts, scatters particle indices into cell order, then sorts each cell by
// particle index so every pair sum runs in the same order on every run.
//
// Positions and velocities are double-floats (hi, lo) as in the gravity
// solver; the potential is cut at r_c and shifted so it's continuous there.
//
// To stay within WebGPU's default of 8 storage buffers per shader, related
// arrays share a buffer in sections of `segment` elements:
//   state: position hi, position lo, velocity hi, velocity lo, acceleration
//   ints:  cell of each particle, particles in cell order, random state
//   cells: cell starts (total_cells + 1), then the scan's block sums
//
// Prepended at build time: WGSL_FIXED and WGSL_DF64 come after this file.

struct Params {
    count: u32,
    cells: u32,
    total_cells: u32,
    // 0: constant energy (velocity Verlet), 1: Langevin thermostat (BAOAB)
    thermostat: u32,
    box_size: f32,
    cell_size: f32,
    rc2: f32,
    // U(r_c), subtracted so the potential is continuous at the cutoff
    shift: f32,
    dt: f32,
    half_dt: f32,
    one: f32,
    energy_scale: f32,
    momentum_scale: f32,
    // Langevin: v <- c1 v + c2 * gaussian
    c1: f32,
    c2: f32,
    // Elements per section in `state` and `ints`
    segment: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> state: array<vec4f>;
@group(0) @binding(2) var<storage, read_write> virial_fixed: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> cell_count: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read_write> cells: array<u32>;
@group(0) @binding(5) var<storage, read_write> ints: array<u32>;

// ---- Random numbers: per-particle PCG state, so runs are reproducible ----

fn pcg(state: ptr<function, u32>) -> u32 {
    let s = *state * 747796405u + 2891336453u;
    *state = s;
    let word = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
    return (word >> 22u) ^ word;
}

fn uniform01(state: ptr<function, u32>) -> f32 {
    // 24 random bits in (0, 1]
    return (f32(pcg(state) >> 8u) + 1.0) / 16777216.0;
}

fn gaussian2(state: ptr<function, u32>) -> vec2f {
    let r = sqrt(-2.0 * log(uniform01(state)));
    let a = 6.283185307179586 * uniform01(state);
    return r * vec2f(cos(a), sin(a));
}

// ---- Step part 1: half kick and drift (with the Langevin step if on) ----

fn wrap(x: vec2f) -> vec2f {
    var y = x;
    if y.x >= params.box_size {
        y = df_add_f(y, -params.box_size);
    } else if y.x < 0.0 {
        y = df_add_f(y, params.box_size);
    }
    return y;
}

@compute @workgroup_size(256)
fn begin(@builtin(global_invocation_id) id: vec3u) {
    df_init(params.one);
    let i = id.x;
    if i >= params.count {
        return;
    }
    let a = state[4u * params.segment + i];
    var vh = state[2u * params.segment + i];
    var vl = state[3u * params.segment + i];
    var ph = state[i];
    var pl = state[params.segment + i];
    var noise = vec3f(0.0);
    if params.thermostat == 1u {
        var state = ints[2u * params.segment + i];
        let g = gaussian2(&state);
        let h = gaussian2(&state);
        noise = vec3f(g, h.x);
        ints[2u * params.segment + i] = state;
    }
    for (var k = 0u; k < 3u; k++) {
        // B: half kick
        var v = df_add_f(vec2f(vh[k], vl[k]), a[k] * params.half_dt);
        var x = vec2f(ph[k], pl[k]);
        if params.thermostat == 1u {
            // A, O, A: half drift, friction and noise, half drift
            x = df_add(x, df_mul_f(v, params.half_dt));
            v = df_add_f(df_mul_f(v, params.c1), params.c2 * noise[k]);
            x = df_add(x, df_mul_f(v, params.half_dt));
        } else {
            x = df_add(x, df_mul_f(v, params.dt));
        }
        x = wrap(x);
        vh[k] = v.x;
        vl[k] = v.y;
        ph[k] = x.x;
        pl[k] = x.y;
    }
    state[2u * params.segment + i] = vh;
    state[3u * params.segment + i] = vl;
    state[i] = ph;
    state[params.segment + i] = pl;
}

// ---- Cell grid ----

fn cell_coords(x: vec3f) -> vec3u {
    let c = vec3i(floor(x / params.cell_size));
    let m = i32(params.cells) - 1;
    return vec3u(clamp(c, vec3i(0), vec3i(m)));
}

fn cell_index(c: vec3u) -> u32 {
    return (c.z * params.cells + c.y) * params.cells + c.x;
}

@compute @workgroup_size(256)
fn clear_cells(@builtin(global_invocation_id) id: vec3u) {
    if id.x < params.total_cells {
        atomicStore(&cell_count[id.x], 0u);
    }
}

@compute @workgroup_size(256)
fn bin(@builtin(global_invocation_id) id: vec3u) {
    let i = id.x;
    if i >= params.count {
        return;
    }
    let c = cell_index(cell_coords(state[i].xyz));
    ints[i] = c;
    atomicAdd(&cell_count[c], 1u);
}

var<workgroup> scan_tmp: array<u32, 256>;

// Exclusive prefix sum of cell counts within each block of 256 cells
@compute @workgroup_size(256)
fn scan_blocks(@builtin(global_invocation_id) id: vec3u, @builtin(local_invocation_index) li: u32, @builtin(workgroup_id) group: vec3u) {
    let c = id.x;
    var v = 0u;
    if c < params.total_cells {
        v = atomicLoad(&cell_count[c]);
    }
    scan_tmp[li] = v;
    workgroupBarrier();
    for (var offset = 1u; offset < 256u; offset *= 2u) {
        var add = 0u;
        if li >= offset {
            add = scan_tmp[li - offset];
        }
        workgroupBarrier();
        scan_tmp[li] += add;
        workgroupBarrier();
    }
    let inclusive = scan_tmp[li];
    if c < params.total_cells {
        cells[c] = inclusive - v;
    }
    if li == 255u {
        cells[params.total_cells + 1u + group.x] = inclusive;
    }
}

// Exclusive prefix sum of the block totals: one workgroup, four per thread
// (up to 1024 blocks, so up to 262144 cells)
@compute @workgroup_size(256)
fn scan_sums(@builtin(local_invocation_index) li: u32) {
    let blocks = (params.total_cells + 255u) / 256u;
    var prefix: array<u32, 4>;
    var total = 0u;
    for (var k = 0u; k < 4u; k++) {
        let b = li * 4u + k;
        var v = 0u;
        if b < blocks {
            v = cells[params.total_cells + 1u + b];
        }
        prefix[k] = total;
        total += v;
    }
    scan_tmp[li] = total;
    workgroupBarrier();
    for (var offset = 1u; offset < 256u; offset *= 2u) {
        var add = 0u;
        if li >= offset {
            add = scan_tmp[li - offset];
        }
        workgroupBarrier();
        scan_tmp[li] += add;
        workgroupBarrier();
    }
    let before = scan_tmp[li] - total;
    for (var k = 0u; k < 4u; k++) {
        let b = li * 4u + k;
        if b < blocks {
            cells[params.total_cells + 1u + b] = before + prefix[k];
        }
    }
}

// Global cell starts; counts become fill cursors for the scatter
@compute @workgroup_size(256)
fn scan_add(@builtin(global_invocation_id) id: vec3u, @builtin(workgroup_id) group: vec3u) {
    let c = id.x;
    if c < params.total_cells {
        cells[c] += cells[params.total_cells + 1u + group.x];
        atomicStore(&cell_count[c], 0u);
    }
    if c == 0u {
        cells[params.total_cells] = params.count;
    }
}

@compute @workgroup_size(256)
fn scatter(@builtin(global_invocation_id) id: vec3u) {
    let i = id.x;
    if i >= params.count {
        return;
    }
    let c = ints[i];
    ints[params.segment + cells[c] + atomicAdd(&cell_count[c], 1u)] = i;
}

// Insertion sort of each cell by particle index: the scatter's order depends
// on thread timing, this makes it the same every run
@compute @workgroup_size(256)
fn sort_cells(@builtin(global_invocation_id) id: vec3u) {
    let c = id.x;
    if c >= params.total_cells {
        return;
    }
    let start = cells[c];
    let end = cells[c + 1u];
    for (var s = start + 1u; s < end; s++) {
        let key = ints[params.segment + s];
        var t = s;
        while t > start && ints[params.segment + t - 1u] > key {
            ints[params.segment + t] = ints[params.segment + t - 1u];
            t -= 1u;
        }
        ints[params.segment + t] = key;
    }
}

// ---- Forces ----

struct Pair {
    // Acceleration (mass 1), this particle's half of the potential energy
    // and of the virial sum r . F over its pairs
    a: vec3f,
    pe: vec2f,
    w: vec2f,
}

fn interactions(i: u32) -> Pair {
    let xh = state[i].xyz;
    let xl = state[params.segment + i].xyz;
    let c = ints[i];
    let n = params.cells;
    let cc = vec3i(i32(c % n), i32((c / n) % n), i32(c / (n * n)));
    var out = Pair(vec3f(0.0), vec2f(0.0), vec2f(0.0));
    for (var dz = -1; dz <= 1; dz++) {
        for (var dy = -1; dy <= 1; dy++) {
            for (var dx = -1; dx <= 1; dx++) {
                let nc = vec3u((cc + vec3i(dx, dy, dz) + vec3i(i32(n))) % vec3i(i32(n)));
                let cell = cell_index(nc);
                for (var s = cells[cell]; s < cells[cell + 1u]; s++) {
                    let j = ints[params.segment + s];
                    if j == i {
                        continue;
                    }
                    var d = (state[j].xyz - xh) + (state[params.segment + j].xyz - xl);
                    // Nearest periodic image
                    d -= params.box_size * round(d / params.box_size);
                    let r2 = dot(d, d);
                    if r2 < params.rc2 {
                        let inv2 = 1.0 / r2;
                        let s6 = inv2 * inv2 * inv2;
                        let s12 = s6 * s6;
                        // r . F for this pair, and the force on i
                        let rf = 24.0 * (2.0 * s12 - s6);
                        out.a -= d * (rf * inv2);
                        out.pe = df_add_f(out.pe, 2.0 * (s12 - s6) - 0.5 * params.shift);
                        out.w = df_add_f(out.w, 0.5 * rf);
                    }
                }
            }
        }
    }
    return out;
}

@compute @workgroup_size(256)
fn prime(@builtin(global_invocation_id) id: vec3u) {
    df_init(params.one);
    if id.x < params.count {
        state[4u * params.segment + id.x] = vec4f(interactions(id.x).a, 0.0);
    }
}

// New forces, then the second half kick
@compute @workgroup_size(256)
fn force_kick(@builtin(global_invocation_id) id: vec3u) {
    df_init(params.one);
    let i = id.x;
    if i >= params.count {
        return;
    }
    let a = interactions(i).a;
    state[4u * params.segment + i] = vec4f(a, 0.0);
    var vh = state[2u * params.segment + i];
    var vl = state[3u * params.segment + i];
    for (var k = 0u; k < 3u; k++) {
        let v = df_add_f(vec2f(vh[k], vl[k]), a[k] * params.half_dt);
        vh[k] = v.x;
        vl[k] = v.y;
    }
    state[2u * params.segment + i] = vh;
    state[3u * params.segment + i] = vl;
}

// Slots: 0 kinetic, 1 potential, 2 virial sum, 3-5 momentum
@compute @workgroup_size(256)
fn measure(@builtin(global_invocation_id) id: vec3u, @builtin(local_invocation_index) li: u32) {
    df_init(params.one);
    fixed_wg_clear(li);
    workgroupBarrier();
    let i = id.x;
    var ke = vec2f(0.0);
    var pe = vec2f(0.0);
    var w = vec2f(0.0);
    var momentum = vec3f(0.0);
    if i < params.count {
        let p = interactions(i);
        pe = p.pe;
        w = p.w;
        let vh = state[2u * params.segment + i];
        let vl = state[3u * params.segment + i];
        for (var k = 0u; k < 3u; k++) {
            let v = vec2f(vh[k], vl[k]);
            ke = df_add(ke, df_mul(v, v));
            momentum[k] = vh[k] + vl[k];
        }
        ke = df_mul_f(ke, 0.5);
    }
    let es = params.energy_scale;
    fixed_wg_add(0u, ke.x * es);
    fixed_wg_add(0u, ke.y * es);
    fixed_wg_add(1u, pe.x * es);
    fixed_wg_add(1u, pe.y * es);
    fixed_wg_add(2u, w.x * es);
    fixed_wg_add(2u, w.y * es);
    for (var k = 0u; k < 3u; k++) {
        fixed_wg_add(3u + k, momentum[k] * params.momentum_scale);
    }
    workgroupBarrier();
    fixed_wg_flush(li, 0u);
}

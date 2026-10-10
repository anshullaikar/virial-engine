// All-pairs gravity with velocity Verlet. Positions and velocities are
// double-floats split across two buffers (hi, lo); forces are computed from
// differences (x_j - x_i) taken per part, so each body effectively sits at the
// origin of its own coordinates and keeps full precision anywhere in space.
//
// Prepended at build time: the bindings below, WGSL_FIXED and WGSL_DF64.

struct Params {
    count: u32,
    g: f32,
    eps2: f32,
    dt: f32,
    half_dt: f32,
    // 1.0, read at run time for df_init
    one: f32,
    energy_scale: f32,
    momentum_scale: f32,
    // Low part of softening^2, for measurements
    eps2_lo: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
// Position in xyz and mass in w: hi parts, then lo parts
@group(0) @binding(1) var<storage, read_write> pos_hi: array<vec4f>;
@group(0) @binding(2) var<storage, read_write> pos_lo: array<vec4f>;
@group(0) @binding(3) var<storage, read_write> vel_hi: array<vec4f>;
@group(0) @binding(4) var<storage, read_write> vel_lo: array<vec4f>;
@group(0) @binding(5) var<storage, read_write> acc: array<vec4f>;
@group(0) @binding(6) var<storage, read_write> virial_fixed: array<atomic<u32>>;

const TILE: u32 = 256u;
var<workgroup> tile_hi: array<vec4f, TILE>;
var<workgroup> tile_lo: array<vec4f, TILE>;

// First half kick, then drift: v += a dt/2, x += v dt
@compute @workgroup_size(TILE)
fn drift(@builtin(global_invocation_id) id: vec3u) {
    df_init(params.one);
    let i = id.x;
    if i >= params.count {
        return;
    }
    let a = acc[i];
    var vh = vel_hi[i];
    var vl = vel_lo[i];
    var ph = pos_hi[i];
    var pl = pos_lo[i];
    for (var k = 0u; k < 3u; k++) {
        let v = df_add_f(vec2f(vh[k], vl[k]), a[k] * params.half_dt);
        vh[k] = v.x;
        vl[k] = v.y;
        let x = df_add(vec2f(ph[k], pl[k]), df_mul_f(v, params.dt));
        ph[k] = x.x;
        pl[k] = x.y;
    }
    vel_hi[i] = vh;
    vel_lo[i] = vl;
    pos_hi[i] = ph;
    pos_lo[i] = pl;
}

fn load_tile(t: u32, li: u32) {
    let j = t * TILE + li;
    if j < params.count {
        tile_hi[li] = pos_hi[j];
        tile_lo[li] = pos_lo[j];
    } else {
        // Padding: no mass, so no force
        tile_hi[li] = vec4f(0.0);
        tile_lo[li] = vec4f(0.0);
    }
}

struct Pair {
    a0: vec3f,
    a1: vec3f,
}

// Accelerations on two bodies, i0 = base + li and i1 = i0 + TILE, from every
// other body, one tile of TILE at a time through workgroup memory. Two bodies
// per thread halves the workgroup-memory reads per interaction. Every thread
// runs every barrier.
fn accelerations(base: u32, li: u32) -> Pair {
    let i0 = base + li;
    let i1 = i0 + TILE;
    var xh0 = vec3f(0.0);
    var xl0 = vec3f(0.0);
    var xh1 = vec3f(0.0);
    var xl1 = vec3f(0.0);
    if i0 < params.count {
        xh0 = pos_hi[i0].xyz;
        xl0 = pos_lo[i0].xyz;
    }
    if i1 < params.count {
        xh1 = pos_hi[i1].xyz;
        xl1 = pos_lo[i1].xyz;
    }
    // With softening, a body's pull on itself and on padding (mass 0) is
    // exactly zero, so the per-pair check is only needed for exact gravity
    let exact = params.eps2 == 0.0;
    // Each tile's 256 terms are summed in f32, then the tile totals in
    // double-float: one rounding per tile instead of one per body
    var a0h = vec3f(0.0);
    var a0l = vec3f(0.0);
    var a1h = vec3f(0.0);
    var a1l = vec3f(0.0);
    let tiles = (params.count + TILE - 1u) / TILE;
    for (var t = 0u; t < tiles; t++) {
        load_tile(t, li);
        workgroupBarrier();
        var t0 = vec3f(0.0);
        var t1 = vec3f(0.0);
        for (var k = 0u; k < TILE; k++) {
            let p = tile_hi[k];
            let q = tile_lo[k].xyz;
            let d0 = (p.xyz - xh0) + (q - xl0);
            let d1 = (p.xyz - xh1) + (q - xl1);
            var inv0 = inverseSqrt(dot(d0, d0) + params.eps2);
            var inv1 = inverseSqrt(dot(d1, d1) + params.eps2);
            if exact {
                let j = t * TILE + k;
                let pad = j >= params.count;
                inv0 = select(inv0, 0.0, pad || j == i0);
                inv1 = select(inv1, 0.0, pad || j == i1);
            }
            t0 += d0 * (p.w * inv0 * inv0 * inv0);
            t1 += d1 * (p.w * inv1 * inv1 * inv1);
        }
        // Components written out: indexing a vector with a loop variable
        // pushes it out of registers
        let x0 = df_add_f(vec2f(a0h.x, a0l.x), t0.x);
        let y0 = df_add_f(vec2f(a0h.y, a0l.y), t0.y);
        let z0 = df_add_f(vec2f(a0h.z, a0l.z), t0.z);
        let x1 = df_add_f(vec2f(a1h.x, a1l.x), t1.x);
        let y1 = df_add_f(vec2f(a1h.y, a1l.y), t1.y);
        let z1 = df_add_f(vec2f(a1h.z, a1l.z), t1.z);
        a0h = vec3f(x0.x, y0.x, z0.x);
        a0l = vec3f(x0.y, y0.y, z0.y);
        a1h = vec3f(x1.x, y1.x, z1.x);
        a1l = vec3f(x1.y, y1.y, z1.y);
        workgroupBarrier();
    }
    let a0 = a0h + a0l;
    let a1 = a1h + a1l;
    return Pair(a0 * params.g, a1 * params.g);
}

fn kick(i: u32, a: vec3f) {
    acc[i] = vec4f(a, 0.0);
    var vh = vel_hi[i];
    var vl = vel_lo[i];
    for (var k = 0u; k < 3u; k++) {
        let v = df_add_f(vec2f(vh[k], vl[k]), a[k] * params.half_dt);
        vh[k] = v.x;
        vl[k] = v.y;
    }
    vel_hi[i] = vh;
    vel_lo[i] = vl;
}

// Accelerations only: used once before the first step
@compute @workgroup_size(TILE)
fn prime(@builtin(workgroup_id) group: vec3u, @builtin(local_invocation_index) li: u32) {
    df_init(params.one);
    let base = group.x * 2u * TILE;
    let a = accelerations(base, li);
    if base + li < params.count {
        acc[base + li] = vec4f(a.a0, 0.0);
    }
    if base + li + TILE < params.count {
        acc[base + li + TILE] = vec4f(a.a1, 0.0);
    }
}

// New accelerations, then the second half kick: v += a dt/2
@compute @workgroup_size(TILE)
fn force_kick(@builtin(workgroup_id) group: vec3u, @builtin(local_invocation_index) li: u32) {
    df_init(params.one);
    let base = group.x * 2u * TILE;
    let a = accelerations(base, li);
    if base + li < params.count {
        kick(base + li, a.a0);
    }
    if base + li + TILE < params.count {
        kick(base + li + TILE, a.a1);
    }
}

// Bodies per measurement chunk, and the word in virial_fixed holding the
// first body of the next chunk
const MEASURE_CHUNK: u32 = 1024u;
const CURSOR: u32 = 16u;

// Moves the measurement on to the next chunk
@compute @workgroup_size(1)
fn advance() {
    atomicAdd(&virial_fixed[CURSOR], MEASURE_CHUNK);
}

// Energy and momentum into fixed-point slots: 0 kinetic, 1 potential,
// 2-4 momentum xyz. Everything here is double-float, distances and masses
// included, so a measurement is limited by fixed-point resolution (2^-52 of
// the total), not by f32. About 10x the cost of a force pass, so it runs in
// chunks of MEASURE_CHUNK bodies (each against all bodies), on a snapshot of
// the state bound in place of the live buffers.
@compute @workgroup_size(TILE)
fn measure(@builtin(global_invocation_id) id: vec3u, @builtin(local_invocation_index) li: u32) {
    df_init(params.one);
    fixed_wg_clear(li);
    let i = atomicLoad(&virial_fixed[CURSOR]) + id.x;
    let in_range = i < params.count;
    var xh = vec4f(0.0);
    var xl = vec4f(0.0);
    if in_range {
        xh = pos_hi[i];
        xl = pos_lo[i];
    }
    let eps2 = vec2f(params.eps2, params.eps2_lo);
    var pe = vec2f(0.0);
    let tiles = (params.count + TILE - 1u) / TILE;
    for (var t = 0u; t < tiles; t++) {
        load_tile(t, li);
        workgroupBarrier();
        for (var k = 0u; k < TILE; k++) {
            let j = t * TILE + k;
            if j == i || j >= params.count {
                continue;
            }
            let ph = tile_hi[k];
            let pl = tile_lo[k];
            var r2 = eps2;
            for (var c = 0u; c < 3u; c++) {
                let d = df_sub(vec2f(ph[c], pl[c]), vec2f(xh[c], xl[c]));
                r2 = df_add(r2, df_mul(d, d));
            }
            pe = df_add(pe, df_div(vec2f(ph.w, pl.w), df_sqrt(r2)));
        }
        workgroupBarrier();
    }

    var ke = vec2f(0.0);
    var momentum = vec3f(0.0);
    if in_range {
        let m = vec2f(xh.w, xl.w);
        let vh = vel_hi[i];
        let vl = vel_lo[i];
        for (var c = 0u; c < 3u; c++) {
            let v = vec2f(vh[c], vl[c]);
            ke = df_add(ke, df_mul(v, v));
            momentum[c] = df_to_f32(df_mul(m, v));
        }
        ke = df_mul_f(df_mul(ke, m), 0.5);
        // Each pair is counted from both ends, so half of it belongs to i
        pe = df_mul_f(df_mul(pe, m), -0.5 * params.g);
    } else {
        // Threads past the last body ran the loop too; drop what they summed
        pe = vec2f(0.0);
    }
    let es = params.energy_scale;
    fixed_wg_add(0u, ke.x * es);
    fixed_wg_add(0u, ke.y * es);
    fixed_wg_add(1u, pe.x * es);
    fixed_wg_add(1u, pe.y * es);
    for (var c = 0u; c < 3u; c++) {
        fixed_wg_add(2u + c, momentum[c] * params.momentum_scale);
    }
    workgroupBarrier();
    fixed_wg_flush(li, 0u);
}


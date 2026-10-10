// Granular material: spheres in contact, in SI units. Normal force is Hertz
// elasticity with viscoelastic damping (Brilliantov et al. 1996),
// F_n = k delta^(3/2) + A k sqrt(delta) d(delta)/dt with k = 4/3 E* sqrt(R*),
// clamped so contacts never pull. Tangential force is regularised Coulomb
// friction, -min(gamma_t |v_t|, mu F_n) along v_t. Grains don't rotate yet.
// Gravity points along -z; a tilted floor and four side walls hold the grains.
//
// Buffer layout as in lennard_jones.wgsl; the position's w holds the radius.
// Appended at build time: grid.wgsl, WGSL_FIXED and WGSL_DF64.

struct Params {
    count: u32,
    cells: u32,
    total_cells: u32,
    segment: u32,
    cell_size: f32,
    box_size: f32,
    dt: f32,
    half_dt: f32,
    one: f32,
    gravity: f32,
    density: f32,
    // Effective Young's modulus E*
    modulus: f32,
    // Dissipative constant A (seconds)
    damping: f32,
    friction: f32,
    friction_gamma: f32,
    energy_scale: f32,
    momentum_scale: f32,
    force_scale: f32,
    // Floor: points with dot(floor_normal, x) < floor_offset are below it
    floor_nx: f32,
    floor_nz: f32,
    floor_offset: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> state: array<vec4f>;
@group(0) @binding(2) var<storage, read_write> virial_fixed: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> cell_count: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read_write> cells: array<u32>;
@group(0) @binding(5) var<storage, read_write> ints: array<u32>;

fn mass_of(radius: f32) -> f32 {
    return params.density * 4.18879020478639 * radius * radius * radius;
}

// Half kick, then drift
@compute @workgroup_size(256)
fn begin(@builtin(global_invocation_id) id: vec3u) {
    df_init(params.one);
    let i = id.x;
    if i >= params.count {
        return;
    }
    let s = params.segment;
    let a = state[4u * s + i];
    var vh = state[2u * s + i];
    var vl = state[3u * s + i];
    var ph = state[i];
    var pl = state[s + i];
    for (var k = 0u; k < 3u; k++) {
        let v = df_add_f(vec2f(vh[k], vl[k]), a[k] * params.half_dt);
        let x = df_add(vec2f(ph[k], pl[k]), df_mul_f(v, params.dt));
        vh[k] = v.x;
        vl[k] = v.y;
        ph[k] = x.x;
        pl[k] = x.y;
    }
    state[2u * s + i] = vh;
    state[3u * s + i] = vl;
    state[i] = ph;
    state[s + i] = pl;
}

struct Contact {
    force: vec3f,
    // Elastic energy stored in the contact: 2/5 k delta^(5/2)
    energy: f32,
}

// Force on a sphere from a contact with normal n (pointing towards the
// sphere), overlap delta, relative velocity v (sphere minus other), effective
// radius and mass
fn contact(n: vec3f, delta: f32, v: vec3f, r_eff: f32) -> Contact {
    let k = 1.3333333333 * params.modulus * sqrt(r_eff);
    let sd = sqrt(delta);
    let approach = -dot(v, n);
    let fn_mag = max(k * delta * sd + params.damping * k * sd * approach, 0.0);
    var force = n * fn_mag;
    let vt = v + n * approach;
    let speed_t = length(vt);
    if speed_t > 0.0 {
        force -= vt * (min(params.friction_gamma * speed_t, params.friction * fn_mag) / speed_t);
    }
    return Contact(force, 0.4 * k * delta * delta * sd);
}

struct Forces {
    total: vec3f,
    // Force from the floor and walls only, and their stored elastic energy
    support: vec3f,
    elastic: f32,
}

fn forces(i: u32) -> Forces {
    let s = params.segment;
    let ph = state[i];
    let xh = ph.xyz;
    let xl = state[s + i].xyz;
    let r = ph.w;
    let v = state[2u * s + i].xyz + state[3u * s + i].xyz;
    var out = Forces(vec3f(0.0), vec3f(0.0), 0.0);

    let n = params.cells;
    let c = ints[i];
    let cc = vec3i(i32(c % n), i32((c / n) % n), i32(c / (n * n)));
    for (var dz = -1; dz <= 1; dz++) {
        for (var dy = -1; dy <= 1; dy++) {
            for (var dx = -1; dx <= 1; dx++) {
                let nc = cc + vec3i(dx, dy, dz);
                if any(nc < vec3i(0)) || any(nc >= vec3i(i32(n))) {
                    continue;
                }
                let cell = cell_index(vec3u(nc));
                for (var t = cells[cell]; t < cells[cell + 1u]; t++) {
                    let j = ints[s + t];
                    if j == i {
                        continue;
                    }
                    let pj = state[j];
                    let d = (xh - pj.xyz) + (xl - state[s + j].xyz);
                    let dist2 = dot(d, d);
                    let reach = r + pj.w;
                    if dist2 < reach * reach && dist2 > 0.0 {
                        let dist = sqrt(dist2);
                        let vj = state[2u * s + j].xyz + state[3u * s + j].xyz;
                        let r_eff = r * pj.w / reach;
                        let k = contact(d / dist, reach - dist, v - vj, r_eff);
                        out.total += k.force;
                        // Each contact is seen from both spheres
                        out.elastic += 0.5 * k.energy;
                    }
                }
            }
        }
    }

    // Floor and walls: planes with inward normals
    let x = xh + xl;
    let floor_n = vec3f(params.floor_nx, 0.0, params.floor_nz);
    let normals = array(floor_n, vec3f(1.0, 0.0, 0.0), vec3f(-1.0, 0.0, 0.0), vec3f(0.0, 1.0, 0.0), vec3f(0.0, -1.0, 0.0));
    let offsets = array(params.floor_offset, 0.0, -params.box_size, 0.0, -params.box_size);
    for (var w = 0u; w < 5u; w++) {
        let gap = dot(normals[w], x) - offsets[w];
        if gap < r {
            let k = contact(normals[w], r - gap, v, r);
            out.total += k.force;
            out.support += k.force;
            out.elastic += k.energy;
        }
    }
    let m = mass_of(r);
    out.total += vec3f(0.0, 0.0, -params.gravity * m);
    return out;
}

@compute @workgroup_size(256)
fn prime(@builtin(global_invocation_id) id: vec3u) {
    df_init(params.one);
    let i = id.x;
    if i < params.count {
        let f = forces(i);
        state[4u * params.segment + i] = vec4f(f.total / mass_of(state[i].w), 0.0);
    }
}

@compute @workgroup_size(256)
fn force_kick(@builtin(global_invocation_id) id: vec3u) {
    df_init(params.one);
    let i = id.x;
    if i >= params.count {
        return;
    }
    let s = params.segment;
    let a = forces(i).total / mass_of(state[i].w);
    state[4u * s + i] = vec4f(a, 0.0);
    var vh = state[2u * s + i];
    var vl = state[3u * s + i];
    for (var k = 0u; k < 3u; k++) {
        let v = df_add_f(vec2f(vh[k], vl[k]), a[k] * params.half_dt);
        vh[k] = v.x;
        vl[k] = v.y;
    }
    state[2u * s + i] = vh;
    state[3u * s + i] = vl;
}

// Slots: 0 kinetic, 1 gravitational, 2 elastic, 3-5 momentum, 6 vertical
// support force from floor and walls, 7 number of grains
@compute @workgroup_size(256)
fn measure(@builtin(global_invocation_id) id: vec3u, @builtin(local_invocation_index) li: u32) {
    df_init(params.one);
    fixed_wg_clear(li);
    workgroupBarrier();
    let i = id.x;
    var ke = vec2f(0.0);
    var pe = 0.0;
    var elastic = 0.0;
    var momentum = vec3f(0.0);
    var support = 0.0;
    var one = 0.0;
    if i < params.count {
        let s = params.segment;
        let p = state[i];
        let m = mass_of(p.w);
        let f = forces(i);
        let vh = state[2u * s + i];
        let vl = state[3u * s + i];
        for (var k = 0u; k < 3u; k++) {
            let v = vec2f(vh[k], vl[k]);
            ke = df_add(ke, df_mul(v, v));
            momentum[k] = m * (vh[k] + vl[k]);
        }
        ke = df_mul_f(ke, 0.5 * m);
        pe = m * params.gravity * (p.z + state[s + i].z);
        elastic = f.elastic;
        support = f.support.z;
        one = 1.0;
    }
    let es = params.energy_scale;
    fixed_wg_add(0u, ke.x * es);
    fixed_wg_add(0u, ke.y * es);
    fixed_wg_add(1u, pe * es);
    fixed_wg_add(2u, elastic * es);
    for (var k = 0u; k < 3u; k++) {
        fixed_wg_add(3u + k, momentum[k] * params.momentum_scale);
    }
    fixed_wg_add(6u, support * params.force_scale);
    fixed_wg_add(7u, one);
    workgroupBarrier();
    fixed_wg_flush(li, 0u);
}

// Granular material: spheres in contact, in SI units. Normal force is Hertz
// elasticity with viscoelastic damping (Brilliantov et al. 1996),
// F_n = k delta^(3/2) + A k sqrt(delta) d(delta)/dt with k = 4/3 E* sqrt(R*),
// clamped so contacts never pull. Tangential force follows Cundall and Strack:
// each contact keeps a spring stretched by the sliding of the contact point
// (which includes the grains' spin), F_t = -k_t xi - gamma_t v_t, capped at
// mu F_n, beyond which the contact slides and the spring is reset to the
// limit. So friction holds still below the friction angle and slides at
// Coulomb's value above it. Friction turns the grains through torque (moment
// of inertia 2/5 m r^2); optional rolling resistance opposes relative spin.
//
// Springs live in per-grain contact tables, double-buffered: each step reads
// last step's table (chosen by the parity word) and writes the other.
// Gravity points along -z; a tilted floor and four side walls hold the grains.
//
// Buffer layout as in lennard_jones.wgsl, plus angular velocity (section 5)
// and angular acceleration (section 6); the position's w holds the radius.
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
    rolling_friction: f32,
    _pad1: f32,
    _pad2: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> state: array<vec4f>;
@group(0) @binding(2) var<storage, read_write> virial_fixed: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> cell_count: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read_write> cells: array<u32>;
@group(0) @binding(5) var<storage, read_write> ints: array<u32>;
// Contact tables: spring vectors, and the partner of each entry (a grain
// index, a wall WALL + w, or EMPTY). Entry k of grain i in table t is at
// (t * CONTACTS + k) * segment + i.
@group(0) @binding(6) var<storage, read_write> springs: array<vec4f>;
@group(0) @binding(7) var<storage, read_write> partners: array<u32>;

fn grid_cells() -> vec3u {
    return vec3u(params.cells);
}

fn grid_cell_size() -> vec3f {
    return vec3f(params.cell_size);
}

const CONTACTS: u32 = 16u;
const EMPTY: u32 = 0xffffffffu;
const WALL: u32 = 0xfffffff0u;
// Word in virial_fixed whose low bit says which table holds the latest springs
const PARITY: u32 = 16u;

fn mass_of(radius: f32) -> f32 {
    return params.density * 4.18879020478639 * radius * radius * radius;
}

fn inertia_of(radius: f32) -> f32 {
    return 0.4 * mass_of(radius) * radius * radius;
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
    // Spin: half kick (a sphere's orientation doesn't affect its contacts)
    state[5u * s + i] = vec4f(state[5u * s + i].xyz + state[6u * s + i].xyz * params.half_dt, 0.0);
}

struct Contact {
    force: vec3f,
    normal_force: f32,
    // Elastic energy stored in the contact: 2/5 k delta^(5/2) for the normal
    // spring, plus 1/2 k_t |xi|^2 for the tangential one
    energy: f32,
    spring: vec3f,
}

// Force on a sphere from a contact with normal n (pointing towards the
// sphere), overlap delta, relative velocity v of the contact point (sphere's
// surface minus the other's), effective radius, and the contact's spring
// from the last step
fn contact(n: vec3f, delta: f32, v: vec3f, r_eff: f32, spring: vec3f) -> Contact {
    let k = 1.3333333333 * params.modulus * sqrt(r_eff);
    let sd = sqrt(delta);
    let approach = -dot(v, n);
    let fn_mag = max(k * delta * sd + params.damping * k * sd * approach, 0.0);
    let vt = v + n * approach;
    // Project the old spring onto the current tangent plane. (Rescaling it
    // back to its old length divides by the projection, which can be near
    // zero; contacts turn little per step, so the length lost is negligible.)
    var xi = spring - n * dot(spring, n) + vt * params.dt;
    // 2/7 of Hertz's incremental normal stiffness 3/2 k sqrt(delta)
    let kt = 0.42857142857 * k * sd;
    var ft = -kt * xi - params.friction_gamma * vt;
    let limit = params.friction * fn_mag;
    let size = length(ft);
    if size > limit {
        // Sliding: force at the Coulomb limit, and the spring set to carry it.
        // (Not including the damping term: k_t is near zero as a contact
        // forms, and gamma v_t / k_t would make a huge spring.)
        ft *= limit / max(size, 1e-30);
        if kt > 0.0 {
            xi = -ft / kt;
        } else {
            xi = vec3f(0.0);
        }
    }
    return Contact(n * fn_mag + ft, fn_mag, 0.4 * k * delta * delta * sd + 0.5 * kt * dot(xi, xi), xi);
}

// Last step's spring for grain i's contact with `partner` (zero if new)
fn old_spring(i: u32, partner: u32) -> vec3f {
    let table = atomicLoad(&virial_fixed[PARITY]) & 1u;
    let seg = params.segment;
    for (var k = 0u; k < CONTACTS; k++) {
        let slot = (table * CONTACTS + k) * seg + i;
        let p = partners[slot];
        if p == EMPTY {
            break;
        }
        if p == partner {
            return springs[slot].xyz;
        }
    }
    return vec3f(0.0);
}

fn store_spring(i: u32, entry: u32, partner: u32, spring: vec3f) {
    if entry >= CONTACTS {
        return;
    }
    let table = 1u - (atomicLoad(&virial_fixed[PARITY]) & 1u);
    let slot = (table * CONTACTS + entry) * params.segment + i;
    partners[slot] = partner;
    springs[slot] = vec4f(spring, 0.0);
}

struct Forces {
    total: vec3f,
    torque: vec3f,
    // Force from the floor and walls only, and their stored elastic energy
    support: vec3f,
    elastic: f32,
}

// Rolling resistance: a torque against the relative spin, up to
// mu_r r* F_n, regularised at low spin like sliding friction
fn rolling(relative_spin: vec3f, normal_force: f32, r_eff: f32) -> vec3f {
    let rate = length(relative_spin);
    if rate == 0.0 || params.rolling_friction == 0.0 {
        return vec3f(0.0);
    }
    let limit = params.rolling_friction * r_eff * normal_force;
    return -relative_spin * (min(params.friction_gamma * r_eff * r_eff * rate, limit) / rate);
}

// Forces on grain i. With `update`, also writes this step's springs into the
// other table (measurements don't).
fn forces(i: u32, update: bool) -> Forces {
    let s = params.segment;
    let ph = state[i];
    let xh = ph.xyz;
    let xl = state[s + i].xyz;
    let r = ph.w;
    let v = state[2u * s + i].xyz + state[3u * s + i].xyz;
    let w = state[5u * s + i].xyz;
    var out = Forces(vec3f(0.0), vec3f(0.0), vec3f(0.0), 0.0);
    var entries = 0u;

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
                    let dist = sqrt(dist2);
                    // Tested on the overlap itself: dist2 < reach^2 can hold
                    // while sqrt(dist2) rounds up to reach, which would give a
                    // zero or negative overlap and a NaN from its square root
                    if reach - dist > 0.0 && dist2 > 0.0 {
                        let normal = d / dist;
                        let vj = state[2u * s + j].xyz + state[3u * s + j].xyz;
                        let wj = state[5u * s + j].xyz;
                        let r_eff = r * pj.w / reach;
                        // The contact point sits mid-overlap: arm_i from i
                        // along -normal, arm_j from j along +normal. The arms
                        // then meet, so torques conserve angular momentum.
                        let overlap = reach - dist;
                        let arm_i = r - 0.5 * overlap;
                        let arm_j = pj.w - 0.5 * overlap;
                        let v_contact = (v - vj) - cross(arm_i * w + arm_j * wj, normal);
                        let k = contact(normal, overlap, v_contact, r_eff, old_spring(i, j));
                        if update {
                            store_spring(i, entries, j, k.spring);
                            entries += 1u;
                        }
                        out.total += k.force;
                        out.torque += cross(-arm_i * normal, k.force) + rolling(w - wj, k.normal_force, r_eff);
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
    let spin = w;
    for (var wall = 0u; wall < 5u; wall++) {
        let gap = dot(normals[wall], x) - offsets[wall];
        if gap < r {
            let normal = normals[wall];
            // The contact point is on the wall, `gap` from the centre
            let k = contact(normal, r - gap, v - gap * cross(spin, normal), r, old_spring(i, WALL + wall));
            if update {
                store_spring(i, entries, WALL + wall, k.spring);
                entries += 1u;
            }
            out.total += k.force;
            out.torque += cross(-gap * normal, k.force) + rolling(spin, k.normal_force, r);
            out.support += k.force;
            out.elastic += k.energy;
        }
    }
    if update && entries < CONTACTS {
        store_spring(i, entries, EMPTY, vec3f(0.0));
    }
    let m = mass_of(r);
    out.total += vec3f(0.0, 0.0, -params.gravity * m);
    return out;
}

// Makes the springs just written the latest
@compute @workgroup_size(1)
fn flip() {
    atomicAdd(&virial_fixed[PARITY], 1u);
}

@compute @workgroup_size(256)
fn prime(@builtin(global_invocation_id) id: vec3u) {
    df_init(params.one);
    let i = id.x;
    if i < params.count {
        let f = forces(i, true);
        let r = state[i].w;
        state[4u * params.segment + i] = vec4f(f.total / mass_of(r), 0.0);
        state[6u * params.segment + i] = vec4f(f.torque / inertia_of(r), 0.0);
    }
}

// Forces and torques from the current state. Kicking happens in `kick`, a
// separate dispatch: contact forces read neighbours' velocities and spins,
// which must not change while any grain is still computing.
@compute @workgroup_size(256)
fn force(@builtin(global_invocation_id) id: vec3u) {
    df_init(params.one);
    let i = id.x;
    if i >= params.count {
        return;
    }
    let s = params.segment;
    let f = forces(i, true);
    let r = state[i].w;
    state[4u * s + i] = vec4f(f.total / mass_of(r), 0.0);
    state[6u * s + i] = vec4f(f.torque / inertia_of(r), 0.0);
}

// Second half kick of velocity and spin
@compute @workgroup_size(256)
fn kick(@builtin(global_invocation_id) id: vec3u) {
    df_init(params.one);
    let i = id.x;
    if i >= params.count {
        return;
    }
    let s = params.segment;
    let a = state[4u * s + i];
    var vh = state[2u * s + i];
    var vl = state[3u * s + i];
    for (var k = 0u; k < 3u; k++) {
        let v = df_add_f(vec2f(vh[k], vl[k]), a[k] * params.half_dt);
        vh[k] = v.x;
        vl[k] = v.y;
    }
    state[2u * s + i] = vh;
    state[3u * s + i] = vl;
    state[5u * s + i] = vec4f(state[5u * s + i].xyz + state[6u * s + i].xyz * params.half_dt, 0.0);
}

// Slots: 0 kinetic (translation), 1 gravitational, 2 elastic, 3-5 momentum,
// 6 vertical support force from floor and walls, 7 kinetic (rotation)
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
    var spin_energy = 0.0;
    if i < params.count {
        let s = params.segment;
        let p = state[i];
        let m = mass_of(p.w);
        let f = forces(i, false);
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
        let w = state[5u * s + i].xyz;
        spin_energy = 0.5 * inertia_of(p.w) * dot(w, w);
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
    fixed_wg_add(7u, spin_energy * es);
    workgroupBarrier();
    fixed_wg_flush(li, 0u);
}

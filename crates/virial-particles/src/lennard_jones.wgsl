// Lennard-Jones particles in a periodic rectangular box, in reduced units
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
//   state: position hi, position lo, velocity hi, velocity lo, acceleration,
//          position at the last neighbour-list build, packed position
//
// The packed position is what neighbours read: the hi part in xyz and the lo
// part in w as three 9-bit fractions of the hi part's last-bit size, about 33
// significant bits in 16 bytes instead of 32. Neighbour reads are most of the
// memory traffic, so this nearly halves it.
//
// Forces use Verlet neighbour lists: everything within cutoff + skin, built
// from the cell grid. Each step `decide` checks whether any particle has moved
// more than skin / 2 since the last build and writes the dispatch sizes of the
// rebuild kernels into `dispatch` (zero when no rebuild is needed), so the GPU
// skips the rebuild without a round trip to the CPU. With that rule no pair
// within the cutoff is ever missing from a list.
//   ints:  cell of each particle, particles in cell order, random state,
//          original index of the particle in each slot
//   scratch: room to reorder the six state sections plus (random state,
//          original index) when the lists rebuild
//
// When the lists rebuild, particles are also moved into cell order, so that
// spatial neighbours sit next to each other in memory: neighbour reads then
// mostly hit the cache. `ints` section 3 remembers each slot's original index.
//   cells: cell starts (total_cells + 1), then the scan's block sums
//
// Prepended at build time: WGSL_FIXED and WGSL_DF64 come after this file.

struct Params {
    // Box sides, cell sides and cells along each axis, in xyz
    box_size: vec4f,
    cell_size: vec4f,
    cells: vec4u,
    // Heat exchange: hot slab from x to y, cold slab from z to w
    hex_slabs: vec4f,
    count: u32,
    total_cells: u32,
    // 0: constant energy (velocity Verlet), 1: Langevin thermostat (BAOAB),
    // 2: Nose-Hoover thermostat
    thermostat: u32,
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
    // Shifted-force cutoff: F(r_c), subtracted so the force also reaches
    // zero at the cutoff (0 for the plain shifted potential)
    force_shift: f32,
    // (cutoff + skin)^2 and (skin / 2)^2
    list_r2: f32,
    half_skin2: f32,
    // Neighbours stored per particle (the list holds a count, then these)
    capacity: u32,
    // Nose-Hoover: g k T (g degrees of freedom) and the thermostat mass Q
    nh_gkt: f32,
    nh_q: f32,
    // Profiles along z: number of slabs and their width
    profile_bins: u32,
    bin_width: f32,
    // Neighbours within 1.5 sigma that make a molecule count as condensed
    near_count: u32,
    // Heat exchange: energy moved from the cold slab to the hot one per step
    hex_energy: f32,
    // What measurements leave in each velocity's w: 0 neighbours within
    // 1.5 sigma, 1 local temperature
    drawn: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> state: array<vec4f>;
@group(0) @binding(2) var<storage, read_write> virial_fixed: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> cell_count: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read_write> cells: array<u32>;
@group(0) @binding(5) var<storage, read_write> ints: array<u32>;
@group(0) @binding(6) var<storage, read_write> neighbors: array<u32>;
@group(0) @binding(8) var<storage, read_write> scratch: array<vec4f>;
// Workgroup counts for the rebuild kernels: (cells, 1, 1), (particles, 1, 1),
// (1, 1, 1), each zeroed when no rebuild is needed
@group(0) @binding(7) var<storage, read_write> dispatch: array<u32>;

fn grid_cells() -> vec3u {
    return params.cells.xyz;
}

fn grid_cell_size() -> vec3f {
    return params.cell_size.xyz;
}

// Slots in virial_fixed past the measurement accumulators
const MAX_DISPLACEMENT: u32 = 16u;
const REBUILDS: u32 = 17u;
const OVERFLOW: u32 = 18u;
const NEVER_BUILT: u32 = 19u;
// Nose-Hoover: kinetic energy sum (fixed-point slot 10, words 20-21), the
// thermostat's xi and eta as double-floats, and the current velocity scale
// minus one. (The scale itself sits within about 1e-4 of 1, where f32 has
// only about 6e-8 resolution: storing it directly mis-books about 0.06% of
// the energy exchanged with the thermostat every half step.)
const KE_SLOT: u32 = 10u;
const XI: u32 = 22u;
const ETA: u32 = 24u;
const SCALE: u32 = 26u;
// Heat exchange: per slab (hot, then cold) HEX_SLOTS_PER_SLAB fixed-point
// slots from HEX_SLOT, then HEX_WORDS_PER_SLAB words per slab from HEX_WORDS
const HEX_SLOT: u32 = 14u;
const HEX_WORDS: u32 = 64u;

// ---- Packed positions ----

// Size of the last bit of h, from its exponent (integer operations only)
fn last_bit(h: f32) -> f32 {
    let e = (bitcast<u32>(h) >> 23u) & 0xffu;
    return bitcast<f32>((max(e, 24u) - 23u) << 23u);
}

// lo as three 9-bit fractions of each component's last bit. Bits 27-30 are
// 0100, so the word is always a normal float: never NaN, never subnormal.
fn pack_lo(hi: vec3f, lo: vec3f) -> f32 {
    var bits = 0x20000000u;
    for (var k = 0u; k < 3u; k++) {
        let q = i32(round(clamp(lo[k] / last_bit(hi[k]), -0.5, 0.5) * 510.0));
        bits |= u32(q + 256) << (9u * k);
    }
    return bitcast<f32>(bits);
}

fn unpack_lo(hi: vec3f, packed: f32) -> vec3f {
    let bits = bitcast<u32>(packed);
    var lo = vec3f(0.0);
    for (var k = 0u; k < 3u; k++) {
        let q = f32(i32((bits >> (9u * k)) & 511u) - 256);
        lo[k] = q / 510.0 * last_bit(hi[k]);
    }
    return lo;
}

fn store_packed(i: u32, hi: vec3f, lo: vec3f) {
    state[6u * params.segment + i] = vec4f(hi, pack_lo(hi, lo));
}

// Packs every position (before the first step)
@compute @workgroup_size(256)
fn pack(@builtin(global_invocation_id) id: vec3u) {
    if id.x < params.count {
        store_packed(id.x, state[id.x].xyz, state[params.segment + id.x].xyz);
    }
}

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

fn wrap(x: vec2f, side: f32) -> vec2f {
    var y = x;
    if y.x >= side {
        y = df_add_f(y, -side);
    } else if y.x < 0.0 {
        y = df_add_f(y, side);
    }
    return y;
}

// Nearest periodic image of a separation
fn nearest(d: vec3f) -> vec3f {
    let side = params.box_size.xyz;
    return d - side * round(d / side);
}

@compute @workgroup_size(256)
fn begin(@builtin(global_invocation_id) id: vec3u, @builtin(local_invocation_index) li: u32) {
    df_init(params.one);
    if li == 0u {
        atomicStore(&group_moved, 0u);
    }
    workgroupBarrier();
    let i = id.x;
    if i < params.count {
        step_particle(i);
    }
    workgroupBarrier();
    // One global atomic per workgroup: a million threads hitting the same
    // word every step would queue up one behind another
    if li == 0u {
        atomicMax(&virial_fixed[MAX_DISPLACEMENT], atomicLoad(&group_moved));
    }
}

var<workgroup> group_moved: atomic<u32>;

fn step_particle(i: u32) {
    let a = state[4u * params.segment + i];
    var vh = state[2u * params.segment + i];
    var vl = state[3u * params.segment + i];
    if params.thermostat == 2u {
        // The first Nose-Hoover half step's velocity scaling: v + v (s - 1)
        let s_minus_1 = bitcast<f32>(atomicLoad(&virial_fixed[SCALE]));
        for (var k = 0u; k < 3u; k++) {
            let old = vec2f(vh[k], vl[k]);
            let v = df_add(old, df_mul_f(old, s_minus_1));
            vh[k] = v.x;
            vl[k] = v.y;
        }
    }
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
        x = wrap(x, params.box_size[k]);
        vh[k] = v.x;
        vl[k] = v.y;
        ph[k] = x.x;
        pl[k] = x.y;
    }
    state[2u * params.segment + i] = vh;
    state[3u * params.segment + i] = vl;
    state[i] = ph;
    state[params.segment + i] = pl;
    store_packed(i, ph.xyz, pl.xyz);
    // Squared displacement since the last list build (nearest image), as
    // bits: for non-negative floats, integer order matches float order
    let moved = nearest(ph.xyz - state[5u * params.segment + i].xyz);
    atomicMax(&group_moved, bitcast<u32>(dot(moved, moved)));
}

// Sets the rebuild kernels' dispatch sizes: full when any particle has moved
// more than skin / 2 since the last build (or there has been no build yet),
// zero otherwise
@compute @workgroup_size(1)
fn decide() {
    let moved = bitcast<f32>(atomicLoad(&virial_fixed[MAX_DISPLACEMENT]));
    atomicStore(&virial_fixed[MAX_DISPLACEMENT], 0u);
    let rebuild = moved > params.half_skin2 || atomicLoad(&virial_fixed[NEVER_BUILT]) == 0u;
    let cells = select(0u, (params.total_cells + 255u) / 256u, rebuild);
    let particles = select(0u, (params.count + 255u) / 256u, rebuild);
    dispatch[0] = cells;
    dispatch[1] = 1u;
    dispatch[2] = 1u;
    dispatch[3] = particles;
    dispatch[4] = 1u;
    dispatch[5] = 1u;
    dispatch[6] = select(0u, 1u, rebuild);
    dispatch[7] = 1u;
    dispatch[8] = 1u;
    // Reordering keeps neighbours close in memory; that decays slowly, so
    // only every fourth rebuild pays for it
    let count = atomicLoad(&virial_fixed[REBUILDS]);
    dispatch[9] = select(0u, particles, rebuild && count % 4u == 0u);
    dispatch[10] = 1u;
    dispatch[11] = 1u;
    if rebuild {
        atomicAdd(&virial_fixed[REBUILDS], 1u);
        atomicStore(&virial_fixed[NEVER_BUILT], 1u);
    }
}

// ---- Nose-Hoover thermostat (Tuckerman's time-reversible splitting) ----
//
// Each step is: thermostat half step, velocity Verlet, thermostat half step.
// A half step sums the kinetic energy (ke_sum), then one thread (nh_update)
// advances xi and eta and sets the velocity scale exp(-xi dt / 2), which
// `begin` (first half) or `scale` (second half) applies. The conserved
// quantity is KE + PE + Q xi^2 / 2 + g k T eta.

@compute @workgroup_size(256)
fn ke_sum(@builtin(global_invocation_id) id: vec3u, @builtin(local_invocation_index) li: u32) {
    df_init(params.one);
    fixed_wg_clear(li);
    workgroupBarrier();
    var ke = vec2f(0.0);
    if id.x < params.count {
        let vh = state[2u * params.segment + id.x];
        let vl = state[3u * params.segment + id.x];
        for (var k = 0u; k < 3u; k++) {
            let v = vec2f(vh[k], vl[k]);
            ke = df_add(ke, df_mul(v, v));
        }
        ke = df_mul_f(ke, 0.5);
    }
    fixed_wg_add(0u, ke.x * params.energy_scale);
    fixed_wg_add(0u, ke.y * params.energy_scale);
    workgroupBarrier();
    fixed_wg_flush(li, KE_SLOT);
}

fn load_df(word: u32) -> vec2f {
    return vec2f(bitcast<f32>(atomicLoad(&virial_fixed[word])), bitcast<f32>(atomicLoad(&virial_fixed[word + 1u])));
}

fn store_df(word: u32, x: vec2f) {
    atomicStore(&virial_fixed[word], bitcast<u32>(x.x));
    atomicStore(&virial_fixed[word + 1u], bitcast<u32>(x.y));
}

@compute @workgroup_size(1)
fn nh_update() {
    df_init(params.one);
    // Kinetic energy from the 64-bit fixed-point sum, kept in double-float:
    // the low word goes in as two 16-bit halves so no bits are lost
    let low = atomicLoad(&virial_fixed[2u * KE_SLOT]);
    let high = atomicLoad(&virial_fixed[2u * KE_SLOT + 1u]);
    atomicStore(&virial_fixed[2u * KE_SLOT], 0u);
    atomicStore(&virial_fixed[2u * KE_SLOT + 1u], 0u);
    var raw = df_add_f(df_from(f32(bitcast<i32>(high))), f32(low >> 16u) / 65536.0);
    raw = df_add_f(raw, f32(low & 0xffffu) / 4294967296.0);
    let ke = df_mul_f(raw, 1.0 / params.energy_scale);

    var xi = load_df(XI);
    var eta = load_df(ETA);
    let quarter = 0.25 * params.dt;
    // xi += (2 KE - g k T) / Q * dt / 4
    let g1 = df_mul_f(df_add_f(df_mul_f(ke, 2.0), -params.nh_gkt), 1.0 / params.nh_q);
    xi = df_add(xi, df_mul_f(g1, quarter));
    // s - 1 = exp(-x) - 1 for small x, by series: accurate relative to s - 1
    let x = df_to_f32(xi) * params.half_dt;
    let s_minus_1 = -x * (1.0 - x * (0.5 - x * (1.0 / 6.0 - x / 24.0)));
    eta = df_add(eta, df_mul_f(xi, params.half_dt));
    // Kinetic energy after scaling: KE s^2 = KE + KE (2 (s - 1) + (s - 1)^2)
    let ke2 = df_add(ke, df_mul_f(ke, s_minus_1 * (2.0 + s_minus_1)));
    let g2 = df_mul_f(df_add_f(df_mul_f(ke2, 2.0), -params.nh_gkt), 1.0 / params.nh_q);
    xi = df_add(xi, df_mul_f(g2, quarter));
    store_df(XI, xi);
    store_df(ETA, eta);
    atomicStore(&virial_fixed[SCALE], bitcast<u32>(s_minus_1));
}

// The second Nose-Hoover half step's velocity scaling
@compute @workgroup_size(256)
fn nh_scale(@builtin(global_invocation_id) id: vec3u) {
    df_init(params.one);
    if id.x >= params.count {
        return;
    }
    let s_minus_1 = bitcast<f32>(atomicLoad(&virial_fixed[SCALE]));
    var vh = state[2u * params.segment + id.x];
    var vl = state[3u * params.segment + id.x];
    for (var k = 0u; k < 3u; k++) {
        let old = vec2f(vh[k], vl[k]);
        let v = df_add(old, df_mul_f(old, s_minus_1));
        vh[k] = v.x;
        vl[k] = v.y;
    }
    state[2u * params.segment + id.x] = vh;
    state[3u * params.segment + id.x] = vl;
}

// ---- Cell grid: grid.wgsl, appended at build time ----

// ---- Forces ----

struct Pair {
    // Acceleration (mass 1), this particle's half of the potential energy
    // and of the virial sum r . F over its pairs, and its neighbours within
    // NEAR_R2
    a: vec3f,
    pe: vec2f,
    w: vec2f,
    near: u32,
    // Sum of |v_j|^2 over neighbours within the cutoff, and their number
    v2: f32,
    within: u32,
}

// Condensed: at least params.near_count neighbours within 1.5 sigma
const NEAR_R2: f32 = 2.25;

// Copies every particle into its slot in cell order: slot s gets particle
// ints[segment + s] (the sorted order)
@compute @workgroup_size(256)
fn reorder_gather(@builtin(global_invocation_id) id: vec3u) {
    let s = id.x;
    if s >= params.count {
        return;
    }
    let seg = params.segment;
    let i = ints[seg + s];
    for (var k = 0u; k < 7u; k++) {
        scratch[k * seg + s] = state[k * seg + i];
    }
    scratch[7u * seg + s] = vec4f(bitcast<f32>(ints[2u * seg + i]), bitcast<f32>(ints[3u * seg + i]), 0.0, 0.0);
}

// Writes the gathered copy back; slots now hold particles in cell order, so
// the sorted order becomes the identity and each slot's cell is its cell
@compute @workgroup_size(256)
fn reorder_scatter(@builtin(global_invocation_id) id: vec3u) {
    let s = id.x;
    if s >= params.count {
        return;
    }
    let seg = params.segment;
    for (var k = 0u; k < 7u; k++) {
        state[k * seg + s] = scratch[k * seg + s];
    }
    let extra = scratch[7u * seg + s];
    ints[2u * seg + s] = bitcast<u32>(extra.x);
    ints[3u * seg + s] = bitcast<u32>(extra.y);
    ints[s] = cell_index(cell_coords(state[s].xyz));
    ints[seg + s] = s;
}

// Lists everything within cutoff + skin, in sorted cell order, and records
// the position the list was built at
@compute @workgroup_size(256)
fn build_list(@builtin(global_invocation_id) id: vec3u) {
    let i = id.x;
    if i >= params.count {
        return;
    }
    let xh = state[i].xyz;
    let xl = state[params.segment + i].xyz;
    let c = ints[i];
    let n = vec3i(params.cells.xyz);
    let ci = i32(c);
    let cc = vec3i(ci % n.x, (ci / n.x) % n.y, ci / (n.x * n.y));
    // Transposed layout: the count at [i], entry k at [segment * (k + 1) + i],
    // so neighbouring threads read neighbouring words
    let seg = params.segment;
    var found = 0u;
    for (var dz = -1; dz <= 1; dz++) {
        for (var dy = -1; dy <= 1; dy++) {
            for (var dx = -1; dx <= 1; dx++) {
                let nc = vec3u((cc + vec3i(dx, dy, dz) + n) % n);
                let cell = cell_index(nc);
                for (var s = cells[cell]; s < cells[cell + 1u]; s++) {
                    let j = ints[params.segment + s];
                    if j == i {
                        continue;
                    }
                    let pj = state[6u * params.segment + j];
                    let d = nearest((pj.xyz - xh) + (unpack_lo(pj.xyz, pj.w) - xl));
                    if dot(d, d) < params.list_r2 {
                        if found < params.capacity {
                            neighbors[seg * (found + 1u) + i] = j;
                        }
                        found += 1u;
                    }
                }
            }
        }
    }
    neighbors[i] = min(found, params.capacity);
    if found > params.capacity {
        atomicMax(&virial_fixed[OVERFLOW], found);
    }
    state[5u * params.segment + i] = vec4f(xh, 0.0);
}

// Forces from the neighbour list. With `energies`, also the potential energy
// and virial (only measurements need them).
fn interactions(i: u32, energies: bool) -> Pair {
    let xh = state[i].xyz;
    let xl = state[params.segment + i].xyz;
    let seg = params.segment;
    let count = neighbors[i];
    var out = Pair(vec3f(0.0), vec2f(0.0), vec2f(0.0), 0u, 0.0, 0u);
    for (var k = 0u; k < count; k++) {
        let j = neighbors[seg * (k + 1u) + i];
        let pj = state[6u * params.segment + j];
        let d = nearest((pj.xyz - xh) + (unpack_lo(pj.xyz, pj.w) - xl));
        let r2 = dot(d, d);
        if r2 < params.rc2 {
            let p = pair(r2);
            out.a -= d * (p.x / r2);
            if energies {
                out.pe = df_add_f(out.pe, 0.5 * p.y);
                out.w = df_add_f(out.w, 0.5 * p.x);
                out.near += select(0u, 1u, r2 < NEAR_R2);
                if params.drawn == 1u {
                    let vj = state[2u * params.segment + j].xyz;
                    out.v2 += dot(vj, vj);
                    out.within += 1u;
                }
            }
        }
    }
    return out;
}

// r . F and the potential energy of a pair at squared distance r2 < rc2
fn pair(r2: f32) -> vec2f {
    let inv2 = 1.0 / r2;
    let s6 = inv2 * inv2 * inv2;
    let s12 = s6 * s6;
    var rf = 24.0 * (2.0 * s12 - s6);
    var u = 4.0 * (s12 - s6) - params.shift;
    if params.force_shift != 0.0 {
        let r = sqrt(r2);
        rf -= r * params.force_shift;
        u += (r - sqrt(params.rc2)) * params.force_shift;
    }
    return vec2f(rf, u);
}

@compute @workgroup_size(256)
fn prime(@builtin(global_invocation_id) id: vec3u) {
    df_init(params.one);
    if id.x < params.count {
        state[4u * params.segment + id.x] = vec4f(interactions(id.x, false).a, 0.0);
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
    let a = interactions(i, false).a;
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

// Slots: 0 kinetic, 1 potential, 2 virial sum, 3-5 momentum, 6 condensed
// molecules. Also leaves each molecule's neighbour count within 1.5 sigma
// (or its local temperature, by params.drawn) in its velocity's w, for
// drawing.
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
    var condensed = 0.0;
    if i < params.count {
        let p = interactions(i, true);
        pe = p.pe;
        w = p.w;
        condensed = select(0.0, 1.0, p.near >= params.near_count);
        var drawn = f32(p.near);
        if params.drawn == 1u {
            // Local temperature: m v^2 / 3 averaged over the molecule and
            // its neighbours within the cutoff
            let vi = state[2u * params.segment + i].xyz;
            drawn = (dot(vi, vi) + p.v2) / (3.0 * f32(p.within + 1u));
        }
        state[2u * params.segment + i].w = drawn;
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
    fixed_wg_add(6u, condensed);
    workgroupBarrier();
    fixed_wg_flush(li, 0u);
}

// ---- Profiles along z ----
//
// Bound with the profile buffer in place of virial_fixed, so the fixed-point
// helpers add into it. Every call adds one sample to the running sums. Slab b
// has PROFILE_SLOTS slots from b * PROFILE_SLOTS: particle count, then
// m v_a^2 for a = x, y, z, then the virial tensor's diagonal W_aa, then
// potential energy. After the slabs come three slots of whole-box totals
// of W_aa, summed pair by pair rather than through the slabs.
//
// Kinetic terms and potential energy go to the slab holding the particle
// (half of each pair's energy to each end). The virial follows Irving and
// Kirkwood: each pair's share is spread over the slabs that the line between
// the two particles crosses, in proportion to the length inside each. That
// makes the normal pressure P_zz(z) exactly constant across an interface in
// mechanical equilibrium; other contours don't.

const PROFILE_SLOTS: u32 = 8u;

fn slab_of(z: f32) -> u32 {
    return u32(clamp(i32(floor(z / params.bin_width)), 0, i32(params.profile_bins) - 1));
}

fn profile_add(slab: u32, quantity: u32, x: f32) {
    fixed_add(slab * PROFILE_SLOTS + quantity, x);
}

@compute @workgroup_size(256)
fn profile(@builtin(global_invocation_id) id: vec3u, @builtin(local_invocation_index) li: u32) {
    df_init(params.one);
    fixed_wg_clear(li);
    workgroupBarrier();
    let i = id.x;
    var total = vec3f(0.0);
    if i < params.count {
        let seg = params.segment;
        let xh = state[i].xyz;
        let xl = state[seg + i].xyz;
        let home = slab_of(xh.z);
        let v = state[2u * seg + i].xyz + state[3u * seg + i].xyz;
        profile_add(home, 0u, 1.0);
        for (var k = 0u; k < 3u; k++) {
            profile_add(home, 1u + k, v[k] * v[k]);
        }
        let bins = i32(params.profile_bins);
        let width = params.bin_width;
        var pe = 0.0;
        let count = neighbors[i];
        for (var k = 0u; k < count; k++) {
            let j = neighbors[seg * (k + 1u) + i];
            let pj = state[6u * seg + j];
            let d = nearest((pj.xyz - xh) + (unpack_lo(pj.xyz, pj.w) - xl));
            let r2 = dot(d, d);
            if r2 >= params.rc2 {
                continue;
            }
            let p = pair(r2);
            pe += 0.5 * p.y;
            // This particle's half of the pair's virial tensor
            let w = 0.5 * d * d * (p.x / r2);
            total += w;
            let lo = xh.z + min(d.z, 0.0);
            let hi = xh.z + max(d.z, 0.0);
            let length = hi - lo;
            if length < 1e-6 * width {
                for (var a = 0u; a < 3u; a++) {
                    profile_add(home, 4u + a, w[a]);
                }
                continue;
            }
            // Slabs from the one holding lo to the one holding hi; the line
            // may cross the periodic boundary, so slab numbers wrap. The last
            // slab gets what's left, so the shares add up to exactly one.
            var b = i32(floor(lo / width));
            var left = 1.0;
            loop {
                let last = f32(b + 1) * width >= hi;
                let start = max(lo, f32(b) * width);
                let end = min(hi, f32(b + 1) * width);
                var share = select((end - start) / length, left, last);
                share = max(share, 0.0);
                left -= share;
                let slab = u32(((b % bins) + bins) % bins);
                for (var a = 0u; a < 3u; a++) {
                    profile_add(slab, 4u + a, w[a] * share);
                }
                if last {
                    break;
                }
                b += 1;
            }
        }
        profile_add(home, 7u, pe);
    }
    for (var a = 0u; a < 3u; a++) {
        fixed_wg_add(a, total[a]);
    }
    workgroupBarrier();
    if li < 3u {
        let low = atomicLoad(&virial_fixed_wg[2u * li]);
        let high = atomicLoad(&virial_fixed_wg[2u * li + 1u]);
        fixed_add_parts(params.profile_bins * PROFILE_SLOTS + li, vec2u(low, high));
    }
}

// ---- Heat exchange (eHEX: Wirnsberger, Frenkel and Dellago 2015) ----
//
// After each step, the hot slab gains hex_energy and the cold slab loses it,
// by scaling velocities about each slab's centre-of-mass velocity
// (Ikeshoji and Hafskjold): momentum stays put and exactly hex_energy moves.
// The original scheme drifts in energy through the splitting error; eHEX
// cancels the leading term with a correction to the positions of order
// dt^3 (their eq. 20, all energy supplied at the end of the step, "eHEX/a").
// hex_sum totals each slab, hex_update (one thread) works out the scales
// and correction coefficients, hex_apply scales and corrects.

const HEX_SLOTS_PER_SLAB: u32 = 9u;

// 0 hot, 1 cold, 2 neither
fn hex_slab(z: f32) -> u32 {
    let b = params.hex_slabs;
    if z >= b.x && z < b.y {
        return 0u;
    }
    if z >= b.z && z < b.w {
        return 1u;
    }
    return 2u;
}

// Per slab: count, momentum (3), sum of v^2, sum of f . v, sum of f (3)
@compute @workgroup_size(256)
fn hex_sum(@builtin(global_invocation_id) id: vec3u) {
    let i = id.x;
    if i >= params.count {
        return;
    }
    let seg = params.segment;
    let slab = hex_slab(state[i].z);
    if slab == 2u {
        return;
    }
    let v = state[2u * seg + i].xyz + state[3u * seg + i].xyz;
    let f = state[4u * seg + i].xyz;
    let first = HEX_SLOT + HEX_SLOTS_PER_SLAB * slab;
    fixed_add(first, 1.0);
    for (var k = 0u; k < 3u; k++) {
        fixed_add(first + 1u + k, v[k] * params.momentum_scale);
        fixed_add(first + 6u + k, f[k] * params.momentum_scale);
    }
    fixed_add(first + 4u, dot(v, v) * params.energy_scale);
    fixed_add(first + 5u, dot(f, v) * params.energy_scale);
}

// A fixed-point slot as a double-float, then cleared
fn take_fixed(slot: u32) -> vec2f {
    let low = atomicLoad(&virial_fixed[2u * slot]);
    let high = atomicLoad(&virial_fixed[2u * slot + 1u]);
    atomicStore(&virial_fixed[2u * slot], 0u);
    atomicStore(&virial_fixed[2u * slot + 1u], 0u);
    // The low word goes in as two 16-bit halves so no bits are lost
    let x = df_add_f(df_from(f32(bitcast<i32>(high))), f32(low >> 16u) / 65536.0);
    return df_add_f(x, f32(low & 0xffffu) / 4294967296.0);
}

// Words per slab from HEX_WORDS: centre-of-mass velocity (3), velocity scale
// minus one, the two correction coefficients, mean force (3)
const HEX_WORDS_PER_SLAB: u32 = 9u;

fn store_word(w: u32, x: f32) {
    atomicStore(&virial_fixed[w], bitcast<u32>(x));
}

fn load_word(w: u32) -> f32 {
    return bitcast<f32>(atomicLoad(&virial_fixed[w]));
}

@compute @workgroup_size(1)
fn hex_update() {
    df_init(params.one);
    let dt3 = params.dt * params.dt * params.dt;
    for (var slab = 0u; slab < 2u; slab++) {
        let first = HEX_SLOT + HEX_SLOTS_PER_SLAB * slab;
        let m = df_to_f32(take_fixed(first));
        var p = vec3f(0.0);
        var f = vec3f(0.0);
        for (var k = 0u; k < 3u; k++) {
            p[k] = df_to_f32(take_fixed(first + 1u + k)) / params.momentum_scale;
            f[k] = df_to_f32(take_fixed(first + 6u + k)) / params.momentum_scale;
        }
        let v2 = df_mul_f(take_fixed(first + 4u), 1.0 / params.energy_scale);
        let fv = df_to_f32(take_fixed(first + 5u)) / params.energy_scale;
        var com = vec3f(0.0);
        var s_minus_1 = 0.0;
        var c_velocity = 0.0;
        var c_force = 0.0;
        var mean_force = vec3f(0.0);
        if m > 0.0 {
            com = p / m;
            mean_force = f / m;
            // Kinetic energy about the centre of mass, K
            let kinetic = df_to_f32(df_add_f(df_mul_f(v2, 0.5), -0.5 * dot(p, p) / m));
            // Energy into this slab per step and per unit time (F)
            let energy = select(-1.0, 1.0, slab == 0u) * params.hex_energy;
            let flux = energy / params.dt;
            let x = energy / kinetic;
            // sqrt(1 + x) - 1, accurate when x is small
            s_minus_1 = x / (1.0 + sqrt(1.0 + x));
            // eq. 20 with eta_i = F / (2K) (v_i - v_com):
            // E_i = eta_i / K [F / 48 + 1/6 sum_j f_j . (v_j - v_com)]
            //       - F / (12 K) (f_i - mean f)
            let power = fv - dot(f, com);
            c_velocity = dt3 * flux / (2.0 * kinetic * kinetic) * (flux / 48.0 + power / 6.0);
            c_force = dt3 * flux / (12.0 * kinetic);
        }
        let w = HEX_WORDS + HEX_WORDS_PER_SLAB * slab;
        for (var k = 0u; k < 3u; k++) {
            store_word(w + k, com[k]);
            store_word(w + 6u + k, mean_force[k]);
        }
        store_word(w + 3u, s_minus_1);
        store_word(w + 4u, c_velocity);
        store_word(w + 5u, c_force);
    }
}

@compute @workgroup_size(256)
fn hex_apply(@builtin(global_invocation_id) id: vec3u) {
    df_init(params.one);
    let i = id.x;
    if i >= params.count {
        return;
    }
    let seg = params.segment;
    var ph = state[i];
    var pl = state[seg + i];
    let slab = hex_slab(ph.z);
    if slab == 2u {
        return;
    }
    let w = HEX_WORDS + HEX_WORDS_PER_SLAB * slab;
    let s_minus_1 = load_word(w + 3u);
    let c_velocity = load_word(w + 4u);
    let c_force = load_word(w + 5u);
    let f = state[4u * seg + i].xyz;
    var vh = state[2u * seg + i];
    var vl = state[3u * seg + i];
    for (var k = 0u; k < 3u; k++) {
        let com = load_word(w + k);
        let v = vec2f(vh[k], vl[k]);
        let relative = df_add_f(v, -com);
        // Position correction -dt^3 E_i, from the velocity before scaling
        let shift = c_velocity * df_to_f32(relative) - c_force * (f[k] - load_word(w + 6u + k));
        let x = wrap(df_add_f(vec2f(ph[k], pl[k]), -shift), params.box_size[k]);
        ph[k] = x.x;
        pl[k] = x.y;
        // v + (s - 1)(v - com)
        let out = df_add(v, df_mul_f(relative, s_minus_1));
        vh[k] = out.x;
        vl[k] = out.y;
    }
    state[2u * seg + i] = vh;
    state[3u * seg + i] = vl;
    state[i] = ph;
    state[seg + i] = pl;
    store_packed(i, ph.xyz, pl.xyz);
}

// ---- Heat current, for Green-Kubo ----
//
// J = sum_i e_i v_i + 1/2 sum_i sum_j (F_ij . v_i) r_ij, with e_i the
// particle's kinetic energy plus half of each of its pair energies. Bound
// with a series buffer in place of virial_fixed: slot 0 counts samples, and
// sample k goes to slots 1 + 3k (x, y, z). heat_current adds one sample,
// heat_current_advance moves the count on.

@compute @workgroup_size(256)
fn heat_current(@builtin(global_invocation_id) id: vec3u, @builtin(local_invocation_index) li: u32) {
    df_init(params.one);
    fixed_wg_clear(li);
    workgroupBarrier();
    let i = id.x;
    var j_i = vec3f(0.0);
    if i < params.count {
        let seg = params.segment;
        let xh = state[i].xyz;
        let xl = state[seg + i].xyz;
        let v = state[2u * seg + i].xyz + state[3u * seg + i].xyz;
        var e = 0.5 * dot(v, v);
        var pairs = vec3f(0.0);
        let count = neighbors[i];
        for (var k = 0u; k < count; k++) {
            let j = neighbors[seg * (k + 1u) + i];
            let pj = state[6u * seg + j];
            let d = nearest((pj.xyz - xh) + (unpack_lo(pj.xyz, pj.w) - xl));
            let r2 = dot(d, d);
            if r2 < params.rc2 {
                let p = pair(r2);
                e += 0.5 * p.y;
                // r_ij (F_ij . v_i) with r_ij = -d and F_ij = -d rf / r^2
                pairs += d * (dot(d, v) * p.x / r2);
            }
        }
        j_i = e * v + 0.5 * pairs;
    }
    for (var k = 0u; k < 3u; k++) {
        fixed_wg_add(k, j_i[k]);
    }
    workgroupBarrier();
    if li < 3u {
        let sample = atomicLoad(&virial_fixed[0]);
        let low = atomicLoad(&virial_fixed_wg[2u * li]);
        let high = atomicLoad(&virial_fixed_wg[2u * li + 1u]);
        fixed_add_parts(1u + 3u * sample + li, vec2u(low, high));
    }
}

@compute @workgroup_size(1)
fn heat_current_advance() {
    atomicAdd(&virial_fixed[0], 1u);
}

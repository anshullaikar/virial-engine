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
    count: u32,
    cells: u32,
    total_cells: u32,
    // 0: constant energy (velocity Verlet), 1: Langevin thermostat (BAOAB),
    // 2: Nose-Hoover thermostat
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
    _pad0: u32,
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
    store_packed(i, ph.xyz, pl.xyz);
    // Squared displacement since the last list build (nearest image), as
    // bits: for non-negative floats, integer order matches float order
    var moved = ph.xyz - state[5u * params.segment + i].xyz;
    moved -= params.box_size * round(moved / params.box_size);
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
    // and of the virial sum r . F over its pairs
    a: vec3f,
    pe: vec2f,
    w: vec2f,
}

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
    let n = params.cells;
    let cc = vec3i(i32(c % n), i32((c / n) % n), i32(c / (n * n)));
    // Transposed layout: the count at [i], entry k at [segment * (k + 1) + i],
    // so neighbouring threads read neighbouring words
    let seg = params.segment;
    var found = 0u;
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
                    let pj = state[6u * params.segment + j];
                    var d = (pj.xyz - xh) + (unpack_lo(pj.xyz, pj.w) - xl);
                    d -= params.box_size * round(d / params.box_size);
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
    var out = Pair(vec3f(0.0), vec2f(0.0), vec2f(0.0));
    for (var k = 0u; k < count; k++) {
        let j = neighbors[seg * (k + 1u) + i];
        let pj = state[6u * params.segment + j];
        var d = (pj.xyz - xh) + (unpack_lo(pj.xyz, pj.w) - xl);
        // Nearest periodic image
        d -= params.box_size * round(d / params.box_size);
        let r2 = dot(d, d);
        if r2 < params.rc2 {
            let inv2 = 1.0 / r2;
            let s6 = inv2 * inv2 * inv2;
            let s12 = s6 * s6;
            // r . F for this pair, and the force on i
            var rf = 24.0 * (2.0 * s12 - s6);
            var u = 4.0 * (s12 - s6) - params.shift;
            if params.force_shift != 0.0 {
                let r = sqrt(r2);
                rf -= r * params.force_shift;
                u += (r - sqrt(params.rc2)) * params.force_shift;
            }
            out.a -= d * (rf * inv2);
            if energies {
                out.pe = df_add_f(out.pe, 0.5 * u);
                out.w = df_add_f(out.w, 0.5 * rf);
            }
        }
    }
    return out;
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
        let p = interactions(i, true);
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

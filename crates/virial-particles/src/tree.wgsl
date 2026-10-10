// Barnes-Hut gravity on the GPU, after Karras (2012).
//
// Each force evaluation:
//   1. bounding box of all bodies (order-independent atomics on sortable bits)
//   2. 30-bit Morton code per body
//   3. stable LSD radix sort of (code, body), 8 passes of 4 bits
//   4. binary radix tree over the sorted codes, one internal node per thread
//   5. mass, centre of mass and bounding box per node, level by level: a node
//      is summed in pass p only from children stamped in earlier passes, so
//      no invocation reads another's writes from the same dispatch
//   6. each body walks the tree: a node of size s at distance d from its
//      centre of mass counts as one mass plus its quadrupole moment when
//      s / d < theta, else it opens
//
// Every step is deterministic: same input, same bits.
//
// Node numbering: internal nodes 0 .. N-2 (root 0), leaf l is node N-1+l.

struct Params {
    count: u32,
    groups: u32,
    // Section length in `sort` and `links`
    stride: u32,
    node_stride: u32,
    theta2: f32,
    eps2: f32,
    g: f32,
    _pad: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> positions: array<vec4f>;
// Sections: keys A, bodies A, keys B, bodies B (ping-pong for the sort)
@group(0) @binding(2) var<storage, read_write> sort: array<u32>;
@group(0) @binding(3) var<storage, read_write> hist: array<u32>;
// Sections: (centre of mass, mass), box min, box max, and the traceless
// quadrupole Q = sum m (3 s s^T - |s|^2 I) about the centre of mass as
// (xx, yy, zz, xy) and (xz, yz, -, -)
@group(0) @binding(4) var<storage, read_write> nodes: array<vec4f>;
// Sections: left child, right child, parent, ready stamp
@group(0) @binding(5) var<storage, read_write> links: array<u32>;
// 0-2 box min, 3-5 box max (sortable bits), 6 sort pass, 7 reduce pass
@group(0) @binding(6) var<storage, read_write> control: array<atomic<u32>>;
@group(0) @binding(7) var<storage, read_write> acc: array<vec4f>;

const SORT_PASS: u32 = 6u;
const REDUCE_PASS: u32 = 7u;

// ---- 1. Bounding box ----

// Order-preserving map from f32 to u32
fn sortable(x: f32) -> u32 {
    let b = bitcast<u32>(x);
    return select(b ^ 0x80000000u, ~b, (b & 0x80000000u) != 0u);
}

fn unsortable(u: u32) -> f32 {
    return bitcast<f32>(select(~u, u ^ 0x80000000u, (u & 0x80000000u) != 0u));
}

@compute @workgroup_size(1)
fn begin_tree() {
    for (var k = 0u; k < 3u; k++) {
        atomicStore(&control[k], 0xffffffffu);
        atomicStore(&control[3u + k], 0u);
    }
    atomicStore(&control[SORT_PASS], 0u);
    atomicStore(&control[REDUCE_PASS], 2u);
}

var<workgroup> wg_min: array<atomic<u32>, 3>;
var<workgroup> wg_max: array<atomic<u32>, 3>;

@compute @workgroup_size(256)
fn bounds(@builtin(global_invocation_id) id: vec3u, @builtin(local_invocation_index) li: u32) {
    if li < 3u {
        atomicStore(&wg_min[li], 0xffffffffu);
        atomicStore(&wg_max[li], 0u);
    }
    workgroupBarrier();
    if id.x < params.count {
        let p = positions[id.x].xyz;
        for (var k = 0u; k < 3u; k++) {
            atomicMin(&wg_min[k], sortable(p[k]));
            atomicMax(&wg_max[k], sortable(p[k]));
        }
    }
    workgroupBarrier();
    if li < 3u {
        atomicMin(&control[li], atomicLoad(&wg_min[li]));
        atomicMax(&control[3u + li], atomicLoad(&wg_max[li]));
    }
}

// ---- 2. Morton codes ----

// Spreads the low 10 bits of v so there are two zero bits between each
fn spread(v: u32) -> u32 {
    var x = v & 0x3ffu;
    x = (x | (x << 16u)) & 0x030000ffu;
    x = (x | (x << 8u)) & 0x0300f00fu;
    x = (x | (x << 4u)) & 0x030c30c3u;
    x = (x | (x << 2u)) & 0x09249249u;
    return x;
}

@compute @workgroup_size(256)
fn morton(@builtin(global_invocation_id) id: vec3u) {
    let i = id.x;
    if i >= params.count {
        return;
    }
    var lo = vec3f(0.0);
    var hi = vec3f(0.0);
    for (var k = 0u; k < 3u; k++) {
        lo[k] = unsortable(atomicLoad(&control[k]));
        hi[k] = unsortable(atomicLoad(&control[3u + k]));
    }
    let extent = max(hi - lo, vec3f(1e-30));
    let q = vec3u(clamp((positions[i].xyz - lo) / extent * 1023.0, vec3f(0.0), vec3f(1023.0)));
    sort[i] = (spread(q.x) << 2u) | (spread(q.y) << 1u) | spread(q.z);
    sort[params.stride + i] = i;
}

// ---- 3. Radix sort: 8 passes of 4 bits, stable ----

fn sort_source() -> u32 {
    return select(0u, 2u * params.stride, (atomicLoad(&control[SORT_PASS]) & 1u) == 1u);
}

fn digit_of(key: u32) -> u32 {
    return (key >> (4u * atomicLoad(&control[SORT_PASS]))) & 15u;
}

var<workgroup> wg_digits: array<u32, 256>;
var<workgroup> wg_counts: array<atomic<u32>, 16>;

// Digit counts per workgroup, stored digit-major: hist[d * groups + g]
@compute @workgroup_size(256)
fn radix_count(@builtin(global_invocation_id) id: vec3u, @builtin(local_invocation_index) li: u32, @builtin(workgroup_id) group: vec3u) {
    if li < 16u {
        atomicStore(&wg_counts[li], 0u);
    }
    workgroupBarrier();
    if id.x < params.count {
        atomicAdd(&wg_counts[digit_of(sort[sort_source() + id.x])], 1u);
    }
    workgroupBarrier();
    if li < 16u {
        hist[li * params.groups + group.x] = atomicLoad(&wg_counts[li]);
    }
}

var<workgroup> wg_scan: array<u32, 256>;

// Exclusive prefix sum of all 16 * groups counts, in one workgroup
@compute @workgroup_size(256)
fn radix_scan(@builtin(local_invocation_index) li: u32) {
    let total = 16u * params.groups;
    let chunk = (total + 255u) / 256u;
    let first = li * chunk;
    var sum = 0u;
    for (var k = 0u; k < chunk; k++) {
        if first + k < total {
            sum += hist[first + k];
        }
    }
    wg_scan[li] = sum;
    workgroupBarrier();
    for (var offset = 1u; offset < 256u; offset *= 2u) {
        var add = 0u;
        if li >= offset {
            add = wg_scan[li - offset];
        }
        workgroupBarrier();
        wg_scan[li] += add;
        workgroupBarrier();
    }
    var running = wg_scan[li] - sum;
    for (var k = 0u; k < chunk; k++) {
        if first + k < total {
            let v = hist[first + k];
            hist[first + k] = running;
            running += v;
        }
    }
}

// Each element goes to its digit's start plus the number of earlier elements
// in its workgroup with the same digit: stable
@compute @workgroup_size(256)
fn radix_scatter(@builtin(global_invocation_id) id: vec3u, @builtin(local_invocation_index) li: u32, @builtin(workgroup_id) group: vec3u) {
    let source = sort_source();
    let destination = 2u * params.stride - source;
    let in_range = id.x < params.count;
    var key = 0u;
    var digit = 16u;
    if in_range {
        key = sort[source + id.x];
        digit = digit_of(key);
    }
    wg_digits[li] = digit;
    workgroupBarrier();
    if in_range {
        var rank = 0u;
        for (var j = 0u; j < li; j++) {
            rank += select(0u, 1u, wg_digits[j] == digit);
        }
        let dest = hist[digit * params.groups + group.x] + rank;
        sort[destination + dest] = key;
        sort[destination + params.stride + dest] = sort[source + params.stride + id.x];
    }
}

@compute @workgroup_size(1)
fn radix_next() {
    atomicAdd(&control[SORT_PASS], 1u);
}

// ---- 4. Binary radix tree (Karras 2012) ----

fn key_at(i: i32) -> u32 {
    return sort[u32(i)];
}

// Length of the common prefix of sorted entries i and j; equal codes are
// told apart by their position, so every pair differs
fn delta(i: i32, j: i32) -> i32 {
    if j < 0 || j >= i32(params.count) {
        return -1;
    }
    let a = key_at(i);
    let b = key_at(j);
    if a == b {
        return 32 + i32(countLeadingZeros(u32(i) ^ u32(j)));
    }
    return i32(countLeadingZeros(a ^ b));
}

@compute @workgroup_size(256)
fn build(@builtin(global_invocation_id) id: vec3u) {
    let n = i32(params.count);
    let i = i32(id.x);
    if i >= n - 1 {
        return;
    }
    let d = select(-1, 1, delta(i, i + 1) - delta(i, i - 1) > 0);
    let d_min = delta(i, i - d);
    var l_max = 2;
    while delta(i, i + l_max * d) > d_min {
        l_max *= 2;
    }
    var l = 0;
    var t = l_max / 2;
    while t >= 1 {
        if delta(i, i + (l + t) * d) > d_min {
            l += t;
        }
        t /= 2;
    }
    let j = i + l * d;
    let d_node = delta(i, j);
    var s = 0;
    var div = 2;
    loop {
        let step = (l + div - 1) / div;
        if delta(i, i + (s + step) * d) > d_node {
            s += step;
        }
        if step <= 1 {
            break;
        }
        div *= 2;
    }
    let split = i + s * d + min(d, 0);
    let leaf_base = u32(n - 1);
    let left = select(u32(split), leaf_base + u32(split), min(i, j) == split);
    let right = select(u32(split + 1), leaf_base + u32(split + 1), max(i, j) == split + 1);
    let stride = params.node_stride;
    links[u32(i)] = left;
    links[stride + u32(i)] = right;
    links[2u * stride + left] = u32(i);
    links[2u * stride + right] = u32(i);
    links[3u * stride + u32(i)] = 0u;
}

// ---- 5. Node mass, centre of mass and box ----

@compute @workgroup_size(256)
fn leaves(@builtin(global_invocation_id) id: vec3u) {
    let l = id.x;
    if l >= params.count {
        return;
    }
    let node = params.count - 1u + l;
    let p = positions[sort[params.stride + l]];
    let ns = params.node_stride;
    nodes[node] = p;
    nodes[ns + node] = vec4f(p.xyz, 0.0);
    nodes[2u * ns + node] = vec4f(p.xyz, 0.0);
    nodes[3u * ns + node] = vec4f(0.0);
    nodes[4u * ns + node] = vec4f(0.0);
    links[3u * ns + node] = 1u;
}

@compute @workgroup_size(256)
fn reduce(@builtin(global_invocation_id) id: vec3u) {
    let i = id.x;
    if i + 1u >= params.count {
        return;
    }
    let ns = params.node_stride;
    let stamp = atomicLoad(&control[REDUCE_PASS]);
    if links[3u * ns + i] != 0u {
        return;
    }
    let left = links[i];
    let right = links[ns + i];
    let sl = links[3u * ns + left];
    let sr = links[3u * ns + right];
    // Children summed in earlier passes only
    if sl == 0u || sr == 0u || sl >= stamp || sr >= stamp {
        return;
    }
    let a = nodes[left];
    let b = nodes[right];
    let m = a.w + b.w;
    var centre = 0.5 * (a.xyz + b.xyz);
    if m > 0.0 {
        centre = (a.xyz * a.w + b.xyz * b.w) / m;
    }
    nodes[i] = vec4f(centre, m);
    nodes[ns + i] = min(nodes[ns + left], nodes[ns + right]);
    nodes[2u * ns + i] = max(nodes[2u * ns + left], nodes[2u * ns + right]);
    // Quadrupole about the new centre: the children's own moments plus
    // each child's mass at its offset (parallel axis theorem)
    var q1 = nodes[3u * ns + left] + nodes[3u * ns + right];
    var q2 = nodes[4u * ns + left] + nodes[4u * ns + right];
    for (var c = 0u; c < 2u; c++) {
        let child = select(right, left, c == 0u);
        let mc = nodes[child];
        let s = mc.xyz - centre;
        let s2 = dot(s, s);
        q1 += mc.w * vec4f(3.0 * s.x * s.x - s2, 3.0 * s.y * s.y - s2, 3.0 * s.z * s.z - s2, 3.0 * s.x * s.y);
        q2 += mc.w * vec4f(3.0 * s.x * s.z, 3.0 * s.y * s.z, 0.0, 0.0);
    }
    nodes[3u * ns + i] = q1;
    nodes[4u * ns + i] = q2;
    links[3u * ns + i] = stamp;
}

@compute @workgroup_size(1)
fn reduce_next() {
    atomicAdd(&control[REDUCE_PASS], 1u);
}

// ---- 6. Forces ----

// One thread per body in sorted order, so neighbouring threads walk similar
// paths through the tree
@compute @workgroup_size(256)
fn forces(@builtin(global_invocation_id) id: vec3u) {
    let l = id.x;
    if l >= params.count {
        return;
    }
    let body = sort[params.stride + l];
    let x = positions[body].xyz;
    let ns = params.node_stride;
    let leaf_base = params.count - 1u;
    var a = vec3f(0.0);
    var stack: array<u32, 64>;
    var top = 1u;
    stack[0] = 0u;
    if params.count == 1u {
        top = 0u;
    }
    while top > 0u {
        top -= 1u;
        let node = stack[top];
        let mc = nodes[node];
        let d = mc.xyz - x;
        let r2 = dot(d, d);
        var open = false;
        if node < leaf_base {
            let lo = nodes[ns + node].xyz;
            let hi = nodes[2u * ns + node].xyz;
            let size = max(max(hi.x - lo.x, hi.y - lo.y), hi.z - lo.z);
            let inside = all(x >= lo) && all(x <= hi);
            open = inside || size * size >= params.theta2 * r2;
        } else if node - leaf_base == l {
            continue;
        }
        if open && top + 2u <= 64u {
            stack[top] = links[node];
            stack[top + 1u] = links[ns + node];
            top += 2u;
        } else {
            let inv = inverseSqrt(r2 + params.eps2);
            let inv2 = inv * inv;
            let inv3 = inv2 * inv;
            a += d * (mc.w * inv3);
            if node < leaf_base {
                // Quadrupole term, with d from the body to the centre of mass:
                // G (-Q d / r^5 + 5/2 (d Q d) d / r^7)
                let q1 = nodes[3u * ns + node];
                let q2 = nodes[4u * ns + node];
                let qd = vec3f(
                    q1.x * d.x + q1.w * d.y + q2.x * d.z,
                    q1.w * d.x + q1.y * d.y + q2.y * d.z,
                    q2.x * d.x + q2.y * d.y + q1.z * d.z,
                );
                let inv5 = inv3 * inv2;
                a += -qd * inv5 + d * (2.5 * dot(d, qd) * inv5 * inv2);
            }
        }
    }
    acc[body] = vec4f(a * params.g, 0.0);
}

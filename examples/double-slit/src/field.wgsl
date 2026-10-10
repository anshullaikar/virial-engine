// Draws the wavefunction as an image and the detector screen as a bar
// graph beside it, both in one full-screen pass.
//
// Prepended at build time: colormaps.wgsl.

struct View {
    // Image rectangle in pixels: left, top, side
    image: vec4f,
    // Graph rectangle in pixels: left, top, width, height
    graph: vec4f,
    // Shown region: x min, y max, side; grid spacing
    domain: vec4f,
    // Detector x; |psi|^2 that maps to full brightness; smallest screen
    // value that fills the graph (so early noise stays small); background
    misc: vec4f,
    n: u32,
    // 0: density through inferno; 1: phase as hue, |psi| as brightness
    mode: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> view: View;
@group(0) @binding(1) var<storage, read> psi: array<vec2f>;
@group(0) @binding(2) var<storage, read> potential: array<f32>;
@group(0) @binding(3) var<storage, read> detector: array<f32>;
@group(0) @binding(4) var<storage, read> peak: array<f32>;

@vertex
fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
    let p = vec2f(f32((i << 1u) & 2u), f32(i & 2u));
    return vec4f(p * 2.0 - 1.0, 0.0, 1.0);
}

fn cell(x: f32, y: f32) -> vec2f {
    let n = f32(view.n);
    return vec2f(x / view.domain.w + n * 0.5, y / view.domain.w + n * 0.5);
}

fn at(i: i32, j: i32) -> vec2f {
    let n = i32(view.n);
    return psi[u32(clamp(j, 0, n - 1) * n + clamp(i, 0, n - 1))];
}

fn hue(a: f32) -> vec3f {
    let t = a / 6.2831853 + 0.5;
    return clamp(abs(fract(t + vec3f(0.0, 2.0, 1.0) / 3.0) * 6.0 - 3.0) - 1.0, vec3f(0.0), vec3f(1.0));
}

@fragment
fn fs(@builtin(position) frag: vec4f) -> @location(0) vec4f {
    let bg = vec3f(view.misc.w);
    let px = frag.xy;
    let side = view.image.z;
    let u = (px - view.image.xy) / side;
    let y = view.domain.y - (px.y - view.image.y) / side * view.domain.z;

    if all(u >= vec2f(0.0)) && all(u < vec2f(1.0)) {
        let x = view.domain.x + u.x * view.domain.z;
        let c = cell(x, y);
        let i = i32(floor(c.x));
        let j = i32(floor(c.y));
        let f = c - floor(c);
        // Bilinear in psi itself keeps the phase smooth
        let p = mix(mix(at(i, j), at(i + 1, j), f.x), mix(at(i, j + 1), at(i + 1, j + 1), f.x), f.y);
        let density = dot(p, p);
        let level = sqrt(clamp(density / view.misc.y, 0.0, 1.0));
        var colour: vec3f;
        if view.mode == 0u {
            colour = colormap(1u, level);
        } else {
            colour = hue(atan2(p.y, p.x)) * level;
        }
        let ni = u32(clamp(i32(round(c.x)), 0, i32(view.n) - 1));
        let nj = u32(clamp(i32(round(c.y)), 0, i32(view.n) - 1));
        if potential[nj * view.n + ni] > 0.0 {
            colour = vec3f(0.42, 0.44, 0.5);
        }
        // The screen, as a thin line
        let pixel = view.domain.z / side;
        if abs(x - view.misc.x) < 0.75 * pixel {
            colour = mix(colour, vec3f(0.55, 0.75, 1.0), 0.6);
        }
        return vec4f(colour, 1.0);
    }

    let g = view.graph;
    if px.x >= g.x && px.x < g.x + g.z && px.y >= g.y && px.y < g.y + g.w {
        let c = cell(0.0, y);
        let j = u32(clamp(i32(round(c.y)), 0, i32(view.n) - 1));
        let value = detector[j] / max(peak[0], view.misc.z);
        let t = (px.x - g.x) / g.z;
        if t < value {
            return vec4f(colormap(1u, 0.45 + 0.5 * value), 1.0);
        }
        return vec4f(bg + 0.025, 1.0);
    }
    return vec4f(bg, 1.0);
}

// The largest screen count, so the graph fills its width
var<workgroup> partial: array<f32, 256>;

@group(0) @binding(5) var<storage, read_write> peak_out: array<f32>;

@compute @workgroup_size(256)
fn find_peak(@builtin(local_invocation_index) li: u32) {
    var m = 0.0;
    for (var j = li; j < view.n; j += 256u) {
        m = max(m, detector[j]);
    }
    partial[li] = m;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s >>= 1u) {
        if li < s {
            partial[li] = max(partial[li], partial[li + s]);
        }
        workgroupBarrier();
    }
    if li == 0u {
        peak_out[0] = partial[0];
    }
}

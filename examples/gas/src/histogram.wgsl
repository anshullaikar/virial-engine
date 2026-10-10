// Speed histogram on the GPU, drawn as bars with the Maxwell-Boltzmann curve
// for the measured temperature over them. Prepended at build time:
// virial_render::COLORMAPS_WGSL.

struct Histogram {
    // Panel in pixels: x, y, width, height (y down)
    rect: vec4f,
    viewport: vec2f,
    speed_max: f32,
    count: u32,
    temperature: f32,
    // Probability density at the top of the panel
    density_max: f32,
    bins: u32,
    _pad: f32,
}

@group(0) @binding(0) var<uniform> h: Histogram;
@group(0) @binding(1) var<storage, read> velocities: array<vec4f>;
@group(0) @binding(2) var<storage, read_write> counts: array<atomic<u32>>;

@compute @workgroup_size(64)
fn clear(@builtin(global_invocation_id) id: vec3u) {
    if id.x < h.bins {
        atomicStore(&counts[id.x], 0u);
    }
}

@compute @workgroup_size(256)
fn fill(@builtin(global_invocation_id) id: vec3u) {
    if id.x >= h.count {
        return;
    }
    let speed = length(velocities[id.x].xyz);
    let bin = min(u32(speed / h.speed_max * f32(h.bins)), h.bins - 1u);
    atomicAdd(&counts[bin], 1u);
}

// Drawing reads the counts through a second, read-only binding
@group(0) @binding(3) var<storage, read> counts_read: array<u32>;

struct Out {
    @builtin(position) position: vec4f,
    @location(0) color: vec4f,
}

const CORNERS = array(vec2f(0.0, 0.0), vec2f(1.0, 0.0), vec2f(0.0, 1.0), vec2f(0.0, 1.0), vec2f(1.0, 0.0), vec2f(1.0, 1.0));

fn to_clip(pixel: vec2f) -> vec4f {
    return vec4f(pixel.x / h.viewport.x * 2.0 - 1.0, 1.0 - pixel.y / h.viewport.y * 2.0, 0.0, 1.0);
}

@vertex
fn vs_bars(@builtin(vertex_index) v: u32, @builtin(instance_index) bin: u32) -> Out {
    let width = h.speed_max / f32(h.bins);
    let density = f32(counts_read[bin]) / (f32(h.count) * width);
    let height = min(density / h.density_max, 1.0) * h.rect.w;
    let bar = h.rect.z / f32(h.bins);
    let corner = CORNERS[v];
    let pixel = vec2f(h.rect.x + (f32(bin) + 0.1 + corner.x * 0.8) * bar, h.rect.y + h.rect.w - corner.y * height);
    var out: Out;
    out.position = to_clip(pixel);
    out.color = vec4f(colormap(1u, (f32(bin) + 0.5) / f32(h.bins)), 1.0);
    return out;
}

// The Maxwell-Boltzmann density as a chain of small segments
@vertex
fn vs_curve(@builtin(vertex_index) v: u32, @builtin(instance_index) i: u32) -> Out {
    let segments = 160.0;
    let t = h.temperature;
    let s0 = f32(i) / segments * h.speed_max;
    let s1 = f32(i + 1u) / segments * h.speed_max;
    let norm = pow(1.0 / (6.283185307 * t), 1.5) * 12.566370614;
    let d0 = norm * s0 * s0 * exp(-s0 * s0 / (2.0 * t));
    let d1 = norm * s1 * s1 * exp(-s1 * s1 / (2.0 * t));
    let p0 = vec2f(h.rect.x + s0 / h.speed_max * h.rect.z, h.rect.y + h.rect.w * (1.0 - min(d0 / h.density_max, 1.0)));
    let p1 = vec2f(h.rect.x + s1 / h.speed_max * h.rect.z, h.rect.y + h.rect.w * (1.0 - min(d1 / h.density_max, 1.0)));
    // A quad of width 2.5 px along the segment
    let along = normalize(p1 - p0);
    let across = vec2f(-along.y, along.x) * 1.25 * max(h.viewport.y / 1080.0, 1.0);
    let corner = CORNERS[v];
    let pixel = mix(p0, p1, corner.x) + across * (corner.y * 2.0 - 1.0);
    var out: Out;
    out.position = to_clip(pixel);
    out.color = vec4f(0.35, 0.85, 1.0, 1.0);
    return out;
}

@fragment
fn fs_main(in: Out) -> @location(0) vec4f {
    return in.color;
}

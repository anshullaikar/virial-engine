// The gpu-playground WebGPU demo's kernels, unchanged apart from names.

struct Particle {
    pos: vec2f,
    vel: vec2f,
}

struct Params {
    scale: vec2f,
    dt: f32,
    strength: f32,
    damping: f32,
    count: u32,
    softening: f32,
    intensity: f32,
    attractor_count: u32,
}

const MAX_ATTRACTORS: u32 = 16u;

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> particles: array<Particle>;
// Attractor positions in .xy (uniform arrays need 16-byte elements)
@group(0) @binding(2) var<uniform> attractors: array<vec4f, MAX_ATTRACTORS>;

@compute @workgroup_size(256)
fn simulate(@builtin(global_invocation_id) id: vec3u) {
    let i = id.x;
    if i >= params.count {
        return;
    }
    var p = particles[i];
    let pull = params.strength * params.dt / f32(params.attractor_count);
    for (var a = 0u; a < params.attractor_count; a++) {
        let d = attractors[a].xy - p.pos;
        p.vel += d * (pull / (dot(d, d) + params.softening));
    }
    p.vel *= params.damping;
    p.pos += p.vel * params.dt;
    particles[i] = p;
}

@group(0) @binding(0) var<uniform> draw_params: Params;
@group(0) @binding(1) var<storage, read> draw_particles: array<Particle>;

struct VertexOut {
    @builtin(position) position: vec4f,
    @location(0) color: vec3f,
}

@vertex
fn vs_main(@builtin(vertex_index) i: u32) -> VertexOut {
    let p = draw_particles[i];
    var out: VertexOut;
    out.position = vec4f(p.pos * draw_params.scale, 0.0, 1.0);
    // A hash of the index picks each particle's colour
    let h = f32((i * 2654435761u) >> 8u) / 16777216.0;
    out.color = mix(vec3f(0.35, 0.5, 1.0), vec3f(1.0, 0.45, 0.3), h) * draw_params.intensity;
    return out;
}

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4f {
    return vec4f(in.color, 1.0);
}

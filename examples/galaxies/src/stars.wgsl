// Stars as soft glowing sprites, added together so dense regions shine.

struct View {
    view_proj: mat4x4f,
    viewport: vec2f,
    // Sprite radius in pixels
    size: f32,
    // Bodies [0, half) belong to the first galaxy
    half: u32,
    intensity: f32,
    heavy_mass: f32,
    _pad: vec2f,
}

@group(0) @binding(0) var<uniform> view: View;
@group(0) @binding(1) var<storage, read> positions: array<vec4f>;
@group(0) @binding(2) var<storage, read> velocities: array<vec4f>;

struct Out {
    @builtin(position) position: vec4f,
    @location(0) uv: vec2f,
    @location(1) color: vec3f,
}

const CORNERS = array(vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(-1.0, 1.0), vec2f(-1.0, 1.0), vec2f(1.0, -1.0), vec2f(1.0, 1.0));

@vertex
fn vs_main(@builtin(vertex_index) v: u32, @builtin(instance_index) i: u32) -> Out {
    let p = positions[i];
    let corner = CORNERS[v];
    var clip = view.view_proj * vec4f(p.xyz, 1.0);
    let heavy = p.w >= view.heavy_mass;
    let radius = select(view.size, view.size * 5.0, heavy);
    // Offset in screen pixels, whatever the depth
    clip += vec4f(corner * radius * 2.0 / view.viewport * clip.w, 0.0, 0.0);

    // Cool blue for one galaxy, warm amber for the other; faster stars whiter
    let speed = length(velocities[i].xyz);
    let base = select(vec3f(1.0, 0.62, 0.32), vec3f(0.42, 0.62, 1.0), i < view.half);
    var color = mix(base, vec3f(1.0), clamp(speed * 0.35 - 0.1, 0.0, 0.6)) * view.intensity;
    if heavy {
        color = vec3f(1.0, 0.95, 0.85);
    }

    var out: Out;
    out.position = clip;
    out.uv = corner;
    out.color = color;
    return out;
}

@fragment
fn fs_main(in: Out) -> @location(0) vec4f {
    let r2 = dot(in.uv, in.uv);
    if r2 > 1.0 {
        discard;
    }
    let glow = exp(-4.0 * r2);
    return vec4f(in.color * glow, 1.0);
}

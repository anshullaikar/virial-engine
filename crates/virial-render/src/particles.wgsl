// Particles as camera-facing quads: soft glowing sprites (additive, no depth)
// or shaded spheres (depth-tested, with the depth of the sphere's surface).
// Prepended at build time: colormaps.wgsl.

struct Uniforms {
    view: mat4x4f,
    projection: mat4x4f,
    viewport: vec2f,
    // Glow: sprite radius in pixels. Spheres: radius used when a position's w is 0
    size: f32,
    // 0 one colour, 1 two groups split at `split`, 2 speed or 3 the velocity's
    // w through a colour map
    coloring: u32,
    color_a: vec4f,
    color_b: vec4f,
    split: u32,
    colormap: u32,
    value_min: f32,
    value_max: f32,
    // Glow: brightness of each sprite; spheres: unused
    intensity: f32,
    // Glow: sprites for bodies at least this heavy (w) are drawn larger and white
    heavy: f32,
    _pad: vec2f,
}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> positions: array<vec4f>;
@group(0) @binding(2) var<storage, read> velocities: array<vec4f>;

struct Out {
    @builtin(position) position: vec4f,
    @location(0) uv: vec2f,
    @location(1) color: vec3f,
    // Sphere centre in view space and radius
    @location(2) centre: vec3f,
    @location(3) radius: f32,
}

const CORNERS = array(vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(-1.0, 1.0), vec2f(-1.0, 1.0), vec2f(1.0, -1.0), vec2f(1.0, 1.0));

fn particle_color(i: u32) -> vec3f {
    if u.coloring == 1u {
        return select(u.color_b.rgb, u.color_a.rgb, i < u.split);
    }
    if u.coloring == 2u {
        let speed = length(velocities[i].xyz);
        return colormap(u.colormap, (speed - u.value_min) / (u.value_max - u.value_min));
    }
    if u.coloring == 3u {
        return colormap(u.colormap, (velocities[i].w - u.value_min) / (u.value_max - u.value_min));
    }
    return u.color_a.rgb;
}

@vertex
fn vs_glow(@builtin(vertex_index) v: u32, @builtin(instance_index) i: u32) -> Out {
    let p = positions[i];
    let corner = CORNERS[v];
    var clip = u.projection * u.view * vec4f(p.xyz, 1.0);
    let heavy = u.heavy > 0.0 && p.w >= u.heavy;
    let radius = select(u.size, u.size * 5.0, heavy);
    clip += vec4f(corner * radius * 2.0 / u.viewport * clip.w, 0.0, 0.0);
    var out: Out;
    out.position = clip;
    out.uv = corner;
    out.color = select(particle_color(i) * u.intensity, vec3f(1.0, 0.95, 0.85), heavy);
    return out;
}

@fragment
fn fs_glow(in: Out) -> @location(0) vec4f {
    let r2 = dot(in.uv, in.uv);
    if r2 > 1.0 {
        discard;
    }
    return vec4f(in.color * exp(-4.0 * r2), 1.0);
}

@vertex
fn vs_sphere(@builtin(vertex_index) v: u32, @builtin(instance_index) i: u32) -> Out {
    let p = positions[i];
    let radius = select(u.size, p.w, p.w > 0.0);
    let corner = CORNERS[v];
    let centre = (u.view * vec4f(p.xyz, 1.0)).xyz;
    // A quad slightly larger than the sphere, facing the camera
    let corner_view = centre + vec3f(corner * radius * 1.15, 0.0);
    var out: Out;
    out.position = u.projection * vec4f(corner_view, 1.0);
    out.uv = corner * 1.15;
    out.color = particle_color(i);
    out.centre = centre;
    out.radius = radius;
    return out;
}

struct SphereFragment {
    @location(0) color: vec4f,
    @builtin(frag_depth) depth: f32,
}

@fragment
fn fs_sphere(in: Out) -> SphereFragment {
    let r2 = dot(in.uv, in.uv);
    if r2 > 1.0 {
        discard;
    }
    let normal = vec3f(in.uv, sqrt(1.0 - r2));
    // Depth of the visible surface point, not the quad
    let surface = in.centre + normal * in.radius;
    let clip = u.projection * vec4f(surface, 1.0);
    let light = normalize(vec3f(-0.4, 0.6, 0.7));
    let diffuse = max(dot(normal, light), 0.0);
    let shade = 0.28 + 0.72 * diffuse;
    let rim = pow(1.0 - normal.z, 3.0) * 0.15;
    var out: SphereFragment;
    out.color = vec4f(in.color * shade + rim, 1.0);
    out.depth = clip.z / clip.w;
    return out;
}

// ---- Planes: flat quads with a faint grid, drawn before spheres ----

struct Plane {
    // Corners a, b, c, d in order around the quad; w unused
    corners: array<vec4f, 4>,
    color: vec4f,
    // x: grid spacing in world units (0: none)
    grid: vec4f,
}

@group(0) @binding(3) var<storage, read> planes: array<Plane>;

struct PlaneOut {
    @builtin(position) position: vec4f,
    // Distance along the plane's two edges, in world units
    @location(0) along: vec2f,
    @location(1) @interpolate(flat) index: u32,
}

@vertex
fn vs_plane(@builtin(vertex_index) v: u32, @builtin(instance_index) i: u32) -> PlaneOut {
    let order = array(0u, 1u, 2u, 0u, 2u, 3u);
    let uv = array(vec2f(0.0, 0.0), vec2f(1.0, 0.0), vec2f(1.0, 1.0), vec2f(0.0, 1.0));
    let c = planes[i].corners;
    let k = order[v];
    var out: PlaneOut;
    out.position = u.projection * u.view * vec4f(c[k].xyz, 1.0);
    out.along = uv[k] * vec2f(length(c[1].xyz - c[0].xyz), length(c[3].xyz - c[0].xyz));
    out.index = i;
    return out;
}

@fragment
fn fs_plane(in: PlaneOut) -> @location(0) vec4f {
    let plane = planes[in.index];
    var color = plane.color.rgb;
    let spacing = plane.grid.x;
    if spacing > 0.0 {
        // Grid in the plane's own coordinates, a pixel wide at any distance
        let g = in.along / spacing;
        let width = fwidth(g);
        let line = step(fract(g), width) + step(1.0 - width, fract(g));
        let on = clamp(max(line.x, line.y), 0.0, 1.0);
        color = mix(color, color * 1.5 + 0.006, on);
    }
    return vec4f(color, 1.0);
}

struct VertexOut {
    @builtin(position) position: vec4f,
    @location(0) color: vec3f,
}

// Amber, orange and green corners
const POSITIONS = array(vec2f(0.0, 0.6), vec2f(-0.6, -0.5), vec2f(0.6, -0.5));
const COLORS = array(vec3f(0.89, 0.48, 0.07), vec3f(0.86, 0.21, 0.10), vec3f(0.04, 0.65, 0.33));

@vertex
fn vs_main(@builtin(vertex_index) i: u32) -> VertexOut {
    var out: VertexOut;
    out.position = vec4f(POSITIONS[i], 0.0, 1.0);
    out.color = COLORS[i];
    return out;
}

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4f {
    return vec4f(in.color, 1.0);
}

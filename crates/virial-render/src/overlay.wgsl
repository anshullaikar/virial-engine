// Text, panels and colour bars drawn over the frame, in pixel coordinates.
// Prepended at build time: colormaps.wgsl.

struct Screen {
    viewport: vec2f,
    _pad: vec2f,
    // font8x8: glyph c is rows 0-3 in words 2c and rows 4-7 in 2c + 1
    glyphs: array<vec4u, 64>,
}

struct Item {
    // x, y, width, height in pixels (y down)
    @location(0) rect: vec4f,
    @location(1) color: vec4f,
    // 0 glyph, 1 solid rectangle, 2 colour bar
    @location(2) kind: u32,
    @location(3) code: u32,
}

@group(0) @binding(0) var<uniform> screen: Screen;

struct Out {
    @builtin(position) position: vec4f,
    @location(0) uv: vec2f,
    @location(1) color: vec4f,
    @location(2) @interpolate(flat) kind: u32,
    @location(3) @interpolate(flat) code: u32,
}

const CORNERS = array(vec2f(0.0, 0.0), vec2f(1.0, 0.0), vec2f(0.0, 1.0), vec2f(0.0, 1.0), vec2f(1.0, 0.0), vec2f(1.0, 1.0));

@vertex
fn vs_main(@builtin(vertex_index) v: u32, item: Item) -> Out {
    let corner = CORNERS[v];
    let pixel = item.rect.xy + corner * item.rect.zw;
    var out: Out;
    out.position = vec4f(pixel.x / screen.viewport.x * 2.0 - 1.0, 1.0 - pixel.y / screen.viewport.y * 2.0, 0.0, 1.0);
    out.uv = corner;
    out.color = item.color;
    out.kind = item.kind;
    out.code = item.code;
    return out;
}

fn glyph_word(index: u32) -> u32 {
    return screen.glyphs[index / 4u][index % 4u];
}

@fragment
fn fs_main(in: Out) -> @location(0) vec4f {
    if in.kind == 1u {
        return in.color;
    }
    if in.kind == 2u {
        return vec4f(colormap(in.code, in.uv.x), 1.0);
    }
    let col = min(u32(in.uv.x * 8.0), 7u);
    let row = min(u32(in.uv.y * 8.0), 7u);
    let word = glyph_word(in.code * 2u + row / 4u);
    let bits = (word >> ((row % 4u) * 8u)) & 0xffu;
    if ((bits >> col) & 1u) == 0u {
        discard;
    }
    return in.color;
}

// Double-float ("df64") arithmetic: a value is the unevaluated sum hi + lo of
// two f32s, stored as vec2f(hi, lo). About 48 significant bits (14 decimal
// digits) on hardware that only has f32. Roughly 10-20x the cost of f32, so
// use it where accuracy matters: long integrations and totals.
//
// The algorithms are the classic error-free transformations (Dekker, Knuth).
// They capture rounding errors that a compiler in fast-math mode (Metal's
// default, which wgpu leaves on) would simplify away: (a + b) - a becomes b.
// So every add, subtract and multiply here goes through df_guard, which XORs
// the float's bits with a zero the compiler can't see. Float algebra can't
// cross an integer operation, so the compiler can neither simplify across a
// guard nor regroup sums of guarded terms. (A multiply by an unknown 1.0
// isn't enough: fast math may distribute it, (a + b)*one - a -> b*one.)
// Call df_init with a 1.0 read from a buffer at the start of each kernel;
// without it the mask is a constant and the guard may be optimised out.

var<private> df_mask: u32 = 0u;

fn df_init(one_from_a_buffer: f32) {
    // Zero at run time, unknown at compile time
    df_mask = bitcast<u32>(one_from_a_buffer) ^ 0x3f800000u;
}

fn df_guard(x: f32) -> f32 {
    return bitcast<f32>(bitcast<u32>(x) ^ df_mask);
}

fn df_two_sum(a: f32, b: f32) -> vec2f {
    let s = df_guard(a + b);
    let bb = df_guard(s - a);
    let ea = df_guard(a - df_guard(s - bb));
    let eb = df_guard(b - bb);
    return vec2f(s, df_guard(ea + eb));
}

fn df_quick_two_sum(a: f32, b: f32) -> vec2f {
    let s = df_guard(a + b);
    return vec2f(s, df_guard(b - df_guard(s - a)));
}

// Splits a into two 12-bit halves (Dekker), for an exact product without fma.
// (WGSL's fma isn't guaranteed fused on every backend.)
fn df_split(a: f32) -> vec2f {
    let c = df_guard(4097.0 * a);
    let hi = df_guard(c - df_guard(c - a));
    return vec2f(hi, df_guard(a - hi));
}

fn df_two_prod(a: f32, b: f32) -> vec2f {
    let p = df_guard(a * b);
    let x = df_split(a);
    let y = df_split(b);
    var e = df_guard(df_guard(x.x * y.x) - p);
    e = df_guard(e + df_guard(x.x * y.y));
    e = df_guard(e + df_guard(x.y * y.x));
    e = df_guard(e + df_guard(x.y * y.y));
    return vec2f(p, e);
}

fn df_from(a: f32) -> vec2f {
    return vec2f(a, 0.0);
}

fn df_to_f32(a: vec2f) -> f32 {
    return a.x + a.y;
}

fn df_add(a: vec2f, b: vec2f) -> vec2f {
    let s = df_two_sum(a.x, b.x);
    let t = df_two_sum(a.y, b.y);
    let u = df_quick_two_sum(s.x, df_guard(s.y + t.x));
    return df_quick_two_sum(u.x, df_guard(u.y + t.y));
}

fn df_neg(a: vec2f) -> vec2f {
    return -a;
}

fn df_sub(a: vec2f, b: vec2f) -> vec2f {
    return df_add(a, -b);
}

fn df_add_f(a: vec2f, b: f32) -> vec2f {
    let s = df_two_sum(a.x, b);
    return df_quick_two_sum(s.x, df_guard(s.y + a.y));
}

fn df_mul(a: vec2f, b: vec2f) -> vec2f {
    let p = df_two_prod(a.x, b.x);
    let cross = df_guard(df_guard(a.x * b.y) + df_guard(a.y * b.x));
    return df_quick_two_sum(p.x, df_guard(p.y + cross));
}

fn df_mul_f(a: vec2f, b: f32) -> vec2f {
    let p = df_two_prod(a.x, b);
    return df_quick_two_sum(p.x, df_guard(p.y + df_guard(a.y * b)));
}

fn df_div(a: vec2f, b: vec2f) -> vec2f {
    let q1 = df_guard(a.x / b.x);
    var r = df_sub(a, df_mul_f(b, q1));
    let q2 = df_guard(r.x / b.x);
    r = df_sub(r, df_mul_f(b, q2));
    let q3 = df_guard(r.x / b.x);
    return df_add_f(df_quick_two_sum(q1, q2), q3);
}

fn df_sqrt(a: vec2f) -> vec2f {
    if a.x <= 0.0 {
        return vec2f(0.0);
    }
    // One Newton step from the f32 root doubles the correct bits
    let s = sqrt(a.x);
    let e = df_sub(a, df_two_prod(s, s));
    return df_add_f(df_from(s), df_guard(df_to_f32(e) / df_guard(2.0 * s)));
}

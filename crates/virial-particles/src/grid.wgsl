// Cell grid shared by the short-range solvers. Bins particles into box-shaped
// cells, prefix-sums the counts, scatters particle indices into cell order
// and sorts each cell by particle index, so neighbour loops visit pairs in
// the same order on every run.
//
// The including shader declares `params` (with count, total_cells,
// segment), `state` (position hi parts at state[i]), `cell_count`, `cells`
// and `ints`, laid out as in lennard_jones.wgsl, and two functions:
// grid_cells() (cells along each axis) and grid_cell_size() (their sides).

fn cell_coords(x: vec3f) -> vec3u {
    let c = vec3i(floor(x / grid_cell_size()));
    let m = vec3i(grid_cells()) - 1;
    return vec3u(clamp(c, vec3i(0), m));
}

fn cell_index(c: vec3u) -> u32 {
    let n = grid_cells();
    return (c.z * n.y + c.y) * n.x + c.x;
}

@compute @workgroup_size(256)
fn clear_cells(@builtin(global_invocation_id) id: vec3u) {
    if id.x < params.total_cells {
        atomicStore(&cell_count[id.x], 0u);
    }
}

@compute @workgroup_size(256)
fn bin(@builtin(global_invocation_id) id: vec3u) {
    let i = id.x;
    if i >= params.count {
        return;
    }
    let c = cell_index(cell_coords(state[i].xyz));
    ints[i] = c;
    atomicAdd(&cell_count[c], 1u);
}

var<workgroup> scan_tmp: array<u32, 256>;

// Exclusive prefix sum of cell counts within each block of 256 cells
@compute @workgroup_size(256)
fn scan_blocks(@builtin(global_invocation_id) id: vec3u, @builtin(local_invocation_index) li: u32, @builtin(workgroup_id) group: vec3u) {
    let c = id.x;
    var v = 0u;
    if c < params.total_cells {
        v = atomicLoad(&cell_count[c]);
    }
    scan_tmp[li] = v;
    workgroupBarrier();
    for (var offset = 1u; offset < 256u; offset *= 2u) {
        var add = 0u;
        if li >= offset {
            add = scan_tmp[li - offset];
        }
        workgroupBarrier();
        scan_tmp[li] += add;
        workgroupBarrier();
    }
    let inclusive = scan_tmp[li];
    if c < params.total_cells {
        cells[c] = inclusive - v;
    }
    if li == 255u {
        cells[params.total_cells + 1u + group.x] = inclusive;
    }
}

// Exclusive prefix sum of the block totals: one workgroup, sixteen per thread
// (up to 4096 blocks, so up to 1,048,576 cells)
@compute @workgroup_size(256)
fn scan_sums(@builtin(local_invocation_index) li: u32) {
    let blocks = (params.total_cells + 255u) / 256u;
    var prefix: array<u32, 16>;
    var total = 0u;
    for (var k = 0u; k < 16u; k++) {
        let b = li * 16u + k;
        var v = 0u;
        if b < blocks {
            v = cells[params.total_cells + 1u + b];
        }
        prefix[k] = total;
        total += v;
    }
    scan_tmp[li] = total;
    workgroupBarrier();
    for (var offset = 1u; offset < 256u; offset *= 2u) {
        var add = 0u;
        if li >= offset {
            add = scan_tmp[li - offset];
        }
        workgroupBarrier();
        scan_tmp[li] += add;
        workgroupBarrier();
    }
    let before = scan_tmp[li] - total;
    for (var k = 0u; k < 16u; k++) {
        let b = li * 16u + k;
        if b < blocks {
            cells[params.total_cells + 1u + b] = before + prefix[k];
        }
    }
}

// Global cell starts; counts become fill cursors for the scatter
@compute @workgroup_size(256)
fn scan_add(@builtin(global_invocation_id) id: vec3u, @builtin(workgroup_id) group: vec3u) {
    let c = id.x;
    if c < params.total_cells {
        cells[c] += cells[params.total_cells + 1u + group.x];
        atomicStore(&cell_count[c], 0u);
    }
    if c == 0u {
        cells[params.total_cells] = params.count;
    }
}

@compute @workgroup_size(256)
fn scatter(@builtin(global_invocation_id) id: vec3u) {
    let i = id.x;
    if i >= params.count {
        return;
    }
    let c = ints[i];
    ints[params.segment + cells[c] + atomicAdd(&cell_count[c], 1u)] = i;
}

// Insertion sort of each cell by particle index: the scatter's order depends
// on thread timing, this makes it the same every run
@compute @workgroup_size(256)
fn sort_cells(@builtin(global_invocation_id) id: vec3u) {
    let c = id.x;
    if c >= params.total_cells {
        return;
    }
    let start = cells[c];
    let end = cells[c + 1u];
    for (var s = start + 1u; s < end; s++) {
        let key = ints[params.segment + s];
        var t = s;
        while t > start && ints[params.segment + t - 1u] > key {
            ints[params.segment + t] = ints[params.segment + t - 1u];
            t -= 1u;
        }
        ints[params.segment + t] = key;
    }
}


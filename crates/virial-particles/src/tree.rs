//! Barnes-Hut gravity: the GPU builds a tree over the bodies every step and
//! each body treats distant groups as single masses. O(N log N) instead of
//! O(N²), with an error set by `theta`. See tree.wgsl.

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

/// Reduction passes: one per tree level. Morton codes have 30 bits and ties
/// are broken by position (20 bits for a million bodies), so no tree is
/// deeper than 50.
const LEVELS: u32 = 64;
const WORKGROUP: u32 = 256;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    count: u32,
    groups: u32,
    stride: u32,
    node_stride: u32,
    theta2: f32,
    eps2: f32,
    g: f32,
    _pad: f32,
}

pub(crate) struct Tree {
    group: wgpu::BindGroup,
    pipelines: Vec<(&'static str, wgpu::ComputePipeline)>,
    groups: u32,
    internal_groups: u32,
}

impl Tree {
    pub(crate) fn new(
        device: &wgpu::Device,
        positions: &wgpu::Buffer,
        acc: &wgpu::Buffer,
        count: u32,
        theta: f64,
        softening: f64,
        g: f64,
    ) -> Self {
        assert!(count >= 2, "Barnes-Hut needs at least two bodies");
        let groups = count.div_ceil(WORKGROUP);
        let stride = count.div_ceil(64) * 64;
        let node_stride = (2 * count).div_ceil(64) * 64;
        let params = Params {
            count,
            groups,
            stride,
            node_stride,
            theta2: (theta * theta) as f32,
            eps2: (softening * softening) as f32,
            g: g as f32,
            _pad: 0.0,
        };
        let usage = wgpu::BufferUsages::STORAGE;
        let buffer = |label: &str, size: u64| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage,
                mapped_at_creation: false,
            })
        };
        let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("tree params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let sort = buffer("tree sort", 4 * stride as u64 * 4);
        let hist = buffer("tree digit counts", 16 * groups as u64 * 4);
        let nodes = buffer("tree nodes", 5 * node_stride as u64 * 16);
        let links = buffer("tree links", 4 * node_stride as u64 * 4);
        let control = buffer("tree control", 8 * 4);

        let entry = |binding: u32, ty| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let rw = wgpu::BufferBindingType::Storage { read_only: false };
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("tree"),
            entries: &[
                entry(0, wgpu::BufferBindingType::Uniform),
                entry(1, wgpu::BufferBindingType::Storage { read_only: true }),
                entry(2, rw),
                entry(3, rw),
                entry(4, rw),
                entry(5, rw),
                entry(6, rw),
                entry(7, rw),
            ],
        });
        let buffers = [
            &params_buffer,
            positions,
            &sort,
            &hist,
            &nodes,
            &links,
            &control,
            acc,
        ];
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("tree"),
            layout: &layout,
            entries: &buffers
                .iter()
                .enumerate()
                .map(|(i, b)| wgpu::BindGroupEntry {
                    binding: i as u32,
                    resource: b.as_entire_binding(),
                })
                .collect::<Vec<_>>(),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("tree"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("tree"),
            source: wgpu::ShaderSource::Wgsl(include_str!("tree.wgsl").into()),
        });
        let pipelines = [
            "begin_tree",
            "bounds",
            "morton",
            "radix_count",
            "radix_scan",
            "radix_scatter",
            "radix_next",
            "build",
            "leaves",
            "reduce",
            "reduce_next",
            "forces",
        ]
        .map(|entry| {
            let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&pipeline_layout),
                module: &module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            });
            (entry, pipeline)
        })
        .to_vec();
        Self {
            group,
            pipelines,
            groups,
            internal_groups: (count - 1).div_ceil(WORKGROUP),
        }
    }

    fn run(&self, pass: &mut wgpu::ComputePass<'_>, name: &str, groups: u32) {
        let pipeline = &self
            .pipelines
            .iter()
            .find(|(n, _)| *n == name)
            .expect("known pipeline")
            .1;
        pass.set_pipeline(pipeline);
        pass.dispatch_workgroups(groups, 1, 1);
    }

    /// Records a full force evaluation into `acc`. Leaves the tree's bind
    /// group set: callers rebind their own afterwards.
    pub(crate) fn forces(&self, pass: &mut wgpu::ComputePass<'_>) {
        pass.set_bind_group(0, &self.group, &[]);
        self.run(pass, "begin_tree", 1);
        self.run(pass, "bounds", self.groups);
        self.run(pass, "morton", self.groups);
        for _ in 0..8 {
            self.run(pass, "radix_count", self.groups);
            self.run(pass, "radix_scan", 1);
            self.run(pass, "radix_scatter", self.groups);
            self.run(pass, "radix_next", 1);
        }
        self.run(pass, "build", self.internal_groups);
        self.run(pass, "leaves", self.groups);
        for _ in 0..LEVELS {
            self.run(pass, "reduce", self.internal_groups);
            self.run(pass, "reduce_next", 1);
        }
        self.run(pass, "forces", self.groups);
    }
}

//! Particles drawn from position and velocity buffers (`vec4f` each, as the
//! Virial solvers lay them out: position in xyz and mass or radius in w).

use bytemuck::{Pod, Zeroable};

use crate::{Camera, Colormap, Mat4};

/// How particles look.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Style {
    /// Soft glowing sprites, added together (stars). `heavy`: bodies with
    /// mass (w) at least this are drawn larger and white; 0 turns that off.
    Glow {
        radius_px: f32,
        intensity: f32,
        heavy: f32,
    },
    /// Shaded, depth-tested spheres. Radius from each position's w, or
    /// `radius` where w is 0.
    Spheres { radius: f32 },
}

/// A flat quad drawn under the particles: a floor or wall. Corners go around
/// the quad in order. `grid`: spacing of faint grid lines in world units, 0
/// for none.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Plane {
    pub corners: [[f32; 3]; 4],
    pub color: [f32; 3],
    pub grid: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct PlaneGpu {
    corners: [[f32; 4]; 4],
    color: [f32; 4],
    grid: [f32; 4],
}

const MAX_PLANES: u64 = 8;

/// How particles are coloured. Colours are linear RGB.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Coloring {
    Uniform([f32; 3]),
    /// Particles below `split` get `a`, the rest `b`
    Groups {
        split: u32,
        a: [f32; 3],
        b: [f32; 3],
    },
    /// Speed mapped from `min`..`max` through `map`
    Speed {
        map: Colormap,
        min: f32,
        max: f32,
    },
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Uniforms {
    view: Mat4,
    projection: Mat4,
    viewport: [f32; 2],
    size: f32,
    coloring: u32,
    color_a: [f32; 4],
    color_b: [f32; 4],
    split: u32,
    colormap: u32,
    value_min: f32,
    value_max: f32,
    intensity: f32,
    heavy: f32,
    _pad: [f32; 2],
}

/// Draws particles into a frame.
pub struct Particles {
    layout: wgpu::BindGroupLayout,
    glow: wgpu::RenderPipeline,
    spheres: wgpu::RenderPipeline,
    planes_pipeline: wgpu::RenderPipeline,
    uniforms: wgpu::Buffer,
    planes: wgpu::Buffer,
    plane_count: u32,
    depth: Option<(wgpu::TextureView, u32, u32)>,
}

const DEPTH: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

impl Particles {
    pub fn new(device: &wgpu::Device, format: wgpu::TextureFormat) -> Self {
        let entry = |binding, ty| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let storage = wgpu::BufferBindingType::Storage { read_only: true };
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("particles"),
            entries: &[
                entry(0, wgpu::BufferBindingType::Uniform),
                entry(1, storage),
                entry(2, storage),
                entry(3, storage),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("particles"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let source = format!(
            "{}\n{}",
            include_str!("colormaps.wgsl"),
            include_str!("particles.wgsl")
        );
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("particles"),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
        let pipeline = |vs: &str, fs: &str, blend: Option<wgpu::BlendState>, depth: bool| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(vs),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &module,
                    entry_point: Some(vs),
                    compilation_options: Default::default(),
                    buffers: &[],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &module,
                    entry_point: Some(fs),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format,
                        blend,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: depth.then(|| wgpu::DepthStencilState {
                    format: DEPTH,
                    depth_write_enabled: Some(true),
                    depth_compare: Some(wgpu::CompareFunction::Less),
                    stencil: Default::default(),
                    bias: Default::default(),
                }),
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            })
        };
        let additive = wgpu::BlendComponent {
            src_factor: wgpu::BlendFactor::One,
            dst_factor: wgpu::BlendFactor::One,
            operation: wgpu::BlendOperation::Add,
        };
        let glow = pipeline(
            "vs_glow",
            "fs_glow",
            Some(wgpu::BlendState {
                color: additive,
                alpha: additive,
            }),
            false,
        );
        let spheres = pipeline("vs_sphere", "fs_sphere", None, true);
        let planes_pipeline = pipeline("vs_plane", "fs_plane", None, true);
        let planes = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("planes"),
            size: MAX_PLANES * size_of::<PlaneGpu>() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let uniforms = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("particles"),
            size: size_of::<Uniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self {
            layout,
            glow,
            spheres,
            planes_pipeline,
            uniforms,
            planes,
            plane_count: 0,
            depth: None,
        }
    }

    /// Floors and walls drawn (with depth) under sphere-style particles. Up
    /// to 8.
    pub fn set_planes(&mut self, queue: &wgpu::Queue, planes: &[Plane]) {
        let gpu: Vec<PlaneGpu> = planes
            .iter()
            .take(MAX_PLANES as usize)
            .map(|p| PlaneGpu {
                corners: p.corners.map(|c| [c[0], c[1], c[2], 1.0]),
                color: [p.color[0], p.color[1], p.color[2], 1.0],
                grid: [p.grid, 0.0, 0.0, 0.0],
            })
            .collect();
        if !gpu.is_empty() {
            queue.write_buffer(&self.planes, 0, bytemuck::cast_slice(&gpu));
        }
        self.plane_count = gpu.len() as u32;
    }

    /// Binds the buffers to draw from: `vec4f` positions and velocities.
    pub fn bind(
        &self,
        device: &wgpu::Device,
        positions: wgpu::BindingResource<'_>,
        velocities: wgpu::BindingResource<'_>,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("particles"),
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.uniforms.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: positions,
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: velocities,
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.planes.as_entire_binding(),
                },
            ],
        })
    }

    fn depth_view(&mut self, device: &wgpu::Device, width: u32, height: u32) -> &wgpu::TextureView {
        if self
            .depth
            .as_ref()
            .is_none_or(|(_, w, h)| (*w, *h) != (width, height))
        {
            let texture = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("particle depth"),
                size: wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: DEPTH,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            });
            self.depth = Some((texture.create_view(&Default::default()), width, height));
        }
        &self.depth.as_ref().expect("just created").0
    }

    /// Clears `view` to `background` and draws `count` particles.
    #[allow(clippy::too_many_arguments)]
    pub fn draw(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        size: (u32, u32),
        camera: &Camera,
        group: &wgpu::BindGroup,
        count: u32,
        style: Style,
        coloring: Coloring,
        background: [f64; 3],
        timestamp_writes: Option<wgpu::RenderPassTimestampWrites<'_>>,
    ) {
        let (width, height) = size;
        let rgb4 = |c: [f32; 3]| [c[0], c[1], c[2], 1.0];
        let mut u = Uniforms {
            view: camera.view(),
            projection: camera.projection(width as f32 / height.max(1) as f32),
            viewport: [width as f32, height as f32],
            size: 0.0,
            coloring: 0,
            color_a: [1.0; 4],
            color_b: [1.0; 4],
            split: 0,
            colormap: 0,
            value_min: 0.0,
            value_max: 1.0,
            intensity: 1.0,
            heavy: 0.0,
            _pad: [0.0; 2],
        };
        match coloring {
            Coloring::Uniform(c) => u.color_a = rgb4(c),
            Coloring::Groups { split, a, b } => {
                u.coloring = 1;
                u.split = split;
                u.color_a = rgb4(a);
                u.color_b = rgb4(b);
            }
            Coloring::Speed { map, min, max } => {
                u.coloring = 2;
                u.colormap = map as u32;
                u.value_min = min;
                u.value_max = max.max(min + 1e-12);
            }
        }
        let spheres = match style {
            Style::Glow {
                radius_px,
                intensity,
                heavy,
            } => {
                u.size = radius_px;
                u.intensity = intensity;
                u.heavy = heavy;
                false
            }
            Style::Spheres { radius } => {
                u.size = radius;
                true
            }
        };
        queue.write_buffer(&self.uniforms, 0, bytemuck::bytes_of(&u));
        let depth = spheres.then(|| self.depth_view(device, width, height).clone());
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("particles"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color {
                        r: background[0],
                        g: background[1],
                        b: background[2],
                        a: 1.0,
                    }),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: depth.as_ref().map(|view| {
                wgpu::RenderPassDepthStencilAttachment {
                    view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Discard,
                    }),
                    stencil_ops: None,
                }
            }),
            timestamp_writes,
            ..Default::default()
        });
        pass.set_bind_group(0, group, &[]);
        if spheres && self.plane_count > 0 {
            pass.set_pipeline(&self.planes_pipeline);
            pass.draw(0..6, 0..self.plane_count);
        }
        pass.set_pipeline(if spheres { &self.spheres } else { &self.glow });
        pass.draw(0..6, 0..count);
    }
}

//! Text, panels and colour bars drawn over a frame in pixel coordinates.
//! Because they're part of the frame, they appear in recorded videos.

use bytemuck::{Pod, Zeroable};

use crate::Colormap;
use crate::font::GLYPHS;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Item {
    rect: [f32; 4],
    color: [f32; 4],
    kind: u32,
    code: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Screen {
    viewport: [f32; 2],
    _pad: [f32; 2],
    glyphs: [u32; 256],
}

/// Collects text and shapes for a frame, then draws them in one pass.
pub struct Overlay {
    pipeline: wgpu::RenderPipeline,
    screen: wgpu::Buffer,
    group: wgpu::BindGroup,
    items: Vec<Item>,
    buffer: Option<wgpu::Buffer>,
}

/// Width in pixels of `text` at `scale` (pixels per font pixel).
pub fn text_width(text: &str, scale: f32) -> f32 {
    text.chars().count() as f32 * 8.0 * scale
}

impl Overlay {
    pub fn new(device: &wgpu::Device, format: wgpu::TextureFormat) -> Self {
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("overlay"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let screen = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("overlay screen"),
            size: size_of::<Screen>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("overlay"),
            layout: &layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: screen.as_entire_binding(),
            }],
        });
        let source = format!(
            "{}\n{}",
            include_str!("colormaps.wgsl"),
            include_str!("overlay.wgsl")
        );
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("overlay"),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("overlay"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("overlay"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[Some(wgpu::VertexBufferLayout {
                    array_stride: size_of::<Item>() as u64,
                    step_mode: wgpu::VertexStepMode::Instance,
                    attributes: &wgpu::vertex_attr_array![0 => Float32x4, 1 => Float32x4, 2 => Uint32, 3 => Uint32],
                })],
            },
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });
        Self {
            pipeline,
            screen,
            group,
            items: Vec::new(),
            buffer: None,
        }
    }

    /// Text with its top-left corner at (x, y); `scale` is pixels per font
    /// pixel (glyphs are 8 × 8). Characters outside ASCII draw as '?'.
    pub fn text(&mut self, x: f32, y: f32, scale: f32, color: [f32; 4], text: &str) {
        let size = 8.0 * scale;
        for (i, c) in text.chars().enumerate() {
            let code = if c.is_ascii() { c as u32 } else { '?' as u32 };
            if code == ' ' as u32 {
                continue;
            }
            self.items.push(Item {
                rect: [x + i as f32 * size, y, size, size],
                color,
                kind: 0,
                code,
            });
        }
    }

    /// A filled rectangle (use alpha below 1 for a translucent panel).
    pub fn panel(&mut self, x: f32, y: f32, width: f32, height: f32, color: [f32; 4]) {
        self.items.push(Item {
            rect: [x, y, width, height],
            color,
            kind: 1,
            code: 0,
        });
    }

    /// A horizontal colour bar with `label` above it and the range below it.
    #[allow(clippy::too_many_arguments)]
    pub fn color_bar(
        &mut self,
        x: f32,
        y: f32,
        width: f32,
        height: f32,
        scale: f32,
        map: Colormap,
        range: (&str, &str),
        label: &str,
    ) {
        let line = 8.0 * scale;
        let white = [0.92, 0.93, 0.96, 1.0];
        let pad = 0.6 * line;
        let total = line + pad + height + pad + line;
        self.panel(
            x - pad,
            y - pad,
            width + 2.0 * pad,
            total + 2.0 * pad,
            [0.03, 0.035, 0.05, 0.78],
        );
        self.text(x, y, scale, white, label);
        let bar_y = y + line + pad;
        self.items.push(Item {
            rect: [x, bar_y, width, height],
            color: [1.0; 4],
            kind: 2,
            code: map as u32,
        });
        let below = bar_y + height + pad;
        self.text(x, below, scale, white, range.0);
        self.text(
            x + width - text_width(range.1, scale),
            below,
            scale,
            white,
            range.1,
        );
    }

    /// Lines of text on a translucent panel, top-left at (x, y). Returns the
    /// panel's height.
    pub fn lines(&mut self, x: f32, y: f32, scale: f32, lines: &[String]) -> f32 {
        let line = 8.0 * scale;
        let gap = 0.5 * line;
        let pad = 0.8 * line;
        let width = lines
            .iter()
            .map(|l| text_width(l, scale))
            .fold(0.0, f32::max);
        let height = lines.len() as f32 * (line + gap) - gap;
        self.panel(
            x,
            y,
            width + 2.0 * pad,
            height + 2.0 * pad,
            [0.03, 0.035, 0.05, 0.78],
        );
        for (i, l) in lines.iter().enumerate() {
            self.text(
                x + pad,
                y + pad + i as f32 * (line + gap),
                scale,
                [0.92, 0.93, 0.96, 1.0],
                l,
            );
        }
        height + 2.0 * pad
    }

    /// Draws everything collected since the last draw over `view`, then clears.
    pub fn draw(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        size: (u32, u32),
    ) {
        if self.items.is_empty() {
            return;
        }
        let screen = Screen {
            viewport: [size.0 as f32, size.1 as f32],
            _pad: [0.0; 2],
            glyphs: GLYPHS,
        };
        queue.write_buffer(&self.screen, 0, bytemuck::bytes_of(&screen));
        let bytes = (self.items.len() * size_of::<Item>()) as u64;
        if self.buffer.as_ref().is_none_or(|b| b.size() < bytes) {
            self.buffer = Some(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("overlay items"),
                size: bytes.next_power_of_two(),
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
        }
        let buffer = self.buffer.as_ref().expect("just created");
        queue.write_buffer(buffer, 0, bytemuck::cast_slice(&self.items));
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("overlay"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            })],
            ..Default::default()
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.group, &[]);
        pass.set_vertex_buffer(0, buffer.slice(..bytes));
        pass.draw(0..6, 0..self.items.len() as u32);
        drop(pass);
        self.items.clear();
    }
}

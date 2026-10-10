//! Milestone 1: the same binary draws a triangle in the browser and natively.

use virial::{Config, Context, Frame, Gpu, Scene, wgpu};

struct Triangle {
    pipeline: wgpu::RenderPipeline,
}

impl Scene for Triangle {
    fn new(gpu: &Gpu) -> Self {
        let shader = gpu
            .device
            .create_shader_module(wgpu::include_wgsl!("triangle.wgsl"));
        let pipeline = gpu
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("triangle"),
                layout: None,
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    compilation_options: Default::default(),
                    buffers: &[],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_main"),
                    compilation_options: Default::default(),
                    targets: &[Some(gpu.surface_format().into())],
                }),
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            });
        Self { pipeline }
    }

    fn render(&mut self, _ctx: &mut Context<'_>, frame: Frame<'_>) {
        let mut pass = frame
            .encoder
            .begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("triangle"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: frame.view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        // Near-black, slightly blue background
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.0,
                            g: 0.0,
                            b: 0.003,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                ..Default::default()
            });
        pass.set_pipeline(&self.pipeline);
        pass.draw(0..3, 0..1);
    }
}

fn main() {
    virial::run::<Triangle>(Config {
        title: "Virial: triangle",
    });
}

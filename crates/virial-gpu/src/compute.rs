//! A GPU without a window: for tests, benchmarks and offline rendering.

use crate::GpuError;

/// Device and queue with no surface.
pub struct Compute {
    pub adapter: wgpu::Adapter,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
}

impl Compute {
    /// Opens the best GPU available, with the same limits as a windowed [`crate::Gpu`].
    pub async fn new() -> Result<Self, GpuError> {
        let instance = crate::instance();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
                apply_limit_buckets: false,
            })
            .await
            .map_err(GpuError::NoAdapter)?;
        let (device, queue) = crate::request_device(&adapter).await?;
        Ok(Self {
            adapter,
            device,
            queue,
        })
    }

    /// Like [`Compute::new`], but refuses software adapters (such as WARP on
    /// Windows CI machines): tests that check GPU behaviour skip there rather
    /// than run for minutes on an emulator.
    pub async fn hardware() -> Result<Self, GpuError> {
        let gpu = Self::new().await?;
        if gpu.adapter.get_info().device_type == wgpu::DeviceType::Cpu {
            return Err(GpuError::SoftwareOnly(gpu.adapter.get_info().name));
        }
        Ok(gpu)
    }

    /// Copies `buffer` (which needs `COPY_SRC`) back to the CPU, waiting for
    /// the GPU. Native only: the browser can't block.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn read_back(&self, buffer: &wgpu::Buffer) -> Vec<u8> {
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("read back"),
            size: buffer.size(),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, buffer.size());
        self.queue.submit([encoder.finish()]);
        staging.slice(..).map_async(wgpu::MapMode::Read, |result| {
            result.expect("could not read the buffer back")
        });
        self.device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            })
            .expect("the GPU did not finish");
        let bytes = staging
            .slice(..)
            .get_mapped_range()
            .expect("the buffer was just mapped")
            .to_vec();
        staging.unmap();
        bytes
    }

    /// Waits until all submitted GPU work has finished. Native only.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn wait(&self) {
        self.device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            })
            .expect("the GPU did not finish");
    }
}

//! GPU device and surface setup for Virial.
//!
//! Wraps `wgpu` so the rest of the engine gets one [`Gpu`] holding the device,
//! queue and the surface the frame is drawn to. The same code runs on WebGPU in
//! the browser and on Metal, Vulkan or DX12 natively.

use std::fmt;

pub use wgpu;

mod timer;
pub use timer::GpuTimer;

/// Limits the engine's kernels are written against. These are WebGPU's
/// defaults, so any browser that supports WebGPU can run them.
pub fn required_limits() -> wgpu::Limits {
    wgpu::Limits::default()
}

/// Why the GPU could not be set up. Shown to the player, so each message says
/// what to do about it.
#[derive(Debug)]
pub enum GpuError {
    Surface(wgpu::CreateSurfaceError),
    NoAdapter(wgpu::RequestAdapterError),
    LimitsTooLow(String),
    Device(wgpu::RequestDeviceError),
}

impl fmt::Display for GpuError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GpuError::Surface(e) => write!(f, "Could not draw to the page or window: {e}"),
            GpuError::NoAdapter(e) => write!(
                f,
                "No WebGPU-capable GPU was found ({e}). Use a recent Chrome, Edge or Safari, and check that hardware acceleration is on."
            ),
            GpuError::LimitsTooLow(which) => write!(
                f,
                "This GPU is below WebGPU's default limits ({which}). Try another browser or device."
            ),
            GpuError::Device(e) => write!(f, "The GPU refused to start: {e}"),
        }
    }
}

impl std::error::Error for GpuError {}

/// The device, queue and window surface.
pub struct Gpu {
    pub adapter: wgpu::Adapter,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub surface: wgpu::Surface<'static>,
    pub config: wgpu::SurfaceConfiguration,
}

impl Gpu {
    /// Creates the device and a surface for `target` (a winit window).
    /// `width` and `height` are the target's size in physical pixels.
    pub async fn new(
        target: impl Into<wgpu::SurfaceTarget<'static>>,
        width: u32,
        height: u32,
    ) -> Result<Self, GpuError> {
        let mut descriptor = wgpu::InstanceDescriptor::new_without_display_handle_from_env();
        descriptor.backends = wgpu::Backends::PRIMARY | wgpu::Backends::BROWSER_WEBGPU;
        let instance = wgpu::Instance::new(descriptor);
        let surface = instance.create_surface(target).map_err(GpuError::Surface)?;
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
                apply_limit_buckets: false,
            })
            .await
            .map_err(GpuError::NoAdapter)?;

        let available = adapter.limits();
        let mut too_low = Vec::new();
        required_limits().check_limits_with_fail_fn(&available, false, |name, want, have| {
            too_low.push(format!("{name}: has {have}, needs {want}"));
        });
        if !too_low.is_empty() {
            return Err(GpuError::LimitsTooLow(too_low.join(", ")));
        }
        // Kernels are written for the defaults, but scene size is memory, so
        // take all the buffer memory the adapter offers
        let required = wgpu::Limits {
            max_buffer_size: available.max_buffer_size,
            max_storage_buffer_binding_size: available.max_storage_buffer_binding_size,
            ..required_limits()
        };

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("virial"),
                required_features: wgpu::Features::TIMESTAMP_QUERY & adapter.features(),
                required_limits: required,
                ..Default::default()
            })
            .await
            .map_err(GpuError::Device)?;

        let info = adapter.get_info();
        log::info!(
            "GPU: {} ({:?}, {:?})",
            info.name,
            info.backend,
            info.device_type
        );

        let mut config = surface
            .get_default_config(&adapter, width.max(1), height.max(1))
            .expect("the adapter was chosen to be compatible with this surface");
        config.present_mode = wgpu::PresentMode::AutoVsync;
        surface.configure(&device, &config);

        Ok(Self {
            adapter,
            device,
            queue,
            surface,
            config,
        })
    }

    /// Call when the window changes size (physical pixels).
    pub fn resize(&mut self, width: u32, height: u32) {
        if width == 0 || height == 0 {
            return;
        }
        self.config.width = width;
        self.config.height = height;
        self.surface.configure(&self.device, &self.config);
    }

    pub fn surface_format(&self) -> wgpu::TextureFormat {
        self.config.format
    }

    /// Width divided by height of the surface.
    pub fn aspect(&self) -> f32 {
        self.config.width as f32 / self.config.height.max(1) as f32
    }

    /// Whether GPU pass timings are available on this device.
    pub fn has_timestamps(&self) -> bool {
        self.device
            .features()
            .contains(wgpu::Features::TIMESTAMP_QUERY)
    }
}

//! Offline rendering: a scene drawn frame by frame on a fixed video clock,
//! with no window. Each frame takes as long as it takes, so accuracy never
//! competes with frame rate.
//!
//! Output in the chosen directory:
//! - `video.mp4`: H.264 via `ffmpeg` (unless `video=0`)
//! - `frames.csv`: a 64-bit hash of every frame's pixels, to check that a
//!   rerun is identical
//! - `measurements.csv`, `*.npy`: whatever the scene logs through `ctx.data`
//! - `run.json`: settings, GPU, Virial version and git commit

use std::io::Write;
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::time::Instant;

use crate::data::DataLog;
use crate::manifest::{Manifest, fnv1a};
use crate::{Context, Frame, Gpu, Input, Scene, Stats, Time, arg, args, wgpu};

/// What to record.
#[derive(Clone, Debug)]
pub struct RecordSettings {
    pub dir: PathBuf,
    pub width: u32,
    pub height: u32,
    pub fps: u32,
    pub frames: u32,
    /// Encode `video.mp4` with ffmpeg. Off, only frame hashes and data are written.
    pub video: bool,
}

impl RecordSettings {
    /// From the settings `record=<dir> width= height= fps= seconds= video=`.
    pub fn from_args(dir: &str) -> Self {
        let fps = arg("fps", 60u32).max(1);
        Self {
            dir: PathBuf::from(dir),
            width: arg("width", 1920u32).max(16),
            height: arg("height", 1080u32).max(16),
            fps,
            frames: (arg("seconds", 10.0f64) * fps as f64).round().max(1.0) as u32,
            video: arg("video", 1u32) != 0,
        }
    }
}

/// Whether a hardware GPU is available (tests use this to skip on machines
/// with none, or only a software adapter).
pub fn gpu_available() -> bool {
    pollster::block_on(virial_gpu::Compute::hardware()).is_ok()
}

/// What a recording produced.
pub struct Recording {
    /// FNV-1a hash of each frame's RGBA pixels
    pub frame_hashes: Vec<u64>,
    pub seconds: f64,
}

fn git_commit() -> Option<String> {
    let out = Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output()
        .ok()?;
    out.status
        .success()
        .then(|| String::from_utf8_lossy(&out.stdout).trim().to_string())
}

/// Whether the working tree has uncommitted changes, so the commit alone
/// doesn't describe the code that ran
fn git_dirty() -> Option<bool> {
    let out = Command::new("git")
        .args(["status", "--porcelain"])
        .output()
        .ok()?;
    out.status.success().then_some(!out.stdout.is_empty())
}

fn start_ffmpeg(settings: &RecordSettings) -> Child {
    let size = format!("{}x{}", settings.width, settings.height);
    let fps = settings.fps.to_string();
    let output = settings.dir.join("video.mp4");
    Command::new("ffmpeg")
        .args([
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgba",
            "-s",
            &size,
            "-r",
            &fps,
            "-i",
            "-",
        ])
        // One thread and fixed settings: the same frames give the same file
        .args([
            "-c:v", "libx264", "-preset", "medium", "-crf", "16", "-pix_fmt", "yuv420p",
            "-threads", "1",
        ])
        .args(["-movflags", "+faststart"])
        .arg(&output)
        .stdin(Stdio::piped())
        .spawn()
        .unwrap_or_else(|e| {
            panic!("could not start ffmpeg ({e}); install it, or record with video=0")
        })
}

/// Renders `settings.frames` frames of `S` offline.
pub fn record<S: Scene>(title: &str, settings: &RecordSettings) -> Recording {
    std::fs::create_dir_all(&settings.dir)
        .unwrap_or_else(|e| panic!("could not create {}: {e}", settings.dir.display()));
    let started = Instant::now();
    // Read before rendering: a commit made during a long render must not be
    // recorded as the code that ran
    let (commit, dirty) = (git_commit(), git_dirty());
    let gpu = pollster::block_on(Gpu::headless(settings.width, settings.height))
        .unwrap_or_else(|e| panic!("{e}"));
    let info = gpu.adapter.get_info();

    let mut scene = S::new(&gpu);
    let mut input = Input::default();
    input.set_surface_size(settings.width, settings.height);
    let mut stats = Stats::new("record");
    let mut data = DataLog::recording(settings.dir.clone());

    let target = gpu.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("frame"),
        size: wgpu::Extent3d {
            width: settings.width,
            height: settings.height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: gpu.config.format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = target.create_view(&Default::default());
    let row = settings.width * 4;
    let padded_row =
        row.div_ceil(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT) * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    let readback = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("frame readback"),
        size: (padded_row * settings.height) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let mut ffmpeg = settings.video.then(|| start_ffmpeg(settings));
    let mut hashes = Vec::with_capacity(settings.frames as usize);
    let mut pixels = vec![0u8; (row * settings.height) as usize];
    let mut last_report = Instant::now();
    let dt = 1.0 / settings.fps as f32;

    for frame in 0..settings.frames {
        let time = Time {
            dt,
            elapsed: frame as f64 / settings.fps as f64,
            frame: frame as u64,
        };
        let mut ctx = Context {
            gpu: &gpu,
            input: &input,
            time,
            stats: &mut stats,
            data: &mut data,
        };
        scene.update(&mut ctx);
        let mut encoder = gpu.device.create_command_encoder(&Default::default());
        scene.render(
            &mut ctx,
            Frame {
                encoder: &mut encoder,
                view: &view,
                timer: None,
            },
        );
        encoder.copy_texture_to_buffer(
            target.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &readback,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded_row),
                    rows_per_image: Some(settings.height),
                },
            },
            target.size(),
        );
        gpu.queue.submit([encoder.finish()]);
        readback
            .slice(..)
            .map_async(wgpu::MapMode::Read, |r| r.expect("frame read back"));
        gpu.device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            })
            .expect("the GPU did not finish");
        {
            let mapped = readback.slice(..).get_mapped_range().expect("mapped");
            for (y, chunk) in pixels.chunks_exact_mut(row as usize).enumerate() {
                let start = y * padded_row as usize;
                chunk.copy_from_slice(&mapped[start..start + row as usize]);
            }
        }
        readback.unmap();
        hashes.push(fnv1a(&pixels));
        if let Some(child) = &mut ffmpeg {
            child
                .stdin
                .as_mut()
                .expect("ffmpeg stdin")
                .write_all(&pixels)
                .expect("ffmpeg stopped accepting frames");
        }
        input.end_frame();

        if last_report.elapsed().as_secs_f32() >= 2.0 || frame + 1 == settings.frames {
            last_report = Instant::now();
            log::info!(
                "frame {}/{} ({:.0}%), {:.1} s elapsed",
                frame + 1,
                settings.frames,
                (frame + 1) as f64 * 100.0 / settings.frames as f64,
                started.elapsed().as_secs_f64()
            );
        }
    }

    let time = Time {
        dt,
        elapsed: settings.frames as f64 / settings.fps as f64,
        frame: settings.frames as u64,
    };
    let mut ctx = Context {
        gpu: &gpu,
        input: &input,
        time,
        stats: &mut stats,
        data: &mut data,
    };
    scene.finish(&mut ctx);

    if let Some(mut child) = ffmpeg {
        drop(child.stdin.take());
        let status = child.wait().expect("ffmpeg did not exit");
        assert!(status.success(), "ffmpeg failed: {status}");
    }

    let write = |name: &str, text: String| {
        let path = settings.dir.join(name);
        std::fs::write(&path, text)
            .unwrap_or_else(|e| panic!("could not write {}: {e}", path.display()));
    };
    let mut frames_csv = String::from("frame,hash\n");
    for (i, h) in hashes.iter().enumerate() {
        frames_csv += &format!("{i},{h:016x}\n");
    }
    write("frames.csv", frames_csv);
    write("measurements.csv", data.csv());

    let seconds = started.elapsed().as_secs_f64();
    let manifest = Manifest {
        title,
        commit,
        uncommitted_changes: dirty,
        gpu: format!("{} ({:?})", info.name, info.backend),
        width: settings.width,
        height: settings.height,
        fps: settings.fps,
        frames: settings.frames,
        seconds,
        settings: args().into_iter().collect(),
    };
    let json = manifest.json(&data);
    write("run.json", json);

    log::info!(
        "recorded {} frames to {} in {seconds:.1} s",
        settings.frames,
        settings.dir.display()
    );
    Recording {
        frame_hashes: hashes,
        seconds,
    }
}

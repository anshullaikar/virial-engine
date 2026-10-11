//! Offline rendering in the browser, on the same fixed video clock as the
//! native recorder: each frame is rendered, read back, hashed and encoded
//! with WebCodecs (VP9), however long that takes. The page then shows
//! links to download `video.webm`, `frames.csv`, `measurements.csv` and
//! `run.json`, and leaves them on `window.virialRecording` for scripts.
//!
//! Start it with the page settings `?record=1&seconds=10` (plus `width`,
//! `height`, `fps` and the scene's own settings). Keep the tab in front:
//! browsers throttle background tabs.

use std::cell::{Cell, RefCell};
use std::future::Future;
use std::pin::Pin;
use std::rc::Rc;
use std::task::{Context as TaskContext, Poll, Waker};

use wasm_bindgen::JsCast;
use wasm_bindgen::prelude::*;
use web_sys::{
    EncodedVideoChunk, EncodedVideoChunkType, LatencyMode, VideoEncoder, VideoEncoderConfig,
    VideoEncoderEncodeOptions, VideoEncoderInit, VideoFrame, VideoFrameBufferInit,
    VideoPixelFormat,
};

use crate::data::DataLog;
use crate::manifest::{Manifest, fnv1a};
use crate::webm::WebmWriter;
use crate::{Context, Frame, Gpu, Input, Scene, Stats, Time, arg, args, wgpu};

/// Starts recording `S` instead of running it live.
pub(crate) fn start<S: Scene>(title: &'static str) {
    wasm_bindgen_futures::spawn_local(async move {
        if let Err(error) = record::<S>(title).await {
            status(&format!("Recording failed: {error:?}"));
            log::error!("recording failed: {error:?}");
        }
    });
}

/// Shows `text` in a status line on the page.
fn status(text: &str) {
    let Some(document) = web_sys::window().and_then(|w| w.document()) else {
        return;
    };
    let element = match document.get_element_by_id("virial-record-status") {
        Some(element) => element,
        None => {
            let Ok(element) = document.create_element("p") else {
                return;
            };
            element.set_id("virial-record-status");
            let _ = element.set_attribute(
                "style",
                "font: 14px system-ui, sans-serif; color: #e9ebf3; margin: 16px",
            );
            if let Some(body) = document.body() {
                let _ = body.append_child(&element);
            }
            element
        }
    };
    element.set_text_content(Some(text));
}

/// A future that finishes when the callback it hands out is called (for
/// `map_async`, whose result arrives through the browser's event loop).
#[derive(Default)]
struct Signal {
    done: Cell<bool>,
    waker: RefCell<Option<Waker>>,
}

impl Signal {
    fn fire(&self) {
        self.done.set(true);
        if let Some(waker) = self.waker.borrow_mut().take() {
            waker.wake();
        }
    }
}

struct Wait(Rc<Signal>);

impl Future for Wait {
    type Output = ();
    fn poll(self: Pin<&mut Self>, cx: &mut TaskContext<'_>) -> Poll<()> {
        if self.0.done.get() {
            Poll::Ready(())
        } else {
            *self.0.waker.borrow_mut() = Some(cx.waker().clone());
            Poll::Pending
        }
    }
}

/// Yields to the browser for `ms` milliseconds.
async fn sleep(ms: i32) {
    let promise = js_sys::Promise::new(&mut |resolve, _| {
        if let Some(window) = web_sys::window() {
            let _ = window.set_timeout_with_callback_and_timeout_and_arguments_0(&resolve, ms);
        }
    });
    let _ = wasm_bindgen_futures::JsFuture::from(promise).await;
}

/// A link on the page to download `bytes` as `name`.
fn download(name: &str, bytes: &[u8], mime: &str) -> Result<(), JsValue> {
    let array = js_sys::Uint8Array::from(bytes);
    let parts = js_sys::Array::of1(&array);
    let options = web_sys::BlobPropertyBag::new();
    options.set_type(mime);
    let blob = web_sys::Blob::new_with_u8_array_sequence_and_options(&parts, &options)?;
    let url = web_sys::Url::create_object_url_with_blob(&blob)?;
    let document = web_sys::window()
        .and_then(|w| w.document())
        .ok_or("no document")?;
    let link: web_sys::HtmlAnchorElement = document.create_element("a")?.dyn_into()?;
    link.set_href(&url);
    link.set_download(name);
    link.set_text_content(Some(name));
    let _ = link.set_attribute(
        "style",
        "display: block; font: 14px system-ui, sans-serif; color: #34d399; margin: 4px 16px",
    );
    if let Some(body) = document.body() {
        body.append_child(&link)?;
    }
    Ok(())
}

async fn record<S: Scene>(title: &str) -> Result<(), JsValue> {
    let width = arg("width", 1920u32).max(16) & !1;
    let height = arg("height", 1080u32).max(16) & !1;
    let fps = arg("fps", 60u32).max(1);
    let frames = (arg("seconds", 10.0f64) * fps as f64).round().max(1.0) as u32;
    let started = web_time::Instant::now();
    status(&format!("Recording {frames} frames at {width} x {height}…"));

    let gpu = Gpu::headless(width, height)
        .await
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    let info = gpu.adapter.get_info();
    // The browser reports invalid shaders and pipelines only to an error
    // callback, and drops the work: catch that rather than record zeros
    let scope = gpu.device.push_error_scope(wgpu::ErrorFilter::Validation);
    let mut scene = S::new(&gpu);
    let mut input = Input::default();
    input.set_surface_size(width, height);
    let mut stats = Stats::new("record");
    let mut data = DataLog::recording_in_memory();

    let target = gpu.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("frame"),
        size: wgpu::Extent3d {
            width,
            height,
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
    let row = width * 4;
    let padded_row =
        row.div_ceil(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT) * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    let readback = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("frame readback"),
        size: (padded_row * height) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    // The surface format may be BGRA; WebCodecs takes either
    let pixel_format = match gpu.config.format {
        wgpu::TextureFormat::Bgra8Unorm | wgpu::TextureFormat::Bgra8UnormSrgb => {
            VideoPixelFormat::Bgra
        }
        _ => VideoPixelFormat::Rgba,
    };

    let writer = Rc::new(RefCell::new(WebmWriter::new(width, height, fps)));
    let failure = Rc::new(RefCell::new(None::<String>));
    let output = {
        let writer = writer.clone();
        Closure::<dyn FnMut(EncodedVideoChunk, JsValue)>::new(
            move |chunk: EncodedVideoChunk, _: JsValue| {
                let mut bytes = vec![0u8; chunk.byte_length() as usize];
                if chunk.copy_to_with_u8_slice(&mut bytes).is_ok() {
                    writer
                        .borrow_mut()
                        .push(chunk.type_() == EncodedVideoChunkType::Key, bytes);
                }
            },
        )
    };
    let error = {
        let failure = failure.clone();
        Closure::<dyn FnMut(JsValue)>::new(move |e: JsValue| {
            *failure.borrow_mut() = Some(format!("{e:?}"));
        })
    };
    let encoder = VideoEncoder::new(&VideoEncoderInit::new(
        error.as_ref().unchecked_ref(),
        output.as_ref().unchecked_ref(),
    ))?;
    let config = VideoEncoderConfig::new("vp09.00.41.08", height, width);
    config.set_framerate(fps as f64);
    // About 0.2 bits per pixel per frame: crisp for particles at 1080p60
    config.set_bitrate_f64(0.2 * (width * height * fps) as f64);
    config.set_latency_mode(LatencyMode::Quality);
    encoder.configure(&config)?;

    let mut first_scope = Some(scope);
    let mut hashes = Vec::with_capacity(frames as usize);
    let mut pixels = vec![0u8; (row * height) as usize];
    let dt = 1.0 / fps as f32;
    let mut last_report = web_time::Instant::now();
    for frame in 0..frames {
        let time = Time {
            dt,
            elapsed: frame as f64 / fps as f64,
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
        let mut commands = gpu.device.create_command_encoder(&Default::default());
        scene.render(
            &mut ctx,
            Frame {
                encoder: &mut commands,
                view: &view,
                timer: None,
            },
        );
        commands.copy_texture_to_buffer(
            target.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &readback,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded_row),
                    rows_per_image: Some(height),
                },
            },
            target.size(),
        );
        gpu.queue.submit([commands.finish()]);
        if frame == 0 {
            // Scene setup and the first frame were all valid
            if let Some(error) = first_scope
                .take()
                .expect("set before the first frame")
                .pop()
                .await
            {
                return Err(JsValue::from_str(&format!("GPU validation error: {error}")));
            }
        }
        let signal = Rc::new(Signal::default());
        {
            let signal = signal.clone();
            readback
                .slice(..)
                .map_async(wgpu::MapMode::Read, move |_| signal.fire());
        }
        Wait(signal).await;
        {
            let mapped = readback.slice(..).get_mapped_range().expect("mapped");
            for (y, chunk) in pixels.chunks_exact_mut(row as usize).enumerate() {
                let start = y * padded_row as usize;
                chunk.copy_from_slice(&mapped[start..start + row as usize]);
            }
        }
        readback.unmap();
        hashes.push(fnv1a(&pixels));

        let init = VideoFrameBufferInit::new(
            height,
            width,
            pixel_format,
            // Microseconds, as i32 in this binding: up to about 35 minutes
            (frame as u64 * 1_000_000 / fps as u64) as i32,
        );
        let video_frame =
            VideoFrame::new_with_u8_slice_and_video_frame_buffer_init(&mut pixels, &init)?;
        let options = VideoEncoderEncodeOptions::new();
        // A keyframe every two seconds, so the file seeks well
        options.set_key_frame(frame % (2 * fps) == 0);
        encoder.encode_with_options(&video_frame, &options)?;
        video_frame.close();
        while encoder.encode_queue_size() > 4 {
            sleep(1).await;
        }
        if let Some(e) = failure.borrow().clone() {
            return Err(JsValue::from_str(&format!("video encoder: {e}")));
        }
        input.end_frame();
        if last_report.elapsed().as_secs_f32() >= 0.5 || frame + 1 == frames {
            last_report = web_time::Instant::now();
            status(&format!(
                "Recording: frame {} of {frames} ({:.0}%), {:.0} s so far. Keep this tab in front.",
                frame + 1,
                (frame + 1) as f64 * 100.0 / frames as f64,
                started.elapsed().as_secs_f64()
            ));
        }
    }
    let time = Time {
        dt,
        elapsed: frames as f64 / fps as f64,
        frame: frames as u64,
    };
    let mut ctx = Context {
        gpu: &gpu,
        input: &input,
        time,
        stats: &mut stats,
        data: &mut data,
    };
    scene.finish(&mut ctx);
    wasm_bindgen_futures::JsFuture::from(encoder.flush()).await?;
    encoder.close()?;
    if let Some(e) = failure.borrow().clone() {
        return Err(JsValue::from_str(&format!("video encoder: {e}")));
    }

    let seconds = started.elapsed().as_secs_f64();
    let video = writer.borrow().finish();
    let mut frames_csv = String::from("frame,hash\n");
    for (i, h) in hashes.iter().enumerate() {
        frames_csv += &format!("{i},{h:016x}\n");
    }
    let manifest = Manifest {
        title,
        commit: option_env!("VIRIAL_GIT_COMMIT").map(str::to_string),
        uncommitted_changes: option_env!("VIRIAL_GIT_DIRTY").map(|d| d == "1"),
        gpu: format!("{} ({:?}), in {}", info.name, info.backend, user_agent()),
        width,
        height,
        fps,
        frames,
        seconds,
        settings: args().into_iter().collect(),
    };
    // For scripts: the recorded files as strings and bytes
    let result = js_sys::Object::new();
    let set = |key: &str, value: &JsValue| js_sys::Reflect::set(&result, &key.into(), value);
    set("video", &js_sys::Uint8Array::from(video.as_slice()).into())?;
    set("frames", &frames_csv.as_str().into())?;
    set("measurements", &data.csv().into())?;
    set("run", &manifest.json(&data).into())?;
    if let Some(window) = web_sys::window() {
        js_sys::Reflect::set(&window, &"virialRecording".into(), &result)?;
    }
    download("video.webm", &video, "video/webm")?;
    download("frames.csv", frames_csv.as_bytes(), "text/csv")?;
    download("measurements.csv", data.csv().as_bytes(), "text/csv")?;
    download(
        "run.json",
        manifest.json(&data).as_bytes(),
        "application/json",
    )?;
    status(&format!(
        "Recorded {frames} frames in {seconds:.0} s ({} encoded frames, {:.1} MB). Downloads below.",
        writer.borrow().frames(),
        video.len() as f64 / 1e6
    ));
    drop((output, error));
    Ok(())
}

fn user_agent() -> String {
    web_sys::window()
        .and_then(|w| w.navigator().user_agent().ok())
        .unwrap_or_else(|| "a browser".into())
}

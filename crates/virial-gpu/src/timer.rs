//! GPU pass timings from timestamp queries.
//!
//! Each frame, passes ask for timestamp writes by label. At the end of the
//! frame the timestamps are resolved and copied into one of a few staging
//! buffers, read back a frame or two later without ever waiting on the GPU.

use std::cell::{Cell, RefCell};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

/// Most timed passes per frame.
const MAX_PASSES: u32 = 16;
/// Staging buffers in flight. With three, the CPU never waits for a readback.
const RING: usize = 3;
const BYTES: u64 = MAX_PASSES as u64 * 2 * 8;

struct Staging {
    buffer: wgpu::Buffer,
    labels: Vec<&'static str>,
    /// Set by the map callback once the results can be read.
    ready: Arc<AtomicBool>,
    in_flight: bool,
}

/// Times GPU passes. Created by the engine when the device supports
/// timestamp queries; scenes reach it through `Frame::timer`.
pub struct GpuTimer {
    query_set: wgpu::QuerySet,
    resolve: wgpu::Buffer,
    ring: Vec<Staging>,
    /// Staging buffer this frame writes into, if one was free
    current: Cell<Option<usize>>,
    labels: RefCell<Vec<&'static str>>,
    period_ns: f32,
    results: Vec<(&'static str, f32)>,
}

impl GpuTimer {
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Self {
        let query_set = device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("pass timings"),
            ty: wgpu::QueryType::Timestamp,
            count: MAX_PASSES * 2,
        });
        let resolve = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("pass timings resolve"),
            size: BYTES,
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let ring = (0..RING)
            .map(|_| Staging {
                buffer: device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("pass timings readback"),
                    size: BYTES,
                    usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }),
                labels: Vec::new(),
                ready: Arc::new(AtomicBool::new(false)),
                in_flight: false,
            })
            .collect();
        Self {
            query_set,
            resolve,
            ring,
            current: Cell::new(None),
            labels: RefCell::new(Vec::new()),
            period_ns: queue.get_timestamp_period(),
            results: Vec::new(),
        }
    }

    /// Starts a frame: collects any finished readbacks and picks a free
    /// staging buffer. Timing is skipped for a frame when none is free.
    pub fn begin_frame(&mut self) {
        for staging in &mut self.ring {
            if staging.in_flight && staging.ready.load(Ordering::Acquire) {
                let ticks: Vec<u64> = {
                    let view = staging
                        .buffer
                        .slice(..)
                        .get_mapped_range()
                        .expect("the map callback reported success");
                    view.as_chunks::<8>()
                        .0
                        .iter()
                        .map(|b| u64::from_le_bytes(*b))
                        .collect()
                };
                staging.buffer.unmap();
                self.results = staging
                    .labels
                    .iter()
                    .enumerate()
                    .map(|(i, label)| {
                        let elapsed = ticks[2 * i + 1].saturating_sub(ticks[2 * i]);
                        (*label, elapsed as f32 * self.period_ns / 1e6)
                    })
                    .collect();
                staging.in_flight = false;
                staging.ready.store(false, Ordering::Release);
            }
        }
        self.current
            .set(self.ring.iter().position(|s| !s.in_flight));
        self.labels.borrow_mut().clear();
    }

    fn slot(&self, label: &'static str) -> Option<u32> {
        self.current.get()?;
        let mut labels = self.labels.borrow_mut();
        let index = labels.len() as u32;
        if index >= MAX_PASSES {
            return None;
        }
        labels.push(label);
        Some(index * 2)
    }

    /// Timestamp writes for a compute pass, or `None` when timing is off this frame.
    pub fn compute(&self, label: &'static str) -> Option<wgpu::ComputePassTimestampWrites<'_>> {
        let first = self.slot(label)?;
        Some(wgpu::ComputePassTimestampWrites {
            query_set: &self.query_set,
            beginning_of_pass_write_index: Some(first),
            end_of_pass_write_index: Some(first + 1),
        })
    }

    /// Timestamp writes for a render pass, or `None` when timing is off this frame.
    pub fn render(&self, label: &'static str) -> Option<wgpu::RenderPassTimestampWrites<'_>> {
        let first = self.slot(label)?;
        Some(wgpu::RenderPassTimestampWrites {
            query_set: &self.query_set,
            beginning_of_pass_write_index: Some(first),
            end_of_pass_write_index: Some(first + 1),
        })
    }

    /// Records the resolve and copy at the end of the frame's commands.
    pub fn resolve(&mut self, encoder: &mut wgpu::CommandEncoder) {
        let Some(index) = self.current.get() else {
            return;
        };
        let labels = std::mem::take(&mut *self.labels.borrow_mut());
        if labels.is_empty() {
            return;
        }
        let count = labels.len() as u32 * 2;
        encoder.resolve_query_set(&self.query_set, 0..count, &self.resolve, 0);
        encoder.copy_buffer_to_buffer(&self.resolve, 0, &self.ring[index].buffer, 0, BYTES);
        self.ring[index].labels = labels;
    }

    /// Call after the frame's commands are submitted.
    pub fn after_submit(&mut self) {
        let Some(index) = self.current.take() else {
            return;
        };
        let staging = &mut self.ring[index];
        if staging.labels.is_empty() {
            return;
        }
        staging.in_flight = true;
        let ready = staging.ready.clone();
        staging
            .buffer
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| {
                if result.is_ok() {
                    ready.store(true, Ordering::Release);
                }
            });
    }

    /// The most recent timings: (label, milliseconds) per timed pass.
    pub fn results(&self) -> &[(&'static str, f32)] {
        &self.results
    }
}

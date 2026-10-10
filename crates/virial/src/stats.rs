//! Frame rate, GPU pass timings and scene readouts. Shown in a panel in the
//! browser and in the window title natively.

use web_time::Instant;
use winit::window::Window;

pub struct Stats {
    #[cfg_attr(target_arch = "wasm32", allow(dead_code))]
    title: &'static str,
    fps: f32,
    frame_ms: f32,
    gpu: Vec<(&'static str, f32)>,
    lines: Vec<(String, String)>,
    last_shown: Option<Instant>,
    last_logged: Option<Instant>,
}

impl Stats {
    pub(crate) fn new(title: &'static str) -> Self {
        Self {
            title,
            fps: 0.0,
            frame_ms: 0.0,
            gpu: Vec::new(),
            lines: Vec::new(),
            last_shown: None,
            last_logged: None,
        }
    }

    /// Sets a readout shown under the frame rate, such as `("Energy drift", "3e-7")`.
    /// Setting the same label again replaces it.
    pub fn set(&mut self, label: &str, value: impl Into<String>) {
        let value = value.into();
        match self.lines.iter_mut().find(|(l, _)| l == label) {
            Some(line) => line.1 = value,
            None => self.lines.push((label.to_string(), value)),
        }
    }

    /// Frames per second, smoothed.
    pub fn fps(&self) -> f32 {
        self.fps
    }

    /// The latest GPU time of a pass, in milliseconds.
    pub fn gpu_ms(&self, label: &str) -> Option<f32> {
        self.gpu
            .iter()
            .find(|(l, _)| *l == label)
            .map(|(_, ms)| *ms)
    }

    pub(crate) fn frame(&mut self, dt: f32) {
        if dt <= 0.0 {
            return;
        }
        let ms = dt * 1000.0;
        // Smooth over roughly half a second
        let k = (dt * 2.0).min(1.0);
        self.frame_ms = if self.frame_ms == 0.0 {
            ms
        } else {
            self.frame_ms + (ms - self.frame_ms) * k
        };
        self.fps = 1000.0 / self.frame_ms;
    }

    pub(crate) fn set_gpu_timings(&mut self, results: &[(&'static str, f32)]) {
        for &(label, ms) in results {
            match self.gpu.iter_mut().find(|(l, _)| *l == label) {
                Some(entry) => entry.1 += (ms - entry.1) * 0.1,
                None => self.gpu.push((label, ms)),
            }
        }
    }

    fn text(&self) -> String {
        let mut out = format!("{:.0} fps  {:.1} ms/frame\n", self.fps, self.frame_ms);
        for (label, ms) in &self.gpu {
            out += &format!("GPU {label}: {ms:.2} ms\n");
        }
        for (label, value) in &self.lines {
            out += &format!("{label}: {value}\n");
        }
        out
    }

    /// Updates the display at most four times a second.
    pub(crate) fn show(&mut self, window: &Window) {
        let now = Instant::now();
        if self
            .last_shown
            .is_some_and(|t| now.duration_since(t).as_secs_f32() < 0.25)
        {
            return;
        }
        self.last_shown = Some(now);

        #[cfg(not(target_arch = "wasm32"))]
        {
            let mut title = format!("{}  ·  {:.0} fps", self.title, self.fps);
            for (label, ms) in &self.gpu {
                title += &format!("  ·  {label} {ms:.2} ms");
            }
            window.set_title(&title);
            if self
                .last_logged
                .is_none_or(|t| now.duration_since(t).as_secs_f32() >= 5.0)
            {
                self.last_logged = Some(now);
                log::info!("\n{}", self.text().trim_end());
            }
        }
        #[cfg(target_arch = "wasm32")]
        {
            let _ = (window, &self.last_logged);
            show_in_page(&self.text());
        }
    }
}

/// Writes the stats into `<pre id="virial-stats">`, creating it if needed.
#[cfg(target_arch = "wasm32")]
fn show_in_page(text: &str) {
    let Some(document) = web_sys::window().and_then(|w| w.document()) else {
        return;
    };
    let panel = match document.get_element_by_id("virial-stats") {
        Some(panel) => panel,
        None => {
            let Ok(panel) = document.create_element("pre") else {
                return;
            };
            panel.set_id("virial-stats");
            if let Some(body) = document.body() {
                let _ = body.append_child(&panel);
            }
            panel
        }
    };
    panel.set_text_content(Some(text));
}

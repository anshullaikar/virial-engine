//! What a recording writes about itself (`run.json`), shared by the native
//! and browser recorders.

use crate::data::{DataLog, json_string};

/// FNV-1a hash of a frame's pixels, for checking that reruns are identical
pub fn fnv1a(bytes: &[u8]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x0100_0000_01b3);
    }
    h
}

/// The facts `run.json` records.
pub struct Manifest<'a> {
    pub title: &'a str,
    pub commit: Option<String>,
    pub uncommitted_changes: Option<bool>,
    pub gpu: String,
    pub width: u32,
    pub height: u32,
    pub fps: u32,
    pub frames: u32,
    pub seconds: f64,
    pub settings: Vec<(String, String)>,
}

impl Manifest<'_> {
    /// `run.json`, with the scene's own facts from `data`
    pub fn json(&self, data: &DataLog) -> String {
        let mut settings = self.settings.clone();
        settings.sort();
        let mut json = String::from("{\n");
        let mut field = |k: &str, v: String| json += &format!("  {}: {v},\n", json_string(k));
        field("scene", json_string(self.title));
        field("virial_version", json_string(env!("CARGO_PKG_VERSION")));
        field(
            "git_commit",
            self.commit.as_deref().map_or("null".into(), json_string),
        );
        field(
            "git_uncommitted_changes",
            self.uncommitted_changes
                .map_or("null".into(), |d| d.to_string()),
        );
        field("gpu", json_string(&self.gpu));
        field("width", self.width.to_string());
        field("height", self.height.to_string());
        field("fps", self.fps.to_string());
        field("frames", self.frames.to_string());
        field("render_seconds", format!("{:.3}", self.seconds));
        let pairs: Vec<String> = settings
            .iter()
            .map(|(k, v)| format!("{}: {}", json_string(k), json_string(v)))
            .collect();
        field("settings", format!("{{{}}}", pairs.join(", ")));
        let scene_meta: Vec<String> = data
            .meta_entries()
            .iter()
            .map(|(k, v)| format!("{}: {}", json_string(k), json_string(v)))
            .collect();
        json += &format!("  \"scene_meta\": {{{}}}\n}}\n", scene_meta.join(", "));
        json
    }
}

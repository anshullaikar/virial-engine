//! A minimal WebM writer for one VP9 video track: enough for a browser
//! recording (WebCodecs gives encoded frames, not a file). Frames go into
//! clusters that start at each keyframe (or every 5 s), and a cue index at
//! the end makes the file seekable.
//!
//! Matroska elements are an ID, a size and a body; IDs below are the
//! standard ones (matroska.org/technical/elements.html).

/// Encoded frames collected for one video.
pub struct WebmWriter {
    width: u32,
    height: u32,
    fps: u32,
    frames: Vec<(bool, Vec<u8>)>,
}

impl WebmWriter {
    pub fn new(width: u32, height: u32, fps: u32) -> Self {
        Self {
            width,
            height,
            fps,
            frames: Vec::new(),
        }
    }

    /// Adds the next frame (frame k shows at k / fps seconds).
    pub fn push(&mut self, keyframe: bool, data: Vec<u8>) {
        self.frames.push((keyframe, data));
    }

    pub fn frames(&self) -> usize {
        self.frames.len()
    }

    /// The finished file.
    pub fn finish(&self) -> Vec<u8> {
        let ms = |k: usize| (k as u64 * 1000 + self.fps as u64 / 2) / self.fps as u64;
        let duration_ms = ms(self.frames.len());

        let info = element(
            0x1549_A966,
            &[
                // Timestamps in milliseconds
                uint(0x2A_D7B1, 1_000_000),
                string(0x4D80, "virial"),
                string(0x5741, "virial"),
                float(0x4489, duration_ms as f64),
            ]
            .concat(),
        );
        let video = element(
            0xE0,
            &[
                uint(0xB0, self.width as u64),
                uint(0xBA, self.height as u64),
            ]
            .concat(),
        );
        let track = element(
            0xAE,
            &[
                uint(0xD7, 1),
                uint(0x73C5, 1),
                // Video
                uint(0x83, 1),
                string(0x86, "V_VP9"),
                uint(0x23_E383, 1_000_000_000 / self.fps as u64),
                video,
            ]
            .concat(),
        );
        let tracks = element(0x1654_AE6B, &track);

        // Clusters, remembering where each starts (relative to the segment
        // body) for the cues
        let mut clusters = Vec::new();
        let mut cues = Vec::new();
        let before_clusters = info.len() + tracks.len();
        let mut k = 0;
        while k < self.frames.len() {
            let start = ms(k);
            let mut body = uint(0xE7, start);
            let mut end = k;
            while end < self.frames.len()
                && (end == k || !self.frames[end].0)
                && ms(end) - start < 5000
            {
                let (key, data) = &self.frames[end];
                let mut block = vec![0x81];
                block.extend(((ms(end) - start) as i16).to_be_bytes());
                block.push(if *key { 0x80 } else { 0x00 });
                block.extend(data);
                body.extend(element(0xA3, &block));
                end += 1;
            }
            if self.frames[k].0 {
                cues.push((start, before_clusters + clusters.len()));
            }
            clusters.extend(element(0x1F43_B675, &body));
            k = end;
        }
        let cue_points: Vec<u8> = cues
            .iter()
            .flat_map(|&(time, position)| {
                element(
                    0xBB,
                    &[
                        uint(0xB3, time),
                        element(0xB7, &[uint(0xF7, 1), uint(0xF1, position as u64)].concat()),
                    ]
                    .concat(),
                )
            })
            .collect();
        let segment = [info, tracks, clusters, element(0x1C53_BB6B, &cue_points)].concat();

        let header = element(
            0x1A45_DFA3,
            &[
                uint(0x4286, 1),
                uint(0x42F7, 1),
                uint(0x42F2, 4),
                uint(0x42F3, 8),
                string(0x4282, "webm"),
                uint(0x4287, 4),
                uint(0x4285, 2),
            ]
            .concat(),
        );
        [header, element(0x1853_8067, &segment)].concat()
    }
}

/// An element: ID bytes (the ID already carries its length marker), size as
/// an 8-byte variable-length integer, then the body
fn element(id: u32, body: &[u8]) -> Vec<u8> {
    let mut out: Vec<u8> = id
        .to_be_bytes()
        .into_iter()
        .skip_while(|&b| b == 0)
        .collect();
    out.push(0x01);
    out.extend(&(body.len() as u64).to_be_bytes()[1..]);
    out.extend(body);
    out
}

fn uint(id: u32, value: u64) -> Vec<u8> {
    let bytes: Vec<u8> = value
        .to_be_bytes()
        .into_iter()
        .skip_while(|&b| b == 0)
        .collect();
    element(id, if bytes.is_empty() { &[0] } else { &bytes })
}

fn float(id: u32, value: f64) -> Vec<u8> {
    element(id, &value.to_be_bytes())
}

fn string(id: u32, value: &str) -> Vec<u8> {
    element(id, value.as_bytes())
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::*;
    use std::process::Command;

    fn ffmpeg_available() -> bool {
        Command::new("ffmpeg").arg("-version").output().is_ok()
    }

    /// Frames from an IVF file (32-byte header, then a 12-byte header per
    /// frame: size, timestamp); keyframes from the VP9 frame header
    fn read_ivf(bytes: &[u8]) -> Vec<(bool, Vec<u8>)> {
        let mut out = Vec::new();
        let mut at = 32;
        while at + 12 <= bytes.len() {
            let size = u32::from_le_bytes(bytes[at..at + 4].try_into().expect("4 bytes")) as usize;
            let data = bytes[at + 12..at + 12 + size].to_vec();
            // Profile 0: frame marker, two profile bits, show_existing_frame,
            // then frame_type (0 for a keyframe)
            let keyframe = data[0] & 0x04 == 0;
            out.push((keyframe, data));
            at += 12 + size;
        }
        out
    }

    /// Per-frame hashes of what ffmpeg decodes from `path`
    fn decoded_hashes(path: &std::path::Path) -> String {
        let out = Command::new("ffmpeg")
            .args(["-loglevel", "error", "-i"])
            .arg(path)
            .args(["-f", "framemd5", "-"])
            .output()
            .expect("ffmpeg runs");
        assert!(
            out.status.success(),
            "ffmpeg could not decode {}",
            path.display()
        );
        // Only the hashes: the timestamp columns differ between containers
        String::from_utf8_lossy(&out.stdout)
            .lines()
            .filter(|l| !l.starts_with('#'))
            .map(|l| l.rsplit(',').next().unwrap_or("").trim().to_string())
            .collect::<Vec<_>>()
            .join("\n")
    }

    #[test]
    fn ffmpeg_reads_back_every_frame() {
        if !ffmpeg_available() {
            eprintln!("skipped: no ffmpeg");
            return;
        }
        let dir = std::env::temp_dir().join(format!("virial-webm-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("temp dir");
        let ivf = dir.join("frames.ivf");
        // 3 s of a moving test pattern at 30 fps, a keyframe every second
        let status = Command::new("ffmpeg")
            .args(["-loglevel", "error", "-y", "-f", "lavfi", "-i"])
            .arg("testsrc2=size=320x180:rate=30:duration=3")
            .args([
                "-c:v",
                "libvpx-vp9",
                "-g",
                "30",
                "-b:v",
                "500k",
                "-f",
                "ivf",
            ])
            .arg(&ivf)
            .status()
            .expect("ffmpeg runs");
        assert!(status.success());
        let frames = read_ivf(&std::fs::read(&ivf).expect("ivf written"));
        assert_eq!(frames.len(), 90);
        assert_eq!(frames.iter().filter(|f| f.0).count(), 3, "keyframes");
        let mut writer = WebmWriter::new(320, 180, 30);
        for (key, data) in frames {
            writer.push(key, data);
        }
        let webm = dir.join("video.webm");
        std::fs::write(&webm, writer.finish()).expect("webm written");
        let (from_ivf, from_webm) = (decoded_hashes(&ivf), decoded_hashes(&webm));
        assert_eq!(from_webm.lines().count(), 90, "frames decoded from WebM");
        assert_eq!(from_ivf, from_webm, "decoded frames differ");
        // Duration as players see it
        let probe = Command::new("ffprobe")
            .args([
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "csv=p=0",
            ])
            .arg(&webm)
            .output()
            .expect("ffprobe runs");
        let duration: f64 = String::from_utf8_lossy(&probe.stdout)
            .trim()
            .parse()
            .expect("a duration");
        assert!((duration - 3.0).abs() < 0.05, "duration {duration}");
        let _ = std::fs::remove_dir_all(&dir);
    }
}

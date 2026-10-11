//! Data a scene produces: measurement rows, array snapshots and run metadata.
//!
//! When recording (`record=<dir>`), rows go to `measurements.csv`, snapshots
//! to `<name>.npy` and metadata to `run.json` in that directory. In the live
//! preview, rows are kept in memory and snapshots are skipped.
//!
//! Numbers are written in their shortest exact form: reading a value back
//! gives exactly the f64 that was written.

use std::fmt::Write as _;
#[cfg(not(target_arch = "wasm32"))]
use std::path::PathBuf;

/// Where a scene's data goes.
pub struct DataLog {
    #[cfg(not(target_arch = "wasm32"))]
    dir: Option<PathBuf>,
    /// Recording (to disk, or in memory for a browser download)
    recording: bool,
    columns: Vec<String>,
    rows: Vec<Vec<f64>>,
    meta: Vec<(String, String)>,
}

impl DataLog {
    pub(crate) fn preview() -> Self {
        Self {
            #[cfg(not(target_arch = "wasm32"))]
            dir: None,
            recording: false,
            columns: Vec::new(),
            rows: Vec::new(),
            meta: Vec::new(),
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) fn recording(dir: PathBuf) -> Self {
        Self {
            dir: Some(dir),
            recording: true,
            ..Self::preview()
        }
    }

    /// Recording, with everything kept in memory (the browser recorder
    /// offers it as downloads; snapshots are skipped)
    #[cfg(target_arch = "wasm32")]
    pub(crate) fn recording_in_memory() -> Self {
        Self {
            recording: true,
            ..Self::preview()
        }
    }

    /// Whether this run is being recorded to disk.
    pub fn is_recording(&self) -> bool {
        self.recording
    }

    /// Adds a measurement row. The first row sets the columns; later rows must
    /// use the same names in the same order.
    pub fn row(&mut self, values: &[(&str, f64)]) {
        if self.columns.is_empty() {
            self.columns = values.iter().map(|(name, _)| name.to_string()).collect();
        }
        debug_assert!(
            values
                .iter()
                .map(|(n, _)| *n)
                .eq(self.columns.iter().map(String::as_str)),
            "measurement columns changed between rows"
        );
        self.rows.push(values.iter().map(|(_, v)| *v).collect());
    }

    /// Records a setting or fact about the run for `run.json`.
    pub fn meta(&mut self, key: &str, value: impl ToString) {
        let value = value.to_string();
        match self.meta.iter_mut().find(|(k, _)| k == key) {
            Some(entry) => entry.1 = value,
            None => self.meta.push((key.to_string(), value)),
        }
    }

    /// Writes `data` (row-major, `shape` dimensions) to `<name>.npy` when
    /// recording. Load it in Python with `numpy.load`.
    pub fn snapshot(&mut self, name: &str, shape: &[usize], data: &[f64]) {
        #[cfg(target_arch = "wasm32")]
        let _ = name;
        assert_eq!(
            shape.iter().product::<usize>(),
            data.len(),
            "shape doesn't match data"
        );
        #[cfg(not(target_arch = "wasm32"))]
        if let Some(dir) = &self.dir {
            let path = dir.join(format!("{name}.npy"));
            std::fs::write(&path, npy(shape, data))
                .unwrap_or_else(|e| panic!("could not write {}: {e}", path.display()));
        }
    }

    /// The rows so far, as CSV text.
    pub fn csv(&self) -> String {
        let mut out = self.columns.join(",");
        out.push('\n');
        for row in &self.rows {
            let cells: Vec<String> = row.iter().map(|v| exact(*v)).collect();
            out += &cells.join(",");
            out.push('\n');
        }
        out
    }

    pub(crate) fn meta_entries(&self) -> &[(String, String)] {
        &self.meta
    }
}

/// Shortest decimal that reads back as exactly `v`.
pub fn exact(v: f64) -> String {
    let mut s = String::new();
    let _ = write!(s, "{v:?}");
    s
}

/// NumPy `.npy` (format 1.0) bytes for a little-endian f64 array.
pub fn npy(shape: &[usize], data: &[f64]) -> Vec<u8> {
    let dims: Vec<String> = shape.iter().map(|d| d.to_string()).collect();
    let shape_text = match dims.len() {
        1 => format!("({},)", dims[0]),
        _ => format!("({})", dims.join(", ")),
    };
    let mut header = format!("{{'descr': '<f8', 'fortran_order': False, 'shape': {shape_text}, }}");
    // Magic (6) + version (2) + length (2) + header + newline, padded to 64
    let unpadded = 10 + header.len() + 1;
    header.push_str(&" ".repeat((64 - unpadded % 64) % 64));
    header.push('\n');
    let mut out = Vec::with_capacity(10 + header.len() + data.len() * 8);
    out.extend_from_slice(b"\x93NUMPY\x01\x00");
    out.extend_from_slice(&(header.len() as u16).to_le_bytes());
    out.extend_from_slice(header.as_bytes());
    for v in data {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

/// A JSON string literal.
pub(crate) fn json_string(s: &str) -> String {
    let mut out = String::from("\"");
    for c in s.chars() {
        match c {
            '"' => out += "\\\"",
            '\\' => out += "\\\\",
            '\n' => out += "\\n",
            c if (c as u32) < 0x20 => out += &format!("\\u{:04x}", c as u32),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn numbers_round_trip_exactly() {
        for v in [
            0.1,
            -1.0 / 3.0,
            6.02214076e23,
            1e-300,
            -0.0,
            12345.678901234567,
        ] {
            assert_eq!(exact(v).parse::<f64>().unwrap().to_bits(), v.to_bits());
        }
    }

    #[test]
    fn npy_header_is_aligned_and_sized() {
        let bytes = npy(&[3, 2], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(&bytes[..8], b"\x93NUMPY\x01\x00");
        let header_len = u16::from_le_bytes([bytes[8], bytes[9]]) as usize;
        assert_eq!((10 + header_len) % 64, 0);
        assert_eq!(bytes.len(), 10 + header_len + 6 * 8);
        let header = std::str::from_utf8(&bytes[10..10 + header_len]).unwrap();
        assert!(header.contains("'shape': (3, 2)") && header.ends_with('\n'));
    }
}

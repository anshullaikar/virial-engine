use std::collections::HashSet;

use winit::keyboard::KeyCode;

/// Mouse and keyboard state for the current frame.
///
/// Positions are in normalised device coordinates: x and y run from -1 to 1
/// across the surface, with y pointing up.
#[derive(Default)]
pub struct Input {
    cursor: Option<[f32; 2]>,
    previous: Option<[f32; 2]>,
    size: [f32; 2],
    pub(crate) left: bool,
    pub(crate) right: bool,
    pub(crate) scroll: f32,
    held: HashSet<KeyCode>,
    pressed: HashSet<KeyCode>,
}

impl Input {
    /// Cursor position, or `None` when it's outside the surface.
    pub fn cursor(&self) -> Option<[f32; 2]> {
        self.cursor
    }

    /// How far the cursor moved this frame while the left button was held.
    pub fn drag(&self) -> [f32; 2] {
        match (self.left, self.cursor, self.previous) {
            (true, Some(c), Some(p)) => [c[0] - p[0], c[1] - p[1]],
            _ => [0.0, 0.0],
        }
    }

    pub fn left_down(&self) -> bool {
        self.left
    }

    pub fn right_down(&self) -> bool {
        self.right
    }

    /// Scroll this frame, in lines. Positive scrolls up (away from the user).
    pub fn scroll(&self) -> f32 {
        self.scroll
    }

    /// Whether `key` is held down.
    pub fn held(&self, key: KeyCode) -> bool {
        self.held.contains(&key)
    }

    /// Whether `key` went down this frame.
    pub fn pressed(&self, key: KeyCode) -> bool {
        self.pressed.contains(&key)
    }

    pub(crate) fn set_surface_size(&mut self, width: u32, height: u32) {
        self.size = [width.max(1) as f32, height.max(1) as f32];
    }

    pub(crate) fn cursor_moved(&mut self, x: f64, y: f64) {
        let [w, h] = self.size;
        if w == 0.0 {
            return;
        }
        self.cursor = Some([(x as f32 / w) * 2.0 - 1.0, 1.0 - (y as f32 / h) * 2.0]);
    }

    pub(crate) fn cursor_left(&mut self) {
        self.cursor = None;
        self.previous = None;
    }

    pub(crate) fn key(&mut self, code: KeyCode, down: bool, repeat: bool) {
        if down {
            if !repeat {
                self.pressed.insert(code);
            }
            self.held.insert(code);
        } else {
            self.held.remove(&code);
        }
    }

    pub(crate) fn end_frame(&mut self) {
        self.previous = self.cursor;
        self.scroll = 0.0;
        self.pressed.clear();
    }
}

//! Raw RGB frame + nearest-neighbour resize + normalization.
//!
//! Vision models want a fixed 224x224 input. Decoding aside, this resize + byte
//! layout is the hot path that would choke Python, so it lives in Rust. No image
//! crate needed for nearest-neighbour.

use base64::Engine;

pub const TARGET: usize = 224;

#[derive(Clone)]
pub struct RawFrame {
    pub width: usize,
    pub height: usize,
    pub rgb: Vec<u8>, // width*height*3, row-major RGB
}

impl RawFrame {
    pub fn new(width: usize, height: usize) -> Self {
        RawFrame { width, height, rgb: vec![0u8; width * height * 3] }
    }

    #[inline]
    pub fn set(&mut self, x: usize, y: usize, r: u8, g: u8, b: u8) {
        let i = (y * self.width + x) * 3;
        self.rgb[i] = r;
        self.rgb[i + 1] = g;
        self.rgb[i + 2] = b;
    }

    /// Nearest-neighbour resize to `TARGET`x`TARGET`.
    pub fn resize_square(&self) -> RawFrame {
        let mut out = RawFrame::new(TARGET, TARGET);
        if self.width == 0 || self.height == 0 {
            return out;
        }
        for ty in 0..TARGET {
            let sy = ty * self.height / TARGET;
            for tx in 0..TARGET {
                let sx = tx * self.width / TARGET;
                let si = (sy * self.width + sx) * 3;
                let di = (ty * TARGET + tx) * 3;
                out.rgb[di] = self.rgb[si];
                out.rgb[di + 1] = self.rgb[si + 1];
                out.rgb[di + 2] = self.rgb[si + 2];
            }
        }
        out
    }

    /// Base64 of the raw RGB bytes — what the ML service decodes for CLIP.
    pub fn tensor_b64(&self) -> String {
        base64::engine::general_purpose::STANDARD.encode(&self.rgb)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resize_produces_224_square() {
        let mut f = RawFrame::new(640, 480);
        f.set(0, 0, 10, 20, 30);
        let r = f.resize_square();
        assert_eq!(r.width, TARGET);
        assert_eq!(r.height, TARGET);
        assert_eq!(r.rgb.len(), TARGET * TARGET * 3);
    }

    #[test]
    fn tensor_b64_roundtrips_len() {
        let f = RawFrame::new(2, 2);
        let b64 = f.tensor_b64();
        let bytes = base64::engine::general_purpose::STANDARD
            .decode(b64)
            .unwrap();
        assert_eq!(bytes.len(), 2 * 2 * 3);
    }
}

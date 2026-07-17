//! Synthetic frame source.
//!
//! Without the `ffmpeg` feature (the default), there's no real decoder, so this
//! produces deterministic RGB frames on a scripted timeline. Each frame carries
//! ground-truth `labels`, which the ML service uses in offline mode — so the full
//! decode -> resize -> embed -> index path runs with no video file.

use crate::frame::RawFrame;

pub struct SynthFrame {
    pub timestamp_ms: u64,
    pub frame: RawFrame,
    pub labels: Vec<String>,
}

/// (timestamp_ms, labels) — mirrors the ML demo so search returns sensible hits.
fn script() -> Vec<(u64, Vec<&'static str>)> {
    vec![
        (0, vec!["daytime", "car"]),
        (1000, vec!["daytime", "car", "turning left"]),
        (2000, vec!["daytime", "crosswalk", "person"]),
        (3000, vec!["person", "red jacket"]),
        (4000, vec!["person", "red jacket", "backpack"]),
        (5000, vec!["person", "backpack", "stopped"]),
        (6000, vec!["backpack", "crosswalk"]),
        (7000, vec!["crowd", "traffic light"]),
        (8000, vec!["bicycle", "turning right"]),
        (9000, vec!["car", "speeding", "night"]),
        (10000, vec!["truck", "night"]),
        (11000, vec!["dog", "person"]),
    ]
}

fn hash(s: &str) -> u64 {
    let mut h: u64 = 1469598103934665603;
    for b in s.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(1099511628211);
    }
    h
}

fn colored_frame(labels: &[&str]) -> RawFrame {
    let mut f = RawFrame::new(320, 240);
    let seed = labels.iter().fold(0u64, |a, l| a.wrapping_add(hash(l)));
    let (r, g, b) = (
        (seed & 0xff) as u8,
        ((seed >> 8) & 0xff) as u8,
        ((seed >> 16) & 0xff) as u8,
    );
    for y in 0..f.height {
        for x in 0..f.width {
            f.set(x, y, r ^ (x as u8), g ^ (y as u8), b);
        }
    }
    f
}

pub fn generate() -> Vec<SynthFrame> {
    script()
        .into_iter()
        .map(|(ts, labels)| SynthFrame {
            timestamp_ms: ts,
            frame: colored_frame(&labels),
            labels: labels.iter().map(|s| s.to_string()).collect(),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generates_scripted_frames_with_labels() {
        let frames = generate();
        assert_eq!(frames.len(), 12);
        assert!(frames.iter().any(|f| f.labels.contains(&"red jacket".to_string())));
        // frames are real RGB buffers
        assert_eq!(frames[0].frame.rgb.len(), 320 * 240 * 3);
    }
}

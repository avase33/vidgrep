//! vidgrep processor CLI.
//!
//! Samples frames (synthetic by default, or real video with `--features ffmpeg`),
//! resizes each to 224x224, and POSTs the normalized tensor + metadata to the ML
//! service for embedding & indexing.
//!
//!   vidgrep-processor --video-id v-123 --ml-url http://localhost:8000
//!   vidgrep-processor --dry-run          # build tensors but don't POST

use serde_json::json;
use vidgrep_processor::synthetic;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut video_id = "demo".to_string();
    let mut ml_url = std::env::var("VIDGREP_ML_URL").unwrap_or_else(|_| "http://localhost:8000".to_string());
    let mut dry = false;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--video-id" if i + 1 < args.len() => {
                i += 1;
                video_id = args[i].clone();
            }
            "--ml-url" if i + 1 < args.len() => {
                i += 1;
                ml_url = args[i].clone();
            }
            "--dry-run" => dry = true,
            _ => {}
        }
        i += 1;
    }

    let frames = synthetic::generate();
    println!("processor: {} frames for video '{}'", frames.len(), video_id);

    let mut sent = 0usize;
    for sf in &frames {
        let resized = sf.frame.resize_square();
        let body = json!({
            "video_id": video_id,
            "timestamp_ms": sf.timestamp_ms,
            "width": 224,
            "height": 224,
            "tensor_b64": resized.tensor_b64(),
            "labels": sf.labels,
            "index": true,
        });
        if dry {
            sent += 1;
            continue;
        }
        match ureq::post(&format!("{}/embed/image", ml_url)).send_json(body) {
            Ok(_) => sent += 1,
            Err(e) => eprintln!("  frame {}ms failed: {}", sf.timestamp_ms, e),
        }
    }
    println!("processor: sent {}/{} frames to {}", sent, frames.len(), ml_url);
}

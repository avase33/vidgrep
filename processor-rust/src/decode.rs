//! Real video decoding via FFmpeg (compiled only with `--features ffmpeg`).
//!
//! Samples one frame per `interval_ms`, converts to RGB24, and hands each frame
//! to the resize pipeline. Requires system FFmpeg libraries; the default build
//! uses the synthetic source instead so no system deps are needed.

use ffmpeg_next as ffmpeg;

use crate::frame::RawFrame;

/// Decode `path`, returning (timestamp_ms, RawFrame) sampled every `interval_ms`.
pub fn sample_frames(path: &str, interval_ms: u64) -> Result<Vec<(u64, RawFrame)>, String> {
    ffmpeg::init().map_err(|e| e.to_string())?;
    let mut ictx = ffmpeg::format::input(&path).map_err(|e| e.to_string())?;
    let input = ictx
        .streams()
        .best(ffmpeg::media::Type::Video)
        .ok_or("no video stream")?;
    let stream_index = input.index();
    let time_base = f64::from(input.time_base());

    let ctx = ffmpeg::codec::context::Context::from_parameters(input.parameters())
        .map_err(|e| e.to_string())?;
    let mut decoder = ctx.decoder().video().map_err(|e| e.to_string())?;

    let mut scaler = ffmpeg::software::scaling::context::Context::get(
        decoder.format(),
        decoder.width(),
        decoder.height(),
        ffmpeg::format::Pixel::RGB24,
        decoder.width(),
        decoder.height(),
        ffmpeg::software::scaling::flag::Flags::BILINEAR,
    )
    .map_err(|e| e.to_string())?;

    let mut out = Vec::new();
    let mut next_ms = 0u64;

    let mut receive = |decoder: &mut ffmpeg::decoder::Video| -> Result<(), String> {
        let mut frame = ffmpeg::frame::Video::empty();
        while decoder.receive_frame(&mut frame).is_ok() {
            let ts_ms = (frame.pts().unwrap_or(0) as f64 * time_base * 1000.0) as u64;
            if ts_ms + interval_ms < next_ms {
                continue;
            }
            next_ms = ts_ms + interval_ms;
            let mut rgb = ffmpeg::frame::Video::empty();
            scaler.run(&frame, &mut rgb).map_err(|e| e.to_string())?;
            let w = rgb.width() as usize;
            let h = rgb.height() as usize;
            let data = rgb.data(0);
            let stride = rgb.stride(0);
            let mut raw = RawFrame::new(w, h);
            for y in 0..h {
                for x in 0..w {
                    let i = y * stride + x * 3;
                    raw.set(x, y, data[i], data[i + 1], data[i + 2]);
                }
            }
            out.push((ts_ms, raw));
        }
        Ok(())
    };

    for (stream, packet) in ictx.packets() {
        if stream.index() == stream_index {
            decoder.send_packet(&packet).map_err(|e| e.to_string())?;
            receive(&mut decoder)?;
        }
    }
    decoder.send_eof().map_err(|e| e.to_string())?;
    receive(&mut decoder)?;
    Ok(out)
}

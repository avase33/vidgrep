//! vidgrep frame processor — decode/sample -> resize 224x224 -> normalize.

pub mod frame;
pub mod synthetic;

#[cfg(feature = "ffmpeg")]
pub mod decode;

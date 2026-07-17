# Changelog

Format: [Keep a Changelog](https://keepachangelog.com/); versioning: [SemVer](https://semver.org/).

## [0.1.0] - 2026-07-17

Initial release — a four-language multimodal video search engine.

### Added
- **Python ML pipeline**: CLIP-style shared-space embedder (deterministic mock for
  image+text, optional real `clip-vit-base-patch32`), in-memory cosine vector store
  + optional Qdrant adapter, FastAPI (`/embed/image`, `/embed/text`, `/search`), CLI,
  offline demo + pytest suite.
- **Rust processor**: nearest-neighbour resize to 224x224, RGB tensor + base64,
  synthetic scripted frame source, optional FFmpeg decoder behind a feature flag,
  POSTs frames to the ML service. Unit tests.
- **Go gateway**: chunked multipart upload staging, in-process job queue, concurrent
  worker pool that spawns the Rust processor per video, `/search` proxy, `/status`
  + WebSocket progress, `/health`. Tests.
- **Next.js UI**: upload, natural-language search, HTML5 player with a canvas
  bounding-box overlay, and a clickable results timeline that seeks the video.
- Shared JSON protocol (`proto/protocol.md`), docker-compose (+ optional Qdrant),
  per-language Dockerfiles, multi-language CI, Makefile, offline verifier, MIT license.

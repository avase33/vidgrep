# vidgrep architecture

Search *inside* videos with natural language. Each language owns the layer it's
best at; one JSON protocol (`proto/protocol.md`) is the only coupling.

```
┌──────────────────────────────────────────────────────────────────────┐
│ Browser · Next.js + TypeScript                                        │
│ upload · NL search box · <video> + canvas bounding boxes · timeline    │
└───────────────┬───────────────────────────────────────────────────────┘
                │ HTTP  (/upload, /search, /status, WS /ws/status)
┌───────────────▼───────────────────────────────────────────────────────┐
│ Gateway · Go                                                          │
│ chunked upload → temp disk · job queue · worker pool · search proxy    │
└───────────────┬───────────────────────────────────────────────────────┘
                │ spawn per video (subprocess)
┌───────────────▼───────────────────────────────────────────────────────┐
│ Processor · Rust                                                      │
│ decode/sample frames · resize 224x224 · normalize → tensor            │
└───────────────┬───────────────────────────────────────────────────────┘
                │ HTTP  POST /embed/image  (tensor + metadata)
┌───────────────▼───────────────────────────────────────────────────────┐
│ ML · Python                                                           │
│ CLIP-style embed (image + text, shared space) · vector store · search │
└────────────────────────────────────────────────────────────────────────┘
```

## Why each language

| Layer | Language | Reason |
| --- | --- | --- |
| Interface | TypeScript / Next.js | Canvas overlays on playing video, interactive timeline, reactive state. |
| Gateway | Go | Absorbs multi-GB uploads and high-throughput streaming with cheap concurrency. |
| Processor | Rust | Video decode + pixel-matrix work at near-C speed, no GC pauses. |
| ML | Python | PyTorch / CLIP / vector search ecosystem. |

## Indexing flow

1. Browser `POST /upload`s a video; Go streams it to disk and enqueues a job.
2. A Go worker pops the job and spawns the Rust processor for that video.
3. Rust samples frames (1 fps), resizes to 224×224, normalizes, and `POST`s each
   frame tensor to Python `/embed/image`.
4. Python embeds each frame into a 512-dim vector and upserts it into the vector
   store keyed by `(video_id, timestamp_ms)`.

## Search flow

1. Browser `POST /search {video_id, query}` → Go proxies to Python.
2. Python embeds the **text** query into the *same* 512-dim space and ranks the
   video's frame vectors by cosine similarity.
3. It returns the top timestamps (with scores, labels, and a bounding box). The UI
   drops clickable markers on the timeline; clicking one seeks the `<video>` to
   that frame and draws the box on the canvas overlay.

## The shared embedding space

The whole thing hinges on image frames and text queries living in **one** vector
space (CLIP's key property). Offline, a deterministic mock projects a shared
concept vocabulary into 512-dim: frames carry ground-truth `labels`, a query is
matched to those concepts, and cosine similarity ranks correctly — so the entire
index→search path runs with zero ML weights. Set `VIDGREP_CLIP=real` (with the
`clip` extra) to run an actual `clip-vit-base-patch32` on real pixels; nothing else
changes.

## Offline-first

- **ML**: mock CLIP + in-memory vector store → no weights, no Qdrant, no GPU.
- **Rust**: synthetic scripted frames (real RGB buffers) → no FFmpeg/system libs;
  enable the `ffmpeg` feature for real decoding.
- **Go**: in-process queue → no Redis; the interface is ready for a Redis/Kafka
  backend.

So `docker compose up` (or `make ml-demo`) gives a working, searchable pipeline
immediately, and each layer is independently testable.

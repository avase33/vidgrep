# vidgrep 🎞️🔎

**Search *inside* videos with natural language.** Not by title or tags — vidgrep
processes video frame-by-frame and lets you ask things like *"find the frame where
a person in a red jacket has a backpack"* or *"a car turning left"* and jump
straight to the matching moment.

Four languages, each doing what it's best at, joined by one JSON protocol:

```
Browser ──HTTP──▶ Go gateway ──spawn──▶ Rust processor ──HTTP──▶ Python ML ──▶ vector store
(Next.js UI)      (upload+queue)        (decode+resize)          (CLIP embed)     (search)
        ◀───────────── timestamps + scores + bounding boxes ─────────────────────┘
```

| Layer | Language | Owns |
| --- | --- | --- |
| **UI** | TypeScript / Next.js | Upload, NL search, `<video>` + canvas boxes, clickable timeline |
| **Gateway** | Go | Multi-GB upload streaming, job queue, worker pool, search proxy |
| **Processor** | Rust | Frame decode/sample, resize 224×224, normalize to tensors |
| **ML** | Python | CLIP-style image+text embedding (shared space), vector search |

It runs **offline with zero ML weights and no video files**: a deterministic mock
CLIP embeds a shared concept space, the Rust processor emits synthetic frames, and
the Go queue is in-process. Flip env vars for real CLIP (`clip-vit-base-patch32`),
FFmpeg decoding, and Qdrant.

## Quickstart — the ML core, offline

```bash
cd ml-pipeline-python && pip install -e .
python -m vidgrep_ml.cli demo
```

```
vidgrep-ml demo — embedder=mock  indexed 12 frames
  query: "person in a red jacket"
      3.0s  score=1.000  labels=['person', 'red jacket']  box=[...]
      4.0s  score=0.816  labels=['person', 'red jacket', 'backpack']  box=[...]
  query: "a car turning left"
      1.0s  score=0.816  labels=['daytime', 'car', 'turning left']  box=[...]
```

Offline end-to-end check:

```bash
python scripts/verify.py     # RESULT: N passed, 0 failed
```

## Quickstart — the whole stack

```bash
docker compose up --build
# UI:        http://localhost:3000   (ingest the demo, search, click the timeline)
# Gateway:   http://localhost:8080/health
# ML:        http://localhost:8000/health
```

Or run layers standalone:

```bash
cd ml-pipeline-python && pip install -e ".[server]" && vidgrep-ml serve      # :8000
cd processor-rust     && cargo run -- --video-id demo --ml-url http://localhost:8000
cd gateway-go         && VIDGREP_ML_URL=http://localhost:8000 go run .        # :8080
cd frontend-ts        && npm install && npm run dev                           # :3000
```

## Going real (production adapters)

```bash
# real CLIP on real pixels
pip install -e "ml-pipeline-python[clip]"   &&  export VIDGREP_CLIP=real
# real video decoding
cargo build --release --features ffmpeg -p vidgrep-processor
# real vector DB
export VIDGREP_VECTORDB=qdrant QDRANT_URL=http://localhost:6333
```

No code changes — the factories in `embedder.py`, `vector_store.py`, the Rust
feature flag, and the Go config pick the implementation from the environment.

## The interesting engineering

- **Shared CLIP space** — image frames and text queries embed into one 512-dim
  space, which is what makes "search by meaning" work. `ml-pipeline-python/vidgrep_ml/`
- **Rust decode path** — nearest-neighbour resize + RGB tensor layout, the hot
  loop that would choke Python. `processor-rust/src/frame.rs`
- **Go ingestion** — chunk-streams uploads to disk, a bounded queue + worker pool
  spawns the Rust binary per video and tracks progress over WebSocket. `gateway-go/`
- **Canvas overlay** — bounding boxes drawn over the playing `<video>`, scaled from
  the 224×224 model frame to the display. `frontend-ts/app/page.tsx`

## Testing

```bash
make test                    # rust + python + go
cd ml-pipeline-python && pytest -q
cd processor-rust     && cargo test
cd gateway-go         && go test ./...
cd frontend-ts        && npm run build
```

## Layout

```
proto/               shared JSON wire protocol
frontend-ts/         Next.js UI (player, canvas boxes, timeline)
gateway-go/          Go ingestion gateway (upload, queue, workers, proxy)
processor-rust/      Rust frame processor (resize/normalize, +optional ffmpeg)
ml-pipeline-python/  CLIP-style embedder + vector store + FastAPI
scripts/verify.py    offline end-to-end check
docs/ARCHITECTURE.md
```

## License

MIT © 2026 Akhil Vase

# vidgrep wire protocol

Search *inside* videos with natural language. Every layer speaks JSON over HTTP
so any one can be rewritten without touching its neighbours.

```
Browser ──HTTP/WS──▶ Go gateway ──spawn──▶ Rust processor ──HTTP──▶ Python ML ──▶ vector store
(Next.js)           (upload+queue)        (decode frames)         (CLIP embed)     (search)
        ◀──search results (timestamps + boxes)──────────────────────────────────┘
```

## 1. Browser ⇄ Gateway (Go)

```jsonc
POST /upload            (multipart: video=@file.mp4)   -> { "video_id": "v-ab12", "status": "queued" }
GET  /status/{video_id}                                 -> { "video_id": ..., "frames": 120, "done": true }
POST /search  { "video_id": "v-ab12", "query": "person in a red jacket", "top_k": 5 }
      -> { "results": [ { "timestamp_ms": 42000, "score": 0.83, "box": [x,y,w,h] }, ... ] }
GET  /health
WS   /ws/status/{video_id}   -> streamed { "processed": n, "total": m, "done": bool }
```

## 2. Rust processor ⇄ Python ML

The Rust binary decodes/samples frames and posts each to Python:

```jsonc
POST /embed/image
{
  "video_id": "v-ab12",
  "timestamp_ms": 42000,
  "width": 224, "height": 224,
  "tensor_b64": "<base64 of normalized RGB bytes>",   // real path
  "labels": ["red jacket", "backpack"]                 // offline/simulated path
}
-> { "vector": [ ...512 floats... ], "indexed": true }
```

## 3. Browser/Gateway ⇄ Python ML (search)

```jsonc
POST /embed/text   { "text": "person in a red jacket" } -> { "vector": [ ...512 floats... ] }
POST /search       { "video_id": "v-ab12", "query": "...", "top_k": 5 }
      -> { "results": [ { "timestamp_ms": 42000, "score": 0.83, "labels": [...] }, ... ] }
GET  /health
```

## Shared types

### Frame

```jsonc
{ "video_id": "v-ab12", "timestamp_ms": 42000, "width": 224, "height": 224,
  "tensor_b64": "...", "labels": ["red jacket"] }
```

### SearchResult

```jsonc
{ "timestamp_ms": 42000, "score": 0.83, "labels": ["red jacket"], "box": [88, 40, 120, 200] }
```

## Embedding space

Image frames and text queries embed into the **same** 512-dim space (CLIP-style),
so cosine similarity ranks frames by how well they match the query. In offline
mode a deterministic mock encoder maps a shared concept vocabulary into that space
(frames carry ground-truth `labels`); with `VIDGREP_CLIP=real` the same endpoints
run an actual CLIP model on real pixels.

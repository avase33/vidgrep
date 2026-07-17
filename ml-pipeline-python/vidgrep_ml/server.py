"""FastAPI service for the vidgrep ML pipeline.

Endpoints (see proto/protocol.md): /embed/image, /embed/text, /index/upsert,
/search, /health. Requires the ``server`` extra.
"""

from __future__ import annotations

from typing import Any

from .models import Frame
from .service import IndexService

try:
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel
except ImportError as e:  # pragma: no cover
    raise RuntimeError("Install server extras: pip install 'vidgrep-ml[server]'") from e

app = FastAPI(title="vidgrep-ml", version="0.1.0")
_svc = IndexService()


class EmbedImageReq(BaseModel):
    video_id: str
    timestamp_ms: int
    width: int = 224
    height: int = 224
    tensor_b64: str | None = None
    labels: list[str] = []
    index: bool = True


class EmbedTextReq(BaseModel):
    text: str


class SearchReq(BaseModel):
    video_id: str
    query: str
    top_k: int = 5


@app.get("/health")
def health() -> JSONResponse:
    return JSONResponse({"status": "ok", "service": "ml", "embedder": _svc.embedder.name, "dim": _svc.embedder.dim})


@app.post("/embed/image")
def embed_image(req: EmbedImageReq) -> dict[str, Any]:
    frame = Frame(req.video_id, req.timestamp_ms, req.width, req.height, req.tensor_b64, req.labels)
    if req.index:
        vec = _svc.index_frame(frame)
    else:
        vec = _svc.embedder.embed_image(frame)
    return {"vector": vec, "indexed": req.index}


@app.post("/embed/text")
def embed_text(req: EmbedTextReq) -> dict[str, Any]:
    return {"vector": _svc.embedder.embed_text(req.text)}


@app.post("/search")
def search(req: SearchReq) -> dict[str, Any]:
    results = _svc.search(req.video_id, req.query, req.top_k)
    return {"results": [r.to_dict() for r in results]}

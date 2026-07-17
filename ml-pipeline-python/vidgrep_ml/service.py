"""Index + search orchestration."""

from __future__ import annotations

import hashlib
from typing import Optional

from .embedder import BaseEmbedder, build_embedder
from .models import Frame, SearchResult
from .vector_store import VectorStore, build_store


def _box_for(labels: list[str]) -> list[int]:
    """A deterministic bounding box (in the 224x224 frame) for the top label,
    so the UI has something to draw. Real detection would come from the model."""
    key = (labels[0] if labels else "object")
    h = int(hashlib.md5(key.encode()).hexdigest()[:6], 16)
    x = 20 + (h % 120)
    y = 20 + ((h >> 6) % 120)
    w = 40 + ((h >> 12) % 60)
    ht = 60 + ((h >> 4) % 80)
    return [x, y, min(w, 224 - x), min(ht, 224 - y)]


class IndexService:
    def __init__(self, embedder: Optional[BaseEmbedder] = None, store: Optional[VectorStore] = None) -> None:
        self.embedder = embedder or build_embedder()
        self.store = store or build_store(self.embedder.dim)

    def index_frame(self, frame: Frame) -> list[float]:
        vec = self.embedder.embed_image(frame)
        self.store.upsert(frame.video_id, frame.timestamp_ms, vec, frame.labels)
        return vec

    def index_frames(self, frames: list[Frame]) -> int:
        for f in frames:
            self.index_frame(f)
        return len(frames)

    def search(self, video_id: str, query: str, top_k: int = 5) -> list[SearchResult]:
        qvec = self.embedder.embed_text(query)
        hits = self.store.search(video_id, qvec, top_k)
        results: list[SearchResult] = []
        for rec, score in hits:
            results.append(
                SearchResult(
                    timestamp_ms=rec.timestamp_ms,
                    score=float(score),
                    labels=rec.labels,
                    box=_box_for(rec.labels),
                )
            )
        return results

    def count(self, video_id: str) -> int:
        return self.store.count(video_id)

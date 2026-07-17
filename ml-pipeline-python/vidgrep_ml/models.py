"""Shared data types (see proto/protocol.md)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Frame:
    video_id: str
    timestamp_ms: int
    width: int = 224
    height: int = 224
    tensor_b64: Optional[str] = None       # real path: normalized RGB bytes, base64
    labels: list[str] = field(default_factory=list)  # offline path: ground-truth concepts


@dataclass
class SearchResult:
    timestamp_ms: int
    score: float
    labels: list[str] = field(default_factory=list)
    box: Optional[list[int]] = None        # [x, y, w, h] in the 224x224 frame

    def to_dict(self) -> dict:
        d = {"timestamp_ms": self.timestamp_ms, "score": round(self.score, 4), "labels": self.labels}
        if self.box is not None:
            d["box"] = self.box
        return d


EMBED_DIM = 512

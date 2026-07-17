"""Vector store for frame embeddings.

`InMemoryStore` keeps per-video frame vectors and ranks them by cosine similarity
— no dependencies, perfect for tests and the offline demo. `QdrantStore` mirrors
the same interface against a real Qdrant instance when you install the ``qdrant``
extra and set ``VIDGREP_VECTORDB=qdrant``.
"""

from __future__ import annotations

import abc
import math
import os
from dataclasses import dataclass


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)


@dataclass
class Record:
    timestamp_ms: int
    vector: list[float]
    labels: list[str]


class VectorStore(abc.ABC):
    @abc.abstractmethod
    def upsert(self, video_id: str, timestamp_ms: int, vector: list[float], labels: list[str]) -> None: ...

    @abc.abstractmethod
    def search(self, video_id: str, query: list[float], top_k: int) -> list[tuple[Record, float]]: ...

    @abc.abstractmethod
    def count(self, video_id: str) -> int: ...


class InMemoryStore(VectorStore):
    def __init__(self) -> None:
        self._by_video: dict[str, list[Record]] = {}

    def upsert(self, video_id: str, timestamp_ms: int, vector: list[float], labels: list[str]) -> None:
        recs = self._by_video.setdefault(video_id, [])
        for r in recs:
            if r.timestamp_ms == timestamp_ms:
                r.vector, r.labels = vector, labels
                return
        recs.append(Record(timestamp_ms, vector, labels))

    def search(self, video_id: str, query: list[float], top_k: int) -> list[tuple[Record, float]]:
        recs = self._by_video.get(video_id, [])
        scored = [(r, cosine(query, r.vector)) for r in recs]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[: max(1, top_k)]

    def count(self, video_id: str) -> int:
        return len(self._by_video.get(video_id, []))


class QdrantStore(VectorStore):  # pragma: no cover - needs a running Qdrant
    def __init__(self, url: str, dim: int = 512) -> None:
        from qdrant_client import QdrantClient  # type: ignore
        from qdrant_client.models import Distance, VectorParams  # type: ignore

        self._c = QdrantClient(url=url)
        self._dim = dim
        self._Distance = Distance
        self._VectorParams = VectorParams
        self._seq = 0

    def _ensure(self, video_id: str) -> None:
        name = f"vid_{video_id}"
        if not self._c.collection_exists(name):
            self._c.create_collection(
                name, vectors_config=self._VectorParams(size=self._dim, distance=self._Distance.COSINE)
            )

    def upsert(self, video_id, timestamp_ms, vector, labels) -> None:
        from qdrant_client.models import PointStruct  # type: ignore

        self._ensure(video_id)
        self._seq += 1
        self._c.upsert(
            f"vid_{video_id}",
            points=[PointStruct(id=self._seq, vector=vector,
                                payload={"timestamp_ms": timestamp_ms, "labels": labels})],
        )

    def search(self, video_id, query, top_k):
        hits = self._c.search(f"vid_{video_id}", query_vector=query, limit=max(1, top_k))
        out = []
        for h in hits:
            p = h.payload or {}
            out.append((Record(p.get("timestamp_ms", 0), [], p.get("labels", [])), float(h.score)))
        return out

    def count(self, video_id):
        try:
            return self._c.count(f"vid_{video_id}").count
        except Exception:
            return 0


def build_store(dim: int = 512) -> VectorStore:
    if os.environ.get("VIDGREP_VECTORDB", "memory").lower() == "qdrant":
        return QdrantStore(os.environ.get("QDRANT_URL", "http://localhost:6333"), dim)
    return InMemoryStore()

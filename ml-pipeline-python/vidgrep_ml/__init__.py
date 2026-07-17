"""vidgrep ML pipeline — CLIP-style multimodal embedding + vector search."""

from .embedder import BaseEmbedder, MockCLIP, build_embedder
from .models import EMBED_DIM, Frame, SearchResult
from .service import IndexService
from .vector_store import InMemoryStore, build_store

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "BaseEmbedder",
    "MockCLIP",
    "build_embedder",
    "EMBED_DIM",
    "Frame",
    "SearchResult",
    "IndexService",
    "InMemoryStore",
    "build_store",
]

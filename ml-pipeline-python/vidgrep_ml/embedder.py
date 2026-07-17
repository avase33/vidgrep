"""CLIP-style multimodal embedders.

`MockCLIP` is the offline default: it embeds image frames from their concept
labels and text from the query, into one shared space. `RealCLIP` runs an actual
``openai/clip-vit-base-patch32`` model when you install the ``clip`` extra and set
``VIDGREP_CLIP=real``. Both produce 512-dim vectors, so the rest of the system is
identical either way.
"""

from __future__ import annotations

import abc
import base64
import os

from . import concepts
from .models import EMBED_DIM, Frame


class BaseEmbedder(abc.ABC):
    name = "base"
    dim = EMBED_DIM

    @abc.abstractmethod
    def embed_image(self, frame: Frame) -> list[float]: ...

    @abc.abstractmethod
    def embed_text(self, text: str) -> list[float]: ...


class MockCLIP(BaseEmbedder):
    name = "mock"

    def embed_image(self, frame: Frame) -> list[float]:
        # Offline: the frame's ground-truth labels define its content.
        if frame.labels:
            return concepts.vector_for_labels(frame.labels)
        # No labels but a tensor: derive a weak signal from pixel statistics so the
        # frame is still embedded deterministically.
        if frame.tensor_b64:
            raw = base64.b64decode(frame.tensor_b64)
            buckets = [0.0] * EMBED_DIM
            for i, byte in enumerate(raw[:4096]):
                buckets[i % EMBED_DIM] += byte / 255.0
            return concepts.normalize(buckets)
        return [0.0] * EMBED_DIM

    def embed_text(self, text: str) -> list[float]:
        return concepts.vector_for_text(text)


class RealCLIP(BaseEmbedder):  # pragma: no cover - needs torch + weights
    name = "clip"

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32") -> None:
        import torch  # type: ignore
        from transformers import CLIPModel, CLIPProcessor  # type: ignore

        self._torch = torch
        self._model = CLIPModel.from_pretrained(model_name)
        self._proc = CLIPProcessor.from_pretrained(model_name)
        self._model.eval()

    def _to_image(self, frame: Frame):
        from PIL import Image  # type: ignore

        raw = base64.b64decode(frame.tensor_b64 or "")
        return Image.frombytes("RGB", (frame.width, frame.height), raw)

    def embed_image(self, frame: Frame) -> list[float]:
        image = self._to_image(frame)
        inputs = self._proc(images=image, return_tensors="pt")
        with self._torch.no_grad():
            feats = self._model.get_image_features(**inputs)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats[0].tolist()

    def embed_text(self, text: str) -> list[float]:
        inputs = self._proc(text=[text], return_tensors="pt", padding=True)
        with self._torch.no_grad():
            feats = self._model.get_text_features(**inputs)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats[0].tolist()


def build_embedder() -> BaseEmbedder:
    if os.environ.get("VIDGREP_CLIP", "mock").lower() == "real":
        return RealCLIP(os.environ.get("VIDGREP_CLIP_MODEL", "openai/clip-vit-base-patch32"))
    return MockCLIP()

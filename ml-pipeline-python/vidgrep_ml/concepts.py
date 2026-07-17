"""Shared concept space for the offline (mock) CLIP encoder.

Both the image encoder and the text encoder project into the *same* 512-dim
space by summing deterministic unit vectors, one per concept phrase. Frames carry
ground-truth ``labels`` (what's visually in them); a text query is matched to the
concept vocabulary. Cosine similarity between the two then ranks frames by how
well they answer the query — exactly the property real CLIP gives you, simulated
with zero dependencies.
"""

from __future__ import annotations

import hashlib
import math
import random

from .models import EMBED_DIM

# A small but expressive vocabulary of visual concepts.
CONCEPTS: list[str] = [
    "person", "red jacket", "blue shirt", "backpack", "suitcase",
    "car", "truck", "bus", "bicycle", "motorcycle",
    "turning left", "turning right", "stopped", "speeding",
    "dog", "cat", "crowd", "traffic light", "crosswalk",
    "night", "daytime", "rain", "fire", "smoke", "weapon",
]


def _concept_vector(phrase: str) -> list[float]:
    seed = int(hashlib.md5(phrase.encode()).hexdigest()[:8], 16)
    rng = random.Random(seed)
    v = [rng.gauss(0.0, 1.0) for _ in range(EMBED_DIM)]
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


# Precompute the concept basis once.
_BASIS: dict[str, list[float]] = {c: _concept_vector(c) for c in CONCEPTS}


def normalize(v: list[float]) -> list[float]:
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


def vector_for_labels(labels: list[str]) -> list[float]:
    acc = [0.0] * EMBED_DIM
    hit = False
    for label in labels:
        basis = _BASIS.get(label.lower().strip())
        if basis is None:
            basis = _concept_vector(label.lower().strip())  # unknown label still deterministic
        for i, x in enumerate(basis):
            acc[i] += x
        hit = True
    if not hit:
        return [0.0] * EMBED_DIM
    return normalize(acc)


def extract_concepts(text: str) -> list[str]:
    low = text.lower()
    found = [c for c in CONCEPTS if c in low]
    if found:
        return found
    # fallback: single-word concept tokens (e.g. "car", "dog")
    words = set(low.replace(",", " ").split())
    return [c for c in CONCEPTS if c in words]


def vector_for_text(text: str) -> list[float]:
    concepts = extract_concepts(text)
    if concepts:
        return vector_for_labels(concepts)
    # no known concept: hash the raw tokens into the space so search still returns
    # a stable (if weakly grounded) ordering.
    return vector_for_labels(text.lower().split()[:5])

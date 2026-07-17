#!/usr/bin/env python3
"""Offline end-to-end check of the vidgrep ML core.

Indexes a synthetic video (frames with ground-truth labels) through the CLIP-style
mock embedder + in-memory vector store, then runs natural-language searches and
checks the right frames come back — no video files, no CLIP weights, no services.

    python scripts/verify.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ml-pipeline-python"))

from vidgrep_ml.demo import demo_video  # noqa: E402
from vidgrep_ml.service import IndexService  # noqa: E402

_passed = 0
_failed = 0


def check(label: str, cond: bool) -> None:
    global _passed, _failed
    if cond:
        _passed += 1
        print(f"  [PASS] {label}")
    else:
        _failed += 1
        print(f"  [FAIL] {label}")


def main() -> int:
    print("=" * 70)
    print("vidgrep - offline end-to-end verification")
    print("=" * 70)
    svc = IndexService()
    frames = demo_video("demo")
    n = svc.index_frames(frames)
    print(f"  embedder={svc.embedder.name}  indexed {n} frames  ({svc.count('demo')} in store)")

    rj = svc.search("demo", "person in a red jacket", top_k=3)
    check("finds the red-jacket frame first", rj and rj[0].timestamp_ms == 3000)
    check("result carries a bounding box", bool(rj and rj[0].box and len(rj[0].box) == 4))

    car = svc.search("demo", "a car turning left", top_k=3)
    check("finds the turning-left car", bool(car) and car[0].timestamp_ms == 1000)

    bag = svc.search("demo", "someone carrying a backpack", top_k=3)
    check("surfaces a backpack frame", any("backpack" in r.labels for r in bag))

    night = svc.search("demo", "night time", top_k=4)
    check("returns top_k sorted by score",
          [r.score for r in night] == sorted((r.score for r in night), reverse=True))

    print("-" * 70)
    print(f"RESULT: {_passed} passed, {_failed} failed")
    return 0 if _failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""A scripted synthetic 'video' — frames at 1 fps, each with ground-truth labels.

Lets the whole pipeline (index + search) run head-to-tail offline with no real
video file or CLIP weights.
"""

from __future__ import annotations

from .models import Frame

# (timestamp_ms, labels)
DEMO_SCENES: list[tuple[int, list[str]]] = [
    (0, ["daytime", "car"]),
    (1000, ["daytime", "car", "turning left"]),
    (2000, ["daytime", "crosswalk", "person"]),
    (3000, ["person", "red jacket"]),
    (4000, ["person", "red jacket", "backpack"]),
    (5000, ["person", "backpack", "stopped"]),
    (6000, ["backpack", "crosswalk"]),
    (7000, ["crowd", "traffic light"]),
    (8000, ["bicycle", "turning right"]),
    (9000, ["car", "speeding", "night"]),
    (10000, ["truck", "night"]),
    (11000, ["dog", "person"]),
]


def demo_video(video_id: str = "demo") -> list[Frame]:
    return [Frame(video_id=video_id, timestamp_ms=ts, labels=labels) for ts, labels in DEMO_SCENES]

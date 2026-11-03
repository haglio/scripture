from __future__ import annotations

from dataclasses import dataclass

from scripture.frames import frame_of_ms


@dataclass
class Scene:
    start_frame: int
    end_frame: int


def scenes_from_splits(splits: list[int], total_frames: int) -> list[Scene]:
    boundaries = sorted({0, *splits, total_frames})
    return [Scene(boundaries[i], boundaries[i + 1]) for i in range(len(boundaries) - 1)]


def actions_by_scene(
    actions: list[dict],
    scenes: list[Scene],
    fps: float,
) -> dict[int, list[dict]]:
    buckets: dict[int, list[dict]] = {}
    for action in actions:
        frame = frame_of_ms(action["at"], fps)
        for i, scene in enumerate(scenes):
            if scene.start_frame <= frame < scene.end_frame:
                buckets.setdefault(i, []).append(action)
                break
    return buckets

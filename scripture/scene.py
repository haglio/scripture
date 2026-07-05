"""Scene data model and split-point logic."""

from dataclasses import dataclass


@dataclass
class Scene:
    start_frame: int
    end_frame: int


def scenes_from_splits(splits: list[int], total_frames: int) -> list[Scene]:
    """Derive an ordered list of scenes from a list of split-point frame indices."""
    boundaries = sorted(set([0] + splits + [total_frames]))
    return [Scene(boundaries[i], boundaries[i + 1]) for i in range(len(boundaries) - 1)]


def actions_by_scene(
    actions: list[dict],
    scenes: list[Scene],
    fps: float,
) -> dict[int, list[dict]]:
    """Bucket funscript actions into scenes by their frame position.

    Scenes with no actions get no entry (matching how the GUI treats an
    unprocessed scene).
    """
    buckets: dict[int, list[dict]] = {}
    for action in actions:
        frame = action["at"] / 1000.0 * fps
        for i, scene in enumerate(scenes):
            if scene.start_frame <= frame < scene.end_frame:
                buckets.setdefault(i, []).append(action)
                break
    return buckets

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

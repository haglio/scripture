"""The `.scripture` project file: what it holds, and how it reads and writes.

The schema used to live inside the Qt window -- the key names, the `str()`-keyed
scene indices, the tuple/list coercions, the numpy round trip -- which is the
least reachable file in the repo for a format another program opens: evolver
globs the sessions directory, reads the top-level video path and writes the
file back whole.

That makes the shape a contract rather than an implementation detail, so it
says which shape it is.  Evolver rewrites only a version it was written for
and leaves anything else alone, which is what turns a rename here into a
refusal there instead of a silent mangling; `tests/test_project.py` holds both
the version and the field name against this module.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np

from scripture.auto_funscript import (
    PipelineResult,
    pipeline_result_from_state,
    pipeline_result_to_state,
)
from scripture.motion_tracker import AxisDefinition, TrackingResult

Label = dict[str, object]

#: Which shape a saved project is.  A file written before this key existed is
#: the first, and reads as one.
PROJECT_FORMAT_VERSION = 1

#: Where the video this project was cut against is recorded.  Evolver moves
#: videos and repoints this in place, so the name is a contract with it.
VIDEO_PATH_FIELD = "video_path"


@dataclass
class ProjectDocument:
    """One video's worth of work: where it was split, and what was marked."""

    video_path: str | None = None
    splits: list[int] = field(default_factory=list)
    axes: dict[int, AxisDefinition] = field(default_factory=dict)
    actions: dict[int, list[dict]] = field(default_factory=dict)
    tracking: dict[int, TrackingResult] = field(default_factory=dict)
    labels: dict[int, dict[int, Label]] = field(default_factory=dict)
    auto: PipelineResult | None = None
    current_frame: int = 0


def state_from_document(document: ProjectDocument) -> dict:
    """The document as the plain JSON object that reaches the file."""
    return {
        "version": PROJECT_FORMAT_VERSION,
        VIDEO_PATH_FIELD: document.video_path,
        "splits": document.splits,
        "axes": {str(i): {"tip": list(a.tip), "base": list(a.base), "frame": a.frame}
                 for i, a in document.axes.items()},
        "actions": {str(i): a for i, a in document.actions.items()},
        "tracking": {str(i): _tracking_state(r) for i, r in document.tracking.items()},
        "ground_truth": {str(scene): {str(frame): _label_state(label)
                                      for frame, label in frames.items()}
                         for scene, frames in document.labels.items()},
        "auto": pipeline_result_to_state(document.auto) if document.auto else None,
        "current_frame": document.current_frame,
    }


def document_from_state(state: dict) -> ProjectDocument:
    """The document a saved state describes, tolerating what older ones omit."""
    auto = state.get("auto")
    return ProjectDocument(
        video_path=state[VIDEO_PATH_FIELD],
        splits=state["splits"],
        axes={int(k): AxisDefinition(tip=tuple(v["tip"]), base=tuple(v["base"]),
                                     frame=v.get("frame", 0))
              for k, v in state.get("axes", {}).items()},
        actions={int(k): v for k, v in state.get("actions", {}).items()},
        tracking={int(k): _tracking_from_state(v)
                  for k, v in state.get("tracking", {}).items()},
        labels={int(scene): {int(frame): _label_from_state(label)
                             for frame, label in frames.items()}
                for scene, frames in state.get("ground_truth", {}).items()},
        auto=pipeline_result_from_state(auto) if auto else None,
        current_frame=state.get("current_frame", 0),
    )


def _tracking_state(result: TrackingResult) -> dict:
    entry = {
        "timestamps_ms": result.timestamps_ms.tolist(),
        "positions": result.positions.tolist(),
    }
    if result.tip_coords is not None:
        entry["tip_coords"] = result.tip_coords.tolist()
    if result.base_coords is not None:
        entry["base_coords"] = result.base_coords.tolist()
    return entry


def _tracking_from_state(entry: dict) -> TrackingResult:
    return TrackingResult(
        timestamps_ms=np.array(entry["timestamps_ms"]),
        positions=np.array(entry["positions"]),
        tip_coords=np.array(entry["tip_coords"]) if "tip_coords" in entry else None,
        base_coords=np.array(entry["base_coords"]) if "base_coords" in entry else None,
    )


def _label_state(label: Label) -> dict:
    return {
        "tip": list(label["tip"]) if label.get("tip") else None,
        "base": list(label["base"]) if label.get("base") else None,
        "contact": list(label["contact"]) if label.get("contact") else None,
        "is_action": label.get("is_action", False),
    }


def _label_from_state(entry: dict) -> Label:
    return {
        "tip": tuple(entry["tip"]) if entry.get("tip") else None,
        "base": tuple(entry["base"]) if entry.get("base") else None,
        "contact": tuple(entry["contact"]) if entry.get("contact") else None,
        "is_action": entry.get("is_action", False),
    }


def save_project(path: str, state: dict) -> None:
    with open(path, "w") as f:
        json.dump(state, f, indent=2)


def load_project(path: str) -> dict:
    with open(path) as f:
        return json.load(f)

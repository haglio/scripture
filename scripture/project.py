from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from app_support import video_ids

from scripture.auto_funscript import (
    PipelineResult,
    pipeline_result_from_state,
    pipeline_result_to_state,
)
from scripture.motion_tracker import AxisDefinition, TrackingResult

Label = dict[str, object]

PROJECT_FORMAT_VERSION = 2
PROJECTS_DIR = Path(__file__).resolve().parent.parent / "sessions"

_log = logging.getLogger("scripture")
_DAMAGED_PROJECT_ERRORS = (OSError, ValueError, LookupError, TypeError, AttributeError, StopIteration)


@dataclass
class Project:

    video_path: str | None = None
    splits: list[int] = field(default_factory=list)
    axes: dict[int, AxisDefinition] = field(default_factory=dict)
    actions: dict[int, list[dict]] = field(default_factory=dict)
    tracking: dict[int, TrackingResult] = field(default_factory=dict)
    labels: dict[int, dict[int, Label]] = field(default_factory=dict)
    auto: PipelineResult | None = None
    current_frame: int = 0


def state_from_project(project: Project) -> dict:
    return {
        "version": PROJECT_FORMAT_VERSION,
        **_reference_to(project.video_path),
        "splits": project.splits,
        "axes": {str(i): {"tip": list(a.tip), "base": list(a.base), "frame": a.frame}
                 for i, a in project.axes.items()},
        "actions": {str(i): a for i, a in project.actions.items()},
        "tracking": {str(i): _tracking_state(r) for i, r in project.tracking.items()},
        "ground_truth": {str(scene): {str(frame): _label_state(label)
                                      for frame, label in frames.items()}
                         for scene, frames in project.labels.items()},
        "auto": pipeline_result_to_state(project.auto) if project.auto else None,
        "current_frame": project.current_frame,
    }


def project_from_state(state: dict) -> Project:
    auto = state.get("auto")
    return Project(
        video_path=_video_named_by(state),
        splits=state["splits"],
        axes={int(k): AxisDefinition(tip=tuple(v["tip"]), base=tuple(v["base"]),
                                     frame=v["frame"])
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


def _reference_to(video_path: str | None) -> dict:
    if video_path is None:
        return {video_ids.REFERENCE_PATH_KEY: None}
    return video_ids.reference_to(video_path)


def _video_named_by(state: dict) -> str | None:
    if video_ids.REFERENCE_ID_KEY not in state:
        return state[video_ids.REFERENCE_PATH_KEY]
    return video_ids.video_named_by(state)


def name_the_videos_by_id(projects_dir: Path) -> None:
    for path in sorted(Path(projects_dir).glob("*.scripture")):
        try:
            moved = video_ids.named_by_id(load_project(str(path)))
        except _DAMAGED_PROJECT_ERRORS:
            _log.warning("Could not read the project %s", path, exc_info=True)
            continue
        if moved is not None:
            save_project(str(path), {**moved, "version": PROJECT_FORMAT_VERSION})


def provenance_of_actions(project: Project) -> dict | None:
    auto = project.auto.provenance if project.auto else None
    stamps = [project.tracking[scene].provenance if scene in project.tracking else auto
              for scene in project.actions]
    if project.auto:
        stamps.append(auto)
    if None in stamps:
        return None
    first, *rest = stamps
    return {key: value if all(stamp[key] == value for stamp in rest) else None
            for key, value in first.items()}


def _tracking_state(result: TrackingResult) -> dict:
    entry = {
        "timestamps_ms": result.timestamps_ms.tolist(),
        "positions": result.positions.tolist(),
        "provenance": result.provenance,
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
        provenance=entry.get("provenance"),
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


def reopen_last_project(pointer: Path, open_project: Callable[[str], None]) -> None:
    path = pointer
    try:
        if pointer.exists():
            path = pointer.read_text().strip()
            if path and Path(path).exists():
                open_project(path)
    except _DAMAGED_PROJECT_ERRORS:
        _log.warning("Could not reopen the last project: %s", path, exc_info=True)

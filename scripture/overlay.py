from __future__ import annotations

import bisect
from itertools import pairwise

import numpy as np

from scripture.auto_funscript import PipelineResult
from scripture.frames import half_frame_ms, ms_of_frame
from scripture.motion_tracker import AxisDefinition, TrackingResult
from scripture.scene import Scene

_DETECTION_REACH = 6

_STILL = 0.001


def interpolate_actions(actions: list[dict], frame_ms: float,
                        half_a_frame: float) -> tuple[int, bool]:
    for a in actions:
        if abs(a["at"] - frame_ms) < half_a_frame:
            return a["pos"], True
    if frame_ms <= actions[0]["at"]:
        return actions[0]["pos"], False
    if frame_ms >= actions[-1]["at"]:
        return actions[-1]["pos"], False
    for earlier, later in pairwise(actions):
        if earlier["at"] <= frame_ms <= later["at"]:
            across = (frame_ms - earlier["at"]) / (later["at"] - earlier["at"])
            return round(earlier["pos"]
                         + across * (later["pos"] - earlier["pos"])), False
    return 50, False


def tracked_overlay(*, axis: AxisDefinition, scene: Scene, frame_idx: int, fps: float,
                    actions: list[dict], result: TrackingResult | None) -> dict | None:
    frame_ms = ms_of_frame(frame_idx, fps)
    half_a_frame = half_frame_ms(fps)
    local_idx = frame_idx - scene.start_frame

    if result is not None:
        if local_idx < 0 or local_idx >= len(result.positions):
            return None
        pos_frac = float(result.positions[local_idx])
        pos_100 = round(pos_frac * 100)
        stamp = result.timestamps_ms[local_idx]
        on_action = [a for a in actions if abs(a["at"] - stamp) < half_a_frame]
        is_action = bool(on_action)
        if is_action:
            pos_100 = on_action[0]["pos"]
    elif actions:
        pos_100, is_action = interpolate_actions(actions, frame_ms, half_a_frame)
        pos_frac = pos_100 / 100.0
    else:
        return None

    tip, base = _ends(axis, result, local_idx)
    contact = base + pos_frac * (tip - base)
    return {
        "axis": AxisDefinition(tip=_point(tip), base=_point(base), frame=axis.frame),
        "contact_pt": _point(contact),
        "pos": pos_100,
        "is_action": is_action,
        "direction": _direction(result, local_idx),
    }


def auto_overlay(*, result: PipelineResult, frame_idx: int, fps: float,
                 actions: list[dict], detection_frames: list[int],
                 last_anchor: np.ndarray | None) -> dict | None:
    local = frame_idx - result.start_frame
    if local < 0 or local >= len(result.positions):
        return None

    frame_ms = ms_of_frame(frame_idx, fps)
    half_a_frame = half_frame_ms(fps)
    lock = result.signal.lock[local]
    beliefs = result.signal.beliefs
    remembered = (beliefs[local] if local < len(beliefs)
                  and lock in ("contact", "coast") else None)
    return {
        "roi": result.signal.rois[local],
        "detections": _last_sighting(result, detection_frames, local),
        "pos": round(result.positions[local]),
        "active": lock != "none",
        "lock": lock,
        "belief": remembered,
        "belief_age_s": _age(remembered, last_anchor, local, fps),
        "is_action": any(abs(a["at"] - frame_ms) < half_a_frame for a in actions),
    }


def _ends(axis: AxisDefinition, result: TrackingResult | None,
          local_idx: int) -> tuple[np.ndarray, np.ndarray]:
    if (result is not None
            and result.tip_coords is not None
            and result.base_coords is not None
            and 0 <= local_idx < len(result.tip_coords)):
        return result.tip_coords[local_idx], result.base_coords[local_idx]
    return (np.array(axis.tip, dtype=np.float64),
            np.array(axis.base, dtype=np.float64))


def _direction(result: TrackingResult | None, local_idx: int) -> int:
    if result is None or not 0 < local_idx < len(result.positions):
        return 0
    delta = result.positions[local_idx] - result.positions[local_idx - 1]
    if abs(delta) <= _STILL:
        return 0
    return 1 if delta > 0 else -1


def _last_sighting(result: PipelineResult, detection_frames: list[int],
                   local: int) -> list | None:
    if not detection_frames:
        return None
    i = bisect.bisect_right(detection_frames, local) - 1
    if i < 0 or local - detection_frames[i] > _DETECTION_REACH:
        return None
    return result.signal.detections[detection_frames[i]]


def _age(remembered, last_anchor: np.ndarray | None, local: int,
         fps: float) -> float | None:
    if remembered is None or last_anchor is None:
        return None
    seen = last_anchor[local]
    return (local - seen) / fps if seen >= 0 else None


def _point(coords) -> tuple[int, int]:
    return round(coords[0]), round(coords[1])

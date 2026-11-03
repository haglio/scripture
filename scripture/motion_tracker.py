from __future__ import annotations

from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass

import cv2
import numpy as np

from scripture import cotracker_tracking
from scripture.frames import ms_of_frame
from scripture.funscript import made_by

TRACKER = "axis_tracking"
TRACKER_VERSION = "1"


@dataclass
class AxisDefinition:
    tip: tuple[int, int]
    base: tuple[int, int]
    frame: int = 0


@dataclass
class TrackingResult:
    timestamps_ms: np.ndarray
    positions: np.ndarray
    tip_coords: np.ndarray | None = None
    base_coords: np.ndarray | None = None
    provenance: dict | None = None


@contextmanager
def _released_after(capture):
    try:
        yield capture
    finally:
        capture.release()


def track_motion(video_path: str, axis: AxisDefinition,
                 start_frame: int, end_frame: int,
                 on_frame: Callable[[int], None] | None = None) -> TrackingResult:
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    ct_result = cotracker_tracking.cotrack_axis(video_path, axis, start_frame, end_frame)
    tip_coords = ct_result.tip_coords
    base_coords = ct_result.base_coords

    n_frames = end_frame - start_frame
    raw_positions = np.full(n_frames, 0.5)

    with _released_after(cap):
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        for i in range(n_frames):
            ret, frame = cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            base_f = base_coords[i]
            tip_f = tip_coords[i]
            axis_vec = tip_f - base_f
            if np.linalg.norm(axis_vec) < 1:
                continue

            t_vals, intensities = cotracker_tracking.sample_axis_intensity(
                gray, base_f, axis_vec, n=200, strip_w=20,
            )
            raw_positions[i] = cotracker_tracking.find_contact_gradient(t_vals, intensities)

            if on_frame is not None:
                on_frame(i)

    positions = cotracker_tracking.sanitize_positions(raw_positions, fps=fps)
    timestamps = ms_of_frame(start_frame + np.arange(n_frames), fps)

    return TrackingResult(
        timestamps_ms=timestamps,
        positions=positions,
        tip_coords=tip_coords,
        base_coords=base_coords,
        provenance=made_by(TRACKER, TRACKER_VERSION),
    )

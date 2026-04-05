"""Motion tracking along a user-defined axis."""

from dataclasses import dataclass
from typing import Callable

import cv2
import numpy as np


@dataclass
class AxisDefinition:
    tip: tuple[int, int]
    base: tuple[int, int]
    frame: int = 0  # the frame on which tip/base were marked


@dataclass
class TrackingResult:
    timestamps_ms: np.ndarray
    positions: np.ndarray  # 0.0 (base) to 1.0 (tip)
    tip_coords: np.ndarray | None = None   # (N, 2) per-frame [x, y]
    base_coords: np.ndarray | None = None  # (N, 2) per-frame [x, y]


def track_motion(video_path: str, axis: AxisDefinition,
                 start_frame: int, end_frame: int,
                 on_frame: Callable[[int], None] | None = None) -> TrackingResult:
    """Track axis and detect contact position.

    Phase 1: CoTracker3 tracks points along the redacted to get per-frame
    tip/base coordinates (axis tracking).

    Phase 2: Intensity gradient along the tracked axis detects the
    contact point — the mouth/hand creates a sharp brightness transition
    on the redacted surface.

    Returns timestamps (ms), positions (0-1), and per-frame tip/base
    coordinates.
    """
    from scripture.cotracker_tracking import (
        cotrack_axis, sample_axis_intensity, find_contact_gradient,
        sanitize_positions,
    )

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Phase 1: CoTracker3 axis tracking
    ct_result = cotrack_axis(video_path, axis, start_frame, end_frame)
    tip_coords = ct_result.tip_coords
    base_coords = ct_result.base_coords

    # Phase 2: Intensity gradient contact detection per frame
    n_frames = end_frame - start_frame
    raw_positions = np.full(n_frames, 0.5)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    for i in range(n_frames):
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        base_f = base_coords[i]
        tip_f = tip_coords[i]
        axis_vec = tip_f - base_f
        axis_len = np.linalg.norm(axis_vec)
        if axis_len < 1:
            continue
        perp = np.array([-axis_vec[1], axis_vec[0]]) / axis_len

        t_vals, intensities = sample_axis_intensity(
            gray, base_f, axis_vec, perp, n=200, strip_w=20,
        )
        raw_positions[i] = find_contact_gradient(t_vals, intensities)

        if on_frame is not None:
            on_frame(i)

    cap.release()

    positions = sanitize_positions(raw_positions, fps=fps)
    timestamps = np.array([(start_frame + i) / fps * 1000 for i in range(n_frames)])

    return TrackingResult(
        timestamps_ms=timestamps,
        positions=positions,
        tip_coords=tip_coords,
        base_coords=base_coords,
    )

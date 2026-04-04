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
                 margin: int = 80,
                 on_frame: Callable[[int], None] | None = None) -> TrackingResult:
    """Track axis and detect contact position using CoTracker3.

    Uses CoTracker3 to track points along the redacted, then derives the
    contact position from the visibility transition boundary — where
    tracked points go from visible (exposed redacted) to occluded (covered
    by hand/mouth).

    Returns timestamps (ms), positions (0-1), and per-frame tip/base
    coordinates.
    """
    from scripture.cotracker_tracking import cotrack_axis

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    result = cotrack_axis(video_path, axis, start_frame, end_frame)

    n_frames = end_frame - start_frame
    timestamps = np.array([(start_frame + i) / fps * 1000 for i in range(n_frames)])

    return TrackingResult(
        timestamps_ms=timestamps,
        positions=result.positions,
        tip_coords=result.tip_coords,
        base_coords=result.base_coords,
    )

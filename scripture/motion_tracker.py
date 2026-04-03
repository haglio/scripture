"""Optical-flow-based motion tracking along a user-defined axis."""

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


def compute_axis_unit_vector(axis: AxisDefinition) -> np.ndarray:
    """Return unit vector from tip toward base."""
    direction = np.array(axis.base, dtype=np.float64) - np.array(axis.tip, dtype=np.float64)
    length = np.linalg.norm(direction)
    if length == 0:
        raise ValueError("Tip and base must be different points")
    return direction / length


def build_roi_mask(frame_shape: tuple[int, int], axis: AxisDefinition, margin: int = 80) -> np.ndarray:
    """Build a rectangular mask around the tip-base axis with given margin."""
    h, w = frame_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    tip = np.array(axis.tip)
    base = np.array(axis.base)
    pts = np.array([tip, base])

    x_min = max(0, int(pts[:, 0].min()) - margin)
    x_max = min(w, int(pts[:, 0].max()) + margin)
    y_min = max(0, int(pts[:, 1].min()) - margin)
    y_max = min(h, int(pts[:, 1].max()) + margin)

    mask[y_min:y_max, x_min:x_max] = 255
    return mask


def build_axis_strip_mask(
    frame_shape: tuple[int, int],
    axis: AxisDefinition,
    half_width: int = 15,
) -> np.ndarray:
    """Build a narrow strip mask along the axis line.

    Creates a polygon mask +-half_width pixels perpendicular to the tip-base
    line.  Returns a uint8 mask (0 or 255) with shape frame_shape[:2].
    """
    h, w = frame_shape[:2]
    tip = np.array(axis.tip, dtype=np.float64)
    base = np.array(axis.base, dtype=np.float64)
    direction = base - tip
    length = np.linalg.norm(direction)
    if length == 0:
        raise ValueError("Tip and base must be different points")
    perp = np.array([-direction[1], direction[0]]) / length
    offset = perp * half_width
    corners = np.array([
        tip + offset,
        tip - offset,
        base - offset,
        base + offset,
    ], dtype=np.int32)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [corners], 255)
    return mask


def rolling_normalize(signal: np.ndarray, window_frames: int) -> np.ndarray:
    """Normalize signal using a rolling min/max window.

    Unlike global normalization, this preserves stroke amplitude even when
    the signal drifts.  Uses scipy.ndimage for O(n) performance.
    """
    from scipy.ndimage import minimum_filter1d, maximum_filter1d

    local_min = minimum_filter1d(signal, size=window_frames, mode="nearest")
    local_max = maximum_filter1d(signal, size=window_frames, mode="nearest")
    span = local_max - local_min
    eps = 1e-8
    safe_span = np.where(span > eps, span, 1.0)
    normalized = (signal - local_min) / safe_span
    return np.where(span > eps, normalized, 0.5)


def subtract_camera_motion(roi_motion: float, bg_motion: float) -> float:
    """Subtract estimated camera/body motion from ROI motion."""
    return roi_motion - bg_motion


def compute_crop_bounds(
    axis: AxisDefinition,
    half_width: int,
    frame_shape: tuple[int, int],
    padding: int = 30,
) -> tuple[int, int, int, int]:
    """Compute tight (y_min, y_max, x_min, x_max) crop around the axis strip.

    The bounding box covers the strip mask plus *padding* extra pixels,
    clamped to frame boundaries.
    """
    h, w = frame_shape[:2]
    tip = np.array(axis.tip, dtype=np.float64)
    base = np.array(axis.base, dtype=np.float64)
    expand = half_width + padding
    x_min = max(0, int(min(tip[0], base[0]) - expand))
    x_max = min(w, int(max(tip[0], base[0]) + expand))
    y_min = max(0, int(min(tip[1], base[1]) - expand))
    y_max = min(h, int(max(tip[1], base[1]) + expand))
    return y_min, y_max, x_min, x_max


def track_motion(video_path: str, axis: AxisDefinition,
                 start_frame: int, end_frame: int,
                 margin: int = 80,
                 on_frame: Callable[[int], None] | None = None) -> TrackingResult:
    """Track motion along the defined axis using dense optical flow.

    Uses a narrow strip mask along the axis (not a wide rectangle), crops
    frames before computing flow (10-14x faster), subtracts camera motion,
    and applies rolling normalization instead of global normalization.

    Returns timestamps (ms) and normalized position (0=base, 1=tip) arrays.
    """
    half_width = 15

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    ret, prev_frame = cap.read()
    if not ret:
        raise RuntimeError(f"Cannot read frame {start_frame}")

    frame_shape = prev_frame.shape[:2]
    strip_mask = build_axis_strip_mask(frame_shape, axis, half_width)
    y_min, y_max, x_min, x_max = compute_crop_bounds(
        axis, half_width, frame_shape, padding=30,
    )

    # Crop the strip mask to match the cropped frames
    strip_crop = strip_mask[y_min:y_max, x_min:x_max]
    strip_pixels = strip_crop > 0
    bg_pixels = ~strip_pixels

    unit_vec = compute_axis_unit_vector(axis)

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)[y_min:y_max, x_min:x_max]

    cumulative_displacement = 0.0
    displacements = [0.0]
    timestamps = [start_frame / fps * 1000]

    for frame_idx in range(start_frame + 1, end_frame):
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)[y_min:y_max, x_min:x_max]
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, gray, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
        )

        # Project flow onto axis direction
        projected = flow[:, :, 0] * unit_vec[0] + flow[:, :, 1] * unit_vec[1]

        # Median motion on the strip (redacted + hand)
        if strip_pixels.any():
            roi_motion = float(np.median(projected[strip_pixels]))
        else:
            roi_motion = 0.0

        # Median background motion (camera shake / body movement)
        if bg_pixels.any():
            bg_motion = float(np.median(projected[bg_pixels]))
        else:
            bg_motion = 0.0

        net_motion = subtract_camera_motion(roi_motion, bg_motion)
        cumulative_displacement += net_motion
        displacements.append(cumulative_displacement)
        timestamps.append(frame_idx / fps * 1000)

        prev_gray = gray

        if on_frame is not None:
            on_frame(frame_idx)

    cap.release()

    displacements = np.array(displacements)
    timestamps = np.array(timestamps)

    # Rolling normalization (preserves strokes even with drift)
    norm_window = max(3, min(int(fps * 10), len(displacements)))
    # Positive displacement along tip→base means moving toward base (pos=0)
    positions = 1.0 - rolling_normalize(displacements, norm_window)

    return TrackingResult(timestamps_ms=timestamps, positions=positions)

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
    tip_coords: np.ndarray | None = None   # (N, 2) per-frame [x, y]
    base_coords: np.ndarray | None = None  # (N, 2) per-frame [x, y]


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

    Uses CoTracker3 to track tip/base through the scene, then computes
    Farneback flow along the tracked axis for position estimation.

    Returns timestamps (ms), normalized positions, and per-frame
    tip/base coordinates.
    """
    from scripture.cotracker_tracking import cotrack_axis

    half_width = 15

    # ── Phase 1: Track axis via CoTracker3 ────────────────────────
    tip_coords, base_coords = cotrack_axis(video_path, axis, start_frame, end_frame)

    if on_frame is not None:
        on_frame(end_frame - start_frame)  # Phase 1 complete

    # ── Phase 2: Forward optical flow with per-frame strip masks ──
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    ret, prev_frame = cap.read()
    if not ret:
        raise RuntimeError(f"Cannot read frame {start_frame}")

    frame_shape = prev_frame.shape[:2]
    # Compute crop large enough to contain all tracked positions
    all_x = np.concatenate([tip_coords[:, 0], base_coords[:, 0]])
    all_y = np.concatenate([tip_coords[:, 1], base_coords[:, 1]])
    crop_pad = half_width + 30
    y_min = max(0, int(all_y.min()) - crop_pad)
    y_max = min(frame_shape[0], int(all_y.max()) + crop_pad)
    x_min = max(0, int(all_x.min()) - crop_pad)
    x_max = min(frame_shape[1], int(all_x.max()) + crop_pad)
    crop_shape = (y_max - y_min, x_max - x_min)

    unit_vec = compute_axis_unit_vector(axis)
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)[y_min:y_max, x_min:x_max]

    n_frames = end_frame - start_frame

    def _build_strip(tip_frame, base_frame):
        tip_crop = (int(round(tip_frame[0] - x_min)), int(round(tip_frame[1] - y_min)))
        base_crop = (int(round(base_frame[0] - x_min)), int(round(base_frame[1] - y_min)))
        fa = AxisDefinition(tip=tip_crop, base=base_crop)
        mask = build_axis_strip_mask(crop_shape, fa, half_width)
        return mask > 0

    cumulative_displacement = 0.0
    displacements = [0.0]
    timestamps = [start_frame / fps * 1000]
    progress = n_frames  # Phase 1 already counted

    for i in range(1, n_frames):
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)[y_min:y_max, x_min:x_max]

        strip_pixels = _build_strip(tip_coords[i], base_coords[i])
        bg_pixels = ~strip_pixels

        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, gray, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
        )
        projected = flow[:, :, 0] * unit_vec[0] + flow[:, :, 1] * unit_vec[1]

        roi_motion = float(np.median(projected[strip_pixels])) if strip_pixels.any() else 0.0
        bg_motion = float(np.median(projected[bg_pixels])) if bg_pixels.any() else 0.0

        net_motion = subtract_camera_motion(roi_motion, bg_motion)
        cumulative_displacement += net_motion
        displacements.append(cumulative_displacement)
        timestamps.append((start_frame + i) / fps * 1000)

        prev_gray = gray
        progress += 1
        if on_frame is not None:
            on_frame(progress)

    cap.release()

    displacements = np.array(displacements)
    timestamps = np.array(timestamps)

    norm_window = max(3, min(int(fps * 10), len(displacements)))
    positions = 1.0 - rolling_normalize(displacements, norm_window)

    return TrackingResult(
        timestamps_ms=timestamps,
        positions=positions,
        tip_coords=tip_coords,
        base_coords=base_coords,
    )

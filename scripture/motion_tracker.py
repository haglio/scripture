"""Optical-flow-based motion tracking along a user-defined axis."""

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class AxisDefinition:
    tip: tuple[int, int]
    base: tuple[int, int]


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


def track_motion(video_path: str, axis: AxisDefinition,
                 start_frame: int, end_frame: int,
                 margin: int = 80) -> TrackingResult:
    """Track motion along the defined axis using dense optical flow.

    Returns timestamps (ms) and normalized position (0=base, 1=tip) arrays.
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    ret, prev_frame = cap.read()
    if not ret:
        raise RuntimeError(f"Cannot read frame {start_frame}")

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    mask = build_roi_mask(prev_gray.shape, axis, margin)
    unit_vec = compute_axis_unit_vector(axis)
    axis_length = np.linalg.norm(
        np.array(axis.base, dtype=np.float64) - np.array(axis.tip, dtype=np.float64)
    )

    cumulative_displacement = 0.0
    displacements = [0.0]
    timestamps = [start_frame / fps * 1000]

    for frame_idx in range(start_frame + 1, end_frame):
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, gray, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        )

        # Project flow onto axis direction within ROI
        flow_x = flow[:, :, 0]
        flow_y = flow[:, :, 1]
        projected = flow_x * unit_vec[0] + flow_y * unit_vec[1]

        # Average projected motion within the ROI
        roi_pixels = mask > 0
        if roi_pixels.any():
            avg_motion = np.mean(projected[roi_pixels])
        else:
            avg_motion = 0.0

        cumulative_displacement += avg_motion
        displacements.append(cumulative_displacement)
        timestamps.append(frame_idx / fps * 1000)

        prev_gray = gray

    cap.release()

    displacements = np.array(displacements)
    timestamps = np.array(timestamps)

    # Normalize: project cumulative displacement to 0..1 range
    # Positive displacement along tip→base means moving toward base (pos=0)
    if displacements.max() - displacements.min() > 0:
        positions = 1.0 - (displacements - displacements.min()) / (displacements.max() - displacements.min())
    else:
        positions = np.full_like(displacements, 0.5)

    return TrackingResult(timestamps_ms=timestamps, positions=positions)

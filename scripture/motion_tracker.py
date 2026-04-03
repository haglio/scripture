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






class LKPointTracker:
    """Track a point across frames using Lucas-Kanade sparse optical flow.

    Seeds a cluster of feature points near the target, tracks them
    frame-to-frame, takes the median displacement, and re-seeds lost points.
    """

    _lk_params = dict(
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )

    def __init__(self, gray_frame: np.ndarray, center: tuple[int, int],
                 radius: int = 30, n_points: int = 50):
        self._prev_gray = gray_frame.copy()
        self._center = np.array(center, dtype=np.float64)
        self._radius = radius
        self._n_points = n_points
        self._points = self._seed_points(gray_frame, center, radius, n_points)

    @staticmethod
    def _seed_points(gray: np.ndarray, center: tuple[int, int],
                     radius: int, n_points: int) -> np.ndarray:
        """Find good feature points in a region around center."""
        h, w = gray.shape[:2]
        cx, cy = int(center[0]), int(center[1])
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(mask, (cx, cy), radius, 255, -1)
        corners = cv2.goodFeaturesToTrack(
            gray, maxCorners=n_points, qualityLevel=0.01,
            minDistance=3, mask=mask,
        )
        if corners is None or len(corners) == 0:
            # Fall back to a grid if no good features found
            pts = []
            for dy in range(-radius, radius + 1, max(1, radius // 3)):
                for dx in range(-radius, radius + 1, max(1, radius // 3)):
                    if dx * dx + dy * dy <= radius * radius:
                        px = max(0, min(w - 1, cx + dx))
                        py = max(0, min(h - 1, cy + dy))
                        pts.append([[float(px), float(py)]])
            return np.array(pts, dtype=np.float32)
        return corners

    def update(self, gray_frame: np.ndarray) -> tuple[float, float]:
        """Track points into gray_frame, return updated center (x, y)."""
        if len(self._points) == 0:
            return (float(self._center[0]), float(self._center[1]))

        new_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            self._prev_gray, gray_frame, self._points, None, **self._lk_params,
        )

        # Keep only points that were successfully tracked
        good_mask = status.ravel() == 1
        if good_mask.sum() > 0:
            old_good = self._points[good_mask]
            new_good = new_pts[good_mask]
            # Median displacement (robust to outliers)
            displacements = new_good.reshape(-1, 2) - old_good.reshape(-1, 2)
            med_dx = float(np.median(displacements[:, 0]))
            med_dy = float(np.median(displacements[:, 1]))
            self._center = self._center + np.array([med_dx, med_dy])
            self._points = new_good.reshape(-1, 1, 2)
        # else: keep previous center and points

        # Re-seed if we've lost too many points
        if len(self._points) < self._n_points // 3:
            cx, cy = int(round(self._center[0])), int(round(self._center[1]))
            self._points = self._seed_points(
                gray_frame, (cx, cy), self._radius, self._n_points,
            )

        self._prev_gray = gray_frame.copy()
        return (float(self._center[0]), float(self._center[1]))



def track_motion(video_path: str, axis: AxisDefinition,
                 start_frame: int, end_frame: int,
                 margin: int = 80,
                 on_frame: Callable[[int], None] | None = None) -> TrackingResult:
    """Track motion along the defined axis using dense optical flow.

    Tracks the base point per-frame using Lucas-Kanade sparse optical flow,
    bidirectionally from axis.frame.  Builds a per-frame strip mask from the
    tracked coordinates.

    Returns timestamps (ms), normalized positions, and per-frame
    tip/base coordinates.
    """
    half_width = 15

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    # ── Read the representative frame and set up geometry ─────────
    cap.set(cv2.CAP_PROP_POS_FRAMES, axis.frame)
    ret_ref, ref_frame = cap.read()
    if not ret_ref:
        raise RuntimeError(f"Cannot read reference frame {axis.frame}")
    ref_gray = cv2.cvtColor(ref_frame, cv2.COLOR_BGR2GRAY)
    axis_vector = np.array(axis.tip, dtype=np.float64) - np.array(axis.base, dtype=np.float64)

    frame_shape = ref_frame.shape[:2]
    y_min, y_max, x_min, x_max = compute_crop_bounds(
        axis, half_width, frame_shape, padding=80,
    )
    crop_shape = (y_max - y_min, x_max - x_min)
    unit_vec = compute_axis_unit_vector(axis)

    n_frames = end_frame - start_frame
    ref_local = max(0, min(axis.frame - start_frame, n_frames - 1))
    base_in_crop_ref = (axis.base[0] - x_min, axis.base[1] - y_min)

    # ── Phase 1: Track base coordinates bidirectionally from axis.frame ─
    progress = 0
    base_coords_crop = [None] * n_frames
    base_coords_crop[ref_local] = base_in_crop_ref
    ref_crop_gray = ref_gray[y_min:y_max, x_min:x_max]

    # Backward: ref_local-1 down to 0 (LK tracker, needs seeks)
    if ref_local > 0:
        tracker_bwd = LKPointTracker(ref_crop_gray, base_in_crop_ref, radius=40)
        for i in range(ref_local - 1, -1, -1):
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame + i)
            ret, frame = cap.read()
            if not ret:
                base_coords_crop[i] = base_coords_crop[i + 1]
            else:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)[y_min:y_max, x_min:x_max]
                pos = tracker_bwd.update(gray)
                base_coords_crop[i] = pos
            progress += 1
            if on_frame is not None:
                on_frame(progress)

    # Forward: ref_local+1 to end (LK tracker, sequential read)
    if ref_local < n_frames - 1:
        tracker_fwd = LKPointTracker(ref_crop_gray, base_in_crop_ref, radius=40)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame + ref_local + 1)
        for i in range(ref_local + 1, n_frames):
            ret, frame = cap.read()
            if not ret:
                base_coords_crop[i] = base_coords_crop[i - 1]
            else:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)[y_min:y_max, x_min:x_max]
                pos = tracker_fwd.update(gray)
                base_coords_crop[i] = pos
            progress += 1
            if on_frame is not None:
                on_frame(progress)

    # Convert to frame coords
    all_bases = [(bc[0] + x_min, bc[1] + y_min) for bc in base_coords_crop]
    all_tips = [(bc[0] + x_min + axis_vector[0], bc[1] + y_min + axis_vector[1])
                for bc in base_coords_crop]

    def _build_strip(base_c, tip_c):
        fa = AxisDefinition(
            tip=(int(round(tip_c[0])), int(round(tip_c[1]))),
            base=(int(round(base_c[0])), int(round(base_c[1]))),
        )
        mask = build_axis_strip_mask(crop_shape, fa, half_width)
        return mask > 0

    # ── Phase 2: Forward optical flow with pre-tracked strip masks ─
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    ret, prev_frame = cap.read()
    if not ret:
        raise RuntimeError(f"Cannot read frame {start_frame}")
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)[y_min:y_max, x_min:x_max]

    cumulative_displacement = 0.0
    displacements = [0.0]
    timestamps = [start_frame / fps * 1000]

    for i in range(1, n_frames):
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)[y_min:y_max, x_min:x_max]

        # Per-frame strip mask from pre-tracked coordinates
        bc = base_coords_crop[i]
        tc = (bc[0] + axis_vector[0], bc[1] + axis_vector[1])
        strip_pixels = _build_strip(bc, tc)
        bg_pixels = ~strip_pixels

        # Optical flow
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

    # Rolling normalization (preserves strokes even with drift)
    norm_window = max(3, min(int(fps * 10), len(displacements)))
    positions = 1.0 - rolling_normalize(displacements, norm_window)

    return TrackingResult(
        timestamps_ms=timestamps,
        positions=positions,
        tip_coords=np.array(all_tips, dtype=np.float64),
        base_coords=np.array(all_bases, dtype=np.float64),
    )

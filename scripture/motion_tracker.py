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


def extract_base_template(
    gray_frame: np.ndarray,
    base_point: tuple[int, int],
    radius: int = 25,
) -> np.ndarray:
    """Extract a square grayscale template patch centered on base_point.

    Returns a (2*radius+1, 2*radius+1) uint8 array.  Pixels outside
    the frame are filled via replicate border padding.
    """
    bx, by = int(base_point[0]), int(base_point[1])
    h, w = gray_frame.shape[:2]
    size = 2 * radius + 1

    # Compute how much padding is needed on each side
    pad_left = max(0, radius - bx)
    pad_right = max(0, (bx + radius + 1) - w)
    pad_top = max(0, radius - by)
    pad_bottom = max(0, (by + radius + 1) - h)

    if pad_left or pad_right or pad_top or pad_bottom:
        padded = cv2.copyMakeBorder(
            gray_frame, pad_top, pad_bottom, pad_left, pad_right,
            cv2.BORDER_REPLICATE,
        )
        # Shift coordinates into padded frame
        bx += pad_left
        by += pad_top
        return padded[by - radius:by + radius + 1, bx - radius:bx + radius + 1].copy()

    return gray_frame[by - radius:by + radius + 1, bx - radius:bx + radius + 1].copy()


def track_base_in_frame(
    gray_crop: np.ndarray,
    template: np.ndarray,
    last_pos_in_crop: tuple[int, int],
    search_radius: int = 60,
    min_confidence: float = 0.3,
) -> tuple[tuple[int, int], float]:
    """Find the base position in gray_crop via template matching.

    Searches within a window around last_pos_in_crop.  Uses normalised
    cross-correlation.  Returns (new_pos_in_crop, confidence).  Falls
    back to last_pos_in_crop when confidence < min_confidence.
    """
    th, tw = template.shape[:2]
    half_t = tw // 2
    half_t_h = th // 2
    ch, cw = gray_crop.shape[:2]

    lx, ly = int(last_pos_in_crop[0]), int(last_pos_in_crop[1])

    # Search region bounds (must leave room for the template to fit)
    sx_min = max(half_t, lx - search_radius)
    sx_max = min(cw - half_t - 1, lx + search_radius)
    sy_min = max(half_t_h, ly - search_radius)
    sy_max = min(ch - half_t_h - 1, ly + search_radius)

    # If search region is too small for template matching, fall back
    region_w = sx_max - sx_min + tw
    region_h = sy_max - sy_min + th
    if region_w < tw or region_h < th:
        return (lx, ly), 0.0

    search_sub = gray_crop[sy_min - half_t_h:sy_max + half_t_h + 1,
                           sx_min - half_t:sx_max + half_t + 1]

    if search_sub.shape[0] < th or search_sub.shape[1] < tw:
        return (lx, ly), 0.0

    result = cv2.matchTemplate(search_sub, template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)

    # max_loc is (x, y) offset in the result map
    new_x = sx_min + max_loc[0]
    new_y = sy_min + max_loc[1]

    if max_val >= min_confidence:
        return (new_x, new_y), float(max_val)
    return (lx, ly), float(max_val)


def track_motion(video_path: str, axis: AxisDefinition,
                 start_frame: int, end_frame: int,
                 margin: int = 80,
                 on_frame: Callable[[int], None] | None = None) -> TrackingResult:
    """Track motion along the defined axis using dense optical flow.

    Template-matches the base point per-frame so the axis follows the
    anatomy.  Tracks **bidirectionally** from axis.frame (where the user
    defined the axis) to avoid drift from a distant start_frame.

    Returns timestamps (ms), normalized positions, and per-frame
    tip/base coordinates.
    """
    half_width = 15
    template_radius = 25

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    # ── Extract base template from the representative frame ───────
    cap.set(cv2.CAP_PROP_POS_FRAMES, axis.frame)
    ret_ref, ref_frame = cap.read()
    if not ret_ref:
        raise RuntimeError(f"Cannot read reference frame {axis.frame}")
    ref_gray = cv2.cvtColor(ref_frame, cv2.COLOR_BGR2GRAY)
    base_template = extract_base_template(ref_gray, axis.base, radius=template_radius)
    axis_vector = np.array(axis.tip, dtype=np.float64) - np.array(axis.base, dtype=np.float64)

    # ── Set up the fixed crop ─────────────────────────────────────
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    ret, first_frame = cap.read()
    if not ret:
        raise RuntimeError(f"Cannot read frame {start_frame}")

    frame_shape = first_frame.shape[:2]
    y_min, y_max, x_min, x_max = compute_crop_bounds(
        axis, half_width, frame_shape, padding=80,
    )
    crop_shape = (y_max - y_min, x_max - x_min)
    unit_vec = compute_axis_unit_vector(axis)

    n_frames = end_frame - start_frame
    ref_local = max(0, min(axis.frame - start_frame, n_frames - 1))
    base_in_crop_ref = (axis.base[0] - x_min, axis.base[1] - y_min)

    # ── Phase 1: Track base coordinates bidirectionally from axis.frame ─
    base_coords_crop = [None] * n_frames
    base_coords_crop[ref_local] = base_in_crop_ref

    # Backward: ref_local-1 down to 0 (seeks per frame, template match only)
    last = base_in_crop_ref
    for i in range(ref_local - 1, -1, -1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame + i)
        ret, frame = cap.read()
        if not ret:
            base_coords_crop[i] = last
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)[y_min:y_max, x_min:x_max]
        last, _ = track_base_in_frame(gray, base_template, last, search_radius=60)
        base_coords_crop[i] = last

    # Forward: ref_local+1 to end (seeks per frame, template match only)
    last = base_in_crop_ref
    for i in range(ref_local + 1, n_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame + i)
        ret, frame = cap.read()
        if not ret:
            base_coords_crop[i] = last
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)[y_min:y_max, x_min:x_max]
        last, _ = track_base_in_frame(gray, base_template, last, search_radius=60)
        base_coords_crop[i] = last

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

        if on_frame is not None:
            on_frame(start_frame + i)

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

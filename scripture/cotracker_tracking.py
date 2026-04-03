"""CoTracker3-based axis tracking for per-frame tip/base coordinates."""

from typing import Callable

import cv2
import numpy as np

from scripture.motion_tracker import AxisDefinition

_cotracker_model = None


def _get_model():
    """Load CoTracker3 (cached after first call)."""
    import torch

    global _cotracker_model
    if _cotracker_model is None:
        _cotracker_model = torch.hub.load(
            "facebookresearch/co-tracker", "cotracker3_offline",
        )
        _cotracker_model = _cotracker_model.to("cuda")
    return _cotracker_model


def fit_axis_from_points(
    points: np.ndarray,
    t_params: np.ndarray,
    visible: np.ndarray,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Fit a line through visible tracked points and extrapolate tip/base.

    points:   (N, 2) array of [x, y] positions for each tracked point.
    t_params: (N,) parametric positions along the axis (0=base, 1=tip).
    visible:  (N,) boolean mask of which points are visible.

    Returns (tip_xy, base_xy) as 1-D arrays, or None if <2 points visible.
    """
    mask = visible.astype(bool)
    if mask.sum() < 2:
        return None
    t_vis = t_params[mask]
    pts_vis = points[mask]
    # Fit x = a_x * t + b_x, y = a_y * t + b_y via least squares
    A = np.column_stack([t_vis, np.ones(len(t_vis))])
    coeffs_x, _, _, _ = np.linalg.lstsq(A, pts_vis[:, 0], rcond=None)
    coeffs_y, _, _, _ = np.linalg.lstsq(A, pts_vis[:, 1], rcond=None)
    base = np.array([coeffs_x[1], coeffs_y[1]])            # t=0
    tip = np.array([coeffs_x[0] + coeffs_x[1],             # t=1
                    coeffs_y[0] + coeffs_y[1]])
    return tip, base


def scale_coords(
    coords: np.ndarray,
    from_size: tuple[int, int],
    to_size: tuple[int, int],
) -> np.ndarray:
    """Scale (x, y) coordinates between two frame sizes.

    from_size and to_size are (height, width).
    """
    from_h, from_w = from_size
    to_h, to_w = to_size
    scale_x = to_w / from_w
    scale_y = to_h / from_h
    out = coords.copy().astype(np.float64)
    out[..., 0] *= scale_x
    out[..., 1] *= scale_y
    return out


_TARGET_LONG_SIDE = 384


def _read_scene_frames(video_path: str, start_frame: int, end_frame: int):
    """Read frames from video, downscale to _TARGET_LONG_SIDE on long edge.

    Returns (frames_tensor, orig_size, scaled_size).
    frames_tensor: (1, T, 3, H, W) float32 on GPU.
    """
    import torch

    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    n_frames = end_frame - start_frame

    ret, first = cap.read()
    if not ret:
        raise RuntimeError(f"Cannot read frame {start_frame}")
    orig_h, orig_w = first.shape[:2]
    scale = _TARGET_LONG_SIDE / max(orig_h, orig_w)
    new_w, new_h = int(orig_w * scale), int(orig_h * scale)
    orig_size = (orig_h, orig_w)
    scaled_size = (new_h, new_w)

    frames = []
    resized = cv2.resize(first, (new_w, new_h))
    frames.append(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))

    for _ in range(1, n_frames):
        ret, frame = cap.read()
        if not ret:
            break
        resized = cv2.resize(frame, (new_w, new_h))
        frames.append(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))

    cap.release()

    arr = np.stack(frames)  # (T, H, W, 3)
    tensor = torch.from_numpy(arr).permute(0, 3, 1, 2).float()  # (T, 3, H, W)
    tensor = tensor.unsqueeze(0).to("cuda")  # (1, T, 3, H, W)
    return tensor, orig_size, scaled_size


def cotrack_axis(
    video_path: str,
    axis: AxisDefinition,
    start_frame: int,
    end_frame: int,
    n_points: int = 8,
    on_progress: Callable[[int], None] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Track tip and base through a scene using CoTracker3.

    Places n_points along the axis on the representative frame, tracks
    all of them, then reconstructs per-frame tip/base via line fitting
    from whichever points are visible.

    Returns (tip_coords, base_coords) as (N_frames, 2) arrays in
    original frame coordinates.
    """
    import torch

    model = _get_model()
    n_frames = end_frame - start_frame
    ref_local = max(0, min(axis.frame - start_frame, n_frames - 1))

    # Read and downscale video
    video, orig_size, scaled_size = _read_scene_frames(video_path, start_frame, end_frame)
    actual_frames = video.shape[1]

    # Generate query points along the axis
    tip = np.array(axis.tip, dtype=np.float64)
    base = np.array(axis.base, dtype=np.float64)
    t_params = np.linspace(0, 1, n_points)  # 0=base, 1=tip
    axis_points = np.array([base + t * (tip - base) for t in t_params])  # (N, 2) in orig coords
    axis_points_scaled = scale_coords(axis_points, orig_size, scaled_size)

    # Build queries: (1, N, 3) as (frame_idx, x, y)
    queries = torch.zeros(1, n_points, 3, device="cuda")
    queries[0, :, 0] = ref_local
    queries[0, :, 1] = torch.from_numpy(axis_points_scaled[:, 0]).float()
    queries[0, :, 2] = torch.from_numpy(axis_points_scaled[:, 1]).float()

    # Run CoTracker3
    with torch.no_grad():
        pred_tracks, pred_visibility = model(
            video, queries=queries, backward_tracking=True,
        )
    # pred_tracks: (1, T, N, 2), pred_visibility: (1, T, N)
    tracks = pred_tracks[0].cpu().numpy()       # (T, N, 2)
    visibility = pred_visibility[0].cpu().numpy()  # (T, N)

    # Free GPU memory
    del video, pred_tracks, pred_visibility, queries
    torch.cuda.empty_cache()

    # Reconstruct per-frame tip/base via line fitting
    tip_coords = np.zeros((actual_frames, 2), dtype=np.float64)
    base_coords = np.zeros((actual_frames, 2), dtype=np.float64)
    last_tip = axis_points_scaled[n_points - 1]
    last_base = axis_points_scaled[0]

    for i in range(actual_frames):
        vis = visibility[i] > 0.5
        result = fit_axis_from_points(tracks[i], t_params, vis)
        if result is not None:
            last_tip, last_base = result
        tip_coords[i] = last_tip
        base_coords[i] = last_base

    # Scale back to original resolution
    tip_coords = scale_coords(tip_coords, scaled_size, orig_size)
    base_coords = scale_coords(base_coords, scaled_size, orig_size)

    return tip_coords, base_coords

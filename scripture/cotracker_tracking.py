"""CoTracker3-based axis tracking for per-frame tip/base coordinates."""

from dataclasses import dataclass
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


def visibility_to_position(t_params: np.ndarray, vis: np.ndarray) -> int:
    """Convert per-point visibility into a contact position (0-100).

    The contact point is where the hand/mouth meets the redacted.  Points
    on the base-side of the contact are exposed (visible), points on the
    tip-side are covered (occluded).  The position is the t-parameter at
    the visibility transition boundary.

    t_params: (N,) parametric positions along axis (0=base, 1=tip).
    vis:      (N,) visibility scores (>0.5 = visible).

    Returns pos 0-100 where 0=base, 100=tip.
    """
    visible = vis > 0.5
    if visible.all():
        return 100  # nothing covering the redacted
    if not visible.any():
        return 0    # fully covered

    # The contact point is the leading edge of the occluded region — where
    # the hand first meets the exposed redacted.  Walk from base (t=0) to
    # tip (t=1): the last visible point before a run of occluded points
    # is the contact boundary.
    #
    # If the base end is occluded (hand from below), walk from tip
    # downward instead.
    n = len(t_params)
    # Check which end is visible to determine scan direction
    base_visible = visible[:n // 4].sum() > visible[3 * n // 4:].sum()

    if base_visible:
        # Hand from tip side: find last visible point scanning base→tip
        for i in range(n - 1, -1, -1):
            if visible[i]:
                return int(round(t_params[i] * 100))
    else:
        # Hand from base side: find last visible point scanning tip→base
        for i in range(n):
            if visible[i]:
                return int(round(t_params[i] * 100))

    return 50


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
_MAX_CHUNK_FRAMES = 300  # CoTracker3 internal attention uses ~50x the video tensor size


def _get_video_geometry(video_path: str, start_frame: int):
    """Read one frame to determine original and scaled sizes."""
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError(f"Cannot read frame {start_frame}")
    orig_h, orig_w = frame.shape[:2]
    scale = _TARGET_LONG_SIDE / max(orig_h, orig_w)
    new_w, new_h = int(orig_w * scale), int(orig_h * scale)
    return (orig_h, orig_w), (new_h, new_w)


def _read_chunk(video_path: str, start_frame: int, n_frames: int,
                scaled_size: tuple[int, int]):
    """Read n_frames starting at start_frame, downscale, return GPU tensor."""
    import torch

    new_h, new_w = scaled_size
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frames = []
    for _ in range(n_frames):
        ret, frame = cap.read()
        if not ret:
            break
        resized = cv2.resize(frame, (new_w, new_h))
        frames.append(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
    cap.release()

    arr = np.stack(frames)  # (T, H, W, 3)
    tensor = torch.from_numpy(arr).permute(0, 3, 1, 2).float()  # (T, 3, H, W)
    return tensor.unsqueeze(0).to("cuda")  # (1, T, 3, H, W)


def _track_chunk(model, video_chunk, queries):
    """Run CoTracker3 on a single chunk, return tracks and visibility on CPU."""
    import torch

    with torch.no_grad():
        pred_tracks, pred_visibility = model(
            video_chunk, queries=queries, backward_tracking=True,
        )
    tracks = pred_tracks[0].cpu().numpy()
    visibility = pred_visibility[0].cpu().numpy()
    del video_chunk, pred_tracks, pred_visibility
    torch.cuda.empty_cache()
    return tracks, visibility


@dataclass
class CoTrackResult:
    """Full output from cotrack_axis."""
    tip_coords: np.ndarray     # (N_frames, 2) in original frame coords
    base_coords: np.ndarray    # (N_frames, 2) in original frame coords
    visibility: np.ndarray     # (N_frames, n_points) visibility scores
    t_params: np.ndarray       # (n_points,) parametric positions 0=base, 1=tip
    positions: np.ndarray      # (N_frames,) contact position 0.0-1.0


def cotrack_axis(
    video_path: str,
    axis: AxisDefinition,
    start_frame: int,
    end_frame: int,
    n_points: int = 30,
    on_progress: Callable[[int], None] | None = None,
) -> CoTrackResult:
    """Track tip/base and detect contact position using CoTracker3.

    Places n_points along the axis on the representative frame, tracks
    all of them, reconstructs per-frame tip/base via line fitting from
    visible points, and derives the contact position from the visibility
    transition boundary.

    For long scenes, processes in chunks of _MAX_CHUNK_FRAMES.
    """
    import torch

    model = _get_model()
    n_frames = end_frame - start_frame
    ref_local = max(0, min(axis.frame - start_frame, n_frames - 1))

    orig_size, scaled_size = _get_video_geometry(video_path, start_frame)

    # Generate query points along the axis (in scaled coordinates)
    tip = np.array(axis.tip, dtype=np.float64)
    base = np.array(axis.base, dtype=np.float64)
    t_params = np.linspace(0, 1, n_points)
    axis_points = np.array([base + t * (tip - base) for t in t_params])
    axis_points_scaled = scale_coords(axis_points, orig_size, scaled_size)

    # Allocate output arrays (in scaled coords, converted at the end)
    all_tracks = np.zeros((n_frames, n_points, 2), dtype=np.float64)
    all_vis = np.zeros((n_frames, n_points), dtype=np.float64)

    if n_frames <= _MAX_CHUNK_FRAMES:
        # Small enough to process in one shot
        video = _read_chunk(video_path, start_frame, n_frames, scaled_size)
        queries = torch.zeros(1, n_points, 3, device="cuda")
        queries[0, :, 0] = ref_local
        queries[0, :, 1] = torch.from_numpy(axis_points_scaled[:, 0]).float()
        queries[0, :, 2] = torch.from_numpy(axis_points_scaled[:, 1]).float()
        tracks, vis = _track_chunk(model, video, queries)
        actual = min(tracks.shape[0], n_frames)
        all_tracks[:actual] = tracks[:actual]
        all_vis[:actual] = vis[:actual]
    else:
        # Process in overlapping chunks, expanding outward from the
        # reference frame.  The chunk containing ref_local is processed
        # first (seeded from the user's axis points).  Subsequent chunks
        # are seeded from the overlap region of the previous chunk,
        # ensuring continuity without drift.
        chunk_size = _MAX_CHUNK_FRAMES
        overlap = min(100, chunk_size // 4)

        def _run_chunk(c_start, c_end, q_frame, q_points):
            c_len = c_end - c_start
            video = _read_chunk(video_path, start_frame + c_start, c_len, scaled_size)
            queries = torch.zeros(1, n_points, 3, device="cuda")
            queries[0, :, 0] = q_frame
            queries[0, :, 1] = torch.from_numpy(q_points[:, 0].copy()).float()
            queries[0, :, 2] = torch.from_numpy(q_points[:, 1].copy()).float()
            tracks, vis = _track_chunk(model, video, queries)
            actual = min(tracks.shape[0], c_len)
            for i in range(actual):
                gi = c_start + i
                if gi < n_frames:
                    all_tracks[gi] = tracks[i]
                    all_vis[gi] = vis[i]

        # 1. Process the chunk containing the reference frame
        ref_chunk_start = max(0, ref_local - chunk_size // 2)
        ref_chunk_end = min(n_frames, ref_chunk_start + chunk_size)
        ref_chunk_start = max(0, ref_chunk_end - chunk_size)
        _run_chunk(ref_chunk_start, ref_chunk_end,
                   ref_local - ref_chunk_start, axis_points_scaled)

        # 2. Expand forward from ref chunk
        pos = ref_chunk_end - overlap
        while pos < n_frames:
            c_end = min(pos + chunk_size, n_frames)
            # Seed from overlap region (tracked by previous chunk)
            seed_frame = pos  # global index where we have good data
            q_points = all_tracks[seed_frame]
            _run_chunk(pos, c_end, 0, q_points)
            pos = c_end - overlap
            if c_end == n_frames:
                break

        # 3. Expand backward from ref chunk
        pos = ref_chunk_start + overlap
        while pos > 0:
            c_start = max(0, pos - chunk_size)
            # Seed from overlap region (tracked by previous chunk)
            seed_frame = pos - 1  # global index where we have good data
            q_frame = seed_frame - c_start
            q_points = all_tracks[seed_frame]
            _run_chunk(c_start, pos, q_frame, q_points)
            pos = c_start + overlap
            if c_start == 0:
                break

    # Use outermost visible tracked points as tip/base for the overlay.
    # This shows the visible portion of the redacted truthfully — no
    # extrapolation to guess where occluded endpoints are.
    # The pos value comes from visibility_to_position, which is
    # independent of these coordinates.
    tip_coords = np.zeros((n_frames, 2), dtype=np.float64)
    base_coords = np.zeros((n_frames, 2), dtype=np.float64)
    positions = np.full(n_frames, 0.5)
    last_tip = axis_points_scaled[n_points - 1]
    last_base = axis_points_scaled[0]

    for i in range(n_frames):
        visible = all_vis[i] > 0.5
        if visible.any():
            vis_indices = np.where(visible)[0]
            last_base = all_tracks[i, vis_indices[0]]   # lowest t visible
            last_tip = all_tracks[i, vis_indices[-1]]    # highest t visible
        tip_coords[i] = last_tip
        base_coords[i] = last_base
        positions[i] = visibility_to_position(t_params, all_vis[i]) / 100.0

    # Scale back to original resolution
    tip_coords = scale_coords(tip_coords, scaled_size, orig_size)
    base_coords = scale_coords(base_coords, scaled_size, orig_size)

    return CoTrackResult(
        tip_coords=tip_coords,
        base_coords=base_coords,
        visibility=all_vis,
        t_params=t_params,
        positions=positions,
    )

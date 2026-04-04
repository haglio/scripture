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


def motion_divergence_position(
    t_params: np.ndarray,
    tracks_prev: np.ndarray,
    tracks_curr: np.ndarray,
) -> int:
    """Compute contact position from per-point motion between two frames.

    Points on the exposed redacted are relatively stationary.  Points covered
    by the hand/mouth move WITH the hand.  The contact boundary is where
    the motion pattern changes.

    t_params:    (N,) parametric positions along axis (0=base, 1=tip).
    tracks_prev: (N, 2) point positions on previous frame.
    tracks_curr: (N, 2) point positions on current frame.

    Returns pos 0-100 where 0=base, 100=tip.
    """
    displacements = np.linalg.norm(tracks_curr - tracks_prev, axis=1)
    median_disp = np.median(displacements)

    # If nothing is moving, nothing is covering the redacted
    if median_disp < 0.5:
        return 100

    # Classify each point as "moving" (with hand) or "stationary" (exposed)
    # Use the median as a threshold — points moving more than the median
    # are likely under the hand
    threshold = max(1.0, median_disp * 0.5)
    moving = displacements > threshold

    if not moving.any():
        return 100
    if moving.all():
        return 0

    # The contact point is the boundary between stationary and moving.
    # Find the lowest t-param of a moving point that's above a stationary point.
    stationary_t = t_params[~moving]
    moving_t = t_params[moving]

    # The contact is at the edge of the moving region closest to stationary
    if stationary_t.min() < moving_t.min():
        # Stationary at base, moving at tip → contact = min of moving t
        contact_t = float(moving_t.min())
    else:
        # Stationary at tip, moving at base → contact = max of moving t
        contact_t = float(moving_t.max())

    return int(round(contact_t * 100))


def sanitize_positions(positions: np.ndarray, fps: float = 30.0) -> np.ndarray:
    """Apply temporal smoothing and physical constraints to raw pos signal.

    - Smooths the signal to remove single-frame noise
    - Enforces a maximum speed (full stroke in no less than ~0.25s)
    - Clamps output to [0, 1]
    """
    from scipy.signal import savgol_filter

    n = len(positions)
    if n < 5:
        return positions.copy()

    # 1. Median filter to kill single-frame spikes (non-linear, preserves edges)
    from scipy.ndimage import median_filter
    cleaned = median_filter(positions, size=5, mode="nearest")

    # 2. Light Savitzky-Golay smoothing
    window = max(5, min(int(fps * 0.2), n))
    if window % 2 == 0:
        window += 1
    window = min(window, n)
    if window >= 5:
        cleaned = savgol_filter(cleaned, window, 3)

    # 3. Enforce max speed: full stroke (0→1) takes at least 0.25s
    max_delta_per_frame = 1.0 / (fps * 0.25)
    for i in range(1, n):
        delta = cleaned[i] - cleaned[i - 1]
        if abs(delta) > max_delta_per_frame:
            cleaned[i] = cleaned[i - 1] + np.sign(delta) * max_delta_per_frame

    return np.clip(cleaned, 0.0, 1.0)


def sample_axis_intensity(
    gray: np.ndarray,
    base: np.ndarray,
    axis_vec: np.ndarray,
    perp: np.ndarray,
    n: int = 200,
    strip_w: int = 20,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample averaged intensity along the axis in a strip.

    Returns (t_values, intensities) where t_values is [0, 1] parametric
    and intensities is the mean pixel value across the strip at each t.
    """
    from scipy.ndimage import uniform_filter1d

    t_values = np.linspace(0, 1, n)
    intensities = np.zeros(n)
    h, w = gray.shape[:2]
    for j, t in enumerate(t_values):
        pt = base + t * axis_vec
        total = 0.0
        count = 0
        for off in range(-strip_w, strip_w + 1):
            px = pt + off * perp
            x, y = int(round(px[0])), int(round(px[1]))
            if 0 <= x < w and 0 <= y < h:
                total += float(gray[y, x])
                count += 1
        intensities[j] = total / max(1, count)
    return t_values, uniform_filter1d(intensities, size=5)


def find_contact_gradient(
    t_values: np.ndarray,
    intensities: np.ndarray,
    search_min: float = 0.4,
    search_max: float = 1.0,
) -> float:
    """Find the contact point as the largest intensity gradient along the axis.

    Returns t in [0, 1] where 0=base, 1=tip.
    """
    gradient = np.abs(np.diff(intensities))
    search = (t_values[:-1] >= search_min) & (t_values[:-1] <= search_max)
    g = gradient.copy()
    g[~search] = 0
    if g.max() < 1e-6:
        return (search_min + search_max) / 2
    return float(t_values[g.argmax()])


def compute_pos_from_points(
    base: tuple[int, int],
    tip: tuple[int, int],
    contact: tuple[int, int],
) -> int:
    """Project contact point onto the base→tip axis. Returns 0-100."""
    base_a = np.array(base, dtype=np.float64)
    tip_a = np.array(tip, dtype=np.float64)
    contact_a = np.array(contact, dtype=np.float64)
    axis_vec = tip_a - base_a
    dot_aa = np.dot(axis_vec, axis_vec)
    if dot_aa < 1e-8:
        return 50
    t = np.dot(contact_a - base_a, axis_vec) / dot_aa
    return int(round(max(0.0, min(1.0, t)) * 100))


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

    # CoTracker3 interpretation:
    # - The lowest-t visible point tracks the BASE well
    # - The highest-t visible point tracks the CONTACT (not the tip!)
    # - The actual TIP is further along the same direction
    #
    # Strategy:
    # 1. base_coords = lowest visible tracked point (good tracking)
    # 2. contact_coords = highest visible tracked point (good tracking)
    # 3. tip_coords = base + shaft_direction * reference_axis_length
    #    where shaft_direction = unit(contact - base)
    # 4. pos = intensity gradient along the base->tip axis
    ref_axis_len_scaled = np.linalg.norm(axis_points_scaled[-1] - axis_points_scaled[0])

    base_coords_s = np.zeros((n_frames, 2), dtype=np.float64)
    contact_coords_s = np.zeros((n_frames, 2), dtype=np.float64)
    tip_coords_s = np.zeros((n_frames, 2), dtype=np.float64)
    last_base = axis_points_scaled[0]
    last_contact = axis_points_scaled[-1]

    for i in range(n_frames):
        visible = all_vis[i] > 0.5
        if visible.any():
            vis_indices = np.where(visible)[0]
            last_base = all_tracks[i, vis_indices[0]]
            last_contact = all_tracks[i, vis_indices[-1]]
        base_coords_s[i] = last_base
        contact_coords_s[i] = last_contact
        # Derive tip from base + redacted direction * reference length
        bc_vec = last_contact - last_base
        bc_len = np.linalg.norm(bc_vec)
        if bc_len > 1:
            shaft_dir = bc_vec / bc_len
            tip_coords_s[i] = last_base + shaft_dir * ref_axis_len_scaled
        else:
            tip_coords_s[i] = last_contact

    # Scale to original resolution
    base_coords = scale_coords(base_coords_s, scaled_size, orig_size)
    tip_coords = scale_coords(tip_coords_s, scaled_size, orig_size)

    # Pos is computed by track_motion via intensity gradient (not here)
    positions = np.full(n_frames, 0.5)

    return CoTrackResult(
        tip_coords=tip_coords,
        base_coords=base_coords,
        visibility=all_vis,
        t_params=t_params,
        positions=positions,
    )

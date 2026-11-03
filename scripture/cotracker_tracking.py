from __future__ import annotations

from dataclasses import dataclass
from functools import cache

import cv2
import numpy as np

from scripture.motion_tracker import AxisDefinition


@cache
def _get_model():
    import torch  # noqa: PLC0415

    return torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline").to("cuda")


_FILTER_WIDTH = 5


def sanitize_positions(positions: np.ndarray, fps: float = 30.0) -> np.ndarray:
    from scipy.ndimage import median_filter  # noqa: PLC0415
    from scipy.signal import savgol_filter  # noqa: PLC0415

    n = len(positions)
    if n < _FILTER_WIDTH:
        return positions.copy()

    cleaned = median_filter(positions, size=_FILTER_WIDTH, mode="nearest")

    window = max(_FILTER_WIDTH, min(int(fps * 0.2), n))
    if window % 2 == 0:
        window += 1
    window = min(window, n)
    cleaned = savgol_filter(cleaned, window, 3)

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
    from scipy.ndimage import uniform_filter1d  # noqa: PLC0415

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


_SEARCH_MAX = 1.0
_FAINTEST_EDGE = 1e-6


def find_contact_gradient(
    t_values: np.ndarray,
    intensities: np.ndarray,
    search_min: float = 0.4,
) -> float:
    gradient = np.abs(np.diff(intensities))
    search = (t_values[:-1] >= search_min) & (t_values[:-1] <= _SEARCH_MAX)
    g = gradient.copy()
    g[~search] = 0
    if g.max() < _FAINTEST_EDGE:
        return (search_min + _SEARCH_MAX) / 2
    return float(t_values[g.argmax()])


_NO_LENGTH = 1e-8


def compute_pos_from_points(
    base: tuple[int, int],
    tip: tuple[int, int],
    contact: tuple[int, int],
) -> int:
    base_a = np.array(base, dtype=np.float64)
    tip_a = np.array(tip, dtype=np.float64)
    contact_a = np.array(contact, dtype=np.float64)
    axis_vec = tip_a - base_a
    dot_aa = np.dot(axis_vec, axis_vec)
    if dot_aa < _NO_LENGTH:
        return 50
    t = np.dot(contact_a - base_a, axis_vec) / dot_aa
    return int(round(max(0.0, min(1.0, t)) * 100))


def scale_coords(
    coords: np.ndarray,
    from_size: tuple[int, int],
    to_size: tuple[int, int],
) -> np.ndarray:
    from_h, from_w = from_size
    to_h, to_w = to_size
    scale_x = to_w / from_w
    scale_y = to_h / from_h
    out = coords.copy().astype(np.float64)
    out[..., 0] *= scale_x
    out[..., 1] *= scale_y
    return out


_TARGET_LONG_SIDE = 384
_AXIS_POINTS = 30
_MAX_CHUNK_FRAMES = 300
_VISIBLE_ABOVE = 0.5


def _get_video_geometry(video_path: str, start_frame: int):
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
    import torch  # noqa: PLC0415

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

    arr = np.stack(frames)
    tensor = torch.from_numpy(arr).permute(0, 3, 1, 2).float()
    return tensor.unsqueeze(0).to("cuda")


def _track_chunk(model, video_chunk, queries):
    import torch  # noqa: PLC0415

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
    tip_coords: np.ndarray
    base_coords: np.ndarray


def cotrack_axis(
    video_path: str,
    axis: AxisDefinition,
    start_frame: int,
    end_frame: int,
) -> CoTrackResult:
    import torch  # noqa: PLC0415

    model = _get_model()
    n_frames = end_frame - start_frame
    ref_local = max(0, min(axis.frame - start_frame, n_frames - 1))

    orig_size, scaled_size = _get_video_geometry(video_path, start_frame)

    tip = np.array(axis.tip, dtype=np.float64)
    base = np.array(axis.base, dtype=np.float64)
    t_params = np.linspace(0, 1, _AXIS_POINTS)
    axis_points = np.array([base + t * (tip - base) for t in t_params])
    axis_points_scaled = scale_coords(axis_points, orig_size, scaled_size)

    all_tracks = np.zeros((n_frames, _AXIS_POINTS, 2), dtype=np.float64)
    all_vis = np.zeros((n_frames, _AXIS_POINTS), dtype=np.float64)

    if n_frames <= _MAX_CHUNK_FRAMES:
        video = _read_chunk(video_path, start_frame, n_frames, scaled_size)
        queries = torch.zeros(1, _AXIS_POINTS, 3, device="cuda")
        queries[0, :, 0] = ref_local
        queries[0, :, 1] = torch.from_numpy(axis_points_scaled[:, 0]).float()
        queries[0, :, 2] = torch.from_numpy(axis_points_scaled[:, 1]).float()
        tracks, vis = _track_chunk(model, video, queries)
        actual = min(tracks.shape[0], n_frames)
        all_tracks[:actual] = tracks[:actual]
        all_vis[:actual] = vis[:actual]
    else:
        chunk_size = _MAX_CHUNK_FRAMES
        overlap = min(100, chunk_size // 4)

        def _run_chunk(c_start, c_end, q_frame, q_points):
            c_len = c_end - c_start
            video = _read_chunk(video_path, start_frame + c_start, c_len, scaled_size)
            queries = torch.zeros(1, _AXIS_POINTS, 3, device="cuda")
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

        ref_chunk_start = max(0, ref_local - chunk_size // 2)
        ref_chunk_end = min(n_frames, ref_chunk_start + chunk_size)
        ref_chunk_start = max(0, ref_chunk_end - chunk_size)
        _run_chunk(ref_chunk_start, ref_chunk_end,
                   ref_local - ref_chunk_start, axis_points_scaled)

        pos = ref_chunk_end - overlap
        while pos < n_frames:
            c_end = min(pos + chunk_size, n_frames)
            seed_frame = pos
            q_points = all_tracks[seed_frame]
            _run_chunk(pos, c_end, 0, q_points)
            pos = c_end - overlap
            if c_end == n_frames:
                break

        pos = ref_chunk_start + overlap
        while pos > 0:
            c_start = max(0, pos - chunk_size)
            seed_frame = pos - 1
            q_frame = seed_frame - c_start
            q_points = all_tracks[seed_frame]
            _run_chunk(c_start, pos, q_frame, q_points)
            pos = c_start + overlap
            if c_start == 0:
                break

    ref_axis_len_scaled = np.linalg.norm(axis_points_scaled[-1] - axis_points_scaled[0])

    base_coords_s = np.zeros((n_frames, 2), dtype=np.float64)
    contact_coords_s = np.zeros((n_frames, 2), dtype=np.float64)
    tip_coords_s = np.zeros((n_frames, 2), dtype=np.float64)
    last_base = axis_points_scaled[0]
    last_contact = axis_points_scaled[-1]

    for i in range(n_frames):
        visible = all_vis[i] > _VISIBLE_ABOVE
        if visible.any():
            vis_indices = np.where(visible)[0]
            last_base = all_tracks[i, vis_indices[0]]
            last_contact = all_tracks[i, vis_indices[-1]]
        base_coords_s[i] = last_base
        contact_coords_s[i] = last_contact
        bc_vec = last_contact - last_base
        bc_len = np.linalg.norm(bc_vec)
        if bc_len > 1:
            anchor_dir = bc_vec / bc_len
            tip_coords_s[i] = last_base + anchor_dir * ref_axis_len_scaled
        else:
            tip_coords_s[i] = last_contact

    base_coords = scale_coords(base_coords_s, scaled_size, orig_size)
    tip_coords = scale_coords(tip_coords_s, scaled_size, orig_size)

    return CoTrackResult(tip_coords=tip_coords, base_coords=base_coords)

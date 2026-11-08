from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from functools import cache
from typing import TYPE_CHECKING

import cv2
import numpy as np
from scipy.ndimage import median_filter, uniform_filter1d
from scipy.signal import savgol_filter

if TYPE_CHECKING:
    from scripture.motion_tracker import AxisDefinition


def _torch():
    import torch  # noqa: PLC0415

    return torch


@cache
def _get_model():
    torch = _torch()

    return torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline").to("cuda")


_FILTER_WIDTH = 5


def sanitize_positions(positions: np.ndarray, fps: float = 30.0) -> np.ndarray:
    n = len(positions)
    if n < _FILTER_WIDTH:
        return positions.copy()

    cleaned = _remove_single_frame_spikes_keeping_edges(positions)

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
    n: int = 200,
    strip_w: int = 20,
) -> tuple[np.ndarray, np.ndarray]:
    perp = np.array([-axis_vec[1], axis_vec[0]]) / np.linalg.norm(axis_vec)
    t_values = np.linspace(0, 1, n)
    intensities = np.zeros(n)
    h, w = gray.shape[:2]
    for j, t in enumerate(t_values):
        pt = base + t * axis_vec
        total = 0.0
        count = 0
        for off in range(-strip_w, strip_w + 1):
            px = pt + off * perp
            x, y = round(px[0]), round(px[1])
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
    return round(max(0.0, min(1.0, t)) * 100)


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
_CHUNK_OVERLAP = 75
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
    torch = _torch()

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
    torch = _torch()

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


@dataclass(frozen=True)
class TrackingChunk:
    start: int
    end: int
    query_frame: int
    seed: int | None


def chunks_outward(n_frames: int, reference: int) -> list[TrackingChunk]:
    if n_frames <= _MAX_CHUNK_FRAMES:
        return [TrackingChunk(0, n_frames, reference, seed=None)]
    first_start = max(0, reference - _MAX_CHUNK_FRAMES // 2)
    first_end = min(n_frames, first_start + _MAX_CHUNK_FRAMES)
    first_start = max(0, first_end - _MAX_CHUNK_FRAMES)
    return [
        TrackingChunk(first_start, first_end, reference - first_start, seed=None),
        *_chunks_forward(first_end - _CHUNK_OVERLAP, n_frames),
        *_chunks_backward(first_start + _CHUNK_OVERLAP),
    ]


def _chunks_forward(start: int, n_frames: int) -> Iterator[TrackingChunk]:
    while start < n_frames:
        end = min(start + _MAX_CHUNK_FRAMES, n_frames)
        yield TrackingChunk(start, end, 0, seed=start)
        if end == n_frames:
            return
        start = end - _CHUNK_OVERLAP


def _chunks_backward(end: int) -> Iterator[TrackingChunk]:
    while end > 0:
        start = max(0, end - _MAX_CHUNK_FRAMES)
        seed = end - 1
        yield TrackingChunk(start, end, seed - start, seed=seed)
        if start == 0:
            return
        end = start + _CHUNK_OVERLAP


def points_along_axis(axis: AxisDefinition) -> np.ndarray:
    tip = np.array(axis.tip, dtype=np.float64)
    base = np.array(axis.base, dtype=np.float64)
    return np.array([base + t * (tip - base) for t in np.linspace(0, 1, _AXIS_POINTS)])


def ends_from_visible_points(
    tracks: np.ndarray,
    visibility: np.ndarray,
    drawn_points: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    drawn_length = np.linalg.norm(drawn_points[-1] - drawn_points[0])
    base_coords = np.zeros((len(tracks), 2), dtype=np.float64)
    tip_coords = np.zeros((len(tracks), 2), dtype=np.float64)
    base, contact = drawn_points[0], drawn_points[-1]
    for i, (points, visible) in enumerate(zip(tracks, visibility > _VISIBLE_ABOVE, strict=True)):
        if visible.any():
            seen = points[visible]
            base, contact = seen[0], seen[-1]
        base_coords[i] = base
        along = contact - base
        length = np.linalg.norm(along)
        tip_coords[i] = base + along / length * drawn_length if length > 1 else contact
    return base_coords, tip_coords


def _queries_at(query_frame: int, points: np.ndarray):
    torch = _torch()

    queries = torch.zeros(1, _AXIS_POINTS, 3, device="cuda")
    queries[0, :, 0] = query_frame
    queries[0, :, 1] = torch.from_numpy(points[:, 0].copy()).float()
    queries[0, :, 2] = torch.from_numpy(points[:, 1].copy()).float()
    return queries


def cotrack_axis(
    video_path: str,
    axis: AxisDefinition,
    start_frame: int,
    end_frame: int,
) -> CoTrackResult:
    model = _get_model()
    n_frames = end_frame - start_frame
    orig_size, scaled_size = _get_video_geometry(video_path, start_frame)
    drawn_points = scale_coords(points_along_axis(axis), orig_size, scaled_size)
    all_tracks = np.zeros((n_frames, _AXIS_POINTS, 2), dtype=np.float64)
    all_vis = np.zeros((n_frames, _AXIS_POINTS), dtype=np.float64)

    reference = max(0, min(axis.frame - start_frame, n_frames - 1))
    for chunk in chunks_outward(n_frames, reference):
        length = chunk.end - chunk.start
        video = _read_chunk(video_path, start_frame + chunk.start, length, scaled_size)
        seed_points = drawn_points if chunk.seed is None else all_tracks[chunk.seed]
        tracks, vis = _track_chunk(model, video, _queries_at(chunk.query_frame, seed_points))
        reached = min(tracks.shape[0], length)
        all_tracks[chunk.start:chunk.start + reached] = tracks[:reached]
        all_vis[chunk.start:chunk.start + reached] = vis[:reached]

    base_coords, tip_coords = ends_from_visible_points(all_tracks, all_vis, drawn_points)
    return CoTrackResult(tip_coords=scale_coords(tip_coords, scaled_size, orig_size),
                         base_coords=scale_coords(base_coords, scaled_size, orig_size))


def _remove_single_frame_spikes_keeping_edges(positions: np.ndarray) -> np.ndarray:
    return median_filter(positions, size=_FILTER_WIDTH, mode="nearest")

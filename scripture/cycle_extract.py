from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks, savgol_filter

from scripture.frames import ms_of_frame

_MIDPOINT = 0.5


def smooth_signal(positions: np.ndarray, window: int = 15, polyorder: int = 3) -> np.ndarray:
    if len(positions) < window:
        return positions
    return savgol_filter(positions, window, polyorder)


def remove_drift(positions: np.ndarray, cutoff_period_frames: int = 300) -> np.ndarray:
    if len(positions) < cutoff_period_frames:
        return positions - np.mean(positions) + _MIDPOINT

    window = cutoff_period_frames if cutoff_period_frames % 2 == 1 else cutoff_period_frames + 1
    window = min(window, len(positions))
    if window % 2 == 0:
        window -= 1
    drift = savgol_filter(positions, window, 2)
    corrected = positions - drift + np.mean(positions)
    return np.clip(corrected, 0, 1)


def _adaptive_prominence(positions: np.ndarray, fps: float,
                         floor: float = 0.05) -> float:
    window = max(3, int(fps * 5))
    local_mean = uniform_filter1d(positions.astype(np.float64), size=window, mode="nearest")
    local_sq_mean = uniform_filter1d((positions ** 2).astype(np.float64), size=window, mode="nearest")
    local_std = np.sqrt(np.maximum(local_sq_mean - local_mean ** 2, 0))
    median_std = np.median(local_std)
    return max(floor, 0.3 * median_std * 1.35)


_MIN_CYCLE_DISTANCE_MS = 200.0


def _enforce_alternating(indices: np.ndarray, values: np.ndarray) -> np.ndarray:
    if len(indices) <= 1:
        return indices
    result = [indices[0]]
    for i in range(1, len(indices)):
        prev_val = values[result[-1]]
        curr_val = values[indices[i]]
        prev_is_peak = prev_val > _MIDPOINT
        curr_is_peak = curr_val > _MIDPOINT
        if prev_is_peak == curr_is_peak:
            if prev_is_peak:
                if curr_val > prev_val:
                    result[-1] = indices[i]
            elif curr_val < prev_val:
                result[-1] = indices[i]
        else:
            result.append(indices[i])
    return np.array(result, dtype=indices.dtype)


def extract_cycles(positions: np.ndarray, timestamps_ms: np.ndarray,
                   max_prominence: float = 0.15,
                   fps: float = 30.0) -> list[dict]:
    detrended = remove_drift(positions, cutoff_period_frames=max(5, int(fps * 10)))
    smooth_window = max(5, min(15, int(fps * 0.2)))
    if smooth_window % 2 == 0:
        smooth_window += 1
    smoothed = smooth_signal(detrended, window=smooth_window)

    avg_frame_interval_ms = np.mean(np.diff(timestamps_ms)) if len(timestamps_ms) > 1 else ms_of_frame(1, fps)
    min_distance_frames = max(1, int(_MIN_CYCLE_DISTANCE_MS / avg_frame_interval_ms))

    prominence = _adaptive_prominence(smoothed, fps)
    effective_prominence = min(prominence, max_prominence)

    peaks, _ = find_peaks(smoothed, distance=min_distance_frames, prominence=effective_prominence)

    valleys, _ = find_peaks(-smoothed, distance=min_distance_frames, prominence=effective_prominence)

    extrema_indices = np.sort(np.concatenate([peaks, valleys]))
    if len(extrema_indices) > 0:
        extrema_indices = _enforce_alternating(extrema_indices, smoothed)

    return [
        {"at": round(timestamps_ms[idx]), "pos": round(np.clip(smoothed[idx], 0, 1) * 100)}
        for idx in extrema_indices
    ]

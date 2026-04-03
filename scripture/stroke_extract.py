"""Extract stroke peaks and valleys from a position signal."""

import numpy as np
from scipy.signal import find_peaks, savgol_filter


def smooth_signal(positions: np.ndarray, window: int = 15, polyorder: int = 3) -> np.ndarray:
    """Apply Savitzky-Golay filter to smooth the position signal."""
    if len(positions) < window:
        return positions
    return savgol_filter(positions, window, polyorder)


def remove_drift(positions: np.ndarray, cutoff_period_frames: int = 300) -> np.ndarray:
    """Remove low-frequency drift via high-pass filtering.

    Subtracts a heavily-smoothed version of the signal (the drift component).
    """
    if len(positions) < cutoff_period_frames:
        return positions - np.mean(positions) + 0.5

    window = cutoff_period_frames if cutoff_period_frames % 2 == 1 else cutoff_period_frames + 1
    window = min(window, len(positions))
    if window % 2 == 0:
        window -= 1
    drift = savgol_filter(positions, window, 2)
    corrected = positions - drift + np.mean(positions)
    return np.clip(corrected, 0, 1)


def _adaptive_prominence(positions: np.ndarray, fps: float,
                         floor: float = 0.05) -> float:
    """Compute prominence threshold based on local signal variation."""
    from scipy.ndimage import uniform_filter1d
    window = max(3, int(fps * 5))
    # Rolling IQR approximation: use std * 1.35 as IQR proxy
    local_mean = uniform_filter1d(positions.astype(np.float64), size=window, mode="nearest")
    local_sq_mean = uniform_filter1d((positions ** 2).astype(np.float64), size=window, mode="nearest")
    local_std = np.sqrt(np.maximum(local_sq_mean - local_mean ** 2, 0))
    median_std = np.median(local_std)
    # Use 30% of the local IQR proxy as prominence, with a floor
    return max(floor, 0.3 * median_std * 1.35)


def _enforce_alternating(indices: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Keep only alternating extrema — when two peaks/valleys are consecutive,
    keep the more extreme one."""
    if len(indices) < 2:
        return indices
    result = [indices[0]]
    for i in range(1, len(indices)):
        prev_val = values[result[-1]]
        curr_val = values[indices[i]]
        # Determine if both are on the same side of the midpoint
        prev_is_peak = prev_val > 0.5
        curr_is_peak = curr_val > 0.5
        if prev_is_peak == curr_is_peak:
            # Same type — keep the more extreme one
            if prev_is_peak:
                if curr_val > prev_val:
                    result[-1] = indices[i]
            else:
                if curr_val < prev_val:
                    result[-1] = indices[i]
        else:
            result.append(indices[i])
    return np.array(result, dtype=indices.dtype)


def extract_strokes(positions: np.ndarray, timestamps_ms: np.ndarray,
                    min_stroke_height: float = 0.15,
                    min_stroke_distance_ms: float = 200.0,
                    fps: float = 30.0) -> list[dict]:
    """Find stroke turnaround points (peaks and valleys).

    Returns a list of {"at": timestamp_ms, "pos": 0-100} dicts, containing
    only the extrema — the minimal representation of the stroke pattern.
    """
    # Remove drift before smoothing
    detrended = remove_drift(positions, cutoff_period_frames=max(5, int(fps * 10)))
    smooth_window = max(5, min(15, int(fps * 0.2)))
    if smooth_window % 2 == 0:
        smooth_window += 1
    smoothed = smooth_signal(detrended, window=smooth_window)

    avg_frame_interval_ms = np.mean(np.diff(timestamps_ms)) if len(timestamps_ms) > 1 else 1000 / fps
    min_distance_frames = max(1, int(min_stroke_distance_ms / avg_frame_interval_ms))

    prominence = _adaptive_prominence(smoothed, fps)
    # Use the smaller of adaptive and the caller's threshold
    effective_prominence = min(prominence, min_stroke_height)

    # Find peaks (high positions = near tip)
    peaks, _ = find_peaks(smoothed, distance=min_distance_frames, prominence=effective_prominence)

    # Find valleys (low positions = near base)
    valleys, _ = find_peaks(-smoothed, distance=min_distance_frames, prominence=effective_prominence)

    # Merge, sort, and enforce alternation
    extrema_indices = np.sort(np.concatenate([peaks, valleys]))
    if len(extrema_indices) > 0:
        extrema_indices = _enforce_alternating(extrema_indices, smoothed)

    actions = []
    for idx in extrema_indices:
        actions.append({
            "at": int(round(timestamps_ms[idx])),
            "pos": int(round(np.clip(smoothed[idx], 0, 1) * 100)),
        })

    return actions

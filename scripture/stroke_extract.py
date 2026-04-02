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

    drift = savgol_filter(positions, min(cutoff_period_frames | 1, len(positions)), 2)
    corrected = positions - drift + np.mean(positions)
    return np.clip(corrected, 0, 1)


def extract_strokes(positions: np.ndarray, timestamps_ms: np.ndarray,
                    min_stroke_height: float = 0.15,
                    min_stroke_distance_ms: float = 200.0,
                    fps: float = 30.0) -> list[dict]:
    """Find stroke turnaround points (peaks and valleys).

    Returns a list of {"at": timestamp_ms, "pos": 0-100} dicts, containing
    only the extrema — the minimal representation of the stroke pattern.
    """
    smoothed = smooth_signal(positions)

    avg_frame_interval_ms = np.mean(np.diff(timestamps_ms)) if len(timestamps_ms) > 1 else 1000 / fps
    min_distance_frames = max(1, int(min_stroke_distance_ms / avg_frame_interval_ms))

    # Find peaks (high positions = near tip)
    peaks, _ = find_peaks(smoothed, distance=min_distance_frames, prominence=min_stroke_height)

    # Find valleys (low positions = near base)
    valleys, _ = find_peaks(-smoothed, distance=min_distance_frames, prominence=min_stroke_height)

    # Merge and sort by time
    extrema_indices = np.sort(np.concatenate([peaks, valleys]))

    actions = []
    for idx in extrema_indices:
        actions.append({
            "at": int(round(timestamps_ms[idx])),
            "pos": int(round(smoothed[idx] * 100)),
        })

    return actions

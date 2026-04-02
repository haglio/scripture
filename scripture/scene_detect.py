"""Automatic scene/shot change detection.

Combines histogram correlation (catches color distribution shifts) with
frame differencing (catches spatial layout changes between similar-colored
shots).  Either method triggering counts as a scene change.
"""

from dataclasses import dataclass
from typing import Callable

import cv2
import numpy as np


@dataclass
class Scene:
    start_frame: int
    end_frame: int
    is_content: bool
    representative_frame: int


def filter_scenes(scenes: list[Scene], min_frames: int = 5) -> list[Scene]:
    """Remove non-content and too-short scenes."""
    return [
        s for s in scenes
        if s.is_content and (s.end_frame - s.start_frame) >= min_frames
    ]


def detect_scene_changes(
    video_path: str,
    hist_threshold: float = 0.4,
    diff_threshold: float = 30.0,
    subsample: int = 2,
    progress: Callable[[float], None] | None = None,
) -> list[int]:
    """Return frame indices where scene changes occur.

    Uses two complementary methods:
    - Histogram correlation: catches overall color distribution shifts
      (e.g. black-to-content transitions)
    - Frame differencing: mean absolute pixel difference catches spatial
      changes between shots with similar color palettes

    `subsample` controls how many frames to skip between comparisons
    (1 = every frame, 2 = every other frame, etc.)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    prev_gray = None
    prev_hist = None
    scene_changes: list[int] = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % subsample == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Downscale for faster differencing
            small = cv2.resize(gray, (160, 90))

            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            cv2.normalize(hist, hist)

            if prev_gray is not None:
                # Method 1: histogram correlation
                correlation = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL)
                hist_cut = correlation < hist_threshold

                # Method 2: mean absolute frame difference
                diff = cv2.absdiff(prev_gray, small)
                mean_diff = float(diff.mean())
                diff_cut = mean_diff > diff_threshold

                if hist_cut or diff_cut:
                    scene_changes.append(frame_idx)

            prev_gray = small
            prev_hist = hist

        frame_idx += 1

        if progress and total_frames > 0 and frame_idx % 500 == 0:
            progress(frame_idx / total_frames)

    cap.release()

    if progress:
        progress(1.0)

    return scene_changes


def _frame_variance(cap: cv2.VideoCapture, frame_idx: int) -> float:
    """Return the Laplacian variance of a frame (higher = more visual content)."""
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    if not ret:
        return 0.0
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _frame_brightness(cap: cv2.VideoCapture, frame_idx: int) -> float:
    """Return mean brightness of a frame (0-255)."""
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    if not ret:
        return 0.0
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(gray.mean())


def build_scenes(
    video_path: str,
    hist_threshold: float = 0.4,
    diff_threshold: float = 30.0,
    subsample: int = 2,
    black_brightness: float = 15.0,
    num_samples: int = 10,
    progress: Callable[[float], None] | None = None,
) -> list[Scene]:
    """Detect scenes, classify as content/non-content, pick representative frames.

    Returns only content scenes with sufficient length (black/blank scenes
    and zero-length scenes are filtered out).
    """
    changes = detect_scene_changes(
        video_path, hist_threshold, diff_threshold, subsample, progress,
    )
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    boundaries = [0] + changes + [total_frames]
    scenes: list[Scene] = []

    for i in range(len(boundaries) - 1):
        start = boundaries[i]
        end = boundaries[i + 1]
        length = end - start

        sample_count = min(num_samples, length)
        if sample_count <= 0:
            continue

        if sample_count == 1:
            sample_indices = [start]
        else:
            sample_indices = [
                start + int(j * (length - 1) / (sample_count - 1))
                for j in range(sample_count)
            ]

        brightnesses = [_frame_brightness(cap, idx) for idx in sample_indices]
        dark_count = sum(1 for b in brightnesses if b < black_brightness)
        is_content = dark_count < len(brightnesses) / 2

        best_idx = sample_indices[0]
        best_var = -1.0
        for idx in sample_indices:
            v = _frame_variance(cap, idx)
            if v > best_var:
                best_var = v
                best_idx = idx

        scenes.append(Scene(
            start_frame=start,
            end_frame=end,
            is_content=is_content,
            representative_frame=best_idx,
        ))

    cap.release()
    return filter_scenes(scenes)

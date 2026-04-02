"""Automatic scene/shot change detection via frame histogram comparison."""

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Scene:
    start_frame: int
    end_frame: int
    is_content: bool
    representative_frame: int


def detect_scene_changes(video_path: str, threshold: float = 0.4) -> list[int]:
    """Return a list of frame indices where scene changes occur.

    Uses histogram correlation between consecutive frames. A drop below
    `threshold` indicates a cut.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    prev_hist = None
    scene_changes: list[int] = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        cv2.normalize(hist, hist)

        if prev_hist is not None:
            correlation = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL)
            if correlation < threshold:
                scene_changes.append(frame_idx)

        prev_hist = hist
        frame_idx += 1

    cap.release()
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


def build_scenes(video_path: str, threshold: float = 0.4,
                 black_brightness: float = 15.0,
                 num_samples: int = 10) -> list[Scene]:
    """Detect scenes, classify as content/non-content, pick representative frames.

    A scene is non-content if the majority of sampled frames are near-black.
    The representative frame is the sampled frame with the highest Laplacian
    variance (most visual detail).
    """
    changes = detect_scene_changes(video_path, threshold)
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    boundaries = [0] + changes + [total_frames]
    scenes: list[Scene] = []

    for i in range(len(boundaries) - 1):
        start = boundaries[i]
        end = boundaries[i + 1]
        length = end - start

        # Sample frames evenly across the scene
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

        # Check brightness to classify content vs non-content
        brightnesses = [_frame_brightness(cap, idx) for idx in sample_indices]
        dark_count = sum(1 for b in brightnesses if b < black_brightness)
        is_content = dark_count < len(brightnesses) / 2

        # Pick representative frame: highest Laplacian variance
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
    return scenes

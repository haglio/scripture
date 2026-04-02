"""Scene detection using PySceneDetect's ContentDetector.

ContentDetector uses HSV color-space analysis with adaptive thresholds,
which correctly distinguishes actual scene cuts from fast motion within
a scene.
"""

from dataclasses import dataclass
from typing import Callable

import cv2
import numpy as np
from scenedetect import open_video, SceneManager, ContentDetector


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
    min_scene_len_sec: float = 2.0,
    black_brightness: float = 15.0,
    num_samples: int = 10,
    progress: Callable[[float], None] | None = None,
) -> list[Scene]:
    """Detect scenes via PySceneDetect, classify, pick representative frames.

    Uses ContentDetector for cut detection with a minimum scene length to
    prevent motion-triggered false positives.  Returns only content scenes
    (black/blank and very short scenes are filtered out).
    """
    video = open_video(video_path)
    fps = video.frame_rate
    total_frames = video.duration.get_frames()
    min_scene_frames = int(min_scene_len_sec * fps)

    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(min_scene_len=min_scene_frames))

    frame_count = [0]

    def _on_frame(_frame: np.ndarray, frame_num: int) -> None:
        frame_count[0] = frame_num
        if progress and total_frames > 0 and frame_num % 200 == 0:
            progress(frame_num / total_frames)

    scene_manager.detect_scenes(video=video, callback=_on_frame)

    if progress:
        progress(1.0)

    scene_list = scene_manager.get_scene_list(start_in_scene=True)

    # Convert to our Scene dataclass with representative frame selection
    cap = cv2.VideoCapture(video_path)
    scenes: list[Scene] = []

    for start_tc, end_tc in scene_list:
        start = start_tc.get_frames()
        end = end_tc.get_frames()
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

        # Classify content vs black
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
    return filter_scenes(scenes)

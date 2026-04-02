"""Automatic scene/shot change detection via frame histogram comparison."""

import cv2
import numpy as np


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

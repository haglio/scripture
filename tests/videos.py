from __future__ import annotations

import numpy as np


def a_flat_video(path, frames=30, size=(160, 120)):
    """A short clip of one unchanging textured frame, written to `path`."""
    import cv2

    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, size)
    frame = np.random.default_rng(3).integers(
        0, 255, (size[1], size[0], 3)).astype(np.uint8)
    for _ in range(frames):
        writer.write(frame)
    writer.release()
    return str(path)

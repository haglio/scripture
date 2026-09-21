"""The video being annotated: opened once, asked for a frame at a time.

Everything OpenCV about reading footage lives here, so the window holds a video
rather than a capture plus four numbers it read off one.  It also puts the
"a capture that would not open is not a video" rule in one place: the window
checked it where a project named a missing file and not where a file was chosen,
which is how a released capture stayed installed and every later frame came back
empty with nothing said.
"""
from __future__ import annotations

import cv2

#: What to read a video at when it will not say its own frame rate.
_ASSUMED_FPS = 30.0


class VideoSource:
    """One open video file: its shape, and the frame at an index."""

    def __init__(self, capture: cv2.VideoCapture) -> None:
        self._capture = capture
        self.fps = capture.get(cv2.CAP_PROP_FPS) or _ASSUMED_FPS
        self.total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    @classmethod
    def opened(cls, path: str) -> VideoSource | None:
        """The video at `path`, or None -- having let the file go -- if it will
        not open, so a refusal can never leave a dead one installed."""
        capture = cv2.VideoCapture(path)
        if not capture.isOpened():
            capture.release()
            return None
        return cls(capture)

    def frame_at(self, index: int):
        """The frame at that index, or None when the video cannot give it."""
        self._capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        read, frame = self._capture.read()
        return frame if read else None

    def release(self) -> None:
        self._capture.release()

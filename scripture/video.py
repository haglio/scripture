from __future__ import annotations

import cv2

_ASSUMED_FPS = 30.0


class VideoSource:

    def __init__(self, capture: cv2.VideoCapture) -> None:
        self._capture = capture
        self.fps = capture.get(cv2.CAP_PROP_FPS) or _ASSUMED_FPS
        self.total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    @classmethod
    def opened(cls, path: str) -> VideoSource | None:
        capture = cv2.VideoCapture(path)
        if not capture.isOpened():
            capture.release()
            return None
        return cls(capture)

    def frame_at(self, index: int):
        self._capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        read, frame = self._capture.read()
        return frame if read else None

    def release(self) -> None:
        self._capture.release()


class _NoVideo:
    fps = _ASSUMED_FPS
    total_frames = 0

    def frame_at(self, _index: int) -> None:
        return None

    def release(self) -> None:
        pass


NO_VIDEO = _NoVideo()

from __future__ import annotations

_MS_PER_SECOND = 1000


def ms_of_frame(frame, fps: float):
    return frame / fps * _MS_PER_SECOND


def frame_of_ms(ms: float, fps: float) -> int:
    return round(ms / _MS_PER_SECOND * fps)


def half_frame_ms(fps: float) -> float:
    return ms_of_frame(0.5, fps)

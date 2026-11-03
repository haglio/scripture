from __future__ import annotations

import math
from dataclasses import dataclass

from app_support.overlay import overlay_value

from scripture.content import LOCAL_CONTENT, load_content

_CONTENT = load_content()

ANCHOR_CLASSES: tuple[str, ...] = tuple(overlay_value(_CONTENT, "anchor_classes", path=LOCAL_CONTENT))


@dataclass
class Detection:
    class_name: str
    confidence: float
    rect: tuple[int, int, int, int]


_INTERACTION_DISTANCE_FACTOR = 0.85

_CONTACT_CLASSES: tuple[str, ...] = tuple(overlay_value(_CONTENT, "contact_classes", path=LOCAL_CONTENT))


def _center(rect: tuple[int, int, int, int]) -> tuple[float, float]:
    x, y, w, h = rect
    return x + w / 2, y + h / 2


def _half_diagonal(rect: tuple[int, int, int, int]) -> float:
    _, _, w, h = rect
    return math.hypot(w, h) / 2


def _within_interaction_distance(
    rect_a: tuple[int, int, int, int],
    rect_b: tuple[int, int, int, int],
) -> bool:
    acx, acy = _center(rect_a)
    bcx, bcy = _center(rect_b)
    dist = math.hypot(bcx - acx, bcy - acy)
    max_dist = (_half_diagonal(rect_a) + _half_diagonal(rect_b)) * _INTERACTION_DISTANCE_FACTOR
    return dist < max_dist


def find_interacting(anchor: Detection, detections: list[Detection]) -> list[Detection]:
    return [
        d for d in detections
        if d.class_name != ANCHOR_CLASSES[0]
        and _within_interaction_distance(anchor.rect, d.rect)
    ]


def contact_near_rect(
    rect: tuple[int, int, int, int],
    detections: list[Detection],
) -> list[Detection]:
    return [
        d for d in detections
        if d.class_name in _CONTACT_CLASSES
        and _within_interaction_distance(rect, d.rect)
    ]


def combine_roi(
    rects: list[tuple[int, int, int, int]],
    frame_size: tuple[int, int],
    padding: int = 20,
    min_size: int = 128,
) -> tuple[int, int, int, int]:
    frame_h, frame_w = frame_size
    x1 = max(0, min(b[0] for b in rects) - padding)
    y1 = max(0, min(b[1] for b in rects) - padding)
    x2 = min(frame_w, max(b[0] + b[2] for b in rects) + padding)
    y2 = min(frame_h, max(b[1] + b[3] for b in rects) + padding)

    w, h = x2 - x1, y2 - y1
    if w < min_size:
        x1 = max(0, x1 - (min_size - w) // 2)
        w = min_size
    if h < min_size:
        y1 = max(0, y1 - (min_size - h) // 2)
        h = min_size

    x1 = min(x1, max(0, frame_w - w))
    y1 = min(y1, max(0, frame_h - h))
    return int(x1), int(y1), int(w), int(h)


def smooth_roi(
    previous: tuple[int, int, int, int] | None,
    new: tuple[int, int, int, int],
    factor: float,
) -> tuple[int, int, int, int]:
    if previous is None:
        return new
    return tuple(int(p * factor + n * (1 - factor)) for p, n in zip(previous, new, strict=True))


def _best_anchor(detections: list[Detection]) -> Detection | None:
    for cls in ANCHOR_CLASSES:
        candidates = [d for d in detections if d.class_name == cls]
        if candidates:
            return max(candidates, key=lambda d: d.confidence)
    return None


class RoiLock:
    def __init__(self, *, detect_every: int, padding: int, min_size: int, smoothing: float,
                 persistence: int) -> None:
        self._detect_every = detect_every
        self._padding = padding
        self._min_size = min_size
        self._smoothing = smoothing
        self._persistence = persistence
        self.roi: tuple[int, int, int, int] | None = None
        self.belief: tuple[int, int, int, int] | None = None
        self.mode = "none"
        self._frames_unrefreshed = 0

    def lose(self) -> None:
        self.roi = None
        self.belief = None
        self.mode = "none"
        self._frames_unrefreshed = 0

    def wants_detection_on(self, idx: int) -> bool:
        return idx % self._detect_every == 0 or self.roi is None

    def _aim_at(self, rects: list[tuple[int, int, int, int]], mode: str, frame_size: tuple[int, int]) -> None:
        self.mode = mode
        self._frames_unrefreshed = 0
        target = combine_roi(rects, frame_size, self._padding, self._min_size)
        self.roi = smooth_roi(self.roi, target, self._smoothing)

    def look_at(self, detections: list[Detection], frame_size: tuple[int, int]) -> bool:
        anchor = _best_anchor(detections)
        if anchor is not None:
            self.belief = anchor.rect
            near = find_interacting(anchor, detections)
            self._aim_at([anchor.rect, *(d.rect for d in near)], "anchor", frame_size)
            return True
        if self.roi is None or self.belief is None:
            return False
        contacts = contact_near_rect(self.belief, detections)
        if contacts:
            self._aim_at([self.belief, *(d.rect for d in contacts)], "contact", frame_size)
            return True
        self.mode = "coast"
        return False

    def grow_older(self) -> None:
        if self.roi is None:
            return
        self._frames_unrefreshed += 1
        if self._frames_unrefreshed > self._persistence:
            self.lose()

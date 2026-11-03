from __future__ import annotations

from collections.abc import Callable


def schedule_frames(start: int, end: int, stride: int) -> list[int]:
    return list(range(start, end + 1, stride))


def next_scheduled(schedule: list[int], annotated: set[int],
                   after: int) -> int | None:
    pending = [f for f in schedule if f not in annotated]
    if not pending:
        return None
    for frame in pending:
        if frame > after:
            return frame
    return pending[0]


class LabelSession:

    def __init__(self) -> None:
        self.active = False
        self.target: int | None = None
        self._undo: list[tuple[int, int]] = []

    @staticmethod
    def progress(schedule: list[int], annotated: set[int]) -> tuple[int, int]:
        return len(annotated & set(schedule)), len(schedule)

    def stop(self) -> None:
        self.active = False
        self.target = None

    def advance(self, schedule: list[int], annotated: set[int],
                after: int) -> int | None:
        nxt = next_scheduled(schedule, annotated, after)
        if nxt is None:
            self.stop()
        else:
            self.target = nxt
        return nxt

    def anchor(self, fallback: int) -> int:
        return self.target if self.target is not None else fallback

    def after_label(self, scene: int, frame: int) -> int:
        self._undo.append((scene, frame))
        return self.anchor(frame) - 1

    def undo_last(self) -> tuple[int, int] | None:
        return self._undo.pop() if self._undo else None

    def forget_scene(self, scene: int) -> None:
        self._undo = [entry for entry in self._undo if entry[0] != scene]

    def reindex(self, scene_of_frame: Callable[[int], int]) -> None:
        self._undo = [(scene_of_frame(frame), frame) for _scene, frame in self._undo]

    def clear(self) -> None:
        self.stop()
        self._undo.clear()

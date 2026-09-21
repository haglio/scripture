"""Sparse ground-truth labeling: which frames to decide, and where you are.

The operator walks an evenly-strided schedule over one scene deciding each frame
in turn, and may scrub away and come back, skip, or undo.  All of that is state
with rules between it -- running out of frames ends the session, an undo has to
name the scene the label went to -- so it lives in one place rather than as five
attributes on the window.
"""
from __future__ import annotations

from collections.abc import Callable


def schedule_frames(start: int, end: int, stride: int) -> list[int]:
    """Evenly-strided frame indices covering [start, end]."""
    return list(range(start, end + 1, stride))


def next_scheduled(schedule: list[int], annotated: set[int],
                   after: int) -> int | None:
    """The next frame to decide: the first undecided entry past `after`, wrapping
    round to the earliest undecided one; None when they are all done."""
    pending = [f for f in schedule if f not in annotated]
    if not pending:
        return None
    for frame in pending:
        if frame > after:
            return frame
    return pending[0]


class LabelSession:
    """Whether a session is running, which frame it is on, and what to undo."""

    def __init__(self) -> None:
        self.active = False
        self.target: int | None = None
        self._undo: list[tuple[int, int]] = []

    @staticmethod
    def progress(schedule: list[int], annotated: set[int]) -> tuple[int, int]:
        """How many of the scheduled frames are decided, out of how many."""
        return len(annotated & set(schedule)), len(schedule)

    def stop(self) -> None:
        """Pause or finish: nothing is being decided any more."""
        self.active = False
        self.target = None

    def advance(self, schedule: list[int], annotated: set[int],
                after: int) -> int | None:
        """Aim at the next frame still to decide, ending the session when there
        is none -- which is the rule its one caller used to spell out."""
        nxt = next_scheduled(schedule, annotated, after)
        if nxt is None:
            self.stop()
        else:
            self.target = nxt
        return nxt

    def anchor(self, fallback: int) -> int:
        """Where the next search starts: the frame being decided, if any."""
        return self.target if self.target is not None else fallback

    def after_label(self, scene: int, frame: int) -> int:
        """Record a decision, and say where to search from -- just before the
        frame being decided, so labeling around it never skips it."""
        self._undo.append((scene, frame))
        return self.anchor(frame) - 1

    def undo_last(self) -> tuple[int, int] | None:
        """The most recent decision, taken off the stack."""
        return self._undo.pop() if self._undo else None

    def forget_scene(self, scene: int) -> None:
        self._undo = [entry for entry in self._undo if entry[0] != scene]

    def reindex(self, scene_of_frame: Callable[[int], int]) -> None:
        """Renumber the scene each entry names, after a split or unsplit."""
        self._undo = [(scene_of_frame(frame), frame) for _scene, frame in self._undo]

    def clear(self) -> None:
        self.stop()
        self._undo.clear()

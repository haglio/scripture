"""Sparse ground-truth labeling sessions: which frames to label, and what's next."""


def schedule_frames(start: int, end: int, stride: int) -> list[int]:
    """Evenly-strided frame indices covering [start, end]."""
    return list(range(start, end + 1, stride))


def next_scheduled(schedule: list[int], annotated: set[int],
                   after: int) -> int | None:
    """The next frame to label: first unannotated schedule entry past `after`,
    wrapping around to the earliest unannotated one; None when all are done."""
    pending = [f for f in schedule if f not in annotated]
    if not pending:
        return None
    for frame in pending:
        if frame > after:
            return frame
    return pending[0]

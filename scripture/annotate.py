"""Sparse ground-truth labeling sessions: which frames to label, what's next,
and flattening the labels for training."""


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


def collect_labels(ground_truth: dict) -> list[dict]:
    """Flatten {scene: {frame: entry}} into training rows.

    Each row: scene, frame, tip, base, contact (None = explicit no-contact),
    pos (0-100 projection of contact onto the base->tip axis, None if no
    contact). Only explicit entries are emitted — inherited/derived frames are
    the GUI's display concern, not labels.
    """
    from scripture.cotracker_tracking import compute_pos_from_points

    rows = []
    for scene_idx, frames in ground_truth.items():
        for frame_idx, entry in frames.items():
            contact = entry.get("contact")
            tip, base = entry.get("tip"), entry.get("base")
            pos = None
            if contact is not None and tip and base:
                pos = float(compute_pos_from_points(base, tip, contact))
            rows.append({"scene": scene_idx, "frame": frame_idx,
                         "tip": tip, "base": base,
                         "contact": contact, "pos": pos})
    return rows

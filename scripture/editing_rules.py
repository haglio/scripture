from __future__ import annotations


def moving_the_axis_needs_asking(axis, frame):
    return axis is not None and axis.frame != frame


def both_scenes_have_an_axis(axes, left, right):
    return left in axes and right in axes


def next_frame(frames, current):
    if not frames:
        return None
    return next((frame for frame in frames if frame > current), frames[0])


def previous_frame(frames, current):
    if not frames:
        return None
    return next((frame for frame in reversed(frames) if frame < current), frames[-1])

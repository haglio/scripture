"""A split renumbers the scenes, and the hand-labeled frames must follow.

`_rebuild_scenes` remapped the axis, the actions and the tracking positions
from old scene index to new and left the ground truth on its old key. Every
label past the split point then belonged to the scene before it: the canvas
read the wrong bucket, the label session counted the wrong frames done, and
"Delete all N labeled frames in this scene?" deleted another scene's work.
"""

from __future__ import annotations


def _label(contact):
    """A hand-placed label. The numbers are invented; only the keys matter."""
    return {"tip": (40, 30), "base": (40, 150), "contact": contact, "is_action": True}


def _with_one_scene(window, total_frames=1000):
    window.total_frames = total_frames
    window.splits = []
    window._rebuild_scenes()
    return window


def test_a_split_before_a_labeled_frame_carries_the_label_across(window):
    _with_one_scene(window)
    window.ground_truth = {0: {800: _label((40, 90))}}
    window.current_frame_idx = 800

    window._do_split_at(400)

    assert window._current_scene_idx() == 1
    assert window._get_gt_for_frame(1, 800) is not None


def test_unsplitting_gathers_both_scenes_labels_into_the_merged_one(window):
    _with_one_scene(window)
    window.splits = [400]
    window._rebuild_scenes()
    window.ground_truth = {0: {100: _label((40, 60))}, 1: {800: _label((40, 90))}}
    window.current_frame_idx = 400

    window._do_unsplit(400)

    assert sorted(window.ground_truth) == [0]
    assert sorted(window.ground_truth[0]) == [100, 800]


def test_undo_after_a_split_deletes_the_label_it_was_recorded_for(window):
    """The undo stack names scenes too, so it has to move with the labels."""
    _with_one_scene(window)
    window.ground_truth = {0: {800: _label((40, 90))}}
    window._session_undo = [(0, 800)]
    window.label_session = True
    window.current_frame_idx = 800

    window._do_split_at(400)
    window._session_undo_last()

    assert window._get_gt_for_frame(1, 800) is None

"""Per-scene annotations: the record, and the rule that keeps it consistent.

The four dicts this replaces were keyed by scene index with nothing owning the
invariant between them, so every caller had to remember that changing a scene's
axis makes the actions and the tracked positions derived from the old one stale.
"""
from __future__ import annotations

import numpy as np

from scripture.annotations import SceneAnnotations
from scripture.motion_tracker import AxisDefinition, TrackingResult


def _axis(tip=(10, 20), base=(30, 40), frame=0):
    return AxisDefinition(tip=tip, base=base, frame=frame)


def _tracked():
    return TrackingResult(timestamps_ms=np.array([0.0]), positions=np.array([0.5]))


class TestTheAxisChangeRule:

    def test_a_new_axis_drops_the_actions_and_positions_read_off_the_old_one(self):
        annotations = SceneAnnotations()
        annotations.set_axis(0, _axis())
        annotations.set_result(0, [{"at": 0, "pos": 50}], _tracked())

        annotations.set_axis(0, _axis(frame=99))

        assert annotations.axes[0].frame == 99
        assert 0 not in annotations.actions
        assert 0 not in annotations.tracking

    def test_taking_the_axis_away_hands_it_back_and_drops_the_derived_pair(self):
        annotations = SceneAnnotations()
        annotations.set_axis(0, _axis(frame=7))
        annotations.set_result(0, [{"at": 0, "pos": 50}], _tracked())

        taken = annotations.drop_axis(0)

        assert taken.frame == 7
        assert annotations.axes == {}
        assert annotations.actions == {}
        assert annotations.tracking == {}

    def test_discarding_a_run_keeps_the_axis_it_was_measured_against(self):
        annotations = SceneAnnotations()
        annotations.set_axis(0, _axis(frame=7))
        annotations.set_result(0, [{"at": 0, "pos": 50}], _tracked())

        annotations.discard_result(0)

        assert annotations.axes[0].frame == 7
        assert annotations.actions == {}
        assert annotations.tracking == {}


class TestReindexingOnASplit:
    """A split renumbers the scenes, so every mark has to find its new one."""

    def test_an_axis_lands_on_the_scene_holding_the_frame_it_was_drawn_on(self):
        annotations = SceneAnnotations()
        annotations.set_axis(0, _axis(frame=700))
        annotations.set_result(0, [{"at": 0, "pos": 50}], _tracked())

        annotations.reindex(lambda frame: 1 if frame >= 500 else 0)

        assert annotations.axes[1].frame == 700
        assert 1 in annotations.actions
        assert 1 in annotations.tracking
        assert 0 not in annotations.axes

    def test_a_label_lands_on_the_scene_holding_its_own_frame(self):
        annotations = SceneAnnotations()
        annotations.set_axis(0, _axis(frame=10))
        annotations.set_label(0, 10, {"is_action": False})
        annotations.set_label(0, 700, {"is_action": True})

        annotations.reindex(lambda frame: 1 if frame >= 500 else 0)

        assert annotations.labels[0] == {10: {"is_action": False}}
        assert annotations.labels[1] == {700: {"is_action": True}}

    def test_a_scene_with_only_labels_keeps_them(self):
        annotations = SceneAnnotations()
        annotations.set_label(3, 20, {"is_action": True})

        annotations.reindex(lambda frame: 0)

        assert annotations.labels == {0: {20: {"is_action": True}}}


class TestActionsForTheWholeVideo:
    """The automatic pipeline answers for every scene at once."""

    def test_a_fresh_bucketing_replaces_what_the_per_scene_runs_left(self):
        annotations = SceneAnnotations()
        annotations.set_axis(0, _axis())
        annotations.set_result(0, [{"at": 0, "pos": 1}], _tracked())

        annotations.replace_actions({1: [{"at": 9, "pos": 2}]})

        assert annotations.actions == {1: [{"at": 9, "pos": 2}]}
        assert annotations.axes[0].tip == (10, 20)

    def test_the_tracked_positions_can_be_dropped_on_their_own(self):
        annotations = SceneAnnotations()
        annotations.set_axis(0, _axis())
        annotations.set_result(0, [{"at": 0, "pos": 1}], _tracked())

        annotations.clear_tracking()

        assert annotations.tracking == {}
        assert annotations.actions == {0: [{"at": 0, "pos": 1}]}


class TestTheLabels:

    def test_a_scene_reset_clears_only_that_scene(self):
        annotations = SceneAnnotations()
        annotations.set_label(0, 5, {"is_action": False})
        annotations.set_label(1, 5, {"is_action": True})

        annotations.drop_labels(0)

        assert annotations.labels == {1: {5: {"is_action": True}}}

    def test_one_label_can_go_without_the_rest(self):
        annotations = SceneAnnotations()
        annotations.set_label(0, 5, {"is_action": False})
        annotations.set_label(0, 9, {"is_action": True})

        annotations.drop_label(0, 5)

        assert list(annotations.labels[0]) == [9]

    def test_dropping_the_last_label_leaves_the_scene_unlabeled(self):
        annotations = SceneAnnotations()
        annotations.set_label(0, 5, {"is_action": False})

        annotations.drop_label(0, 5)

        assert annotations.labels == {}


class TestLoadingADocument:

    def test_a_document_fills_the_four_kinds_of_mark(self):
        annotations = SceneAnnotations()
        annotations.load(axes={0: _axis(frame=3)}, actions={0: [{"at": 0, "pos": 1}]},
                         tracking={0: _tracked()}, labels={0: {4: {"is_action": True}}})

        assert annotations.axes[0].frame == 3
        assert annotations.actions[0] == [{"at": 0, "pos": 1}]
        assert 0 in annotations.tracking
        assert annotations.labels[0][4] == {"is_action": True}

    def test_clearing_leaves_nothing_behind(self):
        annotations = SceneAnnotations()
        annotations.set_axis(0, _axis())
        annotations.set_label(0, 1, {})

        annotations.clear()

        assert annotations.axes == {} and annotations.labels == {}

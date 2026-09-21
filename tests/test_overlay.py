"""What gets drawn over a frame, worked out without a window.

Both overlay builders lived on the Qt window, which nothing in the suite could
import, so the geometry that decides where the contact dot lands and which way
the motion is going had never been exercised at all.
"""
from __future__ import annotations

import numpy as np

from scripture.auto_funscript import PipelineResult, TrackSignal
from scripture.motion_tracker import AxisDefinition, TrackingResult
from scripture.overlay import auto_overlay, interpolate_actions, tracked_overlay
from scripture.scene import Scene

_AXIS = AxisDefinition(tip=(100, 100), base=(100, 300), frame=0)
_SCENE = Scene(0, 10)
_ACTIONS = [{"at": 0, "pos": 20}, {"at": 1000, "pos": 80}, {"at": 2000, "pos": 40}]


def _tracked(positions, **extra):
    return TrackingResult(
        timestamps_ms=np.arange(len(positions)) * 1000.0 / 30.0,
        positions=np.array(positions), **extra)


def _overlay(frame, actions=(), result=None):
    return tracked_overlay(axis=_AXIS, scene=_SCENE, frame_idx=frame, fps=30.0,
                           actions=list(actions), result=result)


class TestTheTrackedOverlay:

    def test_the_contact_dot_sits_that_far_from_the_base_toward_the_tip(self):
        overlay = _overlay(0, result=_tracked([0.25]))

        assert overlay["pos"] == 25
        assert overlay["contact_pt"] == (100, 250)

    def test_a_rising_position_reads_as_motion_toward_the_tip(self):
        assert _overlay(1, result=_tracked([0.2, 0.6]))["direction"] == 1
        assert _overlay(1, result=_tracked([0.6, 0.2]))["direction"] == -1

    def test_a_position_that_barely_moved_reads_as_still(self):
        assert _overlay(1, result=_tracked([0.5, 0.5]))["direction"] == 0

    def test_a_frame_past_the_tracked_run_has_nothing_to_draw(self):
        assert _overlay(5, result=_tracked([0.5])) is None

    def test_a_frame_with_neither_a_run_nor_an_action_has_nothing_to_draw(self):
        assert _overlay(0) is None

    def test_without_a_run_the_position_is_read_off_the_actions(self):
        overlay = _overlay(15, actions=_ACTIONS)  # frame 15 at 30fps == 500ms

        assert overlay["pos"] == 50
        assert overlay["is_action"] is False

    def test_a_frame_the_tracker_marked_takes_that_action_s_position(self):
        overlay = _overlay(0, actions=[{"at": 0, "pos": 90}], result=_tracked([0.25]))

        assert overlay["pos"] == 90
        assert overlay["is_action"] is True

    def test_tracked_ends_are_drawn_where_they_were_seen_not_where_drawn(self):
        overlay = _overlay(0, result=_tracked(
            [0.5], tip_coords=np.array([[10.0, 20.0]]),
            base_coords=np.array([[10.0, 40.0]])))

        assert overlay["axis"].tip == (10, 20)
        assert overlay["axis"].base == (10, 40)
        assert overlay["axis"].frame == _AXIS.frame


def _pipeline(lock, positions, **signal):
    return PipelineResult(
        signal=TrackSignal(dy=np.zeros(len(positions)), lock=lock,
                           rois=[None] * len(positions), **signal),
        positions=np.array(positions), actions=[], fps=30.0,
        start_frame=0, total_frames=len(positions))


def _auto(result, frame, **extra):
    fields = {"actions": [], "detection_frames": [], "last_anchor": None}
    fields.update(extra)
    return auto_overlay(result=result, frame_idx=frame, fps=30.0, **fields)


class TestTheAutomaticOverlay:

    def test_a_frame_past_the_run_has_nothing_to_draw(self):
        assert _auto(_pipeline(["anchor"], [50.0]), 5) is None

    def test_a_frame_the_tracker_never_locked_onto_reads_as_inactive(self):
        assert _auto(_pipeline(["none"], [50.0]), 0)["active"] is False

    def test_the_last_sighting_within_six_frames_is_the_one_shown(self):
        result = _pipeline(["anchor"] * 10, [50.0] * 10,
                           detections={0: ["near"], 8: ["far"]})

        assert _auto(result, 6, detection_frames=[0, 8])["detections"] == ["near"]
        assert _auto(result, 7, detection_frames=[0, 8])["detections"] is None
        assert _auto(result, 8, detection_frames=[0, 8])["detections"] == ["far"]

    def test_the_remembered_anchor_shows_only_while_the_sighting_is_lost(self):
        seen = _pipeline(["anchor"], [50.0], beliefs=[(1, 2, 3, 4)])
        lost = _pipeline(["coast"], [50.0], beliefs=[(1, 2, 3, 4)])

        assert _auto(seen, 0)["belief"] is None
        assert _auto(lost, 0)["belief"] == (1, 2, 3, 4)

    def test_the_remembered_anchor_says_how_long_since_it_was_seen(self):
        lost = _pipeline(["anchor", "coast", "coast"], [50.0] * 3,
                         beliefs=[None, (1, 2, 3, 4), (1, 2, 3, 4)])

        overlay = _auto(lost, 2, last_anchor=np.array([0, 0, 0]))

        assert overlay["belief_age_s"] == 2 / 30.0


class TestInterpolatingBetweenActions:
    """Kept from the window's own static method, which had these four pins."""

    def test_a_frame_on_an_action_takes_its_position_and_is_marked_one(self):
        assert interpolate_actions(_ACTIONS, 1010, 16.0) == (80, True)

    def test_a_frame_before_the_first_action_holds_its_position(self):
        assert interpolate_actions(_ACTIONS, -500, 16.0) == (20, False)

    def test_a_frame_after_the_last_action_holds_its_position(self):
        assert interpolate_actions(_ACTIONS, 5000, 16.0) == (40, False)

    def test_a_frame_between_two_actions_lands_on_the_line_between_them(self):
        assert interpolate_actions(_ACTIONS, 500, 16.0) == (50, False)
        assert interpolate_actions(_ACTIONS, 1500, 16.0) == (60, False)

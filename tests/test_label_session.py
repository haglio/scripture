"""A labeling session: which frame is being decided, and what can be undone.

The state machine was five attributes on the Qt window and eleven methods around
them, so the rule that finishing the schedule ends the session was written out
by hand at its one call site, and nothing could exercise the undo stack without
standing a window up.
"""
from __future__ import annotations

from scripture.label_session import LabelSession, next_scheduled, schedule_frames


class TestTheSchedule:

    def test_the_frames_to_decide_are_evenly_strided(self):
        assert schedule_frames(100, 400, stride=100) == [100, 200, 300, 400]

    def test_the_next_frame_to_label_is_the_first_undecided_one_past_here(self):
        assert next_scheduled([100, 200, 300, 400], annotated={200}, after=150) == 300

    def test_the_search_wraps_to_the_earliest_still_undecided(self):
        assert next_scheduled([100, 200, 300], annotated={300}, after=350) == 100

    def test_nothing_is_next_once_every_frame_is_decided(self):
        assert next_scheduled([100, 200], annotated={100, 200}, after=0) is None

    def test_progress_counts_only_the_scheduled_frames_that_are_done(self):
        assert LabelSession.progress([100, 200, 300], {100, 999}) == (1, 3)


class TestAdvancing:

    def test_aiming_lands_on_the_next_undecided_frame(self):
        session = LabelSession()
        session.active = True

        assert session.advance([100, 200], annotated=set(), after=100) == 200
        assert session.target == 200

    def test_running_out_of_frames_ends_the_session_itself(self):
        session = LabelSession()
        session.active = True
        session.target = 100

        assert session.advance([100], annotated={100}, after=0) is None
        assert not session.active
        assert session.target is None

    def test_pausing_forgets_the_frame_being_decided(self):
        session = LabelSession()
        session.active = True
        session.target = 100

        session.stop()

        assert not session.active
        assert session.target is None


class TestTheUndoStack:

    def test_a_label_is_recorded_and_the_next_search_starts_before_the_target(self):
        session = LabelSession()
        session.active = True
        session.target = 300

        assert session.after_label(scene=0, frame=250) == 299
        assert session.undo_last() == (0, 250)

    def test_with_no_target_the_search_starts_before_the_frame_just_labeled(self):
        session = LabelSession()
        session.active = True

        assert session.after_label(scene=0, frame=250) == 249

    def test_an_empty_stack_has_nothing_to_undo(self):
        assert LabelSession().undo_last() is None

    def test_a_reset_scene_loses_its_entries_and_keeps_the_others(self):
        session = LabelSession()
        session.after_label(scene=0, frame=10)
        session.after_label(scene=1, frame=20)

        session.forget_scene(0)

        assert session.undo_last() == (1, 20)
        assert session.undo_last() is None

    def test_a_split_renumbers_the_scene_each_entry_names(self):
        session = LabelSession()
        session.after_label(scene=0, frame=800)

        session.reindex(lambda frame: 1 if frame >= 400 else 0)

        assert session.undo_last() == (1, 800)

    def test_a_new_video_leaves_no_session_behind(self):
        session = LabelSession()
        session.active = True
        session.target = 100
        session.after_label(scene=0, frame=100)

        session.clear()

        assert not session.active
        assert session.target is None
        assert session.undo_last() is None


class TestTheAnchorToSearchFrom:

    def test_the_frame_being_decided_wins_over_the_one_on_screen(self):
        session = LabelSession()
        session.target = 300

        assert session.anchor(fallback=120) == 300

    def test_with_nothing_being_decided_the_frame_on_screen_is_it(self):
        assert LabelSession().anchor(fallback=120) == 120

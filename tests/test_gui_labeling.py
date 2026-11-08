from __future__ import annotations

import numpy as np
import pytest
from PyQt6.QtCore import QPoint, QPointF, Qt
from PyQt6.QtGui import QWheelEvent

from scripture.auto_funscript import PipelineResult, TrackSignal
from scripture.editing_rules import both_scenes_have_an_axis, moving_the_axis_needs_asking
from scripture.gui import TimelineState, TimelineWidget, Viewport
from scripture.motion_tracker import AxisDefinition
from scripture.scene import Scene
from tests.gui_doubles import Footage


class _FootageWithPictures(Footage):
    def frame_at(self, _index):
        return np.zeros((120, 160, 3), dtype=np.uint8)


@pytest.fixture
def scene(window):
    window.video = _FootageWithPictures(total_frames=1000)
    window.splits = []
    window._rebuild_scenes()
    return window


def _label(**fields):
    return {"tip": (40, 30), "base": (40, 150), "contact": (40, 90), "is_action": True, **fields}


def _wheel(x, delta):
    where = QPointF(x, 10)
    return QWheelEvent(where, where, QPoint(0, 0), QPoint(0, delta),
                       Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier,
                       Qt.ScrollPhase.NoScrollPhase, False)


class TestInheritingALabel:

    def test_a_frame_shows_the_nearest_label_rather_than_the_first(self, scene):
        scene.annotations.set_label(0, 100, _label(contact=(40, 60)))
        scene.annotations.set_label(0, 800, _label(contact=(40, 120)))

        assert scene._label_of_frame(0, 700)["contact"] == (40, 120)

    def test_a_frame_showing_another_frame_s_label_is_not_an_action_frame(self, scene):
        scene.annotations.set_label(0, 100, _label())

        assert scene._label_of_frame(0, 200)["is_action"] is False

    def test_a_new_label_starts_from_the_nearest_one(self, scene):
        scene.annotations.set_label(0, 100, _label(contact=(40, 60)))
        scene.annotations.set_label(0, 800, _label(contact=(40, 120)))
        scene.current_frame_idx = 700

        assert scene._ensure_label_of_frame()["contact"] == (40, 120)

    def test_changing_a_new_label_leaves_the_one_it_started_from_alone(self, scene):
        scene.annotations.set_label(0, 100, _label())
        scene.current_frame_idx = 200

        scene._ensure_label_of_frame()["contact"] = (1, 1)

        assert scene.annotations.label_at(0, 100)["contact"] == (40, 90)

    def test_dragging_a_label_s_point_moves_it(self, scene):
        scene.annotations.set_label(0, 100, _label())
        scene.current_frame_idx = 100

        scene._on_point_dragged("contact", 7, 8)

        assert scene.annotations.label_at(0, 100)["contact"] == (7, 8)


class TestNoContact:

    def test_n_gives_a_label_with_no_contact_one_at_the_middle_of_its_axis(self, scene):
        scene.annotations.set_label(0, 100, _label(tip=(10, 30), base=(50, 150), contact=None))
        scene.current_frame_idx = 100

        scene._label_no_contact()

        assert scene.annotations.label_at(0, 100)["contact"] == (30, 90)

    def test_in_a_label_session_n_records_no_contact_and_moves_to_the_next_frame(self, scene):
        scene.btn_label_session.setChecked(True)
        scene._toggle_label_session()
        first = scene.current_frame_idx

        scene._label_no_contact()

        assert scene.annotations.label_at(0, first)["contact"] is None
        assert scene.current_frame_idx == first + 60


class TestTheLabelSession:

    def test_the_session_asks_about_a_frame_every_two_seconds(self, scene):
        assert scene._session_schedule()[:3] == [0, 60, 120]

    def test_g_moves_on_without_labeling(self, scene):
        scene.btn_label_session.setChecked(True)
        scene._toggle_label_session()
        first = scene.current_frame_idx

        scene._session_skip()

        assert scene.current_frame_idx == first + 60
        assert scene.annotations.labels == {}

    def test_home_comes_back_to_the_frame_being_decided_after_scrubbing(self, scene):
        scene.btn_label_session.setChecked(True)
        scene._toggle_label_session()
        target = scene.current_frame_idx
        scene._show_frame(target + 7)

        scene._session_home()

        assert scene.current_frame_idx == target


class TestMovingTheAxis:

    def test_moving_a_drawn_axis_to_another_frame_is_asked_first(self):
        drawn = AxisDefinition(tip=(1, 2), base=(3, 4), frame=10)

        assert moving_the_axis_needs_asking(drawn, 20)

    def test_moving_it_on_its_own_frame_is_not_asked(self):
        drawn = AxisDefinition(tip=(1, 2), base=(3, 4), frame=10)

        assert not moving_the_axis_needs_asking(drawn, 10)

    def test_with_no_axis_there_is_nothing_to_ask(self):
        assert not moving_the_axis_needs_asking(None, 20)


class TestSplitting:

    def test_two_scenes_that_both_have_an_axis_will_not_merge(self):
        assert both_scenes_have_an_axis({0: "an axis", 1: "an axis"}, 0, 1)
        assert not both_scenes_have_an_axis({0: "an axis"}, 0, 1)

    @pytest.mark.parametrize("frame", [0, 1000, 400])
    def test_a_split_at_either_end_or_where_one_already_is_does_nothing(self, scene, frame):
        scene._do_split_at(400)

        scene._do_split_at(frame)

        assert scene.splits == [400]

    def test_s_on_a_frame_with_no_split_adds_one(self, scene):
        scene.current_frame_idx = 300

        scene._split_or_unsplit()

        assert scene.splits == [300]

    def test_a_split_sorts_the_automatic_pass_s_actions_into_the_new_scenes(self, scene):
        scene._set_auto_result(PipelineResult(
            signal=TrackSignal(dy=np.zeros(0), lock=[]), positions=np.zeros(0),
            actions=[{"at": 1000, "pos": 10}, {"at": 20000, "pos": 90}],
            fps=30.0, start_frame=0, total_frames=1000))

        scene._do_split_at(400)

        assert scene.annotations.actions == {
            0: [{"at": 1000, "pos": 10}], 1: [{"at": 20000, "pos": 90}]}


def test_the_left_arrow_stops_at_the_first_frame(scene):
    scene._show_frame(0)

    scene._frame_back()

    assert scene.current_frame_idx == 0


class TestTheTimelineWheel:

    @staticmethod
    def _timeline():
        timeline = TimelineWidget()
        timeline.resize(400, 60)
        timeline.set_state(TimelineState(scenes=[Scene(0, 8000)], total_frames=8000))
        return timeline

    def test_zooming_keeps_the_frame_under_the_pointer_under_it(self, qt_app):
        timeline = self._timeline()
        before = timeline._x_to_frame(100)

        timeline.wheelEvent(_wheel(100, 120))

        assert abs(timeline._x_to_frame(100) - before) <= 1
        assert timeline._zoom > 1.0


class TestTheViewport:

    def test_zoom_stays_within_its_limits(self, qt_app):
        viewport = Viewport()

        viewport.zoom = 99
        assert viewport.zoom == 3.0

        viewport.zoom = 0.0001
        assert viewport.zoom == 0.1

    def test_a_label_s_point_catches_the_click_ahead_of_the_axis_end_under_it(self, qt_app):
        viewport = Viewport()
        viewport.resize(320, 240)
        viewport._frame_w, viewport._frame_h = 320, 240
        viewport.set_axis(AxisDefinition(tip=(100, 100), base=(100, 200), frame=0))
        viewport.set_label({"contact": None, "tip": None, "base": (100, 100)})

        assert viewport._hit_point(*viewport._frame_to_viewport(100, 100)) == "base"


class _FootageCountingReads(Footage):
    def __init__(self, total_frames):
        super().__init__(total_frames)
        self.reads = 0

    def frame_at(self, _index):
        self.reads += 1
        return np.zeros((120, 160, 3), dtype=np.uint8)


def test_dragging_a_labels_point_reads_no_frame_from_the_video(window):
    footage = _FootageCountingReads(total_frames=1000)
    window.video = footage
    window.splits = []
    window._rebuild_scenes()
    window.annotations.set_label(0, 100, _label())
    window.current_frame_idx = 100
    footage.reads = 0

    window._on_point_dragged("contact", 7, 8)

    assert footage.reads == 0


def test_a_new_label_in_a_scene_with_no_drawn_axis_has_no_tip_and_no_base_the_automatic_sessions_draw_none_and_the_trainer_derives_the_axis_from_the_anchor_track(scene):
    scene.current_frame_idx = 100

    entry = scene._ensure_label_of_frame()

    assert (entry["tip"], entry["base"]) == (None, None)

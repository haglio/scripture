from __future__ import annotations

import numpy as np
import pytest
from PyQt6.QtCore import QEvent, QPointF, Qt
from PyQt6.QtGui import QMouseEvent, QPixmap

from scripture.editing_rules import next_frame, previous_frame
from scripture.gui import TimelineState, TimelineWidget, Viewport, action_dot_frames
from scripture.motion_tracker import AxisDefinition
from scripture.roi_lock import Detection
from scripture.scene import Scene


def _right_click_at(x, y):
    where = QPointF(x, y)
    return QMouseEvent(
        QEvent.Type.MouseButtonPress, where, where,
        Qt.MouseButton.RightButton, Qt.MouseButton.RightButton,
        Qt.KeyboardModifier.NoModifier)


@pytest.fixture
def emitted():
    return []


def test_the_timeline_asks_for_a_menu_at_the_frame_under_the_click(qt_app, emitted):
    timeline = TimelineWidget()
    timeline.resize(400, 60)
    timeline.set_state(TimelineState(total_frames=800))
    timeline.context_menu_requested.connect(emitted.append)

    timeline.mousePressEvent(_right_click_at(200, 30))

    assert emitted == [timeline._x_to_frame(200)]


def test_the_viewport_asks_for_a_menu_at_the_video_point_under_the_click(qt_app, emitted):
    viewport = Viewport()
    viewport.resize(320, 240)
    viewport._frame_w, viewport._frame_h = 320, 240
    viewport.context_menu_requested.connect(lambda fx, fy: emitted.append((fx, fy)))

    viewport.mousePressEvent(_right_click_at(160, 120))

    assert emitted == [viewport._viewport_to_frame(160, 120)]


def test_the_timeline_paints_a_scene_with_labels_and_actions(qt_app):
    timeline = TimelineWidget()
    timeline.resize(400, 60)
    timeline.set_state(TimelineState(
        scenes=[Scene(0, 400), Scene(400, 800)],
        scene_axes={0: AxisDefinition(tip=(40, 30), base=(40, 150), frame=5)},
        scene_actions={0: [{"at": 100, "pos": 50}]},
        splits=[400],
        total_frames=800,
        current_frame=200,
        labels={1: {600: {"tip": None, "base": None,
                                "contact": (40, 90), "is_action": True}}},
        fps=30.0,
    ))

    timeline.render(QPixmap(timeline.size()))


def test_a_scene_with_labels_shows_its_labeled_action_frames_and_not_its_computed_ones():
    state = TimelineState(
        scene_actions={0: [{"at": 1000, "pos": 50}], 1: [{"at": 2000, "pos": 50}]},
        labels={0: {7: {"is_action": True}, 8: {"is_action": False}}},
        fps=30.0,
    )

    assert action_dot_frames(state) == [7, 60]


def test_the_next_frame_of_a_list_wraps_round_to_its_first():
    assert next_frame([10, 20], 10) == 20
    assert next_frame([10, 20], 20) == 10


def test_the_previous_frame_of_a_list_wraps_round_to_its_last():
    assert previous_frame([10, 20], 20) == 10
    assert previous_frame([10, 20], 10) == 20


def test_an_empty_list_of_frames_goes_nowhere():
    assert next_frame([], 5) is None
    assert previous_frame([], 5) is None


def test_the_viewport_paints_what_the_automatic_tracker_saw(qt_app):
    viewport = Viewport()
    viewport.resize(320, 240)
    viewport.set_frame(np.zeros((240, 320, 3), dtype=np.uint8))
    viewport.set_auto_overlay({
        "detections": [Detection("made up", 0.5, (10, 10, 40, 40))],
        "belief": (20, 20, 30, 30), "belief_age_s": 2.0,
        "roi": (5, 5, 100, 100), "lock": "coast",
        "active": True, "pos": 40, "is_action": True,
    })

    viewport.render(QPixmap(viewport.size()))


def test_the_viewport_paints_no_target_when_the_tracker_has_none(qt_app):
    viewport = Viewport()
    viewport.resize(320, 240)
    viewport.set_frame(np.zeros((240, 320, 3), dtype=np.uint8))
    viewport.set_auto_overlay({"active": False})

    viewport.render(QPixmap(viewport.size()))

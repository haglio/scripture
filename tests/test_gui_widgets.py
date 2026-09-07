"""What the two custom widgets emit when they are clicked.

Neither had any test, so both carried a right-click payload -- the screen
coordinates of the click -- that both handlers dropped on the floor, papered
over by whitelist entries reading "the signal emits them; the slot must
accept". These are the app's own signals; it decides what they carry.
"""

from __future__ import annotations

import pytest
from PyQt6.QtCore import QEvent, QPointF, Qt
from PyQt6.QtGui import QMouseEvent, QPixmap

from scripture.gui import FrameCanvas, TimelineWidget
from scripture.motion_tracker import AxisDefinition
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
    timeline.total_frames = 800
    timeline.context_menu_requested.connect(emitted.append)

    timeline.mousePressEvent(_right_click_at(200, 30))

    assert emitted == [timeline._x_to_frame(200)]


def test_the_canvas_asks_for_a_menu_at_the_video_point_under_the_click(qt_app, emitted):
    canvas = FrameCanvas()
    canvas.resize(320, 240)
    canvas._frame_w, canvas._frame_h = 320, 240
    canvas.context_menu_requested.connect(lambda fx, fy: emitted.append((fx, fy)))

    canvas.mousePressEvent(_right_click_at(160, 120))

    assert emitted == [canvas._canvas_to_frame(160, 120)]


def test_the_timeline_paints_a_scene_with_labels_and_actions(qt_app):
    """paintEvent is the widget's longest method and had no coverage at all.
    Rendering it offscreen is what makes changes to what it reads safe."""
    timeline = TimelineWidget()
    timeline.resize(400, 60)
    timeline.set_state(
        scenes=[Scene(0, 400), Scene(400, 800)],
        scene_axes={0: AxisDefinition(tip=(40, 30), base=(40, 150), frame=5)},
        scene_actions={0: [{"at": 100, "pos": 50}]},
        splits=[400],
        total_frames=800,
        current_frame=200,
        ground_truth={1: {600: {"tip": None, "base": None,
                                "contact": (40, 90), "is_action": True}}},
        fps=30.0,
    )

    timeline.render(QPixmap(timeline.size()))

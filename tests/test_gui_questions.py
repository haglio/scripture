from __future__ import annotations

from PySide6.QtWidgets import QMessageBox
from shared_ui.testing import questions_answered

from scripture import gui
from tests.gui_doubles import CloseEvent


class _Labels:
    def __init__(self):
        self.dropped = []

    def labels_of(self, _idx):
        return [10, 20]

    def axis_of(self, _idx):
        return object()

    def drop_labels(self, idx):
        self.dropped.append(idx)


class _Running:
    stopped = False

    def isRunning(self):
        return True

    def stop(self):
        self.stopped = True


def _yes_is_the_default(box):
    return box.default == QMessageBox.StandardButton.Yes


def test_resetting_a_scene_s_labels_asks_with_yes_as_the_default(window, monkeypatch):
    labels = _Labels()
    monkeypatch.setattr(window, "annotations", labels)
    monkeypatch.setattr(window, "_current_scene_idx", lambda: 0)
    asked = questions_answered(monkeypatch, QMessageBox.StandardButton.No)

    window._reset_scene_labels()

    [box] = asked
    assert box.text == "Delete all 2 labeled frames in this scene?"
    assert _yes_is_the_default(box)
    assert labels.dropped == []


def test_moving_the_axis_asks_with_yes_as_the_default(window, monkeypatch):
    monkeypatch.setattr(window, "annotations", _Labels())
    monkeypatch.setattr(gui, "moving_the_axis_needs_asking", lambda *_args: True)
    asked = questions_answered(monkeypatch, QMessageBox.StandardButton.Yes)

    moved = window._may_move_the_axis(0)

    [box] = asked
    assert moved
    assert box.text == "Move axis to this frame?"
    assert _yes_is_the_default(box)


def test_closing_while_processing_asks_with_yes_as_the_default(window, monkeypatch):
    running = _Running()
    monkeypatch.setattr(window, "_worker", running)
    asked = questions_answered(monkeypatch, QMessageBox.StandardButton.No)
    event = CloseEvent()

    window.closeEvent(event)

    [box] = asked
    assert box.text == "Processing is still running. Abort and close?"
    assert _yes_is_the_default(box)
    assert (event.accepted, running.stopped) == (False, False)


def test_asking_to_open_a_video_first_offers_nothing_to_copy(window, monkeypatch):
    shown = []
    monkeypatch.setattr(gui, "show_warning",
                        lambda title, message, **how: shown.append((title, how)))
    window.video_path = None

    window._require_video()

    assert shown == [("No video", {"copyable": False})]

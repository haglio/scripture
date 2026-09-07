"""Closing with "Save" must not close over a save that never happened.

The close guard asked "Save before closing?", called `_save_project()` and then
accepted the event whatever came back -- and `_save_project` has two ways to
return having written nothing. So the one answer that means "keep my work" was
the one that threw it away.
"""

from __future__ import annotations

from PyQt6.QtWidgets import QMessageBox

from scripture import gui
from scripture.project import load_project


class _CloseEvent:
    """Stands in for the QCloseEvent Qt hands to `closeEvent`."""

    def __init__(self):
        self.accepted = None

    def accept(self):
        self.accepted = True

    def ignore(self):
        self.accepted = False


class _Dialogs:
    """Stands in for `gui.QMessageBox`, answering with a queued button."""

    StandardButton = QMessageBox.StandardButton

    def __init__(self, *answers):
        self._answers = list(answers)
        self.warnings = []

    def question(self, *_args):
        return self._answers.pop(0)

    def warning(self, _parent, title, text):
        self.warnings.append((title, text))


class _SaveDialog:
    """Stands in for `gui.QFileDialog`, offering one Save As answer."""

    def __init__(self, chosen=""):
        self._chosen = chosen

    def getSaveFileName(self, *_args):  # noqa: N802 - Qt spells it this way
        return self._chosen, ""


def _close_asking_to_save(window, monkeypatch, save_dialog):
    monkeypatch.setattr(gui, "QMessageBox", _Dialogs(QMessageBox.StandardButton.Save))
    monkeypatch.setattr(gui, "QFileDialog", save_dialog)
    event = _CloseEvent()
    window.closeEvent(event)
    return event


def test_a_cancelled_save_as_dialog_leaves_the_window_open(window, monkeypatch, tmp_path):
    window.video_path = str(tmp_path / "example clip.mp4")
    window._mark_dirty()

    event = _close_asking_to_save(window, monkeypatch, _SaveDialog(chosen=""))

    assert event.accepted is False
    assert window._dirty, "the unsaved work is still unsaved"


def test_closing_with_no_video_open_leaves_the_window_open(window, monkeypatch):
    """The other branch that returns having written nothing."""
    window._mark_dirty()

    event = _close_asking_to_save(window, monkeypatch, _SaveDialog())

    assert event.accepted is False
    assert window._dirty


def test_choosing_a_file_saves_it_and_closes(window, monkeypatch, tmp_path):
    window.video_path = str(tmp_path / "example clip.mp4")
    window.splits = [120]
    window._mark_dirty()
    chosen = tmp_path / "example clip.scripture"

    event = _close_asking_to_save(window, monkeypatch, _SaveDialog(chosen=str(chosen)))

    assert event.accepted is True
    assert not window._dirty
    assert load_project(chosen)["splits"] == [120]


def test_discarding_closes_without_writing_anything(window, monkeypatch, tmp_path):
    window.video_path = str(tmp_path / "example clip.mp4")
    window._mark_dirty()
    monkeypatch.setattr(
        gui, "QMessageBox", _Dialogs(QMessageBox.StandardButton.Discard))
    monkeypatch.setattr(gui, "QFileDialog", _SaveDialog())
    event = _CloseEvent()

    window.closeEvent(event)

    assert event.accepted is True
    assert not list(tmp_path.glob("*.scripture"))

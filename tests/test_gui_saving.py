from __future__ import annotations

from PySide6.QtWidgets import QMessageBox
from shared_ui.testing import alert_shown

from scripture import gui
from scripture.project import load_project
from tests.gui_doubles import (
    CloseEvent,
    Dialogs,
    FileDialog,
)


def _close_asking_to_save(window, monkeypatch, save_dialog):
    monkeypatch.setattr(gui, "QMessageBox", Dialogs(QMessageBox.StandardButton.Save))
    monkeypatch.setattr(gui, "QFileDialog", save_dialog)
    event = CloseEvent()
    window.closeEvent(event)
    return event


def test_a_cancelled_save_as_dialog_leaves_the_window_open(window, monkeypatch, tmp_path):
    window.video_path = str(tmp_path / "example clip.mp4")
    window._mark_dirty()

    event = _close_asking_to_save(window, monkeypatch, FileDialog(chosen=""))

    assert event.accepted is False
    assert window._dirty, "the unsaved work is still unsaved"


def test_closing_with_no_video_open_leaves_the_window_open(window, monkeypatch):
    window._mark_dirty()

    with alert_shown() as shown:
        event = _close_asking_to_save(window, monkeypatch, FileDialog())

    assert event.accepted is False
    assert window._dirty
    assert shown.call_args.args[0] == "No video"


def test_choosing_a_file_saves_it_and_closes(window, monkeypatch, tmp_path):
    window.video_path = str(tmp_path / "example clip.mp4")
    window.splits = [120]
    window._mark_dirty()
    chosen = tmp_path / "example clip.scripture"

    event = _close_asking_to_save(window, monkeypatch, FileDialog(chosen=str(chosen)))

    assert event.accepted is True
    assert not window._dirty
    assert load_project(chosen)["splits"] == [120]


def test_discarding_closes_without_writing_anything(window, monkeypatch, tmp_path):
    window.video_path = str(tmp_path / "example clip.mp4")
    window._mark_dirty()
    monkeypatch.setattr(
        gui, "QMessageBox", Dialogs(QMessageBox.StandardButton.Discard))
    monkeypatch.setattr(gui, "QFileDialog", FileDialog())
    event = CloseEvent()

    window.closeEvent(event)

    assert event.accepted is True
    assert not list(tmp_path.glob("*.scripture"))


def test_the_unsaved_changes_question_has_save_as_the_default(window, monkeypatch, tmp_path):
    window.video_path = str(tmp_path / "example clip.mp4")
    window._mark_dirty()
    dialogs = Dialogs(QMessageBox.StandardButton.Discard)
    monkeypatch.setattr(gui, "QMessageBox", dialogs)
    monkeypatch.setattr(gui, "QFileDialog", FileDialog())

    window.closeEvent(CloseEvent())

    [(_window, title, _text, _buttons, default)] = dialogs.asked
    assert (title, default) == ("Unsaved changes", QMessageBox.StandardButton.Save)

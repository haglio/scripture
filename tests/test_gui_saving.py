from __future__ import annotations

from PyQt6.QtWidgets import QMessageBox

from scripture import gui
from scripture.project import load_project
from tests.gui_doubles import (
    CloseEvent,
    DialogsOfferingOnlyWhatTheWindowUsesSoMoreFailsLoudly,
    FileDialogOfferingOnlyWhatTheWindowUsesSoMoreFailsLoudly,
)


def _close_asking_to_save(window, monkeypatch, save_dialog):
    monkeypatch.setattr(gui, "QMessageBox", DialogsOfferingOnlyWhatTheWindowUsesSoMoreFailsLoudly(QMessageBox.StandardButton.Save))
    monkeypatch.setattr(gui, "QFileDialog", save_dialog)
    event = CloseEvent()
    window.closeEvent(event)
    return event


def test_a_cancelled_save_as_dialog_leaves_the_window_open(window, monkeypatch, tmp_path):
    window.video_path = str(tmp_path / "example clip.mp4")
    window._mark_dirty()

    event = _close_asking_to_save(window, monkeypatch, FileDialogOfferingOnlyWhatTheWindowUsesSoMoreFailsLoudly(chosen=""))

    assert event.accepted is False
    assert window._dirty, "the unsaved work is still unsaved"


def test_closing_with_no_video_open_leaves_the_window_open(window, monkeypatch):
    window._mark_dirty()

    event = _close_asking_to_save(window, monkeypatch, FileDialogOfferingOnlyWhatTheWindowUsesSoMoreFailsLoudly())

    assert event.accepted is False
    assert window._dirty


def test_choosing_a_file_saves_it_and_closes(window, monkeypatch, tmp_path):
    window.video_path = str(tmp_path / "example clip.mp4")
    window.splits = [120]
    window._mark_dirty()
    chosen = tmp_path / "example clip.scripture"

    event = _close_asking_to_save(window, monkeypatch, FileDialogOfferingOnlyWhatTheWindowUsesSoMoreFailsLoudly(chosen=str(chosen)))

    assert event.accepted is True
    assert not window._dirty
    assert load_project(chosen)["splits"] == [120]


def test_discarding_closes_without_writing_anything(window, monkeypatch, tmp_path):
    window.video_path = str(tmp_path / "example clip.mp4")
    window._mark_dirty()
    monkeypatch.setattr(
        gui, "QMessageBox", DialogsOfferingOnlyWhatTheWindowUsesSoMoreFailsLoudly(QMessageBox.StandardButton.Discard))
    monkeypatch.setattr(gui, "QFileDialog", FileDialogOfferingOnlyWhatTheWindowUsesSoMoreFailsLoudly())
    event = CloseEvent()

    window.closeEvent(event)

    assert event.accepted is True
    assert not list(tmp_path.glob("*.scripture"))

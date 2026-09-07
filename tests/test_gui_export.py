"""Export is what the app is for, and nothing had exercised the write.

Replacing `save_funscript`'s body with `return None` left the suite at its exact
count: `_export` gathers every scene's actions, builds the funscript and writes
it, and none of those three steps was reached by a test.
"""

from __future__ import annotations

import json

from scripture import gui
from tests.gui_doubles import Dialogs, FileDialog


def _ready_to_export(window, tmp_path):
    window.video_path = str(tmp_path / "example clip.mp4")
    window.total_frames = 600
    window.fps = 30.0
    window.scene_actions = {
        0: [{"at": 0, "pos": 50}, {"at": 1000, "pos": 90}],
        1: [{"at": 2000, "pos": 10}],
    }
    return window


def test_export_writes_every_scene_s_actions_in_time_order(
        window, tmp_path, monkeypatch):
    _ready_to_export(window, tmp_path)
    target = tmp_path / "example clip.funscript"
    monkeypatch.setattr(gui, "QFileDialog", FileDialog(chosen=str(target)))

    window._export()

    written = json.loads(target.read_text(encoding="utf-8"))
    assert [action["at"] for action in written["actions"]] == [0, 1000, 2000]
    assert written["metadata"]["creator"] == "scripture"
    assert written["metadata"]["duration"] == 20


def test_a_cancelled_export_dialog_writes_nothing(window, tmp_path, monkeypatch):
    _ready_to_export(window, tmp_path)
    monkeypatch.setattr(gui, "QFileDialog", FileDialog(chosen=""))

    window._export()

    assert not list(tmp_path.glob("*.funscript"))


def test_exporting_before_a_scene_is_processed_says_so_and_writes_nothing(
        window, tmp_path, monkeypatch):
    window.video_path = str(tmp_path / "example clip.mp4")
    dialogs = Dialogs()
    monkeypatch.setattr(gui, "QMessageBox", dialogs)
    monkeypatch.setattr(
        gui, "QFileDialog", FileDialog(chosen=str(tmp_path / "example clip.funscript")))

    window._export()

    assert [title for title, _text in dialogs.shown] == ["No data"]
    assert not list(tmp_path.glob("*.funscript"))

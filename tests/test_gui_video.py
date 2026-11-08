from __future__ import annotations

import ast
from pathlib import Path

from scripture import gui
from scripture.video import VideoSource
from tests.gui_doubles import (
    Capture,
    Dialogs,
    FileDialog,
    Footage,
)


def _labeled(window, frame):
    window.video = Footage(total_frames=1000)
    window.splits = []
    window._rebuild_scenes()
    window.annotations.set_label(0, frame, {
        "tip": (40, 30), "base": (40, 150), "contact": (40, 90), "is_action": True})
    window.session.after_label(0, frame)


def _open(window, monkeypatch, path):
    monkeypatch.setattr(gui, "QFileDialog", FileDialog(chosen=str(path)))
    monkeypatch.setattr(gui.cv2, "VideoCapture", lambda _path: Capture())
    window._open_video()


def test_a_new_video_is_not_saved_with_the_last_one_s_labels(
        window, tmp_path, monkeypatch):
    _labeled(window, 800)

    _open(window, monkeypatch, tmp_path / "another clip.mp4")

    assert window._build_state()["ground_truth"] == {}


def test_a_new_video_does_not_inherit_the_label_session(window, tmp_path, monkeypatch):
    _labeled(window, 800)
    window.session.active = True
    window.btn_label_session.setChecked(True)
    window.session.target = 800

    _open(window, monkeypatch, tmp_path / "another clip.mp4")

    assert not window.session.active
    assert not window.btn_label_session.isChecked()
    assert window.session.target is None
    assert window.session.undo_last() is None


def test_a_video_that_will_not_open_says_so_and_keeps_the_one_in_hand(
        window, tmp_path, monkeypatch):
    playing = Capture()
    window.video = VideoSource(playing)
    kept = window.video
    dialogs = Dialogs()
    monkeypatch.setattr(gui, "QMessageBox", dialogs)
    monkeypatch.setattr(
        gui, "QFileDialog", FileDialog(chosen=str(tmp_path / "unreadable clip.mp4")))
    monkeypatch.setattr(gui.cv2, "VideoCapture", lambda _path: Capture(opened=False))

    window._open_video()

    assert window.video is kept
    assert not playing.released
    assert [title for title, _text in dialogs.shown] == ["Video not found"]


def test_the_window_opens_no_capture_of_its_own_video_py_reads_its_footage():
    package = Path(__file__).resolve().parents[1] / "scripture"
    opened_in_the_window = []
    for source in (package / "gui.py", package / "overlay.py"):
        for node in ast.walk(ast.parse(source.read_text(encoding="utf-8"))):
            names_a_capture = (
                (isinstance(node, ast.Attribute) and node.attr == "VideoCapture")
                or (isinstance(node, ast.ImportFrom) and any(alias.name == "VideoCapture" for alias in node.names)))
            if names_a_capture:
                opened_in_the_window.append(f"{source.name}:{node.lineno}")
    assert opened_in_the_window == []

"""Opening a video starts a new document; nothing of the last one may survive.

`_open_video` cleared the splits, the scenes and the three per-scene dicts and
left the hand-placed labels, the label session and its undo stack alone. The
timeline then painted another video's labels over this one, the session counted
them as frames already done, and the next save wrote them into the new video's
project file.
"""

from __future__ import annotations

from scripture import gui
from scripture.video import VideoSource
from tests.gui_doubles import Capture, Dialogs, FileDialog


def _labeled(window, frame):
    window.total_frames = 1000
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
    """The same answer _do_load already gave a project whose video had moved."""
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

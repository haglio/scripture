"""Opening a video starts a new document; nothing of the last one may survive.

`_open_video` cleared the splits, the scenes and the three per-scene dicts and
left the hand-placed labels, the label session and its undo stack alone. The
timeline then painted another video's labels over this one, the session counted
them as frames already done, and the next save wrote them into the new video's
project file.
"""

from __future__ import annotations

from scripture import gui
from tests.gui_doubles import Capture, FileDialog


def _labeled(window, frame):
    window.total_frames = 1000
    window.splits = []
    window._rebuild_scenes()
    window.ground_truth = {0: {frame: {
        "tip": (40, 30), "base": (40, 150), "contact": (40, 90), "is_action": True}}}
    window._session_undo = [(0, frame)]


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
    window.label_session = True
    window.btn_label_session.setChecked(True)
    window._session_target = 800

    _open(window, monkeypatch, tmp_path / "another clip.mp4")

    assert not window.label_session
    assert not window.btn_label_session.isChecked()
    assert window._session_target is None
    assert window._session_undo == []

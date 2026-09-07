"""Opening a project must not put the window out of the video business.

`_do_load` released the open capture and only then tried the project's video.
When that video had moved, the "Video not found" dialog left `self.cap` pointing
at a released capture -- still truthy, so `_show_frame`'s only guard passed and
every frame read after it came back empty. The canvas simply stopped updating,
with nothing said after that one dialog.
"""

from __future__ import annotations

from scripture import gui
from scripture.project import save_project
from tests.gui_doubles import Capture, Dialogs


def _project_naming(video, path):
    save_project(str(path), {"video_path": str(video), "splits": []})
    return str(path)


def test_a_project_whose_video_moved_leaves_the_open_one_playing(
        window, tmp_path, monkeypatch):
    playing = Capture()
    window.cap = playing
    dialogs = Dialogs()
    monkeypatch.setattr(gui, "QMessageBox", dialogs)
    project = _project_naming(tmp_path / "moved clip.mp4", tmp_path / "one.scripture")

    window._do_load(project)

    assert window.cap is playing
    assert not playing.released
    assert [title for title, _text in dialogs.shown] == ["Video not found"]


def test_a_project_that_opens_releases_the_capture_it_replaces(
        window, tmp_path, monkeypatch):
    replaced = Capture()
    window.cap = replaced
    opening = Capture()
    monkeypatch.setattr(gui.cv2, "VideoCapture", lambda _path: opening)
    project = _project_naming(tmp_path / "example clip.mp4", tmp_path / "two.scripture")

    window._do_load(project)

    assert window.cap is opening
    assert replaced.released


def test_a_loaded_project_starts_its_own_labeling_session(
        window, tmp_path, monkeypatch):
    window.ground_truth = {0: {800: {
        "tip": None, "base": None, "contact": (40, 90), "is_action": True}}}
    window._session_undo = [(0, 800)]
    window.label_session = True
    window.btn_label_session.setChecked(True)
    monkeypatch.setattr(gui.cv2, "VideoCapture", lambda _path: Capture())
    project = _project_naming(tmp_path / "example clip.mp4", tmp_path / "three.scripture")

    window._do_load(project)

    assert window.ground_truth == {}
    assert window._session_undo == []
    assert not window.label_session
    assert not window.btn_label_session.isChecked()

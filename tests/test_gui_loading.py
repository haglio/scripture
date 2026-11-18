from __future__ import annotations

from shared_ui.alert import Level
from shared_ui.testing import alert_shown

from scripture import gui
from scripture.project import save_project
from scripture.video import VideoSource
from tests.gui_doubles import (
    Capture,
)


def _project_naming(video, path):
    save_project(str(path), {"video_path": str(video), "splits": []})
    return str(path)


def test_a_project_whose_video_moved_leaves_the_open_one_playing(
        window, tmp_path, monkeypatch):
    playing = Capture()
    window.video = VideoSource(playing)
    kept = window.video
    project = _project_naming(tmp_path / "moved clip.mp4", tmp_path / "one.scripture")

    with alert_shown() as shown:
        window._do_load(project)

    assert window.video is kept
    assert not playing.released
    assert [call.args[0] for call in shown.call_args_list] == ["Video not found"]
    assert shown.call_args.kwargs["level"] is Level.ERROR


def test_a_project_that_opens_releases_the_capture_it_replaces(
        window, tmp_path, monkeypatch):
    replaced = Capture()
    window.video = VideoSource(replaced)
    opening = Capture()
    monkeypatch.setattr(gui.cv2, "VideoCapture", lambda _path: opening)
    project = _project_naming(tmp_path / "example clip.mp4", tmp_path / "two.scripture")

    window._do_load(project)

    assert window.video.frame_at(0) is None
    assert replaced.released


def test_a_loaded_project_starts_its_own_labeling_session(
        window, tmp_path, monkeypatch):
    window.annotations.set_label(0, 800, {
        "tip": None, "base": None, "contact": (40, 90), "is_action": True})
    window.session.after_label(0, 800)
    window.session.active = True
    window.btn_label_session.setChecked(True)
    monkeypatch.setattr(gui.cv2, "VideoCapture", lambda _path: Capture())
    project = _project_naming(tmp_path / "example clip.mp4", tmp_path / "three.scripture")

    window._do_load(project)

    assert window.annotations.labels == {}
    assert window.session.undo_last() is None
    assert not window.session.active
    assert not window.btn_label_session.isChecked()

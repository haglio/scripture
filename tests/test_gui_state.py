"""The shape of a `.scripture` file, which is not this repo's alone to change.

`_build_state` and `_do_load` are the whole schema -- the key names, the
`str()`-keyed scene indices, the tuple/list coercions, the numpy round trip --
and they sit inside the window, where no test could reach them. Meanwhile
evolver reads every `sessions/*.scripture`, takes the top-level `video_path`
and writes the file back whole, so a key that moves here breaks a repo that
never imported this one.
"""

from __future__ import annotations

import json

import numpy as np

from scripture import gui
from scripture.motion_tracker import AxisDefinition, TrackingResult
from scripture.project import PROJECT_FORMAT_VERSION, save_project
from tests.gui_doubles import Capture

_LABEL = {"tip": (40, 30), "base": (40, 150), "contact": (40, 90), "is_action": True}
_LABEL_ON_DISK = {"tip": [40, 30], "base": [40, 150], "contact": [40, 90],
                  "is_action": True}


def _populated(window, video_path):
    """A document with one of everything the schema can hold."""
    window.video_path = str(video_path)
    window.total_frames = 1000
    window.splits = [400]
    window._rebuild_scenes()
    window.scene_axes[0] = AxisDefinition(tip=(40, 30), base=(40, 150), frame=5)
    window.scene_actions[0] = [{"at": 0, "pos": 50}, {"at": 120, "pos": 90}]
    window.scene_positions[0] = TrackingResult(
        timestamps_ms=np.array([0.0, 33.0]),
        positions=np.array([0.5, 0.9]),
        tip_coords=np.array([[40.0, 30.0], [41.0, 31.0]]),
        base_coords=np.array([[40.0, 150.0], [40.0, 151.0]]),
    )
    window.ground_truth[1] = {800: dict(_LABEL)}
    window.current_frame_idx = 800
    return window


def _saved(window, tmp_path, name="example clip.scripture"):
    path = tmp_path / name
    save_project(str(path), window._build_state())
    return path


def _reopened(project, tmp_path, monkeypatch):
    """A second window with that project loaded, its video answered for."""
    monkeypatch.setattr(gui, "_LAST_PROJECT_FILE", tmp_path / ".last_session")
    monkeypatch.setattr(gui.cv2, "VideoCapture", lambda _path: Capture())
    reopened = gui.App()
    reopened._do_load(str(project))
    return reopened


def test_a_saved_project_holds_exactly_these_top_level_keys(window, tmp_path):
    state = _populated(window, tmp_path / "example clip.mp4")._build_state()

    assert sorted(state) == [
        "actions", "auto", "axes", "current_frame", "ground_truth",
        "splits", "tracking", "version", "video_path",
    ]


def test_the_video_path_another_repo_rewrites_is_a_top_level_string(window, tmp_path):
    video = tmp_path / "example clip.mp4"

    state = _populated(window, video)._build_state()

    assert state["video_path"] == str(video)


def test_a_saved_project_says_which_shape_the_repo_that_rewrites_it_is_reading(
    window, tmp_path
):
    """Evolver writes this file back whole.  Without a version it had to assume
    the shape, so a change here would have reached it as a wrong rewrite rather
    than as a refusal."""
    state = _populated(window, tmp_path / "example clip.mp4")._build_state()

    assert state["version"] == PROJECT_FORMAT_VERSION


def test_the_scene_indices_are_strings_so_the_file_is_plain_json(window, tmp_path):
    state = _populated(window, tmp_path / "example clip.mp4")._build_state()

    assert list(state["axes"]) == ["0"]
    assert list(state["actions"]) == ["0"]
    assert list(state["tracking"]) == ["0"]
    assert list(state["ground_truth"]) == ["1"]
    assert list(state["ground_truth"]["1"]) == ["800"]


def test_each_scene_entry_carries_the_keys_the_loader_reads(window, tmp_path):
    """The round trip below reads what _build_state writes, so a field it stops
    writing goes missing from both sides of it and only this notices."""
    state = _populated(window, tmp_path / "example clip.mp4")._build_state()

    assert sorted(state["axes"]["0"]) == ["base", "frame", "tip"]
    assert sorted(state["tracking"]["0"]) == [
        "base_coords", "positions", "provenance", "timestamps_ms", "tip_coords"]
    assert sorted(state["ground_truth"]["1"]["800"]) == [
        "base", "contact", "is_action", "tip"]


def test_a_populated_project_comes_back_the_way_it_went_in(
        window, tmp_path, monkeypatch):
    saved = _populated(window, tmp_path / "example clip.mp4")._build_state()
    project = _saved(window, tmp_path)

    reopened = _reopened(project, tmp_path, monkeypatch)

    assert reopened._build_state() == saved


def test_a_project_whose_video_path_was_rewritten_opens_against_the_new_one(
        window, tmp_path, monkeypatch):
    _populated(window, tmp_path / "example clip.mp4")
    project = _saved(window, tmp_path)
    moved = tmp_path / "archive" / "example clip.mp4"
    payload = json.loads(project.read_text(encoding="utf-8"))
    payload["video_path"] = str(moved)
    project.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    reopened = _reopened(project, tmp_path, monkeypatch)

    assert reopened.video_path == str(moved)
    assert reopened._build_state()["ground_truth"] == {"1": {"800": _LABEL_ON_DISK}}

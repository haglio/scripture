from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pytest
from app_support import haglio_settings, video_ids

from scripture.auto_funscript import PipelineResult, TrackSignal
from scripture.motion_tracker import AxisDefinition, TrackingResult
from scripture.project import (
    PROJECT_FORMAT_VERSION,
    Project,
    load_project,
    name_the_videos_by_id,
    project_from_state,
    provenance_of_actions,
    reopen_last_project,
    save_project,
    state_from_project,
)
from tests.stamps import A_STAMP, METHOD_KEY_ON_DISK


def _tracked(provenance):
    return TrackingResult(
        timestamps_ms=np.array([0.0]), positions=np.array([0.5]), provenance=provenance)


def _auto_pass(provenance):
    return PipelineResult(
        signal=TrackSignal(dy=np.array([0.0]), lock=["none"], rois=[None]),
        positions=np.array([50.0]), actions=[], fps=30.0, start_frame=0, total_frames=1,
        provenance=provenance)


class TestProjectPersistence:

    def test_round_trip(self, tmp_path):
        path = tmp_path / "test.scripture"
        state = {
            "video_path": "C:/videos/test.mp4",
            "splits": [100, 500, 1200],
            "axes": {
                "0": {"tip": [10, 20], "base": [30, 40], "frame": 50},
                "2": {"tip": [100, 200], "base": [300, 400], "frame": 1000},
            },
            "actions": {
                "0": [{"at": 1000, "pos": 50}, {"at": 2000, "pos": 100}],
            },
        }
        save_project(str(path), state)
        loaded = load_project(str(path))
        assert loaded == state

    def test_empty_project(self, tmp_path):
        path = tmp_path / "empty.scripture"
        state = {
            "video_path": "C:/videos/test.mp4",
            "splits": [],
            "axes": {},
            "actions": {},
        }
        save_project(str(path), state)
        loaded = load_project(str(path))
        assert loaded == state

    def test_tracking_data_round_trip(self, tmp_path):
        path = tmp_path / "tracking.scripture"
        state = {
            "video_path": "C:/videos/test.mp4",
            "splits": [],
            "axes": {},
            "actions": {},
            "tracking": {
                "1": {
                    "timestamps_ms": [0.0, 33.3, 66.6],
                    "positions": [0.5, 0.6, 0.4],
                    "tip_coords": [[100, 50], [101, 51], [102, 52]],
                    "base_coords": [[100, 350], [101, 351], [102, 352]],
                }
            },
        }
        save_project(str(path), state)
        loaded = load_project(str(path))
        assert loaded["tracking"]["1"]["tip_coords"] == [[100, 50], [101, 51], [102, 52]]
        assert loaded["tracking"]["1"]["positions"] == [0.5, 0.6, 0.4]


class TestDocumentSchema:

    def _project(self):
        return Project(
            video_path="C:/videos/example clip.mp4",
            splits=[400],
            axes={0: AxisDefinition(tip=(40, 30), base=(40, 150), frame=5)},
            actions={0: [{"at": 0, "pos": 50}]},
            tracking={0: TrackingResult(
                timestamps_ms=np.array([0.0, 33.0]),
                positions=np.array([0.5, 0.9]),
                tip_coords=np.array([[40.0, 30.0], [41.0, 31.0]]),
                base_coords=np.array([[40.0, 150.0], [40.0, 151.0]]))},
            labels={1: {800: {"tip": (40, 30), "base": (40, 150),
                              "contact": (40, 90), "is_action": True}}},
            current_frame=800,
        )

    def _through_json(self, project):
        return project_from_state(json.loads(json.dumps(state_from_project(project))))

    def test_the_scene_keyed_maps_come_back_on_integer_keys(self):
        restored = self._through_json(self._project())

        assert list(restored.axes) == [0]
        assert list(restored.tracking) == [0]
        assert list(restored.labels) == [1]
        assert list(restored.labels[1]) == [800]

    def test_the_axis_comes_back_as_a_pair_of_points_and_a_frame(self):
        restored = self._through_json(self._project())

        assert restored.axes[0] == AxisDefinition(tip=(40, 30), base=(40, 150), frame=5)

    def test_the_tracked_arrays_come_back_as_arrays(self):
        restored = self._through_json(self._project())

        np.testing.assert_allclose(restored.tracking[0].positions, [0.5, 0.9])
        np.testing.assert_allclose(restored.tracking[0].tip_coords, [[40, 30], [41, 31]])

    def test_what_tracked_a_scene_comes_back_with_its_tracking(self):
        project = self._project()
        project.tracking[0] = _tracked(A_STAMP)

        restored = self._through_json(project)

        assert restored.tracking[0].provenance == A_STAMP

    def test_a_scene_tracked_before_the_stamp_existed_opens_unstamped(self):
        restored = project_from_state({
            "video_path": "C:/videos/example clip.mp4", "splits": [],
            "tracking": {"0": {"timestamps_ms": [0.0], "positions": [0.5]}}})

        assert restored.tracking[0].provenance is None

    def test_a_label_comes_back_with_its_points_as_pairs(self):
        restored = self._through_json(self._project())

        assert restored.labels[1][800] == {
            "tip": (40, 30), "base": (40, 150), "contact": (40, 90), "is_action": True}

    def test_a_project_saved_before_these_keys_existed_still_opens(self):
        restored = project_from_state(
            {"video_path": "C:/videos/example clip.mp4", "splits": []})

        assert restored.axes == {}
        assert restored.labels == {}
        assert restored.auto is None
        assert restored.current_frame == 0

    def test_an_axis_without_its_frame_is_a_damaged_file_never_an_axis_on_the_video_s_first_frame(self):
        frameless = {"video_path": "C:/videos/example clip.mp4", "splits": [100],
                     "axes": {"0": {"tip": [10, 20], "base": [30, 40]},
                              "1": {"tip": [50, 60], "base": [70, 80]}}}

        with pytest.raises(LookupError):
            project_from_state(frameless)


class TestTheVideoAProjectWasMadeOn:

    @pytest.fixture
    def library(self, tmp_path, monkeypatch):
        settings = tmp_path / "settings.json"
        settings.write_text(json.dumps({"library_root": str(tmp_path / "library")}),
                            encoding="utf-8")
        monkeypatch.setenv(haglio_settings.SETTINGS_ENV, str(settings))
        return tmp_path / "library" / "videos" / "videos"

    def _video(self, folder):
        video = folder / "example clip.mp4"
        video.parent.mkdir(parents=True, exist_ok=True)
        video.write_bytes(b"")
        return video

    def test_a_project_saved_before_the_version_key_existed_is_the_first_shape_and_still_loads(self):
        state = state_from_project(Project(video_path="C:/videos/example clip.mp4", splits=[]))
        del state["version"]

        assert project_from_state(state).video_path == "C:/videos/example clip.mp4"

    def test_a_library_video_is_named_by_its_id_and_found_again_after_it_moves(self, library):
        video = self._video(library / "2D")
        state = state_from_project(Project(video_path=str(video), splits=[]))
        moved = library / "elsewhere" / "example clip.mp4"
        for was, now in ((video_ids.record_of(video), video_ids.record_of(moved)), (video, moved)):
            now.parent.mkdir(parents=True, exist_ok=True)
            was.replace(now)

        assert (state["version"], "video_path" in state) == (PROJECT_FORMAT_VERSION, False)
        assert project_from_state(state).video_path == str(moved)

    def test_a_video_outside_the_library_is_named_by_its_path(self, library, tmp_path):
        outside = str(tmp_path / "downloads" / "example clip.mp4")

        assert state_from_project(Project(video_path=outside, splits=[]))["video_path"] == outside

    def test_the_saved_projects_of_library_videos_are_moved_to_their_ids_once(
        self, library, tmp_path,
    ):
        video = self._video(library / "2D")
        projects = tmp_path / "sessions"
        projects.mkdir()
        (projects / "one.scripture").write_text(json.dumps(
            {"version": 1, "video_path": str(video), "splits": [3]}), encoding="utf-8")

        name_the_videos_by_id(projects)

        assert json.loads((projects / "one.scripture").read_text(encoding="utf-8")) == {
            "version": PROJECT_FORMAT_VERSION, "splits": [3],
            "video_id": video_ids.id_of(video_ids.record_of(video))}


class TestWhatMadeTheActions:

    def test_what_every_scene_s_tracking_agrees_on_is_kept_and_the_rest_is_unknown(self):
        later = {**A_STAMP, "stamped_at": "2026-01-02T00:00:00+00:00"}
        project = Project(
            actions={0: [{"at": 0, "pos": 50}], 1: [{"at": 900, "pos": 90}]},
            tracking={0: _tracked(A_STAMP), 1: _tracked(later)})

        assert provenance_of_actions(project) == {**A_STAMP, "stamped_at": None}

    def test_a_scene_nothing_vouches_for_leaves_the_script_unvouched(self):
        project = Project(
            actions={0: [{"at": 0, "pos": 50}], 1: [{"at": 900, "pos": 90}]},
            tracking={0: _tracked(A_STAMP)})

        assert provenance_of_actions(project) is None

    def test_scenes_the_auto_pass_filled_carry_its_stamp(self):
        auto_stamp = {**A_STAMP, METHOD_KEY_ON_DISK: "roi_flow"}
        project = Project(
            actions={0: [{"at": 0, "pos": 50}], 1: [{"at": 900, "pos": 90}]},
            auto=_auto_pass(auto_stamp))

        assert provenance_of_actions(project) == auto_stamp

    def test_a_tracked_scene_may_hold_the_auto_pass_s_actions_so_both_count(self):
        project = Project(
            actions={0: [{"at": 0, "pos": 50}]},
            tracking={0: _tracked(A_STAMP)},
            auto=_auto_pass({**A_STAMP, METHOD_KEY_ON_DISK: "roi_flow"}))

        assert provenance_of_actions(project) == {**A_STAMP, METHOD_KEY_ON_DISK: None}


def _pointing_at(project: Path, tmp_path: Path) -> Path:
    pointer = tmp_path / ".last_session"
    pointer.write_text(str(project), encoding="utf-8")
    return pointer


def test_the_last_project_is_reopened(tmp_path):
    project = tmp_path / "one.scripture"
    project.write_text("{}", encoding="utf-8")
    opened = []

    reopen_last_project(_pointing_at(project, tmp_path), opened.append)

    assert opened == [str(project)]


def test_no_last_project_opens_nothing(tmp_path):
    opened = []

    reopen_last_project(tmp_path / ".last_session", opened.append)
    reopen_last_project(_pointing_at(tmp_path / "gone.scripture", tmp_path), opened.append)

    assert opened == []


def test_a_damaged_last_project_is_logged_by_name_and_opens_nothing(tmp_path, caplog):
    damaged = tmp_path / "damaged.scripture"
    damaged.write_text("{ not a project", encoding="utf-8")

    with caplog.at_level(logging.WARNING, logger="scripture"):
        reopen_last_project(_pointing_at(damaged, tmp_path), load_project)

    assert str(damaged) in caplog.text


def test_a_bug_while_reopening_is_not_taken_for_a_damaged_file(tmp_path):
    project = tmp_path / "fine.scripture"
    project.write_text("{}", encoding="utf-8")

    def a_bug(_path):
        raise RuntimeError("an invented bug")

    with pytest.raises(RuntimeError):
        reopen_last_project(_pointing_at(project, tmp_path), a_bug)

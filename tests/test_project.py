"""The project file: the JSON wrapper, and the schema it wraps.

Only the wrapper was reachable from here before -- the schema lived inside the
Qt window, so nothing could round-trip it without standing a window up.
"""

import json

import numpy as np

from scripture.motion_tracker import AxisDefinition, TrackingResult
from scripture.project import (
    ProjectDocument,
    document_from_state,
    load_project,
    save_project,
    state_from_document,
)


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

    def _document(self):
        return ProjectDocument(
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

    def _through_json(self, document):
        return document_from_state(json.loads(json.dumps(state_from_document(document))))

    def test_the_scene_keyed_maps_come_back_on_integer_keys(self):
        restored = self._through_json(self._document())

        assert list(restored.axes) == [0]
        assert list(restored.tracking) == [0]
        assert list(restored.labels) == [1]
        assert list(restored.labels[1]) == [800]

    def test_the_axis_comes_back_as_a_pair_of_points_and_a_frame(self):
        restored = self._through_json(self._document())

        assert restored.axes[0] == AxisDefinition(tip=(40, 30), base=(40, 150), frame=5)

    def test_the_tracked_arrays_come_back_as_arrays(self):
        restored = self._through_json(self._document())

        np.testing.assert_allclose(restored.tracking[0].positions, [0.5, 0.9])
        np.testing.assert_allclose(restored.tracking[0].tip_coords, [[40, 30], [41, 31]])

    def test_a_label_comes_back_with_its_points_as_pairs(self):
        restored = self._through_json(self._document())

        assert restored.labels[1][800] == {
            "tip": (40, 30), "base": (40, 150), "contact": (40, 90), "is_action": True}

    def test_a_project_saved_before_these_keys_existed_still_opens(self):
        """Every key but the video and the splits is optional, which is what
        lets a file written by an older build load at all."""
        restored = document_from_state(
            {"video_path": "C:/videos/example clip.mp4", "splits": []})

        assert restored.axes == {} and restored.labels == {}
        assert restored.auto is None and restored.current_frame == 0

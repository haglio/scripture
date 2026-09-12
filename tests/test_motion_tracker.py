from __future__ import annotations

import numpy as np

from scripture import cotracker_tracking
from scripture.cotracker_tracking import CoTrackResult
from scripture.motion_tracker import RECIPE_VERSION, AxisDefinition, track_motion
from tests.videos import a_flat_video


def test_a_tracked_result_says_which_tracker_made_it_and_at_which_version(tmp_path, monkeypatch):
    video_path = a_flat_video(tmp_path / "example clip.mp4", frames=5)
    monkeypatch.setattr(cotracker_tracking, "cotrack_axis", lambda *_args: CoTrackResult(
        tip_coords=np.tile([80.0, 20.0], (5, 1)), base_coords=np.tile([80.0, 100.0], (5, 1))))

    result = track_motion(video_path, AxisDefinition(tip=(80, 20), base=(80, 100)), 0, 5)

    made = result.provenance
    assert (made["app"], made["recipe"]) == ("scripture", "axis_tracking")
    assert made["recipe_version"] == RECIPE_VERSION

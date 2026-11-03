from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.ndimage import median_filter

from scripture import cotracker_tracking
from scripture.cotracker_tracking import (
    TrackingChunk,
    _get_video_geometry,
    chunks_outward,
    compute_pos_from_points,
    ends_from_visible_points,
    find_contact_gradient,
    points_along_axis,
    sample_axis_intensity,
    sanitize_positions,
    scale_coords,
)
from scripture.motion_tracker import AxisDefinition
from tests.videos import a_flat_video


class TestSanitizePositions:

    def test_smooth_signal_unchanged(self):
        t = np.linspace(0, 4 * np.pi, 200)
        positions = (np.sin(t) + 1) / 2
        result = sanitize_positions(positions, fps=30)
        assert np.corrcoef(positions, result)[0, 1] > 0.9

    def test_removes_single_frame_spikes(self):
        positions = np.full(100, 0.5)
        positions[50] = 0.0

        result = sanitize_positions(positions, fps=30)

        assert result[50] == pytest.approx(0.5, abs=0.01)

    def test_enforces_max_speed(self):
        positions = np.zeros(100)
        positions[50:] = 1.0
        result = sanitize_positions(positions, fps=30)
        assert result[51] < 0.8

    def test_output_range(self):
        rng = np.random.RandomState(42)
        positions = rng.rand(200)
        result = sanitize_positions(positions, fps=30)
        assert result.min() >= 0.0
        assert result.max() <= 1.0


class TestWhatSanitizingDoes:

    def test_smoothing_follows_the_median_filter(self):
        noisy = 0.5 + np.random.default_rng(11).normal(0, 0.02, 300)

        assert sanitize_positions(noisy, fps=30).std() < median_filter(noisy, size=5, mode="nearest").std()

    def test_the_smoothing_s_overshoot_past_a_step_is_clamped(self):
        step = np.r_[np.zeros(20), np.ones(20)]

        result = sanitize_positions(step, fps=1)

        assert (result.min(), result.max()) == (0.0, 1.0)

    def test_a_signal_too_short_to_filter_comes_back_as_it_was(self):
        short = np.array([0.1, 0.9, 0.2])

        np.testing.assert_array_equal(sanitize_positions(short), short)


class TestSamplingAlongTheAxis:

    @staticmethod
    def _along_row(gray, row, **kw):
        base = np.array([0, row], dtype=np.float64)
        axis_vec = np.array([gray.shape[1] - 1, 0], dtype=np.float64)
        return sample_axis_intensity(gray, base, axis_vec, **kw)[1]

    def test_it_averages_a_strip_across_the_axis(self):
        gray = np.zeros((100, 200), dtype=np.uint8)
        gray[50, :] = 220

        assert self._along_row(gray, 50, strip_w=5) == pytest.approx(20.0)

    def test_what_falls_outside_the_frame_is_left_out_of_the_average(self):
        gray = np.full((100, 200), 100, dtype=np.uint8)
        gray[-5:, :] = 0

        assert self._along_row(gray, 0, strip_w=5) == pytest.approx(100.0)

    def test_the_result_is_smoothed_along_the_axis(self):
        gray = np.full((100, 200), 200, dtype=np.uint8)
        gray[:, 100] = 0

        assert self._along_row(gray, 50, n=200, strip_w=0).min() == pytest.approx(160.0, abs=1.0)


def test_coordinates_scale_by_each_side_s_own_ratio():
    scaled = scale_coords(np.array([[10.0, 10.0]]), from_size=(100, 200), to_size=(200, 200))

    assert scaled.tolist() == [[10.0, 20.0]]


def test_a_video_is_tracked_at_a_size_whose_long_side_is_384(tmp_path):
    video = a_flat_video(tmp_path / "example clip.mp4", frames=2, size=(160, 120))

    assert _get_video_geometry(video, 0) == ((120, 160), (288, 384))


class TestIntensityGradientContact:

    def test_the_search_starts_part_way_along_the_axis_by_default(self):
        t_values = np.linspace(0, 1, 11)
        intensities = np.array([9.0, 9, 0, 0, 0, 0, 0, 1, 1, 1, 1])

        assert find_contact_gradient(t_values, intensities) == pytest.approx(0.6)

    def test_sharp_edge_detected(self):
        gray = np.full((100, 200), 200, dtype=np.uint8)
        gray[:, 120:] = 50
        base = np.array([0, 50], dtype=np.float64)
        tip = np.array([199, 50], dtype=np.float64)
        axis_vec = tip - base
        t_vals, intensities = sample_axis_intensity(gray, base, axis_vec, n=100, strip_w=5)
        pred_t = find_contact_gradient(t_vals, intensities, search_min=0.0)
        assert abs(pred_t - 0.60) < 0.05

    def test_no_edge_falls_back_to_the_middle_of_the_search_range(self):
        gray = np.full((100, 200), 128, dtype=np.uint8)
        base = np.array([0, 50], dtype=np.float64)
        tip = np.array([199, 50], dtype=np.float64)
        axis_vec = tip - base
        t_vals, intensities = sample_axis_intensity(gray, base, axis_vec, n=100, strip_w=5)

        pred_t = find_contact_gradient(t_vals, intensities, search_min=0.0)

        assert pred_t == pytest.approx(0.5)


class TestComputePosFromPoints:

    def test_an_axis_with_no_length_reads_as_the_middle(self):
        assert compute_pos_from_points((5, 5), (5, 5), (9, 9)) == 50

    def test_contact_at_base(self):
        assert compute_pos_from_points((100, 800), (100, 100), (100, 800)) == 0

    def test_contact_at_tip(self):
        assert compute_pos_from_points((100, 800), (100, 100), (100, 100)) == 100

    def test_contact_at_midpoint(self):
        assert compute_pos_from_points((100, 800), (100, 100), (100, 450)) == 50

    def test_diagonal_axis(self):
        pos = compute_pos_from_points((0, 0), (100, 100), (50, 50))
        assert 45 <= pos <= 55

    def test_clamps_below_0(self):
        pos = compute_pos_from_points((100, 100), (100, 500), (100, 50))
        assert pos == 0

    def test_clamps_above_100(self):
        pos = compute_pos_from_points((100, 100), (100, 500), (100, 550))
        assert pos == 100


class TestScaleCoords:

    def test_round_trip(self):
        orig = np.array([[960, 540], [100, 200]], dtype=np.float64)
        orig_size = (1080, 1920)
        scaled_size = (216, 384)
        scaled = scale_coords(orig, orig_size, scaled_size)
        restored = scale_coords(scaled, scaled_size, orig_size)
        np.testing.assert_allclose(restored, orig, atol=1)

    def test_scale_factor(self):
        orig = np.array([[1920, 1080]], dtype=np.float64)
        scaled = scale_coords(orig, (1080, 1920), (216, 384))
        np.testing.assert_allclose(scaled, [[384, 216]], atol=1)


class TestTheTrackingModel:

    def test_is_loaded_once_and_kept_for_every_later_track(self, monkeypatch):
        loads = []

        class _Model:
            def to(self, _device):
                return self

        def load(repo, name):
            loads.append((repo, name))
            return _Model()

        monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(hub=SimpleNamespace(load=load)))
        cotracker_tracking._get_model.cache_clear()
        try:
            first, second = cotracker_tracking._get_model(), cotracker_tracking._get_model()
        finally:
            cotracker_tracking._get_model.cache_clear()

        assert first is second
        assert loads == [("facebookresearch/co-tracker", "cotracker3_offline")]


class TestTheChunksOfALongScene:

    def test_a_scene_that_fits_in_one_chunk_runs_once_from_the_drawn_axis(self):
        assert chunks_outward(300, reference=120) == [TrackingChunk(0, 300, 120, seed=None)]

    def test_the_chunk_holding_the_drawn_axis_runs_first_then_forward_then_backward(self):
        assert chunks_outward(700, reference=350) == [
            TrackingChunk(200, 500, 150, seed=None),
            TrackingChunk(425, 700, 0, seed=425),
            TrackingChunk(0, 275, 274, seed=274),
        ]

    def test_each_later_chunk_overlaps_the_one_before_and_starts_from_what_it_tracked(self):
        assert chunks_outward(1000, reference=0) == [
            TrackingChunk(0, 300, 0, seed=None),
            TrackingChunk(225, 525, 0, seed=225),
            TrackingChunk(450, 750, 0, seed=450),
            TrackingChunk(675, 975, 0, seed=675),
            TrackingChunk(900, 1000, 0, seed=900),
            TrackingChunk(0, 75, 74, seed=74),
        ]


class TestTheEndsOfTheAxisFromWhatWasVisible:

    @staticmethod
    def _ends(tracks, visibility):
        drawn = np.array([[0.0, 0.0], [0.0, 100.0]])
        return ends_from_visible_points(
            np.array(tracks, dtype=np.float64), np.array(visibility, dtype=np.float64), drawn)

    def test_the_lowest_visible_point_is_the_base(self):
        base, _tip = self._ends([[[9.0, 9.0], [0.0, 10.0], [0.0, 50.0]]], [[0.2, 0.9, 0.9]])

        assert base.tolist() == [[0.0, 10.0]]

    def test_the_highest_visible_point_is_the_contact_and_the_tip_lies_the_drawn_length_past_the_base(self):
        _base, tip = self._ends([[[0.0, 10.0], [0.0, 50.0], [9.0, 9.0]]], [[0.9, 0.9, 0.2]])

        assert tip.tolist() == [[0.0, 110.0]]

    def test_the_tip_points_the_way_the_contact_lies_from_the_base(self):
        _base, tip = self._ends([[[10.0, 0.0], [10.0, -30.0]]], [[0.9, 0.9]])

        assert tip.tolist() == [[10.0, -100.0]]

    def test_a_frame_with_nothing_visible_keeps_the_ends_of_the_frame_before(self):
        base, tip = self._ends([[[0.0, 10.0], [0.0, 50.0]], [[7.0, 7.0], [8.0, 8.0]]],
                               [[0.9, 0.9], [0.1, 0.1]])

        assert base.tolist() == [[0.0, 10.0], [0.0, 10.0]]
        assert tip.tolist() == [[0.0, 110.0], [0.0, 110.0]]

    def test_before_anything_is_visible_the_ends_are_where_they_were_drawn(self):
        base, tip = self._ends([[[7.0, 7.0], [8.0, 8.0]]], [[0.1, 0.1]])

        assert base.tolist() == [[0.0, 0.0]]
        assert tip.tolist() == [[0.0, 100.0]]

    def test_a_contact_on_top_of_the_base_gives_no_direction_so_the_tip_stays_on_the_contact(self):
        _base, tip = self._ends([[[3.0, 3.0], [3.0, 3.5]]], [[0.9, 0.9]])

        assert tip.tolist() == [[3.0, 3.5]]

    def test_a_point_is_visible_only_above_one_half(self):
        base, _tip = self._ends([[[9.0, 9.0], [0.0, 10.0], [0.0, 50.0]]], [[0.5, 0.51, 0.9]])

        assert base.tolist() == [[0.0, 10.0]]


def test_the_points_handed_to_the_tracker_run_evenly_from_base_to_tip():
    points = points_along_axis(AxisDefinition(tip=(0, 29), base=(0, 0)))

    assert points[0].tolist() == [0.0, 0.0]
    assert points[-1].tolist() == [0.0, 29.0]
    assert points[1].tolist() == [0.0, 1.0]

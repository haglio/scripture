import numpy as np

from scripture.cotracker_tracking import (
    fit_axis_from_points, scale_coords, visibility_to_position,
)


class TestFitAxisFromPoints:

    def test_all_visible_returns_exact_endpoints(self):
        """With all points visible and collinear, should recover exact tip/base."""
        # 8 points along y=100..800 at x=500 (vertical axis)
        t_params = np.linspace(0, 1, 8)  # 0=base, 1=tip
        points = np.array([[500, 800 - 700 * t] for t in t_params], dtype=np.float64)
        visible = np.ones(8, dtype=bool)
        tip, base = fit_axis_from_points(points, t_params, visible)
        np.testing.assert_allclose(base, [500, 800], atol=1)
        np.testing.assert_allclose(tip, [500, 100], atol=1)

    def test_tip_occluded_extrapolates(self):
        """With tip-end points hidden, should extrapolate from visible ones."""
        t_params = np.linspace(0, 1, 8)
        points = np.array([[500, 800 - 700 * t] for t in t_params], dtype=np.float64)
        visible = np.array([True, True, True, True, True, True, False, False])
        tip, base = fit_axis_from_points(points, t_params, visible)
        np.testing.assert_allclose(base, [500, 800], atol=5)
        np.testing.assert_allclose(tip, [500, 100], atol=5)

    def test_base_offscreen_extrapolates(self):
        """With base-end points hidden, should extrapolate from visible ones."""
        t_params = np.linspace(0, 1, 8)
        points = np.array([[500, 800 - 700 * t] for t in t_params], dtype=np.float64)
        visible = np.array([False, False, False, True, True, True, True, True])
        tip, base = fit_axis_from_points(points, t_params, visible)
        np.testing.assert_allclose(base, [500, 800], atol=5)
        np.testing.assert_allclose(tip, [500, 100], atol=5)

    def test_diagonal_axis(self):
        t_params = np.linspace(0, 1, 8)
        points = np.array([[100 + 300 * t, 400 - 200 * t] for t in t_params])
        visible = np.ones(8, dtype=bool)
        tip, base = fit_axis_from_points(points, t_params, visible)
        np.testing.assert_allclose(base, [100, 400], atol=1)
        np.testing.assert_allclose(tip, [400, 200], atol=1)

    def test_fewer_than_2_visible_returns_none(self):
        t_params = np.linspace(0, 1, 8)
        points = np.zeros((8, 2))
        visible = np.array([False, False, False, False, False, False, False, True])
        result = fit_axis_from_points(points, t_params, visible)
        assert result is None


class TestVisibilityToPosition:

    def test_all_visible_returns_100(self):
        """All redacted points visible = nothing covering it = pos 100 (tip)."""
        t_params = np.linspace(0, 1, 30)
        vis = np.ones(30)
        assert visibility_to_position(t_params, vis) == 100

    def test_all_occluded_returns_0(self):
        """All redacted points occluded = fully covered = pos 0 (base)."""
        t_params = np.linspace(0, 1, 30)
        vis = np.zeros(30)
        assert visibility_to_position(t_params, vis) == 0

    def test_hand_at_midpoint(self):
        """Base half visible, tip half occluded → contact near 50."""
        t_params = np.linspace(0, 1, 30)
        vis = np.zeros(30)
        vis[:15] = 1.0  # base side visible
        pos = visibility_to_position(t_params, vis)
        assert 40 <= pos <= 60

    def test_hand_covers_base_end(self):
        """Hand covers base end, tip exposed → contact at hand's leading edge."""
        t_params = np.linspace(0, 1, 30)
        vis = np.zeros(30)
        vis[25:] = 1.0  # only tip-end visible, hand covers base through ~t=0.83
        pos = visibility_to_position(t_params, vis)
        # Hand's leading edge is near t=0.83 → pos ≈ 83
        assert 75 <= pos <= 90

    def test_hand_covers_tip_end(self):
        """Hand covers tip end, base exposed → contact at hand's leading edge."""
        t_params = np.linspace(0, 1, 30)
        vis = np.zeros(30)
        vis[:5] = 1.0  # only base-end visible, hand covers from ~t=0.17 up
        pos = visibility_to_position(t_params, vis)
        # Hand's leading edge is near t=0.14 → pos ≈ 14
        assert 10 <= pos <= 25


class TestScaleCoords:

    def test_round_trip(self):
        """Scaling down then up should preserve coordinates."""
        orig = np.array([[960, 540], [100, 200]], dtype=np.float64)
        orig_size = (1080, 1920)  # (h, w)
        scaled_size = (216, 384)
        scaled = scale_coords(orig, orig_size, scaled_size)
        restored = scale_coords(scaled, scaled_size, orig_size)
        np.testing.assert_allclose(restored, orig, atol=1)

    def test_scale_factor(self):
        orig = np.array([[1920, 1080]], dtype=np.float64)
        scaled = scale_coords(orig, (1080, 1920), (216, 384))
        np.testing.assert_allclose(scaled, [[384, 216]], atol=1)

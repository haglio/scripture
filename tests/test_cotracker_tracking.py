import numpy as np

from scripture.cotracker_tracking import (
    fit_axis_from_points, scale_coords, visibility_to_position,
    motion_divergence_position, sanitize_positions,
    compute_pos_from_points, sample_axis_intensity,
    find_contact_gradient,
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


class TestMotionDivergencePosition:

    def _make_tracks(self, n_frames, n_points, contact_t_per_frame):
        """Build synthetic tracks where points below contact_t are stationary
        and points above contact_t move with the hand (offset by 5px/frame).
        """
        t_params = np.linspace(0, 1, n_points)
        # All points start at their t-position * 100 along y, x=50
        tracks = np.zeros((n_frames, n_points, 2), dtype=np.float64)
        for f in range(n_frames):
            ct = contact_t_per_frame[f]
            for j, t in enumerate(t_params):
                tracks[f, j, 0] = 50  # x stays constant
                tracks[f, j, 1] = t * 100  # base y position
                if t > ct:
                    # Points above contact move with hand
                    tracks[f, j, 1] += f * 5
        return tracks, t_params

    def test_stationary_scene_returns_100(self):
        """No motion anywhere → nothing covering redacted → pos=100."""
        n_points = 30
        t_params = np.linspace(0, 1, n_points)
        tracks_prev = np.array([[50, t * 100] for t in t_params], dtype=np.float64)
        tracks_curr = tracks_prev.copy()  # identical
        pos = motion_divergence_position(t_params, tracks_prev, tracks_curr)
        assert pos >= 90

    def test_hand_at_midpoint(self):
        """Points 0-14 stationary, 15-29 moving → contact near 50."""
        n_points = 30
        t_params = np.linspace(0, 1, n_points)
        tracks_prev = np.array([[50, t * 100] for t in t_params], dtype=np.float64)
        tracks_curr = tracks_prev.copy()
        # Move tip-side points
        tracks_curr[15:, 1] += 10
        pos = motion_divergence_position(t_params, tracks_prev, tracks_curr)
        assert 35 <= pos <= 65

    def test_hand_near_tip(self):
        """Only last few points moving → contact near tip → high pos."""
        n_points = 30
        t_params = np.linspace(0, 1, n_points)
        tracks_prev = np.array([[50, t * 100] for t in t_params], dtype=np.float64)
        tracks_curr = tracks_prev.copy()
        tracks_curr[25:, 1] += 10  # only tip-end moving
        pos = motion_divergence_position(t_params, tracks_prev, tracks_curr)
        assert pos >= 70

    def test_hand_near_base(self):
        """Most points moving, only base few stationary → contact near base."""
        n_points = 30
        t_params = np.linspace(0, 1, n_points)
        tracks_prev = np.array([[50, t * 100] for t in t_params], dtype=np.float64)
        tracks_curr = tracks_prev.copy()
        tracks_curr[3:, 1] += 10  # almost everything moving
        pos = motion_divergence_position(t_params, tracks_prev, tracks_curr)
        assert pos <= 25


class TestSanitizePositions:

    def test_smooth_signal_unchanged(self):
        """A clean sine wave should survive sanitization mostly intact."""
        t = np.linspace(0, 4 * np.pi, 200)
        positions = (np.sin(t) + 1) / 2
        result = sanitize_positions(positions, fps=30)
        # Should correlate strongly with original
        assert np.corrcoef(positions, result)[0, 1] > 0.9

    def test_removes_single_frame_spikes(self):
        """A spike from 50 to 0 back to 50 in one frame should be smoothed."""
        positions = np.full(100, 0.5)
        positions[50] = 0.0  # single-frame spike
        result = sanitize_positions(positions, fps=30)
        # The spike should be gone or greatly reduced
        assert result[50] > 0.3

    def test_enforces_max_speed(self):
        """A jump from 0 to 100 in one frame should be limited."""
        positions = np.zeros(100)
        positions[50:] = 1.0  # instant jump
        result = sanitize_positions(positions, fps=30)
        # The transition should be spread over multiple frames
        assert result[51] < 0.8  # can't reach 1.0 in one frame

    def test_output_range(self):
        """Output should stay in [0, 1]."""
        rng = np.random.RandomState(42)
        positions = rng.rand(200)
        result = sanitize_positions(positions, fps=30)
        assert result.min() >= 0.0
        assert result.max() <= 1.0


class TestIntensityGradientContact:

    def test_sharp_edge_detected(self):
        """A bright-to-dark transition at t=0.6 should be found."""
        # Simulate: bright redacted (200) with dark mouth region (50) starting at 60%
        gray = np.full((100, 200), 200, dtype=np.uint8)
        gray[:, 120:] = 50  # dark from x=120 onward
        base = np.array([0, 50], dtype=np.float64)
        tip = np.array([199, 50], dtype=np.float64)
        axis_vec = tip - base
        perp = np.array([0, 1], dtype=np.float64)
        t_vals, intensities = sample_axis_intensity(gray, base, axis_vec, perp, n=100, strip_w=5)
        pred_t = find_contact_gradient(t_vals, intensities, search_min=0.0)
        assert abs(pred_t - 0.60) < 0.05

    def test_no_edge_returns_within_range(self):
        """Uniform image should still return something in [0, 1]."""
        gray = np.full((100, 200), 128, dtype=np.uint8)
        base = np.array([0, 50], dtype=np.float64)
        tip = np.array([199, 50], dtype=np.float64)
        axis_vec = tip - base
        perp = np.array([0, 1], dtype=np.float64)
        t_vals, intensities = sample_axis_intensity(gray, base, axis_vec, perp, n=100, strip_w=5)
        pred_t = find_contact_gradient(t_vals, intensities, search_min=0.0)
        assert 0 <= pred_t <= 1


class TestComputePosFromPoints:

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

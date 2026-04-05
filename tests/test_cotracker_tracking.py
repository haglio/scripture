import numpy as np

from scripture.cotracker_tracking import (
    scale_coords, sanitize_positions,
    compute_pos_from_points, sample_axis_intensity,
    find_contact_gradient,
)


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

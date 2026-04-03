import tempfile

import cv2
import numpy as np

from scripture.motion_tracker import (
    AxisDefinition, build_axis_strip_mask, compute_crop_bounds,
    subtract_camera_motion, rolling_normalize, track_motion, TrackingResult,
    LKPointTracker,
)


class TestBuildAxisStripMask:

    def test_shape_matches_frame(self):
        axis = AxisDefinition(tip=(100, 100), base=(100, 500))
        mask = build_axis_strip_mask((600, 400), axis, half_width=15)
        assert mask.shape == (600, 400)
        assert mask.dtype == np.uint8

    def test_pixel_count_proportional_to_axis_length(self):
        axis = AxisDefinition(tip=(100, 100), base=(100, 500))
        mask = build_axis_strip_mask((600, 400), axis, half_width=15)
        # Vertical axis length = 400, width = 30 => ~12000 pixels
        count = np.count_nonzero(mask)
        assert 10000 < count < 15000

    def test_much_smaller_than_rect_roi(self):
        """Strip mask should use far fewer pixels than the old rectangle ROI."""
        from scripture.motion_tracker import build_roi_mask
        axis = AxisDefinition(tip=(929, 362), base=(807, 1047))
        frame_shape = (1080, 1920)
        strip = build_axis_strip_mask(frame_shape, axis, half_width=15)
        rect = build_roi_mask(frame_shape, axis, margin=80)
        assert np.count_nonzero(strip) < np.count_nonzero(rect) * 0.20

    def test_diagonal_axis(self):
        axis = AxisDefinition(tip=(50, 50), base=(350, 450))
        mask = build_axis_strip_mask((500, 400), axis, half_width=15)
        count = np.count_nonzero(mask)
        length = np.linalg.norm([300, 400])  # 500
        expected = length * 30  # ~15000
        assert count > expected * 0.7
        assert count < expected * 1.3

    def test_axis_near_frame_edge(self):
        axis = AxisDefinition(tip=(5, 5), base=(5, 100))
        mask = build_axis_strip_mask((120, 30), axis, half_width=15)
        assert mask.shape == (120, 30)
        # Should not crash; pixels outside frame are clipped
        assert np.count_nonzero(mask) > 0

    def test_short_axis(self):
        axis = AxisDefinition(tip=(100, 100), base=(110, 110))
        mask = build_axis_strip_mask((200, 200), axis, half_width=15)
        assert np.count_nonzero(mask) > 0


class TestComputeCropBounds:

    def test_basic_vertical_axis(self):
        axis = AxisDefinition(tip=(100, 100), base=(100, 500))
        y_min, y_max, x_min, x_max = compute_crop_bounds(axis, half_width=15, frame_shape=(600, 400), padding=30)
        # expand = half_width + padding = 45 in all dimensions
        assert x_min == 55   # 100 - 45
        assert x_max == 145  # 100 + 45
        assert y_min == 55   # 100 - 45
        assert y_max == 545  # 500 + 45

    def test_clamps_to_frame(self):
        axis = AxisDefinition(tip=(5, 5), base=(5, 100))
        y_min, y_max, x_min, x_max = compute_crop_bounds(axis, half_width=15, frame_shape=(120, 30), padding=30)
        assert x_min >= 0
        assert y_min >= 0
        assert x_max <= 30
        assert y_max <= 120

    def test_diagonal_contains_endpoints(self):
        axis = AxisDefinition(tip=(50, 50), base=(350, 450))
        y_min, y_max, x_min, x_max = compute_crop_bounds(axis, half_width=15, frame_shape=(500, 400), padding=30)
        # Crop must contain both tip and base
        assert x_min <= 50
        assert x_max >= 350
        assert y_min <= 50
        assert y_max >= 450


class TestSubtractCameraMotion:

    def test_zero_background(self):
        assert subtract_camera_motion(5.0, 0.0) == 5.0

    def test_nonzero_background(self):
        assert subtract_camera_motion(5.0, 3.0) == 2.0

    def test_negative_motion(self):
        assert subtract_camera_motion(-2.0, 1.0) == -3.0

    def test_both_negative(self):
        assert subtract_camera_motion(-2.0, -3.0) == 1.0


class TestRollingNormalize:

    def test_sine_wave_spans_full_range(self):
        signal = np.sin(np.linspace(0, 10 * np.pi, 1000))
        result = rolling_normalize(signal, window_frames=100)
        assert result.min() >= 0.0
        assert result.max() <= 1.0
        assert result.max() > 0.9
        assert result.min() < 0.1

    def test_drifting_sine_still_spans_range(self):
        t = np.linspace(0, 10 * np.pi, 1000)
        signal = np.sin(t) + np.linspace(0, 5, 1000)
        result = rolling_normalize(signal, window_frames=100)
        assert result.max() > 0.9
        assert result.min() < 0.1

    def test_constant_signal(self):
        signal = np.full(100, 5.0)
        result = rolling_normalize(signal, window_frames=30)
        # All values should be the same (no variation to normalize)
        assert np.allclose(result, result[0])

    def test_short_signal(self):
        signal = np.array([0.0, 1.0, 0.0])
        result = rolling_normalize(signal, window_frames=10)
        assert result.shape == signal.shape
        assert result.max() <= 1.0
        assert result.min() >= 0.0


class TestLKPointTracker:

    def test_stationary_scene_no_drift(self):
        """On identical frames, tracked point should not move."""
        rng = np.random.RandomState(42)
        frame = rng.randint(0, 255, (200, 200), dtype=np.uint8)
        tracker = LKPointTracker(frame, center=(100, 100), radius=30)
        pos = tracker.update(frame)
        assert abs(pos[0] - 100) <= 2
        assert abs(pos[1] - 100) <= 2

    def test_tracks_known_shift(self):
        """A textured patch shifted by a known amount should be tracked."""
        rng = np.random.RandomState(42)
        texture = rng.randint(0, 255, (200, 200), dtype=np.uint8)
        # Frame 1: texture as-is
        frame1 = texture.copy()
        # Frame 2: shift texture 5px right, 3px down (via translation matrix)
        M = np.float32([[1, 0, 5], [0, 1, 3]])
        frame2 = cv2.warpAffine(texture, M, (200, 200))
        tracker = LKPointTracker(frame1, center=(100, 100), radius=30)
        pos = tracker.update(frame2)
        assert abs(pos[0] - 105) <= 2
        assert abs(pos[1] - 103) <= 2

    def test_survives_multiple_steps(self):
        """Track through 10 small shifts without crashing or losing track."""
        rng = np.random.RandomState(42)
        texture = rng.randint(0, 255, (300, 300), dtype=np.uint8)
        frame = texture.copy()
        tracker = LKPointTracker(frame, center=(150, 150), radius=30)
        cumulative_dx, cumulative_dy = 0.0, 0.0
        for _ in range(10):
            dx, dy = 2, 1
            cumulative_dx += dx
            cumulative_dy += dy
            M = np.float32([[1, 0, cumulative_dx], [0, 1, cumulative_dy]])
            frame = cv2.warpAffine(texture, M, (300, 300))
            pos = tracker.update(frame)
        expected_x = 150 + cumulative_dx
        expected_y = 150 + cumulative_dy
        assert abs(pos[0] - expected_x) <= 5
        assert abs(pos[1] - expected_y) <= 5


class TestTrackingResultCompat:

    def test_default_none_fields(self):
        result = TrackingResult(timestamps_ms=np.array([0.0]), positions=np.array([0.5]))
        assert result.tip_coords is None
        assert result.base_coords is None

    def test_with_coords(self):
        tips = np.array([[100, 50], [101, 51]], dtype=np.float64)
        bases = np.array([[100, 350], [101, 351]], dtype=np.float64)
        result = TrackingResult(
            timestamps_ms=np.array([0.0, 33.3]),
            positions=np.array([0.5, 0.6]),
            tip_coords=tips,
            base_coords=bases,
        )
        assert result.tip_coords.shape == (2, 2)
        assert result.base_coords.shape == (2, 2)


def _make_synthetic_video(path, n_frames=60, fps=30, width=200, height=400):
    """Create a synthetic video with a textured bar oscillating vertically.

    The bar moves from y=100 to y=350 in a sinusoidal pattern, simulating
    a stroking motion along a vertical axis.  Both bar and background have
    noise texture so optical flow can track features.
    """
    rng = np.random.RandomState(42)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (width, height))
    bar_h = 40
    # Static background texture (dark gray with noise)
    bg_texture = rng.randint(20, 60, (height, width, 3), dtype=np.uint8)
    # Static bar texture (bright with noise)
    bar_texture = rng.randint(180, 255, (bar_h, 40, 3), dtype=np.uint8)
    for i in range(n_frames):
        frame = bg_texture.copy()
        t = i / n_frames
        y_center = int(200 + 100 * np.sin(2 * np.pi * 2 * t))
        y_top = max(0, y_center - bar_h // 2)
        y_bot = min(height, y_top + bar_h)
        h_slice = y_bot - y_top
        frame[y_top:y_bot, 80:120] = bar_texture[:h_slice]
        writer.write(frame)
    writer.release()


class TestTrackMotion:

    def test_returns_correct_types(self, tmp_path):
        video_path = str(tmp_path / "test.mp4")
        _make_synthetic_video(video_path, n_frames=30)
        axis = AxisDefinition(tip=(100, 50), base=(100, 350))
        result = track_motion(video_path, axis, 0, 30)
        assert isinstance(result, TrackingResult)
        assert isinstance(result.positions, np.ndarray)
        assert isinstance(result.timestamps_ms, np.ndarray)
        assert len(result.positions) == len(result.timestamps_ms)

    def test_positions_in_valid_range(self, tmp_path):
        video_path = str(tmp_path / "test.mp4")
        _make_synthetic_video(video_path, n_frames=60)
        axis = AxisDefinition(tip=(100, 50), base=(100, 350))
        result = track_motion(video_path, axis, 0, 60)
        assert result.positions.min() >= 0.0
        assert result.positions.max() <= 1.0

    def test_oscillating_bar_produces_oscillating_signal(self, tmp_path):
        video_path = str(tmp_path / "test.mp4")
        _make_synthetic_video(video_path, n_frames=60, fps=30)
        axis = AxisDefinition(tip=(100, 50), base=(100, 350))
        result = track_motion(video_path, axis, 0, 60)
        # The signal should have variation (not flat)
        assert np.std(result.positions) > 0.1

    def test_on_frame_callback_called(self, tmp_path):
        video_path = str(tmp_path / "test.mp4")
        _make_synthetic_video(video_path, n_frames=20)
        axis = AxisDefinition(tip=(100, 50), base=(100, 350))
        frames_seen = []
        result = track_motion(video_path, axis, 0, 20, on_frame=lambda f: frames_seen.append(f))
        # Called during both Phase 1 (tracking) and Phase 2 (flow)
        assert len(frames_seen) >= 19

    def test_returns_per_frame_coords(self, tmp_path):
        video_path = str(tmp_path / "test.mp4")
        _make_synthetic_video(video_path, n_frames=30)
        axis = AxisDefinition(tip=(100, 50), base=(100, 350))
        result = track_motion(video_path, axis, 0, 30)
        assert result.tip_coords is not None
        assert result.base_coords is not None
        assert result.tip_coords.shape == (30, 2)
        assert result.base_coords.shape == (30, 2)

    def test_tip_base_vector_preserved(self, tmp_path):
        """Per-frame tip - base should equal original axis vector."""
        video_path = str(tmp_path / "test.mp4")
        _make_synthetic_video(video_path, n_frames=30)
        axis = AxisDefinition(tip=(100, 50), base=(100, 350))
        result = track_motion(video_path, axis, 0, 30)
        expected_vec = np.array(axis.tip) - np.array(axis.base)  # (0, -300)
        actual_vecs = result.tip_coords - result.base_coords
        for i in range(len(actual_vecs)):
            np.testing.assert_array_equal(actual_vecs[i], expected_vec)

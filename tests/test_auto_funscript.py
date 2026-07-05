"""Tests for the automatic YOLO+flow funscript pipeline."""

import numpy as np
import pytest

from scripture.auto_funscript import (
    Detection,
    TrackConfig,
    TrackSignal,
    anti_plateau_normalize,
    combine_roi,
    find_interacting,
    flow_to_position,
    is_scene_cut,
    parse_args,
    signal_to_actions,
    smooth_roi,
    track_flow_signal,
    weighted_flow,
)


def det(cls, x, y, w, h, conf=0.9):
    return Detection(class_name=cls, confidence=conf, box=(x, y, w, h))


class TestFindInteracting:
    def test_nearby_hand_interacts(self):
        # Anchor at (100,100) 50x100, hand overlapping its center region
        anchor = det("anchor", 100, 100, 50, 100)
        hand = det("hand", 110, 120, 40, 40)
        assert find_interacting(anchor, [anchor, hand]) == [hand]

    def test_distant_object_does_not_interact(self):
        anchor = det("anchor", 100, 100, 50, 100)
        far_face = det("face", 500, 400, 60, 60)
        assert find_interacting(anchor, [anchor, far_face]) == []


class TestCombineRoi:
    def test_union_with_padding(self):
        boxes = [(200, 200, 100, 200), (180, 150, 80, 80)]
        # Union: x 180-300, y 150-400. Padding 20 -> 160-320, 130-420.
        assert combine_roi(boxes, frame_size=(544, 960), padding=20) == (160, 130, 160, 290)

    def test_clamped_to_frame(self):
        boxes = [(0, 0, 200, 200)]
        x, y, w, h = combine_roi(boxes, frame_size=(544, 960), padding=30)
        assert (x, y) == (0, 0)
        assert (w, h) == (230, 230)

    def test_minimum_size_enforced(self):
        boxes = [(400, 300, 20, 20)]
        x, y, w, h = combine_roi(boxes, frame_size=(544, 960), padding=0, min_size=128)
        assert w == 128 and h == 128
        # Still centered on the small box and inside the frame
        assert 0 <= x <= 960 - 128 and 0 <= y <= 544 - 128


class TestSmoothRoi:
    def test_first_roi_passes_through(self):
        assert smooth_roi(None, (10, 20, 100, 200), factor=0.6) == (10, 20, 100, 200)

    def test_blend_weights_previous_by_factor(self):
        # factor 0.6 keeps 60% of previous, takes 40% of new
        assert smooth_roi((0, 0, 100, 100), (100, 100, 200, 200), factor=0.6) == (
            40, 40, 140, 140)


class TestWeightedFlow:
    def test_small_moving_object_dominates_static_background(self):
        # 100x100 flow field, all static except a 10x10 patch moving down 5px
        flow = np.zeros((100, 100, 2), dtype=np.float32)
        flow[40:50, 40:50, 1] = 5.0
        dy, dx = weighted_flow(flow)
        # Magnitude weighting: only moving pixels carry weight, so dy ~= 5
        assert dy == pytest.approx(5.0, abs=0.01)
        assert dx == pytest.approx(0.0, abs=0.01)

    def test_no_motion_returns_zero(self):
        flow = np.zeros((50, 50, 2), dtype=np.float32)
        dy, dx = weighted_flow(flow)
        assert dy == 0.0 and dx == 0.0


class TestFlowToPosition:
    def test_maps_flow_velocity_around_center(self):
        dy = np.array([0.0, 2.0, -2.0, 10.0, -10.0])
        pos = flow_to_position(dy, gain=10.0, median_window=1)
        assert pos.tolist() == [50.0, 70.0, 30.0, 100.0, 0.0]

    def test_median3_prefilter_kills_single_frame_spike(self):
        dy = np.array([0.0, 0.0, 8.0, 0.0, 0.0])
        pos = flow_to_position(dy, gain=10.0, median_window=3)
        assert pos[2] == 50.0


class TestAntiPlateauNormalize:
    def test_moderate_wave_expands_to_full_range(self):
        t = np.arange(600)
        wave = 50 + 20 * np.sin(2 * np.pi * t / 30)  # 30-70 oscillation
        out = anti_plateau_normalize(wave, window=120, threshold=15.0)
        assert out.max() > 95 and out.min() < 5

    def test_tiny_jitter_is_not_amplified(self):
        rng = np.random.default_rng(42)
        jitter = 50 + rng.normal(0, 1.5, 600)  # p90-p10 well under threshold
        out = anti_plateau_normalize(jitter, window=120, threshold=15.0)
        np.testing.assert_allclose(out, jitter)

    def test_flat_signal_unchanged(self):
        flat = np.full(300, 50.0)
        out = anti_plateau_normalize(flat, window=120, threshold=15.0)
        np.testing.assert_allclose(out, flat)


class TestSceneCut:
    def test_identical_frames_are_not_a_cut(self):
        frame = np.full((30, 40), 128, dtype=np.uint8)
        assert not is_scene_cut(frame, frame)

    def test_total_change_is_a_cut(self):
        black = np.zeros((30, 40), dtype=np.uint8)
        white = np.full((30, 40), 255, dtype=np.uint8)
        assert is_scene_cut(black, white)

    def test_small_motion_is_not_a_cut(self):
        rng = np.random.default_rng(7)
        frame = rng.integers(0, 255, (30, 40)).astype(np.uint8)
        shifted = np.roll(frame, 2, axis=0)  # small vertical pan
        assert not is_scene_cut(frame, shifted)


def textured_frame(seed=0, shape=(200, 300)):
    """BGR frame with enough texture that flow patches are meaningful."""
    rng = np.random.default_rng(seed)
    gray = rng.integers(0, 255, shape).astype(np.uint8)
    return np.stack([gray] * 3, axis=-1)


class TestTrackFlowSignal:
    def _constant_flow_fn(self, dy_value):
        def flow_fn(prev_gray, gray):
            flow = np.zeros((*gray.shape, 2), dtype=np.float32)
            flow[..., 1] = dy_value
            return flow
        return flow_fn

    def test_tracks_dy_when_anchor_and_contact_detected(self):
        frames = [textured_frame(0) for _ in range(6)]
        detections = [
            Detection("anchor", 0.9, (100, 60, 60, 90)),
            Detection("hand", 0.9, (110, 70, 50, 50)),
        ]
        result = track_flow_signal(
            iter(frames), lambda f: detections, self._constant_flow_fn(3.0),
            TrackConfig(detect_every=2),
        )
        assert len(result.dy) == 6
        # First frame has no previous patch; afterwards flow is constant 3
        assert result.dy[0] == 0.0
        np.testing.assert_allclose(result.dy[1:], 3.0)
        assert result.roi_active.all()

    def test_no_detections_means_flat_signal(self):
        frames = [textured_frame(0) for _ in range(4)]
        result = track_flow_signal(
            iter(frames), lambda f: [], self._constant_flow_fn(3.0),
            TrackConfig(),
        )
        np.testing.assert_allclose(result.dy, 0.0)
        assert not result.roi_active.any()

    def test_roi_persists_after_detection_loss_then_expires(self):
        frames = [textured_frame(0) for _ in range(10)]
        dets = [Detection("anchor", 0.9, (100, 60, 60, 90))]
        calls = {"n": 0}

        def detect_fn(frame):
            calls["n"] += 1
            return dets if calls["n"] == 1 else []

        result = track_flow_signal(
            iter(frames), detect_fn, self._constant_flow_fn(2.0),
            TrackConfig(detect_every=1, roi_persistence_frames=3),
        )
        # ROI from frame 0 persists 3 frames after loss, then clears
        assert result.roi_active[:4].all()
        assert not result.roi_active[-3:].any()

    def test_scene_cut_resets_flow_state(self):
        bright = np.full((200, 300, 3), 230, dtype=np.uint8)
        frames = [textured_frame(1)] * 3 + [bright] * 3
        detections = [Detection("anchor", 0.9, (100, 60, 60, 90))]
        result = track_flow_signal(
            iter(frames), lambda f: detections, self._constant_flow_fn(4.0),
            TrackConfig(detect_every=1),
        )
        # Flow must not bridge the cut at index 3
        assert result.dy[3] == 0.0
        assert 3 in result.cuts


class TestSignalToActions:
    def test_oscillating_flow_yields_alternating_strokes(self):
        fps = 30.0
        n = 300
        t = np.arange(n)
        # ~1 stroke/sec oscillation in flow velocity, active throughout
        dy = 3.0 * np.sin(2 * np.pi * t / 30)
        signal = TrackSignal(
            dy=dy, roi_active=np.ones(n, dtype=bool))
        actions = signal_to_actions(signal, fps=fps, config=TrackConfig())
        # ~10 strokes -> ~20 turnarounds; allow slack for edge handling
        assert 12 <= len(actions) <= 28
        pos = np.array([a["pos"] for a in actions])
        # Normalization should spread turnarounds wide
        assert pos.max() >= 85 and pos.min() <= 15
        # Timestamps in ms and increasing
        ts = np.array([a["at"] for a in actions])
        assert (np.diff(ts) > 0).all()
        assert ts[-1] <= n / fps * 1000

    def test_inactive_signal_yields_no_actions(self):
        n = 200
        signal = TrackSignal(
            dy=np.zeros(n), roi_active=np.zeros(n, dtype=bool))
        actions = signal_to_actions(signal, fps=30.0, config=TrackConfig())
        assert actions == []

    def test_timestamps_offset_by_start_frame(self):
        n = 90
        t = np.arange(n)
        dy = 3.0 * np.sin(2 * np.pi * t / 30)
        signal = TrackSignal(
            dy=dy, roi_active=np.ones(n, dtype=bool))
        base = signal_to_actions(signal, fps=30.0, config=TrackConfig())
        shifted = signal_to_actions(
            signal, fps=30.0, config=TrackConfig(), start_frame=300)
        assert len(base) == len(shifted)
        # 300 frames at 30fps = 10s offset
        for a, b in zip(base, shifted):
            assert b["at"] - a["at"] == 10000


class TestParseArgs:
    def test_output_defaults_next_to_video(self):
        args = parse_args([r"C:\vids\clip.mp4"])
        assert args.video == r"C:\vids\clip.mp4"
        assert args.output == r"C:\vids\clip.funscript"
        assert args.start_frame == 0
        assert args.end_frame is None

    def test_explicit_options(self):
        args = parse_args([
            "v.mp4", "--output", "out.funscript",
            "--start-frame", "3231", "--end-frame", "14478",
            "--detect-every", "2",
        ])
        assert args.output == "out.funscript"
        assert args.start_frame == 3231
        assert args.end_frame == 14478
        assert args.detect_every == 2

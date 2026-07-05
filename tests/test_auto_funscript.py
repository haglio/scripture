"""Tests for the automatic YOLO+flow funscript pipeline."""

import numpy as np
import pytest

from scripture.auto_funscript import (
    Detection,
    TrackConfig,
    TrackSignal,
    anti_plateau_normalize,
    background_flow,
    combine_roi,
    compute_positions,
    contact_near_box,
    find_interacting,
    flow_to_position,
    is_scene_cut,
    parse_args,
    pipeline_result_from_state,
    pipeline_result_to_state,
    run_pipeline,
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


class TestContactNearBox:
    def test_face_overlapping_box_is_contact(self):
        anchor_box = (100, 100, 50, 100)
        face = det("face", 90, 80, 120, 130)
        assert contact_near_box(anchor_box, [face]) == [face]

    def test_distant_contact_ignored(self):
        anchor_box = (100, 100, 50, 100)
        far_hand = det("hand", 600, 400, 50, 50)
        assert contact_near_box(anchor_box, [far_hand]) == []

    def test_non_contact_classes_ignored(self):
        anchor_box = (100, 100, 50, 100)
        anchor_tip = det("anchor_tip", 100, 100, 30, 30)
        navel = det("navel", 110, 110, 20, 20)
        assert contact_near_box(anchor_box, [anchor_tip, navel]) == []


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


class TestBackgroundFlow:
    def test_uniform_shift_is_recovered(self):
        flow = np.zeros((40, 60, 2), dtype=np.float32)
        flow[..., 0] = -3.0
        flow[..., 1] = 2.0
        dy, dx = background_flow(flow)
        assert (dy, dx) == (2.0, -3.0)

    def test_minority_mover_is_ignored(self):
        # Camera still, small object moving fast: background flow is zero
        flow = np.zeros((40, 60, 2), dtype=np.float32)
        flow[10:18, 20:30, 1] = 25.0
        dy, dx = background_flow(flow)
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

    def test_contact_over_lost_anchor_keeps_roi_alive(self):
        # Anchor visible at frame 0 only; face then covers its position.
        # Contact hold must outlast the plain persistence window.
        frames = [textured_frame(0) for _ in range(12)]
        anchor = Detection("anchor", 0.9, (100, 60, 60, 90))
        face_over_anchor = Detection("face", 0.9, (80, 40, 110, 130))
        calls = {"n": 0}

        def detect_fn(frame):
            calls["n"] += 1
            return [anchor, face_over_anchor] if calls["n"] == 1 else [face_over_anchor]

        result = track_flow_signal(
            iter(frames), detect_fn, self._constant_flow_fn(2.0),
            TrackConfig(detect_every=1, roi_persistence_frames=3),
        )
        assert result.roi_active.all()
        assert result.lock[0] == "anchor"
        assert set(result.lock[1:]) == {"contact"}

    def test_contact_far_from_lost_anchor_lets_roi_expire(self):
        frames = [textured_frame(0) for _ in range(12)]
        anchor = Detection("anchor", 0.9, (100, 60, 60, 90))
        far_face = Detection("face", 0.9, (600, 350, 80, 80))
        calls = {"n": 0}

        def detect_fn(frame):
            calls["n"] += 1
            return [anchor, far_face] if calls["n"] == 1 else [far_face]

        result = track_flow_signal(
            iter(frames), detect_fn, self._constant_flow_fn(2.0),
            TrackConfig(detect_every=1, roi_persistence_frames=3),
        )
        # Far face is not holding the lock: coast, then expire
        assert result.lock[1] == "coast"
        assert not result.roi_active[-3:].any()
        assert result.lock[-1] == "none"

    def test_belief_rides_global_motion_during_occlusion(self):
        # Anchor seen once, then nothing at all; the whole frame drifts
        # down 2px/frame, and the remembered box must drift with it.
        frames = [textured_frame(0) for _ in range(8)]
        anchor = Detection("anchor", 0.9, (100, 60, 60, 90))
        calls = {"n": 0}

        def detect_fn(frame):
            calls["n"] += 1
            return [anchor] if calls["n"] == 1 else []

        config = TrackConfig(detect_every=1, roi_persistence_frames=30)
        result = track_flow_signal(
            iter(frames), detect_fn, self._constant_flow_fn(2.0), config)
        assert result.beliefs[0] == (100, 60, 60, 90)
        # Global flow is measured on the downscaled frame, so dy=2 there
        # is 2 * downscale full-resolution pixels per frame, for 7 frames.
        expected_dy = 2 * config.global_motion_downscale * 7
        assert result.beliefs[-1][1] == pytest.approx(60 + expected_dy, abs=4)
        assert result.beliefs[-1][0] == pytest.approx(100, abs=4)

    def test_belief_leans_toward_held_contact(self):
        # Static camera; face holds the lock offset to the right of the
        # remembered spot: the belief should creep toward it, not stay put.
        frames = [textured_frame(0) for _ in range(12)]
        anchor = Detection("anchor", 0.9, (100, 100, 40, 60))
        face = Detection("face", 0.9, (140, 80, 100, 120))  # center (190, 140)
        calls = {"n": 0}

        def detect_fn(frame):
            calls["n"] += 1
            return [anchor, face] if calls["n"] == 1 else [face]

        result = track_flow_signal(
            iter(frames), detect_fn, self._constant_flow_fn(0.0),
            TrackConfig(detect_every=1, roi_persistence_frames=30),
        )
        assert set(result.lock[1:]) == {"contact"}
        start_cx = 100 + 40 / 2
        end_cx = result.beliefs[-1][0] + result.beliefs[-1][2] / 2
        assert end_cx > start_cx + 20   # moved toward the contact
        assert end_cx < 190             # but not snapped onto it

    def test_records_rois_and_detections_for_visualization(self):
        frames = [textured_frame(0) for _ in range(4)]
        dets = [
            Detection("anchor", 0.9, (100, 60, 60, 90)),
            Detection("hand", 0.8, (110, 70, 50, 50)),
        ]
        result = track_flow_signal(
            iter(frames), lambda f: dets, self._constant_flow_fn(1.0),
            TrackConfig(detect_every=2),
        )
        # One ROI tuple per frame (persisted between detections)
        assert len(result.rois) == 4
        assert all(r is not None and len(r) == 4 for r in result.rois)
        # Detections recorded only on the frames where YOLO ran
        assert sorted(result.detections.keys()) == [0, 2]
        assert [d.class_name for d in result.detections[0]] == ["anchor", "hand"]


class TestComputePositions:
    def test_full_pipeline_position_series(self):
        n = 300
        t = np.arange(n)
        dy = 3.0 * np.sin(2 * np.pi * t / 30)
        signal = TrackSignal(dy=dy, lock=["anchor"] * n)
        pos = compute_positions(signal, TrackConfig())
        assert len(pos) == n
        # Gain + normalization drive the wave to (nearly) full range
        assert pos.max() > 95 and pos.min() < 5


class TestSignalToActions:
    def test_oscillating_flow_yields_alternating_strokes(self):
        fps = 30.0
        n = 300
        t = np.arange(n)
        # ~1 stroke/sec oscillation in flow velocity, active throughout
        dy = 3.0 * np.sin(2 * np.pi * t / 30)
        signal = TrackSignal(dy=dy, lock=["anchor"] * n)
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
        signal = TrackSignal(dy=np.zeros(n), lock=["none"] * n)
        actions = signal_to_actions(signal, fps=30.0, config=TrackConfig())
        assert actions == []

    def test_timestamps_offset_by_start_frame(self):
        n = 90
        t = np.arange(n)
        dy = 3.0 * np.sin(2 * np.pi * t / 30)
        signal = TrackSignal(dy=dy, lock=["anchor"] * n)
        base = signal_to_actions(signal, fps=30.0, config=TrackConfig())
        shifted = signal_to_actions(
            signal, fps=30.0, config=TrackConfig(), start_frame=300)
        assert len(base) == len(shifted)
        # 300 frames at 30fps = 10s offset
        for a, b in zip(base, shifted):
            assert b["at"] - a["at"] == 10000


class TestRunPipeline:
    def test_processes_video_file_with_injected_stages(self, tmp_path):
        import cv2

        video_path = str(tmp_path / "clip.mp4")
        writer = cv2.VideoWriter(
            video_path, cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (160, 120))
        rng = np.random.default_rng(3)
        base = rng.integers(0, 255, (120, 160, 3)).astype(np.uint8)
        for _ in range(30):
            writer.write(base)
        writer.release()

        dets = [Detection("anchor", 0.9, (40, 30, 40, 60))]

        def fake_flow(prev_gray, gray):
            flow = np.zeros((*gray.shape, 2), dtype=np.float32)
            flow[..., 1] = 2.0
            return flow

        progress = []
        result = run_pipeline(
            video_path, config=TrackConfig(detect_every=5),
            on_frame=progress.append,
            detect_fn=lambda f: dets, flow_fn=fake_flow,
        )
        assert len(result.positions) == 30
        assert len(result.signal.rois) == 30
        assert result.fps == pytest.approx(30.0, abs=0.1)
        assert isinstance(result.actions, list)
        assert progress == list(range(30))


class TestPipelineResultSerialization:
    def test_json_round_trip(self):
        import json

        from scripture.auto_funscript import PipelineResult

        signal = TrackSignal(
            dy=np.array([0.0, 1.5, -2.0]),
            lock=["anchor", "contact", "none"],
            cuts=[2],
            rois=[(10, 20, 130, 140), (12, 22, 130, 140), None],
            detections={0: [Detection("anchor", 0.9, (10, 20, 30, 40))]},
            beliefs=[(10, 20, 30, 40), (11, 21, 30, 40), None],
        )
        original = PipelineResult(
            signal=signal,
            positions=np.array([50.0, 65.0, 30.0]),
            actions=[{"at": 0, "pos": 50}],
            fps=29.97,
            start_frame=100,
            total_frames=500,
        )
        state = json.loads(json.dumps(pipeline_result_to_state(original)))
        restored = pipeline_result_from_state(state)

        np.testing.assert_allclose(restored.signal.dy, signal.dy)
        assert restored.signal.lock == ["anchor", "contact", "none"]
        np.testing.assert_array_equal(
            restored.signal.roi_active, [True, True, False])
        assert restored.signal.cuts == [2]
        assert restored.signal.rois == [(10, 20, 130, 140), (12, 22, 130, 140), None]
        assert restored.signal.beliefs == [(10, 20, 30, 40), (11, 21, 30, 40), None]
        det = restored.signal.detections[0][0]
        assert det.class_name == "anchor" and det.box == (10, 20, 30, 40)
        np.testing.assert_allclose(restored.positions, original.positions)
        assert restored.actions == original.actions
        assert restored.fps == pytest.approx(29.97)
        assert restored.start_frame == 100 and restored.total_frames == 500


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

"""Tests for the automatic YOLO+flow funscript pipeline."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripture.auto_funscript import (
    ANCHOR_CLASSES,
    RECIPE_VERSION,
    Detection,
    PipelineResult,
    TrackConfig,
    TrackSignal,
    anti_plateau_normalize,
    combine_roi,
    compute_positions,
    contact_near_rect,
    find_interacting,
    flow_to_position,
    generate_funscript,
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
from tests.videos import a_flat_video

# The class vocabulary is private; take it from whichever overlay is loaded.
ANCHOR, ANCHOR_TIP = ANCHOR_CLASSES[0], ANCHOR_CLASSES[1]


def _flow_of(dy):
    """A flow field of constant vertical motion, whatever the frames."""
    def flow_fn(_prev_gray, gray):
        flow = np.zeros((*gray.shape, 2), dtype=np.float32)
        flow[..., 1] = dy
        return flow
    return flow_fn


def det(cls, x, y, w, h, conf=0.9):
    return Detection(class_name=cls, confidence=conf, rect=(x, y, w, h))


class TestFindInteracting:
    def test_nearby_hand_interacts(self):
        # Anchor at (100,100) 50x100, hand overlapping its center region
        anchor = det(ANCHOR, 100, 100, 50, 100)
        hand = det("hand", 110, 120, 40, 40)
        assert find_interacting(anchor, [anchor, hand]) == [hand]

    def test_distant_object_does_not_interact(self):
        anchor = det(ANCHOR, 100, 100, 50, 100)
        far_face = det("face", 500, 400, 60, 60)
        assert find_interacting(anchor, [anchor, far_face]) == []


class TestContactNearBox:
    def test_face_overlapping_box_is_contact(self):
        anchor_box = (100, 100, 50, 100)
        face = det("face", 90, 80, 120, 130)
        assert contact_near_rect(anchor_box, [face]) == [face]

    def test_distant_contact_ignored(self):
        anchor_box = (100, 100, 50, 100)
        far_hand = det("hand", 600, 400, 50, 50)
        assert contact_near_rect(anchor_box, [far_hand]) == []

    def test_non_contact_classes_ignored(self):
        anchor_box = (100, 100, 50, 100)
        anchor_tip = det(ANCHOR_TIP, 100, 100, 30, 30)
        navel = det("navel", 110, 110, 20, 20)
        assert contact_near_rect(anchor_box, [anchor_tip, navel]) == []


class TestCombineRoi:
    def test_union_with_padding(self):
        rects = [(200, 200, 100, 200), (180, 150, 80, 80)]
        # Union: x 180-300, y 150-400. Padding 20 -> 160-320, 130-420.
        assert combine_roi(rects, frame_size=(544, 960), padding=20) == (160, 130, 160, 290)

    def test_clamped_to_frame(self):
        rects = [(0, 0, 200, 200)]
        x, y, w, h = combine_roi(rects, frame_size=(544, 960), padding=30)
        assert (x, y) == (0, 0)
        assert (w, h) == (230, 230)

    def test_minimum_size_enforced(self):
        rects = [(400, 300, 20, 20)]
        x, y, w, h = combine_roi(rects, frame_size=(544, 960), padding=0, min_size=128)
        assert w == 128 and h == 128
        # Still centered on the small rect and inside the frame
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
    def test_tracks_dy_when_anchor_and_contact_detected(self):
        frames = [textured_frame(0) for _ in range(6)]
        detections = [
            Detection(ANCHOR, 0.9, (100, 60, 60, 90)),
            Detection("hand", 0.9, (110, 70, 50, 50)),
        ]
        result = track_flow_signal(
            iter(frames), lambda f: detections, _flow_of(3.0),
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
            iter(frames), lambda f: [], _flow_of(3.0),
            TrackConfig(),
        )
        np.testing.assert_allclose(result.dy, 0.0)
        assert not result.roi_active.any()

    def test_roi_persists_after_detection_loss_then_expires(self):
        frames = [textured_frame(0) for _ in range(10)]
        dets = [Detection(ANCHOR, 0.9, (100, 60, 60, 90))]
        calls = {"n": 0}

        def detect_fn(frame):
            calls["n"] += 1
            return dets if calls["n"] == 1 else []

        result = track_flow_signal(
            iter(frames), detect_fn, _flow_of(2.0),
            TrackConfig(detect_every=1, roi_persistence_frames=3),
        )
        # ROI from frame 0 persists 3 frames after loss, then clears
        assert result.roi_active[:4].all()
        assert not result.roi_active[-3:].any()

    def test_scene_cut_resets_flow_state(self):
        bright = np.full((200, 300, 3), 230, dtype=np.uint8)
        frames = [textured_frame(1)] * 3 + [bright] * 3
        detections = [Detection(ANCHOR, 0.9, (100, 60, 60, 90))]
        result = track_flow_signal(
            iter(frames), lambda f: detections, _flow_of(4.0),
            TrackConfig(detect_every=1),
        )
        # Flow must not bridge the cut at index 3
        assert result.dy[3] == 0.0
        assert 3 in result.cuts

    def test_contact_over_lost_anchor_keeps_roi_alive(self):
        # Anchor visible at frame 0 only; face then covers its position.
        # Contact hold must outlast the plain persistence window.
        frames = [textured_frame(0) for _ in range(12)]
        anchor = Detection(ANCHOR, 0.9, (100, 60, 60, 90))
        face_over_anchor = Detection("face", 0.9, (80, 40, 110, 130))
        calls = {"n": 0}

        def detect_fn(frame):
            calls["n"] += 1
            return [anchor, face_over_anchor] if calls["n"] == 1 else [face_over_anchor]

        result = track_flow_signal(
            iter(frames), detect_fn, _flow_of(2.0),
            TrackConfig(detect_every=1, roi_persistence_frames=3),
        )
        assert result.roi_active.all()
        assert result.lock[0] == "anchor"
        assert set(result.lock[1:]) == {"contact"}

    def test_contact_far_from_lost_anchor_lets_roi_expire(self):
        frames = [textured_frame(0) for _ in range(12)]
        anchor = Detection(ANCHOR, 0.9, (100, 60, 60, 90))
        far_face = Detection("face", 0.9, (600, 350, 80, 80))
        calls = {"n": 0}

        def detect_fn(frame):
            calls["n"] += 1
            return [anchor, far_face] if calls["n"] == 1 else [far_face]

        result = track_flow_signal(
            iter(frames), detect_fn, _flow_of(2.0),
            TrackConfig(detect_every=1, roi_persistence_frames=3),
        )
        # Far face is not holding the lock: coast, then expire
        assert result.lock[1] == "coast"
        assert not result.roi_active[-3:].any()
        assert result.lock[-1] == "none"

    def test_belief_stays_frozen_during_occlusion(self):
        # Measured on real footage: every attempt to move the remembered
        # rect (global-motion translation, contact pinning) predicted the
        # re-appearing anchor WORSE than simply freezing it. Freeze it.
        frames = [textured_frame(0) for _ in range(10)]
        anchor = Detection(ANCHOR, 0.9, (100, 60, 60, 90))
        face_over_anchor = Detection("face", 0.9, (80, 40, 110, 130))
        calls = {"n": 0}

        def detect_fn(frame):
            calls["n"] += 1
            return [anchor, face_over_anchor] if calls["n"] == 1 else [face_over_anchor]

        result = track_flow_signal(
            iter(frames), detect_fn, _flow_of(2.0),
            TrackConfig(detect_every=1, roi_persistence_frames=30),
        )
        assert set(result.lock[1:]) == {"contact"}
        assert all(b == (100, 60, 60, 90) for b in result.beliefs)

    def test_records_rois_and_detections_for_visualization(self):
        frames = [textured_frame(0) for _ in range(4)]
        dets = [
            Detection(ANCHOR, 0.9, (100, 60, 60, 90)),
            Detection("hand", 0.8, (110, 70, 50, 50)),
        ]
        result = track_flow_signal(
            iter(frames), lambda f: dets, _flow_of(1.0),
            TrackConfig(detect_every=2),
        )
        # One ROI tuple per frame (persisted between detections)
        assert len(result.rois) == 4
        assert all(r is not None and len(r) == 4 for r in result.rois)
        # Detections recorded only on the frames where YOLO ran
        assert sorted(result.detections.keys()) == [0, 2]
        assert [d.class_name for d in result.detections[0]] == [ANCHOR, "hand"]


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
    def test_oscillating_flow_yields_alternating_cycles(self):
        fps = 30.0
        n = 300
        t = np.arange(n)
        # ~1 cycle/sec oscillation in flow velocity, active throughout
        dy = 3.0 * np.sin(2 * np.pi * t / 30)
        signal = TrackSignal(dy=dy, lock=["anchor"] * n)
        actions = signal_to_actions(signal, fps=fps, config=TrackConfig())
        # ~10 cycles -> ~20 turnarounds; allow slack for edge handling
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
        video_path = a_flat_video(tmp_path / "example clip.mp4")
        dets = [Detection(ANCHOR, 0.9, (40, 30, 40, 60))]

        progress = []
        result = run_pipeline(
            video_path, config=TrackConfig(detect_every=5),
            on_frame=progress.append,
            detect_fn=lambda f: dets, flow_fn=_flow_of(2.0),
        )
        assert len(result.positions) == 30
        assert len(result.signal.rois) == 30
        assert result.fps == pytest.approx(30.0, abs=0.1)
        assert isinstance(result.actions, list)
        assert progress == list(range(30))

    def test_the_result_says_which_tracker_made_it_and_at_which_version(self, tmp_path):
        video_path = a_flat_video(tmp_path / "example clip.mp4")

        result = run_pipeline(video_path, detect_fn=lambda _frame: [], flow_fn=_flow_of(0.0))

        made = result.provenance
        assert (made["app"], made["recipe"]) == ("scripture", "roi_flow")
        assert made["recipe_version"] == RECIPE_VERSION


class TestGenerateFunscript:
    def test_a_video_becomes_a_funscript_file_with_injected_stages(self, tmp_path):
        """The whole headless composition -- video in, file out. Nothing had
        run it end to end, because the stages could not be injected past
        run_pipeline."""

        video_path = a_flat_video(tmp_path / "example clip.mp4")
        output_path = str(tmp_path / "example clip.funscript")

        actions = generate_funscript(
            video_path, output_path, config=TrackConfig(detect_every=5),
            detect_fn=lambda _frame: [Detection(ANCHOR, 0.9, (40, 30, 40, 60))],
            flow_fn=_flow_of(2.0))

        written = json.loads(Path(output_path).read_text(encoding="utf-8"))
        assert written["metadata"]["creator"] == "scripture"
        assert written["actions"] == sorted(actions, key=lambda a: a["at"])

    def test_the_file_says_which_tracker_made_it_outside_the_block_players_read(self, tmp_path):
        video_path = a_flat_video(tmp_path / "example clip.mp4")
        output_path = tmp_path / "example clip.funscript"

        generate_funscript(
            video_path, str(output_path), detect_fn=lambda _frame: [], flow_fn=_flow_of(0.0))

        written = json.loads(output_path.read_text(encoding="utf-8"))
        assert written["provenance"]["recipe"] == "roi_flow"
        assert "provenance" not in written["metadata"]


class TestPipelineResultSerialization:
    def test_json_round_trip(self):
        signal = TrackSignal(
            dy=np.array([0.0, 1.5, -2.0]),
            lock=["anchor", "contact", "none"],
            cuts=[2],
            rois=[(10, 20, 130, 140), (12, 22, 130, 140), None],
            detections={0: [Detection(ANCHOR, 0.9, (10, 20, 30, 40))]},
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
        assert det.class_name == ANCHOR and det.rect == (10, 20, 30, 40)
        np.testing.assert_allclose(restored.positions, original.positions)
        assert restored.actions == original.actions
        assert restored.fps == pytest.approx(29.97)
        assert restored.start_frame == 100 and restored.total_frames == 500

    def test_what_made_the_result_comes_back_with_it(self):
        made = {"schema": 1, "app": "scripture", "app_commit": None, "app_dirty": None,
                "recipe": "roi_flow", "recipe_version": "1",
                "stamped_at": "2026-01-01T00:00:00+00:00"}
        original = PipelineResult(
            signal=TrackSignal(dy=np.array([0.0]), lock=["none"], rois=[None]),
            positions=np.array([50.0]), actions=[], fps=30.0, start_frame=0, total_frames=1,
            provenance=made)

        restored = pipeline_result_from_state(
            json.loads(json.dumps(pipeline_result_to_state(original))))

        assert restored.provenance == made

    def test_a_project_saved_before_the_field_was_renamed_still_loads(self):
        """A saved detection's rect used to sit under a different key.

        That key cannot be named in this tree any more, so the loader finds it
        by elimination -- a saved detection carries the class name, the
        confidence, and one other value, which is the rect. Projects are saved
        wherever the user chose, so there is no directory to migrate ahead of
        time; this read is the migration. Without it the load raises KeyError,
        and the caller that opens the last session swallows it, so the user's
        tracking would vanish with no message.
        """
        state = {
            "dy": [0.0, 1.0],
            "lock": ["anchor", "none"],
            "cuts": [],
            "rois": [(10, 20, 130, 140), None],
            "beliefs": [(10, 20, 30, 40), None],
            "detections": {
                "0": [{"class_name": ANCHOR, "confidence": 0.9,
                       "the_older_key": [10, 20, 30, 40]}],
            },
            "positions": [50.0, 65.0],
            "actions": [],
            "fps": 30.0,
            "start_frame": 0,
            "total_frames": 2,
        }

        restored = pipeline_result_from_state(state)

        det = restored.signal.detections[0][0]
        assert det.class_name == ANCHOR and det.rect == (10, 20, 30, 40)


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

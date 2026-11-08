from __future__ import annotations

import argparse
import os
import sys
import time
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from app_support.funscript import write as write_funscript
from app_support.overlay import overlay_value
from numpy.lib.stride_tricks import sliding_window_view
from scipy.ndimage import median_filter

from scripture.content import LOCAL_CONTENT, load_content
from scripture.cycle_extract import extract_cycles
from scripture.frames import ms_of_frame
from scripture.funscript import build_funscript, made_by
from scripture.roi_lock import Detection, RoiLock
from scripture.video import VideoSource

TRACKER = "roi_flow"
TRACKER_VERSION = "1"

_CONTENT = load_content()

DEFAULT_MODEL_PATH = overlay_value(_CONTENT, "model_path", path=LOCAL_CONTENT)

def flow_to_position(
    dy: np.ndarray,
    gain: float = 10.0,
    median_window: int = 3,
) -> np.ndarray:
    smoothed = median_filter(dy.astype(np.float64), size=median_window, mode="nearest")
    return np.clip(50.0 + gain * smoothed, 0.0, 100.0)


def anti_plateau_normalize(
    positions: np.ndarray,
    window: int = 120,
    threshold: float = 15.0,
) -> np.ndarray:
    n = len(positions)
    if n <= 1:
        return positions.astype(np.float64).copy()

    pad = min(window // 2, n)
    padded = np.pad(positions.astype(np.float64), (pad, pad), mode="edge")
    win = min(2 * pad + 1, len(padded))
    windows = sliding_window_view(padded, win)[:n]
    p10 = np.percentile(windows, 10, axis=1)
    p90 = np.percentile(windows, 90, axis=1)
    band = p90 - p10

    out = positions.astype(np.float64).copy()
    active = band > threshold
    out[active] = np.clip(
        (positions[active] - p10[active]) / band[active] * 100.0, 0.0, 100.0)
    return out


def is_scene_cut(
    prev_gray: np.ndarray,
    gray: np.ndarray,
    threshold: float = 0.5,
) -> bool:
    h1, _ = np.histogram(prev_gray, bins=32, range=(0, 256))
    h2, _ = np.histogram(gray, bins=32, range=(0, 256))
    h1 = h1 / max(1, h1.sum())
    h2 = h2 / max(1, h2.sum())
    return float(np.abs(h1 - h2).sum()) > threshold


def weighted_flow(flow: np.ndarray) -> tuple[float, float]:
    magnitudes = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
    total = magnitudes.sum()
    if total <= 0:
        return 0.0, 0.0
    dy = float((flow[..., 1] * magnitudes).sum() / total)
    dx = float((flow[..., 0] * magnitudes).sum() / total)
    return dy, dx


@dataclass
class TrackConfig:
    detect_every: int = 3
    conf_threshold: float = 0.4
    roi_padding: int = 20
    roi_min_size: int = 128
    roi_smoothing: float = 0.6
    roi_persistence_frames: int = 180
    patch_size: int = 256
    gain: float = 10.0
    median_window: int = 3
    norm_window: int = 120
    norm_threshold: float = 15.0
    cut_threshold: float = 0.5


@dataclass
class TrackSignal:
    dy: np.ndarray
    lock: list[str]
    rois: list[tuple[int, int, int, int] | None] = field(default_factory=list)
    detections: dict[int, list[Detection]] = field(default_factory=dict)
    beliefs: list[tuple[int, int, int, int] | None] = field(default_factory=list)

    @property
    def roi_active(self) -> np.ndarray:
        return np.array([state != "none" for state in self.lock], dtype=bool)


def compute_positions(signal: TrackSignal, config: TrackConfig) -> np.ndarray:
    positions = flow_to_position(
        signal.dy, gain=config.gain, median_window=config.median_window)
    return anti_plateau_normalize(
        positions, window=config.norm_window, threshold=config.norm_threshold)


def signal_to_actions(
    signal: TrackSignal,
    fps: float,
    config: TrackConfig,
    start_frame: int = 0,
) -> list[dict]:
    if not signal.roi_active.any():
        return []

    positions = compute_positions(signal, config)
    timestamps_ms = ms_of_frame(start_frame + np.arange(len(positions)), fps)
    return extract_cycles(positions / 100.0, timestamps_ms, fps=fps)


_CUT_DETECT_SIZE = (48, 27)


class _RoiFlow:
    def __init__(self, flow_fn: Callable[[np.ndarray, np.ndarray], np.ndarray], patch_size: int) -> None:
        self._flow_fn = flow_fn
        self._patch_size = patch_size
        self._previous: np.ndarray | None = None

    def forget(self) -> None:
        self._previous = None

    def dy_within(self, gray: np.ndarray, roi: tuple[int, int, int, int] | None) -> float:
        if roi is None:
            self.forget()
            return 0.0
        x, y, w, h = roi
        patch = gray[y:y + h, x:x + w]
        if patch.size == 0:
            return 0.0
        patch = cv2.resize(patch, (self._patch_size, self._patch_size))
        dy = 0.0
        if self._previous is not None:
            dy, _ = weighted_flow(self._flow_fn(self._previous, patch))
        self._previous = patch
        return dy


def track_flow_signal(
    frames: Iterable[np.ndarray] | Iterator[np.ndarray],
    detect_fn: Callable[[np.ndarray], list[Detection]],
    flow_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    config: TrackConfig,
    on_frame: Callable[[int], None] | None = None,
) -> TrackSignal:
    dy_list: list[float] = []
    lock_log: list[str] = []
    roi_log: list[tuple[int, int, int, int] | None] = []
    belief_log: list[tuple[int, int, int, int] | None] = []
    detection_log: dict[int, list[Detection]] = {}

    lock = RoiLock(detect_every=config.detect_every, padding=config.roi_padding,
                   min_size=config.roi_min_size, smoothing=config.roi_smoothing,
                   persistence=config.roi_persistence_frames)
    flow = _RoiFlow(flow_fn, config.patch_size)
    prev_small: np.ndarray | None = None

    for idx, frame in enumerate(frames):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, _CUT_DETECT_SIZE)
        if prev_small is not None and is_scene_cut(prev_small, small, config.cut_threshold):
            lock.lose()
            flow.forget()
        prev_small = small

        refreshed = False
        if lock.wants_detection_on(idx):
            detection_log[idx] = detect_fn(frame)
            refreshed = lock.look_at(detection_log[idx], gray.shape)
        if not refreshed:
            lock.grow_older()

        dy_list.append(flow.dy_within(gray, lock.roi))
        lock_log.append(lock.mode if lock.roi is not None else "none")
        roi_log.append(lock.roi)
        belief_log.append(lock.belief)
        if on_frame is not None:
            on_frame(idx)

    return TrackSignal(
        dy=np.array(dy_list),
        lock=lock_log,
        rois=roi_log,
        detections=detection_log,
        beliefs=belief_log,
    )


def _read_frames(video_path: str, start_frame: int, end_frame: int | None):
    cap = cv2.VideoCapture(video_path)
    if start_frame:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    idx = start_frame
    try:
        while end_frame is None or idx < end_frame:
            ret, frame = cap.read()
            if not ret:
                break
            yield frame
            idx += 1
    finally:
        cap.release()


def _yolo():
    from ultralytics import YOLO  # noqa: PLC0415

    return YOLO


def _make_yolo_detector(model_path: str, conf_threshold: float):
    model = _yolo()(model_path, task="detect")

    def detect(frame: np.ndarray) -> list[Detection]:
        results = model(frame, device=0, verbose=False, conf=conf_threshold)
        detections = []
        for result in results:
            for found in result.boxes:
                x1, y1, x2, y2 = found.xyxy[0].tolist()
                detections.append(Detection(
                    class_name=result.names[int(found.cls[0])],
                    confidence=float(found.conf[0]),
                    rect=(int(x1), int(y1), int(x2 - x1), int(y2 - y1)),
                ))
        return detections

    return detect


def _make_dis_flow():
    dis = cv2.DISOpticalFlow.create(cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST)

    def flow(prev_gray: np.ndarray, gray: np.ndarray) -> np.ndarray:
        return dis.calc(prev_gray, gray, None)

    return flow


@dataclass
class PipelineResult:
    signal: TrackSignal
    positions: np.ndarray
    actions: list[dict]
    fps: float
    start_frame: int
    total_frames: int
    provenance: dict | None = None


@dataclass(frozen=True)
class PipelineRun:
    model_path: str = DEFAULT_MODEL_PATH
    start_frame: int = 0
    end_frame: int | None = None
    config: TrackConfig = field(default_factory=TrackConfig)
    detect_fn: Callable[[np.ndarray], list[Detection]] | None = None
    flow_fn: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None


def run_pipeline(
    video_path: str,
    run: PipelineRun | None = None,
    *,
    on_frame: Callable[[int], None] | None = None,
) -> PipelineResult:
    run = run or PipelineRun()
    config = run.config

    source = VideoSource(cv2.VideoCapture(video_path))
    fps, total_frames = source.fps, source.total_frames
    source.release()

    signal = track_flow_signal(
        _read_frames(video_path, run.start_frame, run.end_frame),
        run.detect_fn or _make_yolo_detector(run.model_path, config.conf_threshold),
        run.flow_fn or _make_dis_flow(),
        config,
        on_frame=on_frame,
    )

    return PipelineResult(
        signal=signal,
        positions=compute_positions(signal, config),
        actions=signal_to_actions(signal, fps=fps, config=config,
                                  start_frame=run.start_frame),
        fps=fps,
        start_frame=run.start_frame,
        total_frames=total_frames,
        provenance=made_by(TRACKER, TRACKER_VERSION),
    )


def pipeline_result_to_state(result: PipelineResult) -> dict:
    sig = result.signal
    return {
        "dy": [round(v, 4) for v in sig.dy.tolist()],
        "lock": list(sig.lock),
        "rois": [list(r) if r is not None else None for r in sig.rois],
        "beliefs": [list(b) if b is not None else None for b in sig.beliefs],
        "detections": {
            str(idx): [
                {"class_name": d.class_name, "confidence": d.confidence,
                 "rect": list(d.rect)}
                for d in dets
            ]
            for idx, dets in sig.detections.items()
        },
        "positions": [round(v, 2) for v in result.positions.tolist()],
        "actions": result.actions,
        "fps": result.fps,
        "start_frame": result.start_frame,
        "total_frames": result.total_frames,
        "provenance": result.provenance,
    }


def _detection_from_state(saved: dict) -> Detection:
    rect = saved["rect"] if "rect" in saved else next(
        value for key, value in saved.items()
        if key not in ("class_name", "confidence"))
    return Detection(class_name=saved["class_name"],
                     confidence=saved["confidence"],
                     rect=tuple(rect))


def pipeline_result_from_state(state: dict) -> PipelineResult:
    lock = state.get("lock")
    if lock is None:
        lock = ["anchor" if a else "none" for a in state["roi_active"]]
    signal = TrackSignal(
        dy=np.array(state["dy"]),
        lock=lock,
        rois=[tuple(r) if r is not None else None for r in state["rois"]],
        beliefs=[tuple(b) if b is not None else None
                 for b in state.get("beliefs", [None] * len(state["rois"]))],
        detections={
            int(idx): [_detection_from_state(d) for d in dets]
            for idx, dets in state["detections"].items()
        },
    )
    return PipelineResult(
        signal=signal,
        positions=np.array(state["positions"]),
        actions=state["actions"],
        fps=state["fps"],
        start_frame=state["start_frame"],
        total_frames=state["total_frames"],
        provenance=state.get("provenance"),
    )


def generate_funscript(
    video_path: str,
    output_path: str,
    run: PipelineRun | None = None,
    *,
    on_frame: Callable[[int], None] | None = None,
) -> list[dict]:
    result = run_pipeline(video_path, run, on_frame=on_frame)
    funscript = build_funscript(
        result.actions, duration_seconds=int(result.total_frames / result.fps),
        provenance=result.provenance)
    write_funscript(Path(output_path), funscript)
    return result.actions


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="auto_funscript",
        description="Generate a funscript from a video, no human input.")
    parser.add_argument("video", help="path to the video file")
    parser.add_argument("--output", help="output .funscript path "
                        "(default: next to the video)")
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH,
                        help="YOLO detection model path")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int, default=None)
    parser.add_argument("--detect-every", type=int,
                        default=TrackConfig.detect_every,
                        help="run YOLO every N frames")
    args = parser.parse_args(argv)
    if args.output is None:
        args.output = os.path.splitext(args.video)[0] + ".funscript"
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    config = TrackConfig(detect_every=args.detect_every)

    started = time.time()
    n_frames_hint = None

    def report(idx: int) -> None:
        if idx % 300 == 0 and idx:
            rate = idx / (time.time() - started)
            msg = f"\rframe {idx}"
            if n_frames_hint:
                msg += f"/{n_frames_hint} ({idx / n_frames_hint:.0%})"
            msg += f"  {rate:.0f} fps"
            print(msg, end="", flush=True)

    cap = cv2.VideoCapture(args.video)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    n_frames_hint = (args.end_frame or total) - args.start_frame

    actions = generate_funscript(
        args.video, args.output,
        PipelineRun(model_path=args.model, start_frame=args.start_frame,
                    end_frame=args.end_frame, config=config),
        on_frame=report,
    )
    elapsed = time.time() - started
    print(f"\n{len(actions)} actions -> {args.output}  ({elapsed:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

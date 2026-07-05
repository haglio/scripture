"""Fully automatic funscript generation: YOLO detection + ROI optical flow.

Offline port of the only pipeline that has produced a usable funscript on
2D POV footage (FunGen's LIVE_YOLO_ROI tracker): YOLO finds the anchor and
whatever is interacting with it, their union defines a region of interest,
and dense optical flow inside that ROI drives the position signal.

CLI:  python -m scripture.auto_funscript VIDEO [--start-frame N] [--end-frame N]
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field

import numpy as np

DEFAULT_MODEL_PATH = (
    r"C:\path\to\suite-root\projects\FunGenApp\FunGen\models\FunGen-12s-pov-1.1.0.pt")


@dataclass
class Detection:
    """One YOLO detection with box in (x, y, w, h) pixel coords."""
    class_name: str
    confidence: float
    box: tuple[int, int, int, int]


# An object interacts with the anchor when its center is closer than this
# fraction of the two boxes' combined half-diagonals.
_INTERACTION_DISTANCE_FACTOR = 0.85

# Classes that can be the thing touching the anchor.  When the anchor itself
# is occluded (gripped, in mouth), one of these overlapping its last known
# position is evidence the interaction is still happening there.
_CONTACT_CLASSES = ("face", "hand", "region_a", "redacted", "redacted", "foot")


def _center(box: tuple[int, int, int, int]) -> tuple[float, float]:
    x, y, w, h = box
    return x + w / 2, y + h / 2


def _half_diagonal(box: tuple[int, int, int, int]) -> float:
    _, _, w, h = box
    return math.hypot(w, h) / 2


def _within_interaction_distance(
    box_a: tuple[int, int, int, int],
    box_b: tuple[int, int, int, int],
) -> bool:
    acx, acy = _center(box_a)
    bcx, bcy = _center(box_b)
    dist = math.hypot(bcx - acx, bcy - acy)
    max_dist = (_half_diagonal(box_a) + _half_diagonal(box_b)) * _INTERACTION_DISTANCE_FACTOR
    return dist < max_dist


def find_interacting(anchor: Detection, detections: list[Detection]) -> list[Detection]:
    """Return the non-anchor detections close enough to interact with it."""
    return [
        d for d in detections
        if d.class_name != "anchor" and _within_interaction_distance(anchor.box, d.box)
    ]


def contact_near_box(
    box: tuple[int, int, int, int],
    detections: list[Detection],
) -> list[Detection]:
    """Contact-class detections close enough to `box` to be touching it."""
    return [
        d for d in detections
        if d.class_name in _CONTACT_CLASSES
        and _within_interaction_distance(box, d.box)
    ]


def combine_roi(
    boxes: list[tuple[int, int, int, int]],
    frame_size: tuple[int, int],
    padding: int = 20,
    min_size: int = 128,
) -> tuple[int, int, int, int]:
    """Union of boxes, padded and clamped to the frame, at least min_size.

    frame_size is (height, width); the returned ROI is (x, y, w, h).
    """
    frame_h, frame_w = frame_size
    x1 = max(0, min(b[0] for b in boxes) - padding)
    y1 = max(0, min(b[1] for b in boxes) - padding)
    x2 = min(frame_w, max(b[0] + b[2] for b in boxes) + padding)
    y2 = min(frame_h, max(b[1] + b[3] for b in boxes) + padding)

    w, h = x2 - x1, y2 - y1
    if w < min_size:
        x1 = max(0, x1 - (min_size - w) // 2)
        w = min_size
    if h < min_size:
        y1 = max(0, y1 - (min_size - h) // 2)
        h = min_size

    x1 = min(x1, max(0, frame_w - w))
    y1 = min(y1, max(0, frame_h - h))
    return int(x1), int(y1), int(w), int(h)


def smooth_roi(
    previous: tuple[int, int, int, int] | None,
    new: tuple[int, int, int, int],
    factor: float,
) -> tuple[int, int, int, int]:
    """Blend the new ROI toward the previous one; factor is the weight kept."""
    if previous is None:
        return new
    return tuple(int(p * factor + n * (1 - factor)) for p, n in zip(previous, new))


def flow_to_position(
    dy: np.ndarray,
    gain: float = 10.0,
    median_window: int = 3,
) -> np.ndarray:
    """Map per-frame flow velocity to a raw 0-100 position series.

    This is the mapping FunGen's live tracker actually ships: the position
    is the (median-filtered) flow velocity scaled around a center of 50.
    For rhythmic stroking a scaled velocity is itself a stroke wave, just
    phase-shifted, which is why this works.
    """
    from scipy.ndimage import median_filter

    smoothed = median_filter(dy.astype(np.float64), size=median_window, mode="nearest")
    return np.clip(50.0 + gain * smoothed, 0.0, 100.0)


def anti_plateau_normalize(
    positions: np.ndarray,
    window: int = 120,
    threshold: float = 15.0,
) -> np.ndarray:
    """Rescale positions so local stroke ranges span the full 0-100.

    For each sample, the surrounding window's p10..p90 band is stretched to
    0..100 — but only where that band exceeds the threshold, so idle jitter
    stays small instead of being blown up into fake strokes.  Offline we can
    center the window, avoiding the lag FunGen's causal version has.
    """
    from numpy.lib.stride_tricks import sliding_window_view

    n = len(positions)
    if n < 2:
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
    """Detect a hard cut by comparing grayscale histograms.

    Histograms ignore in-scene motion (pans, strokes) but shift sharply
    when the shot changes.  threshold is the L1 distance between the two
    normalized histograms (0 = identical, 2 = disjoint).
    """
    h1, _ = np.histogram(prev_gray, bins=32, range=(0, 256))
    h2, _ = np.histogram(gray, bins=32, range=(0, 256))
    h1 = h1 / max(1, h1.sum())
    h2 = h2 / max(1, h2.sum())
    return float(np.abs(h1 - h2).sum()) > threshold


def weighted_flow(flow: np.ndarray) -> tuple[float, float]:
    """Magnitude-weighted mean (dy, dx) of a dense flow field.

    Weighting each vector by its own magnitude keeps a small moving object
    (hand, face) from being diluted by a large static background.
    """
    magnitudes = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
    total = magnitudes.sum()
    if total <= 0:
        return 0.0, 0.0
    dy = float((flow[..., 1] * magnitudes).sum() / total)
    dx = float((flow[..., 0] * magnitudes).sum() / total)
    return dy, dx


@dataclass
class TrackConfig:
    """Tuning knobs; defaults mirror the FunGen run that produced a usable
    script (padding/persistence/smoothing) plus offline-only additions."""
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
    """Per-frame motion signal plus diagnostics from the tracking pass.

    lock records how the ROI was justified on each frame: "anchor" (anchor or
    anchor_tip detected), "contact" (anchor occluded but a contact-class object
    covers its last known position), "coast" (nothing relevant detected,
    persistence window still open), or "none" (no ROI).  rois has one entry
    per frame; detections holds the YOLO boxes for exactly the frames where
    detection ran.  All of it exists so a GUI can replay what the tracker saw.
    """
    dy: np.ndarray
    lock: list[str]
    cuts: list[int] = field(default_factory=list)
    rois: list[tuple[int, int, int, int] | None] = field(default_factory=list)
    detections: dict[int, list[Detection]] = field(default_factory=dict)

    @property
    def roi_active(self) -> np.ndarray:
        return np.array([state != "none" for state in self.lock], dtype=bool)


def compute_positions(signal: TrackSignal, config: TrackConfig) -> np.ndarray:
    """The final per-frame 0-100 position series behind the actions."""
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
    """Turn the per-frame flow signal into funscript turnaround actions."""
    from scripture.stroke_extract import extract_strokes

    if not signal.roi_active.any():
        return []

    positions = compute_positions(signal, config)
    frame_ms = 1000.0 / fps
    timestamps_ms = (start_frame + np.arange(len(positions))) * frame_ms
    return extract_strokes(positions / 100.0, timestamps_ms, fps=fps)


def _best_anchor(detections: list[Detection]) -> Detection | None:
    """Pick the box that anchors the ROI: the anchor, falling back to the
    anchor_tip (which the model often still sees when the redacted is gripped)."""
    for cls in ("anchor", "anchor_tip"):
        candidates = [d for d in detections if d.class_name == cls]
        if candidates:
            return max(candidates, key=lambda d: d.confidence)
    return None


_CUT_DETECT_SIZE = (48, 27)


def track_flow_signal(
    frames: Iterable[np.ndarray] | Iterator[np.ndarray],
    detect_fn: Callable[[np.ndarray], list[Detection]],
    flow_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    config: TrackConfig,
    on_frame: Callable[[int], None] | None = None,
) -> TrackSignal:
    """Run the detection + ROI + optical-flow loop over a frame stream.

    detect_fn maps a BGR frame to Detections; flow_fn maps two equal-size
    grayscale patches to a dense flow field.  Both are injected so the loop
    is testable without a GPU.
    """
    import cv2

    dy_list: list[float] = []
    lock_log: list[str] = []
    cuts: list[int] = []
    roi_log: list[tuple[int, int, int, int] | None] = []
    detection_log: dict[int, list[Detection]] = {}

    roi: tuple[int, int, int, int] | None = None
    last_anchor_box: tuple[int, int, int, int] | None = None
    lock_mode = "none"
    prev_patch: np.ndarray | None = None
    prev_small: np.ndarray | None = None
    frames_since_lock = 0

    for idx, frame in enumerate(frames):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, _CUT_DETECT_SIZE)
        if prev_small is not None and is_scene_cut(prev_small, small, config.cut_threshold):
            cuts.append(idx)
            roi = None
            last_anchor_box = None
            lock_mode = "none"
            prev_patch = None
            frames_since_lock = 0
        prev_small = small

        lock_refreshed = False
        if idx % config.detect_every == 0 or roi is None:
            detections = detect_fn(frame)
            detection_log[idx] = detections
            anchor = _best_anchor(detections)
            if anchor is not None:
                last_anchor_box = anchor.box
                lock_mode = "anchor"
                lock_refreshed = True
                boxes = [anchor.box] + [
                    d.box for d in find_interacting(anchor, detections)]
                target = combine_roi(
                    boxes, gray.shape, config.roi_padding, config.roi_min_size)
                roi = smooth_roi(roi, target, config.roi_smoothing)
            elif roi is not None and last_anchor_box is not None:
                # Anchor occluded: something touching its last known spot
                # is evidence the interaction continues right there.
                contacts = contact_near_box(last_anchor_box, detections)
                if contacts:
                    lock_mode = "contact"
                    lock_refreshed = True
                    target = combine_roi(
                        [last_anchor_box] + [d.box for d in contacts],
                        gray.shape, config.roi_padding, config.roi_min_size)
                    roi = smooth_roi(roi, target, config.roi_smoothing)
                else:
                    lock_mode = "coast"

        if lock_refreshed:
            frames_since_lock = 0
        elif roi is not None:
            frames_since_lock += 1
            if frames_since_lock > config.roi_persistence_frames:
                roi = None
                last_anchor_box = None
                lock_mode = "none"
                prev_patch = None

        dy = 0.0
        if roi is not None:
            x, y, w, h = roi
            patch = gray[y:y + h, x:x + w]
            if patch.size > 0:
                patch = cv2.resize(patch, (config.patch_size, config.patch_size))
                if prev_patch is not None:
                    dy, _ = weighted_flow(flow_fn(prev_patch, patch))
                prev_patch = patch
        else:
            prev_patch = None

        dy_list.append(dy)
        lock_log.append(lock_mode if roi is not None else "none")
        roi_log.append(roi)
        if on_frame is not None:
            on_frame(idx)

    return TrackSignal(
        dy=np.array(dy_list),
        lock=lock_log,
        cuts=cuts,
        rois=roi_log,
        detections=detection_log,
    )


def _read_frames(video_path: str, start_frame: int, end_frame: int | None):
    """Yield BGR frames from start_frame up to (exclusive) end_frame."""
    import cv2

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


def _make_yolo_detector(model_path: str, conf_threshold: float):
    """Wrap an ultralytics model as detect_fn(frame) -> list[Detection]."""
    from ultralytics import YOLO

    model = YOLO(model_path, task="detect")

    def detect(frame: np.ndarray) -> list[Detection]:
        results = model(frame, device=0, verbose=False, conf=conf_threshold)
        detections = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                detections.append(Detection(
                    class_name=result.names[int(box.cls[0])],
                    confidence=float(box.conf[0]),
                    box=(int(x1), int(y1), int(x2 - x1), int(y2 - y1)),
                ))
        return detections

    return detect


def _make_dis_flow():
    """Wrap OpenCV DIS optical flow as flow_fn(prev_gray, gray) -> flow."""
    import cv2

    dis = cv2.DISOpticalFlow.create(cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST)

    def flow(prev_gray: np.ndarray, gray: np.ndarray) -> np.ndarray:
        return dis.calc(prev_gray, gray, None)

    return flow


@dataclass
class PipelineResult:
    """Everything the tracking pass produced for a stretch of video."""
    signal: TrackSignal
    positions: np.ndarray
    actions: list[dict]
    fps: float
    start_frame: int
    total_frames: int


def run_pipeline(
    video_path: str,
    model_path: str = DEFAULT_MODEL_PATH,
    start_frame: int = 0,
    end_frame: int | None = None,
    config: TrackConfig | None = None,
    on_frame: Callable[[int], None] | None = None,
    detect_fn: Callable[[np.ndarray], list[Detection]] | None = None,
    flow_fn: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None,
) -> PipelineResult:
    """Run the full pipeline in memory; detect_fn/flow_fn are injectable."""
    import cv2

    config = config or TrackConfig()

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if end_frame is None or end_frame > total_frames:
        end_frame = total_frames

    signal = track_flow_signal(
        _read_frames(video_path, start_frame, end_frame),
        detect_fn or _make_yolo_detector(model_path, config.conf_threshold),
        flow_fn or _make_dis_flow(),
        config,
        on_frame=on_frame,
    )

    return PipelineResult(
        signal=signal,
        positions=compute_positions(signal, config),
        actions=signal_to_actions(signal, fps=fps, config=config,
                                  start_frame=start_frame),
        fps=fps,
        start_frame=start_frame,
        total_frames=total_frames,
    )


def pipeline_result_to_state(result: PipelineResult) -> dict:
    """JSON-serializable form of a PipelineResult for project persistence."""
    sig = result.signal
    return {
        "dy": [round(v, 4) for v in sig.dy.tolist()],
        "lock": list(sig.lock),
        "cuts": list(sig.cuts),
        "rois": [list(r) if r is not None else None for r in sig.rois],
        "detections": {
            str(idx): [
                {"class_name": d.class_name, "confidence": d.confidence,
                 "box": list(d.box)}
                for d in dets
            ]
            for idx, dets in sig.detections.items()
        },
        "positions": [round(v, 2) for v in result.positions.tolist()],
        "actions": result.actions,
        "fps": result.fps,
        "start_frame": result.start_frame,
        "total_frames": result.total_frames,
    }


def pipeline_result_from_state(state: dict) -> PipelineResult:
    """Inverse of pipeline_result_to_state."""
    lock = state.get("lock")
    if lock is None:
        # Sessions saved before lock states existed only recorded activity
        lock = ["anchor" if a else "none" for a in state["roi_active"]]
    signal = TrackSignal(
        dy=np.array(state["dy"]),
        lock=lock,
        cuts=list(state["cuts"]),
        rois=[tuple(r) if r is not None else None for r in state["rois"]],
        detections={
            int(idx): [
                Detection(class_name=d["class_name"],
                          confidence=d["confidence"],
                          box=tuple(d["box"]))
                for d in dets
            ]
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
    )


def generate_funscript(
    video_path: str,
    output_path: str,
    model_path: str = DEFAULT_MODEL_PATH,
    start_frame: int = 0,
    end_frame: int | None = None,
    config: TrackConfig | None = None,
    on_frame: Callable[[int], None] | None = None,
) -> list[dict]:
    """Video in, funscript out.  Returns the action list it wrote."""
    from scripture.funscript import build_funscript, save_funscript

    result = run_pipeline(
        video_path, model_path=model_path, start_frame=start_frame,
        end_frame=end_frame, config=config, on_frame=on_frame)
    funscript = build_funscript(
        result.actions, duration_seconds=int(result.total_frames / result.fps))
    save_funscript(funscript, output_path)
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

    import cv2
    cap = cv2.VideoCapture(args.video)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    n_frames_hint = (args.end_frame or total) - args.start_frame

    actions = generate_funscript(
        args.video, args.output, model_path=args.model,
        start_frame=args.start_frame, end_frame=args.end_frame,
        config=config, on_frame=report,
    )
    elapsed = time.time() - started
    print(f"\n{len(actions)} actions -> {args.output}  ({elapsed:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

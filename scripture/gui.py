"""PyQt6 GUI for scripture: manual scene splitting, axis annotation, and export."""

import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QRect
from PyQt6.QtGui import QIcon, QImage, QPixmap, QPainter, QPen, QBrush, QColor, QShortcut, QKeySequence
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QLabel, QPushButton, QSlider, QFileDialog, QMessageBox, QProgressBar,
)

from shared_ui.colors import (
    BG_PRIMARY, BG_SECONDARY, BG_TERTIARY, BG_BUTTON,
    TEXT_PRIMARY, TEXT_SECONDARY, TEXT_MUTED,
    BLUE, BORDER_SUBTLE,
)
from shared_ui.fonts import SIZE_BODY, SIZE_SMALL, make_font
from shared_ui.spacing import MARGIN_STANDARD, GAP_SMALL, GAP_MEDIUM

from scripture.scene import Scene, scenes_from_splits
from scripture.motion_tracker import AxisDefinition, track_motion
from scripture.stroke_extract import extract_strokes
from scripture.funscript import build_funscript, save_funscript
from scripture.project import save_project, load_project


_BTN_STYLE = f"""
    QPushButton {{
        color: {TEXT_PRIMARY.name()};
        background: {BG_BUTTON.name()};
        border: 1px solid {BORDER_SUBTLE.name()};
        padding: 4px 10px;
        border-radius: 3px;
    }}
    QPushButton:hover {{
        background: {BG_TERTIARY.name()};
    }}
    QPushButton:disabled {{
        color: {TEXT_MUTED.name()};
        background: {BG_SECONDARY.name()};
    }}
"""

_BTN_ACTIVE_STYLE = f"""
    QPushButton {{
        color: {TEXT_PRIMARY.name()};
        background: {BLUE.name()};
        border: 1px solid {BLUE.name()};
        padding: 4px 10px;
        border-radius: 3px;
    }}
    QPushButton:hover {{
        background: {BLUE.name()};
    }}
"""

_SLIDER_STYLE = f"""
    QSlider::groove:horizontal {{
        background: {BG_TERTIARY.name()};
        height: 6px;
        border-radius: 3px;
    }}
    QSlider::handle:horizontal {{
        background: {BLUE.name()};
        width: 14px;
        margin: -4px 0;
        border-radius: 7px;
    }}
"""

_PROGRESS_STYLE = f"""
    QProgressBar {{
        background: {BG_TERTIARY.name()};
        border: 1px solid {BORDER_SUBTLE.name()};
        border-radius: 3px;
        text-align: center;
        color: {TEXT_PRIMARY.name()};
        height: 18px;
    }}
    QProgressBar::chunk {{
        background: {BLUE.name()};
        border-radius: 2px;
    }}
"""

# Timeline scene state colors
_SCENE_EMPTY = QColor(35, 35, 35)       # no annotation
_SCENE_ANNOTATED = QColor(55, 65, 80)   # has tip+base
_SCENE_PROCESSED = QColor(70, 100, 70)  # has been processed
_SCENE_BORDER = QColor(80, 80, 80)
_REP_FRAME_COLOR = QColor(255, 220, 80)


class ProcessWorker(QThread):
    """Runs optical flow processing off the main thread."""
    frame_progress = pyqtSignal(int)
    scene_done = pyqtSignal(int, list)  # scene_idx, actions
    finished = pyqtSignal()
    error = pyqtSignal(int, str)

    def __init__(self, video_path: str, jobs: list[tuple[int, 'Scene', AxisDefinition]], fps: float):
        super().__init__()
        self.video_path = video_path
        self.jobs = jobs
        self.fps = fps

    def run(self):
        offsets: dict[int, int] = {}
        cumulative = 0
        for idx, scene, _axis in self.jobs:
            offsets[idx] = cumulative
            cumulative += scene.end_frame - scene.start_frame

        for idx, scene, axis in self.jobs:
            offset = offsets[idx]
            try:
                result = track_motion(
                    self.video_path, axis, scene.start_frame, scene.end_frame,
                    on_frame=lambda f, _o=offset, _s=scene: self.frame_progress.emit(_o + f - _s.start_frame),
                )
                actions = extract_strokes(result.positions, result.timestamps_ms, fps=self.fps)
                self.scene_done.emit(idx, actions)
            except Exception as e:
                self.error.emit(idx, str(e))
        self.finished.emit()


class TimelineWidget(QWidget):
    """Timeline bar showing scene breaks, annotation state, and representative frames."""

    clicked = pyqtSignal(int)  # frame index

    def __init__(self):
        super().__init__()
        self.setFixedHeight(28)
        self.setMinimumWidth(100)
        self.scenes: list[Scene] = []
        self.scene_axes: dict[int, AxisDefinition] = {}
        self.scene_actions: dict[int, list[dict]] = {}
        self.total_frames: int = 0
        self.current_frame: int = 0

    def set_state(self, scenes, scene_axes, scene_actions, total_frames, current_frame):
        self.scenes = scenes
        self.scene_axes = scene_axes
        self.scene_actions = scene_actions
        self.total_frames = total_frames
        self.current_frame = current_frame
        self.update()

    def paintEvent(self, event):
        if self.total_frames == 0:
            return
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        w, h = self.width(), self.height()

        for i, scene in enumerate(self.scenes):
            x1 = int(scene.start_frame / self.total_frames * w)
            x2 = int(scene.end_frame / self.total_frames * w)

            # Scene state color
            if i in self.scene_actions:
                color = _SCENE_PROCESSED
            elif i in self.scene_axes:
                color = _SCENE_ANNOTATED
            else:
                color = _SCENE_EMPTY

            p.fillRect(x1, 0, x2 - x1, h, color)

            # Scene border (left edge)
            if i > 0:
                p.setPen(QPen(_SCENE_BORDER, 1))
                p.drawLine(x1, 0, x1, h)

            # Representative frame marker
            if i in self.scene_axes:
                rep = self.scene_axes[i].frame
                rx = int(rep / self.total_frames * w)
                p.setPen(QPen(_REP_FRAME_COLOR, 2))
                p.drawLine(rx, 0, rx, h)

        # Current frame cursor
        cx = int(self.current_frame / self.total_frames * w)
        p.setPen(QPen(Qt.GlobalColor.white, 2))
        p.drawLine(cx, 0, cx, h)

        # Outer border
        p.setPen(QPen(_SCENE_BORDER))
        p.drawRect(0, 0, w - 1, h - 1)

        p.end()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self.total_frames > 0:
            frame = int(event.position().x() / self.width() * self.total_frames)
            frame = max(0, min(frame, self.total_frames - 1))
            self.clicked.emit(frame)


class FrameCanvas(QWidget):
    """Displays a video frame scaled to 1/3 canvas, accepts clicks for axis annotation."""

    clicked = pyqtSignal(int, int)  # frame-space x, y

    def __init__(self):
        super().__init__()
        self.setMinimumSize(400, 300)
        self._pixmap: QPixmap | None = None
        self._frame_w: int = 0
        self._frame_h: int = 0
        self._axis: AxisDefinition | None = None
        self._pending_tip: tuple[int, int] | None = None
        self._pending_base: tuple[int, int] | None = None

    def set_frame(self, frame_bgr: np.ndarray):
        h, w = frame_bgr.shape[:2]
        self._frame_w = w
        self._frame_h = h
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888)
        self._pixmap = QPixmap.fromImage(qimg)
        self.update()

    def set_axis(self, axis: AxisDefinition | None):
        self._axis = axis
        self._pending_tip = None
        self._pending_base = None
        self.update()

    def set_pending_tip(self, tip: tuple[int, int] | None):
        self._pending_tip = tip
        self.update()

    def set_pending_base(self, base: tuple[int, int] | None):
        self._pending_base = base
        self.update()

    def clear(self):
        self._pixmap = None
        self._axis = None
        self._pending_tip = None
        self._pending_base = None
        self.update()

    def _display_scale(self) -> float:
        if self._frame_w == 0 or self._frame_h == 0:
            return 1.0
        return min(
            self.width() / (3 * self._frame_w),
            self.height() / (3 * self._frame_h),
        )

    def _frame_to_canvas(self, fx: int, fy: int) -> tuple[int, int]:
        scale = self._display_scale()
        disp_w = int(self._frame_w * scale)
        disp_h = int(self._frame_h * scale)
        ox = (self.width() - disp_w) // 2
        oy = (self.height() - disp_h) // 2
        return int(fx * scale) + ox, int(fy * scale) + oy

    def _canvas_to_frame(self, cx: int, cy: int) -> tuple[int, int]:
        scale = self._display_scale()
        disp_w = int(self._frame_w * scale)
        disp_h = int(self._frame_h * scale)
        ox = (self.width() - disp_w) // 2
        oy = (self.height() - disp_h) // 2
        return int((cx - ox) / scale), int((cy - oy) / scale)

    def paintEvent(self, event):
        p = QPainter(self)
        p.fillRect(self.rect(), QBrush(BG_PRIMARY))

        if self._pixmap is None:
            p.setPen(QPen(TEXT_MUTED))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Open a video to begin")
            p.end()
            return

        scale = self._display_scale()
        disp_w = int(self._frame_w * scale)
        disp_h = int(self._frame_h * scale)
        ox = (self.width() - disp_w) // 2
        oy = (self.height() - disp_h) // 2

        p.setPen(QPen(BORDER_SUBTLE))
        p.drawRect(ox - 1, oy - 1, disp_w + 1, disp_h + 1)

        scaled = self._pixmap.scaled(
            disp_w, disp_h,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        p.drawPixmap(ox, oy, scaled)

        if self._axis is not None:
            self._draw_axis(p, self._axis)
        else:
            if self._pending_tip is not None:
                self._draw_point(p, self._pending_tip, QColor(255, 80, 80), "TIP")
            if self._pending_base is not None:
                self._draw_point(p, self._pending_base, QColor(80, 140, 255), "BASE")

        p.end()

    def _draw_point(self, p: QPainter, point: tuple[int, int], color: QColor, label: str):
        cx, cy = self._frame_to_canvas(*point)
        p.setPen(QPen(Qt.GlobalColor.white, 1))
        p.setBrush(QBrush(color))
        p.drawEllipse(cx - 5, cy - 5, 10, 10)
        p.setPen(QPen(color))
        p.setFont(make_font(size=SIZE_SMALL))
        p.drawText(cx + 10, cy + 4, label)

    def _draw_axis(self, p: QPainter, axis: AxisDefinition):
        tx, ty = self._frame_to_canvas(*axis.tip)
        bx, by = self._frame_to_canvas(*axis.base)

        pen = QPen(QColor(255, 220, 80), 2, Qt.PenStyle.DashLine)
        p.setPen(pen)
        p.drawLine(tx, ty, bx, by)

        self._draw_point(p, axis.tip, QColor(255, 80, 80), "TIP")
        self._draw_point(p, axis.base, QColor(80, 140, 255), "BASE")

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self._frame_w > 0:
            fx, fy = self._canvas_to_frame(int(event.position().x()), int(event.position().y()))
            self.clicked.emit(fx, fy)


class App(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("scripture")
        self.resize(1200, 800)
        _icon_path = Path(__file__).resolve().parent.parent / "icon.ico"
        if _icon_path.exists():
            self.setWindowIcon(QIcon(str(_icon_path)))
        self.setStyleSheet(f"background: {BG_PRIMARY.name()}; color: {TEXT_PRIMARY.name()};")

        self.video_path: str | None = None
        self.cap: cv2.VideoCapture | None = None
        self.fps: float = 30.0
        self.total_frames: int = 0
        self.frame_w: int = 0
        self.frame_h: int = 0

        self.splits: list[int] = []
        self.scenes: list[Scene] = []
        self.scene_axes: dict[int, AxisDefinition] = {}
        self.scene_actions: dict[int, list[dict]] = {}

        self.current_frame_idx: int = 0
        self.placing: str | None = None  # None, "tip", or "base"
        self.pending_tip: tuple[int, int] | None = None
        self.pending_base: tuple[int, int] | None = None

        self._worker: ProcessWorker | None = None
        self._project_path: str | None = None  # for Save overwrite

        self._build_ui()
        self._build_shortcuts()

    def _build_shortcuts(self):
        QShortcut(QKeySequence("B"), self, self._toggle_base)
        QShortcut(QKeySequence("T"), self, self._toggle_tip)

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(MARGIN_STANDARD, MARGIN_STANDARD, MARGIN_STANDARD, MARGIN_STANDARD)
        root.setSpacing(GAP_MEDIUM)

        # --- Row 1: file + scene splitting ---
        row1 = QHBoxLayout()
        row1.setSpacing(GAP_SMALL)

        self.btn_open = QPushButton("Open Video")
        self.btn_open.setStyleSheet(_BTN_STYLE)
        self.btn_open.clicked.connect(self._open_video)
        row1.addWidget(self.btn_open)

        self.btn_save = QPushButton("Save")
        self.btn_save.setStyleSheet(_BTN_STYLE)
        self.btn_save.clicked.connect(self._save_project)
        row1.addWidget(self.btn_save)

        self.btn_save_as = QPushButton("Save As")
        self.btn_save_as.setStyleSheet(_BTN_STYLE)
        self.btn_save_as.clicked.connect(self._save_project_as)
        row1.addWidget(self.btn_save_as)

        self.btn_load = QPushButton("Load")
        self.btn_load.setStyleSheet(_BTN_STYLE)
        self.btn_load.clicked.connect(self._load_project)
        row1.addWidget(self.btn_load)

        row1.addSpacing(12)

        self.btn_split = QPushButton("Split Here")
        self.btn_split.setStyleSheet(_BTN_STYLE)
        self.btn_split.clicked.connect(self._split_here)
        row1.addWidget(self.btn_split)

        self.btn_merge = QPushButton("Merge \u2190")
        self.btn_merge.setStyleSheet(_BTN_STYLE)
        self.btn_merge.setToolTip("Remove split at start of current scene (merge with previous)")
        self.btn_merge.clicked.connect(self._merge_left)
        row1.addWidget(self.btn_merge)

        row1.addStretch()
        root.addLayout(row1)

        # --- Row 2: nav + annotation + processing ---
        row2 = QHBoxLayout()
        row2.setSpacing(GAP_SMALL)

        self.btn_prev = QPushButton("< Prev")
        self.btn_prev.setFixedWidth(70)
        self.btn_prev.setStyleSheet(_BTN_STYLE)
        self.btn_prev.clicked.connect(self._prev_scene)
        row2.addWidget(self.btn_prev)

        self.btn_next = QPushButton("Next >")
        self.btn_next.setFixedWidth(70)
        self.btn_next.setStyleSheet(_BTN_STYLE)
        self.btn_next.clicked.connect(self._next_scene)
        row2.addWidget(self.btn_next)

        self.scene_label = QLabel("Scene: -/-")
        self.scene_label.setMinimumWidth(200)
        self.scene_label.setFont(make_font(size=SIZE_BODY))
        self.scene_label.setStyleSheet(f"color: {TEXT_SECONDARY.name()};")
        row2.addWidget(self.scene_label)

        row2.addSpacing(12)

        # Tip/Base toggle buttons
        self.btn_tip = QPushButton("Tip (T)")
        self.btn_tip.setFixedWidth(80)
        self.btn_tip.setStyleSheet(_BTN_STYLE)
        self.btn_tip.clicked.connect(self._toggle_tip)
        row2.addWidget(self.btn_tip)

        self.btn_base = QPushButton("Base (B)")
        self.btn_base.setFixedWidth(80)
        self.btn_base.setStyleSheet(_BTN_STYLE)
        self.btn_base.clicked.connect(self._toggle_base)
        row2.addWidget(self.btn_base)

        self.btn_clear_axis = QPushButton("Clear Axis")
        self.btn_clear_axis.setStyleSheet(_BTN_STYLE)
        self.btn_clear_axis.clicked.connect(self._clear_axis)
        row2.addWidget(self.btn_clear_axis)

        row2.addStretch()

        self.btn_process = QPushButton("Process Scene")
        self.btn_process.setStyleSheet(_BTN_STYLE)
        self.btn_process.clicked.connect(self._process_scene)
        row2.addWidget(self.btn_process)

        self.btn_discard = QPushButton("Discard")
        self.btn_discard.setStyleSheet(_BTN_STYLE)
        self.btn_discard.setToolTip("Discard processed actions for current scene")
        self.btn_discard.clicked.connect(self._discard_scene)
        row2.addWidget(self.btn_discard)

        self.btn_process_all = QPushButton("Process All")
        self.btn_process_all.setStyleSheet(_BTN_STYLE)
        self.btn_process_all.clicked.connect(self._process_all)
        row2.addWidget(self.btn_process_all)

        self.btn_export = QPushButton("Export .funscript")
        self.btn_export.setStyleSheet(_BTN_STYLE)
        self.btn_export.clicked.connect(self._export)
        row2.addWidget(self.btn_export)

        root.addLayout(row2)

        # --- Progress bar ---
        self.progress_bar = QProgressBar()
        self.progress_bar.setStyleSheet(_PROGRESS_STYLE)
        self.progress_bar.setVisible(False)
        root.addWidget(self.progress_bar)

        # --- Frame canvas ---
        self.canvas = FrameCanvas()
        self.canvas.clicked.connect(self._on_canvas_click)
        root.addWidget(self.canvas, stretch=1)

        # --- Timeline ---
        self.timeline = TimelineWidget()
        self.timeline.clicked.connect(self._on_timeline_click)
        root.addWidget(self.timeline)

        # --- Scrubber row ---
        scrub_row = QHBoxLayout()
        scrub_row.setSpacing(GAP_SMALL)

        scrub_label = QLabel("Frame:")
        scrub_label.setFont(make_font(size=SIZE_SMALL))
        scrub_label.setStyleSheet(f"color: {TEXT_SECONDARY.name()};")
        scrub_row.addWidget(scrub_label)

        self.frame_scrubber = QSlider(Qt.Orientation.Horizontal)
        self.frame_scrubber.setStyleSheet(_SLIDER_STYLE)
        self.frame_scrubber.setMinimum(0)
        self.frame_scrubber.setMaximum(10000)
        self.frame_scrubber.valueChanged.connect(self._on_scrub)
        scrub_row.addWidget(self.frame_scrubber, stretch=1)

        self.frame_info_label = QLabel("")
        self.frame_info_label.setFixedWidth(150)
        self.frame_info_label.setFont(make_font(size=SIZE_SMALL))
        self.frame_info_label.setStyleSheet(f"color: {TEXT_SECONDARY.name()};")
        scrub_row.addWidget(self.frame_info_label)

        root.addLayout(scrub_row)

        # --- Status bar ---
        self.status = QLabel("Open a video to begin.")
        self.status.setFont(make_font(size=SIZE_SMALL))
        self.status.setStyleSheet(
            f"color: {TEXT_MUTED.name()}; background: {BG_SECONDARY.name()}; "
            f"padding: 3px 6px; border-radius: 2px;"
        )
        root.addWidget(self.status)

    def _set_status(self, text: str):
        self.status.setText(text)

    def _format_time(self, frame_idx: int) -> str:
        seconds = frame_idx / self.fps if self.fps else 0
        m, s = divmod(int(seconds), 60)
        h, m = divmod(m, 60)
        if h:
            return f"{h}:{m:02d}:{s:02d}"
        return f"{m}:{s:02d}"

    def _current_scene_idx(self) -> int:
        for i, scene in enumerate(self.scenes):
            if self.current_frame_idx < scene.end_frame:
                return i
        return max(0, len(self.scenes) - 1)

    def _rebuild_scenes(self, clear_annotations: bool = True):
        self.scenes = scenes_from_splits(self.splits, self.total_frames)
        if clear_annotations:
            self.scene_axes.clear()
            self.scene_actions.clear()

    def _update_timeline(self):
        self.timeline.set_state(
            self.scenes, self.scene_axes, self.scene_actions,
            self.total_frames, self.current_frame_idx,
        )

    def _update_annotation_buttons(self):
        """Update tip/base button styles based on current placing mode."""
        idx = self._current_scene_idx()
        has_axis = idx in self.scene_axes

        if self.placing == "tip":
            self.btn_tip.setStyleSheet(_BTN_ACTIVE_STYLE)
            self.btn_tip.setText("Cancel Tip")
        elif has_axis or self.pending_tip is not None:
            self.btn_tip.setStyleSheet(_BTN_STYLE)
            self.btn_tip.setText("Move Tip (T)")
        else:
            self.btn_tip.setStyleSheet(_BTN_STYLE)
            self.btn_tip.setText("Tip (T)")

        if self.placing == "base":
            self.btn_base.setStyleSheet(_BTN_ACTIVE_STYLE)
            self.btn_base.setText("Cancel Base")
        elif has_axis or self.pending_base is not None:
            self.btn_base.setStyleSheet(_BTN_STYLE)
            self.btn_base.setText("Move Base (B)")
        else:
            self.btn_base.setStyleSheet(_BTN_STYLE)
            self.btn_base.setText("Base (B)")

    # ── Video loading ──────────────────────────────────────────────────

    def _open_video(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Video", "",
            "Video files (*.mp4 *.mkv *.avi *.webm *.mov);;All files (*)",
        )
        if not path:
            return

        self.video_path = path
        if self.cap is not None:
            self.cap.release()
        self.cap = cv2.VideoCapture(path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.splits = []
        self.scenes = [Scene(0, self.total_frames)]
        self.scene_axes.clear()
        self.scene_actions.clear()
        self.current_frame_idx = 0
        self._project_path = None

        duration = self._format_time(self.total_frames)
        self._set_status(
            f"Loaded: {Path(path).name} \u2014 {self.total_frames} frames @ "
            f"{self.fps:.1f} fps \u2014 {duration} \u2014 {self.frame_w}x{self.frame_h}"
        )
        self._show_frame(0)
        self._update_scene_label()
        self._update_timeline()

    # ── Frame display ──────────────────────────────────────────────────

    def _show_frame(self, frame_idx: int):
        if self.cap is None:
            return
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self.cap.read()
        if not ret:
            return

        self.current_frame_idx = frame_idx
        self.canvas._frame_w = self.frame_w
        self.canvas._frame_h = self.frame_h
        self.canvas.set_frame(frame)

        idx = self._current_scene_idx()
        # Only show axis on the representative frame
        if idx in self.scene_axes and self.scene_axes[idx].frame == frame_idx:
            self.canvas.set_axis(self.scene_axes[idx])
        else:
            self.canvas.set_axis(None)
            # Show pending points only if they were placed on this frame
            self.canvas.set_pending_tip(self.pending_tip if self.pending_tip is not None else None)
            self.canvas.set_pending_base(self.pending_base if self.pending_base is not None else None)

        self.frame_info_label.setText(f"#{frame_idx} / {self._format_time(frame_idx)}")
        self._update_scene_label()
        self._update_timeline()
        self._update_annotation_buttons()

    def _update_scene_label(self):
        if not self.scenes:
            self.scene_label.setText("Scene: -/-")
            return
        idx = self._current_scene_idx()
        scene = self.scenes[idx]
        start_t = self._format_time(scene.start_frame)
        end_t = self._format_time(scene.end_frame)
        self.scene_label.setText(
            f"Scene {idx + 1}/{len(self.scenes)}  "
            f"[{start_t} \u2013 {end_t}]  "
            f"frames {scene.start_frame}\u2013{scene.end_frame}"
        )

    # ── Scene splitting ────────────────────────────────────────────────

    def _split_here(self):
        if self.total_frames == 0:
            return
        frame = self.current_frame_idx
        if frame <= 0 or frame >= self.total_frames:
            self._set_status("Cannot split at start or end of video.")
            return
        if frame in self.splits:
            self._set_status(f"Split already exists at frame {frame}.")
            return

        self.splits.append(frame)
        self._rebuild_scenes()
        self._cancel_placing()
        self._update_scene_label()
        self._update_timeline()
        self._set_status(f"Split at frame {frame} ({self._format_time(frame)}). {len(self.scenes)} scenes.")

    def _merge_left(self):
        idx = self._current_scene_idx()
        if idx == 0:
            self._set_status("First scene \u2014 nothing to merge with.")
            return
        split_frame = self.scenes[idx].start_frame
        if split_frame in self.splits:
            self.splits.remove(split_frame)
            self._rebuild_scenes()
            self._cancel_placing()
            self._update_scene_label()
            self._update_timeline()
            self._set_status(f"Removed split at frame {split_frame}. {len(self.scenes)} scenes.")

    # ── Scene navigation ───────────────────────────────────────────────

    def _navigate_to_scene(self, scene_idx: int):
        if scene_idx in self.scene_axes:
            target = self.scene_axes[scene_idx].frame
        else:
            target = self.scenes[scene_idx].start_frame
        self._cancel_placing()
        self.frame_scrubber.blockSignals(True)
        self.frame_scrubber.setValue(int(target / self.total_frames * 10000))
        self.frame_scrubber.blockSignals(False)
        self._show_frame(target)

    def _prev_scene(self):
        idx = self._current_scene_idx()
        if idx > 0:
            self._navigate_to_scene(idx - 1)

    def _next_scene(self):
        idx = self._current_scene_idx()
        if idx < len(self.scenes) - 1:
            self._navigate_to_scene(idx + 1)

    def _on_scrub(self, value: int):
        if self.total_frames == 0:
            return
        frame_idx = int(value / 10000 * self.total_frames)
        frame_idx = min(frame_idx, self.total_frames - 1)
        self._show_frame(frame_idx)

    def _on_timeline_click(self, frame: int):
        self.frame_scrubber.blockSignals(True)
        self.frame_scrubber.setValue(int(frame / self.total_frames * 10000))
        self.frame_scrubber.blockSignals(False)
        self._show_frame(frame)

    # ── Axis annotation ────────────────────────────────────────────────

    def _cancel_placing(self):
        self.placing = None
        self.pending_tip = None
        self.pending_base = None
        self._update_annotation_buttons()

    def _toggle_tip(self):
        if self.placing == "tip":
            self.placing = None
        else:
            self.placing = "tip"
        self._update_annotation_buttons()

    def _toggle_base(self):
        if self.placing == "base":
            self.placing = None
        else:
            self.placing = "base"
        self._update_annotation_buttons()

    def _clear_axis(self):
        idx = self._current_scene_idx()
        if idx in self.scene_axes:
            del self.scene_axes[idx]
        if idx in self.scene_actions:
            del self.scene_actions[idx]
        self.pending_tip = None
        self.pending_base = None
        self.placing = None
        self._show_frame(self.current_frame_idx)
        self._set_status(f"Cleared axis for scene {idx + 1}")

    def _on_canvas_click(self, frame_x: int, frame_y: int):
        if not self.scenes or self.placing is None:
            return

        idx = self._current_scene_idx()

        # Check if changing representative frame
        if idx in self.scene_axes and self.scene_axes[idx].frame != self.current_frame_idx:
            reply = QMessageBox.question(
                self, "Change representative frame?",
                "Are you sure you want to change the representative frame for this scene?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return
            # Moving to new frame — start fresh
            old_axis = self.scene_axes[idx]
            if self.placing == "tip":
                self.pending_base = old_axis.base
            else:
                self.pending_tip = old_axis.tip
            del self.scene_axes[idx]

        if self.placing == "tip":
            self.pending_tip = (frame_x, frame_y)
            self.placing = None
            self.canvas.set_pending_tip(self.pending_tip)
        elif self.placing == "base":
            self.pending_base = (frame_x, frame_y)
            self.placing = None
            self.canvas.set_pending_base(self.pending_base)

        # If both are placed, create the axis
        if self.pending_tip is not None and self.pending_base is not None:
            axis = AxisDefinition(
                tip=self.pending_tip, base=self.pending_base,
                frame=self.current_frame_idx,
            )
            self.scene_axes[idx] = axis
            self.pending_tip = None
            self.pending_base = None
            self.canvas.set_axis(axis)
            self._set_status(f"Axis set for scene {idx + 1}")

        self._update_annotation_buttons()
        self._update_timeline()

    # ── Processing ─────────────────────────────────────────────────────

    def _build_jobs(self, scene_indices: list[int]) -> list[tuple[int, Scene, AxisDefinition]]:
        return [
            (idx, self.scenes[idx], self.scene_axes[idx])
            for idx in scene_indices
        ]

    @staticmethod
    def _fmt_duration(seconds: float) -> str:
        s = int(seconds)
        if s < 60:
            return f"{s}s"
        m, s = divmod(s, 60)
        if m < 60:
            return f"{m}m {s:02d}s"
        h, m = divmod(m, 60)
        return f"{h}h {m:02d}m {s:02d}s"

    def _start_processing(self, jobs: list[tuple[int, Scene, AxisDefinition]]):
        total_frames = sum(scene.end_frame - scene.start_frame for _, scene, _ in jobs)
        self.progress_bar.setVisible(True)
        self.progress_bar.setMaximum(total_frames)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat(f"0 / {total_frames} frames (0%)")
        self.btn_process.setEnabled(False)
        self.btn_process_all.setEnabled(False)

        self._process_total_frames = total_frames
        self._process_start_time = time.monotonic()
        self._worker = ProcessWorker(self.video_path, jobs, self.fps)
        self._worker.frame_progress.connect(self._on_frame_progress)
        self._worker.scene_done.connect(self._on_scene_done)
        self._worker.error.connect(self._on_process_error)
        self._worker.finished.connect(self._on_process_finished)
        self._worker.start()

    def _on_frame_progress(self, frames_done: int):
        self.progress_bar.setValue(frames_done)
        elapsed = time.monotonic() - self._process_start_time
        elapsed_str = self._fmt_duration(elapsed)

        if frames_done > 0:
            eta = elapsed / frames_done * (self._process_total_frames - frames_done)
            eta_str = self._fmt_duration(eta)
            self.progress_bar.setFormat(
                f"{frames_done} / {self._process_total_frames} frames (%p%)  "
                f"\u2014 {elapsed_str} elapsed, ~{eta_str} remaining"
            )
        else:
            self.progress_bar.setFormat(
                f"0 / {self._process_total_frames} frames (0%)  \u2014 {elapsed_str} elapsed"
            )

    def _on_scene_done(self, scene_idx: int, actions: list[dict]):
        self.scene_actions[scene_idx] = actions
        self._update_timeline()

    def _on_process_error(self, scene_idx: int, msg: str):
        self._set_status(f"Error processing scene {scene_idx + 1}: {msg}")

    def _on_process_finished(self):
        elapsed = time.monotonic() - self._process_start_time
        elapsed_str = self._fmt_duration(elapsed)

        self.progress_bar.setVisible(False)
        self.btn_process.setEnabled(True)
        self.btn_process_all.setEnabled(True)

        total_actions = sum(len(a) for a in self.scene_actions.values())
        self._set_status(
            f"Done \u2014 {len(self.scene_actions)} scenes, {total_actions} actions "
            f"in {elapsed_str}."
        )
        self._update_timeline()

    def _process_scene(self):
        idx = self._current_scene_idx()
        if idx not in self.scene_axes:
            QMessageBox.warning(self, "No axis", "Define tip and base for this scene first.")
            return
        self._start_processing(self._build_jobs([idx]))

    def _process_all(self):
        annotated = [i for i in range(len(self.scenes)) if i in self.scene_axes]
        if not annotated:
            QMessageBox.warning(self, "No axes", "Annotate tip/base on at least one scene first.")
            return
        self._start_processing(self._build_jobs(annotated))

    def _discard_scene(self):
        idx = self._current_scene_idx()
        if idx in self.scene_actions:
            del self.scene_actions[idx]
            self._update_timeline()
            self._set_status(f"Discarded actions for scene {idx + 1}.")
        else:
            self._set_status(f"Scene {idx + 1} has no processed actions.")

    # ── Project save/load ────────────────────────────────────────────────

    def _build_state(self) -> dict:
        axes_data = {}
        for idx, axis in self.scene_axes.items():
            axes_data[str(idx)] = {
                "tip": list(axis.tip),
                "base": list(axis.base),
                "frame": axis.frame,
            }
        actions_data = {}
        for idx, actions in self.scene_actions.items():
            actions_data[str(idx)] = actions

        return {
            "video_path": self.video_path,
            "splits": self.splits,
            "axes": axes_data,
            "actions": actions_data,
        }

    def _do_save(self, path: str):
        save_project(path, self._build_state())
        self._project_path = path
        self._set_status(f"Saved to {Path(path).name}")

    def _save_project(self):
        if self.video_path is None:
            QMessageBox.warning(self, "No video", "Open a video first.")
            return
        if self._project_path is not None:
            self._do_save(self._project_path)
        else:
            self._save_project_as()

    def _save_project_as(self):
        if self.video_path is None:
            QMessageBox.warning(self, "No video", "Open a video first.")
            return
        default_name = Path(self.video_path).stem + ".scripture"
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Project As", default_name,
            "Scripture project (*.scripture);;All files (*)",
        )
        if not path:
            return
        self._do_save(path)

    def _load_project(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Project", "",
            "Scripture project (*.scripture);;All files (*)",
        )
        if not path:
            return

        state = load_project(path)
        video_path = state["video_path"]

        if self.cap is not None:
            self.cap.release()
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            QMessageBox.critical(self, "Video not found", f"Cannot open: {video_path}")
            return

        self.video_path = video_path
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.splits = state["splits"]
        self._rebuild_scenes(clear_annotations=False)

        self.scene_axes.clear()
        for idx_str, axis_data in state.get("axes", {}).items():
            self.scene_axes[int(idx_str)] = AxisDefinition(
                tip=tuple(axis_data["tip"]),
                base=tuple(axis_data["base"]),
                frame=axis_data.get("frame", 0),
            )

        self.scene_actions.clear()
        for idx_str, actions in state.get("actions", {}).items():
            self.scene_actions[int(idx_str)] = actions

        self._project_path = path
        self._cancel_placing()
        self.current_frame_idx = 0
        self._show_frame(0)
        self._update_scene_label()
        self._update_timeline()

        n_axes = len(self.scene_axes)
        n_processed = len(self.scene_actions)
        self._set_status(
            f"Loaded: {Path(path).name} \u2014 "
            f"{len(self.scenes)} scenes, {n_axes} annotated, {n_processed} processed"
        )

    # ── Export ─────────────────────────────────────────────────────────

    def _export(self):
        if not self.scene_actions:
            QMessageBox.warning(self, "No data", "Process at least one scene first.")
            return

        all_actions = []
        for actions in self.scene_actions.values():
            all_actions.extend(actions)

        duration_s = int(self.total_frames / self.fps)
        funscript = build_funscript(all_actions, duration_s)

        default_name = Path(self.video_path).stem + ".funscript" if self.video_path else "output.funscript"
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Funscript", default_name,
            "Funscript (*.funscript);;All files (*)",
        )
        if not path:
            return

        save_funscript(funscript, path)
        self._set_status(f"Exported {len(all_actions)} actions to {Path(path).name}")

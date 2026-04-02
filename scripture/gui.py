"""PyQt6 GUI for scripture: manual scene splitting, axis annotation, and export."""

import sys
from pathlib import Path

import cv2
import numpy as np
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap, QPainter, QPen, QBrush, QColor
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QLabel, QPushButton, QSlider, QFileDialog, QMessageBox,
)

from shared_ui.colors import (
    BG_PRIMARY, BG_SECONDARY, BG_TERTIARY, BG_BUTTON,
    TEXT_PRIMARY, TEXT_SECONDARY, TEXT_MUTED,
    ACCENT_PRIMARY, BORDER_SUBTLE,
)
from shared_ui.fonts import SIZE_BODY, SIZE_SMALL, make_font
from shared_ui.spacing import MARGIN_STANDARD, GAP_SMALL, GAP_MEDIUM

from scripture.scene import Scene, scenes_from_splits
from scripture.motion_tracker import AxisDefinition, track_motion
from scripture.stroke_extract import extract_strokes
from scripture.funscript import build_funscript, save_funscript


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

_SLIDER_STYLE = f"""
    QSlider::groove:horizontal {{
        background: {BG_TERTIARY.name()};
        height: 6px;
        border-radius: 3px;
    }}
    QSlider::handle:horizontal {{
        background: {ACCENT_PRIMARY.name()};
        width: 14px;
        margin: -4px 0;
        border-radius: 7px;
    }}
"""


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
        self.update()

    def set_pending_tip(self, tip: tuple[int, int] | None):
        self._pending_tip = tip
        self.update()

    def clear(self):
        self._pixmap = None
        self._axis = None
        self._pending_tip = None
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
        elif self._pending_tip is not None:
            tx, ty = self._frame_to_canvas(*self._pending_tip)
            p.setPen(QPen(QColor(255, 80, 80), 2))
            p.setBrush(QBrush(QColor(255, 80, 80)))
            p.drawEllipse(tx - 5, ty - 5, 10, 10)
            p.setPen(QPen(QColor(255, 80, 80)))
            p.setFont(make_font(size=SIZE_SMALL))
            p.drawText(tx + 10, ty + 4, "TIP")

        p.end()

    def _draw_axis(self, p: QPainter, axis: AxisDefinition):
        tx, ty = self._frame_to_canvas(*axis.tip)
        bx, by = self._frame_to_canvas(*axis.base)

        pen = QPen(QColor(255, 220, 80), 2, Qt.PenStyle.DashLine)
        p.setPen(pen)
        p.drawLine(tx, ty, bx, by)

        p.setPen(QPen(Qt.GlobalColor.white, 1))
        p.setBrush(QBrush(QColor(255, 80, 80)))
        p.drawEllipse(tx - 5, ty - 5, 10, 10)
        p.setPen(QPen(QColor(255, 80, 80)))
        p.setFont(make_font(size=SIZE_SMALL))
        p.drawText(tx + 10, ty + 4, "TIP")

        p.setPen(QPen(Qt.GlobalColor.white, 1))
        p.setBrush(QBrush(QColor(80, 140, 255)))
        p.drawEllipse(bx - 5, by - 5, 10, 10)
        p.setPen(QPen(QColor(80, 140, 255)))
        p.drawText(bx + 10, by + 4, "BASE")

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self._frame_w > 0:
            fx, fy = self._canvas_to_frame(int(event.position().x()), int(event.position().y()))
            self.clicked.emit(fx, fy)


class App(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("scripture")
        self.resize(1200, 800)
        self.setStyleSheet(f"background: {BG_PRIMARY.name()}; color: {TEXT_PRIMARY.name()};")

        self.video_path: str | None = None
        self.cap: cv2.VideoCapture | None = None
        self.fps: float = 30.0
        self.total_frames: int = 0
        self.frame_w: int = 0
        self.frame_h: int = 0

        # Manual scene splitting
        self.splits: list[int] = []
        self.scenes: list[Scene] = []
        self.scene_axes: dict[int, AxisDefinition] = {}
        self.scene_actions: dict[int, list[dict]] = {}

        self.current_frame_idx: int = 0
        self.click_state: str = "tip"
        self.current_tip: tuple[int, int] | None = None

        self._build_ui()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(MARGIN_STANDARD, MARGIN_STANDARD, MARGIN_STANDARD, MARGIN_STANDARD)
        root.setSpacing(GAP_MEDIUM)

        # --- Toolbar ---
        toolbar = QHBoxLayout()
        toolbar.setSpacing(GAP_SMALL)

        self.btn_open = QPushButton("Open Video")
        self.btn_open.setStyleSheet(_BTN_STYLE)
        self.btn_open.clicked.connect(self._open_video)
        toolbar.addWidget(self.btn_open)

        toolbar.addSpacing(12)

        self.btn_split = QPushButton("Split Here")
        self.btn_split.setStyleSheet(_BTN_STYLE)
        self.btn_split.clicked.connect(self._split_here)
        toolbar.addWidget(self.btn_split)

        self.btn_merge = QPushButton("Merge \u2190")
        self.btn_merge.setStyleSheet(_BTN_STYLE)
        self.btn_merge.setToolTip("Remove split at start of current scene (merge with previous)")
        self.btn_merge.clicked.connect(self._merge_left)
        toolbar.addWidget(self.btn_merge)

        toolbar.addSpacing(12)

        self.btn_prev = QPushButton("< Prev")
        self.btn_prev.setFixedWidth(70)
        self.btn_prev.setStyleSheet(_BTN_STYLE)
        self.btn_prev.clicked.connect(self._prev_scene)
        toolbar.addWidget(self.btn_prev)

        self.scene_label = QLabel("Scene: -/-")
        self.scene_label.setFixedWidth(360)
        self.scene_label.setFont(make_font(size=SIZE_BODY))
        self.scene_label.setStyleSheet(f"color: {TEXT_SECONDARY.name()};")
        toolbar.addWidget(self.scene_label)

        self.btn_next = QPushButton("Next >")
        self.btn_next.setFixedWidth(70)
        self.btn_next.setStyleSheet(_BTN_STYLE)
        self.btn_next.clicked.connect(self._next_scene)
        toolbar.addWidget(self.btn_next)

        toolbar.addSpacing(12)

        self.click_label = QLabel("Click: TIP")
        self.click_label.setFixedWidth(90)
        self.click_label.setFont(make_font(size=SIZE_BODY, bold=True))
        self.click_label.setStyleSheet("color: #ff5050;")
        toolbar.addWidget(self.click_label)

        toolbar.addStretch()

        self.btn_process = QPushButton("Process Scene")
        self.btn_process.setStyleSheet(_BTN_STYLE)
        self.btn_process.clicked.connect(self._process_scene)
        toolbar.addWidget(self.btn_process)

        self.btn_process_all = QPushButton("Process All")
        self.btn_process_all.setStyleSheet(_BTN_STYLE)
        self.btn_process_all.clicked.connect(self._process_all)
        toolbar.addWidget(self.btn_process_all)

        self.btn_export = QPushButton("Export .funscript")
        self.btn_export.setStyleSheet(_BTN_STYLE)
        self.btn_export.clicked.connect(self._export)
        toolbar.addWidget(self.btn_export)

        root.addLayout(toolbar)

        # --- Frame canvas ---
        self.canvas = FrameCanvas()
        self.canvas.clicked.connect(self._on_canvas_click)
        root.addWidget(self.canvas, stretch=1)

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
        """Return the index of the scene containing the current frame."""
        for i, scene in enumerate(self.scenes):
            if self.current_frame_idx < scene.end_frame:
                return i
        return max(0, len(self.scenes) - 1)

    def _rebuild_scenes(self):
        """Rebuild scene list from splits. Clears axes and actions."""
        self.scenes = scenes_from_splits(self.splits, self.total_frames)
        self.scene_axes.clear()
        self.scene_actions.clear()

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

        duration = self._format_time(self.total_frames)
        self._set_status(
            f"Loaded: {Path(path).name} — {self.total_frames} frames @ "
            f"{self.fps:.1f} fps — {duration} — {self.frame_w}x{self.frame_h}"
        )
        self._show_frame(0)
        self._update_scene_label()

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
        if idx in self.scene_axes:
            self.canvas.set_axis(self.scene_axes[idx])
        else:
            self.canvas.set_axis(None)
            self.canvas.set_pending_tip(self.current_tip)

        self.frame_info_label.setText(f"#{frame_idx} / {self._format_time(frame_idx)}")
        self._update_scene_label()

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
        self._reset_click_state()
        self._update_scene_label()
        self._set_status(f"Split at frame {frame} ({self._format_time(frame)}). {len(self.scenes)} scenes.")

    def _merge_left(self):
        idx = self._current_scene_idx()
        if idx == 0:
            self._set_status("First scene — nothing to merge with.")
            return

        # The split at the start of the current scene
        split_frame = self.scenes[idx].start_frame
        if split_frame in self.splits:
            self.splits.remove(split_frame)
            self._rebuild_scenes()
            self._reset_click_state()
            self._update_scene_label()
            self._set_status(f"Removed split at frame {split_frame}. {len(self.scenes)} scenes.")

    # ── Scene navigation ───────────────────────────────────────────────

    def _prev_scene(self):
        idx = self._current_scene_idx()
        if idx > 0:
            target = self.scenes[idx - 1].start_frame
            self._reset_click_state()
            self.frame_scrubber.blockSignals(True)
            self.frame_scrubber.setValue(int(target / self.total_frames * 10000))
            self.frame_scrubber.blockSignals(False)
            self._show_frame(target)

    def _next_scene(self):
        idx = self._current_scene_idx()
        if idx < len(self.scenes) - 1:
            target = self.scenes[idx + 1].start_frame
            self._reset_click_state()
            self.frame_scrubber.blockSignals(True)
            self.frame_scrubber.setValue(int(target / self.total_frames * 10000))
            self.frame_scrubber.blockSignals(False)
            self._show_frame(target)

    def _on_scrub(self, value: int):
        if self.total_frames == 0:
            return
        frame_idx = int(value / 10000 * self.total_frames)
        frame_idx = min(frame_idx, self.total_frames - 1)
        self._show_frame(frame_idx)

    # ── Axis annotation ────────────────────────────────────────────────

    def _reset_click_state(self):
        self.click_state = "tip"
        self.click_label.setText("Click: TIP")
        self.click_label.setStyleSheet("color: #ff5050;")
        self.current_tip = None

    def _on_canvas_click(self, frame_x: int, frame_y: int):
        if not self.scenes:
            return

        idx = self._current_scene_idx()

        if self.click_state == "tip":
            self.current_tip = (frame_x, frame_y)
            self.click_state = "base"
            self.click_label.setText("Click: BASE")
            self.click_label.setStyleSheet("color: #508cff;")
            self.canvas.set_pending_tip(self.current_tip)
        else:
            axis = AxisDefinition(tip=self.current_tip, base=(frame_x, frame_y))
            self.scene_axes[idx] = axis
            self.canvas.set_axis(axis)
            self._reset_click_state()
            self._set_status(f"Axis set for scene {idx + 1}")

    # ── Processing ─────────────────────────────────────────────────────

    def _process_scene(self):
        idx = self._current_scene_idx()
        if idx not in self.scene_axes:
            QMessageBox.warning(self, "No axis", "Define tip and base for this scene first.")
            return

        scene = self.scenes[idx]
        axis = self.scene_axes[idx]
        self._set_status(f"Processing scene {idx + 1} (frames {scene.start_frame}\u2013{scene.end_frame})...")
        QApplication.processEvents()

        result = track_motion(self.video_path, axis, scene.start_frame, scene.end_frame)
        actions = extract_strokes(result.positions, result.timestamps_ms, fps=self.fps)
        self.scene_actions[idx] = actions
        self._set_status(f"Scene {idx + 1}: found {len(actions)} stroke points.")

    def _process_all(self):
        unannotated = [i for i in range(len(self.scenes)) if i not in self.scene_axes]
        if unannotated:
            QMessageBox.warning(
                self, "Missing axes",
                f"Scenes {', '.join(str(i+1) for i in unannotated)} need tip/base annotation.",
            )
            return

        for idx in range(len(self.scenes)):
            scene = self.scenes[idx]
            axis = self.scene_axes[idx]
            self._set_status(f"Processing scene {idx + 1}...")
            QApplication.processEvents()
            result = track_motion(self.video_path, axis, scene.start_frame, scene.end_frame)
            actions = extract_strokes(result.positions, result.timestamps_ms, fps=self.fps)
            self.scene_actions[idx] = actions

        self._set_status(f"Processed all {len(self.scenes)} scenes.")

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

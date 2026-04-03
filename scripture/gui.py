"""PyQt6 GUI for scripture: manual scene splitting, axis annotation, and export."""

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import qtawesome as qta
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QPointF
from PyQt6.QtGui import (
    QAction, QIcon, QImage, QPixmap, QPainter, QPen, QBrush, QColor,
    QShortcut, QKeySequence, QPolygonF, QWheelEvent, QCursor,
)
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QToolBar,
    QLabel, QPushButton, QFileDialog, QMessageBox, QProgressBar, QMenu,
)

from shared_ui.colors import (
    BG_PRIMARY, BG_SECONDARY, BG_TERTIARY, BG_BUTTON,
    TEXT_PRIMARY, TEXT_SECONDARY, TEXT_MUTED,
    BLUE, GREEN, RED, BORDER_SUBTLE,
)
from shared_ui.fonts import SIZE_BODY, SIZE_SMALL, make_font
from shared_ui.spacing import MARGIN_STANDARD, GAP_SMALL, GAP_MEDIUM

from scripture.scene import Scene, scenes_from_splits
from scripture.motion_tracker import AxisDefinition, track_motion
from scripture.stroke_extract import extract_strokes
from scripture.funscript import build_funscript, save_funscript
from scripture.project import save_project, load_project

_ICON_COLOR = "#ddd"
_LAST_SESSION_FILE = Path(__file__).resolve().parent.parent / "sessions" / ".last_session"

_MENU_STYLE = f"""
    QMenu {{
        background: {BG_TERTIARY.name()};
        color: {TEXT_PRIMARY.name()};
        border: 1px solid {BORDER_SUBTLE.name()};
        padding: 2px;
    }}
    QMenu::item {{
        padding: 4px 16px 4px 8px;
    }}
    QMenu::item:selected {{
        background: {BLUE.name()};
    }}
    QMenu::separator {{
        height: 1px;
        background: {BORDER_SUBTLE.name()};
        margin: 2px 4px;
    }}
"""

_BTN_STYLE = f"""
    QPushButton {{
        color: {TEXT_PRIMARY.name()};
        background: {BG_BUTTON.name()};
        border: 1px solid {BORDER_SUBTLE.name()};
        padding: 4px 10px;
        border-radius: 3px;
        font-size: {SIZE_SMALL}pt;
    }}
    QPushButton:hover {{
        background: {BG_TERTIARY.name()};
    }}
    QPushButton:disabled {{
        color: {TEXT_MUTED.name()};
        background: {BG_SECONDARY.name()};
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

_SCENE_EMPTY = QColor(35, 35, 35)
_SCENE_ANNOTATED = QColor(55, 65, 80)
_SCENE_PROCESSED = QColor(70, 100, 70)
_SCENE_BORDER = QColor(100, 100, 100)
_REP_FRAME_COLOR = QColor(255, 220, 80)
_HANDLE_HEIGHT = 8
_TIP_COLOR = QColor(255, 80, 80)
_BASE_COLOR = QColor(80, 140, 255)
_AXIS_COLOR = QColor(255, 220, 80)


class ProcessWorker(QThread):
    frame_progress = pyqtSignal(int)
    scene_done = pyqtSignal(int, list)
    finished = pyqtSignal()
    error = pyqtSignal(int, str)

    def __init__(self, video_path, jobs, fps):
        super().__init__()
        self.video_path = video_path
        self.jobs = jobs
        self.fps = fps

    def run(self):
        offsets = {}
        cumulative = 0
        for idx, scene, _ in self.jobs:
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
    frame_changed = pyqtSignal(int)
    context_menu_requested = pyqtSignal(int, int, int)  # frame, global_x, global_y

    def __init__(self):
        super().__init__()
        self.setFixedHeight(28 + _HANDLE_HEIGHT)
        self.setMinimumWidth(100)
        self.scenes = []
        self.scene_axes = {}
        self.scene_actions = {}
        self.splits = []
        self.total_frames = 0
        self.current_frame = 0
        self._dragging = False
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)

    def set_state(self, scenes, scene_axes, scene_actions, splits, total_frames, current_frame):
        self.scenes = scenes
        self.scene_axes = scene_axes
        self.scene_actions = scene_actions
        self.splits = splits
        self.total_frames = total_frames
        self.current_frame = current_frame
        self.update()

    def _frame_to_x(self, frame):
        if self.total_frames == 0:
            return 0
        return int(frame / self.total_frames * self.width())

    def _x_to_frame(self, x):
        if self.width() == 0:
            return 0
        return max(0, min(int(x / self.width() * self.total_frames), self.total_frames - 1))

    def paintEvent(self, event):
        if self.total_frames == 0:
            return
        p = QPainter(self)
        w, h_total = self.width(), self.height()
        bar_y = _HANDLE_HEIGHT
        bar_h = h_total - _HANDLE_HEIGHT

        for i, scene in enumerate(self.scenes):
            x1 = self._frame_to_x(scene.start_frame)
            x2 = self._frame_to_x(scene.end_frame)
            if i in self.scene_actions:
                color = _SCENE_PROCESSED
            elif i in self.scene_axes:
                color = _SCENE_ANNOTATED
            else:
                color = _SCENE_EMPTY
            p.fillRect(x1, bar_y, x2 - x1, bar_h, color)

        for split in self.splits:
            sx = self._frame_to_x(split)
            p.setPen(QPen(_SCENE_BORDER, 3))
            p.drawLine(sx, bar_y, sx, bar_y + bar_h)
            self._draw_handle(p, sx, _SCENE_BORDER)

        for i, axis in self.scene_axes.items():
            rx = self._frame_to_x(axis.frame)
            p.setPen(QPen(_REP_FRAME_COLOR, 2))
            p.drawLine(rx, bar_y, rx, bar_y + bar_h)
            self._draw_handle(p, rx, _REP_FRAME_COLOR)

        cx = self._frame_to_x(self.current_frame)
        p.setPen(QPen(Qt.GlobalColor.white, 2))
        p.drawLine(cx, bar_y, cx, bar_y + bar_h)

        p.setBrush(Qt.BrushStyle.NoBrush)
        p.setPen(QPen(BORDER_SUBTLE))
        p.drawRect(0, bar_y, w - 1, bar_h - 1)
        p.end()

    def _draw_handle(self, p, x, color):
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(color))
        p.drawPolygon(QPolygonF([
            QPointF(x - 4, 0), QPointF(x + 4, 0), QPointF(x, _HANDLE_HEIGHT),
        ]))

    def _nearest_handle_frame(self, x, y):
        if y > _HANDLE_HEIGHT:
            return None
        handles = [(self._frame_to_x(s), s) for s in self.splits]
        handles += [(self._frame_to_x(a.frame), a.frame) for a in self.scene_axes.values()]
        best_frame, best_dist = None, 7
        for hx, hf in handles:
            d = abs(hx - x)
            if d < best_dist:
                best_dist, best_frame = d, hf
        return best_frame

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self.total_frames > 0:
            x, y = int(event.position().x()), int(event.position().y())
            snap = self._nearest_handle_frame(x, y)
            if snap is not None:
                self.frame_changed.emit(snap)
            else:
                self._dragging = True
                self.frame_changed.emit(self._x_to_frame(x))
        elif event.button() == Qt.MouseButton.RightButton and self.total_frames > 0:
            frame = self._x_to_frame(int(event.position().x()))
            gp = self.mapToGlobal(event.position().toPoint())
            self.context_menu_requested.emit(frame, gp.x(), gp.y())

    def mouseMoveEvent(self, event):
        if self._dragging and self.total_frames > 0:
            self.frame_changed.emit(self._x_to_frame(int(event.position().x())))

    def mouseReleaseEvent(self, event):
        self._dragging = False


class FrameCanvas(QWidget):
    clicked = pyqtSignal(int, int)
    point_dragged = pyqtSignal(str, int, int)  # "tip" or "base", frame_x, frame_y
    context_menu_requested = pyqtSignal(int, int, int, int)  # frame_x, frame_y, global_x, global_y

    def __init__(self):
        super().__init__()
        self.setMinimumSize(400, 300)
        self._pixmap = None
        self._frame_w = 0
        self._frame_h = 0
        self._axis = None
        self._pending_tip = None
        self._pending_base = None
        self._zoom = 1.0
        self._dragging_point = None  # "tip", "base", or None

    @property
    def zoom(self):
        return self._zoom

    @zoom.setter
    def zoom(self, value):
        self._zoom = max(0.1, min(value, 3.0))
        self.update()

    def set_frame(self, frame_bgr):
        h, w = frame_bgr.shape[:2]
        self._frame_w, self._frame_h = w, h
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        self._pixmap = QPixmap.fromImage(QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888))
        self.update()

    def set_axis(self, axis):
        self._axis = axis
        self._pending_tip = self._pending_base = None
        self.update()

    def set_pending_tip(self, tip):
        self._pending_tip = tip
        self.update()

    def set_pending_base(self, base):
        self._pending_base = base
        self.update()

    def clear(self):
        self._pixmap = self._axis = self._pending_tip = self._pending_base = None
        self.update()

    def _display_scale(self):
        if not self._frame_w or not self._frame_h:
            return 1.0
        return min(self.width() / self._frame_w, self.height() / self._frame_h) * self._zoom

    def _frame_to_canvas(self, fx, fy):
        s = self._display_scale()
        dw, dh = int(self._frame_w * s), int(self._frame_h * s)
        ox, oy = (self.width() - dw) // 2, (self.height() - dh) // 2
        return int(fx * s) + ox, int(fy * s) + oy

    def _canvas_to_frame(self, cx, cy):
        s = self._display_scale()
        dw, dh = int(self._frame_w * s), int(self._frame_h * s)
        ox, oy = (self.width() - dw) // 2, (self.height() - dh) // 2
        return int((cx - ox) / s), int((cy - oy) / s)

    def _hit_point(self, cx, cy):
        """Return 'tip', 'base', or None based on which point was hit."""
        for label, pt in [("tip", self._get_tip()), ("base", self._get_base())]:
            if pt is None:
                continue
            px, py = self._frame_to_canvas(*pt)
            if (cx - px) ** 2 + (cy - py) ** 2 <= 100:  # 10px radius
                return label
        return None

    def _get_tip(self):
        if self._axis:
            return self._axis.tip
        return self._pending_tip

    def _get_base(self):
        if self._axis:
            return self._axis.base
        return self._pending_base

    def paintEvent(self, event):
        p = QPainter(self)
        p.fillRect(self.rect(), QBrush(BG_PRIMARY))
        if self._pixmap is None:
            p.setPen(QPen(TEXT_MUTED))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Open a video to begin")
            p.end()
            return

        s = self._display_scale()
        dw, dh = int(self._frame_w * s), int(self._frame_h * s)
        ox, oy = (self.width() - dw) // 2, (self.height() - dh) // 2
        p.setPen(QPen(BORDER_SUBTLE))
        p.drawRect(ox - 1, oy - 1, dw + 1, dh + 1)
        p.drawPixmap(ox, oy, self._pixmap.scaled(dw, dh, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))

        if self._axis:
            self._draw_axis(p, self._axis)
        else:
            if self._pending_tip:
                self._draw_point(p, self._pending_tip, _TIP_COLOR, "TIP")
            if self._pending_base:
                self._draw_point(p, self._pending_base, _BASE_COLOR, "BASE")
        p.end()

    def _draw_point(self, p, pt, color, label):
        cx, cy = self._frame_to_canvas(*pt)
        p.setPen(QPen(Qt.GlobalColor.white, 1))
        p.setBrush(QBrush(color))
        p.drawEllipse(cx - 5, cy - 5, 10, 10)
        p.setPen(QPen(color))
        p.setFont(make_font(size=SIZE_SMALL))
        p.drawText(cx + 10, cy + 4, label)

    def _draw_axis(self, p, axis):
        tx, ty = self._frame_to_canvas(*axis.tip)
        bx, by = self._frame_to_canvas(*axis.base)
        p.setPen(QPen(_AXIS_COLOR, 2, Qt.PenStyle.DashLine))
        p.drawLine(tx, ty, bx, by)
        self._draw_point(p, axis.tip, _TIP_COLOR, "TIP")
        self._draw_point(p, axis.base, _BASE_COLOR, "BASE")

    def mousePressEvent(self, event):
        if not self._frame_w:
            return
        if event.button() == Qt.MouseButton.LeftButton:
            cx, cy = int(event.position().x()), int(event.position().y())
            hit = self._hit_point(cx, cy)
            if hit:
                self._dragging_point = hit
            else:
                fx, fy = self._canvas_to_frame(cx, cy)
                self.clicked.emit(fx, fy)
        elif event.button() == Qt.MouseButton.RightButton:
            cx, cy = int(event.position().x()), int(event.position().y())
            fx, fy = self._canvas_to_frame(cx, cy)
            gp = self.mapToGlobal(event.position().toPoint())
            self.context_menu_requested.emit(fx, fy, gp.x(), gp.y())

    def mouseMoveEvent(self, event):
        if self._dragging_point:
            cx, cy = int(event.position().x()), int(event.position().y())
            fx, fy = self._canvas_to_frame(cx, cy)
            self.point_dragged.emit(self._dragging_point, fx, fy)

    def mouseReleaseEvent(self, event):
        self._dragging_point = None

    def wheelEvent(self, event):
        d = event.angleDelta().y()
        if d > 0:
            self.zoom *= 1.15
        elif d < 0:
            self.zoom /= 1.15


class App(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Scripture")
        self.resize(1200, 800)
        _icon_path = Path(__file__).resolve().parent.parent / "icon.ico"
        if _icon_path.exists():
            self.setWindowIcon(QIcon(str(_icon_path)))
        self.setStyleSheet(f"background: {BG_PRIMARY.name()}; color: {TEXT_PRIMARY.name()};")

        self.video_path = None
        self.cap = None
        self.fps = 30.0
        self.total_frames = 0
        self.frame_w = self.frame_h = 0

        self.splits = []
        self.scenes = []
        self.scene_axes = {}
        self.scene_actions = {}
        self.current_frame_idx = 0
        self.placing = None
        self.pending_tip = self.pending_base = None

        self._worker = None
        self._project_path = None
        self._dirty = False

        self._build_ui()
        self._build_shortcuts()
        self._try_load_last_session()

    def _mark_dirty(self):
        self._dirty = True

    def _mark_clean(self):
        self._dirty = False

    def _build_shortcuts(self):
        QShortcut(QKeySequence("B"), self, lambda: self._place_point("base"))
        QShortcut(QKeySequence("T"), self, lambda: self._place_point("tip"))
        QShortcut(QKeySequence("S"), self, self._split_or_unsplit)
        QShortcut(QKeySequence(Qt.Key.Key_Left), self, self._frame_back)
        QShortcut(QKeySequence(Qt.Key.Key_Right), self, self._frame_forward)
        QShortcut(QKeySequence(Qt.Key.Key_Tab), self, self._next_poi)
        QShortcut(QKeySequence("Shift+Tab"), self, self._prev_poi)

    def _frame_back(self):
        if self.current_frame_idx > 0:
            self._show_frame(self.current_frame_idx - 1)

    def _frame_forward(self):
        if self.current_frame_idx < self.total_frames - 1:
            self._show_frame(self.current_frame_idx + 1)

    def _get_points_of_interest(self):
        """All scene breaks and representative frames, sorted."""
        pois = set(self.splits)
        for axis in self.scene_axes.values():
            pois.add(axis.frame)
        return sorted(pois)

    def _next_poi(self):
        pois = self._get_points_of_interest()
        if not pois:
            return
        for p in pois:
            if p > self.current_frame_idx:
                self._show_frame(p)
                return
        self._show_frame(pois[0])  # wrap

    def _prev_poi(self):
        pois = self._get_points_of_interest()
        if not pois:
            return
        for p in reversed(pois):
            if p < self.current_frame_idx:
                self._show_frame(p)
                return
        self._show_frame(pois[-1])  # wrap

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(MARGIN_STANDARD, MARGIN_STANDARD, MARGIN_STANDARD, MARGIN_STANDARD)
        root.setSpacing(GAP_MEDIUM)

        # --- File I/O toolbar ---
        tb = QToolBar()
        tb.setMovable(False)
        tb.setFloatable(False)
        tb.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        tb.setContentsMargins(MARGIN_STANDARD, 0, MARGIN_STANDARD, 0)
        self.addToolBar(tb)

        for icon, label, handler in [
            ("fa5s.folder-open", "New", self._open_video),
            ("fa5s.save", "Save", self._save_project),
            ("fa5s.copy", "Save As", self._save_project_as),
            ("fa5s.folder", "Load", self._load_project),
        ]:
            act = QAction(qta.icon(icon, color=_ICON_COLOR), label, self)
            act.triggered.connect(handler)
            tb.addAction(act)
        tb.addSeparator()
        act = QAction(qta.icon("fa5s.file-export", color=_ICON_COLOR), "Export", self)
        act.triggered.connect(self._export)
        tb.addAction(act)

        # Progress bar embedded in toolbar (no layout shift)
        self._progress_sep = tb.addSeparator()
        self._progress_sep.setVisible(False)
        self.progress_bar = QProgressBar()
        self.progress_bar.setStyleSheet(_PROGRESS_STYLE)
        self.progress_bar.setFixedHeight(20)
        self.progress_bar.setMinimumWidth(200)
        self._progress_action = tb.addWidget(self.progress_bar)
        self._progress_action.setVisible(False)

        # --- Frame canvas ---
        self.canvas = FrameCanvas()
        self.canvas.clicked.connect(self._on_canvas_click)
        self.canvas.point_dragged.connect(self._on_point_dragged)
        self.canvas.context_menu_requested.connect(self._on_canvas_context_menu)
        root.addWidget(self.canvas, stretch=1)

        # --- Timeline ---
        self.timeline = TimelineWidget()
        self.timeline.frame_changed.connect(self._on_timeline_frame)
        self.timeline.context_menu_requested.connect(self._on_timeline_context_menu)
        root.addWidget(self.timeline)

        # --- Bottom bar: Process All + info ---
        bottom = QHBoxLayout()
        bottom.setSpacing(GAP_MEDIUM)

        self.btn_process_all = QPushButton("Process All")
        self.btn_process_all.setStyleSheet(_BTN_STYLE)
        self.btn_process_all.clicked.connect(self._process_all)
        bottom.addWidget(self.btn_process_all)

        self.info_label = QLabel("")
        self.info_label.setFont(make_font(size=SIZE_SMALL))
        self.info_label.setStyleSheet(
            f"color: {TEXT_MUTED.name()}; background: {BG_SECONDARY.name()}; "
            f"padding: 3px 6px; border-radius: 2px;"
        )
        bottom.addWidget(self.info_label, stretch=1)

        root.addLayout(bottom)

    def _set_status(self, text):
        self.info_label.setText(text)

    def _format_time(self, frame_idx):
        s = frame_idx / self.fps if self.fps else 0
        m, s = divmod(int(s), 60)
        h, m = divmod(m, 60)
        return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"

    def _current_scene_idx(self):
        for i, sc in enumerate(self.scenes):
            if self.current_frame_idx < sc.end_frame:
                return i
        return max(0, len(self.scenes) - 1)

    def _rebuild_scenes(self, clear_annotations=True):
        old_axes, old_actions = dict(self.scene_axes), dict(self.scene_actions)
        self.scenes = scenes_from_splits(self.splits, self.total_frames)
        if clear_annotations:
            self.scene_axes.clear()
            self.scene_actions.clear()
            for _oi, axis in old_axes.items():
                for ni, sc in enumerate(self.scenes):
                    if sc.start_frame <= axis.frame < sc.end_frame:
                        self.scene_axes[ni] = axis
                        if _oi in old_actions:
                            self.scene_actions[ni] = old_actions[_oi]
                        break

    def _update_timeline(self):
        self.timeline.set_state(
            self.scenes, self.scene_axes, self.scene_actions,
            self.splits, self.total_frames, self.current_frame_idx,
        )

    def _update_info(self):
        if self.total_frames == 0:
            self._set_status("Open a video to begin.")
            return
        idx = self._current_scene_idx()
        sc = self.scenes[idx]
        sc_info = f"Scene {idx+1}/{len(self.scenes)} [{self._format_time(sc.start_frame)} \u2013 {self._format_time(sc.end_frame)}]"
        fr_info = f"{self.current_frame_idx} / {self.total_frames} \u2014 {self._format_time(self.current_frame_idx)} / {self._format_time(self.total_frames)}"
        self._set_status(f"{sc_info}    {fr_info}")

    # ── Video loading ──────────────────────────────────────────────

    def _open_video(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Video", "",
            "Video files (*.mp4 *.mkv *.avi *.webm *.mov);;All files (*)",
        )
        if not path:
            return
        self._load_video(path)
        self.splits = []
        self.scenes = [Scene(0, self.total_frames)]
        self.scene_axes.clear()
        self.scene_actions.clear()
        self.current_frame_idx = 0
        self._project_path = None
        self._mark_dirty()
        self._show_frame(0)

    def _load_video(self, path):
        self.video_path = path
        if self.cap:
            self.cap.release()
        self.cap = cv2.VideoCapture(path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # ── Frame display ──────────────────────────────────────────────

    def _show_frame(self, frame_idx):
        if not self.cap:
            return
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self.cap.read()
        if not ret:
            return
        self.current_frame_idx = frame_idx
        idx = self._current_scene_idx()
        self.canvas._frame_w = self.frame_w
        self.canvas._frame_h = self.frame_h
        self.canvas.set_frame(frame)

        if idx in self.scene_axes and self.scene_axes[idx].frame == frame_idx:
            self.canvas.set_axis(self.scene_axes[idx])
        else:
            self.canvas.set_axis(None)
            self.canvas.set_pending_tip(self.pending_tip)
            self.canvas.set_pending_base(self.pending_base)

        self._update_info()
        self._update_timeline()

    def _on_timeline_frame(self, frame):
        self._show_frame(frame)

    # ── Scene splitting ────────────────────────────────────────────

    def _split_or_unsplit(self):
        if self.total_frames == 0:
            return
        frame = self.current_frame_idx
        if frame in self.splits:
            self._do_unsplit(frame)
        elif 0 < frame < self.total_frames:
            self.splits.append(frame)
            self._rebuild_scenes()
            self._cancel_placing()
            self._mark_dirty()
            self._show_frame(frame)
            self._set_status(f"Split at frame {frame}. {len(self.scenes)} scenes.")

    def _do_unsplit(self, frame):
        idx = self._current_scene_idx()
        left_idx = idx - 1 if idx > 0 and self.scenes[idx].start_frame == frame else idx
        right_idx = left_idx + 1
        if left_idx in self.scene_axes and right_idx < len(self.scenes) and right_idx in self.scene_axes:
            QMessageBox.warning(self, "Cannot unsplit", "Both adjacent scenes have axes. Delete one axis first.")
            return
        self.splits.remove(frame)
        self._rebuild_scenes()
        self._cancel_placing()
        self._mark_dirty()
        self._show_frame(frame)
        self._set_status(f"Removed split. {len(self.scenes)} scenes.")

    # ── Context menus ──────────────────────────────────────────────

    def _on_canvas_context_menu(self, fx, fy, gx, gy):
        if not self.scenes:
            return
        idx = self._current_scene_idx()
        has_axis = idx in self.scene_axes
        menu = QMenu(self)
        menu.setStyleSheet(_MENU_STYLE)

        if has_axis or self.pending_tip:
            menu.addAction("\u2717 Remove Tip", lambda: self._delete_point("tip"))
        else:
            menu.addAction("Place Tip Here", lambda: self._place_point_at("tip", fx, fy))

        if has_axis or self.pending_base:
            menu.addAction("\u2717 Remove Base", lambda: self._delete_point("base"))
        else:
            menu.addAction("Place Base Here", lambda: self._place_point_at("base", fx, fy))

        menu.exec(QCursor.pos())

    def _on_timeline_context_menu(self, frame, gx, gy):
        if self.total_frames == 0:
            return
        self._show_frame(frame)

        idx = self._current_scene_idx()
        menu = QMenu(self)
        menu.setStyleSheet(_MENU_STYLE)

        if frame in self.splits:
            menu.addAction("Unsplit", lambda: self._do_unsplit(frame))
        elif 0 < frame < self.total_frames:
            menu.addAction("Split Here", lambda: self._do_split_at(frame))

        menu.addSeparator()
        if idx in self.scene_axes:
            if idx in self.scene_actions:
                menu.addAction("Discard Scene Actions", lambda: self._discard_scene(idx))
            elif not self._is_processing():
                menu.addAction("Process Scene Actions", lambda: self._process_scene(idx))

        menu.exec(QCursor.pos())

    def _do_split_at(self, frame):
        if frame in self.splits or frame <= 0 or frame >= self.total_frames:
            return
        self.splits.append(frame)
        self._rebuild_scenes()
        self._cancel_placing()
        self._mark_dirty()
        self._show_frame(frame)
        self._set_status(f"Split at frame {frame}. {len(self.scenes)} scenes.")

    # ── Axis annotation ────────────────────────────────────────────

    def _cancel_placing(self):
        self.placing = None
        self.pending_tip = self.pending_base = None

    def _place_point(self, which):
        """Enter placement mode for tip or base."""
        self.placing = which

    def _place_point_at(self, which, fx, fy):
        """Place a point directly at (fx, fy) without entering placement mode."""
        idx = self._current_scene_idx()

        # If there's an existing axis on a different frame, confirm change
        if idx in self.scene_axes and self.scene_axes[idx].frame != self.current_frame_idx:
            reply = QMessageBox.question(
                self, "Change representative frame?",
                "Move axis to this frame?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return
            old = self.scene_axes.pop(idx)
            if which == "tip":
                self.pending_base = old.base
            else:
                self.pending_tip = old.tip
            if idx in self.scene_actions:
                del self.scene_actions[idx]

        if which == "tip":
            self.pending_tip = (fx, fy)
        else:
            self.pending_base = (fx, fy)

        # If both placed, form axis
        if self.pending_tip and self.pending_base:
            self.scene_axes[idx] = AxisDefinition(
                tip=self.pending_tip, base=self.pending_base, frame=self.current_frame_idx,
            )
            self.pending_tip = self.pending_base = None

        self._mark_dirty()
        self._show_frame(self.current_frame_idx)

    def _delete_point(self, which):
        idx = self._current_scene_idx()
        if idx in self.scene_axes:
            old = self.scene_axes.pop(idx)
            if which == "tip":
                self.pending_base = old.base
            else:
                self.pending_tip = old.tip
            if idx in self.scene_actions:
                del self.scene_actions[idx]
        if which == "tip":
            self.pending_tip = None
        else:
            self.pending_base = None
        self._mark_dirty()
        self._show_frame(self.current_frame_idx)

    def _on_canvas_click(self, fx, fy):
        if not self.scenes or not self.placing:
            return
        idx = self._current_scene_idx()

        # Changing representative frame?
        if idx in self.scene_axes and self.scene_axes[idx].frame != self.current_frame_idx:
            reply = QMessageBox.question(
                self, "Change representative frame?",
                "Move axis to this frame?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return
            old = self.scene_axes.pop(idx)
            if self.placing == "tip":
                self.pending_base = old.base
            else:
                self.pending_tip = old.tip
            if idx in self.scene_actions:
                del self.scene_actions[idx]

        if self.placing == "tip":
            self.pending_tip = (fx, fy)
        elif self.placing == "base":
            self.pending_base = (fx, fy)
        self.placing = None

        if self.pending_tip and self.pending_base:
            self.scene_axes[idx] = AxisDefinition(
                tip=self.pending_tip, base=self.pending_base, frame=self.current_frame_idx,
            )
            self.pending_tip = self.pending_base = None

        self._mark_dirty()
        self._show_frame(self.current_frame_idx)

    def _on_point_dragged(self, which, fx, fy):
        idx = self._current_scene_idx()
        if idx in self.scene_axes:
            old = self.scene_axes[idx]
            if which == "tip":
                self.scene_axes[idx] = AxisDefinition(tip=(fx, fy), base=old.base, frame=old.frame)
            else:
                self.scene_axes[idx] = AxisDefinition(tip=old.tip, base=(fx, fy), frame=old.frame)
            if idx in self.scene_actions:
                del self.scene_actions[idx]
            self._mark_dirty()
            self.canvas.set_axis(self.scene_axes[idx])
            self._update_timeline()
        elif which == "tip" and self.pending_tip:
            self.pending_tip = (fx, fy)
            self.canvas.set_pending_tip(self.pending_tip)
        elif which == "base" and self.pending_base:
            self.pending_base = (fx, fy)
            self.canvas.set_pending_base(self.pending_base)

    # ── Processing ─────────────────────────────────────────────────

    def _build_jobs(self, indices):
        return [(i, self.scenes[i], self.scene_axes[i]) for i in indices]

    @staticmethod
    def _fmt_duration(seconds):
        s = int(seconds)
        if s < 60:
            return f"{s}s"
        m, s = divmod(s, 60)
        if m < 60:
            return f"{m}m {s:02d}s"
        h, m = divmod(m, 60)
        return f"{h}h {m:02d}m {s:02d}s"

    def _start_processing(self, jobs):
        tf = sum(sc.end_frame - sc.start_frame for _, sc, _ in jobs)
        self._progress_sep.setVisible(True)
        self._progress_action.setVisible(True)
        self.progress_bar.setMaximum(tf)
        self.progress_bar.setValue(0)
        self.btn_process_all.setEnabled(False)
        self._process_total_frames = tf
        self._process_start_time = time.monotonic()
        self._worker = ProcessWorker(self.video_path, jobs, self.fps)
        self._worker.frame_progress.connect(self._on_frame_progress)
        self._worker.scene_done.connect(self._on_scene_done)
        self._worker.error.connect(self._on_process_error)
        self._worker.finished.connect(self._on_process_finished)
        self._worker.start()

    def _on_frame_progress(self, done):
        self.progress_bar.setValue(done)
        el = time.monotonic() - self._process_start_time
        if done > 0:
            eta = el / done * (self._process_total_frames - done)
            self.progress_bar.setFormat(
                f"{done}/{self._process_total_frames} (%p%) \u2014 "
                f"{self._fmt_duration(el)} elapsed, ~{self._fmt_duration(eta)} left"
            )

    def _on_scene_done(self, idx, actions):
        self.scene_actions[idx] = actions
        self._mark_dirty()
        self._update_timeline()

    def _on_process_error(self, idx, msg):
        self._set_status(f"Error scene {idx+1}: {msg}")

    def _on_process_finished(self):
        el = time.monotonic() - self._process_start_time
        self._progress_sep.setVisible(False)
        self._progress_action.setVisible(False)
        self.btn_process_all.setEnabled(True)
        self._worker = None
        total = sum(len(a) for a in self.scene_actions.values())
        self._set_status(f"Done \u2014 {total} actions in {self._fmt_duration(el)}.")

    def _is_processing(self):
        return self._worker is not None and self._worker.isRunning()

    def _process_scene(self, idx=None):
        if self._is_processing():
            self._set_status("Already processing \u2014 wait for it to finish.")
            return
        if idx is None:
            idx = self._current_scene_idx()
        if idx not in self.scene_axes:
            return
        self._start_processing(self._build_jobs([idx]))

    def _process_all(self):
        if self._is_processing():
            self._set_status("Already processing \u2014 wait for it to finish.")
            return
        annotated = [i for i in range(len(self.scenes)) if i in self.scene_axes]
        if not annotated:
            QMessageBox.warning(self, "No axes", "Annotate tip/base on at least one scene first.")
            return
        self._start_processing(self._build_jobs(annotated))

    def _discard_scene(self, idx=None):
        if idx is None:
            idx = self._current_scene_idx()
        if idx in self.scene_actions:
            del self.scene_actions[idx]
            self._mark_dirty()
            self._update_timeline()
            self._set_status(f"Discarded actions for scene {idx+1}.")

    # ── Project save/load ──────────────────────────────────────────

    def _build_state(self):
        axes = {str(i): {"tip": list(a.tip), "base": list(a.base), "frame": a.frame}
                for i, a in self.scene_axes.items()}
        acts = {str(i): a for i, a in self.scene_actions.items()}
        return {
            "video_path": self.video_path, "splits": self.splits,
            "axes": axes, "actions": acts, "current_frame": self.current_frame_idx,
        }

    def _do_save(self, path):
        save_project(path, self._build_state())
        self._project_path = path
        self._mark_clean()
        self._save_last_session(path)
        self._set_status(f"Saved to {Path(path).name}")

    def _save_project(self):
        if not self.video_path:
            QMessageBox.warning(self, "No video", "Open a video first.")
            return
        if self._project_path:
            self._do_save(self._project_path)
        else:
            self._save_project_as()

    def _save_project_as(self):
        if not self.video_path:
            QMessageBox.warning(self, "No video", "Open a video first.")
            return
        sd = str(Path(__file__).resolve().parent.parent / "sessions")
        dn = Path(self.video_path).stem + ".scripture"
        path, _ = QFileDialog.getSaveFileName(self, "Save As", str(Path(sd) / dn), "Scripture (*.scripture);;All (*)")
        if path:
            self._do_save(path)

    def _load_project(self):
        sd = str(Path(__file__).resolve().parent.parent / "sessions")
        path, _ = QFileDialog.getOpenFileName(self, "Load", sd, "Scripture (*.scripture);;All (*)")
        if path:
            self._do_load(path)

    def _do_load(self, path):
        state = load_project(path)
        vp = state["video_path"]
        if self.cap:
            self.cap.release()
        cap = cv2.VideoCapture(vp)
        if not cap.isOpened():
            QMessageBox.critical(self, "Video not found", f"Cannot open: {vp}")
            return

        self.cap = cap
        self.video_path = vp
        self.fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.splits = state["splits"]
        self._rebuild_scenes(clear_annotations=False)
        self.scene_axes.clear()
        for k, v in state.get("axes", {}).items():
            self.scene_axes[int(k)] = AxisDefinition(tip=tuple(v["tip"]), base=tuple(v["base"]), frame=v.get("frame", 0))
        self.scene_actions.clear()
        for k, v in state.get("actions", {}).items():
            self.scene_actions[int(k)] = v

        self._project_path = path
        self._mark_clean()
        self._cancel_placing()
        self._save_last_session(path)
        self.current_frame_idx = state.get("current_frame", 0)
        self._show_frame(self.current_frame_idx)

    def _save_last_session(self, path):
        try:
            _LAST_SESSION_FILE.parent.mkdir(parents=True, exist_ok=True)
            _LAST_SESSION_FILE.write_text(path)
        except OSError:
            pass

    def _try_load_last_session(self):
        try:
            if _LAST_SESSION_FILE.exists():
                path = _LAST_SESSION_FILE.read_text().strip()
                if path and Path(path).exists():
                    self._do_load(path)
        except Exception:
            pass

    # ── Export ──────────────────────────────────────────────────────

    def _export(self):
        if not self.scene_actions:
            QMessageBox.warning(self, "No data", "Process at least one scene first.")
            return
        all_a = []
        for a in self.scene_actions.values():
            all_a.extend(a)
        fs = build_funscript(all_a, int(self.total_frames / self.fps))
        dn = Path(self.video_path).stem + ".funscript" if self.video_path else "output.funscript"
        path, _ = QFileDialog.getSaveFileName(self, "Export", dn, "Funscript (*.funscript);;All (*)")
        if path:
            save_funscript(fs, path)
            self._set_status(f"Exported {len(all_a)} actions to {Path(path).name}")

    # ── Close guard ────────────────────────────────────────────────

    def closeEvent(self, event):
        if self._worker and self._worker.isRunning():
            reply = QMessageBox.question(
                self, "Processing in progress",
                "Processing is still running. Abort and close?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                event.ignore()
                return
            self._worker.terminate()
            self._worker.wait()

        if self._dirty:
            reply = QMessageBox.question(
                self, "Unsaved changes",
                "Save before closing?",
                QMessageBox.StandardButton.Save | QMessageBox.StandardButton.Discard | QMessageBox.StandardButton.Cancel,
            )
            if reply == QMessageBox.StandardButton.Save:
                self._save_project()
                event.accept()
            elif reply == QMessageBox.StandardButton.Discard:
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()

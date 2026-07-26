"""PyQt6 GUI for scripture: manual scene splitting, axis annotation, and export."""

import bisect
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
    QShortcut, QKeySequence, QPolygonF, QCursor,
)
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QToolBar,
    QLabel, QPushButton, QFileDialog, QMessageBox, QProgressBar, QMenu, QSizePolicy,
)

from shared_ui.colors import (
    BG_PRIMARY, BG_SECONDARY, BG_TERTIARY, BG_BUTTON,
    TEXT_PRIMARY, TEXT_MUTED,
    BLUE, BORDER_SUBTLE,
)
from shared_ui.fonts import SIZE_BODY, SIZE_SMALL, make_font
from shared_ui.spacing import MARGIN_STANDARD, GAP_MEDIUM

from content import load_content

from scripture.auto_funscript import (
    pipeline_result_from_state, pipeline_result_to_state, run_pipeline,
)
from scripture.scene import Scene, actions_by_scene, scenes_from_splits
from scripture.motion_tracker import AxisDefinition, TrackingResult, track_motion
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
    scene_done = pyqtSignal(int, list, object)  # idx, actions, TrackingResult
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
                    on_frame=lambda f, _o=offset: self.frame_progress.emit(_o + f),
                )
                actions = extract_strokes(result.positions, result.timestamps_ms, fps=self.fps)
                self.scene_done.emit(idx, actions, result)
            except Exception as e:
                import traceback
                traceback.print_exc()
                self.error.emit(idx, str(e))
        self.finished.emit()


class AutoProcessWorker(QThread):
    """Runs the fully automatic YOLO+flow pipeline over the whole video."""
    frame_progress = pyqtSignal(int)
    done = pyqtSignal(object)  # PipelineResult
    error = pyqtSignal(str)

    def __init__(self, video_path):
        super().__init__()
        self.video_path = video_path

    def run(self):
        try:
            result = run_pipeline(
                self.video_path, on_frame=self.frame_progress.emit)
            self.done.emit(result)
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.error.emit(str(e))


_ACTION_DOT_COLOR = QColor(80, 255, 80, 180)

# Detection box colors by class (drawn in the auto-tracking overlay).  The class
# names are private, so the map comes from the content overlay.
_DET_COLORS = {
    name: QColor(*rgb) for name, rgb in load_content()["class_colors"].items()
}
_DET_DEFAULT_COLOR = QColor(200, 200, 200)

# ROI border color by lock state: how the tracker justified this ROI
_LOCK_COLORS = {
    "anchor": QColor(80, 255, 80),     # an anchor class detected
    "contact": QColor(255, 215, 60),   # contact object over last anchor spot
    "coast": QColor(255, 140, 40),     # nothing relevant; persistence window
}
_LOCK_LABELS = {
    "anchor": "lock: anchor",
    "contact": "lock: contact",
    "coast": "coasting",
}


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
        self._zoom = 1.0       # 1.0 = fully zoomed out (all frames visible)
        self._scroll = 0.0     # left edge in frame-fraction (0.0 to 1.0 - 1/zoom)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)

    def set_state(self, scenes, scene_axes, scene_actions, splits, total_frames, current_frame, ground_truth=None):
        self.scenes = scenes
        self.scene_axes = scene_axes
        self.scene_actions = scene_actions
        self.splits = splits
        self.total_frames = total_frames
        self.current_frame = current_frame
        self.ground_truth = ground_truth or {}
        self.update()

    def _frame_to_x(self, frame):
        if self.total_frames == 0:
            return 0
        frac = frame / self.total_frames
        return int((frac - self._scroll) * self._zoom * self.width())

    def _x_to_frame(self, x):
        if self.width() == 0 or self.total_frames == 0:
            return 0
        frac = x / (self._zoom * self.width()) + self._scroll
        return max(0, min(int(frac * self.total_frames), self.total_frames - 1))

    def _clamp_scroll(self):
        max_scroll = max(0.0, 1.0 - 1.0 / self._zoom)
        self._scroll = max(0.0, min(self._scroll, max_scroll))

    def paintEvent(self, event):
        if self.total_frames == 0:
            return
        p = QPainter(self)
        p.setClipRect(0, 0, self.width(), self.height())
        w, h_total = self.width(), self.height()
        bar_y = _HANDLE_HEIGHT
        bar_h = h_total - _HANDLE_HEIGHT

        for i, scene in enumerate(self.scenes):
            x1 = self._frame_to_x(scene.start_frame)
            x2 = self._frame_to_x(scene.end_frame)
            if x2 < 0 or x1 > w:
                continue
            if i in self.scene_actions:
                color = _SCENE_PROCESSED
            elif i in self.scene_axes:
                color = _SCENE_ANNOTATED
            else:
                color = _SCENE_EMPTY
            p.fillRect(x1, bar_y, x2 - x1, bar_h, color)

        # Action frame dots along the center line
        mid_y = bar_y + bar_h // 2
        fps = self._get_fps()
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(_ACTION_DOT_COLOR))
        # Show GT action frames if any annotations exist; otherwise computed
        scenes_with_gt = set(self.ground_truth.keys())
        for scene_idx, frames in self.ground_truth.items():
            for frame_idx, entry in frames.items():
                if entry.get("is_action"):
                    ax = self._frame_to_x(frame_idx)
                    if 0 <= ax <= w:
                        p.drawEllipse(ax - 2, mid_y - 2, 4, 4)
        for scene_idx, actions in self.scene_actions.items():
            if scene_idx in scenes_with_gt:
                continue  # skip computed dots for annotated scenes
            for a in actions:
                frame = int(round(a["at"] / 1000 * fps))
                ax = self._frame_to_x(frame)
                if 0 <= ax <= w:
                    p.drawEllipse(ax - 2, mid_y - 2, 4, 4)

        for split in self.splits:
            sx = self._frame_to_x(split)
            if -5 <= sx <= w + 5:
                p.setPen(QPen(_SCENE_BORDER, 3))
                p.drawLine(sx, bar_y, sx, bar_y + bar_h)
                self._draw_handle(p, sx, _SCENE_BORDER)

        for i, axis in self.scene_axes.items():
            rx = self._frame_to_x(axis.frame)
            if -5 <= rx <= w + 5:
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

    def _get_fps(self):
        """Get fps from parent App if available."""
        parent = self.parent()
        while parent is not None:
            if hasattr(parent, 'fps'):
                return parent.fps
            parent = parent.parent()
        return 30.0

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

    def wheelEvent(self, event):
        if self.total_frames == 0:
            return
        x = int(event.position().x())
        # Frame under cursor before zoom
        frame_at_cursor = self._x_to_frame(x)
        frac_at_cursor = frame_at_cursor / self.total_frames

        # Adjust zoom
        delta = event.angleDelta().y()
        if delta > 0:
            self._zoom = min(self._zoom * 1.3, max(1.0, self.total_frames / 10))
        elif delta < 0:
            self._zoom = max(self._zoom / 1.3, 1.0)

        # Adjust scroll so the frame under cursor stays under cursor
        if self._zoom > 1.0:
            self._scroll = frac_at_cursor - x / (self._zoom * self.width())
        else:
            self._scroll = 0.0
        self._clamp_scroll()
        self.update()


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
        self._dragging_point = None  # "tip", "base", "contact", or None
        self._overlay = None  # dict with axis, contact_pt, pos, is_action
        self._gt = None  # ground truth dict: tip, base, contact, is_action, pos
        self._auto_overlay = None  # dict with roi, detections, pos, active, is_action

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

    def set_overlay(self, overlay):
        """Set debug overlay data (or None to clear).

        overlay dict keys: axis, contact_pt, pos (0-100), is_action (bool).
        """
        self._overlay = overlay
        self.update()

    def set_gt(self, gt):
        """Set ground truth annotation for this frame (or None)."""
        self._gt = gt
        self.update()

    def set_auto_overlay(self, overlay):
        """Set auto-tracking overlay data (or None to clear).

        overlay dict keys: roi (x,y,w,h or None), detections (list of
        Detection or None), pos (0-100), active (bool), is_action (bool).
        """
        self._auto_overlay = overlay
        self.update()

    def clear(self):
        self._pixmap = self._axis = self._pending_tip = self._pending_base = self._overlay = self._gt = None
        self._auto_overlay = None
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
        """Return 'tip', 'base', 'contact', or None based on which point was hit."""
        # Check GT points first (on top visually, so should catch clicks first)
        if self._gt:
            for label in ("contact", "tip", "base"):
                pt = self._gt.get(label)
                if pt is None:
                    continue
                px, py = self._frame_to_canvas(*pt)
                if (cx - px) ** 2 + (cy - py) ** 2 <= 100:
                    return label
        for label, pt in [("tip", self._get_tip()), ("base", self._get_base())]:
            if pt is None:
                continue
            px, py = self._frame_to_canvas(*pt)
            if (cx - px) ** 2 + (cy - py) ** 2 <= 100:
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

        if self._auto_overlay:
            self._draw_auto_overlay(p)
        if self._gt:
            self._draw_gt(p)
        elif not self._auto_overlay:
            if self._overlay:
                self._draw_overlay(p)
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

    def _draw_overlay(self, p):
        """Draw debug overlay: axis, tip/base, contact with direction arrow."""
        ov = self._overlay
        axis = ov["axis"]
        tx, ty = self._frame_to_canvas(*axis.tip)
        bx, by = self._frame_to_canvas(*axis.base)

        # Faint axis line
        faint_axis = QColor(_AXIS_COLOR)
        faint_axis.setAlpha(80)
        p.setPen(QPen(faint_axis, 1))
        p.drawLine(tx, ty, bx, by)

        # Small semi-transparent tip/base dots (no labels)
        for pt, color in [(axis.tip, _TIP_COLOR), (axis.base, _BASE_COLOR)]:
            cx, cy = self._frame_to_canvas(*pt)
            faint = QColor(color)
            faint.setAlpha(100)
            p.setPen(QPen(Qt.GlobalColor.transparent))
            p.setBrush(QBrush(faint))
            p.drawEllipse(cx - 3, cy - 3, 6, 6)

        ct = ov["contact_pt"]
        ccx, ccy = self._frame_to_canvas(*ct)
        contact_color = QColor(80, 255, 80)

        if ov["is_action"]:
            # Action frame: bold circle, no fill
            p.setPen(QPen(contact_color, 3))
            p.setBrush(QBrush(Qt.GlobalColor.transparent))
            p.drawEllipse(ccx - 7, ccy - 7, 14, 14)
        else:
            # Non-action: small dot with direction arrow along the axis
            p.setPen(QPen(Qt.GlobalColor.white, 1))
            p.setBrush(QBrush(contact_color))
            p.drawEllipse(ccx - 3, ccy - 3, 6, 6)

            direction = ov.get("direction", 0)
            if direction != 0 and (tx != bx or ty != by):
                # Arrow along axis: toward tip (+1) or toward base (-1)
                ax_dx, ax_dy = tx - bx, ty - by
                ax_len = (ax_dx ** 2 + ax_dy ** 2) ** 0.5
                if ax_len > 0:
                    ux, uy = ax_dx / ax_len, ax_dy / ax_len
                    if direction < 0:
                        ux, uy = -ux, -uy
                    arrow_len = 15
                    ex, ey = ccx + ux * arrow_len, ccy + uy * arrow_len
                    p.setPen(QPen(contact_color, 2))
                    p.drawLine(ccx, ccy, int(ex), int(ey))
                    # Arrowhead
                    head = 5
                    px, py = -uy, ux  # perpendicular
                    p.drawLine(int(ex), int(ey),
                               int(ex - ux * head + px * head * 0.5),
                               int(ey - uy * head + py * head * 0.5))
                    p.drawLine(int(ex), int(ey),
                               int(ex - ux * head - px * head * 0.5),
                               int(ey - uy * head - py * head * 0.5))

        # Pos value on every frame
        p.setPen(QPen(contact_color))
        p.setFont(make_font(size=SIZE_SMALL, bold=ov["is_action"]))
        p.drawText(ccx + 12, ccy + 4, str(ov["pos"]))

    def _draw_auto_overlay(self, p):
        """Draw the auto-tracker's view: detections, ROI, and a pos gauge."""
        ov = self._auto_overlay

        for det in ov.get("detections") or []:
            x, y, w, h = det.box
            color = _DET_COLORS.get(det.class_name, _DET_DEFAULT_COLOR)
            cx1, cy1 = self._frame_to_canvas(x, y)
            cx2, cy2 = self._frame_to_canvas(x + w, y + h)
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.setPen(QPen(color, 1, Qt.PenStyle.DotLine))
            p.drawRect(cx1, cy1, cx2 - cx1, cy2 - cy1)
            p.setPen(QPen(color))
            p.setFont(make_font(size=SIZE_SMALL))
            p.drawText(cx1 + 2, cy1 + 12, f"{det.class_name} {det.confidence:.2f}")

        belief = ov.get("belief")
        if belief is not None:
            x, y, w, h = belief
            cx1, cy1 = self._frame_to_canvas(x, y)
            cx2, cy2 = self._frame_to_canvas(x + w, y + h)
            memory_color = QColor(60, 200, 120)
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.setPen(QPen(memory_color, 2, Qt.PenStyle.DashLine))
            p.drawRect(cx1, cy1, cx2 - cx1, cy2 - cy1)
            p.setPen(QPen(memory_color))
            p.setFont(make_font(size=SIZE_SMALL))
            age = ov.get("belief_age_s")
            label = "anchor (memory)" if age is None else f"anchor (memory {age:.0f}s)"
            p.drawText(cx1 + 2, cy1 - 4, label)

        lock = ov.get("lock", "none")
        roi = ov.get("roi")
        if roi is not None:
            x, y, w, h = roi
            cx1, cy1 = self._frame_to_canvas(x, y)
            cx2, cy2 = self._frame_to_canvas(x + w, y + h)
            roi_color = _LOCK_COLORS.get(lock, _DET_DEFAULT_COLOR)
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.setPen(QPen(roi_color, 2))
            p.drawRect(cx1, cy1, cx2 - cx1, cy2 - cy1)
            p.setPen(QPen(roi_color))
            p.setFont(make_font(size=SIZE_SMALL, bold=True))
            p.drawText(cx1 + 3, cy2 - 5, _LOCK_LABELS.get(lock, ""))

        # Position gauge along the right edge of the canvas
        gauge_x = self.width() - 26
        gauge_top, gauge_h = 20, max(60, self.height() - 60)
        p.setPen(QPen(BORDER_SUBTLE))
        p.setBrush(QBrush(BG_SECONDARY))
        p.drawRect(gauge_x, gauge_top, 12, gauge_h)
        if ov.get("active"):
            pos = ov.get("pos", 50)
            marker_y = gauge_top + int((100 - pos) / 100 * gauge_h)
            color = QColor(80, 255, 80)
            if ov.get("is_action"):
                p.setPen(QPen(color, 3))
                p.setBrush(Qt.BrushStyle.NoBrush)
                p.drawEllipse(gauge_x - 4, marker_y - 8, 20, 16)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QBrush(color))
            p.drawRect(gauge_x + 1, marker_y - 2, 11, 5)
            p.setPen(QPen(color))
            p.setFont(make_font(size=SIZE_SMALL, bold=ov.get("is_action", False)))
            p.drawText(gauge_x - 24, marker_y + 4, f"{ov.get('pos', 50):3d}")
        else:
            p.setPen(QPen(QColor(255, 150, 50)))
            p.setFont(make_font(size=SIZE_SMALL, bold=True))
            p.drawText(gauge_x - 60, gauge_top + 14, "no target")

    def _draw_gt(self, p):
        """Draw ground truth annotation points."""
        gt = self._gt
        # Axis line from base to tip
        if gt.get("tip") and gt.get("base"):
            tx, ty = self._frame_to_canvas(*gt["tip"])
            bx, by = self._frame_to_canvas(*gt["base"])
            p.setPen(QPen(_AXIS_COLOR, 2, Qt.PenStyle.DashLine))
            p.drawLine(tx, ty, bx, by)
            # Tip marker
            p.setPen(QPen(Qt.GlobalColor.white, 1))
            p.setBrush(QBrush(_TIP_COLOR))
            p.drawEllipse(tx - 5, ty - 5, 10, 10)
            p.setPen(QPen(_TIP_COLOR))
            p.setFont(make_font(size=SIZE_SMALL))
            p.drawText(tx + 10, ty + 4, "T")
            # Base marker
            p.setPen(QPen(Qt.GlobalColor.white, 1))
            p.setBrush(QBrush(_BASE_COLOR))
            p.drawEllipse(bx - 5, by - 5, 10, 10)
            p.setPen(QPen(_BASE_COLOR))
            p.setFont(make_font(size=SIZE_SMALL))
            p.drawText(bx + 10, by + 4, "B")

        # Contact marker
        contact = gt.get("contact")
        contact_color = QColor(80, 255, 80)
        if contact is not None:
            ccx, ccy = self._frame_to_canvas(*contact)
            if gt.get("is_action"):
                p.setPen(QPen(contact_color, 3))
                p.setBrush(QBrush(Qt.GlobalColor.transparent))
                p.drawEllipse(ccx - 8, ccy - 8, 16, 16)
            else:
                p.setPen(QPen(Qt.GlobalColor.white, 1))
                p.setBrush(QBrush(contact_color))
                p.drawEllipse(ccx - 5, ccy - 5, 10, 10)
            # Pos label
            pos = gt.get("pos", "?")
            p.setPen(QPen(contact_color))
            p.setFont(make_font(size=SIZE_BODY, bold=gt.get("is_action", False)))
            p.drawText(ccx + 12, ccy + 5, str(pos))
        else:
            # No contact indicator — show "—" near the axis midpoint
            if gt.get("tip") and gt.get("base"):
                mx = (tx + bx) // 2
                my = (ty + by) // 2
                p.setPen(QPen(QColor(255, 150, 50)))
                p.setFont(make_font(size=SIZE_BODY, bold=True))
                p.drawText(mx + 10, my + 5, "no contact")

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
        self.scene_positions = {}  # idx -> TrackingResult (for debug overlay)
        self.ground_truth = {}    # idx -> {frame_idx: {tip, base, contact, is_action}}
        self.auto_result = None   # PipelineResult from the YOLO+flow pipeline
        self._auto_det_frames = []  # sorted detection frame indices (cache)
        self.current_frame_idx = 0
        self.placing = None
        self.label_session = False
        self._session_target = None   # frame currently being decided
        self._session_undo = []       # [(scene_idx, frame_idx), ...] this session
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
        QShortcut(QKeySequence("]"), self, self._next_action_frame)
        QShortcut(QKeySequence("["), self, self._prev_action_frame)
        QShortcut(QKeySequence("A"), self, self._toggle_gt_action)
        QShortcut(QKeySequence("N"), self, self._gt_no_contact)
        QShortcut(QKeySequence("C"), self, self._place_gt_contact)
        QShortcut(QKeySequence("G"), self, self._session_skip)
        QShortcut(QKeySequence(Qt.Key.Key_Home), self, self._session_home)
        QShortcut(QKeySequence("U"), self, self._session_undo_last)

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

    def _action_frames_for_current_scene(self):
        """Return sorted list of frame indices for actions in the current scene."""
        idx = self._current_scene_idx()
        actions = self.scene_actions.get(idx, [])
        if not actions:
            return []
        return sorted(int(round(a["at"] / 1000 * self.fps)) for a in actions)

    def _next_action_frame(self):
        frames = self._action_frames_for_current_scene()
        if not frames:
            return
        for f in frames:
            if f > self.current_frame_idx:
                self._show_frame(f)
                return
        self._show_frame(frames[0])  # wrap

    def _prev_action_frame(self):
        frames = self._action_frames_for_current_scene()
        if not frames:
            return
        for f in reversed(frames):
            if f < self.current_frame_idx:
                self._show_frame(f)
                return
        self._show_frame(frames[-1])  # wrap

    # ── Ground truth annotation ────────────────────────────────────

    def _ensure_gt_frame(self):
        """Ensure the current frame has a GT entry, inheriting from nearest."""
        idx = self._current_scene_idx()
        frame = self.current_frame_idx
        if idx not in self.ground_truth:
            self.ground_truth[idx] = {}
        gt_scene = self.ground_truth[idx]
        if frame in gt_scene:
            return gt_scene[frame]
        # Inherit from nearest annotated frame
        if gt_scene:
            nearest = min(gt_scene.keys(), key=lambda f: abs(f - frame))
            entry = dict(gt_scene[nearest])  # shallow copy
        elif idx in self.scene_axes:
            axis = self.scene_axes[idx]
            entry = {"tip": axis.tip, "base": axis.base, "contact": None, "is_action": False}
        else:
            # Contact-only label (auto sessions have no manual axis; the
            # trainer derives the axis from the automated anchor track)
            entry = {"tip": None, "base": None, "contact": None, "is_action": False}
        gt_scene[frame] = entry
        return entry

    def _get_gt_for_frame(self, scene_idx, frame_idx):
        """Get GT data for display (read-only, does not create entries)."""
        gt_scene = self.ground_truth.get(scene_idx, {})
        if frame_idx in gt_scene:
            gt = dict(gt_scene[frame_idx])
        elif gt_scene:
            nearest = min(gt_scene.keys(), key=lambda f: abs(f - frame_idx))
            gt = dict(gt_scene[nearest])
            gt["is_action"] = False  # inherited frames aren't action frames
        else:
            return None
        # Compute pos from tip/base/contact
        if gt.get("contact") and gt.get("tip") and gt.get("base"):
            from scripture.cotracker_tracking import compute_pos_from_points
            gt["pos"] = compute_pos_from_points(gt["base"], gt["tip"], gt["contact"])
        else:
            gt["pos"] = "—"
        return gt

    def _toggle_gt_action(self):
        entry = self._ensure_gt_frame()
        if entry is None:
            return
        entry["is_action"] = not entry["is_action"]
        self._mark_dirty()
        self._show_frame(self.current_frame_idx)

    def _gt_no_contact(self):
        """N: in a label session, record an explicit no-contact and advance;
        otherwise toggle no-contact on the current frame."""
        entry = self._ensure_gt_frame()
        if entry is None:
            return
        if self.label_session:
            entry["contact"] = None
            self._mark_dirty()
            self._session_after_label(self.current_frame_idx)
            return
        if entry["contact"] is None:
            # Restore contact — put it at midpoint of axis
            if entry.get("tip") and entry.get("base"):
                mx = (entry["tip"][0] + entry["base"][0]) // 2
                my = (entry["tip"][1] + entry["base"][1]) // 2
                entry["contact"] = (mx, my)
        else:
            entry["contact"] = None
        self._mark_dirty()
        self._show_frame(self.current_frame_idx)

    def _place_gt_contact(self):
        """Enter contact placement mode — next click sets the contact point."""
        self.placing = "contact"
        self._set_status("Click to place CONTACT point (or press C again to cancel)")

    # ── Label session (sparse GT clicking) ─────────────────────────

    def _session_schedule(self):
        from scripture.annotate import schedule_frames

        idx = self._current_scene_idx()
        if not self.scenes:
            return []
        scene = self.scenes[idx]
        stride = max(1, int(2 * self.fps)) if self.fps else 60
        return schedule_frames(scene.start_frame, scene.end_frame - 1, stride)

    def _session_annotated(self):
        idx = self._current_scene_idx()
        return set(self.ground_truth.get(idx, {}).keys())

    def _toggle_label_session(self):
        self.label_session = self.btn_label_session.isChecked()
        if self.label_session:
            self._session_goto_next(after=self.current_frame_idx)
        else:
            self._session_target = None
            self._set_status("Label session paused — progress saves with the project.")

    def _session_goto_next(self, after):
        """Make the next unlabeled scheduled frame past `after` the target."""
        from scripture.annotate import next_scheduled

        if not self.label_session or not self.scenes:
            return
        schedule = self._session_schedule()
        annotated = self._session_annotated()
        nxt = next_scheduled(schedule, annotated, after)
        done, total = len(annotated & set(schedule)), len(schedule)
        if nxt is None:
            self.btn_label_session.setChecked(False)
            self.label_session = False
            self._session_target = None
            self._set_status(f"Label session complete: {done}/{total} frames. "
                             "Save the project to keep them.")
            return
        self._session_target = nxt
        self._show_frame(nxt)
        self._set_status(f"Label {done}/{total} · click = contact · N = none · "
                         "G = skip · U = undo · arrows = scrub · Home = back")

    def _session_after_label(self, labeled_frame):
        """Record for undo, then target the next frame — never skipping a
        still-unlabeled target the user labeled around."""
        self._session_undo.append((self._current_scene_idx(), labeled_frame))
        anchor = self._session_target if self._session_target is not None else labeled_frame
        self._session_goto_next(after=anchor - 1)

    def _session_skip(self):
        """G: give up on the current target and move to the next one."""
        if not self.label_session:
            return
        anchor = self._session_target if self._session_target is not None \
            else self.current_frame_idx
        self._session_goto_next(after=anchor)

    def _session_home(self):
        """Home: return to the frame being decided after scrubbing around."""
        if self.label_session and self._session_target is not None:
            self._show_frame(self._session_target)

    def _session_undo_last(self):
        """U: delete the most recent label and go back to that frame."""
        if not self.label_session or not self._session_undo:
            return
        scene_idx, frame = self._session_undo.pop()
        self.ground_truth.get(scene_idx, {}).pop(frame, None)
        self._mark_dirty()
        self._session_target = frame
        self._show_frame(frame)
        schedule = self._session_schedule()
        done = len(self._session_annotated() & set(schedule))
        self._set_status(f"Undid label at frame {frame} — decide it again "
                         f"({done}/{len(schedule)} done)")

    def _reset_scene_labels(self):
        """Clear every label in the current scene (button, with confirm)."""
        idx = self._current_scene_idx()
        n = len(self.ground_truth.get(idx, {}))
        if not n:
            self._set_status("No labels in this scene to reset.")
            return
        reply = QMessageBox.question(
            self, "Reset labels?",
            f"Delete all {n} labeled frames in this scene?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        self.ground_truth.pop(idx, None)
        self._session_undo = [u for u in self._session_undo if u[0] != idx]
        self._mark_dirty()
        self._show_frame(self.current_frame_idx)
        if self.label_session:
            self._session_goto_next(after=self.current_frame_idx)
        self._set_status(f"Cleared {n} labels in scene {idx}. "
                         "Save the project to make it permanent.")

    def _on_gt_point_dragged(self, which, fx, fy):
        """Handle dragging of a GT point (tip, base, or contact)."""
        idx = self._current_scene_idx()
        entry = self._ensure_gt_frame()
        if entry is None:
            return
        entry[which] = (fx, fy)
        self._mark_dirty()
        # Update display without re-reading the frame
        gt = self._get_gt_for_frame(idx, self.current_frame_idx)
        self.canvas.set_gt(gt)

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
        self.addToolBar(tb)

        # Left pad to match central widget margin
        left_pad = QWidget()
        left_pad.setFixedWidth(MARGIN_STANDARD)
        tb.addWidget(left_pad)

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

        # Spacer — fills toolbar when progress bar is hidden
        self._spacer = QWidget()
        self._spacer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self._spacer_action = tb.addWidget(self._spacer)

        # Progress bar + abort button embedded in toolbar (no layout shift)
        self._progress_sep = tb.addSeparator()
        self._progress_sep.setVisible(False)
        self.progress_bar = QProgressBar()
        self.progress_bar.setStyleSheet(_PROGRESS_STYLE)
        self.progress_bar.setFixedHeight(20)
        self.progress_bar.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self._progress_action = tb.addWidget(self.progress_bar)
        self._progress_action.setVisible(False)
        self._abort_action = QAction(qta.icon("fa5s.stop", color="#ff6666"), "Abort", self)
        self._abort_action.triggered.connect(self._abort_processing)
        self._abort_action.setVisible(False)
        tb.addAction(self._abort_action)

        # Right pad
        right_pad = QWidget()
        right_pad.setFixedWidth(MARGIN_STANDARD)
        tb.addWidget(right_pad)

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

        self.btn_auto_process = QPushButton("Auto Process (YOLO)")
        self.btn_auto_process.setStyleSheet(_BTN_STYLE)
        self.btn_auto_process.clicked.connect(self._auto_process)
        bottom.addWidget(self.btn_auto_process)

        self.btn_process_all = QPushButton("Process All")
        self.btn_process_all.setStyleSheet(_BTN_STYLE)
        self.btn_process_all.clicked.connect(self._process_all)
        bottom.addWidget(self.btn_process_all)

        self.btn_label_session = QPushButton("Label Session")
        self.btn_label_session.setStyleSheet(_BTN_STYLE)
        self.btn_label_session.setCheckable(True)
        self.btn_label_session.clicked.connect(self._toggle_label_session)
        bottom.addWidget(self.btn_label_session)

        self.btn_reset_labels = QPushButton("Reset Labels")
        self.btn_reset_labels.setStyleSheet(_BTN_STYLE)
        self.btn_reset_labels.clicked.connect(self._reset_scene_labels)
        bottom.addWidget(self.btn_reset_labels)

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
        old_positions = dict(self.scene_positions)
        self.scenes = scenes_from_splits(self.splits, self.total_frames)
        if clear_annotations:
            self.scene_axes.clear()
            self.scene_actions.clear()
            self.scene_positions.clear()
            for _oi, axis in old_axes.items():
                for ni, sc in enumerate(self.scenes):
                    if sc.start_frame <= axis.frame < sc.end_frame:
                        self.scene_axes[ni] = axis
                        if _oi in old_actions:
                            self.scene_actions[ni] = old_actions[_oi]
                        if _oi in old_positions:
                            self.scene_positions[ni] = old_positions[_oi]
                        break
        # Auto actions are global; re-bucket them into the new scene layout
        if self.auto_result is not None:
            self.scene_actions = actions_by_scene(
                self.auto_result.actions, self.scenes, self.fps)

    def _update_timeline(self):
        self.timeline.set_state(
            self.scenes, self.scene_axes, self.scene_actions,
            self.splits, self.total_frames, self.current_frame_idx,
            self.ground_truth,
        )

    def _update_info(self):
        if self.total_frames == 0:
            self._set_status("Open a video to begin.")
            return
        idx = self._current_scene_idx()
        sc = self.scenes[idx]
        sc_len = sc.end_frame - sc.start_frame
        sc_info = (
            f"scene {idx+1} / {len(self.scenes)} : "
            f"frames {sc.start_frame} \u2013 {sc.end_frame} "
            f"({self._format_time(sc.start_frame)} \u2013 {self._format_time(sc.end_frame)}), "
            f"total {sc_len} frames ({self._format_time(sc_len)} duration)"
        )
        fr_info = (
            f"frame {self.current_frame_idx} / {self.total_frames} "
            f"({self._format_time(self.current_frame_idx)}  / {self._format_time(self.total_frames)})"
        )
        self._set_status(f"{sc_info}  |  {fr_info}")

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
        self.scene_positions.clear()
        self._set_auto_result(None)
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

    def _build_overlay(self, scene_idx, frame_idx):
        """Build debug overlay dict for a processed scene's frame.

        Uses full per-frame positions when available (just processed), or
        interpolates from stored actions (loaded session).
        """
        axis = self.scene_axes[scene_idx]
        scene = self.scenes[scene_idx]
        frame_ms = frame_idx / self.fps * 1000
        actions = self.scene_actions.get(scene_idx, [])
        half_frame_ms = 500 / self.fps

        local_idx = frame_idx - scene.start_frame
        result = self.scene_positions.get(scene_idx)
        if result is not None:
            if local_idx < 0 or local_idx >= len(result.positions):
                return None
            pos_frac = float(result.positions[local_idx])
            pos_100 = int(round(pos_frac * 100))
            ts = result.timestamps_ms[local_idx]
            is_action = any(abs(a["at"] - ts) < half_frame_ms for a in actions)
            if is_action:
                for a in actions:
                    if abs(a["at"] - ts) < half_frame_ms:
                        pos_100 = a["pos"]
                        break
        elif actions:
            # Interpolate from action list only
            pos_100, is_action = self._interpolate_actions(actions, frame_ms, half_frame_ms)
            pos_frac = pos_100 / 100.0
        else:
            return None

        # Use per-frame tracked coordinates when available
        if (result is not None
                and result.tip_coords is not None
                and result.base_coords is not None
                and 0 <= local_idx < len(result.tip_coords)):
            tip = result.tip_coords[local_idx]
            base = result.base_coords[local_idx]
        else:
            tip = np.array(axis.tip, dtype=np.float64)
            base = np.array(axis.base, dtype=np.float64)

        # Contact point: lerp between base (pos=0) and tip (pos=1)
        contact = base + pos_frac * (tip - base)
        contact_pt = (int(round(contact[0])), int(round(contact[1])))

        frame_axis = AxisDefinition(
            tip=(int(round(tip[0])), int(round(tip[1]))),
            base=(int(round(base[0])), int(round(base[1]))),
            frame=axis.frame,
        )
        # Compute direction of motion: +1 = moving toward tip, -1 = toward base, 0 = still
        direction = 0
        if result is not None and 0 < local_idx < len(result.positions):
            delta = result.positions[local_idx] - result.positions[local_idx - 1]
            if abs(delta) > 0.001:
                direction = 1 if delta > 0 else -1

        return {
            "axis": frame_axis,
            "contact_pt": contact_pt,
            "pos": pos_100,
            "is_action": is_action,
            "direction": direction,
        }

    @staticmethod
    def _interpolate_actions(actions, frame_ms, half_frame_ms):
        """Interpolate pos from action list for a given timestamp."""
        if not actions:
            return 50, False
        is_action = False
        for a in actions:
            if abs(a["at"] - frame_ms) < half_frame_ms:
                return a["pos"], True
        # Before first action
        if frame_ms <= actions[0]["at"]:
            return actions[0]["pos"], False
        # After last action
        if frame_ms >= actions[-1]["at"]:
            return actions[-1]["pos"], False
        # Between two actions — linear interpolation
        for i in range(len(actions) - 1):
            a0, a1 = actions[i], actions[i + 1]
            if a0["at"] <= frame_ms <= a1["at"]:
                t = (frame_ms - a0["at"]) / (a1["at"] - a0["at"])
                pos = a0["pos"] + t * (a1["pos"] - a0["pos"])
                return int(round(pos)), False
        return 50, False

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

        # Auto-tracking overlay takes precedence when the auto pipeline ran
        if self.auto_result is not None:
            self.canvas.set_auto_overlay(self._build_auto_overlay(frame_idx))
        else:
            self.canvas.set_auto_overlay(None)

        # Debug overlay for processed scenes (works with full positions or just actions)
        if idx in self.scene_axes and (idx in self.scene_positions or idx in self.scene_actions):
            self.canvas.set_overlay(self._build_overlay(idx, frame_idx))
        else:
            self.canvas.set_overlay(None)

        # Ground truth annotation layer
        if idx in self.ground_truth:
            self.canvas.set_gt(self._get_gt_for_frame(idx, frame_idx))
        else:
            self.canvas.set_gt(None)

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
                self.scene_positions.pop(idx, None)

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
                self.scene_positions.pop(idx, None)
        if which == "tip":
            self.pending_tip = None
        else:
            self.pending_base = None
        self._mark_dirty()
        self._show_frame(self.current_frame_idx)

    def _on_canvas_click(self, fx, fy):
        if not self.scenes:
            return
        if self.label_session and not self.placing:
            entry = self._ensure_gt_frame()
            if entry is not None:
                entry["contact"] = (fx, fy)
                self._mark_dirty()
                self._session_after_label(self.current_frame_idx)
            return
        if not self.placing:
            return
        idx = self._current_scene_idx()

        if self.placing == "contact":
            entry = self._ensure_gt_frame()
            if entry is not None:
                entry["contact"] = (fx, fy)
                self._mark_dirty()
                self._show_frame(self.current_frame_idx)
            self.placing = None
            self._set_status("Contact point placed.")
            return

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
                self.scene_positions.pop(idx, None)

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
        # GT point drag (contact, or tip/base while GT is active)
        if which in ("contact", "tip", "base") and idx in self.ground_truth:
            self._on_gt_point_dragged(which, fx, fy)
            return
        if idx in self.scene_axes:
            old = self.scene_axes[idx]
            if which == "tip":
                self.scene_axes[idx] = AxisDefinition(tip=(fx, fy), base=old.base, frame=old.frame)
            else:
                self.scene_axes[idx] = AxisDefinition(tip=old.tip, base=(fx, fy), frame=old.frame)
            if idx in self.scene_actions:
                del self.scene_actions[idx]
                self.scene_positions.pop(idx, None)
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

    def _show_progress_ui(self, maximum, fmt):
        self._spacer_action.setVisible(False)
        self._progress_sep.setVisible(True)
        self._progress_action.setVisible(True)
        self._abort_action.setVisible(True)
        self.progress_bar.setMaximum(maximum)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat(fmt)
        self.btn_process_all.setEnabled(False)
        self.btn_auto_process.setEnabled(False)
        self._process_start_time = time.monotonic()

    def _hide_progress_ui(self):
        self._spacer_action.setVisible(True)
        self._progress_sep.setVisible(False)
        self._progress_action.setVisible(False)
        self._abort_action.setVisible(False)
        self.btn_process_all.setEnabled(True)
        self.btn_auto_process.setEnabled(True)

    def _start_processing(self, jobs):
        tf = sum(sc.end_frame - sc.start_frame for _, sc, _ in jobs)
        # Indeterminate (pulsing) during CoTracker3
        self._show_progress_ui(0, "Tracking axis with CoTracker3\u2026")
        self._process_total_frames = tf
        self._worker = ProcessWorker(self.video_path, jobs, self.fps)
        self._worker.frame_progress.connect(self._on_frame_progress)
        self._worker.scene_done.connect(self._on_scene_done)
        self._worker.error.connect(self._on_process_error)
        self._worker.finished.connect(self._on_process_finished)
        self._worker.start()

    def _on_frame_progress(self, done):
        if self.progress_bar.maximum() == 0:
            self.progress_bar.setMaximum(self._process_total_frames)
            self.progress_bar.setFormat("Detecting contact points\u2026 %p%")
        self.progress_bar.setValue(done)

    def _on_scene_done(self, idx, actions, tracking_result):
        self.scene_actions[idx] = actions
        self.scene_positions[idx] = tracking_result
        self._mark_dirty()
        self._update_timeline()

    def _on_process_error(self, idx, msg):
        self._set_status(f"Error scene {idx+1}: {msg}")
        QMessageBox.warning(self, f"Processing Error (Scene {idx+1})", msg)

    def _on_process_finished(self):
        el = time.monotonic() - self._process_start_time
        self._hide_progress_ui()
        self._worker = None
        total = sum(len(a) for a in self.scene_actions.values())
        self._set_status(f"Done \u2014 {total} actions in {self._fmt_duration(el)}.")

    def _abort_processing(self):
        if self._worker and self._worker.isRunning():
            self._worker.terminate()
            self._worker.wait()
            self._worker = None
        self._hide_progress_ui()
        self._set_status("Processing aborted.")

    # \u2500\u2500 Automatic YOLO+flow processing \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500

    def _auto_process(self):
        if self._is_processing():
            self._set_status("Already processing \u2014 wait for it to finish.")
            return
        if not self.video_path:
            QMessageBox.warning(self, "No video", "Open a video first.")
            return
        self._show_progress_ui(self.total_frames, "Auto tracking\u2026 %p%")
        self._worker = AutoProcessWorker(self.video_path)
        self._worker.frame_progress.connect(self.progress_bar.setValue)
        self._worker.done.connect(self._on_auto_done)
        self._worker.error.connect(self._on_auto_error)
        self._worker.finished.connect(self._on_auto_finished)
        self._worker.start()

    def _set_auto_result(self, result):
        """Install a pipeline result: bucket actions and refresh caches."""
        self.auto_result = result
        self._auto_det_frames = sorted(result.signal.detections.keys()) if result else []
        self._auto_last_anchor = None
        if result is not None:
            self.scene_actions = actions_by_scene(result.actions, self.scenes, self.fps)
            self.scene_positions.clear()
            # For each frame, the most recent frame with a direct sighting
            last_anchor = np.full(len(result.signal.lock), -1, dtype=np.int64)
            last = -1
            for i, state in enumerate(result.signal.lock):
                if state == "anchor":
                    last = i
                last_anchor[i] = last
            self._auto_last_anchor = last_anchor

    def _on_auto_done(self, result):
        self._set_auto_result(result)
        self._mark_dirty()
        self._update_timeline()
        self._show_frame(self.current_frame_idx)

    def _on_auto_error(self, msg):
        self._set_status(f"Auto processing error: {msg}")
        QMessageBox.warning(self, "Auto Processing Error", msg)

    def _on_auto_finished(self):
        el = time.monotonic() - self._process_start_time
        self._hide_progress_ui()
        self._worker = None
        if self.auto_result is not None:
            self._set_status(
                f"Auto done \u2014 {len(self.auto_result.actions)} actions "
                f"in {self._fmt_duration(el)}.")

    def _build_auto_overlay(self, frame_idx):
        """Overlay dict showing what the auto tracker saw at this frame."""
        r = self.auto_result
        local = frame_idx - r.start_frame
        if local < 0 or local >= len(r.positions):
            return None

        # Most recent detection result within a couple of detection cycles
        detections = None
        if self._auto_det_frames:
            i = bisect.bisect_right(self._auto_det_frames, local) - 1
            if i >= 0 and local - self._auto_det_frames[i] <= 6:
                detections = r.signal.detections[self._auto_det_frames[i]]

        frame_ms = frame_idx / self.fps * 1000
        half_frame_ms = 500 / self.fps
        idx = self._current_scene_idx()
        actions = self.scene_actions.get(idx, [])
        is_action = any(abs(a["at"] - frame_ms) < half_frame_ms for a in actions)

        lock = r.signal.lock[local]
        beliefs = r.signal.beliefs
        belief = (beliefs[local] if local < len(beliefs)
                  and lock in ("contact", "coast") else None)
        belief_age_s = None
        if belief is not None and self._auto_last_anchor is not None:
            last = self._auto_last_anchor[local]
            if last >= 0:
                belief_age_s = (local - last) / self.fps
        return {
            "roi": r.signal.rois[local],
            "detections": detections,
            "pos": int(round(r.positions[local])),
            "active": lock != "none",
            "lock": lock,
            # Show the remembered anchor whenever it isn't directly seen
            "belief": belief,
            "belief_age_s": belief_age_s,
            "is_action": is_action,
        }

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
            self.scene_positions.pop(idx, None)
            self._mark_dirty()
            self._update_timeline()
            self._set_status(f"Discarded actions for scene {idx+1}.")

    # ── Project save/load ──────────────────────────────────────────

    def _build_state(self):
        axes = {str(i): {"tip": list(a.tip), "base": list(a.base), "frame": a.frame}
                for i, a in self.scene_axes.items()}
        acts = {str(i): a for i, a in self.scene_actions.items()}
        # Serialize per-frame tracking coordinates
        tracking = {}
        for i, result in self.scene_positions.items():
            entry = {
                "timestamps_ms": result.timestamps_ms.tolist(),
                "positions": result.positions.tolist(),
            }
            if result.tip_coords is not None:
                entry["tip_coords"] = result.tip_coords.tolist()
            if result.base_coords is not None:
                entry["base_coords"] = result.base_coords.tolist()
            tracking[str(i)] = entry
        # Serialize ground truth annotations
        gt = {}
        for scene_idx, frames in self.ground_truth.items():
            gt_frames = {}
            for frame_idx, entry in frames.items():
                gt_frames[str(frame_idx)] = {
                    "tip": list(entry["tip"]) if entry.get("tip") else None,
                    "base": list(entry["base"]) if entry.get("base") else None,
                    "contact": list(entry["contact"]) if entry.get("contact") else None,
                    "is_action": entry.get("is_action", False),
                }
            gt[str(scene_idx)] = gt_frames
        return {
            "video_path": self.video_path, "splits": self.splits,
            "axes": axes, "actions": acts, "tracking": tracking,
            "ground_truth": gt,
            "auto": (pipeline_result_to_state(self.auto_result)
                     if self.auto_result is not None else None),
            "current_frame": self.current_frame_idx,
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
        self.scene_positions.clear()
        for k, v in state.get("actions", {}).items():
            self.scene_actions[int(k)] = v
        for k, v in state.get("tracking", {}).items():
            tip_c = np.array(v["tip_coords"]) if "tip_coords" in v else None
            base_c = np.array(v["base_coords"]) if "base_coords" in v else None
            self.scene_positions[int(k)] = TrackingResult(
                timestamps_ms=np.array(v["timestamps_ms"]),
                positions=np.array(v["positions"]),
                tip_coords=tip_c,
                base_coords=base_c,
            )

        # Load ground truth annotations
        self.ground_truth.clear()
        for scene_k, frames in state.get("ground_truth", {}).items():
            scene_gt = {}
            for frame_k, entry in frames.items():
                scene_gt[int(frame_k)] = {
                    "tip": tuple(entry["tip"]) if entry.get("tip") else None,
                    "base": tuple(entry["base"]) if entry.get("base") else None,
                    "contact": tuple(entry["contact"]) if entry.get("contact") else None,
                    "is_action": entry.get("is_action", False),
                }
            self.ground_truth[int(scene_k)] = scene_gt

        # Load auto-pipeline diagnostics
        auto_state = state.get("auto")
        self._set_auto_result(
            pipeline_result_from_state(auto_state) if auto_state else None)

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

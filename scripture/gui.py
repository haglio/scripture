from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from app_support.funscript import write as write_funscript
from app_support.overlay import overlay_value
from PyQt6.QtCore import QPointF, QSize, Qt, pyqtSignal
from PyQt6.QtGui import (
    QAction,
    QBrush,
    QColor,
    QCursor,
    QImage,
    QKeySequence,
    QPainter,
    QPen,
    QPixmap,
    QPolygonF,
    QShortcut,
)
from PyQt6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMenu,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QToolBar,
    QVBoxLayout,
    QWidget,
)
from shared_ui import palette
from shared_ui.chrome import toolbar_rules
from shared_ui.colors import (
    BG_BUTTON,
    BG_BUTTON_ACTIVE,
    BG_PRIMARY,
    BG_SECONDARY,
    BG_TERTIARY,
    BLUE,
    BORDER_DEFAULT,
    BORDER_SUBTLE,
    GREEN,
    RED,
    TEXT_MUTED,
    TEXT_PRIMARY,
    WHITE,
    YELLOW,
)
from shared_ui.fonts import SIZE_BODY, SIZE_SMALL, make_font
from shared_ui.icons import qt_icon
from shared_ui.preview import Preview, window_title
from shared_ui.preview_icon import app_icon
from shared_ui.spacing import BUTTON_ICON, BUTTON_RADIUS_HUD, GAP_MEDIUM, GAP_SMALL, MARGIN_STANDARD

from scripture.annotations import SceneAnnotations
from scripture.content import LOCAL_CONTENT, load_content
from scripture.cotracker_tracking import compute_pos_from_points
from scripture.editing_rules import (
    both_scenes_have_an_axis,
    moving_the_axis_needs_asking,
    next_frame,
    previous_frame,
)
from scripture.frames import frame_of_ms
from scripture.funscript import build_funscript
from scripture.label_session import LabelSession, schedule_frames
from scripture.motion_tracker import AxisDefinition
from scripture.overlay import FrameMoment, auto_overlay, tracked_overlay
from scripture.project import (
    PROJECTS_DIR,
    Project,
    load_project,
    project_from_state,
    provenance_of_actions,
    reopen_last_project,
    save_project,
    state_from_project,
)
from scripture.scene import Scene, actions_by_scene, scenes_from_splits
from scripture.tracking_workers import AutoProcessWorker, ProcessWorker
from scripture.video import NO_VIDEO, VideoSource

_ICON_COLOR = TEXT_PRIMARY
_QT_PULSES_A_BAR_WHOSE_MAXIMUM_IS = 0
_LAST_PROJECT_FILE = PROJECTS_DIR / ".last_session"

_PROGRESS_STYLE = f"""
    QProgressBar {{
        background: {BG_TERTIARY.name()};
        border: 1px solid {BORDER_SUBTLE.name()};
        border-radius: {BUTTON_RADIUS_HUD}px;
        text-align: center;
        color: {TEXT_PRIMARY.name()};
        height: 18px;
    }}
    QProgressBar::chunk {{
        background: {BLUE.name()};
        border-radius: {BUTTON_RADIUS_HUD}px;
    }}
"""

_SCENE_EMPTY = BG_SECONDARY
_SCENE_ANNOTATED = BG_BUTTON
_SCENE_PROCESSED = BG_BUTTON_ACTIVE
_SCENE_BORDER = BORDER_SUBTLE
_REP_FRAME_COLOR = YELLOW
_HANDLE_HEIGHT = 8
_MARKER_SLACK = 5
_GRAB_REACH = 10
_SECONDS_PER_MINUTE = 60
_MINUTES_PER_HOUR = 60
_TIP_COLOR = RED
_BASE_COLOR = BLUE
_AXIS_COLOR = YELLOW


_ACTION_DOT_COLOR = QColor(*palette.GREEN, 180)

_DET_COLORS = {
    name: QColor(*palette.nearest(rgb))
    for name, rgb in overlay_value(load_content(), "class_colors", path=LOCAL_CONTENT).items()
}
_DET_DEFAULT_COLOR = BORDER_DEFAULT

_LOCK_COLORS = {
    "anchor": GREEN,
    "contact": YELLOW,
    "coast": RED,
}
_LOCK_LABELS = {
    "anchor": "lock: anchor",
    "contact": "lock: contact",
    "coast": "coasting",
}


@dataclass(frozen=True)
class TimelineState:
    scenes: list = field(default_factory=list)
    scene_axes: dict = field(default_factory=dict)
    scene_actions: dict = field(default_factory=dict)
    splits: list = field(default_factory=list)
    total_frames: int = 0
    current_frame: int = 0
    labels: dict = field(default_factory=dict)
    fps: float = 30.0


def action_dot_frames(state: TimelineState) -> list[int]:
    labeled = [frame for frames in state.labels.values()
               for frame, entry in frames.items() if entry.get("is_action")]
    computed = [frame_of_ms(action["at"], state.fps)
                for scene_idx, actions in state.scene_actions.items()
                if scene_idx not in state.labels
                for action in actions]
    return labeled + computed


class TimelineWidget(QWidget):
    frame_changed = pyqtSignal(int)
    context_menu_requested = pyqtSignal(int)

    def __init__(self):
        super().__init__()
        self.setFixedHeight(28 + _HANDLE_HEIGHT)
        self.setMinimumWidth(100)
        self._state = TimelineState()
        self.total_frames = 0
        self._dragging = False
        self._zoom = 1.0
        self._scroll = 0.0
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)

    def set_state(self, state: TimelineState):
        self._state = state
        self.total_frames = state.total_frames
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

        self._paint_scenes(p, bar_y, bar_h)
        self._paint_action_dots(p, bar_y + bar_h // 2)
        self._paint_markers(p, bar_y, bar_h)

        cx = self._frame_to_x(self._state.current_frame)
        p.setPen(QPen(WHITE, 2))
        p.drawLine(cx, bar_y, cx, bar_y + bar_h)

        p.setBrush(Qt.BrushStyle.NoBrush)
        p.setPen(QPen(BORDER_SUBTLE))
        p.drawRect(0, bar_y, w - 1, bar_h - 1)
        p.end()

    def _paint_scenes(self, p, bar_y, bar_h):
        state = self._state
        for i, scene in enumerate(state.scenes):
            x1 = self._frame_to_x(scene.start_frame)
            x2 = self._frame_to_x(scene.end_frame)
            if x2 < 0 or x1 > self.width():
                continue
            if i in state.scene_actions:
                color = _SCENE_PROCESSED
            elif i in state.scene_axes:
                color = _SCENE_ANNOTATED
            else:
                color = _SCENE_EMPTY
            p.fillRect(x1, bar_y, x2 - x1, bar_h, color)

    def _paint_action_dots(self, p, mid_y):
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(_ACTION_DOT_COLOR))
        for frame in action_dot_frames(self._state):
            ax = self._frame_to_x(frame)
            if 0 <= ax <= self.width():
                p.drawEllipse(ax - 2, mid_y - 2, 4, 4)

    def _paint_markers(self, p, bar_y, bar_h):
        state = self._state
        markers = [(split, _SCENE_BORDER, 3) for split in state.splits]
        markers += [(axis.frame, _REP_FRAME_COLOR, 2) for axis in state.scene_axes.values()]
        for frame, color, width in markers:
            x = self._frame_to_x(frame)
            if -_MARKER_SLACK <= x <= self.width() + _MARKER_SLACK:
                p.setPen(QPen(color, width))
                p.drawLine(x, bar_y, x, bar_y + bar_h)
                self._draw_handle(p, x, color)

    def _draw_handle(self, p, x, color):
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(color))
        p.drawPolygon(QPolygonF([
            QPointF(x - 4, 0), QPointF(x + 4, 0), QPointF(x, _HANDLE_HEIGHT),
        ]))

    def _nearest_handle_frame(self, x, y):
        if y > _HANDLE_HEIGHT:
            return None
        state = self._state
        handles = [(self._frame_to_x(s), s) for s in state.splits]
        handles += [(self._frame_to_x(a.frame), a.frame) for a in state.scene_axes.values()]
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
            self.context_menu_requested.emit(frame)

    def mouseMoveEvent(self, event):
        if self._dragging and self.total_frames > 0:
            self.frame_changed.emit(self._x_to_frame(int(event.position().x())))

    def mouseReleaseEvent(self, event):
        self._dragging = False

    def wheelEvent(self, event):
        if self.total_frames == 0:
            return
        x = int(event.position().x())
        frame_at_cursor = self._x_to_frame(x)
        frac_at_cursor = frame_at_cursor / self.total_frames

        delta = event.angleDelta().y()
        if delta > 0:
            self._zoom = min(self._zoom * 1.3, max(1.0, self.total_frames / 10))
        elif delta < 0:
            self._zoom = max(self._zoom / 1.3, 1.0)

        if self._zoom > 1.0:
            self._scroll = frac_at_cursor - x / (self._zoom * self.width())
        else:
            self._scroll = 0.0
        self._clamp_scroll()
        self.update()


def _viewport_rect(to_viewport, rect):
    x, y, w, h = rect
    return (*to_viewport(x, y), *to_viewport(x + w, y + h))


def _draw_detections(p, to_viewport, detections):
    for det in detections:
        color = _DET_COLORS.get(det.class_name, _DET_DEFAULT_COLOR)
        cx1, cy1, cx2, cy2 = _viewport_rect(to_viewport, det.rect)
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.setPen(QPen(color, 1, Qt.PenStyle.DotLine))
        p.drawRect(cx1, cy1, cx2 - cx1, cy2 - cy1)
        p.setPen(QPen(color))
        p.setFont(make_font(size=SIZE_SMALL))
        p.drawText(cx1 + 2, cy1 + 12, f"{det.class_name} {det.confidence:.2f}")


def _draw_remembered_anchor(p, to_viewport, overlay):
    cx1, cy1, cx2, cy2 = _viewport_rect(to_viewport, overlay["belief"])
    memory_color = GREEN
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.setPen(QPen(memory_color, 2, Qt.PenStyle.DashLine))
    p.drawRect(cx1, cy1, cx2 - cx1, cy2 - cy1)
    p.setPen(QPen(memory_color))
    p.setFont(make_font(size=SIZE_SMALL))
    age = overlay.get("belief_age_s")
    p.drawText(cx1 + 2, cy1 - 4, "anchor (memory)" if age is None else f"anchor (memory {age:.0f}s)")


def _draw_roi(p, to_viewport, overlay):
    lock = overlay.get("lock", "none")
    cx1, cy1, cx2, cy2 = _viewport_rect(to_viewport, overlay["roi"])
    roi_color = _LOCK_COLORS.get(lock, _DET_DEFAULT_COLOR)
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.setPen(QPen(roi_color, 2))
    p.drawRect(cx1, cy1, cx2 - cx1, cy2 - cy1)
    p.setPen(QPen(roi_color))
    p.setFont(make_font(size=SIZE_SMALL, bold=True))
    p.drawText(cx1 + 3, cy2 - 5, _LOCK_LABELS.get(lock, ""))


def _draw_position_gauge(p, overlay, viewport_size):
    width, height = viewport_size
    gauge_x = width - 26
    gauge_top, gauge_h = 20, max(60, height - 60)
    p.setPen(QPen(BORDER_SUBTLE))
    p.setBrush(QBrush(BG_SECONDARY))
    p.drawRect(gauge_x, gauge_top, 12, gauge_h)
    if not overlay.get("active"):
        p.setPen(QPen(YELLOW))
        p.setFont(make_font(size=SIZE_SMALL, bold=True))
        p.drawText(gauge_x - 60, gauge_top + 14, "no target")
        return
    pos = overlay.get("pos", 50)
    marker_y = gauge_top + int((100 - pos) / 100 * gauge_h)
    if overlay.get("is_action"):
        p.setPen(QPen(GREEN, 3))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawEllipse(gauge_x - 4, marker_y - 8, 20, 16)
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(QBrush(GREEN))
    p.drawRect(gauge_x + 1, marker_y - 2, 11, 5)
    p.setPen(QPen(GREEN))
    p.setFont(make_font(size=SIZE_SMALL, bold=overlay.get("is_action", False)))
    p.drawText(gauge_x - 24, marker_y + 4, f"{pos:3d}")


class Viewport(QWidget):
    clicked = pyqtSignal(int, int)
    point_dragged = pyqtSignal(str, int, int)
    context_menu_requested = pyqtSignal(int, int)

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
        self._dragging_point = None
        self._overlay = None
        self._label = None
        self._auto_overlay = None

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
        self._overlay = overlay
        self.update()

    def set_label(self, label):
        self._label = label
        self.update()

    def set_auto_overlay(self, overlay):
        self._auto_overlay = overlay
        self.update()

    def _display_scale(self):
        if not self._frame_w or not self._frame_h:
            return 1.0
        return min(self.width() / self._frame_w, self.height() / self._frame_h) * self._zoom

    def _frame_to_viewport(self, fx, fy):
        s = self._display_scale()
        dw, dh = int(self._frame_w * s), int(self._frame_h * s)
        ox, oy = (self.width() - dw) // 2, (self.height() - dh) // 2
        return int(fx * s) + ox, int(fy * s) + oy

    def _viewport_to_frame(self, cx, cy):
        s = self._display_scale()
        dw, dh = int(self._frame_w * s), int(self._frame_h * s)
        ox, oy = (self.width() - dw) // 2, (self.height() - dh) // 2
        return int((cx - ox) / s), int((cy - oy) / s)

    def _hit_point(self, cx, cy):
        def reaches(pt):
            px, py = self._frame_to_viewport(*pt)
            return (cx - px) ** 2 + (cy - py) ** 2 <= _GRAB_REACH ** 2

        if self._label:
            for label in ("contact", "tip", "base"):
                pt = self._label.get(label)
                if pt is not None and reaches(pt):
                    return label
        for label, pt in [("tip", self._get_tip()), ("base", self._get_base())]:
            if pt is not None and reaches(pt):
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
        if self._label:
            self._draw_label(p)
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
        cx, cy = self._frame_to_viewport(*pt)
        p.setPen(QPen(WHITE, 1))
        p.setBrush(QBrush(color))
        p.drawEllipse(cx - 5, cy - 5, 10, 10)
        p.setPen(QPen(color))
        p.setFont(make_font(size=SIZE_SMALL))
        p.drawText(cx + 10, cy + 4, label)

    def _draw_axis(self, p, axis):
        tx, ty = self._frame_to_viewport(*axis.tip)
        bx, by = self._frame_to_viewport(*axis.base)
        p.setPen(QPen(_AXIS_COLOR, 2, Qt.PenStyle.DashLine))
        p.drawLine(tx, ty, bx, by)
        self._draw_point(p, axis.tip, _TIP_COLOR, "TIP")
        self._draw_point(p, axis.base, _BASE_COLOR, "BASE")

    def _draw_overlay(self, p):
        ov = self._overlay
        axis = ov["axis"]
        tx, ty = self._frame_to_viewport(*axis.tip)
        bx, by = self._frame_to_viewport(*axis.base)

        faint_axis = QColor(_AXIS_COLOR)
        faint_axis.setAlpha(80)
        p.setPen(QPen(faint_axis, 1))
        p.drawLine(tx, ty, bx, by)

        for pt, color in [(axis.tip, _TIP_COLOR), (axis.base, _BASE_COLOR)]:
            cx, cy = self._frame_to_viewport(*pt)
            faint = QColor(color)
            faint.setAlpha(100)
            p.setPen(QPen(Qt.GlobalColor.transparent))
            p.setBrush(QBrush(faint))
            p.drawEllipse(cx - 3, cy - 3, 6, 6)

        ct = ov["contact_pt"]
        ccx, ccy = self._frame_to_viewport(*ct)
        contact_color = GREEN

        if ov["is_action"]:
            p.setPen(QPen(contact_color, 3))
            p.setBrush(QBrush(Qt.GlobalColor.transparent))
            p.drawEllipse(ccx - 7, ccy - 7, 14, 14)
        else:
            p.setPen(QPen(WHITE, 1))
            p.setBrush(QBrush(contact_color))
            p.drawEllipse(ccx - 3, ccy - 3, 6, 6)

            direction = ov.get("direction", 0)
            if direction != 0 and (tx != bx or ty != by):
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
                    head = 5
                    px, py = -uy, ux
                    p.drawLine(int(ex), int(ey),
                               int(ex - ux * head + px * head * 0.5),
                               int(ey - uy * head + py * head * 0.5))
                    p.drawLine(int(ex), int(ey),
                               int(ex - ux * head - px * head * 0.5),
                               int(ey - uy * head - py * head * 0.5))

        p.setPen(QPen(contact_color))
        p.setFont(make_font(size=SIZE_SMALL, bold=ov["is_action"]))
        p.drawText(ccx + 12, ccy + 4, str(ov["pos"]))

    def _draw_auto_overlay(self, p):
        overlay = self._auto_overlay
        _draw_detections(p, self._frame_to_viewport, overlay.get("detections") or [])
        if overlay.get("belief") is not None:
            _draw_remembered_anchor(p, self._frame_to_viewport, overlay)
        if overlay.get("roi") is not None:
            _draw_roi(p, self._frame_to_viewport, overlay)
        _draw_position_gauge(p, overlay, (self.width(), self.height()))

    def _draw_label(self, p):
        label = self._label
        if label.get("tip") and label.get("base"):
            tx, ty = self._frame_to_viewport(*label["tip"])
            bx, by = self._frame_to_viewport(*label["base"])
            p.setPen(QPen(_AXIS_COLOR, 2, Qt.PenStyle.DashLine))
            p.drawLine(tx, ty, bx, by)
            p.setPen(QPen(WHITE, 1))
            p.setBrush(QBrush(_TIP_COLOR))
            p.drawEllipse(tx - 5, ty - 5, 10, 10)
            p.setPen(QPen(_TIP_COLOR))
            p.setFont(make_font(size=SIZE_SMALL))
            p.drawText(tx + 10, ty + 4, "T")
            p.setPen(QPen(WHITE, 1))
            p.setBrush(QBrush(_BASE_COLOR))
            p.drawEllipse(bx - 5, by - 5, 10, 10)
            p.setPen(QPen(_BASE_COLOR))
            p.setFont(make_font(size=SIZE_SMALL))
            p.drawText(bx + 10, by + 4, "B")

        contact = label.get("contact")
        contact_color = GREEN
        if contact is not None:
            ccx, ccy = self._frame_to_viewport(*contact)
            if label.get("is_action"):
                p.setPen(QPen(contact_color, 3))
                p.setBrush(QBrush(Qt.GlobalColor.transparent))
                p.drawEllipse(ccx - 8, ccy - 8, 16, 16)
            else:
                p.setPen(QPen(WHITE, 1))
                p.setBrush(QBrush(contact_color))
                p.drawEllipse(ccx - 5, ccy - 5, 10, 10)
            pos = label.get("pos", "?")
            p.setPen(QPen(contact_color))
            p.setFont(make_font(size=SIZE_BODY, bold=label.get("is_action", False)))
            p.drawText(ccx + 12, ccy + 5, str(pos))
        elif label.get("tip") and label.get("base"):
            mx = (tx + bx) // 2
            my = (ty + by) // 2
            p.setPen(QPen(YELLOW))
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
                fx, fy = self._viewport_to_frame(cx, cy)
                self.clicked.emit(fx, fy)
        elif event.button() == Qt.MouseButton.RightButton:
            cx, cy = int(event.position().x()), int(event.position().y())
            fx, fy = self._viewport_to_frame(cx, cy)
            self.context_menu_requested.emit(fx, fy)

    def mouseMoveEvent(self, event):
        if self._dragging_point:
            cx, cy = int(event.position().x()), int(event.position().y())
            fx, fy = self._viewport_to_frame(cx, cy)
            self.point_dragged.emit(self._dragging_point, fx, fy)

    def mouseReleaseEvent(self, event):
        self._dragging_point = None

    def wheelEvent(self, event):
        d = event.angleDelta().y()
        if d > 0:
            self.zoom *= 1.15
        elif d < 0:
            self.zoom /= 1.15


def _fmt_duration(seconds):
    s = int(seconds)
    if s < _SECONDS_PER_MINUTE:
        return f"{s}s"
    m, s = divmod(s, _SECONDS_PER_MINUTE)
    if m < _MINUTES_PER_HOUR:
        return f"{m}m {s:02d}s"
    h, m = divmod(m, _MINUTES_PER_HOUR)
    return f"{h}h {m:02d}m {s:02d}s"


class App(QMainWindow):

    def __init__(self, *, preview: Preview | None = None):
        super().__init__()
        self.setWindowTitle(window_title("Scripture", preview))
        self.resize(1200, 800)
        _icon_path = Path(__file__).resolve().parent.parent / "icon.ico"
        if _icon_path.exists():
            self.setWindowIcon(app_icon(_icon_path, preview))

        self.video_path = None
        self.video = NO_VIDEO

        self.splits = []
        self.scenes = []
        self.annotations = SceneAnnotations()
        self.auto_result = None
        self._auto_det_frames = []
        self.current_frame_idx = 0
        self.placing = None
        self.session = LabelSession()
        self.pending_tip = self.pending_base = None

        self._worker = None
        self._project_path = None
        self._dirty = False

        self._build_ui()
        self._build_shortcuts()
        self._try_load_last_project()

    @property
    def fps(self) -> float:
        return self.video.fps

    @property
    def total_frames(self) -> int:
        return self.video.total_frames

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
        QShortcut(QKeySequence(Qt.Key.Key_Tab), self,
                  lambda: self._go_to(next_frame(self._get_points_of_interest(), self.current_frame_idx)))
        QShortcut(QKeySequence("Shift+Tab"), self,
                  lambda: self._go_to(previous_frame(self._get_points_of_interest(), self.current_frame_idx)))
        QShortcut(QKeySequence("]"), self,
                  lambda: self._go_to(next_frame(self._action_frames_for_current_scene(), self.current_frame_idx)))
        QShortcut(QKeySequence("["), self,
                  lambda: self._go_to(previous_frame(self._action_frames_for_current_scene(), self.current_frame_idx)))
        QShortcut(QKeySequence("A"), self, self._toggle_label_action)
        QShortcut(QKeySequence("N"), self, self._label_no_contact)
        QShortcut(QKeySequence("C"), self, self._place_label_contact)
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
        pois = set(self.splits)
        for axis in self.annotations.axes.values():
            pois.add(axis.frame)
        return sorted(pois)

    def _go_to(self, frame):
        if frame is not None:
            self._show_frame(frame)

    def _action_frames_for_current_scene(self):
        idx = self._current_scene_idx()
        actions = self.annotations.actions.get(idx, [])
        if not actions:
            return []
        return sorted(frame_of_ms(a["at"], self.fps) for a in actions)

    def _ensure_label_of_frame(self):
        idx = self._current_scene_idx()
        frame = self.current_frame_idx
        here = self.annotations.label_at(idx, frame)
        if here is not None:
            return here
        labeled = self.annotations.labels_of(idx)
        axis = self.annotations.axis_of(idx)
        if labeled:
            nearest = min(labeled, key=lambda f: abs(f - frame))
            entry = dict(labeled[nearest])
        elif axis is not None:
            entry = {"tip": axis.tip, "base": axis.base, "contact": None, "is_action": False}
        else:
            entry = {"tip": None, "base": None, "contact": None, "is_action": False}
        self.annotations.set_label(idx, frame, entry)
        return entry

    def _label_of_frame(self, scene_idx, frame_idx):
        labels_of_scene = self.annotations.labels_of(scene_idx)
        if frame_idx in labels_of_scene:
            label = dict(labels_of_scene[frame_idx])
        elif labels_of_scene:
            nearest = min(labels_of_scene.keys(), key=lambda f: abs(f - frame_idx))
            label = dict(labels_of_scene[nearest])
            label["is_action"] = False
        else:
            return None
        if label.get("contact") and label.get("tip") and label.get("base"):
            label["pos"] = compute_pos_from_points(label["base"], label["tip"], label["contact"])
        else:
            label["pos"] = "—"
        return label

    def _toggle_label_action(self):
        entry = self._ensure_label_of_frame()
        if entry is None:
            return
        entry["is_action"] = not entry["is_action"]
        self._mark_dirty()
        self._show_frame(self.current_frame_idx)

    def _label_no_contact(self):
        entry = self._ensure_label_of_frame()
        if entry is None:
            return
        if self.session.active:
            entry["contact"] = None
            self._mark_dirty()
            self._session_after_label(self.current_frame_idx)
            return
        if entry["contact"] is None:
            if entry.get("tip") and entry.get("base"):
                mx = (entry["tip"][0] + entry["base"][0]) // 2
                my = (entry["tip"][1] + entry["base"][1]) // 2
                entry["contact"] = (mx, my)
        else:
            entry["contact"] = None
        self._mark_dirty()
        self._show_frame(self.current_frame_idx)

    def _place_label_contact(self):
        self.placing = "contact"
        self._set_status("Click to place CONTACT point (or press C again to cancel)")


    def _session_schedule(self):
        idx = self._current_scene_idx()
        if not self.scenes:
            return []
        scene = self.scenes[idx]
        stride = max(1, int(2 * self.fps))
        return schedule_frames(scene.start_frame, scene.end_frame - 1, stride)

    def _session_annotated(self):
        idx = self._current_scene_idx()
        return set(self.annotations.labels_of(idx))

    def _toggle_label_session(self):
        self.session.active = self.btn_label_session.isChecked()
        if self.session.active:
            self._session_goto_next(after=self.current_frame_idx)
        else:
            self.session.stop()
            self._set_status("Label session paused — progress saves with the project.")

    def _session_goto_next(self, after):
        if not self.session.active or not self.scenes:
            return
        schedule = self._session_schedule()
        annotated = self._session_annotated()
        done, total = LabelSession.progress(schedule, annotated)
        nxt = self.session.advance(schedule, annotated, after)
        if nxt is None:
            self.btn_label_session.setChecked(False)
            self._set_status(f"Label session complete: {done}/{total} frames. "
                             "Save the project to keep them.")
            return
        self._show_frame(nxt)
        self._set_status(f"Label {done}/{total} · click = contact · N = none · "
                         "G = skip · U = undo · arrows = scrub · Home = back")

    def _session_after_label(self, labeled_frame):
        self._session_goto_next(
            after=self.session.after_label(self._current_scene_idx(), labeled_frame))

    def _session_skip(self):
        if not self.session.active:
            return
        self._session_goto_next(after=self.session.anchor(self.current_frame_idx))

    def _session_home(self):
        if self.session.active and self.session.target is not None:
            self._show_frame(self.session.target)

    def _session_undo_last(self):
        undone = self.session.undo_last() if self.session.active else None
        if undone is None:
            return
        scene_idx, frame = undone
        self.annotations.drop_label(scene_idx, frame)
        self._mark_dirty()
        self.session.target = frame
        self._show_frame(frame)
        done, total = LabelSession.progress(self._session_schedule(),
                                            self._session_annotated())
        self._set_status(f"Undid label at frame {frame} — decide it again "
                         f"({done}/{total} done)")

    def _reset_scene_labels(self):
        idx = self._current_scene_idx()
        n = len(self.annotations.labels_of(idx))
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
        self.annotations.drop_labels(idx)
        self.session.forget_scene(idx)
        self._mark_dirty()
        self._show_frame(self.current_frame_idx)
        if self.session.active:
            self._session_goto_next(after=self.current_frame_idx)
        self._set_status(f"Cleared {n} labels in scene {idx}. "
                         "Save the project to make it permanent.")

    def _on_label_point_dragged(self, which, fx, fy):
        idx = self._current_scene_idx()
        entry = self._ensure_label_of_frame()
        if entry is None:
            return
        entry[which] = (fx, fy)
        self._mark_dirty()
        label = self._label_of_frame(idx, self.current_frame_idx)
        self.viewport.set_label(label)

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(MARGIN_STANDARD, MARGIN_STANDARD, MARGIN_STANDARD, MARGIN_STANDARD)
        root.setSpacing(GAP_MEDIUM)

        self._build_toolbar()

        self.viewport = Viewport()
        self.viewport.clicked.connect(self._on_viewport_click)
        self.viewport.point_dragged.connect(self._on_point_dragged)
        self.viewport.context_menu_requested.connect(self._on_viewport_context_menu)
        root.addWidget(self.viewport, stretch=1)

        self.timeline = TimelineWidget()
        self.timeline.frame_changed.connect(self._show_frame)
        self.timeline.context_menu_requested.connect(self._on_timeline_context_menu)
        root.addWidget(self.timeline)

        root.addLayout(self._build_footer())

    def _build_toolbar(self):
        tb = QToolBar()
        tb.setMovable(False)
        tb.setFloatable(False)
        tb.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        tb.setIconSize(QSize(BUTTON_ICON, BUTTON_ICON))
        tb.setStyleSheet(toolbar_rules())
        self.addToolBar(tb)

        left_pad = QWidget()
        left_pad.setFixedWidth(MARGIN_STANDARD)
        tb.addWidget(left_pad)

        self._spacer = QWidget()
        self._spacer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self._spacer_action = tb.addWidget(self._spacer)

        self._progress_sep = tb.addSeparator()
        self._progress_sep.setVisible(False)
        self.progress_bar = QProgressBar()
        self.progress_bar.setStyleSheet(_PROGRESS_STYLE)
        self.progress_bar.setFixedHeight(20)
        self.progress_bar.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self._progress_action = tb.addWidget(self.progress_bar)
        self._progress_action.setVisible(False)
        self._abort_action = QAction(qt_icon("stop", color=RED), "Abort", self)
        self._abort_action.triggered.connect(self._abort_processing)
        self._abort_action.setVisible(False)
        tb.addAction(self._abort_action)

        for icon, label, handler in [
            ("plus", "New", self._open_video),
            ("save", "Save", self._save_project),
            ("copy", "Save As", self._save_project_as),
            ("folder", "Load", self._load_project),
        ]:
            act = QAction(qt_icon(icon, color=_ICON_COLOR), label, self)
            act.triggered.connect(handler)
            tb.addAction(act)
        tb.addSeparator()
        act = QAction(qt_icon("export", color=_ICON_COLOR), "Export", self)
        act.triggered.connect(self._export)
        tb.addAction(act)

        right_pad = QWidget()
        right_pad.setFixedWidth(MARGIN_STANDARD)
        tb.addWidget(right_pad)

    def _build_footer(self):
        footer = QHBoxLayout()
        footer.setSpacing(GAP_MEDIUM)

        self.btn_auto_process = QPushButton("Auto Process (YOLO)")
        self.btn_auto_process.clicked.connect(self._auto_process)
        footer.addWidget(self.btn_auto_process)

        self.btn_process_all = QPushButton("Process All")
        self.btn_process_all.clicked.connect(self._process_all)
        footer.addWidget(self.btn_process_all)

        self.btn_label_session = QPushButton("Label Session")
        self.btn_label_session.setCheckable(True)
        self.btn_label_session.clicked.connect(self._toggle_label_session)
        footer.addWidget(self.btn_label_session)

        self.btn_reset_labels = QPushButton("Reset Labels")
        self.btn_reset_labels.clicked.connect(self._reset_scene_labels)
        footer.addWidget(self.btn_reset_labels)

        self.info_label = QLabel("")
        self.info_label.setFont(make_font(size=SIZE_SMALL))
        self.info_label.setStyleSheet(
            f"color: {TEXT_MUTED.name()}; background: {BG_SECONDARY.name()}; "
            f"padding: {GAP_SMALL}px {GAP_MEDIUM}px; border-radius: {BUTTON_RADIUS_HUD}px;"
        )
        footer.addWidget(self.info_label, stretch=1)
        return footer

    def _set_status(self, text):
        self.info_label.setText(text)

    def _format_time(self, frame_idx):
        s = frame_idx / self.fps
        m, s = divmod(int(s), _SECONDS_PER_MINUTE)
        h, m = divmod(m, _MINUTES_PER_HOUR)
        return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"

    def _scene_index_for_frame(self, frame):
        for i, sc in enumerate(self.scenes):
            if frame < sc.end_frame:
                return i
        return max(0, len(self.scenes) - 1)

    def _current_scene_idx(self):
        return self._scene_index_for_frame(self.current_frame_idx)

    def _rebuild_scenes(self):
        self.scenes = scenes_from_splits(self.splits, self.total_frames)
        self.annotations.reindex(self._scene_index_for_frame)
        self.session.reindex(self._scene_index_for_frame)
        if self.auto_result is not None:
            self.annotations.replace_actions(actions_by_scene(
                self.auto_result.actions, self.scenes, self.fps))

    def _update_timeline(self):
        self.timeline.set_state(TimelineState(
            self.scenes, self.annotations.axes, self.annotations.actions,
            self.splits, self.total_frames, self.current_frame_idx,
            self.annotations.labels, self.fps,
        ))

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


    def _reset_project(self):
        self.splits = []
        self.scenes = []
        self.annotations.clear()
        self.session.clear()
        self.btn_label_session.setChecked(False)
        self._set_auto_result(None)
        self.current_frame_idx = 0
        self._project_path = None

    def _open_video(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Video", "",
            "Video files (*.mp4 *.mkv *.avi *.webm *.mov);;All files (*)",
        )
        if not path or not self._load_video(path):
            return
        self._reset_project()
        self.scenes = [Scene(0, self.total_frames)]
        self._mark_dirty()
        self._show_frame(0)

    def _load_video(self, path):
        source = VideoSource.opened(path)
        if source is None:
            QMessageBox.critical(self, "Video not found", f"Cannot open: {path}")
            return False
        self.video.release()
        self.video = source
        self.video_path = path
        return True


    def _show_frame(self, frame_idx):
        frame = self.video.frame_at(frame_idx)
        if frame is None:
            return
        self.current_frame_idx = frame_idx
        idx = self._current_scene_idx()
        self.viewport.set_frame(frame)

        moment = FrameMoment(frame_idx, self.fps, self.annotations.actions.get(idx, []))
        axis = self.annotations.axis_of(idx)
        if axis is not None and axis.frame == frame_idx:
            self.viewport.set_axis(axis)
        else:
            self.viewport.set_axis(None)
            self.viewport.set_pending_tip(self.pending_tip)
            self.viewport.set_pending_base(self.pending_base)

        if self.auto_result is not None:
            self.viewport.set_auto_overlay(auto_overlay(
                moment, result=self.auto_result,
                detection_frames=self._auto_det_frames,
                last_anchor=self._auto_last_anchor))
        else:
            self.viewport.set_auto_overlay(None)

        if axis is not None and (idx in self.annotations.tracking
                                 or idx in self.annotations.actions):
            self.viewport.set_overlay(tracked_overlay(
                moment, axis=axis, scene=self.scenes[idx],
                result=self.annotations.tracking.get(idx)))
        else:
            self.viewport.set_overlay(None)

        if self.annotations.labels_of(idx):
            self.viewport.set_label(self._label_of_frame(idx, frame_idx))
        else:
            self.viewport.set_label(None)

        self._update_info()
        self._update_timeline()


    def _split_or_unsplit(self):
        if self.total_frames == 0:
            return
        frame = self.current_frame_idx
        if frame in self.splits:
            self._do_unsplit(frame)
        else:
            self._do_split_at(frame)

    def _do_unsplit(self, frame):
        idx = self._current_scene_idx()
        left_idx = idx - 1 if idx > 0 and self.scenes[idx].start_frame == frame else idx
        right_idx = left_idx + 1
        if both_scenes_have_an_axis(self.annotations.axes, left_idx, right_idx):
            QMessageBox.warning(self, "Cannot unsplit", "Both adjacent scenes have axes. Delete one axis first.")
            return
        self.splits.remove(frame)
        self._rebuild_scenes()
        self._cancel_placing()
        self._mark_dirty()
        self._show_frame(frame)
        self._set_status(f"Removed split. {len(self.scenes)} scenes.")


    def _on_viewport_context_menu(self, fx, fy):
        if not self.scenes:
            return
        idx = self._current_scene_idx()
        has_axis = self.annotations.axis_of(idx) is not None
        menu = QMenu(self)

        if has_axis or self.pending_tip:
            menu.addAction(qt_icon("cross"), "Remove Tip", lambda: self._delete_point("tip"))
        else:
            menu.addAction("Place Tip Here", lambda: self._place_point_at("tip", fx, fy))

        if has_axis or self.pending_base:
            menu.addAction(qt_icon("cross"), "Remove Base", lambda: self._delete_point("base"))
        else:
            menu.addAction("Place Base Here", lambda: self._place_point_at("base", fx, fy))

        menu.exec(QCursor.pos())

    def _on_timeline_context_menu(self, frame):
        if self.total_frames == 0:
            return
        self._show_frame(frame)

        idx = self._current_scene_idx()
        menu = QMenu(self)

        if frame in self.splits:
            menu.addAction("Unsplit", lambda: self._do_unsplit(frame))
        elif 0 < frame < self.total_frames:
            menu.addAction("Split Here", lambda: self._do_split_at(frame))

        menu.addSeparator()
        if self.annotations.axis_of(idx) is not None:
            if idx in self.annotations.actions:
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


    def _cancel_placing(self):
        self.placing = None
        self.pending_tip = self.pending_base = None

    def _may_move_the_axis(self, idx):
        if not moving_the_axis_needs_asking(self.annotations.axis_of(idx), self.current_frame_idx):
            return True
        reply = QMessageBox.question(
            self, "Change representative frame?",
            "Move axis to this frame?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        return reply == QMessageBox.StandardButton.Yes

    def _form_axis_if_both_placed(self, idx):
        if self.pending_tip and self.pending_base:
            self.annotations.set_axis(idx, AxisDefinition(
                tip=self.pending_tip, base=self.pending_base, frame=self.current_frame_idx,
            ))
            self.pending_tip = self.pending_base = None
        self._mark_dirty()
        self._show_frame(self.current_frame_idx)

    def _place_point(self, which):
        self.placing = which

    def _place_point_at(self, which, fx, fy):
        idx = self._current_scene_idx()

        if not self._may_move_the_axis(idx):
            return
        old = self.annotations.drop_axis(idx)
        if old is not None:
            if which == "tip":
                self.pending_base = old.base
            else:
                self.pending_tip = old.tip

        if which == "tip":
            self.pending_tip = (fx, fy)
        else:
            self.pending_base = (fx, fy)

        self._form_axis_if_both_placed(idx)

    def _delete_point(self, which):
        idx = self._current_scene_idx()
        old = self.annotations.drop_axis(idx)
        if old is not None:
            if which == "tip":
                self.pending_base = old.base
            else:
                self.pending_tip = old.tip
        if which == "tip":
            self.pending_tip = None
        else:
            self.pending_base = None
        self._mark_dirty()
        self._show_frame(self.current_frame_idx)

    def _on_viewport_click(self, fx, fy):
        if not self.scenes:
            return
        if self.session.active and not self.placing:
            entry = self._ensure_label_of_frame()
            if entry is not None:
                entry["contact"] = (fx, fy)
                self._mark_dirty()
                self._session_after_label(self.current_frame_idx)
            return
        if not self.placing:
            return
        idx = self._current_scene_idx()

        if self.placing == "contact":
            entry = self._ensure_label_of_frame()
            if entry is not None:
                entry["contact"] = (fx, fy)
                self._mark_dirty()
                self._show_frame(self.current_frame_idx)
            self.placing = None
            self._set_status("Contact point placed.")
            return

        if not self._may_move_the_axis(idx):
            return
        old = self.annotations.drop_axis(idx)
        if old is not None:
            if self.placing == "tip":
                self.pending_base = old.base
            else:
                self.pending_tip = old.tip

        if self.placing == "tip":
            self.pending_tip = (fx, fy)
        elif self.placing == "base":
            self.pending_base = (fx, fy)
        self.placing = None
        self._form_axis_if_both_placed(idx)

    def _on_point_dragged(self, which, fx, fy):
        idx = self._current_scene_idx()
        if which in ("contact", "tip", "base") and self.annotations.labels_of(idx):
            self._on_label_point_dragged(which, fx, fy)
            return
        old = self.annotations.axis_of(idx)
        if old is not None:
            moved = (AxisDefinition(tip=(fx, fy), base=old.base, frame=old.frame)
                     if which == "tip"
                     else AxisDefinition(tip=old.tip, base=(fx, fy), frame=old.frame))
            self.annotations.set_axis(idx, moved)
            self._mark_dirty()
            self.viewport.set_axis(moved)
            self._update_timeline()
        elif which == "tip" and self.pending_tip:
            self.pending_tip = (fx, fy)
            self.viewport.set_pending_tip(self.pending_tip)
        elif which == "base" and self.pending_base:
            self.pending_base = (fx, fy)
            self.viewport.set_pending_base(self.pending_base)


    def _build_jobs(self, indices):
        axes = self.annotations.axes
        return [(i, self.scenes[i], axes[i]) for i in indices]

    def _show_progress_ui(self, maximum, fmt):
        self._spacer_action.setVisible(False)
        for action in (self._progress_sep, self._progress_action, self._abort_action):
            action.setVisible(True)
        self.progress_bar.setMaximum(maximum)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat(fmt)
        self.btn_process_all.setEnabled(False)
        self.btn_auto_process.setEnabled(False)
        self._process_start_time = time.monotonic()

    def _hide_progress_ui(self):
        self._spacer_action.setVisible(True)
        for action in (self._progress_sep, self._progress_action, self._abort_action):
            action.setVisible(False)
        self.btn_process_all.setEnabled(True)
        self.btn_auto_process.setEnabled(True)

    def _start_processing(self, jobs):
        tf = sum(sc.end_frame - sc.start_frame for _, sc, _ in jobs)
        self._show_progress_ui(_QT_PULSES_A_BAR_WHOSE_MAXIMUM_IS, "Tracking axis with CoTracker3\u2026")
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
        self.annotations.set_result(idx, actions, tracking_result)
        self._mark_dirty()
        self._update_timeline()

    def _on_process_error(self, idx, msg):
        self._set_status(f"Error scene {idx+1}: {msg}")
        QMessageBox.warning(self, f"Processing Error (Scene {idx+1})", msg)

    def _on_process_finished(self):
        el = time.monotonic() - self._process_start_time
        self._hide_progress_ui()
        self._worker = None
        total = sum(len(a) for a in self.annotations.actions.values())
        self._set_status(f"Done \u2014 {total} actions in {_fmt_duration(el)}.")

    def _abort_processing(self):
        if self._worker and self._worker.isRunning():
            self._worker.stop()
            self._worker = None
        self._hide_progress_ui()
        self._set_status("Processing aborted.")


    def _auto_process(self):
        if self._is_processing():
            self._set_status("Already processing \u2014 wait for it to finish.")
            return
        if not self._require_video():
            return
        self._show_progress_ui(self.total_frames, "Auto tracking\u2026 %p%")
        self._worker = AutoProcessWorker(self.video_path)
        self._worker.frame_progress.connect(self.progress_bar.setValue)
        self._worker.done.connect(self._on_auto_done)
        self._worker.error.connect(self._on_auto_error)
        self._worker.finished.connect(self._on_auto_finished)
        self._worker.start()

    def _set_auto_result(self, result):
        self.auto_result = result
        self._auto_det_frames = sorted(result.signal.detections.keys()) if result else []
        self._auto_last_anchor = None
        if result is not None:
            self.annotations.replace_actions(
                actions_by_scene(result.actions, self.scenes, self.fps))
            self.annotations.clear_tracking()
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
                f"in {_fmt_duration(el)}.")

    def _is_processing(self):
        return self._worker is not None and self._worker.isRunning()

    def _process_scene(self, idx):
        if self._is_processing():
            self._set_status("Already processing \u2014 wait for it to finish.")
            return
        if self.annotations.axis_of(idx) is None:
            return
        self._start_processing(self._build_jobs([idx]))

    def _process_all(self):
        if self._is_processing():
            self._set_status("Already processing \u2014 wait for it to finish.")
            return
        axes = self.annotations.axes
        annotated = [i for i in range(len(self.scenes)) if i in axes]
        if not annotated:
            QMessageBox.warning(self, "No axes", "Annotate tip/base on at least one scene first.")
            return
        self._start_processing(self._build_jobs(annotated))

    def _discard_scene(self, idx):
        if idx in self.annotations.actions:
            self.annotations.discard_result(idx)
            self._mark_dirty()
            self._update_timeline()
            self._set_status(f"Discarded actions for scene {idx+1}.")


    def _project(self):
        return Project(
            video_path=self.video_path,
            splits=self.splits,
            axes=self.annotations.axes,
            actions=self.annotations.actions,
            tracking=self.annotations.tracking,
            labels=self.annotations.labels,
            auto=self.auto_result,
            current_frame=self.current_frame_idx,
        )

    def _build_state(self):
        return state_from_project(self._project())

    def _do_save(self, path):
        save_project(path, self._build_state())
        self._project_path = path
        self._mark_clean()
        self._save_last_project(path)
        self._set_status(f"Saved to {Path(path).name}")
        return True

    def _require_video(self):
        if self.video_path:
            return True
        QMessageBox.warning(self, "No video", "Open a video first.")
        return False

    def _save_project(self):
        if not self._require_video():
            return False
        if self._project_path:
            return self._do_save(self._project_path)
        return self._save_project_as()

    def _save_project_as(self):
        if not self._require_video():
            return False
        sd = str(PROJECTS_DIR)
        dn = Path(self.video_path).stem + ".scripture"
        path, _ = QFileDialog.getSaveFileName(self, "Save As", str(Path(sd) / dn), "Scripture (*.scripture);;All (*)")
        if not path:
            return False
        return self._do_save(path)

    def _load_project(self):
        sd = str(PROJECTS_DIR)
        path, _ = QFileDialog.getOpenFileName(self, "Load", sd, "Scripture (*.scripture);;All (*)")
        if path:
            self._do_load(path)

    def _do_load(self, path):
        project = project_from_state(load_project(path))
        if not self._load_video(project.video_path):
            return

        self._reset_project()
        self.splits = project.splits
        self._rebuild_scenes()
        self.annotations.load(axes=project.axes, actions=project.actions,
                              tracking=project.tracking, labels=project.labels)
        self._set_auto_result(project.auto)

        self._project_path = path
        self._mark_clean()
        self._cancel_placing()
        self._save_last_project(path)
        self.current_frame_idx = project.current_frame
        self._show_frame(self.current_frame_idx)

    def _save_last_project(self, path):
        try:
            _LAST_PROJECT_FILE.parent.mkdir(parents=True, exist_ok=True)
            _LAST_PROJECT_FILE.write_text(path)
        except OSError:
            pass

    def _try_load_last_project(self):
        reopen_last_project(_LAST_PROJECT_FILE, self._do_load)


    def _export(self):
        if not self.annotations.actions:
            QMessageBox.warning(self, "No data", "Process at least one scene first.")
            return
        all_a = []
        for a in self.annotations.actions.values():
            all_a.extend(a)
        fs = build_funscript(all_a, int(self.total_frames / self.fps),
                             provenance=provenance_of_actions(self._project()))
        dn = Path(self.video_path).stem + ".funscript" if self.video_path else "output.funscript"
        path, _ = QFileDialog.getSaveFileName(self, "Export", dn, "Funscript (*.funscript);;All (*)")
        if path:
            write_funscript(Path(path), fs)
            self._set_status(f"Exported {len(all_a)} actions to {Path(path).name}")


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
            self._worker.stop()

        if self._dirty:
            reply = QMessageBox.question(
                self, "Unsaved changes",
                "Save before closing?",
                QMessageBox.StandardButton.Save | QMessageBox.StandardButton.Discard | QMessageBox.StandardButton.Cancel,
            )
            if reply == QMessageBox.StandardButton.Save:
                if self._save_project():
                    event.accept()
                else:
                    event.ignore()
            elif reply == QMessageBox.StandardButton.Discard:
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()

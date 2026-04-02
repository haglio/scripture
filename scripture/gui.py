"""Tkinter GUI for scripture: axis annotation, signal review, and export."""

import tkinter as tk
from tkinter import filedialog, messagebox
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageTk

from scripture.scene_detect import detect_scene_changes
from scripture.motion_tracker import AxisDefinition, track_motion
from scripture.stroke_extract import extract_strokes
from scripture.funscript import build_funscript, save_funscript


class App(tk.Tk):

    def __init__(self):
        super().__init__()
        self.title("scripture")
        self.geometry("1200x800")

        self.video_path: str | None = None
        self.cap: cv2.VideoCapture | None = None
        self.fps: float = 30.0
        self.total_frames: int = 0

        # Scene data: list of (start_frame, end_frame) tuples
        self.scenes: list[tuple[int, int]] = []
        # Per-scene axis definitions: scene_index -> AxisDefinition
        self.scene_axes: dict[int, AxisDefinition] = {}
        # Per-scene actions
        self.scene_actions: dict[int, list[dict]] = {}

        self.current_scene_idx: int = 0
        self.click_state: str = "tip"  # "tip" or "base"
        self.current_tip: tuple[int, int] | None = None
        self.current_base: tuple[int, int] | None = None

        # Display scaling
        self.display_scale: float = 1.0

        self._build_ui()

    def _build_ui(self):
        # Top toolbar
        toolbar = tk.Frame(self)
        toolbar.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

        tk.Button(toolbar, text="Open Video", command=self._open_video).pack(side=tk.LEFT, padx=2)
        tk.Button(toolbar, text="Detect Scenes", command=self._detect_scenes).pack(side=tk.LEFT, padx=2)

        self.scene_label = tk.Label(toolbar, text="Scene: -/-")
        self.scene_label.pack(side=tk.LEFT, padx=10)

        tk.Button(toolbar, text="< Prev Scene", command=self._prev_scene).pack(side=tk.LEFT, padx=2)
        tk.Button(toolbar, text="Next Scene >", command=self._next_scene).pack(side=tk.LEFT, padx=2)

        self.click_label = tk.Label(toolbar, text="Click: TIP", fg="red")
        self.click_label.pack(side=tk.LEFT, padx=10)

        tk.Button(toolbar, text="Process Scene", command=self._process_scene).pack(side=tk.LEFT, padx=2)
        tk.Button(toolbar, text="Process All", command=self._process_all).pack(side=tk.LEFT, padx=2)
        tk.Button(toolbar, text="Export .funscript", command=self._export).pack(side=tk.LEFT, padx=2)

        # Main area: frame display
        self.canvas = tk.Canvas(self, bg="black")
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.canvas.bind("<Button-1>", self._on_canvas_click)

        # Status bar
        self.status = tk.Label(self, text="Open a video to begin.", anchor=tk.W)
        self.status.pack(side=tk.BOTTOM, fill=tk.X, padx=5)

    def _set_status(self, text: str):
        self.status.config(text=text)
        self.update_idletasks()

    def _open_video(self):
        path = filedialog.askopenfilename(
            filetypes=[("Video files", "*.mp4 *.mkv *.avi *.webm *.mov"), ("All files", "*.*")]
        )
        if not path:
            return

        self.video_path = path
        self.cap = cv2.VideoCapture(path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.scenes = []
        self.scene_axes.clear()
        self.scene_actions.clear()
        self.current_scene_idx = 0

        self._set_status(f"Loaded: {Path(path).name} — {self.total_frames} frames @ {self.fps:.1f} fps")
        self._show_frame(0)

    def _show_frame(self, frame_idx: int):
        if self.cap is None:
            return

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self.cap.read()
        if not ret:
            return

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Scale to fit canvas
        canvas_w = self.canvas.winfo_width() or 800
        canvas_h = self.canvas.winfo_height() or 600
        h, w = frame_rgb.shape[:2]
        self.display_scale = min(canvas_w / w, canvas_h / h)
        new_w = int(w * self.display_scale)
        new_h = int(h * self.display_scale)

        img = Image.fromarray(frame_rgb).resize((new_w, new_h), Image.LANCZOS)
        self._photo = ImageTk.PhotoImage(img)

        self.canvas.delete("all")
        self.canvas.create_image(canvas_w // 2, canvas_h // 2, image=self._photo)

        # Draw existing axis for this scene
        if self.current_scene_idx in self.scene_axes:
            axis = self.scene_axes[self.current_scene_idx]
            self._draw_axis(axis)
        elif self.current_tip is not None:
            # Draw partial annotation (tip placed, waiting for base)
            sx = int(self.current_tip[0] * self.display_scale) + (canvas_w - new_w) // 2
            sy = int(self.current_tip[1] * self.display_scale) + (canvas_h - new_h) // 2
            self.canvas.create_oval(sx - 5, sy - 5, sx + 5, sy + 5, fill="red", outline="white")
            self.canvas.create_text(sx + 10, sy, text="TIP", fill="red", anchor=tk.W)

    def _draw_axis(self, axis: AxisDefinition):
        canvas_w = self.canvas.winfo_width() or 800
        canvas_h = self.canvas.winfo_height() or 600
        # Compute offset for centered image
        if self.cap is not None:
            frame_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            disp_w = int(frame_w * self.display_scale)
            disp_h = int(frame_h * self.display_scale)
        else:
            disp_w, disp_h = canvas_w, canvas_h

        ox = (canvas_w - disp_w) // 2
        oy = (canvas_h - disp_h) // 2

        tx = int(axis.tip[0] * self.display_scale) + ox
        ty = int(axis.tip[1] * self.display_scale) + oy
        bx = int(axis.base[0] * self.display_scale) + ox
        by = int(axis.base[1] * self.display_scale) + oy

        self.canvas.create_line(tx, ty, bx, by, fill="yellow", width=2, dash=(4, 4))
        self.canvas.create_oval(tx - 5, ty - 5, tx + 5, ty + 5, fill="red", outline="white")
        self.canvas.create_text(tx + 10, ty, text="TIP", fill="red", anchor=tk.W)
        self.canvas.create_oval(bx - 5, by - 5, bx + 5, by + 5, fill="blue", outline="white")
        self.canvas.create_text(bx + 10, by, text="BASE", fill="blue", anchor=tk.W)

    def _on_canvas_click(self, event):
        if self.cap is None or not self.scenes:
            return

        # Convert canvas coords to frame coords
        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()
        frame_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        disp_w = int(frame_w * self.display_scale)
        disp_h = int(frame_h * self.display_scale)

        ox = (canvas_w - disp_w) // 2
        oy = (canvas_h - disp_h) // 2

        frame_x = int((event.x - ox) / self.display_scale)
        frame_y = int((event.y - oy) / self.display_scale)

        if not (0 <= frame_x < frame_w and 0 <= frame_y < frame_h):
            return

        if self.click_state == "tip":
            self.current_tip = (frame_x, frame_y)
            self.click_state = "base"
            self.click_label.config(text="Click: BASE", fg="blue")
            # Redraw to show tip marker
            start_frame = self.scenes[self.current_scene_idx][0]
            self._show_frame(start_frame)
        else:
            self.current_base = (frame_x, frame_y)
            axis = AxisDefinition(tip=self.current_tip, base=self.current_base)
            self.scene_axes[self.current_scene_idx] = axis
            self.click_state = "tip"
            self.click_label.config(text="Click: TIP", fg="red")
            self.current_tip = None
            self.current_base = None
            # Redraw to show full axis
            start_frame = self.scenes[self.current_scene_idx][0]
            self._show_frame(start_frame)
            self._set_status(f"Axis set for scene {self.current_scene_idx + 1}")

    def _detect_scenes(self):
        if self.video_path is None:
            messagebox.showwarning("No video", "Open a video first.")
            return

        self._set_status("Detecting scene changes...")
        changes = detect_scene_changes(self.video_path)

        # Build scene list: (start, end) frame pairs
        boundaries = [0] + changes + [self.total_frames]
        self.scenes = []
        for i in range(len(boundaries) - 1):
            self.scenes.append((boundaries[i], boundaries[i + 1]))

        self.current_scene_idx = 0
        self._update_scene_display()
        self._set_status(f"Found {len(self.scenes)} scenes.")

    def _update_scene_display(self):
        if not self.scenes:
            return
        self.scene_label.config(text=f"Scene: {self.current_scene_idx + 1}/{len(self.scenes)}")
        start_frame = self.scenes[self.current_scene_idx][0]
        self._show_frame(start_frame)

    def _prev_scene(self):
        if self.scenes and self.current_scene_idx > 0:
            self.current_scene_idx -= 1
            self._reset_click_state()
            self._update_scene_display()

    def _next_scene(self):
        if self.scenes and self.current_scene_idx < len(self.scenes) - 1:
            self.current_scene_idx += 1
            self._reset_click_state()
            self._update_scene_display()

    def _reset_click_state(self):
        self.click_state = "tip"
        self.click_label.config(text="Click: TIP", fg="red")
        self.current_tip = None
        self.current_base = None

    def _process_scene(self):
        idx = self.current_scene_idx
        if idx not in self.scene_axes:
            messagebox.showwarning("No axis", "Define tip and base for this scene first.")
            return

        start, end = self.scenes[idx]
        axis = self.scene_axes[idx]
        self._set_status(f"Processing scene {idx + 1} (frames {start}-{end})...")

        result = track_motion(self.video_path, axis, start, end)
        actions = extract_strokes(result.positions, result.timestamps_ms, fps=self.fps)
        self.scene_actions[idx] = actions
        self._set_status(f"Scene {idx + 1}: found {len(actions)} stroke points.")

    def _process_all(self):
        unannotated = [i for i in range(len(self.scenes)) if i not in self.scene_axes]
        if unannotated:
            messagebox.showwarning(
                "Missing axes",
                f"Scenes {', '.join(str(i+1) for i in unannotated)} need tip/base annotation."
            )
            return

        for idx in range(len(self.scenes)):
            self.current_scene_idx = idx
            self._process_scene()

        self._set_status(f"All {len(self.scenes)} scenes processed.")

    def _export(self):
        if not self.scene_actions:
            messagebox.showwarning("No data", "Process at least one scene first.")
            return

        all_actions = []
        for actions in self.scene_actions.values():
            all_actions.extend(actions)

        duration_s = int(self.total_frames / self.fps)
        funscript = build_funscript(all_actions, duration_s)

        default_name = Path(self.video_path).stem + ".funscript" if self.video_path else "output.funscript"
        path = filedialog.asksaveasfilename(
            defaultextension=".funscript",
            initialfile=default_name,
            filetypes=[("Funscript", "*.funscript"), ("All files", "*.*")],
        )
        if not path:
            return

        save_funscript(funscript, path)
        self._set_status(f"Exported {len(all_actions)} actions to {Path(path).name}")

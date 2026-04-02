"""Tkinter GUI for scripture: axis annotation, signal review, and export."""

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageTk

from scripture.scene_detect import Scene, build_scenes
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
        self.frame_w: int = 0
        self.frame_h: int = 0

        self.scenes: list[Scene] = []
        # Per-scene axis definitions: scene_index -> AxisDefinition
        self.scene_axes: dict[int, AxisDefinition] = {}
        # Per-scene actions
        self.scene_actions: dict[int, list[dict]] = {}

        self.current_scene_idx: int = 0
        self.current_frame_idx: int = 0
        self.click_state: str = "tip"  # "tip" or "base"
        self.current_tip: tuple[int, int] | None = None
        self.current_base: tuple[int, int] | None = None

        # Display scaling (video is 1/3 of canvas in each dimension)
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

        tk.Button(toolbar, text="< Prev", command=self._prev_scene).pack(side=tk.LEFT, padx=2)
        tk.Button(toolbar, text="Next >", command=self._next_scene).pack(side=tk.LEFT, padx=2)

        self.click_label = tk.Label(toolbar, text="Click: TIP", fg="red")
        self.click_label.pack(side=tk.LEFT, padx=10)

        tk.Button(toolbar, text="Process Scene", command=self._process_scene).pack(side=tk.LEFT, padx=2)
        tk.Button(toolbar, text="Process All", command=self._process_all).pack(side=tk.LEFT, padx=2)
        tk.Button(toolbar, text="Export .funscript", command=self._export).pack(side=tk.LEFT, padx=2)

        # Main area: frame display
        self.canvas = tk.Canvas(self, bg="#1a1a1a")
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.canvas.bind("<Button-1>", self._on_canvas_click)

        # Frame scrubber
        scrub_frame = tk.Frame(self)
        scrub_frame.pack(fill=tk.X, padx=5)

        tk.Label(scrub_frame, text="Frame:").pack(side=tk.LEFT)
        self.frame_scrubber = ttk.Scale(
            scrub_frame, from_=0, to=1, orient=tk.HORIZONTAL,
            command=self._on_scrub,
        )
        self.frame_scrubber.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.frame_info_label = tk.Label(scrub_frame, text="", width=30, anchor=tk.W)
        self.frame_info_label.pack(side=tk.LEFT)

        # Status bar
        self.status = tk.Label(self, text="Open a video to begin.", anchor=tk.W)
        self.status.pack(side=tk.BOTTOM, fill=tk.X, padx=5)

    def _set_status(self, text: str):
        self.status.config(text=text)
        self.update_idletasks()

    def _format_time(self, frame_idx: int) -> str:
        seconds = frame_idx / self.fps
        m, s = divmod(int(seconds), 60)
        h, m = divmod(m, 60)
        if h:
            return f"{h}:{m:02d}:{s:02d}"
        return f"{m}:{s:02d}"

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
        self.frame_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.scenes = []
        self.scene_axes.clear()
        self.scene_actions.clear()
        self.current_scene_idx = 0

        duration = self._format_time(self.total_frames)
        self._set_status(
            f"Loaded: {Path(path).name} — {self.total_frames} frames @ "
            f"{self.fps:.1f} fps — {duration} — {self.frame_w}x{self.frame_h}"
        )
        self._show_frame(0)

    def _show_frame(self, frame_idx: int):
        if self.cap is None:
            return

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self.cap.read()
        if not ret:
            return

        self.current_frame_idx = frame_idx
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Scale video to 1/3 of canvas so it sits in the center 1/9th
        canvas_w = self.canvas.winfo_width() or 800
        canvas_h = self.canvas.winfo_height() or 600
        h, w = frame_rgb.shape[:2]
        self.display_scale = min(canvas_w / (3 * w), canvas_h / (3 * h))
        new_w = int(w * self.display_scale)
        new_h = int(h * self.display_scale)

        img = Image.fromarray(frame_rgb).resize((new_w, new_h), Image.LANCZOS)
        self._photo = ImageTk.PhotoImage(img)

        self.canvas.delete("all")

        # Draw a subtle border around the video area
        ox = (canvas_w - new_w) // 2
        oy = (canvas_h - new_h) // 2
        self.canvas.create_rectangle(
            ox - 1, oy - 1, ox + new_w + 1, oy + new_h + 1,
            outline="#555555", width=1,
        )
        self.canvas.create_image(canvas_w // 2, canvas_h // 2, image=self._photo)

        # Draw existing axis for this scene
        if self.current_scene_idx in self.scene_axes:
            self._draw_axis(self.scene_axes[self.current_scene_idx])
        elif self.current_tip is not None:
            sx, sy = self._frame_to_canvas(self.current_tip)
            self.canvas.create_oval(sx - 5, sy - 5, sx + 5, sy + 5, fill="red", outline="white")
            self.canvas.create_text(sx + 10, sy, text="TIP", fill="red", anchor=tk.W)

        # Update frame info
        self.frame_info_label.config(
            text=f"#{frame_idx} / {self._format_time(frame_idx)}"
        )

    def _frame_to_canvas(self, point: tuple[int, int]) -> tuple[int, int]:
        """Convert frame coordinates to canvas coordinates."""
        canvas_w = self.canvas.winfo_width() or 800
        canvas_h = self.canvas.winfo_height() or 600
        disp_w = int(self.frame_w * self.display_scale)
        disp_h = int(self.frame_h * self.display_scale)
        ox = (canvas_w - disp_w) // 2
        oy = (canvas_h - disp_h) // 2
        return (
            int(point[0] * self.display_scale) + ox,
            int(point[1] * self.display_scale) + oy,
        )

    def _canvas_to_frame(self, cx: int, cy: int) -> tuple[int, int]:
        """Convert canvas coordinates to frame coordinates (may be off-screen)."""
        canvas_w = self.canvas.winfo_width() or 800
        canvas_h = self.canvas.winfo_height() or 600
        disp_w = int(self.frame_w * self.display_scale)
        disp_h = int(self.frame_h * self.display_scale)
        ox = (canvas_w - disp_w) // 2
        oy = (canvas_h - disp_h) // 2
        return (
            int((cx - ox) / self.display_scale),
            int((cy - oy) / self.display_scale),
        )

    def _draw_axis(self, axis: AxisDefinition):
        tx, ty = self._frame_to_canvas(axis.tip)
        bx, by = self._frame_to_canvas(axis.base)

        self.canvas.create_line(tx, ty, bx, by, fill="yellow", width=2, dash=(4, 4))
        self.canvas.create_oval(tx - 5, ty - 5, tx + 5, ty + 5, fill="red", outline="white")
        self.canvas.create_text(tx + 10, ty, text="TIP", fill="red", anchor=tk.W)
        self.canvas.create_oval(bx - 5, by - 5, bx + 5, by + 5, fill="blue", outline="white")
        self.canvas.create_text(bx + 10, by, text="BASE", fill="blue", anchor=tk.W)

    def _on_canvas_click(self, event):
        if self.cap is None or not self.scenes:
            return

        # Allow clicks anywhere — coordinates may be off-screen
        frame_x, frame_y = self._canvas_to_frame(event.x, event.y)

        if self.click_state == "tip":
            self.current_tip = (frame_x, frame_y)
            self.click_state = "base"
            self.click_label.config(text="Click: BASE", fg="blue")
            self._show_frame(self.current_frame_idx)
        else:
            self.current_base = (frame_x, frame_y)
            axis = AxisDefinition(tip=self.current_tip, base=self.current_base)
            self.scene_axes[self.current_scene_idx] = axis
            self.click_state = "tip"
            self.click_label.config(text="Click: TIP", fg="red")
            self.current_tip = None
            self.current_base = None
            self._show_frame(self.current_frame_idx)
            self._set_status(f"Axis set for scene {self.current_scene_idx + 1}")

    def _on_scrub(self, value):
        if not self.scenes:
            return
        scene = self.scenes[self.current_scene_idx]
        frame_range = scene.end_frame - scene.start_frame
        frame_idx = scene.start_frame + int(float(value) * frame_range)
        frame_idx = min(frame_idx, scene.end_frame - 1)
        self._show_frame(frame_idx)

    def _detect_scenes(self):
        if self.video_path is None:
            messagebox.showwarning("No video", "Open a video first.")
            return

        self._set_status("Detecting scenes (sampling frames for classification)...")
        self.scenes = build_scenes(self.video_path)

        content_count = sum(1 for s in self.scenes if s.is_content)
        non_content = len(self.scenes) - content_count

        self.current_scene_idx = 0
        # Jump to first content scene
        for i, scene in enumerate(self.scenes):
            if scene.is_content:
                self.current_scene_idx = i
                break

        self._update_scene_display()
        self._set_status(
            f"Found {len(self.scenes)} scenes: "
            f"{content_count} content, {non_content} non-content (black/blank)."
        )

    def _scene_label_text(self) -> str:
        if not self.scenes:
            return "Scene: -/-"
        scene = self.scenes[self.current_scene_idx]
        tag = "" if scene.is_content else " [BLACK]"
        start_t = self._format_time(scene.start_frame)
        end_t = self._format_time(scene.end_frame)
        return (
            f"Scene {self.current_scene_idx + 1}/{len(self.scenes)}{tag}  "
            f"[{start_t} - {end_t}]  "
            f"frames {scene.start_frame}-{scene.end_frame}"
        )

    def _update_scene_display(self):
        if not self.scenes:
            return
        scene = self.scenes[self.current_scene_idx]
        self.scene_label.config(text=self._scene_label_text())
        self.frame_scrubber.set(0)
        self._show_frame(scene.representative_frame)

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

        scene = self.scenes[idx]
        axis = self.scene_axes[idx]
        self._set_status(f"Processing scene {idx + 1} (frames {scene.start_frame}-{scene.end_frame})...")

        result = track_motion(self.video_path, axis, scene.start_frame, scene.end_frame)
        actions = extract_strokes(result.positions, result.timestamps_ms, fps=self.fps)
        self.scene_actions[idx] = actions
        self._set_status(f"Scene {idx + 1}: found {len(actions)} stroke points.")

    def _process_all(self):
        content_scenes = [i for i, s in enumerate(self.scenes) if s.is_content]
        unannotated = [i for i in content_scenes if i not in self.scene_axes]
        if unannotated:
            messagebox.showwarning(
                "Missing axes",
                f"Content scenes {', '.join(str(i+1) for i in unannotated)} need tip/base annotation."
            )
            return

        for idx in content_scenes:
            self.current_scene_idx = idx
            self._process_scene()

        self._set_status(f"Processed {len(content_scenes)} content scenes.")

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

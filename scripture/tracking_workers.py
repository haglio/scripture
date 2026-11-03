from __future__ import annotations

import traceback

from PyQt6.QtCore import QThread, pyqtSignal

from scripture.auto_funscript import run_pipeline
from scripture.cycle_extract import extract_cycles
from scripture.motion_tracker import track_motion

_STOP_WAIT_MS = 5000


def _say(signal, *args):
    signal.emit(*args)


class _Stopped(Exception):
    pass


class _TrackingWorker(QThread):
    frame_progress = pyqtSignal(int)

    def __init__(self, video_path):
        super().__init__()
        self.video_path = video_path
        self._stop_asked = False

    def ask_to_stop(self):
        self._stop_asked = True

    def stop(self):
        self.disconnect()
        self.ask_to_stop()
        if not self.wait(_STOP_WAIT_MS):
            self.terminate()
            self.wait()

    def _report_frame(self, frame):
        if self._stop_asked:
            raise _Stopped
        _say(self.frame_progress, frame)


class ProcessWorker(_TrackingWorker):
    scene_done = pyqtSignal(int, list, object)
    finished = pyqtSignal()
    error = pyqtSignal(int, str)

    def __init__(self, video_path, jobs, fps, *, track=track_motion):
        super().__init__(video_path)
        self.jobs = jobs
        self.fps = fps
        self._track = track

    def run(self):
        offsets = {}
        cumulative = 0
        for idx, scene, _ in self.jobs:
            offsets[idx] = cumulative
            cumulative += scene.end_frame - scene.start_frame
        for idx, scene, axis in self.jobs:
            offset = offsets[idx]
            try:
                result = self._track(
                    self.video_path, axis, scene.start_frame, scene.end_frame,
                    on_frame=lambda f, _o=offset: self._report_frame(_o + f),
                )
                actions = extract_cycles(result.positions, result.timestamps_ms, fps=self.fps)
                _say(self.scene_done, idx, actions, result)
            except _Stopped:
                return
            except Exception as e:
                traceback.print_exc()
                _say(self.error, idx, str(e))
        _say(self.finished)


class AutoProcessWorker(_TrackingWorker):
    done = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(self, video_path, *, track=run_pipeline):
        super().__init__(video_path)
        self._track = track

    def run(self):
        try:
            _say(self.done, self._track(self.video_path, on_frame=self._report_frame))
        except _Stopped:
            return
        except Exception as e:
            traceback.print_exc()
            _say(self.error, str(e))

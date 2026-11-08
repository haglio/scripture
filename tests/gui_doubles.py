from __future__ import annotations

from PyQt6.QtWidgets import QMessageBox


class Footage:
    def __init__(self, total_frames, fps=30.0):
        self.total_frames = total_frames
        self.fps = fps

    def frame_at(self, _index):
        return None

    def release(self):
        pass


class CloseEvent:

    def __init__(self):
        self.accepted = None

    def accept(self):
        self.accepted = True

    def ignore(self):
        self.accepted = False


class Dialogs:

    StandardButton = QMessageBox.StandardButton

    def __init__(self, *answers):
        self._answers = list(answers)
        self.shown = []

    def question(self, *_args):
        return self._answers.pop(0)

    def warning(self, _parent, title, text):
        self.shown.append((title, text))

    def critical(self, _parent, title, text):
        self.shown.append((title, text))


class FileDialog:

    def __init__(self, chosen=""):
        self._chosen = chosen

    def getSaveFileName(self, *_args):
        return self._chosen, ""

    def getOpenFileName(self, *_args):
        return self._chosen, ""


class Capture:

    def __init__(self, opened=True):
        self._opened = opened
        self.released = False

    def isOpened(self):
        return self._opened

    def release(self):
        self.released = True

    def get(self, _prop):
        return 0

    def set(self, *_args):
        return True

    def read(self):
        return False, None

"""Stand-ins for the collaborators `gui` reaches for by module-level name.

`gui.QMessageBox`, `gui.QFileDialog` and `gui.cv2` are all names in the module,
so a test can put one of these in their place and answer for them. Nothing here
subclasses the real thing: what the window uses of each is small, and a stand-in
that only offers that much fails loudly when the window starts asking for more.
"""

from __future__ import annotations

from PyQt6.QtWidgets import QMessageBox


class CloseEvent:
    """The QCloseEvent Qt hands to `closeEvent`."""

    def __init__(self):
        self.accepted = None

    def accept(self):
        self.accepted = True

    def ignore(self):
        self.accepted = False


class Dialogs:
    """`gui.QMessageBox`: answers each question in turn, remembers the rest."""

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


class SaveDialog:
    """`gui.QFileDialog`: one Save As answer, empty for a cancelled dialog."""

    def __init__(self, chosen=""):
        self._chosen = chosen

    def getSaveFileName(self, *_args):  # noqa: N802 - Qt spells it this way
        return self._chosen, ""


class Capture:
    """A `cv2.VideoCapture`, open or not, that never yields a frame."""

    def __init__(self, opened=True):
        self._opened = opened
        self.released = False

    def isOpened(self):  # noqa: N802 - cv2 spells it this way
        return self._opened

    def release(self):
        self.released = True

    def get(self, _prop):
        return 0

    def set(self, *_args):
        return True

    def read(self):
        return False, None

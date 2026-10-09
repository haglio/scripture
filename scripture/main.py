from __future__ import annotations

import contextlib
import logging
import sys
from pathlib import Path

from app_support.logging_utils import install_exception_logging
from app_support.process_identity import ProcessNamer
from app_support.win32 import set_app_user_model_id, stamp_pinned_shortcuts
from shared_ui.preview import Preview, preview_of, taskbar_identity

from scripture.tracker_environment import complaints

APP_USER_MODEL_ID = "FunTime.Scripture"
PROJECT_DIR = Path(__file__).resolve().parent.parent
_ICON = PROJECT_DIR / "icon.ico"


def _set_windows_app_user_model_id(preview: Preview | None) -> None:
    """Claim the identity the pinned shortcut carries, and stamp the pin with it,
    before any window exists.

    Cosmetic: a window under the interpreter's icon is still a window, so a
    refusal costs the icon and nothing else.
    """
    if sys.platform != "win32":
        return
    with contextlib.suppress(OSError):
        set_app_user_model_id(taskbar_identity(APP_USER_MODEL_ID, preview))
    stamp_pinned_shortcuts(APP_USER_MODEL_ID, ["Scripture"])


def _name_this_process() -> None:
    """Leave ``launch_scripture.vbs`` an interpreter that says "Scripture" next
    time.  The console interpreter, because that is the one the launcher runs --
    it redirects the app's output into its log.  Why it is one launch late, and
    why it can never cost the launch: :meth:`ProcessNamer.name_this_process`."""
    ProcessNamer("Scripture", icon=_ICON).name_this_process("Scripture", interpreter="python.exe")


def _report_tracker_environment() -> None:
    """Say on the way up when this machine's torch cannot run the tracker.

    Here rather than in the suite: the merge gate installs the CPU wheel on
    purpose and has no GPU, so a test of this could only ever skip there, while
    the machine that tracks is the one that needs telling.  The launcher sends
    this stream to sessions/scripture_launcher.log.
    """
    for said in complaints():
        print(f"scripture: {said}", file=sys.stderr)


def _log_errors_and_keep_running() -> None:
    install_exception_logging(logging.getLogger("scripture"))


def main():
    _log_errors_and_keep_running()
    preview = preview_of(PROJECT_DIR)
    _set_windows_app_user_model_id(preview)
    _name_this_process()
    _report_tracker_environment()

    # Local: the window, and the toolkit under it, only when a window is wanted.
    from PyQt6.QtWidgets import QApplication  # noqa: PLC0415

    # Local: the window, and the toolkit under it, only when a window is wanted.
    from shared_ui.chrome import family_stylesheet  # noqa: PLC0415

    # Local: the window, and the toolkit under it, only when a window is wanted.
    from scripture.gui import App  # noqa: PLC0415

    app = QApplication.instance() or QApplication(sys.argv)
    # On the application, not the window: the family's tooltip rule reaches a
    # top-level popup only from here.
    app.setStyleSheet(family_stylesheet())
    window = App(preview=preview)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

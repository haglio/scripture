from __future__ import annotations

import contextlib
import logging
import sys
from pathlib import Path

from app_support.logging_utils import install_exception_logging
from app_support.process_identity import ProcessNamer
from app_support.win32 import set_app_user_model_id, stamp_pinned_shortcuts
from shared_ui.preview import Preview, preview_of, taskbar_identity

from scripture import content
from scripture.tracker_environment import complaints

APP_USER_MODEL_ID = "FunTime.Scripture"
PROJECT_DIR = Path(__file__).resolve().parent.parent
_ICON = PROJECT_DIR / "icon.ico"

STEPS = (
    ("Checking the tracker...", 5.0),
    ("Loading Scripture...", 1.0),
    ("Opening the window...", 0.2),
)
LOADING_CAPTION = "Scripture Loading"
CANCEL_HINT = "Press Esc to cancel opening Scripture"


def _set_windows_app_user_model_id(preview: Preview | None) -> None:
    if sys.platform != "win32":
        return
    with contextlib.suppress(OSError):
        set_app_user_model_id(_its_taskbar_identity(preview))
    stamp_pinned_shortcuts(APP_USER_MODEL_ID, ["Scripture"])


def _its_taskbar_identity(preview: Preview | None) -> str:
    return taskbar_identity(APP_USER_MODEL_ID, preview)


def _name_this_process() -> None:
    ProcessNamer("Scripture", icon=_ICON).name_this_process("Scripture", interpreter="python.exe")


def _report_tracker_environment(*, found=complaints) -> None:
    for said in found():
        print(f"scripture: {said}", file=sys.stderr)


def _log_errors_and_keep_running() -> None:
    install_exception_logging(logging.getLogger("scripture"))


def open_scripture(loading, *, preview: Preview | None, dress):
    checking, loading_scripture, opening = (step for step, _weight in STEPS)
    loading.say(checking)
    _report_tracker_environment()
    loading.say(loading_scripture)
    content.refuse_an_incomplete_overlay()
    from scripture.gui import App  # noqa: PLC0415

    dress()
    loading.say(opening)
    return App(preview=preview)


def open_the_loading_screen(preview: Preview | None):
    from shared_ui.loading_process import LoadingProcess  # noqa: PLC0415

    return LoadingProcess.open(
        caption=LOADING_CAPTION, wordmark="Scripture", icon=_ICON, preview=preview,
        steps=STEPS, cancel_hint=CANCEL_HINT,
        app_id=_its_taskbar_identity(preview) if sys.platform == "win32" else None,
    )


def main():
    _log_errors_and_keep_running()
    preview = preview_of(PROJECT_DIR)
    _set_windows_app_user_model_id(preview)
    _name_this_process()

    from PyQt6.QtWidgets import QApplication  # noqa: PLC0415
    from shared_ui.colors import dress_application  # noqa: PLC0415
    from shared_ui.loading_window import Loading, LoadingCanceled  # noqa: PLC0415

    app = QApplication.instance() or QApplication(sys.argv)
    loading = Loading(app, logging.getLogger("scripture"), open_the_loading_screen(preview))
    try:
        # Qt styles a tooltip, a top-level popup, only from the QApplication's rules.
        window = open_scripture(
            loading, preview=preview, dress=lambda: dress_application(app))
    except LoadingCanceled:
        return
    loading.done()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

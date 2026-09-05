import contextlib
import sys
from pathlib import Path

from app_support.process_identity import ProcessNamer
from app_support.win32 import set_app_user_model_id

SCRIPTURE_APP_USER_MODEL_ID = "FunTime.Scripture"
_ICON = Path(__file__).resolve().parent.parent / "icon.ico"


def _set_windows_app_user_model_id() -> None:
    """Claim the identity the pinned shortcut carries, before any window exists.

    Cosmetic: a window under the interpreter's icon is still a window, so a
    refusal costs the icon and nothing else.
    """
    if sys.platform != "win32":
        return
    with contextlib.suppress(OSError):
        set_app_user_model_id(SCRIPTURE_APP_USER_MODEL_ID)


def _name_this_process() -> None:
    """Leave ``launch_scripture.vbs`` an interpreter that says "Scripture" next
    time.  The console interpreter, because that is the one the launcher runs --
    it redirects the app's output into its log.  Why it is one launch behind, and
    why it can never cost the launch: :meth:`ProcessNamer.name_this_process`."""
    ProcessNamer("Scripture", icon=_ICON).name_this_process("Scripture", interpreter="python.exe")


def main():
    _set_windows_app_user_model_id()
    _name_this_process()

    from PyQt6.QtWidgets import QApplication

    from scripture.gui import App

    app = QApplication.instance() or QApplication(sys.argv)
    window = App()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

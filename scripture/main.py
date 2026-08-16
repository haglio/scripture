import sys

SCRIPTURE_APP_USER_MODEL_ID = "FunTime.Scripture"


def _set_windows_app_user_model_id() -> None:
    if sys.platform != "win32":
        return
    try:
        import ctypes
        set_app_id = ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID
        set_app_id.argtypes = [ctypes.c_wchar_p]
        set_app_id.restype = ctypes.c_long
        set_app_id(SCRIPTURE_APP_USER_MODEL_ID)
    except Exception:
        pass


def _name_this_process() -> None:
    """Leave ``launch_scripture.vbs`` an interpreter that says "Scripture" next time.

    Windows takes what it shows about a process from the file it was started
    from -- the Details tab's name, the Processes tab's description, the icon
    beside it -- so a plain interpreter puts Scripture in the task list as one more
    anonymous "Python".  That costs nothing until something strands a process,
    and then the task list is the only way back and cannot say which row is safe
    to end.

    Naming this process on the way in is the one thing that cannot be done:
    writing the copy takes the very interpreter being named.  So each run makes
    it for the run after and the launcher picks it up, which costs one launch,
    once.  A checkout that has never started launches exactly as it used to.
    """
    try:
        from pathlib import Path as _Path

        from app_support.process_identity import ProcessNamer

        icon = _Path(__file__).resolve().parent.parent / "icon.ico"
        ProcessNamer("Scripture", icon=icon).prepare_launcher(
            "Scripture", _Path(sys.executable).with_name("python.exe"))
    except Exception:
        pass  # Cosmetic: costs a name in the task list, never a launch.


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

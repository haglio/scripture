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


def main():
    _set_windows_app_user_model_id()

    from PyQt6.QtWidgets import QApplication
    from scripture.gui import App

    app = QApplication.instance() or QApplication(sys.argv)
    window = App()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

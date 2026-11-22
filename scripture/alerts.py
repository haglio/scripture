from __future__ import annotations

from pathlib import Path

from shared_ui import alert

ICON_FILE = Path(__file__).resolve().parent.parent / "icon.ico"


def show_error(title: str, message: str, *, warning_icon: bool = False) -> bool:
    return alert.show_error(title, message, icon=ICON_FILE,
                            level=alert.Level.WARNING if warning_icon else alert.Level.ERROR)


def show_warning(title: str, message: str) -> bool:
    return alert.show_warning(title, message, icon=ICON_FILE, level=alert.Level.WARNING)

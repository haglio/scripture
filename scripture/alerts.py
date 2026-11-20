from __future__ import annotations

from pathlib import Path

from shared_ui import alert

ICON_FILE = Path(__file__).resolve().parent.parent / "icon.ico"


def show_error(title: str, message: str) -> bool:
    return alert.show_alert(title, message, icon=ICON_FILE, level=alert.Level.ERROR)


def show_warning(title: str, message: str, *, copyable: bool = True) -> bool:
    return alert.show_alert(title, message, icon=ICON_FILE, level=alert.Level.WARNING, copyable=copyable)

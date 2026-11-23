from __future__ import annotations

from pathlib import Path

from shared_ui import alert

ICON_FILE = Path(__file__).resolve().parent.parent / "icon.ico"


def show_error(title: str, message: str) -> bool:
    return alert.show_error(title, message, icon=ICON_FILE)


def show_warning(title: str, message: str) -> bool:
    return alert.show_warning(title, message, icon=ICON_FILE)

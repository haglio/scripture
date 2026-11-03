from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(scope="session")
def qt_app():
    from PyQt6.QtWidgets import QApplication  # noqa: PLC0415

    return QApplication.instance() or QApplication([])


@pytest.fixture
def window(qt_app, tmp_path, monkeypatch):
    from scripture import gui  # noqa: PLC0415

    monkeypatch.setattr(gui, "_LAST_PROJECT_FILE", tmp_path / ".last_session")
    app_window = gui.App()
    yield app_window
    app_window._mark_clean()
    app_window.close()

from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(scope="session")
def qt_app():
    from PyQt6.QtWidgets import QApplication  # noqa: PLC0415

    return QApplication.instance() or QApplication([])


def _the_last_project_pointer_aimed_at_scratch_or_a_new_window_reopens_his_project_and_video(
        gui, monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(gui, "_LAST_PROJECT_FILE", tmp_path / ".last_session")


def _mark_clean_so_the_close_guard_asks_nobody_about_unsaved_work(app_window) -> None:
    app_window._mark_clean()


@pytest.fixture
def window(qt_app, tmp_path, monkeypatch):
    from scripture import gui  # noqa: PLC0415

    _the_last_project_pointer_aimed_at_scratch_or_a_new_window_reopens_his_project_and_video(gui, monkeypatch, tmp_path)
    app_window = gui.App()
    yield app_window
    _mark_clean_so_the_close_guard_asks_nobody_about_unsaved_work(app_window)
    app_window.close()

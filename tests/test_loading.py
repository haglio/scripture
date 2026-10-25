"""What Scripture shows while it opens: the family's loading screen, walking
the steps the launch takes, the slow one first."""

from __future__ import annotations

import json
import sys
import tomllib
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from app_support.overlay import MissingOverlayKey

from scripture import content, main


class _Loading:
    def __init__(self):
        self.said: list[str] = []

    def say(self, step: str) -> None:
        self.said.append(step)


def test_the_launch_says_each_step_before_it_takes_it():
    loading = _Loading()
    with patch.object(main, "_report_tracker_environment") as checked, \
         patch("scripture.gui.App") as window_class:
        window = main.open_scripture(loading, preview=None, dress=lambda: None)

    assert loading.said == [step for step, _weight in main.STEPS]
    checked.assert_called_once_with()
    window_class.assert_called_once_with(preview=None)
    assert window is window_class.return_value


def test_the_loading_screen_runs_in_a_process_of_its_own_wearing_scriptures_taskbar_button(
        monkeypatch):
    monkeypatch.setattr(main.sys, "platform", "win32")

    with patch("shared_ui.loading_process.LoadingProcess.open") as opened:
        main.open_the_loading_screen(None)

    opened.assert_called_once_with(
        caption=main.LOADING_CAPTION, wordmark="Scripture", icon=main.PROJECT_DIR / "icon.ico",
        preview=None, steps=main.STEPS, cancel_hint=main.CANCEL_HINT,
        app_id=main.APP_USER_MODEL_ID)


def test_off_windows_the_loading_screen_wears_no_taskbar_identity(monkeypatch):
    monkeypatch.setattr(main.sys, "platform", "linux")

    with patch("shared_ui.loading_process.LoadingProcess.open") as opened:
        main.open_the_loading_screen(None)

    assert opened.call_args.kwargs["app_id"] is None


def test_the_tracker_check_is_the_long_step_so_the_bar_spends_most_of_its_time_there():
    (first, weight), *rest = main.STEPS

    assert first == "Checking the tracker..."
    assert weight > sum(w for _step, w in rest)


def test_a_launch_on_a_settings_file_missing_a_line_its_example_documents_refuses_before_the_window(
        monkeypatch, tmp_path):
    lines = json.loads(content.EXAMPLE_CONTENT.read_text(encoding="utf-8"))
    del lines["class_colors"]
    short = tmp_path / "content.local.json"
    short.write_text(json.dumps(lines), encoding="utf-8")
    monkeypatch.setattr(content, "LOCAL_CONTENT", short)
    window = MagicMock()
    monkeypatch.setitem(sys.modules, "scripture.gui", SimpleNamespace(App=window))

    with patch.object(main, "_report_tracker_environment"), \
         pytest.raises(MissingOverlayKey, match="class_colors"):
        main.open_scripture(_Loading(), preview=None, dress=lambda: None)

    window.assert_not_called()


def test_the_pyproject_says_the_launch_needs_every_key_of_the_example_in_the_local_overlay():
    declared = tomllib.loads((content.PROJECT_DIR / "pyproject.toml").read_text(encoding="utf-8"))[
        "tool"]["haglio"]["opens-only-with"]

    assert (content.PROJECT_DIR / declared["every-key-of"], content.PROJECT_DIR / declared["in"]) == (
        content.EXAMPLE_CONTENT, content.LOCAL_CONTENT)

from __future__ import annotations

import os
from pathlib import Path


def test_the_suite_asks_for_an_offscreen_platform_here_too_the_window_fixture_builds_a_real_window_in_this_process_and_by_default_so_a_developer_can_watch():
    assert os.environ.get("QT_QPA_PLATFORM") == "offscreen"


def test_no_fixture_sets_qt_up_for_every_test_or_a_launch_that_never_completes_could_pass_green():
    source = (Path(__file__).resolve().parent / "conftest.py").read_text(encoding="utf-8")

    assert "def qt_app(" in source
    assert "autouse" not in source

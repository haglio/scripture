"""Which interpreter launch_scripture.vbs runs the app on, asked of the launcher.

Scripture runs on the project venv and nothing else: it holds the CUDA torch the
tracker needs and the editable shared_ui the GUI imports, and it is what this
suite runs on.  The launcher is rendered from its spec in pyproject.toml by
app_support.launcher, whose own tests hold what every launcher does; what is
Scripture's is asked of this one under the real script host.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
from app_support.launcher import assert_launchers_match_their_specs, dry_run

REPO_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = REPO_ROOT / "launch_scripture.vbs"

on_windows = pytest.mark.skipif(sys.platform != "win32", reason="the Windows script host")


def test_the_launcher_is_where_the_pinned_shortcut_points():
    assert LAUNCHER.is_file()


def test_the_launcher_is_what_its_spec_renders():
    assert_launchers_match_their_specs(REPO_ROOT)


@on_windows
def test_the_launcher_runs_the_package_from_this_checkout_on_its_venv():
    report = dry_run(LAUNCHER)

    assert Path(report.value("interpreter")).parent == REPO_ROOT / ".venv" / "Scripts"
    assert Path(report.value("directory")) == REPO_ROOT
    assert report.value("arguments") == "-m scripture"


@on_windows
def test_a_launch_that_dies_importing_leaves_its_traceback_in_the_sessions_folder():
    report = dry_run(LAUNCHER)

    assert Path(report.value("log")) == REPO_ROOT / "sessions" / "scripture_launcher.log"

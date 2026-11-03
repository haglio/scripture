from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
from app_support.launch_smoke import (
    assert_an_unresolvable_import_is_caught,
    assert_every_import_resolves,
    assert_the_walk_reached,
    launch_imports,
)
from app_support.launcher import dry_run

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "scripture"
LAUNCHER = REPO_ROOT / "launch_scripture.vbs"

LAUNCH_FILES = (
    REPO_ROOT / PACKAGE / "__main__.py",
    REPO_ROOT / PACKAGE / "main.py",
)

_REACHED_ONLY_FROM_INSIDE_MAIN = ("scripture.gui", "PyQt6.QtWidgets")


def _launch_imports() -> list[str]:
    return launch_imports(PACKAGE, LAUNCH_FILES)



def _the_interpreter_that_will_run(named_by_the_launcher: str) -> str:
    if Path(named_by_the_launcher).is_file():
        return named_by_the_launcher
    return sys.executable


def _run_the_launchs_way(statements: list[str]) -> subprocess.CompletedProcess:
    report = dry_run(LAUNCHER)

    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env["QT_QPA_PLATFORM"] = "offscreen"

    driver = "\n".join(
        [
            "from scripture import content as _content",
            "_content.LOCAL_CONTENT = _content.EXAMPLE_CONTENT",
            *statements,
        ]
    )
    return subprocess.run(
        [_the_interpreter_that_will_run(report.value("interpreter")), "-c", driver],
        cwd=report.value("directory"),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )



runs_the_launcher = pytest.mark.skipif(sys.platform != "win32", reason="the Windows script host")


@runs_the_launcher
def test_the_launch_imports_everything_it_names():
    assert_every_import_resolves(_run_the_launchs_way, _launch_imports())


def test_the_walk_reaches_the_imports_buried_in_main():
    assert_the_walk_reached(_launch_imports(), _REACHED_ONLY_FROM_INSIDE_MAIN)


@runs_the_launcher
def test_a_launch_import_that_cannot_resolve_fails_here():
    assert_an_unresolvable_import_is_caught(
        _run_the_launchs_way, _launch_imports(), "scripture.gui")

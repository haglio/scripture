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

LAUNCH_FILES_THE_WHOLE_SEQUENCE_WHILE_EVERY_HELPER_MAIN_CALLS_LIVES_IN_THEM = (
    REPO_ROOT / PACKAGE / "__main__.py",
    REPO_ROOT / PACKAGE / "main.py",
)

_REACHED_ONLY_FROM_INSIDE_MAIN = ("scripture.gui", "PyQt6.QtWidgets")
_THE_PYTHONPATH_THE_LAUNCHER_NEVER_EXPORTS = "PYTHONPATH"
_A_PUBLIC_CHECKOUT_HAS_ONLY_THE_EXAMPLE_SO_THE_LAUNCH_COMES_UP_ON_IT = (
    "from scripture import content as _content",
    "_content.LOCAL_CONTENT = _content.EXAMPLE_CONTENT",
)


def _launch_imports() -> list[str]:
    return launch_imports(PACKAGE, LAUNCH_FILES_THE_WHOLE_SEQUENCE_WHILE_EVERY_HELPER_MAIN_CALLS_LIVES_IN_THEM)



def _the_interpreter_that_will_run_or_the_suites_own_where_the_gate_and_a_worktree_have_no_venv(
        named_by_the_launcher: str) -> str:
    if Path(named_by_the_launcher).is_file():
        return named_by_the_launcher
    return sys.executable


def _run_the_launchs_way(statements: list[str]) -> subprocess.CompletedProcess:
    report = dry_run(LAUNCHER)

    env = {k: v for k, v in os.environ.items() if k != _THE_PYTHONPATH_THE_LAUNCHER_NEVER_EXPORTS}
    env["QT_QPA_PLATFORM"] = "offscreen"

    driver = "\n".join([*_A_PUBLIC_CHECKOUT_HAS_ONLY_THE_EXAMPLE_SO_THE_LAUNCH_COMES_UP_ON_IT, *statements])
    return subprocess.run(
        [_the_interpreter_that_will_run_or_the_suites_own_where_the_gate_and_a_worktree_have_no_venv(
            report.value("interpreter")), "-c", driver],
        cwd=report.value("directory"),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )



runs_the_launcher = pytest.mark.skipif(sys.platform != "win32", reason="the Windows script host")


@runs_the_launcher
def test_the_launch_imports_everything_it_names_the_launcher_gate_resolves_the_command_and_never_runs_it_and_a_failed_launch_shows_only_a_line_in_the_launchers_log():
    assert_every_import_resolves(_run_the_launchs_way, _launch_imports())


def test_the_walk_reaches_the_imports_buried_in_main():
    assert_the_walk_reached(_launch_imports(), _REACHED_ONLY_FROM_INSIDE_MAIN)


@runs_the_launcher
def test_a_launch_import_that_cannot_resolve_fails_here():
    assert_an_unresolvable_import_is_caught(
        _run_the_launchs_way, _launch_imports(), "scripture.gui")


def test_the_window_and_qt_are_imported_only_once_a_window_is_wanted():
    done = _run_the_launchs_way([
        "import sys, scripture.main",
        "print(sorted(name for name in ('scripture.gui', 'PyQt6.QtWidgets') if name in sys.modules))",
    ])

    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "[]"


def test_the_launcher_runs_the_venv_the_suite_runs_on_so_a_green_suite_speaks_for_the_app_that_opens():
    named = Path(dry_run(LAUNCHER).value("interpreter"))

    assert named.parent.parent.name == ".venv"
    assert named.name == "python.exe"


def test_nothing_but_the_package_sits_beside_it_at_the_root_an_installed_copy_would_leave_a_root_module_behind():
    assert sorted(path.name for path in REPO_ROOT.glob("*.py")) == ["vulture_whitelist.py"]

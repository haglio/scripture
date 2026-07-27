"""What launch_scripture.vbs must put on PYTHONPATH for the app to import, and
that the file actually runs.

The launcher prefers the conda interpreter (it has torch+CUDA for the tracker)
over the repo's own venv, and that interpreter has no ``shared_ui`` installed
and no ``.pth`` pointing at it. So whatever the launcher exports is the only
thing making ``shared_ui`` importable there -- and the parent directory alone
does not: it makes ``shared_ui`` a namespace package rooted at the *checkout*,
whose ``colors`` submodule sits one level further down.

Everything above reads the file as text, which is how a launcher that could not
run at all passed this suite for a week: a renamed object left one call site
naming an undeclared variable, and VBScript failed at run time, before the
first log line, so the icon did nothing and nothing recorded why. The last test
here runs the script.
"""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

LAUNCHER = Path(__file__).resolve().parents[1] / "launch_scripture.vbs"


def _launcher_text() -> str:
    return LAUNCHER.read_text(encoding="utf-8", errors="replace")


def test_the_launcher_exists_where_the_shortcut_points():
    assert LAUNCHER.is_file()


def test_pythonpath_carries_the_repo_parent_so_sibling_checkouts_resolve():
    assert "set PYTHONPATH=" in _launcher_text()
    assert "parentDir" in _launcher_text()


def test_pythonpath_also_carries_the_shared_ui_checkout_itself():
    """The parent alone yields a namespace package with no importable submodules."""
    text = _launcher_text()

    assert "shared_ui" in text, (
        "launch_scripture.vbs must put the shared_ui checkout on PYTHONPATH; "
        "without it `from shared_ui.colors import ...` fails under the conda "
        "interpreter the launcher prefers"
    )


def test_the_launcher_runs_the_package_from_the_repo_root():
    text = _launcher_text()

    assert "-m scripture" in text
    assert "projectRoot" in text


def test_the_launcher_declares_its_variables():
    """Option Explicit turns a renamed-and-missed variable into a compile error.

    Without it the same slip is an "Object required" at run time, from a hidden
    window, with no log line -- which is exactly how this file broke.
    """
    assert "Option Explicit" in _launcher_text()


@pytest.mark.skipif(shutil.which("cscript") is None, reason="Windows Script Host only")
def test_the_launcher_runs_end_to_end_and_resolves_a_launch_command():
    """Run the real file with no arguments -- the shortcut's exact invocation.

    The dry-run switch is an environment variable rather than an argument for
    that reason. Deciding it from ``WScript.Arguments(0)`` meant this test
    passed an argument the shortcut never passes, and VBScript's non-short-
    circuiting ``And`` then read element 0 of an empty list on every real
    launch: "Subscript out of range", before the first log line, icon does
    nothing. The test was green throughout.
    """
    result = subprocess.run(
        ["cscript", "//nologo", str(LAUNCHER)],
        capture_output=True,
        text=True,
        cwd=LAUNCHER.parent,
        env={**os.environ, "SCRIPTURE_LAUNCHER_DRY_RUN": "1"},
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.startswith("OK: "), result.stdout + result.stderr
    assert "-m scripture" in result.stdout


def test_the_dry_run_switch_is_not_an_argument():
    """Whatever gates the dry run must not be read off the argument list.

    A launcher that behaves differently under test than under the shortcut is
    not a launch test; the shortcut passes no arguments at all.
    """
    text = _launcher_text()

    assert "WScript.Arguments" not in text
    assert "SCRIPTURE_LAUNCHER_DRY_RUN" in text

"""Which interpreter launch_scripture.vbs runs the app on, and that it runs.

Scripture is launched on the project venv and nothing else. It used to prefer
``%USERPROFILE%\\miniconda3\\python.exe`` because the venv's torch was CPU-only
and ``cotracker_tracking`` pins every tensor to ``cuda`` with no fallback -- so
the app ran on an interpreter this suite never touches and ``pyproject.toml``
never describes, and needed ``PYTHONPATH`` entries to hand it a ``shared_ui``
it had no other way to see. The venv carries the CUDA build now, so all of that
is gone: same interpreter for the tests and the app, and no path juggling.

Most of this reads the file as text, which is how a launcher that could not run
at all passed this suite for a week: a renamed object left one call site naming
an undeclared variable, and VBScript failed at run time, before the first log
line, so the icon did nothing and nothing recorded why. One test here runs the
script.
"""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

LAUNCHER = Path(__file__).resolve().parents[1] / "launch_scripture.vbs"


def _launcher_text() -> str:
    return LAUNCHER.read_text(encoding="utf-8", errors="replace")


def _launcher_code() -> str:
    """The launcher with its comment lines dropped.

    The tests below that assert something is *absent* have to read code only.
    The file explains at length why conda and PYTHONPATH are gone, and that
    explanation is worth keeping -- but a test that greps the whole file would
    read it as the very thing it forbids, and the fix would be deleting the
    comment rather than fixing the launcher.
    """
    lines = _launcher_text().splitlines()
    return "\n".join(line for line in lines if not line.lstrip().startswith("'"))


def test_the_launcher_exists_where_the_shortcut_points():
    assert LAUNCHER.is_file()


def test_the_launcher_runs_the_app_on_the_project_venv():
    """The venv is the only interpreter with this app's dependencies: the CUDA
    torch the tracker needs, and the editable ``shared_ui`` ``gui.py`` imports.
    It is also the one the suite runs on, which is what makes a green run say
    anything about the launch."""
    text = _launcher_text()

    assert ".venv\\Scripts\\python.exe" in text


def test_there_is_no_other_interpreter_to_fall_back_to():
    """A PATH python has neither torch nor shared_ui, so falling back to one is
    not a lesser launch but a broken one -- it dies while importing, before any
    window and before any log line. Conda is gone for the same reason it was
    ever here: it was the only interpreter with CUDA torch, and now it is not."""
    code = _launcher_code()

    assert "miniconda" not in code.lower()
    assert "where " not in code
    assert "py -3" not in code


def test_the_launcher_hands_over_no_pythonpath():
    """The venv resolves ``shared_ui`` through the editable install's .pth and
    the working directory resolves the rest, so an exported ``PYTHONPATH`` would
    only be a second, divergent answer to a question already settled."""
    assert "set PYTHONPATH=" not in _launcher_code()


def test_the_launcher_runs_the_package_from_the_repo_root():
    text = _launcher_text()

    assert "-m scripture" in text
    assert "projectRoot" in text


def test_a_missing_venv_says_so_instead_of_doing_nothing():
    """The launcher starts the app hidden, so a launch that cannot start has no
    console to complain on. Naming the missing interpreter in a dialog is the
    difference between "Scripture is broken" and "Scripture's venv is gone"."""
    text = _launcher_text()

    assert "virtual environment is missing" in text
    assert "vbCritical" in text


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

    assert "WScript.Arguments" not in _launcher_code()
    assert "SCRIPTURE_LAUNCHER_DRY_RUN" in text

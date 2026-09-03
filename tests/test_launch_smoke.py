"""The launch smoke test: everything the launcher's command imports, imported.

``tests/test_launcher_contract.py`` proves the launcher resolves a command and
stops there -- it never runs the command. ``tests/test_gui_import.py`` runs one
import, ``scripture.gui``. Neither covers the launch as a whole, which is where
an app that will not open lives: ``main()`` reaches Qt and the GUI through
imports *inside* the function, and every other test here runs under
``tests/conftest.py``, which stands a QApplication up before the first test
module is collected. So a run can be green on a launch sequence that never
completes.

So this asks the launcher itself what it would run -- interpreter and working
directory -- and drives the whole import phase under exactly that. Nothing here
restates the launcher's decisions, so the launcher changing its mind changes
what is tested rather than leaving this quietly checking the old arrangement.

The walk that reads those imports off the AST, and the three assertions that
replay them, are ``app_support.launch_smoke``: seven repos carried a copy of the
same 200 lines, drifting. What stays here is the half that is this app's --
which files the launch executes, and how its launcher starts an interpreter.
"""
from __future__ import annotations

import os
import re
import shutil
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

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "scripture"
LAUNCHER = REPO_ROOT / "launch_scripture.vbs"

# The two files the launcher's ``-m scripture`` runs: the entrypoint, and the
# module holding main(). Between them they are the whole launch sequence.
LAUNCH_FILES = (
    REPO_ROOT / PACKAGE / "__main__.py",
    REPO_ROOT / PACKAGE / "main.py",
)

# Reached only from inside main(), so a module-level import test never saw it.
# Asserted present, so a walk that silently found nothing cannot pass as a
# clean launch.
_REACHED_ONLY_FROM_INSIDE_MAIN = ("scripture.gui", "PyQt6.QtWidgets")

_INTERPRETER = re.compile(r"&& \"(?P<python>[^\"]+)\" -m " + PACKAGE)
_WORKING_DIR = re.compile(r"cd /d \"(?P<cwd>[^\"]+)\"")


def _launch_imports() -> list[str]:
    return launch_imports(PACKAGE, LAUNCH_FILES)


# --------------------------------------------------------------------------
# Where the launcher would run them
# --------------------------------------------------------------------------

def _launch_command() -> str:
    """The command launch_scripture.vbs resolves, from the launcher itself.

    Deriving it beats restating it: whatever the launcher decides is then what
    gets tested, rather than a copy of its decisions kept here that can drift
    from it without anything noticing.
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
    return result.stdout.strip()


def _the_interpreter_that_will_run(named_by_the_launcher: str) -> str:
    """The launcher's interpreter where there is one, else the suite's own.

    The launcher names ``<checkout>/.venv/Scripts/python.exe`` unconditionally,
    and two places legitimately have no such file: CI, which runs a plain
    checkout with the dependencies installed into the runner's python, and an
    agent's worktree, whose venv lives back in the primary. In both, the
    interpreter running pytest is the one carrying this app's dependencies --
    and on a developer's machine it *is* the venv the launcher named, so the
    fallback changes nothing where the real launch happens.
    """
    if Path(named_by_the_launcher).is_file():
        return named_by_the_launcher
    return sys.executable


def _run_the_launchs_way(statements: list[str]) -> subprocess.CompletedProcess:
    command = _launch_command()
    interpreter = _INTERPRETER.search(command)
    assert interpreter, f"could not read the interpreter out of: {command}"
    working_dir = _WORKING_DIR.search(command)
    assert working_dir, f"the launcher names no working directory: {command}"

    # No PYTHONPATH -- the launcher exports none, so neither does this. What the
    # shell handed pytest is exactly what the icon does not get.
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env["QT_QPA_PLATFORM"] = "offscreen"

    driver = "\n".join(
        [
            # Before anything that reads content at import time: a public
            # checkout has only the committed example, so that is what the
            # launch has to come up on.
            "import content as _content",
            "_content.LOCAL_CONTENT = _content.EXAMPLE_CONTENT",
            *statements,
        ]
    )
    return subprocess.run(
        [_the_interpreter_that_will_run(interpreter["python"]), "-c", driver],
        cwd=working_dir["cwd"],
        env=env,
        capture_output=True,
        text=True,
    )


# --------------------------------------------------------------------------
# The tests
# --------------------------------------------------------------------------

runs_the_launcher = pytest.mark.skipif(
    shutil.which("cscript") is None, reason="Windows Script Host only"
)


@runs_the_launcher
def test_the_launch_imports_everything_it_names():
    """Failing here means the icon does nothing: the traceback goes to
    ``sessions/scripture_launcher.log`` and no window ever appears."""
    assert_every_import_resolves(_run_the_launchs_way, _launch_imports())


def test_the_walk_reaches_the_imports_buried_in_main():
    """The guard above is only worth anything if the walk found the lazy ones --
    the GUI is imported inside main(), which is the whole app."""
    assert_the_walk_reached(_launch_imports(), _REACHED_ONLY_FROM_INSIDE_MAIN)


@runs_the_launcher
def test_a_launch_import_that_cannot_resolve_fails_here():
    """A negative control: if the subprocess reported success regardless, every
    assertion above would pass vacuously and the guard would be decorative."""
    assert_an_unresolvable_import_is_caught(
        _run_the_launchs_way, _launch_imports(), "scripture.gui")


@runs_the_launcher
def test_the_launcher_starts_the_app_in_this_checkout():
    """The working directory is what makes the top-level ``content`` module
    importable, and what keeps a sibling checkout from answering for this one."""
    working_dir = _WORKING_DIR.search(_launch_command())

    assert working_dir
    assert Path(working_dir["cwd"]) == REPO_ROOT

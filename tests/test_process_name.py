"""Scripture says its own name in the Windows task list.

Windows takes what it shows about a process -- the Details tab's name, the
Processes tab's description, the icon beside it -- from the file the process was
started from, so a plain interpreter puts Scripture in the task list as one more
anonymous "Python".  That costs nothing until something strands a process, and
then the task list is the only way back and cannot say which row is safe to end,
among half a dozen identical ones belonging to different apps.

``app_support.process_identity`` makes a copy of the interpreter named,
described and marked for this app.  Naming this process on the way in is the one
thing that cannot be done -- writing the copy takes the very interpreter being
named -- so each run prepares it for the run after and the launcher picks it up.
Both halves are asserted: a launcher that never looks, or an app that never
prepares, leaves the app anonymous for good.
"""
from __future__ import annotations

from pathlib import Path

from app_support.process_identity import ProcessNamer

PROJECT_DIR = Path(__file__).resolve().parent.parent
APP_NAME = "Scripture"
ROLE = "Scripture"

LAUNCHER = (PROJECT_DIR / "launch_scripture.vbs").read_text(encoding="utf-8")
ENTRY_POINT = (PROJECT_DIR / "scripture/main.py").read_text(encoding="utf-8")


def test_the_launcher_prefers_the_copy_named_for_this_app():
    expected = ProcessNamer(APP_NAME).exe_name("python.exe", ROLE)

    assert expected in LAUNCHER, f"the launcher does not look for {expected}"
    # It overrides the plain interpreter rather than being overridden by it:
    # the launcher picks python.exe first and only then swaps in the named copy,
    # so the swap has to come second to win.
    assert LAUNCHER.rindex(expected) > LAUNCHER.index(r"pythonExe = projectRoot")
    assert "pythonExe = projectRoot" in LAUNCHER


def test_the_launcher_still_works_before_any_run_has_named_it():
    """The naming runs one launch behind, so a fresh checkout has no copy to
    find.  That must cost the name and nothing else."""
    assert r"\.venv\Scripts\python.exe" in LAUNCHER


def test_the_app_prepares_that_copy_for_next_time():
    assert "_name_this_process()" in ENTRY_POINT
    assert f'ProcessNamer("{APP_NAME}"' in ENTRY_POINT
    assert f'"{ROLE}"' in ENTRY_POINT


def test_it_prepares_the_interpreter_the_launcher_actually_runs():
    """The launcher runs python.exe -- it redirects the app's output into its
    log -- so naming pythonw would leave a copy nothing ever starts."""
    assert 'with_name("python.exe")' in ENTRY_POINT


def test_the_row_reads_as_the_app_and_nothing_more():
    # One app with one window, so the row is its name, not its name twice.
    assert ProcessNamer(APP_NAME).description(ROLE) == APP_NAME


def test_it_stamps_its_own_mark():
    assert (PROJECT_DIR / "icon.ico").is_file()
    assert "icon.ico" in ENTRY_POINT


def test_naming_never_takes_a_launch_down():
    """A read-only venv or an antivirus hold must cost the name in the task list
    and nothing else."""
    body = ENTRY_POINT[ENTRY_POINT.index("def _name_this_process"):]
    body = body[:body.index("\ndef ", 1)]

    assert "except Exception:" in body

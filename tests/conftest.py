"""What the whole suite runs under.

The QApplication is a fixture a test asks for by name, never autouse. An autouse
one stood here once and no test ever asked for it -- the only PyQt6 import in
`tests/` was the fixture's own -- so its one effect was to leave the process
Qt-initialized before the first test module was collected, which is exactly the
condition `test_launch_smoke.py` exists to rule out: a green run on a launch
sequence that never completes. Asking by name keeps Qt out of every run that
does not build a widget.
"""

import os
import random

import pytest

# Ask Qt for a headless platform. The window fixture below builds a real
# QMainWindow, and the subprocesses that import `scripture.gui` set this in
# their own env -- but the day one of them stops, the difference is a window on
# the screen of the machine this is run from. The merge gate sets it too;
# setdefault lets a developer override it to watch something on a real display.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(scope="session")
def qt_app():
    """The one QApplication a widget test needs, built when one asks for it."""
    from PyQt6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


@pytest.fixture
def window(qt_app, tmp_path, monkeypatch):
    """A real `App`, with the last-session pointer aimed at a scratch file.

    Left alone, `App.__init__` finishes by reopening whatever project the
    developer had open last -- reading their video off disk in a unit test.
    """
    from scripture import gui

    monkeypatch.setattr(gui, "_LAST_PROJECT_FILE", tmp_path / ".last_session")
    app_window = gui.App()
    yield app_window
    # The close guard asks about unsaved work, and a modal question with no one
    # to answer it would hang the suite.
    app_window._mark_clean()
    app_window.close()


def pytest_collection_modifyitems(items):
    """Collect in a different order when asked, so a test that leans on the ones
    beside it fails on the commit that introduces the lean.

    ``TEST_COLLECTION_ORDER=reverse`` collects back to front;
    ``TEST_COLLECTION_ORDER=shuffle`` shuffles with ``TEST_COLLECTION_SEED`` (0
    unless given), so a red run can be repeated exactly.  Unset leaves the order
    alone; anything else is a typo, and a typo that silently ran forward would
    make the gate's second leg a green that proves nothing.
    """
    order = os.environ.get("TEST_COLLECTION_ORDER")
    if order is None:
        return
    if order == "reverse":
        items.reverse()
    elif order == "shuffle":
        random.Random(int(os.environ.get("TEST_COLLECTION_SEED", "0"))).shuffle(items)
    else:
        raise pytest.UsageError(
            f"TEST_COLLECTION_ORDER={order!r}: expected 'reverse' or 'shuffle'"
        )

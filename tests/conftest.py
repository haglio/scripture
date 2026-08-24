"""What the whole suite runs under.

No QApplication. There used to be one here, session-scoped and autouse, and no
test ever asked for it -- the only PyQt6 import in `tests/` was the fixture's
own. Its one effect was to leave the process Qt-initialised before the first
test module was collected, which is exactly the condition
`test_launch_smoke.py` exists to rule out: a green run on a launch sequence that
never completes. When `gui.py` gets widget tests they can ask for one by name.
"""

import os
import random

import pytest

# Ask Qt for a headless platform anyway. Nothing here builds a widget today, and
# the subprocesses that import `scripture.gui` set this in their own env -- but
# the day one of them stops, or a widget test arrives, the difference is a window
# on the screen of the machine this is run from. The merge gate sets it too;
# setdefault lets a developer override it to watch something on a real display.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


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

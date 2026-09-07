"""The suite asks Qt for a headless platform, whoever started it.

This reads the setting rather than asking a QApplication which platform it got.
The QApplication is a fixture the widget tests name, so a run that collects only
this one need not build it -- and building one here would put back the
Qt-initialized process `test_launch_smoke.py` exists to rule out. What it guards
is `conftest.py` keeping the setting, which is the difference between a headless
suite and a window on the screen of whoever ran it.
"""
from __future__ import annotations

import os


def test_the_suite_asks_for_an_offscreen_platform():
    assert os.environ.get("QT_QPA_PLATFORM") == "offscreen"

"""The suite asks Qt for a headless platform, whoever started it.

There is no QApplication here to ask which platform it got -- building one would
put back the Qt-initialised process `test_launch_smoke.py` exists to rule out --
so this reads the setting itself. It guards `conftest.py` against losing it, and
the day a widget test arrives it is already in place.
"""
import os


def test_the_suite_asks_for_an_offscreen_platform():
    assert os.environ.get("QT_QPA_PLATFORM") == "offscreen"

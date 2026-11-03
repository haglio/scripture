from __future__ import annotations

import os


def test_the_suite_asks_for_an_offscreen_platform():
    assert os.environ.get("QT_QPA_PLATFORM") == "offscreen"

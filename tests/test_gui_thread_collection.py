from __future__ import annotations

import gc
from pathlib import Path

import pytest
from shared_ui.gui_thread_collection import GuiThreadCollection
from shared_ui.gui_thread_guard import assert_every_application_collects_on_its_gui_thread

_CHECKOUT = Path(__file__).resolve().parents[1]


def test_the_suite_collects_cycles_on_the_gui_thread_as_scripture_does(qt_app):
    assert not gc.isenabled()
    assert qt_app.findChild(GuiThreadCollection) is not None


@pytest.mark.repo_check
def test_every_application_scripture_makes_collects_cycles_on_its_gui_thread():
    assert_every_application_collects_on_its_gui_thread(_CHECKOUT)

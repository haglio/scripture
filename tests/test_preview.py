from __future__ import annotations

import pytest
from shared_ui.palette import PREVIEW_INK
from shared_ui.preview import Preview

_A_PREVIEW = Preview(feature="the new scene cut")


@pytest.fixture
def a_preview_window(qt_app, tmp_path, monkeypatch):
    from scripture import gui  # noqa: PLC0415

    monkeypatch.setattr(gui, "_LAST_PROJECT_FILE", tmp_path / ".last_session")
    window = gui.App(preview=_A_PREVIEW)
    yield window
    window._mark_clean()
    window.close()


def test_a_preview_is_titled_for_the_feature_it_demos(a_preview_window):
    assert a_preview_window.windowTitle() == "Scripture — preview of the new scene cut"


def test_a_preview_wears_its_letter_in_the_preview_ink(a_preview_window):
    letter = a_preview_window.windowIcon().pixmap(32, 32).toImage()

    middle_of_the_s = letter.pixelColor(16, 16)
    assert (middle_of_the_s.red(), middle_of_the_s.green(), middle_of_the_s.blue()) == PREVIEW_INK

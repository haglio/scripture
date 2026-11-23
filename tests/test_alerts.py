from __future__ import annotations

from shared_ui.alert import Level
from shared_ui.testing import alert_shown

from scripture import alerts


def test_a_warning_is_the_yellow_triangle_window():
    with alert_shown() as shown:
        alerts.show_warning("No video", "Open a video first.")

    assert shown.call_args.kwargs["level"] is Level.WARNING


def test_an_error_is_the_red_x_window():
    with alert_shown() as shown:
        alerts.show_error("Auto processing error", "Something made up went wrong.")

    assert shown.call_args.kwargs.get("level", Level.ERROR) is Level.ERROR

from __future__ import annotations

from shared_ui.testing import alert_shown

from scripture import alerts


def test_a_warning_reaches_the_window_with_nothing_to_copy():
    with alert_shown() as shown:
        alerts.show_warning("No video", "Open a video first.")

    assert shown.call_args.kwargs["copyable"] is False


def test_an_error_offers_its_message_to_copy_whatever_its_icon():
    with alert_shown() as shown:
        alerts.show_error("Auto processing error", "Something made up went wrong.", warning_icon=True)

    assert shown.call_args.kwargs["copyable"] is True

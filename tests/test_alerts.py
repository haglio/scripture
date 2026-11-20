from __future__ import annotations

from shared_ui.testing import alert_shown

from scripture import alerts


def test_a_warning_he_would_never_report_reaches_the_window_with_nothing_to_copy():
    with alert_shown() as shown:
        alerts.show_warning("No video", "Open a video first.", copyable=False)

    assert shown.call_args.kwargs["copyable"] is False


def test_a_warning_keeps_copy_error_message_unless_told_otherwise():
    with alert_shown() as shown:
        alerts.show_warning("Auto processing error", "Something made up went wrong.")

    assert shown.call_args.kwargs["copyable"] is True

"""Scripture's window groups under its pinned taskbar shortcut."""
from __future__ import annotations

import sys

from scripture import main


def test_the_pin_is_stamped_with_the_identity_the_process_claims(monkeypatch):
    claimed: list[str] = []
    stamped: list[tuple[str, tuple[str, ...]]] = []
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(main, "set_app_user_model_id", claimed.append)
    monkeypatch.setattr(main, "stamp_pinned_shortcuts",
                        lambda app_id, names: stamped.append((app_id, tuple(names))) or {})

    main._set_windows_app_user_model_id()

    assert claimed == [main.SCRIPTURE_APP_USER_MODEL_ID]
    assert stamped == [(main.SCRIPTURE_APP_USER_MODEL_ID, ("Scripture",))]


def test_a_process_windows_refuses_an_identity_still_stamps_its_pin(monkeypatch):
    stamped: list[str] = []

    def refuse(app_id: str) -> None:
        raise OSError("SetCurrentProcessExplicitAppUserModelID failed: HRESULT 0x80070057")

    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(main, "set_app_user_model_id", refuse)
    monkeypatch.setattr(main, "stamp_pinned_shortcuts",
                        lambda app_id, names: stamped.append(app_id) or {})

    main._set_windows_app_user_model_id()

    assert stamped == [main.SCRIPTURE_APP_USER_MODEL_ID]

"""Scripture's window groups under its pinned taskbar shortcut."""
from __future__ import annotations

import ast
import sys
from pathlib import Path

from shared_ui.preview import Preview

from scripture import main


def test_the_pin_is_stamped_with_the_identity_the_process_claims(monkeypatch):
    claimed: list[str] = []
    stamped: list[tuple[str, tuple[str, ...]]] = []
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(main, "set_app_user_model_id", claimed.append)
    monkeypatch.setattr(main, "stamp_pinned_shortcuts",
                        lambda app_id, names: stamped.append((app_id, tuple(names))) or {})

    main._set_windows_app_user_model_id(None)

    assert claimed == [main.APP_USER_MODEL_ID]
    assert stamped == [(main.APP_USER_MODEL_ID, ("Scripture",))]


def test_a_process_windows_refuses_an_identity_still_stamps_its_pin(monkeypatch):
    stamped: list[str] = []

    def refuse(app_id: str) -> None:
        raise OSError("SetCurrentProcessExplicitAppUserModelID failed: HRESULT 0x80070057")

    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(main, "set_app_user_model_id", refuse)
    monkeypatch.setattr(main, "stamp_pinned_shortcuts",
                        lambda app_id, names: stamped.append(app_id) or {})

    main._set_windows_app_user_model_id(None)

    assert stamped == [main.APP_USER_MODEL_ID]


def test_a_preview_claims_a_taskbar_button_of_its_own_and_leaves_the_pin_to_the_live_app(monkeypatch):
    claimed: list[str] = []
    stamped: list[str] = []
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(main, "set_app_user_model_id", claimed.append)
    monkeypatch.setattr(main, "stamp_pinned_shortcuts",
                        lambda app_id, names: stamped.append(app_id) or {})

    main._set_windows_app_user_model_id(Preview(feature=None))

    assert claimed == [f"{main.APP_USER_MODEL_ID}.Preview"]
    assert stamped == [main.APP_USER_MODEL_ID]


def _calls_in_main() -> set[str]:
    tree = ast.parse(Path(main.__file__).read_text(encoding="utf-8"))
    body, = (node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
    return {ast.unparse(node) for node in ast.walk(body) if isinstance(node, ast.Call)}


def test_the_launch_hands_this_checkouts_preview_to_the_taskbar_and_the_window():
    calls = _calls_in_main()

    assert {"preview_of(PROJECT_DIR)", "_set_windows_app_user_model_id(preview)",
            "App(preview=preview)"} <= calls

from __future__ import annotations

from pathlib import Path

from shared_ui.app_icon import assert_follows_the_family_spec

import scripture

ICON_PATH = Path(scripture.__file__).resolve().parent.parent / "icon.ico"


def test_the_icon_is_the_familys_s():
    assert_follows_the_family_spec(ICON_PATH, "S")

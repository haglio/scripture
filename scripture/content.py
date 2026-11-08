from __future__ import annotations

from pathlib import Path
from typing import Any

from app_support.overlay import MissingOverlayKey, missing_keys, read_overlay

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = PACKAGE_DIR.parent
_WHERE_HIS_OWN_FILE_IS = PROJECT_DIR
_INSIDE_THE_PACKAGE_AN_INSTALLED_COPY_SHIPS_ONLY_PACKAGES_AND_A_ROOT_NAME_RESOLVES_BY_CWD = PACKAGE_DIR
LOCAL_CONTENT = _WHERE_HIS_OWN_FILE_IS / "content.local.json"
EXAMPLE_CONTENT = _INSIDE_THE_PACKAGE_AN_INSTALLED_COPY_SHIPS_ONLY_PACKAGES_AND_A_ROOT_NAME_RESOLVES_BY_CWD / "content.example.json"


def load_content(
    local_path: Path | None = None,
    example_path: Path | None = None,
) -> dict[str, Any]:
    return read_overlay(LOCAL_CONTENT if local_path is None else local_path,
                        EXAMPLE_CONTENT if example_path is None else example_path)


def refuse_an_incomplete_overlay() -> None:
    if missing := missing_keys(LOCAL_CONTENT, EXAMPLE_CONTENT):
        raise MissingOverlayKey((", ".join(missing),), LOCAL_CONTENT)

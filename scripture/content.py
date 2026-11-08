from __future__ import annotations

from pathlib import Path
from typing import Any

from app_support.overlay import MissingOverlayKey, missing_keys, read_overlay

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = PACKAGE_DIR.parent
LOCAL_CONTENT = PROJECT_DIR / "content.local.json"
EXAMPLE_CONTENT = PACKAGE_DIR / "content.example.json"


def load_content(
    local_path: Path | None = None,
    example_path: Path | None = None,
) -> dict[str, Any]:
    return read_overlay(LOCAL_CONTENT if local_path is None else local_path,
                        EXAMPLE_CONTENT if example_path is None else example_path)


def refuse_an_incomplete_overlay() -> None:
    if missing := missing_keys(LOCAL_CONTENT, EXAMPLE_CONTENT):
        raise MissingOverlayKey((", ".join(missing),), LOCAL_CONTENT)

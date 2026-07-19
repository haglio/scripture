"""Content overlay — the values that must not be published, loaded at runtime.

The detector's class vocabulary and the path to its weights describe the footage
this tool was built for, so they live in ``content.local.json`` (git-ignored)
rather than in source. A committed ``content.example.json`` documents the shape
and is what a fresh or public checkout loads; the tracker behaves the same
either way, since every class name reaches the code through here.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

PROJECT_DIR = Path(__file__).resolve().parent
LOCAL_CONTENT = PROJECT_DIR / "content.local.json"
EXAMPLE_CONTENT = PROJECT_DIR / "content.example.json"


def load_content(
    local_path: Path | None = None,
    example_path: Path | None = None,
) -> dict[str, Any]:
    """The local overlay's content when present, else the committed example."""
    local_path = LOCAL_CONTENT if local_path is None else local_path
    example_path = EXAMPLE_CONTENT if example_path is None else example_path
    path = local_path if local_path.exists() else example_path
    return json.loads(path.read_text(encoding="utf-8"))

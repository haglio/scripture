"""Content overlay — the values that must not be published, loaded at runtime.

The detector's class vocabulary and the path to its weights describe the footage
this tool was built for, so they live in ``content.local.json`` (git-ignored) at
the top of the checkout rather than in source. The committed
``content.example.json`` beside this module documents the shape and is what a
fresh or public checkout loads; the tracker behaves the same either way, since
every class name reaches the code through here.

It sits inside the package, and the example with it, because a module beside the
package is not in the distribution: ``[project.scripts]`` declares a ``scripture``
command, and a plain ``pip install .`` used to produce one that died on
``ModuleNotFoundError: content``. Beside the package it also claimed the bare
name ``content``, which in a venv holding several of these checkouts resolved by
working directory -- from anywhere but this one it was a sibling's overlay, whose
keys are different, so importing the pipeline raised at import time.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from app_support.overlay import read_overlay

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = PACKAGE_DIR.parent
#: Where the real values go, at the top of the checkout, where the owner's is.
LOCAL_CONTENT = PROJECT_DIR / "content.local.json"
#: The placeholder shape, inside the package so a built distribution carries it.
EXAMPLE_CONTENT = PACKAGE_DIR / "content.example.json"


def load_content(
    local_path: Path | None = None,
    example_path: Path | None = None,
) -> dict[str, Any]:
    """The local overlay's content when present, else the committed example."""
    return read_overlay(LOCAL_CONTENT if local_path is None else local_path,
                        EXAMPLE_CONTENT if example_path is None else example_path)

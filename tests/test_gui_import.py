"""The GUI module must import on its own, the way the launcher loads it.

Nothing else in this suite imports ``scripture.gui`` -- the dead-code check
reads it as text, not as a module -- so an import-time error in it survives a
fully green run and only shows up as an app that will not start.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _fresh_interpreter(statement: str) -> subprocess.CompletedProcess:
    """Run *statement* the way launch_scripture.vbs does: repo root as cwd.

    ``content`` is a top-level module beside the package, so the repo root has
    to be importable; the launcher gets that from its working directory.
    """
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env["QT_QPA_PLATFORM"] = "offscreen"
    return subprocess.run(
        [sys.executable, "-c", statement],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        env=env,
    )


def test_gui_module_imports_in_a_fresh_interpreter():
    result = _fresh_interpreter("import scripture.gui as m; print(m.App.__name__)")

    assert result.returncode == 0, result.stderr
    assert "App" in result.stdout


def test_the_detection_colors_are_whatever_the_loaded_overlay_says(tmp_path):
    """This compared `_DET_COLORS` with the dict it was built from, so it held
    for any overlay at all -- renaming every `class_colors` key left it green.
    It substitutes an overlay and asks what the map came out as instead.

    The class names are private, so a fabricated overlay is also the only kind
    a test may name.
    """
    overlay = json.loads(
        (REPO_ROOT / "content.example.json").read_text(encoding="utf-8"))
    overlay["class_colors"] = {"widget": [1, 2, 3], "sprocket": [4, 5, 6]}
    overlay_path = tmp_path / "content.local.json"
    overlay_path.write_text(json.dumps(overlay), encoding="utf-8")

    result = _fresh_interpreter(
        "import content; from pathlib import Path;"
        f"content.LOCAL_CONTENT = Path({str(overlay_path)!r});"
        "import scripture.gui as m;"
        "print(sorted(m._DET_COLORS), m._DET_COLORS['widget'].getRgb()[:3])"
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "['sprocket', 'widget'] (1, 2, 3)"

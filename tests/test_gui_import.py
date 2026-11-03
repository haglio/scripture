from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from shared_ui import palette

REPO_ROOT = Path(__file__).resolve().parents[1]


def _fresh_interpreter(statement: str) -> subprocess.CompletedProcess:
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env["QT_QPA_PLATFORM"] = "offscreen"
    return subprocess.run(
        [sys.executable, "-c", statement],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        env=env,
        check=False,
    )


def test_gui_module_imports_in_a_fresh_interpreter():
    result = _fresh_interpreter("import scripture.gui as m; print(m.App.__name__)")

    assert result.returncode == 0, result.stderr
    assert "App" in result.stdout


def test_the_detection_colors_are_the_familys_nearest_to_what_the_loaded_overlay_says(tmp_path):
    overlay = json.loads(
        (REPO_ROOT / "scripture" / "content.example.json").read_text(encoding="utf-8"))
    overlay["class_colors"] = {"widget": [250, 70, 70], "sprocket": [4, 5, 6]}
    overlay_path = tmp_path / "content.local.json"
    overlay_path.write_text(json.dumps(overlay), encoding="utf-8")

    result = _fresh_interpreter(
        "from scripture import content; from pathlib import Path;"
        f"content.LOCAL_CONTENT = Path({str(overlay_path)!r});"
        "import scripture.gui as m;"
        "print(sorted(m._DET_COLORS), m._DET_COLORS['widget'].getRgb()[:3])"
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == f"['sprocket', 'widget'] {palette.RED}"


def test_opening_the_window_does_not_load_torch_or_the_detector():
    result = _fresh_interpreter(
        "import sys, scripture.gui;"
        "print(sorted({'torch', 'ultralytics'} & set(sys.modules)))")

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "[]"

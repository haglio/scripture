"""The GUI module must import on its own, the way the launcher loads it.

Nothing else in this suite imports ``scripture.gui`` -- the dead-code check
reads it as text, not as a module -- so an import-time error in it survives a
fully green run and only shows up as an app that will not start.
"""

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


def test_the_detection_colors_come_from_the_content_overlay():
    """The class names are private, so the map is built from the overlay."""
    result = _fresh_interpreter(
        "import scripture.gui as m;"
        "from content import load_content;"
        "print(sorted(m._DET_COLORS) == sorted(load_content()['class_colors']))"
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "True"

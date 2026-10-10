from __future__ import annotations

import ast
import os
import subprocess
import sys
import textwrap
from pathlib import Path

from scripture import main

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_an_error_nothing_catches_is_logged_and_the_app_keeps_running():
    child = textwrap.dedent("""
        import sys
        from PyQt6.QtCore import QTimer
        from PyQt6.QtWidgets import QApplication
        from scripture.main import _log_errors_and_keep_running
        app = QApplication(sys.argv[:1])
        _log_errors_and_keep_running()
        def go_wrong():
            raise RuntimeError("an invented error no handler expects")
        QTimer.singleShot(0, go_wrong)
        QTimer.singleShot(100, app.quit)
        app.exec()
        print("still running", flush=True)
    """)
    ran = subprocess.run([sys.executable, "-c", child], cwd=REPO_ROOT, timeout=60,
                         env=os.environ | {"QT_QPA_PLATFORM": "offscreen"},
                         capture_output=True, encoding="utf-8", errors="replace",
                         creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))

    assert "still running" in ran.stdout, ran.stdout + ran.stderr
    assert "an invented error no handler expects" in ran.stderr


def test_the_launch_does_so_before_the_window_exists():
    tree = ast.parse(Path(main.__file__).read_text(encoding="utf-8"))
    body, = (node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
    first_call = {}
    for node in ast.walk(body):
        if isinstance(node, ast.Call):
            name = ast.unparse(node.func)
            first_call[name] = min(node.lineno, first_call.get(name, node.lineno))

    assert first_call["_log_errors_and_keep_running"] < first_call["open_scripture"]

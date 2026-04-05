"""Ensure no dead code accumulates in the scripture package."""

import subprocess
import sys


# Vulture false positives: Qt overrides and signal-slot callback parameters.
# Each entry is "path:name" where path is relative to the repo root.
WHITELIST = {
    # -- Qt method overrides (called by the framework, not by our code) --
    # ProcessWorker (QThread)
    "scripture/gui.py:ProcessWorker.run",
    # TimelineWidget (QWidget)
    "scripture/gui.py:TimelineWidget.paintEvent",
    "scripture/gui.py:TimelineWidget.mousePressEvent",
    "scripture/gui.py:TimelineWidget.mouseMoveEvent",
    "scripture/gui.py:TimelineWidget.mouseReleaseEvent",
    "scripture/gui.py:TimelineWidget.wheelEvent",
    # FrameCanvas (QWidget)
    "scripture/gui.py:FrameCanvas.paintEvent",
    "scripture/gui.py:FrameCanvas.mousePressEvent",
    "scripture/gui.py:FrameCanvas.mouseMoveEvent",
    "scripture/gui.py:FrameCanvas.mouseReleaseEvent",
    "scripture/gui.py:FrameCanvas.wheelEvent",
    # App (QMainWindow)
    "scripture/gui.py:App.closeEvent",
    # -- Signal-slot callback parameters (signal emits them; slot must accept) --
    "scripture/gui.py:_on_canvas_context_menu.gx",
    "scripture/gui.py:_on_canvas_context_menu.gy",
    "scripture/gui.py:_on_timeline_context_menu.gx",
    "scripture/gui.py:_on_timeline_context_menu.gy",
}


def _parse_vulture_line(line: str) -> str | None:
    """Extract 'path:name' from a vulture output line.

    Vulture output format:
        scripture\\gui.py:114: unused method 'run' (60% confidence)
        scripture\\gui.py:1174: unused variable 'gx' (100% confidence)

    We need to map these to whitelist keys.  For methods we look up the
    class that owns them; for variables in a method we use method.var.
    """
    # Strip trailing confidence
    if "unused" not in line:
        return None
    # e.g. "scripture\\gui.py:114: unused method 'run' (60% confidence)"
    parts = line.split(": unused ")
    if len(parts) != 2:
        return None
    file_and_line = parts[0]  # "scripture\\gui.py:114"
    rest = parts[1]           # "method 'run' (60% confidence)"

    file_path = file_and_line.rsplit(":", 1)[0].replace("\\", "/")

    # Extract the name from 'run', 'gx', etc.
    quote_start = rest.find("'")
    quote_end = rest.find("'", quote_start + 1)
    if quote_start == -1 or quote_end == -1:
        return None
    name = rest[quote_start + 1:quote_end]

    return f"{file_path}:{name}"


def _find_whitelist_match(key: str) -> bool:
    """Check whether a vulture finding matches any whitelist entry.

    key is "path:name" (e.g. "scripture/gui.py:run").
    Whitelist entries are "path:Class.method" or "path:method.var".
    We match if any whitelist entry ends with the bare name.
    """
    if key in WHITELIST:
        return True
    # "scripture/gui.py:run" matches "scripture/gui.py:ProcessWorker.run"
    path, name = key.rsplit(":", 1)
    return any(
        entry.startswith(path + ":") and entry.endswith("." + name)
        for entry in WHITELIST
    )


def test_no_dead_code():
    result = subprocess.run(
        [sys.executable, "-m", "vulture", "scripture/", "--min-confidence", "60"],
        capture_output=True, text=True,
    )
    lines = [l.strip() for l in result.stdout.splitlines() + result.stderr.splitlines() if l.strip()]

    unwhitelisted = []
    for line in lines:
        key = _parse_vulture_line(line)
        if key is None:
            continue
        if not _find_whitelist_match(key):
            unwhitelisted.append(line)

    assert not unwhitelisted, (
        "Vulture found dead code not in the whitelist:\n"
        + "\n".join(unwhitelisted)
        + "\n\nIf these are framework false positives, add them to WHITELIST in "
        "tests/test_dead_code.py."
    )

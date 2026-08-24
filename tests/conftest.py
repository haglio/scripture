"""What the whole suite runs under.

No QApplication. There used to be one here, session-scoped and autouse, and no
test ever asked for it -- the only PyQt6 import in `tests/` was the fixture's
own. Its one effect was to leave the process Qt-initialised before the first
test module was collected, which is exactly the condition
`test_launch_smoke.py` exists to rule out: a green run on a launch sequence that
never completes. When `gui.py` gets widget tests they can ask for one by name.
"""

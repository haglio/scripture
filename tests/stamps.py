from __future__ import annotations

# The keys a stamp keeps what made a file, and its version, under: the names
# they were first written with, which every file on disk already holds.
METHOD, METHOD_VERSION = "recipe", "recipe_version"

A_STAMP = {"schema": 1, "app": "scripture", "app_commit": "0123456789abcdef0123456789abcdef01234567",
           "app_dirty": False, METHOD: "axis_tracking", METHOD_VERSION: "1",
           "stamped_at": "2026-01-01T00:00:00+00:00"}

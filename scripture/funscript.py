"""This app's name on the funscripts it writes.

What a funscript document *is* -- its keys, the metadata block an outside
player reads, the order the actions go in -- belongs to every app in this
family that reads or writes one, and lives in :mod:`app_support.funscript`.
All that is this app's is the credit on the scripts it authors.
"""
from __future__ import annotations

from app_support.funscript import document

CREATOR = "scripture"


def build_funscript(actions: list[dict], duration_seconds: int) -> dict:
    """The family's funscript document over *actions*, credited to this app."""
    return document(actions, duration_seconds=duration_seconds, creator=CREATOR)

"""This app's name on what it writes.

What a funscript document *is* -- its keys, the metadata block an outside
player reads, the order the actions go in -- belongs to every app in this
family that reads or writes one, and lives in :mod:`app_support.funscript`.
What is this app's is its name: the credit on the scripts it authors, and the
stamp on every result its trackers make.
"""
from __future__ import annotations

from app_support import provenance
from app_support.funscript import document

CREATOR = "scripture"


def build_funscript(actions: list[dict], duration_seconds: int, *, provenance: dict | None) -> dict:
    """The family's funscript document over *actions*, credited to this app, and
    what made them."""
    return document(actions, duration_seconds=duration_seconds, creator=CREATOR) | {
        "provenance": provenance}


def made_by(recipe: str, recipe_version: str) -> dict:
    return provenance.stamp(CREATOR, anchor=__file__, recipe=recipe, recipe_version=recipe_version)

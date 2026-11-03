from __future__ import annotations

from app_support import provenance
from app_support.funscript import document

CREATOR = "scripture"


def build_funscript(actions: list[dict], duration_seconds: int, *, provenance: dict | None) -> dict:
    return document(actions, duration_seconds=duration_seconds, creator=CREATOR) | {
        "provenance": provenance}


def made_by(tracker: str, tracker_version: str) -> dict:
    return provenance.stamp(CREATOR, anchor=__file__, method=tracker, method_version=tracker_version)

"""Funscript JSON generation."""

import json


def build_funscript(actions: list[dict], duration_seconds: int) -> dict:
    """Build a complete funscript dict from a list of actions."""
    return {
        "actions": sorted(actions, key=lambda a: a["at"]),
        "inverted": False,
        "metadata": {
            "bookmarks": [],
            "chapters": [],
            "creator": "scripture",
            "description": "",
            "duration": duration_seconds,
            "license": "",
            "notes": "",
            "performers": [],
            "script_url": "",
            "tags": [],
            "title": "",
            "type": "basic",
            "video_url": "",
        },
        "range": 100,
        "version": "1.0",
    }


def save_funscript(funscript: dict, output_path: str) -> None:
    """Write funscript dict to a JSON file."""
    with open(output_path, "w") as f:
        json.dump(funscript, f, indent=2)

"""Project save/load — persists splits and axis annotations as JSON."""

import json


def save_project(path: str, state: dict) -> None:
    with open(path, "w") as f:
        json.dump(state, f, indent=2)


def load_project(path: str) -> dict:
    with open(path) as f:
        return json.load(f)

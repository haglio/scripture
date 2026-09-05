"""The file actions sit at the right of the toolbar, the way Evolver's do.

They were flush left behind a margin pad, which left the two apps' top bars
reading as different chrome for the same kind of row.  What puts them right is
the expanding spacer ahead of them -- so this checks the ORDER, which is the
thing that would silently come undone if the actions were ever moved back up.
"""

from __future__ import annotations

import ast
from pathlib import Path

_GUI = Path(__file__).resolve().parent.parent / "scripture" / "gui.py"


def _toolbar_calls() -> list[str]:
    """Every `tb.<something>(...)` in the order the window builds the toolbar."""
    tree = ast.parse(_GUI.read_text(encoding="utf-8"))
    calls = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        target = node.func.value
        if isinstance(target, ast.Name) and target.id == "tb":
            calls.append((node.lineno, node.func.attr, ast.unparse(node)))
    return [f"{attr}:{text}" for _line, attr, text in sorted(calls)]


def test_the_spacer_comes_before_the_file_actions():
    calls = _toolbar_calls()
    spacer = next(i for i, c in enumerate(calls) if "_spacer" in c)
    first_action = next(i for i, c in enumerate(calls) if c.startswith("addAction"))

    assert spacer < first_action, "the actions would sit flush left again"


def test_the_row_still_ends_with_its_margin_pad():
    calls = _toolbar_calls()

    assert "right_pad" in calls[-1], calls[-1]


def test_the_toolbar_colors_come_from_the_family_palette():
    # The icons were tinted with this app's own near-white and the abort mark
    # with its own red.  Read off the syntax tree rather than as text, so a
    # reformat of the assignment cannot turn this red with the color unchanged.
    tree = ast.parse(_GUI.read_text(encoding="utf-8"))
    icon_color = next(
        node.value for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "_ICON_COLOR" for t in node.targets))
    literals = {node.value for node in ast.walk(tree)
                if isinstance(node, ast.Constant) and isinstance(node.value, str)}

    assert ast.unparse(icon_color) == "TEXT_PRIMARY.name()"
    assert "#ddd" not in literals
    assert "#ff6666" not in literals

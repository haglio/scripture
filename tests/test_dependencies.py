"""What this repo needs is what its pyproject says, in each of the ways it says it.

A package that imports something nobody declared works on the machine that
happened to have it and dies on the merge gate, which installs exactly what the
pyproject says; so does a version nobody bounded, a sibling checkout nobody
recorded, and a Python floor no run proves.  The gates are the family's
(``app_support.dependencies``); what is here is which packages are this repo's
own and which trees to read.
"""
from __future__ import annotations

from pathlib import Path

from app_support.dependencies import (
    assert_every_dependency_is_bounded,
    assert_every_import_is_declared,
    assert_every_sibling_is_declared,
    assert_every_sibling_is_pinned,
    assert_the_declared_floor_is_the_one_the_gate_runs,
)

ROOT = Path(__file__).resolve().parent.parent
TREES = [ROOT / "scripture", ROOT / "tests", ROOT / "tools",
         ROOT / "vulture_whitelist.py"]


def test_every_third_party_import_is_declared():
    """`local` naming the package alone is also what holds the package to it:
    a module put back beside the package and imported from inside it reads as an
    undeclared third-party dependency here, which is what it is once installed."""
    assert_every_import_is_declared(
        ROOT, [ROOT / "scripture"], ROOT / "pyproject.toml", local=("scripture",))


def test_every_requirement_has_an_upper_bound():
    assert_every_dependency_is_bounded(ROOT / "pyproject.toml")


def test_every_sibling_this_repo_needs_is_declared():
    assert_every_sibling_is_declared(ROOT, TREES, ROOT / "pyproject.toml")


def test_every_sibling_is_named_at_a_tag():
    assert_every_sibling_is_pinned(ROOT / "pyproject.toml")


def test_the_declared_floor_is_the_one_the_gate_runs():
    assert_the_declared_floor_is_the_one_the_gate_runs(
        ROOT / "pyproject.toml", ROOT / ".github" / "workflows" / "merge-gate.yml")

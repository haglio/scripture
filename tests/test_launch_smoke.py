"""The launch smoke test: everything the launcher's command imports, imported.

``tests/test_launcher_contract.py`` proves the launcher resolves a command, and
stops there -- it never runs the command. ``tests/test_gui_import.py`` imports
``scripture.gui`` in a fresh interpreter, but under *this* interpreter, on
*this* suite's path. Neither covers the gap between them, which is where an app
that will not open lives: the launcher prefers the conda interpreter over this
repo's ``.venv``, and puts ``shared_ui`` on ``PYTHONPATH`` by hand because that
interpreter has no ``.pth`` pointing at it. A module that imports cleanly under
the venv the suite runs on can therefore fail under the interpreter that
actually launches, and the icon just does nothing.

So this asks the launcher itself what it would run -- interpreter, working
directory, ``PYTHONPATH`` -- and drives the launch's import phase under exactly
that. Nothing here restates the launcher's decisions, so the launcher changing
its mind (a different interpreter, a path dropped) changes what is tested
rather than leaving this quietly checking the old arrangement.

The statements come off the AST of the files the launch executes rather than a
list maintained here, so the next import added to ``main()`` is covered without
anyone remembering to add it. They are replayed whole -- ``from X import a, b``
rather than ``import X`` -- so a symbol the launch names but the module no
longer defines fails here too.
"""
from __future__ import annotations

import ast
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "scripture"
LAUNCHER = REPO_ROOT / "launch_scripture.vbs"

# The two files the launcher's ``-m scripture`` runs: the entrypoint, and the
# module holding main(). Between them they are the whole launch sequence.
LAUNCH_FILES = (
    REPO_ROOT / PACKAGE / "__main__.py",
    REPO_ROOT / PACKAGE / "main.py",
)

# Reached only from inside main(), so a module-level import test never saw it.
# Asserted present, so a walk that silently found nothing cannot pass as a
# clean launch.
_REACHED_ONLY_FROM_INSIDE_MAIN = ("scripture.gui", "PyQt6.QtWidgets")

# Only these two. A broad ``except Exception`` around a launch body is an error
# *reporter* -- it puts a dialog on screen or writes a crash log -- so an import
# inside it is required, not optional: it failing is exactly the launch failure
# this file exists to catch.
_TOLERATED_BY = {"ImportError", "ModuleNotFoundError"}

_COMMAND = re.compile(
    r"set PYTHONPATH=(?P<pythonpath>.*?)&&(?P<python>.*?) -m " + PACKAGE
)
_WORKING_DIR = re.compile(r"cd /d \"(?P<cwd>[^\"]+)\"")


# --------------------------------------------------------------------------
# What the launch imports
# --------------------------------------------------------------------------

def _is_type_checking(test: ast.expr) -> bool:
    """``if TYPE_CHECKING:`` bodies are never executed, at launch or anywhere."""
    if isinstance(test, ast.Name):
        return test.id == "TYPE_CHECKING"
    return isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"


def _tolerates_a_missing_module(handlers: list[ast.ExceptHandler]) -> bool:
    for handler in handlers:
        if handler.type is None:  # bare except -- catches everything, promises nothing
            return False
        caught = (
            handler.type.elts if isinstance(handler.type, ast.Tuple) else [handler.type]
        )
        if any(isinstance(n, ast.Name) and n.id in _TOLERATED_BY for n in caught):
            return True
    return False


def _optional_imports(tree: ast.Module) -> set[int]:
    """Imports whose absence the module already handles, so the launch survives
    them and this test must not insist on them."""
    optional: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.If) and _is_type_checking(node.test):
            body = node.body
        elif isinstance(node, ast.Try) and _tolerates_a_missing_module(node.handlers):
            body = node.body
        else:
            continue
        for statement in body:
            for inner in ast.walk(statement):
                optional.add(id(inner))
    return optional


def _render(node: ast.Import | ast.ImportFrom) -> str:
    """The import statement as the launch executes it, relative made absolute.

    Both launch files sit at the top of the package, so a relative import is
    never deeper than one level.
    """
    names = ", ".join(
        alias.name + (f" as {alias.asname}" if alias.asname else "")
        for alias in node.names
    )
    if isinstance(node, ast.Import):
        return f"import {names}"
    assert node.level <= 1, f"unexpected relative import depth in {PACKAGE}"
    module = node.module or ""
    if node.level:
        module = f"{PACKAGE}.{module}" if module else PACKAGE
    return f"from {module} import {names}"


def _is_a_compiler_directive(node: ast.Import | ast.ImportFrom) -> bool:
    """``from __future__ import ...`` loads no module -- it is a flag to the
    compiler, and it is only legal at the top of a file, so replaying it among
    the others is a SyntaxError rather than a check of anything."""
    return isinstance(node, ast.ImportFrom) and node.module == "__future__"


def _launch_imports() -> list[str]:
    statements: list[str] = []
    for path in LAUNCH_FILES:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        optional = _optional_imports(tree)
        for node in ast.walk(tree):
            if not isinstance(node, (ast.Import, ast.ImportFrom)):
                continue
            if id(node) in optional or _is_a_compiler_directive(node):
                continue
            statements.append(_render(node))
    return statements


# --------------------------------------------------------------------------
# Where the launcher would run them
# --------------------------------------------------------------------------

def _launch_command() -> str:
    """The command launch_scripture.vbs resolves, from the launcher itself.

    Deriving it beats restating it: whatever the launcher decides about the
    interpreter and the path is then what gets tested, including in CI, where
    there is no conda and no ``.venv`` and it falls through to PATH.
    """
    result = subprocess.run(
        ["cscript", "//nologo", str(LAUNCHER)],
        capture_output=True,
        text=True,
        cwd=LAUNCHER.parent,
        env={**os.environ, "SCRIPTURE_LAUNCHER_DRY_RUN": "1"},
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.startswith("OK: "), result.stdout + result.stderr
    return result.stdout.strip()


def _argv(python_command: str) -> list[str]:
    """Split the launcher's interpreter into argv, unquoting each token.

    It is ``"C:\\...\\python.exe"`` for conda or the venv and a bare ``python``
    or ``py -3`` off PATH, so it cannot just be handed over whole.
    """
    return [token.strip('"') for token in python_command.strip().split() if token]


def _run_the_launchs_way(statements: list[str]) -> subprocess.CompletedProcess:
    command = _launch_command()
    parts = _COMMAND.search(command)
    assert parts, f"could not read the launch command out of: {command}"
    working_dir = _WORKING_DIR.search(command)
    assert working_dir, f"the launcher names no working directory: {command}"

    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env["PYTHONPATH"] = parts["pythonpath"]
    env["QT_QPA_PLATFORM"] = "offscreen"

    driver = "\n".join(
        [
            # Before anything that reads content at import time: a public
            # checkout has only the committed example, so that is what the
            # launch has to come up on.
            "import content as _content",
            "_content.LOCAL_CONTENT = _content.EXAMPLE_CONTENT",
            *statements,
        ]
    )
    return subprocess.run(
        [*_argv(parts["python"]), "-c", driver],
        cwd=working_dir["cwd"],
        env=env,
        capture_output=True,
        text=True,
    )


# --------------------------------------------------------------------------
# The tests
# --------------------------------------------------------------------------

runs_the_launcher = pytest.mark.skipif(
    shutil.which("cscript") is None, reason="Windows Script Host only"
)


@runs_the_launcher
def test_the_launch_imports_everything_it_names():
    """Failing here means the icon does nothing: the traceback goes to
    ``sessions/scripture_launcher.log`` and no window ever appears."""
    result = _run_the_launchs_way(_launch_imports())

    assert result.returncode == 0, result.stderr


def test_the_walk_reaches_the_imports_buried_in_main():
    """The guard above is only worth anything if the walk found the lazy ones --
    the GUI is imported inside main(), which is the whole app."""
    found = "\n".join(_launch_imports())

    for module in _REACHED_ONLY_FROM_INSIDE_MAIN:
        assert module in found, f"the launch imports {module}; the walk missed it"


@runs_the_launcher
def test_a_launch_import_that_cannot_resolve_fails_here():
    """A negative control: if the subprocess reported success regardless, every
    assertion above would pass vacuously and the guard would be decorative."""
    result = _run_the_launchs_way(
        [*_launch_imports(), "from scripture.gui import NoSuchSymbol"]
    )

    assert result.returncode != 0
    assert "NoSuchSymbol" in result.stderr


@runs_the_launcher
def test_the_launcher_starts_the_app_in_this_checkout():
    """The working directory is what makes the top-level ``content`` module
    importable, and what keeps a sibling checkout from answering for this one."""
    working_dir = _WORKING_DIR.search(_launch_command())

    assert working_dir
    assert Path(working_dir["cwd"]) == REPO_ROOT

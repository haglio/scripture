"""What launch_scripture.vbs must put on PYTHONPATH for the app to import.

The launcher prefers the conda interpreter (it has torch+CUDA for the tracker)
over the repo's own venv, and that interpreter has no ``shared_ui`` installed
and no ``.pth`` pointing at it. So whatever the launcher exports is the only
thing making ``shared_ui`` importable there -- and the parent directory alone
does not: it makes ``shared_ui`` a namespace package rooted at the *checkout*,
whose ``colors`` submodule sits one level further down.
"""

from pathlib import Path

LAUNCHER = Path(__file__).resolve().parents[1] / "launch_scripture.vbs"


def _launcher_text() -> str:
    return LAUNCHER.read_text(encoding="utf-8", errors="replace")


def test_the_launcher_exists_where_the_shortcut_points():
    assert LAUNCHER.is_file()


def test_pythonpath_carries_the_repo_parent_so_sibling_checkouts_resolve():
    assert "set PYTHONPATH=" in _launcher_text()
    assert "parentDir" in _launcher_text()


def test_pythonpath_also_carries_the_shared_ui_checkout_itself():
    """The parent alone yields a namespace package with no importable submodules."""
    text = _launcher_text()

    assert "shared_ui" in text, (
        "launch_scripture.vbs must put the shared_ui checkout on PYTHONPATH; "
        "without it `from shared_ui.colors import ...` fails under the conda "
        "interpreter the launcher prefers"
    )


def test_the_launcher_runs_the_package_from_the_repo_root():
    text = _launcher_text()

    assert "-m scripture" in text
    assert "projectRoot" in text

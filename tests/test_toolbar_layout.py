"""The file actions sit at the right of the toolbar, the way Evolver's do.

They were flush left after a margin pad, which left the two apps' top bars
reading as different chrome for the same kind of row.  What puts them right is
the expanding spacer ahead of them -- so this checks the ORDER, which is the
thing that would silently come undone if the actions were ever moved back up.

Asked of the toolbar the window actually builds.  This read the window's source
instead -- walking its syntax tree for calls on a local spelled `tb` -- so
renaming that local, or annotating the icon color's assignment, turned it red
with the row on screen unchanged (audit scripture/all/tests/002).
"""

from __future__ import annotations

import pytest
import qtawesome as qta
from PyQt6.QtWidgets import QSizePolicy, QToolBar
from shared_ui.colors import RED, TEXT_PRIMARY
from shared_ui.spacing import MARGIN_STANDARD

from scripture import gui

_FILE_ACTIONS = ["New", "Save", "Save As", "Load", "Export"]


@pytest.fixture
def toolbar(window) -> QToolBar:
    (found,) = window.findChildren(QToolBar)
    return found


def _texts(toolbar: QToolBar) -> list[str]:
    return [action.text() for action in toolbar.actions()]


def test_the_spacer_comes_before_the_file_actions(toolbar, window):
    places = toolbar.actions()
    spacer = places.index(window._spacer_action)
    first_action = min(places.index(a) for a in places if a.text() in _FILE_ACTIONS)

    assert spacer < first_action, "the actions would sit flush left again"


def test_the_spacer_is_what_takes_up_the_slack(toolbar, window):
    """An expanding widget is the whole mechanism; a fixed one pushes nothing."""
    spacer = toolbar.widgetForAction(window._spacer_action)

    assert spacer.sizePolicy().horizontalPolicy() == QSizePolicy.Policy.Expanding


def test_the_file_actions_stay_in_their_order(toolbar):
    assert [text for text in _texts(toolbar) if text in _FILE_ACTIONS] == _FILE_ACTIONS


def test_the_row_starts_and_ends_with_its_margin_pad(toolbar):
    places = toolbar.actions()
    first = toolbar.widgetForAction(places[0])
    last = toolbar.widgetForAction(places[-1])

    assert first.width() == MARGIN_STANDARD
    assert last.width() == MARGIN_STANDARD


def _drawn(icon) -> bytes:
    """The icon as pixels, which is the only place its color shows."""
    image = icon.pixmap(32, 32).toImage()
    return image.constBits().asstring(image.sizeInBytes())


def test_the_file_action_marks_are_drawn_in_the_family_ink(toolbar):
    (export,) = [a for a in toolbar.actions() if a.text() == "Export"]

    assert _drawn(export.icon()) == _drawn(
        qta.icon("fa5s.file-export", color=TEXT_PRIMARY.name()))


def test_the_abort_mark_is_drawn_in_the_family_red(window):
    assert _drawn(window._abort_action.icon()) == _drawn(
        qta.icon("fa5s.stop", color=RED.name()))


def test_the_icon_color_is_the_palette_s_and_not_this_app_s_own():
    assert TEXT_PRIMARY.name() == gui._ICON_COLOR

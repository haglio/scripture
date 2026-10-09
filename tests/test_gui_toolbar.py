"""The toolbar's buttons as the family dresses a toolbar's buttons, as Evolver's top bar does."""

from __future__ import annotations

from PyQt6.QtCore import QPoint, QRect
from PyQt6.QtWidgets import QApplication, QToolBar, QToolButton
from shared_ui.colors import BG_BUTTON, BORDER_SUBTLE


def _save_button_as_painted(window):
    window.show()
    QApplication.processEvents()
    [toolbar] = window.findChildren(QToolBar)
    [save] = [widget for widget in map(toolbar.widgetForAction, toolbar.actions())
              if isinstance(widget, QToolButton) and widget.text() == "Save"]
    painted = QApplication.primaryScreen().grabWindow(window.winId()).toImage()
    return painted.copy(QRect(save.mapTo(window, QPoint(0, 0)), save.size()))


def test_a_toolbar_button_wears_the_family_button_s_border_round_its_ground(window):
    save = _save_button_as_painted(window)

    middle = save.height() // 2
    assert save.pixelColor(0, middle).name() == BORDER_SUBTLE.name()
    assert save.pixelColor(3, middle).name() == BG_BUTTON.name()

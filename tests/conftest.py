import sys

import pytest
from PyQt6.QtWidgets import QApplication


@pytest.fixture(scope="session", autouse=True)
def qapp():
    """Ensure a QApplication exists for the test session."""
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    return app

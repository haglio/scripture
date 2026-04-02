import sys

from PyQt6.QtWidgets import QApplication

from scripture.gui import App


def main():
    app = QApplication.instance() or QApplication(sys.argv)
    window = App()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

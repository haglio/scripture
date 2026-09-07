# Vulture whitelist — false positives that are actually called by frameworks.
#
# Qt method overrides (called by the Qt event loop, not by our code)
_.paintEvent    # QWidget override (TimelineWidget, FrameCanvas)
_.mousePressEvent   # QWidget override
_.mouseMoveEvent    # QWidget override
_.mouseReleaseEvent # QWidget override
_.wheelEvent    # QWidget override
_.closeEvent    # QMainWindow override (App)

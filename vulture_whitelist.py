# Vulture whitelist — false positives that are actually called by frameworks.
#
# Qt method overrides (called by the Qt event loop, not by our code)
_.run           # QThread override (ProcessWorker)
_.paintEvent    # QWidget override (TimelineWidget, FrameCanvas)
_.mousePressEvent   # QWidget override
_.mouseMoveEvent    # QWidget override
_.mouseReleaseEvent # QWidget override
_.wheelEvent    # QWidget override
_.closeEvent    # QMainWindow override (App)

# Signal-slot callback parameters — the signal emits them; the slot must accept.
gx  # _on_canvas_context_menu, _on_timeline_context_menu
gy  # _on_canvas_context_menu, _on_timeline_context_menu

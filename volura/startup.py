"""Starting Volura: first the splash screen, then the heavy imports.

``main.py`` (also the entry of the PyInstaller build) and ``python -m volura`` call
:func:`main`; ``volura.app.main`` is the same start without the splash. It makes the
QApplication, shows the splash screen and then loads in steps, so the bar follows what
really happens:

    numpy and manifold3d -> OpenGL and the 3D view -> volura.app -> the main window

The imports of those steps are inside the function on purpose: that is what keeps
``import volura.app`` light, because nothing here is imported before the splash is shown.
"""
from __future__ import annotations

import os
import sys

from .i18n import tr
from .splash import Splash, no_splash


def _load_geometry():
    """The geometry engine: numpy and manifold3d."""
    import manifold3d       # noqa: F401
    import numpy            # noqa: F401


def main(argv=None):
    """Show the splash screen, load Volura per step and run the event loop."""
    argv = list(sys.argv if argv is None else argv)
    quiet = no_splash(argv)
    argv = [a for a in argv if a != "--no-splash"]
    if sys.platform.startswith("linux") and os.environ.get("WAYLAND_DISPLAY") \
            and not os.environ.get("QT_QPA_PLATFORM"):
        os.environ["QT_QPA_PLATFORM"] = "xcb"  # PyOpenGL needs GLX
    from PySide6.QtCore import QSettings, Qt
    from PySide6.QtWidgets import QApplication
    QApplication.setAttribute(Qt.AA_ShareOpenGLContexts)
    app = QApplication(argv)
    app.setApplicationName("Volura")
    app.setOrganizationName("Volura")
    from . import theme
    theme.apply(str(QSettings("Volura", "Volura").value("theme", theme.SYSTEM)), app)
    splash = None if quiet or no_splash() else Splash()   # no_splash() again: now Qt knows the platform
    if splash is not None:
        splash.show()

    def step(text, percent):
        if splash is not None:
            splash.step(text, percent)

    step(tr("Loading geometry engine…"), 15)
    _load_geometry()
    step(tr("Loading 3D view…"), 40)
    from . import viewport                      # noqa: F401  (PyOpenGL is what loads here)
    step(tr("Building the interface…"), 65)
    from . import app as volura_app
    from .icons import icon
    app.setWindowIcon(icon("app"))
    sys.excepthook = volura_app.excepthook
    step(tr("Restoring your settings…"), 85)
    win = volura_app.MainWindow()
    win.show()
    if len(argv) > 1 and os.path.exists(argv[1]):
        step(tr("Opening {file}…").format(file=os.path.basename(argv[1])), 95)
        win.file_open(argv[1])
    if splash is not None:
        splash.finish(win)
    return app.exec()

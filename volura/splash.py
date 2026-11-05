"""The splash screen: what is already to see while Volura is still loading.

Starting Volura takes a moment (numpy, manifold3d, OpenGL and the interface are heavy),
so the first window shows the logo, the name and version, a bar that follows the real
steps and the text of the step Volura is on. `volura.startup` fills it in between the
steps and closes it with :meth:`PySide6.QtWidgets.QSplashScreen.finish` as soon as the
main window is shown.

``--no-splash`` or ``VOLURA_NO_SPLASH=1`` starts without it, and so does running without
a display, which is how the tests run. The colors come from the palette, so the splash
follows the theme (light or dark) without knowing anything about it.
"""
from __future__ import annotations

import os
import sys

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QPainter, QPalette, QPixmap
from PySide6.QtWidgets import QApplication, QLabel, QProgressBar, QSplashScreen

from . import __version__

WIDTH, HEIGHT = 460, 190           # the splash in logical pixels
LOGO, MARGIN = 92, 28              # the logo and the space around everything
BAR_HEIGHT = 8


def offscreen() -> bool:
    """True when Qt runs without a display, which is how the tests run."""
    if os.environ.get("QT_QPA_PLATFORM", "").lower() == "offscreen":
        return True
    app = QApplication.instance()
    return app is not None and app.platformName() == "offscreen"


def no_splash(argv=()) -> bool:
    """No splash screen: asked for with --no-splash or VOLURA_NO_SPLASH=1, or no display."""
    if "--no-splash" in tuple(argv):
        return True
    if os.environ.get("VOLURA_NO_SPLASH", "").strip().lower() in ("1", "true", "yes", "on"):
        return True
    return offscreen()


def logo_path():
    """assets/volura.png: in the source tree or in the bundle PyInstaller made."""
    roots = [getattr(sys, "_MEIPASS", None),
             os.path.dirname(os.path.dirname(os.path.abspath(__file__)))]
    for root in roots:
        if root:
            path = os.path.join(str(root), "assets", "volura.png")
            if os.path.isfile(path):
                return path
    return None


def _device_pixel_ratio() -> float:
    """The scale of the screen, so the logo stays sharp on a HiDPI display."""
    app = QApplication.instance()
    screen = app.primaryScreen() if app is not None else None
    try:
        ratio = float(screen.devicePixelRatio()) if screen is not None else 1.0
    except Exception:                     # Qt versions without it, or no screen at all
        ratio = 1.0
    return ratio if ratio > 0 else 1.0


def _background() -> QPixmap:
    """The picture behind the text: the theme color, the logo and a thin edge."""
    ratio = _device_pixel_ratio()
    pm = QPixmap(round(WIDTH * ratio), round(HEIGHT * ratio))
    pm.setDevicePixelRatio(ratio)
    palette = QApplication.palette()
    window = palette.color(QPalette.Window)
    pm.fill(window)
    p = QPainter(pm)
    p.setRenderHint(QPainter.SmoothPixmapTransform)
    edge = window.darker(130) if window.lightness() > 128 else window.lighter(150)
    p.setPen(edge)
    p.drawRect(0, 0, WIDTH - 1, HEIGHT - 1)
    path = logo_path()
    if path:
        logo = QPixmap(path)
        if not logo.isNull():
            logo = logo.scaled(round(LOGO * ratio), round(LOGO * ratio),
                               Qt.KeepAspectRatio, Qt.SmoothTransformation)
            logo.setDevicePixelRatio(ratio)
            p.drawPixmap(MARGIN, round((HEIGHT - LOGO) / 2), logo)
    p.end()
    return pm


class Splash(QSplashScreen):
    """The splash screen of Volura: the logo, the version, a bar and the current step."""

    def __init__(self):
        super().__init__(_background())
        palette = self.palette()
        text = palette.color(QPalette.WindowText)
        muted = text.darker(150) if text.lightness() > 128 else text.lighter(140)
        left = MARGIN + LOGO + 22

        self.title = QLabel("Volura", self)
        title_font = QFont(self.font())
        title_font.setPointSizeF(max(title_font.pointSizeF(), 9.0) * 1.9)
        title_font.setBold(True)
        self.title.setFont(title_font)
        self.title.setStyleSheet(f"color: {text.name()}; background: transparent;")
        self.title.setGeometry(left, MARGIN, WIDTH - left - MARGIN, 40)

        self.version_label = QLabel(__version__, self)
        self.version_label.setStyleSheet(f"color: {muted.name()}; background: transparent;")
        self.version_label.setGeometry(left, MARGIN + 44, WIDTH - left - MARGIN, 22)

        self.step_label = QLabel(self)
        self.step_label.setStyleSheet(f"color: {text.name()}; background: transparent;")
        self.step_label.setGeometry(MARGIN, HEIGHT - 2 * MARGIN - BAR_HEIGHT - 24,
                                    WIDTH - 2 * MARGIN, 20)

        self.progress = QProgressBar(self)
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setTextVisible(False)
        self.progress.setFixedHeight(BAR_HEIGHT)
        window = palette.color(QPalette.Window)
        track = window.darker(125) if window.lightness() > 128 else window.lighter(135)
        self.progress.setStyleSheet(
            f"QProgressBar {{border: none; background: {track.name()}; border-radius: "
            f"{BAR_HEIGHT // 2}px;}}"
            f"QProgressBar::chunk {{background: {palette.color(QPalette.Highlight).name()};"
            f" border-radius: {BAR_HEIGHT // 2}px;}}")
        self.progress.setGeometry(MARGIN, HEIGHT - MARGIN - BAR_HEIGHT,
                                  WIDTH - 2 * MARGIN, BAR_HEIGHT)

    @property
    def percent(self) -> int:
        return self.progress.value()

    def step(self, text, percent=None):
        """Show the step Volura is on now, and paint it right away."""
        self.step_label.setText(text)
        if percent is not None:
            self.progress.setValue(percent)
        self.raise_()
        QApplication.processEvents()

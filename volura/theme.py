"""Light and dark theme: the choice, the Qt palette and the colors of Volura's own drawing.

Window > Preferences > Theme picks System (follows Windows or the desktop), Light or
Dark. Qt widgets follow the palette by themselves; whatever Volura draws itself (the 3D
view, the 2D worksheet, the panel headers, the switch, the icons) asks here with
:func:`color` for the color of the current theme.

In the dark theme the sheet on the worksheet stays light, like paper or material:
the layer colors of a laser cutter (pure red, blue, black) then stay easy to read.
The 3D view gets a muted gray sky on which both the black edges and preview lines
and the white faces have enough contrast.
"""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QGuiApplication, QPalette

from .i18n import tr

SYSTEM, LIGHT, DARK = "system", "light", "dark"
CHOICES = ((SYSTEM, tr("System")), (LIGHT, tr("Light")), (DARK, tr("Dark")))

COLORS = {
    # messages and text with a meaning
    "ok": ("#2e7d32", "#7ed389"),
    "error": ("#c62828", "#ff8f87"),
    "warn": ("#b94700", "#ffb45e"),
    "info": ("#1565c0", "#8cc4ff"),
    "invalid": ("#c0392b", "#ff6f61"),
    "title": ("#d23b3b", "#ff6b6b"),
    "muted": ("#555c69", "#a9b0bb"),
    # panel headers of the Default Tray
    "tray_bg": ("#ededed", "#30343a"),
    "tray_hover": ("#e2e2e2", "#3b4048"),
    "tray_line_top": ("#d8d8d8", "#41464e"),
    "tray_line_bottom": ("#c4c4c4", "#15171a"),
    "tray_text": ("#1f1f1f", "#f2f4f7"),
    # the 3D | 2D switch
    "switch_on": ("#2f6fd6", "#2563eb"),
    "switch_on_text": ("#ffffff", "#ffffff"),
    "switch_track": ("#e6eaf0", "#2b2f35"),
    "switch_hover": ("#dde2ea", "#353a42"),
    "switch_border": ("#b9c3d0", "#5b636e"),
    "switch_text": ("#4b5563", "#d5dae1"),
    # the 2D worksheet
    "sheet_empty": ("#ffffff", "#e9ebee"),        # without a sheet: the whole area is paper
    "sheet_desk": ("#e8ebef", "#25282c"),         # next to the sheet
    "sheet_plate": ("#ffffff", "#eceef1"),
    "sheet_shadow": ("#c4cad3", "#0e0f11"),
    "sheet_border": ("#4b5563", "#8b95a3"),
    "sheet_label": ("#555c69", "#c9ced6"),
    "sheet_margin": ("#8fa3bf", "#7f95b5"),
    "grid": ("#dfe3e8", "#d6dae0"),
    "grid_major": ("#b9c0c9", "#b0b7c1"),
    "desk_grid": ("#dfe3e8", "#2e3237"),
    "desk_grid_major": ("#b9c0c9", "#3c4148"),
    "origin": ("#2b2b2b", "#2b2b2b"),
    # the 3D view (RGB 0..1 is made from these hex values)
    "sky_top": ("#9eb8d1", "#434a54"),
    "sky_horizon": ("#f7f7f5", "#7a7d82"),
    "ground": ("#cccbc6", "#686864"),
    "view_text": ("#5a5a5a", "#e3e6ea"),
    "overhang": ("#e0491f", "#ff7043"),        # a face that needs support (View > Overhangs)
}

# Dark colors in the SVG icons and their light counterpart in the dark theme.
ICON_INK = {"#2b2b2b": "#e6e8eb", "#1f3b63": "#a9c8f2", "#6b7685": "#b3bdca",
            "#888": "#b5b5b5", "#8a8a8a": "#b5b5b5"}

_choice = SYSTEM
_base_style = None


def choice():
    return _choice


def is_dark(app=None) -> bool:
    """True when the current palette is dark (whatever the choice or the system is)."""
    app = app or QGuiApplication.instance()
    if app is None:
        return False
    return app.palette().color(QPalette.Window).lightness() < 128


def color(name, dark=None) -> str:
    light_value, dark_value = COLORS[name]
    if dark is None:
        dark = is_dark()
    return dark_value if dark else light_value


def qcolor(name, alpha=None, dark=None) -> QColor:
    c = QColor(color(name, dark))
    if alpha is not None:
        c.setAlpha(alpha)
    return c


def rgb(name, dark=None) -> tuple:
    """A theme color as (r, g, b) from 0 to 1, for OpenGL."""
    c = QColor(color(name, dark))
    return (c.redF(), c.greenF(), c.blueF())


def dark_palette() -> QPalette:
    """A dark palette with ample contrast, also for disabled text and hints."""
    p = QPalette()
    window, base, alt = QColor("#202326"), QColor("#16181b"), QColor("#24272b")
    text, bright = QColor("#eceff3"), QColor("#ffffff")
    button = QColor("#2d3136")
    disabled = QColor("#8a919b")
    highlight = QColor("#2563eb")
    for group in (QPalette.Active, QPalette.Inactive, QPalette.Disabled):
        p.setColor(group, QPalette.Window, window)
        p.setColor(group, QPalette.WindowText, text)
        p.setColor(group, QPalette.Base, base)
        p.setColor(group, QPalette.AlternateBase, alt)
        p.setColor(group, QPalette.ToolTipBase, QColor("#2d3136"))
        p.setColor(group, QPalette.ToolTipText, text)
        p.setColor(group, QPalette.PlaceholderText, QColor("#9aa1ab"))
        p.setColor(group, QPalette.Text, text)
        p.setColor(group, QPalette.Button, button)
        p.setColor(group, QPalette.ButtonText, text)
        p.setColor(group, QPalette.BrightText, bright)
        p.setColor(group, QPalette.Light, QColor("#3c4148"))
        p.setColor(group, QPalette.Midlight, QColor("#33373d"))
        p.setColor(group, QPalette.Mid, QColor("#4a5058"))
        p.setColor(group, QPalette.Dark, QColor("#121416"))
        p.setColor(group, QPalette.Shadow, QColor("#000000"))
        p.setColor(group, QPalette.Highlight, highlight)
        p.setColor(group, QPalette.HighlightedText, bright)
        p.setColor(group, QPalette.Link, QColor("#8cc4ff"))
        p.setColor(group, QPalette.LinkVisited, QColor("#c6a6ff"))
        p.setColor(group, QPalette.Accent, highlight)
    p.setColor(QPalette.Inactive, QPalette.Highlight, QColor("#2f5591"))
    for role in (QPalette.WindowText, QPalette.Text, QPalette.ButtonText):
        p.setColor(QPalette.Disabled, role, disabled)
    p.setColor(QPalette.Disabled, QPalette.Base, window)
    p.setColor(QPalette.Disabled, QPalette.Button, QColor("#26292d"))
    p.setColor(QPalette.Disabled, QPalette.Highlight, QColor("#4a5058"))
    return p


def wants_dark(value, app=None) -> bool:
    """Does this choice make it dark? With System, Windows or the desktop decides."""
    if value == DARK:
        return True
    if value == LIGHT:
        return False
    app = app or QGuiApplication.instance()
    if app is None:
        return False
    return app.styleHints().colorScheme() == Qt.ColorScheme.Dark


def apply(value=SYSTEM, app=None):
    """Set the theme: System, Light or Dark. Returns True when it is now dark.

    Dark uses the Fusion style everywhere with its own palette: the Windows styles have
    too little contrast in the dark, and Windows 10 has no dark style. Light uses the
    normal style of the system.
    """
    global _choice, _base_style
    from PySide6.QtWidgets import QApplication
    app = app or QApplication.instance()
    _choice = value if value in (SYSTEM, LIGHT, DARK) else SYSTEM
    if app is None:
        return False
    if _base_style is None:
        _base_style = app.style().name()
    hints = app.styleHints()
    if _choice == SYSTEM:
        hints.unsetColorScheme()
    else:
        hints.setColorScheme(Qt.ColorScheme.Dark if _choice == DARK else Qt.ColorScheme.Light)
    dark = wants_dark(_choice, app)
    if dark:
        if app.style().name().lower() != "fusion":
            app.setStyle("Fusion")
        app.setPalette(dark_palette())
    else:
        if app.style().name().lower() != _base_style.lower():
            app.setStyle(_base_style)
        app.setPalette(QPalette())          # back to the palette of the system
    return dark

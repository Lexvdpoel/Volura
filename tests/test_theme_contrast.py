"""The high contrast theme: black, white and yellow, also for what Volura draws itself.

Runs without a display (Qt offscreen platform), like tests/test_theme_spacemouse.py.

Run with:  python -m pytest tests/test_theme_contrast.py
"""
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest  # noqa: E402
from PySide6.QtGui import QColor, QPalette  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from volura import app as app_module, theme, tray  # noqa: E402
from volura.settings import qsettings  # noqa: E402

_app = QApplication.instance() or QApplication(sys.argv[:1])


def luminance(c):
    c = QColor(c)

    def lin(v):
        return v / 12.92 if v <= 0.03928 else ((v + 0.055) / 1.055) ** 2.4
    return 0.2126 * lin(c.redF()) + 0.7152 * lin(c.greenF()) + 0.0722 * lin(c.blueF())


def contrast(a, b):
    la, lb = sorted((luminance(a), luminance(b)), reverse=True)
    return (la + 0.05) / (lb + 0.05)


@pytest.fixture
def contrast_theme():
    theme.apply(theme.HIGH_CONTRAST)
    yield
    theme.apply(theme.SYSTEM)


@pytest.fixture
def win():
    from volura.app import MainWindow
    w = MainWindow()
    was = w.settings.theme
    yield w
    w.settings.theme = was
    w.settings.save()
    theme.apply(theme.SYSTEM)
    w.doc.dirty = False
    w.close()


# ------------------------------------------------------------------ the choice
def test_the_choice_is_offered_after_dark():
    assert [key for key, _ in theme.CHOICES] == ["system", "light", "dark", "contrast"]
    assert dict(theme.CHOICES)[theme.HIGH_CONTRAST] == "High contrast"


def test_it_is_a_dark_theme(contrast_theme):
    assert theme.choice() == theme.HIGH_CONTRAST
    assert theme.wants_dark(theme.HIGH_CONTRAST)
    assert theme.is_dark()


def test_the_own_colors_are_easy_to_see_on_black(contrast_theme):
    for name in ("ok", "error", "warn", "info", "muted"):
        assert contrast(theme.color(name), "#000000") >= 4.5, name
    assert contrast(theme.color("title"), "#000000") >= 3.0      # large, bold title
    assert contrast(theme.color("tray_text"), theme.color("tray_bg")) >= 7
    assert contrast(theme.color("switch_text"), theme.color("switch_track")) >= 4.5
    assert contrast(theme.color("sheet_label"), theme.color("sheet_desk")) >= 4.5
    assert theme.color("tray_bg") in tray.header_style()


def test_the_high_contrast_colors_win_from_the_dark_ones(contrast_theme):
    assert theme.color("error") == "#ff4d4d"
    assert theme.color("ok") == "#00ff66"
    assert theme.color("warn") == "#ffd400"
    assert theme.color("info") == "#66ccff"
    assert theme.color("sky_top") == "#000000"
    assert theme.color("sheet_desk") == "#000000"


def test_a_name_without_a_third_value_keeps_the_dark_one(contrast_theme):
    for name, values in theme.COLORS.items():
        if len(values) == 2:
            assert theme.color(name) == values[1], name
    assert theme.color("sheet_plate") == "#eceef1"       # the sheet stays light: paper


# ------------------------------------------------------------------ the palette
def test_the_palette_is_black_white_and_yellow(contrast_theme):
    p = _app.palette()
    for role, name in ((QPalette.Window, "#000000"), (QPalette.Base, "#000000"),
                       (QPalette.Button, "#000000"), (QPalette.ToolTipBase, "#000000"),
                       (QPalette.AlternateBase, "#1a1a1a"),
                       (QPalette.Text, "#ffffff"), (QPalette.WindowText, "#ffffff"),
                       (QPalette.ButtonText, "#ffffff"), (QPalette.ToolTipText, "#ffffff"),
                       (QPalette.Highlight, "#ffd400"), (QPalette.HighlightedText, "#000000"),
                       (QPalette.Link, "#66ccff")):
        assert p.color(role).name() == name, role
    assert p.color(QPalette.Disabled, QPalette.Text).name() == "#a0a0a0"
    assert p.color(QPalette.Disabled, QPalette.WindowText).name() == "#a0a0a0"


def test_the_window_follows_the_choice(win):
    win.settings.theme = theme.HIGH_CONTRAST
    assert win.apply_theme()
    assert _app.palette().color(QPalette.Window).name() == "#000000"
    win.settings.theme = theme.LIGHT
    assert not win.apply_theme()


def test_the_choice_round_trips_through_the_settings():
    q = qsettings()
    was = q.value("theme")
    s = app_module.Settings()
    s.theme = theme.HIGH_CONTRAST
    s.save()
    assert app_module.Settings().theme == "contrast"
    q.remove("theme")
    if was is not None:
        q.setValue("theme", was)
    q.sync()


# ------------------------------------------------------------------ the 3D view
def test_the_edges_of_the_3d_view_turn_white(contrast_theme):
    """White edges on the dark faces and sky; light and dark keep the near black they had."""
    from volura import viewport as vp
    assert vp.edge_color() == pytest.approx((1.0, 1.0, 1.0, 1.0))
    for value in (theme.LIGHT, theme.DARK):
        theme.apply(value)
        assert vp.edge_color() == pytest.approx((0.05, 0.05, 0.05, 1), abs=0.002), value


def test_the_preferences_apply_the_new_choice(win):
    from volura.app import PreferencesDialog
    dlg = PreferencesDialog(win, "Display")
    assert dlg.theme.findData("contrast") >= 0
    dlg.theme.setCurrentIndex(dlg.theme.findData("contrast"))
    dlg.accept()
    assert win.settings.theme == "contrast" and theme.is_dark()
    assert _app.palette().color(QPalette.Highlight).name() == "#ffd400"

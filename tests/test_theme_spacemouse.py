"""The dark theme (enough contrast, applied everywhere) and the 3Dconnexion 3D mouse."""
import os
import struct
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from PySide6.QtGui import QColor, QPalette  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from volura import icons, spacemouse, theme, tray  # noqa: E402

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
def dark():
    theme.apply(theme.DARK)
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


# ------------------------------------------------------------------ theme
def test_dark_palette_has_enough_contrast(dark):
    p = _app.palette()
    assert theme.is_dark()
    for bg, fg in ((QPalette.Window, QPalette.WindowText), (QPalette.Base, QPalette.Text),
                   (QPalette.Button, QPalette.ButtonText),
                   (QPalette.Highlight, QPalette.HighlightedText)):
        assert contrast(p.color(bg), p.color(fg)) >= 4.5, (bg, fg)
    assert contrast(p.color(QPalette.Disabled, QPalette.Text), p.color(QPalette.Base)) >= 4.5
    assert contrast(p.color(QPalette.PlaceholderText), p.color(QPalette.Base)) >= 4.5


@pytest.mark.parametrize("is_dark", (False, True))
def test_own_colors_readable_in_both_themes(is_dark):
    window = "#202326" if is_dark else "#f3f3f3"
    assert contrast(theme.color("title", is_dark), window) >= 3.0      # large, bold title
    for name in ("ok", "error", "warn", "info", "muted"):
        assert contrast(theme.color(name, is_dark), window) >= 4.5, name
    assert contrast(theme.color("tray_text", is_dark), theme.color("tray_bg", is_dark)) >= 7
    assert contrast(theme.color("switch_text", is_dark), theme.color("switch_track", is_dark)) >= 4.5
    assert contrast(theme.color("sheet_label", is_dark), theme.color("sheet_desk", is_dark)) >= 4.5
    # on the sheet the layer colors of a laser cutter stay readable
    for layer in ("#ff0000", "#0000ff", "#000000"):
        assert contrast(layer, theme.color("sheet_plate", is_dark)) >= 3.0, layer


def test_switch_between_light_and_dark():
    assert theme.apply(theme.DARK) and theme.is_dark()
    assert _app.style().name().lower() == "fusion"
    assert not theme.apply(theme.LIGHT) and not theme.is_dark()
    theme.apply(theme.SYSTEM)
    assert theme.choice() == theme.SYSTEM


def test_icons_light_in_the_dark(dark):
    ic = icons.icon("arc")
    img = ic.pixmap(24).toImage()
    inks = {img.pixelColor(x, y).name() for x in range(24) for y in range(24)
            if img.pixelColor(x, y).alpha() == 255}
    assert "#e6e8eb" in inks and "#2b2b2b" not in inks
    theme.apply(theme.LIGHT)
    img = ic.pixmap(24).toImage()
    inks = {img.pixelColor(x, y).name() for x in range(24) for y in range(24)
            if img.pixelColor(x, y).alpha() == 255}
    assert "#2b2b2b" in inks                      # the same icon, without new buttons
    assert icons.cursor("line").pixmap().toImage().pixelColor(4, 20).name() != "#e6e8eb"


def test_panel_headers_follow_the_theme(dark):
    assert theme.color("tray_bg") in tray.header_style()
    assert theme.color("tray_text") in tray.header_style()
    assert "#ededed" in tray.header_style(dark=False)


def test_workspace_dark_with_light_sheet(win):
    win.settings.theme = theme.DARK
    win.apply_theme()
    s = win.doc.sheet
    s.sheet_w, s.sheet_h = 100.0, 60.0
    win.set_mode("2d")
    v = win.sheet_view
    v.resize(600, 400)
    v.zoom_extents()
    img = v.grab().toImage()
    plate = v.to_screen((50.0, 30.0))
    assert QColor(img.pixelColor(int(plate.x()) + 3, int(plate.y()) + 3)).lightness() > 200
    assert QColor(img.pixelColor(3, 3)).lightness() < 70
    win.settings.theme = theme.LIGHT
    win.apply_theme()
    img = v.grab().toImage()
    assert QColor(img.pixelColor(3, 3)).lightness() > 200


def test_theme_in_the_preferences(win):
    from volura.app import PreferencesDialog
    dlg = PreferencesDialog(win, "Display")
    assert [dlg.theme.itemData(i) for i in range(dlg.theme.count())] == \
        ["system", "light", "dark", "contrast"]
    dlg.theme.setCurrentIndex(dlg.theme.findData("dark"))
    dlg.accept()
    assert win.settings.theme == "dark" and theme.is_dark()
    assert type(win.settings)().theme == "dark"
    dlg = PreferencesDialog(win, "Display")
    dlg.theme.setCurrentIndex(dlg.theme.findData("system"))
    dlg.accept()
    assert not theme.is_dark()


# ------------------------------------------------------------------ 3D mouse
def report(rid, *values):
    return bytes([rid]) + struct.pack(f"<{len(values)}h", *values)


def test_old_device_sends_translation_and_rotation_separately():
    st = spacemouse.HidState()
    assert st.feed(report(1, 10, 20, 30)) == ("motion", (10, -30, 20, 0, 0, 0))
    assert st.feed(report(2, 1, 2, 3)) == ("motion", (10, -30, 20, 1, -3, 2))


def test_new_device_sends_everything_in_one_report():
    st = spacemouse.HidState()
    assert st.feed(report(1, 10, 20, 30, 1, 2, 3)) == ("motion", (10, -30, 20, 1, -3, 2))


def test_buttons_only_on_press():
    st = spacemouse.HidState()
    assert st.feed(bytes([3, 0b10, 0])) == ("buttons", 0b10)
    assert st.feed(bytes([3, 0b11, 0])) == ("buttons", 0b01)
    assert st.feed(bytes([3, 0, 0])) == ("buttons", 0)
    assert spacemouse.bits(0b101) == [0, 2]


def test_3dxware_events():
    buf = bytearray(spacemouse._EVENT_SIZE)
    struct.pack_into("<i", buf, 0, 2)
    struct.pack_into("<6i", buf, 20, 350, -175, 0, 0, 35, 0)
    struct.pack_into("<i", buf, 44, 16)
    assert spacemouse.parse_si_event(bytes(buf)) == ("motion", (350, -175, 0, 0, 35, 0), 16)
    buf = bytearray(64)
    struct.pack_into("<iI", buf, 0, 12, 2)
    assert spacemouse.parse_si_event(bytes(buf)) == ("press", 2)
    assert spacemouse.V3DK_COMMANDS[2] == "fit"
    assert spacemouse.parse_si_event(bytes(64)) is None


def test_spacenavd_event():
    pkt = struct.pack("=8i", 0, 1, 2, 3, 4, 5, 6, 16)
    assert spacemouse.parse_spnav(pkt) == ("motion", (1, 2, -3, 4, 5, -6), 16)
    assert spacemouse.parse_spnav(struct.pack("=8i", 1, 1, 0, 0, 0, 0, 0, 0)) == ("press", 1)


def test_dead_zone_and_scale():
    m = spacemouse.normalize((350, 3, 0, 0, -700, 0), 0.02)
    assert m.tx == pytest.approx(1.0) and m.ty == 0.0 and m.ry == pytest.approx(-2.0)
    assert m.dt == pytest.approx(0.02)
    assert spacemouse.normalize((1, 2, 3, 4, 5, 6)).is_zero()


def test_signals_of_the_3d_mouse():
    sm = spacemouse.SpaceMouse()
    got, cmds = [], []
    sm.motion.connect(got.append)
    sm.command.connect(cmds.append)
    st = spacemouse.HidState()
    sm.handle_hid(st, report(1, 350, 0, 0, 0, 0, 0))
    sm.handle_hid(st, bytes([3, 0b10]))
    sm.handle_hid(st, report(1, 0, 0, 0, 0, 0, 0))      # at rest: no motion
    assert len(got) == 1 and got[0].tx == pytest.approx(1.0)
    assert cmds == ["fit"]


def test_3d_view_follows_the_cap(win):
    v = win.view
    cam = v.cam
    _, r, u = cam.basis()
    target, dist, yaw, pitch = cam.target.copy(), cam.distance, cam.yaw, cam.pitch
    v.spacemouse_move(spacemouse.Motion(tx=1.0, dt=0.1))
    assert np.dot(cam.target - target, r) < 0          # model to the right = camera to the left
    v.spacemouse_move(spacemouse.Motion(tz=1.0, dt=0.1))
    assert cam.distance < dist                          # pulling towards you zooms in
    v.spacemouse_move(spacemouse.Motion(ry=1.0, rx=1.0, dt=0.1))
    assert cam.yaw < yaw and cam.pitch > pitch


def test_workspace_pans_and_zooms_with_the_cap(win):
    win.set_mode("2d")
    v = win.sheet_view
    x, scale = v.origin.x(), v.scale
    v.spacemouse_move(spacemouse.Motion(tx=1.0, dt=0.1))
    assert v.origin.x() > x
    v.spacemouse_move(spacemouse.Motion(tz=1.0, dt=0.1))
    assert v.scale > scale


def test_button_of_the_3d_mouse(win):
    win.on_spacemouse_command("top")
    assert (win.view.cam.yaw, win.view.cam.pitch) == (-90.0, 89.99)
    assert win.start_spacemouse() is None              # never in a test
    assert win.spacemouse_status() in ("no 3D mouse found", "off")

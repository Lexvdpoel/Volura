"""Label plates: a printable name tag or drawer label with text on it.

`label_plate.label_plate` gives the label as one closed volume that stands on `z = 0` and is
centred on the origin: the text in letters of `height` mm cap height, on a rounded plate that is
the text `margin` mm wider all round and `thickness` mm thick. "raised" stands the letters
`relief` mm on top of the plate, "engraved" cuts them `relief` mm deep in the top face, and a
`hole` drills through the plate `margin` left of the text for a hook or a screw, making the plate
longer for it. A `\n` in the text makes the next line, centred and 1.4 cap heights lower.

These tests weigh a label against what it promises: a raised one is thickness + relief high, an
engraved one keeps the thickness of the plate and holds less plastic than the bare plate, the
plate grows with the text and with the hole, and the input that cannot be printed is refused. The
place in the interface is checked too (Print Tools right after the standoff, one group in one undo
step, and the dialog that remembers its values).

Run with:  python -m pytest tests/test_label_plate.py
"""
import os
import sys
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import manifold3d as mf  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from PySide6.QtCore import QCoreApplication  # noqa: E402
from PySide6.QtWidgets import QApplication, QDialog, QPlainTextEdit  # noqa: E402

import volura.app as app_module  # noqa: E402
from volura import enclosure, fonts, label_plate  # noqa: E402

_app = QApplication.instance() or QApplication(sys.argv[:1])

TEXT = "Volura"
H, T, R, M, RAD = 8.0, 2.0, 0.8, 3.0, 2.0
OPTIONS = dict(text="Tools", height=H, thickness=T, relief=R, mode="raised", margin=M,
               radius=RAD, hole=0.0)
#: the same label with the word the geometry tests measure
BASE = dict(OPTIONS, text=TEXT)


def need_fonts():
    if not fonts.has_fonts():
        pytest.skip("this system has no fonts, so there are no letters to test")


def clean(text):
    return text.replace("&", "").replace("…", "").replace("...", "").strip()


def pump_until(predicate, timeout_ms=15000):
    """Spin the event loop until `predicate()` holds or the time is up; returns its value."""
    deadline = time.monotonic() + timeout_ms / 1000.0
    while not predicate() and time.monotonic() < deadline:
        _app.processEvents()
        time.sleep(0.005)
    return predicate()


def bbox(m):
    """The bounding box of a manifold as (low, high)."""
    b = np.asarray(m.bounding_box(), dtype=float)
    return b[:3], b[3:]


def plate(**kw):
    return label_plate.label_plate(**dict(BASE, **kw))


def bare_plate(m):
    """The plate of `m` without any letters: the same rounded rectangle, `T` mm thick."""
    lo, hi = bbox(m)
    return mf.Manifold.extrude(enclosure.rounded_rect(hi[0] - lo[0], hi[1] - lo[1], RAD), T)


def ink(text=TEXT):
    """(width, depth) of the letters of `text` as the font draws them."""
    pts = np.concatenate(fonts.text_loops(text, None, H))
    return (float(pts[:, 0].max() - pts[:, 0].min()),
            float(pts[:, 1].max() - pts[:, 1].min()))


@pytest.fixture()
def win(monkeypatch, tmp_path):
    monkeypatch.setattr(app_module.errors, "log_dir", lambda: str(tmp_path))
    from volura.app import MainWindow
    w = MainWindow()
    w.doc.dirty = False
    QCoreApplication.processEvents()            # handle the deferred initial view now
    w.resize(900, 600)
    yield w
    w.doc.dirty = False
    w.close()


# ------------------------------------------------------------------ the label

@pytest.mark.parametrize("mode", label_plate.LABEL_MODES)
def test_a_label_is_one_valid_volume_that_stands_on_the_bed(mode):
    need_fonts()
    m = plate(mode=mode)
    assert m.status() == mf.Error.NoError and not m.is_empty()
    assert len(m.decompose()) == 1, "the letters are not fused to the plate"
    lo, hi = bbox(m)
    assert lo[2] == pytest.approx(0.0), "the label does not stand on the bed"
    assert 0.5 * (lo[0] + hi[0]) == pytest.approx(0.0), "the text is not centred"
    assert 0.5 * (lo[1] + hi[1]) == pytest.approx(0.0), "the text is not centred"
    assert m.genus() == 0


def test_a_raised_label_is_the_plate_with_the_letters_on_top():
    need_fonts()
    m = plate(mode="raised")
    lo, hi = bbox(m)
    assert hi[2] == pytest.approx(T + R), "the letters do not stand on the plate"
    plain = bare_plate(m)
    assert (m - plain).volume() == pytest.approx(m.volume() - plain.volume(), rel=1e-6)
    assert (plain - m).volume() == pytest.approx(0.0, abs=1e-6), "a raised label cuts away"


def test_an_engraved_label_keeps_the_thickness_of_the_plate():
    need_fonts()
    m = plate(mode="engraved")
    lo, hi = bbox(m)
    assert hi[2] == pytest.approx(T), "an engraving does not cut into the plate"
    plain = bare_plate(m)
    assert m.volume() < plain.volume(), "the engraving cut nothing away"
    assert (m - plain).volume() == pytest.approx(0.0, abs=1e-6), "an engraving stands up"


def test_the_plate_is_the_text_with_margin_on_every_side():
    need_fonts()
    wide, deep = ink()
    lo, hi = bbox(plate())
    assert hi[0] - lo[0] == pytest.approx(wide + 2.0 * M)
    assert hi[1] - lo[1] == pytest.approx(deep + 2.0 * M)
    assert hi[0] == pytest.approx(0.5 * wide + M)


def test_a_longer_text_makes_a_longer_plate():
    need_fonts()
    short, long_ = plate(text="Tool"), plate(text="Drawer tools")
    assert bbox(long_)[1][0] - bbox(long_)[0][0] > bbox(short)[1][0] - bbox(short)[0][0]
    assert bbox(long_)[1][1] - bbox(long_)[0][1] == pytest.approx(
        bbox(short)[1][1] - bbox(short)[0][1]), "the text became a different height"


def test_two_lines_stand_above_each_other():
    need_fonts()
    one, two = plate(text="Tools"), plate(text="Tools\nDrawers")
    assert bbox(two)[1][1] - bbox(two)[0][1] == pytest.approx(
        bbox(one)[1][1] - bbox(one)[0][1] + label_plate.LINE_SPACING * H)
    lo, hi = bbox(two)
    assert 0.5 * (lo[0] + hi[0]) == pytest.approx(0.0), "the lines are not centred"


def test_a_hole_makes_a_longer_plate_and_a_handle():
    need_fonts()
    d = 4.0
    plain, hung = plate(), plate(hole=d)
    plo, phi = bbox(plain)
    lo, hi = bbox(hung)
    assert hung.genus() == plain.genus() + 1, "the screw cannot go through"
    assert hi[0] - lo[0] == pytest.approx(phi[0] - plo[0] + d + M)
    assert hi[0] == pytest.approx(phi[0]), "the text moved for the hole"
    assert lo[0] == pytest.approx(plo[0] - d - M)
    assert hi[2] == pytest.approx(phi[2]), "the hole pushed the plate up"
    # the hole stands margin left of the text, in the middle of the plate depth
    x = -0.5 * (phi[0] - plo[0] - 2.0 * M) - M - 0.5 * d
    probe = mf.Manifold.cylinder(T + 2.0, 0.5 * d - 0.05, 0.5 * d - 0.05,
                                 label_plate.SEGMENTS).translate((x, 0.0, -1.0))
    assert (hung ^ probe).volume() == pytest.approx(0.0, abs=1e-6), "the hole is elsewhere"


@pytest.mark.parametrize("kw, wrong", (
    ({"text": ""}, "needs text"), ({"text": "   "}, "needs text"),
    ({"text": None}, "needs text"), ({"text": "a\nb\nc\nd\ne"}, "at most"),
    ({"text": "x" * 41}, "at most"),
    ({"height": 2.0}, "text height"), ({"height": 60.0}, "text height"),
    ({"thickness": 0.5}, "thickness"), ({"thickness": 12.0}, "thickness"),
    ({"relief": 0.1}, "relief"), ({"relief": 6.0}, "relief"),
    ({"mode": "debossed"}, "style"), ({"mode": None}, "style"),
    ({"margin": -1.0}, "margin"), ({"margin": 25.0}, "margin"),
    ({"radius": -0.5}, "corner radius"),
    ({"hole": -1.0}, "hole"), ({"hole": 20.0}, "hole"),
    ({"mode": "engraved", "relief": 1.8}, "engraving must leave")))
def test_impossible_labels_raise(kw, wrong):
    with pytest.raises(ValueError, match=wrong):
        plate(**kw)


def test_a_font_that_is_not_there_is_said_plainly(monkeypatch):
    monkeypatch.setattr(fonts, "default_family", lambda *a, **kw: "")
    with pytest.raises(ValueError, match="No font found"):
        plate()


def test_a_font_that_draws_nothing_is_said_plainly(monkeypatch):
    monkeypatch.setattr(fonts, "default_family", lambda *a, **kw: "Nowhere Sans")
    monkeypatch.setattr(fonts, "text_loops", lambda *a, **kw: [])
    with pytest.raises(ValueError, match="No font found"):
        plate()


# ------------------------------------------------------------------ in the window

def test_the_label_is_one_group_that_one_undo_takes_away(win):
    need_fonts()
    steps = len(win.doc.undo_stack)
    win.make_label_plate_model(OPTIONS)
    assert pump_until(lambda: len(win.doc.groups) == 1), "no label plate in the model"
    assert len(win.doc.undo_stack) == steps + 1, "the label is not one undo step"
    g = win.doc.groups[0]
    assert g.name == "Tools", "the group is not named after the first line"
    m = g.ents.to_manifold()
    assert m.status() == mf.Error.NoError
    assert m.volume() == pytest.approx(plate(**OPTIONS).volume(), rel=1e-6)
    assert [g.id for g in win.doc.selected_groups()] == [g.id for g in win.doc.groups]
    assert pump_until(lambda: win.hint.text() == "Label plate added."), win.hint.text()
    win.doc.undo()
    assert win.doc.groups == []


def test_the_label_stands_on_the_bed(win):
    need_fonts()
    win.make_label_plate_model(OPTIONS)
    assert pump_until(lambda: len(win.doc.groups) == 1), "no label plate in the model"
    assert pump_until(lambda: win.doc.groups[0].bbox() is not None)
    lo, hi = np.asarray(win.doc.groups[0].bbox(), dtype=float)
    assert lo[2] == pytest.approx(0.0), "the label does not stand on the bed"
    assert hi[2] == pytest.approx(T + R)
    bx, by, _bz = win.settings.bed_size
    assert 0.5 * (lo[0] + hi[0]) == pytest.approx(0.5 * bx), "the label is not on the bed"
    assert 0.5 * (lo[1] + hi[1]) == pytest.approx(0.5 * by)
    win.make_label_plate_model(dict(OPTIONS, text="Drawer"))
    assert pump_until(lambda: len(win.doc.groups) == 2), "the second label is missing"
    first, second = (np.asarray(g.bbox(), dtype=float) for g in win.doc.groups)
    assert second[0][0] - first[1][0] == pytest.approx(10.0), "the labels are not 10 mm apart"


def test_a_label_that_cannot_be_built_is_a_clear_text(win):
    win.make_label_plate_model(dict(OPTIONS, text=" "))
    assert pump_until(lambda: "Label plate failed" in win.hint.text()), win.hint.text()
    assert win.doc.groups == []


def test_the_label_plate_follows_the_standoff_in_print_tools(win):
    labels = [clean(a.text()) for a in win.print_tools_menu.actions() if not a.isSeparator()]
    assert labels.index("Label Plate") == labels.index("Standoff") + 1


def test_the_dialog_starts_at_the_last_values_and_remembers_the_new_ones(win, monkeypatch):
    need_fonts()
    seen = {}

    def fill(dlg):
        seen["title"] = dlg.windowTitle()
        seen["text"] = dlg.text.toPlainText()
        seen["start"] = [round(dlg.height.value(), 2), round(dlg.thickness.value(), 2),
                         round(dlg.relief.value(), 2), round(dlg.margin.value(), 2),
                         round(dlg.radius.value(), 2), round(dlg.hole.value(), 2)]
        seen["mode"] = dlg.mode.currentData()
        seen["edit"] = isinstance(dlg.findChild(QPlainTextEdit), QPlainTextEdit)
        dlg.text.setPlainText("Drawer A\nTools")
        dlg.height.setValue(12.0)
        dlg.thickness.setValue(3.0)
        dlg.relief.setValue(1.0)
        dlg.margin.setValue(4.0)
        dlg.radius.setValue(1.0)
        dlg.hole.setValue(4.0)
        dlg.mode.setCurrentIndex(dlg.mode.findData("engraved"))
        return QDialog.Accepted

    monkeypatch.setattr(app_module.QDialog, "exec", fill)
    win.settings.label_text, win.settings.label_height = "Old", 6.0
    win.settings.label_thickness, win.settings.label_relief = 1.5, 0.6
    win.settings.label_mode, win.settings.label_margin = "engraved", 2.0
    win.settings.label_radius, win.settings.label_hole = 3.0, 0.0
    win.label_plate_dialog()
    assert seen["title"] == "Label Plate"
    assert seen["text"] == "Old"
    assert seen["start"] == [6.0, 1.5, 0.6, 2.0, 3.0, 0.0]
    assert seen["mode"] == "engraved"
    assert seen["edit"], "the text is not written in a multi-line box"
    assert pump_until(lambda: len(win.doc.groups) == 1), "no label plate in the model"
    assert win.doc.groups[0].name == "Drawer A", "the group is not named after the first line"
    lo, hi = np.asarray(win.doc.groups[0].bbox(), dtype=float)
    assert hi[2] == pytest.approx(3.0), "the style is not as asked"
    assert hi[0] - lo[0] > hi[1] - lo[1], "the hole did not make the plate longer"
    assert win.settings.label_text == "Drawer A\nTools"
    assert win.settings.label_height == pytest.approx(12.0)
    s = app_module.Settings()
    assert (s.label_text, s.label_height, s.label_thickness, s.label_relief, s.label_mode,
            s.label_margin, s.label_radius, s.label_hole) == (
        "Drawer A\nTools", pytest.approx(12.0), pytest.approx(3.0), pytest.approx(1.0),
        "engraved", pytest.approx(4.0), pytest.approx(1.0), pytest.approx(4.0))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))

"""3D text: a part that is made of letters.

`text3d.text3d` gives the text as one closed volume that stands on `z = 0` and is centred on the
origin in plan: the letters of the text in `height` mm cap height, extruded `depth` mm, with the
holes of o, a and 8 in them. A `\n` makes the next line, `line_spacing` cap heights lower, and
`align` says whether those lines stand at the same left edge, centred, or at the same right edge.
The height and the depth are clamped to what still prints as text.

These tests weigh a text against what it promises: it stands on the bed, its depth is its height
in z, an "o" is a ring, two lines are taller than one and the lines are lined up as asked. The
place in the interface is checked too (Draw right after the Parts submenu, one group in one undo
step, and the command that is only there in 3D).

Run with:  python -m pytest tests/test_text3d.py
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
from volura import fonts, text3d  # noqa: E402

_app = QApplication.instance() or QApplication(sys.argv[:1])

TEXT = "Volura"
H, D = 10.0, 3.0
OPTIONS = dict(text="Tools", family="", height=H, depth=D, bold=False, align="left")
#: the same text with the word the geometry tests measure
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


def part(**kw):
    return text3d.text3d(**dict(BASE, **kw))


def ink_of_the_lower_line(m):
    """The (low, high) x of the letters below the middle of the text: the second line."""
    lo, hi = bbox(m)
    box = mf.Manifold.cube([100.0, 0.5 * (hi[1] - lo[1]) + 1.0, 100.0], True) \
        .translate((0.0, lo[1] - 0.5, 0.0))
    cut = mf.Manifold.batch_boolean([m, box], mf.OpType.Intersect)
    assert not cut.is_empty(), "there is no second line to look at"
    return bbox(cut)[0][0], bbox(cut)[1][0]


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


# ------------------------------------------------------------------ the text

def test_a_text_is_one_valid_volume_that_stands_on_the_bed():
    need_fonts()
    m = part()
    assert m.status() == mf.Error.NoError and not m.is_empty()
    assert m.volume() > 0.0
    lo, hi = bbox(m)
    assert lo[2] == pytest.approx(0.0), "the text does not stand on the bed"
    assert 0.5 * (lo[0] + hi[0]) == pytest.approx(0.0), "the text is not centred"
    assert 0.5 * (lo[1] + hi[1]) == pytest.approx(0.0), "the text is not centred"


def test_the_depth_is_the_height_of_the_text():
    need_fonts()
    for d in (0.4, 3.0, 12.0):
        lo, hi = bbox(part(depth=d))
        assert lo[2] == pytest.approx(0.0)
        assert hi[2] - lo[2] == pytest.approx(d)


def test_the_hole_of_an_o_stays_a_hole():
    need_fonts()
    assert part(text="o").genus() == 1, "an o is not a ring"


def test_the_cap_height_is_the_height_of_a_capital():
    need_fonts()
    lo, hi = bbox(part(text="H"))
    assert hi[1] - lo[1] == pytest.approx(H, rel=2e-2)


def test_two_lines_are_taller_than_one():
    need_fonts()
    one, two = part(text="Tools"), part(text="Tools\nDrawers")
    olo, ohi = bbox(one)
    lo, hi = bbox(two)
    assert hi[1] - lo[1] > ohi[1] - olo[1], "the second line did not make the text taller"
    assert hi[1] - lo[1] == pytest.approx(ohi[1] - olo[1] + text3d.LINE_SPACING * H,
                                          rel=5e-2)
    assert 0.5 * (lo[0] + hi[0]) == pytest.approx(0.0), "the lines are not centred"


@pytest.mark.parametrize("align, edge", (("left", "low"), ("centre", "mid"), ("right", "high")))
def test_the_lines_stand_as_asked(align, edge):
    need_fonts()
    m = part(text="AB\nC", align=align)
    lo, hi = bbox(m)
    low, high = ink_of_the_lower_line(m)
    if edge == "low":
        assert low == pytest.approx(lo[0], abs=0.05), "the lines do not start at one left edge"
    elif edge == "high":
        assert high == pytest.approx(hi[0], abs=0.05), "the lines do not end at one right edge"
    else:
        assert low + high == pytest.approx(lo[0] + hi[0], abs=0.1), "the line is not centred"


def test_a_height_and_a_depth_outside_the_range_are_clamped():
    need_fonts()
    # compared with the text at the limit itself: how a font renders a tiny capital differs
    # per platform, the clamping does not
    for asked, limit in ((1.0, text3d.HEIGHT_MIN), (500.0, text3d.HEIGHT_MAX)):
        got, want = bbox(part(height=asked)), bbox(part(height=limit))
        assert got[1] - got[0] == pytest.approx(want[1] - want[0])
    assert bbox(part(depth=0.1))[1][2] == pytest.approx(text3d.DEPTH_MIN)
    assert bbox(part(depth=500.0))[1][2] == pytest.approx(text3d.DEPTH_MAX)


@pytest.mark.parametrize("kw, wrong", (
    ({"text": ""}, "needs text"), ({"text": "   "}, "needs text"),
    ({"text": None}, "needs text"), ({"align": "middle"}, "alignment")))
def test_impossible_texts_raise(kw, wrong):
    with pytest.raises(ValueError, match=wrong):
        part(**kw)


def test_a_font_that_is_not_there_is_said_plainly(monkeypatch):
    monkeypatch.setattr(fonts, "default_family", lambda *a, **kw: "")
    with pytest.raises(ValueError, match="No font found"):
        part()


def test_a_font_that_draws_nothing_is_said_plainly(monkeypatch):
    monkeypatch.setattr(fonts, "default_family", lambda *a, **kw: "Nowhere Sans")
    monkeypatch.setattr(fonts, "text_loops", lambda *a, **kw: [])
    with pytest.raises(ValueError, match="No font found"):
        part()


# ------------------------------------------------------------------ in the window

def test_the_text_is_one_group_that_one_undo_takes_away(win):
    need_fonts()
    steps = len(win.doc.undo_stack)
    win.make_text3d_model(OPTIONS)
    assert pump_until(lambda: len(win.doc.groups) == 1), "no 3D text in the model"
    assert len(win.doc.undo_stack) == steps + 1, "the text is not one undo step"
    g = win.doc.groups[0]
    assert g.name == "Tools", "the group is not named after the first line"
    m = g.ents.to_manifold()
    assert m.status() == mf.Error.NoError
    assert m.volume() == pytest.approx(text3d.text3d(**OPTIONS).volume(), rel=1e-6)
    assert [g.id for g in win.doc.selected_groups()] == [g.id for g in win.doc.groups]
    assert pump_until(lambda: win.hint.text() == "3D text added."), win.hint.text()
    win.doc.undo()
    assert win.doc.groups == []


def test_the_text_stands_on_the_bed(win):
    need_fonts()
    win.make_text3d_model(OPTIONS)
    assert pump_until(lambda: len(win.doc.groups) == 1), "no 3D text in the model"
    assert pump_until(lambda: win.doc.groups[0].bbox() is not None)
    lo, hi = np.asarray(win.doc.groups[0].bbox(), dtype=float)
    assert lo[2] == pytest.approx(0.0), "the text does not stand on the bed"
    assert hi[2] - lo[2] == pytest.approx(D)
    bx, by, _bz = win.settings.bed_size
    assert 0.5 * (lo[0] + hi[0]) == pytest.approx(0.5 * bx), "the text is not on the bed"
    assert 0.5 * (lo[1] + hi[1]) == pytest.approx(0.5 * by)
    win.make_text3d_model(dict(OPTIONS, text="Drawer"))
    assert pump_until(lambda: len(win.doc.groups) == 2), "the second text is missing"
    first, second = (np.asarray(g.bbox(), dtype=float) for g in win.doc.groups)
    assert second[0][0] - first[1][0] == pytest.approx(10.0), "the texts are not 10 mm apart"


def test_a_text_that_cannot_be_built_is_a_clear_text(win):
    win.make_text3d_model(dict(OPTIONS, text=" "))
    assert pump_until(lambda: "3D text failed" in win.hint.text()), win.hint.text()
    assert win.doc.groups == []


def test_the_dialog_starts_at_the_last_values_and_remembers_the_new_ones(win, monkeypatch):
    need_fonts()
    seen = {}

    def fill(dlg):
        seen["title"] = dlg.windowTitle()
        seen["text"] = dlg.text.toPlainText()
        seen["start"] = [round(dlg.height.value(), 2), round(dlg.depth.value(), 2),
                         dlg.bold.isChecked()]
        seen["align"] = dlg.align.currentData()
        seen["edit"] = isinstance(dlg.findChild(QPlainTextEdit), QPlainTextEdit)
        dlg.text.setPlainText("Drawer A\nTools")
        dlg.height.setValue(24.0)
        dlg.depth.setValue(6.0)
        dlg.bold.setChecked(True)
        dlg.align.setCurrentIndex(dlg.align.findData("right"))
        return QDialog.Accepted

    monkeypatch.setattr(app_module.QDialog, "exec", fill)
    win.settings.text3d_text, win.settings.text3d_height = "Old", 6.0
    win.settings.text3d_depth, win.settings.text3d_bold = 1.5, False
    win.settings.text3d_align = "centre"
    win.text3d_dialog()
    assert seen["title"] == "3D Text"
    assert seen["text"] == "Old"
    assert seen["start"] == [6.0, 1.5, False]
    assert seen["align"] == "centre"
    assert seen["edit"], "the text is not written in a multi-line box"
    assert pump_until(lambda: len(win.doc.groups) == 1), "no 3D text in the model"
    assert win.doc.groups[0].name == "Drawer A", "the group is not named after the first line"
    lo, hi = np.asarray(win.doc.groups[0].bbox(), dtype=float)
    assert hi[2] - lo[2] == pytest.approx(6.0), "the depth is not as asked"
    assert win.settings.text3d_text == "Drawer A\nTools"
    assert win.settings.text3d_depth == pytest.approx(6.0)
    s = app_module.Settings()
    assert (s.text3d_text, s.text3d_height, s.text3d_depth, s.text3d_bold, s.text3d_align) == (
        "Drawer A\nTools", pytest.approx(24.0), pytest.approx(6.0), True, "right")


# ------------------------------------------------------------------ in the menus

def test_the_command_is_in_the_draw_menu_after_the_parts(win):
    labels = [clean(a.text()) for a in win.flat_actions(win.menuBar())]
    assert labels.index("3D Text") == labels.index("Pegboard Hook") + 1


def test_the_command_is_only_there_in_3d(win):
    draw = next(m for m in win.menuBar().actions()
                if m.menu() is not None and clean(m.text()) == "Draw")
    assert win.a_text3d in win.flat_actions(draw.menu())
    win.set_mode("3d")
    assert win.a_text3d.isVisible(), "the model has no 3D text"
    win.set_mode("2d")
    assert not win.a_text3d.isVisible(), "the worksheet does not draw 3D text"
    win.set_mode("3d")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))

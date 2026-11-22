"""3D text: a part that is made of letters.

`text3d.text3d` gives the text as one closed volume that stands on `z = 0` and is centred on the
origin in plan: the letters of the text in `height` mm cap height, extruded `depth` mm, with the
holes of o, a and 8 in them. A `\n` makes the next line, `line_spacing` cap heights lower, and
`align` says whether those lines stand at the same left edge, centred, or at the same right edge.
The height and the depth are clamped to what still prints as text.

With `path="arc"` the letters follow a circle instead of a line: `radius` is the circle they stand
on, every letter is turned to the tangent of that circle in the middle of the room it takes, and
the room is walked along the circle, so the letters are as far apart on the arc as on a line. The
centre of the circle is then the origin, the text is centred on the top (with `inside`: on the
bottom), and one line is the most an arc takes.

These tests weigh a text against what it promises: it stands on the bed, its depth is its height
in z, an "o" is a ring, two lines are taller than one and the lines are lined up as asked. The arc
is weighed too: where every letter stands, which way it reads, that the holes survive the placing,
and which radius a text needs. The place in the interface is checked as well (Draw right after the
Parts submenu, one group in one undo step, and the command that is only there in 3D).

Run with:  python -m pytest tests/test_text3d.py
"""
import math
import os
import sys
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import manifold3d as mf  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from PySide6.QtCore import QCoreApplication  # noqa: E402
from PySide6.QtWidgets import (QApplication, QDialog, QDialogButtonBox,  # noqa: E402
                               QPlainTextEdit)

import volura.app as app_module  # noqa: E402
from volura import fonts, text3d, theme  # noqa: E402
from volura.settings import qsettings  # noqa: E402

_app = QApplication.instance() or QApplication(sys.argv[:1])

TEXT = "Volura"
H, D = 10.0, 3.0
OPTIONS = dict(text="Tools", family="", height=H, depth=D, bold=False, align="left")
#: the same text with the word the geometry tests measure
BASE = dict(OPTIONS, text=TEXT)
#: the circle the letters of an arc text stand on
R = 60.0
#: the same text on an arc, outside and inside: what the window is asked to build
ARCOPTS = dict(OPTIONS, path="arc", radius=R)
ARCIN = dict(ARCOPTS, inside=True)


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


def arc(**kw):
    """The same text on an arc of `R` millimetres, as the geometry tests measure it."""
    return text3d.text3d(**dict(dict(BASE, path="arc", radius=R), **kw))


def radii(m):
    """The distance of every point of `m` to the origin: how far the letters stand from the middle
    of the arc, whose centre is the origin."""
    v = np.asarray(m.to_mesh().vert_properties, dtype=float)
    return np.hypot(v[:, 0], v[:, 1])


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
    s = w.settings                              # the values the tests below start from
    s.text3d_text, s.text3d_family = TEXT, ""
    s.text3d_height, s.text3d_depth = H, D
    s.text3d_bold, s.text3d_align = False, "left"
    s.text3d_path, s.text3d_radius, s.text3d_inside = "straight", R, False
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


# ------------------------------------------------------------------ on an arc

@pytest.mark.parametrize("kw", ({}, {"path": "straight"},
                                {"path": "straight", "radius": 12.0, "inside": True},
                                {"radius": 5.0}, {"inside": True}))
def test_the_line_is_still_the_text_of_before(kw):
    """Straight is the default, and the values an arc needs are ignored on a line: the same volume
    and the same box as before arcs existed."""
    need_fonts()
    before, now = part(), part(**kw)
    assert now.volume() == pytest.approx(before.volume(), rel=1e-9)
    assert bbox(now)[0] == pytest.approx(bbox(before)[0], abs=1e-9)
    assert bbox(now)[1] == pytest.approx(bbox(before)[1], abs=1e-9)


@pytest.mark.parametrize("inside", (False, True))
def test_an_arc_text_is_one_valid_volume_of_the_same_letters(inside):
    need_fonts()
    m = arc(inside=inside)
    assert m.status() == mf.Error.NoError and not m.is_empty()
    assert m.volume() > 0.0
    lo, hi = bbox(m)
    assert lo[2] == pytest.approx(0.0), "the arc text does not stand on the bed"
    assert hi[2] - lo[2] == pytest.approx(D)
    assert m.volume() == pytest.approx(part().volume(), rel=1e-3), "the arc made other letters"
    # the foot of a letter rests on the tangent in the middle of its room, so it is a little
    # outside the circle, and a round letter overshoots its baseline a little past it; on the
    # inside the letters hang inward, so the circle is their outer edge there
    low, high = (R, R + H) if not inside else (R - H, R)
    feet, tops = radii(m).min(), radii(m).max()
    assert float(tops) == pytest.approx(high, abs=0.5), "the letters do not stand a cap high"
    assert float(feet) == pytest.approx(low, abs=0.5), "the baseline is not on the circle"


@pytest.mark.parametrize("inside", (False, True))
def test_the_middle_of_every_letter_stands_half_a_cap_height_from_the_arc_centre(inside):
    need_fonts()
    places = text3d.arc_places(TEXT, height=H, radius=R, inside=inside)
    assert len(places) == len(TEXT), "every character should have a place on the arc"
    want = R + 0.5 * H if not inside else R - 0.5 * H
    for x, y, _ang in places:
        assert math.hypot(x, y) == pytest.approx(want, abs=0.2)


def test_the_outside_reads_clockwise_over_the_top_and_the_inside_under_the_bottom():
    need_fonts()
    out = text3d.arc_places(TEXT, height=H, radius=R)
    into = text3d.arc_places(TEXT, height=H, radius=R, inside=True)
    assert out[0][2] > out[-1][2], "the outside letters do not run clockwise"
    assert into[0][2] < into[-1][2], "the inside letters do not run counter clockwise"
    assert out[0][0] < 0.0 < out[-1][0] and out[0][1] > 0.0 and out[-1][1] > 0.0, \
        "the outside text does not start at the left of the top and end at the right of it"
    assert into[0][0] < 0.0 < into[-1][0] and into[0][1] < 0.0 and into[-1][1] < 0.0, \
        "the inside text does not start at the left of the bottom and end at the right of it"


def test_the_room_of_the_letters_is_walked_along_the_circle():
    """The letters stand as far apart on the arc as they do on a line: the room of a letter (its
    advance) is the arc length it takes, and the whole text is centred on the top of the circle."""
    need_fonts()
    letters = fonts.letter_loops(TEXT, fonts.default_family(), H, False)
    assert len(letters) == len(TEXT)
    wide = sum(a for _loops, a in letters)
    assert text3d.text_width(TEXT, height=H) == pytest.approx(wide)
    assert text3d.text_width(TEXT, height=2.0 * H) == pytest.approx(2.0 * wide)
    for inside, sign, middle in ((False, -1.0, 0.5 * math.pi), (True, 1.0, -0.5 * math.pi)):
        run = 0.0
        places = text3d.arc_places(TEXT, height=H, radius=R, inside=inside)
        for (_loops, a), (_x, _y, ang) in zip(letters, places):
            want = run + 0.5 * a - 0.5 * wide
            assert sign * R * (ang - middle) == pytest.approx(want, abs=1e-6)
            run += a
        assert run == pytest.approx(wide), "the letters did not walk the whole of their room"
    # a letter does not fill the room the font gives it, so the room is a little wider than the
    # drawn ink; line and arc still measure the same text
    ink = float(np.concatenate([L for L in fonts.text_loops(TEXT, None, H, False)])[:, 0].max())
    assert wide == pytest.approx(ink, rel=0.08)


@pytest.mark.parametrize("inside", (False, True))
def test_the_hole_of_an_o_stays_a_hole_on_an_arc(inside):
    need_fonts()
    ring = text3d.text3d("O", height=H, depth=D, path="arc", radius=R, inside=inside)
    flat = text3d.text3d("O", height=H, depth=D)
    assert ring.status() == mf.Error.NoError
    assert ring.genus() == 1, "an o on an arc is not a ring"
    assert ring.volume() == pytest.approx(flat.volume(), rel=1e-6), "the hole was closed on the arc"


def test_the_holes_of_letters_that_stand_next_to_each_other_stay_holes():
    need_fonts()
    m = arc(text="oO8")
    assert m.volume() == pytest.approx(part(text="oO8").volume(), rel=1e-6)
    assert sum(c.genus() for c in m.decompose()) == 4, "a hole was lost on the way round"


def test_a_space_takes_room_on_the_arc_but_draws_nothing():
    need_fonts()
    places = text3d.arc_places("A A", height=H, radius=R)
    assert len(places) == 3, "the space between the letters lost its place"
    m = text3d.text3d("A A", height=H, depth=D, path="arc", radius=R)
    assert m.status() == mf.Error.NoError
    assert len(m.decompose()) == 2, "a space was drawn, or a letter was lost"
    assert m.volume() == pytest.approx(2.0 * text3d.text3d("A", height=H, depth=D).volume(),
                                       rel=1e-6)
    close = text3d.arc_places("AA", height=H, radius=R)
    assert abs(places[2][2] - places[0][2]) > abs(close[1][2] - close[0][2]), \
        "a space takes no room on the arc"


def test_the_smallest_radius_of_a_text_is_the_one_it_needs():
    need_fonts()
    wide = text3d.text_width(TEXT, height=H)
    need = text3d.min_radius(TEXT, height=H)
    assert need == pytest.approx(max(text3d.RADIUS_MIN, wide / (2.0 * math.pi)), abs=0.1)
    assert need * 2.0 * math.pi >= wide, "the text would walk past the whole circle"
    assert text3d.min_radius("A", height=H) == text3d.RADIUS_MIN, "a short text may not go to 5 mm"
    assert arc(radius=need).status() == mf.Error.NoError, "the text does not fit its own radius"
    tall = text3d.min_radius(TEXT, height=text3d.HEIGHT_MAX)
    assert text3d.RADIUS_MIN < tall < text3d.RADIUS_MAX


def test_a_text_that_walks_past_the_whole_circle_names_the_radius_it_needs():
    need_fonts()
    tall = text3d.min_radius(TEXT, height=text3d.HEIGHT_MAX)
    assert text3d.RADIUS_MIN < tall < text3d.RADIUS_MAX
    with pytest.raises(ValueError) as ei:
        arc(height=text3d.HEIGHT_MAX, radius=0.6 * tall)
    assert "radius" in str(ei.value)
    assert f"{tall:.1f}" in str(ei.value), "the message does not say which radius fits"
    assert arc(height=text3d.HEIGHT_MAX,
               radius=min(text3d.RADIUS_MAX, 1.5 * tall)).status() == mf.Error.NoError


@pytest.mark.parametrize("text", ("Drawer\nTools", "A\n\nB", "A\nB\n", "\nA\nB \n"))
def test_an_arc_takes_one_line_of_text(text):
    need_fonts()
    with pytest.raises(ValueError, match="one line"):
        arc(text=text)
    with pytest.raises(ValueError, match="one line"):
        text3d.arc_places(text, height=H, radius=R)


@pytest.mark.parametrize("text", ("Tools", "Tools\n", "\n\nTools  ", "  Tools\n"))
def test_the_empty_lines_rounding_one_line_are_not_lines(text):
    need_fonts()
    assert text3d.arc_lines(text) == 1
    assert arc(text=text).volume() == pytest.approx(arc(text="Tools").volume(), rel=1e-9)


@pytest.mark.parametrize("text, lines", (("Tools", 1), ("Tools\n", 1), ("\n\nTools\n\n", 1),
                                         ("A\nB", 2), ("A\n \nB", 2), (" \n \n", 0), ("", 0),
                                         (None, 0)))
def test_arc_lines_counts_the_lines_that_draw_something(text, lines):
    assert text3d.arc_lines(text) == lines


def test_an_unknown_path_is_said_plainly():
    with pytest.raises(ValueError, match="path"):
        part(path="circle")


@pytest.mark.parametrize("text", ("", "   ", None))
def test_an_arc_of_nothing_is_said_plainly(text):
    with pytest.raises(ValueError, match="needs text"):
        arc(text=text)


def test_a_font_that_is_not_there_is_said_plainly_for_an_arc_too(monkeypatch):
    monkeypatch.setattr(fonts, "default_family", lambda *a, **kw: "")
    with pytest.raises(ValueError, match="No font found"):
        arc()


def test_a_font_that_draws_nothing_is_said_plainly_for_an_arc_too(monkeypatch):
    monkeypatch.setattr(fonts, "default_family", lambda *a, **kw: "Nowhere Sans")
    monkeypatch.setattr(fonts, "letter_loops", lambda *a, **kw: [])
    with pytest.raises(ValueError, match="No font found"):
        arc()


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
    assert m.volume() == pytest.approx(text3d.text3d(**OPTIONS).volume(), rel=1e-4)
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


def test_the_arc_text_is_one_group_that_one_undo_takes_away(win):
    need_fonts()
    steps = len(win.doc.undo_stack)
    win.make_text3d_model(ARCOPTS)
    assert pump_until(lambda: len(win.doc.groups) == 1), "no arc text in the model"
    assert len(win.doc.undo_stack) == steps + 1, "the arc text is not one undo step"
    g = win.doc.groups[0]
    assert g.name == "Tools", "the group is not named after the line"
    m = g.ents.to_manifold()
    assert m.status() == mf.Error.NoError
    assert m.volume() == pytest.approx(text3d.text3d(**ARCOPTS).volume(), rel=1e-4)
    assert [g.id for g in win.doc.selected_groups()] == [g.id for g in win.doc.groups]
    assert pump_until(lambda: win.hint.text() == "3D text added."), win.hint.text()
    win.doc.undo()
    assert win.doc.groups == []


def test_the_arc_text_stands_on_the_bed_around_the_middle_of_it(win):
    need_fonts()
    win.make_text3d_model(ARCIN)
    assert pump_until(lambda: len(win.doc.groups) == 1), "no arc text in the model"
    assert pump_until(lambda: win.doc.groups[0].bbox() is not None)
    lo, hi = np.asarray(win.doc.groups[0].bbox(), dtype=float)
    assert lo[2] == pytest.approx(0.0), "the arc text does not stand on the bed"
    assert hi[2] - lo[2] == pytest.approx(D)
    bx, by, _bz = win.settings.bed_size
    assert 0.5 * (lo[0] + hi[0]) == pytest.approx(0.5 * bx), "the arc text is not on the bed"
    assert 0.5 * (lo[1] + hi[1]) == pytest.approx(0.5 * by), "the arc text is not on the bed"


def test_an_arc_that_cannot_be_built_is_a_clear_text(win):
    need_fonts()
    win.make_text3d_model(dict(ARCOPTS, text="Drawer\nTools"))
    assert pump_until(lambda: "3D text failed" in win.hint.text()), win.hint.text()
    assert "one line" in win.hint.text(), win.hint.text()
    assert win.doc.groups == []
    win.make_text3d_model(dict(ARCOPTS, radius=5.0, height=text3d.HEIGHT_MAX))
    assert pump_until(lambda: "radius" in win.hint.text()), win.hint.text()
    assert win.doc.groups == [], "text that walks past the circle was built anyway"


def test_the_dialog_shows_the_arc_fields_only_for_an_arc_and_remembers_them(win, monkeypatch):
    need_fonts()
    seen = {}

    def fill(dlg):
        seen["start"] = [str(dlg.path.currentData()), round(dlg.radius.value(), 2),
                         dlg.inside.isChecked(), str(dlg.align.currentData())]
        seen["arc"] = [dlg.radius.isHidden(), dlg.inside.isHidden(), dlg.align.isHidden(),
                       dlg.bb.button(QDialogButtonBox.Ok).isEnabled()]
        dlg.path.setCurrentIndex(dlg.path.findData("straight"))
        seen["line"] = [dlg.radius.isHidden(), dlg.inside.isHidden(), dlg.align.isHidden()]
        dlg.path.setCurrentIndex(dlg.path.findData("arc"))
        seen["back"] = [dlg.radius.isHidden(), dlg.inside.isHidden(), dlg.align.isHidden()]
        dlg.radius.setValue(120.0)
        dlg.inside.setChecked(True)
        dlg.text.setPlainText("Round\nlid")
        seen["two"] = [dlg.info.text(), theme.color("invalid") in dlg.text.styleSheet(),
                       theme.color("warn") in dlg.info.styleSheet(),
                       dlg.bb.button(QDialogButtonBox.Ok).isEnabled()]
        dlg.text.setPlainText("Round lid")
        seen["one"] = [dlg.info.text(), dlg.text.styleSheet(),
                       dlg.bb.button(QDialogButtonBox.Ok).isEnabled()]
        return QDialog.Accepted

    monkeypatch.setattr(app_module.QDialog, "exec", fill)
    win.settings.text3d_text = "Lid"
    win.settings.text3d_path, win.settings.text3d_radius = "arc", 90.0
    win.settings.text3d_inside, win.settings.text3d_align = True, "right"
    win.text3d_dialog()
    assert seen["start"] == ["arc", 90.0, True, "right"], "the arc values did not come back"
    assert seen["arc"] == [False, False, True, True], "an arc does not show its own fields"
    assert seen["line"] == [True, True, False], "a line does not hide the arc fields"
    assert seen["back"] == [False, False, True], "the fields did not come back with the arc"
    assert seen["two"] == ["An arc takes one line of text", True, True, False], seen["two"]
    assert seen["one"] == ["", "", True], "the refusal stayed after the second line was gone"
    assert pump_until(lambda: len(win.doc.groups) == 1), "no arc text in the model"
    assert win.doc.groups[0].name == "Round lid", "the group is not named after the line"
    lo, hi = np.asarray(win.doc.groups[0].bbox(), dtype=float)
    assert hi[2] - lo[2] == pytest.approx(D), "the depth is not as asked"
    assert (win.settings.text3d_text, win.settings.text3d_path) == ("Round lid", "arc")
    assert win.settings.text3d_radius == pytest.approx(120.0)
    assert win.settings.text3d_inside is True
    s = app_module.Settings()
    assert (s.text3d_path, s.text3d_radius, s.text3d_inside) == ("arc", pytest.approx(120.0), True)


ARC_KEYS = ("text3d_path", "text3d_radius", "text3d_inside")


def test_a_settings_file_without_the_arc_or_with_nonsense_in_it_still_gives_a_path():
    """A settings file of before the arc, or one with nonsense in it: the dialog starts on a
    straight line with a radius it can show, and an arc that is asked for, is taken."""
    for key in ARC_KEYS:
        qsettings().remove(key)
    qsettings().sync()
    s = app_module.Settings()
    assert (s.text3d_path, s.text3d_inside) == ("straight", False)
    assert s.text3d_radius == pytest.approx(text3d.RADIUS_START)
    qsettings().setValue("text3d_path", "spiral")
    qsettings().setValue("text3d_radius", "wide")
    qsettings().setValue("text3d_inside", "sometimes")
    qsettings().sync()
    s = app_module.Settings()
    assert (s.text3d_path, s.text3d_inside) == ("straight", False), "nonsense was believed"
    assert s.text3d_radius == pytest.approx(text3d.RADIUS_START), "nonsense was believed"
    qsettings().setValue("text3d_path", "arc")
    qsettings().setValue("text3d_radius", 1e9)
    qsettings().setValue("text3d_inside", True)
    qsettings().sync()
    s = app_module.Settings()
    assert s.text3d_path == "arc"
    assert s.text3d_inside is True
    assert s.text3d_radius == pytest.approx(text3d.RADIUS_MAX), "the radius went past its limit"
    for key in ARC_KEYS:
        qsettings().remove(key)
    qsettings().sync()


# ------------------------------------------------------------------ in the menus

def test_the_command_is_in_the_draw_menu_after_the_parts(win):
    labels = [clean(a.text()) for a in win.flat_actions(win.menuBar())]
    assert labels.index("3D Text") == labels.index("Wall Hook") + 1


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

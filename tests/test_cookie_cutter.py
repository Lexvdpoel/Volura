"""Cookie cutters: an outline of the worksheet, a drawing or a word, turned into a cutter you press.

`cookie_cutter.cookie_cutter` gives the cutter as closed volume standing on `z = 0` and centred: a
wall of `wall` millimetres that follows every outer outline of the source with round joins and stands
`height` millimetres high, thin to half its thickness over its top `EDGE` millimetres while `edge` is
on, and standing on a flange `flange` millimetres wide and `flange_thickness` millimetres thick. The
whole source is scaled together to `width` millimetres, so several outlines keep their sizes and
their places to one another. A line within an outline cannot stand as a wall that nothing holds, so
with `stamp` such a line becomes a ridge half a wall wide that stands `EDGE` under the cutting edge
and is bridged to the flange by spokes; a ridge that no spoke reaches is left out instead of left
loose.

These tests weigh the cutter against that: one volume on the bed and centred, the wall measured
around the outline and the flange that reaches past it, an edge half the wall and a wall without one
as high as the cutter, a hole that is not cut and a stamp that stays one piece and under the edge and
within the place the cutter takes, held by its flange and by the wall of a cutter whose flange is on
0; outlines that keep their sizes to one another, letters, a drawing,
a source that is very small and one that lies far from the origin; every range reached by clamping
and every number that is no number, every source without a closed outline and every source that is no
source refused; and the window with its dialog: the worksheet choice out of reach without a selection,
a cutter made of what is selected, one group in the middle of the bed that one undo takes away, OK
that waits for a closed outline and the values that come back the next time.

Run with:  python -m pytest tests/test_cookie_cutter.py
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
from PySide6.QtWidgets import QApplication, QDialog, QDialogButtonBox  # noqa: E402

import volura.app as app_module  # noqa: E402
from volura import cookie_cutter, fonts  # noqa: E402
from volura.vector2d import VPath, next_src  # noqa: E402

_app = QApplication.instance() or QApplication(sys.argv[:1])       # the letters need Qt

BASE = dict(source="sheet", width=70.0, wall=0.8, height=15.0, flange=4.0,
            flange_thickness=2.0, edge=True, stamp=False)
#: the outline of the source is a square of 50 mm, which the asked width of 70 mm makes this wide
SIDE = 70.0
#: that scaled outline, so a band can be measured against it
OUTLINE = np.array([[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]]) * SIDE
#: the rounded joins are polygons, so a part is at most this far off
ABS = 0.05
#: mm across a probe that feels for plastic
PROBE = 0.4
#: the drawing of the tests: a square with a square hole in it, and a square beside it
SVG = """<svg xmlns="http://www.w3.org/2000/svg" width="100mm" height="50mm"
     viewBox="0 0 100 50"><path fill-rule="evenodd" d="M0 0 L40 0 L40 40 L0 40 Z
     M10 10 L30 10 L30 30 L10 30 Z"/><rect x="60" y="0" width="40" height="30"/></svg>"""
#: the sizes the drawing is cut with: two cutters of 90 mm together
DRAWING = dict(source="file", width=90.0, wall=1.2, height=12.0, flange=6.0,
               flange_thickness=3.0)


def square(side=50.0, x=0.0, y=0.0, src=1) -> VPath:
    """A closed square of the worksheet, as the selection hands it over."""
    pts = np.array([[x, y], [x + side, y], [x + side, y + side], [x, y + side]], dtype=float)
    return VPath(pts, True, rule="nonzero", src=src)


def ring(r_out=30.0, r_in=12.0, segments=64) -> tuple:
    """A ring of the worksheet: an outline with a hole in it, filled by the even-odd rule.

    The hole is a quarter of the outline wide, so it is 14 mm across once the outline is 70 mm.
    """
    t = np.linspace(0.0, 2.0 * math.pi, segments + 1)[:-1]
    return (VPath(np.c_[r_out * np.cos(t), r_out * np.sin(t)], True, rule="evenodd", src=1),
            VPath(np.c_[r_in * np.cos(t), r_in * np.sin(t)], True, rule="evenodd", src=1))


def star(r_out=30.0, r_in=13.0, points=5) -> VPath:
    """A star with as many sharp points as `points`, the way a cookie cutter is often drawn."""
    pts = []
    for i in range(2 * points):
        r = r_out if i % 2 == 0 else r_in
        a = math.pi / 2.0 + i * math.pi / points
        pts.append((r * math.cos(a), r * math.sin(a)))
    return VPath(np.array(pts, dtype=float), True, rule="nonzero", src=1)


def spec(**kw) -> cookie_cutter.Spec:
    """A cutter of `sheet` - one square of 50 mm by default - with the sizes that are not given."""
    kw.setdefault("sheet", (square(),))
    return cookie_cutter.Spec(**dict(BASE, **kw))


def part(**kw) -> mf.Manifold:
    return cookie_cutter.cookie_cutter(spec(**kw))


def sizes(**kw) -> dict:
    return cookie_cutter.sizes(spec(**kw))


def drawing(tmp_path, text=SVG, name="cutter.svg") -> str:
    """A drawing on disk, the way the file source reads it."""
    path = os.path.join(str(tmp_path), name)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)
    return path


def need_fonts():
    """Skip the case when this system has no fonts, as the other tests with letters do."""
    if not fonts.has_fonts():
        pytest.skip("this system has no fonts, so there are no letters to test")


def bbox(m):
    """The bounding box of a manifold as (low, high)."""
    b = np.asarray(m.bounding_box(), dtype=float)
    return b[:3], b[3:]


def size_of(box) -> np.ndarray:
    """How big a bounding box is in x, y and z."""
    lo, hi = box
    return hi - lo


def area(loop) -> float:
    """The signed area of a closed polygon."""
    p = np.asarray(loop, dtype=float)
    return 0.5 * float(np.dot(p[:, 0], np.roll(p[:, 1], -1))
                       - np.dot(p[:, 1], np.roll(p[:, 0], -1)))


def cut(m, z) -> list:
    """The contours of the cutter where the plane `z` crosses it, the biggest first."""
    return sorted((np.asarray(p, dtype=float) for p in m.slice(z).to_polygons()),
                  key=lambda p: -abs(area(p)))


def reach(loop) -> tuple:
    """How far a contour reaches from the middle of the cutter in x and in y."""
    p = np.asarray(loop, dtype=float)
    return float(np.abs(p[:, 0]).max()), float(np.abs(p[:, 1]).max())


def plug(m, x, y, z0, z1, r=PROBE) -> float:
    """The volume of a round post of radius `r` at (`x`, `y`) between `z0` and `z1`.

    A post that misses the cutter gives nothing back; one that stands in plastic is off by whole
    tenths of a millimetre, so the two are easy to tell apart.
    """
    return (m ^ mf.Manifold.cylinder(z1 - z0, r, r, 32).translate((x, y, z0))).volume()


def band(offset, outline=OUTLINE) -> float:
    """The exact area of a band `offset` millimetres outside a closed outline with round joins.

    Such a band is the outline grown by `offset`: its area grows by the perimeter of the outline
    times `offset`, while the corners that roll open together make one circle of that radius.
    """
    closed = np.vstack([outline, outline[:1]])
    perimeter = float(np.sum(np.linalg.norm(np.diff(closed, axis=0), axis=1)))
    return perimeter * offset + math.pi * offset ** 2


# ------------------------------------------------------------------ the piece

def test_the_cutter_is_one_volume_that_stands_on_its_flange_and_is_centred():
    s, m = sizes(), part()
    assert m.status() == mf.Error.NoError and not m.is_empty()
    assert len(m.decompose()) == 1, "the cutter fell apart"
    lo, hi = bbox(m)
    assert lo[2] == pytest.approx(0.0, abs=1e-6), "the cutter does not stand on its flange"
    assert np.allclose(lo[:2], -0.5 * np.array([s["size_x"], s["size_y"]]), atol=ABS), \
        "not centred in x and y"
    assert size_of(bbox(m))[2] == pytest.approx(BASE["height"], abs=ABS), "not as high as asked"


def test_the_wall_follows_the_outline_and_leaves_the_dough_free():
    """The wall stands just outside the scaled outline and the inside of the outline stays empty:
    the dough within the outline is the cookie."""
    m = part()
    outer, hole = cut(m, 0.5 * BASE["height"])
    assert reach(outer)[0] == pytest.approx(0.5 * SIDE + BASE["wall"], abs=ABS), \
        "the wall is not outside the outline"
    assert reach(hole)[0] == pytest.approx(0.5 * SIDE, abs=ABS), "the wall ate into the outline"
    assert m.slice(0.5 * BASE["height"]).area() == pytest.approx(band(BASE["wall"]), rel=2e-3), \
        "the wall is not as thick as asked"
    assert len(cut(m, 0.5 * BASE["height"])) == 2, "the cutter is not a hollow wall"
    assert plug(m, 0.0, 0.0, 2.0 * BASE["flange_thickness"], BASE["height"] - 2.0) \
        == pytest.approx(0.0, abs=1e-6), "there is plastic where the dough has to come"


def test_the_flange_reaches_past_the_wall_and_is_the_only_thing_that_touches_the_bed():
    low, thick = BASE["wall"] + BASE["flange"], BASE["flange_thickness"]
    m = part()
    assert len(cut(m, 0.5 * thick)) == 2, "the flange is not a ring round the wall"
    assert reach(cut(m, 0.5 * thick)[0])[0] == pytest.approx(0.5 * SIDE + low, abs=ABS), \
        "the flange does not reach as far past the wall as asked"
    assert m.slice(0.5 * thick).area() == pytest.approx(band(low), rel=2e-3), \
        "the flange is not as wide as asked"
    assert m.slice(thick + 0.5).area() == pytest.approx(
        m.slice(0.5 * BASE["height"]).area(), rel=1e-6), \
        "above its flange the cutter is wider than its wall"


def test_the_sharp_edge_is_half_the_wall_and_stands_on_top_of_it():
    m = part()
    top = BASE["height"] - 0.5 * cookie_cutter.EDGE
    assert reach(cut(m, top)[0])[0] == pytest.approx(0.5 * SIDE + 0.5 * BASE["wall"], abs=ABS), \
        "the edge is not half the wall thick"
    assert m.slice(top).area() == pytest.approx(band(0.5 * BASE["wall"]), rel=2e-3), \
        "the edge is not a wall of half the thickness"
    assert reach(cut(m, 0.5 * BASE["height"])[0])[0] == pytest.approx(
        0.5 * SIDE + BASE["wall"], abs=ABS), "the wall is thin where the edge does not reach"
    assert plug(m, 0.5 * SIDE + 0.75 * BASE["wall"], 0.0, BASE["flange_thickness"], top) > 0.0, \
        "nothing of the full wall is left under the edge"


def test_a_cutter_without_a_sharp_edge_keeps_its_wall_to_the_top():
    m = part(edge=False)
    lo, hi = bbox(m)
    assert hi[2] - lo[2] == pytest.approx(BASE["height"], abs=ABS), \
        "a cutter without a sharp edge stands another height"
    assert reach(cut(m, BASE["height"] - 0.5)[0])[0] == pytest.approx(
        0.5 * SIDE + BASE["wall"], abs=ABS), "a cutter without a sharp edge lost wall at the top"


@pytest.mark.parametrize("flange", (0.0, 4.0, 10.0))
def test_the_cutter_is_as_wide_as_its_wall_and_its_flange_together(flange):
    """With 0 the wall stands on the bed itself, which is a cutter of nothing but its wall."""
    m = part(flange=flange)
    assert m.status() == mf.Error.NoError and len(m.decompose()) == 1, "the cutter fell apart"
    assert size_of(bbox(m))[0] == pytest.approx(
        SIDE + 2.0 * (BASE["wall"] + flange), abs=ABS), "not the wall and the flange asked for"
    assert bbox(m)[0][2] == pytest.approx(0.0, abs=1e-6), "the cutter does not stand on the bed"


def test_a_star_of_sharp_points_stays_one_cutter_of_its_own_place():
    """The points of a star are exactly where a round join matters: without one the two sides of the
    wall run on until they cross and the point of the cutter bends away while it is pressed."""
    loop = np.asarray(star().pts, dtype=float)
    span = loop.max(axis=0) - loop.min(axis=0)
    factor = BASE["width"] / float(span[0])
    room = BASE["wall"] + BASE["flange"]
    m = part(sheet=(star(),))
    assert m.status() == mf.Error.NoError and not m.is_empty()
    assert len(m.decompose()) == 1, "the star was cut in two"
    assert size_of(bbox(m))[0] == pytest.approx(factor * span[0] + 2.0 * room, abs=ABS), \
        "the star is not as wide as asked"
    assert size_of(bbox(m))[1] == pytest.approx(factor * span[1] + 2.0 * room, abs=ABS), \
        "the star did not keep its shape while it was scaled"


# ------------------------------------------------------------------ the lines within an outline

def test_a_hole_in_the_source_is_neither_cut_nor_left_as_a_wall_of_its_own():
    """A ring is one cutter of its outer outline. The hole is not cut: a wall that stands in the
    middle of an opening and is held by nothing would break off in the dough."""
    m = part(sheet=ring())
    assert m.status() == mf.Error.NoError and len(m.decompose()) == 1, "the ring was cut in two"
    assert size_of(bbox(m))[0] == pytest.approx(
        BASE["width"] + 2.0 * (BASE["wall"] + BASE["flange"]), abs=ABS)
    assert len(cut(m, 0.5 * BASE["height"])) == 2, "the hole of the ring was cut as well"
    assert plug(m, 14.0, 0.0, BASE["flange_thickness"], BASE["height"] - 2.0) \
        == pytest.approx(0.0, abs=1e-6), "the hole of the ring became a wall of its own"


def test_the_stamp_marks_the_line_within_the_outline_under_the_edge():
    """With the stamp the hole of a ring gets a ridge half a wall wide whose top stands `EDGE` under
    the cutting edge: the dough is marked instead of cut."""
    off, on = part(sheet=ring()), part(sheet=ring(), stamp=True)
    assert on.status() == mf.Error.NoError, "the stamp made an invalid cutter"
    assert len(on.decompose()) == 1, "the stamp fell out of the cutter"
    assert on.volume() > off.volume(), "the stamp added nothing"
    assert len(cut(on, 0.5 * BASE["height"])) == len(cut(off, 0.5 * BASE["height"])) + 2, \
        "the line within the outline did not become a ridge"
    assert plug(on, 14.0, 0.0, BASE["flange_thickness"], BASE["height"] - 2.1) > 0.0, \
        "the stamp left no ridge on the line within the outline"
    assert plug(on, 14.0, 0.0, BASE["height"] - cookie_cutter.EDGE + 0.05, BASE["height"]) \
        == pytest.approx(0.0, abs=1e-6), "the ridge reaches up to the cutting edge"


def test_the_stamp_is_held_by_the_flange_and_stays_within_the_cutter():
    """The spokes are webs in the thickness of the flange: they split the opening there and nowhere
    else, and the cutter takes exactly the place it took without the stamp."""
    off, on = part(sheet=ring()), part(sheet=ring(), stamp=True)
    assert np.allclose(bbox(on)[0][:2], bbox(off)[0][:2], atol=ABS) \
        and np.allclose(bbox(on)[1][:2], bbox(off)[1][:2], atol=ABS), \
        "a spoke sticks out beside the flange"
    assert len(cut(on, 0.5 * BASE["flange_thickness"])) > len(
        cut(off, 0.5 * BASE["flange_thickness"])), "nothing bridges to the ridge"
    assert plug(on, 24.5, 0.0, BASE["flange_thickness"] + 0.1, BASE["height"] - 2.0) \
        == pytest.approx(0.0, abs=1e-6), "a spoke stands in the opening as high as the cutter"


def test_the_stamp_is_also_held_by_the_wall_of_a_cutter_without_a_flange():
    """With the flange on 0 there is no flange to bridge to, so the spoke welds into the wall
    instead: the ridge is held, the cutter is one piece and it takes the same place on the bed."""
    off, on = part(sheet=ring(), flange=0.0), part(sheet=ring(), flange=0.0, stamp=True)
    assert on.status() == mf.Error.NoError and len(on.decompose()) == 1, \
        "a stamp without a flange to hold it fell apart"
    assert on.volume() > off.volume(), "the ridge was left out although the wall can hold it"
    assert bbox(on)[0][2] == pytest.approx(0.0, abs=1e-6), "the cutter does not stand on the bed"
    assert np.allclose(bbox(on)[0][:2], bbox(off)[0][:2], atol=ABS) \
        and np.allclose(bbox(on)[1][:2], bbox(off)[1][:2], atol=ABS), \
        "a spoke sticks out beside a cutter without a flange"
    assert plug(on, 14.0, 0.0, BASE["flange_thickness"], BASE["height"] - 2.1) > 0.0, \
        "the ridge is not there although the wall holds it"
    assert plug(on, 14.0, 0.0, BASE["height"] - cookie_cutter.EDGE + 0.05, BASE["height"]) \
        == pytest.approx(0.0, abs=1e-6), "the ridge reaches up to the cutting edge"


def test_a_ridge_that_no_spoke_reaches_is_left_out():
    """A line out of reach of the outline loses its ridge: a stamp one detail short is still a
    stamp, while a ridge that hangs loose is plastic in the dough."""
    outline = np.array([[-20.0, -20.0], [20.0, -20.0], [20.0, 20.0], [-20.0, 20.0]], dtype=float)
    out_of_reach = np.array([[80.0, 80.0], [90.0, 80.0], [90.0, 90.0], [80.0, 90.0]], dtype=float)
    within_reach = np.array([[-5.0, -5.0], [5.0, -5.0], [5.0, 5.0], [-5.0, 5.0]], dtype=float)
    s = cookie_cutter.numbers(spec())
    grown = cookie_cutter._offset(cookie_cutter._cs(outline), s["wall"] + s["flange"])
    foot = grown - cookie_cutter._cs(outline)
    assert cookie_cutter._stamp(outline, [out_of_reach], grown, foot, s) == [], \
        "a ridge was kept that nothing holds"
    assert cookie_cutter._stamp(outline, [within_reach], grown, foot, s), \
        "a ridge that the flange does reach was left out"


# ------------------------------------------------------------------ several outlines

def test_several_outlines_keep_their_sizes_and_their_places_to_one_another():
    """The whole source is scaled together, so the small cutter stays half as big as the one beside
    it instead of every outline growing to the asked width."""
    small, big = square(20.0, 60.0), square(40.0, 0.0)
    room = BASE["wall"] + BASE["flange"]
    m = part(sheet=(big, small))
    assert m.status() == mf.Error.NoError
    assert len(m.decompose()) == 2, "the two outlines did not become two cutters"
    assert sizes(sheet=(big, small))["cutters"] == 2
    widths = sorted(size_of(bbox(p))[0] - 2.0 * room for p in m.decompose())
    assert widths[1] == pytest.approx(2.0 * widths[0], abs=ABS), \
        "the small outline was scaled to the same width as the big one"
    assert size_of(bbox(m))[0] == pytest.approx(BASE["width"] + 2.0 * room, abs=ABS), \
        "the two cutters do not take the width asked for together"


def test_open_lines_in_a_selection_are_passed_over():
    """A selection that also holds a stray line still makes the cutter of the shape in it: what is
    no outline is on its way out, and the user is not asked to select more carefully."""
    m = part(sheet=(square(), VPath(np.array([[0.0, 0.0], [10.0, 0.0]]), False)))
    assert len(m.decompose()) == 1, "a stray line made a cutter of its own"
    assert size_of(bbox(m))[0] == pytest.approx(SIDE + 2.0 * (BASE["wall"] + BASE["flange"]),
                                                abs=ABS)


# ------------------------------------------------------------------ letters and drawings

def test_letters_become_a_cutter_of_themselves():
    need_fonts()
    s, m = sizes(source="text", text="O"), part(source="text", text="O")
    assert s["cutters"] == 1, "one letter made more than one cutter"
    assert m.status() == mf.Error.NoError and len(m.decompose()) == 1, "the letter fell apart"
    assert size_of(bbox(m))[0] == pytest.approx(s["size_x"], abs=ABS), "not as wide as asked"
    assert size_of(bbox(m))[1] == pytest.approx(s["size_y"], abs=ABS), \
        "the letter did not keep its shape while it was scaled"
    assert size_of(bbox(m))[2] == pytest.approx(BASE["height"], abs=ABS)
    assert len(cut(m, 0.5 * BASE["height"])) == 2, "the middle of the letter was cut as well"


def test_two_letters_become_two_cutters_in_the_width_asked():
    need_fonts()
    s = sizes(source="text", text="OO")
    assert s["cutters"] == 2, "two letters did not become two cutters"
    assert s["size_x"] == pytest.approx(BASE["width"] + 2.0 * (BASE["wall"] + BASE["flange"]),
                                        abs=ABS), "the word was not scaled to the width asked"


def test_a_drawing_makes_one_cutter_per_outer_outline(tmp_path):
    """A drawing of a square with a hole and a square beside it: two cutters, and the hole of the
    drawing is left to the stamp."""
    kw = spec(**dict(DRAWING, file=drawing(tmp_path)))
    s = cookie_cutter.sizes(kw)
    assert s["cutters"] == 2, "the hole of the drawing became a cutter of its own"
    assert s["size_x"] == pytest.approx(90.0 + 2.0 * 7.2, abs=ABS)
    assert s["size_y"] == pytest.approx(36.0 + 2.0 * 7.2, abs=ABS), "the drawing lost its shape"
    m = cookie_cutter.cookie_cutter(kw)
    assert m.status() == mf.Error.NoError and len(m.decompose()) == 2, "not two cutters"
    lo, hi = bbox(m)
    assert lo[2] == pytest.approx(0.0, abs=1e-6) and hi[2] == pytest.approx(12.0, abs=ABS)
    assert len(cut(m, 6.0)) == 4, "the hole in the drawing was cut as well"


@pytest.mark.parametrize("side", (1.0, 0.1))
def test_a_source_that_is_very_small_still_becomes_a_cutter_of_the_asked_width(side):
    """The width is what the source becomes, whatever size its own units were in."""
    m = part(sheet=(square(side),))
    assert size_of(bbox(m))[0] == pytest.approx(SIDE + 2.0 * (BASE["wall"] + BASE["flange"]),
                                                abs=ABS), "a small source came out another size"
    assert size_of(bbox(m))[2] == pytest.approx(BASE["height"], abs=ABS)


def test_a_source_that_lies_far_from_the_origin_is_brought_to_the_middle_of_the_bed():
    """A drawing that was exported around a distant datum still stands centred, at its own size and
    with its flange on the bed."""
    far = np.array([[0.0, 0.0], [50.0, 0.0], [50.0, 50.0], [0.0, 50.0]], dtype=float) + 1e6
    m = part(sheet=(VPath(far, True, rule="nonzero", src=1),))
    lo, hi = bbox(m)
    assert np.allclose(0.5 * (lo[:2] + hi[:2]), 0.0, atol=1e-4), "the cutter is not centred"
    assert lo[2] == pytest.approx(0.0, abs=1e-6), "the cutter does not stand on the bed"
    assert size_of(bbox(m))[0] == pytest.approx(SIDE + 2.0 * (BASE["wall"] + BASE["flange"]),
                                                abs=ABS), "a source far off came out another size"


# ------------------------------------------------------------------ the numbers

def test_sizes_reports_the_cutters_the_source_makes():
    s = sizes()
    assert set(s) == {"cutters", "size_x", "size_y", "height"}
    assert s["cutters"] == 1 and s["height"] == BASE["height"]
    room = 2.0 * (BASE["wall"] + BASE["flange"])
    assert s["size_x"] == pytest.approx(SIDE + room) and s["size_y"] == pytest.approx(SIDE + room)
    assert sizes(sheet=(square(40.0, 0.0), square(20.0, 60.0)))["cutters"] == 2
    assert sizes(sheet=ring())["cutters"] == 1, "the hole of a ring became a cutter of its own"
    assert sizes(sheet=ring(), stamp=True)["cutters"] == 1, "the stamp made a cutter of a line"


def test_the_default_cutter_is_a_word_of_seventy_millimetres():
    s = cookie_cutter.numbers(cookie_cutter.Spec())
    assert s["source"] == cookie_cutter.DEFAULT_SOURCE == "text"
    assert s["text"] == "Volura" and s["family"] == "" and s["bold"] is False
    assert s["file"] == "" and s["sheet"] == ()
    assert s["width"] == 70.0 and s["wall"] == 0.8 and s["height"] == 15.0
    assert s["flange"] == 4.0 and s["flange_thickness"] == 2.0
    assert s["edge"] is True and s["stamp"] is False


def test_the_numbers_are_clamped_to_what_prints():
    small = cookie_cutter.numbers(spec(width=0.0, wall=0.0, height=0.0, flange=-1.0,
                                       flange_thickness=0.0))
    assert small["width"] == cookie_cutter.WIDTH_MIN and small["wall"] == cookie_cutter.WALL_MIN
    assert small["height"] == cookie_cutter.HEIGHT_MIN
    assert small["flange"] == cookie_cutter.FLANGE_MIN
    assert small["flange_thickness"] == cookie_cutter.RIM_MIN
    big = cookie_cutter.numbers(spec(width=999.0, wall=999.0, height=999.0, flange=999.0,
                                     flange_thickness=999.0))
    assert big["width"] == cookie_cutter.WIDTH_MAX and big["wall"] == cookie_cutter.WALL_MAX
    assert big["height"] == cookie_cutter.HEIGHT_MAX and big["flange"] == cookie_cutter.FLANGE_MAX
    assert big["flange_thickness"] == cookie_cutter.RIM_MAX
    for s in (small, big):
        m = cookie_cutter.cookie_cutter(spec(width=s["width"], wall=s["wall"],
                                             height=s["height"], flange=s["flange"],
                                             flange_thickness=s["flange_thickness"]))
        assert size_of(bbox(m))[0] == pytest.approx(
            s["width"] + 2.0 * (s["wall"] + s["flange"]), abs=ABS), "a clamped width was lost"
        assert size_of(bbox(m))[2] == pytest.approx(s["height"], abs=ABS), "a clamped height lost"


@pytest.mark.parametrize("bad", (float("nan"), float("inf"), None, "wide"))
def test_a_number_that_is_no_number_is_refused(bad):
    for key in ("width", "wall", "height", "flange", "flange_thickness"):
        with pytest.raises(ValueError):
            cookie_cutter.sizes(spec(**{key: bad}))
        with pytest.raises(ValueError):
            cookie_cutter.cookie_cutter(spec(**{key: bad}))


@pytest.mark.parametrize("source", ("cube", "Sheet", "", None))
def test_a_source_that_is_no_source_is_refused(source):
    with pytest.raises(ValueError):
        cookie_cutter.sizes(spec(source=source))
    with pytest.raises(ValueError):
        cookie_cutter.cookie_cutter(spec(source=source))


# ------------------------------------------------------------------ what is refused

@pytest.mark.parametrize("shapes", (
    (),                                       # nothing selected at all
    (VPath(np.array([[0.0, 0.0], [30.0, 0.0]]), False),),         # one open line
    (VPath(np.array([[0.0, 0.0], [30.0, 0.0], [30.0, 30.0]]), False),),  # an open corner
    (VPath(np.array([[0.0, 0.0], [30.0, 0.0]]), True),),          # a closed line of two points
    (VPath(np.array([[0.0, 0.0], [30.0, 0.0], [60.0, 0.0],
                     [30.0, 0.0]]), True),),                      # a closed line with no area
    (VPath(np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]), True),),    # all in one corner
))
def test_a_source_without_a_closed_outline_is_refused(shapes):
    with pytest.raises(ValueError):
        cookie_cutter.sizes(spec(sheet=shapes))
    with pytest.raises(ValueError):
        cookie_cutter.cookie_cutter(spec(sheet=shapes))


def test_a_drawing_offered_by_its_name_only_is_refused():
    with pytest.raises(ValueError):
        cookie_cutter.cookie_cutter(cookie_cutter.Spec(source="file", file="   ", width=70.0))


def test_a_drawing_that_is_not_there_is_refused(tmp_path):
    with pytest.raises(ValueError):
        cookie_cutter.cookie_cutter(cookie_cutter.Spec(
            source="file", file=drawing(tmp_path) + "x", width=70.0))


def test_a_drawing_of_open_lines_is_refused(tmp_path):
    open_lines = """<svg xmlns="http://www.w3.org/2000/svg" width="100mm" height="50mm"
     viewBox="0 0 100 50"><path d="M0 0 L40 0 L40 40"/></svg>"""
    with pytest.raises(ValueError):
        cookie_cutter.cookie_cutter(cookie_cutter.Spec(
            source="file", file=drawing(tmp_path, open_lines, "open.svg"), width=70.0))


def test_letters_without_text_are_refused():
    need_fonts()
    for text in ("", "   ", "\n"):
        with pytest.raises(ValueError):
            cookie_cutter.cookie_cutter(cookie_cutter.Spec(source="text", text=text, width=70.0))


# ------------------------------------------------------------------ the interface

DEFAULTS = {"cookie_cutter_source": "text", "cookie_cutter_text": "Volura",
            "cookie_cutter_family": "", "cookie_cutter_bold": False, "cookie_cutter_file": "",
            "cookie_cutter_width": 70.0, "cookie_cutter_wall": 0.8, "cookie_cutter_height": 15.0,
            "cookie_cutter_flange": 4.0, "cookie_cutter_flange_thickness": 2.0,
            "cookie_cutter_edge": True, "cookie_cutter_stamp": False}


@pytest.fixture()
def win(monkeypatch, tmp_path):
    """A window that starts from the cutter this part ships with, whatever a test before left."""
    monkeypatch.setattr(app_module.errors, "log_dir", lambda: str(tmp_path))
    w = app_module.MainWindow()
    for key, value in DEFAULTS.items():
        setattr(w.settings, key, value)
    w.settings.save()
    w.doc.dirty = False
    QCoreApplication.processEvents()
    yield w
    for key, value in DEFAULTS.items():
        setattr(w.settings, key, value)
    w.settings.save()
    w.doc.dirty = False
    w.close()


def wait_for_groups(w, count=1, seconds=30.0):
    """Wait for the cutter the window is still building, as the tests of the label do."""
    deadline = time.monotonic() + seconds
    while len(w.doc.groups) < count and time.monotonic() < deadline:
        _app.processEvents()
        time.sleep(0.005)


def select(win, pts, closed=True):
    """One shape on the worksheet, with the selection of the worksheet around it."""
    sheet = win.doc.sheet
    path = sheet.add_path(pts, closed=closed, layer=sheet.default_layer("cut"),
                          src=next_src(sheet.paths))
    win.set_mode("2d")
    win.sheet_view.set_selection([path])
    return path


def test_the_worksheet_choice_is_out_of_reach_without_a_selection(win):
    """Without closed shapes on the worksheet the dialog cannot promise a cutter of them: that
    choice is greyed out, says why and leaves the source on something that has outlines."""
    from volura.part_dialogs import CookieCutterDialog
    win.settings.cookie_cutter_source = "sheet"
    win.set_mode("3d")
    dlg = CookieCutterDialog(win)
    try:
        item = dlg.source.model().item(dlg.source.findData("sheet"))
        assert not item.isEnabled(), "the worksheet choice can be picked without a selection"
        assert "closed" in item.toolTip().lower(), "the greyed choice does not say why"
        assert dlg.source.currentData() == "text", "the dialog opens on a source without outlines"
        assert dlg.buttonBox.button(QDialogButtonBox.Ok).isEnabled(), \
            "the source the dialog fell back on is refused too"
        dlg.text.setPlainText("")
        assert not dlg.buttonBox.button(QDialogButtonBox.Ok).isEnabled(), \
            "OK stays on for a word of nothing"
    finally:
        dlg.deleteLater()


def test_the_worksheet_choice_is_offered_again_by_selected_shapes(win):
    from volura.part_dialogs import CookieCutterDialog
    select(win, np.array([[0.0, 0.0], [40.0, 0.0], [40.0, 40.0], [0.0, 40.0]]))
    dlg = CookieCutterDialog(win)
    try:
        item = dlg.source.model().item(dlg.source.findData("sheet"))
        assert item.isEnabled(), "the worksheet choice is greyed out with shapes selected"
        dlg.source.setCurrentIndex(dlg.source.findData("sheet"))
        assert dlg.source.currentData() == "sheet", "the dialog will not take the shapes"
        assert len(dlg.sheet_paths) == 1, "the dialog lost the selection on its way in"
        assert "1 cutter stands" in dlg.info.text(), f"the dialog miscounts: {dlg.info.text()}"
        assert "79.6" in dlg.info.text(), f"the dialog does not say how big: {dlg.info.text()}"
    finally:
        dlg.deleteLater()


def test_the_cutter_is_made_of_the_shapes_selected_on_the_worksheet(win, monkeypatch):
    """One command: one group of cutters, standing in the middle of the bed on its flange, and one
    undo takes the whole thing away again."""
    select(win, np.array([[0.0, 0.0], [50.0, 0.0], [50.0, 50.0], [0.0, 50.0]]))

    def fill(dlg):
        dlg.source.setCurrentIndex(dlg.source.findData("sheet"))
        dlg.width.setValue(70.0)
        dlg.stamp.setChecked(True)
        return QDialog.Accepted

    monkeypatch.setattr(app_module.QDialog, "exec", fill)
    win.cookie_cutter_dialog()
    wait_for_groups(win)
    assert len(win.doc.groups) == 1, "the shapes of the worksheet did not become one group"
    group = win.doc.groups[0]
    assert group.name == "Cookie cutter", "the group is not named after the part"
    assert group in win.doc.selected_groups(), "the cutter is not there to be moved on"
    lo, hi = np.asarray(group.bbox(), dtype=float)
    bx, by, _bz = win.settings.bed_size
    assert lo[2] == pytest.approx(0.0, abs=1e-6), "the cutter does not stand on the bed"
    assert 0.5 * (lo[0] + hi[0]) == pytest.approx(0.5 * bx, abs=ABS), "not the middle of the bed"
    assert 0.5 * (lo[1] + hi[1]) == pytest.approx(0.5 * by, abs=ABS), "not the middle of the bed"
    assert hi[0] - lo[0] == pytest.approx(79.6, abs=ABS), "not as wide as the dialog said"
    assert hi[2] - lo[2] == pytest.approx(15.0, abs=ABS), "not as high as the dialog said"
    assert win.settings.cookie_cutter_source == "sheet", "the source was not remembered"
    assert win.settings.cookie_cutter_stamp is True, "the stamp was not remembered"
    win.doc.undo()
    assert not win.doc.groups, "one undo did not take the cutter away again"


def test_a_worksheet_source_without_shapes_leaves_the_model_alone(win):
    """A command that is handed the worksheet source while nothing is selected says what it needs
    and builds nothing; the greyed-out choice of the dialog is what keeps this from happening."""
    from volura.part_dialogs import CookieCutterDialog
    msgs = []
    win.view.status_changed.connect(msgs.append)
    win.set_mode("3d")
    dlg = CookieCutterDialog(win)
    try:
        made = win.make_cookie_cutter_model(dlg.options() | {"source": "sheet", "shapes": ()})
    finally:
        dlg.deleteLater()
    assert made is None, "a cutter was built out of nothing"
    assert not win.doc.groups, "a cutter without an outline stands in the model"
    assert any("closed shapes" in m for m in msgs), msgs


def test_the_dialog_builds_a_cutter_of_a_drawing_and_remembers_it(win, monkeypatch, tmp_path):
    path = drawing(tmp_path)

    def fill(dlg):
        dlg.source.setCurrentIndex(dlg.source.findData("file"))
        assert dlg.text.isHidden(), "a drawing shows the fields of a word"
        assert not dlg.drawing_row.isHidden(), "the drawing has no row to type its name in"
        dlg.file.setText(path)
        dlg.width.setValue(90.0)
        dlg.wall.setValue(1.2)
        dlg.height.setValue(12.0)
        dlg.flange.setValue(6.0)
        dlg.thick.setValue(3.0)
        assert dlg.buttonBox.button(QDialogButtonBox.Ok).isEnabled(), \
            f"OK waits on a drawing that is there: {dlg.info.text()}"
        assert "2 cutters" in dlg.info.text(), f"the dialog miscounts: {dlg.info.text()}"
        assert "104.4" in dlg.info.text(), f"the dialog miscounts: {dlg.info.text()}"
        return QDialog.Accepted

    monkeypatch.setattr(app_module.QDialog, "exec", fill)
    win.cookie_cutter_dialog()
    wait_for_groups(win)
    assert len(win.doc.groups) == 1, "the drawing did not become a cutter"
    lo, hi = np.asarray(win.doc.groups[0].bbox(), dtype=float)
    assert hi[0] - lo[0] == pytest.approx(104.4, abs=ABS), "not as wide as the dialog said"
    assert hi[1] - lo[1] == pytest.approx(50.4, abs=ABS), "the drawing lost its shape"
    assert hi[2] - lo[2] == pytest.approx(12.0, abs=ABS) and lo[2] == pytest.approx(0.0, abs=1e-6)
    settings = app_module.Settings()
    assert settings.cookie_cutter_source == "file" and settings.cookie_cutter_file == path
    assert (settings.cookie_cutter_width, settings.cookie_cutter_wall,
            settings.cookie_cutter_height) == (90.0, 1.2, 12.0)
    assert (settings.cookie_cutter_flange, settings.cookie_cutter_flange_thickness) == (6.0, 3.0)
    assert settings.cookie_cutter_edge is True and settings.cookie_cutter_stamp is False


def test_the_dialog_keeps_the_values_of_the_last_cutter(win):
    """The dialog opens on what was picked last time, the way every part of Draw > Parts does."""
    from volura.part_dialogs import CookieCutterDialog
    for key, value in (("cookie_cutter_source", "file"), ("cookie_cutter_text", "VOLURA"),
                       ("cookie_cutter_bold", True), ("cookie_cutter_width", 120.0),
                       ("cookie_cutter_wall", 2.4), ("cookie_cutter_height", 30.0),
                       ("cookie_cutter_flange", 8.0), ("cookie_cutter_flange_thickness", 4.0),
                       ("cookie_cutter_edge", False), ("cookie_cutter_stamp", True)):
        setattr(win.settings, key, value)
    dlg = CookieCutterDialog(win)
    try:
        assert dlg.source.currentData() == "file"
        assert dlg.text.toPlainText() == "VOLURA", "the word of the last cutter is gone"
        assert dlg.bold.isChecked(), "bold is not remembered"
        assert (dlg.width.value(), dlg.wall.value(), dlg.height.value()) == (120.0, 2.4, 30.0)
        assert (dlg.flange.value(), dlg.thick.value()) == (8.0, 4.0)
        assert not dlg.edge.isChecked(), "the sharp edge is not remembered"
        assert dlg.stamp.isChecked(), "the stamp is not remembered"
        assert not dlg.drawing_row.isHidden(), "a drawing has no row to type its name in"
        assert dlg.text.isHidden() and dlg.family.isHidden(), "a drawing shows the fields of a word"
    finally:
        dlg.deleteLater()


def test_ok_waits_until_the_source_has_a_closed_outline(win, tmp_path):
    """A source that offers nothing - a drawing that is not there, a word of nothing - is said in
    the line under the fields and takes OK away; a source that does offer something gives it back."""
    from volura.part_dialogs import CookieCutterDialog
    need_fonts()
    path = drawing(tmp_path)
    dlg = CookieCutterDialog(win)

    def ok() -> bool:
        return dlg.buttonBox.button(QDialogButtonBox.Ok).isEnabled()

    try:
        dlg.source.setCurrentIndex(dlg.source.findData("file"))
        dlg.file.setText("")
        assert not ok(), "OK is on without a drawing"
        assert "SVG" in dlg.info.text(), dlg.info.text()
        dlg.file.setText(os.path.join(str(tmp_path), "nope.svg"))
        assert not ok(), "OK is on for a drawing that is not there"
        dlg.file.setText(path)
        assert ok(), f"OK waits on a drawing that is there: {dlg.info.text()}"
        assert "2 cutters" in dlg.info.text(), dlg.info.text()
        dlg.source.setCurrentIndex(dlg.source.findData("text"))
        assert dlg.drawing_row.isHidden(), "a word still shows the row of the drawing"
        dlg.text.setPlainText("Volura")
        assert ok(), dlg.info.text()
        assert "79.6" in dlg.info.text() and "mm on the bed" in dlg.info.text(), dlg.info.text()
        dlg.text.setPlainText("\n")
        assert not ok(), "OK is on for a word that draws nothing"
    finally:
        dlg.deleteLater()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))

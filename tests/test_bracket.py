"""Corner brackets: a printed L-shaped angle bracket with screw holes in both legs.

`bracket.bracket` gives the bracket as one closed volume that lies on its side on `z = 0`: two
legs of `thickness` millimetres, `leg_a` and `leg_b` long from the outside corner and `width`
wide, with the inner corner rounded to `fillet` and `holes` teardrop holes per leg through the
thickness, countersunk on the inside when asked, because the screws are driven in from the inside
of the L. Two triangular ribs stand in the inner corner at both ends of the width when `gussets`
is on; the underside of the upper one rises at 45 degrees.

These tests weigh the part against that: it is one piece that lies on the bed and is as big as its
two legs, the fillet adds the plastic it promises, every hole is open through its leg where
`sizes` says it is and leaves the ribs and the fillet whole, and nothing hangs over steeper than
45 degrees, because the bracket prints without support.

Run with:  python -m pytest tests/test_bracket.py
"""
import math
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import manifold3d as mf  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from volura import bracket, geom, threads  # noqa: E402

BASE = dict(leg_a=60.0, leg_b=50.0)
#: the fillet and the holes are polygons, so a part is at most this far off
ABS = 0.05


def part(**kw) -> mf.Manifold:
    return bracket.bracket(**dict(BASE, **kw))


def numbers(**kw) -> dict:
    return bracket.sizes(**dict(BASE, **kw))


def bbox(m):
    """The bounding box of a manifold as (low, high)."""
    b = np.asarray(m.bounding_box(), dtype=float)
    return b[:3], b[3:]


def head_of(screw, sink) -> float:
    """The screw head that sits on a printed hole of `screw`: the countersunk one, or the other."""
    return threads.SCREW_HOLES[screw][3 if sink else 1] + threads.DEFAULT_HOLE_CLEARANCE


def rod(along, across, length, x, y, z) -> mf.Manifold:
    """A square rod `across` wide and `length` long along `along`, centred on (x, y, z)."""
    d = {"x": [length, across, across], "y": [across, length, across],
         "z": [across, across, length]}[along]
    return mf.Manifold.cube(d, True).translate((x, y, z))


def tablet(across, x, y, z) -> mf.Manifold:
    """A thin tablet `across` wide and 0.6 high, `0.2` deep in y, centred on (x, y, z)."""
    return mf.Manifold.cube([across, 0.2, 0.6], True).translate((x, y, z))


def head_disc(head, x, y, z) -> mf.Manifold:
    """The circle a screw head covers on a face: `head` across, 0.2 deep in -y, ending at `y`."""
    return (mf.Manifold.cylinder(0.2, 0.5 * head, 0.5 * head, 64).rotate((-90.0, 0.0, 0.0))
            .translate((x, y - 0.2, z)))


def touches(m, probe) -> bool:
    """Whether a probe runs into the part anywhere."""
    return (m ^ probe).volume() > 1e-9


def rib_room(s) -> mf.Manifold:
    """The whole space a gusset takes: the corner triangle, over the whole width."""
    t, g = s["thickness"], s["gusset"]
    return mf.Manifold.extrude(mf.CrossSection([[(t, t), (t + g, t), (t, t + g)]]), s["width"])


def fillet_room(s) -> mf.Manifold:
    """The whole space an inner fillet takes: the square over the corner, over the whole width."""
    t, r = s["thickness"], s["fillet"]
    return mf.Manifold.extrude(mf.CrossSection([[(t, t), (t + r, t), (t + r, t + r),
                                                 (t, t + r)]]), s["width"])


def corner_room(s) -> mf.Manifold:
    """The leg the fillet and the gussets stand on, from the inner face out to the outside."""
    t, w = s["thickness"], s["width"]
    return mf.Manifold.cube([max(s["gusset"], s["fillet"]), t, w]).translate((t, 0.0, 0.0))


def triangles(m):
    """The unit normals, the centres and the areas of all triangles, pointing out of the part."""
    v, t = geom.mesh_arrays(m)
    a, b, c = v[t[:, 0]], v[t[:, 1]], v[t[:, 2]]
    n = np.cross(b - a, c - a)
    area = 0.5 * np.linalg.norm(n, axis=1)
    return n / np.maximum(2.0 * area[:, None], 1e-30), (a + b + c) / 3.0, area


# ------------------------------------------------------------------ the part

@pytest.mark.parametrize("leg_a, leg_b", ((15.0, 15.0), (60.0, 40.0), (150.0, 90.0)))
@pytest.mark.parametrize("ribs", (True, False))
def test_every_bracket_is_one_piece_that_lies_on_its_side(leg_a, leg_b, ribs):
    kw = dict(leg_a=leg_a, leg_b=leg_b, gussets=ribs)
    s, m = numbers(**kw), part(**kw)
    assert m.status() == mf.Error.NoError and not m.is_empty()
    assert len(m.decompose()) == 1, "the bracket fell apart"
    lo, hi = bbox(m)
    assert np.allclose(lo, 0.0), "the outside corner does not stand at the origin"
    assert hi[0] == pytest.approx(s["leg_a"], abs=ABS), "leg A is not as long as asked"
    assert hi[1] == pytest.approx(s["leg_b"], abs=ABS), "leg B is not as long as asked"
    assert hi[2] == pytest.approx(s["width"], abs=ABS), "the bend is not as wide as asked"
    assert lo[2] == pytest.approx(0.0), "the bracket does not lie on the bed"
    assert m.genus() == len(s["holes_a"]) + len(s["holes_b"]), "a hole is not open through"


@pytest.mark.parametrize("holes, screw", ((0, "M4"), (3, "M3"), (2, "M6")))
@pytest.mark.parametrize("ribs, sink", ((True, True), (False, False)))
def test_the_bracket_is_valid_whatever_it_is_asked_for(holes, screw, ribs, sink):
    m = part(holes=holes, screw=screw, gussets=ribs, countersink=sink)
    assert m.status() == mf.Error.NoError and not m.is_empty()
    assert len(m.decompose()) == 1, "the bracket fell apart"


# ------------------------------------------------------------------ the bend

@pytest.mark.parametrize("fillet", (0.0, 1.0, 4.0, 12.0))
def test_the_inner_fillet_adds_the_plastic_of_a_rounded_corner(fillet):
    """The two legs make a certain area; a fillet of r adds the corner minus a quarter disc."""
    kw = dict(holes=0, gussets=False, fillet=fillet)
    s = numbers(**kw)
    legs = s["thickness"] * (s["leg_a"] + s["leg_b"] - s["thickness"])
    sharp = part(**dict(kw, fillet=0.0)).slice(0.5 * s["width"]).area()
    round_ = part(**kw).slice(0.5 * s["width"]).area()
    assert sharp == pytest.approx(legs, rel=0.005), "the two legs do not weigh what they should"
    assert round_ - sharp == pytest.approx(s["fillet"] ** 2 * (1.0 - math.pi / 4.0),
                                           rel=0.02, abs=1e-3), "the corner is not rounded to r"


# ------------------------------------------------------------------ the holes

@pytest.mark.parametrize("screw", bracket.SCREWS)
@pytest.mark.parametrize("sink", (True, False))
def test_every_hole_is_open_through_the_leg_where_sizes_puts_it(screw, sink):
    kw = dict(holes=3, screw=screw, countersink=sink, gussets=False, fillet=0.0)
    s, m = numbers(**kw), part(**kw)
    t, z = s["thickness"], 0.5 * s["width"]
    assert len(s["holes_a"]) == len(s["holes_b"]) == 3, "these legs are long enough for three"
    for x in s["holes_a"]:
        assert not touches(m, rod("y", 0.6 * s["hole_d"], t + 4.0, x, 0.5 * t, z)), \
            f"there is no hole in leg A at {x}"
        assert touches(m, rod("y", 0.4, t + 4.0, x - 0.5 * s["hole_d"] - 0.3, 0.5 * t, z)), \
            f"the hole in leg A is wider than {s['hole_d']}"
    for y in s["holes_b"]:
        assert not touches(m, rod("x", 0.6 * s["hole_d"], t + 4.0, 0.5 * t, y, z)), \
            f"there is no hole in leg B at {y}"


@pytest.mark.parametrize("sink", (True, False))
def test_a_countersunk_hole_opens_wider_on_the_inside(sink):
    kw = dict(holes=1, screw="M4", countersink=sink)
    s, m = numbers(**kw), part(**kw)
    t, x, z = s["thickness"], s["holes_a"][0], 0.5 * s["width"]
    inside = head_of("M4", sink) if sink else s["hole_d"]
    assert not touches(m, tablet(inside - 0.5, x, t - 0.1, z)), "the hole is not open inside"
    assert touches(m, tablet(inside + 0.5, x, t - 0.1, z)), "the inside opens wider than the head"
    assert not touches(m, tablet(s["hole_d"] - 0.5, x, 0.1, z)), "the hole is blocked outside"
    assert touches(m, tablet(s["hole_d"] + 0.5, x, 0.1, z)), "the hole is not a straight hole"
    assert (not touches(m, tablet(s["hole_d"] + 0.5, x, t - 0.1, z))) == sink, \
        "the room for the head is not there, or not only there"


@pytest.mark.parametrize("ribs", (True, False))
@pytest.mark.parametrize("fillet", (0.0, 4.0))
def test_the_holes_leave_the_gussets_and_the_fillet_whole(ribs, fillet):
    kw = dict(holes=3, gussets=ribs, fillet=fillet)
    s, m = numbers(**kw), part(**kw)
    whole = part(**dict(kw, holes=0))
    for room in ([rib_room(s)] if ribs else []) + ([fillet_room(s)] if fillet else []):
        assert (m ^ room).volume() == pytest.approx((whole ^ room).volume(), rel=1e-6, abs=1e-6), \
            "a hole cuts into the plastic that makes the bend strong"
    for x in s["holes_a"]:
        assert x - 0.5 * s["hole_d"] >= s["thickness"] + max(s["gusset"], s["fillet"]), \
            "a hole starts before the corner is behind you"


@pytest.mark.parametrize("ribs, fillet", ((False, 12.0), (True, 0.0)))
def test_the_head_leaves_the_plastic_of_the_corner_whole(ribs, fillet):
    """The head sits on the inner face, right where the corner plastic begins, so a whole head
    has to fit past it: the widest screw of the book may not undermine a fillet or a rib."""
    kw = dict(holes=3, screw="M6", gussets=ribs, fillet=fillet)
    s, m = numbers(**kw), part(**kw)
    whole = part(**dict(kw, holes=0))
    head, room = head_of(s["screw"], s["countersink"]), corner_room(s)
    edge = s["thickness"] + max(s["gusset"], s["fillet"]) + bracket.ROOM
    assert s["holes_a"] and s["holes_b"], "these legs are long enough for a hole on each"
    for places in (s["holes_a"], s["holes_b"]):
        assert min(places) - 0.5 * head >= edge - 1e-9, "the head reaches into the corner"
    for x in s["holes_a"]:
        assert not touches(head_disc(head, x, s["thickness"], 0.5 * s["width"]), room), \
            "the head is cut out of the corner plastic"
    assert ((whole - m) ^ room).volume() == pytest.approx(0.0, abs=1e-6), \
        "drilling takes plastic away from under the corner"


def test_holes_are_spread_evenly_over_the_free_part_of_a_leg():
    kw = dict(holes=3, leg_a=150.0, leg_b=150.0, gussets=False, fillet=6.0)
    s = numbers(**kw)
    for places in (s["holes_a"], s["holes_b"]):
        steps = np.diff(places)
        assert len(places) == 3 and np.allclose(steps, steps[0]), "the holes are not spread evenly"
        assert places[0] - 0.5 * s["hole_d"] > 4.0 + 6.0, "a hole stands in the fillet"
        assert places[-1] + 0.5 * head_of("M4", True) <= 150.0 + 1e-9, \
            "a head does not fit before the free end"


# ------------------------------------------------------------------ printing on its side

@pytest.mark.parametrize("screw", bracket.SCREWS)
@pytest.mark.parametrize("ribs, sink", ((True, True), (True, False), (False, True)))
def test_nothing_hangs_over_steeper_than_45_degrees(screw, ribs, sink):
    """The L lies on the bed: the teardrops roof the holes and the upper rib rises at 45 degrees."""
    m = part(holes=3, screw=screw, gussets=ribs, countersink=sink)
    n, c, area = triangles(m)
    keep = area > 1e-9
    assert (n[keep][c[keep][:, 2] < 1e-9][:, 2] < -0.99).all(), \
        "the triangles do not face out of the part"
    down = keep & (n[:, 2] < -np.sin(np.pi / 4.0) - 0.01) & (c[:, 2] > 1e-6)
    assert not down.any(), "a face overhangs steeper than 45 degrees, so it needs support"


def test_the_lower_rib_stands_on_the_bed_and_the_upper_one_rises_from_the_inner_faces():
    s, m = numbers(holes=0), part(holes=0)
    t, w, g = s["thickness"], s["width"], s["gusset"]
    q = min(t, 0.5 * w)
    foot = mf.Manifold.extrude(mf.CrossSection([[(t, t), (t + g, t), (t, t + g)]]), q - 0.2)
    assert (m ^ foot).volume() == pytest.approx(foot.volume(), rel=1e-6), \
        "the lower rib does not stand whole on the bed"
    assert not touches(m, rod("z", 0.4, q, t + 0.5 * g, t + 0.5 * g, w - 0.5 * q)), \
        "the upper rib still hangs over the empty corner"
    assert touches(m, rod("z", 0.4, q, t + 0.2, t + 0.5 * g, w - 0.5 * q)), \
        "the upper rib does not stand against the inner face of leg A"
    assert touches(m, rod("z", 0.4, q, t + 0.5 * g, t + 0.2, w - 0.5 * q)), \
        "the upper rib does not stand against the inner face of leg B"


# ------------------------------------------------------------------ the numbers

def test_sizes_reports_the_bracket_and_its_holes():
    s = numbers(leg_a=80.0, leg_b=50.0, width=30.0, thickness=6.0, holes=3, screw="M5",
                countersink=False, gussets=True, fillet=10.0)
    assert s["leg_a"] == 80.0 and s["leg_b"] == 50.0 and s["width"] == 30.0
    assert s["thickness"] == 6.0 and s["holes"] == 3 and s["screw"] == "M5"
    assert s["countersink"] is False and s["gussets"] is True and s["fillet"] == 10.0
    assert s["hole_d"] == pytest.approx(threads.SCREW_HOLES["M5"][0] +
                                        threads.DEFAULT_HOLE_CLEARANCE)
    assert s["gusset"] == pytest.approx(0.4 * 50.0)
    assert len(s["holes_a"]) == 3 and len(s["holes_b"]) == 2, "the short leg gets the same holes"
    assert set(s) == {"leg_a", "leg_b", "width", "thickness", "holes", "screw", "countersink",
                      "gussets", "fillet", "gusset", "hole_d", "holes_a", "holes_b"}


def test_the_default_bracket_gives_every_leg_the_holes_it_asks():
    s = bracket.sizes()
    assert s["holes"] == 2, "the default asks two holes in a leg"
    assert len(s["holes_a"]) == 2 and len(s["holes_b"]) == 2, \
        "the default legs are too short for the two holes they ask"


def test_the_numbers_are_clamped_to_what_prints():
    small = numbers(leg_a=1.0, leg_b=1.0, width=1.0, thickness=1.0, holes=-4, fillet=-2.0)
    assert (small["leg_a"], small["leg_b"], small["width"], small["thickness"]) == (
        bracket.LEG_MIN, bracket.LEG_MIN, bracket.WIDTH_MIN, bracket.THICK_MIN)
    assert small["holes"] == 0 and small["fillet"] == 0.0 and small["holes_a"] == []
    big = numbers(leg_a=999.0, leg_b=999.0, width=999.0, thickness=999.0, holes=99, fillet=999.0)
    assert (big["leg_a"], big["leg_b"], big["width"], big["thickness"]) == (
        bracket.LEG_MAX, bracket.LEG_MAX, bracket.WIDTH_MAX, bracket.THICK_MAX)
    assert big["holes"] == bracket.HOLES_MAX
    assert big["fillet"] == pytest.approx(bracket.FILLET_MAX * bracket.THICK_MAX)
    lo, hi = bbox(part(leg_a=1.0, leg_b=1.0, width=1.0, thickness=1.0, holes=0))
    assert hi[0] == pytest.approx(bracket.LEG_MIN, abs=ABS), "the smallest bracket is not the leg"


def test_a_leg_that_is_too_short_gets_fewer_holes():
    tight = numbers(holes=3, leg_a=15.0, leg_b=15.0, gussets=False, fillet=0.0)
    assert 0 < len(tight["holes_a"]) < 3, "a short leg was not given fewer holes"
    short = numbers(holes=3, leg_a=15.0, leg_b=15.0)
    assert short["holes_a"] == [] and short["holes_b"] == [], "a ribbed short leg still has holes"
    assert len(numbers(holes=3, leg_a=150.0, leg_b=15.0)["holes_a"]) == 3


def test_a_screw_that_does_not_fit_across_the_width_becomes_a_smaller_one():
    assert numbers(width=8.0, screw="M6")["screw"] == "M3"
    assert numbers(width=20.0, screw="M6")["screw"] == "M6"
    narrow = numbers(width=8.0, screw="M6", holes=3)
    m = part(width=8.0, screw="M6", holes=3)
    assert m.status() == mf.Error.NoError and len(m.decompose()) == 1, "the leg was cut in two"
    assert narrow["holes_a"], "a leg wide enough for the smallest screw has no holes"


@pytest.mark.parametrize("bad", (float("nan"), float("inf"), None, "wide"))
def test_a_number_that_is_no_number_is_refused(bad):
    for key in ("leg_a", "leg_b", "width", "thickness", "holes", "fillet"):
        with pytest.raises(ValueError):
            bracket.bracket(**dict(BASE, **{key: bad}))
        with pytest.raises(ValueError):
            bracket.sizes(**dict(BASE, **{key: bad}))


@pytest.mark.parametrize("screw", ("M12", "8", "", None, "m4"))
def test_a_screw_that_is_not_in_the_book_is_refused(screw):
    with pytest.raises(ValueError):
        bracket.bracket(**dict(BASE, screw=screw))
    with pytest.raises(ValueError):
        bracket.sizes(**dict(BASE, screw=screw))


# ------------------------------------------------------------------ the interface

def test_the_dialog_builds_a_bracket_and_remembers_it(monkeypatch, tmp_path):
    import time
    from PySide6.QtCore import QCoreApplication
    from PySide6.QtWidgets import QApplication, QDialog
    app = QApplication.instance() or QApplication(sys.argv[:1])
    import volura.app as app_module
    monkeypatch.setattr(app_module.errors, "log_dir", lambda: str(tmp_path))
    w = app_module.MainWindow()
    w.doc.dirty = False
    QCoreApplication.processEvents()

    def fill(dlg):
        dlg.leg_a.setValue(80.0)
        dlg.leg_b.setValue(60.0)
        dlg.holes.setValue(3)
        assert dlg.gussets.isChecked(), "the gussets are made along with the bracket"
        assert "4.7" in dlg.info.text(), "the dialog does not show the hole it drills"
        return QDialog.Accepted

    monkeypatch.setattr(app_module.QDialog, "exec", fill)
    try:
        w.bracket_dialog()
        deadline = time.monotonic() + 15.0
        while not w.doc.groups and time.monotonic() < deadline:
            app.processEvents()
            time.sleep(0.005)
        assert len(w.doc.groups) == 1, "no bracket in the model"
        lo, hi = np.asarray(w.doc.groups[0].bbox(), dtype=float)
        s = bracket.sizes(leg_a=80.0, leg_b=60.0)
        assert lo[2] == pytest.approx(0.0, abs=1e-6), "the bracket does not lie on the bed"
        assert hi[2] - lo[2] == pytest.approx(s["width"], abs=1e-4)
        assert hi[0] - lo[0] == pytest.approx(80.0, abs=1e-4), "leg A is not as long as asked"
        settings = app_module.Settings()
        assert (settings.bracket_leg_a, settings.bracket_leg_b, settings.bracket_holes) == (
            pytest.approx(80.0), pytest.approx(60.0), 3)
        labels = [a.text().replace("&", "").replace("…", "") for a in w.flat_actions(w.menuBar())]
        assert labels.index("Corner Bracket") == labels.index("Desk Grommet") + 1
    finally:
        w.settings.bracket_leg_a, w.settings.bracket_leg_b, w.settings.bracket_holes = (
            40.0, 40.0, 2)
        w.settings.save()
        w.doc.dirty = False
        w.close()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))

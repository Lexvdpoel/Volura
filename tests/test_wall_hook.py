"""Wall hooks: a J that is screwed to a wall, for a coat, a bag, headphones or a cable.

`wall_hook.wall_hook` gives the hook as one closed volume that lies on its side on `z = 0`: a back
plate of `plate` millimetres against the wall on `x = 0`, an arm that leaves its lower end along
+X and reaches `reach` millimetres out, and a tip that rises `rise` millimetres at the end of the
arm, all `width` millimetres wide and with both inside bends rounded. The plate gets 1 or 2
teardrop holes through its thickness, countersunk on the front face when asked, because the screws
are driven in from away from the wall.

These tests weigh the part against that: it is one piece that lies on the bed and is as big as its
reach and its plate, the back of the plate is flat and the whole plate, a rod lies on the arm
between the plate and the tip without touching anything, every hole is open through the plate where
`sizes` puts it and its cone is on the front face, the holes leave the bend whole, and nothing
hangs over steeper than 45 degrees, because the hook prints on its side without support.

Run with:  python -m pytest tests/test_wall_hook.py
"""
import math
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import manifold3d as mf  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from volura import geom, threads, wall_hook  # noqa: E402

BASE = dict(reach=30.0, rise=15.0)
#: the bends, the holes and the chamfers are polygons, so a part is at most this far off
ABS = 0.05


def part(**kw) -> mf.Manifold:
    return wall_hook.wall_hook(**dict(BASE, **kw))


def numbers(**kw) -> dict:
    return wall_hook.sizes(**dict(BASE, **kw))


def bbox(m):
    """The bounding box of a manifold as (low, high)."""
    b = np.asarray(m.bounding_box(), dtype=float)
    return b[:3], b[3:]


def touches(m, probe) -> bool:
    """Whether a probe runs into the part anywhere.

    Two solids that only touch leave a sliver of a few ten-millionths of a mm³ behind, so that is
    not a touch; a rod that does not fit is off by whole millimetres.
    """
    return (m ^ probe).volume() > 1e-6


def rod(m, d, x, y) -> mf.Manifold:
    """A rod `d` across along the width of the hook, with its axis through (`x`, `y`)."""
    return mf.Manifold.cylinder(m.bounding_box()[5] + 4.0, 0.5 * d, 0.5 * d, 64) \
        .translate((x, y, -2.0))


def disc(m, d, y, x0) -> mf.Manifold:
    """A disc `d` across and 0.2 thick along X, from `x0`, centred on the middle of the width."""
    return (mf.Manifold.cylinder(0.2, 0.5 * d, 0.5 * d, 48).rotate((0.0, 90.0, 0.0))
            .translate((x0, y, 0.5 * m.bounding_box()[5])))


def slab(x, y, z) -> mf.Manifold:
    """A tablet 0.3 thick along X that starts at `x`, `y` deep and `z` high, with its corner on
    the origin."""
    return mf.Manifold.cube([0.3, y, z]).translate((x, 0.0, 0.0))


def triangles(m):
    """The unit normals, the centres and the areas of all triangles, pointing out of the part."""
    v, t = geom.mesh_arrays(m)
    a, b, c = v[t[:, 0]], v[t[:, 1]], v[t[:, 2]]
    n = np.cross(b - a, c - a)
    area = 0.5 * np.linalg.norm(n, axis=1)
    return n / np.maximum(2.0 * area[:, None], 1e-30), (a + b + c) / 3.0, area


# ------------------------------------------------------------------ the part

@pytest.mark.parametrize("reach, rise", ((10.0, 5.0), (30.0, 15.0), (80.0, 50.0)))
@pytest.mark.parametrize("thickness, plate", ((3.0, 20.0), (5.0, 50.0), (10.0, 150.0)))
def test_the_hook_is_one_piece_that_lies_on_its_side(reach, rise, thickness, plate):
    kw = dict(reach=reach, rise=rise, thickness=thickness, plate=plate)
    s, m = numbers(**kw), part(**kw)
    assert m.status() == mf.Error.NoError and not m.is_empty()
    assert len(m.decompose()) == 1, "the hook fell apart"
    lo, hi = bbox(m)
    assert np.allclose(lo, 0.0), "the wall face does not stand on the plane x = 0"
    assert hi[0] == pytest.approx(s["reach"], abs=ABS), "the arm does not reach as far as asked"
    assert hi[1] == pytest.approx(max(s["plate"], s["rise"]), abs=ABS), "the hook is not as tall"
    assert hi[2] == pytest.approx(s["width"], abs=ABS), "the hook does not stand as high as wide"
    assert m.genus() == len(s["holes_y"]), "a hole is not open through"


@pytest.mark.parametrize("screw", wall_hook.SCREWS)
@pytest.mark.parametrize("countersink, holes", ((True, 1), (False, 2)))
def test_the_hook_is_valid_whatever_it_is_asked_for(screw, countersink, holes):
    m = part(width=8.0, thickness=10.0, plate=150.0, screw=screw, countersink=countersink,
             holes=holes)
    assert m.status() == mf.Error.NoError and not m.is_empty()
    assert len(m.decompose()) == 1, "the hook fell apart"


# ------------------------------------------------------------------ the plate

def test_the_back_of_the_plate_is_flat_and_the_whole_plate():
    """The face that lies against the wall is the plate itself: nothing sticks past it, and it is
    only missing what the chamfer and the screw holes take from it."""
    s, m = numbers(), part()
    assert not touches(m, slab(-0.3, s["plate"] + 1.0, s["width"] + 1.0)), \
        "something reaches past the wall"
    n, c, area = triangles(m)
    back = (area > 1e-9) & (n[:, 0] < -0.999) & (np.abs(c[:, 0]) < 1e-6)
    r = 0.5 * s["hole_d"]
    want = s["plate"] * (s["width"] - wall_hook.CHAMFER) - len(s["holes_y"]) * r * r \
        * (0.75 * math.pi + 0.5)
    assert area[back].sum() == pytest.approx(want, rel=0.05), "the back of the plate is not a plate"


@pytest.mark.parametrize("holes", (1, 2))
def test_the_holes_stay_in_the_plate_above_the_arm(holes):
    """The plate is never shorter than the bend of the arm plus a head and room for every hole,
    and a head sits flat on the front face with plastic left beside it."""
    kw = dict(holes=holes, screw="M5", plate=1.0)
    s, m = numbers(**kw), part(**kw)
    assert s["plate"] == pytest.approx(2.0 * s["thickness"] + 2.0 * wall_hook.HEAD_EDGE
                                   + holes * s["head"] + (holes - 1) * 4.0)
    assert len(s["holes_y"]) == holes
    if holes > 1:
        step = s["plate"] - s["root"] - s["head"] - 2.0 * wall_hook.HEAD_EDGE
        assert np.allclose(np.diff(s["holes_y"]), step), "the holes are not spread evenly"
    for y in s["holes_y"]:
        assert y - 0.5 * s["head"] >= s["root"] + wall_hook.HEAD_EDGE - 1e-9, "a head cuts into the bend"
        assert y + 0.5 * s["head"] <= s["plate"] - wall_hook.HEAD_EDGE + 1e-9, "a head hangs over"
    assert s["holes_y"][0] + s["holes_y"][-1] == pytest.approx(s["root"] + s["plate"])
    bend = mf.Manifold.cube([s["thickness"], s["root"],
                             s["width"] - 2.0 * wall_hook.CHAMFER]) \
        .translate((0.0, 0.0, wall_hook.CHAMFER))
    assert (m ^ bend).volume() == pytest.approx(bend.volume(), rel=1e-3), \
        "a hole cuts the plastic that makes the bend strong"


# ------------------------------------------------------------------ what hangs on it

@pytest.mark.parametrize("thickness", (3.0, 5.0))
def test_a_rod_lies_on_the_arm_between_plate_and_tip(thickness):
    """A rod of 6 mm lies on the arm in the middle of its flat part, with the plate behind it and
    the tip in front of it, and touches nothing there."""
    kw = dict(thickness=thickness)
    s, m = numbers(**kw), part(**kw)
    x = 0.5 * (2.0 * s["thickness"] + s["reach"] - s["thickness"] - s["bend"])
    assert not touches(m, rod(m, 6.0, x, s["thickness"] + 3.0)), "no rod lies on the arm"
    assert touches(m, rod(m, 6.0, s["thickness"], s["thickness"] + 3.0)), \
        "a rod stands in the plate instead of in the hook"


def test_the_top_of_the_tip_is_round():
    """The top of the tip is a full half-round of the thickness: the corner above its outside is
    rounded away and there is plastic right under the middle of the top."""
    s, m = numbers(), part()
    r = 0.5 * s["thickness"]
    cx, cy = s["reach"] - r, s["rise"] - r
    deep = s["width"] - 2.0 * wall_hook.CHAMFER
    corner = mf.Manifold.cube([0.3, 0.3, deep]).translate((cx + 0.8 * r, cy + 0.8 * r,
                                                           wall_hook.CHAMFER))
    assert not touches(m, corner), "the corner above the tip is not rounded away"
    middle = mf.Manifold.cube([0.3, 0.4, deep]).translate((cx - 0.15, cy + r - 0.5,
                                                           wall_hook.CHAMFER))
    assert touches(m, middle), "there is nothing under the middle of the tip"


# ------------------------------------------------------------------ the screw holes

@pytest.mark.parametrize("screw", wall_hook.SCREWS)
@pytest.mark.parametrize("countersink", (True, False))
def test_every_hole_is_open_through_the_plate_where_sizes_puts_it(screw, countersink):
    kw = dict(holes=2, screw=screw, countersink=countersink)
    s, m = numbers(**kw), part(**kw)
    t = s["thickness"]
    assert len(s["holes_y"]) == 2, "this plate is tall enough for two holes"
    for y in s["holes_y"]:
        pin = mf.Manifold.cylinder(t + 4.0, 0.3 * s["hole_d"], 0.3 * s["hole_d"], 32) \
            .rotate((0.0, 90.0, 0.0)).translate((-2.0, y, 0.5 * s["width"]))
        assert not touches(m, pin), f"there is no hole in the plate at {y}"
        assert touches(m, disc(m, s["hole_d"] + 0.5, y, 0.0)), \
            f"the hole in the plate is wider than {s['hole_d']}"


@pytest.mark.parametrize("countersink", (True, False))
def test_a_countersunk_hole_opens_wider_on_the_front_face(countersink):
    """The screws are driven in from away from the wall, so the room for a head is on the front
    face of the plate and the back that lies against the wall stays flat."""
    kw = dict(holes=1, screw="M4", countersink=countersink)
    s, m = numbers(**kw), part(**kw)
    t, y = s["thickness"], s["holes_y"][0]
    assert not touches(m, disc(m, s["hole_d"] - 0.5, y, 0.0)), "the hole is blocked at the wall"
    assert touches(m, disc(m, s["hole_d"] + 0.5, y, 0.0)), "the hole is not a straight hole"
    assert (not touches(m, disc(m, s["head"] - 1.0, y, t - 0.2))) == countersink, \
        "the room for the head is not there, or not only there"
    assert touches(m, disc(m, s["head"] - 1.0, y, 0.0)), "the plate is eaten through from behind"


# ------------------------------------------------------------------ printing on its side

@pytest.mark.parametrize("screw", wall_hook.SCREWS)
@pytest.mark.parametrize("countersink, holes", ((True, 2), (False, 1)))
def test_nothing_hangs_over_steeper_than_45_degrees(screw, countersink, holes):
    """The J lies on the bed: the teardrops roof the holes, the tip is a half-round and the top
    edges are chamfered, so no face hangs over steeper than 45 degrees."""
    m = part(width=8.0, thickness=10.0, plate=150.0, screw=screw, countersink=countersink,
             holes=holes)
    n, c, area = triangles(m)
    keep = area > 1e-9
    assert (n[keep][c[keep][:, 2] < 1e-9][:, 2] < -0.99).all(), \
        "the triangles do not face out of the part"
    down = keep & (n[:, 2] < -np.sin(np.pi / 4.0) - 0.01) & (c[:, 2] > 1e-6)
    assert not down.any(), "a face overhangs steeper than 45 degrees, so it needs support"


def test_every_outside_edge_on_top_is_chamfered():
    """The top is smaller than the footprint by exactly what the 45 degree edges lean inwards,
    over the whole outline."""
    s, m = numbers(), part()
    n, c, area = triangles(m)
    top = (area > 1e-9) & (n[:, 2] > 0.999) & (c[:, 2] > s["height"] - 1e-6)
    slant = (area > 1e-9) & (np.abs(n[:, 2] - math.sqrt(0.5)) < 1e-6)
    slant &= c[:, 2] > s["height"] - wall_hook.CHAMFER
    assert slant.any(), "there is no 45 degree edge on top"
    band = m.slice(0.5).area() - area[top].sum()
    assert (area[slant] * n[slant][:, 2]).sum() == pytest.approx(band, rel=1e-6), \
        "the chamfer does not cover the whole outline"


# ------------------------------------------------------------------ the numbers

def test_sizes_reports_the_hook_and_its_holes():
    s = wall_hook.sizes(reach=45.0, rise=25.0, width=20.0, thickness=6.0, plate=90.0,
                        screw="M5", countersink=False, holes=2)
    assert s["reach"] == 45.0 and s["rise"] == 25.0 and s["width"] == 20.0
    assert s["thickness"] == 6.0 and s["plate"] == 90.0 and s["holes"] == 2
    assert s["screw"] == "M5" and s["countersink"] is False
    assert s["height"] == 20.0, "the hook does not stand as high on the bed as it is wide"
    assert s["bend"] == pytest.approx(min(0.25 * 45.0, wall_hook.BEND_MAX))
    assert s["root"] == pytest.approx(2.0 * 6.0)
    assert s["hole_d"] == pytest.approx(threads.SCREW_HOLES["M5"][0] +
                                        threads.DEFAULT_HOLE_CLEARANCE)
    assert s["head"] == pytest.approx(threads.SCREW_HOLES["M5"][1] +
                                      threads.DEFAULT_HOLE_CLEARANCE)
    assert len(s["holes_y"]) == 2
    assert set(s) == {"reach", "rise", "width", "thickness", "plate", "screw", "countersink",
                      "holes", "height", "holes_y", "hole_d", "head", "bend", "root"}


def test_the_default_hook_is_a_coat_hook():
    s = wall_hook.sizes()
    assert s["reach"] == 30.0 and s["rise"] == 15.0 and s["width"] == 15.0
    assert s["thickness"] == 5.0 and s["plate"] == 50.0 and s["holes"] == 2
    assert s["screw"] == "M4" and s["countersink"] is True
    assert len(s["holes_y"]) == 2, "the default plate is tall enough for the two holes it asks"


@pytest.mark.parametrize("holes", (1, 2))
def test_the_plate_grows_for_every_hole_it_gets(holes):
    """A plate that is too short for its holes is made taller: the bend of the arm, plus a head
    and four millimetres between two holes, with HEAD_EDGE at both ends."""
    s = numbers(plate=1.0, holes=holes, screw="M4")
    assert s["plate"] == pytest.approx(2.0 * s["thickness"] + 2.0 * wall_hook.HEAD_EDGE
                                   + holes * s["head"] + (holes - 1) * 4.0)
    assert s["plate"] > s["root"] + 0.5 * s["head"], "the plate is too short for its own hole"


def test_a_thick_hook_is_never_too_short_to_bend():
    """The reach carries both bends and a piece of straight arm between them, and a tip is never
    lower than its own bend and its own round top."""
    s = numbers(reach=10.0, rise=5.0, thickness=10.0)
    assert s["reach"] > 3.0 * s["thickness"] + s["bend"], "there is no arm left between the bends"
    assert s["rise"] >= 1.5 * s["thickness"] + s["bend"], "the tip is lower than its own bend"
    assert s["bend"] <= s["reach"] - 3.0 * s["thickness"] - wall_hook.ARM_MIN + 1e-9
    assert s["bend"] <= wall_hook.BEND_MAX + 1e-9


def test_the_numbers_are_clamped_to_what_prints():
    small = numbers(reach=1.0, rise=1.0, width=1.0, thickness=1.0, plate=1.0, holes=0)
    assert small["width"] == wall_hook.WIDTH_MIN and small["thickness"] == wall_hook.THICK_MIN
    assert small["holes"] == wall_hook.HOLES_MIN
    assert small["reach"] >= wall_hook.REACH_MIN and small["rise"] >= wall_hook.RISE_MIN
    big = numbers(reach=999.0, rise=999.0, width=999.0, thickness=999.0, plate=999.0, holes=99)
    assert big["width"] == wall_hook.WIDTH_MAX and big["thickness"] == wall_hook.THICK_MAX
    assert big["holes"] == wall_hook.HOLES_MAX and big["plate"] == wall_hook.PLATE_MAX
    assert big["reach"] <= wall_hook.REACH_MAX and big["rise"] <= wall_hook.RISE_MAX
    lo, hi = bbox(part(reach=1.0, rise=1.0, width=1.0, thickness=1.0, plate=1.0, holes=0))
    assert hi[2] == pytest.approx(wall_hook.WIDTH_MIN, abs=ABS), \
        "the smallest hook is not as high on the bed as it is wide"


def test_a_screw_that_does_not_fit_across_the_width_becomes_a_smaller_one():
    assert numbers(width=8.0, screw="M5")["screw"] == "M3"
    assert numbers(width=40.0, screw="M5")["screw"] == "M5"
    narrow = numbers(width=8.0, screw="M5", holes=2)
    m = part(width=8.0, screw="M5", holes=2)
    assert m.status() == mf.Error.NoError and len(m.decompose()) == 1, "the plate was cut in two"
    assert len(narrow["holes_y"]) == 2, "a plate wide enough for the smallest screw has no holes"


@pytest.mark.parametrize("bad", (float("nan"), float("inf"), None, "wide"))
def test_a_number_that_is_no_number_is_refused(bad):
    for key in ("reach", "rise", "width", "thickness", "plate", "holes"):
        with pytest.raises(ValueError):
            wall_hook.wall_hook(**dict(BASE, **{key: bad}))
        with pytest.raises(ValueError):
            wall_hook.sizes(**dict(BASE, **{key: bad}))


@pytest.mark.parametrize("screw", ("M12", "6", "", None, "m4"))
def test_a_screw_that_is_not_in_the_book_is_refused(screw):
    with pytest.raises(ValueError):
        wall_hook.wall_hook(**dict(BASE, screw=screw))
    with pytest.raises(ValueError):
        wall_hook.sizes(**dict(BASE, screw=screw))


# ------------------------------------------------------------------ the interface

def test_the_dialog_builds_a_hook_and_remembers_it(monkeypatch, tmp_path):
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
        dlg.reach.setValue(45.0)
        dlg.rise.setValue(25.0)
        dlg.plate.setValue(20.0)
        dlg.holes.setValue(1)
        assert dlg.countersink.isChecked(), "the head is made to sink into the plate"
        assert "4.7" in dlg.info.text(), "the dialog does not show the hole it drills"
        return QDialog.Accepted

    monkeypatch.setattr(app_module.QDialog, "exec", fill)
    try:
        w.wall_hook_dialog()
        deadline = time.monotonic() + 15.0
        while not w.doc.groups and time.monotonic() < deadline:
            app.processEvents()
            time.sleep(0.005)
        assert len(w.doc.groups) == 1, "no hook in the model"
        lo, hi = np.asarray(w.doc.groups[0].bbox(), dtype=float)
        s = wall_hook.sizes(reach=45.0, rise=25.0, plate=20.0, holes=1)
        assert lo[2] == pytest.approx(0.0, abs=1e-6), "the hook does not lie on the bed"
        assert hi[2] - lo[2] == pytest.approx(s["width"], abs=1e-4)
        assert hi[0] - lo[0] == pytest.approx(45.0, abs=1e-4), "the arm does not reach as far"
        assert hi[1] - lo[1] == pytest.approx(max(s["plate"], s["rise"]), abs=0.2)
        assert s["plate"] > 20.0, "the plate was not made taller for its hole"
        settings = app_module.Settings()
        assert (settings.wall_hook_reach, settings.wall_hook_rise, settings.wall_hook_plate,
                settings.wall_hook_holes) == (45.0, 25.0, 20.0, 1)
        labels = [a.text().replace("&", "").replace("…", "") for a in w.flat_actions(w.menuBar())]
        assert labels.index("Wall Hook") == labels.index("Memory Card Holder") + 1
    finally:
        w.settings.wall_hook_reach, w.settings.wall_hook_rise = 30.0, 15.0
        w.settings.wall_hook_plate, w.settings.wall_hook_holes = 50.0, 2
        w.settings.save()
        w.doc.dirty = False
        w.close()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))

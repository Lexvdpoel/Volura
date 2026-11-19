"""Pen cups: a round cup that stands on a desk and holds pens, with a band of windows in its wall.

`pen_cup.pen_cup` gives the cup as one closed volume on `z = 0`: a cylinder `diameter` millimetres
across and `height` high, with a wall of `wall` millimetres around a floor of `floor` millimetres
and, when `windows` is on, `columns` diamonds round its wall in rows that interleave - every diamond
a square turned 45 degrees, so its sides lean at 45 degrees and the band prints open without
support.

These tests weigh the cup against that: one volume on the bed and centred, `diameter` by `diameter`
by `height`, with one hole through the wall for every window; an inside that is empty down to the
floor and a floor that is there; an open window at the middle of every diamond and wall between two
diamonds of a row; no window above the band it is given; no face that points down by more than 46
degrees, so the cup stands on its floor without support; a whole wall when the windows are off;
every range reached by clamping and every number that is no number refused; and the dialog that
builds the cup and remembers it.

Run with:  python -m pytest tests/test_pen_cup.py
"""
import math
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import manifold3d as mf  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from PySide6.QtWidgets import QApplication  # noqa: E402

from volura import geom, pen_cup  # noqa: E402

_app = QApplication.instance() or QApplication(sys.argv[:1])

BASE = dict(diameter=75.0, height=100.0, wall=2.4, floor=2.4, windows=True, columns=12)
#: the wall is a polygon of 128 sides, so a part is at most this far off
ABS = 0.05
#: mm across a probe that feels for plastic
PROBE = 0.4
#: mm³ of sliver a boolean leaves behind where two faces of the parts coincide exactly
SLIVER = 1e-4


def part(**kw) -> mf.Manifold:
    return pen_cup.pen_cup(**dict(BASE, **kw))


def numbers(**kw) -> dict:
    return pen_cup.sizes(**dict(BASE, **kw))


def bbox(m):
    """The bounding box of a manifold as (low, high)."""
    b = np.asarray(m.bounding_box(), dtype=float)
    return b[:3], b[3:]


def overlap(m, probe) -> float:
    """How much of two solids is the same: nothing when they only touch.

    Two solids that only touch leave a sliver of a few ten-millionths of a mm³ behind, so that is
    not a touch; plastic that is really there is off by whole tenths of a millimetre.
    """
    return (m ^ probe).volume()


def post(x, y, z0, z1, r=PROBE) -> mf.Manifold:
    """A round post of radius `r` at (`x`, `y`), between the heights `z0` and `z1`."""
    return mf.Manifold.cylinder(z1 - z0, r, r, 32).translate((x, y, z0))


def radius(s) -> float:
    """mm from the axis of the cup to the outside of its wall."""
    return 0.5 * s["diameter"]


def diamond(s) -> float:
    """mm a diamond is wide and high: the part of its column it takes, across a square turned 45."""
    return pen_cup.DIAMOND_PART * (2.0 * math.pi * radius(s) / s["columns"])


def spoke(s, angle, z, r=PROBE) -> mf.Manifold:
    """A round rod of radius `r` pointing at the axis of the cup at height `z`, turned `angle`.

    It reaches from outside the wall to the axis, so it feels the wall at that angle and nothing
    else: a window there lets it through, wall stops it.
    """
    return (mf.Manifold.cylinder(radius(s) + 1.0, r, r, 32).rotate((90.0, 0.0, 0.0))
            .translate((0.0, 0.0, z)).rotate((0.0, 0.0, angle)))


def spoke_at(r, angle, z, w=PROBE) -> mf.Manifold:
    """A round rod of radius `w` across the cup, from `r` on one side to `r` on the other.

    It stays inside the wall and feels nothing, so it says the wall is not thicker than asked.
    """
    return (mf.Manifold.cylinder(2.0 * r, w, w, 32).rotate((90.0, 0.0, 0.0))
            .translate((0.0, r, z)).rotate((0.0, 0.0, angle)))


def windows(s):
    """Where every diamond of the band is: (angle round the cup, height of its middle)."""
    return [(360.0 * k / s["columns"] + (180.0 / s["columns"] if row % 2 else 0.0),
             pen_cup.BAND + 0.5 * diamond(s) + row * (0.5 * diamond(s) + pen_cup.ROW_GAP))
            for row in range(s["rows"]) for k in range(s["columns"])]


def faces(m):
    """The unit normals, the areas and the heights of the middles of all triangles."""
    v, t = geom.mesh_arrays(m)
    a, b, c = v[t[:, 0]], v[t[:, 1]], v[t[:, 2]]
    n = np.cross(b - a, c - a)
    area = 0.5 * np.linalg.norm(n, axis=1)
    return n / np.maximum(2.0 * area[:, None], 1e-30), area, v[t].mean(axis=1)[:, 2]


SIZES = (dict(),
         dict(diameter=40.0, height=50.0, wall=1.6, floor=1.6, columns=6),
         dict(diameter=120.0, height=160.0, wall=4.0, floor=4.0, columns=24))


# ------------------------------------------------------------------ the piece

@pytest.mark.parametrize("kw", SIZES + (dict(windows=False),))
def test_the_cup_is_one_volume_that_stands_on_the_bed_and_is_centred(kw):
    s, m = numbers(**kw), part(**kw)
    assert m.status() == mf.Error.NoError and not m.is_empty()
    assert len(m.decompose()) == 1, "the cup fell apart"
    assert m.genus() == s["rows"] * s["columns"], "the windows are not the only holes in the wall"
    lo, hi = bbox(m)
    assert lo[2] == pytest.approx(0.0, abs=1e-6), "the cup does not stand on the bed"
    assert hi[0] - lo[0] == pytest.approx(s["diameter"], abs=ABS), "not as wide as asked"
    assert hi[1] - lo[1] == pytest.approx(s["diameter"], abs=ABS), "not as wide as asked"
    assert hi[2] - lo[2] == pytest.approx(s["height"], abs=ABS), "the cup does not stand as high"
    assert np.allclose(lo[:2], -hi[:2], atol=ABS), "not centred in x and y"


# ------------------------------------------------------------------ the inside

@pytest.mark.parametrize("kw", SIZES)
def test_the_cup_is_hollow_down_to_its_floor(kw):
    """The inside is empty from the floor to the rim, so pens go in as deep as the cup is tall, and
    the floor of `floor` millimetres under it is the face the cup stands on."""
    s, m = numbers(**kw), part(**kw)
    assert overlap(m, post(0.0, 0.0, s["floor"] + 0.2, s["height"] - 0.2)) \
        == pytest.approx(0.0, abs=SLIVER), "there is no room for the pens"
    assert overlap(m, post(0.0, 0.0, 0.2, 0.5 * s["floor"])) > 0.0, "there is no floor"


@pytest.mark.parametrize("kw", SIZES)
def test_the_wall_is_as_thick_as_asked(kw):
    """The wall of the cup is `wall` millimetres thick all round: the inside is cut out of the
    cylinder from the floor up, so what is left of the wall is where it should be."""
    s, m = numbers(**kw), part(**kw)
    z = 0.5 * (s["floor"] + pen_cup.BAND)      # below the band, so the wall is always whole
    for angle in (0.0, 90.0, 180.0, 270.0):
        assert overlap(m, spoke(s, angle, z)) > 0.0, "the wall is not there"
        assert overlap(m, spoke_at(radius(s) - s["wall"] - 0.5 * PROBE - 0.3, angle, z)) \
            == pytest.approx(0.0, abs=SLIVER), "the wall is thicker than asked"


# ------------------------------------------------------------------ the windows

@pytest.mark.parametrize("kw", SIZES)
def test_every_diamond_is_a_window_through_the_wall(kw):
    """Every diamond of the band is cut right through the wall, and the wall between two diamonds of
    a row is left standing, so the cup is still a cup."""
    s, m = numbers(**kw), part(**kw)
    assert len(windows(s)) == s["rows"] * s["columns"], "another window was counted than is cut"
    for angle, z in windows(s):
        assert overlap(m, spoke(s, angle, z)) == pytest.approx(0.0, abs=SLIVER), \
            "a window is not open through the wall"
        assert overlap(m, spoke(s, angle + 180.0 / s["columns"], z)) > 0.0, \
            "there is no wall left between two diamonds of a row"


@pytest.mark.parametrize("kw", SIZES)
def test_the_band_leaves_the_rim_and_the_floor_whole(kw):
    """The band hangs `BAND` millimetres clear of the rim and of the floor, so the cup keeps a
    strong edge to hold pens with and a strong bottom to stand on."""
    s, m = numbers(**kw), part(**kw)
    for z in (pen_cup.BAND - 1.0, s["height"] - pen_cup.BAND + 1.0):
        for k in range(s["columns"]):
            assert overlap(m, spoke(s, 360.0 * k / s["columns"], z)) > 0.0, \
                "a window is cut outside the band"


def test_a_cup_without_windows_has_a_whole_wall():
    """Without windows the wall stands all round, from the floor to the rim, and nothing is cut
    through it: the pens are hidden and the cup is as stiff as it can be."""
    s, m = numbers(windows=False), part(windows=False)
    assert s["rows"] == 0, "a cup without windows counts rows of them"
    assert m.genus() == 0, "a cup without windows has a hole in its wall"
    for z in (pen_cup.BAND, 0.5 * s["height"], s["height"] - pen_cup.BAND):
        for k in range(2 * s["columns"]):
            assert overlap(m, spoke(s, 180.0 * k / s["columns"], z)) > 0.0, \
                "the wall is not whole"


# ------------------------------------------------------------------ printing

@pytest.mark.parametrize("kw", SIZES + (dict(windows=False),))
def test_nothing_above_the_floor_points_down_by_more_than_46_degrees(kw):
    """The lower sides of a diamond lean at 45 degrees, which a printer bridges on its own, and
    nothing else overhangs: the only face that looks straight down is the bottom of the floor on the
    bed, so the cup prints standing up without a gram of support."""
    n, area, z = faces(part(**kw))
    keep = (area > 1e-9) & (z > 0.1) & (n[:, 2] < 0.0)
    lean = np.degrees(np.arcsin(np.clip(-n[:, 2], 0.0, 1.0)))
    assert (lean <= 46.0 + 1e-6)[keep].all(), "a face points down, so the cup would need support"


# ------------------------------------------------------------------ the numbers

def test_sizes_reports_the_cup_it_makes():
    s = pen_cup.sizes(diameter=90.0, height=120.0, wall=3.0, floor=2.0, windows=True, columns=16)
    assert s["diameter"] == 90.0 and s["height"] == 120.0
    assert s["wall"] == 3.0 and s["floor"] == 2.0 and s["columns"] == 16
    assert s["windows"] is True and s["rows"] > 0
    assert s["size_x"] == pytest.approx(90.0)
    assert s["size_y"] == pytest.approx(90.0)
    assert s["size_z"] == pytest.approx(120.0)
    assert set(s) == {"diameter", "height", "wall", "floor", "windows", "columns", "rows",
                      "size_x", "size_y", "size_z"}


def test_the_default_cup_fits_a_handful_of_pens():
    s = pen_cup.sizes()
    assert s["diameter"] == 75.0 and s["height"] == 100.0
    assert s["wall"] == 2.4 and s["floor"] == 2.4
    assert s["windows"] is True and s["columns"] == 12


def test_the_numbers_are_clamped_to_what_prints():
    small = numbers(diameter=0.0, height=0.0, wall=0.0, floor=0.0, columns=0)
    assert small["diameter"] == pen_cup.DIAMETER_MIN and small["height"] == pen_cup.HEIGHT_MIN
    assert small["wall"] == pen_cup.WALL_MIN and small["floor"] == pen_cup.FLOOR_MIN
    assert small["columns"] == pen_cup.COLUMNS_MIN
    big = numbers(diameter=999.0, height=999.0, wall=999.0, floor=999.0, columns=999)
    assert big["diameter"] == pen_cup.DIAMETER_MAX and big["height"] == pen_cup.HEIGHT_MAX
    assert big["wall"] == pen_cup.WALL_MAX and big["floor"] == pen_cup.FLOOR_MAX
    assert big["columns"] == pen_cup.COLUMNS_MAX
    for kw in (dict(diameter=0.0, height=0.0, wall=0.0, floor=0.0, columns=0),
               dict(diameter=999.0, height=999.0, wall=999.0, floor=999.0, columns=999)):
        s, m = numbers(**kw), part(**kw)
        lo, hi = bbox(m)
        assert hi[0] - lo[0] == pytest.approx(s["diameter"], abs=ABS), "a clamped size was lost"
        assert hi[2] - lo[2] == pytest.approx(s["height"], abs=ABS), "a clamped height was lost"
        assert m.genus() == s["rows"] * s["columns"], "a clamped column count was lost"


@pytest.mark.parametrize("bad", (float("nan"), float("inf"), None, "wide"))
def test_a_number_that_is_no_number_is_refused(bad):
    for key in ("diameter", "height", "wall", "floor", "columns"):
        with pytest.raises(ValueError):
            pen_cup.pen_cup(**dict(BASE, **{key: bad}))
        with pytest.raises(ValueError):
            pen_cup.sizes(**dict(BASE, **{key: bad}))


# ------------------------------------------------------------------ the interface

def test_the_dialog_builds_a_cup_and_remembers_it(monkeypatch, tmp_path):
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
        dlg.diameter.setValue(90.0)
        dlg.height.setValue(120.0)
        dlg.wall.setValue(3.0)
        dlg.floor.setValue(3.0)
        dlg.columns.setValue(16)
        s = pen_cup.sizes(diameter=90.0, height=120.0, wall=3.0, floor=3.0, windows=True,
                          columns=16)
        assert "90.0" in dlg.info.text(), "the dialog does not show how wide the cup is"
        assert str(s["rows"] * s["columns"]) in dlg.info.text(), \
            "the dialog does not say how many windows the cup has"
        dlg.windows.setChecked(False)
        assert not dlg.columns.isEnabled(), "the columns can be changed while there are no windows"
        dlg.windows.setChecked(True)
        assert dlg.columns.isEnabled(), "the columns are still greyed out"
        return QDialog.Accepted

    monkeypatch.setattr(app_module.QDialog, "exec", fill)
    try:
        w.pen_cup_dialog()
        deadline = time.monotonic() + 20.0
        while len(w.doc.groups) < 1 and time.monotonic() < deadline:
            app.processEvents()
            time.sleep(0.005)
        assert len(w.doc.groups) == 1, "no cup in the model"
        assert [g.name for g in w.doc.groups] == ["Pen cup"], "the group has another name"
        s = pen_cup.sizes(diameter=90.0, height=120.0, wall=3.0, floor=3.0, windows=True,
                          columns=16)
        lo, hi = np.asarray(w.doc.groups[0].bbox(), dtype=float)
        assert lo[2] == pytest.approx(0.0, abs=1e-6), "the cup does not stand on the bed"
        assert hi[0] - lo[0] == pytest.approx(s["diameter"], abs=ABS), "not as wide as computed"
        assert hi[2] - lo[2] == pytest.approx(s["height"], abs=ABS), "not as high as computed"
        assert w.doc.groups[0].ents.to_manifold().genus() == s["rows"] * s["columns"], \
            "the windows are not the only holes in the wall"
        settings = app_module.Settings()
        assert (settings.pen_cup_diameter, settings.pen_cup_height, settings.pen_cup_wall,
                settings.pen_cup_floor, settings.pen_cup_windows, settings.pen_cup_columns) == \
            (90.0, 120.0, 3.0, 3.0, True, 16)
        labels = [a.text().replace("&", "").replace("…", "")
                  for a in w.flat_actions(w.parts_category("Home and Desk"))]
        assert labels.index("Pen Cup") == labels.index("Lithophane") + 1
    finally:
        for key, value in (("pen_cup_diameter", 75.0), ("pen_cup_height", 100.0),
                           ("pen_cup_wall", 2.4), ("pen_cup_floor", 2.4),
                           ("pen_cup_windows", True), ("pen_cup_columns", 12)):
            setattr(w.settings, key, value)
        w.settings.save()
        w.doc.dirty = False
        w.close()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))

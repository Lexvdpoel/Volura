"""Door wedges: a wedge you push under a door to hold it open.

`door_wedge.door_wedge` gives the wedge as one closed volume on `z = 0`: the side profile of a block
that stands `height` millimetres at its thick end and `tip` millimetres at its thin one, with saw
teeth on the slope that bite into the floor when the door tries to push the wedge out and a hole
through the thick end to hang it on a hook, all extruded `width` millimetres straight up.

These tests weigh the wedge against that: one volume on the bed and centred, `length` long and
`width` high; the slope from `height` down to `tip` and the angle it makes; teeth above the slope
when the ridges are on and a plain slope when they are off; an open hole through the thick end that
is not there when it is off or when the wedge is too low to carry it; no face that is not vertical or
horizontal, so nothing overhangs; every range reached by clamping and every number that is no number
refused; and the dialog that builds the wedge and remembers it.

Run with:  python -m pytest tests/test_door_wedge.py
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

from volura import door_wedge, geom  # noqa: E402

_app = QApplication.instance() or QApplication(sys.argv[:1])

BASE = dict(length=120.0, height=25.0, tip=2.0, width=35.0, ridges=True, hole=True)
#: the rounded corners are polygons, so a part is at most this far off
ABS = 0.05
#: mm across a probe that feels for plastic
PROBE = 0.4
#: mm³ of sliver a boolean leaves behind where two faces of the parts coincide exactly
SLIVER = 1e-4


def part(**kw) -> mf.Manifold:
    return door_wedge.door_wedge(**dict(BASE, **kw))


def numbers(**kw) -> dict:
    return door_wedge.sizes(**dict(BASE, **kw))


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


def block(x0, x1, y0, y1, z0, z1) -> mf.Manifold:
    """The block between the corners (`x0`, `y0`, `z0`) and (`x1`, `y1`, `z1`)."""
    return mf.Manifold.cube([x1 - x0, y1 - y0, z1 - z0]).translate((x0, y0, z0))


def frame(m) -> dict:
    """The back of the wedge and its floor, in the millimetres the wedge stands on the bed.

    The profile is drawn with the thick end at `x = 0` and the floor that lies on the bed on
    `y = 0`, and the finished wedge is centred, so these two say where they went.
    """
    lo, _hi = bbox(m)
    return {"back": lo[0], "floor": lo[1]}


def slope(s, x) -> float:
    """How high the wedge stands at `x`: the slope runs from `height` down to `tip`."""
    return s["height"] + (s["tip"] - s["height"]) * x / s["length"]


def top(m, s, x) -> float:
    """How high the wedge stands at `x`, measured on the wedge that stands on the bed."""
    f = frame(m)
    hit = m ^ block(f["back"] + x, f["back"] + x + 0.1, f["floor"] - 1.0,
                    f["floor"] + s["height"] + door_wedge.ROUND, -1.0, s["width"] + 1.0)
    return np.asarray(hit.bounding_box(), dtype=float)[4] - f["floor"]


def above(m, s, x, y, r=0.2) -> float:
    """A thin post at `x`, `y` millimetres above the slope, through the width of the wedge."""
    f = frame(m)
    return overlap(m, post(f["back"] + x, f["floor"] + slope(s, x) + y, 1.0,
                           s["width"] - 1.0, r))


def normals(m):
    """The unit normals and the areas of all triangles, pointing out of the part."""
    v, t = geom.mesh_arrays(m)
    a, b, c = v[t[:, 0]], v[t[:, 1]], v[t[:, 2]]
    n = np.cross(b - a, c - a)
    area = 0.5 * np.linalg.norm(n, axis=1)
    return n / np.maximum(2.0 * area[:, None], 1e-30), area


SIZES = (dict(length=60.0, height=10.0, tip=1.0, width=20.0),
         dict(),
         dict(length=200.0, height=50.0, tip=5.0, width=60.0))


# ------------------------------------------------------------------ the piece

@pytest.mark.parametrize("kw", SIZES)
def test_the_wedge_is_one_volume_that_stands_on_the_bed_and_is_centred(kw):
    s, m = numbers(**kw), part(**kw)
    assert m.status() == mf.Error.NoError and not m.is_empty()
    assert len(m.decompose()) == 1, "the wedge fell apart"
    assert m.genus() == (1 if s["hole"] else 0), "the hanging hole is not the only hole through it"
    lo, hi = bbox(m)
    assert lo[2] == pytest.approx(0.0, abs=1e-6), "the wedge does not stand on the bed"
    assert hi[2] - lo[2] == pytest.approx(s["width"], abs=ABS), "not as wide as asked"
    assert hi[0] - lo[0] == pytest.approx(s["length"], abs=ABS), "not as long as asked"
    assert np.allclose(lo[:2], -hi[:2], atol=ABS), "not centred in x and y"
    assert hi[1] <= s["height"] + ABS, "the wedge stands higher than it is high"


@pytest.mark.parametrize("kw", SIZES)
def test_the_wedge_is_height_high_at_the_thick_end_and_tip_at_the_thin_end(kw):
    """The back of the wedge is one straight slope from `height` at the thick end down to `tip` at
    the thin one; the two corners of the thick end round with `ROUND` millimetres, which is all that
    comes off the slope there."""
    s = numbers(**kw)
    m = part(**dict(kw, ridges=False, hole=False))
    for x in (6.0, 0.5 * s["length"], s["length"] - 6.0):
        assert top(m, s, x) == pytest.approx(slope(s, x), abs=ABS), "the slope is not straight"
    high = top(m, s, door_wedge.ROUND)
    assert s["height"] - 2.0 * door_wedge.ROUND <= high <= s["height"], \
        "the thick end does not stand as high as asked"
    low = top(m, s, s["length"] - door_wedge.ROUND)
    assert s["tip"] <= low < s["tip"] + 2.0 * door_wedge.ROUND, \
        "the thin end does not stand as low as asked"


def test_the_slope_is_the_angle_it_reports():
    """The slope is what holds the door, and the dialog says how steep it is."""
    for kw in SIZES:
        s = numbers(**kw)
        assert s["angle"] == pytest.approx(math.degrees(
            math.atan((s["height"] - s["tip"]) / s["length"]))), "another angle than the wedge has"
        m = part(**dict(kw, ridges=False, hole=False))
        x0, x1 = 6.0, s["length"] - 6.0
        assert math.degrees(math.atan2(top(m, s, x0) - top(m, s, x1), x1 - x0)) \
            == pytest.approx(s["angle"], abs=0.1), "the wedge does not stand at that angle"


# ------------------------------------------------------------------ the ridges

@pytest.mark.parametrize("kw", SIZES)
def test_the_ridges_stand_teeth_above_the_slope(kw):
    """Every `RIDGE_PITCH` millimetres a tooth runs along the slope and stands above it with its
    steep side towards the thin end, so the wedge bites when the door pushes it out."""
    s = numbers(**kw)
    on, off = part(**dict(kw, ridges=True)), part(**dict(kw, ridges=False))
    assert on.volume() > off.volume(), "the ridges add nothing"
    x = door_wedge.RIDGE_START + door_wedge.RIDGE_BASE      # the steep side of the first tooth
    assert above(on, s, x, 0.5 * door_wedge.RIDGE_HEIGHT) > 0.0, "nothing stands above the slope"
    assert above(off, s, x, 0.5 * door_wedge.RIDGE_HEIGHT) == pytest.approx(0.0, abs=SLIVER), \
        "a tooth is there although the ridges are off"
    assert above(on, s, x + 0.6, 0.5 * door_wedge.RIDGE_HEIGHT) == pytest.approx(0.0, abs=SLIVER), \
        "the steep side of a tooth does not face the thin end"


@pytest.mark.parametrize("kw", SIZES)
def test_the_ridges_stop_short_of_both_ends(kw):
    """The ridges start and end `RIDGE_START` millimetres from an end, so the wedge is still easy
    to push under a door and to lift out again."""
    s, m = numbers(**kw), part(**kw)
    for x in (6.0, door_wedge.RIDGE_START - 3.0, s["length"] - door_wedge.RIDGE_START + 5.0,
              s["length"] - 5.0):
        assert above(m, s, x, 0.5 * door_wedge.RIDGE_HEIGHT) == pytest.approx(0.0, abs=SLIVER), \
            "a ridge runs past the end of the wedge"


# ------------------------------------------------------------------ the hole

@pytest.mark.parametrize("kw", (dict(length=60.0, height=20.0, tip=1.0, width=20.0),
                                dict(),
                                dict(length=200.0, height=50.0, tip=5.0, width=60.0)))
def test_the_hole_is_open_through_the_thick_end(kw):
    """A hole of `HOLE_D` millimetres through the thick end, `HOLE_X` millimetres from the back and
    halfway the slope, hangs the wedge on a hook."""
    s, m = numbers(**kw), part(**kw)
    assert s["hole"], "this wedge should carry a hole"
    f = frame(m)
    x, y = f["back"] + door_wedge.HOLE_X, f["floor"] + 0.5 * s["height"]
    assert overlap(m, post(x, y, 1.0, s["width"] - 1.0, 0.5 * door_wedge.HOLE_D - 0.5)) \
        == pytest.approx(0.0, abs=SLIVER), "the hole is not open"
    assert overlap(m, post(x, y, 1.0, s["width"] - 1.0, 0.5 * door_wedge.HOLE_D + 0.5)) > 0.0, \
        "the hole is wider than asked"
    above = post(x, y + 0.5 * door_wedge.HOLE_D + 1.0, 1.0, s["width"] - 1.0, 0.5)
    assert overlap(m, above) > 0.0, "there is no plastic above the hole to hang it on"


@pytest.mark.parametrize("kw", (dict(hole=False), dict(height=door_wedge.HOLE_MIN - 1.0),
                                dict(hole=False, height=door_wedge.HOLE_MIN - 1.0)))
def test_there_is_no_hole_when_it_is_off_or_the_wedge_is_too_low(kw):
    """A wedge of less than `HOLE_MIN` millimetres is too low to carry a hole: it would break
    through the slope and hold nothing."""
    s, m = numbers(**kw), part(**kw)
    assert not s["hole"], "this wedge should have no hole"
    f = frame(m)
    assert overlap(m, post(f["back"] + door_wedge.HOLE_X, f["floor"] + 0.5 * s["height"],
                           1.0, s["width"] - 1.0, 0.5 * door_wedge.HOLE_D - 0.5)) > 0.0, \
        "the wedge is drilled although it should not be"


# ------------------------------------------------------------------ printing

@pytest.mark.parametrize("kw", SIZES + (dict(ridges=False, hole=False),
                                        dict(height=door_wedge.HOLE_MIN, ridges=True)))
def test_every_face_is_vertical_or_horizontal(kw):
    """The wedge is one profile pulled straight up, so no face leans and nothing needs support."""
    n, area = normals(part(**kw))
    keep = area > 1e-9
    vertical = np.abs(n[:, 2]) < 1e-6
    horizontal = np.abs(n[:, 2]) > 1.0 - 1e-6
    assert (vertical | horizontal)[keep].all(), "a face leans, so the wedge would overhang"


# ------------------------------------------------------------------ the numbers

def test_sizes_reports_the_wedge_it_makes():
    s = door_wedge.sizes(length=140.0, height=30.0, tip=3.0, width=40.0, ridges=False, hole=False)
    assert s["length"] == 140.0 and s["height"] == 30.0 and s["tip"] == 3.0
    assert s["width"] == 40.0 and s["ridges"] is False and s["hole"] is False
    assert s["angle"] == pytest.approx(math.degrees(math.atan(27.0 / 140.0)))
    assert s["size_x"] == pytest.approx(140.0)
    assert s["size_y"] == pytest.approx(30.0)
    assert s["size_z"] == pytest.approx(40.0)
    assert set(s) == {"length", "height", "tip", "width", "ridges", "hole", "angle",
                      "size_x", "size_y", "size_z"}


def test_the_default_wedge_is_for_a_door_over_a_floor():
    s = door_wedge.sizes()
    assert s["length"] == 120.0 and s["height"] == 25.0 and s["tip"] == 2.0
    assert s["width"] == 35.0 and s["ridges"] is True and s["hole"] is True


def test_the_hole_is_only_there_when_the_wedge_carries_it():
    assert door_wedge.sizes(height=door_wedge.HOLE_MIN)["hole"] is True
    assert door_wedge.sizes(height=door_wedge.HOLE_MIN - 0.1)["hole"] is False
    assert door_wedge.sizes(height=door_wedge.HOLE_MIN, hole=False)["hole"] is False


def test_the_thin_end_stays_under_the_thick_one():
    """The slope is what holds the door, so the thin end never grows up to the thick one."""
    for height in (10.0, 12.0, 25.0):
        s = numbers(height=height, tip=door_wedge.TIP_MAX)
        assert s["tip"] <= s["height"] - door_wedge.TIP_GAP, "there is no slope left"


def test_the_numbers_are_clamped_to_what_prints():
    small = numbers(length=0.0, height=0.0, tip=0.0, width=0.0)
    assert small["length"] == door_wedge.LENGTH_MIN and small["height"] == door_wedge.HEIGHT_MIN
    assert small["tip"] == door_wedge.TIP_MIN and small["width"] == door_wedge.WIDTH_MIN
    big = numbers(length=999.0, height=999.0, tip=999.0, width=999.0)
    assert big["length"] == door_wedge.LENGTH_MAX and big["height"] == door_wedge.HEIGHT_MAX
    assert big["tip"] == door_wedge.TIP_MAX and big["width"] == door_wedge.WIDTH_MAX
    for kw in (dict(length=0.0, height=0.0, tip=0.0, width=0.0),
               dict(length=999.0, height=999.0, tip=999.0, width=999.0)):
        s, m = numbers(**kw), part(**kw)
        lo, hi = bbox(m)
        assert hi[0] - lo[0] == pytest.approx(s["length"], abs=ABS), "a clamped size was lost"
        assert hi[2] - lo[2] == pytest.approx(s["width"], abs=ABS), "a clamped width was lost"


@pytest.mark.parametrize("bad", (float("nan"), float("inf"), None, "wide"))
def test_a_number_that_is_no_number_is_refused(bad):
    for key in ("length", "height", "tip", "width"):
        with pytest.raises(ValueError):
            door_wedge.door_wedge(**dict(BASE, **{key: bad}))
        with pytest.raises(ValueError):
            door_wedge.sizes(**dict(BASE, **{key: bad}))


# ------------------------------------------------------------------ the interface

def test_the_dialog_builds_a_wedge_and_remembers_it(monkeypatch, tmp_path):
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
        dlg.length.setValue(140.0)
        dlg.height.setValue(30.0)
        dlg.tip.setValue(3.0)
        dlg.width.setValue(40.0)
        dlg.ridges.setChecked(False)
        dlg.hole.setChecked(False)
        assert "10.9" in dlg.info.text(), "the dialog does not show the angle of the slope"
        assert "140.0" in dlg.info.text(), "the dialog does not show how long the wedge is"
        return QDialog.Accepted

    monkeypatch.setattr(app_module.QDialog, "exec", fill)
    try:
        w.door_wedge_dialog()
        deadline = time.monotonic() + 20.0
        while len(w.doc.groups) < 1 and time.monotonic() < deadline:
            app.processEvents()
            time.sleep(0.005)
        assert len(w.doc.groups) == 1, "no wedge in the model"
        assert [g.name for g in w.doc.groups] == ["Door wedge"], "the group has another name"
        s = door_wedge.sizes(length=140.0, height=30.0, tip=3.0, width=40.0, ridges=False,
                             hole=False)
        lo, hi = np.asarray(w.doc.groups[0].bbox(), dtype=float)
        assert lo[2] == pytest.approx(0.0, abs=1e-6), "the wedge does not stand on the bed"
        assert hi[0] - lo[0] == pytest.approx(s["length"], abs=ABS), "not as long as computed"
        assert hi[2] - lo[2] == pytest.approx(s["width"], abs=ABS), "not as wide as computed"
        assert w.doc.groups[0].ents.to_manifold().genus() == 0, "the wedge has a hole through it"
        settings = app_module.Settings()
        assert (settings.door_wedge_length, settings.door_wedge_height, settings.door_wedge_tip,
                settings.door_wedge_width, settings.door_wedge_ridges,
                settings.door_wedge_hole) == (140.0, 30.0, 3.0, 40.0, False, False)
        labels = [a.text().replace("&", "").replace("…", "")
                  for a in w.flat_actions(w.parts_category("Home and Desk"))]
        assert labels.index("Door Wedge") == labels.index("Desk Hook") + 1
    finally:
        for key, value in (("door_wedge_length", 120.0), ("door_wedge_height", 25.0),
                           ("door_wedge_tip", 2.0), ("door_wedge_width", 35.0),
                           ("door_wedge_ridges", True), ("door_wedge_hole", True)):
            setattr(w.settings, key, value)
        w.settings.save()
        w.doc.dirty = False
        w.close()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))

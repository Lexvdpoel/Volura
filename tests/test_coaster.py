"""Coasters: a flat plate with a rim that catches what runs down a glass.

`coaster.coaster` gives the coaster as one closed volume on `z = 0`: a plate `size` millimetres
across in `shape` - a circle, a hexagon across its flats or a square with rounded corners -
pulled up `thickness` millimetres, with the top cut out to `thickness` so the edge stands `rim`
millimetres high as a rim, and grooves of `pattern` cut `GROOVE` millimetres deep in what is left
of the top.

These tests weigh the coaster against that: one volume on the bed and centred, `thickness + rim`
high; the width every shape asks for; the rim standing while the recess is cut down to the plate; a
flat top when the rim is 0; the rings and the grid lowering the top where they are and leaving it
alone everywhere else, and no pattern leaving it flat; no face that is not vertical or horizontal,
so nothing overhangs; every range reached by clamping, every shape and pattern that is unknown
fallen back and every number that is no number refused; and the dialog that builds the coaster and
remembers it.

Run with:  python -m pytest tests/test_coaster.py
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

from volura import coaster, geom  # noqa: E402

_app = QApplication.instance() or QApplication(sys.argv[:1])

BASE = dict(shape="round", size=95.0, thickness=4.0, rim=1.5, pattern="rings")
#: the rings and the rounded corners are polygons, so a part is at most this far off
ABS = 0.05


def part(**kw) -> mf.Manifold:
    return coaster.coaster(**dict(BASE, **kw))


def numbers(**kw) -> dict:
    return coaster.sizes(**dict(BASE, **kw))


def bbox(m):
    """The bounding box of a manifold as (low, high)."""
    b = np.asarray(m.bounding_box(), dtype=float)
    return b[:3], b[3:]


def touches(m, probe) -> bool:
    """Whether a probe runs into the part anywhere.

    Two solids that only touch leave a sliver of a few ten-millionths of a mm³ behind, so that is
    not a touch; plastic that is really there is off by whole tenths of a millimetre.
    """
    return (m ^ probe).volume() > 1e-6


def post(x, y, r, z0=-1.0, z1=40.0) -> mf.Manifold:
    """A round post of radius `r` on (`x`, `y`) from `z0` to `z1`, to feel the piece with."""
    return mf.Manifold.cylinder(z1 - z0, r, r, 32).translate((x, y, z0))


def top(m, x, y, r=0.3) -> float:
    """The height of the highest plastic of `m` above (`x`, `y`); 0 where there is none at all."""
    hit = m ^ post(x, y, r)
    return 0.0 if hit.is_empty() else float(np.asarray(hit.bounding_box(), dtype=float)[5])


def triangles(m):
    """The unit normals and the areas of all triangles."""
    v, t = geom.mesh_arrays(m)
    a, b, c = v[t[:, 0]], v[t[:, 1]], v[t[:, 2]]
    n = np.cross(b - a, c - a)
    area = 0.5 * np.linalg.norm(n, axis=1)
    return n / np.maximum(2.0 * area[:, None], 1e-30), area


# ------------------------------------------------------------------ the piece

@pytest.mark.parametrize("kw", (dict(), dict(shape="hexagon", pattern="grid"),
                                dict(shape="square", rim=0.0, pattern="none"),
                                dict(size=60.0, thickness=3.0, rim=0.0),
                                dict(size=150.0, thickness=8.0, rim=3.0, pattern="grid")))
def test_the_coaster_is_one_volume_that_lies_on_the_bed_and_is_centred(kw):
    s, m = numbers(**kw), part(**kw)
    assert m.status() == mf.Error.NoError and not m.is_empty()
    assert len(m.decompose()) == 1, "the coaster fell apart"
    lo, hi = bbox(m)
    assert lo[2] == pytest.approx(0.0, abs=1e-6), "the coaster does not lie on the bed"
    assert hi[2] - lo[2] == pytest.approx(s["height"], abs=ABS), "not as high as asked"
    assert hi[0] - lo[0] == pytest.approx(s["outside"], abs=ABS), "not as wide as computed"
    assert hi[1] - lo[1] == pytest.approx(s["size"], abs=ABS), "not as deep as asked"
    assert lo[0] == pytest.approx(-hi[0], abs=ABS) and lo[1] == pytest.approx(-hi[1], abs=ABS), \
        "not centred in x and y"


@pytest.mark.parametrize("shape, wide", (("round", 95.0),
                                         ("hexagon", 95.0 / math.cos(math.radians(30))),
                                         ("square", 95.0)))
def test_every_shape_is_as_wide_as_it_is_across(shape, wide):
    """The hexagon is measured across its flats, so it is wider still over its corners; the circle
    and the square are as wide as they are deep."""
    s = numbers(shape=shape)
    assert s["outside"] == pytest.approx(wide, abs=ABS), "the dialog says another width"
    lo, hi = bbox(part(shape=shape))
    assert hi[0] - lo[0] == pytest.approx(s["outside"], abs=ABS), "%s is not that wide" % shape
    assert hi[1] - lo[1] == pytest.approx(s["size"], abs=ABS), "%s is not that deep" % shape


# ------------------------------------------------------------------ the rim

def test_the_rim_stands_and_the_recess_is_cut_down_to_the_plate():
    """The top is cut out to the plate, so the edge that is left stands as a rim round it."""
    s, m = numbers(), part()
    r = 0.5 * s["size"]
    assert top(m, 0.0, r - 0.5 * coaster.WALL) == pytest.approx(s["height"], abs=ABS), \
        "the rim does not stand above the plate"
    assert top(m, 0.0, r - coaster.WALL - 1.0) == pytest.approx(s["thickness"], abs=ABS), \
        "the recess is not cut down to the plate"
    assert not touches(m, post(0.0, 0.0, 2.0, s["thickness"] + 0.1, s["height"])), \
        "nothing was cut out of the middle of the top"


def test_without_a_rim_the_top_is_flat_at_the_thickness():
    s, m = numbers(rim=0.0, pattern="none"), part(rim=0.0, pattern="none")
    assert s["height"] == pytest.approx(s["thickness"]), "a coaster without a rim is not a plate"
    r = 0.5 * s["size"]
    for x, y in ((0.0, 0.0), (0.0, r - 6.0), (0.0, r - 0.5 * coaster.WALL)):
        assert top(m, x, y) == pytest.approx(s["thickness"], abs=ABS), "the top is not flat"


# ------------------------------------------------------------------ the grooves

def test_the_rings_groove_the_top_where_they_stand():
    """A ring every `RING_STEP` millimetres from the middle out to `RING_STEP` inside the edge;
    between two of them, in the middle and by the rim the top stays where it was."""
    s, m = numbers(), part()
    plain = part(pattern="none")
    r_out = 0.5 * s["size"]
    floor = s["thickness"] - coaster.GROOVE
    r = coaster.RING_STEP
    while r <= r_out - coaster.RING_STEP:
        assert top(m, 0.0, r) == pytest.approx(floor, abs=ABS), "no ring at %g mm" % r
        assert top(plain, 0.0, r) == pytest.approx(s["thickness"], abs=ABS), \
            "the plate is grooved without the rings"
        r += coaster.RING_STEP
    for r in (0.0, 9.0, 15.0, r_out - coaster.RING_STEP - 2.0):
        assert top(m, 0.0, r) == pytest.approx(s["thickness"], abs=ABS), \
            "the top is grooved at %g mm, where no ring stands" % r
    assert top(m, 0.0, r_out - coaster.WALL - 1.0) == pytest.approx(s["thickness"], abs=ABS), \
        "a ring breaks into the wall of the recess"


def test_the_grid_grooves_the_top_where_it_stands():
    """A line every `GRID_STEP` millimetres through the middle, in x and in y, and nothing between
    two of them."""
    s, m = numbers(pattern="grid"), part(pattern="grid")
    plain = part(pattern="none")
    x = 0.5 * (0.5 * s["size"] - coaster.WALL - coaster.EDGE)
    floor = s["thickness"] - coaster.GROOVE
    for k in (-3, 0, 3):
        d = k * coaster.GRID_STEP
        assert top(m, x, d) == pytest.approx(floor, abs=ABS), "no line at %g mm" % d
        assert top(m, d, x) == pytest.approx(floor, abs=ABS), "no line at %g mm" % d
        assert top(plain, x, d) == pytest.approx(s["thickness"], abs=ABS), \
            "the plate is grooved without the grid"
    for d in (4.0, 12.0):
        assert top(m, x, d) == pytest.approx(s["thickness"], abs=ABS), \
            "the top is grooved at %g mm, between two lines" % d
        assert top(m, d, x) == pytest.approx(s["thickness"], abs=ABS), \
            "the top is grooved at %g mm, between two lines" % d


def test_no_pattern_leaves_the_top_flat():
    s, m = numbers(pattern="none"), part(pattern="none")
    for x, y in ((0.0, 0.0), (0.0, 6.0), (0.0, 24.0), (20.0, 0.0), (20.0, 20.0)):
        assert top(m, x, y) == pytest.approx(s["thickness"], abs=ABS), \
            "the top is grooved at %g, %g without a pattern" % (x, y)


# ------------------------------------------------------------------ printing

@pytest.mark.parametrize("kw", (dict(), dict(shape="hexagon", pattern="grid"),
                                dict(shape="square", rim=0.0),
                                dict(size=60.0, thickness=3.0, rim=3.0, pattern="grid"),
                                dict(size=150.0, thickness=8.0, pattern="none")))
def test_every_face_is_vertical_or_horizontal(kw):
    """The coaster is a plate pulled straight up with vertical cuts, so nothing overhangs and
    nothing needs support."""
    n, area = triangles(part(**kw))
    keep = area > 1e-9
    up = np.abs(n[keep][:, 2])
    assert np.all((up > 0.999) | (up < 1e-6)), "a face is tilted, so the coaster overhangs"


# ------------------------------------------------------------------ the numbers

def test_sizes_reports_the_coaster_it_makes():
    s = coaster.sizes(shape="square", size=120.0, thickness=6.0, rim=2.0, pattern="grid")
    assert (s["shape"], s["size"], s["thickness"], s["rim"], s["pattern"]) == (
        "square", 120.0, 6.0, 2.0, "grid")
    assert s["outside"] == pytest.approx(120.0) and s["height"] == pytest.approx(8.0)
    assert set(s) == {"shape", "size", "thickness", "rim", "pattern", "outside", "height"}


def test_the_default_coaster_fits_an_ordinary_glass():
    s = coaster.sizes()
    assert s["shape"] == "round" and s["size"] == 95.0 and s["thickness"] == 4.0
    assert s["rim"] == 1.5 and s["pattern"] == "rings"
    assert s["height"] == pytest.approx(5.5)


def test_the_numbers_are_clamped_to_what_prints():
    small = numbers(size=0.0, thickness=0.0, rim=-9.0)
    assert small["size"] == coaster.SIZE_MIN and small["thickness"] == coaster.THICKNESS_MIN
    assert small["rim"] == coaster.RIM_MIN
    big = numbers(size=999.0, thickness=999.0, rim=999.0)
    assert big["size"] == coaster.SIZE_MAX and big["thickness"] == coaster.THICKNESS_MAX
    assert big["rim"] == coaster.RIM_MAX
    for kw in (dict(size=0.0, thickness=0.0, rim=-9.0),
               dict(size=999.0, thickness=999.0, rim=999.0)):
        s, m = numbers(**kw), part(**kw)
        lo, hi = bbox(m)
        assert hi[2] - lo[2] == pytest.approx(s["height"], abs=ABS), "a clamped height was lost"
        assert hi[0] - lo[0] == pytest.approx(s["outside"], abs=ABS), "a clamped size was lost"


@pytest.mark.parametrize("shape", (None, "", "blob", "Round"))
def test_a_shape_that_is_no_shape_is_the_round_one(shape):
    assert coaster.sizes(shape=shape)["shape"] == "round"
    assert part(shape=shape).volume() == pytest.approx(part().volume(), rel=1e-9)


@pytest.mark.parametrize("pattern", (None, "", "blob", "Rings"))
def test_a_pattern_that_is_no_pattern_is_the_rings_one(pattern):
    assert coaster.sizes(pattern=pattern)["pattern"] == "rings"
    assert part(pattern=pattern).volume() == pytest.approx(part().volume(), rel=1e-9)


@pytest.mark.parametrize("bad", (float("nan"), float("inf"), None, "wide"))
def test_a_number_that_is_no_number_is_refused(bad):
    for key in ("size", "thickness", "rim"):
        with pytest.raises(ValueError):
            coaster.coaster(**dict(BASE, **{key: bad}))
        with pytest.raises(ValueError):
            coaster.sizes(**dict(BASE, **{key: bad}))


# ------------------------------------------------------------------ the interface

def test_the_dialog_builds_a_coaster_and_remembers_it(monkeypatch, tmp_path):
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
        dlg.shape.setCurrentIndex(dlg.shape.findData("hexagon"))
        dlg.size.setValue(120.0)
        dlg.thickness.setValue(5.0)
        dlg.rim.setValue(2.0)
        dlg.pattern.setCurrentIndex(dlg.pattern.findData("grid"))
        assert "138.6" in dlg.info.text(), "the dialog does not show how wide the coaster is"
        assert "7.0" in dlg.info.text(), "the dialog does not show how high the coaster is"
        return QDialog.Accepted

    monkeypatch.setattr(app_module.QDialog, "exec", fill)
    try:
        w.coaster_dialog()
        deadline = time.monotonic() + 20.0
        while len(w.doc.groups) < 1 and time.monotonic() < deadline:
            app.processEvents()
            time.sleep(0.005)
        assert len(w.doc.groups) == 1, "no coaster in the model"
        assert [g.name for g in w.doc.groups] == ["Coaster"]
        s = coaster.sizes(shape="hexagon", size=120.0, thickness=5.0, rim=2.0, pattern="grid")
        lo, hi = np.asarray(w.doc.groups[0].bbox(), dtype=float)
        assert lo[2] == pytest.approx(0.0, abs=1e-6), "the coaster does not lie on the bed"
        assert hi[0] - lo[0] == pytest.approx(s["outside"], abs=ABS), "not as wide as computed"
        assert hi[1] - lo[1] == pytest.approx(120.0, abs=ABS), "not as deep as asked"
        assert hi[2] - lo[2] == pytest.approx(s["height"], abs=1e-3), "not as high as computed"
        settings = app_module.Settings()
        assert (settings.coaster_shape, settings.coaster_size, settings.coaster_thickness,
                settings.coaster_rim, settings.coaster_pattern) == (
            "hexagon", 120.0, 5.0, 2.0, "grid")
        labels = [a.text().replace("&", "").replace("…", "") for a in w.flat_actions(w.menuBar())]
        assert labels.index("Coaster") == labels.index("Bag Clip") + 1
    finally:
        for key, value in (("coaster_shape", "round"), ("coaster_size", 95.0),
                           ("coaster_thickness", 4.0), ("coaster_rim", 1.5),
                           ("coaster_pattern", "rings")):
            setattr(w.settings, key, value)
        w.settings.save()
        w.doc.dirty = False
        w.close()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))

"""Bit holders: a block that stands screwdriver bits upright in rows.

`bit_holder.bit_holder` gives the holder as one closed volume that lies on `z = 0` and is centred
on the origin in plan: a block with `cols` by `rows` hexagonal sockets for 1/4 inch bits, open at
the top and chamfered at the mouth, with a ledge for a label along the front. The rows lean back by
`angle` degrees, so the bits lean towards you, and the top of the block rises with that tilt.

These tests weigh the part against that: it is one piece that lies on the bed and is as big as its
block, a bit of 6.35 mm across the flats stands in every socket without touching anything and a bit
wider does, every socket keeps two millimetres of floor under its lowest point, a bit that leans the
other way does not fit, the wall between two sockets stays thick enough and nothing hangs over
steeper than 45 degrees, because the holder prints without support.

Run with:  python -m pytest tests/test_bit_holder.py
"""
import math
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import manifold3d as mf  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from volura import bit_holder, geom  # noqa: E402

BASE = dict(cols=6, rows=2, depth=10.0, fit=0.15, pitch=10.0, angle=0.0, label=True)
#: the rounded corners, the chamfers and the hexagons are polygons, so a part is at most this far
#: off
ABS = 0.05


def part(**kw) -> mf.Manifold:
    return bit_holder.bit_holder(**dict(BASE, **kw))


def numbers(**kw) -> dict:
    return bit_holder.sizes(**dict(BASE, **kw))


def bbox(m):
    """The bounding box of a manifold as (low, high)."""
    b = np.asarray(m.bounding_box(), dtype=float)
    return b[:3], b[3:]


def centres(s) -> list:
    """The mouths of the sockets, from the numbers `sizes` reports.

    The sockets are `pitch` apart in both directions and the block is centred on the origin, so the
    row in front stands as far from the front of the block as the wall there is thick: the ledge for
    a label when there is one, a wall like the others when there is not.
    """
    front = bit_holder.LEDGE if s["label"] else bit_holder.WALL
    y0 = -0.5 * s["length"] + front + 0.5 * s["flats"]
    return [((i - 0.5 * (s["cols"] - 1)) * s["pitch"], y0 + j * s["pitch"])
            for j in range(s["rows"]) for i in range(s["cols"])]


def shank(across, length, x, y, s, lean=None, above=0.5) -> mf.Manifold:
    """A hexagonal bit `across` across the flats and `length` long, along the axis of the socket
    whose mouth is at (`x`, `y`), with its top `above` millimetres above that mouth.

    `lean` is how far the bit leans back over X; the sockets lean by the angle of the holder, so a
    bit that leans the other way runs into the wall of its socket.
    """
    tilt = math.radians(s["angle"]) if lean is None else lean
    rot = geom.rotation_matrix((1.0, 0.0, 0.0), tilt)
    cs = mf.CrossSection([geom.circle_points(across / math.sqrt(3.0), 6)])
    down = length - above
    at = (x, y + down * math.sin(tilt), s["height"] - down * math.cos(tilt))
    return mf.Manifold.extrude(cs, length).transform(geom.affine(rot, at))


def touches(m, probe) -> bool:
    """Whether a probe runs into the part anywhere.

    Two solids that only touch leave a sliver of a few ten-millionths of a mm³ behind, so that is
    not a touch; a bit that does not fit is off by whole millimetres.
    """
    return (m ^ probe).volume() > 1e-6


def triangles(m):
    """The unit normals, the centres and the areas of all triangles, pointing out of the part."""
    v, t = geom.mesh_arrays(m)
    a, b, c = v[t[:, 0]], v[t[:, 1]], v[t[:, 2]]
    n = np.cross(b - a, c - a)
    area = 0.5 * np.linalg.norm(n, axis=1)
    return n / np.maximum(2.0 * area[:, None], 1e-30), (a + b + c) / 3.0, area


# ------------------------------------------------------------------ the part

@pytest.mark.parametrize("cols, rows", ((1, 1), (6, 2), (12, 4)))
@pytest.mark.parametrize("angle", (0.0, 15.0, 30.0))
def test_the_holder_is_one_piece_that_lies_on_the_bed(cols, rows, angle):
    kw = dict(cols=cols, rows=rows, angle=angle)
    s, m = numbers(**kw), part(**kw)
    assert m.status() == mf.Error.NoError and not m.is_empty()
    assert len(m.decompose()) == 1, "the holder fell apart"
    lo, hi = bbox(m)
    assert np.allclose(lo, (-0.5 * s["width"], -0.5 * s["length"], 0.0), atol=ABS), \
        "the block is not centred on the origin, or does not lie on the bed"
    assert np.allclose(hi, (0.5 * s["width"], 0.5 * s["length"], s["height"]), atol=ABS), \
        "the block is not as big as sizes says"


@pytest.mark.parametrize("depth, fit", ((6.0, 0.0), (10.0, 0.15), (20.0, 0.4)))
@pytest.mark.parametrize("pitch, angle", ((8.5, 0.0), (10.0, 20.0), (15.0, 30.0)))
def test_the_holder_is_valid_whatever_it_is_asked_for(depth, fit, pitch, angle):
    m = part(depth=depth, fit=fit, pitch=pitch, angle=angle)
    assert m.status() == mf.Error.NoError and not m.is_empty()
    assert len(m.decompose()) == 1, "the holder fell apart"


@pytest.mark.parametrize("label", (True, False))
def test_the_block_is_a_box_with_rounded_corners(label):
    """Below the sockets the block is a rectangle with corners of CORNER, over the whole
    footprint."""
    s, m = numbers(label=label), part(label=label)
    want = s["width"] * s["length"] - (4.0 - math.pi) * bit_holder.CORNER ** 2
    assert m.slice(0.5).area() == pytest.approx(want, rel=0.01), "the corners are not rounded"


# ------------------------------------------------------------------ the sockets

@pytest.mark.parametrize("fit", (0.0, 0.15, 0.4))
@pytest.mark.parametrize("depth, angle", ((6.0, 0.0), (10.0, 15.0), (20.0, 30.0)))
def test_a_bit_stands_in_every_socket(fit, depth, angle):
    kw = dict(fit=fit, depth=depth, angle=angle)
    s, m = numbers(**kw), part(**kw)
    for x, y in centres(s):
        assert not touches(m, shank(bit_holder.BIT, s["depth"], x, y, s)), \
            "a 1/4 inch bit does not stand in the socket"
        assert touches(m, shank(s["flats"] + 0.3, s["depth"], x, y, s)), \
            "the socket is wider than the bit with its play"


@pytest.mark.parametrize("angle", (10.0, 20.0, 30.0))
def test_the_sockets_lean_towards_the_front(angle):
    """A bit in a socket of a tilted holder leans towards the front; one that leans back towards
    the back of the holder runs into the wall."""
    s, m = numbers(angle=angle), part(angle=angle)
    tilt = math.radians(angle)
    for x, y in centres(s):
        assert not touches(m, shank(bit_holder.BIT, s["depth"], x, y, s)), \
            "a bit does not stand in a socket that leans"
        assert touches(m, shank(bit_holder.BIT, s["depth"], x, y, s, lean=-tilt)), \
            "a socket takes a bit that leans the other way, so it does not lean towards you"
    assert s["length"] > numbers(angle=0.0)["length"], \
        "the block does not grow at the back with the tilt"


@pytest.mark.parametrize("depth, angle", ((6.0, 0.0), (10.0, 20.0), (20.0, 30.0)))
def test_the_floor_under_every_socket_is_at_least_two_mm(depth, angle):
    kw = dict(depth=depth, angle=angle)
    s, m = numbers(**kw), part(**kw)
    tilt = math.radians(s["angle"])
    for x, y in centres(s):
        low = s["height"] - s["depth"] * math.cos(tilt) - 0.5 * s["flats"] * math.sin(tilt)
        assert low >= bit_holder.FLOOR - 1e-9, "a socket has too little floor under it"
        y_low = y + s["depth"] * math.sin(tilt) - 0.5 * s["flats"] * math.cos(tilt)
        floor = mf.Manifold.cube([0.6, 0.6, low - 0.04]).translate((x - 0.3, y_low - 0.3, 0.02))
        assert (m ^ floor).volume() == pytest.approx(floor.volume(), rel=1e-3), \
            "there is nothing solid under the lowest point of the socket"


def test_the_mouth_of_a_socket_is_chamfered():
    """The lead-in is a chamfer of MOUTH at 45 degrees: it is gone MOUTH down the socket."""
    s, m = numbers(), part()
    x, y = centres(s)[0]
    plug = s["flats"] + 2.0 * bit_holder.MOUTH - 0.4
    assert not touches(m, shank(plug, 0.6, x, y, s)), "the mouth has no chamfer for a bit to find"
    assert touches(m, shank(plug, 1.2, x, y, s)), \
        "the chamfer runs deeper down the socket than it is wide"


@pytest.mark.parametrize("fit", (0.0, 0.2, 0.4))
def test_the_wall_between_two_sockets_stays_thick(fit):
    """The pitch is raised when it would bring two sockets too close, so the wall between them is
    what the part breaks on."""
    kw = dict(fit=fit, pitch=bit_holder.PITCH_MIN)
    s, m = numbers(**kw), part(**kw)
    assert s["pitch"] > bit_holder.PITCH_MIN, "the pitch was taken as it was asked"
    assert s["pitch"] >= 2.0 * s["flats"] / math.sqrt(3.0) + bit_holder.WALL_MIN - 1e-9
    x, y = centres(s)[0]
    wall = mf.Manifold.cube([1.0, 1.0, 2.0]).translate((x + 0.5 * s["pitch"] - 0.5, y - 0.5, 1.0))
    assert (m ^ wall).volume() == pytest.approx(wall.volume(), rel=1e-3), \
        "there is no wall left between two sockets next to each other"


# ------------------------------------------------------------------ the ledge

@pytest.mark.parametrize("label", (True, False))
def test_the_ledge_is_a_flat_strip_along_the_front(label):
    """In front of the first row there is a strip at the full width, a millimetre below the top of
    the block: with the ledge it is open to lie a label on, without it the wall stands there."""
    kw = dict(label=label)
    s, m = numbers(**kw), part(**kw)
    top = s["height"] - bit_holder.LEDGE_DROP
    for x in (-0.5 * s["width"] + 0.5, 0.0, 0.5 * s["width"] - 0.5):
        air = mf.Manifold.cube([1.0, 1.0, 0.4]).translate((x - 0.5, -0.5 * s["length"] + 3.0,
                                                           top + 0.3))
        assert touches(m, air) != label, "the front is not a ledge a millimetre below the top"
    under = mf.Manifold.cube([1.0, 1.0, 0.4]).translate((-0.5, -0.5 * s["length"] + 3.0,
                                                         top - 0.3))
    assert touches(m, under), "there is nothing under the ledge"


# ------------------------------------------------------------------ printing

@pytest.mark.parametrize("angle", (0.0, 15.0, 30.0))
@pytest.mark.parametrize("label", (True, False))
def test_nothing_hangs_over_steeper_than_45_degrees(angle, label):
    """The holder stands on its block: a tilted hexagonal wall is at most 30 degrees off vertical
    and the floor of a socket looks up, so no face hangs over steeper than 45 degrees."""
    m = part(angle=angle, label=label, depth=20.0, fit=0.4, pitch=8.5)
    n, c, area = triangles(m)
    keep = area > 1e-9
    assert (n[keep][c[keep][:, 2] < 1e-9][:, 2] < -0.99).all(), \
        "the triangles do not face out of the part"
    down = keep & (n[:, 2] < -np.sin(np.pi / 4.0) - 0.01) & (c[:, 2] > 1e-6)
    assert not down.any(), "a face overhangs steeper than 45 degrees, so it needs support"


# ------------------------------------------------------------------ the numbers

def test_sizes_reports_the_block_and_its_sockets():
    s = bit_holder.sizes(cols=4, rows=3, depth=12.0, fit=0.2, pitch=11.0, angle=0.0, label=True)
    assert s["cols"] == 4 and s["rows"] == 3 and s["depth"] == 12.0 and s["fit"] == 0.2
    assert s["pitch"] == 11.0 and s["angle"] == 0.0 and s["label"] is True
    assert s["flats"] == pytest.approx(bit_holder.BIT + 0.4)
    assert s["width"] == pytest.approx(3 * 11.0 +
                                       2.0 * (bit_holder.WALL + s["flats"] / math.sqrt(3.0)))
    assert s["length"] == pytest.approx(2 * 11.0 + 2.0 * bit_holder.WALL + s["flats"] +
                                        bit_holder.LEDGE)
    assert s["height"] == pytest.approx(bit_holder.FLOOR + 12.0)
    assert set(s) == {"cols", "rows", "depth", "fit", "pitch", "angle", "label", "flats",
                      "width", "length", "height"}


def test_the_block_grows_with_the_ledge_and_the_tilt():
    """The ledge is the front wall, lowered: it takes the five millimetres the wall did not have.
    The tilt carries the sockets backwards, and the block grows with them."""
    flat = numbers(angle=0.0)
    assert flat["length"] - numbers(label=False)["length"] == pytest.approx(
        bit_holder.LEDGE - bit_holder.WALL)
    assert numbers(angle=30.0)["length"] - flat["length"] == pytest.approx(
        flat["depth"] * math.sin(math.radians(30.0)))
    assert numbers(angle=30.0)["height"] > flat["height"], "the top does not rise with the tilt"


def test_the_numbers_are_clamped_to_what_prints():
    small = numbers(cols=0, rows=0, depth=1.0, fit=-1.0, pitch=1.0, angle=-5.0)
    assert small["cols"] == bit_holder.COLS_MIN and small["rows"] == bit_holder.ROWS_MIN
    assert small["depth"] == bit_holder.DEPTH_MIN and small["fit"] == bit_holder.FIT_MIN
    assert small["pitch"] >= bit_holder.PITCH_MIN and small["angle"] == bit_holder.ANGLE_MIN
    big = numbers(cols=99, rows=99, depth=99.0, fit=9.0, pitch=99.0, angle=90.0)
    assert big["cols"] == bit_holder.COLS_MAX and big["rows"] == bit_holder.ROWS_MAX
    assert big["depth"] == bit_holder.DEPTH_MAX and big["fit"] == bit_holder.FIT_MAX
    assert big["pitch"] == bit_holder.PITCH_MAX and big["angle"] == bit_holder.ANGLE_MAX
    lo, hi = bbox(part(cols=0, rows=0, depth=1.0, fit=-1.0, pitch=1.0, angle=-5.0, label=False))
    assert hi[0] - lo[0] == pytest.approx(
        2.0 * (bit_holder.WALL + bit_holder.BIT / math.sqrt(3.0)), abs=ABS), \
        "the smallest holder is not one socket with a wall around it"


@pytest.mark.parametrize("bad", (float("nan"), float("inf"), None, "wide"))
def test_a_number_that_is_no_number_is_refused(bad):
    for key in ("cols", "rows", "depth", "fit", "pitch", "angle"):
        with pytest.raises(ValueError):
            bit_holder.bit_holder(**dict(BASE, **{key: bad}))
        with pytest.raises(ValueError):
            bit_holder.sizes(**dict(BASE, **{key: bad}))


# ------------------------------------------------------------------ the interface

def test_the_dialog_builds_a_holder_and_remembers_it(monkeypatch, tmp_path):
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
        dlg.cols.setValue(4)
        dlg.rows.setValue(1)
        dlg.depth.setValue(14.0)
        dlg.label.setChecked(False)
        assert "4 × 1 sockets" in dlg.info.text(), "the dialog does not show the sockets it makes"
        return QDialog.Accepted

    monkeypatch.setattr(app_module.QDialog, "exec", fill)
    try:
        w.bit_holder_dialog()
        deadline = time.monotonic() + 15.0
        while not w.doc.groups and time.monotonic() < deadline:
            app.processEvents()
            time.sleep(0.005)
        assert len(w.doc.groups) == 1, "no holder in the model"
        lo, hi = np.asarray(w.doc.groups[0].bbox(), dtype=float)
        s = bit_holder.sizes(cols=4, rows=1, depth=14.0, label=False)
        assert lo[2] == pytest.approx(0.0, abs=1e-6), "the holder does not lie on the bed"
        assert hi[2] - lo[2] == pytest.approx(s["height"], abs=1e-4)
        assert hi[0] - lo[0] == pytest.approx(s["width"], abs=0.2), "the block is not as wide"
        assert hi[1] - lo[1] == pytest.approx(s["length"], abs=0.2), "the block is not as long"
        settings = app_module.Settings()
        assert (settings.bit_holder_cols, settings.bit_holder_rows,
                settings.bit_holder_depth, settings.bit_holder_label) == (
            4, 1, pytest.approx(14.0), False)
        labels = [a.text().replace("&", "").replace("…", "") for a in w.flat_actions(w.menuBar())]
        assert labels.index("Bit Holder") == labels.index("Zip-Tie Mount") + 1
    finally:
        w.settings.bit_holder_cols, w.settings.bit_holder_rows = 6, 2
        w.settings.bit_holder_depth, w.settings.bit_holder_label = 10.0, True
        w.settings.save()
        w.doc.dirty = False
        w.close()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))

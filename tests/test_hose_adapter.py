"""Hose adapters: a printed reducer with barbs that joins two hoses of different diameter.

`hose_adapter.hose_adapter` gives the adapter as one closed volume that stands on `z = 0` with
its axis along Z: a spigot of `d1` at the bottom, a spigot of `d2` at the top, a stop collar
between them and a bore of `d - 2 * wall` through it all, joined in the collar by a cone of
45 degrees. Each spigot carries `barbs` barbs that widen to `BARB_GROW` of the hose diameter
going towards the middle and step straight back, so a hose rides over the ramp and hangs behind
the step; `straight` makes plain ends with a chamfer at the tip instead.

These tests weigh the part against that: it is one tube that stands on the bed and is as wide at
the collar as asked, the bore is open from end to end, the barbs reach their diameter just before
each step, a straight end is a plain cylinder, the bore never overhangs steeper than 45 degrees,
and the numbers that do not print are clamped or refused.

Run with:  python -m pytest tests/test_hose_adapter.py
"""
import math
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import manifold3d as mf  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from volura import geom, hose_adapter  # noqa: E402

BASE = dict(d1=12.0, d2=8.0)
#: the spigots, the barbs and the bore are polygons, so a part is at most this far off
ABS = 0.01


def adapter(**kw) -> mf.Manifold:
    return hose_adapter.hose_adapter(**dict(BASE, **kw))


def numbers(**kw) -> dict:
    return hose_adapter.sizes(**dict(BASE, **kw))


def bbox(m):
    """The bounding box of a manifold as (low, high)."""
    b = np.asarray(m.bounding_box(), dtype=float)
    return b[:3], b[3:]


def outer_radius(m, z) -> float:
    """How far the part reaches from its axis in the section at `z`."""
    b = np.asarray(m.slice(z).bounds(), dtype=float)
    return max(abs(b[0]), abs(b[1]), b[2], b[3])


def triangles(m):
    """The unit normals and the centres of all triangles, pointing out of the part."""
    v, t = geom.mesh_arrays(m)
    a, b, c = v[t[:, 0]], v[t[:, 1]], v[t[:, 2]]
    n = np.cross(b - a, c - a)
    return n / np.linalg.norm(n, axis=1)[:, None], (a + b + c) / 3.0


# ------------------------------------------------------------------ the part

def test_the_default_adapter_is_one_tube_that_stands_on_the_bed():
    s, m = numbers(), adapter()
    assert m.status() == mf.Error.NoError and not m.is_empty()
    assert m.genus() == 1, "the water way does not go through the adapter"
    lo, hi = bbox(m)
    assert lo[2] == pytest.approx(0.0), "the adapter does not stand on the bed"
    assert hi[2] - lo[2] == pytest.approx(s["length"]), "the adapter is not as high as its length"
    assert hi[0] - lo[0] == pytest.approx(s["collar_d"], abs=ABS)
    assert hi[1] - lo[1] == pytest.approx(s["collar_d"], abs=ABS), "the collar is not round"
    # the default height is exactly the barbs, the chamfer and the collar
    collar = s["barbs"] * s["barb_len1"] + s["chamfer"] + 0.5 * hose_adapter.COLLAR
    assert s["collar_z"] == pytest.approx(s["barbs"] * s["barb_len1"] + s["chamfer"])
    assert outer_radius(m, collar) == pytest.approx(0.5 * s["collar_d"], abs=ABS)


def test_the_collar_is_the_stop_the_hoses_are_pushed_up_against():
    s, m = numbers(), adapter()
    assert s["collar_d"] == pytest.approx(max(s["d1"], s["d2"]) * hose_adapter.BARB_GROW
                                          + hose_adapter.COLLAR_ROOM)
    collar = s["collar_z"] + 0.5 * hose_adapter.COLLAR
    tips = max(outer_radius(m, 0.2), outer_radius(m, s["length"] - 0.2))
    assert outer_radius(m, collar) > tips, "the collar does not stand proud of both spigots"


def test_the_bore_is_the_hose_diameter_minus_two_walls():
    s = numbers(wall=2.0)
    assert s["bore1"] == pytest.approx(s["d1"] - 4.0)
    assert s["bore2"] == pytest.approx(s["d2"] - 4.0)


@pytest.mark.parametrize("d1, d2", ((12.0, 8.0), (8.0, 12.0), (10.0, 10.0)))
def test_the_bore_is_open_from_end_to_end(d1, d2):
    s = hose_adapter.sizes(d1=d1, d2=d2)
    m = hose_adapter.hose_adapter(d1=d1, d2=d2)
    r = 0.5 * (min(s["bore1"], s["bore2"]) - 0.2)
    probe = (mf.Manifold.cylinder(s["length"] + 2.0, r, r, hose_adapter.SEGMENTS)
             .translate((0.0, 0.0, -1.0)))
    assert (m ^ probe).volume() == pytest.approx(0.0, abs=1e-6), "the bore is blocked"


# ------------------------------------------------------------------ the spigots

@pytest.mark.parametrize("d1, d2", ((12.0, 8.0), (8.0, 12.0)))
def test_the_barbs_widen_towards_the_middle_and_step_back(d1, d2):
    s = hose_adapter.sizes(d1=d1, d2=d2)
    m = hose_adapter.hose_adapter(d1=d1, d2=d2)
    for d, z_tip, up, bl in ((s["d1"], 0.0, 1.0, s["barb_len1"]),
                             (s["d2"], s["length"], -1.0, s["barb_len2"])):
        assert outer_radius(m, z_tip + up * 0.05) == pytest.approx(0.5 * d, abs=ABS), \
            "the tip of a spigot is not the diameter of its hose"
        for i in range(1, s["barbs"] + 1):
            near = outer_radius(m, z_tip + up * (i * bl - 0.05))
            assert near == pytest.approx(0.5 * d * hose_adapter.BARB_GROW, abs=ABS), \
                "a barb does not reach its full diameter before it steps back"
            if i < s["barbs"]:
                after = outer_radius(m, z_tip + up * (i * bl + 0.05))
                assert after == pytest.approx(0.5 * d, abs=ABS), "a barb did not step back"


def test_a_straight_end_is_a_plain_cylinder_with_a_chamfer_at_the_tip():
    s, m = numbers(straight=True), adapter(straight=True)
    low, high = s["collar_z"] - s["chamfer"], s["collar_z"] + hose_adapter.COLLAR
    c = min(hose_adapter.CHAMFER, 0.5 * s["wall"])
    assert outer_radius(m, 0.5 * c) == pytest.approx(0.5 * s["d1"] - 0.5 * c, abs=ABS)
    assert outer_radius(m, s["length"] - 0.5 * c) == pytest.approx(0.5 * s["d2"] - 0.5 * c,
                                                                   abs=ABS)
    for z in (c + 0.2, low - 0.2, high + 0.2, s["length"] - c - 0.2):
        d = s["d1"] if z < low else s["d2"]
        assert outer_radius(m, z) == pytest.approx(0.5 * d, abs=ABS), \
            "a straight spigot is not a plain cylinder"


# ------------------------------------------------------------------ printing upright

@pytest.mark.parametrize("d1, d2", ((8.0, 20.0), (20.0, 8.0)))
def test_the_bore_never_overhangs_steeper_than_45_degrees(d1, d2):
    s = hose_adapter.sizes(d1=d1, d2=d2)
    n, c = triangles(hose_adapter.hose_adapter(d1=d1, d2=d2))
    assert (n[c[:, 2] < 1e-9][:, 2] < -0.99).all(), "the triangles do not face out of the part"
    r = np.hypot(c[:, 0], c[:, 1])
    # the wall of the bore: between the two bore diameters, and not one of the flat rings
    band = (r > 0.5 * min(s["bore1"], s["bore2"]) + 0.05) & (r < 0.5 * max(s["bore1"],
                                                                          s["bore2"]) - 0.05)
    band &= np.abs(n[:, 2]) < 0.99
    assert band.any(), "there is no bore cone to look at"
    assert n[band][:, 2].min() > -math.sin(math.pi / 4.0) - 0.01, \
        "the bore overhangs steeper than 45 degrees, so it needs support"


@pytest.mark.parametrize("d1, d2", ((8.0, 20.0), (20.0, 8.0), (12.0, 8.0)))
def test_the_outside_needs_no_support(d1, d2):
    """Under the collar a cone of 45 degrees; only the small steps of the upper barbs face down."""
    s = hose_adapter.sizes(d1=d1, d2=d2)
    n, c = triangles(hose_adapter.hose_adapter(d1=d1, d2=d2))
    r = np.hypot(c[:, 0], c[:, 1])
    down = (n[:, 2] < -math.sin(math.pi / 4.0) - 0.01) & (c[:, 2] > 1e-6)
    steps = (r > 0.5 * s["d2"] - ABS) & (r < 0.5 * s["d2"] * hose_adapter.BARB_GROW + ABS)
    assert (down <= steps).all(), "a face of the outside overhangs and needs support"


@pytest.mark.parametrize("d1, d2", ((8.0, 20.0), (20.0, 8.0), (12.0, 8.0), (5.0, 40.0)))
def test_the_wall_is_never_thinner_than_asked(d1, d2):
    s = hose_adapter.sizes(d1=d1, d2=d2)
    m = hose_adapter.hose_adapter(d1=d1, d2=d2)
    # a polygon of SEGMENTS sides is this much narrower between its corners than at them
    facet = 0.5 * s["collar_d"] * (1.0 - math.cos(math.pi / hose_adapter.SEGMENTS))
    for z in np.linspace(0.1, s["length"] - 0.1, 200):
        polys = m.slice(float(z)).to_polygons()
        radii = [np.hypot(p[:, 0], p[:, 1]) for p in map(np.asarray, polys)]
        outer = max(r.min() for r in radii)
        inner = min(r.max() for r in radii)
        assert outer - inner >= s["wall"] - facet - 0.01, \
            f"the wall at z = {z:.2f} is thinner than {s['wall']} mm"


# ------------------------------------------------------------------ the numbers

def test_the_length_of_the_barbs_and_the_collar_is_the_default_height():
    s = numbers()
    assert s["length"] == pytest.approx(
        s["barbs"] * (s["barb_len1"] + s["barb_len2"]) + s["chamfer"] + hose_adapter.COLLAR)
    assert s["barb_len1"] == pytest.approx(max(hose_adapter.BARB_PART * s["d1"],
                                               hose_adapter.BARB_MIN))


def test_a_length_too_short_for_the_barbs_is_raised():
    s = numbers()
    assert numbers(length=1.0)["length"] == pytest.approx(s["length"]), \
        "a height too short for the barbs was not raised"
    long = numbers(length=s["length"] + 10.0)
    assert long["length"] == pytest.approx(s["length"] + 10.0)
    assert bbox(adapter(length=s["length"] + 10.0))[1][2] == pytest.approx(long["length"])
    assert adapter(length=1.0).genus() == 1, "the shortest adapter is not a tube"


def test_numbers_are_clamped_to_what_prints():
    s = hose_adapter.sizes(d1=0.0, d2=99.0, wall=0.0, barbs=0)
    assert (s["d1"], s["d2"], s["wall"], s["barbs"]) == (hose_adapter.D_MIN, hose_adapter.D_MAX,
                                                         hose_adapter.WALL_MIN,
                                                         hose_adapter.BARBS_MIN)
    s = hose_adapter.sizes(d1=99.0, d2=99.0, wall=99.0, barbs=99)
    assert (s["d1"], s["d2"], s["wall"], s["barbs"]) == (hose_adapter.D_MAX, hose_adapter.D_MAX,
                                                         hose_adapter.WALL_MAX,
                                                         hose_adapter.BARBS_MAX)
    big = hose_adapter.hose_adapter(d1=99.0, d2=99.0, wall=99.0, barbs=99)
    assert big.status() == mf.Error.NoError and big.genus() == 1


@pytest.mark.parametrize("bad", (float("nan"), float("inf"), None, "wide"))
def test_a_number_that_is_no_number_is_refused(bad):
    for key in ("d1", "d2", "wall", "barbs", "length"):
        if key == "length" and bad is None:
            continue                                # no height at all is the automatic one
        with pytest.raises(ValueError):
            hose_adapter.hose_adapter(**dict(BASE, **{key: bad}))
        with pytest.raises(ValueError):
            hose_adapter.sizes(**dict(BASE, **{key: bad}))


@pytest.mark.parametrize("d1, d2, wall", ((3.0, 8.0, 1.6), (12.0, 8.0, 4.0), (6.0, 6.0, 2.6)))
def test_a_water_way_thinner_than_a_millimetre_is_refused(d1, d2, wall):
    with pytest.raises(ValueError, match="bore"):
        hose_adapter.hose_adapter(d1=d1, d2=d2, wall=wall)


# ------------------------------------------------------------------ the interface

def test_the_dialog_builds_an_adapter_and_remembers_it(monkeypatch, tmp_path):
    import time
    from PySide6.QtCore import QCoreApplication
    from PySide6.QtWidgets import QApplication, QDialog, QDialogButtonBox
    app = QApplication.instance() or QApplication(sys.argv[:1])
    import volura.app as app_module
    monkeypatch.setattr(app_module.errors, "log_dir", lambda: str(tmp_path))
    w = app_module.MainWindow()
    w.doc.dirty = False
    QCoreApplication.processEvents()

    def fill(dlg):
        dlg.straight.setChecked(True)
        assert not dlg.barbs.isEnabled(), "plain ends have no barbs"
        dlg.straight.setChecked(False)
        dlg.wall.setValue(5.0)
        assert not dlg.bb.button(QDialogButtonBox.Ok).isEnabled(), "a wall that fills the bore"
        dlg.wall.setValue(2.0)
        dlg.d1.setValue(16.0)
        dlg.d2.setValue(10.0)
        assert dlg.bb.button(QDialogButtonBox.Ok).isEnabled()
        assert "mm" in dlg.info.text()
        return QDialog.Accepted

    monkeypatch.setattr(app_module.QDialog, "exec", fill)
    try:
        w.hose_adapter_dialog()
        deadline = time.monotonic() + 15.0
        while not w.doc.groups and time.monotonic() < deadline:
            app.processEvents()
            time.sleep(0.005)
        assert len(w.doc.groups) == 1, "no adapter in the model"
        lo, hi = np.asarray(w.doc.groups[0].bbox(), dtype=float)
        s = hose_adapter.sizes(d1=16.0, d2=10.0, wall=2.0)
        assert hi[2] - lo[2] == pytest.approx(s["length"], abs=1e-4)
        assert lo[2] == pytest.approx(0.0, abs=1e-6), "the adapter stands on the bed"
        settings = app_module.Settings()
        assert (settings.hose_d1, settings.hose_d2, settings.hose_wall) == (
            pytest.approx(16.0), pytest.approx(10.0), pytest.approx(2.0))
        labels = [a.text().replace("&", "").replace("…", "") for a in w.flat_actions(w.menuBar())]
        assert labels.index("Hose Adapter") == labels.index("Pegboard Hook") + 1
    finally:
        w.settings.hose_d1, w.settings.hose_d2, w.settings.hose_wall = 12.0, 8.0, 1.6
        w.settings.save()
        w.doc.dirty = False
        w.close()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))

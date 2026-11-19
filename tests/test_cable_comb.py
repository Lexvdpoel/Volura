"""Cable combs: a flat comb that keeps a bundle of cables side by side, every one in its own seat.

`cable_comb.cable_comb` gives the comb as one closed volume on `z = 0`: a plate `thickness`
millimetres high that carries `cables` round seats of `cable` millimetres, every one of them reached
through an opening of `snap * cable` millimetres cut through the top edge and led in with a chamfer,
so a cable is snapped into its own seat and stays in line with the rest of the bundle.

These tests weigh the comb against that: one volume on the bed and centred, `thickness` millimetres
high and without a hole through it; a seat at every cable that a cable of the diameter asked fits into
without touching plastic; an opening that is narrower than its cable and exactly as wide as `snap`
times it; plate left between two seats; no face that leans, so the comb is extruded straight up and
prints flat; every range reached by clamping, a comb that would hang off the bed taken back to what
fits and every number that is no number refused; and the dialog that builds the comb and remembers it.

Run with:  python -m pytest tests/test_cable_comb.py
"""
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import manifold3d as mf  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from PySide6.QtWidgets import QApplication  # noqa: E402

from volura import cable_comb, geom  # noqa: E402

_app = QApplication.instance() or QApplication(sys.argv[:1])

BASE = dict(cables=6, cable=6.0, thickness=4.0, snap=0.75)
#: the plate is a polygon and a seat is a polygon of 64 sides, so a part is at most this far off
ABS = 0.05
#: mm across a probe that feels for plastic
PROBE = 0.4
#: mm³ of sliver a boolean leaves behind where two faces of the parts coincide exactly
SLIVER = 1e-4


def part(**kw) -> mf.Manifold:
    return cable_comb.cable_comb(**dict(BASE, **kw))


def numbers(**kw) -> dict:
    return cable_comb.sizes(**dict(BASE, **kw))


def bbox(m):
    """The bounding box of a manifold as (low, high)."""
    b = np.asarray(m.bounding_box(), dtype=float)
    return b[:3], b[3:]


def common(m, probe) -> float:
    """How much of two solids is the same: nothing when they only touch.

    Two solids that only touch leave a sliver of a few ten-millionths of a mm³ behind, so that is
    not a touch; plastic that is really there is off by whole tenths of a millimetre.
    """
    return mf.Manifold.batch_boolean([m, probe], mf.OpType.Intersect).volume()


def post(x, y, z0, z1, r=PROBE) -> mf.Manifold:
    """A round post of radius `r` at (`x`, `y`), between the heights `z0` and `z1`."""
    return mf.Manifold.cylinder(z1 - z0, r, r, 32).translate((x, y, z0))


def cable_post(x, y, s, r) -> mf.Manifold:
    """A post of radius `r` at (`x`, `y`) through the whole thickness of the comb and past both of
    its faces: what a cable of that radius would take, whether plastic is in the way or not."""
    return post(x, y, -0.5, s["thickness"] + 0.5, r)


def seat(s, k):
    """The middle of the seat of cable `k`, in the frame the comb stands in."""
    return (cable_comb.EDGE + 0.5 * s["cable"] + k * s["pitch"] - 0.5 * s["size_x"],
            cable_comb.SEAT_LIFT)


def mouth(s) -> float:
    """The middle of the narrow part of an opening: halfway between the top of its seat and the
    lead-in that widens the opening where it leaves the top edge."""
    low = cable_comb.SEAT_LIFT + 0.5 * (s["cable"] + cable_comb.SEAT_CLEARANCE)
    return 0.5 * (low + 0.5 * s["size_y"] - cable_comb.CHAMFER)


def faces(m):
    """The unit normals, the areas and the heights of the middles of all triangles."""
    v, t = geom.mesh_arrays(m)
    a, b, c = v[t[:, 0]], v[t[:, 1]], v[t[:, 2]]
    n = np.cross(b - a, c - a)
    area = 0.5 * np.linalg.norm(n, axis=1)
    return n / np.maximum(2.0 * area[:, None], 1e-30), area, v[t].mean(axis=1)[:, 2]


SIZES = (dict(),
         dict(cables=2, cable=3.0, thickness=3.0, snap=0.6),
         dict(cables=16, cable=12.0, thickness=8.0, snap=0.9),
         dict(cables=9, cable=7.5, thickness=5.0, snap=0.65))


# ------------------------------------------------------------------ the piece

@pytest.mark.parametrize("kw", SIZES)
def test_the_comb_is_one_volume_that_lies_on_the_bed_and_is_centred(kw):
    s, m = numbers(**kw), part(**kw)
    assert m.status() == mf.Error.NoError and not m.is_empty()
    assert len(m.decompose()) == 1, "the comb fell apart"
    assert m.genus() == 0, "there is a hole straight through the comb"
    lo, hi = bbox(m)
    assert lo[2] == pytest.approx(0.0, abs=1e-6), "the comb does not lie on the bed"
    assert hi[0] - lo[0] == pytest.approx(s["size_x"], abs=ABS), "not as long as asked"
    assert hi[1] - lo[1] == pytest.approx(s["size_y"], abs=ABS), "not as deep as asked"
    assert hi[2] - lo[2] == pytest.approx(s["thickness"], abs=ABS), "the plate is not as thick"
    assert np.allclose(lo[:2], -hi[:2], atol=ABS), "not centred in x and y"


# ------------------------------------------------------------------ the seats

@pytest.mark.parametrize("kw", SIZES)
def test_every_seat_holds_a_cable_of_the_diameter_asked(kw):
    """A cable of `cable` millimetres lies in every seat without touching plastic: a seat is
    `SEAT_CLEARANCE` wider than the cable, so the cable goes in by hand and the comb holds it."""
    s, m = numbers(**kw), part(**kw)
    for k in range(s["cables"]):
        x, y = seat(s, k)
        assert common(m, cable_post(x, y, s, 0.5 * (s["cable"] - 0.1))) \
            == pytest.approx(0.0, abs=SLIVER), "a cable does not fit in its seat"


@pytest.mark.parametrize("kw", SIZES)
def test_an_opening_is_narrower_than_its_cable(kw):
    """A cable is pushed through an opening that is narrower than it, which is what holds it, and
    that opening is as wide as `snap` times the cable and not a hair wider."""
    s, m = numbers(**kw), part(**kw)
    y = mouth(s)
    for k in range(s["cables"]):
        x = seat(s, k)[0]
        assert common(m, cable_post(x, y, s, 0.5 * s["cable"])) > 0.0, \
            "the opening is as wide as the cable, so nothing holds it"
        assert common(m, cable_post(x, y, s, 0.5 * (s["opening"] - 0.1))) \
            == pytest.approx(0.0, abs=SLIVER), "the opening is not as wide as snap * cable"


@pytest.mark.parametrize("kw", SIZES)
def test_plate_is_left_between_two_seats(kw):
    """`GAP` millimetres of plate stand between two seats, so a comb does not tear in half between
    two cables when the bundle is pulled aside."""
    s, m = numbers(**kw), part(**kw)
    for k in range(s["cables"] - 1):
        x, y = seat(s, k)
        probe = post(x + 0.5 * s["pitch"], y, 0.2, s["thickness"] - 0.2)
        assert common(m, probe) == pytest.approx(probe.volume(), abs=SLIVER), \
            "there is no plate between two seats"


# ------------------------------------------------------------------ printing

@pytest.mark.parametrize("kw", SIZES)
def test_every_face_is_vertical_or_horizontal(kw):
    """The comb is one outline extruded straight up, so every wall stands vertical and every face
    lies horizontal: it prints lying flat with nothing above it hanging in the air."""
    n, area, _z = faces(part(**kw))
    keep = area > 1e-9
    vertical = np.abs(n[:, 2]) < 1e-6
    horizontal = np.abs(n[:, 2]) > 1.0 - 1e-6
    assert (vertical | horizontal)[keep].all(), "a face leans, so the comb is not extruded up"


# ------------------------------------------------------------------ the numbers

def test_sizes_reports_the_comb_it_makes():
    s = cable_comb.sizes(cables=9, cable=7.5, thickness=6.0, snap=0.65)
    assert s["cables"] == 9 and s["cable"] == 7.5
    assert s["thickness"] == 6.0 and s["snap"] == 0.65
    assert s["opening"] == pytest.approx(0.65 * 7.5), "an opening is not snap times the cable"
    assert s["pitch"] == pytest.approx(11.0), "two seats do not stand a cable plus the gap apart"
    assert s["size_x"] == pytest.approx(9 * 11.0 + cable_comb.EDGE)
    assert s["size_y"] == pytest.approx(7.5 + 2.0 * cable_comb.EDGE)
    assert s["size_z"] == pytest.approx(6.0)
    assert set(s) == {"cables", "cable", "thickness", "snap", "opening", "pitch",
                      "size_x", "size_y", "size_z"}


def test_the_default_comb_holds_a_bundle_of_usb_cables():
    s = cable_comb.sizes()
    assert s["cables"] == 6 and s["cable"] == 6.0
    assert s["thickness"] == 4.0 and s["snap"] == 0.75


def test_the_numbers_are_clamped_to_what_prints():
    small = numbers(cables=0, cable=0.0, thickness=0.0, snap=0.0)
    assert small["cables"] == cable_comb.CABLES_MIN and small["cable"] == cable_comb.CABLE_MIN
    assert small["thickness"] == cable_comb.THICKNESS_MIN
    assert small["snap"] == cable_comb.SNAP_MIN
    big = numbers(cables=99, cable=99.0, thickness=99.0, snap=99.0)
    assert big["cable"] == cable_comb.CABLE_MAX and big["thickness"] == cable_comb.THICKNESS_MAX
    assert big["snap"] == cable_comb.SNAP_MAX
    for kw in (dict(cables=0, cable=0.0, thickness=0.0, snap=0.0),
               dict(cables=99, cable=6.0, thickness=99.0, snap=99.0)):
        s, m = numbers(**kw), part(**kw)
        lo, hi = bbox(m)
        assert hi[0] - lo[0] == pytest.approx(s["size_x"], abs=ABS), "a clamped length was lost"
        assert hi[1] - lo[1] == pytest.approx(s["size_y"], abs=ABS), "a clamped width was lost"
        assert hi[2] - lo[2] == pytest.approx(s["thickness"], abs=ABS), "a clamped height was lost"
        assert len(m.decompose()) == 1, "a clamped comb fell apart"


def test_a_comb_of_too_many_cables_is_made_to_fit_the_bed():
    """The thickest cables take the whole bed long before the most cables are reached, so the number
    of them is taken back to what fits: a comb is never longer than the bed."""
    kw = dict(cables=cable_comb.CABLES_MAX, cable=cable_comb.CABLE_MAX)
    s = cable_comb.sizes(**kw)
    assert s["cables"] < cable_comb.CABLES_MAX, "the widest cables do not reach the bed"
    assert s["size_x"] <= cable_comb.BED, "the comb hangs off the bed"
    m = cable_comb.cable_comb(**kw)
    lo, hi = bbox(m)
    assert hi[0] - lo[0] == pytest.approx(s["size_x"], abs=ABS), "the bed was not kept"
    assert len(m.decompose()) == 1, "the comb that fits the bed fell apart"


@pytest.mark.parametrize("bad", (float("nan"), float("inf"), None, "wide"))
def test_a_number_that_is_no_number_is_refused(bad):
    for key in ("cables", "cable", "thickness", "snap"):
        with pytest.raises(ValueError):
            cable_comb.cable_comb(**dict(BASE, **{key: bad}))
        with pytest.raises(ValueError):
            cable_comb.sizes(**dict(BASE, **{key: bad}))


# ------------------------------------------------------------------ the interface

def test_the_dialog_builds_a_comb_and_remembers_it(monkeypatch, tmp_path):
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
        dlg.cables.setValue(8)
        dlg.cable.setValue(7.0)
        dlg.thickness.setValue(5.0)
        dlg.snap.setValue(70)
        s = cable_comb.sizes(cables=8, cable=7.0, thickness=5.0, snap=0.7)
        assert f"{s['opening']:.2f}" in dlg.info.text(), \
            "the dialog does not show how wide an opening is"
        assert f"{s['size_x']:.1f}" in dlg.info.text(), \
            "the dialog does not say how big the comb stands on the bed"
        assert f"{s['size_y']:.1f}" in dlg.info.text(), \
            "the dialog does not say how deep the comb stands on the bed"
        return QDialog.Accepted

    monkeypatch.setattr(app_module.QDialog, "exec", fill)
    try:
        w.cable_comb_dialog()
        deadline = time.monotonic() + 20.0
        while len(w.doc.groups) < 1 and time.monotonic() < deadline:
            app.processEvents()
            time.sleep(0.005)
        assert len(w.doc.groups) == 1, "no comb in the model"
        assert [g.name for g in w.doc.groups] == ["Cable comb"], "the group has another name"
        s = cable_comb.sizes(cables=8, cable=7.0, thickness=5.0, snap=0.7)
        lo, hi = np.asarray(w.doc.groups[0].bbox(), dtype=float)
        assert lo[2] == pytest.approx(0.0, abs=1e-6), "the comb does not lie on the bed"
        assert hi[0] - lo[0] == pytest.approx(s["size_x"], abs=ABS), "not as long as computed"
        assert hi[2] - lo[2] == pytest.approx(s["size_z"], abs=ABS), "not as thick as computed"
        assert w.doc.groups[0].ents.to_manifold().genus() == 0, "the comb has a hole through it"
        settings = app_module.Settings()
        assert (settings.cable_comb_cables, settings.cable_comb_cable,
                settings.cable_comb_thickness, settings.cable_comb_snap) == \
            (8, 7.0, 5.0, pytest.approx(0.7))
        labels = [a.text().replace("&", "").replace("…", "")
                  for a in w.flat_actions(w.parts_category("Cables"))]
        assert labels.index("Cable Comb") == labels.index("Cable Clip") + 1
    finally:
        for key, value in (("cable_comb_cables", 6), ("cable_comb_cable", 6.0),
                           ("cable_comb_thickness", 4.0), ("cable_comb_snap", 0.75)):
            setattr(w.settings, key, value)
        w.settings.save()
        w.doc.dirty = False
        w.close()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))

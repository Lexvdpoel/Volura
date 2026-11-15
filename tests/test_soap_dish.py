"""Soap dishes: a dish a bar of soap lies on while the water runs off it.

`soap_dish.soap_dish` gives the dish as one closed volume on `z = 0`: a tray `length` millimetres
long and `width` wide that stands `rim` high, with a wall of `wall` millimetres around a floor of
2 mm, `ribs` bars standing on that floor from wall to wall and a slot through the floor in every
gap between two of them, so the soap lies dry and the water runs down under it.

These tests weigh the dish against that: one volume on the bed and centred, `length` by `width` by
`rim`; an open slot in every gap between the ribs and floor under every rib; the ribs where they are
told and as high as `rim` allows; a wall as thick as asked; the ribs clamped until 6 mm is left
between two of them; no face that is not vertical or horizontal, so nothing overhangs; every range
reached by clamping and every number that is no number refused; and the dialog that builds the dish
and remembers it.

Run with:  python -m pytest tests/test_soap_dish.py
"""
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import manifold3d as mf  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from PySide6.QtWidgets import QApplication  # noqa: E402

from volura import geom, soap_dish  # noqa: E402

_app = QApplication.instance() or QApplication(sys.argv[:1])

BASE = dict(length=110.0, width=75.0, rim=15.0, wall=2.4, ribs=7)
#: the rounded corners are polygons, so a part is at most this far off
ABS = 0.05
#: mm across a probe that feels for plastic
PROBE = 0.4
#: mm³ of sliver a boolean leaves behind where two faces of the parts coincide exactly
SLIVER = 1e-4


def part(**kw) -> mf.Manifold:
    return soap_dish.soap_dish(**dict(BASE, **kw))


def numbers(**kw) -> dict:
    return soap_dish.sizes(**dict(BASE, **kw))


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


def gaps(s):
    """The middle of every gap between two neighbouring ribs: where a drain slot is cut."""
    return [0.5 * (a + b) for a, b in zip(s["rib_x"], s["rib_x"][1:])]


def normals(m):
    """The unit normals and the areas of all triangles, pointing out of the part."""
    v, t = geom.mesh_arrays(m)
    a, b, c = v[t[:, 0]], v[t[:, 1]], v[t[:, 2]]
    n = np.cross(b - a, c - a)
    area = 0.5 * np.linalg.norm(n, axis=1)
    return n / np.maximum(2.0 * area[:, None], 1e-30), area


SIZES = (dict(length=70.0, width=50.0, rim=8.0, wall=1.6),
         dict(),
         dict(length=160.0, width=120.0, rim=30.0, wall=4.0))


# ------------------------------------------------------------------ the piece

@pytest.mark.parametrize("kw", SIZES)
def test_the_dish_is_one_volume_that_stands_on_the_bed_and_is_centred(kw):
    s, m = numbers(**kw), part(**kw)
    assert m.status() == mf.Error.NoError and not m.is_empty()
    assert len(m.decompose()) == 1, "the dish fell apart"
    assert m.genus() == s["ribs"] - 1, "the slots through the floor are not the only holes"
    lo, hi = bbox(m)
    assert lo[2] == pytest.approx(0.0, abs=1e-6), "the dish does not stand on the bed"
    assert hi[0] - lo[0] == pytest.approx(s["length"], abs=ABS), "not as long as asked"
    assert hi[1] - lo[1] == pytest.approx(s["width"], abs=ABS), "not as wide as asked"
    assert hi[2] - lo[2] == pytest.approx(s["rim"], abs=ABS), "the rim does not stand as high"
    assert np.allclose(lo[:2], -hi[:2], atol=ABS), "not centred in x and y"


# ------------------------------------------------------------------ the floor

@pytest.mark.parametrize("kw", SIZES)
def test_the_water_runs_through_a_slot_in_every_gap_between_the_ribs(kw):
    """Every gap between two neighbouring ribs has one slot of `SLOT_W` millimetres through the
    floor of `FLOOR` millimetres, and the soap above it lies on the ribs beside the slot."""
    s, m = numbers(**kw), part(**kw)
    assert len(gaps(s)) == s["ribs"] - 1, "there is a gap without a slot"
    for x in gaps(s):
        assert overlap(m, post(x, 0.0, 0.2, s["rim"] - 0.2)) == pytest.approx(0.0, abs=SLIVER), \
            "the slot is not open through the floor"
        edge = 0.5 * soap_dish.SLOT_W + 0.5 * PROBE + 0.3
        assert overlap(m, post(x + edge, 0.0, 0.2, 0.5 * soap_dish.FLOOR)) > 0.0, \
            "the floor next to the slot is cut away too"


@pytest.mark.parametrize("kw", SIZES)
def test_the_slots_stop_short_of_the_wall(kw):
    """A slot runs over `width - 2 * wall - 16` millimetres, so the floor keeps an edge all round
    the dish and the water still has the whole length of the gap to run through."""
    s, m = numbers(**kw), part(**kw)
    reach = 0.5 * (s["width"] - 2.0 * s["wall"] - 2.0 * soap_dish.SLOT_END)
    for x in gaps(s):
        assert overlap(m, post(x, reach - 1.0, 0.2, 0.5 * soap_dish.FLOOR)) \
            == pytest.approx(0.0, abs=SLIVER), "the slot does not reach that far"
        assert overlap(m, post(x, reach + 1.0, 0.2, 0.5 * soap_dish.FLOOR)) > 0.0, \
            "the slot runs past the end of the floor"


@pytest.mark.parametrize("kw", SIZES)
def test_the_floor_is_two_millimetres_thick_under_the_rim(kw):
    """The floor the dish stands on is `FLOOR` millimetres thick: the basin is cut from there up, so
    the dish is watertight and still low enough for the water to run off."""
    s, m = numbers(**kw), part(**kw)
    reach = 0.5 * (s["width"] - 2.0 * s["wall"] - 2.0 * soap_dish.SLOT_END)
    x, y = gaps(s)[0], reach + 1.0      # the middle of a gap, past the end of its slot
    assert overlap(m, post(x, y, 0.2, 0.5 * soap_dish.FLOOR)) > 0.0, "there is no floor"
    assert overlap(m, post(x, y, soap_dish.FLOOR + 0.2, s["rim"] - 0.2)) \
        == pytest.approx(0.0, abs=SLIVER), "the basin is cut higher than the floor"


# ------------------------------------------------------------------ the ribs

@pytest.mark.parametrize("kw", SIZES)
def test_the_ribs_stand_on_the_floor_from_wall_to_wall(kw):
    """`ribs` bars of `RIB_W` millimetres stand on the floor and reach the walls, so a bar of soap
    lies above the water instead of in it."""
    s, m = numbers(**kw), part(**kw)
    assert len(s["rib_x"]) == s["ribs"], "another rib was counted than is stood up"
    top = soap_dish.FLOOR + soap_dish.RIB_PART * s["rim"]
    for x in s["rib_x"]:
        assert overlap(m, post(x, 0.0, 0.2, 0.5 * soap_dish.FLOOR)) > 0.0, \
            "there is no floor under a rib"
        assert overlap(m, post(x, 0.0, soap_dish.FLOOR + 0.2, top - 0.2)) > 0.0, \
            "a rib does not stand on the floor"
        assert overlap(m, post(x, 0.0, top + 0.2, s["rim"] - 0.2)) \
            == pytest.approx(0.0, abs=SLIVER), "a rib stands higher than it is high"
        edge = 0.5 * soap_dish.RIB_W + 0.5 * PROBE + 0.3
        assert overlap(m, post(x + edge, 0.0, soap_dish.FLOOR + 0.2, top - 0.2)) \
            == pytest.approx(0.0, abs=SLIVER), "a rib is wider than asked"
        near = 0.5 * (s["width"] - 2.0 * s["wall"]) - 0.5
        assert overlap(m, post(x, near, soap_dish.FLOOR + 0.2, top - 0.2)) > 0.0, \
            "a rib does not reach the wall"


@pytest.mark.parametrize("kw", SIZES)
def test_the_first_and_the_last_rib_stand_twelve_mm_from_the_wall(kw):
    """The ribs are evenly over the dish, the first and the last 12 mm from the inside of the end
    walls, so there is room left to lift the soap out with a fingernail."""
    s = numbers(**kw)
    assert s["rib_x"][0] == pytest.approx(-0.5 * s["length"] + s["wall"] + soap_dish.RIB_END)
    assert s["rib_x"][-1] == pytest.approx(0.5 * s["length"] - s["wall"] - soap_dish.RIB_END)
    step = (s["rib_x"][-1] - s["rib_x"][0]) / (s["ribs"] - 1)
    for i, x in enumerate(s["rib_x"]):
        assert x == pytest.approx(s["rib_x"][0] + i * step), "the ribs are not evenly spaced"


def test_the_ribs_keep_six_mm_between_them():
    """Ribs that stand closer together than `RIB_GAP` millimetres clog with soap and drain nothing,
    so fewer ribs are stood up than were asked for."""
    for length in (70.0, 90.0, 110.0, 160.0):
        for wall in (soap_dish.WALL_MIN, soap_dish.WALL_MAX):
            s = soap_dish.sizes(length=length, wall=wall, ribs=soap_dish.RIBS_MAX)
            assert s["ribs"] <= soap_dish.RIBS_MAX
            step = s["rib_x"][1] - s["rib_x"][0]
            assert step - soap_dish.RIB_W >= soap_dish.RIB_GAP - 1e-9, "the ribs stand too close"
            assert step == pytest.approx((s["rib_x"][-1] - s["rib_x"][0]) / (s["ribs"] - 1))
    assert soap_dish.sizes(length=160.0, wall=soap_dish.WALL_MIN,
                           ribs=soap_dish.RIBS_MAX)["ribs"] == soap_dish.RIBS_MAX


# ------------------------------------------------------------------ the wall

@pytest.mark.parametrize("kw", SIZES)
def test_the_wall_is_as_thick_as_asked(kw):
    """The wall of the tray is `wall` millimetres thick all round, and the rim stands on it to the
    height it was given: the basin is cut out of the dish from the floor up."""
    s, m = numbers(**kw), part(**kw)
    in_wall = 0.5 * s["length"] - 0.5 * s["wall"]
    in_side = 0.5 * s["width"] - 0.5 * s["wall"]
    assert overlap(m, post(in_wall, 0.0, 0.2, s["rim"] - 0.2)) > 0.0, "the wall is not there"
    assert overlap(m, post(in_wall + 0.5 * s["wall"] + 0.5 * PROBE + 0.3, 0.0,
                           0.2, s["rim"] - 0.2)) == pytest.approx(0.0, abs=SLIVER), \
        "the wall is thicker than asked"
    assert overlap(m, post(0.0, in_side, 0.2, s["rim"] - 0.2)) > 0.0, "the side wall is not there"
    assert overlap(m, post(0.0, in_side + 0.5 * s["wall"] + 0.5 * PROBE + 0.3,
                           0.2, s["rim"] - 0.2)) == pytest.approx(0.0, abs=SLIVER), \
        "the side wall is thicker than asked"


# ------------------------------------------------------------------ printing

@pytest.mark.parametrize("kw", SIZES + (dict(ribs=soap_dish.RIBS_MAX),
                                        dict(length=70.0, rim=8.0, wall=4.0, ribs=3)))
def test_every_face_is_vertical_or_horizontal(kw):
    """The dish is a tray of extrusions and boxes, so no face leans and it prints flat on its floor
    without a gram of support."""
    n, area = normals(part(**kw))
    keep = area > 1e-9
    vertical = np.abs(n[:, 2]) < 1e-6
    horizontal = np.abs(n[:, 2]) > 1.0 - 1e-6
    assert (vertical | horizontal)[keep].all(), "a face leans, so the dish would overhang"


# ------------------------------------------------------------------ the numbers

def test_sizes_reports_the_dish_it_makes():
    s = soap_dish.sizes(length=130.0, width=90.0, rim=20.0, wall=3.0, ribs=5)
    assert s["length"] == 130.0 and s["width"] == 90.0 and s["rim"] == 20.0
    assert s["wall"] == 3.0 and s["ribs"] == 5
    assert len(s["rib_x"]) == 5
    assert s["rib_x"][0] == pytest.approx(-0.5 * 130.0 + 3.0 + soap_dish.RIB_END)
    assert s["size_x"] == pytest.approx(130.0)
    assert s["size_y"] == pytest.approx(90.0)
    assert s["size_z"] == pytest.approx(20.0)
    assert set(s) == {"length", "width", "rim", "wall", "ribs", "rib_x",
                      "size_x", "size_y", "size_z"}


def test_the_default_dish_is_for_a_bar_of_soap():
    s = soap_dish.sizes()
    assert s["length"] == 110.0 and s["width"] == 75.0 and s["rim"] == 15.0
    assert s["wall"] == 2.4 and s["ribs"] == 7


def test_the_numbers_are_clamped_to_what_prints():
    small = numbers(length=0.0, width=0.0, rim=0.0, wall=0.0, ribs=0)
    assert small["length"] == soap_dish.LENGTH_MIN and small["width"] == soap_dish.WIDTH_MIN
    assert small["rim"] == soap_dish.RIM_MIN and small["wall"] == soap_dish.WALL_MIN
    assert small["ribs"] == soap_dish.RIBS_MIN
    big = numbers(length=999.0, width=999.0, rim=999.0, wall=999.0, ribs=999)
    assert big["length"] == soap_dish.LENGTH_MAX and big["width"] == soap_dish.WIDTH_MAX
    assert big["rim"] == soap_dish.RIM_MAX and big["wall"] == soap_dish.WALL_MAX
    assert big["ribs"] <= soap_dish.RIBS_MAX
    for kw in (dict(length=0.0, width=0.0, rim=0.0, wall=0.0, ribs=0),
               dict(length=999.0, width=999.0, rim=999.0, wall=999.0, ribs=999)):
        s, m = numbers(**kw), part(**kw)
        lo, hi = bbox(m)
        assert hi[0] - lo[0] == pytest.approx(s["length"], abs=ABS), "a clamped length was lost"
        assert hi[1] - lo[1] == pytest.approx(s["width"], abs=ABS), "a clamped width was lost"
        assert hi[2] - lo[2] == pytest.approx(s["rim"], abs=ABS), "a clamped rim was lost"
        assert len(s["rib_x"]) == s["ribs"], "a clamped rib count was lost"


@pytest.mark.parametrize("bad", (float("nan"), float("inf"), None, "wide"))
def test_a_number_that_is_no_number_is_refused(bad):
    for key in ("length", "width", "rim", "wall", "ribs"):
        with pytest.raises(ValueError):
            soap_dish.soap_dish(**dict(BASE, **{key: bad}))
        with pytest.raises(ValueError):
            soap_dish.sizes(**dict(BASE, **{key: bad}))


# ------------------------------------------------------------------ the interface

def test_the_dialog_builds_a_dish_and_remembers_it(monkeypatch, tmp_path):
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
        dlg.length.setValue(130.0)
        dlg.width.setValue(90.0)
        dlg.rim.setValue(20.0)
        dlg.wall.setValue(3.0)
        dlg.ribs.setValue(5)
        assert "130.0" in dlg.info.text(), "the dialog does not show how long the dish is"
        assert "90.0" in dlg.info.text(), "the dialog does not show how wide the dish is"
        return QDialog.Accepted

    monkeypatch.setattr(app_module.QDialog, "exec", fill)
    try:
        w.soap_dish_dialog()
        deadline = time.monotonic() + 20.0
        while len(w.doc.groups) < 1 and time.monotonic() < deadline:
            app.processEvents()
            time.sleep(0.005)
        assert len(w.doc.groups) == 1, "no dish in the model"
        assert [g.name for g in w.doc.groups] == ["Soap dish"], "the group has another name"
        s = soap_dish.sizes(length=130.0, width=90.0, rim=20.0, wall=3.0, ribs=5)
        lo, hi = np.asarray(w.doc.groups[0].bbox(), dtype=float)
        assert lo[2] == pytest.approx(0.0, abs=1e-6), "the dish does not stand on the bed"
        assert hi[0] - lo[0] == pytest.approx(s["length"], abs=ABS), "not as long as computed"
        assert hi[1] - lo[1] == pytest.approx(s["width"], abs=ABS), "not as wide as computed"
        assert hi[2] - lo[2] == pytest.approx(s["rim"], abs=ABS), "not as high as computed"
        assert w.doc.groups[0].ents.to_manifold().genus() == s["ribs"] - 1, \
            "the slots through the floor are not the only holes"
        settings = app_module.Settings()
        assert (settings.soap_dish_length, settings.soap_dish_width, settings.soap_dish_rim,
                settings.soap_dish_wall, settings.soap_dish_ribs) == (130.0, 90.0, 20.0, 3.0, 5)
        labels = [a.text().replace("&", "").replace("…", "") for a in w.flat_actions(w.menuBar())]
        assert labels.index("Soap Dish") == labels.index("Door Wedge") + 1
    finally:
        for key, value in (("soap_dish_length", 110.0), ("soap_dish_width", 75.0),
                           ("soap_dish_rim", 15.0), ("soap_dish_wall", 2.4),
                           ("soap_dish_ribs", 7)):
            setattr(w.settings, key, value)
        w.settings.save()
        w.doc.dirty = False
        w.close()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))

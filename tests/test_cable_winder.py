"""Cable winders: an H-shaped plate that prints flat and that you wind a cable round.

`cable_winder.cable_winder` gives the winder as one closed volume lying on `z = 0`: a plate of
`length` by `width` by `thickness` millimetres with a slot of `slot` millimetres wide and `depth`
millimetres deep cut in the middle of each short end, so the cable winds round the middle part
between the two slots. The outside corners round with `CORNER` millimetres and the closed end of
every slot with half the slot width; an optional keyhole clamps the end of the cable and an
optional label engraves itself in the top face.

These tests weigh the winder against that: one volume on the bed and centred; the outside as long,
wide and thick as asked; an open slot in each end and plastic in the middle part; the slots as wide
and as deep as asked; the slot ends and the outside corners rounded, so a probe in a corner meets
nothing; the clamp there when it is on and gone when it is off; a label that lowers the top face
over its letters and leaves one volume; every face vertical or horizontal, so nothing overhangs;
every range reached by clamping, the slot by the width and the depth by the length; and a number
that is no number refused.

Run with:  python -m pytest tests/test_cable_winder.py
"""
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import manifold3d as mf  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from PySide6.QtWidgets import QApplication  # noqa: E402

from volura import cable_winder, geom  # noqa: E402

_app = QApplication.instance() or QApplication(sys.argv[:1])   # the letters of a label need it

BASE = dict(length=80.0, width=40.0, thickness=3.0, slot=10.0, depth=15.0, keyhole=True, label="")
#: the round ends are polygons, so a part is at most this far off
ABS = 0.05
#: mm across a probe that feels for plastic or for an open slot
PROBE = 0.4


def part(**kw) -> mf.Manifold:
    return cable_winder.cable_winder(**dict(BASE, **kw))


def numbers(**kw) -> dict:
    return cable_winder.sizes(**dict(BASE, **kw))


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


def post(x, y, z0, z1, r=PROBE) -> mf.Manifold:
    """A round post of radius `r` at (`x`, `y`), between the heights `z0` and `z1`."""
    return mf.Manifold.cylinder(z1 - z0, r, r, 32).translate((x, y, z0))


def slab(m, z0, z1) -> float:
    """The volume of the part between the heights `z0` and `z1`."""
    return (m ^ mf.Manifold.cube([1e4, 1e4, z1 - z0]).translate((-5e3, -5e3, z0))).volume()


def normals(m):
    """The unit normals and the areas of all triangles, pointing out of the part."""
    v, t = geom.mesh_arrays(m)
    a, b, c = v[t[:, 0]], v[t[:, 1]], v[t[:, 2]]
    n = np.cross(b - a, c - a)
    area = 0.5 * np.linalg.norm(n, axis=1)
    return n / np.maximum(2.0 * area[:, None], 1e-30), area


# ------------------------------------------------------------------ the piece

@pytest.mark.parametrize("keyhole", (False, True))
@pytest.mark.parametrize("size", ((40.0, 20.0, 2.0), (80.0, 40.0, 3.0), (200.0, 100.0, 6.0)))
def test_the_winder_is_one_volume_that_lies_on_the_bed_and_is_centred(keyhole, size):
    length, width, thickness = size
    kw = dict(length=length, width=width, thickness=thickness, keyhole=keyhole)
    m = part(**kw)
    assert m.status() == mf.Error.NoError and not m.is_empty()
    assert len(m.decompose()) == 1, "the winder fell apart"
    lo, hi = bbox(m)
    assert lo[2] == pytest.approx(0.0, abs=1e-6), "the winder does not lie on the bed"
    assert np.allclose(lo[:2], -0.5 * np.array([length, width]), atol=ABS), "not centred in x and y"
    assert np.allclose(hi[:2], 0.5 * np.array([length, width]), atol=ABS), "not the size asked"


def test_the_outside_is_its_length_by_its_width_by_its_thickness():
    s, m = numbers(), part()
    lo, hi = bbox(m)
    assert hi[0] - lo[0] == pytest.approx(s["length"], abs=ABS), "not as long as asked"
    assert hi[1] - lo[1] == pytest.approx(s["width"], abs=ABS), "not as wide as asked"
    assert hi[2] - lo[2] == pytest.approx(s["thickness"], abs=ABS), "not as thick as asked"


# ------------------------------------------------------------------ the slots and the middle part

def test_there_is_an_open_slot_in_each_end_and_plastic_in_the_middle():
    """A probe in the middle of each slot, above the plate's middle line, meets nothing; a probe
    in the middle part between the two slots meets plastic."""
    s, m = numbers(), part()
    half = 0.5 * s["length"]
    for sign in (1.0, -1.0):
        x = sign * (half - 0.5 * s["depth"])         # halfway along a slot
        assert not touches(m, post(x, 0.25 * s["slot"], 0.5, s["thickness"] - 0.5)), \
            "the slot is not open"
    assert touches(m, post(0.0, 0.0, 0.5, s["thickness"] - 0.5)), "the middle part is not plastic"


def test_the_slots_are_as_wide_and_as_deep_as_asked():
    s, m = numbers(), part()
    half = 0.5 * s["length"]
    x = half - 0.5 * s["depth"]                       # in the straight part of the +x slot
    assert not touches(m, post(x, 0.5 * s["slot"] - 0.4, 0.5, s["thickness"] - 0.5)), \
        "the slot is narrower than asked"
    assert touches(m, post(x, 0.5 * s["slot"] + 0.4, 0.5, s["thickness"] - 0.5)), \
        "the slot is wider than asked"
    # the -x slot carries no clamp, so the plate is whole just past its bottom
    bottom = -half + s["depth"]                       # the closed end of the -x slot
    assert not touches(m, post(bottom - 0.4, 0.0, 0.5, s["thickness"] - 0.5)), \
        "the slot is shallower than asked"
    assert touches(m, post(bottom + 0.4, 0.0, 0.5, s["thickness"] - 0.5)), \
        "the slot is deeper than asked"


# ------------------------------------------------------------------ the roundings

def test_the_outside_corners_are_rounded():
    """A probe in the corner, inside the radius, meets nothing; a probe on the flat meets plastic."""
    s, m = numbers(), part()
    half_l, half_w = 0.5 * s["length"], 0.5 * s["width"]
    for sx in (1.0, -1.0):
        for sy in (1.0, -1.0):
            assert not touches(m, post(sx * (half_l - 0.5), sy * (half_w - 0.5), 0.5,
                                       s["thickness"] - 0.5)), "the corner is not rounded away"
    assert touches(m, post(half_l - cable_winder.CORNER - 0.5, half_w - 0.5, 0.5,
                           s["thickness"] - 0.5)), "the flat beside the corner is gone"


def test_the_slot_ends_are_rounded():
    """The closed end of a slot is a half circle: the plate is whole where a square end would have
    left a corner, so a probe there meets plastic while the middle of the end stays open."""
    s, m = numbers(), part()
    x = -0.5 * s["length"] + s["depth"] - 0.4         # just inside the bottom of the -x slot
    assert not touches(m, post(x, 0.0, 0.5, s["thickness"] - 0.5)), "the slot does not reach its depth"
    assert touches(m, post(x, 0.5 * s["slot"] - 0.4, 0.5, s["thickness"] - 0.5)), \
        "the slot end is square, not round"


# ------------------------------------------------------------------ the clamp

def test_the_clamp_is_there_when_it_is_on_and_gone_when_it_is_off():
    s = numbers()
    x = 0.5 * s["length"] - s["depth"] - 0.5 * cable_winder.KEYHOLE_LEN   # in the keyhole
    on = part(keyhole=True)
    off = part(keyhole=False)
    assert not touches(on, post(x, 0.0, 0.5, s["thickness"] - 0.5)), "the clamp is not cut"
    assert touches(off, post(x, 0.0, 0.5, s["thickness"] - 0.5)), "the clamp is cut when it is off"


# ------------------------------------------------------------------ the label

def test_a_label_lowers_the_top_face_and_leaves_one_volume():
    from volura import fonts
    if not fonts.has_fonts():
        pytest.skip("this system has no fonts, so there are no letters to engrave")
    s, plain = numbers(), part()
    made = part(label="VOLURA")
    assert len(made.decompose()) == 1, "the engraving cut the winder in two"
    assert slab(made, s["thickness"] - 0.55, s["thickness"]) < slab(plain, s["thickness"] - 0.55,
                                                                    s["thickness"]), \
        "the letters do not lower the top face"
    assert slab(made, 0.1, s["thickness"] - cable_winder.ENGRAVE - 0.05) == pytest.approx(
        slab(plain, 0.1, s["thickness"] - cable_winder.ENGRAVE - 0.05), abs=1e-6), \
        "the engraving cuts deeper than it should"


@pytest.mark.parametrize("keyhole", (True, False))
def test_a_long_label_stays_clear_of_the_cable_clamp(keyhole):
    from volura import fonts
    if not fonts.has_fonts():
        pytest.skip("this system has no fonts, so there are no letters to engrave")
    s = numbers(keyhole=keyhole)
    gone = part(keyhole=keyhole) - part(keyhole=keyhole, label="A LONG CABLE FOR THE LAPTOP")
    assert not gone.is_empty(), "nothing was engraved"
    b = np.asarray(gone.bounding_box(), dtype=float)
    room = 0.5 * s["middle"] - cable_winder.LABEL_MARGIN \
        - (cable_winder.KEYHOLE_LEN if keyhole else 0.0)
    assert b[3] - b[0] <= 2.0 * room + 1e-3, "the letters run into the clamp or past the middle"


# ------------------------------------------------------------------ no overhang

@pytest.mark.parametrize("kw", (dict(), dict(keyhole=False), dict(length=40.0, width=20.0),
                                dict(length=200.0, width=100.0, slot=30.0, depth=60.0)))
def test_every_face_is_vertical_or_horizontal(kw):
    """The winder prints lying flat: no face leans, so nothing overhangs and nothing needs support."""
    n, area = normals(part(**kw))
    keep = area > 1e-9
    vertical = np.abs(n[:, 2]) < 1e-6
    horizontal = np.abs(n[:, 2]) > 1.0 - 1e-6
    assert (vertical | horizontal)[keep].all(), "a face leans, so the winder would overhang"


# ------------------------------------------------------------------ the numbers

def test_sizes_reports_the_plate_and_the_middle_part():
    s = cable_winder.sizes(length=120.0, width=60.0, thickness=4.0, slot=20.0, depth=30.0,
                           keyhole=False, label="tools")
    assert s["length"] == 120.0 and s["width"] == 60.0 and s["thickness"] == 4.0
    assert s["slot"] == 20.0 and s["depth"] == 30.0 and s["keyhole"] is False
    assert s["label"] == "tools" and s["middle"] == pytest.approx(60.0)
    assert set(s) == {"length", "width", "thickness", "slot", "depth", "keyhole", "label", "middle"}


def test_the_default_winder_winds_on_a_fifty_millimetre_barrel():
    s = cable_winder.sizes()
    assert s["middle"] == pytest.approx(50.0)


def test_the_numbers_are_clamped_to_what_prints():
    small = numbers(length=0.0, width=0.0, thickness=0.0, slot=0.0, depth=0.0)
    assert small["length"] == cable_winder.LENGTH_MIN and small["width"] == cable_winder.WIDTH_MIN
    assert small["thickness"] == cable_winder.THICKNESS_MIN
    assert small["slot"] == cable_winder.SLOT_MIN and small["depth"] == cable_winder.DEPTH_MIN
    big = numbers(length=999.0, width=999.0, thickness=999.0, slot=999.0, depth=999.0)
    assert big["length"] == cable_winder.LENGTH_MAX and big["width"] == cable_winder.WIDTH_MAX
    assert big["thickness"] == cable_winder.THICKNESS_MAX
    assert big["slot"] == cable_winder.SLOT_MAX and big["depth"] == cable_winder.DEPTH_MAX
    lo, hi = bbox(part(length=0.0, width=0.0, thickness=0.0, slot=0.0, depth=0.0))
    assert hi[2] - lo[2] == pytest.approx(cable_winder.THICKNESS_MIN, abs=ABS)


def test_the_slot_is_limited_by_the_width_and_the_depth_by_the_length():
    """A slot wider than the plate less two flanges is made narrower, and a slot deeper than a
    third of the length is made shallower, so the middle part always stands."""
    narrow = numbers(width=20.0, slot=999.0)
    assert narrow["slot"] == pytest.approx(20.0 - 2.0 * cable_winder.FLANGE)
    short = numbers(length=60.0, depth=999.0)
    assert short["depth"] == pytest.approx(20.0)
    assert short["middle"] == pytest.approx(60.0 - 2.0 * 20.0)
    assert len(bbox(part(width=20.0, slot=999.0))) == 2


@pytest.mark.parametrize("bad", (float("nan"), float("inf"), None, "wide"))
def test_a_number_that_is_no_number_is_refused(bad):
    for key in ("length", "width", "thickness", "slot", "depth"):
        with pytest.raises(ValueError):
            cable_winder.cable_winder(**dict(BASE, **{key: bad}))
        with pytest.raises(ValueError):
            cable_winder.sizes(**dict(BASE, **{key: bad}))


# ------------------------------------------------------------------ the interface

def test_the_dialog_builds_a_winder_and_remembers_it(monkeypatch, tmp_path):
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
        dlg.length.setValue(120.0)
        dlg.width.setValue(60.0)
        dlg.thickness.setValue(4.0)
        dlg.slot.setValue(20.0)
        dlg.depth.setValue(30.0)
        assert dlg.keyhole.isChecked(), "the cable clamp is off by default"
        dlg.keyhole.setChecked(False)
        assert "60.0" in dlg.info.text(), "the dialog does not show the middle part it leaves"
        return QDialog.Accepted

    monkeypatch.setattr(app_module.QDialog, "exec", fill)
    try:
        w.cable_winder_dialog()
        deadline = time.monotonic() + 20.0
        while len(w.doc.groups) < 1 and time.monotonic() < deadline:
            app.processEvents()
            time.sleep(0.005)
        assert len(w.doc.groups) == 1, "no winder in the model"
        assert [g.name for g in w.doc.groups] == ["Cable winder"]
        s = cable_winder.sizes(length=120.0, width=60.0, thickness=4.0, slot=20.0, depth=30.0,
                               keyhole=False)
        lo, hi = np.asarray(w.doc.groups[0].bbox(), dtype=float)
        assert lo[2] == pytest.approx(0.0, abs=1e-6), "the winder does not lie on the bed"
        assert hi[0] - lo[0] == pytest.approx(s["length"], abs=1e-3), "not as long as computed"
        assert hi[1] - lo[1] == pytest.approx(s["width"], abs=1e-3), "not as wide as computed"
        settings = app_module.Settings()
        assert (settings.cable_winder_length, settings.cable_winder_width,
                settings.cable_winder_thickness, settings.cable_winder_slot,
                settings.cable_winder_depth, settings.cable_winder_keyhole) == (
            120.0, 60.0, 4.0, 20.0, 30.0, False)
        labels = [a.text().replace("&", "").replace("…", "") for a in w.flat_actions(w.menuBar())]
        assert labels.index("Cable Winder") == labels.index("Plant Pot") + 1
    finally:
        for key, value in (("cable_winder_length", 80.0), ("cable_winder_width", 40.0),
                           ("cable_winder_thickness", 3.0), ("cable_winder_slot", 10.0),
                           ("cable_winder_depth", 15.0), ("cable_winder_keyhole", True),
                           ("cable_winder_label", "")):
            setattr(w.settings, key, value)
        w.settings.save()
        w.doc.dirty = False
        w.close()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))

"""Zip-tie mounts: a printed cable-tie anchor that is screwed onto a surface.

`tie_mount.tie_mount` gives the mount as one closed volume that lies on its base on `z = 0` and is
centred on the origin: a plate of `base` by `base` with a screw through its middle, and a bridge
over it with a tunnel of `tie_width + 0.6` by `tie_thick + 0.6` through it for the tie, along X and
for `ways` 2 also along Y. The tunnel closes on a roof of 45 degrees and the screw goes in from
above through the bridge, with its head in the top of the base under the tunnel.

These tests weigh the part against that: it is one piece that lies on the bed and is as big as its
plate, a tie slides through every tunnel without touching anything, the screw hole is open from the
top of the bridge to the bed, the head of a sunk screw fits flush in the floor, and nothing hangs
over steeper than 45 degrees, because the mount prints without support.

Run with:  python -m pytest tests/test_tie_mount.py
"""
import math
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import manifold3d as mf  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from volura import geom, tie_mount, threads  # noqa: E402

BASE = dict(base=24.0)
#: the rounded corners, the chamfers and the holes are polygons, so a part is at most this far off
ABS = 0.05


def part(**kw) -> mf.Manifold:
    return tie_mount.tie_mount(**dict(BASE, **kw))


def numbers(**kw) -> dict:
    return tie_mount.sizes(**dict(BASE, **kw))


def bbox(m):
    """The bounding box of a manifold as (low, high)."""
    b = np.asarray(m.bounding_box(), dtype=float)
    return b[:3], b[3:]


def plate_of(screw, countersink) -> float:
    """How thick the base is under a mount for `screw`: a sunk head needs a thicker plate."""
    hole_d, _head_d, _head_h, cs_head_d = threads.SCREW_HOLES[screw]
    cone = 0.5 * (cs_head_d - hole_d)
    return max(tie_mount.BASE, cone + tie_mount.UNDER) if countersink else tie_mount.BASE


def floor_of(s) -> float:
    """The floor the tie lies on: the top of the base, with the play of PLAY under the tunnel."""
    return plate_of(s["screw"], s["countersink"]) + tie_mount.PLAY


def open_of(s) -> float:
    """How wide the opening through the top of the bridge is: the head, with room to pass."""
    _hole_d, head_d, _head_h, cs_head_d = threads.SCREW_HOLES[s["screw"]]
    if not s["countersink"]:
        return head_d + tie_mount.HEAD_ROOM
    # the cone of a sunk head is cut with room past the floor and opens at 45 degrees on its way
    return max(cs_head_d + tie_mount.HEAD_ROOM,
               cs_head_d + threads.DEFAULT_HOLE_CLEARANCE + 2.0 * tie_mount.MARGIN)


def rod(along, across, length, x, y, z) -> mf.Manifold:
    """A square rod `across` wide and `length` long along `along`, centred on (x, y, z)."""
    d = {"x": [length, across, across], "y": [across, length, across],
         "z": [across, across, length]}[along]
    return mf.Manifold.cube(d, True).translate((x, y, z))


def tie(along, s, width, thick) -> mf.Manifold:
    """A cable tie `width` by `thick`, long past the mount, lying on the floor of its tunnel."""
    d = {"x": [4.0 * s["base"], width, thick], "y": [width, 4.0 * s["base"], thick]}[along]
    return mf.Manifold.cube(d, True).translate((0.0, 0.0, floor_of(s) + 0.5 * thick))


def disc(diameter, z0, z1) -> mf.Manifold:
    """A round rod `diameter` across, from `z0` to `z1`, on the axis of the screw."""
    return (mf.Manifold.cylinder(z1 - z0, 0.5 * diameter, 0.5 * diameter, 64)
            .translate((0.0, 0.0, z0)))


def head_cone(screw, wider=0.0) -> mf.Manifold:
    """The 90 degree head of a flat screw of `screw`, as it sits in the top of the base.

    Its top is level with the top of the base and it narrows towards the hole at 45 degrees;
    `wider` makes the whole cone that much wider, so it runs into the plastic beside the cone
    that is cut.
    """
    hole_d, _head_d, _head_h, cs_head_d = threads.SCREW_HOLES[screw]
    r_hole, r_head = 0.5 * hole_d + wider, 0.5 * cs_head_d + wider
    return mf.Manifold.cylinder(r_head - r_hole, r_hole, r_head, 64).translate(
        (0.0, 0.0, plate_of(screw, True) - (r_head - r_hole)))


def touches(m, probe) -> bool:
    """Whether a probe runs into the part anywhere.

    Two solids that only touch leave a sliver of a few ten-millionths of a mm³ behind, so that is
    not a touch; a probe that does not fit is off by whole millimetres.
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

@pytest.mark.parametrize("base", (12.0, 24.0, 40.0))
@pytest.mark.parametrize("ways", (1, 2))
def test_every_mount_is_one_piece_that_lies_on_its_base(base, ways):
    kw = dict(base=base, ways=ways)
    s, m = numbers(**kw), part(**kw)
    assert m.status() == mf.Error.NoError and not m.is_empty()
    assert len(m.decompose()) == 1, "the mount fell apart"
    lo, hi = bbox(m)
    assert np.allclose(lo[:2], -0.5 * s["base"], atol=ABS), "the base is not centred on the origin"
    assert np.allclose(hi[:2], 0.5 * s["base"], atol=ABS), "the base is not as big as asked"
    assert lo[2] == pytest.approx(0.0), "the base does not lie on the bed"
    assert hi[2] == pytest.approx(s["height"], abs=ABS), "the mount does not stand as high as asked"


@pytest.mark.parametrize("screw", tie_mount.SCREWS)
@pytest.mark.parametrize("ways", (1, 2))
@pytest.mark.parametrize("tie_width, tie_thick", ((2.5, 1.0), (5.0, 1.5), (9.0, 2.5)))
def test_the_mount_is_valid_whatever_it_is_asked_for(screw, ways, tie_width, tie_thick):
    m = part(screw=screw, ways=ways, tie_width=tie_width, tie_thick=tie_thick)
    assert m.status() == mf.Error.NoError and not m.is_empty()
    assert len(m.decompose()) == 1, "the mount fell apart"


@pytest.mark.parametrize("countersink", (True, False))
@pytest.mark.parametrize("screw", tie_mount.SCREWS)
def test_the_mount_is_valid_with_a_sunk_head_or_a_straight_hole(countersink, screw):
    m = part(screw=screw, countersink=countersink)
    assert m.status() == mf.Error.NoError and not m.is_empty()
    assert len(m.decompose()) == 1, "the mount fell apart"


# ------------------------------------------------------------------ the tunnel

@pytest.mark.parametrize("ways", (1, 2))
@pytest.mark.parametrize("countersink", (True, False))
def test_the_tie_slides_through_every_tunnel(ways, countersink):
    kw = dict(ways=ways, countersink=countersink)
    s, m = numbers(**kw), part(**kw)
    assert not touches(m, tie("x", s, s["tie_width"], s["tie_thick"])), \
        "the tie does not slide through the tunnel along X"
    if ways == 2:
        assert not touches(m, tie("y", s, s["tie_width"], s["tie_thick"])), \
            "the tie does not slide through the tunnel along Y"
    else:
        assert touches(m, tie("y", s, s["tie_width"], s["tie_thick"])), \
            "the tie goes along Y through a mount that has one tunnel"
    assert touches(m, tie("x", s, s["tie_width"] + 0.8, s["tie_thick"])), \
        "the tunnel is wider than the tie plus its play"
    assert touches(m, tie("x", s, s["tie_width"], s["tie_thick"] + 2.0 * tie_mount.PLAY + 0.2)), \
        "the tunnel is higher than the tie plus its play"


@pytest.mark.parametrize("tie_width, tie_thick", ((2.5, 1.0), (5.0, 1.5), (9.0, 2.5)))
def test_the_tunnel_is_the_tie_with_play_all_round(tie_width, tie_thick):
    s = numbers(tie_width=tie_width, tie_thick=tie_thick)
    assert s["slot_w"] == pytest.approx(tie_width + tie_mount.PLAY)
    assert s["slot_h"] == pytest.approx(tie_thick + tie_mount.PLAY)
    assert s["roof"] >= tie_mount.ROOF - 1e-9, "too little plastic over the roof of the tunnel"


# ------------------------------------------------------------------ the screw

@pytest.mark.parametrize("ways", (1, 2))
@pytest.mark.parametrize("screw", tie_mount.SCREWS)
def test_the_screw_hole_is_open_from_the_top_of_the_bridge_to_the_bed(ways, screw):
    kw = dict(ways=ways, screw=screw)
    s, m = numbers(**kw), part(**kw)
    plate = plate_of(screw, s["countersink"])
    assert not touches(m, rod("z", 0.6, 4.0 * s["base"], 0.0, 0.0, 0.5 * s["height"])), \
        "there is no hole through the middle of the mount"
    assert touches(m, rod("z", s["hole_d"] + 0.6, plate, 0.0, 0.0, 0.5 * plate)), \
        "the hole in the base is wider than the screw hole of the book"
    assert not touches(m, disc(open_of(s) - 0.4, s["height"] - 0.5, s["height"])), \
        "the screwdriver does not come through the top of the bridge"
    assert touches(m, disc(open_of(s) + 0.4, s["height"] - 0.5, s["height"])), \
        "the opening in the top of the bridge is wider than the head"


@pytest.mark.parametrize("screw", tie_mount.SCREWS)
def test_the_head_of_a_sunk_screw_is_flush_in_the_floor_of_the_tunnel(screw):
    s, m = numbers(screw=screw), part(screw=screw)
    assert not touches(m, head_cone(screw)), "the head of a flat screw does not fit in the floor"
    assert touches(m, head_cone(screw, wider=0.4)), "the cone in the floor is wider than the head"
    assert not touches(m, tie("x", s, s["tie_width"], s["tie_thick"])), \
        "the tie does not lie flat over the head"


@pytest.mark.parametrize("screw", tie_mount.SCREWS)
def test_a_straight_hole_leaves_no_room_for_the_head(screw):
    m = part(screw=screw, countersink=False)
    assert touches(m, head_cone(screw)), "a straight hole got a cone for the head anyway"


def test_a_sunk_head_leaves_plastic_under_it():
    """The plate under a sunk head is a millimetre of plastic: the cone ends above the bed."""
    for screw in tie_mount.SCREWS:
        m = part(screw=screw)
        t = plate_of(screw, True)
        hole_d, _head_d, _head_h, cs_head_d = threads.SCREW_HOLES[screw]
        cone, head = 0.5 * (cs_head_d - hole_d), 0.5 * cs_head_d
        assert t - cone >= tie_mount.UNDER - 1e-9
        assert touches(m, disc(2.0 * head - 0.4, t - cone - 0.4, t - cone)), \
            "there is nothing left under the head of a sunk screw"


# ------------------------------------------------------------------ the bridge

@pytest.mark.parametrize("ways", (1, 2))
@pytest.mark.parametrize("screw", tie_mount.SCREWS)
def test_the_bridge_leaves_wall_beside_the_tunnel_and_the_head(ways, screw):
    kw = dict(ways=ways, screw=screw, tie_width=5.0)
    s, m = numbers(**kw), part(**kw)
    assert s["bridge"] == pytest.approx(max(s["slot_w"], open_of(s)) + 2.0 * tie_mount.WALL)
    z = plate_of(screw, s["countersink"]) + 1.0
    assert touches(m, rod("x", 0.4, 4.0 * s["base"], 0.0, 0.5 * s["bridge"] - 0.2, z)), \
        "the bridge is not as wide as sizes says"
    assert not touches(m, rod("x", 0.4, 4.0 * s["base"], 0.0, 0.5 * s["bridge"] + 0.2, z)), \
        "the bridge hangs over the base"
    if ways == 2:
        assert touches(m, rod("y", 0.4, 4.0 * s["base"], 0.5 * s["bridge"] - 0.2, 0.0, z)), \
            "the bridge is not square, so the tie has no wall beside the tunnel along Y"


def test_a_bridge_too_wide_for_the_base_is_made_to_fit_it():
    s = numbers(base=12.0, screw="M5", tie_width=9.0)
    assert s["bridge"] == pytest.approx(12.0 - 2.0 * tie_mount.CHAMFER)
    m = part(base=12.0, screw="M5", tie_width=9.0)
    lo, hi = bbox(m)
    assert np.allclose(hi[:2], 6.0, atol=ABS), "the bridge hangs over the base"
    assert len(m.decompose()) == 1, "the mount fell apart"


def test_the_base_is_a_plate_with_rounded_corners():
    """The plate is a square with rounded corners, and the screw hole goes through it."""
    s, m = numbers(base=30.0), part(base=30.0)
    want = (30.0 ** 2 - (4.0 - math.pi) * tie_mount.CORNER ** 2
            - math.pi * (0.5 * s["hole_d"]) ** 2)
    assert m.slice(0.2).area() == pytest.approx(want, rel=0.01), "the corners are not rounded"


# ------------------------------------------------------------------ printing

@pytest.mark.parametrize("screw", tie_mount.SCREWS)
@pytest.mark.parametrize("ways, countersink", ((1, True), (2, True), (2, False)))
def test_nothing_hangs_over_steeper_than_45_degrees(screw, ways, countersink):
    """The mount stands on its base: the tunnel roofs itself and the cone in the floor is a hole."""
    m = part(screw=screw, ways=ways, countersink=countersink, tie_width=9.0, tie_thick=2.5)
    n, c, area = triangles(m)
    keep = area > 1e-9
    assert (n[keep][c[keep][:, 2] < 1e-9][:, 2] < -0.99).all(), \
        "the triangles do not face out of the part"
    down = keep & (n[:, 2] < -np.sin(np.pi / 4.0) - 0.01) & (c[:, 2] > 1e-6)
    assert not down.any(), "a face overhangs steeper than 45 degrees, so it needs support"


# ------------------------------------------------------------------ the numbers

def test_sizes_reports_the_mount_and_its_tunnel():
    s = tie_mount.sizes(base=30.0, tie_width=6.0, tie_thick=2.0, height=6.0, screw="M5",
                        countersink=True, ways=2)
    assert s["base"] == 30.0 and s["tie_width"] == 6.0 and s["tie_thick"] == 2.0
    assert s["screw"] == "M5" and s["countersink"] is True and s["ways"] == 2
    assert s["slot_w"] == pytest.approx(6.6) and s["slot_h"] == pytest.approx(2.6)
    assert s["hole_d"] == pytest.approx(threads.SCREW_HOLES["M5"][0] +
                                        threads.DEFAULT_HOLE_CLEARANCE)
    assert s["bridge"] == pytest.approx(open_of(s) + 2.0 * tie_mount.WALL)
    assert s["roof"] >= tie_mount.ROOF - 1e-9
    assert set(s) == {"base", "tie_width", "tie_thick", "height", "screw", "countersink", "ways",
                      "slot_w", "slot_h", "bridge", "roof", "hole_d"}


def test_the_height_keeps_roof_over_the_peak_of_the_tunnel():
    """The tunnel is high, so a mount that was asked to be low is made as tall as the roof needs."""
    s = numbers(tie_width=5.0, tie_thick=1.5, height=2.0)
    assert s["height"] > 2.0, "a mount of 2 mm was taken for the height it asks"
    assert s["roof"] == pytest.approx(tie_mount.ROOF), "the height was not raised to the roof"
    tall = numbers(height=99.0)
    assert tall["height"] == tie_mount.HEIGHT_MAX and tall["roof"] > tie_mount.ROOF


def test_the_numbers_are_clamped_to_what_prints():
    small = numbers(base=1.0, tie_width=0.2, tie_thick=0.1, height=1.0, ways=0)
    assert small["base"] == tie_mount.BASE_MIN
    assert small["tie_width"] == tie_mount.TIE_W_MIN and small["tie_thick"] == tie_mount.TIE_T_MIN
    assert small["ways"] == tie_mount.WAYS_MIN and small["height"] > 1.0
    big = numbers(base=999.0, tie_width=99.0, tie_thick=99.0, height=99.0, ways=9)
    assert big["base"] == tie_mount.BASE_MAX
    assert big["tie_width"] == tie_mount.TIE_W_MAX and big["tie_thick"] == tie_mount.TIE_T_MAX
    assert big["ways"] == tie_mount.WAYS_MAX and big["height"] == tie_mount.HEIGHT_MAX
    lo, hi = bbox(part(base=1.0, tie_width=0.2, tie_thick=0.1, height=1.0, ways=0))
    assert hi[0] - lo[0] == pytest.approx(tie_mount.BASE_MIN, abs=ABS), \
        "the smallest mount is not the smallest base"


@pytest.mark.parametrize("bad", (float("nan"), float("inf"), None, "wide"))
def test_a_number_that_is_no_number_is_refused(bad):
    for key in ("base", "tie_width", "tie_thick", "height", "ways"):
        with pytest.raises(ValueError):
            tie_mount.tie_mount(**dict(BASE, **{key: bad}))
        with pytest.raises(ValueError):
            tie_mount.sizes(**dict(BASE, **{key: bad}))


@pytest.mark.parametrize("screw", ("M12", "2.5", "", None, "m4"))
def test_a_screw_that_is_not_in_the_book_is_refused(screw):
    with pytest.raises(ValueError):
        tie_mount.tie_mount(**dict(BASE, screw=screw))
    with pytest.raises(ValueError):
        tie_mount.sizes(**dict(BASE, screw=screw))


# ------------------------------------------------------------------ the interface

def test_the_dialog_builds_a_mount_and_remembers_it(monkeypatch, tmp_path):
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
        dlg.base.setValue(30.0)
        dlg.tie_width.setValue(4.0)
        dlg.ways.setValue(1)
        assert dlg.countersink.isChecked(), "the head is sunk along with the mount"
        assert "Tunnel 4.6" in dlg.info.text(), "the dialog does not show the tunnel it makes"
        return QDialog.Accepted

    monkeypatch.setattr(app_module.QDialog, "exec", fill)
    try:
        w.tie_mount_dialog()
        deadline = time.monotonic() + 15.0
        while not w.doc.groups and time.monotonic() < deadline:
            app.processEvents()
            time.sleep(0.005)
        assert len(w.doc.groups) == 1, "no mount in the model"
        lo, hi = np.asarray(w.doc.groups[0].bbox(), dtype=float)
        s = tie_mount.sizes(base=30.0, tie_width=4.0, ways=1)
        assert lo[2] == pytest.approx(0.0, abs=1e-6), "the mount does not lie on the bed"
        assert hi[2] - lo[2] == pytest.approx(s["height"], abs=1e-4)
        assert hi[0] - lo[0] == pytest.approx(30.0, abs=1e-4), "the base is not as big as asked"
        settings = app_module.Settings()
        assert (settings.tie_mount_base, settings.tie_mount_tie_width, settings.tie_mount_ways) == (
            pytest.approx(30.0), pytest.approx(4.0), 1)
        labels = [a.text().replace("&", "").replace("…", "") for a in w.flat_actions(w.menuBar())]
        assert labels.index("Zip-Tie Mount") == labels.index("Corner Bracket") + 1
    finally:
        w.settings.tie_mount_base, w.settings.tie_mount_tie_width, w.settings.tie_mount_ways = (
            20.0, 5.0, 2)
        w.settings.save()
        w.doc.dirty = False
        w.close()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))

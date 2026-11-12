"""Phone stands: a stand a phone or a tablet leans in, printed lying on its side.

`phone_stand.phone_stand` gives the stand as one closed volume that lies on its side on `z = 0` and
is centred in x and y: a base of `wall` millimetres on the bed, a back support that leans `angle`
degrees back from the bed and rises `back` millimetres up that lean with a half-round top, and in
front of it a lip that comes `lip` millimetres up above the ledge, so the slot between support and
lip is a millimetre wider than the device it holds. The length of the base is computed from the
lean, so the centre of a leaning tablet stays over the base, and a gap of 12 millimetres in the
middle of the width lets a charging cable run in behind the stand and up to the foot of the phone.

These tests weigh the part against that: it is one piece that lies on the bed and is as big as its
base, its support and its height, a phone of `device` millimetres fits in the slot and a thicker
one does not, the support leans at the angle it asks, the lip is as high above the ledge as asked,
the cable slot is open through the base and the lip when it is on and closed when it is off or the
stand is too narrow for it, a leaning device stays over the base at the extremes, every range is
reached by clamping and nothing hangs over while there is no cable slot, because the profile is
pulled straight up from the bed.

Run with:  python -m pytest tests/test_phone_stand.py
"""
import math
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import manifold3d as mf  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from volura import geom, phone_stand  # noqa: E402
from volura.phone_stand import WALL  # noqa: E402

BASE = dict(width=70.0, angle=65.0, back=90.0, device=11.0, lip=10.0)
#: the fillet and the rounded top are polygons, so a part is at most this far off
ABS = 0.05


def part(**kw) -> mf.Manifold:
    return phone_stand.phone_stand(**dict(BASE, **kw))


def numbers(**kw) -> dict:
    return phone_stand.sizes(**dict(BASE, **kw))


def bbox(m):
    """The bounding box of a manifold as (low, high)."""
    b = np.asarray(m.bounding_box(), dtype=float)
    return b[:3], b[3:]


def touches(m, probe) -> bool:
    """Whether a probe runs into the part anywhere.

    Two solids that only touch leave a sliver of a few ten-millionths of a mm³ behind, so that is
    not a touch; a phone that does not fit is off by whole tenths of a millimetre.
    """
    return (m ^ probe).volume() > 1e-6


def box(s, x0, x1, y0, y1, z0, z1) -> mf.Manifold:
    """A box between two corners, in the frame the profile is drawn in.

    The profile is drawn with the back edge of the base on `x = 0` and the bed on `y = 0`, while
    the stand comes back centred in x and y, so every probe is drawn the way the profile is and
    moved along with it.
    """
    return mf.Manifold.cube([x1 - x0, y1 - y0, z1 - z0]).translate(
        (x0 - 0.5 * s["depth"], y0 - 0.5 * s["height"], z0))


def lean(s):
    """The tangent of the lean: how far a face falls back for every millimetre it rises."""
    return 1.0 / math.tan(math.radians(s["angle"]))


def support_front(s, y) -> float:
    """Where the front of the support is at height `y`, in the frame of the profile."""
    return s["support_x"] - (y - WALL) * lean(s)


def lip_side(s, y) -> float:
    """Where the side of the lip that looks into the slot is at height `y`."""
    return s["support_x"] + s["slot"] / math.sin(math.radians(s["angle"])) - (y - WALL) * lean(s)


def slot_probe(s, thick, up=3.0, length=2.0) -> mf.Manifold:
    """A slab `thick` thick that leans in the slot with its back against the support.

    It stands `up` millimetres up the lean and reaches `length` millimetres further, so it feels
    the slot itself and not the fillet in its corner, and it stays beside the cable slot. Its back
    is a twentieth of a millimetre off the support, so it does not touch the face it leans on.
    """
    a = math.radians(s["angle"])
    corner = (s["support_x"] - up * math.cos(a) + 0.05 * math.sin(a) - 0.5 * s["depth"],
              WALL + up * math.sin(a) + 0.05 * math.cos(a) - 0.5 * s["height"],
              0.15 * s["width"])
    return mf.Manifold.cube([thick, length, 0.2 * s["width"]]) \
        .rotate((0.0, 0.0, 90.0 - s["angle"])).translate(corner)


def back_x(m, s, y) -> float:
    """How far back the support is at height `y`, found by walking a probe in from the back edge.

    Nothing stands behind the support, so the first plastic a probe meets on its way from the back
    edge of the base to the front is the back of the support itself.
    """
    lo, hi = 0.0, s["support_x"]
    while hi - lo > 1e-6:
        mid = 0.5 * (lo + hi)
        if touches(m, box(s, 0.0, mid, y - 0.1, y + 0.1, 0.15 * s["width"], 0.35 * s["width"])):
            hi = mid
        else:
            lo = mid
    return hi


def cable_probes(s):
    """The two probes that say where a cable runs: one through the base, one through the lip.

    Both are 10 millimetres across and stand in the middle of the width, where the gap is, and
    both stop short of the walls beside the gap, so they only run into plastic if the gap is not
    there.
    """
    t, slot, lip = s["wall"], s["slot"], s["lip"]
    front = s["support_x"] + (slot + t) / math.sin(math.radians(s["angle"]))
    back = lip_side(s, WALL + lip)
    z0, z1 = 0.5 * (s["width"] - 10.0), 0.5 * (s["width"] + 10.0)
    return (box(s, 0.5, front - 0.5, 0.5, t - 0.5, z0, z1),
            box(s, back + 0.5, front - 0.5, WALL + 0.5, WALL + lip + 0.5, z0, z1))


def triangles(m):
    """The unit normals, the centres and the areas of all triangles, pointing out of the part."""
    v, t = geom.mesh_arrays(m)
    a, b, c = v[t[:, 0]], v[t[:, 1]], v[t[:, 2]]
    n = np.cross(b - a, c - a)
    area = 0.5 * np.linalg.norm(n, axis=1)
    return n / np.maximum(2.0 * area[:, None], 1e-30), (a + b + c) / 3.0, area


# ------------------------------------------------------------------ the part

@pytest.mark.parametrize("width, angle, back", ((30.0, 50.0, 40.0), (70.0, 65.0, 90.0),
                                                (200.0, 80.0, 160.0)))
@pytest.mark.parametrize("device, lip", ((5.0, 4.0), (11.0, 10.0), (16.0, 20.0)))
def test_the_stand_is_one_piece_that_lies_on_its_side(width, angle, back, device, lip):
    kw = dict(width=width, angle=angle, back=back, device=device, lip=lip)
    s, m = numbers(**kw), part(**kw)
    assert m.status() == mf.Error.NoError and not m.is_empty()
    assert len(m.decompose()) == 1, "the stand fell apart"
    lo, hi = bbox(m)
    assert lo[2] == pytest.approx(0.0, abs=1e-6), "the stand does not lie on the bed"
    assert hi[2] - lo[2] == pytest.approx(s["width"], abs=ABS), "the stand does not stand as high as wide"
    assert hi[0] - lo[0] == pytest.approx(s["depth"], abs=ABS), "the base is not as long as sizes says"
    assert hi[1] - lo[1] == pytest.approx(s["height"], abs=ABS), "the support does not come as high"
    assert lo[0] == pytest.approx(-0.5 * s["depth"], abs=ABS), "the stand is not centred in x"
    assert lo[1] == pytest.approx(-0.5 * s["height"], abs=ABS), "the stand is not centred in y"


@pytest.mark.parametrize("angle, back", ((50.0, 40.0), (65.0, 90.0), (80.0, 160.0)))
def test_the_base_is_under_the_bed_and_the_lip_stands_on_it(angle, back):
    """The bed side of the base is the whole length of the base, and the lip stands on its front
    part: what holds the device is rooted in the plate that lies on the bed."""
    kw = dict(angle=angle, back=back)
    s, m = numbers(**kw), part(**kw)
    assert not touches(m, box(s, -1.0, 0.0, 0.0, s["height"], 0.0, s["width"])), \
        "something reaches past the back of the base"
    assert touches(m, box(s, s["depth"] - 0.5, s["depth"], 0.0, WALL, 0.1 * s["width"],
                          0.9 * s["width"])), "the front of the base is not there"
    assert touches(m, box(s, s["support_x"] - 1.0, s["support_x"], WALL - 0.5, WALL,
                          0.1 * s["width"], 0.9 * s["width"])), "the support does not stand on the base"


# ------------------------------------------------------------------ the slot

@pytest.mark.parametrize("device, lip", ((5.0, 10.0), (11.0, 10.0), (16.0, 20.0)))
def test_the_slot_holds_the_device_it_is_made_for(device, lip):
    """The slot is a millimetre wider than the device, so a phone with its case in it goes in and
    a thicker one does not."""
    s, m = numbers(device=device, lip=lip), part(device=device, lip=lip)
    assert s["slot"] == pytest.approx(device + phone_stand.SLOT_PLAY)
    a = math.radians(s["angle"])
    up = phone_stand.FILLET * math.tan(0.5 * a) + 0.2      # above the fillet in its corner
    tall = (lip - (device + 1.2) * math.cos(a)) / math.sin(a)   # below the top of the lip
    length = tall - up - 0.2
    assert length > 0.5, "this stand leaves no room to measure the slot in"
    assert not touches(m, slot_probe(s, device + 0.8, up, length)), "a device of that size does not fit"
    assert touches(m, slot_probe(s, device + 1.2, up, length)), "the slot is wider than its play"


def test_the_floor_of_the_slot_is_the_top_of_the_base():
    """The device stands on the ledge, which is the top of the base: `wall` above the bed there is
    plastic and a tenth of a millimetre higher there is the slot."""
    s, m = numbers(), part()
    below = 0.5 * (support_front(s, WALL - 0.6) + lip_side(s, WALL - 0.6))
    above = 0.5 * (support_front(s, WALL + 0.6) + lip_side(s, WALL + 0.6))
    assert touches(m, box(s, below - 1.0, below + 1.0, WALL - 1.0, WALL - 0.2,
                          0.15 * s["width"], 0.35 * s["width"])), "there is no ledge to stand on"
    assert not touches(m, box(s, above - 1.0, above + 1.0, WALL + 0.2, WALL + 1.0,
                              0.15 * s["width"], 0.35 * s["width"])), "the slot is blocked by its floor"


@pytest.mark.parametrize("lip", (4.0, 10.0, 20.0))
def test_the_lip_is_as_high_above_the_ledge_as_asked(lip):
    """The lip keeps the foot of the device in the slot: plastic up to `lip` above the ledge and
    nothing above that."""
    s, m = numbers(lip=lip), part(lip=lip)
    in_lip = 0.5 * (lip_side(s, WALL + lip - 0.4) + lip_side(s, WALL + lip - 0.4) + s["wall"]
                    / math.sin(math.radians(s["angle"])))
    above = 0.5 * (lip_side(s, WALL + lip + 0.4) + lip_side(s, WALL + lip + 0.4)
                   + s["wall"] / math.sin(math.radians(s["angle"])))
    assert touches(m, box(s, in_lip - 0.5, in_lip + 0.5, WALL + lip - 0.8, WALL + lip - 0.2,
                          0.15 * s["width"], 0.35 * s["width"])), "the lip does not come that high"
    assert not touches(m, box(s, above - 0.5, above + 0.5, WALL + lip + 0.2, WALL + lip + 0.8,
                              0.15 * s["width"], 0.35 * s["width"])), "the lip comes higher than asked"


# ------------------------------------------------------------------ the lean

@pytest.mark.parametrize("angle", (50.0, 65.0, 80.0))
@pytest.mark.parametrize("back", (40.0, 90.0, 160.0))
def test_the_support_leans_at_the_angle_it_asks(angle, back):
    """The back of the support is one straight face that falls back as it rises: from where it is
    at two heights the slope follows, and that slope is the tangent of the lean."""
    kw = dict(angle=angle, back=back)
    s, m = numbers(**kw), part(**kw)
    rise = back * math.sin(math.radians(angle))
    y1, y2 = WALL + 0.2 * rise, WALL + 0.8 * rise
    slope = (back_x(m, s, y2) - back_x(m, s, y1)) / (y2 - y1)
    assert slope == pytest.approx(-lean(s), abs=0.01), "the support does not lean at that angle"
    assert back_x(m, s, y2) < back_x(m, s, y1), "the support leans forwards instead of backwards"


def test_the_top_of_the_support_is_round():
    """The support ends on a half-round of its own thickness: the corner outside that round is
    rounded away and there is plastic right under the middle of the top."""
    s, m = numbers(), part()
    a = math.radians(s["angle"])
    r = 0.5 * s["wall"]
    cx = s["support_x"] - s["back"] * math.cos(a) - r * math.sin(a)
    cy = WALL + s["back"] * math.sin(a) - r * math.cos(a)
    bx, by = (math.sin(a) - math.cos(a)) / math.sqrt(2.0), (math.sin(a) + math.cos(a)) / math.sqrt(2.0)
    px, py = cx + 1.3 * r * bx, cy + 1.3 * r * by
    assert not touches(m, box(s, px - 0.2, px + 0.2, py - 0.2, py + 0.2,
                              0.15 * s["width"], 0.35 * s["width"])), \
        "the corner outside the round top is not rounded away"
    assert touches(m, box(s, cx - 0.2, cx + 0.2, cy + 0.6 * r, cy + 0.9 * r,
                          0.15 * s["width"], 0.35 * s["width"])), \
        "there is nothing under the middle of the top"


# ------------------------------------------------------------------ the cable slot

def test_the_cable_slot_is_open_through_the_base_and_the_lip():
    base, lip = cable_probes(numbers())
    m = part()
    assert not touches(m, base), "there is no room for a cable through the base"
    assert not touches(m, lip), "there is no room for a cable through the lip"


def test_a_stand_without_a_cable_slot_is_closed_under_the_device():
    base, lip = cable_probes(numbers(cable=False))
    m = part(cable=False)
    assert touches(m, base), "the base is open without a cable slot"
    assert touches(m, lip), "the lip is open without a cable slot"


@pytest.mark.parametrize("width", (30.0, 39.0))
def test_a_narrow_stand_has_no_room_for_a_cable_slot(width):
    """Beside a gap of 12 millimetres has to stay stand enough to stand on, so a narrow stand
    keeps its base and its lip whole."""
    s = numbers(width=width)
    assert s["cable"] is False, "a stand under 40 mm wide gets no cable slot"
    base, lip = cable_probes(s)
    m = part(width=width)
    assert touches(m, base) and touches(m, lip), "a narrow stand was cut open anyway"
    assert len(m.decompose()) == 1


def test_the_cable_slot_leaves_the_stand_in_one_piece():
    for width in (40.0, 70.0, 200.0):
        m = part(width=width)
        assert len(m.decompose()) == 1, f"the cable slot cut the {width:g} mm stand in two"
        assert m.genus() == 1, "the cable does not run under the foot of the support"


# ------------------------------------------------------------------ standing on its own

@pytest.mark.parametrize("angle, back, device", ((50.0, 40.0, 16.0), (80.0, 40.0, 16.0),
                                                 (50.0, 160.0, 5.0), (80.0, 160.0, 5.0),
                                                 (65.0, 90.0, 11.0)))
def test_a_leaning_device_stays_over_the_base(angle, back, device):
    """A device of `back` × 1,8 millimetres leans against the support with its centre half that
    height up the lean: the base is long enough that this centre stays `BACK_SPARE` millimetres in
    front of its back edge, so the stand cannot tip over backwards."""
    kw = dict(angle=angle, back=back, device=device)
    s, m = numbers(**kw), part(**kw)
    assert s["device_x"] >= phone_stand.BACK_SPARE - 1e-9, "the device leans over the back edge"
    assert s["device_x"] <= s["depth"], "the device leans past the front of the base"
    lo, hi = bbox(m)
    assert lo[0] + s["device_x"] > lo[0], "the device centre is not over the base"
    assert len(m.decompose()) == 1


def test_the_base_reaches_under_the_back_of_the_support():
    """The back edge of the base is under the back of the rounded top, so nothing of the stand
    leans past the plate that lies on the bed; a longer support leans further back and makes the
    base longer with it."""
    short, long = numbers(back=40.0), numbers(back=160.0)
    assert long["depth"] > short["depth"], "a longer support does not need a longer base"
    for kw in (dict(back=40.0), dict(back=160.0), dict(angle=50.0, back=160.0)):
        s, m = numbers(**kw), part(**kw)
        assert not touches(m, box(s, -0.5, 0.0, 0.0, s["height"], 0.0, s["width"])), \
            "something leans past the back of the base"
        assert s["device_x"] >= phone_stand.BACK_SPARE - 1e-9


# ------------------------------------------------------------------ printing

def test_nothing_hangs_over_because_the_profile_is_pulled_straight_up():
    """The profile lies on the bed and every wall of the stand is pulled straight up from it, so
    the only face that looks down is the one that lies on the bed."""
    m = part(cable=False, width=30.0, angle=80.0, back=160.0, device=16.0, lip=20.0)
    n, c, area = triangles(m)
    keep = area > 1e-9
    assert (n[keep][c[keep][:, 2] < 1e-9][:, 2] < -0.99).all(), \
        "the triangles do not face out of the part"
    down = keep & (n[:, 2] < -1e-6) & (c[:, 2] > 1e-6)
    assert not down.any(), "a face hangs over, so the stand would need support"


def test_the_cable_slot_is_the_only_thing_that_hangs_over():
    """The gap for the cable is the one place where plastic looks down, right above the gap and as
    high as the top of it: what hangs over is a wall of the base and the lip, and bridges on its
    own."""
    s, m = numbers(), part()
    n, c, area = triangles(m)
    down = (area > 1e-9) & (n[:, 2] < -0.99) & (c[:, 2] > 1e-6)
    assert down.any(), "the cable slot leaves no ceiling at all"
    assert np.allclose(c[down][:, 2], 0.5 * (s["width"] + phone_stand.CABLE), atol=1e-6), \
        "something hangs over here that is not the ceiling of the cable slot"


# ------------------------------------------------------------------ the numbers

def test_sizes_reports_the_stand_and_its_slot():
    s = phone_stand.sizes(width=100.0, angle=70.0, back=120.0, device=13.0, lip=15.0, cable=False)
    assert s["width"] == 100.0 and s["angle"] == 70.0 and s["back"] == 120.0
    assert s["device"] == 13.0 and s["lip"] == 15.0 and s["cable"] is False
    assert s["wall"] == phone_stand.WALL
    assert s["slot"] == pytest.approx(14.0), "the slot is a millimetre wider than the device"
    assert s["height"] == pytest.approx(WALL + 120.0 * math.sin(math.radians(70.0))
                                        + 0.5 * WALL * (1.0 - math.cos(math.radians(70.0))))
    assert s["depth"] > s["device_x"], "the device leans past the front of the base"
    assert set(s) == {"width", "angle", "back", "device", "lip", "cable", "wall", "slot",
                      "depth", "height", "support_x", "device_x"}


def test_the_default_stand_holds_a_phone_with_its_case():
    s = phone_stand.sizes()
    assert s["width"] == 70.0 and s["angle"] == 65.0 and s["back"] == 90.0
    assert s["device"] == 11.0 and s["lip"] == 10.0 and s["cable"] is True
    assert s["slot"] == 12.0 and s["wall"] == 3.0
    assert s["depth"] > s["support_x"], "the lip does not stand on the base"


@pytest.mark.parametrize("width", (30.0, 39.0, 40.0, 200.0))
def test_the_cable_slot_follows_the_width(width):
    want = width >= phone_stand.CABLE_MIN_WIDTH
    assert phone_stand.sizes(width=width)["cable"] is want


def test_the_numbers_are_clamped_to_what_prints():
    small = numbers(width=0.0, angle=0.0, back=0.0, device=0.0, lip=0.0)
    assert small["width"] == phone_stand.WIDTH_MIN and small["angle"] == phone_stand.ANGLE_MIN
    assert small["back"] == phone_stand.BACK_MIN and small["device"] == phone_stand.DEVICE_MIN
    assert small["lip"] == phone_stand.LIP_MIN
    big = numbers(width=999.0, angle=999.0, back=999.0, device=999.0, lip=999.0)
    assert big["width"] == phone_stand.WIDTH_MAX and big["angle"] == phone_stand.ANGLE_MAX
    assert big["back"] == phone_stand.BACK_MAX and big["device"] == phone_stand.DEVICE_MAX
    assert big["lip"] == phone_stand.LIP_MAX
    lo, hi = bbox(part(width=0.0, angle=0.0, back=0.0, device=0.0, lip=0.0))
    assert hi[2] - lo[2] == pytest.approx(phone_stand.WIDTH_MIN, abs=ABS), \
        "the smallest stand is not as high on the bed as it is wide"
    assert len(part(width=999.0, angle=999.0, back=999.0, device=999.0, lip=999.0).decompose()) == 1


@pytest.mark.parametrize("bad", (float("nan"), float("inf"), None, "wide"))
def test_a_number_that_is_no_number_is_refused(bad):
    for key in ("width", "angle", "back", "device", "lip"):
        with pytest.raises(ValueError):
            phone_stand.phone_stand(**dict(BASE, **{key: bad}))
        with pytest.raises(ValueError):
            phone_stand.sizes(**dict(BASE, **{key: bad}))


# ------------------------------------------------------------------ the interface

def test_the_dialog_builds_a_stand_and_remembers_it(monkeypatch, tmp_path):
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
        dlg.width.setValue(100.0)
        dlg.angle.setValue(70.0)
        dlg.back.setValue(120.0)
        dlg.lip.setValue(15.0)
        assert dlg.cable.isChecked(), "the cable slot is there unless you say otherwise"
        assert "62.9" in dlg.info.text(), "the dialog does not show the base it computes"
        return QDialog.Accepted

    monkeypatch.setattr(app_module.QDialog, "exec", fill)
    try:
        w.phone_stand_dialog()
        deadline = time.monotonic() + 15.0
        while not w.doc.groups and time.monotonic() < deadline:
            app.processEvents()
            time.sleep(0.005)
        assert len(w.doc.groups) == 1, "no stand in the model"
        lo, hi = np.asarray(w.doc.groups[0].bbox(), dtype=float)
        s = phone_stand.sizes(width=100.0, angle=70.0, back=120.0, lip=15.0)
        assert lo[2] == pytest.approx(0.0, abs=1e-6), "the stand does not lie on the bed"
        assert hi[2] - lo[2] == pytest.approx(s["width"], abs=1e-4)
        assert hi[0] - lo[0] == pytest.approx(s["depth"], abs=1e-4), "the base is not as computed"
        assert hi[1] - lo[1] == pytest.approx(s["height"], abs=0.01), "the support is not as high"
        settings = app_module.Settings()
        assert (settings.phone_stand_width, settings.phone_stand_angle,
                settings.phone_stand_back, settings.phone_stand_device,
                settings.phone_stand_lip, settings.phone_stand_cable) == (
            100.0, 70.0, 120.0, 11.0, 15.0, True)
        labels = [a.text().replace("&", "").replace("…", "") for a in w.flat_actions(w.menuBar())]
        assert labels.index("Phone Stand") == labels.index("Wall Hook") + 1
    finally:
        w.settings.phone_stand_width, w.settings.phone_stand_angle = 70.0, 65.0
        w.settings.phone_stand_back, w.settings.phone_stand_device = 90.0, 11.0
        w.settings.phone_stand_lip, w.settings.phone_stand_cable = 10.0, True
        w.settings.save()
        w.doc.dirty = False
        w.close()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))

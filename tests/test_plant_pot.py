"""Plant pots: a tapered flower pot that prints upright, with a saucer beside it.

`plant_pot.plant_pot` gives the pot and, when it is asked for, its saucer as closed volumes that
stand side by side on `z = 0`: the pot is `diameter` millimetres across at the top and `height`
millimetres high, narrowing to `taper` percent of that at the bottom, with a wall of `wall` and a
floor of `FLOOR`, its rim rounded outward by a bead of `BEAD`, its floor drained by `holes` holes
with channels under the floor that lead the water out. The saucer is the bottom of the pot plus
`SAUCER_ROOM` millimetres all round and `SAUCER_H` millimetres high, of the same shape as the pot,
which is round or six-sided across the flats.

These tests weigh the pair against that: one volume each, standing on the bed, five millimetres
apart and centred together; the pot as wide as asked at the bead on its rim and as narrow at the
bottom as the taper says; the wall leaning out by at most 45 degrees for every taper that is
allowed; an inside that is hollow down to the floor; an open drain hole on every place that is
promised and a whole floor when none are asked for; channels from the holes out from under the
pot, and the rest of the bottom on the bed; the bottom of the pot dropping into the saucer with
room to spare; six flats on a hexagon; every range reached by clamping and a number that is no
number refused.

Run with:  python -m pytest tests/test_plant_pot.py
"""
import math
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import manifold3d as mf  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from volura import plant_pot  # noqa: E402

BASE = dict(diameter=100.0, height=90.0, taper=75.0, wall=2.0, holes=5, shape="round", saucer=True)
#: the bead and the holes are polygons, so a part is at most this far off
ABS = 0.05
#: mm across a probe that feels for plastic or for an open hole
PROBE = 0.4


def parts(**kw) -> list:
    return plant_pot.plant_pot(**dict(BASE, **kw))


def pot(**kw) -> mf.Manifold:
    """The pot on its own, without the saucer beside it."""
    return parts(**{**kw, "saucer": False})[0]


def dish(**kw) -> mf.Manifold:
    """The saucer on its own, without the pot beside it."""
    return parts(**{**kw, "saucer": True})[1]


def numbers(**kw) -> dict:
    return plant_pot.sizes(**dict(BASE, **kw))


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


def middle(m):
    """The middle of a piece in plan, so a probe sits in the frame of the piece it tests."""
    lo, hi = bbox(m)
    return 0.5 * (lo[0] + hi[0]), 0.5 * (lo[1] + hi[1])


def width(m, z0, z1) -> tuple:
    """How wide a piece is between the heights `z0` and `z1`, in x and in y."""
    slab = mf.Manifold.cube([1e4, 1e4, z1 - z0]).translate((-5e3, -5e3, z0))
    lo, hi = bbox(m ^ slab)
    return hi[0] - lo[0], hi[1] - lo[1]


def wall_radius(s, z) -> float:
    """The radius of the outside of the wall at height `z`, measured to its flat."""
    lean = s["height"] - 2.0 * plant_pot.BEAD - plant_pot.CHANNEL_H
    return 0.5 * (s["bottom"] + (s["diameter"] - s["bottom"]) * (z - plant_pot.CHANNEL_H) / lean)


# ------------------------------------------------------------------ the pieces

@pytest.mark.parametrize("shape", ("round", "hexagon"))
@pytest.mark.parametrize("holes", (0, 5, 9))
@pytest.mark.parametrize("with_saucer", (False, True))
def test_every_piece_is_one_volume_that_stands_on_the_bed(shape, holes, with_saucer):
    kw = dict(shape=shape, holes=holes, saucer=with_saucer)
    s, made = numbers(**kw), parts(**kw)
    assert len(made) == (2 if with_saucer else 1), "the saucer is not a piece of its own"
    for m in made:
        assert m.status() == mf.Error.NoError and not m.is_empty()
        assert len(m.decompose()) == 1, "a piece fell apart"
        lo, hi = bbox(m)
        assert lo[2] == pytest.approx(0.0, abs=1e-6), "the piece does not stand on the bed"
    lo, hi = bbox(made[0])
    assert hi[2] - lo[2] == pytest.approx(s["height"], abs=ABS), "the pot is not as high"
    assert hi[1] - lo[1] == pytest.approx(s["diameter"] + 2.0 * plant_pot.BEAD, abs=ABS), \
        "the pot is not as wide as the bead makes it"
    if with_saucer:
        lo, hi = bbox(made[1])
        assert hi[2] - lo[2] == pytest.approx(plant_pot.SAUCER_H, abs=ABS), "the saucer is not as high"
        assert hi[1] - lo[1] == pytest.approx(s["saucer_inside"] + 2.0 * plant_pot.SAUCER_WALL,
                                              abs=ABS), "the saucer is not the bottom plus its room"


def test_the_pot_and_the_saucer_stand_side_by_side_and_are_centred_in_plan():
    """The pair is one print: the saucer lies beside the pot with the gap between them, and what
    stands on the bed is centred in x and y."""
    made = parts()
    assert len(made) == 2
    boxes = [np.asarray(m.bounding_box(), dtype=float) for m in made]
    assert boxes[1][0] - boxes[0][3] == pytest.approx(plant_pot.GAP, abs=ABS), \
        "the saucer does not lie 5 mm beside the pot"
    for b in boxes:
        assert b[1] == pytest.approx(-b[4], abs=ABS), "a piece is not centred in y"
    assert min(b[0] for b in boxes) == pytest.approx(-max(b[3] for b in boxes), abs=ABS), \
        "the pair is not centred in x"


# ------------------------------------------------------------------ the shape of the pot

@pytest.mark.parametrize("shape", ("round", "hexagon"))
@pytest.mark.parametrize("taper", (50.0, 75.0, 100.0))
def test_the_pot_is_its_diameter_across_at_the_top_and_the_taper_at_the_bottom(shape, taper):
    """The widest place of the pot is the bead on its rim, one bead wider than the diameter on
    every side; the bottom it stands on is `taper` percent of that diameter."""
    s, m = numbers(shape=shape, taper=taper), pot(shape=shape, taper=taper)
    at = s["height"] - plant_pot.BEAD
    wx, wy = width(m, at - 0.05, at + 0.05)
    assert wy == pytest.approx(s["diameter"] + 2.0 * plant_pot.BEAD, abs=ABS), "not the top diameter"
    if shape == "round":
        assert wx == pytest.approx(s["diameter"] + 2.0 * plant_pot.BEAD, abs=ABS)
    else:
        assert wx == pytest.approx((s["diameter"] + 2.0 * plant_pot.BEAD) * plant_pot.CORNER,
                                   abs=ABS), "a hexagon is wider at its corners"
    wx, wy = width(m, plant_pot.CHANNEL_H, plant_pot.CHANNEL_H + 0.05)
    assert wy == pytest.approx(s["bottom"], abs=ABS), "the bottom is not the taper of the top"
    assert s["bottom"] == pytest.approx(s["diameter"] * taper / 100.0)


@pytest.mark.parametrize("z", (3.0, 45.0, 85.0))
@pytest.mark.parametrize("shape", ("round", "hexagon"))
def test_the_wall_leans_outwards_in_a_straight_line(shape, z):
    s, m = numbers(shape=shape), pot(shape=shape)
    r = wall_radius(s, z)
    assert touches(m, post(0.0, r - 0.5, z - 0.05, z + 0.05)), f"there is no wall at {z} mm"
    assert not touches(m, post(0.0, r + 0.5, z - 0.05, z + 0.05)), f"the wall reaches too far at {z} mm"


@pytest.mark.parametrize("diameter", (40.0, 100.0, 250.0))
@pytest.mark.parametrize("height", (30.0, 90.0, 250.0))
@pytest.mark.parametrize("taper", (50.0, 75.0, 100.0))
def test_the_wall_never_leans_out_further_than_it_stands_high(diameter, height, taper):
    """A wall that leans out by as much as it stands high leans 45 degrees, which is the steepest
    one that prints upright without support; a taper that leans further is made milder."""
    s = numbers(diameter=diameter, height=height, taper=taper)
    assert 0.5 * (s["diameter"] - s["bottom"]) <= \
        s["height"] - 2.0 * plant_pot.BEAD - plant_pot.CHANNEL_H + 1e-9, "the wall leans past 45 degrees"
    assert s["taper"] >= taper - 1e-9, "the taper was made steeper, not milder"
    assert s["bottom"] <= s["diameter"] + 1e-9, "the pot is wider at the bottom than at the top"


def test_a_taper_that_leans_too_far_makes_the_bottom_larger():
    s = numbers(diameter=250.0, height=30.0, taper=50.0)
    assert s["taper"] > plant_pot.TAPER_MIN, "the taper was not clamped"
    assert s["bottom"] == pytest.approx(s["diameter"]
                                        - 2.0 * (s["height"] - 2.0 * plant_pot.BEAD - plant_pot.CHANNEL_H))


# ------------------------------------------------------------------ the inside

def test_the_inside_is_hollow_down_to_the_floor():
    s, m = numbers(holes=0), pot(holes=0)
    assert not touches(m, post(0.0, 0.0, plant_pot.CHANNEL_H + plant_pot.FLOOR + 0.5, s["height"] + 1.0)), \
        "the pot is not hollow"
    assert touches(m, post(0.0, 0.0, plant_pot.CHANNEL_H + 0.5,
                           plant_pot.CHANNEL_H + plant_pot.FLOOR - 0.5)), "there is no floor under the soil"


def test_the_wall_keeps_its_thickness_from_the_floor_to_the_rim():
    """The inside leans with the outside, so `wall` millimetres of plastic stay everywhere."""
    s, m = numbers(), pot()
    for z in (5.0, 45.0, s["height"] - 2.0 * plant_pot.BEAD - 0.5):
        r = wall_radius(s, z)
        assert touches(m, post(0.0, r - 0.5 * s["wall"], z - 0.05, z + 0.05)), f"thin wall at {z} mm"
        assert not touches(m, post(0.0, r - s["wall"] - 0.5, z - 0.05, z + 0.05)), \
            f"the inside is wider than asked at {z} mm"


# ------------------------------------------------------------------ the drain holes

@pytest.mark.parametrize("holes", (0, 1, 2, 5, 9))
def test_the_drain_holes_are_open_where_they_stand(holes):
    """One in the middle of the floor and the rest on the circle halfway to the wall; a probe
    through every one of them meets no plastic, and none of them is there when 0 are asked for."""
    s, m = numbers(holes=holes), pot(holes=holes)
    at = _hole_places(s)
    assert len(at) == holes
    for x, y in at:
        assert not touches(m, post(x, y, 0.1, s["height"] - 1.0, r=0.3)), \
            f"the drain hole at ({x:.1f}, {y:.1f}) is not open"
    if holes == 0:
        assert touches(m, post(0.0, 0.0, 0.1, s["height"] - 1.0, r=0.3)), \
            "the floor is drilled even without drain holes"


@pytest.mark.parametrize("holes", (2, 5, 9))
def test_the_floor_between_two_drain_holes_stays_whole(holes):
    """Halfway between two of them, in the floor itself, is plastic."""
    s, m = numbers(holes=holes), pot(holes=holes)
    r = 0.5 * s["hole_ring"]
    a = math.pi / (holes - 1)
    assert touches(m, post(r * math.cos(a), r * math.sin(a), plant_pot.CHANNEL_H + 0.5,
                           plant_pot.CHANNEL_H + plant_pot.FLOOR - 0.5, r=0.3)), \
        "two drain holes ran into one another"


def test_the_drain_holes_are_eight_millimetres_across_a_normal_pot():
    s = numbers()
    assert s["hole"] == pytest.approx(plant_pot.HOLE_D)
    assert s["hole_ring"] == pytest.approx(0.5 * (s["bottom"] - 2.0 * s["wall"]))


def _hole_places(s) -> list:
    """Where the drain holes of a pot stand: the middle of the floor and the circle halfway to
    the wall."""
    if s["holes"] <= 0:
        return []
    r = 0.5 * s["hole_ring"]
    n = max(1, s["holes"] - 1)
    return [(0.0, 0.0)] + [(r * math.cos(2.0 * math.pi * k / n), r * math.sin(2.0 * math.pi * k / n))
                           for k in range(s["holes"] - 1)]


# ------------------------------------------------------------------ the channels

@pytest.mark.parametrize("shape", ("round", "hexagon"))
@pytest.mark.parametrize("holes, channels", ((1, plant_pot.CHANNEL_N), (5, 4), (9, 8)))
def test_channels_lead_the_water_out_from_under_the_pot(shape, holes, channels):
    """Under every channel the bed is open from the middle to past the edge, the floor bridges
    it, and between the channels the pot stands on its bottom."""
    s, m = numbers(shape=shape, holes=holes), pot(shape=shape, holes=holes)
    lo, hi = bbox(m)
    assert lo[2] == pytest.approx(0.0, abs=1e-6), "the pot does not stand on the bed"
    r = 0.5 * s["bottom"]
    low = (0.3, plant_pot.CHANNEL_H - 0.3)
    for k in range(channels):
        a = 2.0 * math.pi * k / channels
        for rad in (0.3 * r, 0.7 * r, r - 0.3):
            assert not touches(m, post(rad * math.cos(a), rad * math.sin(a), *low)), \
                "a channel is blocked"
        assert touches(m, post(0.7 * r * math.cos(a), 0.7 * r * math.sin(a),
                               plant_pot.CHANNEL_H + 0.3, plant_pot.CHANNEL_H + 0.6)), \
            "no floor over a channel"
    for k in range(channels):                # between two channels the pot stands on the bed
        a = 2.0 * math.pi * (k + 0.5) / channels
        rad = 0.85 * r
        assert touches(m, post(rad * math.cos(a), rad * math.sin(a), *low)), \
            "the bottom does not stand on the bed between the channels"


def test_without_drain_holes_the_bottom_is_whole():
    s, m = numbers(holes=0), pot(holes=0)
    r = 0.5 * s["bottom"]
    for k in range(12):
        a = 2.0 * math.pi * k / 12
        assert touches(m, post(0.7 * r * math.cos(a), 0.7 * r * math.sin(a), 0.3, 1.0))


# ------------------------------------------------------------------ the saucer

def test_the_bottom_of_the_pot_fits_into_the_saucer_with_room_to_spare():
    s, sc = numbers(), dish()
    assert s["saucer_inside"] == pytest.approx(s["bottom"] + 2.0 * plant_pot.SAUCER_ROOM)
    cx, cy = middle(sc)
    above = plant_pot.SAUCER_FLOOR + 1.0
    assert not touches(sc, post(cx + 0.5 * s["bottom"], cy, above, plant_pot.SAUCER_H - 1.0)), \
        "the bottom of the pot does not fit into the saucer"
    assert not touches(sc, post(cx + 0.5 * s["saucer_inside"] - 0.5, cy, above,
                                plant_pot.SAUCER_H - 1.0)), "the saucer has no room to spare"
    assert touches(sc, post(cx + 0.5 * s["saucer_inside"] + 0.5 * plant_pot.SAUCER_WALL, cy, above,
                            plant_pot.SAUCER_H - 1.0)), "the saucer has no wall"
    assert not touches(sc, post(cx, cy, above, plant_pot.SAUCER_H - 1.0)), \
        "the saucer's floor is thicker than asked"
    assert touches(sc, post(cx, cy, 0.5, plant_pot.SAUCER_FLOOR - 0.5)), "the saucer has no floor"


@pytest.mark.parametrize("shape", ("round", "hexagon"))
def test_the_saucer_has_the_shape_of_the_pot(shape):
    s, m = numbers(shape=shape), parts(shape=shape)[1]
    wx, wy = width(m, 1.0, 2.0)
    out = s["saucer_inside"] + 2.0 * plant_pot.SAUCER_WALL
    assert wy == pytest.approx(out, abs=ABS)
    if shape == "hexagon":
        assert wx == pytest.approx(out * plant_pot.CORNER, abs=ABS)


# ------------------------------------------------------------------ the hexagon

@pytest.mark.parametrize("shape", ("round", "hexagon"))
def test_the_hexagon_has_six_flat_sides(shape):
    """A hexagon sticks out in six corners and is drawn in between them: a probe in the middle of
    the wall meets plastic on six of the twelve directions round the pot, and nothing at all on
    the other six. A round pot has no corners, so it meets nothing anywhere."""
    s, m = numbers(shape=shape), pot(shape=shape)
    z = 0.5 * s["height"]
    r = wall_radius(s, z)
    out = 0.5 * plant_pot.CORNER * (2.0 * r - s["wall"])
    at = [touches(m, post(out * math.cos(math.pi * k / 6.0), out * math.sin(math.pi * k / 6.0),
                          z - 0.05, z + 0.05)) for k in range(12)]
    if shape == "hexagon":
        assert at == [k % 2 == 0 for k in range(12)], "not six corners between six flats"
    else:
        assert not any(at), "a round pot has corners"


# ------------------------------------------------------------------ the numbers

def test_sizes_reports_the_pot_and_its_saucer():
    s = plant_pot.sizes(diameter=120.0, height=100.0, taper=70.0, wall=2.4, holes=4,
                        shape="hexagon", saucer=True)
    assert s["diameter"] == 120.0 and s["height"] == 100.0 and s["taper"] == 70.0
    assert s["wall"] == 2.4 and s["holes"] == 4 and s["shape"] == "hexagon" and s["saucer"] is True
    assert s["bottom"] == pytest.approx(84.0)
    assert s["hole"] == pytest.approx(plant_pot.HOLE_D)
    assert s["saucer_inside"] == pytest.approx(104.0)
    assert s["capacity"] == pytest.approx(0.810, abs=1e-3)
    assert set(s) == {"diameter", "height", "taper", "wall", "holes", "shape", "saucer", "bottom",
                      "hole", "hole_ring", "saucer_inside", "capacity"}


def test_the_default_pot_holds_about_half_a_litre():
    s = plant_pot.sizes()
    assert s["bottom"] == pytest.approx(75.0) and s["hole_ring"] == pytest.approx(35.5)
    assert s["saucer_inside"] == pytest.approx(95.0)
    assert s["capacity"] == pytest.approx(0.482, abs=1e-3)


@pytest.mark.parametrize("holes, want", ((-3, 0), (0, 0), (1, 1), (5, 5), (9, 9), (99, 9)))
def test_the_number_of_drain_holes_follows_what_is_asked(holes, want):
    assert plant_pot.sizes(holes=holes)["holes"] == want


@pytest.mark.parametrize("shape, want", ((None, "round"), ("", "round"), ("triangle", "round"),
                                         ("HEXAGON", "hexagon"), ("hexagon", "hexagon")))
def test_a_shape_that_is_not_there_makes_a_round_pot(shape, want):
    assert plant_pot.sizes(shape=shape)["shape"] == want


def test_the_numbers_are_clamped_to_what_prints():
    small = numbers(diameter=0.0, height=0.0, taper=0.0, wall=0.0, holes=-1)
    assert small["diameter"] == plant_pot.D_MIN and small["height"] == plant_pot.HEIGHT_MIN
    assert small["taper"] == plant_pot.TAPER_MIN and small["wall"] == plant_pot.WALL_MIN
    assert small["holes"] == plant_pot.HOLES_MIN
    big = numbers(diameter=999.0, height=999.0, taper=999.0, wall=999.0, holes=99)
    assert big["diameter"] == plant_pot.D_MAX and big["height"] == plant_pot.HEIGHT_MAX
    assert big["taper"] == plant_pot.TAPER_MAX and big["wall"] == plant_pot.WALL_MAX
    assert big["holes"] == plant_pot.HOLES_MAX
    lo, hi = bbox(pot(diameter=0.0, height=0.0, taper=0.0, wall=0.0, holes=-1))
    assert hi[2] - lo[2] == pytest.approx(plant_pot.HEIGHT_MIN, abs=ABS)
    assert len(parts(diameter=999.0, height=999.0, taper=999.0, wall=999.0, holes=99)) == 2


@pytest.mark.parametrize("bad", (float("nan"), float("inf"), None, "wide"))
def test_a_number_that_is_no_number_is_refused(bad):
    for key in ("diameter", "height", "taper", "wall", "holes"):
        with pytest.raises(ValueError):
            plant_pot.plant_pot(**dict(BASE, **{key: bad}))
        with pytest.raises(ValueError):
            plant_pot.sizes(**dict(BASE, **{key: bad}))


# ------------------------------------------------------------------ the interface

def test_the_dialog_builds_a_pot_and_a_saucer_as_two_groups_and_remembers_it(monkeypatch, tmp_path):
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
        dlg.diameter.setValue(120.0)
        dlg.height.setValue(100.0)
        dlg.taper.setValue(70.0)
        dlg.wall.setValue(2.4)
        dlg.holes.setValue(4)
        assert dlg.saucer.isChecked(), "the saucer is off by default"
        assert "84.0" in dlg.info.text(), "the dialog does not show the bottom it makes"
        assert "0.81" in dlg.info.text(), "the dialog does not say what the pot holds"
        return QDialog.Accepted

    monkeypatch.setattr(app_module.QDialog, "exec", fill)
    try:
        w.plant_pot_dialog()
        deadline = time.monotonic() + 20.0
        while len(w.doc.groups) < 2 and time.monotonic() < deadline:
            app.processEvents()
            time.sleep(0.005)
        assert len(w.doc.groups) == 2, "no pot and saucer in the model"
        assert [g.name for g in w.doc.groups] == ["Plant pot", "Saucer"]
        s = plant_pot.sizes(diameter=120.0, height=100.0, taper=70.0, wall=2.4, holes=4,
                            shape="hexagon")
        lo, hi = np.asarray(w.doc.groups[0].bbox(), dtype=float)
        assert lo[2] == pytest.approx(0.0, abs=1e-6), "the pot does not stand on the bed"
        assert hi[2] - lo[2] == pytest.approx(s["height"], abs=1e-4)
        assert hi[1] - lo[1] == pytest.approx(s["diameter"] + 2.0 * plant_pot.BEAD, abs=1e-3), \
            "not as computed"
        lo, hi = np.asarray(w.doc.groups[1].bbox(), dtype=float)
        assert hi[2] - lo[2] == pytest.approx(plant_pot.SAUCER_H, abs=1e-4), "not the saucer"
        settings = app_module.Settings()
        assert (settings.plant_pot_diameter, settings.plant_pot_height, settings.plant_pot_taper,
                settings.plant_pot_wall, settings.plant_pot_holes, settings.plant_pot_shape,
                settings.plant_pot_saucer) == (120.0, 100.0, 70.0, 2.4, 4, "hexagon", True)
        labels = [a.text().replace("&", "").replace("…", "") for a in w.flat_actions(w.menuBar())]
        assert labels.index("Plant Pot") == labels.index("Drawer Organizer") + 1
    finally:
        for key, value in (("plant_pot_diameter", 100.0), ("plant_pot_height", 90.0),
                           ("plant_pot_taper", 75.0), ("plant_pot_wall", 2.0),
                           ("plant_pot_holes", 5), ("plant_pot_shape", "round"),
                           ("plant_pot_saucer", True)):
            setattr(w.settings, key, value)
        w.settings.save()
        w.doc.dirty = False
        w.close()


@pytest.mark.parametrize("shape", ("round", "hexagon"))
def test_the_bead_grows_out_of_the_wall_at_45_degrees(shape):
    """The rim prints upright without support: under the bead nothing sticks out flat."""
    s, m = numbers(shape=shape), pot(shape=shape)
    top = s["height"] - plant_pot.BEAD
    for dz in (0.25, 0.5, 1.0, 1.5, 2.0):
        _wx, below = width(m, top - dz - 0.05, top - dz)
        _wx, above = width(m, top - dz + 0.2, top - dz + 0.25)
        assert above - below <= 2.0 * 0.3 + 1e-3, f"the bead overhangs at {top - dz + 0.25:.2f} mm"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))

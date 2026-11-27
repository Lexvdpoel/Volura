"""Timing pulleys and belt idlers: the wheel a toothed belt runs in, and the smooth one it runs over.

`pulley.timing_pulley` gives the wheel as one closed volume that stands on `z = 0` and is centred on
the origin, cut for a belt of GT2, GT3 or HTD 3M: it is turned to the published outside diameter of
its tooth count, which is its pitch circle less twice the pitch line differential of the profile,
and carries one groove per tooth over the height the belt runs between. `pulley.idler` gives the
smooth wheel such a belt runs over, bored for a 625 bearing or for a bush.

These tests weigh a wheel against what it promises: the diameter of the supplier's table, as many
grooves in a cross-section of the band as teeth were asked for, an axle with a flat that turns in
the bore without rubbing while the hub keeps its wall, a set screw that runs in from outside the hub
into the bore with its nut standing in a slot that is open at the top, collars that stand proud of
the teeth with the room of the belt between them, a smooth idler that runs on the circle its belt's
back stands on, and a part of which nothing looks down but the face on the bed, the slope of 45
degrees under the collar in the air and the roof that bridges the mouth of one groove. A size that
cannot be printed becomes the nearest one that can, and a wheel whose grooves do not fit or whose
bore would eat them is refused. The place in the interface is checked too (Parts under the
standoffs, one undo step, the dialog that starts at the old values and remembers the new ones), and
a real slice says the part needs no support.

Run with:  python -m pytest tests/test_pulley.py
"""
import math
import os
import re
import sys
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import manifold3d as mf  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from PySide6.QtCore import QCoreApplication  # noqa: E402
from PySide6.QtWidgets import (QApplication, QDialog, QDialogButtonBox,  # noqa: E402
                               QDoubleSpinBox)

import volura.app as app_module  # noqa: E402
from volura import cura_profiles, pulley, slicer, threads  # noqa: E402

_app = QApplication.instance() or QApplication(sys.argv[:1])

#: the fields of the dialog, as the window remembers them
BASE = dict(profile="gt2", teeth=20, clearance=0.1, width=6.0, flanges="both", flange=1.0,
            hub=True, hub_d=16.0, hub_h=10.0, bore=5.0, flat=0.0, nut=True)
OPTIONS = dict(BASE, idler=False, bearing=False)
BUSH = dict(profile="gt2", teeth=20, width=6.0, flange=1.0, bore=5.0, bearing=False, flat=0.0)
ROLLER = dict(BUSH, bearing=True, bore=8.0)     # asked for a bush of 8 mm, gets the bearing bore

#: how far the smooth back of a belt stands under its own pitch line, per profile in millimetres:
#: the thickness minus the pitch line to back, minus the height of a tooth under that line
BACK = {"gt2": 0.376, "gt3": 0.879, "htd3m": 0.739}

ENGINE, RESOURCES = slicer.find_engine(), slicer.find_resources()
needs_engine = pytest.mark.skipif(
    ENGINE is None or RESOURCES is None or slicer.engine_problem(ENGINE) is not None,
    reason="no CuraEngine that runs here (tools/fetch_curaengine.py; Linux: glibc 2.38+)")

#: a probe that is small enough to fall into a groove and big enough to be worth a test
PROBE = mf.Manifold.cylinder(0.06, 0.03, 0.03, 8)
PROBE_VOLUME = PROBE.volume()


def clean(text):
    return text.replace("&", "").replace("…", "").replace("...", "").strip()


def pump_until(predicate, timeout_ms=20000):
    """Spin the event loop until `predicate()` holds or the time is up; returns its value."""
    deadline = time.monotonic() + timeout_ms / 1000.0
    while not predicate() and time.monotonic() < deadline:
        _app.processEvents()
        time.sleep(0.005)
    return predicate()


def wheel(**kw):
    """A pulley with the fields of the dialog, with these answers over the defaults."""
    return pulley.timing_pulley(**dict(BASE, **kw))


def numbers(**kw):
    """The numbers that belong to that wheel, as the dialog shows them under the fields."""
    return pulley.sizes(**dict(BASE, **kw))


def bbox(m):
    """The bounding box of a manifold as (low, high)."""
    b = np.asarray(m.bounding_box(), dtype=float)
    return b[:3], b[3:]


def solid(m, x, y, z):
    """Whether a small probe at (x, y, z) falls in material."""
    return ((PROBE.translate((x, y, z))) - m).volume() < 0.02 * PROBE_VOLUME


def beside(m, r, y, z):
    """Whether a probe falls in material at the radius `r`, shifted `y` millimetres sideways;
    `y` has to stay well inside `r`, so the probe really stands at that radius."""
    assert abs(y) < 0.8 * r, "that probe would not stand where it is asked to"
    return solid(m, math.sqrt(r * r - y * y), y, z)


def tube(r0, r1, z0, z1):
    """The tube between the cylinders `r0` and `r1`, over the height from `z0` to `z1`."""
    h = max(z1 - z0, 1e-3)
    out = mf.Manifold.cylinder(h, max(r1, 1e-3), max(r1, 1e-3), 128)
    if r0 > 1e-6:
        out = out - mf.Manifold.cylinder(h + 2.0, r0, r0, 128).translate((0.0, 0.0, -1.0))
    return out.translate((0.0, 0.0, z0))


def in_tube(m, r0, r1, z0, z1):
    """How much of the part lies in that tube, and how much the tube itself is."""
    t = tube(r0, r1, z0, z1)
    return (m ^ t).volume(), t.volume()


def arcs(m, z, r0, r1):
    """How many pieces a thin horizontal slice of the part shows between two circles."""
    return len((m ^ tube(r0, r1, z - 0.05, z + 0.05)).decompose())


def facets(m):
    """The unit normals, the centroids and the areas of the triangles of a manifold."""
    mesh = m.to_mesh64()
    v = np.asarray(mesh.vert_properties, dtype=float)[:, :3]
    t = np.asarray(mesh.tri_verts, dtype=np.int64).reshape(-1, 3)
    a, b, c = v[t[:, 0]], v[t[:, 1]], v[t[:, 2]]
    n = np.cross(b - a, c - a)
    length = np.linalg.norm(n, axis=1)
    ok = length > 1e-12
    return n[ok] / length[ok][:, None], (a + b + c)[ok] / 3.0, length[ok] / 2.0


def looks_down(m, min_area=0.02):
    """The faces above the bed that look down: how far they lean away from the vertical, their
    radius, their height and their area. A face that leans by more than 45 degrees needs support."""
    n, centre, area = facets(m)
    keep = (n[:, 2] < -0.005) & (area > min_area) & (centre[:, 2] > 1e-6)
    lean = np.degrees(np.arcsin(np.clip(-n[:, 2], -1.0, 1.0)))
    radius = np.hypot(centre[:, 0], centre[:, 1])
    return lean[keep], radius[keep], centre[keep][:, 2], area[keep]


def roof_z(n):
    """The height of the roof over a groove: above the belt the wheel is one piece, so a groove
    is bridged half a `SIDE_ROOM` above the belt; where a collar stands there is no roof."""
    return n["ze"] if n["far"] else n["zb"] + n["width"] + 0.5 * pulley.SIDE_ROOM


def on_tunnel(radius, z, n):
    """Whether a face stands on the wall of the tunnel of the set screw. That hole of `screw`
    millimetres runs straight from the slot of the nut out through the wall of the hub, so it is a
    bridge between two walls of plastic and far too short to need support; the slice with Cura in
    this file is what says so of it."""
    if n["screw"] <= 0.0:
        return np.zeros(len(radius), dtype=bool)
    return ((radius >= 0.5 * n["slot_x"] - 0.05) & (radius <= 0.5 * n["hub_d"] + 0.05)
            & (z >= n["screw_z"] - 0.5 * n["screw"] - 0.05)
            & (z <= n["screw_z"] + 0.5 * n["screw"] + 0.05))


def hub_stand(n):
    """The radius of the face a hub stands on, and how far the hub of these numbers is wider than
    that face: the collar in the air carries the hub when there is one, the band of the belt when
    there is no collar there."""
    stand = n["rf"] if n["far"] else n["rim"]
    return stand, max(0.0, 0.5 * n["hub_d"] - stand)


def hub_cone(radius, z, n):
    """Whether a face is the cone that carries a hub wider than the face of the wheel it stands on.
    The cone rises from the edge of that face `SINK` millimetres into the wheel, which makes it
    steeper than 45 degrees from the vertical, so it prints without support."""
    stand, climb = hub_stand(n)
    if climb <= 1e-9:
        return np.zeros(len(radius), dtype=bool)
    return ((radius >= stand - 0.05) & (radius <= stand + climb + 0.05)
            & (z >= n["zt"] - pulley.SINK - 0.05) & (z <= n["hub_base"] + 0.05))


def needs_support(m, n):
    """The faces that would need support: those that lean more than 45 degrees away from the
    vertical, and are neither the roof the wheel makes over the mouth of one groove, nor the wall of
    the short tunnel the set screw is drilled through, nor the cone that carries the hub."""
    lean, radius, z, area = looks_down(m)
    roof = ((np.abs(z - roof_z(n)) < 1e-6) & (radius >= 0.5 * n["root"] - 0.05)
            & (radius <= n["rim"] + 0.05))
    keep = (lean > 45.0 + 0.6) & ~roof & ~on_tunnel(radius, z, n) & ~hub_cone(radius, z, n)
    return lean[keep], radius[keep], z[keep], area[keep]


def bed_islands(m):
    """The pieces a part touches the bed with, each by its widest reach."""
    slab = mf.Manifold.cylinder(0.02, 60.0, 60.0, 64).translate((0.0, 0.0, 0.01))
    out = []
    for part in (m ^ slab).decompose():
        b = np.asarray(part.bounding_box(), dtype=float)
        out.append(max(abs(b[0]), abs(b[1]), b[3], b[4]))
    return sorted(out)


def box(x0, x1, y0, y1, z0, z1):
    """A block of space, to ask the part whether it leaves a place free."""
    face = mf.CrossSection.square([max(x1 - x0, 1e-6), max(y1 - y0, 1e-6)])
    return mf.Manifold.extrude(face, max(z1 - z0, 1e-6)).translate((x0, y0, z0))


def shaft_of(n):
    """An axle of the bore, filed flat `flat` millimetres deep on one side, long enough to stand
    out of the hub: what the bore has to take."""
    r = 0.5 * n["bore"]
    out = mf.Manifold.cylinder(n["height"] - 0.2, r, r, 96).translate((0.0, 0.0, 0.1))
    if n["flat"] > 0.0:
        out = out - box(-r - 1.0, r + 1.0, -r - 1.0, -(r - n["flat"]), -1.0, n["height"] + 1.0)
    return out


def nut_in_its_slot(n, size=pulley.SET_SCREW):
    """The nut of the set screw as it stands in its slot: a hexagon of `s` millimetres across its
    flats, lying on its side along the screw with a corner up and a corner down, so it is gripped by
    the walls of the slot, with its inner face `NUT_GAP` millimetres from the bore so an axle of the
    bore still comes in beside it."""
    s, m = threads.HEX_NUTS[size]
    corner = s / math.sqrt(3.0)
    loop = np.asarray([[corner * math.cos(math.radians(60.0 * k)),
                        corner * math.sin(math.radians(60.0 * k))] for k in range(6)])
    prism = mf.Manifold.extrude(mf.CrossSection([loop]), m).rotate((0.0, 90.0, 0.0))
    return prism.translate((0.5 * n["bore"] + pulley.NUT_GAP, 0.0, n["screw_z"]))


def set_settings(win, **over):
    """Every value the pulley dialog keeps, at the defaults of this file with these answers over
    it: no test may depend on what another test left in the settings."""
    for key, value in dict(OPTIONS, **over).items():
        setattr(win.settings, "pulley_" + key, value)


@pytest.fixture()
def win(monkeypatch, tmp_path):
    monkeypatch.setattr(app_module.errors, "log_dir", lambda: str(tmp_path))
    from volura.app import MainWindow
    w = MainWindow()
    set_settings(w)
    w.doc.dirty = False
    QCoreApplication.processEvents()            # handle the deferred initial view now
    w.resize(900, 600)
    yield w
    w.doc.dirty = False
    w.close()


# ------------------------------------------------------------------ the circles of the wheel

def test_the_default_wheel_is_one_valid_volume_that_stands_on_the_bed():
    m, n = wheel(), numbers()
    assert m.status() == mf.Error.NoError, "the wheel is not a valid volume"
    assert not m.is_empty() and m.volume() > 0.0
    assert len(m.decompose()) == 1, "the wheel is not one piece"
    lo, hi = bbox(m)
    assert lo[2] == pytest.approx(0.0), "the wheel does not stand on the bed"
    assert hi[2] == pytest.approx(n["height"]), "the wheel is not as tall as it says"
    assert 0.5 * (lo[0] + hi[0]) == pytest.approx(0.0), "the wheel is not centred"
    assert 0.5 * (lo[1] + hi[1]) == pytest.approx(0.0), "the wheel is not centred"
    assert hi[0] - lo[0] == pytest.approx(max(n["hub_d"], 2.0 * n["rf"]), abs=0.05), \
        "the wheel is not as wide as its collars or its hub"
    assert bed_islands(m) == [pytest.approx(n["rf"], abs=0.05)], "the part does not stand on one"


def test_the_outside_diameter_is_the_pitch_circle_less_twice_the_pitch_line_differential():
    for profile, belt in pulley.BELTS.items():
        for teeth in (10, 20, 40, 120):
            pitch = teeth * belt["pitch"] / math.pi
            assert pulley.pitch_diameter(teeth, profile) == pytest.approx(pitch, abs=1e-9)
            assert pulley.outside_diameter(teeth, profile) == pytest.approx(
                pitch - 2.0 * belt["pld"], abs=1e-9), f"{profile} of {teeth} teeth"
    # the diameters a supplier's table gives for these belts, measured over the teeth
    assert round(pulley.outside_diameter(20, "gt2"), 2) == 12.22
    assert round(pulley.outside_diameter(40, "gt2"), 2) == 24.96
    assert round(pulley.outside_diameter(20, "gt3"), 2) == 18.34
    assert round(pulley.outside_diameter(20, "htd3m"), 2) == 18.34
    n = numbers()
    assert n["od"] == pytest.approx(pulley.outside_diameter(20, "gt2"))
    assert n["pitch_d"] - n["od"] == pytest.approx(2.0 * pulley.BELTS["gt2"]["pld"])


def test_the_teeth_stand_under_the_pitch_circle_and_not_above_it():
    """A belt is flexible and its teeth have to run into the wheel right to the bottom of the
    grooves, so a belt wheel is turned under its pitch circle; a spur gear is turned over it,
    because two stiff wheels touch each other on that circle."""
    from volura import gears
    n = numbers()
    assert n["od"] < n["pitch_d"], "the teeth of a belt wheel stand above the pitch circle"
    assert gears.pitch_diameter(20, 2.0) > pulley.outside_diameter(20, "gt2")
    assert n["root"] < n["od"], "the grooves do not reach under the outside circle"
    assert n["root"] == pytest.approx(n["od"] - 2.0 * (pulley.BELTS["gt2"]["groove"]
                                                       + n["clearance"]))


def test_the_wheel_is_turned_to_that_diameter_and_cut_to_that_groove():
    m, n = wheel(), numbers()
    crest, total = in_tube(m, 0.5 * n["od"] - 0.05, 0.5 * n["od"] - 0.01, n["zb"], n["ze"])
    over, _t = in_tube(m, 0.5 * n["od"] + 0.01, 0.5 * n["od"] + 0.05, n["zb"], n["ze"])
    assert crest > 0.0, "nothing stands on the outside circle the table gives"
    assert crest < total, "the wheel is a drum: it has no grooves"
    assert over == pytest.approx(0.0, abs=1e-6), "the wheel is wider than its outside diameter"
    floor, f_total = in_tube(m, 0.5 * n["root"] - 0.05, 0.5 * n["root"] - 0.01, n["zb"], n["ze"])
    assert floor == pytest.approx(f_total, abs=1e-3), "the grooves reach under the root circle"


@pytest.mark.parametrize("kw", [
    dict(), dict(teeth=10, bore=0.0), dict(teeth=36), dict(teeth=120, width=15.0),
    dict(profile="gt3", teeth=12), dict(profile="gt3", teeth=24),
    dict(profile="htd3m", teeth=15), dict(profile="htd3m", teeth=60),
    dict(flanges="one"), dict(flanges="none"), dict(hub=False), dict(clearance=0.0),
])
def test_a_cross_section_of_the_band_shows_the_grooves_it_was_asked_for(kw):
    m, n = wheel(**kw), numbers(**kw)
    got = arcs(m, 0.5 * (n["zb"] + n["ze"]), 0.5 * n["root"] + 0.05, 0.5 * n["od"] - 0.05)
    assert got == n["teeth"], f"{kw}: the slice shows {got} teeth, not {n['teeth']}"


@pytest.mark.parametrize("kw", [dict(), dict(flanges="one"), dict(flanges="none"),
                                dict(hub=False), dict(teeth=120, width=15.0),
                                dict(profile="htd3m", teeth=24, width=15.0)])
def test_the_grooves_are_open_over_the_whole_height_the_belt_runs_between(kw):
    m, n = wheel(**kw), numbers(**kw)
    r, roof = 0.5 * (0.5 * n["root"] + n["rim"]), roof_z(n)
    for z in (n["zb"] + 0.15, 0.5 * (n["zb"] + roof), roof - 0.05):
        assert not solid(m, r, 0.0, z), f"{kw}: a groove is closed at z={z:.2f}"
    half, land = math.pi / n["teeth"], n["rim"] - 0.08
    assert solid(m, land * math.cos(half), land * math.sin(half), 0.5 * (n["zb"] + n["ze"])), \
        f"{kw}: the land between two grooves is gone"
    assert solid(m, r, 0.0, roof + 0.05), f"{kw}: the wheel has nothing over the belt to carry it"


# ------------------------------------------------------------------ the belt and its collars

@pytest.mark.parametrize("flanges", ("both", "one", "none"))
def test_the_collars_stand_proud_of_the_teeth_and_the_belt_has_room_between_them(flanges):
    m, n = wheel(flanges=flanges), numbers(flanges=flanges)
    proud = n["rf"] - n["rim"]
    if flanges == "none":
        assert proud == pytest.approx(0.0), "a wheel without collars still has one"
    else:
        assert proud == pytest.approx(n["flange"]), \
            f"{flanges}: the collars do not stand {n['flange']} mm past the teeth"
        band, total = in_tube(m, n["rim"] + 0.02, n["rf"] - 0.02, 0.05, pulley.FLANGE_T - 0.05)
        assert band == pytest.approx(total, abs=1e-3), "the collar on the bed is not a full ring"
    if flanges == "both":                # the collar in the air is a ring over its whole height
        band, total = in_tube(m, n["rim"] + 0.02, n["rf"] - 0.02, n["zs"] + 0.05, n["zt"] - 0.05)
        assert band == pytest.approx(total, abs=1e-3), "the collar in the air is not a full ring"
    # the body of the belt runs from the crest circle out to under the collars, over its whole
    # width, and the part may not be in that space
    free, _t = in_tube(m, n["rim"] + 0.01, n["rf"] - 0.01, n["zb"], n["zb"] + n["width"])
    assert free == pytest.approx(0.0, abs=1e-6), f"{flanges}: the part is in the way of the belt"
    near = n["zb"] - (pulley.FLANGE_T if n["near"] else 0.0)
    assert near == pytest.approx(0.5 * pulley.SIDE_ROOM if n["near"] else 0.0)
    assert roof_z(n) - (n["zb"] + n["width"]) == pytest.approx(0.5 * pulley.SIDE_ROOM), \
        f"{flanges}: a belt of {n['width']} mm has no room at its free end"


def test_a_moulded_belt_tooth_fits_the_groove_and_the_clearance_is_on_the_groove():
    belt = pulley.BELTS["gt2"]
    n = numbers()
    assert n["mouth"] == pytest.approx(belt["mouth"] + 2.0 * n["clearance"])
    assert n["groove"] == pytest.approx(belt["groove"] + n["clearance"])
    assert n["mouth"] < math.pi * n["od"] / n["teeth"], "the grooves touch one another"
    assert n["tooth"] == pytest.approx(belt["tooth"])
    assert 0.5 * n["root"] < 0.5 * n["od"] - n["tooth"], "a belt tooth cannot reach its depth"
    # what the clearance gives a groove is room; the wheel keeps the diameter of the table. Only a
    # wheel with room between its grooves can take it; a small one would see its grooves merge.
    thin, deep = numbers(teeth=60, clearance=0.0), numbers(teeth=60, clearance=0.2)
    assert deep["od"] == pytest.approx(thin["od"]) and deep["pitch_d"] == pytest.approx(thin["pitch_d"])
    assert deep["root"] < thin["root"], "a deeper clearance did not deepen the grooves"
    assert deep["mouth"] > thin["mouth"], "a wider clearance did not open the mouths"
    for teeth in (10, 20, 40):
        with pytest.raises(ValueError) as exc:
            numbers(teeth=teeth, clearance=pulley.CLEAR_MAX)
        assert "grooves" in str(exc.value).lower(), str(exc.value)


@pytest.mark.parametrize("kw", [dict(clearance=0.0), dict(teeth=60, clearance=0.2),
                                dict(width=pulley.BELT_MIN), dict(width=pulley.BELT_MAX),
                                dict(flange=pulley.FLANGE_MIN), dict(flange=pulley.FLANGE_MAX),
                                dict(teeth=10, bore=0.0), dict(teeth=120, width=15.0, flange=2.0),
                                dict(hub_d=pulley.HUB_MAX, hub_h=pulley.HUBH_MAX),
                                dict(bore=0.0), dict(hub=False), dict(flanges="none"),
                                dict(flanges="one", hub_h=pulley.HUBH_MIN),
                                dict(flanges="one", hub_d=pulley.HUB_MAX, bore=0.0)])
def test_a_wheel_prints_bare_whatever_it_was_asked(kw):
    """Nothing looks down by more than 45 degrees from the vertical: the face on the bed, the slope
    under the collar in the air, the cone of a hub that is wider than the face it stands on and the
    roof over the mouth of a groove are all that leans over, so the wheel prints without support in
    every choice the dialog offers."""
    m, n = wheel(**kw), numbers(**kw)
    assert m.status() == mf.Error.NoError and len(m.decompose()) == 1, f"{kw}: not one volume"
    lean, radius, z, area = needs_support(m, n)
    assert not len(lean), (f"{kw}: something leans over that should not: "
                           f"{sorted(set(np.round(lean, 1)))[:6]} at r="
                           f"{[round(q, 2) for q in radius[:6]]}, z={[round(q, 2) for q in z[:6]]}")
    assert bed_islands(m) == [pytest.approx(n["rf"] if n["near"] else n["rim"], abs=0.12)], \
        f"{kw}: the part does not stand on one ring"


def test_the_roof_over_a_groove_bridges_no_more_than_the_mouth_of_that_groove():
    m, n = wheel(), numbers()
    lean, radius, z, area = looks_down(m)
    flat = (lean > 85.0) & ~on_tunnel(radius, z, n)
    roof = flat & (np.abs(z - roof_z(n)) < 1e-6)
    assert roof.any(), "a groove was expected to have a roof above the belt"
    assert area[flat & ~roof].sum() == pytest.approx(0.0), "something else looks straight down"
    assert radius[roof].min() >= 0.5 * n["root"] - 0.05, "a roof reaches into the wheel"
    assert area[roof].sum() < n["teeth"] * n["mouth"] * (n["groove"] + 0.5), \
        "the roofs cover more of the wheel than the mouths of the grooves"


def test_the_slope_under_the_collar_in_the_air_is_45_degrees():
    m, n = wheel(), numbers()
    lean, radius, z, _area = looks_down(m)
    allowed = on_tunnel(radius, z, n) | hub_cone(radius, z, n)
    slope = (lean < 85.0) & ~allowed
    assert slope.any() and set(np.round(lean[slope], 1)) == {45.0}, \
        "the collar in the air does not rise out of the band on a slope of 45 degrees"
    assert (z[slope] > n["ze"] - 1e-6).all() and (z[slope] < n["zs"] + 1e-6).all(), \
        "the slope is not between the belt and the collar that is in the air"
    # the hub of these numbers is wider than the collar it stands on, so a cone carries it; standing
    # SINK into the wheel makes that cone steeper than that slope of 45 degrees, which prints bare
    _stand, climb = hub_stand(n)
    assert climb > 1e-9, "this wheel was expected to stand a hub wider than its own face"
    cone = hub_cone(radius, z, n)
    assert cone.any(), "the cone that carries the hub is not there"
    assert set(np.round(lean[cone], 1)) == {
        round(math.degrees(math.atan(climb / (climb + pulley.SINK))), 1)}, \
        "the cone under the hub is not the slope it was calculated to be"


def test_a_wheel_without_collars_is_a_plain_toothed_ring_that_bridges_only_its_grooves():
    m, n = wheel(flanges="none"), numbers(flanges="none")
    lean, radius, z, _a = looks_down(m)
    assert len(lean) > 0, "a toothed wheel bridges nothing"
    flat_or_lean = (lean > 85.0) | on_tunnel(radius, z, n) | hub_cone(radius, z, n)
    assert flat_or_lean.all(), "a ring with no collars has a slope of its own: " \
        f"{sorted(set(np.round(lean[~flat_or_lean], 1)))}"
    assert not len(needs_support(m, n)[0]), "the ring hangs over more than the mouths of its grooves"
    assert n["rf"] == pytest.approx(n["rim"]) and n["zb"] == pytest.approx(0.0)
    assert arcs(m, 0.5 * n["ze"], 0.5 * n["root"] + 0.05, 0.5 * n["od"] - 0.05) == n["teeth"]


# ------------------------------------------------------------------ the hub, the bore, the screw

def test_the_bore_is_a_hole_straight_through_the_part():
    m, n = wheel(), numbers()
    assert n["bore"] == pytest.approx(5.0)
    assert m.genus() == (2 if n["screw"] > 0.0 else 1), \
        "the bore and the tunnel of the set screw are the only holes in the part"
    hole, _t = in_tube(m, 0.0, 0.5 * n["bore"] - 0.02, 0.05, n["height"] - 0.05)
    assert hole == pytest.approx(0.0, abs=1e-6), "the bore is blocked somewhere"
    assert wheel(bore=0.0).genus() == 0, "a wheel without a bore has a hole in it"
    assert numbers(bore=0.0)["screw"] == 0.0, "a set screw would screw into nothing"


@pytest.mark.parametrize("kw", [dict(), dict(flat=1.0), dict(flat=1.0, hub_d=pulley.HUB_MIN),
                                dict(flat=2.0, bore=8.0, teeth=40), dict(flat=0.4, bore=2.0),
                                dict(flat=1.0, flanges="none"),
                                dict(flat=1.0, hub_d=pulley.HUB_MAX, bore=8.0, teeth=40)])
def test_the_bore_takes_an_axle_with_a_flat_and_the_hub_keeps_its_wall_all_round(kw):
    n, m = numbers(**kw), wheel(**kw)
    assert (m ^ shaft_of(n)).volume() == pytest.approx(0.0, abs=1e-6), f"{kw}: the axle rubs"
    if not n["hub"]:
        return
    if n["screw"] > 0.0:                         # under the screw and the slot that feeds it, so
        z = n["slot_z"] - 0.5                    # the probe only meets the wall of the hub
        assert z > n["hub_base"] + 0.2, f"{kw}: the hub has no wall under its screw"
    else:
        z = 0.5 * (n["hub_base"] + n["hub_top"])
    for k in range(12):                          # the flat is a chord of the bore, not a slot
        a = 2.0 * math.pi * k / 12.0
        walk = 0.5 * n["hub_d"] - 0.25
        assert solid(m, walk * math.cos(a), walk * math.sin(a), z), \
            f"{kw}: the flat cut the hub open at {math.degrees(a):.0f}°"
    if n["flat"] > 0.0:                          # what stands under the plane of the flat stays
        assert solid(m, 0.0, -(0.5 * n["bore"] + 0.2), z), f"{kw}: the flat is cut too deep"


def test_the_hub_is_grown_until_it_holds_the_bore_with_a_wall():
    thin = numbers(hub_d=pulley.HUB_MIN, bore=8.0, teeth=40, nut=False)
    assert thin["hub_d"] == pytest.approx(8.0 + 2.0 * pulley.WALL_MIN), "the hub stayed too thin"
    assert thin["hub_top"] - thin["hub_base"] == pytest.approx(thin["hub_h"])
    assert wheel(hub_d=pulley.HUB_MIN, bore=8.0, teeth=40, nut=False).genus() == 1
    wide = numbers(hub_d=pulley.HUB_MAX)
    assert wide["height"] == pytest.approx(wide["hub_top"]), "the hub does not crown the part"
    off = numbers(hub=False)
    assert off["hub_d"] == 0.0 and off["height"] == pytest.approx(off["zt"]), \
        "a wheel without a hub is still as tall as one"


def test_the_hub_is_grown_until_the_nut_stands_behind_a_wall_of_it():
    s, m_nut = threads.HEX_NUTS[pulley.SET_SCREW]
    for kw in (dict(), dict(hub_d=pulley.HUB_MIN), dict(hub_d=pulley.HUB_MIN, bore=8.0, teeth=40)):
        n, m = numbers(**kw), wheel(**kw)
        assert n["slot_x"] == pytest.approx(0.5 * n["bore"] + pulley.NUT_GAP + m_nut
                                            + pulley.NUT_PLAY), f"{kw}: the wrong slot depth"
        assert 0.5 * n["hub_d"] - n["slot_x"] >= pulley.NUT_WALL - 1e-9, \
            f"{kw}: the wall in front of the nut is too thin"
        assert n["screw"] > 0.0, f"{kw}: a hub that fits the nut lost its screw"
        assert (m ^ nut_in_its_slot(n)).volume() == pytest.approx(0.0, abs=1e-6), \
            f"{kw}: the nut does not fit in the hub that was grown for it"
    thin = numbers(hub_d=pulley.HUB_MIN)
    assert thin["hub_d"] >= 2.0 * (0.5 * thin["bore"] + pulley.NUT_GAP + m_nut + pulley.NUT_PLAY
                                   + pulley.NUT_WALL), "a hub of 6 mm stayed too thin for an M3 nut"


def test_the_set_screw_runs_in_from_outside_the_hub_straight_to_the_axle():
    m, n = wheel(), numbers()
    assert n["screw"] == pytest.approx(threads.SCREW_HOLES[pulley.SET_SCREW][0]), \
        "the hole is not the tapping hole of an M3"
    assert n["hub_base"] < n["screw_z"] < n["hub_top"], "the screw is not in the hub"
    assert not solid(m, 0.5 * n["hub_d"] - 0.3, 0.0, n["screw_z"]), "the screw hole is not drilled"
    assert not solid(m, 0.5 * n["bore"] + 0.3, 0.0, n["screw_z"]), "the hole stops before the bore"
    assert solid(m, -(0.5 * n["hub_d"] - 0.3), 0.0, n["screw_z"]), \
        "the far side of the hub is drilled open as well"
    wall, half = 0.5 * n["hub_d"] - 0.4, 0.5 * n["slot_w"] + 0.5
    assert beside(m, wall, half, n["screw_z"]), "the wall beside the screw is gone"
    assert beside(m, wall, -half, n["screw_z"]), "the wall beside the screw is gone"
    assert m.genus() == 2, "the screw has no tunnel through the wall in front of its nut"


@pytest.mark.parametrize("kw", [dict(nut=False), dict(bore=0.0), dict(hub=False),
                                dict(hub_h=pulley.HUBH_MIN),
                                dict(hub_d=pulley.HUB_MIN, hub_h=pulley.HUBH_MIN),
                                dict(hub_d=pulley.HUB_MIN + 2.0, bore=8.0, teeth=40,
                                     hub_h=pulley.HUBH_MIN),
                                dict(hub_h=7.0)])
def test_a_screw_the_hub_has_no_room_for_is_left_out(kw):
    n, m = numbers(**kw), wheel(**kw)
    assert m.status() == mf.Error.NoError and m.volume() > 0.0, f"{kw}: nothing left to print"
    assert len(m.decompose()) == 1, f"{kw}: the part fell apart"
    if n["screw"] > 0.0:
        assert n["nut"] and n["hub"] and n["bore"] > 0.0, f"{kw}: a screw that screws into nothing"
        assert n["hub_top"] - n["screw_z"] >= 0.0 and not solid(
            m, 0.5 * n["hub_d"] - 0.3, 0.0, n["screw_z"]), f"{kw}: the screw hole is missing"
    elif n["hub"]:
        assert solid(m, 0.5 * n["hub_d"] - 0.3, 0.0, 0.5 * (n["hub_base"] + n["hub_top"])), \
            f"{kw}: a hole was drilled anyway"


def test_only_a_hub_that_is_too_short_for_the_nut_loses_its_screw():
    s = threads.HEX_NUTS[pulley.SET_SCREW][0]
    room = 2.0 * s / math.sqrt(3.0) + pulley.NUT_FLOOR + pulley.NUT_ROOM
    thin = numbers(hub_d=pulley.HUB_MIN)
    assert thin["hub_d"] > pulley.HUB_MIN and thin["screw"] > 0.0, \
        "a hub that is only too thin for the nut lost its screw instead of growing"
    assert numbers(hub_h=room)["screw"] > 0.0, "a hub with just enough room lost its screw"
    assert numbers(hub_h=room - 0.1)["screw"] == 0.0, "a hub too short for the nut kept its screw"
    assert numbers(nut=False)["slot_x"] == 0.0, "a wheel without a screw got a nut slot anyway"
    assert pulley.sizes(**dict(BUSH, bearing=True), idler=True)["slot_x"] == 0.0, \
        "an idler got a nut slot"


def test_the_nut_of_that_screw_stands_in_a_slot_that_is_closed_on_the_outside():
    m, n = wheel(), numbers()
    s, m_nut = threads.HEX_NUTS[pulley.SET_SCREW]
    assert n["slot_w"] == pytest.approx(s + pulley.NUT_PLAY), "the slot is not the nut plus its play"
    assert n["slot_z"] == pytest.approx(n["screw_z"] - s / math.sqrt(3.0) - pulley.NUT_FLOOR)
    assert n["slot_x"] == pytest.approx(0.5 * n["bore"] + pulley.NUT_GAP + m_nut
                                        + pulley.NUT_PLAY), "the slot is not as deep as the nut"
    assert 0.5 * n["hub_d"] - n["slot_x"] >= pulley.NUT_WALL - 1e-9, \
        "the wall left in front of the nut is thinner than NUT_WALL"
    assert n["slot_z"] > n["hub_base"], "the slot cuts through the wheel it stands on"
    corner = s / math.sqrt(3.0)
    assert n["screw_z"] - corner > n["slot_z"], "the nut has no floor to stand on"
    assert n["screw_z"] + corner < n["hub_top"], "the nut cannot drop into the hub from above"
    assert m.genus() == 2, "the screw has no tunnel through the wall in front of the nut"
    # the channel of the nut is empty from its inner face to its outer face and from its floor to
    # over the top of the part, so the nut drops in from above and nothing hangs over the bed
    channel = box(0.5 * n["bore"] + pulley.NUT_GAP + 0.05, n["slot_x"] - 0.05,
                  -0.5 * n["slot_w"] + 0.05, 0.5 * n["slot_w"] - 0.05,
                  n["slot_z"] + 0.05, n["height"] + 0.5)
    assert (m ^ channel).volume() == pytest.approx(0.0, abs=1e-6), "the slot of the nut is not free"
    assert solid(m, 0.5 * n["hub_d"] - 0.4, 0.0, n["slot_z"] - 0.3), "the slot has no floor"
    grip, side = 0.5 * n["hub_d"] - 0.4, 0.5 * n["slot_w"] + 0.4
    assert beside(m, grip, side, n["screw_z"]), "the walls that grip the nut are not there"
    assert beside(m, grip, -side, n["screw_z"]), "the walls that grip the nut are not there"
    # the wall in front of the nut stays: it is only pierced by the hole of the screw, so the screw
    # bears on the nut instead of pushing the nut out of the hub
    front = 0.5 * (n["slot_x"] + 0.5 * n["hub_d"])
    for side in (0.5 * n["slot_w"] - 0.3, -(0.5 * n["slot_w"] - 0.3)):
        assert beside(m, front, side, n["screw_z"]), \
            f"the wall in front of the nut is cut away at y={side:.2f}"
    assert not solid(m, front, 0.0, n["screw_z"]), "the hole of the screw does not reach the nut"
    assert (m ^ nut_in_its_slot(n)).volume() == pytest.approx(0.0, abs=1e-6), \
        "the nut of the screw does not fit its slot"
    assert (nut_in_its_slot(n) ^ shaft_of(n)).volume() == pytest.approx(0.0, abs=1e-6), \
        "the nut stands in the bore the axle has to go into"
    assert m_nut < n["slot_x"] - 0.5 * n["bore"], "no room for a nut between the bore and the wall"


# ------------------------------------------------------------------ the idler

@pytest.mark.parametrize("profile", pulley.PROFILES)
def test_the_idler_runs_on_the_circle_the_back_of_its_belt_stands_on(profile):
    belt = pulley.BELTS[profile]
    back = belt["thickness"] - belt["pld"] - belt["tooth"]
    assert pulley.back_depth(profile) == pytest.approx(BACK[profile], abs=5e-4), \
        "the back of the belt does not stand where the published tooth height puts it"
    assert pulley.back_depth(profile) == pytest.approx(
        max(pulley.BACK_MIN, min(pulley.BACK_MAX, back)))
    for teeth in (10, 20, 40):
        pitch = teeth * belt["pitch"] / math.pi
        assert pulley.idler_diameter(teeth, profile) == pytest.approx(
            pitch - 2.0 * max(pulley.BACK_MIN, min(pulley.BACK_MAX, back)), abs=1e-9)
    n = pulley.sizes(**dict(BUSH, profile=profile), idler=True)
    assert n["surface"] == pytest.approx(pulley.idler_diameter(20, profile))
    assert n["surface"] < n["pitch_d"], "the idler runs outside the pitch circle of its belt"
    assert n["pitch_d"] - n["surface"] == pytest.approx(2.0 * pulley.back_depth(profile))


def test_a_plain_idler_is_a_smooth_wheel_bored_for_a_bush():
    m, n = pulley.idler(**BUSH), pulley.sizes(**BUSH, idler=True)
    assert m.status() == mf.Error.NoError and len(m.decompose()) == 1
    assert m.genus() == 1, "the idler is not bored through"
    z = 0.5 * (n["zb"] + n["ze"])
    assert arcs(m, z, 0.5 * n["bore"] + 0.5, 0.5 * n["surface"] - 0.05) == 1, \
        "the running surface of an idler is toothed"
    crest, total = in_tube(m, 0.5 * n["surface"] - 0.05, 0.5 * n["surface"] - 0.01,
                           n["zb"], n["ze"])
    over, _t = in_tube(m, 0.5 * n["surface"] + 0.01, 0.5 * n["surface"] + 0.05, n["zb"], n["ze"])
    assert crest == pytest.approx(total, abs=1e-3), "the running surface is not round"
    assert over == pytest.approx(0.0, abs=1e-6), "the idler is wider than its running surface"
    assert n["rf"] - 0.5 * n["surface"] == pytest.approx(n["flange"]), "the collars are not proud"
    free, _t = in_tube(m, n["rim"] + 0.01, n["rf"] - 0.01, n["zb"], n["zb"] + n["width"])
    assert free == pytest.approx(0.0, abs=1e-6), "the idler is in the way of its belt"


def test_an_idler_for_a_bearing_is_pocketed_bored_and_prints_bare():
    m, n = pulley.idler(**ROLLER), pulley.sizes(**ROLLER, idler=True)
    assert n["bore"] == pytest.approx(pulley.BEARING_BORE), "a bearing let the bush bore through"
    assert n["pocket"] == pytest.approx(pulley.BEARING_D + 2.0 * pulley.BEARING_PLAY)
    assert n["surface"] >= n["pocket"] + 2.0 * pulley.WEB_MIN - 1e-9, "the wall is too thin"
    pocket, _t = in_tube(m, 0.0, 0.5 * n["pocket"] - 0.02, n["zt"] - pulley.BEARING_W + 0.02,
                         n["zt"] - 0.02)
    assert pocket == pytest.approx(0.0, abs=1e-6), "the pocket of the bearing is not cut"
    floor = n["zt"] - pulley.BEARING_W - pulley.POCKET_OVER
    # the ring of floor under the outer ring of the bearing stays solid: that is where the wheel is
    # held, and it is a ring because the hole under it is wider than the bearing bore
    seat, s_total = in_tube(m, 0.5 * pulley.BEARING_RELIEF + 0.2, 0.5 * n["pocket"] + 0.4,
                            floor - 0.4, floor - 0.1)
    assert seat == pytest.approx(s_total, abs=1e-3), "no floor is left under the outer ring"
    relief, r_total = in_tube(m, 0.0, 0.5 * pulley.BEARING_RELIEF - 0.02, 0.05, floor - 0.05)
    assert relief == pytest.approx(0.0, abs=1e-6), "the hole under the pocket is not wide enough"
    # what stands still goes through that hole without touching the wheel that has to turn round it:
    # an axle of the bearing bore and a disc as wide as the inner ring of a 625 bearing
    for what, r in (("axle", 0.5 * pulley.BEARING_BORE), ("inner ring", 4.5)):
        still = mf.Manifold.cylinder(floor + 1.0, r, r, 96).translate((0.0, 0.0, -1.0))
        assert (m ^ still).volume() == pytest.approx(0.0, abs=1e-6), f"the {what} rubs on the wheel"
    assert m.genus() == 1 and len(m.decompose()) == 1
    lean, _r, _z, _a = needs_support(m, n)
    assert not len(lean), f"the idler hangs over: {sorted(set(np.round(lean, 1)))[:6]}"
    assert bed_islands(m) == [pytest.approx(n["rf"], abs=0.05)], "the idler stands on more than one"


def test_a_small_wheel_grows_until_the_bearing_stands_in_it_with_a_wall():
    small = pulley.sizes(idler=True, teeth=10, profile="gt2", bearing=True, width=6.0,
                         flange=1.0, bore=5.0, flat=0.0)
    assert small["surface"] == pytest.approx(small["pocket"] + 2.0 * pulley.WEB_MIN)
    assert small["surface"] > pulley.idler_diameter(10, "gt2"), "the wheel stayed too small"
    big = pulley.sizes(idler=True, teeth=120, bearing=True, width=6.0, flange=1.0, bore=5.0,
                       flat=0.0, profile="gt2")
    assert big["surface"] == pytest.approx(pulley.idler_diameter(120, "gt2")), \
        "a wheel with room for the bearing was grown anyway"


@pytest.mark.parametrize("kw", [dict(), dict(teeth=10, bore=1.5), dict(teeth=120, width=20.0),
                                dict(bearing=False), dict(bearing=False, flat=1.0),
                                dict(profile="gt3", teeth=12), dict(profile="htd3m", teeth=24),
                                dict(flange=pulley.FLANGE_MAX), dict(width=pulley.BELT_MIN)])
def test_an_idler_prints_bare_whatever_it_was_asked(kw):
    size = dict(dict(BUSH, bearing=False), **kw)
    m, n = pulley.idler(**size), pulley.sizes(**size, idler=True)
    assert m.status() == mf.Error.NoError and len(m.decompose()) == 1, f"{kw}: not one volume"
    lean, radius, z, _a = needs_support(m, n)
    assert not len(lean), (f"{kw}: something leans over: {sorted(set(np.round(lean, 1)))[:6]} "
                           f"at r={[round(q, 2) for q in radius[:6]]}")


def test_an_idler_refuses_a_bush_that_leaves_too_little_wheel_around_it():
    with pytest.raises(ValueError):
        pulley.idler(**dict(BUSH, bore=11.0))
    with pytest.raises(ValueError):
        pulley.idler(**dict(BUSH, teeth=20.5))
    with pytest.raises(ValueError):
        pulley.idler(**dict(BUSH, profile="gt9"))


# ------------------------------------------------------------------ sizes that have to give

@pytest.mark.parametrize("kw", [
    dict(width=1.0), dict(width=99.0), dict(clearance=-1.0), dict(flange=0.0),
    dict(flange=99.0), dict(hub_d=1.0), dict(hub_h=1.0), dict(hub_h=99.0), dict(bore=-2.0),
    dict(flat=99.0), dict(hub_d=pulley.HUB_MAX, hub_h=pulley.HUBH_MAX, width=pulley.BELT_MAX,
                           flange=pulley.FLANGE_MAX, teeth=120),
])
def test_a_size_that_cannot_be_printed_becomes_the_nearest_one_that_can(kw):
    size = dict(BASE, **kw)
    n, m = pulley.sizes(**size), pulley.timing_pulley(**size)
    assert pulley.BELT_MIN <= n["width"] <= pulley.BELT_MAX, f"{kw}: the belt width"
    assert 0.0 <= n["clearance"] <= pulley.CLEAR_MAX, f"{kw}: the clearance"
    assert pulley.FLANGE_MIN <= n["flange"] <= pulley.FLANGE_MAX, f"{kw}: the collar"
    assert pulley.HUB_MIN <= n["hub_d"] <= pulley.HUB_MAX, f"{kw}: the hub"
    assert pulley.HUBH_MIN <= n["hub_h"] <= pulley.HUBH_MAX, f"{kw}: the height of the hub"
    assert 0.0 <= n["bore"] <= n["surface"], f"{kw}: the bore"
    assert 0.0 <= n["flat"] <= 0.4 * n["bore"] + 1e-9, f"{kw}: the flat"
    assert n["teeth"] == size["teeth"], f"{kw}: the tooth count was changed"
    assert m.status() == mf.Error.NoError and m.volume() > 0.0, f"{kw}: nothing left to print"
    assert len(m.decompose()) == 1, f"{kw}: the part fell apart"
    assert m.bounding_box()[2] == pytest.approx(0.0), f"{kw}: the part does not stand on the bed"


@pytest.mark.parametrize("kw", [
    dict(bore=11.5), dict(bore=99.0), dict(profile="gt5"), dict(profile=""), dict(profile=None),
    dict(teeth=9), dict(teeth=121), dict(teeth=20.5), dict(teeth="many"), dict(teeth=None),
    dict(flanges="left"), dict(flanges=None), dict(width="wide"), dict(clearance=None),
    dict(hub_d=None), dict(hub_h="ten"), dict(bore=None), dict(flat="flat"),
    dict(teeth=10, clearance=0.5), dict(teeth=20, clearance=0.5), dict(clearance=pulley.CLEAR_MAX,
                                                                       teeth=11),
])
def test_a_wheel_that_cannot_take_its_belt_is_refused_with_a_message(kw):
    size = dict(BASE, **kw)
    with pytest.raises(ValueError) as exc:
        pulley.timing_pulley(**size)
    assert str(exc.value), "the refusal says nothing"
    with pytest.raises(ValueError):
        pulley.sizes(**size)


def test_a_bore_that_would_eat_the_bottom_of_the_grooves_is_refused():
    with pytest.raises(ValueError) as exc:
        pulley.timing_pulley(**dict(BASE, teeth=20, bore=11.5))
    assert "bore" in str(exc.value).lower()
    # the same bore on a wheel of the same belt has room, because that wheel is larger
    assert pulley.sizes(**dict(BASE, teeth=60, bore=20.0))["bore"] == pytest.approx(20.0)


# ------------------------------------------------------------------ a real slice

def _mesh(m):
    placed = m.to_mesh()
    return np.asarray(placed.vert_properties)[:, :3], np.asarray(placed.tri_verts), 0


def _first_layer_moves(text):
    """The (x, y) of every extrusion move of the first layer, as the printer lays them down."""
    body = text.split(";LAYER:1", 1)[0].split(";LAYER:0", 1)[-1]
    x = y = 0.0
    points = []
    for line in body.splitlines():
        code = line.split(";", 1)[0].strip()
        if not code.startswith(("G0", "G1")):
            continue
        got = {a: re.search(rf"\b{a}(-?\d*\.?\d+)", code) for a in "XYZE"}
        for axis in ("X", "Y"):
            if got[axis] is not None:
                if axis == "X":
                    x = float(got[axis].group(1))
                else:
                    y = float(got[axis].group(1))
        if got["E"] is not None:
            points.append((x, y))
    return points


@needs_engine
@pytest.mark.parametrize("part, hole", [
    (wheel, 0.0),                                       # a pulley is bored right through the middle
    (lambda: pulley.idler(**ROLLER),                    # an idler stands on the ring round the hole
     0.5 * pulley.BEARING_RELIEF)])                     # that is relieved under the bearing pocket
def test_the_part_slices_without_support_and_stands_on_the_first_layer(tmp_path, part, hole):
    lib = cura_profiles.Library(RESOURCES)
    p = lib.printer("creality_ender3")
    v, m = p.default_variant(), p.default_material(p.default_variant())
    stacks = p.stacks(v, m, "standard", {"infill_sparse_density": 0.0, "support_enabled": False})
    out = tmp_path / "pulley.gcode"
    slicer.slice_meshes(stacks, [_mesh(part())], str(out))
    text = out.read_text(encoding="utf-8")
    assert ";TYPE:SUPPORT" not in text, "the part needs support it is not meant to have"
    assert text.count(";LAYER:") > 10, "barely anything was sliced"
    moves = _first_layer_moves(text)
    assert moves, "nothing is extruded on the first layer"
    xs, ys = [q[0] for q in moves], [q[1] for q in moves]
    middle = (0.5 * (min(xs) + max(xs)), 0.5 * (min(ys) + max(ys)))
    machine = slicer.resolved_settings(stacks)["global"]
    assert middle[0] == pytest.approx(0.5 * float(machine["machine_width"]), abs=2.0), \
        "the part is not printed in the middle of the bed"
    assert middle[1] == pytest.approx(0.5 * float(machine["machine_depth"]), abs=2.0)
    reach = [math.hypot(px - middle[0], py - middle[1]) for px, py in moves]
    assert max(reach) > 5.0, "the part does not cover a wide ring on the bed"
    assert min(reach) < hole + 3.0, \
        f"the plastic on the bed does not come within 3 mm of the {hole:.1f} mm hole in the middle"


# ------------------------------------------------------------------ in the window

def test_the_wheel_comes_into_the_model_as_one_undo_step(win):
    wanted = wheel()
    steps = len(win.doc.undo_stack)
    win.make_pulley_model(OPTIONS)
    assert pump_until(lambda: len(win.doc.groups) == 1), "no pulley in the model"
    assert len(win.doc.undo_stack) == steps + 1, "the wheel is not one undo step"
    group = win.doc.groups[0]
    assert group.name == "Timing pulley"
    assert pump_until(lambda: group.ents.to_manifold().volume() > 0.0)
    got = group.ents.to_manifold()
    assert got.status() == mf.Error.NoError
    assert got.volume() == pytest.approx(wanted.volume(), rel=1e-3), "not the wheel that was built"
    lo, hi = bbox(got)
    assert lo[2] == pytest.approx(0.0), "the wheel does not stand on the bed"
    bx, by, _bz = win.settings.bed_size
    assert 0.5 * (lo[0] + hi[0]) == pytest.approx(0.5 * bx), "the wheel is not on the bed"
    assert 0.5 * (lo[1] + hi[1]) == pytest.approx(0.5 * by), "the wheel is not on the bed"
    assert pump_until(lambda: "pulley" in win.hint.text().lower()), win.hint.text()
    win.doc.undo()
    assert win.doc.groups == []


def test_a_smooth_wheel_comes_in_as_a_belt_idler_next_to_the_first(win):
    win.make_pulley_model(dict(OPTIONS, idler=True, hub=False, nut=False, bearing=True))
    assert pump_until(lambda: len(win.doc.groups) == 1), "no idler in the model"
    assert win.doc.groups[0].name == "Belt idler"
    assert pump_until(lambda: win.doc.groups[0].ents.to_manifold().volume() > 0.0)
    assert win.doc.groups[0].ents.to_manifold().genus() == 1, "the idler is not bored through"
    first = np.asarray(win.doc.groups[0].bbox(), dtype=float)
    win.make_pulley_model(OPTIONS)
    assert pump_until(lambda: len(win.doc.groups) == 2), "the pulley is missing"
    assert pump_until(lambda: all(g.bbox() is not None for g in win.doc.groups))
    second = np.asarray(win.doc.groups[1].bbox(), dtype=float)
    assert second[0][0] - first[1][0] == pytest.approx(10.0), "the parts are not 10 mm apart"
    assert [g.name for g in win.doc.groups] == ["Belt idler", "Timing pulley"]


def test_the_pulley_stands_under_the_standoffs_in_parts(win):
    labels = [clean(a.text()) for a in win.flat_actions(win.parts_category("Mechanical"))]
    assert labels.index("Timing Pulley") == labels.index("Standoff") + 1
    assert win.a_timing_pulley in win.parts_category("Mechanical").actions()


def test_a_wheel_that_cannot_be_built_leaves_the_model_as_it_was(win):
    win.make_pulley_model(dict(OPTIONS, bore=11.5))
    assert pump_until(lambda: "Timing pulley failed" in win.hint.text()), win.hint.text()
    assert "bore" in win.hint.text().lower(), win.hint.text()
    assert win.doc.groups == []
    win.make_pulley_model(dict(OPTIONS, idler=True, teeth=10, bore=5.0))
    assert pump_until(lambda: "Belt idler failed" in win.hint.text()), win.hint.text()
    assert win.doc.groups == [], "a part that could not be made still came into the model"


def test_the_dialog_starts_at_a_thing_when_the_stored_choice_is_gone(win, monkeypatch):
    seen = {}

    def fill(dlg):
        seen["start"] = [dlg.belt.currentData(), dlg.flanges.currentData(), int(dlg.teeth.value()),
                         round(dlg.width.value(), 2), round(dlg.hub_d.value(), 2),
                         round(dlg.bore.value(), 2)]
        seen["info"] = (dlg.info.text(), dlg.info.styleSheet())
        seen["ok"] = dlg.bb.button(QDialogButtonBox.Ok).isEnabled()
        return QDialog.Accepted

    set_settings(win, profile="gt9", flanges="left", teeth=999, width=1.0, hub_d=1.0, bore=2.0,
                 nut=False)
    monkeypatch.setattr(app_module.QDialog, "exec", fill)
    win.pulley_dialog()
    assert seen["start"] == ["gt2", "none", pulley.TEETH_MAX, pulley.BELT_MIN, pulley.HUB_MIN, 2.0]
    assert seen["ok"], "a wheel that prints was blocked: " + seen["info"][0]
    assert seen["info"][0].startswith("<i>") and seen["info"][1] == "", seen["info"]
    assert "120 grooves" in seen["info"][0], seen["info"][0]
    assert pump_until(lambda: len(win.doc.groups) == 1), "no pulley in the model"
    assert win.doc.groups[0].name == "Timing pulley"
    made = win.doc.groups[0].ents.to_manifold()
    assert pump_until(lambda: made.volume() > 0.0)
    assert made.status() == mf.Error.NoError and len(made.decompose()) == 1
    back = app_module.Settings()
    assert (back.pulley_profile, back.pulley_flanges, back.pulley_teeth) == (
        "gt2", "none", pulley.TEETH_MAX)
    assert back.pulley_width == pytest.approx(pulley.BELT_MIN)
    assert back.pulley_hub_d == pytest.approx(pulley.HUB_MIN)


def test_the_dialog_starts_at_the_last_values_and_remembers_the_new_ones(win, monkeypatch):
    seen = {}

    def fill(dlg):
        seen["title"] = dlg.windowTitle()
        seen["start"] = [dlg.kind.currentData(), dlg.belt.currentData(), int(dlg.teeth.value()),
                         round(dlg.width.value(), 2), round(dlg.clearance.value(), 2),
                         dlg.flanges.currentData(), round(dlg.flange.value(), 2),
                         dlg.hub.isChecked(), round(dlg.hub_d.value(), 2),
                         round(dlg.hub_h.value(), 2), round(dlg.bore.value(), 2),
                         round(dlg.flat.value(), 2), dlg.screw.isChecked(),
                         dlg.bearing.isChecked()]
        seen["spins"] = len(dlg.findChildren(QDoubleSpinBox))
        seen["info"] = dlg.info.text()
        seen["calm"] = dlg.info.styleSheet()
        seen["ok"] = dlg.bb.button(QDialogButtonBox.Ok).isEnabled()
        # a smooth wheel is asked for here: it has no collars to choose and no hub to build
        seen["idler_fields"] = [dlg.flanges.isEnabled(), dlg.hub.isEnabled(),
                                dlg.hub_d.isEnabled(), dlg.screw.isEnabled(),
                                dlg.bearing.isEnabled(), dlg.bore.isEnabled(),
                                dlg.flat.isEnabled()]
        dlg.bearing.setChecked(True)          # a bearing takes the middle: no plain bore, no flat
        seen["bearing_fields"] = [dlg.bore.isEnabled(), dlg.flat.isEnabled()]
        dlg.bearing.setChecked(False)
        dlg.kind.setCurrentIndex(dlg.kind.findData("pulley"))
        seen["pulley_fields"] = [dlg.flanges.isEnabled(), dlg.hub.isEnabled(),
                                 dlg.hub_d.isEnabled(), dlg.screw.isEnabled(),
                                 dlg.bearing.isEnabled(), dlg.bore.isEnabled(),
                                 dlg.flat.isEnabled()]
        assert dlg.flanges.currentData() == "one", "a toothed wheel lost the collars it had"
        dlg.kind.setCurrentIndex(dlg.kind.findData("idler"))
        dlg.kind.setCurrentIndex(dlg.kind.findData("pulley"))
        dlg.belt.setCurrentIndex(dlg.belt.findData("htd3m"))
        dlg.teeth.setValue(24)
        dlg.width.setValue(9.0)
        dlg.flange.setValue(2.0)
        dlg.hub_h.setValue(12.0)
        dlg.flat.setValue(1.0)
        seen["new"] = dlg.options()
        seen["still_ok"] = dlg.bb.button(QDialogButtonBox.Ok).isEnabled()
        dlg.bore.setValue(29.0)              # a bore that would eat the bottom of the grooves
        seen["refused"] = (dlg.bb.button(QDialogButtonBox.Ok).isEnabled(), dlg.info.text())
        dlg.bore.setValue(6.0)
        dlg.hub_d.setValue(6.0)              # a hub that cannot hold that bore has to grow
        seen["grown"] = (dlg.info.text(), dlg.info.styleSheet())
        dlg.hub_d.setValue(16.0)             # roomy again, and there is nothing left to say
        seen["quiet"] = (dlg.info.text(), dlg.info.styleSheet())
        return QDialog.Accepted

    set_settings(win, idler=True, profile="gt3", teeth=36, width=9.0, clearance=0.15,
                 flanges="one", flange=1.5, hub=True, hub_d=16.0, hub_h=12.0, bore=6.0,
                 flat=0.5, nut=False, bearing=False)
    monkeypatch.setattr(app_module.QDialog, "exec", fill)
    win.pulley_dialog()
    assert seen["title"] == "Timing Pulley"
    assert seen["start"] == ["idler", "gt3", 36, 9.0, 0.15, "one", 1.5, True, 16.0, 12.0, 6.0,
                             0.5, False, False]
    assert seen["spins"] == 7, "every field but the tooth count carries its millimetres"
    assert seen["info"].startswith("<i>") and "mm" in seen["info"], seen["info"]
    assert "smooth wheel" in seen["info"], seen["info"]
    assert f"Ø{pulley.idler_diameter(36, 'gt3'):.2f}" in seen["info"], seen["info"]
    assert seen["calm"] == "", "a wheel that fits is shown as a problem"
    assert seen["idler_fields"] == [False, False, False, False, True, True, True], seen["idler_fields"]
    assert seen["bearing_fields"] == [False, False], "a bearing left the bore and the flat behind"
    assert seen["pulley_fields"] == [True, True, True, True, False, True, True], seen["pulley_fields"]
    assert seen["ok"] and seen["still_ok"], "a wheel that can be made was blocked"
    assert not seen["refused"][0], "a bore that eats the grooves was not refused"
    assert "bore" in seen["refused"][1].lower(), seen["refused"][1]
    assert "Ø9.0" in seen["grown"][0] and seen["grown"][1], seen["grown"]
    assert seen["quiet"][1] == "" and seen["quiet"][0].startswith("<i>"), seen["quiet"]
    assert seen["new"]["idler"] is False and seen["new"]["profile"] == "htd3m"
    assert seen["new"]["teeth"] == 24 and seen["new"]["bearing"] is False
    assert seen["new"]["nut"] is False and seen["new"]["flanges"] == "one"
    assert pump_until(lambda: len(win.doc.groups) == 1), "no pulley in the model"
    assert win.doc.groups[0].name == "Timing pulley"
    assert pump_until(lambda: win.doc.groups[0].ents.to_manifold().volume() > 0.0)
    made = win.doc.groups[0].ents.to_manifold()
    assert made.volume() == pytest.approx(
        wheel(**{k: seen["new"][k] for k in BASE}).volume(), rel=1e-3)
    assert made.genus() == 1, "the bore is not a hole through the part that came in"
    back = app_module.Settings()
    assert (back.pulley_profile, back.pulley_teeth, back.pulley_idler) == ("htd3m", 24, False)
    assert back.pulley_width == pytest.approx(9.0)
    assert back.pulley_clearance == pytest.approx(0.15)
    assert back.pulley_flanges == "one"
    assert back.pulley_flange == pytest.approx(2.0)
    assert back.pulley_hub_d == pytest.approx(16.0)
    assert back.pulley_hub_h == pytest.approx(12.0)
    assert back.pulley_bore == pytest.approx(6.0)
    assert back.pulley_flat == pytest.approx(1.0)
    assert back.pulley_nut is False and back.pulley_bearing is False


def test_the_dialog_says_what_it_grew_and_what_it_left_out(win, monkeypatch):
    seen = {}

    def fill(dlg):
        dlg.kind.setCurrentIndex(dlg.kind.findData("pulley"))
        dlg.bore.setValue(8.0)               # a hub of Ø10 mm cannot hold a bore of Ø8 mm
        seen["grown"] = (dlg.info.text(), dlg.info.styleSheet())
        dlg.screw.setChecked(False)          # without the nut only the bore makes the hub wider
        seen["bore_only"] = (dlg.info.text(), dlg.info.styleSheet())
        dlg.screw.setChecked(True)
        dlg.bore.setValue(5.0)
        dlg.flat.setValue(6.0)               # more flat than a bore of Ø5 mm can carry
        seen["flat"] = (dlg.info.text(), dlg.info.styleSheet())
        dlg.flat.setValue(0.0)
        dlg.hub_h.setValue(5.0)              # too short a hub for the nut of an M3
        seen["short"] = (dlg.info.text(), dlg.info.styleSheet())
        dlg.hub.setChecked(False)            # without a hub there is no screw to complain about
        seen["nohub"] = (dlg.info.text(), dlg.info.styleSheet())
        dlg.kind.setCurrentIndex(dlg.kind.findData("idler"))
        dlg.bearing.setChecked(True)
        dlg.teeth.setValue(10)               # too small a wheel for a bearing of 16 mm
        seen["bearing"] = (dlg.info.text(), dlg.info.styleSheet())
        return QDialog.Rejected

    set_settings(win, hub_d=10.0)
    monkeypatch.setattr(app_module.QDialog, "exec", fill)
    before = len(win.doc.groups)
    assert win.pulley_dialog() is None
    assert len(win.doc.groups) == before, "a cancelled dialog still built a part"
    assert "Ø16.4" in seen["grown"][0] and "nut" in seen["grown"][0].lower(), seen["grown"]
    assert seen["grown"][1], "a hub that was grown for a nut is not shown as a warning"
    assert "Ø11.0" in seen["bore_only"][0], seen["bore_only"]
    assert "nut" not in seen["bore_only"][0].lower(), seen["bore_only"]
    assert seen["bore_only"][1], seen["bore_only"]
    assert "flat" in seen["flat"][0].lower() and seen["flat"][1], seen["flat"]
    assert "Ø13.4" in seen["short"][0], seen["short"]
    assert "nut" in seen["short"][0].lower() and seen["short"][1], seen["short"]
    assert seen["nohub"][1] == "", seen["nohub"]
    assert "bearing" in seen["bearing"][0].lower() and seen["bearing"][1], seen["bearing"]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))

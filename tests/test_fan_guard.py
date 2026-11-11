"""Fan guards: a printed plate with the screw holes of a fan and a grille over its blade.

`fan_guard.fan_guard` gives the guard as one closed volume that lies on `z = 0` and is centred on
the origin: a plate of `fan` millimetres with rounded corners, the four screw holes of that fan
at their standard spacing, and the blade circle of `fan - 2` millimetres open in the middle and
filled with a grille. `rings` are concentric rings held by four bars, `spokes` are eight radial
bars with a ring halfway, and `grid` is a square grid; `bar` is the width of every bar and `gap`
the openings between them, which is what keeps fingers off the blade.

These tests weigh the part against that: every size and every grille is one piece that lies flat
on the bed and is as wide as its fan, the screw holes are the size and the place the fan bolts
into, no opening of a ring or grid grille is wider than `gap`, the hub is solid plate, and the
numbers that do not print are clamped or refused.

Run with:  python -m pytest tests/test_fan_guard.py
"""
import math
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import manifold3d as mf  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from volura import fan_guard  # noqa: E402

BASE = dict(fan=40)
#: the bars and the rings are polygons, so a part is at most this far off
ABS = 0.01
#: how many points of the opening a grille is searched with, and how often they are looked for
SAMPLES, TRIES = 200, 60
#: the angle between the arms of a sunflower, which spreads sample points evenly over a disc
GOLDEN = 2.399963229728653


def guard(**kw) -> mf.Manifold:
    return fan_guard.fan_guard(**dict(BASE, **kw))


def numbers(**kw) -> dict:
    return fan_guard.sizes(**dict(BASE, **kw))


def bbox(m):
    """The bounding box of a manifold as (low, high)."""
    b = [float(v) for v in m.bounding_box()]
    return b[:3], b[3:]


def top(m) -> mf.CrossSection:
    """The guard seen from above: the section halfway its height."""
    lo, hi = bbox(m)
    return m.slice(0.5 * (lo[2] + hi[2]))


def probe(m, x, y, radius) -> bool:
    """Whether a cylinder of `radius` standing over (x, y) runs into the guard."""
    return (mf.Manifold.cylinder(3.0, radius).translate([x, y, -0.5]) ^ m).volume() > 1e-9


def open_points(sec, radius, count=SAMPLES) -> list:
    """`count` points inside the opening that are not plastic, spread over a sunflower spiral."""
    points, i = [], 0
    while len(points) < count and i < count * TRIES:
        r = radius * math.sqrt((i + 0.5) / (count * TRIES))
        p = (r * math.cos(i * GOLDEN), r * math.sin(i * GOLDEN))
        i += 1
        if (mf.CrossSection.circle(0.05, 8).translate(p) ^ sec).area() <= 1e-9:
            points.append(p)
    assert len(points) == count, "there is nothing left to see through the grille"
    return points


def touches(sec, p, radius) -> bool:
    """Whether a disc of `radius` around `p` reaches the plastic anywhere."""
    return (mf.CrossSection.circle(radius, fan_guard.SEGMENTS).translate(p) ^ sec).area() > 0


# ------------------------------------------------------------------ the part

@pytest.mark.parametrize("pattern", fan_guard.PATTERNS)
@pytest.mark.parametrize("fan", sorted(fan_guard.FANS))
def test_every_grille_is_one_plate_that_lies_on_the_bed(fan, pattern):
    s, m = numbers(fan=fan, pattern=pattern), guard(fan=fan, pattern=pattern)
    assert m.status() == mf.Error.NoError and not m.is_empty()
    assert len(m.decompose()) == 1, "the grille does not grow into the plate"
    lo, hi = bbox(m)
    assert lo[2] == pytest.approx(0.0), "the guard does not lie on the bed"
    assert hi[2] - lo[2] == pytest.approx(s["thickness"]), "the plate is not as thick as asked"
    assert hi[0] - lo[0] == pytest.approx(fan, abs=ABS), "the guard is not as wide as its fan"
    assert hi[1] - lo[1] == pytest.approx(fan, abs=ABS), "the guard is not square"


def test_the_corners_of_the_plate_are_rounded():
    sec = top(guard(fan=80))
    edge = mf.CrossSection.circle(0.2, 32)
    assert (edge.translate([39.9, 0.0]) ^ sec).area() > 0.0, "the plate is not square"
    assert (edge.translate([39.9, 39.9]) ^ sec).area() <= 1e-9, "the corners are not rounded"


@pytest.mark.parametrize("fan", sorted(fan_guard.FANS))
def test_the_screw_holes_are_the_size_and_the_place_of_the_fan(fan):
    s, m = numbers(fan=fan), guard(fan=fan)
    assert s["hole_spacing"] == fan_guard.FANS[fan]
    assert s["screw_hole"] == fan_guard.SCREW_HOLE[fan]
    for x in (-0.5 * s["hole_spacing"], 0.5 * s["hole_spacing"]):
        for y in (-0.5 * s["hole_spacing"], 0.5 * s["hole_spacing"]):
            assert not probe(m, x, y, 0.5 * (s["screw_hole"] - 0.1)), "the screw will not fit"
            assert probe(m, x, y, 0.5 * (s["screw_hole"] + 1.0)), "there is no hole here"


# ------------------------------------------------------------------ the grille

@pytest.mark.parametrize("pattern", ("rings", "grid"))
@pytest.mark.parametrize("fan", sorted(fan_guard.FANS))
def test_no_opening_of_a_guard_is_wider_than_gap(fan, pattern):
    s, m = numbers(fan=fan, pattern=pattern), guard(fan=fan, pattern=pattern)
    sec = top(m)
    for p in open_points(sec, 0.5 * s["opening"]):
        assert touches(sec, p, 0.5 * s["gap"] + 0.01), f"a finger fits through at {p}"


@pytest.mark.parametrize("pattern", ("rings", "spokes"))
def test_the_hub_is_a_solid_disc_in_the_middle(pattern):
    s = numbers(pattern=pattern)
    assert s["hub"] == pytest.approx(min(0.25 * s["fan"], 25.0))
    hub = mf.CrossSection.circle(0.5 * (s["hub"] - 0.1), fan_guard.SEGMENTS)
    assert (hub - top(guard(pattern=pattern))).area() == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize("fan", [40, 80, 140])
def test_the_spokes_go_all_the_way_round(fan):
    s = numbers(fan=fan, pattern="spokes")
    opening = mf.CrossSection.circle(0.5 * s["opening"], fan_guard.SEGMENTS)
    sec = top(guard(fan=fan, pattern="spokes")) ^ opening       # the grille, not the square plate
    for turn in (45.0, 90.0, 180.0):
        assert (sec - sec.rotate(turn)).area() < 0.01 * sec.area(), \
            f"the spokes are not the same {turn:g} degrees further round"


@pytest.mark.parametrize("fan", [40, 80, 140])
def test_a_grid_leaves_no_crumbs_at_the_edge_of_the_opening(fan):
    s = numbers(fan=fan, pattern="grid")
    opening = mf.CrossSection.circle(0.5 * s["opening"], fan_guard.SEGMENTS)
    holes = (opening - top(guard(fan=fan, pattern="grid"))).decompose()
    assert holes, "the grid has no openings"
    smallest = min(h.area() for h in holes)
    assert smallest >= fan_guard.SLIVER * s["gap"] ** 2 - 0.01, \
        f"an opening of {smallest:.3f} mm² at the edge prints as a crumb"


def test_the_default_rings_leave_most_of_the_opening_open():
    s = numbers(fan=80)
    assert s["opening"] == pytest.approx(78.0)
    opening = mf.CrossSection.circle(0.5 * s["opening"], fan_guard.SEGMENTS).area()
    assert 0.3 * opening < s["open_area"] < 0.9 * opening, "the grille is a plate or nothing"


def test_the_grille_stops_in_the_plate_and_never_covers_a_screw_hole():
    kw = dict(fan=140, bar=5.0, gap=2.0)
    for pattern in fan_guard.PATTERNS:
        s, m = numbers(pattern=pattern, **kw), guard(pattern=pattern, **kw)
        assert s["open_area"] > 0 and len(m.decompose()) == 1
        d = 0.5 * s["hole_spacing"]
        assert not probe(m, d, d, 0.5 * (s["screw_hole"] - 0.1)), "the grille is in the way"


# ------------------------------------------------------------------ the numbers

def test_sizes_reports_the_fan_and_the_grille():
    s = numbers(fan=80, thickness=3.0, pattern="grid", bar=2.0, gap=3.0)
    assert s["fan"] == 80 and s["pattern"] == "grid"
    assert s["thickness"] == 3.0 and s["bar"] == 2.0 and s["gap"] == 3.0
    assert s["hub"] == pytest.approx(20.0) and s["open_area"] > 0
    assert set(s) == {"fan", "thickness", "pattern", "bar", "gap", "hole_spacing", "screw_hole",
                      "opening", "hub", "open_area"}


def test_the_numbers_are_clamped_to_what_prints():
    thin = numbers(thickness=0.2, bar=0.1, gap=1.0)
    assert thin["thickness"] == fan_guard.THICK_MIN
    assert thin["bar"] == fan_guard.BAR_MIN and thin["gap"] == fan_guard.GAP_MIN
    thick = numbers(thickness=20.0, bar=99.0, gap=99.0)
    assert thick["thickness"] == fan_guard.THICK_MAX
    assert thick["bar"] == fan_guard.BAR_MAX and thick["gap"] == fan_guard.GAP_MAX
    lo, hi = bbox(guard(thickness=20.0))
    assert hi[2] - lo[2] == pytest.approx(fan_guard.THICK_MAX)


def test_a_fan_this_does_not_know_is_refused():
    for make in (guard, numbers):
        with pytest.raises(ValueError) as e:
            make(fan=33)
        assert "fan" in str(e.value)


def test_a_grille_this_does_not_make_is_refused():
    for make in (guard, numbers):
        with pytest.raises(ValueError) as e:
            make(pattern="stars")
        assert "pattern" in str(e.value)


@pytest.mark.parametrize("kw", (dict(thickness=float("nan")), dict(bar=float("nan")),
                                dict(gap=float("nan")), dict(thickness="wide")))
def test_a_number_that_is_not_a_number_is_refused(kw):
    with pytest.raises(ValueError):
        guard(**kw)


# ------------------------------------------------------------------ the interface

def test_the_dialog_builds_a_guard_and_remembers_it(monkeypatch, tmp_path):
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
        dlg.fan.setCurrentIndex(dlg.fan.findData(120))
        dlg.pattern.setCurrentIndex(dlg.pattern.findData("grid"))
        dlg.thickness.setValue(3.0)
        assert "105 mm apart" in dlg.info.text()
        return QDialog.Accepted

    monkeypatch.setattr(app_module.QDialog, "exec", fill)
    try:
        w.fan_guard_dialog()
        deadline = time.monotonic() + 15.0
        while not w.doc.groups and time.monotonic() < deadline:
            app.processEvents()
            time.sleep(0.005)
        assert len(w.doc.groups) == 1, "no guard in the model"
        lo, hi = np.asarray(w.doc.groups[0].bbox(), dtype=float)
        assert hi - lo == pytest.approx([120.0, 120.0, 3.0], abs=1e-4)
        s = app_module.Settings()
        assert (s.fan_size, s.fan_pattern, s.fan_thickness) == (120, "grid", pytest.approx(3.0))
        labels = [a.text().replace("&", "").replace("…", "") for a in w.flat_actions(w.menuBar())]
        assert labels.index("Fan Guard") == labels.index("Hose Adapter") + 1
    finally:
        w.settings.fan_size, w.settings.fan_pattern, w.settings.fan_thickness = 80, "rings", 2.0
        w.settings.save()
        w.doc.dirty = False
        w.close()

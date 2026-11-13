"""Drawer organizers: a tray that drops into a drawer and divides it into compartments.

`drawer_organizer.drawer_organizer` gives the tray as one closed volume that stands on its floor on
`z = 0` and is centred in x and y: the inside of the drawer (`length` by `width`) less `clearance`
on every side, `height` high, its walls and dividers `wall` and its floor `floor` thick, divided
into `cols` by `rows` equal compartments whose inside corners are rounded with `radius` and whose
inner walls carry a half-round finger notch at their middle when `finger` is on.

These tests weigh the part against that: it is one piece that stands on the bed and is as big as
the drawer less its clearance, the compartments are a grid you can probe between, the floor is as
thick as asked, the corners are rounded so a probe in one touches a wall, the finger notches lower
the inner walls only when they are on and never with a single compartment, the radius is limited for
small compartments, every range is reached by clamping and a number that is no number is refused.

Run with:  python -m pytest tests/test_drawer_organizer.py
"""
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import manifold3d as mf  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from volura import drawer_organizer  # noqa: E402

BASE = dict(length=200.0, width=150.0, height=50.0, clearance=1.0, cols=3, rows=2,
            wall=1.6, floor=1.2, radius=4.0, finger=True)
#: the corners and notches are polygons, so a part is at most this far off
ABS = 0.05
#: mm across a probe that feels for a wall or an empty compartment
PROBE = 0.4


def part(**kw) -> mf.Manifold:
    return drawer_organizer.drawer_organizer(**dict(BASE, **kw))


def numbers(**kw) -> dict:
    return drawer_organizer.sizes(**dict(BASE, **kw))


def bbox(m):
    """The bounding box of a manifold as (low, high)."""
    b = np.asarray(m.bounding_box(), dtype=float)
    return b[:3], b[3:]


def touches(m, probe) -> bool:
    """Whether a probe runs into the part anywhere.

    Two solids that only touch leave a sliver of a few ten-millionths of a mm³ behind, so that is
    not a touch; a wall that is really there is off by whole tenths of a millimetre.
    """
    return (m ^ probe).volume() > 1e-6


def post(x, y, z0, z1, w=PROBE) -> mf.Manifold:
    """A thin upright square of `w` millimetres at (`x`, `y`), between the heights `z0` and `z1`."""
    return mf.Manifold.cube([w, w, z1 - z0]).translate((x - 0.5 * w, y - 0.5 * w, z0))


def layout(**kw) -> dict:
    """The clamped numbers with the places of the compartments, the dividers and the walls."""
    s = numbers(**kw)
    inner_l = s["outer_length"] - 2.0 * s["wall"]
    inner_w = s["outer_width"] - 2.0 * s["wall"]
    cl, cw, t = s["compartment_length"], s["compartment_width"], s["wall"]
    s["cx"] = [-0.5 * inner_l + 0.5 * cl + i * (cl + t) for i in range(s["cols"])]
    s["cy"] = [-0.5 * inner_w + 0.5 * cw + j * (cw + t) for j in range(s["rows"])]
    s["dx"] = [-0.5 * inner_l + i * cl + (i - 0.5) * t for i in range(1, s["cols"])]
    s["dy"] = [-0.5 * inner_w + j * cw + (j - 0.5) * t for j in range(1, s["rows"])]
    s["wall_x"] = 0.5 * s["outer_length"] - 0.5 * t
    s["wall_y"] = 0.5 * s["outer_width"] - 0.5 * t
    return s


# ------------------------------------------------------------------ the part

@pytest.mark.parametrize("length, width, height", ((40.0, 40.0, 15.0), (200.0, 150.0, 50.0),
                                                   (300.0, 300.0, 120.0)))
@pytest.mark.parametrize("cols, rows", ((1, 1), (3, 2), (4, 3)))
def test_the_tray_is_one_volume_that_stands_on_the_bed_and_is_centred(length, width, height,
                                                                      cols, rows):
    kw = dict(length=length, width=width, height=height, cols=cols, rows=rows)
    s, m = numbers(**kw), part(**kw)
    assert m.status() == mf.Error.NoError and not m.is_empty()
    assert len(m.decompose()) == 1, "the tray fell apart"
    lo, hi = bbox(m)
    assert lo[2] == pytest.approx(0.0, abs=1e-6), "the tray does not stand on the bed"
    assert hi[2] - lo[2] == pytest.approx(s["height"], abs=ABS), "the tray is not as high"
    assert hi[0] - lo[0] == pytest.approx(s["outer_length"], abs=ABS), "the tray is not as long"
    assert hi[1] - lo[1] == pytest.approx(s["outer_width"], abs=ABS), "the tray is not as wide"
    assert lo[0] == pytest.approx(-0.5 * s["outer_length"], abs=ABS), "the tray is not centred in x"
    assert lo[1] == pytest.approx(-0.5 * s["outer_width"], abs=ABS), "the tray is not centred in y"


@pytest.mark.parametrize("clearance", (0.0, 1.0, 3.0))
def test_the_outside_is_the_drawer_less_the_clearance_on_every_side(clearance):
    """The size you enter is the inside of the drawer; the tray is `clearance` smaller on every
    side, so it drops in without forcing."""
    s, m = numbers(clearance=clearance), part(clearance=clearance)
    assert s["outer_length"] == pytest.approx(s["length"] - 2.0 * clearance)
    assert s["outer_width"] == pytest.approx(s["width"] - 2.0 * clearance)
    lo, hi = bbox(m)
    assert hi[0] - lo[0] == pytest.approx(s["length"] - 2.0 * clearance, abs=ABS)
    assert hi[1] - lo[1] == pytest.approx(s["width"] - 2.0 * clearance, abs=ABS)


# ------------------------------------------------------------------ the compartments

@pytest.mark.parametrize("cols, rows", ((1, 1), (2, 2), (3, 2), (4, 3)))
def test_the_compartments_are_a_grid_of_equal_pockets(cols, rows):
    """A probe in the middle of every compartment, above the floor, touches nothing; a probe on the
    line of every wall, below the notches, touches the wall."""
    s, m = layout(cols=cols, rows=rows), part(cols=cols, rows=rows)
    for x in s["cx"]:
        for y in s["cy"]:
            assert not touches(m, post(x, y, s["floor"] + 1.0, s["height"] - 1.0)), \
                f"the compartment at ({x:.1f}, {y:.1f}) is not empty"
    for x in s["dx"]:
        assert touches(m, post(x, 0.0, 2.0, s["height"] - 20.0)), "a wall between columns is missing"
    for y in s["dy"]:
        assert touches(m, post(0.0, y, 2.0, s["height"] - 20.0)), "a wall between rows is missing"
    for x in (s["wall_x"], -s["wall_x"]):
        assert touches(m, post(x, 0.0, 2.0, s["height"] - 20.0)), "a side wall is missing"
    for y in (s["wall_y"], -s["wall_y"]):
        assert touches(m, post(0.0, y, 2.0, s["height"] - 20.0)), "a front wall is missing"


@pytest.mark.parametrize("floor", (0.8, 1.2, 3.0))
def test_the_floor_is_as_thick_as_asked(floor):
    s, m = layout(floor=floor), part(floor=floor)
    x, y = s["cx"][0], s["cy"][0]
    assert touches(m, post(x, y, 0.1, floor - 0.1)), "the floor is thinner than asked"
    assert not touches(m, post(x, y, floor + 0.1, s["height"] - 1.0)), \
        "the compartment does not open above the floor"


def test_the_inside_corners_of_a_compartment_are_rounded():
    """A probe in the corner of a compartment, inside the radius, touches the wall because the
    corner is rounded away; with a radius of 0 the pocket reaches its own corner and it does not."""
    s, m = layout(), part()
    eps = 0.3
    x = s["cx"][0] + 0.5 * s["compartment_length"] - eps
    y = s["cy"][0] + 0.5 * s["compartment_width"] - eps
    assert touches(m, post(x, y, s["floor"] + 1.0, s["height"] - 1.0)), \
        "the corner of the compartment is not rounded"
    assert not touches(part(radius=0.0), post(x, y, s["floor"] + 1.0, s["height"] - 1.0)), \
        "a square pocket should reach into its own corner"


# ------------------------------------------------------------------ the finger notches

def test_the_finger_notch_lowers_the_inner_walls_at_their_middle():
    """With the notch on, the top of an inner wall is cut away at its middle; with it off the wall
    stands whole up to the top edge."""
    s, m = layout(), part()
    x = s["dx"][0]
    assert not touches(m, post(x, 0.0, s["height"] - 5.0, s["height"] - 0.5)), \
        "the finger notch is not cut in the wall"
    assert touches(part(finger=False), post(x, 0.0, s["height"] - 5.0, s["height"] - 0.5)), \
        "the wall is cut even without the notch"


@pytest.mark.parametrize("cols, rows", ((1, 1), (2, 1), (1, 2), (3, 2)))
def test_the_finger_notch_is_only_on_inner_walls(cols, rows):
    """A single compartment has no inner wall, so the notch makes no difference there; with any
    divider it takes plastic away."""
    on, off = part(cols=cols, rows=rows, finger=True), part(cols=cols, rows=rows, finger=False)
    if cols == 1 and rows == 1:
        assert on.volume() == pytest.approx(off.volume()), "a lone compartment got a notch anyway"
    else:
        assert on.volume() < off.volume(), "the notch did not cut anything out of the walls"


# ------------------------------------------------------------------ the numbers

def test_the_radius_is_limited_for_small_compartments():
    """The radius never exceeds a third of the smallest compartment, so a corner stays a scoop and
    not a whole round that eats the wall."""
    s = numbers(length=40.0, width=40.0, clearance=0.0, cols=8, rows=8, wall=4.0, floor=0.8,
                radius=15.0)
    comp = min(s["compartment_length"], s["compartment_width"])
    assert s["radius"] == pytest.approx(comp / 3.0), "the radius is not capped to a third"
    assert s["radius"] < drawer_organizer.RADIUS_MAX, "the cap did not bite"


def test_sizes_reports_the_tray_and_its_compartments():
    s = drawer_organizer.sizes(length=210.0, width=160.0, height=45.0, clearance=1.0, cols=2,
                               rows=2, wall=2.0, floor=1.5, radius=5.0, finger=True)
    assert s["length"] == 210.0 and s["width"] == 160.0 and s["height"] == 45.0
    assert s["clearance"] == 1.0 and s["cols"] == 2 and s["rows"] == 2
    assert s["wall"] == 2.0 and s["floor"] == 1.5 and s["radius"] == 5.0 and s["finger"] is True
    assert s["outer_length"] == pytest.approx(208.0) and s["outer_width"] == pytest.approx(158.0)
    assert s["compartment_length"] == pytest.approx(101.0)
    assert s["compartment_width"] == pytest.approx(76.0)
    assert set(s) == {"length", "width", "height", "clearance", "cols", "rows", "wall", "floor",
                      "radius", "finger", "outer_length", "outer_width", "compartment_length",
                      "compartment_width"}


def test_the_default_tray_divides_a_two_hundred_by_a_hundred_and_fifty():
    s = drawer_organizer.sizes()
    assert s["outer_length"] == pytest.approx(198.0) and s["outer_width"] == pytest.approx(148.0)
    assert s["cols"] == 3 and s["rows"] == 2 and s["height"] == 50.0
    assert s["compartment_length"] == pytest.approx(63.8667, abs=1e-3)
    assert s["compartment_width"] == pytest.approx(71.6)


@pytest.mark.parametrize("cols, want", ((0, 1), (1, 1), (3, 3), (8, 8), (99, 8)))
def test_the_number_of_compartments_follows_what_fits(cols, want):
    assert drawer_organizer.sizes(cols=cols)["cols"] == want
    assert drawer_organizer.sizes(rows=cols)["rows"] == want


def test_the_numbers_are_clamped_to_what_prints():
    small = numbers(length=0.0, width=0.0, height=0.0, clearance=-1.0, cols=0, rows=0, wall=0.0,
                    floor=0.0, radius=-1.0)
    assert small["length"] == drawer_organizer.LENGTH_MIN
    assert small["width"] == drawer_organizer.WIDTH_MIN
    assert small["height"] == drawer_organizer.HEIGHT_MIN
    assert small["clearance"] == drawer_organizer.CLEAR_MIN
    assert small["cols"] == drawer_organizer.COLS_MIN and small["rows"] == drawer_organizer.ROWS_MIN
    assert small["wall"] == drawer_organizer.WALL_MIN and small["floor"] == drawer_organizer.FLOOR_MIN
    assert small["radius"] == drawer_organizer.RADIUS_MIN
    big = numbers(length=999.0, width=999.0, height=999.0, clearance=99.0, cols=99, rows=99,
                  wall=99.0, floor=99.0, radius=99.0)
    assert big["length"] == drawer_organizer.LENGTH_MAX and big["width"] == drawer_organizer.WIDTH_MAX
    assert big["height"] == drawer_organizer.HEIGHT_MAX and big["clearance"] == drawer_organizer.CLEAR_MAX
    assert big["cols"] == drawer_organizer.COLS_MAX and big["rows"] == drawer_organizer.ROWS_MAX
    assert big["wall"] == drawer_organizer.WALL_MAX and big["floor"] == drawer_organizer.FLOOR_MAX
    assert drawer_organizer.sizes(radius=99.0)["radius"] == drawer_organizer.RADIUS_MAX
    lo, hi = bbox(part(length=0.0, width=0.0, height=0.0, cols=0, rows=0))
    assert hi[2] - lo[2] == pytest.approx(drawer_organizer.HEIGHT_MIN, abs=ABS)
    assert len(part(length=999.0, width=999.0, cols=99, rows=99, wall=99.0).decompose()) == 1


@pytest.mark.parametrize("bad", (float("nan"), float("inf"), None, "wide"))
def test_a_number_that_is_no_number_is_refused(bad):
    for key in ("length", "width", "height", "clearance", "cols", "rows", "wall", "floor",
                "radius"):
        with pytest.raises(ValueError):
            drawer_organizer.drawer_organizer(**dict(BASE, **{key: bad}))
        with pytest.raises(ValueError):
            drawer_organizer.sizes(**dict(BASE, **{key: bad}))


# ------------------------------------------------------------------ the interface

def test_the_dialog_builds_an_organizer_and_remembers_it(monkeypatch, tmp_path):
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
        dlg.length.setValue(210.0)
        dlg.width.setValue(160.0)
        dlg.height.setValue(45.0)
        dlg.clearance.setValue(1.0)
        dlg.cols.setValue(2)
        dlg.rows.setValue(2)
        dlg.wall.setValue(2.0)
        assert dlg.finger.isChecked(), "finger notches are on by default"
        assert "101.0" in dlg.info.text(), "the dialog does not show the compartment it makes"
        assert "76.0" in dlg.info.text(), "the dialog does not show the compartment it makes"
        dlg.finger.setChecked(False)
        return QDialog.Accepted

    monkeypatch.setattr(app_module.QDialog, "exec", fill)
    try:
        w.drawer_organizer_dialog()
        deadline = time.monotonic() + 15.0
        while not w.doc.groups and time.monotonic() < deadline:
            app.processEvents()
            time.sleep(0.005)
        assert len(w.doc.groups) == 1, "no organizer in the model"
        lo, hi = np.asarray(w.doc.groups[0].bbox(), dtype=float)
        s = drawer_organizer.sizes(length=210.0, width=160.0, height=45.0, clearance=1.0, cols=2,
                                   rows=2, wall=2.0, finger=False)
        assert lo[2] == pytest.approx(0.0, abs=1e-6), "the tray does not stand on the bed"
        assert hi[2] - lo[2] == pytest.approx(s["height"], abs=1e-4)
        assert hi[0] - lo[0] == pytest.approx(s["outer_length"], abs=1e-3), "not as computed"
        settings = app_module.Settings()
        assert (settings.drawer_organizer_length, settings.drawer_organizer_width,
                settings.drawer_organizer_height, settings.drawer_organizer_clearance,
                settings.drawer_organizer_cols, settings.drawer_organizer_rows,
                settings.drawer_organizer_wall, settings.drawer_organizer_finger) == (
            210.0, 160.0, 45.0, 1.0, 2, 2, 2.0, False)
        labels = [a.text().replace("&", "").replace("…", "") for a in w.flat_actions(w.menuBar())]
        assert labels.index("Drawer Organizer") == labels.index("Spool Adapter") + 1
    finally:
        for key, value in (("drawer_organizer_length", 200.0), ("drawer_organizer_width", 150.0),
                           ("drawer_organizer_height", 50.0), ("drawer_organizer_clearance", 1.0),
                           ("drawer_organizer_cols", 3), ("drawer_organizer_rows", 2),
                           ("drawer_organizer_wall", 1.6), ("drawer_organizer_floor", 1.2),
                           ("drawer_organizer_radius", 4.0), ("drawer_organizer_finger", True)):
            setattr(w.settings, key, value)
        w.settings.save()
        w.doc.dirty = False
        w.close()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))

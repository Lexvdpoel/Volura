"""Stringing Test: thin towers the nozzle travels between, which shows whether a printer strings.

`stringing_test.stringing_test` gives the test as one closed volume standing on `z = 0`: a strip
with `towers` towers of `diameter` millimetres on it, `distance` millimetres apart between their
centres and `height` millimetres high. Nothing connects the towers, so the nozzle crosses open
air on every layer and the hairs of plastic it leaves between them say whether retraction and
temperature work together. With `cone` every tower ends in a point, which prints without support.
The row is clamped to the bed.

These tests (Qt offscreen) weigh that: one volume on the bed and centred, plastic in every tower
and nothing between two of them above the strip, the row as high as asked, a pointed tower
narrower near its top than a flat one is, the strip of the thickness it always is, the row clamped
to the bed, every range reached by clamping and a number that is no number refused, and the place
in the interface (3D Printing, right after the bridge test, one group, remembered values).

Run with:  python -m pytest tests/test_stringing_test.py
"""
import os
import sys
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import manifold3d as mf  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from PySide6.QtWidgets import (QApplication, QCheckBox, QDialog, QDoubleSpinBox,  # noqa: E402
                               QSpinBox)

from volura import app as app_module, stringing_test  # noqa: E402
from volura.settings import qsettings  # noqa: E402

_app = QApplication.instance() or QApplication(sys.argv[:1])   # the dialog needs Qt

NUMBERS = dict(towers=2, distance=40.0, height=30.0, diameter=6.0)
BASE = dict(NUMBERS, cone=True)
#: the towers are polygons, so a part is at most this far off
ABS = 0.05
STRIP = stringing_test.STRIP
EDGE = stringing_test.EDGE


class FakeBox:
    """Stands in for QMessageBox: remembers what the window asked."""

    def __init__(self):
        self.asked = []

    def warning(self, *args, **kwargs):
        self.asked.append(args[1:])

    def information(self, *args, **kwargs):
        self.asked.append(args[1:])


@pytest.fixture()
def win(monkeypatch, tmp_path):
    monkeypatch.setattr(app_module.errors, "log_dir", lambda: str(tmp_path))
    monkeypatch.setattr(app_module, "autosave_path", lambda: str(tmp_path / "autosave.volura"))
    monkeypatch.setattr(app_module, "QMessageBox", FakeBox())
    w = app_module.MainWindow()
    w.doc.dirty = False
    yield w
    w.doc.dirty = False
    w.close()


def made(**kw) -> mf.Manifold:
    return stringing_test.stringing_test(**dict(BASE, **kw))


def numbers(**kw) -> dict:
    return stringing_test.sizes(**dict(NUMBERS, **kw))


def bbox(m):
    """The bounding box of a manifold as (low, high)."""
    b = np.asarray(m.bounding_box(), dtype=float)
    return b[:3], b[3:]


def centres(s):
    """The middle of every tower, on the row as it stands centred on the bed."""
    return [(k - 0.5 * (s["towers"] - 1)) * s["distance"] for k in range(s["towers"])]


def probe(m, x, y, z, size=1.0) -> float:
    """The volume of the test inside a small cube centred on (x, y, z): does it touch plastic?"""
    box = mf.Manifold.cube([size, size, size]).translate((x - 0.5 * size, y - 0.5 * size,
                                                          z - 0.5 * size))
    return (m ^ box).volume()


def width_at(m, x, z, span) -> float:
    """How wide the test is where the plane at `z` cuts the tower on `x`, measured across x."""
    slab = mf.Manifold.cube([span, span, 0.2]).translate((x - 0.5 * span, -0.5 * span, z - 0.1))
    cut = m ^ slab
    if cut.is_empty():
        return 0.0
    b = np.asarray(cut.bounding_box(), dtype=float)
    return float(b[3] - b[0])


def pump_until(predicate, timeout_ms=15000):
    """Spin the event loop until `predicate()` holds or the time is up; returns its value."""
    deadline = time.monotonic() + timeout_ms / 1000.0
    while not predicate() and time.monotonic() < deadline:
        _app.processEvents()
        time.sleep(0.005)
    return predicate()


# ------------------------------------------------------------------ the row

@pytest.mark.parametrize("cone", (False, True))
def test_the_test_is_one_volume_that_stands_on_the_bed_and_is_centred(cone):
    s, m = numbers(), made(cone=cone)
    assert m.status() == mf.Error.NoError and not m.is_empty()
    assert len(m.decompose()) == 1, "the row fell apart"
    length, depth, top = s["size"]
    lo, hi = bbox(m)
    assert lo[2] == pytest.approx(0.0, abs=1e-6), "the test does not stand on the bed"
    assert np.allclose(lo, [-0.5 * length, -0.5 * depth, 0.0], atol=ABS), "not centred in x and y"
    assert np.allclose(hi, [0.5 * length, 0.5 * depth, top], atol=ABS), "not the size asked"


@pytest.mark.parametrize("towers, distance", ((2, 40.0), (3, 30.0), (4, 20.0)))
def test_there_is_a_tower_on_every_place_of_the_row_and_nothing_between_them(towers, distance):
    """Every tower is plastic from the strip to its top and the air between two neighbouring
    towers is free, so the nozzle really has to travel from one to the next."""
    s, m = numbers(towers=towers, distance=distance), made(towers=towers, distance=distance)
    xs = centres(s)
    assert len(xs) == s["towers"] == towers
    assert max(xs) - min(xs) == pytest.approx((towers - 1) * distance, abs=ABS)
    for x in xs:
        assert probe(m, x, 0.0, 0.5 * s["height"]) > 0.0, "a tower is not where it should be"
    for a, b in zip(xs, xs[1:]):
        assert probe(m, 0.5 * (a + b), 0.0, 0.5 * s["height"]) == pytest.approx(0.0, abs=1e-6), \
            "something stands between two towers"


@pytest.mark.parametrize("height", (10.0, 30.0, 80.0))
def test_the_towers_stand_as_high_as_asked(height):
    s, m = numbers(height=height), made(height=height)
    lo, hi = bbox(m)
    assert hi[2] - lo[2] == pytest.approx(height, abs=ABS), "the row is not as high as asked"
    for x in centres(s):
        assert probe(m, x, 0.0, 0.5 * height) > 0.0, "a tower stops short of its height"


def test_a_pointed_tower_is_narrower_near_its_top_than_halfway_up():
    s, m = numbers(), made()
    x = centres(s)[0]
    wide = width_at(m, x, 0.5 * s["height"], s["diameter"])
    narrow = width_at(m, x, 0.9 * s["height"], s["diameter"])
    assert wide == pytest.approx(s["diameter"], abs=ABS), "a tower is not as thick as asked"
    assert narrow < 0.5 * wide, "the top of a pointed tower is not pointed"
    assert narrow > stringing_test.TIP - ABS, "a pointed tower ends in nothing"


def test_a_tower_without_a_point_keeps_its_width_to_its_top():
    s, m = numbers(), made(cone=False)
    x = centres(s)[0]
    assert width_at(m, x, 0.9 * s["height"], s["diameter"]) \
        == pytest.approx(width_at(m, x, 0.5 * s["height"], s["diameter"]), abs=ABS), \
        "a tower without a point is not a cylinder"


@pytest.mark.parametrize("cone", (False, True))
def test_the_strip_under_the_row_is_of_the_thickness_it_always_is(cone):
    """The strip is the only thing between two towers, so what touches in the middle of the row
    says how thick it is: plastic up to `STRIP` and free air above it."""
    s, m = numbers(), made(cone=cone)
    lo, _hi = bbox(m)
    mid = 0.5 * (centres(s)[0] + centres(s)[1])
    assert probe(m, mid, 0.0, 0.5 * STRIP) > 0.0, "there is no strip under the row"
    assert probe(m, mid, 0.0, STRIP + 0.3, 0.4) == pytest.approx(0.0, abs=1e-6), \
        "the strip is thicker than it is"
    assert probe(m, mid, 0.5 * s["diameter"] + 0.5 * EDGE, 0.5 * STRIP) > 0.0, \
        "the strip does not reach beside the row"


# ------------------------------------------------------------------ the numbers of the row

def test_sizes_reports_the_row_and_its_size():
    s = stringing_test.sizes(towers=3, distance=25.0, height=50.0, diameter=8.0)
    assert s["towers"] == 3
    assert s["distance"] == 25.0
    assert s["height"] == 50.0
    assert s["diameter"] == 8.0
    assert s["size"] == (2 * 25.0 + 2 * 8.0, 8.0 + 2 * EDGE, 50.0)
    assert stringing_test.sizes()["size"] == (40.0 + 12.0, 10.0, 30.0)


@pytest.mark.parametrize("distance, diameter", ((100.0, 12.0), (90.0, 10.0)))
def test_the_row_drops_its_last_towers_to_fit_the_bed(distance, diameter):
    s = stringing_test.sizes(towers=4, distance=distance, diameter=diameter)
    assert s["towers"] < 4, "the row kept towers that do not fit the bed"
    assert s["size"][0] <= stringing_test.BED
    lo, hi = bbox(stringing_test.stringing_test(towers=4, distance=distance,
                                                diameter=diameter))
    assert hi[0] - lo[0] == pytest.approx(s["size"][0], abs=ABS)
    assert hi[0] - lo[0] <= stringing_test.BED + ABS, "the row is too long for the bed"


def test_the_numbers_are_clamped_to_what_prints():
    small = numbers(towers=1, distance=0.0, height=0.0, diameter=0.0)
    assert small["towers"] == stringing_test.TOWERS_MIN
    assert small["distance"] == stringing_test.DISTANCE_MIN
    assert small["height"] == stringing_test.HEIGHT_MIN
    assert small["diameter"] == stringing_test.DIAMETER_MIN
    big = numbers(towers=99, distance=999.0, height=999.0, diameter=999.0)
    assert big["towers"] <= stringing_test.TOWERS_MAX
    assert big["distance"] == stringing_test.DISTANCE_MAX
    assert big["height"] == stringing_test.HEIGHT_MAX
    assert big["diameter"] == stringing_test.DIAMETER_MAX
    assert numbers(towers=4, distance=20.0)["towers"] == stringing_test.TOWERS_MAX
    lo, hi = bbox(made(towers=99, height=999.0, diameter=999.0))
    assert hi[2] - lo[2] == pytest.approx(stringing_test.HEIGHT_MAX, abs=ABS)


@pytest.mark.parametrize("bad", (float("nan"), float("inf"), None, "wide"))
def test_a_number_that_is_no_number_is_refused(bad):
    for key in ("towers", "distance", "height", "diameter"):
        with pytest.raises(ValueError):
            stringing_test.stringing_test(**dict(BASE, **{key: bad}))
        with pytest.raises(ValueError):
            stringing_test.sizes(**dict(NUMBERS, **{key: bad}))


# -------------------------------------------------------------------- the window

def forget_the_values(settings):
    """The stringing test settings back at what the program starts with."""
    for key, value in (("stringing_test_towers", 2), ("stringing_test_distance", 40.0),
                       ("stringing_test_height", 30.0), ("stringing_test_diameter", 6.0),
                       ("stringing_test_cone", True)):
        setattr(settings, key, value)
    settings.save()


def test_the_dialog_builds_a_test_and_remembers_it(win, monkeypatch):
    seen = {}

    def accept(dlg):
        spins = dlg.findChildren(QDoubleSpinBox)
        towers = dlg.findChild(QSpinBox)
        pointed = dlg.findChild(QCheckBox)
        seen["start"] = [towers.value(), spins[0].value(), spins[1].value(),
                         spins[2].value(), pointed.isChecked()]
        towers.setValue(3)
        spins[0].setValue(55.0)
        pointed.setChecked(False)
        return QDialog.Accepted

    monkeypatch.setattr(app_module.QDialog, "exec", accept)
    win.settings.stringing_test_towers = 2
    win.settings.stringing_test_distance = 40.0
    win.settings.stringing_test_height = 30.0
    win.settings.stringing_test_diameter = 6.0
    win.settings.stringing_test_cone = True
    try:
        win.stringing_test_dialog()
        assert seen["start"] == [2, pytest.approx(40.0), pytest.approx(30.0),
                                 pytest.approx(6.0), True]
        assert pump_until(lambda: len(win.doc.groups) == 1), "no stringing test"
        g = win.doc.groups[0]
        assert g.name == "Stringing test"
        assert g.ents.to_manifold().status() == mf.Error.NoError
        assert [i[2] for i in win.doc.selection] == [g.id]
        s = stringing_test.sizes(towers=3, distance=55.0)
        lo, hi = bbox(g.ents.to_manifold())
        assert lo[2] == pytest.approx(0.0, abs=1e-6), "the test does not stand on the bed"
        assert hi[0] - lo[0] == pytest.approx(s["size"][0], abs=1e-3), "not as long as asked"
        assert hi[2] - lo[2] == pytest.approx(s["size"][2], abs=1e-3), "not as high as asked"
        assert pump_until(lambda: "towers" in win.hint.text()), win.hint.text()
        st = app_module.Settings()
        assert st.stringing_test_towers == 3
        assert st.stringing_test_distance == pytest.approx(55.0)
        assert st.stringing_test_height == pytest.approx(30.0)
        assert st.stringing_test_diameter == pytest.approx(6.0)
        assert st.stringing_test_cone is False
    finally:
        forget_the_values(win.settings)


def test_one_undo_takes_the_whole_test_away(win):
    steps = len(win.doc.undo_stack)
    win._apply_stringing_test(2, 40.0, 30.0, 6.0, True)
    assert pump_until(lambda: len(win.doc.groups) == 1)
    assert len(win.doc.undo_stack) == steps + 1, "the test is not one undo step"
    win.undo()
    assert win.doc.groups == []


def test_the_settings_clamp_what_they_read():
    q = qsettings()
    keys = ("stringing_test_towers", "stringing_test_distance", "stringing_test_height",
            "stringing_test_diameter")
    saved = {k: q.value(k) for k in keys}
    try:
        for k, v in zip(keys, (99.0, 999.0, 999.0, 999.0)):
            q.setValue(k, v)
        q.sync()
        s = app_module.Settings()
        assert s.stringing_test_towers == 4
        assert s.stringing_test_distance == 100.0
        assert s.stringing_test_height == 80.0
        assert s.stringing_test_diameter == 12.0
    finally:
        for k, v in saved.items():
            q.remove(k) if v is None else q.setValue(k, v)
        q.sync()


def test_the_action_is_in_the_3d_printing_menu_after_the_bridge_test(win):
    labels = [a.text().replace("&", "").replace("…", "").replace("...", "").strip()
              for a in win.flat_actions(win.print_tools_menu)]
    assert labels.index("Stringing Test") == labels.index("Bridge Test") + 1


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))

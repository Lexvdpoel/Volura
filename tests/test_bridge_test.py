"""Bridge Test: a row of bridges over gaps that grow wider, which shows how far a printer
bridges a gap without support.

`bridge_test.bridge_test` gives the test as one closed volume standing on `z = 0`: a plate with
`count` pillars of four millimetres and a bridge over every gap between them, the gaps starting at
`first` millimetres and growing by `step` from left to right. Every bridge is a flat slab that
hangs `height` millimetres above the bed and is `width` millimetres deep, so its underside is an
overhang with nothing under it, and its span stands engraved in the top above that gap. Without
fonts the row simply carries no numbers. The widest spans are dropped when the row would grow past
the bed.

These tests (Qt offscreen) weigh that: one volume on the bed and centred, the gap under every
bridge as wide as asked and plastic in every pillar, every bridge at the height asked, the numbers
cut into the tops of the bridges, the row clamped to the bed, every range reached by clamping and a
number that is no number refused, and the place in the interface (3D Printing, right after the
calibration cube, one group, remembered values).

Run with:  python -m pytest tests/test_bridge_test.py
"""
import os
import sys
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import manifold3d as mf  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from PySide6.QtWidgets import QApplication, QDialog, QDoubleSpinBox, QSpinBox  # noqa: E402

from volura import app as app_module, bridge_test, fonts, ops  # noqa: E402
from volura.settings import qsettings  # noqa: E402

_app = QApplication.instance() or QApplication(sys.argv[:1])   # the numbers need Qt

BASE = dict(first=10.0, step=10.0, count=5, height=8.0, width=8.0)
#: the numbers are polygons, so a part is at most this far off
ABS = 0.05
PLATE = bridge_test.PLATE
PILLAR = bridge_test.PILLAR


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


def nofonts(monkeypatch):
    """No fonts: the row is then exactly the plate with the pillars and the bridges on it."""
    monkeypatch.setattr(fonts, "has_fonts", lambda: False)
    monkeypatch.setattr(fonts, "text_loops", lambda *a, **kw: [])


def need_fonts():
    """Skip the case when this system has no fonts, as the other tests with text do."""
    if not fonts.has_fonts():
        pytest.skip("this system has no fonts, so there are no numbers to test")


def made(**kw) -> mf.Manifold:
    return bridge_test.bridge_test(**dict(BASE, **kw))


def numbers(**kw) -> dict:
    return bridge_test.sizes(**dict(BASE, **kw))


def bbox(m):
    """The bounding box of a manifold as (low, high)."""
    b = np.asarray(m.bounding_box(), dtype=float)
    return b[:3], b[3:]


def places(s):
    """The middle of every gap and of every pillar, measured from the left edge of the row."""
    gaps, pillars, x = [], [], 0.0
    for span in s["spans"]:
        pillars.append(x + 0.5 * PILLAR)
        gaps.append(x + PILLAR + 0.5 * span)
        x += PILLAR + span
    pillars.append(x + 0.5 * PILLAR)
    return gaps, pillars


def block(m, x0, x1, z0, z1, deep) -> float:
    """The volume of the test inside the block from `x0` to `x1` and `z0` to `z1`, `deep` mm
    across the row; the bridges lie centred on y = 0, so the block does too."""
    box = mf.Manifold.cube([x1 - x0, deep, z1 - z0]) \
        .translate((x0, -0.5 * deep, z0))
    return (m ^ box).volume()


def pump_until(predicate, timeout_ms=15000):
    """Spin the event loop until `predicate()` holds or the time is up; returns its value."""
    deadline = time.monotonic() + timeout_ms / 1000.0
    while not predicate() and time.monotonic() < deadline:
        _app.processEvents()
        time.sleep(0.005)
    return predicate()


# ------------------------------------------------------------------ the row

@pytest.mark.parametrize("labels", (False, True))
def test_the_test_is_one_volume_that_stands_on_the_bed_and_is_centred(labels, monkeypatch):
    if not labels:
        nofonts(monkeypatch)
    s, m = numbers(), made()
    assert m.status() == mf.Error.NoError and not m.is_empty()
    assert len(m.decompose()) == 1, "the row fell apart"
    length, depth, top = s["size"]
    lo, hi = bbox(m)
    assert lo[2] == pytest.approx(0.0, abs=1e-6), "the test does not stand on the bed"
    assert np.allclose(lo, [-0.5 * length, -0.5 * depth, 0.0], atol=ABS), "not centred in x and y"
    assert np.allclose(hi, [0.5 * length, 0.5 * depth, top], atol=ABS), "not the size asked"


def test_the_row_is_the_plate_with_the_pillars_and_the_bridges_on_it(monkeypatch):
    """Without fonts nothing is cut away, so the row is exactly the boxes it is made of."""
    nofonts(monkeypatch)
    s, m = numbers(), made()
    length, depth, top = s["size"]
    plate = length * depth * PLATE
    pillars = (s["count"] + 1) * PILLAR * s["width"] * (top - PLATE)
    bridges = sum(span * s["width"] * (top - s["height"]) for span in s["spans"])
    assert m.volume() == pytest.approx(plate + pillars + bridges, rel=1e-6)


@pytest.mark.parametrize("height", (4.0, 8.0, 20.0))
def test_there_is_a_gap_of_the_right_span_under_every_bridge(height, monkeypatch):
    """The gap under a bridge is free from the plate to the bridge and plastic on both sides of
    it, so the bridge really spans the span it is named for."""
    nofonts(monkeypatch)
    s, m = numbers(height=height), made(height=height)
    lo, _hi = bbox(m)
    gaps, pillars = places(s)
    assert len(gaps) == s["count"]
    for span, gap in zip(s["spans"], gaps):
        x = lo[0] + gap
        deep = 0.8 * s["width"]
        assert block(m, x - 0.5 * span + 1.0, x + 0.5 * span - 1.0,
                     PLATE + 0.5, s["height"] - 0.5, deep) == pytest.approx(0.0, abs=1e-6), \
            "the gap under a bridge is not free"
        for edge in (x - 0.5 * span, x + 0.5 * span):
            assert block(m, edge - 1.0, edge + 1.0, PLATE + 0.5, s["height"] - 0.5,
                         deep) > 0.0, "the pillar next to a gap is not where it should be"


def test_every_pillar_is_plastic_from_the_plate_to_the_bridge(monkeypatch):
    nofonts(monkeypatch)
    s, m = numbers(), made()
    lo, _hi = bbox(m)
    _gaps, pillars = places(s)
    assert len(pillars) == s["count"] + 1
    for x in pillars:
        got = block(m, lo[0] + x - 1.5, lo[0] + x + 1.5, PLATE + 0.5, s["height"] - 0.5,
                    0.8 * s["width"])
        assert got == pytest.approx(3.0 * 0.8 * s["width"] * (s["height"] - PLATE - 1.0),
                                    rel=1e-6), "a pillar is not plastic"


@pytest.mark.parametrize("height", (4.0, 8.0, 20.0))
def test_every_bridge_hangs_at_the_height_asked(height, monkeypatch):
    """Just under a bridge there is nothing and just above `height` there is all of it: the
    underside of a bridge is a flat overhang that starts where it was asked to."""
    nofonts(monkeypatch)
    s, m = numbers(height=height), made(height=height)
    lo, _hi = bbox(m)
    for gap in places(s)[0]:
        x = lo[0] + gap
        deep = 0.8 * s["width"]
        assert block(m, x - 2.0, x + 2.0, s["height"] - 0.4, s["height"] - 0.05,
                     deep) == pytest.approx(0.0, abs=1e-6), "there is plastic under a bridge"
        assert block(m, x - 2.0, x + 2.0, s["height"] + 0.05, s["height"] + 0.45, deep) \
            == pytest.approx(4.0 * deep * 0.4, rel=1e-6), "the bridge is not at that height"


# ------------------------------------------------------------------ the numbers

def whole_slab(s, thick) -> float:
    """The volume of a slab of the row `thick` millimetres high: pillar and bridge together cover
    the row from its first edge to its last one, but only as wide as a bridge is."""
    return s["size"][0] * s["width"] * thick


def top_slab(m, s, z0, z1) -> float:
    """The volume of the test that lies between the planes that stand on z at `z0` and `z1`,
    across the whole row."""
    length, depth, _top = s["size"]
    return block(m, -0.5 * length, 0.5 * length, z0, z1, depth)


def test_the_numbers_lower_the_top_of_the_bridges():
    """Just under its top the row holds less plastic than a whole slab of that height is: the
    span of a bridge is engraved in the top of it."""
    need_fonts()
    s = numbers()
    room = 0.5 * bridge_test.ENGRAVE
    top = s["size"][2]
    assert top_slab(made(), s, top - room, top) < whole_slab(s, room), \
        "nothing was engraved in the top of the bridges"


def test_the_numbers_do_not_cut_deeper_than_asked():
    """The slab under the engraving is whole, so the cut stops at `ENGRAVE`."""
    need_fonts()
    s = numbers()
    under = s["size"][2] - bridge_test.ENGRAVE - 0.2
    assert top_slab(made(), s, under, under + 0.2) == pytest.approx(whole_slab(s, 0.2),
                                                                    rel=1e-6), \
        "the engraving cuts too deep"


def test_a_test_without_fonts_carries_no_numbers(monkeypatch):
    nofonts(monkeypatch)
    s = numbers()
    room = 0.5 * bridge_test.ENGRAVE
    top = s["size"][2]
    assert top_slab(made(), s, top - room, top) == pytest.approx(whole_slab(s, room), rel=1e-6), \
        "the top of a row without numbers is not whole"


# ------------------------------------------------------------------ the numbers of the row

def test_sizes_reports_the_row_and_its_size():
    s = bridge_test.sizes(first=8.0, step=6.0, count=3, height=12.0, width=6.0)
    assert s["spans"] == [8.0, 14.0, 20.0]
    assert s["count"] == 3
    assert s["size"] == (8.0 + 14.0 + 20.0 + 4 * PILLAR, 12.0, 12.0 + bridge_test.BRIDGE)
    assert bridge_test.sizes()["spans"] == [10.0, 20.0, 30.0, 40.0, 50.0]


def test_the_row_drops_its_widest_spans_to_fit_the_bed():
    s = bridge_test.sizes(first=30.0, step=20.0, count=8)
    assert s["count"] < 8, "the row kept spans that do not fit the bed"
    assert s["spans"] == [30.0, 50.0, 70.0]
    assert s["size"][0] <= bridge_test.BED
    lo, hi = bbox(bridge_test.bridge_test(first=30.0, step=20.0, count=8))
    assert hi[0] - lo[0] == pytest.approx(s["size"][0], abs=ABS)
    assert hi[0] - lo[0] <= bridge_test.BED + ABS, "the row is too long for the bed"


def test_the_numbers_are_clamped_to_what_prints():
    small = numbers(first=0.0, step=0.0, count=1, height=0.0, width=0.0)
    assert small["first"] == bridge_test.FIRST_MIN
    assert small["step"] == bridge_test.STEP_MIN
    assert small["count"] == bridge_test.COUNT_MIN
    assert small["height"] == bridge_test.HEIGHT_MIN
    assert small["width"] == bridge_test.WIDTH_MIN
    big = numbers(first=999.0, step=999.0, height=999.0, width=999.0)
    assert big["first"] == bridge_test.FIRST_MAX
    assert big["step"] == bridge_test.STEP_MAX
    assert big["height"] == bridge_test.HEIGHT_MAX
    assert big["width"] == bridge_test.WIDTH_MAX
    assert numbers(first=5.0, step=5.0, count=99)["count"] == bridge_test.COUNT_MAX
    lo, hi = bbox(made(first=999.0, height=999.0, width=999.0))
    assert hi[2] - lo[2] == pytest.approx(bridge_test.HEIGHT_MAX + bridge_test.BRIDGE, abs=ABS)


@pytest.mark.parametrize("bad", (float("nan"), float("inf"), None, "wide"))
def test_a_number_that_is_no_number_is_refused(bad):
    for key in ("first", "step", "count", "height", "width"):
        with pytest.raises(ValueError):
            bridge_test.bridge_test(**dict(BASE, **{key: bad}))
        with pytest.raises(ValueError):
            bridge_test.sizes(**dict(BASE, **{key: bad}))


# -------------------------------------------------------------------- the window

def forget_the_values(settings):
    """The bridge test settings back at what the program starts with."""
    for key, value in (("bridge_test_first", 10.0), ("bridge_test_step", 10.0),
                       ("bridge_test_count", 5), ("bridge_test_height", 8.0),
                       ("bridge_test_width", 8.0)):
        setattr(settings, key, value)
    settings.save()


def test_the_dialog_builds_a_test_and_remembers_it(win, monkeypatch):
    seen = {}

    def accept(dlg):
        spins = dlg.findChildren(QDoubleSpinBox)
        count = dlg.findChild(QSpinBox)
        seen["start"] = [spins[0].value(), spins[1].value(), count.value(),
                         spins[2].value(), spins[3].value()]
        spins[0].setValue(12.0)
        count.setValue(4)
        return QDialog.Accepted

    monkeypatch.setattr(app_module.QDialog, "exec", accept)
    win.settings.bridge_test_first = 10.0
    win.settings.bridge_test_step = 10.0
    win.settings.bridge_test_count = 5
    win.settings.bridge_test_height = 8.0
    win.settings.bridge_test_width = 8.0
    try:
        win.bridge_test_dialog()
        assert seen["start"] == [pytest.approx(10.0), pytest.approx(10.0), 5,
                                 pytest.approx(8.0), pytest.approx(8.0)]
        assert pump_until(lambda: len(win.doc.groups) == 1), "no bridge test"
        g = win.doc.groups[0]
        assert g.name == "Bridge test"
        assert g.ents.to_manifold().status() == mf.Error.NoError
        assert [i[2] for i in win.doc.selection] == [g.id]
        s = bridge_test.sizes(first=12.0, step=10.0, count=4)
        lo, hi = bbox(g.ents.to_manifold())
        assert lo[2] == pytest.approx(0.0, abs=1e-6), "the test does not stand on the bed"
        assert hi[0] - lo[0] == pytest.approx(s["size"][0], abs=1e-3), "not as long as asked"
        assert hi[2] - lo[2] == pytest.approx(s["size"][2], abs=1e-3), "not as high as asked"
        assert pump_until(lambda: "bridges" in win.hint.text()), win.hint.text()
        st = app_module.Settings()
        assert st.bridge_test_first == pytest.approx(12.0)
        assert st.bridge_test_step == pytest.approx(10.0)
        assert st.bridge_test_count == 4
        assert st.bridge_test_height == pytest.approx(8.0)
        assert st.bridge_test_width == pytest.approx(8.0)
    finally:
        forget_the_values(win.settings)


def test_one_undo_takes_the_whole_test_away(win):
    steps = len(win.doc.undo_stack)
    win._apply_bridge_test(10.0, 10.0, 2, 8.0, 8.0)
    assert pump_until(lambda: len(win.doc.groups) == 1)
    assert len(win.doc.undo_stack) == steps + 1, "the test is not one undo step"
    win.undo()
    assert win.doc.groups == []


def test_the_test_lands_next_to_what_is_already_in_the_model(win):
    ops.group_from_manifold(win.doc, mf.Manifold.cube([20.0, 20.0, 10.0]), "#cccccc", "block")
    win._apply_bridge_test(10.0, 10.0, 2, 8.0, 8.0)
    assert pump_until(lambda: len(win.doc.groups) == 2), "no bridge test"
    row = win.doc.groups[1]
    assert pump_until(lambda: row.bbox() is not None and row.bbox()[0][0] >= 29.9), \
        "the test did not land 10 mm to the right of the model"
    assert row.bbox()[0][2] == pytest.approx(0.0), "the test does not stand on the bed"


def test_the_settings_clamp_what_they_read():
    q = qsettings()
    keys = ("bridge_test_first", "bridge_test_step", "bridge_test_count",
            "bridge_test_height", "bridge_test_width")
    saved = {k: q.value(k) for k in keys}
    try:
        for k, v in zip(keys, (200.0, 200.0, 99.0, 99.0, 99.0)):
            q.setValue(k, v)
        q.sync()
        s = app_module.Settings()
        assert s.bridge_test_first == 30.0
        assert s.bridge_test_step == 20.0
        assert s.bridge_test_count == 8
        assert s.bridge_test_height == 20.0
        assert s.bridge_test_width == 20.0
    finally:
        for k, v in saved.items():
            q.remove(k) if v is None else q.setValue(k, v)
        q.sync()


def test_the_action_is_in_the_3d_printing_menu_after_the_calibration_cube(win):
    labels = [a.text().replace("&", "").replace("…", "").replace("...", "").strip()
              for a in win.flat_actions(win.print_tools_menu)]
    assert labels.index("Bridge Test") == labels.index("Calibration Cube") + 1


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))

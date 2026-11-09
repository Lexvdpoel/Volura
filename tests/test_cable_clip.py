"""Cable clips: a printable clip that snaps round a cable and is screwed to a wall or a desk.

`cable_clip.cable_clip` gives the clip as one closed volume that stands on `z = 0` and is centred
on the origin in plan, `width` millimetres high, printed lying on its side: a ring of `cable_d`
millimetres inside and `wall` millimetres thick, open at the top with a mouth of `opening` times
the cable diameter, on a flat foot of `base` millimetres thick that runs out on one side and takes
a screw of `bore` through its thickness. `count` stands that many clips in a row along X, 4 mm of
bed apart, as one part.

These tests weigh a clip against what it promises: it stands on the bed and is as high as it is
wide, the screw hole is the only handle, the mouth really is open and narrower than the cable it
holds, and a row is that many clips at the right distance. The input that cannot be printed is
refused. The place in the interface is checked too (Print Tools right after the label plate, one
group in one undo step, and the dialog that remembers its values).

Run with:  python -m pytest tests/test_cable_clip.py
"""
import math
import os
import sys
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import manifold3d as mf  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from PySide6.QtCore import QCoreApplication  # noqa: E402
from PySide6.QtWidgets import (QApplication, QDialog, QDoubleSpinBox,  # noqa: E402
                               QSpinBox)

import volura.app as app_module  # noqa: E402
from volura import cable_clip, threads  # noqa: E402

_app = QApplication.instance() or QApplication(sys.argv[:1])

D, W, WALL, OPEN, FOOT = 6.0, 8.0, 1.6, 0.8, 3.0
BASE = dict(cable_d=D, width=W, wall=WALL, opening=OPEN, base=FOOT, bore="M3", count=1)
OPTIONS = dict(BASE)


def clean(text):
    return text.replace("&", "").replace("…", "").replace("...", "").strip()


def pump_until(predicate, timeout_ms=15000):
    """Spin the event loop until `predicate()` holds or the time is up; returns its value."""
    deadline = time.monotonic() + timeout_ms / 1000.0
    while not predicate() and time.monotonic() < deadline:
        _app.processEvents()
        time.sleep(0.005)
    return predicate()


def bbox(m):
    """The bounding box of a manifold as (low, high)."""
    b = np.asarray(m.bounding_box(), dtype=float)
    return b[:3], b[3:]


def clip(**kw):
    return cable_clip.cable_clip(**dict(BASE, **kw))


def bore_centre(m):
    """The centre of the bore of the ring: the ring is the low-left corner of the clip."""
    lo, _hi = bbox(m)
    r_out = 0.5 * D + WALL
    return lo[0] + r_out, lo[1] + r_out


def mouth_slot(m):
    """The mouth in the world: (x of its middle, the y between the tips of its walls)."""
    cx, cy = bore_centre(m)
    x = 0.5 * OPEN * D                       # the tips stand on the bore at half the mouth
    r_in, r_out = 0.5 * D, 0.5 * D + WALL
    return cx, cy + math.sqrt(r_in * r_in - x * x), cy + math.sqrt(r_out * r_out - x * x)


def mouth_probe(m, width, height=1.0):
    """A box of `width` across the mouth, over the whole height of the clip and past it."""
    x, y0, y1 = mouth_slot(m)
    return mf.Manifold.cube([width, y1 - y0, W + 2.0 * height]) \
        .translate((x - 0.5 * width, y0, -height))


def screw_rod(m, radius, dx=0.0, dz=0.0):
    """A rod along Y through the screw hole of the foot, `dx` and `dz` off its centre."""
    cx, cy = bore_centre(m)
    x = cx + 0.5 * (0.5 * D + (D + 2.0 * WALL + cable_clip.TAB_EXTRA))
    return mf.Manifold.cylinder(20.0, radius, radius, cable_clip.SEGMENTS) \
        .translate((0.0, 0.0, -10.0)).rotate((90.0, 0.0, 0.0)) \
        .translate((x + dx, cy, 0.5 * W + dz))


@pytest.fixture()
def win(monkeypatch, tmp_path):
    monkeypatch.setattr(app_module.errors, "log_dir", lambda: str(tmp_path))
    from volura.app import MainWindow
    w = MainWindow()
    w.doc.dirty = False
    QCoreApplication.processEvents()            # handle the deferred initial view now
    w.resize(900, 600)
    yield w
    w.doc.dirty = False
    w.close()


# ------------------------------------------------------------------ the clip

def test_the_default_clip_is_one_valid_volume_that_stands_on_the_bed():
    m = clip()
    assert m.status() == mf.Error.NoError and not m.is_empty()
    assert len(m.decompose()) == 1, "the foot is not fused to the ring"
    lo, hi = bbox(m)
    assert lo[2] == pytest.approx(0.0), "the clip does not stand on the bed"
    assert hi[2] - lo[2] == pytest.approx(W), "the clip is not as high as it is wide"
    assert m.genus() == 1, "the screw cannot go through the foot"


def test_the_foot_is_a_tab_that_runs_out_beside_the_ring():
    m = clip()
    lo, hi = bbox(m)
    assert hi[0] - lo[0] == pytest.approx(0.5 * D + WALL + D + 2.0 * WALL +
                                          cable_clip.TAB_EXTRA, abs=0.05)
    tab = mf.Manifold.cube([2.0, 4.0 * FOOT, W + 2.0]).translate((hi[0] - 3.0, -2.0 * FOOT, -1.0))
    assert (m ^ tab).volume() == pytest.approx(2.0 * FOOT * W, rel=1e-3), \
        "the foot is not a tab of the thickness asked"


def test_the_screw_hole_goes_through_the_middle_of_the_foot():
    m = clip()
    hole_d = threads.SCREW_HOLES["M3"][0]
    assert (m ^ screw_rod(m, 0.5 * hole_d - 0.05)).volume() == pytest.approx(0.0, abs=1e-6), \
        "the screw does not fit through the foot"
    assert (m ^ screw_rod(m, 0.5 * hole_d + 0.05)).volume() > 0.0, \
        "the hole is wider than the screw clearance"
    for dx, dz in ((-1.0, 0.0), (1.0, 0.0), (0.0, -1.0), (0.0, 1.0)):
        assert (m ^ screw_rod(m, 0.5 * hole_d - 0.05, dx=dx, dz=dz)).volume() > 0.0, \
            "the hole is not in the middle of the tab and of the height"


def test_the_mouth_is_open_and_narrower_than_the_cable():
    m = clip()
    x, y0, y1 = mouth_slot(m)
    thin = mf.Manifold.cube([0.4, 0.4, W + 2.0]).translate((x - 0.2, 0.5 * (y0 + y1) - 0.2, -1.0))
    assert (m ^ thin).volume() == pytest.approx(0.0, abs=1e-6), "the mouth is closed"
    loose = mouth_probe(m, OPEN * D - 0.2)
    assert (m ^ loose).volume() == pytest.approx(0.0, abs=1e-6), "the mouth is wider than asked"
    tight = mouth_probe(m, OPEN * D + 0.2)
    assert (m ^ tight).volume() > 0.0, "the mouth is wider than the cable, so nothing holds it"
    assert OPEN * D < D, "a cable that walks out of the mouth is not held"


@pytest.mark.parametrize("kw, wrong", (
    ({"cable_d": 1.0}, "cable diameter"), ({"cable_d": 45.0}, "cable diameter"),
    ({"width": 3.0}, "width of a cable clip"), ({"width": 45.0}, "width of a cable clip"),
    ({"wall": 0.4}, "wall of a cable clip"), ({"wall": 6.0}, "wall of a cable clip"),
    ({"opening": 0.2}, "opening of a cable clip"), ({"opening": 0.99}, "opening of a cable clip"),
    ({"base": 1.0}, "foot of a cable clip"), ({"base": 9.0}, "foot of a cable clip"),
    ({"bore": "M9"}, "unknown screw size"), ({"bore": None}, "unknown screw size"),
    ({"count": 0}, "number of cable clips"), ({"count": 13}, "number of cable clips"),
    ({"width": 4.0}, "screw hole does not fit"),
    ({"width": 10.0, "bore": "M8"}, "screw hole does not fit")))
def test_impossible_clips_raise(kw, wrong):
    with pytest.raises(ValueError, match=wrong):
        clip(**kw)


def test_a_row_is_that_many_clips_in_a_line_along_x():
    n = 3
    one = clip()
    m = clip(count=n)
    parts = m.decompose()
    assert len(parts) == n, "the row is not one clip per piece"
    assert [p.volume() for p in parts] == pytest.approx([one.volume()] * n, rel=1e-6)
    spacing = bbox(one)[1][0] - bbox(one)[0][0] + cable_clip.ROW_ROOM
    xs = sorted(float(np.asarray(p.bounding_box())[0]) for p in parts)
    assert [x - xs[0] for x in xs] == pytest.approx([i * spacing for i in range(n)])
    lo, hi = bbox(m)
    assert lo[2] == pytest.approx(0.0) and hi[2] == pytest.approx(W)
    assert hi[1] - lo[1] == pytest.approx(bbox(one)[1][1] - bbox(one)[0][1]), \
        "the row runs along something else than X"
    assert 0.5 * (lo[0] + hi[0]) == pytest.approx(0.0), "the row is not centred"


# ------------------------------------------------------------------ in the window

def test_the_clips_are_one_group_that_one_undo_takes_away(win):
    steps = len(win.doc.undo_stack)
    win.make_cable_clip_model(OPTIONS)
    assert pump_until(lambda: len(win.doc.groups) == 1), "no cable clips in the model"
    assert len(win.doc.undo_stack) == steps + 1, "the clips are not one undo step"
    g = win.doc.groups[0]
    assert g.name == "Cable clip"
    m = g.ents.to_manifold()
    assert m.status() == mf.Error.NoError
    assert m.volume() == pytest.approx(clip().volume(), rel=1e-6)
    assert [g.id for g in win.doc.selected_groups()] == [g.id for g in win.doc.groups]
    win.doc.undo()
    assert win.doc.groups == []


def test_the_row_stands_on_the_bed_to_the_right_of_the_model(win):
    win.make_cable_clip_model(dict(OPTIONS, count=3))
    assert pump_until(lambda: len(win.doc.groups) == 1), "no cable clips in the model"
    assert pump_until(lambda: win.doc.groups[0].bbox() is not None)
    lo, hi = np.asarray(win.doc.groups[0].bbox(), dtype=float)
    assert lo[2] == pytest.approx(0.0), "the clips do not stand on the bed"
    assert hi[2] == pytest.approx(W)
    bx, by, _bz = win.settings.bed_size
    assert 0.5 * (lo[0] + hi[0]) == pytest.approx(0.5 * bx), "the row is not on the bed"
    assert 0.5 * (lo[1] + hi[1]) == pytest.approx(0.5 * by)
    win.make_cable_clip_model(dict(OPTIONS, count=3, width=12.0))
    assert pump_until(lambda: len(win.doc.groups) == 2), "the second row is missing"
    first, second = (np.asarray(g.bbox(), dtype=float) for g in win.doc.groups)
    assert second[0][0] - first[1][0] == pytest.approx(10.0), "the rows are not 10 mm apart"
    assert pump_until(lambda: "cable clips for" in win.hint.text()), win.hint.text()
    assert win.hint.text() == "3 cable clips for 6 mm cable added."


def test_a_clip_that_cannot_be_built_is_a_clear_text(win):
    win.make_cable_clip_model(dict(OPTIONS, width=4.0))
    assert pump_until(lambda: "Cable clip failed" in win.hint.text()), win.hint.text()
    assert win.doc.groups == []


def test_the_cable_clip_follows_the_label_plate_in_print_tools(win):
    labels = [clean(a.text()) for a in win.flat_actions(win.menuBar())]
    assert labels.index("Cable Clip") == labels.index("Label Plate") + 1


def test_the_dialog_starts_at_the_last_values_and_remembers_the_new_ones(win, monkeypatch):
    seen = {}

    def fill(dlg):
        seen["title"] = dlg.windowTitle()
        seen["start"] = [round(dlg.cable_d.value(), 2), round(dlg.width.value(), 2),
                         round(dlg.wall.value(), 2), round(dlg.opening.value(), 2),
                         round(dlg.base.value(), 2)]
        seen["count"] = dlg.findChild(QSpinBox).value()
        seen["bore"] = dlg.size.currentData()
        seen["spins"] = len(dlg.findChildren(QDoubleSpinBox))
        seen["percent"] = dlg.opening.suffix()
        dlg.cable_d.setValue(10.0)
        dlg.width.setValue(12.0)
        dlg.wall.setValue(2.0)
        dlg.opening.setValue(65.0)
        dlg.base.setValue(4.0)
        dlg.size.setCurrentIndex(dlg.size.findData("M4"))
        dlg.count.setValue(3)
        return QDialog.Accepted

    monkeypatch.setattr(app_module.QDialog, "exec", fill)
    win.settings.clip_cable_d, win.settings.clip_width = 4.0, 6.0
    win.settings.clip_wall, win.settings.clip_opening = 1.2, 0.7
    win.settings.clip_base, win.settings.clip_bore, win.settings.clip_count = 2.0, "M2.5", 2
    win.cable_clip_dialog()
    assert seen["title"] == "Cable clip"
    assert seen["start"] == [4.0, 6.0, 1.2, 70.0, 2.0]
    assert seen["count"] == 2
    assert seen["bore"] == "M2.5"
    assert seen["spins"] == 5 and seen["percent"] == " %"
    assert pump_until(lambda: len(win.doc.groups) == 1), "no cable clips in the model"
    assert len(win.doc.groups[0].ents.to_manifold().decompose()) == 3, \
        "the count did not reach the model"
    lo, hi = np.asarray(win.doc.groups[0].bbox(), dtype=float)
    assert hi[2] - lo[2] == pytest.approx(12.0), "the width is not as asked"
    assert win.settings.clip_cable_d == pytest.approx(10.0)
    assert win.settings.clip_count == 3
    s = app_module.Settings()
    assert (s.clip_cable_d, s.clip_width, s.clip_wall, s.clip_opening, s.clip_base,
            s.clip_bore, s.clip_count) == (pytest.approx(10.0), pytest.approx(12.0),
                                           pytest.approx(2.0), pytest.approx(0.65),
                                           pytest.approx(4.0), "M4", 3)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))

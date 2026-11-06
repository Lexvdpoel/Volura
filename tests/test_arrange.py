"""Arrange on Bed: lay the groups side by side on the print bed, like a slicer.

`ops.arrange_on_bed()` nests the footprints of the groups (the convex hull of their
vertices on XY) with the same algorithm as the worksheet: largest part first, in the
lowest and then leftmost free spot, turned 0, 90, 180 or 270 degrees, with a margin along
the edge and a gap between the parts; afterwards every part stands on the bed. These
tests (Qt offscreen) check:

* three cubes that were stacked on the same spot end up inside the bed, with the gap free
  between them, all on z = 0 and with the same volume;
* a part that is too large for the bed raises and changes nothing;
* a part is turned 90 degrees when that is the only way it fits;
* a locked group stays where it is;
* the menu action takes every visible group when nothing is selected, in one undo step,
  and shows a warning when the parts do not fit.
"""
import itertools
import os
import sys
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import manifold3d as mf  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from volura import app as app_module, ops  # noqa: E402
from volura.model import Document  # noqa: E402

_app = QApplication.instance() or QApplication(sys.argv[:1])

BED = (256.0, 256.0, 256.0)
CUBE = 50.0
GAP = 5.0
MARGIN = 5.0


class FakeBox:
    """Stands in for QMessageBox: remembers what the window asked."""

    def __init__(self):
        self.asked = []

    def warning(self, *args, **kwargs):
        self.asked.append(args[1:])


@pytest.fixture()
def win(monkeypatch, tmp_path):
    """A window that logs and autosaves into the folder of the test."""
    monkeypatch.setattr(app_module.errors, "log_dir", lambda: str(tmp_path))
    monkeypatch.setattr(app_module, "autosave_path", lambda: str(tmp_path / "autosave.volura"))
    box = FakeBox()
    monkeypatch.setattr(app_module, "QMessageBox", box)
    w = app_module.MainWindow()
    w.box = box
    w.settings.bed_size = BED
    w.doc.dirty = False
    yield w
    w.doc.dirty = False
    w.close()


def add_cube(doc, at, size=CUBE, name="cube"):
    """A closed cube of `size` mm with its corner on `at`, as its own group."""
    m = mf.Manifold.cube((float(size), float(size), float(size))).translate(at)
    return ops.group_from_manifold(doc, m, "#cccccc", name)


def stacked_cubes(doc, n=3):
    """`n` cubes of 50 mm, all on the same spot on the bed, one on top of the other."""
    return [add_cube(doc, (10.0, 10.0, k * CUBE), name=f"cube {k}") for k in range(n)]


def gap_between(a, b):
    """The free space between two bounding boxes on the axis that separates them, -1 if they overlap."""
    for axis in (0, 1):
        d = max(a[0][axis] - b[1][axis], b[0][axis] - a[1][axis])
        if d > 1e-6:
            return d
    return -1.0


def assert_on_bed(groups, bed=BED, gap=GAP):
    """Every group inside the bed with its margin, on the bed, and `gap` mm from the others."""
    boxes = [g.ents.bbox() for g in groups]
    for lo, hi in boxes:
        assert lo[2] == pytest.approx(0.0), "the part does not stand on the bed"
        assert (lo[:2] >= [MARGIN - 1e-6] * 2).all(), f"outside the margin: {lo}"
        assert (hi[:2] <= np.array(bed[:2]) - MARGIN + 1e-6).all(), f"outside the bed: {hi}"
    for a, b in itertools.combinations(boxes, 2):
        assert gap_between(a, b) >= gap - 1e-6, f"parts overlap or touch: {a} vs {b}"


def pump_until(predicate, timeout_ms=15000):
    """Spin the event loop until `predicate()` holds or the time is up; returns its value."""
    deadline = time.monotonic() + timeout_ms / 1000.0
    while not predicate() and time.monotonic() < deadline:
        _app.processEvents()
        time.sleep(0.005)
    return predicate()


def spots(groups):
    """The XY corner of every group, so we can see whether the parts moved apart."""
    return {tuple(np.round(g.ents.bbox()[0][:2], 3)) for g in groups}


# ---------------------------------------------------------------- the operation

def test_stacked_cubes_end_up_side_by_side():
    doc = Document()
    gs = stacked_cubes(doc)
    volumes = [g.ents.to_manifold().volume() for g in gs]
    assert ops.arrange_on_bed(doc, [("g", g.id, g.id) for g in gs], BED) == 0
    assert_on_bed(gs)
    for g, v in zip(gs, volumes):
        assert g.ents.to_manifold().volume() == pytest.approx(v)


def test_parts_keep_their_shape_while_arranging():
    """Only turning about the vertical axis and moving: nothing is scaled or skewed."""
    doc = Document()
    g = add_cube(doc, (10.0, 10.0, 20.0))
    ops.arrange_on_bed(doc, [("g", g.id, g.id)], BED)
    lo, hi = g.ents.bbox()
    assert np.allclose(hi - lo, [CUBE, CUBE, CUBE])
    assert lo[2] == pytest.approx(0.0)


def test_a_part_that_is_too_large_raises_and_changes_nothing():
    doc = Document()
    big = ops.group_from_manifold(doc, mf.Manifold.cube((300.0, 300.0, 300.0)), "#cccccc", "big")
    small = add_cube(doc, (10.0, 10.0, 0.0))
    before = [g.ents.bbox().copy() for g in (big, small)]
    with pytest.raises(ValueError) as e:
        ops.arrange_on_bed(doc, ops.whole_items(doc), BED)
    assert "256" in str(e.value)
    for g, b in zip((big, small), before):
        assert np.allclose(g.ents.bbox(), b), "a part moved while the bed was too small"


def test_a_part_is_turned_when_that_is_the_only_way_it_fits():
    doc = Document()
    a = ops.group_from_manifold(doc, mf.Manifold.cube((90.0, 50.0, 10.0)), "#cccccc", "bar a")
    b = ops.group_from_manifold(doc, mf.Manifold.cube((30.0, 60.0, 10.0)), "#cccccc", "bar b")
    ops.arrange_on_bed(doc, [("g", a.id, a.id), ("g", b.id, b.id)], (100.0, 100.0, 100.0))
    lo, hi = b.ents.bbox()
    assert np.allclose(hi[:2] - lo[:2], [60.0, 30.0]), "bar b was not turned 90 degrees"
    assert_on_bed([a, b], bed=(100.0, 100.0, 100.0))


def test_a_locked_group_stays_where_it_is():
    doc = Document()
    a = add_cube(doc, (0.0, 0.0, 0.0), name="a")
    b = add_cube(doc, (0.0, 0.0, 0.0), name="b")
    b.locked = True
    ops.arrange_on_bed(doc, [("g", a.id, a.id), ("g", b.id, b.id)], BED)
    assert np.allclose(b.ents.bbox(), [[0.0, 0.0, 0.0], [CUBE, CUBE, CUBE]])
    assert a.ents.bbox()[0][0] == pytest.approx(MARGIN)


# ---------------------------------------------------------------- the window

def test_arrange_takes_every_visible_group_when_nothing_is_selected(win):
    gs = stacked_cubes(win.doc)
    hidden = add_cube(win.doc, (10.0, 10.0, 3 * CUBE), name="hidden")
    hidden.visible = False
    before = hidden.ents.bbox().copy()
    win.a_arrange.trigger()
    assert pump_until(lambda: len(spots(gs)) == len(gs)), "the parts did not move apart"
    assert_on_bed(gs)
    assert np.allclose(hidden.ents.bbox(), before), "a hidden group was arranged too"


def test_arrange_is_one_undo_step(win):
    gs = stacked_cubes(win.doc)
    before = [g.ents.bbox().copy() for g in gs]
    win.a_arrange.trigger()
    assert pump_until(lambda: len(spots(gs)) == len(gs))
    win.doc.undo()
    for g, b in zip(gs, before):
        assert np.allclose(g.ents.bbox(), b), "one undo did not restore the old place"


def test_arrange_of_a_too_large_part_warns_and_changes_nothing(win):
    g = ops.group_from_manifold(win.doc, mf.Manifold.cube((300.0, 300.0, 300.0)), "#cccccc", "big")
    before = g.ents.bbox().copy()
    steps = len(win.doc.undo_stack)
    win.a_arrange.trigger()
    assert pump_until(lambda: bool(win.box.asked)), "no warning was shown"
    assert "256" in win.box.asked[0][-1]
    assert np.allclose(g.ents.bbox(), before), "the model changed although nothing fitted"
    assert len(win.doc.undo_stack) == steps, "nothing to undo: the model did not change"

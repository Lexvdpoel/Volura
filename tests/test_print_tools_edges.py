"""The print tools at their edges: what a selection means, and when nothing is undone.

Elephant Foot Chamfer, Mouse Ears, Orient for Printing and Arrange on Bed work on the groups
the selection points at, or on every visible group when nothing is selected. Lay Flat on This
Face turns the group of the right-clicked face onto that face. These tests (Qt offscreen)
check the edges those tools share:

* a face or an edge clicked inside a group means the group: the part is chamfered, given
  ears, turned or arranged, not silently skipped;
* with nothing selected a hidden group is left exactly as it was, only the visible ones move;
* loose geometry at the top level and an empty group change nothing and raise nothing: no
  exception dialog and no undo step;
* when nothing moves there is no undo step (the undo stack does not grow), and when a part
  does move one undo brings the old shape or place back;
* Lay Flat on a face that already lies on the bed does not take an undo step either.

Run with:  python -m pytest tests/test_print_tools_edges.py
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
from PySide6.QtWidgets import QApplication, QDialog  # noqa: E402

from volura import app as app_module, ops  # noqa: E402
from volura.brep import Entities  # noqa: E402
from volura.geom import Plane, rotation_matrix  # noqa: E402
from volura.model import ROOT, Document  # noqa: E402

_app = QApplication.instance() or QApplication(sys.argv[:1])

CUBE = 20.0
BOX = (20.0, 20.0, 10.0)
BOX_VOLUME = BOX[0] * BOX[1] * BOX[2]
BED = (256.0, 256.0, 256.0)
MARGIN = 5.0
DIAMETER = 10.0
HEIGHT = 0.4


class FakeAsk:
    """Stands in for QInputDialog: answers the chamfer question with `value`."""

    def __init__(self, value=0.5):
        self.value = float(value)
        self.asked = []

    def getDouble(self, *args, **kwargs):
        self.asked.append(args[1:])
        return self.value, True


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
    """A window that asks for a size, never shows a box and nests on a normal bed."""
    monkeypatch.setattr(app_module.errors, "log_dir", lambda: str(tmp_path))
    monkeypatch.setattr(app_module, "autosave_path", lambda: str(tmp_path / "autosave.volura"))
    box = FakeBox()
    monkeypatch.setattr(app_module, "QInputDialog", FakeAsk())
    monkeypatch.setattr(app_module, "QMessageBox", box)
    w = app_module.MainWindow()
    w.box = box
    w.settings.bed_size = BED
    w.doc.dirty = False
    yield w
    w.doc.dirty = False
    w.close()


def add_cube(doc, size=CUBE, at=(0.0, 0.0, 0.0), name="cube"):
    """A closed cube of `size` mm with its corner on `at`, as its own group."""
    m = mf.Manifold.cube((float(size), float(size), float(size))).translate(at)
    return ops.group_from_manifold(doc, m, "#cccccc", name)


def add_box(doc, size=BOX, name="box"):
    """A closed box of `size` mm standing on the bed, as its own group."""
    return ops.group_from_manifold(doc, mf.Manifold.cube(size), "#cccccc", name)


def add_cylinder(doc, name="cylinder"):
    """A cylinder of r = 10 and h = 20 mm on the bed, out of 32 segments (no sharp corner)."""
    return ops.group_from_manifold(doc, mf.Manifold.cylinder(20.0, 10.0, 10.0, 32), "#cccccc", name)


def tilt_about_a_vertex(doc, name="tilted"):
    """A cube turned 45° about X and then 45° about Y: it stands on one vertex."""
    g = add_cube(doc, name=name)
    for axis in (np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])):
        g.ents.transform(rotation_matrix(axis, math.radians(45)), np.zeros(3))
    return g


def cone(doc, name="cone"):
    """A cone of r = 10 and h = 20 mm that stands on its tip, so it wants turning."""
    g = ops.group_from_manifold(doc, mf.Manifold.cylinder(20.0, 10.0, 0.0, 64), "#cccccc", name)
    g.ents.transform(rotation_matrix(np.array([1.0, 0.0, 0.0]), math.pi), np.zeros(3))
    return g


def face_id(g):
    """The id of any face of a group, to click a face of the part."""
    return next(iter(g.ents.F))


def bottom_face_id(g):
    """The id of the face of `g` whose outward normal points straight down."""
    for fid, face in g.ents.F.items():
        if np.allclose(np.asarray(face.normal, dtype=float), (0.0, 0.0, -1.0), atol=1e-6):
            return fid
    raise AssertionError("no face points down")


def loose_and_empty(w):
    """A loose rectangle on the top level and an empty group, as nothing-selected targets."""
    loop = np.array([[0, 0], [40, 0], [40, 30], [0, 30]], dtype=float)
    fid = w.doc.root.add_shape(Plane((0, 0, 0), (0, 0, 1)), [loop])[0]
    return fid, w.doc.add_group(Entities(), "#cccccc", "empty")


def vertices(g):
    """The vertices of a group in a fixed order, so undo can be compared."""
    P = np.array(list(g.ents.V.values()), dtype=float)
    return P[np.lexsort(P.T)]


def spots(groups):
    """The XY corner of every group, so a move apart can be seen."""
    return {tuple(np.round(g.ents.bbox()[0][:2], 3)) for g in groups}


def pump_until(predicate, timeout_ms=15000):
    """Spin the event loop until `predicate()` holds or the time is up; returns its value."""
    deadline = time.monotonic() + timeout_ms / 1000.0
    while not predicate() and time.monotonic() < deadline:
        _app.processEvents()
        time.sleep(0.005)
    return predicate()


# ---------------------------------------------------------------- Elephant Foot Chamfer

def test_chamfer_a_clicked_face_chamfers_its_group(win):
    g = add_cube(win.doc)
    win.doc.set_selection([("f", g.id, face_id(g))])
    win.a_elephant.trigger()
    assert pump_until(lambda: g.ents.to_manifold().volume() < CUBE ** 3 - 1), "the part was not chamfered"


def test_chamfer_a_hidden_group_is_left_alone(win):
    a, b = add_cube(win.doc, name="a"), add_cube(win.doc, name="b")
    hidden = add_cube(win.doc, name="hidden")
    hidden.visible = False
    before = hidden.ents.to_manifold().volume()
    win.a_elephant.trigger()
    ready = lambda: a.ents.to_manifold().volume() < CUBE ** 3 - 1 and \
        b.ents.to_manifold().volume() < CUBE ** 3 - 1
    assert pump_until(ready), "not every visible group was chamfered"
    assert hidden.ents.to_manifold().volume() == pytest.approx(before), "a hidden group was chamfered"


def test_chamfer_loose_geometry_and_an_empty_group_change_nothing(win):
    fid, empty = loose_and_empty(win)
    steps = len(win.doc.undo_stack)
    win.a_elephant.trigger()
    assert pump_until(lambda: not win._busy), "the job did not finish"
    assert len(win.doc.undo_stack) == steps, "nothing changed: there is nothing to undo"
    assert not empty.ents.V and fid in win.doc.root.F, "the empty group or the loose face changed"
    assert not win.box.asked, "an empty group should not raise a dialog"


def test_chamfer_a_part_without_bottom_edges_leaves_no_undo_step(win):
    tilted = tilt_about_a_vertex(win.doc)
    steps = len(win.doc.undo_stack)
    win.doc.set_selection([("g", tilted.id, tilted.id)])
    win.a_elephant.trigger()
    assert pump_until(lambda: "None of these parts" in win.hint.text()), win.hint.text()
    assert len(win.doc.undo_stack) == steps, "nothing changed: there is nothing to undo"


def test_chamfer_one_undo_restores_the_part(win):
    g = add_cube(win.doc)
    win.doc.set_selection([("g", g.id, g.id)])
    win.a_elephant.trigger()
    assert pump_until(lambda: g.ents.to_manifold().volume() < CUBE ** 3 - 1)
    win.doc.undo()
    assert g.ents.to_manifold().volume() == pytest.approx(CUBE ** 3), "one undo did not restore the part"


# ---------------------------------------------------------------- Mouse Ears

def test_ears_a_clicked_face_gives_ears_to_its_group(win, monkeypatch):
    monkeypatch.setattr(app_module.QDialog, "exec", lambda dlg: QDialog.Accepted)
    g = add_box(win.doc)
    win.doc.set_selection([("f", g.id, face_id(g))])
    win.a_mouse_ears.trigger()
    assert pump_until(lambda: g.ents.to_manifold().volume() > BOX_VOLUME + 1), "the part got no ears"


def test_ears_a_hidden_group_is_left_alone(win, monkeypatch):
    monkeypatch.setattr(app_module.QDialog, "exec", lambda dlg: QDialog.Accepted)
    a, b = add_box(win.doc, name="a"), add_box(win.doc, name="b")
    hidden = add_box(win.doc, name="hidden")
    hidden.visible = False
    before = hidden.ents.to_manifold().volume()
    win.a_mouse_ears.trigger()
    ready = lambda: a.ents.to_manifold().volume() > BOX_VOLUME + 1 and \
        b.ents.to_manifold().volume() > BOX_VOLUME + 1
    assert pump_until(ready), "not every visible group got ears"
    assert hidden.ents.to_manifold().volume() == pytest.approx(before), "a hidden group got ears"


def test_ears_loose_geometry_and_an_empty_group_change_nothing(win, monkeypatch):
    monkeypatch.setattr(app_module.QDialog, "exec", lambda dlg: QDialog.Accepted)
    fid, empty = loose_and_empty(win)
    steps = len(win.doc.undo_stack)
    win.a_mouse_ears.trigger()
    assert pump_until(lambda: not win._busy), "the job did not finish"
    assert len(win.doc.undo_stack) == steps, "nothing changed: there is nothing to undo"
    assert not empty.ents.V and fid in win.doc.root.F, "the empty group or the loose face changed"
    assert not win.box.asked, "an empty group should not raise a dialog"


def test_ears_a_part_without_sharp_corners_leaves_no_undo_step(win, monkeypatch):
    monkeypatch.setattr(app_module.QDialog, "exec", lambda dlg: QDialog.Accepted)
    cyl = add_cylinder(win.doc)
    steps = len(win.doc.undo_stack)
    win.doc.set_selection([("g", cyl.id, cyl.id)])
    win.a_mouse_ears.trigger()
    assert pump_until(lambda: "None of these parts" in win.hint.text()), win.hint.text()
    assert len(win.doc.undo_stack) == steps, "nothing changed: there is nothing to undo"


def test_ears_one_undo_restores_the_part(win, monkeypatch):
    monkeypatch.setattr(app_module.QDialog, "exec", lambda dlg: QDialog.Accepted)
    g = add_box(win.doc)
    win.doc.set_selection([("g", g.id, g.id)])
    win.a_mouse_ears.trigger()
    assert pump_until(lambda: g.ents.to_manifold().volume() > BOX_VOLUME + 1)
    win.doc.undo()
    assert g.ents.to_manifold().volume() == pytest.approx(BOX_VOLUME), "one undo did not restore the part"


# ---------------------------------------------------------------- Orient for Printing

def test_orient_a_clicked_face_turns_its_group(win):
    g = cone(win.doc)
    win.doc.set_selection([("f", g.id, face_id(g))])
    win.a_orient_print.trigger()
    ready = lambda: "Turned 1 parts to need the least support." in win.hint.text()
    assert pump_until(ready), win.hint.text()
    assert g.ents.bbox()[0][2] == pytest.approx(0.0), "the cone does not stand on the bed"


def test_orient_a_hidden_group_is_left_alone(win):
    a, b = cone(win.doc, "a"), cone(win.doc, "b")
    hidden = cone(win.doc, "hidden")
    hidden.visible = False
    before = vertices(hidden)
    win.a_orient_print.trigger()
    ready = lambda: "Turned 2 parts to need the least support." in win.hint.text()
    assert pump_until(ready), win.hint.text()
    assert np.allclose(vertices(hidden), before), "a hidden group was turned"


def test_orient_loose_geometry_and_an_empty_group_change_nothing(win):
    fid, empty = loose_and_empty(win)
    steps = len(win.doc.undo_stack)
    win.a_orient_print.trigger()
    assert pump_until(lambda: not win._busy), "the job did not finish"
    assert len(win.doc.undo_stack) == steps, "nothing changed: there is nothing to undo"
    assert not empty.ents.V and fid in win.doc.root.F, "the empty group or the loose face changed"
    assert not win.box.asked, "an empty group should not raise a dialog"


def test_orient_a_part_that_already_stands_right_leaves_no_undo_step(win):
    g = add_box(win.doc, size=(30.0, 30.0, 10.0))
    steps = len(win.doc.undo_stack)
    win.doc.set_selection([("g", g.id, g.id)])
    win.a_orient_print.trigger()
    ready = lambda: "already the right way round" in win.hint.text()
    assert pump_until(ready), win.hint.text()
    assert len(win.doc.undo_stack) == steps, "nothing moved: there is nothing to undo"


def test_orient_one_undo_restores_the_part(win):
    g = cone(win.doc)
    before = vertices(g)
    win.doc.set_selection([("g", g.id, g.id)])
    win.a_orient_print.trigger()
    assert pump_until(lambda: "Turned 1 parts" in win.hint.text()), win.hint.text()
    win.doc.undo()
    assert np.allclose(vertices(g), before), "one undo did not bring the old place back"


# ---------------------------------------------------------------- Arrange on Bed

def test_arrange_a_clicked_face_arranges_its_group(win):
    g = add_cube(win.doc, at=(0.0, 0.0, 0.0))
    win.doc.set_selection([("f", g.id, face_id(g))])
    win.a_arrange.trigger()
    assert pump_until(lambda: not win._busy), "the job did not finish"
    assert g.ents.bbox()[0][0] == pytest.approx(MARGIN), "the part was not moved to its place"


def test_arrange_a_hidden_group_is_left_alone(win):
    a, b = add_cube(win.doc, at=(10.0, 10.0, 0.0), name="a"), add_cube(win.doc, at=(10.0, 10.0, 40.0), name="b")
    hidden = add_cube(win.doc, at=(10.0, 10.0, 80.0), name="hidden")
    hidden.visible = False
    before = hidden.ents.bbox().copy()
    win.a_arrange.trigger()
    assert pump_until(lambda: len(spots([a, b])) == 2), "the parts did not move apart"
    assert np.allclose(hidden.ents.bbox(), before), "a hidden group was arranged too"


def test_arrange_loose_geometry_and_an_empty_group_change_nothing(win):
    fid, empty = loose_and_empty(win)
    steps = len(win.doc.undo_stack)
    win.a_arrange.trigger()
    assert pump_until(lambda: not win._busy), "the job did not finish"
    assert len(win.doc.undo_stack) == steps, "nothing changed: there is nothing to undo"
    assert not empty.ents.V and fid in win.doc.root.F, "the empty group or the loose face changed"
    assert not win.box.asked, "an empty group should not raise a dialog"


def test_arrange_a_part_that_is_already_placed_leaves_no_undo_step(win):
    g = add_cube(win.doc, at=(0.0, 0.0, 0.0))
    ops.arrange_on_bed(win.doc, [("g", g.id, g.id)], BED)     # put it exactly on its nest spot
    before = g.ents.bbox().copy()
    steps = len(win.doc.undo_stack)
    win.doc.set_selection([("g", g.id, g.id)])
    win.a_arrange.trigger()
    assert pump_until(lambda: not win._busy), "the job did not finish"
    assert len(win.doc.undo_stack) == steps, "nothing moved: there is nothing to undo"
    assert np.allclose(g.ents.bbox(), before), "the part moved although it was already placed"


def test_arrange_one_undo_restores_both_parts(win):
    a, b = add_cube(win.doc, at=(10.0, 10.0, 0.0), name="a"), add_cube(win.doc, at=(10.0, 10.0, 40.0), name="b")
    before = [g.ents.bbox().copy() for g in (a, b)]
    win.doc.set_selection([("g", a.id, a.id), ("g", b.id, b.id)])
    win.a_arrange.trigger()
    assert pump_until(lambda: len(spots([a, b])) == 2), "the parts did not move apart"
    win.doc.undo()
    for g, bb in zip((a, b), before):
        assert np.allclose(g.ents.bbox(), bb), "one undo did not restore the old place"


# ---------------------------------------------------------------- Lay Flat on This Face

def test_lay_flat_a_face_that_already_lies_on_the_bed_leaves_no_undo_step(win):
    g = add_box(win.doc, size=(10.0, 20.0, 30.0))
    before = vertices(g)
    steps = len(win.doc.undo_stack)
    win.doc.set_selection([("g", g.id, g.id)])
    win._context_face = (g.id, bottom_face_id(g))
    win.lay_flat_on_face()
    assert len(win.doc.undo_stack) == steps, "nothing moved: there is nothing to undo"
    assert np.allclose(vertices(g), before), "a part that already lies flat was moved"

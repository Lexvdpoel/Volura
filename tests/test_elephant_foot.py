"""Elephant Foot Chamfer: a small chamfer along the bottom edges of a part.

The first layer of a print is squashed a little wider than the rest of the part, so it
sticks out: an "elephant's foot". `ops.elephant_foot()` chamfers the bottom edges of the
given groups; the bottom edges of a part are those that lie at its lowest z and belong to
a face that points straight down. A chamfer of d mm along a right angle takes a triangle
of d*d/2 per millimetre of edge off the part, so a cube of 20 mm with a chamfer of 0.5 mm
loses 4 * 20 * 0.125 = 10 mm³, apart from the corners where the cuts overlap a little.
These tests (Qt offscreen) check:

* the volume of a chamfered cube, its untouched top face and its smaller bottom face;
* the bottom rim of a cylinder, which is one chain of edges;
* a part that only touches the bed with a vertex, a locked part, a size that does not fit
  and a chamfer that raises are skipped and stay exactly as they were;
* the menu action: every visible group when nothing is selected, the size in the question,
  one undo for the whole operation and the hint in the status bar.

Run with:  python -m pytest tests/test_elephant_foot.py
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
from PySide6.QtWidgets import QApplication  # noqa: E402

from volura import app as app_module, fillet, ops  # noqa: E402
from volura.geom import rotation_matrix  # noqa: E402
from volura.model import Document  # noqa: E402

_app = QApplication.instance() or QApplication(sys.argv[:1])

CUBE = 20.0
SIZE = 0.5


class FakeAsk:
    """Stands in for QInputDialog: remembers the question and answers with `value`."""

    def __init__(self, value=SIZE):
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


@pytest.fixture()
def win(monkeypatch, tmp_path):
    """A window that asks for a size and logs into the folder of the test."""
    monkeypatch.setattr(app_module.errors, "log_dir", lambda: str(tmp_path))
    monkeypatch.setattr(app_module, "autosave_path", lambda: str(tmp_path / "autosave.volura"))
    ask = FakeAsk()
    monkeypatch.setattr(app_module, "QInputDialog", ask)
    monkeypatch.setattr(app_module, "QMessageBox", FakeBox())
    w = app_module.MainWindow()
    w.ask = ask
    w.doc.dirty = False
    yield w
    w.doc.dirty = False
    w.close()


def add_cube(doc, size=CUBE, name="cube", color="#cccccc"):
    """A closed cube of `size` mm standing on the bed, as its own group."""
    m = mf.Manifold.cube((float(size), float(size), float(size)))
    return ops.group_from_manifold(doc, m, color, name)


def tilt_about_a_vertex(doc, name="tilted"):
    """A cube turned 45° about X and then 45° about Y: it stands on one vertex."""
    g = add_cube(doc, name=name)
    for axis in (np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])):
        g.ents.transform(rotation_matrix(axis, math.radians(45)), np.zeros(3))
    return g


def add_cylinder(doc, name="cylinder"):
    """A cylinder of r = 10 mm and h = 20 mm on the bed, out of 32 segments."""
    m = mf.Manifold.cylinder(20.0, 10.0, 10.0, 32)
    return ops.group_from_manifold(doc, m, "#cccccc", name)


def flat_face(ents, z):
    """The horizontal face lying at height `z`, as (lo, hi) of its points; None if absent."""
    for fid, f in ents.F.items():
        if abs(float(f.normal[2])) < 0.999:
            continue
        pts = np.vstack(ents.face_points(fid))
        if np.all(np.abs(pts[:, 2] - z) < 1e-6):
            return pts.min(axis=0), pts.max(axis=0)
    return None


def pump_until(predicate, timeout_ms=15000):
    """Spin the event loop until `predicate()` holds or the time is up; returns its value."""
    deadline = time.monotonic() + timeout_ms / 1000.0
    while not predicate() and time.monotonic() < deadline:
        _app.processEvents()
        time.sleep(0.005)
    return predicate()


# ---------------------------------------------------------------- the operation

def test_the_bottom_edges_of_a_cube_are_chamfered():
    doc = Document()
    g = add_cube(doc)
    assert ops.bottom_edges(g.ents) and len(ops.bottom_edges(g.ents)) == 4
    assert ops.elephant_foot(doc, [("g", g.id, g.id)], SIZE) == (1, 0)
    lost = 4 * CUBE * SIZE * SIZE / 2               # four prisms of a triangle of 0.125 mm²
    assert g.ents.to_manifold().volume() == pytest.approx(CUBE ** 3 - lost, abs=0.5)
    assert g.ents.bbox()[0][2] == pytest.approx(0.0), "the part does not stand on the bed"


def test_the_top_face_of_a_cube_is_untouched():
    doc = Document()
    g = add_cube(doc)
    ops.elephant_foot(doc, [("g", g.id, g.id)], SIZE)
    lo, hi = flat_face(g.ents, CUBE)
    assert np.allclose(hi[:2] - lo[:2], [CUBE, CUBE], atol=1e-6)


def test_the_bottom_face_of_a_cube_becomes_smaller():
    doc = Document()
    g = add_cube(doc)
    ops.elephant_foot(doc, [("g", g.id, g.id)], SIZE)
    lo, hi = flat_face(g.ents, 0.0)
    assert np.allclose(hi[:2] - lo[:2], [CUBE - 2 * SIZE, CUBE - 2 * SIZE], atol=1e-6)


def test_the_bottom_rim_of_a_cylinder_is_chamfered():
    doc = Document()
    g = add_cylinder(doc)
    before = g.ents.to_manifold().volume()
    assert len(ops.bottom_edges(g.ents)) == 32, "the rim is one chain of 32 edges"
    assert ops.elephant_foot(doc, [("g", g.id, g.id)], SIZE) == (1, 0)
    after = g.ents.to_manifold().volume()
    assert after < before
    assert before - after == pytest.approx(2 * math.pi * 10.0 * SIZE * SIZE / 2, abs=0.5)
    lo, hi = flat_face(g.ents, 0.0)
    assert np.allclose(hi[:2] - lo[:2], [19.0, 19.0], atol=0.05), "the bottom face did not shrink"
    _lo, hi = flat_face(g.ents, 20.0)
    assert np.allclose(hi[:2] - _lo[:2], [20.0, 20.0], atol=1e-6), "the top face changed"


def test_a_group_keeps_its_name_and_color():
    doc = Document()
    g = add_cube(doc, name="my part", color="#d23b3b")
    ops.elephant_foot(doc, [("g", g.id, g.id)], SIZE)
    assert g.name == "my part" and g.color == "#d23b3b"


def test_a_part_that_stands_on_one_vertex_is_skipped():
    doc = Document()
    g = tilt_about_a_vertex(doc)
    before = g.ents.to_manifold().volume()
    assert ops.bottom_edges(g.ents) == []
    assert ops.elephant_foot(doc, [("g", g.id, g.id)], SIZE) == (0, 1)
    assert g.ents.to_manifold().volume() == pytest.approx(before)


def test_a_locked_group_is_skipped():
    doc = Document()
    g = add_cube(doc)
    g.locked = True
    before = g.ents.to_manifold().volume()
    assert ops.elephant_foot(doc, [("g", g.id, g.id)], SIZE) == (0, 1)
    assert g.ents.to_manifold().volume() == pytest.approx(before)


def test_a_size_that_does_not_fit_leaves_the_group_alone():
    doc = Document()
    g = add_cube(doc, size=1.0)
    assert ops.elephant_foot(doc, [("g", g.id, g.id)], 2.0) == (0, 1)
    assert g.ents.to_manifold().volume() == pytest.approx(1.0)


def test_a_chamfer_that_raises_leaves_the_group_alone(monkeypatch):
    doc = Document()
    g = add_cube(doc)
    before = g.ents.to_manifold().volume()

    def too_large(*args, **kwargs):
        raise ValueError("the size is too large for this volume")

    monkeypatch.setattr(fillet, "apply", too_large)
    assert ops.elephant_foot(doc, [("g", g.id, g.id)], SIZE) == (0, 1)
    assert g.ents.to_manifold().volume() == pytest.approx(before)


def test_only_the_groups_of_the_items_are_counted_once():
    doc = Document()
    g = add_cube(doc)
    items = [("g", g.id, g.id), ("g", g.id, g.id), ("f", 0, 1)]
    assert ops.elephant_foot(doc, items, SIZE) == (1, 0)


# ---------------------------------------------------------------- the window

def test_the_action_chamfers_the_selected_group(win):
    g = add_cube(win.doc)
    win.doc.set_selection([("g", g.id, g.id)])
    win.a_elephant.trigger()
    assert pump_until(lambda: g.ents.to_manifold().volume() < CUBE ** 3 - 1), "nothing was chamfered"
    assert g.ents.to_manifold().volume() == pytest.approx(CUBE ** 3 - 10, abs=0.5)
    assert pump_until(lambda: "Chamfered the bottom edges of 1 part." in win.hint.text())


def test_the_action_takes_every_visible_group_when_nothing_is_selected(win):
    a, b = add_cube(win.doc, name="a"), add_cube(win.doc, name="b")
    hidden = add_cube(win.doc, name="hidden")
    hidden.visible = False
    before = hidden.ents.to_manifold().volume()
    win.a_elephant.trigger()
    ready = lambda: a.ents.to_manifold().volume() < CUBE ** 3 - 1 and \
        b.ents.to_manifold().volume() < CUBE ** 3 - 1
    assert pump_until(ready), "not every group was chamfered"
    assert hidden.ents.to_manifold().volume() == pytest.approx(before)


def test_the_action_counts_the_parts_it_skipped(win):
    g, tilted = add_cube(win.doc, name="a"), tilt_about_a_vertex(win.doc, name="tilted")
    win.doc.set_selection([("g", g.id, g.id), ("g", tilted.id, tilted.id)])
    win.a_elephant.trigger()
    ready = lambda: "1 part was skipped" in win.hint.text()
    assert pump_until(ready), win.hint.text()
    assert "Chamfered the bottom edges of 1 part." in win.hint.text()


def test_a_part_without_bottom_edges_leaves_no_undo_step(win):
    tilted = tilt_about_a_vertex(win.doc)
    steps = len(win.doc.undo_stack)
    win.doc.set_selection([("g", tilted.id, tilted.id)])
    win.a_elephant.trigger()
    assert pump_until(lambda: "None of these parts" in win.hint.text()), win.hint.text()
    assert len(win.doc.undo_stack) == steps, "nothing changed: there is nothing to undo"


def test_the_action_is_one_undo_step(win):
    g = add_cube(win.doc)
    win.doc.set_selection([("g", g.id, g.id)])
    win.a_elephant.trigger()
    assert pump_until(lambda: g.ents.to_manifold().volume() < CUBE ** 3 - 1)
    win.doc.undo()
    assert g.ents.to_manifold().volume() == pytest.approx(CUBE ** 3), "one undo did not restore the part"


def test_the_action_asks_the_last_size_and_remembers_the_new_one(win):
    win.settings.elephant_foot_size = 0.4
    g = add_cube(win.doc)
    win.doc.set_selection([("g", g.id, g.id)])
    win.a_elephant.trigger()
    assert pump_until(lambda: g.ents.to_manifold().volume() < CUBE ** 3 - 1)
    assert win.ask.asked[0][2] == pytest.approx(0.4), "the question did not start at the last size"
    assert win.settings.elephant_foot_size == pytest.approx(SIZE)

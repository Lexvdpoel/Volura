"""Orient for Printing: turn a part so it stands on the face that needs the least support.

`ops.orient_for_printing()` looks for the way a part prints with the least support, like the
auto-orient of a slicer. The faces a part can stand on are the faces of its convex hull,
merged within a degree and with the largest 40 left over; for each of them the part is turned
until that face points straight down and the area of the triangles that `ops.overhang_mask()`
marks is added. The least of that area wins; a tie of 1% goes to the largest face, because
that stands the most steadily on the bed, and a tie that is left to the way the part already
stands. These tests (Qt offscreen) check:

* a cube of 20 mm needs no support whichever way it stands, so it is not turned;
* a cone of r = 10 and h = 20 that stands on its tip is turned onto its base: 0 mm² of
  support, the lowest vertex on the bed and a base of pi · 10² mm² underneath;
* a box that already lies on its largest face does not move;
* a mushroom (a stem of 10 × 10 × 20 under a cap of 30 × 30 × 5) goes from 800 mm² of
  overhang on its stem to 0 mm² on its cap;
* the overhang limit of the settings is the limit that is used;
* a locked group and a group that is not closed are skipped, and one undo brings the old
  places back;
* the menu action: after Mouse Ears in Print Tools, on the selected groups or on all visible
  ones, in the background, with the counts in the status bar.

Run with:  python -m pytest tests/test_orient.py
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

from volura import app as app_module, ops  # noqa: E402
from volura.brep import Entities  # noqa: E402
from volura.geom import mesh_arrays, rotation_matrix  # noqa: E402
from volura.model import Document  # noqa: E402
from volura.vector2d import polygon_area  # noqa: E402

_app = QApplication.instance() or QApplication(sys.argv[:1])

LIMIT = 45.0
DOWN = np.array([0.0, 0.0, -1.0])
ACTION = "Orient for Printing"


class FakeBox:
    """Stands in for QMessageBox: remembers what the window asked."""

    def __init__(self):
        self.asked = []

    def warning(self, *args, **kwargs):
        self.asked.append(args[1:])


@pytest.fixture()
def win(monkeypatch, tmp_path):
    """A window that does not write error logs and logs into the folder of the test."""
    monkeypatch.setattr(app_module.errors, "log_dir", lambda: str(tmp_path))
    monkeypatch.setattr(app_module, "autosave_path", lambda: str(tmp_path / "autosave.volura"))
    monkeypatch.setattr(app_module, "QMessageBox", FakeBox())
    w = app_module.MainWindow()
    w.doc.dirty = False
    yield w
    w.doc.dirty = False
    w.close()


def box(doc, size=(30.0, 30.0, 10.0), name="box"):
    """A closed box of `size` mm with its corner in the origin, as its own group."""
    return ops.group_from_manifold(doc, mf.Manifold.cube(size), "#cccccc", name)


def cone(doc, name="cone"):
    """A cone of r = 10 and h = 20 that stands on its tip: the base is the face on top."""
    g = ops.group_from_manifold(doc, mf.Manifold.cylinder(20.0, 10.0, 0.0, 64), "#cccccc", name)
    g.ents.transform(rotation_matrix(np.array([1.0, 0.0, 0.0]), math.pi), np.zeros(3))
    return g


def mushroom(doc, name="mushroom"):
    """A stem of 10 × 10 × 20 under a cap of 30 × 30 × 5, centred, as its own group."""
    stem = mf.Manifold.cube((10.0, 10.0, 20.0)).translate((10.0, 10.0, 0.0))
    cap = mf.Manifold.cube((30.0, 30.0, 5.0)).translate((0.0, 0.0, 20.0))
    return ops.group_from_manifold(doc, ops.union_all([stem, cap]), "#cccccc", name)


def open_box(doc, name="open"):
    """A box of 20 mm with one face removed, so it is no closed volume."""
    ents = Entities.from_manifold(mf.Manifold.cube((20.0, 20.0, 20.0)))
    ents.remove([next(iter(ents.F))], [])
    return doc.add_group(ents, "#cccccc", name)


def vertices(g):
    """The vertices of a group in a fixed order, so a move can be seen."""
    P = np.array(list(g.ents.V.values()), dtype=float)
    return P[np.lexsort(P.T)]


def bottom_face(g, z=0.0):
    """The area of the largest flat face of `g` that lies at height `z`, 0.0 when there is none."""
    area = 0.0
    for face in g.ents.F.values():
        if abs(float(face.normal[2])) < 0.999:
            continue
        for loop in face.loops:
            P = np.array([g.ents.V[v] for v in loop], dtype=float)
            if np.all(np.abs(P[:, 2] - z) <= 1e-6):
                area = max(area, abs(polygon_area(P[:, :2])))
    return area


def pump_until(predicate, timeout_ms=15000):
    """Spin the event loop until `predicate()` holds or the time is up; returns its value."""
    deadline = time.monotonic() + timeout_ms / 1000.0
    while not predicate() and time.monotonic() < deadline:
        _app.processEvents()
        time.sleep(0.005)
    return predicate()


def menu_labels(menu):
    return [a.text().replace("&", "") for a in menu.actions() if not a.isSeparator()]


# ---------------------------------------------------------------- the search for the best way

def test_a_cube_needs_no_support_whichever_way_it_stands():
    V, T = mesh_arrays(mf.Manifold.cube((20.0, 20.0, 20.0)))
    d, score = ops.best_down_direction(V, T)
    assert score == pytest.approx(0.0)
    assert ops.is_down(d), "a cube that prints without support is turned for nothing"


def test_the_base_of_a_cone_is_the_way_it_has_to_point_down():
    V, T = mesh_arrays(mf.Manifold.cylinder(20.0, 10.0, 0.0, 64))
    V = V @ rotation_matrix(np.array([1.0, 0.0, 0.0]), math.pi).T      # standing on its tip
    d, score = ops.best_down_direction(V, T)
    assert score == pytest.approx(0.0), "the base of a cone needs no support"
    assert np.allclose(d, [0.0, 0.0, 1.0], atol=1e-9), "the cone was not put on its base"


def test_the_overhang_limit_is_the_limit_of_the_search():
    """A cone on its side hangs over more than 45° but less than 89° from the vertical."""
    V, T = mesh_arrays(mf.Manifold.cylinder(20.0, 10.0, 0.0, 64))
    side = np.array([1.0, 0.0, 0.0])
    assert ops.overhang_area(V, T, side, 45.0) == pytest.approx(153.6, abs=0.5)
    assert ops.overhang_area(V, T, side, 89.0) == pytest.approx(0.0)


def test_the_cap_of_a_mushroom_needs_far_less_support_than_the_stem():
    V, T = mesh_arrays(mushroom(Document()).ents.to_manifold())
    assert ops.overhang_area(V, T, DOWN, LIMIT) == pytest.approx(800.0), "the underside of the cap"
    assert ops.overhang_area(V, T, -DOWN, LIMIT) == pytest.approx(0.0), "a cap that lies down"
    d, score = ops.best_down_direction(V, T)
    assert np.allclose(d, [0.0, 0.0, 1.0], atol=1e-9), "the mushroom was not put on its cap"
    assert score < ops.overhang_area(V, T, DOWN, LIMIT)


# ---------------------------------------------------------------- the operation

def test_a_cone_that_stands_on_its_tip_is_turned_onto_its_base():
    doc = Document()
    g = cone(doc)
    volume = g.ents.to_manifold().volume()
    assert ops.orient_for_printing(doc, [("g", g.id, g.id)], LIMIT) == (1, 0, 0)
    lo, hi = g.ents.bbox()
    assert lo[2] == pytest.approx(0.0), "the cone does not stand on the bed"
    assert bottom_face(g) == pytest.approx(math.pi * 10.0 ** 2, abs=1.0), "not the base underneath"
    assert g.ents.to_manifold().volume() == pytest.approx(volume)


def test_a_box_that_lies_on_its_largest_face_does_not_move():
    doc = Document()
    g = box(doc)
    before = vertices(g)
    assert ops.orient_for_printing(doc, [("g", g.id, g.id)], LIMIT) == (0, 1, 0)
    assert np.allclose(vertices(g), before), "a box that already lies flat was moved"


def test_a_mushroom_ends_with_its_cap_on_the_bed():
    doc = Document()
    g = mushroom(doc)
    assert ops.orient_for_printing(doc, [("g", g.id, g.id)], LIMIT) == (1, 0, 0)
    lo, hi = g.ents.bbox()
    assert lo[2] == pytest.approx(0.0), "the cap does not lie on the bed"
    assert bottom_face(g) == pytest.approx(30.0 * 30.0), "the part does not stand on the cap"
    assert hi[2] - lo[2] == pytest.approx(25.0)


def test_a_group_that_is_only_dropped_is_counted_as_unchanged():
    doc = Document()
    g = box(doc)
    g.ents.transform(np.eye(3), np.array([0.0, 0.0, 40.0]))       # a floating plate
    assert ops.orient_for_printing(doc, [("g", g.id, g.id)], LIMIT) == (0, 1, 0)
    assert g.ents.bbox()[0][2] == pytest.approx(0.0), "the plate was not dropped on the bed"


def test_a_locked_group_is_skipped():
    doc = Document()
    g = cone(doc)
    g.locked = True
    before = vertices(g)
    assert ops.orient_for_printing(doc, [("g", g.id, g.id)], LIMIT) == (0, 0, 1)
    assert np.allclose(vertices(g), before), "a locked group was turned"


def test_a_group_that_is_not_closed_is_skipped():
    doc = Document()
    g = open_box(doc)
    before = vertices(g)
    assert ops.orient_for_printing(doc, [("g", g.id, g.id)], LIMIT) == (0, 0, 1)
    assert np.allclose(vertices(g), before), "a group that is no volume was turned"


def test_every_group_of_the_items_is_counted_once():
    doc = Document()
    g, other = cone(doc), box(doc, name="box")
    items = [("g", g.id, g.id), ("g", g.id, g.id), ("g", other.id, other.id)]
    assert ops.orient_for_printing(doc, items, LIMIT) == (1, 1, 0)


def test_one_undo_brings_the_old_places_back():
    doc = Document()
    g = cone(doc)
    before = vertices(g)
    doc.checkpoint()
    ops.orient_for_printing(doc, [("g", g.id, g.id)], LIMIT)
    assert not np.allclose(vertices(g), before), "the cone did not turn"
    doc.undo()
    assert np.allclose(vertices(g), before), "one undo did not restore the cone"


# ---------------------------------------------------------------- the window

def test_the_action_stands_after_the_print_check_in_3d_printing(win):
    labels = menu_labels(win.print_tools_menu)
    assert labels.index("Print &Check".replace("&", "")) + 1 == labels.index(ACTION), labels


def test_the_action_turns_the_selected_cone(win):
    g = cone(win.doc)
    win.doc.set_selection([("g", g.id, g.id)])
    win.a_orient_print.trigger()
    ready = lambda: "Turned 1 parts to need the least support." in win.hint.text()
    assert pump_until(ready), win.hint.text()
    lo, hi = g.ents.bbox()
    assert lo[2] == pytest.approx(0.0) and hi[2] == pytest.approx(20.0), "the cone did not turn"
    assert bottom_face(g) == pytest.approx(math.pi * 10.0 ** 2, abs=1.0)


def test_the_action_takes_every_visible_group_when_nothing_is_selected(win):
    a, b = cone(win.doc, "a"), mushroom(win.doc, "b")
    hidden = cone(win.doc, "hidden")
    hidden.visible = False
    before = vertices(hidden)
    ready = lambda: "Turned 2 parts to need the least support." in win.hint.text()
    win.a_orient_print.trigger()
    assert pump_until(ready), win.hint.text()
    assert a.ents.bbox()[1][2] == pytest.approx(20.0), "the cone did not turn"
    assert b.ents.bbox()[0][2] == pytest.approx(0.0) and bottom_face(b) == pytest.approx(900.0)
    assert np.allclose(vertices(hidden), before), "a hidden group was turned"


def test_the_action_counts_what_it_left_alone(win):
    g, plate, locked = cone(win.doc, "cone"), box(win.doc, name="plate"), box(win.doc, name="locked")
    locked.locked = True
    win.doc.set_selection([("g", g.id, g.id), ("g", plate.id, plate.id), ("g", locked.id, locked.id)])
    win.a_orient_print.trigger()
    ready = lambda: "1 parts were skipped" in win.hint.text()
    assert pump_until(ready), win.hint.text()
    text = win.hint.text()
    assert "Turned 1 parts to need the least support." in text
    assert "1 parts were already the right way round." in text
    assert "locked or not a closed volume" in text


def test_the_action_passes_the_overhang_limit_of_the_settings(win, monkeypatch):
    seen = []
    real = ops.orient_direction

    def spy(g, limit_deg):
        seen.append(limit_deg)
        return real(g, limit_deg)

    monkeypatch.setattr(ops, "orient_direction", spy)
    g = cone(win.doc)
    win.settings.overhang_limit = 89
    win.doc.set_selection([("g", g.id, g.id)])
    win.a_orient_print.trigger()
    ready = lambda: "Turned 1 parts to need the least support." in win.hint.text()
    assert pump_until(ready), win.hint.text()
    assert seen == [89.0], "the search did not use the limit of the settings"


def test_groups_that_cannot_be_turned_leave_no_undo_step(win):
    g = open_box(win.doc)
    g.locked = True
    steps = len(win.doc.undo_stack)
    win.doc.set_selection([("g", g.id, g.id)])
    win.a_orient_print.trigger()
    ready = lambda: "None of these parts can be turned" in win.hint.text()
    assert pump_until(ready), win.hint.text()
    assert len(win.doc.undo_stack) == steps, "nothing changed: there is nothing to undo"


def test_the_action_is_one_undo_step(win):
    g = cone(win.doc)
    before = vertices(g)
    win.doc.set_selection([("g", g.id, g.id)])
    win.a_orient_print.trigger()
    assert pump_until(lambda: "Turned 1 parts" in win.hint.text()), win.hint.text()
    win.doc.undo()
    assert np.allclose(vertices(g), before), "one undo did not bring the old place back"

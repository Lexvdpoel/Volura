"""Lay Flat on This Face: turn a part so the chosen face lies on the print bed.

`ops.lay_flat()` turns a group about the centre of its own bounding box until the outward
normal of the chosen face points straight down - the shortest way round, so a face that
already lies down turns nothing and a face that points up turns the part over - and drops
the group on the bed afterwards, like "place on face" of a slicer. These tests (Qt
offscreen) check:

* a box of 10 × 20 × 30 that stands on its 10 × 20 face and is laid flat on its 20 × 30
  side face is 10 mm high, stands on the bed and keeps its volume;
* the top face (pointing up) turns the box over: still 30 mm high, now standing on the bed;
* a face that already lies down and a locked group change nothing;
* the sloped face of a wedge comes to lie horizontal on the bed, with all its vertices on
  z = 0;
* one undo brings the old vertices back;
* the right-click menu of a face that lies in a group offers the action, loose geometry
  does not, and the action is one undo step.
"""
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import manifold3d as mf  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from volura import app as app_module, ops  # noqa: E402
from volura.geom import Plane, manifold_from_mesh  # noqa: E402
from volura.model import ROOT, Document  # noqa: E402

_app = QApplication.instance() or QApplication(sys.argv[:1])

ACTION = "Lay Flat on This Face"


@pytest.fixture()
def win():
    from volura.app import MainWindow
    w = MainWindow()
    yield w
    w.doc.dirty = False
    w.close()


def box(doc, size=(10.0, 20.0, 30.0)):
    """A closed box of `size` mm with its corner in the origin, as its own group."""
    return ops.group_from_manifold(doc, mf.Manifold.cube(size), "#cccccc", "box")


def wedge(doc):
    """A wedge 20 long, 10 wide and 10 high with a sloped face, as its own group."""
    V = [(0, 0, 0), (20, 0, 0), (20, 10, 0), (0, 10, 0), (0, 0, 10), (0, 10, 10)]
    T = [(0, 2, 1), (0, 3, 2),      # bottom
         (0, 5, 3), (0, 4, 5),      # the vertical back
         (0, 1, 4), (3, 5, 2),      # the two sides
         (1, 2, 5), (1, 5, 4)]      # the sloped face
    m = manifold_from_mesh(np.array(V, dtype=float), np.array(T, dtype=np.int64))
    return ops.group_from_manifold(doc, m, "#cccccc", "wedge")


def face_with_normal(g, n):
    """The id of the face of `g` whose outward normal points along `n`."""
    n = np.asarray(n, dtype=float)
    for fid, face in g.ents.F.items():
        if np.allclose(np.asarray(face.normal, dtype=float), n, atol=1e-6):
            return fid
    raise AssertionError(f"no face pointing along {n}")


def sloped_face(g):
    """The id of the one face of the wedge that is neither flat nor vertical."""
    for fid, face in g.ents.F.items():
        if face.normal[0] > 0.3 and face.normal[2] > 0.3:
            return fid
    raise AssertionError("the wedge has no sloped face")


def face_points(g, fid):
    return np.concatenate(g.ents.face_points(fid))


def vertices(g):
    """The vertices of a group in a fixed order, so undo can be compared."""
    P = np.array(list(g.ents.V.values()), dtype=float)
    return P[np.lexsort(P.T)]


def menu_labels(menu):
    return [a.text().replace("&", "") for a in menu.actions() if not a.isSeparator()]


def lay_flat_action(win):
    menu = win.build_context_menu()
    return next((a for a in menu.actions() if a.text().replace("&", "") == ACTION), None)


# ---------------------------------------------------------------- the operation

def test_a_box_is_laid_flat_on_its_side_face():
    doc = Document()
    g = box(doc)
    volume = g.ents.to_manifold().volume()
    ops.lay_flat(doc, g.id, face_with_normal(g, (1.0, 0.0, 0.0)))
    lo, hi = g.ents.bbox()
    assert hi[2] - lo[2] == pytest.approx(10.0), "the box did not end up on its 20 × 30 face"
    assert lo[2] == pytest.approx(0.0), "the box does not stand on the bed"
    assert g.ents.to_manifold().volume() == pytest.approx(volume)


def test_the_face_that_is_laid_flat_points_down():
    doc = Document()
    g = box(doc)
    fid = face_with_normal(g, (0.0, 1.0, 0.0))
    ops.lay_flat(doc, g.id, fid)
    assert np.allclose(g.ents.F[fid].normal, (0.0, 0.0, -1.0))
    assert np.allclose(face_points(g, fid)[:, 2], 0.0, atol=1e-6)


def test_the_top_face_turns_the_box_over():
    doc = Document()
    g = box(doc)
    top = face_with_normal(g, (0.0, 0.0, 1.0))
    ops.lay_flat(doc, g.id, top)
    lo, hi = g.ents.bbox()
    assert hi[2] - lo[2] == pytest.approx(30.0), "the box is no longer 30 mm high"
    assert lo[2] == pytest.approx(0.0), "the box does not stand on the bed"
    assert np.allclose(g.ents.F[top].normal, (0.0, 0.0, -1.0)), "the box did not turn over"


def test_a_face_that_already_lies_down_changes_nothing():
    doc = Document()
    g = box(doc)
    before = vertices(g)
    ops.lay_flat(doc, g.id, face_with_normal(g, (0.0, 0.0, -1.0)))
    assert np.allclose(vertices(g), before), "a part that already lies flat was moved"


def test_a_locked_group_stays_where_it_is():
    doc = Document()
    g = box(doc)
    g.locked = True
    before = vertices(g)
    ops.lay_flat(doc, g.id, face_with_normal(g, (1.0, 0.0, 0.0)))
    assert np.allclose(vertices(g), before), "a locked group was turned"


def test_a_sloped_face_comes_to_lie_on_the_bed():
    doc = Document()
    g = wedge(doc)
    fid = sloped_face(g)
    volume = g.ents.to_manifold().volume()
    ops.lay_flat(doc, g.id, fid)
    assert np.allclose(face_points(g, fid)[:, 2], 0.0, atol=1e-6), "the sloped face is not flat"
    assert g.ents.bbox()[0][2] == pytest.approx(0.0), "the wedge does not stand on the bed"
    assert g.ents.to_manifold().volume() == pytest.approx(volume)


def test_undo_brings_the_old_vertices_back():
    doc = Document()
    g = box(doc)
    before = vertices(g)
    doc.checkpoint()
    ops.lay_flat(doc, g.id, face_with_normal(g, (1.0, 0.0, 0.0)))
    assert not np.allclose(vertices(g), before), "the box did not turn"
    doc.undo()
    assert np.allclose(vertices(g), before), "one undo did not restore the old vertices"


# ---------------------------------------------------------------- the window

def test_a_face_of_a_group_offers_laying_it_flat(win):
    g = box(win.doc)
    win.doc.set_selection([("g", g.id, g.id)])
    win._context_face = (g.id, face_with_normal(g, (1.0, 0.0, 0.0)))
    assert ACTION in menu_labels(win.build_context_menu())


def test_a_face_selected_in_an_open_group_offers_laying_it_flat(win):
    g = box(win.doc)
    fid = face_with_normal(g, (0.0, 1.0, 0.0))
    win.doc.set_selection([("f", g.id, fid)])
    assert ACTION in menu_labels(win.build_context_menu())


def test_loose_geometry_does_not_offer_laying_a_face_flat(win):
    loop = np.array([[0, 0], [40, 0], [40, 30], [0, 30]], dtype=float)
    created = win.doc.root.add_shape(Plane((0, 0, 0), (0, 0, 1)), [loop])
    assert created, "rectangle gave no face"
    win.doc.set_selection([("f", ROOT, created[0])])
    win._context_face = (ROOT, created[0])
    assert ACTION not in menu_labels(win.build_context_menu())


def test_the_menu_action_lays_the_box_flat_in_one_undo_step(win):
    g = box(win.doc)
    before = vertices(g)
    win.doc.set_selection([("g", g.id, g.id)])
    win._context_face = (g.id, face_with_normal(g, (1.0, 0.0, 0.0)))
    action = lay_flat_action(win)
    assert action is not None, "the right-click menu has no action to lay the face flat"
    action.trigger()
    lo, hi = g.ents.bbox()
    assert hi[2] - lo[2] == pytest.approx(10.0)
    assert lo[2] == pytest.approx(0.0)
    win.doc.undo()
    assert np.allclose(vertices(g), before), "one undo did not restore the box"


def test_the_menu_action_of_a_locked_group_warns_and_changes_nothing(win):
    g = box(win.doc)
    g.locked = True
    before = vertices(g)
    steps = len(win.doc.undo_stack)
    win.doc.set_selection([("g", g.id, g.id)])
    win._context_face = (g.id, face_with_normal(g, (1.0, 0.0, 0.0)))
    lay_flat_action(win).trigger()
    assert np.allclose(vertices(g), before), "a locked group was turned"
    assert len(win.doc.undo_stack) == steps, "nothing to undo: the model did not change"

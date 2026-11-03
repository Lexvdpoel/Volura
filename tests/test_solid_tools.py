"""The SketchUp 2017 volume tools: Buitenschil, Bijsnijden and Vlakken snijden.

The box is the same 40 x 30 x 10 mm box as box() in tests/test_brep.py (12000 mm3),
so the expected volumes are the ones SketchUp 2017 gives for the same shapes.
"""
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import manifold3d as mf  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from test_brep import box, edge_at, face_at, rect  # noqa: E402
from volura import ops  # noqa: E402
from volura.brep import Entities  # noqa: E402
from volura.geom import Plane  # noqa: E402
from volura.model import ROOT, Document  # noqa: E402

_app = QApplication.instance() or QApplication(sys.argv[:1])


def clean(text):
    return text.replace("&", "").replace("…", "").replace("...", "").strip()


def volume(ents):
    m = ents.to_manifold()
    assert m.status().name == "NoError", m.status()
    return round(m.volume(), 2)


def moved_box(offset):
    """The same box, moved (so two of them overlap for half of their footprint)."""
    e = box()
    e.transform(np.eye(3), np.asarray(offset, dtype=float))
    return e


def plate():
    """A plate of 60 x 6 x 25 mm straight through the box (y 12..18, z -5..20)."""
    e = Entities()
    rect(e, -10, 12, 50, 18, 20)
    e.pushpull(face_at(e, 20, 15, 20), -25)
    return e


def two_boxes(doc):
    a = doc.add_group(box(), "#f2f2f2", "A")
    b = doc.add_group(moved_box([20, 15, 0]), "#2f6fd6", "B")
    return a, b


# ------------------------------------------------------------------ Buitenschil

def test_outer_shell_of_two_overlapping_boxes_is_the_union():
    doc = Document()
    a, b = two_boxes(doc)
    assert (volume(a.ents), volume(b.ents)) == (12000.0, 12000.0)
    res = ops.outer_shell(doc, [a, b])
    assert volume(res.ents) == 21000.0            # 12000 + 12000 - 3000 overlap
    assert len(doc.groups) == 1


def test_outer_shell_keeps_the_name_and_color_of_the_first_group():
    doc = Document()
    a, b = two_boxes(doc)
    res = ops.outer_shell(doc, [a, b])
    assert res is a and res.name == "A" and res.color == "#f2f2f2"


def test_outer_shell_of_one_box_is_that_box():
    doc = Document()
    a = doc.add_group(box(), "#f2f2f2", "A")
    res = ops.outer_shell(doc, [a])
    assert volume(res.ents) == 12000.0


def test_outer_shell_removes_a_fully_enclosed_cavity():
    doc = Document()
    hollow = mf.Manifold.cube((10, 10, 10)) - mf.Manifold.cube((6, 6, 6)).translate((2, 2, 2))
    g = ops.group_from_manifold(doc, hollow, "#f2f2f2", "hol")
    assert volume(g.ents) == 784.0
    res = ops.outer_shell(doc, [g])
    assert volume(res.ents) == 1000.0             # only the outside is left


def test_outer_shell_drops_a_part_that_lies_inside_another():
    doc = Document()
    outer = ops.group_from_manifold(doc, mf.Manifold.cube((10, 10, 10)), "#f2f2f2", "bol")
    inner = ops.group_from_manifold(doc, mf.Manifold.cube((2, 2, 2)).translate((4, 4, 4)),
                                    "#d23b3b", "bin")
    res = ops.outer_shell(doc, [outer, inner])
    assert volume(res.ents) == 1000.0
    assert len(doc.groups) == 1


def test_outer_shell_needs_a_closed_volume():
    doc = Document()
    e = Entities()
    rect(e, 0, 0, 40, 30)
    g = doc.add_group(e, "#f2f2f2", "vlak")
    with pytest.raises(ValueError):
        ops.outer_shell(doc, [g, doc.add_group(box(), "#2f6fd6", "B")])


# ------------------------------------------------------------------- Bijsnijden

def test_trim_cuts_the_second_group_with_the_first():
    doc = Document()
    a, b = two_boxes(doc)
    res = ops.trim(doc, a, b)
    assert res is b
    assert volume(b.ents) == 9000.0               # 12000 - 3000 overlap
    assert len(doc.groups) == 2


def test_trim_leaves_the_first_group_unchanged():
    doc = Document()
    a, b = two_boxes(doc)
    before = volume(a.ents)
    ops.trim(doc, a, b)
    assert volume(a.ents) == before == 12000.0
    assert a.name == "A" and a.color == "#f2f2f2"


def test_trim_of_a_group_inside_another_removes_it():
    doc = Document()
    outer = ops.group_from_manifold(doc, mf.Manifold.cube((10, 10, 10)), "#f2f2f2", "bol")
    inner = ops.group_from_manifold(doc, mf.Manifold.cube((2, 2, 2)).translate((4, 4, 4)),
                                    "#d23b3b", "bin")
    assert ops.trim(doc, outer, inner) is None
    assert [g.name for g in doc.groups] == ["bol"]


def test_trim_needs_two_closed_volumes():
    doc = Document()
    a, _ = two_boxes(doc)
    e = Entities()
    rect(e, 0, 0, 40, 30)
    flat = doc.add_group(e, "#3fa34d", "vlak")
    with pytest.raises(ValueError):
        ops.trim(doc, a, flat)


# ------------------------------------------------------------- Vlakken snijden

def box_and_plate(doc):
    doc.root = box()
    g = doc.add_group(plate(), "#2f6fd6", "plaat")
    return g


def test_intersect_faces_with_model_splits_the_box():
    doc = Document()
    g = box_and_plate(doc)
    items_a = [("g", g.id, g.id)]
    items_b = [it for it in ops.whole_items(doc) if it not in set(items_a)]
    assert ops.intersect_faces(doc, items_a, items_b)
    assert len(doc.root.F) > 6                    # the sides of the box are split
    assert len(doc.root.E) > 12                   # the cut lines are edges now
    assert volume(doc.groups[0].ents) == 9000.0   # the plate itself is untouched


def test_intersect_faces_with_model_adds_edges_on_the_cut_lines():
    doc = Document()
    g = box_and_plate(doc)
    ops.intersect_faces(doc, [("g", g.id, g.id)],
                        [it for it in ops.whole_items(doc) if it != ("g", g.id, g.id)])
    edge_at(doc.root, 0, 12, 5)                   # the plate cuts the left side
    edge_at(doc.root, 40, 12, 5)                  # and the right side
    assert len(doc.root.faces_in_plane(Plane((0, 0, 0), (1, 0, 0)))) == 3


def test_intersect_faces_with_selection_only_cuts_selected_items():
    doc = Document()
    g = box_and_plate(doc)
    sel = [("g", g.id, g.id)]
    assert ops.intersect_faces(doc, sel, sel) == []
    assert (len(doc.root.F), len(doc.root.E)) == (6, 12)


def test_intersect_faces_with_selection_cuts_two_selected_groups():
    doc = Document()
    a, b = two_boxes(doc)
    sel = [("g", a.id, a.id), ("g", b.id, b.id)]
    assert ops.intersect_faces(doc, sel, sel)
    assert doc.root.E                             # the cut lines are loose edges
    assert (len(a.ents.F), len(b.ents.F)) == (6, 6)


def test_intersect_faces_does_not_cut_parallel_faces():
    doc = Document()
    a = doc.add_group(box(), "#f2f2f2", "A")
    b = doc.add_group(moved_box([0, 0, 20]), "#2f6fd6", "B")
    sel = [("g", a.id, a.id), ("g", b.id, b.id)]
    assert ops.intersect_faces(doc, sel, sel) == []


# ---------------------------------------------------------------- in the window

@pytest.fixture()
def win():
    from volura.app import MainWindow
    w = MainWindow()
    yield w
    w.doc.dirty = False
    w.close()


def test_volume_tools_are_no_longer_placeholders(win):
    for a, text in ((win.a_shell, "Buitenschil"), (win.a_trim, "Bijsnijden"),
                    (win.a_cut_face_model, "Met model"), (win.a_cut_face_sel, "Met selectie")):
        assert clean(a.text()) == text
        assert a.isEnabled(), f"{text} is nog een placeholder"


def test_outer_shell_action_is_one_undo_step(win):
    a, b = two_boxes(win.doc)
    win.doc.set_selection([("g", a.id, a.id), ("g", b.id, b.id)])
    win.outer_shell()
    assert len(win.doc.groups) == 1 and volume(win.doc.groups[0].ents) == 21000.0
    win.undo()
    assert len(win.doc.groups) == 2
    assert sorted(volume(g.ents) for g in win.doc.groups) == [12000.0, 12000.0]


def test_trim_action_cuts_the_second_selected_group(win):
    a, b = two_boxes(win.doc)
    win.doc.set_selection([("g", a.id, a.id), ("g", b.id, b.id)])
    win.trim_solids()
    assert volume(a.ents) == 12000.0 and volume(b.ents) == 9000.0
    win.undo()
    assert volume(win.doc.group(b.id).ents) == 12000.0


def test_trim_action_follows_the_selection_order(win):
    a, b = two_boxes(win.doc)
    win.doc.set_selection([("g", b.id, b.id), ("g", a.id, a.id)])
    win.trim_solids()
    assert volume(b.ents) == 12000.0 and volume(a.ents) == 9000.0


def test_intersect_faces_action_splits_the_box(win):
    g = box_and_plate(win.doc)
    win.doc.set_selection([("g", g.id, g.id)])
    win.intersect_faces(True)
    assert len(win.doc.root.F) > 6
    win.undo()
    assert (len(win.doc.root.F), len(win.doc.root.E)) == (6, 12)


def test_intersect_faces_action_with_selection(win):
    a, b = two_boxes(win.doc)
    win.doc.set_selection([("g", a.id, a.id), ("g", b.id, b.id)])
    win.intersect_faces(False)
    assert win.doc.root.E                         # the cut lines are loose edges
    assert (len(a.ents.F), len(b.ents.F)) == (6, 6)
    win.undo()
    assert not win.doc.root.E


def test_intersect_faces_action_without_cut_lines(win):
    """Groepen die elkaar niet raken: er is niets te snijden en er blijft geen undo-stap."""
    a = win.doc.add_group(box(), "#f2f2f2", "A")
    win.doc.add_group(moved_box([0, 0, 30]), "#2f6fd6", "B")
    win.doc.set_selection([("g", a.id, a.id), ("g", win.doc.groups[1].id, win.doc.groups[1].id)])
    steps = len(win.doc.undo_stack)
    win.intersect_faces(True)
    assert len(win.doc.undo_stack) == steps
    assert all(len(g.ents.F) == 6 and len(g.ents.E) == 12 for g in win.doc.groups)
    assert not win.doc.root.E


def test_intersect_faces_is_in_the_context_menu(win):
    fid = win.doc.root.add_shape(Plane((0, 0, 0), (0, 0, 1)),
                                 [np.array([[0, 0], [40, 0], [40, 30], [0, 30]], dtype=float)])[0]
    win.doc.set_selection([("f", ROOT, fid)])
    menu = win.build_context_menu()
    cut = next(a.menu() for a in menu.actions() if clean(a.text()) == "Vlakken snijden")
    assert [clean(a.text()) for a in cut.actions()] == ["Met model", "Met selectie"]
    assert all(a.isEnabled() for a in cut.actions()), "Vlakken snijden moet werken"


def test_solid_tools_are_in_the_group_context_menu(win):
    a, b = two_boxes(win.doc)
    win.doc.set_selection([("g", a.id, a.id), ("g", b.id, b.id)])
    items = [clean(a.text()) for a in win.build_context_menu().actions() if not a.isSeparator()]
    assert "Buitenschil" in items and "Bijsnijden" in items

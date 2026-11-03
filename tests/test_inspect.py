"""Printcontrole (zoals de extensie "Solid Inspector"): problemen vinden en repareren.

De geometrie-hulpjes `box`, `rect` en `face_at` zijn overgenomen uit tests/test_brep.py.
"""
import os
import sys

import numpy as np
import manifold3d as mf

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from volura import ops                              # noqa: E402
from volura.brep import Entities                    # noqa: E402
from volura.geom import Plane                       # noqa: E402
from volura.model import ROOT, Document             # noqa: E402


def P(x, y, z=0):
    return np.array([x, y, z], dtype=float)


def rect(ents, x0, y0, x1, y1, z=0):
    pl = Plane((0, 0, z), (0, 0, 1))
    hit = [f for f in ents.faces_in_plane(pl)]
    if hit:
        pl = ents.face_plane(hit[0])
        pl = Plane((0, 0, z), pl.n)
    loop = pl.to_local(np.array([P(x0, y0, z), P(x1, y0, z), P(x1, y1, z), P(x0, y1, z)]))
    return ents.add_shape(pl, [loop])


def face_at(ents, x, y, z):
    p = P(x, y, z)
    from volura.geom import point_in_polys
    for f in ents.F:
        pl = ents.face_plane(f)
        if abs(pl.distance(p)) > 1e-6:
            continue
        polys = [pl.to_local(L) for L in ents.face_points(f)]
        if point_in_polys(polys, pl.to_local(p)):
            return f
    raise AssertionError(f"no face at {p}")


def box(w=40, d=30, h=10):
    ents = Entities()
    rect(ents, 0, 0, w, d)
    f = face_at(ents, w / 2, d / 2, 0)
    ents.pushpull(f, -h)
    return ents


def doc_of(ents, group=False):
    """A document with the geometry as loose geometry, or in one group."""
    doc = Document()
    if group:
        doc.add_group(ents, "#f2f2f2")
    else:
        doc.root = ents
    return doc


def volume(ents):
    m = ents.to_manifold()
    assert m.status().name == "NoError", m.status()
    return round(m.volume(), 2)


def normals_point_out(ents):
    """Every face normal points to the empty side: solid behind it, air in front."""
    m = ents.to_manifold()
    assert m.status().name == "NoError", m.status()
    for f in ents.F:
        c = np.concatenate(ents.face_points(f)).mean(axis=0)
        n = ents.F[f].normal
        for sign, want in ((1, False), (-1, True)):
            at = np.asarray(c + n * (0.05 * sign) - 0.02, dtype=float)
            probe = mf.Manifold.cube((0.04, 0.04, 0.04)).translate(at.tolist())
            got = (m ^ probe).volume() > 1e-6
            assert got is want, f"de normaal van vlak {f} wijst niet naar buiten"


def test_clean_box_has_no_problems():
    assert ops.inspect(doc_of(box())) == []


def test_box_without_top_has_open_edges():
    ents = box()
    ents.erase_faces([face_at(ents, 20, 15, 10)])
    probs = ops.inspect(doc_of(ents))
    assert [p.kind for p in probs] == [ops.OPEN_EDGE]
    assert len(probs[0].edges) == 4
    assert probs[0].ctx == ROOT
    assert probs[0].point is not None


def test_reversed_face_is_repaired_with_normals_out():
    ents = box()
    ents.reverse_faces([face_at(ents, 20, 15, 10)])
    doc = doc_of(ents)
    assert [p.kind for p in ops.inspect(doc)] == [ops.REVERSED_FACE]
    ops.repair_all(doc)
    assert ops.inspect(doc) == []
    assert volume(doc.root) == 12000.0
    normals_point_out(doc.root)


def test_loose_edge_at_a_corner_is_repaired_away():
    ents = box()
    ents.add_line(P(0, 0, 0), P(0, 0, -8))
    doc = doc_of(ents)
    probs = ops.inspect(doc)
    assert [p.kind for p in probs] == [ops.LOOSE_EDGE]
    assert len(probs[0].edges) == 1
    ops.repair_all(doc)
    assert doc.root.loose_edges() == []
    assert ops.inspect(doc) == []
    assert volume(doc.root) == 12000.0


def test_interior_wall_is_repaired_away():
    ents = box()
    rect(ents, 0, 0, 40, 30, 5)                 # a wall that splits the closed box in two
    doc = doc_of(ents)
    assert ops.INTERIOR_FACE in [p.kind for p in ops.inspect(doc)]
    ops.repair_all(doc)
    assert ops.inspect(doc) == []
    assert volume(doc.root) == 12000.0


def test_an_edge_with_three_faces_is_reported():
    ents = box()
    ents.merge_from(Entities.from_manifold(mf.Manifold.cube((15, 10, 5)).translate((10, 10, 10))))
    probs = ops.inspect(doc_of(ents))
    assert [p.kind for p in probs] == [ops.MANY_FACES]
    assert len(probs[0].edges) == 4


def test_a_group_is_reported_with_its_own_context():
    ents = box()
    ents.erase_faces([face_at(ents, 20, 15, 10)])
    doc = doc_of(ents, group=True)
    probs = ops.inspect(doc)
    assert probs, "een groep met een gat hoort een probleem te geven"
    assert probs[0].ctx == doc.groups[0].id
    assert probs[0].part == doc.groups[0].name
    assert probs[0].items()[0][1] == doc.groups[0].id


def test_repair_is_one_undo_step():
    ents = box()
    ents.add_line(P(0, 0, 0), P(0, 0, -8))
    doc = doc_of(ents)
    before = len(doc.root.E)
    ops.repair_all(doc)
    assert len(doc.root.E) == before - 1
    doc.undo()
    assert len(doc.root.E) == before


def test_panel_marks_and_repairs_problems():
    from PySide6.QtWidgets import QApplication
    from volura.app import MainWindow
    QApplication.instance() or QApplication(sys.argv[:1])
    win = MainWindow()
    try:
        assert not win.tray.is_open("Printcontrole"), "het paneel staat standaard uit"
        assert "Printcontrole" in win.tray.extra_titles
        assert list(win.tray.panel_titles) == ["Entiteitsinfo", "Materialen", "Overzicht"]
        win.tray_actions["Printcontrole"].trigger()
        assert win.tray.is_open("Printcontrole"), "Venster > Standaardvenster zet het paneel aan"
        ents = box()
        ents.erase_faces([face_at(ents, 20, 15, 10)])
        win.doc.add_group(ents, "#f23b3b")
        win.print_panel.refresh()
        head = win.print_panel.tree.topLevelItem(0)
        assert head is not None and head.childCount() == 1
        win.print_panel.tree.setCurrentItem(head.child(0))
        assert win.view.problem_items, "klikken op een probleem markeert de randen"
        assert all(k == "e" for k, _, _ in win.view.problem_items)
        win.print_panel.repair()
        assert win.print_panel.problems == []
        assert volume(win.doc.groups[0].ents) == 12000.0
    finally:
        win.doc.dirty = False
        win.close()

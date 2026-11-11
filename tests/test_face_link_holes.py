"""The holes of a linked face travel back into the model.

A link of a *face* was a one-way road: the outline of the face came to the worksheet and
what was drawn there stayed there. The holes of that outline now go back too. A round
hole or a vent pattern drawn inside the linked wall of a box is cut through that wall -
through the wall and no further, so the wall on the far side keeps its plastic - and a
hole that was deleted on the sheet closes again with plastic as deep as the wall around
it: right through a plate, or only as deep as a pocket went. The outline of the face
itself still only goes from 3D to 2D.

These tests (Qt offscreen) link the front wall of a hollow box, change the holes on the
sheet and look at the solid.

Run with:  python -m pytest tests/test_face_link_holes.py
"""
import math
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import manifold3d as mf  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from PySide6.QtCore import QCoreApplication  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

import volura.app as app_module  # noqa: E402
from volura import linking, ops  # noqa: E402
from volura.brep import Entities  # noqa: E402
from volura.vector2d import polygon_area  # noqa: E402

_app = QApplication.instance() or QApplication(sys.argv[:1])

X, Y, Z = 60.0, 40.0, 30.0     # mm, the outside of the box
WALL = 2.0                     # mm, the thickness of its walls
R = 3.0                        # mm, the radius of the round holes
VENT = {"shape": "hex", "size": 6.0, "web": 2.0, "margin": 3.0}


@pytest.fixture()
def win(monkeypatch, tmp_path):
    monkeypatch.setattr(app_module.errors, "log_dir", lambda: str(tmp_path))
    from volura.app import MainWindow
    w = MainWindow()
    w.doc.dirty = False
    QCoreApplication.processEvents()
    yield w
    w.doc.dirty = False
    w.close()


def hollow_box():
    """The box: 60 x 40 x 30 with walls of 2 mm, open at the top."""
    return mf.Manifold.cube([X, Y, Z]) - mf.Manifold.cube(
        [X - 2 * WALL, Y - 2 * WALL, Z]).translate((WALL, WALL, WALL))


def box(win, man=None):
    """The hollow box in the model, at the origin."""
    g = ops.group_from_manifold(win.doc, hollow_box() if man is None else man, "#3b7dd8", "Box")
    win.doc.notify()
    return g


def wall_face(win, g):
    """The outside of the front wall (+y): the face that goes to the worksheet."""
    return next(f for f, F in g.ents.F.items() if F.normal[1] > 0.999 and F.loops
                and abs(g.ents.V[F.loops[0][0]][1] - Y) < 1e-6)


def to_sheet(win, g, fid):
    """That face on the worksheet, as "To Laser or Mill" puts it there."""
    win.set_mode("3d")
    win.doc.set_selection([("f", g.id, fid)])
    cut = win.doc.sheet.default_layer("cut")
    added = win.send_selection_to_sheet(layer=cut)
    assert added, "the face went to the worksheet"
    win.set_mode("3d")                            # that command ends on the worksheet
    return int(added[0].src)


def link_of(win, src):
    return next(l for l in win.doc.links if int(l.src) == src)


def loops(win, src):
    """The closed paths of the link on the worksheet, largest loop first."""
    return linking.sorted_loops([p.pts for p in win.doc.sheet.paths
                                 if int(p.src) == src and p.closed and len(p.pts) >= 3])


def outer_path(win, src):
    """The outline on the worksheet; the other paths of that source element are its holes."""
    return max((p for p in win.doc.sheet.paths if int(p.src) == src and p.closed),
               key=lambda p: abs(polygon_area(p.pts)))


def hole_paths(win, src):
    return [p for p in win.doc.sheet.paths
            if int(p.src) == src and p.closed and p is not outer_path(win, src)]


def circle(c, r=R, n=32):
    """The points of a round hole on the worksheet."""
    a = np.arange(n) * (2.0 * math.pi / n)
    return np.column_stack([c[0] + r * np.cos(a), c[1] + r * np.sin(a)])


def model_point(win, link, pt):
    """A point of the linked outline on the worksheet, as it lies in the model."""
    M = linking.invert(win._link_placement(link.shape, loops(win, int(link.src))))
    q = linking.place([np.asarray(pt, dtype=float).reshape(1, 2)], M)
    return linking.from_plane(q, linking.frame_of(link.frame))[0]


def add_holes(win, link, spots, r=R):
    """Round holes on the worksheet, at fractions of the linked outline.

    Returns the middle of every hole as it lies in the model, so a probe can go there.
    """
    src, outer = int(link.src), outer_path(win, int(link.src))
    q = np.asarray(outer.pts, dtype=float)
    lo, hi = q.min(axis=0), q.max(axis=0)
    centres = [lo + np.array([fx, fy]) * (hi - lo) for fx, fy in spots]
    for c in centres:
        win.doc.sheet.add_path(circle(c, r), closed=True, layer=outer.layer, src=src)
    win.doc.notify()
    return [model_point(win, link, c) for c in centres]


def along_y(at, length, r=2.0):
    """A cylinder of `length` mm along +y, starting at `at`."""
    M = np.array([[1.0, 0.0, 0.0, at[0]], [0.0, 0.0, 1.0, at[1]], [0.0, -1.0, 0.0, at[2]]])
    return mf.Manifold.cylinder(length, r, r, 32).transform(M)


def front_probe(centre):
    """A rod of 4 mm across through the front wall, where a hole is."""
    return along_y((centre[0], Y - 2 * WALL, centre[2]), 2 * WALL + 1.0)


def back_probe(centre):
    """The same rod through the back wall, at the same x and height."""
    return along_y((centre[0], 0.0, centre[2]), WALL)


def probe(win, g, rod):
    """How much of the rod is still plastic, and how much it is in all."""
    solid = win.doc.group(g.id).ents.to_manifold()
    return (mf.Manifold.batch_boolean([solid, rod], mf.OpType.Intersect).volume(), rod.volume())


def cut(win, g, tool):
    """What a boolean leaves of the part: the box minus `tool`."""
    win.set_mode("3d")
    win.doc.group(g.id).ents = Entities.from_manifold(
        win.doc.group(g.id).ents.to_manifold() - tool)
    win.doc.notify()


def genus(win, g):
    return win.doc.group(g.id).ents.to_manifold().genus()


def volume(win, g):
    return win.doc.group(g.id).ents.to_manifold().volume()


def test_holes_drawn_on_the_sheet_are_cut_through_the_wall(win):
    g = box(win)
    src = to_sheet(win, g, wall_face(win, g))
    link = link_of(win, src)
    win.set_mode("2d")
    centres = add_holes(win, link, [(0.2, 0.3), (0.45, 0.62), (0.75, 0.35)])
    win.set_mode("3d")
    assert genus(win, g) == 3, "one handle per hole through the wall"
    assert win.hint.text() == "3 holes cut into the model."
    for c in centres:
        got, full = probe(win, g, front_probe(c))
        assert got == pytest.approx(0.0, abs=1e-6), "the front wall is open where the hole is"
        got, full = probe(win, g, back_probe(c))
        assert got == pytest.approx(full, rel=1e-3), "the back wall kept its plastic"


def test_a_vent_pattern_on_the_linked_outline_is_cut_in_3d(win):
    g = box(win)
    src = to_sheet(win, g, wall_face(win, g))
    win.set_mode("2d")
    win.sheet_view.set_selection([outer_path(win, src)])
    made = win.apply_vent_pattern(dict(VENT))
    assert made, "the pattern made holes"
    win.set_mode("3d")
    assert genus(win, g) == len(made), "one handle per hole of the grille"


def test_the_holes_are_one_undo_step(win):
    g = box(win)
    src = to_sheet(win, g, wall_face(win, g))
    link = link_of(win, src)
    win.set_mode("2d")
    add_holes(win, link, [(0.25, 0.35), (0.5, 0.5), (0.75, 0.65)])
    steps = len(win.doc.undo_stack)
    win.set_mode("3d")
    assert len(win.doc.undo_stack) == steps + 1
    win.doc.undo()
    assert genus(win, g) == 0, "the box is whole again"


def test_a_hole_deleted_on_the_sheet_closes_again(win):
    g = box(win)
    src = to_sheet(win, g, wall_face(win, g))
    link = link_of(win, src)
    win.set_mode("2d")
    centres = add_holes(win, link, [(0.2, 0.3), (0.45, 0.62), (0.75, 0.35)])
    win.set_mode("3d")
    assert genus(win, g) == 3
    win.set_mode("2d")
    win.doc.sheet.paths.remove(hole_paths(win, src)[0])
    win.doc.notify()
    was = volume(win, g)
    win.set_mode("3d")
    assert genus(win, g) == 2
    assert win.hint.text() == "1 hole closed in the model."
    got, full = probe(win, g, along_y((centres[0][0], Y - WALL, centres[0][2]), WALL))
    assert got == pytest.approx(full, rel=1e-3), "the wall is whole where the hole was"
    assert volume(win, g) - was == pytest.approx(math.pi * R ** 2 * WALL, rel=0.02)
    for c in centres[1:]:
        got, _full = probe(win, g, front_probe(c))
        assert got == pytest.approx(0.0, abs=1e-6), "the other holes stayed open"


def test_a_hole_made_in_3d_closes_when_it_goes_off_the_sheet(win):
    g = box(win)
    cut(win, g, along_y((30.0, Y - 2 * WALL, 15.0), 2 * WALL + 1.0, R))
    src = to_sheet(win, g, wall_face(win, g))
    assert genus(win, g) == 1
    win.set_mode("2d")
    assert len(hole_paths(win, src)) == 1, "the hole came to the sheet with the outline"
    win.doc.sheet.paths.remove(hole_paths(win, src)[0])
    win.doc.notify()
    win.set_mode("3d")
    assert genus(win, g) == 0, "the hole closed through the wall"
    assert win.hint.text() == "1 hole closed in the model."


def test_a_pocket_in_the_linked_face_closes_only_that_deep(win):
    g = box(win)
    cut(win, g, along_y((30.0, Y - 1.0, 15.0), 1.0, 4.0))     # 1 mm deep, 8 mm across
    src = to_sheet(win, g, wall_face(win, g))
    win.set_mode("2d")
    win.doc.sheet.paths.remove(hole_paths(win, src)[0])
    win.doc.notify()
    was = volume(win, g)
    win.set_mode("3d")
    assert volume(win, g) - was == pytest.approx(math.pi * 4.0 ** 2 * 1.0, rel=0.05), \
        "the pocket filled, the wall behind it stayed empty"


def test_a_changed_outline_still_only_goes_from_3d_to_2d(win):
    g = box(win)
    src = to_sheet(win, g, wall_face(win, g))
    was, steps = volume(win, g), len(win.doc.undo_stack)
    win.set_mode("2d")
    outer = outer_path(win, src)
    q = np.asarray(outer.pts, dtype=float)
    q[q[:, 0] > q[:, 0].max() - 1e-6, 0] += 10.0
    outer.pts = np.ascontiguousarray(q)
    win.doc.notify()
    win.set_mode("3d")
    assert win.hint.text() == ("This link only goes from 3D to 2D: a change on the worksheet "
                               "is not put back in the model.")
    assert genus(win, g) == 0 and volume(win, g) == pytest.approx(was)
    assert len(win.doc.undo_stack) == steps, "nothing changed, so nothing to undo"


def test_a_sheet_that_did_not_change_is_no_undo_step(win):
    g = box(win)
    src = to_sheet(win, g, wall_face(win, g))
    link = link_of(win, src)
    win.set_mode("2d")
    add_holes(win, link, [(0.3, 0.4), (0.62, 0.5)])
    win.set_mode("3d")
    assert genus(win, g) == 2
    steps = len(win.doc.undo_stack)
    win.update_links()
    assert len(win.doc.undo_stack) == steps
    assert win.hint.text() == "Nothing to update."


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))

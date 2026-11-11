"""The holes of a linked face travel back into the model.

A link of a *face* was a one-way road: the outline of the face came to the worksheet and
what was drawn there stayed there. The holes of that outline now go back too. A round
hole or a vent pattern drawn inside the linked wall of a box is cut through that wall -
through the wall and no further, so the wall on the far side keeps its plastic - and a
hole that was deleted on the sheet closes again with plastic as deep as the wall around
it: right through a plate, or only as deep as a pocket went. And the outline of the face
goes back as well: a corner notched out on the sheet is cut out of the wall, an edge that
was moved out builds the wall on, as thick as it is.

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
    M = linking.invert(win._link_sheet_map(link, loops(win, int(link.src))))
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
    first = min(hole_paths(win, src), key=lambda p: np.linalg.norm(
        model_point(win, link, p.pts.mean(axis=0)) - centres[0]))
    win.doc.sheet.paths.remove(first)
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


def corner_notched(pts, corner, size):
    """The outline with a square of `size` mm cut out at point `corner`."""
    q = [np.asarray(x, dtype=float) for x in pts]
    i = corner
    c, a, b = q[i], q[i - 1], q[(i + 1) % len(q)]
    ua, ub = (a - c) / np.linalg.norm(a - c), (b - c) / np.linalg.norm(b - c)
    return np.array(q[:i] + [c + size * ua, c + size * (ua + ub), c + size * ub] + q[i + 1:])


def slab(win, g, y0, y1):
    """The volume of the part between y0 and y1: one wall of the box."""
    solid = win.doc.group(g.id).ents.to_manifold()
    cube = mf.Manifold.cube([X + 40, y1 - y0, Z + 40]).translate((-20, y0, -20))
    return mf.Manifold.batch_boolean([solid, cube], mf.OpType.Intersect).volume()


def test_a_notched_corner_on_the_sheet_is_cut_out_of_the_wall(win):
    g = box(win)
    src = to_sheet(win, g, wall_face(win, g))
    link = link_of(win, src)
    win.set_mode("2d")
    outer = outer_path(win, src)
    q = np.asarray(outer.pts, dtype=float)
    i = int(np.argmax(q[:, 0] + q[:, 1]))
    inside = model_point(win, link, q[i] - 5.0)          # in the middle of the notch
    front, back, was = slab(win, g, Y - WALL, Y), slab(win, g, 0, WALL), volume(win, g)
    outer.pts = np.ascontiguousarray(corner_notched(q, i, 10.0))
    win.doc.notify()
    win.set_mode("3d")
    assert win.hint.text() == "The outline of the face was carried to the model."
    assert probe(win, g, front_probe(inside))[0] == pytest.approx(0.0, abs=1e-6)
    assert probe(win, g, back_probe(inside))[0] > 0.0, "the wall at the back is untouched"
    assert slab(win, g, 0, WALL) == pytest.approx(back)
    assert front - slab(win, g, Y - WALL, Y) == pytest.approx(10 * 10 * WALL, rel=0.05)
    assert was - volume(win, g) == pytest.approx(10 * 10 * WALL, rel=0.08), \
        "the cut goes no further than the wall, half a millimetre past it at most"


def test_the_sheet_shows_the_wall_as_seen_from_outside(win):
    """A rectangle fits itself four ways; the link knows which one the sheet shows.

    Seen from in front of the +y wall, the +x end of the box is on the left. A hole drawn
    in the top right corner of the outline is in the top corner at x = 0 in the model.
    """
    g = box(win)
    src = to_sheet(win, g, wall_face(win, g))
    link = link_of(win, src)
    win.set_mode("2d")
    centre = add_holes(win, link, [(0.85, 0.8)])[0]
    win.set_mode("3d")
    assert centre[0] < 0.5 * X and centre[2] > 0.5 * Z
    rod, whole = probe(win, g, front_probe(centre))
    assert rod < 0.6 * whole, "the hole is there"


def test_an_edge_moved_out_on_the_sheet_builds_the_wall_on(win):
    g = box(win)
    src = to_sheet(win, g, wall_face(win, g))
    link = link_of(win, src)
    win.set_mode("2d")
    outer = outer_path(win, src)
    q = np.asarray(outer.pts, dtype=float)
    top = q[:, 1] > q[:, 1].max() - 1e-6
    mid = np.array([0.5 * (q[:, 0].min() + q[:, 0].max()), q[:, 1].max()])
    above = model_point(win, link, mid + [0.0, 5.0])     # in the new strip
    below = model_point(win, link, mid - [0.0, 5.0])     # in the old wall
    was = volume(win, g)
    q[top, 1] += 10.0
    outer.pts = np.ascontiguousarray(q)
    win.doc.notify()
    win.set_mode("3d")
    assert win.hint.text() == "The outline of the face was carried to the model."
    rod, whole = probe(win, g, along_y((above[0], Y - WALL + 0.5, above[2]), WALL - 1.0, 1.0))
    assert rod == pytest.approx(whole, rel=1e-3), "the new strip is plastic, in the wall"
    assert probe(win, g, along_y((above[0], Y - WALL - 3.0, above[2]), 2.0, 1.0))[0] == \
        pytest.approx(0.0, abs=1e-6), "and only as thick as the wall"
    assert probe(win, g, along_y((above[0], Y + 0.5, above[2]), 3.0, 1.0))[0] == \
        pytest.approx(0.0, abs=1e-6), "nothing in front of the wall"
    assert probe(win, g, front_probe(below))[0] > 0.0
    assert volume(win, g) - was == pytest.approx(60 * 10 * WALL, rel=0.05)
    assert genus(win, g) == 0, "the strip is one with the wall"


def test_the_outline_follows_a_part_moved_on_the_sheet(win):
    g = box(win)
    src = to_sheet(win, g, wall_face(win, g))
    link = link_of(win, src)
    win.set_mode("2d")
    outer = outer_path(win, src)
    q = np.asarray(outer.pts, dtype=float)
    i = int(np.argmax(q[:, 0] + q[:, 1]))
    inside = model_point(win, link, q[i] - 5.0)
    was = volume(win, g)
    outer.pts = np.ascontiguousarray(corner_notched(q, i, 10.0) + [120.0, 35.0])
    win.doc.notify()
    win.set_mode("3d")
    assert probe(win, g, front_probe(inside))[0] == pytest.approx(0.0, abs=1e-6), \
        "the corner that was notched, not the middle of the part on the sheet"
    assert was - volume(win, g) == pytest.approx(10 * 10 * WALL, rel=0.08)


def test_an_outline_that_does_not_overlap_the_face_changes_nothing(win, monkeypatch):
    assert linking.outline_diff([(0, 0), (10, 0), (10, 10), (0, 10)],
                                [(20, 0), (30, 0), (30, 10), (20, 10)]) is None
    g = box(win)
    src = to_sheet(win, g, wall_face(win, g))
    was, steps = volume(win, g), len(win.doc.undo_stack)
    win.set_mode("2d")
    outer = outer_path(win, src)
    q = np.asarray(outer.pts, dtype=float)
    outer.pts = np.ascontiguousarray(corner_notched(q, int(np.argmax(q[:, 0] + q[:, 1])), 10))
    win.doc.notify()
    far = np.eye(3)
    far[:2, 2] = (-500.0, -500.0)
    monkeypatch.setattr(linking, "outline_place", lambda *args, **kw: far)
    win.set_mode("3d")
    assert win.hint.text() == ("The new outline does not overlap the face; the model is left "
                               "as it is.")
    assert genus(win, g) == 0 and volume(win, g) == pytest.approx(was)
    assert len(win.doc.undo_stack) == steps, "nothing changed, so nothing to undo"


def test_a_changed_outline_is_one_undo_step(win):
    g = box(win)
    src = to_sheet(win, g, wall_face(win, g))
    link = link_of(win, src)
    win.set_mode("2d")
    outer = outer_path(win, src)
    q = np.asarray(outer.pts, dtype=float)
    outer.pts = np.ascontiguousarray(corner_notched(q, int(np.argmax(q[:, 0] + q[:, 1])), 10))
    add_holes(win, link, [(0.3, 0.4)])
    was, steps = volume(win, g), len(win.doc.undo_stack)
    win.set_mode("3d")
    assert win.hint.text() == ("The outline of the face was carried to the model. "
                               "1 hole cut into the model.")
    assert genus(win, g) == 1
    assert len(win.doc.undo_stack) == steps + 1
    win.doc.undo()
    assert genus(win, g) == 0 and volume(win, g) == pytest.approx(was)


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

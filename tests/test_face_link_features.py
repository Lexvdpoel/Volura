"""A linked face carries the holes and notches that are made in it to the worksheet.

A link of a *face* reads the model again in the plane of that face. Reading the whole
silhouette along the normal only shows what goes through the solid, so a pocket, a blind
hole, a notch in the edge or a corner that was cut off never reached the sheet. The link
reads the face itself now: the flat triangles of that plane, unioned into the outline of
the face, with everything that was cut into it.

These tests (Qt offscreen) link the top of a block, change the top in 3D and look at the
outline on the worksheet.

Run with:  python -m pytest tests/test_face_link_features.py
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


def box(win, man=None, size=(60.0, 40.0, 20.0)):
    """A block in the model, at the origin."""
    g = ops.group_from_manifold(win.doc, mf.Manifold.cube(list(size)) if man is None else man,
                                "#3b7dd8", "Block")
    win.doc.notify()
    return g


def top_faces(win, g):
    """The faces on top of the block, leftmost first."""
    up = [(float(g.ents.V[F.loops[0][0]][2]),
           float(np.mean([g.ents.V[v][0] for v in F.loops[0]])), f)
          for f, F in g.ents.F.items() if F.normal[2] > 0.99 and F.loops and F.loops[0]]
    hi = max(z for z, _x, _f in up)
    return [f for z, _x, f in sorted(up) if abs(z - hi) <= 1e-6]


def to_sheet(win, g, fid):
    """That face on the worksheet, as "To Laser or Mill" puts it there."""
    win.set_mode("3d")
    win.doc.set_selection([("f", g.id, fid)])
    cut = win.doc.sheet.default_layer("cut")
    added = win.send_selection_to_sheet(layer=cut)
    assert added, "the face went to the worksheet"
    win.set_mode("3d")                            # that command ends on the worksheet
    return int(added[0].src)


def cut(win, g, tool):
    """What a boolean leaves of the part: the block minus `tool`."""
    win.set_mode("3d")
    win.doc.group(g.id).ents = Entities.from_manifold(
        win.doc.group(g.id).ents.to_manifold() - tool)
    win.doc.notify()


def loops(win, src):
    """The closed paths of the link on the worksheet, largest loop first."""
    return linking.sorted_loops([p.pts for p in win.doc.sheet.paths
                                 if int(p.src) == src and p.closed and len(p.pts) >= 3])


def outline(win, src):
    pts = np.vstack([p.pts for p in win.doc.sheet.paths if int(p.src) == src and p.closed])
    return pts.min(axis=0), pts.max(axis=0)


def move_on_sheet(win, src, d):
    for p in win.doc.sheet.paths:
        if int(p.src) == src:
            p.pts = np.asarray(p.pts, dtype=float) + np.asarray(d, dtype=float)
    win.doc.notify()


def pocket(at, z_from, z_to, diameter=10.0):
    """A round hole at (x, y) from `z_from` to `z_to`."""
    return mf.Manifold.cylinder(z_to - z_from, diameter / 2, diameter / 2, 32).translate(
        (at[0], at[1], z_from))


def test_face_region_reads_the_face_and_not_the_silhouette():
    box = mf.Manifold.cube([60.0, 40.0, 20.0])
    frame = linking.face_frame((0.0, 0.0, 1.0), (0.0, 0.0, 20.0))
    loops = linking.face_region(box, frame)
    assert loops is not None and len(loops) == 1 and len(loops[0]) == 4
    assert polygon_area(loops[0]) == pytest.approx(2400.0), "the outer loop, counter-clockwise"
    assert linking.face_region(box, linking.face_frame((0.0, 0.0, 1.0), (0.0, 0.0, 21.0))) is None
    buried = box - pocket((30.0, 20.0), 5.0, 10.0)   # a hole that does not reach the top
    assert len(linking.face_region(buried, frame)) == 1, "a hole below the face is not in it"
    blind = box - pocket((30.0, 20.0), 15.0, 20.0)   # a pocket in the top, not through it
    assert len(linking.face_region(blind, frame)) == 2, "the face carries its pocket"
    assert len(linking.face_shape(blind, frame) or []) == 1, "the silhouette does not see it"
    rect = [np.array([[0.0, 0.0], [60.0, 0.0], [60.0, 40.0], [0.0, 40.0]])]
    corner = [np.array([[0.0, 0.0], [60.0, 0.0], [60.0, 30.0],
                        [50.0, 30.0], [50.0, 40.0], [0.0, 40.0]])]
    assert linking.shape_changed(rect, corner), "a corner that is gone is another shape"
    assert not linking.shape_changed(rect, [np.array([[0.0, 0.0], [30.0, 0.0], [60.0, 0.0],
                                                      [60.0, 40.0], [0.0, 40.0]])]), \
        "a point on a straight line is not"


def test_a_blind_pocket_in_the_linked_face_comes_to_the_sheet(win):
    g = box(win)
    src = to_sheet(win, g, top_faces(win, g)[0])
    cut(win, g, pocket((30.0, 20.0), 15.0, 20.0))    # 10 mm wide, 5 mm deep in the top
    win.set_mode("2d")
    ls = loops(win, src)
    assert len(ls) == 2, "the pocket came with the outline"
    assert abs(polygon_area(ls[0])) == pytest.approx(2400.0, abs=1e-3)
    assert abs(polygon_area(ls[1])) == pytest.approx(math.pi * 5.0 ** 2, rel=0.03)


def test_a_notch_in_the_linked_face_stays_where_it_lies_on_the_sheet(win):
    g = box(win)
    src = to_sheet(win, g, top_faces(win, g)[0])
    move_on_sheet(win, src, (100.0, 50.0))           # arranged on the sheet
    lo0, hi0 = outline(win, src)
    cut(win, g, mf.Manifold.cube([10.0, 10.0, 5.0]).translate((50.0, 30.0, 15.0)))
    win.set_mode("2d")
    ls = loops(win, src)
    assert len(ls) == 1 and len(ls[0]) == 6, "the corner of the top is cut off"
    lo, hi = outline(win, src)
    assert np.linalg.norm(0.5 * (lo + hi) - 0.5 * (lo0 + hi0)) < 1e-6, "it keeps its place"


def test_a_hole_through_the_solid_still_comes_to_the_sheet(win):
    g = box(win)
    src = to_sheet(win, g, top_faces(win, g)[0])
    cut(win, g, mf.Manifold.cylinder(30.0, 4.0, 4.0, 32).translate((15.0, 10.0, -5.0)))
    win.set_mode("2d")
    ls = loops(win, src)
    assert len(ls) == 2
    assert abs(polygon_area(ls[1])) == pytest.approx(math.pi * 4.0 ** 2, rel=0.03)


def test_a_face_that_did_not_change_is_no_undo_step(win):
    g = box(win)
    to_sheet(win, g, top_faces(win, g)[0])
    steps = len(win.doc.undo_stack)
    win.update_links()
    assert len(win.doc.undo_stack) == steps
    assert win.hint.text() == "Nothing to update."


def test_two_tops_in_one_plane_keep_the_linked_face(win):
    man = mf.Manifold.cube([60.0, 40.0, 20.0]) - mf.Manifold.cube([10.0, 40.0, 6.0]).translate(
        (25.0, 0.0, 14.0))                            # a step: two tops at the same height
    g = box(win, man)
    faces = top_faces(win, g)
    assert len(faces) == 2
    src = to_sheet(win, g, faces[0])                  # the left top
    cut(win, g, pocket((45.0, 20.0), 15.0, 20.0))    # a pocket in the right top
    win.set_mode("2d")
    ls = loops(win, src)
    assert len(ls) == 1, "the linked top is still the one on the sheet"
    assert abs(polygon_area(ls[0])) == pytest.approx(25.0 * 40.0, abs=1e-3)

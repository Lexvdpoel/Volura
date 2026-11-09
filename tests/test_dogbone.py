"""Dogbone corners: a round relief in every inside corner, so a milled corner is clear.

A round cutter cannot mill a sharp inside corner: it rolls along the two edges and leaves
a fillet of its own radius standing in the corner, so a square tab does not fit into a
milled square slot. `sheettools.dogbone_corners` puts a circle of the cutter diameter
through every such corner, with its centre one cutter radius away from it in the direction
of the waste, and lets the arc of that circle take the place of the corner. An inside
corner of the material is a reflex vertex of an outer contour and a convex vertex of a
hole; which way the contour runs comes from its signed area, so both directions give the
same contour back. A corner whose next edge is too short for the circle is left alone.

In the window (Qt offscreen) the selected closed contours get their reliefs in one undo
step, with the material side of every contour taken from its nesting depth on its own
layer; nothing selected and no inside corners leave the undo stack untouched and say so in
the status bar.

Run with:  python -m pytest tests/test_dogbone.py
"""
import hashlib
import math
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from PySide6.QtCore import QCoreApplication  # noqa: E402
from PySide6.QtWidgets import QApplication, QDialog, QDoubleSpinBox  # noqa: E402

import volura.app as app_module  # noqa: E402
from volura.sheettools import dogbone_corners, dogbone_spots, seg_distance  # noqa: E402
from volura.vector2d import next_src, polygon_area  # noqa: E402

_app = QApplication.instance() or QApplication(sys.argv[:1])

TOOL = 6.0
RECT = np.array([[0.0, 0.0], [40.0, 0.0], [40.0, 20.0], [0.0, 20.0]])
L = np.array([[0.0, 0.0], [40.0, 0.0], [40.0, 20.0], [20.0, 20.0], [20.0, 40.0], [0.0, 40.0]])


def area(pts):
    return abs(polygon_area(pts))


def has_point(pts, p, tol=1e-6):
    return bool(np.all(np.abs(np.asarray(pts, float) - np.asarray(p, float)) <= tol,
                       axis=1).any())


def on_contour(pts, p) -> float:
    """The distance from `p` to the closed contour `pts`."""
    q = np.asarray(pts, dtype=float)
    return min(seg_distance(p, q[i], q[(i + 1) % len(q)]) for i in range(len(q)))


def as_set(pts):
    return sorted((round(float(p[0]), 6), round(float(p[1]), 6)) for p in np.asarray(pts, float))


def zigzag(count=200, r_out=40.0, r_in=25.0) -> np.ndarray:
    """A closed star of `count` points, its radius out and in from point to point."""
    t = np.linspace(0.0, 2.0 * math.pi, count + 1)[:-1]
    rad = np.where(np.arange(count) % 2 == 0, r_out, r_in)
    return np.column_stack([rad * np.cos(t), rad * np.sin(t)])


def digest(q) -> str:
    """The fingerprint of a contour: its shape and every point rounded to a nanometre."""
    q = np.ascontiguousarray(np.asarray(q, dtype=float).reshape(-1, 2), dtype=float)
    return hashlib.sha256(np.round(q, 9).tobytes() + str(q.shape).encode()).hexdigest()[:16]


# ------------------------------------------------------------------ the function

def test_a_slot_gets_a_relief_in_every_corner():
    assert len(dogbone_spots(RECT, TOOL, material_inside=False)) == 4
    assert len(dogbone_corners(RECT, TOOL, material_inside=False)) > len(RECT)


def test_the_corner_points_of_a_slot_stay_on_the_new_contour():
    q = dogbone_corners(RECT, TOOL, material_inside=False)
    for v in RECT:
        assert on_contour(q, v) < 1e-6, "the relief reaches exactly to the corner point"


def test_the_new_points_are_edges_or_relief_circles():
    q = dogbone_corners(RECT, TOOL, material_inside=False)
    spots = dogbone_spots(RECT, TOOL, material_inside=False)
    r = 0.5 * TOOL
    for p in q:
        edge = min(seg_distance(p, RECT[i], RECT[(i + 1) % len(RECT)]) for i in range(len(RECT)))
        ring = min(abs(float(np.hypot(*(p - c))) - r) for c in spots)
        assert min(edge, ring) <= 1e-9, "a point is on an edge of the slot or on a relief"


def test_a_slot_grows_and_a_part_gets_smaller():
    assert area(dogbone_corners(RECT, TOOL, material_inside=False)) > area(RECT)
    assert area(dogbone_corners(L, TOOL, material_inside=True)) < area(L)


def test_an_outer_contour_of_a_rectangle_has_no_inside_corners():
    assert dogbone_spots(RECT, TOOL) == []
    assert np.array_equal(dogbone_corners(RECT, TOOL), RECT)


def test_an_l_shaped_part_gets_one_relief_in_its_reflex_corner():
    spots = dogbone_spots(L, TOOL, material_inside=True)
    assert len(spots) == 1
    want = np.array([20.0, 20.0]) + (0.5 * TOOL) * np.array([math.sqrt(0.5), math.sqrt(0.5)])
    assert np.allclose(spots[0], want), "the centre stands one radius from the corner, in the waste"
    q = dogbone_corners(L, TOOL, material_inside=True)
    assert has_point(q, (20.0, 20.0)), "the relief reaches exactly to the corner point"
    assert area(q) < area(L), "the relief cuts material away"


def test_both_directions_give_the_same_contour_back():
    for inside in (True, False):
        assert as_set(dogbone_corners(RECT[::-1].copy(), TOOL, material_inside=inside)) \
            == as_set(dogbone_corners(RECT, TOOL, material_inside=inside))


def test_a_corner_without_room_on_its_edge_is_left_alone():
    narrow = np.array([[0.0, 0.0], [40.0, 0.0], [40.0, 20.0], [38.0, 20.0],
                       [38.0, 40.0], [0.0, 40.0]])
    assert dogbone_spots(narrow, TOOL, material_inside=True) == []
    assert np.array_equal(dogbone_corners(narrow, TOOL, material_inside=True), narrow)
    assert len(dogbone_spots(narrow, 2.0, material_inside=True)) == 1


def test_a_blunt_corner_is_no_corner():
    assert dogbone_spots(L, TOOL, material_inside=True, min_angle=90.0) == []
    assert np.array_equal(dogbone_corners(L, TOOL, material_inside=True, min_angle=90.0), L)


def test_a_cutter_without_a_diameter_cuts_nothing():
    for tool_d in (0.0, -1.0):
        assert np.array_equal(dogbone_corners(RECT, tool_d, material_inside=False), RECT)
        assert dogbone_spots(RECT, tool_d, material_inside=False) == []


def test_a_repeated_closing_point_gives_the_same_contour():
    closed = np.vstack([RECT, RECT[:1]])         # imports often close a contour by hand
    assert np.allclose(dogbone_corners(closed, TOOL, material_inside=False),
                       dogbone_corners(RECT, TOOL, material_inside=False))


# ------------------------------------------------------------------ the exact contour

#: the fingerprint of the relieved contour for a few contours; a rewrite of the arithmetic
#: may not move a single point, so a relief that comes out elsewhere fails here
PINNED = [
    ("hole-rectangle", RECT, dict(material_inside=False), "796638ad66831c21"),
    ("hole-rectangle-backwards", RECT[::-1].copy(), dict(material_inside=False),
     "fd4cf382ef20aeb3"),
    ("L-part", L, dict(material_inside=True), "3640d5a69e6762f7"),
    ("star-200", zigzag(), dict(material_inside=True), "9e3259cbecae701c"),
    ("star-200-as-a-slot", zigzag(), dict(material_inside=False), "e50dde9dc3cc35ea"),
]


@pytest.mark.parametrize("name, pts, kw, want", PINNED, ids=[p[0] for p in PINNED])
def test_the_relieved_contour_is_pinned(name, pts, kw, want):
    assert digest(dogbone_corners(pts, TOOL, **kw)) == want


# ------------------------------------------------------------------ in the window

@pytest.fixture()
def win(monkeypatch, tmp_path):
    monkeypatch.setattr(app_module.errors, "log_dir", lambda: str(tmp_path))
    from volura.app import MainWindow
    w = MainWindow()
    w.doc.dirty = False
    QCoreApplication.processEvents()
    w.resize(900, 600)
    yield w
    w.doc.dirty = False
    w.close()


def accept_dialog(monkeypatch):
    monkeypatch.setattr(app_module.QDialog, "exec", lambda dlg: QDialog.Accepted)


def contour(win, pts, op="cut", closed=True, select=True):
    """A path on a layer of the worksheet by operation, by default selected."""
    sheet = win.doc.sheet
    layer = sheet.default_layer(op) or sheet.free_layer_name(op)
    p = sheet.add_path(pts, closed=closed, layer=layer, src=next_src(sheet.paths))
    win.set_mode("2d")
    if select:
        win.sheet_view.add_to_selection([p])
    return p


def test_only_the_hole_of_a_part_gets_its_four_dogbones(win, monkeypatch):
    accept_dialog(monkeypatch)
    win.settings.dogbone_tool_d = 6.0
    outer = contour(win, [[0, 0], [60, 0], [60, 40], [0, 40]])
    hole = contour(win, [[20, 15], [40, 15], [40, 25], [20, 25]])
    before = list(win.doc.sheet.paths)
    steps = len(win.doc.undo_stack)
    made = win.dogbone_corners_dialog()
    assert len(win.doc.undo_stack) == steps + 1, "one undo step for all the contours"
    assert made is not None and len(made) == 1
    assert made[0] in win.doc.sheet.paths and outer in win.doc.sheet.paths
    assert hole not in win.doc.sheet.paths, "the hole is replaced by its relieved copy"
    assert made[0].closed and made[0].layer == hole.layer and int(made[0].src) == int(hole.src), \
        "same layer and source, so the part still moves and nests as one"
    assert len(made[0]) > len(hole), "the arcs are new points"
    assert win.hint.text() == "4 inside corners got a dogbone."
    win.doc.undo()
    assert win.doc.sheet.paths == before


def test_a_part_without_holes_has_no_inside_corners(win, monkeypatch):
    accept_dialog(monkeypatch)
    win.settings.dogbone_tool_d = 6.0
    contour(win, [[0, 0], [60, 0], [60, 40], [0, 40]])
    before = list(win.doc.sheet.paths)
    steps = len(win.doc.undo_stack)
    assert win.dogbone_corners_dialog() is None
    assert len(win.doc.undo_stack) == steps, "nothing changed, so no undo step"
    assert win.doc.sheet.paths == before
    assert win.hint.text() == "1 contours have no inside corners."


def test_nothing_selected_is_no_undo_step(win, monkeypatch):
    def no_dialog(dlg):
        raise AssertionError("nothing selected: the dialog should not open")

    monkeypatch.setattr(app_module.QDialog, "exec", no_dialog)
    win.set_mode("2d")
    win.sheet_view.clear_selection()
    before = list(win.doc.sheet.paths)
    steps = len(win.doc.undo_stack)
    assert win.dogbone_corners_dialog() is None
    assert len(win.doc.undo_stack) == steps
    assert win.doc.sheet.paths == before
    assert win.hint.text() == "Select the closed contours for the dogbones first."


def test_the_dialog_starts_at_the_last_diameter_and_remembers_the_new_one(win, monkeypatch):
    win.settings.dogbone_tool_d = 8.0
    contour(win, [[20, 15], [40, 15], [40, 25], [20, 25]])
    seen = {}

    def fill(dlg):
        seen["start"] = round(dlg.findChild(QDoubleSpinBox).value(), 3)
        seen["range"] = (dlg.findChild(QDoubleSpinBox).minimum(),
                         dlg.findChild(QDoubleSpinBox).maximum())
        dlg.findChild(QDoubleSpinBox).setValue(4.0)
        return QDialog.Accepted

    monkeypatch.setattr(app_module.QDialog, "exec", fill)
    win.dogbone_corners_dialog()
    assert seen["start"] == 8.0
    assert seen["range"] == (0.5, 25.0)
    assert win.settings.dogbone_tool_d == pytest.approx(4.0)

"""Draw: "Freehand" (Draw > Lines) and "Rotated Rectangle" (Draw > Shapes).

The geometry behind them is in volura/geom.py (simplify_polyline and rotated_rect);
the tools themselves are exercised here too, as tests/test_guides.py does.
Runs without a display (Qt offscreen platform).
"""
import math
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtWidgets import QApplication, QToolBar, QToolButton  # noqa: E402

_app = QApplication.instance() or QApplication(sys.argv[:1])

from volura.brep import Entities  # noqa: E402
from volura.geom import Plane, rotated_rect, simplify_polyline  # noqa: E402

GROUND = Plane((0, 0, 0), (0, 0, 1))


def P(x, y, z=0.0):
    return np.array([x, y, z], dtype=float)


def clean(text):
    return text.replace("&", "").replace("…", "").strip()


# ------------------------------------------------------------------ freehand geometry

def test_simplify_drops_points_that_barely_deviate():
    """A straight stroke that wobbles a little stays a straight line."""
    t = np.linspace(0, 1, 60)
    pts = np.column_stack([t * 100.0, np.sin(t * 40) * 0.2, np.zeros(60)])
    out = simplify_polyline(pts, 0.5)
    assert len(out) == 2
    assert np.allclose(out[0], pts[0], atol=1e-12)
    assert np.allclose(out[-1], pts[-1], atol=1e-12)


def test_simplify_keeps_the_corners():
    """Every corner that deviates more than the tolerance survives."""
    pts = np.vstack([np.linspace((0, 0, 0), (40, 0, 0), 25),
                     np.linspace((40, 1, 0), (40, 30, 0), 25)[1:],
                     np.linspace((39, 30, 0), (10, 30, 0), 25)[1:]])
    out = simplify_polyline(pts, 0.5)
    assert len(out) == 4
    assert np.allclose(out, [[0, 0, 0], [40, 0, 0], [40, 30, 0], [10, 30, 0]], atol=0.6)


def test_simplify_drops_segments_shorter_than_min_len():
    pts = np.array([[0, 0, 0], [10, 0, 0], [10.3, 0.1, 0], [10.6, 0, 0],
                    [30, 20, 0], [31.2, 20.8, 0], [50, 20, 0]], dtype=float)
    assert len(simplify_polyline(pts, 0.05)) == 7, "without a minimum length everything stays"
    out = simplify_polyline(pts, 0.05, min_len=2.0)
    assert np.allclose(out, [[0, 0, 0], [10, 0, 0], [30, 20, 0], [50, 20, 0]], atol=1e-9)
    assert np.min(np.linalg.norm(np.diff(out, axis=0), axis=1)) >= 2.0


def test_simplify_keeps_the_end_points_of_short_strokes():
    pts = np.array([[0, 0, 0], [1, 0.2, 0], [2, 0, 0]], dtype=float)
    assert np.allclose(simplify_polyline(pts, 0.5), [[0, 0, 0], [2, 0, 0]])
    assert len(simplify_polyline(pts[:2], 0.5)) == 2
    assert len(simplify_polyline([], 0.5)) == 0


# ------------------------------------------------------------------ rotated rectangle

def test_rotated_rect_with_a_horizontal_first_side():
    loop = rotated_rect(P(0, 0), P(40, 0), 20, GROUND)
    assert loop.shape == (4, 3)
    assert np.allclose(loop, [[0, 0, 0], [40, 0, 0], [40, 20, 0], [0, 20, 0]], atol=1e-9)


def test_rotated_rect_with_a_slanted_first_side():
    """SketchUp: the first two corners may be in any direction, the width is perpendicular."""
    loop = rotated_rect(P(0, 0), P(30, 40), 10, GROUND)
    assert np.allclose(loop[0], (0, 0, 0), atol=1e-9)
    assert np.allclose(loop[1], (30, 40, 0), atol=1e-9)
    assert np.allclose(loop[2], (22, 46, 0), atol=1e-9)     # p1 + 10 * n x direction
    assert np.allclose(loop[3], (-8, 6, 0), atol=1e-9)
    sides = [loop[(i + 1) % 4] - loop[i] for i in range(4)]
    lengths = [float(np.linalg.norm(s)) for s in sides]
    assert np.allclose(lengths, [50, 10, 50, 10], atol=1e-9)
    for a, b in zip(sides[:2], sides[1:]):                # everything at right angles
        assert abs(float(np.dot(a, b))) < 1e-9
    assert np.allclose([GROUND.distance(p) for p in loop], 0.0, atol=1e-9)


def test_rotated_rect_width_on_both_sides_of_the_first_side():
    up = rotated_rect(P(10, 5), P(10, 45), 15, GROUND)
    down = rotated_rect(P(10, 5), P(10, 45), -15, GROUND)
    assert np.allclose(up[2], (-5, 45, 0), atol=1e-9)
    assert np.allclose(down[2], (25, 45, 0), atol=1e-9)


def test_rotated_rect_in_a_tilted_plane():
    plane = Plane((1, 2, 3), (0.3, -0.5, 0.8))
    p0, p1 = plane.to_world((0, 0)), plane.to_world((25, 0))
    loop = rotated_rect(p0, p1, 8, plane)
    assert np.allclose(loop[0], p0, atol=1e-9) and np.allclose(loop[1], p1, atol=1e-9)
    assert np.max(np.abs([plane.distance(p) for p in loop])) < 1e-9
    assert np.allclose(np.linalg.norm(loop[2] - loop[1]), 8.0, atol=1e-9)
    assert abs(float(np.dot(loop[2] - loop[1], p1 - p0))) < 1e-9


def test_rotated_rect_without_a_side_is_empty():
    assert len(rotated_rect(P(0, 0), P(0, 0), 10, GROUND)) == 0


# ------------------------------------------------------------------ edges and one face

def blob(tol=0.4, n=90):
    """A freehand stroke that ends near where it started: a wobbly ellipse."""
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([25 + 15 * np.cos(a) + tol * np.sin(9 * a),
                            25 + 10 * np.sin(a) + tol * np.cos(7 * a),
                            np.zeros(n)])


def test_closed_freehand_shape_gives_exactly_one_face():
    """The tool adds the simplified stroke as edges; a closed one becomes a single face."""
    pts = simplify_polyline(blob(), 0.6, min_len=1.0)
    assert len(pts) > 4
    assert float(np.linalg.norm(pts[-1] - pts[0])) < 1.0, "start and end are close together"
    loop = np.concatenate([pts, pts[:1]])                 # last point = first point
    ents = Entities()
    for a, b in zip(loop[:-1], loop[1:]):
        ents.add_line(a, b)
    assert len(ents.F) == 1
    assert len(ents.E) == len(pts)
    assert len(ents.loose_edges()) == 0


def test_open_freehand_stroke_gives_no_face():
    pts = simplify_polyline(blob()[:60], 0.6, min_len=1.0)
    ents = Entities()
    for a, b in zip(pts[:-1], pts[1:]):
        ents.add_line(a, b)
    assert len(ents.F) == 0
    assert len(ents.E) == len(pts) - 1


# ------------------------------------------------------------------ the tools

def test_both_tools_are_registered():
    from volura.tools import TOOL_CLASSES
    tools = {cls.name: cls for cls in TOOL_CLASSES}
    assert "freehand" in tools, "tool freehand is missing from TOOL_CLASSES"
    assert "rotrect" in tools, "tool rotrect is missing from TOOL_CLASSES"


@pytest.fixture()
def win():
    from volura.app import MainWindow
    w = MainWindow()
    w.view.cam.width, w.view.cam.height = 800, 600
    w.view.cam.set_view("iso")
    w.view.cam.target = np.array([25.0, 25.0, 0.0])
    w.view.cam.distance = 300.0
    yield w
    w.doc.dirty = False
    w.close()


class Ev:
    """A stand-in for a Qt mouse/keyboard event."""

    def __init__(self, key=None, mods=Qt.KeyboardModifiers(0), buttons=Qt.MouseButton(0)):
        self._key, self._mods, self._buttons = key, mods, buttons

    def key(self):
        return self._key

    def isAutoRepeat(self):
        return False

    def modifiers(self):
        return self._mods

    def buttons(self):
        return self._buttons


def px(win, p):
    """The widget pixel of a world point."""
    s, ok = win.view.cam.project(np.asarray(p, dtype=float)[None])
    assert ok[0], "point is off screen"
    return float(s[0][0]), float(s[0][1])


def move(tool, win, p):
    tool.mouse_move(Ev(), *px(win, p))


def click(tool, win, p):
    tool.mouse_press(Ev(), *px(win, p))


def drag(tool, win, points):
    """Press the left mouse button, drag and release."""
    tool.mouse_press(Ev(), *px(win, points[0]))
    for p in points[1:]:
        tool.mouse_move(Ev(buttons=Qt.LeftButton), *px(win, p))
    tool.mouse_release(Ev(), *px(win, points[-1]))


def corners(win, fid):
    loop = win.doc.root.face_points(fid)[0]
    return sorted(tuple(np.round(p, 3)) for p in loop)


def the_face(win):
    assert len(win.doc.root.F) == 1, f"exactly one face expected, got {len(win.doc.root.F)}"
    return next(iter(win.doc.root.F))


# ------------------------------------------------------------------ menu and toolbar

def test_menu_items_are_no_longer_placeholders(win):
    menus = {clean(a.text()): a.menu() for a in win.menuBar().actions() if a.menu() is not None}
    lines = next(a.menu() for a in menus["Draw"].actions() if clean(a.text()) == "Lines")
    shapes = next(a.menu() for a in menus["Draw"].actions() if clean(a.text()) == "Shapes")
    free = next(a for a in lines.actions() if clean(a.text()) == "Freehand")
    rot = next(a for a in shapes.actions() if clean(a.text()) == "Rotated Rectangle")
    assert free.isEnabled() and rot.isEnabled(), "both tools must work"
    assert free is win.a_freehand and rot is win.a_rotrect
    free.trigger()
    assert win.view.tool.name == "freehand"
    rot.trigger()
    assert win.view.tool.name == "rotrect"


def test_toolbar_dropdowns_offer_both_tools(win):
    tb = [b for b in win.findChildren(QToolBar) if win.toolBarArea(b).name == "TopToolBarArea"][0]
    for label, action in (("Line", win.a_freehand), ("Rectangle", win.a_rotrect)):
        act = next(a for a in tb.actions() if clean(a.text()) == label)
        btn = tb.widgetForAction(act)
        assert isinstance(btn, QToolButton) and btn.menu() is not None
        items = [a for a in btn.menu().actions() if clean(a.text()) == clean(action.text())]
        assert items and items[0].isEnabled() and items[0] is action


# ------------------------------------------------------------------ freehand

def test_freehand_drag_makes_one_face(win):
    win.set_tool("freehand")
    tool = win.view.tools["freehand"]
    drag(tool, win, [P(*p) for p in blob()])
    fid = the_face(win)
    assert len(win.doc.root.E) >= 4
    assert abs(GROUND.distance(win.doc.root.face_plane(fid).origin)) < 1e-6, "in the plane under the cursor"
    assert not tool.busy(), "after the release the tool is done"


def test_freehand_simplifies_the_stroke(win):
    """A stroke of 90 samples becomes a handful of edges, not 89."""
    win.set_tool("freehand")
    tool = win.view.tools["freehand"]
    tool.mouse_press(Ev(), *px(win, blob()[0]))
    for p in blob():
        tool.mouse_move(Ev(buttons=Qt.LeftButton), *px(win, p))
    assert len(tool.samples) == 90
    pts = tool.points()
    assert 3 < len(pts) < 30, "the stroke is simplified"
    assert np.allclose(pts[0], pts[-1], atol=1e-9), "start and end coincide: the shape closes"


def test_freehand_open_stroke_stays_open(win):
    win.set_tool("freehand")
    tool = win.view.tools["freehand"]
    drag(tool, win, [P(x, 40 + 0.2 * np.sin(x), 0) for x in np.linspace(0, 60, 70)])
    assert len(win.doc.root.F) == 0, "an open stroke does not become a face"
    assert len(win.doc.root.E) == 1, "a straight stroke stays one edge"


def test_freehand_draws_in_the_plane_under_the_cursor(win):
    wall = Plane((20, 0, 0), (0, 1, 0))
    win.doc.root.add_shape(wall, [np.array([[0, 0], [40, 0], [40, 30], [0, 30]], float)])
    win.view.zoom_extents()
    win.set_tool("freehand")
    tool = win.view.tools["freehand"]
    centre = wall.to_world(np.array([20.0, 15.0]))
    drag(tool, win, [wall.to_world(np.array([20 + 6 * math.cos(a), 15 + 6 * math.sin(a)]))
                     for a in np.linspace(0, 2 * np.pi, 60)])
    assert np.max(np.abs([wall.distance(v) for v in win.doc.root.V.values()])) < 1e-6
    assert len(win.doc.root.F) == 2, "the closed shape on the wall is a second face"


# ------------------------------------------------------------------ rotated rectangle

def test_rotrect_three_clicks_make_a_rotated_face(win):
    win.set_tool("rotrect")
    tool = win.view.tools["rotrect"]
    click(tool, win, P(0, 0))
    assert tool.plane is not None and abs(tool.plane.n[2]) > 0.99
    move(tool, win, P(30, 40))
    assert tool.vcb_value == "50 mm", "during step 2 the Measurements show the length"
    click(tool, win, P(30, 40))
    assert np.allclose(tool.p1, (30, 40, 0), atol=1e-9)
    move(tool, win, P(30, 30))
    assert tool.vcb_value == "6 mm", "during step 3 the Measurements show the width"
    click(tool, win, P(30, 30))
    fid = the_face(win)
    assert corners(win, fid) == sorted([tuple(np.round(p, 3)) for p in
                                        rotated_rect(P(0, 0), P(30, 40), -6, GROUND)])
    assert len(win.doc.root.E) == 4
    assert not tool.busy()


def test_rotrect_first_side_can_be_any_direction(win):
    win.set_tool("rotrect")
    tool = win.view.tools["rotrect"]
    for p0, p1, width in ((P(5, 5), P(55, 5), 20), (P(5, 5), P(35, 45), 12),
                          (P(60, 8), P(12, 44), 9)):
        click(tool, win, p0)
        click(tool, win, p1)
        assert np.allclose(tool.p1, p1, atol=1e-6), "the first side may have any direction"
        tool.width = width
        tool.finish()
        fid = the_face(win)
        assert corners(win, fid) == sorted(tuple(np.round(p, 3)) for p in
                                           rotated_rect(p0, p1, width, GROUND))
        win.doc.undo()
        assert len(win.doc.root.F) == 0


def test_rotrect_drag_from_corner_to_corner(win):
    """Dragging from the first to the second corner also gives the first side."""
    win.set_tool("rotrect")
    tool = win.view.tools["rotrect"]
    drag(tool, win, [P(0, 0), P(40, 0)])
    assert np.allclose(tool.p1, (40, 0, 0), atol=1e-9)
    assert tool.p0 is not None, "after the drag only the width is still clicked"
    drag(tool, win, [P(40, 0), P(40, 25)])
    fid = the_face(win)
    assert corners(win, fid) == sorted([tuple(np.round(p, 3)) for p in
                                        rotated_rect(P(0, 0), P(40, 0), 25, GROUND)])


def test_rotrect_typed_length_angle_and_width(win):
    win.set_tool("rotrect")
    tool = win.view.tools["rotrect"]
    click(tool, win, P(10, 10))
    move(tool, win, P(60, 10))
    tool.vcb_enter("50;30")
    p1 = (10 + 50 * math.cos(math.radians(30)), 10 + 50 * math.sin(math.radians(30)), 0.0)
    assert np.allclose(tool.p1, p1, atol=1e-9), "length;angle sets the first side"
    move(tool, win, P(40, 30))
    tool.vcb_enter("20")
    fid = the_face(win)
    assert corners(win, fid) == sorted(tuple(np.round(p, 3)) for p in
                                       rotated_rect(P(10, 10), p1, 20, GROUND))


def test_rotrect_typed_length_only_keeps_the_mouse_direction(win):
    win.set_tool("rotrect")
    tool = win.view.tools["rotrect"]
    click(tool, win, P(0, 0))
    move(tool, win, P(30, 40))
    tool.vcb_enter("25")
    assert np.allclose(tool.p1, (15, 20, 0), atol=1e-9), "25 mm along the direction of the mouse"
    tool.vcb_enter("10")
    assert the_face(win) is not None


def test_rotrect_arrow_keys_choose_the_drawing_plane(win):
    win.set_tool("rotrect")
    tool = win.view.tools["rotrect"]
    assert tool.key_press(Ev(Qt.Key_Right)), "right arrow chooses the plane perpendicular to the red axis"
    click(tool, win, P(20, 20))
    assert np.allclose(np.abs(tool.plane.n), (1, 0, 0), atol=1e-9)
    click(tool, win, P(20, 55))
    tool.width = 10
    tool.finish()
    the_face(win)
    xs = [v[0] for v in win.doc.root.V.values()]
    assert max(xs) - min(xs) < 1e-6, "the rectangle is in the chosen drawing plane"


def test_rotrect_esc_stops_without_drawing(win):
    win.set_tool("rotrect")
    tool = win.view.tools["rotrect"]
    click(tool, win, P(0, 0))
    click(tool, win, P(40, 0))
    tool.cancel()
    assert not tool.busy() and tool.p0 is None
    assert len(win.doc.root.F) == 0 and len(win.doc.root.E) == 0

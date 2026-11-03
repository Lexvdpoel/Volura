"""Filleting and chamfering edges.

The expected volumes are worked out by hand, looking along the edge: a chamfer
of d mm along a right angle removes a triangle of d*d/2 per mm of length, a fillet
of r mm at a right angle the wedge r*r - pi*r*r/4. For the cylinder it is the
cross-section area of the wedge, revolved (Pappus).

Runs without a display (Qt offscreen platform), like tests/test_ui_spec.py.
"""
import math
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import manifold3d as mf  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from PySide6.QtWidgets import QApplication, QToolBar  # noqa: E402

from volura import fillet, ops  # noqa: E402
from volura.brep import Entities  # noqa: E402
from volura.geom import GROUND, mesh_arrays  # noqa: E402
from volura.tools import TOOL_CLASSES  # noqa: E402

_app = QApplication.instance() or QApplication(sys.argv[:1])


def clean(text):
    return text.replace("&", "").replace("…", "").replace("...", "").strip()


def P(*p):
    return np.array(p, dtype=float)


def prism(poly2d, h=10.0):
    """A prism along z through a polygon (counterclockwise)."""
    return mf.Manifold.extrude(mf.CrossSection([np.asarray(poly2d, dtype=float)]), h)


def cube():
    return mf.Manifold.cube((10, 10, 10))          # 0..10 in x, y and z


def top_edge():
    """The top edge along x of the cube (y = 10, z = 10) with its two faces."""
    return (P(0, 10, 10), P(10, 10, 10), P(0, 0, 1), P(0, 1, 0))


def rim(man, z, inside=False):
    """All edges of the ring at height z: (p0, p1, top normal, wall normal)."""
    V, _T = mesh_arrays(man)
    ring = [p for p in V if abs(p[2] - z) < 1e-6]
    ring.sort(key=lambda p: math.atan2(p[1], p[0]))
    n = len(ring)
    out = []
    for i in range(n):
        a, b = P(*ring[i]), P(*ring[(i + 1) % n])
        mid = 0.5 * (a + b)
        w = P(mid[0], mid[1], 0)
        w = w / np.linalg.norm(w)
        out.append((a, b, P(0, 0, 1), -w if inside else w))
    return out


def volume(man):
    assert man.status().name == "NoError", man.status()
    return man.volume()


# ------------------------------------------------------------------ right-angled edge

def test_chamfer_of_one_cube_edge():
    m = cube()
    assert volume(m) == pytest.approx(1000.0)
    res = fillet.apply(m, [top_edge()], 2.0, "chamfer")
    assert volume(res) == pytest.approx(1000 - 0.5 * 2 * 2 * 10, abs=1e-3)


def test_fillet_of_one_cube_edge():
    m = cube()
    res = fillet.apply(m, [top_edge()], 2.0, "fillet")
    exact = 1000 - (4 - math.pi) * 10
    assert volume(res) == pytest.approx(exact, rel=0.01)
    assert 1000 - volume(res) == pytest.approx((4 - math.pi) * 10, rel=0.01)


def test_fillet_is_smaller_and_chamfer_lies_inside_it():
    m = cube()
    f = volume(fillet.apply(m, [top_edge()], 2.0, "fillet"))
    c = volume(fillet.apply(m, [top_edge()], 2.0, "chamfer"))
    assert f < volume(m) and c < f            # a chamfer removes more


def test_edge_is_convex_or_concave():
    m = cube()
    p0, p1, n1, n2 = top_edge()
    assert fillet.is_convex(m, p0, p1, n1, n2, 2.0)
    l = l_profile()
    assert not fillet.is_convex(l, P(0, 0, 0), P(0, 0, 10), P(1, 0, 0), P(0, 1, 0), 2.0)


# ------------------------------------------------------------------ slanted faces

def wedge():
    """Prism with an inner angle of 135° between a horizontal and a slanted face."""
    return prism([(0, -5), (15, -5), (10, 0), (0, 0)])


def test_fillet_of_edge_between_slanted_and_flat_face():
    m = wedge()
    assert volume(m) == pytest.approx(625.0)
    edge = (P(10, 0, 0), P(10, 0, 10), P(0, 1, 0), P(1, 1, 0) / math.sqrt(2))
    res = fillet.apply(m, [edge], 2.0, "fillet")
    assert res.status().name == "NoError"
    assert volume(res) < volume(m)
    assert volume(res) > volume(m) - 2 * 2 * 10          # no more removed than the wedge of the corner


# ------------------------------------------------------------------ along a curve

def cylinder():
    return mf.Manifold.cylinder(10, 10, 10, 48)


def test_fillet_of_whole_cylinder_rim():
    m = cylinder()
    edges = rim(m, 10.0)
    assert len(edges) == 48
    res = fillet.apply(m, edges, 2.0, "fillet")
    assert res.status().name == "NoError"
    removed = volume(m) - volume(res)
    a = (1 - math.pi / 4) * 2 ** 2          # cross-section of the wedge at a right angle
    assert a * 2 * math.pi * 8 < removed < a * 2 * math.pi * 10


def test_fillet_of_a_smooth_rim_stays_a_solid():
    """The chain of the cylinder rim stays one closed volume, also with a concave
    edge in between (the inner rim of a tube)."""
    outer = mf.Manifold.extrude(mf.CrossSection([np.column_stack([
        np.cos(np.arange(48) * 2 * math.pi / 48) * 12,
        np.sin(np.arange(48) * 2 * math.pi / 48) * 12])]), 10)
    inner = mf.Manifold.extrude(mf.CrossSection([np.column_stack([
        np.cos(np.arange(48) * 2 * math.pi / 48) * 6,
        np.sin(np.arange(48) * 2 * math.pi / 48) * 6])]), 10)
    tube = outer - inner
    res = fillet.apply(tube, rim(tube, 10.0, inside=True), 2.0, "fillet")
    assert res.status().name == "NoError"
    assert volume(res) < volume(tube)
    assert volume(Entities.from_manifold(res).to_manifold()) == pytest.approx(volume(res), rel=1e-4)


def cube_edges():
    """The 12 edges of the cube 0..10 with the outward normals of their two faces."""
    out = []
    for axis in range(3):
        others = [i for i in range(3) if i != axis]
        for s in (0, 10):
            for t in (0, 10):
                p0, p1 = np.zeros(3), np.zeros(3)
                p1[axis] = 10.0
                p0[others[0]] = p1[others[0]] = float(s)
                p0[others[1]] = p1[others[1]] = float(t)
                n1, n2 = np.zeros(3), np.zeros(3)
                n1[others[0]] = 1.0 if s else -1.0
                n2[others[1]] = 1.0 if t else -1.0
                out.append((p0, p1, n1, n2))
    return out


def test_fillet_of_every_cube_edge_is_a_solid():
    m = cube()
    edges = cube_edges()
    assert len(edges) == 12
    res = fillet.apply(m, edges, 1.5, "fillet")
    assert res.status().name == "NoError"
    assert 900 < volume(res) < 1000


# ------------------------------------------------------------------ concave corner

def l_profile():
    """L profile of 10 x 10 with an inner angle of 270° at (0, 0)."""
    return prism([(-5, -5), (5, -5), (5, 0), (0, 0), (0, 5), (-5, 5)])


def test_fillet_of_a_concave_corner_adds_material():
    m = l_profile()
    edge = (P(0, 0, 0), P(0, 0, 10), P(1, 0, 0), P(0, 1, 0))
    res = fillet.apply(m, [edge], 2.0, "fillet")
    assert res.status().name == "NoError"
    assert volume(res) > volume(m)
    assert volume(res) - volume(m) == pytest.approx((4 - math.pi) * 10, rel=0.02)


# ------------------------------------------------------------------ the tools

def test_tools_are_registered():
    tools = {cls.name: cls for cls in TOOL_CLASSES}
    assert "fillet" in tools and "chamfer" in tools
    assert tools["fillet"].mode == "fillet" and tools["chamfer"].mode == "chamfer"
    assert [c.name for c in TOOL_CLASSES].count("fillet") == 1


@pytest.fixture()
def win():
    from volura.app import MainWindow
    w = MainWindow()
    w.doc.dirty = False
    yield w
    w.doc.dirty = False
    w.close()


def test_fillet_and_chamfer_are_in_the_print_tools_menu_and_toolbar(win):
    menu = win.print_tools_menu
    texts = [clean(a.text()) for a in menu.actions() if not a.isSeparator()]
    assert "Fillet" in texts and "Chamfer" in texts
    bar = win.findChild(QToolBar, "PrintToolsToolbar")
    assert bar is not None and not bar.isHidden()
    assert win.toolBarArea(bar).name == "RightToolBarArea"
    buttons = [clean(a.text()) for a in bar.actions() if not a.isSeparator()]
    assert "Fillet" in buttons and "Chamfer" in buttons


def cube_group(win):
    g = ops.group_from_manifold(win.doc, cube(), "#f2f2f2", "block")
    win.doc.enter_group(g.id)
    return g


def top_edge_id(ents):
    for e in ents.E:
        a, b = ents.edge_points(e)
        if abs(a[1] - 10) < 1e-6 and abs(a[2] - 10) < 1e-6 and abs(b[1] - 10) < 1e-6 \
                and abs(b[2] - 10) < 1e-6:
            return e
    raise AssertionError("no top edge")


def test_fillet_tool_applies_the_typed_radius_to_a_group(win):
    g = cube_group(win)
    tool = win.view.tools["fillet"]
    tool.reset()
    tool.picked = [(g.id, top_edge_id(g.ents))]
    steps = len(win.doc.undo_stack)
    tool.vcb_enter("2")
    assert volume(g.ents.to_manifold()) == pytest.approx(1000 - (4 - math.pi) * 10, rel=0.01)
    assert g.name == "block" and g.color == "#f2f2f2"
    assert len(win.doc.undo_stack) == steps + 1          # one undo step
    win.undo()
    assert volume(g.ents.to_manifold()) == pytest.approx(1000.0)


def test_chamfer_tool_uses_the_distance_label(win):
    assert win.view.tools["chamfer"].vcb_label() == "Distance"
    assert win.view.tools["fillet"].vcb_label() == "Radius"
    g = cube_group(win)
    tool = win.view.tools["chamfer"]
    tool.reset()
    tool.picked = [(g.id, top_edge_id(g.ents))]
    tool.vcb_enter("2")
    assert volume(g.ents.to_manifold()) == pytest.approx(980.0, abs=1e-3)


def test_fillet_tool_marks_the_chosen_edges(win):
    g = cube_group(win)
    win.set_tool("fillet")
    tool = win.view.tools["fillet"]
    tool.reset()
    tool.picked = [(g.id, top_edge_id(g.ents))]
    tool._show()
    assert win.view.hover == ("edges", [(g.id, tool.picked[0][1])])
    assert "Radius" in tool.status_text() or "radius" in tool.status_text()


def test_fillet_tool_needs_a_closed_volume(win):
    ents = Entities()
    ents.add_shape(GROUND, [np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=float)])
    g = win.doc.add_group(ents, "#f2f2f2", "face")
    win.doc.enter_group(g.id)
    win.set_tool("fillet")
    tool = win.view.tools["fillet"]
    tool.reset()
    tool.picked = [(g.id, next(iter(ents.E)))]
    tool.apply(2.0)
    assert "Select a group" in win.hint.text()
    assert len(g.ents.F) == 1 and len(g.ents.E) == 4        # nothing was done to the face


def test_fillet_tool_picks_the_whole_smooth_chain(win):
    """A click on one segment of the smooth cylinder rim picks the whole chain."""
    m = cylinder()
    g = ops.group_from_manifold(win.doc, m, "#f2f2f2", "cylinder")
    win.doc.enter_group(g.id)
    ents = g.ents
    soft = [e for e in ents.E if e in ents.soft]
    assert soft, "the edges of the cylinder are not smooth"
    chain = fillet.soft_chain(ents, soft[0])
    assert len(chain) > 2
    assert set(chain) <= set(soft)


def test_fillet_tool_works_on_loose_geometry(win):
    win.doc.root = Entities.from_manifold(cube())
    win.set_tool("fillet")
    tool = win.view.tools["fillet"]
    tool.reset()
    tool.picked = [(win.doc.active, top_edge_id(win.doc.root))]
    tool.vcb_enter("2")
    assert volume(win.doc.root.to_manifold()) == pytest.approx(1000 - (4 - math.pi) * 10, rel=0.01)


def test_fillet_tool_without_selection_says_so(win):
    win.set_tool("fillet")
    tool = win.view.tools["fillet"]
    tool.reset()
    tool.apply(2.0)
    assert "edge" in win.hint.text().lower()


def test_fillet_rejects_a_zero_radius(win):
    win.set_tool("fillet")
    tool = win.view.tools["fillet"]
    tool.reset()
    with pytest.raises(ValueError):
        tool.vcb_enter("0")

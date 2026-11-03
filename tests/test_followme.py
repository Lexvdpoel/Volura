"""Tools > Follow Me (SketchUp's Follow Me).

The expected volumes are the ones SketchUp 2017 makes: a profile of 2 x 2 mm
along a straight path of 10 mm is a bar of 4 x 10 = 40 mm3, and the same profile
along a closed square path of 20 x 20 mm (profile perpendicular to the path, on
the outside) is a frame between the path and the offset 2 mm outward:
(24 x 24 - 20 x 20) x 2 = 352 mm3.
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
from PySide6.QtWidgets import QApplication  # noqa: E402

from volura import ops  # noqa: E402
from volura.geom import Plane  # noqa: E402
from volura.model import ROOT  # noqa: E402

_app = QApplication.instance() or QApplication(sys.argv[:1])

TOL = 0.01      # 1 %


def P(x, y, z=0.0):
    return np.array([x, y, z], dtype=float)


def square(normal, center, a, b):
    """A 2 x 2 profile loop around `center`, `a` and `b` its side directions."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    return np.array([center - a - b, center + a - b, center + a + b, center - a + b], dtype=float)


def volume(m):
    assert m.status().name == "NoError", m.status()
    return m.volume()


# ---------------------------------------------------------------- the kernel

def test_square_profile_along_a_straight_open_path_is_a_bar():
    prof = [square((1, 0, 0), P(0, 0, 0), P(0, 1, 0), P(0, 0, 1))]
    path = np.array([P(0, 0, 0), P(10, 0, 0)])
    m = ops.follow_me(prof, path, False)
    assert abs(volume(m) - 40.0) <= TOL * 40.0, m.volume()


def test_square_profile_around_a_closed_square_path():
    path = np.array([P(0, 0, 0), P(20, 0, 0), P(20, 20, 0), P(0, 20, 0)])
    prof = [square((0, 1, 0), P(-1, 0, 1), P(-1, 0, 0), P(0, 0, 1))]
    m = ops.follow_me(prof, path, True)
    assert m.status().name == "NoError", m.status()
    assert abs(volume(m) - 352.0) <= TOL * 352.0, m.volume()
    assert m.genus() == 1                       # a frame is a torus


def test_open_path_with_a_corner_mitres():
    prof = [square((1, 0, 0), P(0, 0, 0), P(0, 1, 0), P(0, 0, 1))]
    path = np.array([P(0, 0, 0), P(10, 0, 0), P(10, 10, 0)])
    m = ops.follow_me(prof, path, False)
    assert abs(volume(m) - 80.0) <= TOL * 80.0, m.volume()


def test_profile_with_a_hole_sweeps_a_tube():
    outer = square((1, 0, 0), P(0, 0, 0), P(0, 1, 0), P(0, 0, 1))
    hole = square((1, 0, 0), P(0, 0, 0), P(0, 0.5, 0), P(0, 0, 0.5))
    path = np.array([P(0, 0, 0), P(10, 0, 0)])
    m = ops.follow_me([outer, hole], path, False)
    assert abs(volume(m) - 30.0) <= TOL * 30.0, m.volume()


def test_closed_path_needs_a_closed_profile():
    """A profile that is swept into itself must stay a valid solid."""
    path = np.array([P(0, 0, 0), P(20, 0, 0), P(20, 20, 0), P(0, 20, 0)])
    prof = [square((0, 1, 0), P(0, 0, 0), P(0, 1, 0), P(0, 0, 1))]
    m = ops.follow_me(prof, path, True)
    assert m.status().name == "NoError", m.status()
    assert abs(volume(m) - 320.0) <= TOL * 320.0, m.volume()


# ------------------------------------------------------------------ the tool

@pytest.fixture()
def win():
    from volura.app import MainWindow
    w = MainWindow()
    yield w
    w.doc.dirty = False
    w.close()


def draw_path(win, pts):
    ids = []
    for a, b in zip(pts[:-1], pts[1:]):
        ids += win.doc.root.add_line(a, b)
    return ids


def draw_profile(win, normal, center, a, b):
    loop = square(normal, center, a, b)
    pl = Plane(center, normal)
    fids = win.doc.root.add_shape(pl, [pl.to_local(loop)])
    assert fids, "the profile gave no face"
    return fids[0]


def follow(win, fid):
    win.set_tool("followme")
    win.view.tools["followme"].apply(fid)


def test_menu_action_selects_the_tool(win):
    assert win.a_follow_me.isEnabled(), "Follow Me is still a placeholder"
    win.a_follow_me.trigger()
    assert win.view.tool.name == "followme"


def test_follow_me_sweeps_the_selected_path_over_the_profile(win):
    root = win.doc.root
    edges = draw_path(win, [P(0, 0, 0), P(10, 0, 0)])
    fid = draw_profile(win, (1, 0, 0), P(0, 0, 0), P(0, 1, 0), P(0, 0, 1))
    win.doc.set_selection([("e", ROOT, e) for e in edges])
    follow(win, fid)
    m = root.to_manifold()
    assert m.status().name == "NoError", m.status()
    assert abs(m.volume() - 40.0) <= TOL * 40.0, m.volume()
    assert fid not in root.F, "SketchUp uses up the profile"
    assert not root.open_edges(), "the result is not closed"


def test_follow_me_cleans_up_the_path_edges(win):
    root = win.doc.root
    edges = draw_path(win, [P(0, 0, 0), P(10, 0, 0)])
    fid = draw_profile(win, (1, 0, 0), P(0, 0, 0), P(0, 1, 0), P(0, 0, 1))
    win.doc.set_selection([("e", ROOT, e) for e in edges])
    follow(win, fid)
    assert not [e for e in root.E if not root.e2f.get(e)], "the path is still in the model as loose edges"


def test_follow_me_around_a_closed_path_in_the_model(win):
    root = win.doc.root
    draw_path(win, [P(0, 0, 0), P(20, 0, 0), P(20, 20, 0), P(0, 20, 0), P(0, 0, 0)])
    root.erase_faces(list(root.F))      # the closed path made a ground face, the path is the edge
    edges = list(root.E)
    fid = draw_profile(win, (0, 1, 0), P(-1, 0, 1), P(-1, 0, 0), P(0, 0, 1))
    win.doc.set_selection([("e", ROOT, e) for e in edges])
    follow(win, fid)
    m = root.to_manifold()
    assert m.status().name == "NoError", m.status()
    assert abs(m.volume() - 352.0) <= TOL * 352.0, m.volume()


def test_follow_me_can_sweep_the_outer_edge_of_a_selected_face(win):
    root = win.doc.root
    draw_profile(win, (0, 0, 1), P(0, 0, 0), P(10, 0, 0), P(0, 10, 0))     # the path source
    fid = draw_profile(win, (1, 0, 0), P(10, -11, 5), P(0, -1, 0), P(0, 0, 1))
    win.doc.set_selection([("f", ROOT, f) for f in root.F if f != fid])
    follow(win, fid)
    vols = []
    for shell in root.shells():
        m = root.to_manifold(shell)
        if m.status().name == "NoError" and m.volume() > 1e-6:
            vols.append(m.volume())
    assert any(abs(v - 352.0) <= TOL * 352.0 for v in vols), vols


def test_follow_me_does_nothing_when_the_path_is_not_one_chain(win):
    root = win.doc.root
    a = draw_path(win, [P(0, 0, 0), P(10, 0, 0)])
    b = draw_path(win, [P(0, 5, 0), P(10, 5, 0)])
    fid = draw_profile(win, (1, 0, 0), P(0, 0, 0), P(0, 1, 0), P(0, 0, 1))
    before = dict(root.F)
    msgs = []
    win.view.status_changed.connect(msgs.append)
    win.doc.set_selection([("e", ROOT, e) for e in a + b])
    win.set_tool("followme")
    win.view.tools["followme"].apply(fid)
    assert dict(root.F) == before, "a broken path must do nothing"
    assert any("path" in m.lower() for m in msgs), msgs


def test_follow_me_needs_a_profile_face(win):
    root = win.doc.root
    edges = draw_path(win, [P(0, 0, 0), P(10, 0, 0)])
    before = dict(root.F)
    win.doc.set_selection([("e", ROOT, e) for e in edges])
    win.set_tool("followme")
    win.view.tools["followme"].mouse_press(None, 0, 0)      # nothing under the cursor
    assert dict(root.F) == before


def test_swept_solid_is_printable(win):
    root = win.doc.root
    edges = draw_path(win, [P(0, 0, 0), P(10, 0, 0)])
    fid = draw_profile(win, (1, 0, 0), P(0, 0, 0), P(0, 1, 0), P(0, 0, 1))
    win.doc.set_selection([("e", ROOT, e) for e in edges])
    follow(win, fid)
    parts = ops.parts(win.doc)
    assert parts and parts[0].problem is None, getattr(parts[0], "problem", None)
    assert parts[0].manifold is not None
    assert parts[0].manifold.status() == mf.Error.NoError


def test_follow_me_tool_is_registered_once():
    from volura.tools import TOOL_CLASSES, FollowMeTool
    assert FollowMeTool in TOOL_CLASSES
    assert [c.name for c in TOOL_CLASSES].count("followme") == 1


def test_follow_me_says_so_when_nothing_is_selected_as_a_path(win):
    root = win.doc.root
    fid = draw_profile(win, (1, 0, 0), P(0, 0, 0), P(0, 1, 0), P(0, 0, 1))
    before = dict(root.F)
    msgs = []
    win.view.status_changed.connect(msgs.append)
    win.set_tool("followme")
    win.view.tools["followme"].apply(fid)
    assert dict(root.F) == before, "without a path nothing may change"
    assert any("path" in m.lower() for m in msgs), msgs


def test_follow_me_previews_the_solid_while_hovering(win):
    root = win.doc.root
    edges = draw_path(win, [P(0, 0, 0), P(10, 0, 0)])
    fid = draw_profile(win, (1, 0, 0), P(0, 0, 0), P(0, 1, 0), P(0, 0, 1))
    win.doc.set_selection([("e", ROOT, e) for e in edges])
    win.set_tool("followme")
    tool = win.view.tools["followme"]
    win.view.cam.width, win.view.cam.height = 800, 600
    win.view.set_view("iso")
    win.view.zoom_extents()
    xy, _ = win.view.cam.project(P(0, 0, 0))
    tool.mouse_move(None, float(xy[0][0]), float(xy[0][1]))
    assert win.view.hover == ("face", ROOT, fid), win.view.hover
    assert tool.preview is not None and len(tool.preview.tris) == 12, "no preview of the swept solid"
    assert tool.path is not None and len(tool.path[0]) == 2


# --------------------------------------------------------- more SketchUp shapes

def test_round_profile_along_a_straight_path_is_a_bar():
    seg = 48
    a = np.linspace(0, 2 * np.pi, seg, endpoint=False)
    circ = np.stack([np.zeros(seg), np.cos(a), np.sin(a)], axis=1)
    m = ops.follow_me([circ], np.array([P(0, 0, 0), P(10, 0, 0)]), False)
    assert abs(volume(m) - np.pi * 10) <= 0.02 * np.pi * 10, m.volume()


def test_profile_along_a_curved_path(win):
    """A quarter circle arc of r=20 with a 2 x 2 profile: 4 x 10 pi mm3."""
    root = win.doc.root
    pts = [P(20 * math.cos(t), 20 * math.sin(t)) for t in np.linspace(0, math.pi / 2, 13)]
    ids = draw_path(win, pts)
    t = pts[1] - pts[0]
    t = t / np.linalg.norm(t)
    fid = draw_profile(win, t, pts[0], P(0, 0, 1), np.cross(P(0, 0, 1), t))
    win.doc.set_selection([("e", ROOT, e) for e in ids])
    follow(win, fid)
    m = root.to_manifold()
    assert m.status().name == "NoError", m.status()
    assert abs(m.volume() - 40 * math.pi) <= 0.01 * 40 * math.pi, m.volume()
    assert root.soft, "the folds of an arc are softened, as in SketchUp"

"""The arc geometry behind the four SketchUp arc tools (volura/geom.py).

The expected values are the circles SketchUp 2017 draws: a half circle of 40 mm
chord has radius 20, an arc of 90° around (0, 0) with radius 30 ends at (0, 30).
"""
import math
import os
import sys

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from volura.brep import Entities  # noqa: E402
from volura.geom import (Plane, arc_from_bulge, arc_from_center,  # noqa: E402
                         arc_through_3)

GROUND = Plane((0, 0, 0), (0, 0, 1))
SEG = 12


def P(x, y, z=0.0):
    return np.array([x, y, z], dtype=float)


# ------------------------------------------------------------------ geometry

def test_half_circle_from_two_points():
    pts = arc_from_bulge(P(0, 0), P(40, 0), P(20, 20), GROUND, SEG)
    assert pts.shape == (SEG + 1, 3)
    assert np.allclose(pts[0], (0, 0, 0)) and np.allclose(pts[-1], (40, 0, 0))
    assert np.allclose(pts[SEG // 2], (20, 20, 0), atol=1e-9)  # top of the bulge
    c = P(20, 0)
    assert np.allclose(np.linalg.norm(pts - c, axis=1), 20.0, atol=1e-9)  # centre + radius
    assert np.allclose(pts[:, 2], 0.0)


def test_bulge_only_uses_the_height_above_the_chord():
    a = arc_from_bulge(P(0, 0), P(40, 0), P(11, 8), GROUND, SEG)
    b = arc_from_bulge(P(0, 0), P(40, 0), P(29, 8), GROUND, SEG)
    assert np.allclose(a, b, atol=1e-9)
    assert np.allclose(a[SEG // 2][1], 8.0, atol=1e-9)


def test_arc_from_center_quarter_and_negative_angle():
    pts = arc_from_center(P(0, 0), P(30, 0), math.radians(90), GROUND, 8)
    assert pts.shape == (9, 3)
    assert np.allclose(pts[0], (30, 0, 0), atol=1e-9)
    assert np.allclose(pts[-1], (0, 30, 0), atol=1e-9)
    assert np.allclose(np.linalg.norm(pts, axis=1), 30.0, atol=1e-9)
    back = arc_from_center(P(0, 0), P(30, 0), math.radians(-90), GROUND, 8)
    assert np.allclose(back[-1], (0, -30, 0), atol=1e-9)


def test_arc_from_center_full_circle_ends_where_it_started():
    pts = arc_from_center(P(5, 5), P(25, 5), 2 * math.pi, GROUND, 24)
    assert np.allclose(pts[0], pts[-1], atol=1e-9)
    assert np.allclose(np.linalg.norm(pts - P(5, 5), axis=1), 20.0, atol=1e-9)


def test_arc_through_three_points():
    pts = arc_through_3(P(10, 0), P(0, 10), P(-10, 0), GROUND, SEG)
    assert pts.shape == (SEG + 1, 3)
    assert np.allclose(pts[0], (10, 0, 0)) and np.allclose(pts[-1], (-10, 0, 0))
    assert np.allclose(pts[SEG // 2], (0, 10, 0), atol=1e-9)
    assert np.allclose(np.linalg.norm(pts, axis=1), 10.0, atol=1e-9)  # circle around (0, 0)


def test_arc_through_three_points_more_than_half_a_circle():
    pts = arc_through_3(P(10, 0), P(0, 10), P(0, -10), GROUND, SEG)
    assert np.allclose(np.linalg.norm(pts, axis=1), 10.0, atol=1e-9)
    assert np.allclose(pts[SEG // 2], (-7.0710678, 7.0710678, 0), atol=1e-6)  # 270°: 135° in
    assert np.allclose(pts[-1], (0, -10, 0), atol=1e-9)


def test_arcs_lie_in_the_drawing_plane():
    wall = Plane((5, -2, 3), (0, 1, 0))
    p0, p1 = wall.to_world((0, 0)), wall.to_world((40, 0))
    pts = arc_from_bulge(p0, p1, wall.to_world((20, 15)), wall, SEG)
    assert np.allclose(pts[0], p0, atol=1e-9) and np.allclose(pts[-1], p1, atol=1e-9)
    assert np.max(np.abs([wall.distance(p) for p in pts])) < 1e-9
    tilted = Plane((1, 2, 3), (1, 1, 2))
    pts = arc_from_center(tilted.origin, tilted.to_world((20, 0)), math.radians(120), tilted, SEG)
    assert np.max(np.abs([tilted.distance(p) for p in pts])) < 1e-9


def test_degenerate_arcs_give_the_chord():
    assert len(arc_from_bulge(P(0, 0), P(40, 0), P(20, 0), GROUND, SEG)) == 2
    assert len(arc_through_3(P(0, 0), P(20, 0), P(40, 0), GROUND, SEG)) == 2
    assert len(arc_from_bulge(P(0, 0), P(0, 0), P(20, 20), GROUND, SEG)) == 2


# ------------------------------------------------------------------ edges and faces

def arc_edges(ents, pts):
    """The arc as a chain of edges, exactly like the arc tools add it."""
    for a, b in zip(pts[:-1], pts[1:]):
        ents.add_line(a, b)


def test_half_circle_and_chord_become_one_face():
    e = Entities()
    pts = arc_from_bulge(P(0, 0), P(40, 0), P(20, 20), GROUND, SEG)
    arc_edges(e, pts)
    assert len(e.F) == 0 and len(e.E) == SEG  # an open arc is no face yet
    e.add_line(pts[-1], pts[0])                # the chord closes the shape
    assert len(e.F) == 1
    assert len(e.E) == SEG + 1
    tri, _ = e.face_triangles(next(iter(e.F)))
    area = float(np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1).sum() / 2)
    assert round(area, 1) == 621.2              # 12 chords of a circle with r = 20


def test_pie_closes_itself_into_a_face():
    e = Entities()
    pts = arc_from_center(P(0, 0), P(30, 0), math.radians(90), GROUND, 8)
    arc_edges(e, pts)
    e.add_line(pts[0], P(0, 0))                 # the two radii of the pie
    e.add_line(P(0, 0), pts[-1])
    assert len(e.F) == 1 and len(e.E) == 10


def test_arc_over_a_face_splits_it():
    e = Entities()
    pl = Plane((0, 0, 0), (0, 0, 1))
    e.add_shape(pl, [pl.to_local(np.array([P(0, 0), P(40, 0), P(40, 20), P(0, 20)]))])
    arc_edges(e, arc_from_bulge(P(0, 0), P(40, 0), P(20, 10), GROUND, SEG))
    assert len(e.F) == 2
    assert len(e.loose_edges()) == 0


# ------------------------------------------------------------------ tools

def test_arc_tools_are_registered():
    from volura.tools import TOOL_CLASSES
    tools = {cls.name: cls for cls in TOOL_CLASSES}
    for name in ("arc", "arc2", "arc3", "pie"):
        assert name in tools, f"tool {name} is missing from TOOL_CLASSES"
    assert tools["arc2"].default_segments == 12


if __name__ == "__main__":
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            try:
                fn()
                print("ok  ", name)
            except Exception as ex:
                fails += 1
                print("FAIL", name, type(ex).__name__, ex)
    print("failures:", fails)

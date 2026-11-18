"""3D toolpaths from the model: the height map, the drop cutter, roughing and finishing.

`volura/toolpath3d.py` is the core without Qt. A height map of the model is the highest
point in every cell of a grid; `tool_surface` slides the shape of the tool over it and
says how high the tip has to stay; `roughing` takes the stock away level by level and
`finishing` runs lines over the surface. `to_gcode` writes it with Z = 0 on the top of the
stock, and `indexed` turns the model about the X axis for a 4th axis that is turned by
hand between the sides.

Run with:  python -m pytest tests/test_toolpath3d.py
"""
import math
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import manifold3d as mf  # noqa: E402
import numpy as np  # noqa: E402

from volura.geom import mesh_arrays  # noqa: E402
from volura.toolpath3d import (Stock, Tool, estimate, finishing, height_map, indexed,  # noqa: E402
                               rotate_about_x, roughing, stock_around, to_gcode,
                               tool_surface)

# a box of corners x + 2y + 4z, its faces counter-clockwise seen from outside
CORNERS = np.array([(x, y, z) for z in (0, 1) for y in (0, 1) for x in (0, 1)], dtype=float)
QUADS = [(0, 2, 3, 1), (4, 5, 7, 6), (0, 1, 5, 4), (2, 6, 7, 3), (0, 4, 6, 2), (1, 3, 7, 5)]
TRIS = np.array([t for q in QUADS for t in ((q[0], q[1], q[2]), (q[0], q[2], q[3]))])


def box(sx=10.0, sy=10.0, sz=5.0):
    """A box from the origin to (sx, sy, sz), as the model would give it."""
    return CORNERS * np.array([sx, sy, sz], dtype=float), TRIS


def stack(*boxes):
    """Several boxes in one model, the way two solids of a part share one mesh."""
    vs, tris = [], []
    for i, (size, at) in enumerate(boxes):
        vs.append(CORNERS * np.array(size, dtype=float) + np.array(at, dtype=float))
        tris.append(TRIS + 8 * i)
    return np.concatenate(vs), np.concatenate(tris)


def hemisphere(r=10.0):
    """A hemisphere of radius `r` on the pad, centred on the origin."""
    dome = mf.Manifold.batch_boolean(
        [mf.Manifold.sphere(r, 200),
         mf.Manifold.cube((4 * r, 4 * r, 2 * r), True).translate((0.0, 0.0, r))],
        mf.OpType.Intersect)
    return mesh_arrays(dome)


def grid_of(vertices, margin=2.0, res=0.5):
    """The height map of the model over its own stock."""
    s = stock_around(vertices, margin)
    xs, ys, Z = height_map(vertices, box()[1], s.x0, s.y0, s.x1, s.y1, res)
    return xs, ys, Z, s


def distance_outside(xs, ys, lo=(0.0, 0.0), hi=(10.0, 10.0)):
    """How far every cell centre stands outside the footprint of the box (0 = inside)."""
    x, y = np.meshgrid(xs, ys)
    return np.maximum(np.maximum(lo[0] - x, x - hi[0]),
                      np.maximum(lo[1] - y, y - hi[1])).clip(0.0)


# ------------------------------------------------------------------ the stock

def test_the_stock_is_the_box_around_the_model():
    s = stock_around(box()[0], margin=2.0)
    assert (s.x0, s.y0, s.x1, s.y1) == (-2.0, -2.0, 12.0, 12.0)
    assert (s.z0, s.z1) == (0.0, 6.0)      # the top is what the mill zeroes on


# ------------------------------------------------------------------ the height map

def test_the_height_map_of_a_box_is_the_top_inside_and_the_bottom_outside():
    xs, ys, Z, s = grid_of(box()[0], res=0.5)
    assert (Z.max(), Z.min()) == (5.0, s.z0)
    out = distance_outside(xs, ys) > 0.25
    assert np.all(Z[~out] == 5.0) and np.all(Z[out] == s.z0)


def test_the_height_map_keeps_the_highest_of_two_stacks():
    """A small block on a tall one: the map shows the top of the small one, not the tall."""
    vs, tris = stack(((20.0, 20.0, 10.0), (0, 0, 0)), ((4.0, 4.0, 8.0), (8, 8, 10)))
    s = stock_around(vs, margin=2.0)
    xs, ys, Z = height_map(vs, tris, s.x0, s.y0, s.x1, s.y1, 0.5)
    assert Z.max() == 18.0
    assert Z[int(round((10.0 - s.y0) / 0.5)), int(round((10.0 - s.x0) / 0.5))] == 18.0
    assert Z[0, 0] == 0.0


# ------------------------------------------------------------------ the drop cutter

def test_a_ball_tool_over_a_flat_top_stands_on_the_top():
    xs, ys, Z, s = grid_of(box()[0], res=0.5)
    S = tool_surface(Z, 0.5, Tool("ball", 3.0))
    assert np.isclose(S[20, 20], 5.0)      # well inside the top: the tip touches it


def test_a_flat_tool_next_to_a_wall_stays_up_within_half_its_diameter():
    xs, ys, Z, s = grid_of(box()[0], margin=4.0, res=0.5)
    tool = Tool("flat", 2.0)
    S = tool_surface(Z, 0.5, tool)
    row = S[int(round((5.0 - s.y0) / 0.5))]        # the line across the middle of the wall

    def at(d):
        return row[int(round((10.0 + d - s.x0) / 0.5))]

    assert at(0.5) == 5.0 and at(1.0) == 5.0       # the wall is still under the cutter
    assert at(1.5) == s.z0                         # half a diameter further it falls


def test_a_ball_tool_rises_along_a_wall():
    """The ball hangs over the edge and touches the top of the wall with its side."""
    xs, ys, Z, s = grid_of(box()[0], margin=6.0, res=0.5)
    tool = Tool("ball", 4.0)
    S = tool_surface(Z, 0.5, tool)
    row = S[int(round((5.0 - s.y0) / 0.5))]

    def at(d):
        return row[int(round((10.0 + d - s.x0) / 0.5))]

    for d in (0.5, 1.0, 1.5):
        want = 5.0 - tool.radius + math.sqrt(tool.radius ** 2 - d * d)
        assert abs(at(d) - want) < 0.05
    assert at(2.5) == s.z0                         # beyond the radius: nothing to touch


# ------------------------------------------------------------------ roughing

def test_roughing_of_a_box_stays_out_of_the_model():
    vs, tris = box()
    s = stock_around(vs, margin=10.0)
    xs, ys, Z = height_map(vs, tris, s.x0, s.y0, s.x1, s.y1, 0.5)
    tool = Tool("flat", 4.0)
    leave = 0.3
    paths = roughing(Z, xs, ys, 0.5, s, tool, step_down=2.0, stock_to_leave=leave)
    assert paths
    surface = tool_surface(Z, 0.5, tool)
    stepover = 0.4 * tool.diameter
    for path in paths:
        for x, y, z in path:
            assert z >= s.z0 - 1e-9                    # never below the bottom
            assert s.x0 - 1e-9 <= x <= s.x1 + 1e-9     # and never outside the stock
            assert s.y0 - 1e-9 <= y <= s.y1 + 1e-9
            i = round((x - s.x0) / 0.5), round((y - s.y0) / 0.5)
            assert z >= surface[i[1], i[0]] - 1e-6 + leave
    lines = [p for p in paths if abs(p[0, 1] - p[-1, 1]) < 1e-9]
    assert len(lines) == len(paths)                    # every line runs in X
    ys_used = sorted({round(p[0, 1], 6) for p in paths})
    assert all(b - a >= stepover - 0.5 for a, b in zip(ys_used, ys_used[1:]))


def test_roughing_goes_down_to_the_bottom_of_the_model():
    vs, tris = box(20.0, 20.0, 6.0)
    s = stock_around(vs, margin=10.0)
    xs, ys, Z = height_map(vs, tris, s.x0, s.y0, s.x1, s.y1, 0.5)
    paths = roughing(Z, xs, ys, 0.5, s, Tool("flat", 4.0), step_down=2.0,
                     stock_to_leave=0.0)
    assert min(p[:, 2].min() for p in paths) == s.z0


def test_roughing_leaves_the_skin_it_was_asked_to():
    vs, tris = box(20.0, 20.0, 6.0)
    s = stock_around(vs, margin=10.0)
    xs, ys, Z = height_map(vs, tris, s.x0, s.y0, s.x1, s.y1, 0.5)
    surface = tool_surface(Z, 0.5, Tool("flat", 4.0))
    paths = roughing(Z, xs, ys, 0.5, s, Tool("flat", 4.0), step_down=2.0,
                     stock_to_leave=0.3)
    for p in paths:
        for x, y, z in p:
            i = round((x - s.x0) / 0.5), round((y - s.y0) / 0.5)
            assert surface[i[1], i[0]] <= z - 0.3 + 1e-6


def test_roughing_splits_a_line_where_the_tool_has_to_stop():
    """A block in the middle of the stock: a line that meets it comes out in two pieces."""
    vs, tris = stack(((30.0, 30.0, 1.0), (0, 0, 0)), ((4.0, 30.0, 9.0), (13, 0, 1)))
    s = stock_around(vs, margin=6.0)
    xs, ys, Z = height_map(vs, tris, s.x0, s.y0, s.x1, s.y1, 0.5)
    paths = roughing(Z, xs, ys, 0.5, s, Tool("flat", 3.0), step_down=3.0)
    level = [p for p in paths if abs(p[0, 2] - 5.0) < 1e-9]
    rows = {}
    for p in level:
        rows.setdefault(round(p[0, 1], 6), []).append(p)
    assert max(len(v) for v in rows.values()) >= 2     # the wall splits the line
    for y, pieces in rows.items():
        if not 0.0 <= y <= 30.0:
            continue                                   # a row that misses the wall
        for p in pieces:                               # no piece runs through it
            assert p[0, 0] < 13.0 or p[-1, 0] > 17.0


# ------------------------------------------------------------------ finishing

def test_finishing_of_a_hemisphere_follows_the_offset_surface():
    """The tip of a ball nose on a sphere stands R + r from its centre, less the radius."""
    r, tool_r = 10.0, 1.5
    vs, tris = hemisphere(r)
    s = stock_around(vs, margin=4.0)
    xs, ys, Z = height_map(vs, tris, s.x0, s.y0, s.x1, s.y1, 0.2)
    tool = Tool("ball", 2.0 * tool_r)
    paths = finishing(Z, xs, ys, 0.2, tool)
    assert paths
    for path in paths:
        for x, y, z in path:
            d = math.hypot(x, y)
            if d > 0.6 * r:                            # near the rim the map runs out
                continue
            want = math.sqrt((r + tool_r) ** 2 - d * d) - tool_r
            assert abs(z - want) < 0.05


def test_finishing_lines_are_parallel_and_as_far_apart_as_asked():
    vs, tris = box(20.0, 20.0, 4.0)
    s = stock_around(vs, margin=2.0)
    xs, ys, Z = height_map(vs, tris, s.x0, s.y0, s.x1, s.y1, 0.5)
    paths = finishing(Z, xs, ys, 0.5, Tool("flat", 6.0), stepover=1.0)
    used = sorted({round(p[0, 1], 6) for p in paths})
    assert len(used) > 5
    assert all(abs(b - a - 1.0) < 0.5 for a, b in zip(used, used[1:]))
    assert all(abs(p[0, 1] - p[-1, 1]) < 1e-9 for p in paths)


def test_finishing_at_90_degrees_runs_in_y():
    vs, tris = box(20.0, 20.0, 4.0)
    s = stock_around(vs, margin=2.0)
    xs, ys, Z = height_map(vs, tris, s.x0, s.y0, s.x1, s.y1, 0.5)
    paths = finishing(Z, xs, ys, 0.5, Tool("flat", 6.0), angle=90, stepover=1.0)
    assert paths and all(abs(p[0, 0] - p[-1, 0]) < 1e-9 for p in paths)


def test_finishing_does_not_run_over_the_waste_board():
    vs, tris = box(20.0, 20.0, 4.0)
    s = stock_around(vs, margin=6.0)
    xs, ys, Z = height_map(vs, tris, s.x0, s.y0, s.x1, s.y1, 0.5)
    paths = finishing(Z, xs, ys, 0.5, Tool("ball", 3.0))
    for path in paths:
        assert np.all(path[:, 2] > s.z0 + 1e-9)          # nothing cut in thin air
        out = np.maximum(np.maximum(-path[:, 0], path[:, 0] - 20.0),
                         np.maximum(-path[:, 1], path[:, 1] - 20.0)).clip(0.0)
        assert np.all(out <= 1.5 + 0.5)                  # no further than the ball reaches


# ------------------------------------------------------------------ the G-code

def gcode_of_box(tool=Tool("ball", 3.0), **kw):
    vs, tris = box(20.0, 20.0, 4.0)
    s = stock_around(vs, margin=4.0)
    xs, ys, Z = height_map(vs, tris, s.x0, s.y0, s.x1, s.y1, 0.5)
    paths = finishing(Z, xs, ys, 0.5, tool, **kw)
    return to_gcode(paths, tool, 800.0, 300.0, 10000, 5.0, s), paths, s


def test_the_gcode_looks_like_the_mill_expects_it():
    text, paths, s = gcode_of_box()
    head = text.splitlines()[:6]
    assert "G21" in head and "G90" in head
    assert any(line.startswith("M3 S10000") for line in text.splitlines())
    assert "G4 P2" in text
    tail = text.rstrip().splitlines()[-3:]
    assert tail[1] == "M5" and tail[2] == "G0 X0 Y0"
    assert "G1 " in text


def test_the_gcode_zeroes_on_the_top_of_the_stock():
    text, paths, s = gcode_of_box()
    cuts = [float(line.split("Z")[1].split()[0]) for line in text.splitlines()
            if line.startswith("G1 ") and "Z" in line]
    assert cuts and max(cuts) <= 0.0 and min(cuts) >= -(s.z1 - s.z0) - 1e-9
    assert any(line == "G0 Z5.000" for line in text.splitlines())


def test_the_gcode_drops_the_points_on_a_straight_line():
    vs, tris = box(20.0, 20.0, 4.0)
    s = stock_around(vs, margin=2.0)
    xs, ys, Z = height_map(vs, tris, s.x0, s.y0, s.x1, s.y1, 0.2)
    flat = finishing(Z, xs, ys, 0.2, Tool("flat", 6.0), stepover=2.0)
    text = to_gcode(flat, Tool("flat", 6.0), 800.0, 300.0, 10000, 5.0, s)
    moves = [line for line in text.splitlines() if line.startswith("G1 X")]
    assert len(moves) < sum(len(p) for p in flat) / 10


# ------------------------------------------------------------------ what it costs

def test_the_estimate_counts_the_cut_and_the_travel():
    straight = [np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])]
    info = estimate(straight, 1000.0, 300.0, 3000.0)
    assert abs(info["cut_mm"] - 10.0) < 1e-9
    assert abs(info["seconds"] - 0.6) < 1e-9
    back = estimate([straight[0], straight[0]], 1000.0, 300.0, 3000.0)
    assert abs(back["travel_mm"] - 10.0) < 1e-9


def test_a_plunge_goes_at_the_plunge_feed():
    down = [np.array([[0.0, 0.0, 5.0], [0.0, 0.0, 0.0]])]
    info = estimate(down, 1000.0, 300.0, 3000.0)
    assert abs(info["seconds"] - 5.0 / 300.0 * 60.0) < 1e-9


# ------------------------------------------------------------------ the 4th axis

def test_rotating_about_x_turns_the_model_upside_down():
    vs, _ = box(10.0, 4.0, 6.0)
    turned = rotate_about_x(vs, 180.0)
    assert np.allclose(turned[:, 0], vs[:, 0])          # the A axis runs along X
    assert np.allclose(turned.min(axis=0), (0.0, 0.0, 0.0))
    assert np.allclose(turned.max(axis=0), (10.0, 4.0, 6.0))
    assert np.allclose(turned[0], (0.0, 4.0, 6.0))      # the near bottom is the far top


def test_a_quarter_turn_moves_y_into_z():
    vs, _ = box(10.0, 4.0, 6.0)
    turned = rotate_about_x(vs, 90.0)
    assert np.allclose(turned[:, 1].min(), -1.0) and np.allclose(turned[:, 1].max(), 5.0)
    assert np.allclose(turned[:, 2].min(), 1.0) and np.allclose(turned[:, 2].max(), 5.0)


def test_indexed_milling_writes_the_angle_of_every_side():
    vs, tris = box(20.0, 20.0, 4.0)
    paths, angles, s, kinds = indexed(vs, tris, (0.0, 180.0), rough_tool=Tool("flat", 4.0),
                                      finish_tool=Tool("ball", 3.0), res=0.5, margin=4.0)
    assert paths and len(angles) == len(paths) == len(kinds)
    assert set(angles) == {0.0, 180.0}
    assert set(kinds) == {"rough", "finish"}
    text = to_gcode(paths, Tool("ball", 3.0), 800.0, 300.0, 10000, 5.0, s, angles=angles)
    assert "G0 A180.000" in text
    assert "G0 A0" not in text                          # the first side needs no word


def test_indexed_stock_holds_every_side():
    vs, tris = box(20.0, 6.0, 12.0)
    _paths, _angles, s, _kinds = indexed(vs, tris, (0.0, 180.0),
                                         finish_tool=Tool("ball", 3.0), res=1.0, margin=2.0)
    assert s.z1 >= 12.0 + 1.0 - 1e-9 and s.z0 <= 1e-9
    assert s.x0 <= -2.0 + 1e-9 and s.x1 >= 22.0 - 1e-9

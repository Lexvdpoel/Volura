"""The skins of painted faces (volura/paint_split.py): what a painted face prints as.

The Paint Bucket gives one face of a part a colour of its own, but a face is not a volume, so
the slicer cannot print it. `split_painted` makes a skin of it: the part cut free under the
painted surface, `depth` mm deep, in the filament of the face's own colour, and what is left of
the part in the colour of the part itself. Together the pieces are exactly the part again - they
do not overlap and nothing is lost - so one place in the print never holds two filaments.

Run with:  python -m pytest tests/test_paint_split.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import manifold3d as mf  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from volura import brep, fileio, ops, paint_split  # noqa: E402

RED, BLUE, GREEN = "#d23b3b", "#2f6fd6", "#3aa655"
AXES = {"x": 0, "y": 1, "z": 2}


def _tagged(man, colors):
    """The same Manifold with the color of every triangle written on it, as a group is."""
    mesh = man.to_mesh()
    v = np.asarray(mesh.vert_properties, dtype=np.float64)
    t = np.asarray(mesh.tri_verts, dtype=np.uint64)
    tags = np.array([ops.color_id(c) for c in colors], dtype=np.uint64)
    return brep.manifold_from_tris(v, t, tags)


def _corners(man):
    """The three corners of every triangle of `man`."""
    mesh = man.to_mesh()
    return np.asarray(mesh.vert_properties)[:, :3][np.asarray(mesh.tri_verts).reshape(-1, 3)]


def _painted_cube(size=20.0, colors=(("z", BLUE),)):
    """A cube in RED with whole faces painted: `colors` is [(axis, color)], where the face at
    the far end of that axis takes the color."""
    man = mf.Manifold.cube((size, size, size))
    corner = _corners(man)
    out = [RED] * len(corner)
    for axis, color in colors:
        on = np.all(corner[:, :, AXES[axis]] > size - 1e-6, axis=1)
        out = [color if hit else keep for hit, keep in zip(on, out)]
    return _tagged(man, out)


def _painted_dome(radius=15.0, segments=60, color=BLUE):
    """A ball in RED with the faces of its upper half painted `color`."""
    man = mf.Manifold.sphere(radius, segments)
    up = np.all(_corners(man)[:, :, AXES["z"]] > -1e-9, axis=1)
    return _tagged(man, [color if hit else RED for hit in up])


def _area(man, colors, color):
    """The area of the triangles of `man` that carry `color`."""
    corner = _corners(man)
    a, b, c = corner[:, 0], corner[:, 1], corner[:, 2]
    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    return float(sum(ar for ar, col in zip(area, colors) if col == color))


def _pieces(man, depth=1.0):
    """The pieces of the painted part `man`, the way the slice builds them."""
    colors = fileio.triangle_colors(man)
    return colors, paint_split.split_painted(man, colors, RED, depth)


def test_only_a_colour_other_than_the_part_is_painted():
    assert paint_split.painted_colors([None, RED, None], RED) == []
    assert paint_split.painted_colors([None, BLUE, RED, BLUE], RED) == [BLUE]
    assert paint_split.painted_colors(["#2F6FD6"], RED) == [BLUE], "the same colour, spelled otherwise"


def test_an_unpainted_part_stays_one_piece():
    man = mf.Manifold.cube((20.0, 20.0, 20.0))
    colors = [RED] * len(man.to_mesh().tri_verts)
    assert paint_split.split_painted(man, colors, RED, 1.0) == [(man, RED)]


def test_a_painted_face_prints_a_skin_of_its_own_colour():
    man = _painted_cube()
    colors, pieces = _pieces(man)
    assert paint_split.painted_colors(colors, RED) == [BLUE]
    got = {color: volume for volume, color in
           ((m.volume(), color) for m, color in pieces)}
    assert got[BLUE] == pytest.approx(400.0, abs=1e-6), "20 × 20 mm, 1 mm deep"
    assert got[RED] == pytest.approx(7600.0, abs=1e-6)
    skin = next(m for m, color in pieces if color == BLUE)
    z = np.asarray(skin.to_mesh().vert_properties)[:, 2]
    assert z.max() == pytest.approx(20.0) and z.min() == pytest.approx(19.0), \
        "the skin hangs under the painted surface"


def test_the_depth_is_taken_from_the_settings():
    man = _painted_cube()
    for depth in (0.4, 1.0, 2.5):
        _colors, pieces = _pieces(man, depth)
        skin = next(m.volume() for m, color in pieces if color == BLUE)
        assert skin == pytest.approx(400.0 * depth, rel=1e-6)


def test_the_pieces_together_are_exactly_the_part():
    man = _painted_cube()
    _colors, pieces = _pieces(man, 1.5)
    assert sum(m.volume() for m, _c in pieces) == pytest.approx(8000.0, abs=1e-6)
    a, b = (m for m, _c in pieces)
    shared = mf.Manifold.batch_boolean([a, b], mf.OpType.Intersect)
    assert shared.volume() == pytest.approx(0.0, abs=1e-6), "two filaments in one place"


def test_two_painted_faces_share_the_edge_where_their_skins_meet():
    man = _painted_cube(colors=(("z", BLUE), ("x", GREEN)))
    _colors, pieces = _pieces(man)
    got = {color: m.volume() for m, color in pieces}
    assert got[RED] == pytest.approx(7220.0, abs=1e-6)
    assert sorted([got[BLUE], got[GREEN]]) == pytest.approx([380.0, 400.0], abs=1e-6), \
        "the colour that comes later keeps the corner both skins want"
    assert sum(got.values()) == pytest.approx(8000.0, abs=1e-6)
    assert len(pieces) == 3


def test_the_skin_of_a_painted_curved_surface_lies_against_it():
    man = _painted_dome()
    colors, pieces = _pieces(man)
    assert sum(m.volume() for m, _c in pieces) == pytest.approx(man.volume(), rel=1e-6)
    skin = next(m for m, color in pieces if color == BLUE)
    r = np.linalg.norm(np.asarray(skin.to_mesh().vert_properties)[:, :3], axis=1)
    assert r.min() >= 15.0 - 1.0 - 0.05 and r.max() <= 15.0 + 0.05, \
        "the skin is neither thicker than the depth nor off the ball"
    assert skin.volume() == pytest.approx(_area(man, colors, BLUE), rel=0.15), \
        "a skin of 1 mm deep covers its painted faces"


def test_a_skin_that_is_nothing_is_left_out():
    """A face painted in the colour of the part is no painting at all."""
    man = _painted_cube(colors=(("z", RED),))
    _colors, pieces = _pieces(man)
    assert len(pieces) == 1 and pieces[0][1] == RED
    assert pieces[0][0].volume() == pytest.approx(8000.0, abs=1e-6)

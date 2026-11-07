"""Performance and watertightness of a bigger model.

A light version of benchmark stap (c) in tools/bench.py: a grid of squares on
the ground, each pushed up on its own. The full 20x20 grid is too slow for a
unit test, so this uses 8x8 (64 boxes); it must finish well within 5 s and the
result must still be one watertight solid (no open edges, a valid volume).

The timing is a guard against the quadratic blow-ups that this task is about:
scanning the whole model on every push/pull, or rebuilding every face of a plane
that did not change. The correctness checks make sure a speed-up is never bought
by producing broken geometry (which is how the pocket push/pull regressed).

The last two guards are about the same on a heavy model: the check that Entity
Info and an export do (triangulating every face of a group) and the overlap part
of a print check, which only has to intersect the parts that can touch at all.
"""
import os
import sys
import time

import manifold3d as mf
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from volura import ops  # noqa: E402
from volura.brep import Entities, _loop_triangles, triangulate_payload  # noqa: E402
from volura.geom import Plane  # noqa: E402

GROUND = Plane((0, 0, 0), (0, 0, 1))
LIMIT = 5.0  # seconds
TRI_LIMIT = 2.0  # seconds for 200k triangular faces
APART_LIMIT = 2.0  # seconds for a print check of ten heavy parts that lie apart


def _new_face(ents, fids, vertex_id):
    """The face made by add_shape whose outer loop runs through vertex_id."""
    for f in fids:
        if vertex_id in ents.F[f].loops[0]:
            return f
    raise AssertionError("face not found")


def grid(n=8, size=5.0, gap=1.0):
    """An n x n grid of `size` mm squares `gap` mm apart, each pulled up 1..5 mm."""
    ents = Entities()
    step = size + gap
    for iy in range(n):
        for ix in range(n):
            x0, y0 = ix * step, iy * step
            loop = np.array([[x0, y0], [x0 + size, y0],
                             [x0 + size, y0 + size], [x0, y0 + size]])
            corner = ents.vertex(np.array([x0, y0, 0.0]))
            f = _new_face(ents, ents.add_shape(GROUND, [loop]), corner)
            ents.pushpull(f, -float(1 + (ix + iy) % 5))
    return ents


def _grid8():
    """The 8x8 grid, built once and shared by the tests below."""
    global _GRID8
    if _GRID8 is None:
        _GRID8 = grid(8)
    return _GRID8


_GRID8 = None


def test_grid_pushpull_is_fast_and_watertight():
    t0 = time.perf_counter()
    ents = grid(8)
    dt = time.perf_counter() - t0

    # 64 boxes, each with a top, a bottom and four sides
    assert len(ents.F) == 64 * 6

    # a solid for 3D-printing: no edge borders anything but two faces
    assert ents.open_edges() == []

    # ... and it turns into a valid, positive-volume manifold
    man = ents.to_manifold()
    assert man.status().name == "NoError", man.status()
    assert man.volume() > 0.0

    assert dt < LIMIT, f"8x8 grid took {dt:.2f} s (limit {LIMIT} s)"


def test_grid_volume_matches_the_boxes():
    # every box is 5 x 5 mm and 1..5 mm high, so the volume is 25 * sum(heights)
    ents = _grid8()
    want = 25.0 * sum(1 + (ix + iy) % 5 for iy in range(8) for ix in range(8))
    man = ents.to_manifold()
    assert man.status().name == "NoError", man.status()
    assert abs(man.volume() - want) < 1e-6


def triangle_payload(n=200000):
    """`n` triangular faces in a strip, straight as a ``mesh_payload``.

    What an imported mesh looks like: nearly every face is one loop of three vertices.
    Made as arrays here because turning a sphere into geometry takes longer than the
    triangulation this measures."""
    x = np.arange(n + 1, dtype=float)
    verts = np.stack([np.repeat(x, 2), np.tile([0.0, 10.0], n + 1), np.zeros(2 * n + 2)], axis=1)
    loops = [((2 * i, 2 * i + 1, 2 * i + 2),) for i in range(n)]
    loops.append(((2 * n - 2, 2 * n, 2 * n + 1, 2 * n - 1),))     # one face to clip
    normals = np.tile([0.0, 0.0, -1.0], (n + 1, 1))
    tags = np.arange(1, n + 2, dtype=np.uint64)
    return verts, loops, normals, tags


def _triangles_face_by_face(payload):
    """The triangles of a payload the way it reads the faces one by one."""
    verts, loops, normals, tags = payload
    tris, out = [], []
    for i, ls in enumerate(loops):
        ids = [v for loop in ls for v in loop]
        if not ids:
            continue
        t = _loop_triangles(verts, ls, normals[i])
        for a, b, c in t:
            tris.append((ids[a], ids[b], ids[c]))
            out.append(int(tags[i]))
    return np.asarray(tris, dtype=np.int64), np.asarray(out, dtype=np.uint64)


def test_triangulating_a_mesh_of_triangles_is_fast():
    """Entity Info, an export and a print check all triangulate a whole group."""
    payload = triangle_payload()
    t0 = time.perf_counter()
    tris, tags = triangulate_payload(payload)
    dt = time.perf_counter() - t0

    want, want_tags = _triangles_face_by_face(payload)
    assert tris.shape == want.shape
    assert np.array_equal(tris, want)
    assert np.array_equal(tags, want_tags)

    assert dt < TRI_LIMIT, f"200k faces triangulated in {dt:.2f} s (limit {TRI_LIMIT} s)"


class Counted:
    """A part manifold that counts how often the print check intersects it."""

    def __init__(self, man):
        self.man = man
        self.xors = 0

    def __xor__(self, other):
        self.xors += 1
        return self.man ^ (other.man if isinstance(other, Counted) else other)

    def __getattr__(self, name):
        return getattr(self.man, name)


def _sphere(x, sections=200):
    return mf.Manifold.sphere(30, sections).translate((x, 0, 0))


def test_print_check_only_intersects_parts_that_can_touch():
    """Parts whose boxes lie apart share no volume; intersecting them is work for nothing."""
    far = [ops.Part("A", "#d23b3b", Counted(_sphere(0.0))),
           ops.Part("B", "#2f6fd6", Counted(_sphere(500.0)))]
    msgs = ops.check_parts(far, None)
    assert not [m for m in msgs if "overlap" in m[1]]
    assert far[0].manifold.xors == 0 and far[1].manifold.xors == 0

    near = [ops.Part("A", "#d23b3b", Counted(_sphere(0.0))),
            ops.Part("B", "#2f6fd6", Counted(_sphere(20.0)))]
    msgs = ops.check_parts(near, None)
    assert [m for m in msgs if "overlap" in m[1]]
    assert near[0].manifold.xors == 1


def _apart_parts():
    """Ten heavy spheres in ten colors, far enough apart to print side by side."""
    global _APART
    if _APART is None:
        colors = ["#%06x" % (0x210000 + 0x0d0d0d * i) for i in range(10)]
        _APART = [ops.Part("Part %d" % (i + 1), colors[i], _sphere(400.0 * i, 630))
                  for i in range(10)]
    return _APART


_APART = None


def test_print_check_of_parts_apart_is_fast():
    parts = _apart_parts()
    t0 = time.perf_counter()
    msgs = ops.check_parts(parts, None)
    dt = time.perf_counter() - t0

    assert not [m for _level, m in msgs if "overlap" in m]
    assert dt < APART_LIMIT, f"print check took {dt:.2f} s (limit {APART_LIMIT} s)"


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

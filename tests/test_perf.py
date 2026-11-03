"""Performance and watertightness of a bigger model.

A light version of benchmark stap (c) in tools/bench.py: a grid of squares on
the ground, each pushed up on its own. The full 20x20 grid is too slow for a
unit test, so this uses 8x8 (64 boxes); it must finish well within 5 s and the
result must still be one watertight solid (no open edges, a valid volume).

The timing is a guard against the quadratic blow-ups that this task is about:
scanning the whole model on every push/pull, or rebuilding every face of a plane
that did not change. The correctness checks make sure a speed-up is never bought
by producing broken geometry (which is how the pocket push/pull regressed).
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from volura.brep import Entities  # noqa: E402
from volura.geom import Plane  # noqa: E402

GROUND = Plane((0, 0, 0), (0, 0, 1))
LIMIT = 5.0  # seconds


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

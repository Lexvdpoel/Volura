"""The fast paths of the geometry give the same answers as the general ones.

* ``Plane`` without an explicit u: its frame is orthonormal also for a normal that
  is tilted a little away from Z (world X, the default u there, is not in the
  plane), so local -> world -> local is exact. A skewed frame put an inlay on a
  2 degree tilted face millimetres off.
* ``Entities.from_manifold`` merges near vertices with a grid, not a sweep on x:
  a cylinder along X has thousands of vertices on two planes of constant x, which
  made the sweep quadratic.
"""
import math
import os
import sys
import time

import manifold3d as mf
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from volura import brep  # noqa: E402
from volura.brep import Entities  # noqa: E402
from volura.geom import Plane  # noqa: E402


@pytest.mark.parametrize("degrees", [0.5, 1.0, 2.0, 2.5])
def test_plane_frame_is_orthonormal_on_a_slightly_tilted_normal(degrees):
    a = math.radians(degrees)
    pl = Plane((3.0, -2.0, 5.0), (math.sin(a), 0.0, math.cos(a)))
    assert abs(float(pl.u @ pl.n)) < 1e-12
    assert np.linalg.norm(pl.u) == pytest.approx(1.0, abs=1e-12)
    assert np.linalg.norm(pl.v) == pytest.approx(1.0, abs=1e-12)
    p = pl.to_world(np.array((7.0, 4.0)))
    assert np.allclose(pl.to_local(p), (7.0, 4.0), atol=1e-12)
    assert abs(float((p - pl.origin) @ pl.n)) < 1e-12, "the point left the plane"


def test_merge_within_tol_joins_only_near_points():
    pts = np.array([(0.0, 0.0, 0.0), (0.0, 0.0, 0.5 * brep.TOL), (0.0, 1.0, 0.0),
                    (0.0, 1.0, 2 * brep.TOL), (5.0, 0.0, 0.0)])
    rep, kept = brep._merge_within_tol(pts)
    assert kept.tolist() == [True, False, True, True, True]
    assert rep.tolist() == [0, 0, 2, 3, 4]


def test_from_manifold_of_a_cylinder_along_x_is_fast():
    segs = 4000
    m = mf.Manifold.cylinder(10, 20, 20, segs).rotate((0, 90, 0))
    t = time.perf_counter()
    ents = Entities.from_manifold(m)
    elapsed = time.perf_counter() - t
    assert len(ents.V) == 2 * segs
    assert len(ents.F) == segs + 2
    assert ents.to_manifold().volume() == pytest.approx(m.volume(), rel=1e-6)
    assert elapsed < 5.0, f"from_manifold took {elapsed:.1f} s"

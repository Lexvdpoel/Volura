"""Benchmarks voor grotere modellen.

    .venv\\Scripts\\python tools\\bench.py

Tijden in seconden, gemeten op dezelfde machine (Python 3.13, numpy 2.x,
manifold3d). "voor" = de eerste versie, "na" = na de snelheidsverbeteringen.
Elke stap (a)-(e) moet onder 1 s, of minstens 3x sneller dan vooraf.

    stap  omschrijving                                   voor     na
    (a)   cirkel 96 segmenten + pushpull 20 mm           0.101    0.098
    (b)   20x cirkel 24 segmenten op het bovenvlak, -5   0.361    0.300
    (c)   raster 20x20 rechthoeken 5x5 mm + optrekken    9.903    1.006
    (d)   200x infer over het resultaat van (c)          0.196    0.088
    (e)   to_manifold + STL-export van (c)               0.030    0.030

Stap (c) is het zwaarst: elk van de 400 rechthoeken herbouwt de hele vlakrang
van de grond (400 rechthoeken, 3200 randen), dus elk werk per rechthoek wordt
O(n) en de hele stap O(n^2). De verbeteringen staan in volura/brep.py
(``_plane_faces``/``_plane_edges`` houden per vlak bij welke randen en gezichten
erbij horen, ``rebuild_plane`` doet maar de helft van het werk voor een vlak dat
onverdeeld blijft) en in ``volura/inference.py`` (de snap-lijst van ``infer``
wordt in de GeoCache bijgehouden in plaats van per aanroep in Python opgebouwd).
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from volura.brep import Entities                      # noqa: E402
from volura.camera import Camera                      # noqa: E402
from volura.fileio import write_stl                   # noqa: E402
from volura.geom import Plane, circle_points, point_in_polys  # noqa: E402
from volura.inference import InferenceEngine          # noqa: E402
from volura.model import Document                     # noqa: E402

GROUND = Plane((0, 0, 0), (0, 0, 1))


def face_at(ents, p, plane=None):
    """The face that contains the world point p (like a click in the viewport)."""
    p = np.asarray(p, dtype=float)
    for f in ents.F:
        pl = ents.face_plane(f)
        if abs(pl.distance(p)) > 1e-6:
            continue
        polys = [pl.to_local(L) for L in ents.face_points(f)]
        if point_in_polys(polys, pl.to_local(p)):
            return f
    raise AssertionError(f"no face at {p}")


def circle(ents, plane, center, radius, segments):
    """Circle tool: a closed loop of `segments` sides in `plane` around `center`.

    Returns (created faces, a vertex id on the loop) so the caller can find the
    circle face again without searching the whole model."""
    c = plane.to_local(np.asarray(center, dtype=float))
    loop = circle_points(radius, segments) + c
    anchor = ents.vertex(plane.to_world(loop[0]))
    return ents.add_shape(plane, [loop], hint=(0, 0, 50)), anchor


def new_face(ents, fids, vertex_id):
    """The face made by `add_shape` whose outer loop runs through `vertex_id`."""
    for f in fids:
        if vertex_id in ents.F[f].loops[0]:
            return f
    raise AssertionError("face not found")


def step_a():
    """(a) a 96-sided circle on the ground, pushed up 20 mm."""
    ents = Entities()
    fids, v = circle(ents, GROUND, (0, 0, 0), 30.0, 96)
    ents.pushpull(new_face(ents, fids, v), -20.0)
    return ents


def step_b(ents):
    """(b) 20 circles of 24 sides on the top face, each pushed in 5 mm."""
    z = 20.0
    plane = Plane((0.0, 0.0, z), (0, 0, 1))
    for i in range(20):
        a = 2 * np.pi * i / 20
        c = np.array([24.0 * np.cos(a), 24.0 * np.sin(a), z])
        circle(ents, plane, c, 3.0, 24)
        ents.pushpull(face_at(ents, c), -5.0)
    return ents


def step_c():
    """(c) a 20x20 grid of 5x5 mm squares with 1 mm gaps, pulled up 1..5 mm."""
    ents = Entities()
    for iy in range(20):
        for ix in range(20):
            x0, y0 = ix * 6.0, iy * 6.0
            loop = np.array([[x0, y0], [x0 + 5, y0], [x0 + 5, y0 + 5], [x0, y0 + 5]])
            corner = ents.vertex(np.array([x0, y0, 0.0]))
            f = new_face(ents, ents.add_shape(GROUND, [loop]), corner)
            ents.pushpull(f, -float(1 + (ix + iy) % 5))
    return ents


def step_d(ents, n=200, seed=7):
    """(d) n inference calls on random screen points of the model."""
    doc = Document()
    doc.root = ents
    cam = Camera()
    cam.width, cam.height = 1280, 800
    cam.fit(ents.bbox())
    engine = InferenceEngine(doc, cam)
    rng = np.random.default_rng(seed)
    pts = rng.uniform([20.0, 20.0], [cam.width - 20.0, cam.height - 20.0], size=(n, 2))
    return [engine.infer(float(x), float(y)) for x, y in pts]


def step_e(ents, path):
    """(e) the solid of the model as an STL file."""
    man = ents.to_manifold()
    write_stl(path, man)
    return man


def main():
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bench_out.stl")
    res = {}

    t0 = time.perf_counter()
    ents = step_a()
    res["a"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    step_b(ents)
    res["b"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    grid = step_c()
    res["c"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    step_d(grid)
    res["d"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    step_e(grid, out)
    res["e"] = time.perf_counter() - t0
    os.remove(out)

    print(f"model (c): {len(grid.F)} faces, {len(grid.E)} edges, {len(grid.V)} vertices")
    for k, label in (("a", "cirkel 96 + pushpull 20 mm"),
                     ("b", "20x cirkel 24 op het bovenvlak, -5 mm"),
                     ("c", "raster 20x20 rechthoeken + optrekken"),
                     ("d", "200x infer op het raster"),
                     ("e", "to_manifold + STL-export")):
        print(f"({k}) {label:<40} {res[k]:7.3f} s")
    print(f"totaal                                   {sum(res.values()):7.3f} s")
    return res


if __name__ == "__main__":
    main()

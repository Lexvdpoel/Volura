"""Random stress test of the B-rep core.

Per seed (0..39), 6..12 random operations are done on an empty model
(rectangles, circles, lines, push/pull, erasing an edge). After every
operation the B-rep invariants must still hold, and if the model is closed
at that moment, it must be a valid solid.
"""
import os
import random
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from volura.brep import Entities  # noqa: E402
from volura.geom import Plane, circle_points, point_in_polys  # noqa: E402

GROUND = Plane((0, 0, 0), (0, 0, 1))
MIN_VERTEX_DIST = 1e-4   # mm: closer together is the same vertex
MAX_LOOP_OFF_PLANE = 1e-6
CIRCLE_SEGMENTS = 16


# ----------------------------------------------------------------- invariants
def check_invariants(e, ctx):
    V, E, F = e.V, e.E, e.F

    for eid, (a, b) in E.items():
        for v in (a, b):
            if v not in V:
                pytest.fail(f"{ctx}: edge {eid} refers to missing vertex {v}")
        if a == b:
            pytest.fail(f"{ctx}: edge {eid} starts and ends at vertex {a}")
        if float(np.linalg.norm(V[a] - V[b])) < 1e-9:
            pytest.fail(f"{ctx}: edge {eid} has zero length")

    seen = {}
    for eid, (a, b) in E.items():
        k = (a, b) if a < b else (b, a)
        if k in seen:
            pytest.fail(f"{ctx}: edges {seen[k]} and {eid} connect the same vertices {k}")
        seen[k] = eid

    if len(V) > 1:
        P = np.array([V[v] for v in V])
        d = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=2)
        iu = np.triu_indices(len(P), 1)
        worst = float(d[iu].min())
        if worst < MIN_VERTEX_DIST:
            pytest.fail(f"{ctx}: two vertices are only {worst:g} mm apart")

    for fid, face in F.items():
        for loop in face.loops:
            if len(loop) < 3:
                pytest.fail(f"{ctx}: face {fid} has a loop of {len(loop)} points")
            for v in loop:
                if v not in V:
                    pytest.fail(f"{ctx}: face {fid} refers to missing vertex {v}")
        pts = np.array([V[v] for loop in face.loops for v in loop])
        off = float(np.max(np.abs((pts - V[face.loops[0][0]]) @ face.normal)))
        if off > MAX_LOOP_OFF_PLANE:
            pytest.fail(f"{ctx}: face {fid} is {off:g} mm off its plane")


def check_closed_solid(e, ctx):
    """A model without open edges must be a valid solid."""
    if not e.F or e.open_edges():
        return
    m = e.to_manifold()
    status = m.status()
    if status.name != "NoError":
        pytest.fail(f"{ctx}: closed model gives manifold status {status}")
    if m.volume() <= 0:
        pytest.fail(f"{ctx}: closed model has volume {m.volume()}")


# ------------------------------------------------------------------ generators
def ground_plane(e):
    hit = e.faces_in_plane(GROUND)
    if hit:
        return Plane((0, 0, 0), e.face_plane(hit[0]).n)
    return GROUND


def face_polys(e, pl, fid):
    return [np.asarray(pl.to_local(L), dtype=float) for L in e.face_points(fid)]


def inside(polys, pts):
    return all(point_in_polys(polys, np.asarray(p, dtype=float)) for p in pts)


def rect_in_polys(rng, polys):
    """A rectangle on the 1 mm grid with all corners and edges inside `polys`."""
    P = np.concatenate(polys)
    lo, hi = np.floor(P.min(axis=0)) + 1, np.ceil(P.max(axis=0)) - 1
    if hi[0] - lo[0] < 4 or hi[1] - lo[1] < 4:
        return None
    for _ in range(15):
        x0 = rng.randint(int(lo[0]), int(hi[0]) - 3)
        x1 = rng.randint(x0 + 3, int(hi[0]))
        y0 = rng.randint(int(lo[1]), int(hi[1]) - 3)
        y1 = rng.randint(y0 + 3, int(hi[1]))
        corners = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=float)
        probes = list(corners)
        probes += [(corners[i] + corners[(i + 1) % 4]) / 2 for i in range(4)]
        probes += [corners.mean(axis=0)]
        if inside(polys, probes):
            return corners
    return None


def circle_in_polys(rng, polys):
    """A circle on the 1 mm grid centre lying completely inside `polys`."""
    P = np.concatenate(polys)
    lo, hi = np.floor(P.min(axis=0)) + 1, np.ceil(P.max(axis=0)) - 1
    if hi[0] - lo[0] < 8 or hi[1] - lo[1] < 8:
        return None
    rmax = min(8, int((hi[0] - lo[0]) // 2), int((hi[1] - lo[1]) // 2))
    if rmax < 3:
        return None
    for _ in range(15):
        r = rng.randint(3, rmax)
        cx = rng.randint(int(lo[0]) + r, int(hi[0]) - r)
        cy = rng.randint(int(lo[1]) + r, int(hi[1]) - r)
        loop = circle_points(r, CIRCLE_SEGMENTS) + np.array([cx, cy], dtype=float)
        if inside(polys, loop):
            return loop
    return None


def free_loop(rng, kind):
    if kind == "circle":
        return circle_points(rng.randint(3, 9), CIRCLE_SEGMENTS) + \
            np.array([rng.randint(-15, 15), rng.randint(-15, 15)], dtype=float)
    x0, y0 = rng.randint(-20, 15), rng.randint(-20, 15)
    w, h = rng.randint(3, 15), rng.randint(3, 15)
    return np.array([[x0, y0], [x0 + w, y0], [x0 + w, y0 + h], [x0, y0 + h]], dtype=float)


# ----------------------------------------------------------------- operations
def op_shape_ground(e, rng, kind):
    e.add_shape(ground_plane(e), [free_loop(rng, kind)])
    return f"add_shape({kind}) on ground"


def op_shape_face(e, rng, kind):
    fid = rng.choice(list(e.F))
    pl = e.face_plane(fid)
    polys = face_polys(e, pl, fid)
    loop = rect_in_polys(rng, polys) if kind == "rect" else circle_in_polys(rng, polys)
    if loop is None:
        e.add_shape(ground_plane(e), [free_loop(rng, kind)])
        return f"add_shape({kind}) on face {fid} (did not fit: on ground)"
    e.add_shape(pl, [loop])
    return f"add_shape({kind}) on face {fid}"


def op_line(e, rng):
    vs = list(e.V)
    for _ in range(25):
        a, b = rng.choice(vs), rng.choice(vs)
        if a != b and float(np.linalg.norm(e.V[a] - e.V[b])) >= 1.0:
            e.add_line(e.V[a], e.V[b])
            return f"add_line({a} -> {b})"
    return "add_line (no pair of vertices 1 mm apart)"


def op_pushpull(e, rng):
    fid = rng.choice(list(e.F))
    d = 0
    while d == 0:
        d = rng.randint(-5, 20)
    e.pushpull(fid, float(d))
    return f"pushpull(face {fid}, {d})"


def op_erase(e, rng):
    eid = rng.choice(list(e.E))
    e.erase_edges([eid])
    return f"erase_edges({eid})"


CHOICES = [
    (5, "rect", op_shape_ground),
    (3, "circle", op_shape_ground),
    (4, "rect", op_shape_face),
    (3, "circle", op_shape_face),
    (5, None, op_line),
    (6, None, op_pushpull),
    (2, None, op_erase),
]


def run_ops(e, rng, seed, n):
    for i in range(n):
        cands = [(w, k, fn) for (w, k, fn) in CHOICES
                 if fn is not op_shape_face or e.F
                 if fn is not op_pushpull or e.F
                 if fn is not op_erase or e.E
                 if fn is not op_line or len(e.V) > 1]
        _, kind, fn = rng.choices(cands, weights=[c[0] for c in cands], k=1)[0]
        desc = fn.__name__ + (f" {kind}" if kind else "")
        try:
            desc = fn(e, rng, kind) if fn in (op_shape_ground, op_shape_face) else fn(e, rng)
            check_invariants(e, f"seed {seed} after op {i} ({desc})")
            check_closed_solid(e, f"seed {seed} after op {i} ({desc})")
        except AssertionError:
            raise
        except Exception as ex:  # noqa: BLE001 - seed and op belong in the report
            raise AssertionError(
                f"seed {seed} op {i} ({desc}): {type(ex).__name__}: {ex}") from ex


def run_seed(seed, n=None):
    rng = random.Random(seed)
    e = Entities()
    run_ops(e, rng, seed, rng.randint(6, 12) if n is None else n)
    check_invariants(e, f"seed {seed} at the end")
    check_closed_solid(e, f"seed {seed} at the end")
    return e


@pytest.mark.parametrize("seed", list(range(40)))
def test_fuzz_brep(seed):
    run_seed(seed)


# Seeds that only show up with more than 12 operations; the bugs are fixed
# in volura/brep.py and stay recorded here.
#   10146: a side face of a push/pull was 3.3e-4 mm out of plane
REGRESSION = [(10146, 15)]


@pytest.mark.parametrize("seed,n", REGRESSION)
def test_regression_from_extended_fuzz(seed, n):
    run_seed(seed, n)


if __name__ == "__main__":
    bad = 0
    for s in range(40):
        try:
            run_seed(s)
        except Exception as ex:  # noqa: BLE001
            bad += 1
            print("FAIL", s, type(ex).__name__, str(ex)[:200])
    print("failures:", bad)

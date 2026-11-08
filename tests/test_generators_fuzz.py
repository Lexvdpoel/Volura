"""Randomised stress test of the Qt-free generators.

`gears.spur_gear`, `boxgen.vent_pattern`, `boxgen.living_hinge`, `sheettools.holding_tabs`
and `sheettools.polar_angles` are pure functions, so they can be driven with random input.
Per seed `numpy.random.default_rng` draws a wheel, a grille, a panel of slits, a contour
with tabs and a circular array; what the tests of those generators state for one example
must hold for every one of them. The outline of a wheel is a simple counter-clockwise loop
between the root and the tip circle whose CrossSection area is its shoelace area, and its
bore lies inside the root circle. Every hole of a grille stands inside the contour shrunk by
the margin, no two holes overlap and two neighbours keep `web` of material between them; a
contour that runs the other way round gives the same holes. The pieces of holding tabs add up
to the perimeter less the tabs, are open, are not zero long and lie on the contour. A polar
array starts on zero and rises strictly to its total. No slit of a living hinge crosses the
margin and none has zero length.

The seeds are fixed, so a failure can be reproduced. Whether two edges of a wheel cross each
other is quadratic in the number of points, so that check only runs on wheels of at most 60
teeth; the wheels are sampled with a slightly coarser chord tolerance, which leaves the
root and the tip circle they must stay between alone.

Run with:  python -m pytest tests/test_generators_fuzz.py
"""
import math
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import manifold3d as mf  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from volura import boxgen, cam, gears  # noqa: E402
from volura.sheettools import circle_loop, holding_tabs, polar_angles  # noqa: E402

SEEDS = list(range(24))
GEAR_SEEDS = list(range(16))
CASES = 2                # cases per seed of the cheap generators
SIMPLE_MAX = 60          # above this many teeth the crossing check is too slow
GEAR_TOL = 0.05          # coarser than the default, so a wheel of 300 teeth stays quick
AREA_TOL = 1e-6          # relative: a CrossSection area against the shoelace area


# ------------------------------------------------------------------- measuring
def area(loop):
    """The shoelace area of a loop: positive counter-clockwise, negative clockwise."""
    p = np.asarray(loop, dtype=float)
    x, y = p[:, 0], p[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def section(*loops):
    return mf.CrossSection([np.ascontiguousarray(np.asarray(q, dtype=float)) for q in loops],
                           mf.FillRule.NonZero)


def radii(loop, center=(0.0, 0.0)):
    p = np.asarray(loop, dtype=float) - np.asarray(center, dtype=float)
    return np.hypot(p[:, 0], p[:, 1])


def perimeter(pts):
    """Length of the closed loop `pts` (last point back to the first)."""
    q = np.asarray(pts, dtype=float)
    return float(np.sum(np.hypot(*(np.diff(np.vstack([q, q[:1]]), axis=0)).T)))


def piece_length(piece):
    return float(np.sum(np.hypot(*(np.diff(np.asarray(piece, dtype=float), axis=0)).T)))


def on_contour(pts, loop, tol) -> bool:
    """Does every point of `pts` lie within `tol` of the closed loop `loop`?"""
    a = np.asarray(loop, dtype=float)
    d = np.roll(a, -1, axis=0) - a
    L2 = np.maximum(np.sum(d * d, axis=1), 1e-30)
    q = np.asarray(pts, dtype=float)[:, None, :] - a[None, :, :]
    t = np.clip(np.einsum("mnj,nj->mn", q, d) / L2, 0.0, 1.0)
    r = q - t[:, :, None] * d[None, :, :]
    return bool(np.sqrt(np.einsum("mnj,mnj->mn", r, r)).min(axis=1).max() <= tol)


def self_crossings(loop) -> int:
    """How many pairs of edges of the closed loop `loop` cross each other.

    Edges that share an end are left out: the contour touches itself there by definition.
    """
    p = np.asarray(loop, dtype=float)
    a, b = p, np.roll(p, -1, axis=0)
    lo, hi = np.minimum(a, b), np.maximum(a, b)
    d = b - a
    n = len(p)
    hits = 0
    for k in range(n):
        j = np.flatnonzero((lo[:, 0] <= hi[k, 0]) & (hi[:, 0] >= lo[k, 0])
                           & (lo[:, 1] <= hi[k, 1]) & (hi[:, 1] >= lo[k, 1])
                           & (np.arange(n) > k))
        j = j[(j != (k + 1) % n) & (j != (k - 1) % n)]
        if not len(j):
            continue
        q = a[j] - a[k]
        den = d[k, 0] * d[j, 1] - d[k, 1] * d[j, 0]
        ok = np.abs(den) > 1e-12
        t = (q[:, 0] * d[j, 1] - q[:, 1] * d[j, 0]) / np.where(ok, den, 1.0)
        u = (q[:, 0] * d[k, 1] - q[:, 1] * d[k, 0]) / np.where(ok, den, 1.0)
        hits += int(np.count_nonzero(ok & (t > 1e-9) & (t < 1.0 - 1e-9)
                                     & (u > 1e-9) & (u < 1.0 - 1e-9)))
    return hits


def outline(rng):
    """A random closed contour: a star (often non-convex), a rectangle or a circle."""
    kind = int(rng.integers(0, 3))
    if kind == 1:
        w, h = rng.uniform(40.0, 120.0, 2)
        x0, y0 = rng.uniform(-40.0, 40.0, 2)
        return np.array([[x0, y0], [x0 + w, y0], [x0 + w, y0 + h], [x0, y0 + h]])
    if kind == 2:
        return circle_loop(rng.uniform(-30.0, 30.0, 2), float(rng.uniform(30.0, 60.0)))
    n = int(rng.integers(5, 13))                    # a star of 5..12 points
    r_out = float(rng.uniform(45.0, 90.0))
    r_in = r_out * float(rng.uniform(0.55, 0.95))   # below about 0.8 the contour is non-convex
    a = np.linspace(0.0, 2.0 * math.pi, 2 * n, endpoint=False) + rng.uniform(0.0, math.pi)
    rad = np.where(np.arange(2 * n) % 2 == 0, r_out, r_in)
    return np.column_stack([rad * np.cos(a), rad * np.sin(a)])


# ---------------------------------------------------------------- the spur gear
def check_wheel(loop, z, m, ctx, tol=GEAR_TOL):
    """The outline of a wheel: counter-clockwise, whole between the two circles."""
    r, rf = 0.5 * m * z, 0.5 * m * z - 1.25 * m
    assert area(loop) > 0.0, f"{ctx}: the outline of the wheel is not counter-clockwise"
    assert section(loop).area() == pytest.approx(area(loop), rel=AREA_TOL, abs=1e-6), \
        f"{ctx}: the CrossSection area is not the shoelace area of the outline"
    rr = radii(loop)
    assert rr.max() <= r + m + tol, f"{ctx}: a tooth stands above the tip circle"
    assert rr.min() >= rf - tol, f"{ctx}: a tooth dips below the root circle"


@pytest.mark.parametrize("seed", GEAR_SEEDS)
def test_spur_gear_of_random_wheels(seed):
    rng = np.random.default_rng(seed)
    z = int(rng.integers(gears.TEETH_MIN, gears.TEETH_MAX + 1))
    m = float(rng.uniform(gears.MODULE_MIN, gears.MODULE_MAX))
    pa = float(rng.uniform(gears.ANGLE_MIN, gears.ANGLE_MAX))
    play = float(rng.uniform(0.0, 0.3))
    bore = float(rng.uniform(0.0, m * (z - 4.5)))   # the largest bore the wheel allows
    ctx = (f"seed {seed}: {z} teeth, module {m:g}, angle {pa:g}, "
           f"backlash {play:g}, bore {bore:g}")
    loops = gears.spur_gear(z, module=m, pressure_angle=pa, backlash=play, bore=bore,
                            tol=GEAR_TOL)
    outer = loops[0]
    check_wheel(outer, z, m, ctx)
    if z <= SIMPLE_MAX:
        crossed = self_crossings(outer)
        assert crossed == 0, f"{ctx}: {crossed} pairs of edges of the outline cross each other"
    if not bore:
        return
    hole = loops[1]
    assert area(hole) < 0.0, f"{ctx}: the bore is not clockwise"
    assert radii(hole).max() < 0.5 * m * z - 1.25 * m, \
        f"{ctx}: the bore is not inside the root circle"
    assert section(outer, hole).area() == pytest.approx(section(outer).area() + area(hole),
                                                        rel=AREA_TOL, abs=1e-6), \
        f"{ctx}: the bore is not a hole in the wheel"


@pytest.mark.parametrize("z, m, pa", (
    (6, 0.3, 14.5), (6, 0.3, 30.0), (6, 20.0, 14.5), (6, 20.0, 30.0),
    (300, 0.3, 14.5), (300, 0.3, 30.0), (300, 20.0, 14.5), (300, 20.0, 30.0)))
def test_spur_gear_on_the_limits_of_the_input(z, m, pa):
    outer = gears.spur_gear(z, module=m, pressure_angle=pa, bore=0.5 * m * (z - 4.5),
                            tol=GEAR_TOL)[0]
    check_wheel(outer, z, m, f"{z} teeth, module {m}, angle {pa}")


# ---------------------------------------------------------------- vent pattern
def as_sets(holes):
    """The holes as a sorted list of rounded points, so two patterns can be compared."""
    return sorted(np.round(q, 9).tolist() for q in holes)


def check_grille(pts, shape, size, web, margin, ctx):
    """The holes of a grille: inside the margin, whole and with the web between them."""
    holes = boxgen.vent_pattern(pts, shape=shape, size=size, web=web, margin=margin)
    if not holes:
        return holes
    room = cam.offset_loop(pts, -margin)
    if room:
        outside = mf.CrossSection.batch_boolean([section(*holes), section(*room)],
                                                mf.OpType.Subtract)
        assert outside.area() < 1e-6, \
            f"{ctx}: a hole stands outside the contour shrunk by the margin"
    # the centres stand a step size + web apart on the grid, so the material between two
    # neighbours is everywhere `web` wide
    c = np.array([h.mean(axis=0) for h in holes])
    d = np.hypot(c[:, None, 0] - c[None, :, 0], c[:, None, 1] - c[None, :, 1])
    np.fill_diagonal(d, np.inf)
    assert float(d.min()) >= size + web - 1e-6, \
        f"{ctx}: two holes have less than the web between them"
    assert sum(abs(area(h)) for h in holes) - section(*holes).area() < 1e-6, \
        f"{ctx}: two holes lie over each other"
    return holes


@pytest.mark.parametrize("seed", SEEDS)
def test_vent_pattern_of_random_outlines(seed):
    rng = np.random.default_rng(1000 + seed)
    pts = outline(rng)
    shape = "hex" if seed % 2 else "round"
    size = float(rng.uniform(3.0, 8.0))
    web = float(rng.uniform(1.0, 3.0))
    margin = float(rng.uniform(0.5, 3.5))
    ctx = f"seed {seed}: {shape} holes of {size:g}, web {web:g}, margin {margin:g}"
    holes = check_grille(pts, shape, size, web, margin, ctx)
    assert holes, f"{ctx}: a panel this big holds no hole at all"
    back = check_grille(pts[::-1], shape, size, web, margin, f"{ctx}, clockwise contour")
    assert as_sets(back) == as_sets(holes), f"{ctx}: the direction of the contour changes it"


# ----------------------------------------------------------------- holding tabs
@pytest.mark.parametrize("seed", SEEDS)
def test_holding_tabs_of_random_contours(seed):
    rng = np.random.default_rng(2000 + seed)
    for _ in range(CASES):
        pts = outline(rng)
        per = perimeter(pts)
        count = int(rng.integers(1, 13))
        width = float(rng.uniform(0.05, 0.4 * per / count))   # the tabs leave room to cut
        pieces = holding_tabs(pts, count, width)
        ctx = f"seed {seed}: {count} tabs of {width:g} mm on a contour of {per:g} mm"
        assert len(pieces) == count, f"{ctx}: not one piece between every two tabs"
        assert sum(piece_length(q) for q in pieces) == pytest.approx(per - count * width,
                                                                     rel=1e-9, abs=1e-9), \
            f"{ctx}: the pieces are not the perimeter less the tabs"
        for q in pieces:
            assert len(q) >= 2, f"{ctx}: a piece of one point is nothing to cut"
            assert piece_length(q) > 1e-9, f"{ctx}: a piece has zero length"
            assert on_contour(q, pts, 1e-9 * max(1.0, per)), f"{ctx}: a piece leaves the contour"


# ---------------------------------------------------------------- polar angles
@pytest.mark.parametrize("seed", SEEDS)
def test_polar_angles_of_random_arrays(seed):
    rng = np.random.default_rng(3000 + seed)
    for _ in range(4 * CASES):
        count = int(rng.integers(2, 401))
        total = 360.0 if rng.random() < 0.5 else float(rng.uniform(0.1, 720.0))
        ctx = f"seed {seed}: {count} copies over {total:g}°"
        a = polar_angles(count, total)
        assert len(a) == count, f"{ctx}: not every copy has an angle"
        assert a[0] == 0.0, f"{ctx}: the first copy is the original itself"
        assert all(b > c for c, b in zip(a, a[1:])), f"{ctx}: the angles do not rise"
        if abs(total - 360.0) <= 1e-9:
            assert a[-1] < 360.0, f"{ctx}: the last copy stands on the first again"
        else:
            assert a[-1] == pytest.approx(total), f"{ctx}: the arc does not end on its total"


# ---------------------------------------------------------------- living hinge
@pytest.mark.parametrize("seed", SEEDS)
def test_living_hinge_of_random_panels(seed):
    rng = np.random.default_rng(4000 + seed)
    for _ in range(CASES):
        w, h = rng.uniform(20.0, 200.0, 2)
        margin = float(rng.uniform(0.0, 0.15 * min(w, h)))
        vertical = bool(rng.integers(0, 2))
        across, along = (w, h) if vertical else (h, w)
        pitch = (across - 2.0 * margin) * float(rng.uniform(0.05, 0.9))
        inner = along - 2.0 * margin
        gap = float(rng.uniform(0.2, 0.1 * inner))
        slit = float(rng.uniform(2.0 * gap, 0.8 * inner))
        ctx = (f"seed {seed}: {w:g} x {h:g} mm, slit {slit:g}, gap {gap:g}, "
               f"pitch {pitch:g}, margin {margin:g}")
        slits = boxgen.living_hinge(0.0, 0.0, w, h, slit=slit, gap=gap, pitch=pitch,
                                    margin=margin, vertical=vertical)
        assert slits, f"{ctx}: a panel with room for two columns holds no slit"
        for s in slits:
            assert float(np.hypot(*(s[1] - s[0]))) > 0.0, f"{ctx}: a slit has zero length"
            for q in s:
                inside = (margin - 1e-9 <= q[0] <= w - margin + 1e-9
                          and margin - 1e-9 <= q[1] <= h - margin + 1e-9)
                assert inside, f"{ctx}: a slit crosses the margin"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))

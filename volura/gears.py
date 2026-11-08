"""Spur gears: the computational core for cutting a gear on the worksheet.

No Qt in this file. It gives the closed contours of an involute spur gear as plain numpy
loops, so it can be tested on its own and reused in the worksheet, the export and the 3D
model.

A **spur gear** is a wheel of straight teeth that stand radially in the rim and mesh with
another wheel in the same plane. The shape of a flank is not a matter of taste: the right
shape is the **involute**, the curve that a taut string describes while it unwinds from
the base circle. Two wheels with the same module and the same pressure angle then drive
each other at a constant speed, in every position, because the flanks touch each other on
one straight line, the line of action, which runs along the pitch circle at the pressure
angle. Round or cycloidal flanks only run evenly in one position; an involute runs evenly
also when the axes stand a little too far from each other, which is what a printed or cut
gear needs.

The size comes from the **module** `m`, the pitch diameter per tooth: the pitch radius is
`m × z / 2`, a tooth stands `m` above the pitch circle (the addendum) and the bottom of
the space between two teeth lies `1.25 × m` below it (the dedendum), so that the tooth of
the one wheel does not touch the root of the other. The base circle, from which the
involute unwinds, is the pitch radius times the cosine of the pressure angle; where the
base circle lies outside the root, the flank goes straight down to the root circle.

The pitch circle is also where `backlash` is measured: the play a pair of gears needs so
that the flanks do not press on each other and the oil has a place. Every tooth is thinned
by that amount, half on each flank, which is how a gear is cut for a loose axle.
"""
from __future__ import annotations

import math

import numpy as np

from .i18n import tr
from .vector2d import arc_points

#: the range of the input fields; below six teeth the tooth eats the whole root
TEETH_MIN, TEETH_MAX = 6, 300
MODULE_MIN, MODULE_MAX = 0.3, 20.0
ANGLE_MIN, ANGLE_MAX = 14.5, 30.0
#: half a tooth in radians below which the tip is a point instead of a flat
POINTED = 1e-4


def pitch_diameter(teeth, module) -> float:
    """The pitch diameter of a gear: the circle on which it rolls without slipping."""
    return float(teeth) * float(module)


def center_distance(t1, t2, module) -> float:
    """The distance between the axes of two gears of `t1` and `t2` teeth."""
    return 0.5 * (float(t1) + float(t2)) * float(module)


def _involute(a) -> float:
    """The involute function tan(a) − a: how far the flank has turned at angle `a`."""
    return math.tan(a) - a


def _sample(f, t0, t1, tol, out, depth=0):
    """The curve `f` between `t0` and `t1` as points in `out`, with every chord near it.

    A chord is subdivided as long as the curve stands more than `tol` from its midpoint,
    the same rule that `arc_points` uses for a circle, so a flank is as finely cut as an
    arc of the same height.
    """
    a, b, m = f(t0), f(t1), f(0.5 * (t0 + t1))
    d = b - a
    L2 = float(d @ d)
    off = float(np.linalg.norm(m - a - np.clip(((m - a) @ d) / L2, 0.0, 1.0) * d)) \
        if L2 > 0.0 else 0.0
    if depth < 20 and off > tol:
        tm = 0.5 * (t0 + t1)
        _sample(f, t0, tm, tol, out, depth + 1)
        _sample(f, tm, t1, tol, out, depth + 1)
        return
    out.append(m)


def _loop(pts) -> np.ndarray:
    """The points of a closed contour, without consecutive duplicates and without a
    closing point: the worksheet closes the contour itself."""
    out: list[np.ndarray] = []
    for q in np.asarray(pts, dtype=float).reshape(-1, 2):
        if out and np.allclose(out[-1], q, rtol=0.0, atol=1e-9):
            continue
        out.append(np.asarray(q, dtype=float))
    while len(out) > 1 and np.allclose(out[0], out[-1], rtol=0.0, atol=1e-9):
        out.pop()
    return np.ascontiguousarray(np.asarray(out, dtype=float).reshape(-1, 2))


def spur_gear(teeth, module=2.0, pressure_angle=20.0, bore=0.0, backlash=0.1,
              center=(0, 0), tol=0.02) -> list[np.ndarray]:
    """The closed contours of an involute spur gear: the wheel, and its bore when asked.

    The first loop is the outline of the wheel, counter-clockwise: per tooth the involute
    flank up to the tip circle, an arc over the tip, the flank back down and an arc over
    the root circle to the next tooth. The second loop, when `bore` is greater than zero,
    is the bore, clockwise, so the two together are a wheel with a hole in it.

    `teeth` is the number of teeth and `module` the pitch diameter per tooth in
    millimetres, so the pitch diameter is `teeth × module`. `pressure_angle` is the angle
    of the line of action, usually 20 degrees. `backlash` is the play of a pair of gears,
    in millimetres measured on the pitch circle; every tooth is thinned by that amount,
    half on each flank. `center` shifts the gear on the sheet and `tol` is how far a
    chord of the flank may stand from the true curve, in millimetres.
    """
    z = int(teeth)
    if z != teeth or z < TEETH_MIN or z > TEETH_MAX:
        raise ValueError(tr("The number of teeth must be between 6 and 300."))
    m = float(module)
    if m < MODULE_MIN or m > MODULE_MAX:
        raise ValueError(tr("The module must be between 0.3 and 20 mm."))
    pa = float(pressure_angle)
    if pa < ANGLE_MIN or pa > ANGLE_MAX:
        raise ValueError(tr("The pressure angle must be between 14.5 and 30 degrees."))
    d_bore = float(bore)
    if d_bore < 0.0:
        raise ValueError(tr("The bore may not be negative."))
    play = float(backlash)
    if play < 0.0:
        raise ValueError(tr("The backlash may not be negative."))
    r = 0.5 * m * z                       # pitch circle
    ra, rf = r + m, r - 1.25 * m          # tip and root circle
    phi = math.radians(pa)
    rb = r * math.cos(phi)                # base circle, the involute unwinds from here
    if d_bore >= 2.0 * rf - 2.0 * m:
        raise ValueError(tr("The bore leaves too little material between the hole "
                            "and the teeth."))
    c = np.asarray(center, dtype=float).reshape(2)
    tol = max(1e-6, float(tol))
    # half a tooth on the pitch circle, minus the half of the backlash on that flank
    half = math.pi / (2.0 * z) - 0.5 * play / r
    if half <= POINTED:
        raise ValueError(tr("The backlash leaves nothing of the teeth."))
    k = half + _involute(phi)             # half a tooth at radius ρ: k − involute(α)

    # the tip circle can fall above the point of the tooth: then the wheel is cut where
    # the tooth becomes pointed, so the outline never crosses itself
    a_tip = math.acos(min(1.0, rb / ra))
    if k - _involute(a_tip) < POINTED:
        lo, hi = 0.0, a_tip
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if k - _involute(mid) >= POINTED else (lo, mid)
        a_tip = lo
    ra = min(ra, rb / math.cos(a_tip))
    a_root = 0.0 if rb >= rf else math.acos(rb / rf)   # no involute below the base circle
    r_start = max(rb, rf)
    b_start = k - _involute(a_root)      # half a tooth where the flank begins
    b_tip = k - _involute(a_tip)

    def flank(psi, sign, a):
        rho = rb / math.cos(a)
        ang = psi + sign * (k - _involute(a))
        return rho * np.array([math.cos(ang), math.sin(ang)])

    pts: list[np.ndarray] = []
    for i in range(z):
        psi = 2.0 * math.pi * i / z
        pts += list(arc_points(np.zeros(2), rf, psi - 2.0 * math.pi / z + b_start,
                               psi - b_start, tol)[:-1])
        if r_start > rf:
            pts.append(rf * np.array([math.cos(psi - b_start), math.sin(psi - b_start)]))
        flank_pts: list[np.ndarray] = []
        _sample(lambda a: flank(psi, -1.0, a), a_root, a_tip, tol, flank_pts)
        pts += [r_start * np.array([math.cos(psi - b_start), math.sin(psi - b_start)])]
        pts += flank_pts
        pts += list(arc_points(np.zeros(2), ra, psi - b_tip, psi + b_tip, tol)[:-1])
        flank_pts = []
        _sample(lambda a: flank(psi, +1.0, a), a_tip, a_root, tol, flank_pts)
        pts += [ra * np.array([math.cos(psi + b_tip), math.sin(psi + b_tip)])]
        pts += flank_pts
        pts += [r_start * np.array([math.cos(psi + b_start), math.sin(psi + b_start)])]
        if r_start > rf:
            pts.append(rf * np.array([math.cos(psi + b_start), math.sin(psi + b_start)]))
    loops = [_loop(np.asarray(pts, dtype=float) + c)]
    if d_bore > 0.0:
        loops.append(arc_points(c, 0.5 * d_bore, 2.0 * math.pi, 0.0, tol)[:-1])
    return loops

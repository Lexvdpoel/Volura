"""Print-in-place ball joints: the computational core for a ball that prints caught in its socket.

No Qt in this file. It gives the joint as two closed volumes that are already assembled, so it can
be tested on its own and reused in the model and the export.

A **print-in-place ball joint** is a ball joint that comes off the printer ready to turn: the
socket, the ball, the stem and the little foot the ball stands on during the print all print in one
go, and the joint moves as soon as it is off the bed. It is used on a printed lamp, an arm, a
camera holder or a model whose part has to point any way it is put, and there is nothing to press
in, to screw or to glue.

The **socket** is what is mounted: a floor that lies flat on the bed, a wall of `wall` millimetres
around it and a short neck on top. Its inside is a bowl — the **seat** — that is the ball with
`clearance` millimetres of play all round, cut away above so the ball can be pressed in and the
stem can come out. The bowl ends in the **neck**: a ring of the same radius as the widest circle of
the bowl that is still printable, which is what holds the ball. How much of the ball hides behind
that ring is the **capture**; without capture the ball is simply a ball in a cup and falls out.

The **ball** is what turns: a sphere of `ball_d` millimetres with a **stem** of `stem_d`
millimetres out of its top, which is what the joint drives. The stem tilts in a cone of `swing`
degrees to either side of upright, and the neck is made wide enough for the stem to do that and
narrow enough to keep the ball. A fat stem needs a wide neck, and a wide neck holds little of the
ball, so the stem is what sets how far a joint of this kind can turn: with a stem of half the ball
there is nothing left to hold the ball, and the swing asked for then becomes the largest swing that
still holds.

The **tip** is what makes the ball print at all: a ball lies on one point, and a sphere wider than
a few millimetres needs support under its widest circle. So the ball carries a slender foot,
`FOOT` millimetres in radius where it stands, down to the bed at a slope that prints, and the joint
is broken off that foot
afterwards with a twist — the same way a model maker breaks a model off its sprue. The socket's
floor carries the **relief** that matches that foot: the room the foot and the lower ball sweep
when the stem tilts, with `clearance` millimetres of play on it, so the joint turns to the swing
asked for and then stops against the neck.

The **grip slits** are what holds a position: `grip` slots sawn in the wall of the socket down from
its top, which turn the wall into springs that press the neck closed around the ball. With `grip`
at 0 the joint stays loose and falls to whatever position it is put in; with slits it has to be
moved with a little force and then stays where it is moved to.

The joint stands on the bed exactly as it prints: the socket's floor and the ball's foot on the bed,
the ball in its seat and the stem upright. A size that cannot be printed becomes the nearest one
that can, like the other parts, and `sizes` says what a set of numbers really comes down to.
"""
from __future__ import annotations

import math

import manifold3d as mf

#: the smallest ball that is worth a joint and the largest that prints in one go
BALL_MIN, BALL_MAX = 8.0, 40.0
#: the thinnest stem that prints and the thickest that is worth putting through a neck
STEM_MIN, STEM_MAX = 1.6, 16.0
#: the least play that survives printing and the most that still leaves a neck worth having
CLEARANCE_MIN, CLEARANCE_MAX = 0.15, 0.6
#: the least tilt that is worth a joint and the most a ball of these sizes can reach
SWING_MIN, SWING_MAX = 5.0, 60.0
#: the thinnest wall that prints and the thickest that is still a socket
WALL_MIN, WALL_MAX = 1.6, 8.0
#: the thinnest floor under the seat and the thickest that is still worth printing. A big ball
#: stands high off the bed, because its tip has to reach the bed at a printable slope; `sizes`
#: says what the floor of a joint of these numbers really is.
BASE_MIN, BASE_MAX = 1.0, 14.0
#: the shortest stem that is worth gripping and the longest that prints straight
STEM_LEN_MIN, STEM_LEN_MAX = 4.0, 60.0
#: no slits at all (a loose joint) and the most that leave a wall between them
GRIP_MIN, GRIP_MAX = 0, 8
#: the default: four springs, one per quarter of the socket, which is what holds a position
GRIP_DEFAULT = 4
#: the steepest overhang that prints without support, in degrees from the vertical
PRINT_LIMIT = 45.0
#: degrees above the printable circle of the seat that the tip is taken to, so the flat faces the
#: ball itself is made of, just above where the tip runs into it, stay on the printable side too
PRINT_MARGIN = 3.0
#: the steepest slope the tip may stand in, so the tip prints and still breaks off cleanly
TIP_LIMIT = 40.0
#: mm of ball that has to hide behind the neck, so the ball is caught instead of retained by luck
CAPTURE_MIN = 0.6
#: mm of neck above the seat, the ring that presses on the ball
NECK = 0.6
#: mm of bed between the wall of the socket and the edge of the floor
LIP = 2.0
#: mm in radius: the foot the ball stands on while it prints, broken off afterwards
FOOT = 0.8
#: mm the seat is made roomier than the promise, so the printed ball is never forced
SEAT_SLACK = 0.03
#: degrees of tilt past the swing that the relief is swept for, so the stop is past the promise
RELIEF_SLACK = 1.0
#: mm wide the grip slits are sawn
SLIT = 1.2
#: mm the relief is dug past the seat where it runs out of room of its own
RELIEF_DEPTH = 0.15
#: the straight pieces a ball, a seat and a tip are approximated with
SEGMENTS = 64
#: degrees of polar angle the relief is swept in
RELIEF_STEP = 1.0


def _clamp(value, lo, hi):
    """`value` between `lo` and `hi`: a size that cannot be printed becomes the nearest one that
    can, so a joint is never refused because of a number that was typed in."""
    return max(lo, min(hi, float(value)))


def _psi(x, y, cz):
    """The polar angle of a profile point `x, y` about the centre `(0, cz)`, in degrees from
    straight down, so that 0 is the lowest point of the seat and 90 is its widest."""
    return math.degrees(math.atan2(x, cz - y))


def _radial_hit(psi, edges, cz):
    """How far out a ray from `(0, cz)` at that polar angle runs through the profile's `edges`."""
    t = math.radians(psi)
    dx, dy = math.sin(t), -math.cos(t)
    hits = []
    for (x1, y1), (x2, y2) in edges:
        ax, ay = x1, y1 - cz
        ex, ey = x2 - x1, y2 - y1
        den = dx * ey - dy * ex
        if abs(den) < 1e-12:
            continue
        reach = (ax * ey - ay * ex) / den
        u = (ax * dy - ay * dx) / den
        if reach > 1e-9 and -1e-9 <= u <= 1.0 + 1e-9:
            hits.append(reach)
    return max(hits) if hits else None


def _relief_profile(quad, cz, swing, play, seat_r, extra=RELIEF_SLACK, step=RELIEF_STEP):
    """The profile of the relief the foot and the low ball sweep when the joint tilts.

    `quad` is the profile of the tip, `swing` the half-angle it turns through, `play` the room that
    is left on it and `seat_r` the radius of the seat it runs into. The relief is the room the tip
    takes at every tilt up to `swing` and `extra` degrees past it, `play` clear all round, and where
    that runs into the seat the relief stops of its own accord: the seat already gives the tip room
    there. The `extra` degrees are why the joint turns a little past the swing it promises before the
    neck takes hold of it.
    """
    edges = [(quad[i], quad[(i + 1) % len(quad)]) for i in range(len(quad))]
    top_psi = max(_psi(x, y, cz) for x, y in quad)
    seen = []
    psi = 0.0
    while psi <= top_psi + 1e-9:
        hit = _radial_hit(psi, edges, cz)
        if hit is not None:
            seen.append((psi, hit))
        psi += step / 3.0
    def window(width, at):
        return [r for (a, r) in seen if abs(a - at) <= width]

    lowest = seat_r - RELIEF_DEPTH
    walls, psi = [], 0.0
    stop = top_psi + swing + extra + math.degrees(play / max(1.0, lowest))
    while psi <= stop + 1e-9:
        near = window(swing + extra, psi)
        if near:
            reach = max(near) + play
            if reach <= lowest:
                break
            lean = math.degrees(play / max(1.0, reach))
            wall = max(max(window(swing + extra + lean, psi)) + play, lowest)
            t = math.radians(psi)
            walls.append((wall * math.sin(t), cz - wall * math.cos(t)))
        psi += step
    return [(0.0, cz)] + walls


def _neck_edge(R, rs, tilt, c):
    """The polar angle of the seat's edge: the circle the bowl of the seat narrows into, wide enough
    for a stem of radius `rs` to pass at a tilt of `tilt` radians with `c` of play on it, no steeper
    to the bed than the printable limit and no wider than the widest circle of the seat itself."""
    open_edge = tilt + math.asin(min(1.0, (rs + NECK * math.sin(tilt) + c) / R))
    return min(math.radians(90.0), max(math.radians(PRINT_LIMIT), open_edge))


def _max_swing(R, Rb, rs, c):
    """The widest half-angle in degrees that still leaves CAPTURE_MIN millimetres of ball behind
    the neck: the neck has to open far enough for a tilted stem and stay narrow enough to catch."""

    def capture(swing):
        return Rb - R * math.sin(_neck_edge(R, rs, math.radians(swing), c))

    lo, hi = SWING_MIN, SWING_MAX
    if capture(hi) >= CAPTURE_MIN:
        return hi
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if capture(mid) >= CAPTURE_MIN:
            lo = mid
        else:
            hi = mid
    return lo


def _max_stem(R, Rb, c):
    """The thickest stem in millimetres that leaves CAPTURE_MIN millimetres of ball behind the neck
    at the smallest swing: a stem past that swallows the neck, and nothing holds the ball."""

    def capture(rs):
        return Rb - R * math.sin(_neck_edge(R, rs, math.radians(SWING_MIN), c))

    lo, hi = 0.5 * STEM_MIN, 0.5 * STEM_MAX
    if capture(hi) >= CAPTURE_MIN:
        return 2.0 * hi
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if capture(mid) >= CAPTURE_MIN:
            lo = mid
        else:
            hi = mid
    return 2.0 * lo


def sizes(ball_d=12.0, clearance=0.3, stem_d=3.0, swing=35.0, wall=2.5, base=3.0,
          stem_len=12.0, grip=GRIP_DEFAULT) -> dict:
    """What a joint of these numbers really is, in millimetres and degrees.

    Everything asked for comes back clamped to what prints, and beside it the numbers that follow
    from it: `seat_r` the radius of the bowl the ball lies in, `centre` the height of the ball's
    centre above the bed, `opening` the diameter of the hole the stem comes out through, `capture`
    the millimetres of ball that hide behind it, `swing` how far the stem tilts to either side — what
    was asked for, or as much of it as these numbers still hold, which `swing_asked` says — `floor`
    the thickness of the socket's floor, which is thicker than the `base` asked for when a big ball
    needs a shallower tip, and `height` and `base_d` the height of the stem's top and the width of
    the whole joint.
    """
    Rb = 0.5 * _clamp(ball_d, BALL_MIN, BALL_MAX)
    c = _clamp(clearance, CLEARANCE_MIN, CLEARANCE_MAX)
    R = Rb + c
    rs = 0.5 * min(_clamp(stem_d, STEM_MIN, STEM_MAX), _max_stem(R, Rb, c))
    w = _clamp(wall, WALL_MIN, WALL_MAX)
    ln = _clamp(stem_len, STEM_LEN_MIN, STEM_LEN_MAX)
    ask = _clamp(swing, SWING_MIN, SWING_MAX)
    sw = min(ask, _max_swing(R, Rb, rs, c))
    a = math.radians(sw)

    # the floor under the seat is as thick as asked, and thicker when a big ball needs a shallow tip.
    # The tip is taken to a point on the ball a little above the printable circle of the seat, so
    # that the flat faces the ball is made of just above that seam lean the printable way too.
    floor = _clamp(base, BASE_MIN, BASE_MAX)
    for _ in range(200):
        cz = floor + Rb + c
        need_z = cz - math.sin(math.radians(PRINT_LIMIT - PRINT_MARGIN)) * Rb + 0.15
        need_r = math.sqrt(max(1e-9, Rb * Rb - (cz - need_z) ** 2))
        foot = min(FOOT, 0.5 * Rb, 0.8 * need_r)
        slope = math.atan2(need_r - foot, need_z)
        if math.degrees(slope) <= TIP_LIMIT or floor >= BASE_MAX:
            break
        floor += 0.25

    # the tip ends where it is well lost in the ball: on the line from the foot to the seat, a
    # little way past where that line enters the ball
    k = math.tan(slope)
    qa, qb = k * k + 1.0, 2.0 * (k * foot - cz)
    disc = qb * qb - 4.0 * (qa * (foot * foot + cz * cz - Rb * Rb))
    if disc > 0.0:
        root = math.sqrt(disc) / (2.0 * qa)
        # the line of the tip runs into the ball between these two heights; the tip ends a third of
        # the way from the first to the second, so what is left of it is well lost in the ball
        tip_z = -qb / (2.0 * qa) - 0.30 * root
    else:
        tip_z = need_z
    tip_r = foot + k * tip_z

    edge = _neck_edge(R, rs, a, c)
    neck_r = R * math.sin(edge)
    neck_z = cz + R * math.cos(edge)
    top_z = neck_z + NECK
    out_r = R + w
    return dict(ball=2.0 * Rb, clearance=c, stem=2.0 * rs, swing=sw, swing_asked=ask, wall=w,
                base=_clamp(base, BASE_MIN, BASE_MAX), stem_len=ln,
                grip=int(_clamp(grip, GRIP_MIN, GRIP_MAX)), seat_r=R + SEAT_SLACK, centre=cz,
                floor=floor, foot=foot, tip_z=tip_z, tip_r=tip_r, tip_slope=math.degrees(slope),
                opening=2.0 * neck_r, neck_r=neck_r, edge=math.degrees(edge),
                capture=Rb - neck_r, neck_z=neck_z, top_z=top_z,
                stem_top=top_z + ln, base_d=2.0 * (out_r + LIP), height=top_z + ln)


def ball_joint(ball_d=12.0, clearance=0.3, stem_d=3.0, swing=35.0, wall=2.5, base=3.0,
               stem_len=12.0, grip=GRIP_DEFAULT):
    """A ball joint that prints in one piece and turns as soon as it is off the printer.

    The joint stands on the bed as it prints: the socket on its floor with the ball in its seat and
    the stem upright, and the ball on a slender foot that is broken off afterwards. Everything is
    centred on the axis and stands on `z = 0`, so the two volumes come back where they print, one
    inside the other: `(the socket, the ball with its stem)`. Both are needed together — the socket
    is `sizes(...)["base_d"]` millimetres wide and `height` millimetres high.

    The seat, the neck and the relief are surfaces of the ball's own centre, so every facet of the
    joint that is not the bed itself leans no more than 45 degrees from the vertical and the joint
    prints without a single strand of support.
    """
    n = sizes(ball_d=ball_d, clearance=clearance, stem_d=stem_d, swing=swing, wall=wall,
              base=base, stem_len=stem_len, grip=grip)
    Rb, c, rs = 0.5 * n["ball"], n["clearance"], 0.5 * n["stem"]
    cz, R_seat = n["centre"], n["seat_r"]
    edge = math.radians(n["edge"])

    ball = mf.Manifold.sphere(Rb, SEGMENTS).translate((0.0, 0.0, cz))
    tip = mf.Manifold.cylinder(n["tip_z"], n["foot"], n["tip_r"], SEGMENTS)
    stem = mf.Manifold.cylinder(n["stem_top"] - cz, rs, rs, SEGMENTS).translate((0.0, 0.0, cz))
    moving = mf.Manifold.batch_boolean([ball, tip, stem], mf.OpType.Add)

    out_r = 0.5 * n["base_d"] - LIP
    body = _revolve([(0.0, 0.0), (out_r + LIP, 0.0), (out_r + LIP, n["floor"]),
                     (out_r, n["floor"]), (out_r, n["top_z"]), (0.0, n["top_z"])])
    bowl = [(0.0, n["floor"])]
    for i in range(33):
        t = math.pi + (edge - math.pi) * i / 32.0
        bowl.append((R_seat * math.sin(t), cz + R_seat * math.cos(t)))
    bowl += [(n["opening"] / 2.0, n["top_z"]), (0.0, n["top_z"])]
    quad = [(0.0, 0.0), (n["foot"], 0.0), (n["tip_r"], n["tip_z"]), (0.0, n["tip_z"])]
    relief = _revolve(_relief_profile(quad, cz, n["swing"], c + SEAT_SLACK, R_seat))
    socket = body - (_revolve(bowl) + relief)

    if n["grip"]:
        slits = []
        for i in range(n["grip"]):
            cut = mf.Manifold.cube([out_r + 2.0 * LIP, SLIT, n["top_z"] + 1.0 - n["floor"]]) \
                .translate((0.0, -0.5 * SLIT, n["floor"]))
            slits.append(cut.rotate((0.0, 0.0, 360.0 * (i + 0.5) / n["grip"])))
        socket = socket - mf.Manifold.batch_boolean(slits, mf.OpType.Add)
    return socket, moving


def _revolve(profile) -> mf.Manifold:
    """The profile, in `(radius, height)`, turned whole round the Z axis."""
    cs = mf.CrossSection([[(float(x), float(y)) for x, y in profile]], mf.FillRule.NonZero)
    return mf.Manifold.revolve(cs, SEGMENTS, 360.0)


def joint(**options) -> mf.Manifold:
    """The whole joint as one volume of two parts that are not fused, like the hinge.

    `ball_joint` gives the two volumes; this puts them together again for the export and for the
    tests that weigh the joint as one print.
    """
    socket, moving = ball_joint(**options)
    return socket + moving

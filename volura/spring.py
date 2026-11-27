"""Springs: the computational core for a printable zigzag spring and a printable spiral spring.

No Qt in this file. It gives both springs as closed volumes that lie flat on `z = 0` and are
centred in x and y, so they can be tested on their own and reused in the model and the export.

A **zigzag spring** is a flat strip folded back and forth on the bed. It has `folds` straight
beams of length `L`, thickness `t` and height `h`, stacked with a gap `g` between them and joined
alternately at the right and at the left by half rounds. The middle line of a joint stands halfway
between the middle lines of the two beams it joins; the hollow of the joint keeps the gap, so the
strip has the same thickness everywhere and no sharp inside corner where a printed spring would
tear. The two ends finish in square pads, each big enough to take the screw hole the spring is
fastened with. A spring of `n` beams is one long beam of `n` times the beam length, folded into a
part barely taller than one beam.

A **spiral spring** is the same kind of strip wound flat on the bed around a hub. Its middle line
is an Archimedean spiral: every turn moves outward by `t + g`, so the strip of thickness `t` leaves
the gap `g` between its neighbouring turns. The inner end is built into the hub, which carries a
D-shaped bore for a shaft; the outer end is built into a square lug with a screw hole. The spring
turns round the bore instead of being compressed, and the D keeps a shaft from turning inside it.

**Stiffness.** For small deflections a zigzag spring behaves like `n` cantilever beams of length
`L`, thickness `t` and rectangular cross-section `t × h`. Their stiffness, measured by the force
per millimetre of compression, is

    k = E h t^3 / (n L^3)

newtons per millimetre, where `E` is the tensile modulus of the material. A spring whose working
strain is `STRAIN` can be compressed about

    d_max = n STRAIN L^2 / (3 t)

before its outer layer is worked hard, and the force there is `k d_max`.

A spiral spring is worked in torsion. Its strip is a coil spring printed flat, and the stiffness
per radian of rotation around the bore is

    k_t = E h t^3 / (12 length)

newton millimetres per radian, where `length` is the length of the middle line of the spiral. The
dialog gives the same stiffness per degree, because that is what a user turns a shaft by.

**Printability.** Both springs are extruded from a flat contour, so the only faces that look down
are their bottom faces on the bed: the walls, holes and bore are straight vertical cuts, and no
bridge or overhang is needed. The spiral hub keeps the inner end of the strip out of the middle;
a spiral whose strip would be folded past its own middle is refused, because it would no longer be
one printable strip.
"""
from __future__ import annotations

import math

import manifold3d as mf
import numpy as np

from .i18n import tr

#: the kinds of spring the dialog and the part builder know
KINDS = ("zigzag", "spiral")
#: the material a spring is printed in and its tensile modulus in MPa
MATERIALS = {"PLA": 3500.0, "PETG": 2100.0, "ABS": 2200.0, "Nylon": 1400.0,
             "TPU 95A": 26.0}

#: the fewest and the most straight beams in a zigzag spring
FOLDS_MIN, FOLDS_MAX = 2, 30
#: the shortest and the longest straight beam of a zigzag spring
LENGTH_MIN, LENGTH_MAX = 5.0, 80.0
#: the thinnest and thickest spring strip that is still a spring and still prints
THICKNESS_MIN, THICKNESS_MAX = 0.6, 4.0
#: the narrowest and widest room between two turns of the strip
GAP_MIN, GAP_MAX = 1.0, 10.0
#: how flat and how tall a spring printed on its side may be
HEIGHT_MIN, HEIGHT_MAX = 2.0, 30.0
#: the fewest and the most turns of a spiral spring
TURNS_MIN, TURNS_MAX = 1, 15
#: the innermost and outermost radius of the middle line of a spiral spring
INNER_MIN, INNER_MAX = 1.0, 40.0
#: the widest screw hole that is still a useful fastening for a small spring
HOLE_MAX = 10.0
#: mm of pad at the end of a zigzag spring with no hole
PAD = 3.0
#: mm of plastic that has to stand around the hole in an end pad
PAD_WALL = 1.2
#: mm by mm of the square lug at the outer end of a spiral spring
LUG = 6.0
#: mm of plastic that has to stand around the hole in that lug
LUG_WALL = 1.2
#: the largest bore that the dialog asks for before the hub makes it smaller
BORE_MAX = 20.0
#: how far one side of the D-shaped bore is filed flat
BORE_FLAT = 0.5
#: mm of hub wall that has to stay around the bore
HUB_WALL = 1.2
#: mm by which a strip end is pushed into the hub, the lug or the next beam, so the union is safe
JOIN_EPS = 0.02
#: mm by which a cutting tool reaches past the part, so a cut leaves no skin behind
CUT_MARGIN = 1.0
#: mm of middle-line radius that has to remain after a strip has been folded
INNER_EDGE_ROOM = 0.5
#: mm of curvature radius that a strip needs before its own inner side crosses itself
CURVE_ROOM = 0.1
#: degrees of strain at which the dialog says the spring is being worked hard
STRAIN = 0.02
#: the straight pieces a round joint, hole or bore is made of
SEGMENTS = 64
#: degrees between the samples of a spiral strip
STRIDE = 3.0


def modulus(material) -> float:
    """The tensile modulus of `material` in MPa; an unknown material is refused."""
    try:
        return float(MATERIALS[material])
    except KeyError:
        raise ValueError(tr("the material of a spring must be one of: {names}")
                         .format(names=", ".join(MATERIALS))) from None


def _number(value, lo, hi, what) -> float:
    """A size that is asked for, taken to what can be printed; no number at all is refused."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        v = math.nan
    if not math.isfinite(v):
        raise ValueError(tr("the {what} of a spring must be a number").format(what=what))
    return max(lo, min(hi, v))


def _whole(value, lo, hi, what) -> int:
    """A count that is asked for, as a whole number in the range a spring is made in."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        raise ValueError(tr("the {what} of a spring must be a whole number")
                         .format(what=what)) from None
    if not math.isfinite(v):
        raise ValueError(tr("the {what} of a spring must be a whole number")
                         .format(what=what))
    return int(max(lo, min(hi, round(v))))


def _kind(kind) -> str:
    """The kind of spring that is asked for; another kind is refused."""
    key = str(kind)
    if key not in KINDS:
        raise ValueError(tr("the kind of spring must be {names}")
                         .format(names=", ".join(KINDS)))
    return key


def _hole(value) -> float:
    """The diameter of a spring hole; zero means that there is no hole."""
    d = _number(value, 0.0, HOLE_MAX, "screw hole")
    return 0.0 if d <= 1e-9 else d


def _pad_width(hole) -> float:
    """The size of an end pad that keeps `PAD_WALL` millimetres around a hole of `hole`."""
    d = _hole(hole)
    return d, (max(PAD, d + 2.0 * PAD_WALL) if d > 0.0 else PAD)


def _common_numbers(folds, length, thickness, gap, height, turns, inner, hole):
    """The sizes that both kinds share, taken to what can be printed."""
    n = _whole(folds, FOLDS_MIN, FOLDS_MAX, "number of folds")
    L = _number(length, LENGTH_MIN, LENGTH_MAX, "length")
    t = _number(thickness, THICKNESS_MIN, THICKNESS_MAX, "thickness")
    g = _number(gap, GAP_MIN, GAP_MAX, "gap")
    h = _number(height, HEIGHT_MIN, HEIGHT_MAX, "height")
    z = _whole(turns, TURNS_MIN, TURNS_MAX, "number of turns")
    r0 = _number(inner, INNER_MIN, INNER_MAX, "inner radius")
    d, pad = _pad_width(hole)
    return {"folds": n, "L": L, "t": t, "g": g, "h": h, "turns": z, "inner": r0,
            "hole": d, "pad": pad}


def _spiral_middle(turns, thickness, gap, inner) -> dict:
    """The middle line of a spiral spring and the strip that stands around it."""
    z = _whole(turns, TURNS_MIN, TURNS_MAX, "number of turns")
    t = _number(thickness, THICKNESS_MIN, THICKNESS_MAX, "thickness")
    g = _number(gap, GAP_MIN, GAP_MAX, "gap")
    r0 = _number(inner, INNER_MIN, INNER_MAX, "inner radius")
    pitch = t + g
    b = pitch / (2.0 * math.pi)
    theta = 2.0 * math.pi * z
    outer = r0 + b * theta
    count = max(2, int(math.ceil(theta / math.radians(STRIDE)))) + 1
    outer_side, inner_side = [], []
    for i in range(count):
        a = theta * i / (count - 1)
        r = r0 + b * a
        c, s = math.cos(a), math.sin(a)
        speed = math.hypot(b, r)
        nx = -(b * s + r * c) / speed
        ny = (b * c - r * s) / speed
        px, py = r * c, r * s
        outer_side.append((px + 0.5 * t * nx, py + 0.5 * t * ny))
        inner_side.append((px - 0.5 * t * nx, py - 0.5 * t * ny))
    return {"turns": z, "t": t, "g": g, "inner": r0, "pitch": pitch, "b": b, "theta": theta,
            "outer": outer, "loop": np.array(outer_side + inner_side[::-1], dtype=float)}


def _spiral_footprint(loop, hub, theta, outer, lug) -> tuple:
    """The space the strip, hub and lug of a spiral spring take on the bed."""
    points = [loop, np.array([[-hub, 0.0], [hub, 0.0], [0.0, -hub], [0.0, hub]], dtype=float)]
    if lug > 1e-9:
        c, s = math.cos(theta), math.sin(theta)
        cx = (outer + 0.5 * LUG - JOIN_EPS) * c
        cy = (outer + 0.5 * LUG - JOIN_EPS) * s
        hx, hy = 0.5 * LUG * c, 0.5 * LUG * s
        vx, vy = -0.5 * LUG * s, 0.5 * LUG * c
        points.append(np.array([(cx + hx + vx, cy + hy + vy), (cx - hx + vx, cy - hy + vy),
                                (cx - hx - vx, cy - hy - vy), (cx + hx - vx, cy + hy - vy)],
                               dtype=float))
    pts = np.vstack(points)
    low, high = pts.min(axis=0), pts.max(axis=0)
    return float(high[0] - low[0]), float(high[1] - low[1])


def _spiral_sizes(inner, thickness, gap, turns, bore, lug, height=HEIGHT_MIN) -> dict:
    """The numbers of a spiral spring without building it."""
    h = _number(height, HEIGHT_MIN, HEIGHT_MAX, "height")
    m = _spiral_middle(turns, thickness, gap, inner)
    t, g, r0, z, pitch, b = m["t"], m["g"], m["inner"], m["turns"], m["pitch"], m["b"]
    theta, outer = m["theta"], m["outer"]
    hub = r0 + 0.5 * t
    if r0 - 0.5 * t <= INNER_EDGE_ROOM:
        raise ValueError(tr("a spiral spring whose strip reaches the middle cannot be made: make "
                            "the inner radius bigger"))
    radius_of_curvature = ((r0 * r0 + b * b) ** 1.5) / (r0 * r0)
    if radius_of_curvature <= 0.5 * t + CURVE_ROOM:
        raise ValueError(tr("a spiral spring that is wound too tightly cannot be made: make the "
                            "inner radius bigger"))
    bore_d = _number(bore, 0.0, BORE_MAX, "bore")
    if bore_d > 0.0:
        bore_d = min(bore_d, max(0.0, 2.0 * (hub - HUB_WALL)))
        if bore_d <= 1e-9:
            bore_d = 0.0
    lug_d = _number(lug, 0.0, LUG, "lug hole")
    if lug_d > 0.0:
        lug_d = min(lug_d, max(0.0, LUG - 2.0 * LUG_WALL))
        if lug_d <= 1e-9:
            lug_d = 0.0
    length = spiral_length(r0, pitch, z)
    width, depth = _spiral_footprint(m["loop"], hub + JOIN_EPS, theta, outer, lug_d)
    return {"turns": z, "inner": r0, "t": t, "g": g, "h": h, "bore": bore_d,
            "lug": lug_d, "pitch": pitch, "outer": outer, "hub": hub, "width": width,
            "height": depth, "radius": 0.5 * max(width, depth), "length": length}


def spiral_length(inner, pitch, turns) -> float:
    """The length of the middle line of a spiral spring, as exactly as the formula allows.

    The middle line is `r = inner + pitch * theta / (2 pi)`. With `b = pitch / (2 pi)` its length
    from the start to the radius `outer` is the closed form of the integral of
    `sqrt(r^2 + b^2)`.
    """
    r0 = _number(inner, INNER_MIN, INNER_MAX, "inner radius")
    p = _number(pitch, THICKNESS_MIN + GAP_MIN, THICKNESS_MAX + GAP_MAX, "pitch")
    z = _whole(turns, TURNS_MIN, TURNS_MAX, "number of turns")
    b = p / (2.0 * math.pi)
    outer = r0 + b * 2.0 * math.pi * z

    def primitive(u):
        return 0.5 * (u * math.sqrt(u * u + b * b) + b * b * math.log(u + math.sqrt(u * u + b * b)))

    return max(0.0, (primitive(outer) - primitive(r0)) / b)


def zigzag_stiffness(material, folds, length, thickness, height) -> dict:
    """The compression stiffness, working travel and full-travel force of a zigzag spring."""
    E = modulus(material)
    n = _whole(folds, FOLDS_MIN, FOLDS_MAX, "number of folds")
    L = _number(length, LENGTH_MIN, LENGTH_MAX, "length")
    t = _number(thickness, THICKNESS_MIN, THICKNESS_MAX, "thickness")
    h = _number(height, HEIGHT_MIN, HEIGHT_MAX, "height")
    k = E * h * t ** 3 / (n * L ** 3)
    travel = n * STRAIN * L ** 2 / (3.0 * t)
    return {"E": E, "k": k, "travel": travel, "force": k * travel}


def spiral_stiffness(material, turns, inner, thickness, gap, height=HEIGHT_MIN) -> dict:
    """The torsional stiffness of a spiral spring around its bore."""
    E = modulus(material)
    n = _spiral_sizes(inner, thickness, gap, turns, 0.0, 0.0, height)
    kt = E * n["h"] * n["t"] ** 3 / (12.0 * n["length"])
    return {"E": E, "length": n["length"], "kt": kt, "kt_deg": kt * math.pi / 180.0}


def sizes(kind="zigzag", **kw) -> dict:
    """The numbers that the dialog shows under the fields of `kind` of spring."""
    kind = _kind(kind)
    common = _common_numbers(kw.get("folds", 6), kw.get("length", 20.0), kw.get("thickness", 1.2),
                             kw.get("gap", 2.5), kw.get("height", 5.0), kw.get("turns", 4),
                             kw.get("inner", 4.0), kw.get("hole", 3.2))
    material = kw.get("material", "PLA")
    E = modulus(material)
    out = dict(common)
    out["kind"] = kind
    out["material"] = material
    out["E"] = E
    if kind == "zigzag":
        n = out["folds"]
        out["pitch"] = out["t"] + out["g"]
        out["strip"] = n * out["t"] + (n - 1) * out["g"]
        reach = (out["g"] + 2.0 * out["t"]) if n > 2 else 0.5 * (out["g"] + 2.0 * out["t"])
        out["width"] = max(out["L"] + reach, out["pad"])
        out["height"] = out["strip"] + 2.0 * out["pad"]
        out.update(zigzag_stiffness(material, n, out["L"], out["t"], out["h"]))
    else:
        s = _spiral_sizes(out["inner"], out["t"], out["g"], out["turns"],
                          kw.get("bore", 5.0), kw.get("lug", 3.2), out["h"])
        out.update(s)
        out.update(spiral_stiffness(material, out["turns"], out["inner"], out["t"], out["g"],
                                    out["h"]))
    return out


def _prism(loop, height) -> mf.Manifold:
    """A straight prism of `height` millimetres, standing on the bed, from one closed loop."""
    loop = np.asarray(loop, dtype=float)
    if np.sum(loop[:, 0] * np.roll(loop[:, 1], -1) -
              np.roll(loop[:, 0], -1) * loop[:, 1]) < 0.0:
        loop = loop[::-1]
    return mf.Manifold.extrude(mf.CrossSection([loop]), max(float(height), 1e-6))


def _rect_prism(x0, x1, y0, y1, height) -> mf.Manifold:
    """A rectangular prism standing on the bed."""
    return _prism(np.array([(x0, y0), (x1, y0), (x1, y1), (x0, y1)], dtype=float), height)


def _sector_prism(cx, cy, r_in, r_out, right, height) -> mf.Manifold:
    """A half annulus standing on the bed, open to the right or to the left."""
    start, end = (-0.5 * math.pi, 0.5 * math.pi) if right else (0.5 * math.pi, 1.5 * math.pi)
    outer, inner = [], []
    for i in range(SEGMENTS + 1):
        a = start + (end - start) * i / SEGMENTS
        outer.append((cx + r_out * math.cos(a), cy + r_out * math.sin(a)))
        inner.append((cx + r_in * math.cos(a), cy + r_in * math.sin(a)))
    return _prism(np.array(outer + inner[::-1], dtype=float), height)


def _circle_prism(x, y, d, height, z0=0.0) -> mf.Manifold:
    """A cylindrical prism standing on the bed, or from `z0` up."""
    m = mf.Manifold.cylinder(max(float(height), 1e-6), max(0.5 * float(d), 1e-6),
                             max(0.5 * float(d), 1e-6), SEGMENTS)
    return m.translate((float(x), float(y), float(z0)))


def _hole_prism(x, y, d, height) -> mf.Manifold:
    """A cutting cylinder that goes clean through a flat part."""
    d = max(float(d), 1e-6)
    m = mf.Manifold.cylinder(height + 2.0 * CUT_MARGIN, 0.5 * d, 0.5 * d, SEGMENTS)
    return m.translate((float(x), float(y), -CUT_MARGIN))


def _d_prism(d, flat, height) -> mf.Manifold:
    """A D-shaped cutting prism: a round bore with one side filed flat."""
    d = max(float(d), 1e-6)
    r = 0.5 * d
    flat = max(0.0, min(float(flat), 0.95 * d))
    if flat <= 1e-9:
        return mf.Manifold.cylinder(height + 2.0 * CUT_MARGIN, r, r, SEGMENTS).translate(
            (0.0, 0.0, -CUT_MARGIN))
    x0 = -r + flat
    y0 = math.sqrt(max(0.0, r * r - x0 * x0))
    alpha = math.acos(max(-1.0, min(1.0, x0 / r)))
    loop = [(x0, -y0)]
    for i in range(1, SEGMENTS):
        a = -alpha + 2.0 * alpha * i / SEGMENTS
        loop.append((r * math.cos(a), r * math.sin(a)))
    loop.append((x0, y0))
    return _prism(loop, height + 2.0 * CUT_MARGIN).translate((0.0, 0.0, -CUT_MARGIN))


def _spiral_strip(turns, thickness, gap, inner, height) -> mf.Manifold:
    """The flat strip of a spiral spring, without hub or lug yet."""
    h = _number(height, HEIGHT_MIN, HEIGHT_MAX, "height")
    return _prism(_spiral_middle(turns, thickness, gap, inner)["loop"], h)


def _lug_prism(theta, radius, hole, height) -> tuple:
    """The square lug at the end of a spiral strip, and the centre of the hole in it."""
    d = max(0.0, float(hole))
    if d <= 1e-9:
        return None, None
    c, s = math.cos(theta), math.sin(theta)
    cx = (radius + 0.5 * LUG - JOIN_EPS) * c
    cy = (radius + 0.5 * LUG - JOIN_EPS) * s
    hx, hy = 0.5 * LUG * c, 0.5 * LUG * s
    vx, vy = -0.5 * LUG * s, 0.5 * LUG * c
    loop = [(cx + hx + vx, cy + hy + vy), (cx - hx + vx, cy - hy + vy),
            (cx - hx - vx, cy - hy - vy), (cx + hx - vx, cy + hy - vy)]
    return _prism(loop, height), (cx, cy)


def zigzag(folds=6, length=20.0, thickness=1.2, gap=2.5, height=5.0, hole=3.2) -> mf.Manifold:
    """A flat zigzag spring, lying on the bed and centred in x and y.

    It has `folds` straight beams of `length` by `thickness` millimetres, `height` millimetres
    tall, stacked with `gap` millimetres between them and joined alternately right and left. The
    end pads are square; with a positive `hole` each of them gets a round hole of that diameter,
    and a pad grows so that it keeps `PAD_WALL` millimetres around the hole.
    """
    n = _whole(folds, FOLDS_MIN, FOLDS_MAX, "number of folds")
    L = _number(length, LENGTH_MIN, LENGTH_MAX, "length")
    t = _number(thickness, THICKNESS_MIN, THICKNESS_MAX, "thickness")
    g = _number(gap, GAP_MIN, GAP_MAX, "gap")
    h = _number(height, HEIGHT_MIN, HEIGHT_MAX, "height")
    d, pad = _pad_width(hole)
    pitch = t + g
    strip = n * t + (n - 1) * g
    pieces = []

    for i in range(n):
        pieces.append(_rect_prism(-JOIN_EPS, L + JOIN_EPS, i * pitch, i * pitch + t, h))
    for i in range(n - 1):
        right = i % 2 == 0
        cy = i * pitch + t + 0.5 * g
        cx = L if right else 0.0
        pieces.append(_sector_prism(cx, cy, 0.5 * g, 0.5 * (g + 2.0 * t), right, h))
    pieces.append(_rect_prism(0.5 * (L - pad), 0.5 * (L + pad), -pad, JOIN_EPS, h))
    pieces.append(_rect_prism(0.5 * (L - pad), 0.5 * (L + pad), strip - JOIN_EPS, strip + pad, h))

    spring = mf.Manifold.batch_boolean(pieces, mf.OpType.Add)
    if d > 0.0:
        for cy in (-0.5 * pad, strip + 0.5 * pad):
            spring = spring - _hole_prism(0.5 * L, cy, d, h)
    spring = spring.translate((-0.5 * L, -0.5 * strip, 0.0))
    if spring.status() != mf.Error.NoError or spring.is_empty():
        raise ValueError(tr("the zigzag spring did not become one valid volume"))
    return spring


def spiral(turns=4, inner=4.0, thickness=1.2, gap=2.5, height=5.0, bore=5.0,
           lug=3.2) -> mf.Manifold:
    """A flat spiral spring, lying on the bed and centred in x and y.

    Its middle line winds `turns` times outward from radius `inner`, with a strip `thickness`
    millimetres wide and `height` millimetres tall and a gap `gap` between its turns. The hub at
    the inner end carries a D-shaped bore of `bore` millimetres; zero leaves a solid hub. The
    square lug at the outer end carries a hole of `lug` millimetres; zero leaves no lug.
    """
    n = _spiral_sizes(inner, thickness, gap, turns, bore, lug, height)
    pieces = [_spiral_strip(n["turns"], n["t"], n["g"], n["inner"], n["h"]),
              _circle_prism(0.0, 0.0, 2.0 * (n["hub"] + JOIN_EPS), n["h"])]
    theta = 2.0 * math.pi * n["turns"]
    tab, tab_c = _lug_prism(theta, n["outer"], n["lug"], n["h"])
    if tab is not None:
        pieces.append(tab)
    spring = mf.Manifold.batch_boolean(pieces, mf.OpType.Add) if len(pieces) > 1 else pieces[0]
    if n["bore"] > 0.0:
        spring = spring - _d_prism(n["bore"], BORE_FLAT, n["h"])
    if tab_c is not None:
        spring = spring - _hole_prism(tab_c[0], tab_c[1], n["lug"], n["h"])

    bb = np.asarray(spring.bounding_box(), dtype=float)
    spring = spring.translate((-0.5 * (bb[0] + bb[3]), -0.5 * (bb[1] + bb[4]), -bb[2]))
    if spring.status() != mf.Error.NoError or spring.is_empty():
        raise ValueError(tr("the spiral spring did not become one valid volume"))
    return spring


def spring(kind="zigzag", **kw) -> mf.Manifold:
    """`zigzag` or `spiral`, depending on `kind`; the remaining keywords go to that builder."""
    kind = _kind(kind)
    if "material" in kw:
        modulus(kw["material"])
    params = {"folds", "length", "thickness", "gap", "height", "hole"} if kind == "zigzag" \
        else {"turns", "inner", "thickness", "gap", "height", "bore", "lug"}
    return (zigzag if kind == "zigzag" else spiral)(
        **{k: v for k, v in kw.items() if k in params})

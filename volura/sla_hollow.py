"""Hollowing a resin print: a shell with drain holes, for File > Slice for Resin.

No Qt in this file. A solid resin print of a large part uses a lot of resin and pulls hard on
the vat; a shell a few millimetres thick prints just as well and takes a fraction of the resin.
The inside has to drain, or the part comes out of the vat full of resin and pops off the build
plate, so a hollow part gets **drain holes** near the bottom: holes straight down through the
floor, out of which the resin runs.

The shell is made per layer, the way the print is cut anyway. The cavity of a layer is what is
at least `wall` away from the surface there *and* in the layers around it, so the floor and the
roof of a cavity are as thick as its sides: the cross-section of a layer `dz` above or below,
moved inwards by `sqrt(wall² - dz²)`, intersected over the layers within `wall` of it - the
part eroded by `wall` in 3D, sampled at nine heights. A part thinner than twice the wall has no
cavity anywhere and stays solid.

Only the holes are cut into the solid; the cavity is subtracted per layer while slicing, by
`volura.sla_slicer.write_sl1`. The holes are cut before the supports are made, so that no
support point lands in one and the cavity - which is not part of the solid the supports see -
gets none.
"""
from __future__ import annotations

import math

import manifold3d as mf
import numpy as np

#: mm, the default thickness of the shell
WALL = 2.5
#: mm2: a cavity smaller than this is not worth a shell round it
MIN_CAVITY = 1.0
#: the layers a cavity is made of: this many, spread over `wall` below and above it
NEIGHBOURS = 9
#: mm, the layer height a part is sampled at when the layers of the print are not known yet
SAMPLE = 0.25
#: how many drain holes a hollow part gets
DRAIN_HOLES = 2
#: mm across a drain hole: wide enough for resin to run out of in a second or two
DRAIN_DIAMETER = 3.0
#: pieces a round drain hole is made of
SEGMENTS = 24
#: mm under the bottom of the part the drain holes start
BELOW = 1.0
#: at most this many points of a cavity are compared when the holes are placed
CANDIDATES = 1024


def _step(wall: float) -> float:
    """The height between the layers a cavity is sampled at: `wall` / 4, never finer than
    SAMPLE, so nine of them reach exactly `wall` below and above the layer."""
    return max(SAMPLE, 2.0 * wall / (NEIGHBOURS - 1))


def _sampled(solid: mf.Manifold, wall: float) -> list[float]:
    """The heights to place the drain holes at when the layers of the print are not known."""
    lo, hi = float(solid.bounding_box()[2]), float(solid.bounding_box()[5])
    step = _step(wall)
    n = max(1, int(math.ceil((hi - lo) / step - 1e-9)))
    return [lo + (i + 0.5) * step for i in range(n)]


def _picked(zs: list[float], z: float, wall: float) -> list[float]:
    """The heights of `zs` within `wall` of `z`, at most NEIGHBOURS of them, evenly picked and
    with the layer at `z` itself among them: that one is the strongest constraint of all."""
    near = [w for w in zs if z - wall - 1e-9 <= w <= z + wall + 1e-9]
    if len(near) <= NEIGHBOURS:
        return near
    last = len(near) - 1
    idx = {round(k * last / (NEIGHBOURS - 1)) for k in range(NEIGHBOURS)}
    mid = min(range(len(near)), key=lambda i: abs(near[i] - z))
    if mid not in idx:
        idx.add(mid)
        idx.discard(min(idx - {mid}, key=lambda i: abs(near[i] - z)))
    return [near[i] for i in sorted(idx)]


def _cavity(solid: mf.Manifold, z: float, zs: list[float], wall: float, cache: dict,
            span) -> mf.CrossSection | None:
    """The cavity of the layer at `z`, or None where there is no room for one.

    Of every layer within `wall` of `z` only the middle survives: a point that is `dz` away
    from that layer has `sqrt(wall² - dz²)` of wall left to keep it out of the surface, so the
    cross-section of that layer, moved inwards by that much, is what may be cavity there.
    `span` is the lowest and the highest height of the part: `wall` below `z` and above it has
    to be part of it, or the floor or the roof of the cavity would be thinner than the wall.
    """
    if z - wall < span[0] - 1e-9 or z + wall > span[1] + 1e-9:
        return None
    near = _picked(zs, z, wall)
    if not near:
        return None
    shapes = []
    for w in near:
        if w not in cache:
            cache[w] = solid.slice(w)
        cs = cache[w]
        d = math.sqrt(max(0.0, wall * wall - (w - z) * (w - z)))
        if d > 1e-9:
            cs = cs.offset(-d, mf.JoinType.Round)
        if cs.is_empty():
            return None
        shapes.append(cs)
    if len(shapes) == 1:
        return shapes[0]
    return mf.CrossSection.batch_boolean(shapes, mf.OpType.Intersect)


def hollow_layers(solid: mf.Manifold, zs, wall: float) -> dict:
    """The cavity of every layer of a hollow part: {z: CrossSection} for the heights of `zs`.

    A height without a cavity - one too near the bottom or the top of the part, or one where
    the part is thinner than twice the wall - is left out of the result. Every height is
    sliced once.
    """
    zs = sorted(float(z) for z in zs)
    box = solid.bounding_box()
    span = (float(box[2]), float(box[5]))
    cache: dict = {}
    out = {}
    for z in zs:
        cs = _cavity(solid, z, zs, wall, cache, span)
        if cs is not None and not cs.is_empty() and cs.area() >= MIN_CAVITY:
            out[z] = cs
    return out


def _farthest(cs: mf.CrossSection, count: int) -> list:
    """Points of a cross-section as far apart as possible: the two ends of its longest extent
    and, for more holes, every next point as far as possible from the ones before it."""
    polys = [np.asarray(p, dtype=float) for p in cs.to_polygons() if len(p)]
    if not polys:
        return []
    pts = np.concatenate(polys)
    if len(pts) > CANDIDATES:
        pts = pts[::len(pts) // CANDIDATES + 1]
    d = np.hypot(pts[:, None, 0] - pts[None, :, 0], pts[:, None, 1] - pts[None, :, 1])
    if float(d.max()) <= 1e-9:
        return [(float(pts[0, 0]), float(pts[0, 1]))]
    chosen = list(np.unravel_index(int(np.argmax(d)), d.shape))
    while len(chosen) < count:
        reach = np.max(d[:, chosen], axis=1)
        k = int(np.argmax(reach))
        if reach[k] <= 1e-9:
            break
        chosen.append(k)
    return [(float(pts[k, 0]), float(pts[k, 1])) for k in chosen[:count]]


def drain_holes(solid: mf.Manifold, cavity_bottom_z: float, count: int = DRAIN_HOLES,
                diameter: float = DRAIN_DIAMETER, wall: float = WALL,
                zs=None) -> tuple[list, float]:
    """The places of the drain holes, low in the cavity: ([(x, y), …], diameter).

    They go straight down from the lowest layer that has a cavity, through the floor of the
    part, so they are as near the bottom of the print as the cavity allows. In that cavity they
    lie as far apart as possible, and so far from its wall that a hole of `diameter` stays
    inside it. Empty when the part has nothing to drain.
    """
    zs = ([cavity_bottom_z + (k - NEIGHBOURS // 2) * _step(wall) for k in range(NEIGHBOURS)]
          if zs is None else sorted(float(z) for z in zs))
    box = solid.bounding_box()
    cs = _cavity(solid, cavity_bottom_z, zs, wall, {}, (float(box[2]), float(box[5])))
    if cs is None or cs.is_empty():
        return [], diameter
    room = cs.offset(-0.5 * diameter, mf.JoinType.Round)
    return _farthest(room if not room.is_empty() else cs, count), diameter


def hollowed(solid: mf.Manifold, wall: float, zs=None,
             holes: bool = True) -> tuple[mf.Manifold, dict]:
    """The part as a shell, with the drain holes cut, and the cavity of every layer.

    Returns (the part to print, {z: the cavity of that layer}); the slicer subtracts those
    cavities, so the hollow part stays one solid everywhere but in its holes. `zs` are the
    heights the print is cut at; without them the part is only sampled high enough to place the
    holes. A part that is too thin for a shell comes back as it was.
    """
    zs = _sampled(solid, wall) if zs is None else sorted(float(z) for z in zs)
    cavities = hollow_layers(solid, zs, wall)
    if not holes or not cavities:
        return solid, cavities
    bottom = min(cavities)
    points, diameter = drain_holes(solid, bottom, wall=wall, zs=zs)
    if not points:
        return solid, cavities
    z0 = float(solid.bounding_box()[2]) - BELOW
    r = 0.5 * diameter
    tubes = [mf.Manifold.cylinder(bottom - z0, r, r, SEGMENTS).translate((x, y, z0))
             for x, y in points]
    cut = mf.Manifold.batch_boolean(tubes, mf.OpType.Add) if len(tubes) > 1 else tubes[0]
    shell = solid - cut
    return (shell, cavities) if not shell.is_empty() else (solid, cavities)

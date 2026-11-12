"""Spool adapters: the computational core of a hub that fills the hole of a filament spool.

No Qt in this file. It gives the adapter as one manifold, so it can be tested on its own and
reused in the model and the export.

A **spool adapter** is a hub of `hub` millimetres across that fills the big hole in the middle of
a filament spool, so the spool turns true on a rod that is much thinner than that hole or on the
608 bearing of a spool holder. It is `depth` millimetres long and stands on a **flange** of
`hub + 2 * lip` millimetres that is `FLANGE` millimetres thick and is what keeps the adapter from
sliding into the spool. The adapter prints **flat on that flange**: every layer is a ring, the
step from the flange to the hub faces outwards and nothing hangs over.

The hub is made `HUB_PLAY` millimetres smaller than the hole it is asked for, so it goes in by
hand, and the outside edge of its top is chamfered `CHAMFER` millimetres, so pushing the adapter
in does not cut into the wall of the hole. The **bore** in the middle is `rod` plus `ROD_PLAY`
millimetres, for the rod that turns through it; with `bearing` it is instead a pocket of
`BEARING_D` plus `BEARING_PLAY` millimetres, `BEARING_DEPTH` deep from the top, that a 608 bearing
presses into, with a hole of `AXLE_D` millimetres through below it for the axle.

A hub that only has to hold a spool round does not have to be solid. With `spokes` the plastic
between an **inner ring** of `RING_IN` millimetres around the bore and an **outer ring** of
`RING_OUT` millimetres inside the outside of the hub is cut away, except for that many bars of
`SPOKE_W` millimetres that run from the middle out and carry the spool round. The pockets between
them stand on the flange and are open at the top, so they save plastic without costing a wall;
`spokes` = 0 keeps the hub solid.

The bore always leaves `MIN_WALL` millimetres of plastic to the outside of the hub: a rod or a
bearing that is too fat for the hole asked for makes the **hub larger** instead of thin, so the
adapter is never a shell of two millimetres that splits as soon as the spool hangs on it.
"""
from __future__ import annotations

import math

import manifold3d as mf

from .i18n import tr

#: mm the hole of the spool that the hub is made smaller than, so it goes in by hand
HUB_PLAY = 0.4
#: mm the bore is made wider than the rod that turns through it
ROD_PLAY = 0.3
#: mm a 608 bearing is across, the play on it and mm deep the pocket that holds it is
BEARING_D, BEARING_PLAY, BEARING_DEPTH = 22.0, 0.15, 7.0
#: mm the axle of a 608 bearing is, which is the hole that stays under its pocket
AXLE_D = 12.0
#: mm thick the flange is that keeps the adapter from sliding into the spool
FLANGE = 2.0
#: mm of plastic that always stays between the bore and the outside of the hub
MIN_WALL = 3.0
#: mm the outside edge of the top of the hub is chamfered, so it slides into the hole
CHAMFER = 1.0
#: mm of wall that stays around the bore, and mm of wall that stays inside the hub
RING_IN, RING_OUT = 4.0, 2.5
#: mm wide a spoke is
SPOKE_W = 4.0
#: the fewest and the most spokes; 0 keeps the hub solid
SPOKES_MIN, SPOKES_MAX = 3, 8
#: the smallest and the largest hole of a spool this fills
HUB_MIN, HUB_MAX = 25.0, 90.0
#: mm the hub goes into the spool
DEPTH_MIN, DEPTH_MAX = 5.0, 40.0
#: mm the flange reaches past the hub all round
LIP_MIN, LIP_MAX = 2.0, 10.0
#: mm the rod that turns through the adapter is
ROD_MIN, ROD_MAX = 5.0, 30.0
#: the straight pieces a round part is approximated with
SEGMENTS = 96
#: mm past a face that a cut reaches, so it does not stop in the face it cuts
OVER = 1.0


def _number(value, lo, hi):
    """A number of the adapter, clamped to what prints; anything but a number is refused."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        v = math.nan
    if not math.isfinite(v):
        raise ValueError(tr("the sizes of a spool adapter must be numbers"))
    return max(lo, min(hi, v))


def _spokes(value):
    """How many spokes hold the hub together: 0 stays 0 (a solid hub), the rest is 3 to 8."""
    n = int(_number(value, 0.0, float(SPOKES_MAX)))
    return 0 if n == 0 else max(SPOKES_MIN, n)


def _sizes(hub, depth, lip, rod, bearing, spokes) -> dict:
    """The clamped numbers and the places of the features, in the frame the adapter prints in.

    In that frame the flange lies on the bed: it stands from `z = 0` to `z = FLANGE` and is
    `flange` millimetres across, and the hub of `hub` millimetres rises from its top to
    `z = height`. `bore` is the diameter of the hole through the middle and `pocket` that of the
    bearing pocket, which is cut `BEARING_DEPTH` down from the top (0 when the adapter is made for
    a rod); `wall` is the plastic that is left between the widest of them and the outside of the
    hub. `ring_in` and `ring_out` are the radii of the annulus the spokes hold together, and
    `hollow` says whether that annulus is a spoke wide enough to cut anything out of.
    """
    hole = _number(hub, HUB_MIN, HUB_MAX)
    dep = _number(depth, DEPTH_MIN, DEPTH_MAX)
    lip_ = _number(lip, LIP_MIN, LIP_MAX)
    rod_ = _number(rod, ROD_MIN, ROD_MAX)
    bear = bool(bearing)
    n = _spokes(spokes)
    bore = rod_ + ROD_PLAY if not bear else AXLE_D
    pocket = BEARING_D + BEARING_PLAY if bear else 0.0
    wide = max(bore, pocket)          # the widest thing that goes through the middle
    # the hub is a fifth of a millimetre under the hole all round, but never so small that what
    # goes through the middle leaves less than MIN_WALL of plastic to the outside
    hub_ = max(hole - HUB_PLAY, wide + 2.0 * MIN_WALL)
    ring_in, ring_out = 0.5 * wide + RING_IN, 0.5 * hub_ - RING_OUT
    return {"hub": hub_, "depth": dep, "lip": lip_, "rod": rod_, "bearing": bear, "spokes": n,
            "bore": bore, "pocket": pocket, "flange": hub_ + 2.0 * lip_, "height": FLANGE + dep,
            "wall": 0.5 * (hub_ - wide), "ring_in": ring_in, "ring_out": ring_out,
            "hollow": n > 0 and ring_out - ring_in >= SPOKE_W}


def sizes(hub=53.0, depth=12.0, lip=4.0, rod=8.0, bearing=False, spokes=5) -> dict:
    """The numbers a spool adapter is made of, so a dialog can show them.

    `hub` is the hole of the spool, `depth` how far the hub goes into it and `lip` how far the
    flange reaches past the hub all round. `rod` is the rod that turns through the middle, or
    `bearing` makes a pocket for a 608 bearing with an axle hole under it; `spokes` is how many of
    them hold the hub together, 0 being a solid hub. `hub`, `bore`, `pocket`, `flange`, `height`
    and `wall` are what the adapter is actually made with: the hub is `HUB_PLAY` millimetres under
    the hole unless that would leave too little wall around the bore, in which case it is made
    larger. Every number is clamped to what prints; anything but a number is a ValueError.
    """
    g = _sizes(hub, depth, lip, rod, bearing, spokes)
    return {key: g[key] for key in ("hub", "depth", "lip", "rod", "bearing", "spokes", "bore",
                                    "pocket", "flange", "height", "wall", "hollow")}


def _revolved(points) -> mf.Manifold:
    """The solid of revolution of (radius, height) points that rise from the bottom to the top.

    Consecutive points are joined by a cone, so a step in the radius becomes a flat ring. The
    profile is closed along the axis and turned round it in one piece.
    """
    profile = [(0.0, points[0][1])]
    for p in list(points) + [(0.0, points[-1][1])]:
        if abs(p[0] - profile[-1][0]) > 1e-9 or abs(p[1] - profile[-1][1]) > 1e-9:
            profile.append((float(p[0]), float(p[1])))
    return mf.Manifold.revolve(mf.CrossSection([profile]), SEGMENTS)


def _body(g) -> mf.Manifold:
    """The flange with the hub on it and the chamfer on the top edge of that hub."""
    rf, rh, h = 0.5 * g["flange"], 0.5 * g["hub"], g["height"]
    c = min(CHAMFER, g["depth"], g["wall"])
    return _revolved([(rf, 0.0), (rf, FLANGE), (rh, FLANGE), (rh, h - c), (rh - c, h)])


def _cylinder(d, z0, z1) -> mf.Manifold:
    """A cylinder of `d` millimetres across along the axis of the adapter, from `z0` to `z1`."""
    return mf.Manifold.cylinder(z1 - z0, 0.5 * d, 0.5 * d, SEGMENTS).translate((0.0, 0.0, z0))


def _bores(g) -> list:
    """The cylinders that cut the middle out: the bore for a rod, or the pocket and the axle."""
    if not g["bearing"]:
        return [_cylinder(g["bore"], -OVER, g["height"] + OVER)]
    # the axle reaches a little into the pocket above it, so the two cuts do not only touch
    return [_cylinder(g["pocket"], g["height"] - BEARING_DEPTH, g["height"] + OVER),
            _cylinder(g["bore"], -OVER, g["height"] - BEARING_DEPTH + OVER)]


def _pockets(g) -> mf.Manifold:
    """The plastic that is cut out of a spoked hub; empty when the hub stays solid.

    The annulus between the two rings is taken out over the whole height of the hub, except for
    the spokes, which run from the middle out and are kept. What is left stands on the flange and
    is open at the top, so the pockets cost no overhang.
    """
    if not g["hollow"]:
        return mf.Manifold()
    reach = g["ring_out"] + OVER
    spoke = mf.CrossSection.square([SPOKE_W, reach]).translate([0.0, 0.5 * reach])
    bars = mf.CrossSection()
    for i in range(g["spokes"]):
        bars = bars + spoke.rotate(i * 360.0 / g["spokes"])
    annulus = (mf.CrossSection.circle(g["ring_out"], SEGMENTS)
               - mf.CrossSection.circle(g["ring_in"], SEGMENTS))
    return mf.Manifold.extrude(annulus - bars, g["height"] + OVER - FLANGE) \
        .translate((0.0, 0.0, FLANGE))


def spool_adapter(hub=53.0, depth=12.0, lip=4.0, rod=8.0, bearing=False,
                  spokes=5) -> mf.Manifold:
    """One printable spool adapter: a hub that fills the hole of a filament spool.

    The adapter stands on its flange on `z = 0` and is centred in x and y: the flange is
    `hub + 2 * lip` millimetres across and `FLANGE` millimetres thick, and the hub of `hub`
    millimetres goes `depth` millimetres into the spool above it, with its top edge chamfered so
    it slides in. The middle is bored for a rod of `rod` millimetres plus `ROD_PLAY`, or with
    `bearing` for a 608 bearing: a pocket of `BEARING_D` plus `BEARING_PLAY` millimetres,
    `BEARING_DEPTH` deep from the top, and a hole of `AXLE_D` millimetres through below it.
    `spokes` cuts the hub open between the ring around the bore and the ring inside its outside,
    leaving that many bars of `SPOKE_W` millimetres to carry the spool round; 0 keeps it solid.

    The hub is made `HUB_PLAY` millimetres smaller than the hole it is asked for, but never so
    small that the bore leaves less than `MIN_WALL` millimetres of plastic to the outside: a rod
    or a bearing that is too fat for that hole makes the hub larger instead. Every number is
    clamped to what prints; anything but a number is a ValueError.
    """
    g = _sizes(hub, depth, lip, rod, bearing, spokes)
    m = _body(g)
    for cut in _bores(g):
        m = m - cut
    pockets = _pockets(g)
    return m if pockets.is_empty() else m - pockets

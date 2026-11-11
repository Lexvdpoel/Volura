"""Zip-tie mounts: the computational core of a printed cable-tie anchor you screw onto a surface.

No Qt in this file. It gives the mount as one manifold, so it can be tested on its own and reused
in the model and the export.

A **zip-tie mount** is the small part a cable tie goes through, so a bundle of cables can be held
to a wall, a desk top or the inside of a machine without providing a place for the tie itself. It
is a flat **base** that lies against the surface and one **screw** through its middle, and a raised
**bridge** that the tie slides through. The tie is held by the tunnel it slides into and the mount
by the screw, and neither pulls on the other.

The **tunnel** is `tie_width + 0.6` millimetres wide and `tie_thick + 0.6` high, measured at its
sides, and it starts `PLAY` millimetres above the top of the base, so a tie still slides in when
the first layer squashed out a little wide. It is horizontal in the print, so its ceiling may not
lie flat over the bed: from both sides the top closes on a roof of 45 degrees to a point, the way
a teardrop hole does, and the mount prints without support. `ways` is how many tunnels there are:
one along X, or two that cross so the tie can go either way.

The **screw** stands in the middle, where the tunnel or the two tunnels cross. It goes in from
above through the opening in the top of the bridge, past the tunnel, and its head ends in the top
of the base under it, so the tie lies flat over the head. `countersink` sinks a flat head into the
base until it is flush with that floor, and the base is then as thick as that cone is deep plus
`UNDER` millimetres, which is all that stays under a head; without it the hole is straight. The
opening through the top of the bridge is as wide as the head plus `HEAD_ROOM` millimetres, and as
wide as the cone of a sunk head where it is widest, so a screwdriver and the head come through from
above and the hole never turns into a ceiling on its way up.

The **bridge** is as wide as the wider of the tunnel and that opening, with `WALL` millimetres of
wall beside them on both sides, and it stands inside the chamfer of the base, so nothing of it
hangs over the plate. A bridge that does not fit on the base is made as wide as the base allows.
"""
from __future__ import annotations

import math

import manifold3d as mf

from . import enclosure, geom, threads
from .i18n import tr

#: mm the smallest and the largest base plate
BASE_MIN, BASE_MAX = 12.0, 40.0
#: mm the narrowest and the widest cable tie, and the thinnest and the thickest
TIE_W_MIN, TIE_W_MAX = 2.5, 9.0
TIE_T_MIN, TIE_T_MAX = 1.0, 2.5
#: mm the tallest a mount gets: above that it is a post, not an anchor
HEIGHT_MAX = 15.0
#: mm thick a base plate is when no screw head sinks into it
BASE = 2.0
#: mm of plastic that always stays under the head of a screw
UNDER = 1.0
#: mm the corners of the base and of the bridge are rounded to, and mm of chamfer on their top edge
CORNER, CHAMFER = 2.0, 0.5
#: mm of play between the tie and its tunnel, and mm of bridge that stays over the roof
PLAY, ROOF = 0.6, 1.6
#: mm of wall that always stays beside the tunnel and beside the opening for the head
WALL = 2.4
#: mm the opening for the screwdriver and the head is wider than the head
HEAD_ROOM = 0.4
#: the screws a mount is made for
SCREWS = ("M3", "M4", "M5")
#: one tunnel for the tie, or two that cross
WAYS_MIN, WAYS_MAX = 1, 2
#: the straight pieces a round hole is approximated with
SEGMENTS = 64
#: mm past a cut, so it leaves no skin behind
MARGIN = 0.5
#: mm of slab at either end of a chamfer, of which the hull of a chamfer is made
SLIVER = 0.2


def _number(value, lo, hi):
    """A number of the mount, clamped to what prints; anything but a number is refused."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        v = math.nan
    if not math.isfinite(v):
        raise ValueError(tr("the sizes of a zip-tie mount must be numbers"))
    return max(lo, min(hi, v))


def _ways(value) -> int:
    """How many tunnels the tie gets: a whole number of them, 1 or 2."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        v = math.nan
    if not math.isfinite(v):
        raise ValueError(tr("the number of ways of a zip-tie mount must be a whole number"))
    return max(WAYS_MIN, min(WAYS_MAX, int(v)))


def _sizes(base, tie_width, tie_thick, height, screw, countersink, ways) -> dict:
    """The clamped numbers and the places of the features, in the frame the mount prints in.

    In that frame the base lies on the bed and is centred on the origin: it is `base` millimetres
    square and `plate` thick, and the bridge stands on it up to `height`. The tunnel runs along X
    from `x = -base / 2` to `x = +base / 2` and, for two of them, also along Y; its floor is the
    plane `floor`, its sides are `slot_h` high and its peak is `peak`. The screw goes through the
    origin: the opening of `open_d` comes down to the top of the base, where the head of the screw
    ends, and the hole of `hole_d` goes on through the plate to the bed. The bridge is `bridge`
    wide and `length` long, with corners of `radius`, and stands inside the chamfer of the base,
    which is why it is never longer than `base - 2 * CHAMFER`.
    """
    b = _number(base, BASE_MIN, BASE_MAX)
    w = _number(tie_width, TIE_W_MIN, TIE_W_MAX)
    t = _number(tie_thick, TIE_T_MIN, TIE_T_MAX)
    size = str(screw)
    if size not in SCREWS:
        raise ValueError(tr("unknown screw size {size}").format(size=size))
    n, sink = _ways(ways), bool(countersink)
    hole_d, head_d, _head_h, cs_head_d = threads.SCREW_HOLES[size]
    # the head that comes through the top of the bridge: the flat one, or the socket one
    head = cs_head_d if sink else head_d
    # a cone of 90 degrees is as deep below the floor as it is wider than the hole it ends in
    cone = 0.5 * (cs_head_d - hole_d)
    plate = max(BASE, cone + UNDER) if sink else BASE
    slot_w, slot_h = w + PLAY, t + PLAY
    floor = plate + PLAY
    peak = floor + slot_h + 0.5 * slot_w        # the roof rises 45 degrees from both sides
    # a head that sinks needs a thicker plate, and a thicker plate stands the tunnel higher
    h = _number(height, peak + ROOF, HEIGHT_MAX)
    # the opening the screwdriver and the head come through: as wide as the head plus HEAD_ROOM,
    # and for a sunk head as wide as the widest point of the cone below it - a hole that becomes
    # narrower on its way up is a ceiling the print would have to hang from
    open_d = head + HEAD_ROOM
    if sink:
        open_d = max(open_d, cs_head_d + threads.DEFAULT_HOLE_CLEARANCE + 2.0 * MARGIN)
    # the bridge stands on the top face of the chamfered plate, which is that much smaller
    span = b - 2.0 * CHAMFER
    wide = min(max(slot_w, open_d) + 2.0 * WALL, span)
    return {"base": b, "tie_width": w, "tie_thick": t, "height": h, "screw": size,
            "countersink": sink, "ways": n, "plate": plate, "cone": cone, "floor": floor,
            "slot_w": slot_w, "slot_h": slot_h, "peak": peak, "roof": h - peak,
            "hole_d": hole_d + threads.DEFAULT_HOLE_CLEARANCE, "head": head,
            "open_d": open_d, "bridge": wide, "length": span if n == 1 else wide,
            "radius": min(CORNER, 0.5 * wide)}


def sizes(base=20.0, tie_width=5.0, tie_thick=1.5, height=6.0, screw="M4", countersink=True,
          ways=2) -> dict:
    """The numbers a zip-tie mount is made of, so a dialog can show them.

    `base` is the size of the plate that lies against the surface and `tie_width` and `tie_thick`
    are the cable tie that slides through the mount. `height` is how high the mount stands on the
    bed, which is raised when it would leave less than `ROOF` millimetres over the peak of the
    tunnel; `screw` is one of `SCREWS`, `countersink` whether a flat head sinks into the floor of
    the tunnel and `ways` how many tunnels cross there. `slot_w` and `slot_h` are the tunnel wide
    and high at its sides, `bridge` is the bridge wide, `roof` the plastic that stays above the
    peak of the tunnel and `hole_d` the screw hole as it prints. Every number is clamped to what
    prints; anything but a number is a ValueError.
    """
    s = _sizes(base, tie_width, tie_thick, height, screw, countersink, ways)
    return {key: s[key] for key in ("base", "tie_width", "tie_thick", "height", "screw",
                                    "countersink", "ways", "slot_w", "slot_h", "bridge", "roof",
                                    "hole_d")}


def _box(length, width, radius, height, z=0.0) -> mf.Manifold:
    """A box `length` by `width` with corners of `radius`, `height` high, standing at `z` and
    centred on the origin in plan."""
    return mf.Manifold.extrude(enclosure.rounded_rect(length, width, radius),
                               height).translate((0.0, 0.0, z))


def _chamfered(length, width, radius, height, z=0.0) -> mf.Manifold:
    """The same box with its top edge chamfered `CHAMFER` millimetres.

    The chamfer is the hull of a thin slab of the one outline and a thin slab of the same outline
    inset by `CHAMFER`: inset a rounded rectangle and its corners get a `CHAMFER` tighter, so the
    hull leans 45 degrees along the whole edge, also round the corner.
    """
    c = min(CHAMFER, 0.5 * height, radius)
    parts = [_box(length, width, radius, height - c, z)]
    parts.append(mf.Manifold.batch_hull([_box(length, width, radius, SLIVER, z + height - c),
                                         _box(length - 2.0 * c, width - 2.0 * c, radius - c,
                                              SLIVER, z + height - SLIVER)]))
    return mf.Manifold.batch_boolean(parts, mf.OpType.Add)


def _sideways(cs, axis, reach) -> mf.Manifold:
    """The prism over the cross-section `cs`, drawn out along `axis` from -`reach` to +`reach`.

    `cs` is written in the plane you look at that axis from: (y, z) for "x" and (x, z) for "y",
    so the roof of a tunnel is drawn the way it is measured. That turn lays the extrusion along
    +x but along -y, so a prism along y grows back from `reach`.
    """
    rot = (geom.rotation_matrix((1.0, 1.0, 1.0), 2.0 * math.pi / 3.0) if axis == "x"
           else geom.rotation_matrix((1.0, 0.0, 0.0), 0.5 * math.pi))
    at = (-reach, 0.0, 0.0) if axis == "x" else (0.0, reach, 0.0)
    return mf.Manifold.extrude(cs, 2.0 * reach).transform(geom.affine(rot, at))


def _tunnel(s) -> mf.CrossSection:
    """The tunnel as you look into it: the slot the tie slides through, closed on a roof.

    The slot is `slot_w` wide and `slot_h` high between its sides, and from both sides the top
    rises at 45 degrees to a point `slot_w / 2` higher. That roof is why a tunnel that runs
    sideways needs no support: no layer hangs over the one below it.
    """
    w, h, z = s["slot_w"], s["slot_h"], s["floor"]
    return mf.CrossSection([[(-0.5 * w, z), (0.5 * w, z), (0.5 * w, z + h),
                             (0.0, z + h + 0.5 * w), (-0.5 * w, z + h)]])


def tie_mount(base=20.0, tie_width=5.0, tie_thick=1.5, height=6.0, screw="M4", countersink=True,
              ways=2) -> mf.Manifold:
    """One printable zip-tie mount, lying on its base on `z = 0` and centred on the origin.

    The base is a plate of `base` by `base` millimetres with corners of `CORNER` and its top edge
    chamfered, and out of the middle of it rises the bridge, `height` millimetres above the bed in
    total. Through that bridge runs the tunnel the tie slides through: `tie_width + 0.6` wide and
    `tie_thick + 0.6` high at its sides, closed on a roof of 45 degrees, along X and for `ways` 2
    also along Y. The screw goes in from above through the bridge and its head ends in the top of
    the base under the tunnel, so the tie lies flat over it: `screw` of `SCREWS`, with a flat head
    sunk flush into that floor when `countersink` and a straight hole when it is off.

    Every number is clamped to what prints, the height is raised when it would leave too little
    roof over the tunnel, a bridge too wide for the base is made to fit it, and nothing hangs over
    steeper than 45 degrees; anything but a number, or a screw that is not in `SCREWS`, is a
    ValueError.
    """
    s = _sizes(base, tie_width, tie_thick, height, screw, countersink, ways)
    solid = _chamfered(s["base"], s["base"], CORNER, s["plate"])
    solid = solid + _chamfered(s["length"], s["bridge"], s["radius"], s["height"] - s["plate"],
                               s["plate"])
    reach = 0.5 * s["base"] + MARGIN
    tunnels = [_sideways(_tunnel(s), "x", reach)]
    if s["ways"] == 2:
        tunnels.append(_sideways(_tunnel(s), "y", reach))
    solid = solid - mf.Manifold.batch_boolean(tunnels, mf.OpType.Add)
    # the room the screwdriver and the head come through, down to the top of the base
    room = mf.Manifold.cylinder(s["height"] - s["plate"] + MARGIN, 0.5 * s["open_d"],
                                0.5 * s["open_d"], SEGMENTS).translate((0.0, 0.0, s["plate"]))
    kind = "countersink" if s["countersink"] else "through"
    return threads.screw_hole(solid - room, (0.0, 0.0, s["plate"]), (0.0, 0.0, -1.0), s["plate"],
                              s["screw"], kind, segments=SEGMENTS)

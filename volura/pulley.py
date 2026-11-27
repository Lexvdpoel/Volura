"""Timing pulleys and idlers: the computational core for a printed pulley that takes a belt.

No Qt in this file. It gives the pulley and the idler as closed volumes that stand on `z = 0`
and are centred on the origin, so they can be tested on their own and reused in the model and
the export.

A **timing belt** is a belt of `width` millimetres with teeth moulded on its inside, and a
**timing pulley** is the wheel those teeth ride in. The wheel is not a gear: the belt is flexible
and its teeth have to run into the wheel right to the bottom of the grooves, so the size of a
pulley comes from the **pitch** `p` of the belt — the distance from one tooth to the next,
measured on the **pitch circle**, the circle on which the belt rolls without slipping:

    pitch diameter = teeth × pitch / π

The teeth of the wheel stand *under* that circle, not above it: the pitch line of a belt lies in
its reinforcement, at the cords, and a belt tooth reaches from the cords down into the groove of
the wheel and stands `PLD` (the **pitch line differential**) above the cords with the rest of its
body. The circle the wheel is turned to is therefore

    outside diameter = pitch diameter − 2 × PLD

which is what a supplier's table says for a pulley and what you measure on it with callipers. A
spur gear (see `volura/gears.py`) is the opposite: its teeth stand above the pitch circle, because
two stiff wheels touch each other *on* that circle.

**The numbers, and where each comes from.** `PLD` is the belt's pitch line differential:
0.254 mm (0.010 in) for GT2, 0.381 mm (0.015 in) for GT3 and for HTD 3M — the correction every
supplier of those profiles prints beside the pitch and the outside diameter, and the reason a GT2
wheel of 20 teeth is 20 × 2 / π = 12.73 mm on the pitch circle and 12.22 mm across its teeth. The
**groove** a belt tooth runs into is given per profile as the depth the wheel is cut to and the
width of its mouth, measured on the outside circle: 0.764 and 1.494 mm for GT2, 1.169 and
2.310 mm for GT3, 1.289 and 2.270 mm for HTD 3M. Those are the pocket dimensions of the SDP/SI
timing belt tables — "GT2 2 mm", "GT2 3 mm" and "HTD 3 mm" — the tables the open printed-pulley
models are built from. The height of a **belt tooth** above the backing (0.75 mm for GT2, 1.14 mm
for GT3, 1.28 mm for HTD 3M) and the **thickness** of the whole belt (1.38, 2.40 and 2.40 mm) are
the cross-section figures of the matching belt tables, the numbers on the "2GT", "3GT" and "3M"
lines of any supplier's belt table. A belt tooth reaches `PLD + tooth` under the pitch circle, so
the smooth back of the belt runs `thickness − PLD − tooth` outside it — 0.376 mm for GT2,
0.879 mm for GT3 and 0.739 mm for HTD 3M — and that is what an idler has to keep.

The **groove** is drawn as the belt tooth it has to take: a mouth that opens on the outside circle,
two flanks that lean `flank` degrees off the radius as they rise into the wheel, and a flat bottom
at the depth of the pocket the table gives, which is the circle the tip of a belt tooth stands on.
The pockets of the real profiles curve a little and are rounded in their root corners, so a printed
wheel runs a touch tighter than the table says and the print clearance is what makes that up; a
printed groove with straight flanks and a flat bottom is what takes these belts well. `flank` is
15 degrees for the GT profiles and 20 degrees for HTD, whose tooth is lower and wider. The mouth
corners lie on the outside circle, so the width of a mouth is measured on the wheel and the
diameter over the teeth is the published outside diameter.

**Print clearance** goes on the groove and not on the wheel: a printed wall comes out fatter than
drawn and a belt tooth is moulded plastic, so a groove is made `clearance` deeper and
`2 × clearance` wider at its mouth. The outside diameter stays what the table says, because that
is where the pitch line of the belt has to end up; it is the bottom of the groove that gives.

**Flanges** are the collars that keep the belt on the wheel. They stand `flange` millimetres past
the teeth, on one or on both sides of the belt, and are `FLANGE_T` millimetres thick. The collar
the part stands on is a straight ring out of the bed, so the only thing it looks down at is the
bed. The collar in the air cannot stand on nothing, so it rises out of the band at its edge on a
**45 degree slope** to its full height, which is the flattest slope that prints without support;
its knife edge is where the band ends, so the slope is clear of the belt that runs between the
collars. Besides that slope the only face of the band that looks down is the roof the wheel itself
makes over the mouth of each groove, which is no wider than that mouth. Between two collars a belt
has `SIDE_ROOM` millimetres of room, because printed collars come out fatter than drawn and a belt
has to slide between them. On the side that carries no collar the belt runs off the wheel and gets
half of that room there, with the `SINK` of the hub on top of it, so the hub never stands in the
path of the belt.

The **hub** is the boss the wheel is built around. It is bored for the axle, can have a **D flat**
so the wheel cannot spin on a shaft that is filed flat, and can carry an **M3 set screw** that
runs radially through the hub into the bore. That screw finds its threads in a nut that is
captured in a **nut slot**: a slot with straight vertical walls, as wide across as the nut plus
`NUT_PLAY`, whose inner face stands `NUT_GAP` millimetres from the bore and whose outer face is
the height of the nut plus its play further out. The wall in front of the nut therefore stays —
only the screw hole pierces it — so the screw bears on the nut and presses on the axle instead of
pushing the nut away from it, and an axle of the bore still goes in beside the nut. The slot is
open at the top and has straight walls, so the nut drops in from above and nothing in it hangs
over the bed. The hub grows until that nut fits in it with `NUT_WALL` millimetres of wall around
it, as it grows to hold the bore; only a hub that is too **short** for the nut is made without the
screw.

**An idler** is the smooth wheel the belt runs over, toothless, on the side of the belt that has
no teeth: a tensioner or a return roller. Its running surface stands one thickness of belt between
the cords and the back under the pitch circle of the tooth count it is asked for, so the cords run
over the idler on the circle they run on over a toothed wheel of that tooth count and the same
belt keeps the same path. It carries a flange on each side of the belt and is bored for a 625
bearing — 16 mm across and 5 mm wide — or for a plain bush. Under the pocket of that bearing the
hole through the wheel is `BEARING_RELIEF` millimetres, wider than the inner ring of the bearing,
so the floor the outer ring stands on never touches the axle that does not turn: the wheel turns on
the bearing and not on its axle.

**Printing.** The part stands on `z = 0` on the collar that is on that side, or on the end face of
its band when it carries no collar, and a hub stands up in the air above the part it belongs to.
No face looks down by more than 45 degrees from the vertical: band, collars and hub are straight
cylinders, the mouth of every groove is bridged by the wheel above it, which is a bridge no wider
than that mouth, a hub that is wider than the face it stands on leans back to the edge of that face
on a cone that stands `SINK` into the wheel and is therefore steeper than 45 degrees, the slopes
under the collars in the air are those 45 degrees, the bore is a hole straight up, the set screw
hole is a horizontal cylinder that is open at both of its ends and the nut slot has straight walls
and no roof. The part can also be turned over, so that it
stands on the collar that carries the slope and the hub stands on the bed under everything; then the
other collar has nothing to lean on and bridges across, which prints when a collar stands little
past the teeth and is the reason the slope is on the collar in the air here.
"""
from __future__ import annotations

import math

import manifold3d as mf
import numpy as np

from .i18n import tr
from . import threads

#: the fewest and the most teeth a wheel of these profiles can have: below ten the groove of one
#: tooth runs into the next and a belt cannot seat
TEETH_MIN, TEETH_MAX = 10, 120
#: the narrowest and the widest belt; the widths a supplier sells are BELT_WIDTHS, and any other
#: number in the range is a belt that is wound on the wheel
BELT_MIN, BELT_MAX = 3.0, 20.0
#: the belt widths of these profiles, shortest first
BELT_WIDTHS = (6.0, 9.0, 10.0, 15.0)
#: mm of print clearance that is asked for by default, and the most that still leaves a groove
CLEAR_DEFAULT, CLEAR_MAX = 0.1, 0.5
#: mm a flange stands past the teeth, and mm how thick it is at its outer edge
FLANGE_MIN, FLANGE_MAX = 0.5, 4.0
FLANGE_T = 2.0
#: mm a belt needs between two printed collars, half of it on each side of it
SIDE_ROOM = 0.6
#: the smallest and the largest hub diameter and height the fields take
HUB_MIN, HUB_MAX = 6.0, 80.0
HUBH_MIN, HUBH_MAX = 4.0, 40.0
#: mm of hub that stays around the bore, and mm of hub outside the nut in its slot
WALL_MIN, NUT_WALL = 1.5, 1.2
#: mm of hub between the bore and the inner face of the nut, mm under it and mm over it in its slot
NUT_GAP, NUT_FLOOR, NUT_ROOM = 0.4, 0.4, 0.6
#: the play on the flats of the nut in its slot
NUT_PLAY = 0.2
#: the screw that clamps the wheel on the axle
SET_SCREW = "M3"
#: the 625 bearing an idler is bored for: mm across, mm wide, mm of its bore and the play on it
BEARING_D, BEARING_W, BEARING_BORE, BEARING_PLAY = 16.0, 5.0, 5.0, 0.15
#: mm of hole through the floor under the pocket of a bearing, so the axle and the inner ring of
#: the bearing stand clear of it and only the outer ring stands on the wheel
BEARING_RELIEF = 10.0
#: mm of room under the pocket of a bearing, so the bearing can drop in
POCKET_OVER = 1.0
#: mm a hub is sunk into the wheel it stands on, so the two are one piece
SINK = 0.5
#: mm of wheel that has to stay around a bearing, and mm that has to stay under a groove
WEB_MIN, BORE_MIN = 1.5, 2.0
#: the fewest and the most millimetres an idler surface may stand under the pitch circle
BACK_MIN, BACK_MAX = 0.2, 3.0
#: the straight pieces a round hole, a rim or a bearing is made of
SEGMENTS = 96
#: mm past a face that a cut reaches, so that a cut leaves no skin behind
MARGIN = 1.0

#: the belt profiles a wheel can be cut for
PROFILES = ("gt2", "gt3", "htd3m")
#: the choices of the flange field
FLANGES = ("none", "one", "both")
#: the choices of the bore of an idler
IDLER_BORES = ("bearing", "plain")

# profile: the pitch, the pitch line differential, the depth and the mouth of the groove, the
# height of a belt tooth and the thickness of the belt in millimetres, and the slope of the
# groove flanks in degrees; the module docstring says which published table each number is from
BELTS = {
    "gt2": {"pitch": 2.0, "pld": 0.254, "groove": 0.764, "mouth": 1.494, "tooth": 0.75,
            "thickness": 1.38, "flank": 15.0},
    "gt3": {"pitch": 3.0, "pld": 0.381, "groove": 1.169, "mouth": 2.310, "tooth": 1.14,
            "thickness": 2.40, "flank": 15.0},
    "htd3m": {"pitch": 3.0, "pld": 0.381, "groove": 1.289, "mouth": 2.270, "tooth": 1.28,
              "thickness": 2.40, "flank": 20.0},
}


def _number(value, lo, hi, what) -> float:
    """A size that is asked for, taken to what can be printed; no number at all is refused."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        v = math.nan
    if not math.isfinite(v):
        raise ValueError(tr("the size of a pulley {what} must be a number").format(what=what))
    return max(lo, min(hi, v))


def _belt(profile) -> dict:
    """The table of the belt that is asked for; another profile is refused."""
    key = str(profile)
    if key not in BELTS:
        raise ValueError(tr("unknown belt profile {profile}").format(profile=profile))
    return BELTS[key]


def _teeth(value) -> int:
    """The number of teeth, as a whole number in the range these profiles are made in."""
    try:
        f, z = float(value), int(float(value))
    except (TypeError, ValueError):
        raise ValueError(tr("the number of teeth of a pulley must be a whole number")) from None
    if z != f:
        raise ValueError(tr("the number of teeth of a pulley must be a whole number"))
    if z < TEETH_MIN or z > TEETH_MAX:
        raise ValueError(tr("the number of teeth must be between {lo} and {hi}")
                         .format(lo=TEETH_MIN, hi=TEETH_MAX))
    return z


def pitch_diameter(teeth, profile="gt2") -> float:
    """The pitch diameter of a wheel: the circle the belt runs on without slipping."""
    return _teeth(teeth) * float(_belt(profile)["pitch"]) / math.pi


def outside_diameter(teeth, profile="gt2") -> float:
    """The diameter over the teeth of a toothed wheel, as a supplier's table gives it."""
    return pitch_diameter(teeth, profile) - 2.0 * float(_belt(profile)["pld"])


def back_depth(profile) -> float:
    """How far under the pitch circle the smooth back of the belt runs: the surface of an idler.

    The backing of the belt lies above its cords and a tooth rises `tooth` below them, so what is
    left of the thickness stands outside the pitch circle: `thickness - pld - tooth`.
    """
    b = _belt(profile)
    return max(BACK_MIN, min(BACK_MAX, float(b["thickness"]) - float(b["pld"])
                             - float(b["tooth"])))


def idler_diameter(teeth, profile="gt2") -> float:
    """The diameter of the smooth surface of an idler that takes that belt."""
    return pitch_diameter(teeth, profile) - 2.0 * back_depth(profile)


def _wheel(profile, teeth, clearance, toothed=True) -> dict:
    """The circles of a toothed wheel, the shape of its grooves and how far its teeth lean out.

    With `toothed` the groove is what a belt tooth runs into and has to fit between its neighbours;
    an idler is turned to the same circles but is smooth, so those checks do not apply to it.
    """
    b = _belt(profile)
    z = _teeth(teeth)
    c = _number(clearance, 0.0, CLEAR_MAX, "clearance")
    pitch_d = z * float(b["pitch"]) / math.pi
    od = pitch_d - 2.0 * float(b["pld"])
    # a printed groove is made deeper and wider at its mouth, so a moulded belt tooth fits
    mouth, depth = float(b["mouth"]) + 2.0 * c, float(b["groove"]) + c
    phi = math.radians(float(b["flank"]))
    if od - 2.0 * depth < 1.0:
        raise ValueError(tr("The clearance leaves nothing of the teeth."))
    pitch_ang = 2.0 * math.pi / z
    wide = 2.0 * math.asin(min(1.0, mouth / od))
    if toothed and wide >= 0.98 * pitch_ang:
        raise ValueError(tr("The grooves of a wheel of {n} teeth of this profile run into one "
                            "another.").format(n=z))
    return {"profile": str(profile), "teeth": z, "clearance": c, "pitch": float(b["pitch"]),
            "pld": float(b["pld"]), "tooth": float(b["tooth"]), "flank": float(b["flank"]),
            "thickness": float(b["thickness"]), "pitch_d": pitch_d, "od": od,
            "root": od - 2.0 * depth, "mouth": mouth, "groove": depth,
            "tan": math.tan(phi), "pitch_ang": pitch_ang, "back": back_depth(profile)}


def _groove(w) -> np.ndarray:
    """One groove of a wheel, as a closed loop in the frame of the groove centred on +x.

    A groove is the negative of a belt tooth: a mouth that opens on the outside circle, two flanks
    that lean `flank` degrees off the radius as they rise into the wheel and a flat bottom at the
    depth of the pocket the table gives, which is the circle the tip of a belt tooth stands on. The
    mouth corners lie on the outside circle, so the width of a mouth is measured on the wheel and
    the diameter over the teeth is the published outside diameter. The loop is led `MARGIN`
    millimetres out past that circle along the same flanks, so the cut goes through the rim of the
    wheel instead of only touching it.
    """
    r, t = 0.5 * w["od"], w["tan"]
    a = 0.5 * w["mouth"]                      # half a mouth, on the outside circle
    u = r + MARGIN                            # how far the cut reaches out of the wheel
    v = max(0.1, a - w["groove"] * t)         # half the bottom of the groove
    return np.asarray([(r - w["groove"], -v), (r, -a), (u, -a - MARGIN * t), (u, a + MARGIN * t),
                       (r, a), (r - w["groove"], v)], dtype=float)


def _loops(w) -> list:
    """The grooves of a whole wheel, as closed loops in one plane."""
    out = []
    for i in range(w["teeth"]):
        a = i * w["pitch_ang"]
        c, s = math.cos(a), math.sin(a)
        p = _groove(w)
        out.append(np.ascontiguousarray(np.column_stack([p[:, 0] * c - p[:, 1] * s,
                                                         p[:, 0] * s + p[:, 1] * c])))
    return out


def _cyl(d, z0, z1) -> mf.Manifold:
    """A cylinder of `d` millimetres along the axis of the part, from `z0` to `z1`."""
    return mf.Manifold.cylinder(max(z1 - z0, 1e-6), max(0.5 * d, 1e-6), max(0.5 * d, 1e-6),
                                SEGMENTS).translate((0.0, 0.0, z0))


def _box(x0, x1, y0, y1, z0, z1) -> mf.Manifold:
    """A block of the part, for the flat of a bore and the slot of a nut."""
    cs = mf.CrossSection.square([max(x1 - x0, 1e-6), max(y1 - y0, 1e-6)])
    return mf.Manifold.extrude(cs, max(z1 - z0, 1e-6)).translate((x0, y0, z0))


def _radial(d, x0, x1, y, z) -> mf.Manifold:
    """A cylinder along the x of the part, for the screw that clamps the wheel on the axle."""
    m = mf.Manifold.cylinder(max(x1 - x0, 1e-6), max(0.5 * d, 1e-6), max(0.5 * d, 1e-6), SEGMENTS)
    return m.rotate((0.0, 90.0, 0.0)).translate((x0, y, z))


def _revolve(points) -> mf.Manifold:
    """The solid that a line of (radius, height) points makes as it turns round the axis.

    Consecutive points are joined by a straight cone, so a step in radius becomes a flat ring and a
    slope stays a slope. The line is closed along the axis, which makes one piece of it.
    """
    prof = [(0.0, float(points[0][1]))]
    for r, z in list(points) + [(0.0, float(points[-1][1]))]:
        if abs(r - prof[-1][0]) > 1e-9 or abs(z - prof[-1][1]) > 1e-9:
            prof.append((float(r), float(z)))
    return mf.Manifold.revolve(mf.CrossSection([prof]), SEGMENTS)


def _sizes(profile, teeth, clearance, width, flanges, flange, hub, hub_d, hub_h, bore, flat,
           nut, bearing, idler) -> dict:
    """Every number of the part, taken to what can be printed and made to fit in each other.

    `rim` is the radius the wheel is turned to: the outside circle of a toothed wheel, the radius
    of the smooth surface of an idler; `rf` is the radius its collars reach to. The part stands on
    `z = 0`: the collar that is on the bed is `FLANGE_T` millimetres thick, the band the belt runs
    on runs from `zb` to `ze` and the collar in the air rises from `ze` on a 45 degree slope to
    `zs`, `FLANGE_T` millimetres below the top of the part at `zt`. `hub_base` and `hub_top` are
    the ends of the hub and `height` is the top of the whole part.

    The sizes are made to fit in each other: the hub is grown until it holds the bore with a wall
    of `WALL_MIN` millimetres around it and until the nut of the set screw fits in it with
    `NUT_WALL` millimetres of wall in front of it, and a hub that is wider than the face it stands
    on is supported by that 45 degree slope. A set screw for which the hub is too short, or that
    would have nothing to screw into, is left out; a hub that is only too thin for that nut grows
    instead of losing the screw. An idler that is too small for the bearing it is bored for is
    grown until that bearing stands in it with a wall of `WEB_MIN` millimetres around it. A bore
    that would eat the bottom of the grooves and a tooth count whose grooves would run into one
    another are refused.
    """
    w = _wheel(profile, teeth, clearance, toothed=not idler)
    wide = _number(width, BELT_MIN, BELT_MAX, "belt width")
    fl = str(flanges)
    if fl not in FLANGES:
        raise ValueError(tr("unknown flange choice {choice}").format(choice=flanges))
    fh = _number(flange, FLANGE_MIN, FLANGE_MAX, "flange height")
    bear = bool(bearing) and bool(idler)
    # the circle the belt runs on: the wheel is toothed, the idler is smooth and stands back
    surface = idler_diameter(w["teeth"], w["profile"]) if idler else w["od"]
    rim = 0.5 * surface
    pocket = BEARING_D + 2.0 * BEARING_PLAY if bear else 0.0
    if bear and pocket + 2.0 * WEB_MIN > surface:
        # a bearing that is too fat for the wheel makes the wheel larger, as in a spool adapter:
        # an idler may be bigger than a belt likes, but it may not be thinner than a shell
        surface = pocket + 2.0 * WEB_MIN
        rim = 0.5 * surface
    near, far = fl in ("one", "both"), fl == "both"
    want = bool(hub) and not idler
    # the band of the belt, the collars on each side of it and the top of the part: `SIDE_ROOM` is
    # the room a belt needs between two printed collars, half of it on each side; on a side that
    # carries no collar the belt runs off the wheel and the hub stands there, so that side gets its
    # half of the room with the `SINK` of the hub on top of it, and the hub never stands in the
    # path of the belt
    rf = rim + fh if near else rim
    zb = (FLANGE_T + 0.5 * SIDE_ROOM) if near else 0.0
    free = 0.5 * SIDE_ROOM + (0.0 if far else (SINK if want else 0.0))
    ze = zb + wide + free
    zs, zt = (ze + fh, ze + fh + FLANGE_T) if far else (ze, ze)
    if bear:                       # the pocket of a bearing has to have a floor under it
        zt = max(zt, BEARING_W + POCKET_OVER + WEB_MIN)
    b = float(BEARING_BORE) if bear else _number(bore, 0.0, surface, "bore")
    if b > surface - 2.0 * BORE_MIN:
        raise ValueError(tr("The bore leaves too little material between the hole and the "
                            "{what}; use a smaller bore or more teeth.")
                         .format(what="surface of the idler" if idler else "teeth"))
    f = _number(flat, 0.0, 0.4 * b, "flat")
    tap = threads.SCREW_HOLES[SET_SCREW][0]
    s, m = threads.HEX_NUTS[SET_SCREW]
    floor, head = s / math.sqrt(3.0) + NUT_FLOOR, s / math.sqrt(3.0) + NUT_ROOM
    # the set screw that clamps the wheel on the axle and the nut it finds its threads in: the nut
    # stands in its slot on two corners and is gripped by its flats, so it cannot turn round, and
    # its inner face stands `NUT_GAP` from the bore, so an axle of the bore still goes in beside it
    xo = 0.5 * b + NUT_GAP + m + NUT_PLAY         # the outside face of the slot, in the hub wall
    want_nut = want and bool(nut) and b > 0.0 and not bear
    # the hub: wide enough to hold the bore and to stand the nut behind a wall of its own, standing
    # on the highest face of the wheel that is under it
    hd = max(_number(hub_d, HUB_MIN, HUB_MAX, "hub diameter"), b + 2.0 * WALL_MIN) if want else 0.0
    if want_nut:
        hd = max(hd, 2.0 * (xo + NUT_WALL))
    hh = _number(hub_h, HUBH_MIN, HUBH_MAX, "hub height") if want else 0.0
    stand, foot = (rf, zt) if far else (rim, zt)
    climb = max(0.0, 0.5 * hd - stand)
    hub_base, hub_top = (foot + climb, foot + climb + hh) if want else (zt, zt)
    # the screw is only left out when the hub is too short for the nut to drop into from above, or
    # when there is nothing for it to press on; a hub that is too thin for it simply grows
    screw = want_nut and hh >= floor + head
    screw_z = 0.5 * (hub_base + hub_top) if screw else 0.0
    return dict(w, width=wide, flanges=fl, flange=fh, near=near, far=far, rf=rf, zb=zb, ze=ze,
                zs=zs, zt=zt, surface=surface, rim=rim, idler=bool(idler), bearing=bear,
                pocket=pocket, bore=b, flat=f, hub=want, hub_d=hd, hub_h=hh, hub_base=hub_base,
                hub_top=hub_top, stand=stand, foot=foot, climb=climb, height=hub_top if want else zt,
                screw=tap if screw else 0.0, screw_z=screw_z, nut=screw, slot_z=screw_z - floor,
                slot_w=s + NUT_PLAY, slot_x=xo if want_nut else 0.0, nut_h=m, room=SIDE_ROOM)


#: the numbers `sizes` gives back
INFO_KEYS = ("profile", "teeth", "clearance", "pitch", "pld", "tooth", "thickness", "flank",
             "pitch_d", "od", "root", "mouth", "groove", "back", "width", "flanges", "flange",
             "near", "far", "rf", "zb", "ze", "zs", "zt", "surface", "rim", "idler", "bearing",
             "pocket", "bore", "flat", "hub", "hub_d", "hub_h", "hub_base", "hub_top", "screw",
             "screw_z", "nut", "slot_z", "slot_w", "slot_x", "height")


def sizes(profile="gt2", teeth=20, clearance=0.1, width=6.0, flanges="both", flange=1.0,
          hub=True, hub_d=16.0, hub_h=10.0, bore=5.0, flat=0.0, nut=True, bearing=False,
          idler=False) -> dict:
    """The numbers a pulley or an idler is made of, so a dialog or a test can look at them.

    The arguments are those of `timing_pulley` and `idler` together: `bearing` and `idler` make an
    idler, the rest a toothed wheel. The diameters are the published ones: `pitch_d` is the circle
    the belt runs on, `od` the circle over the teeth of a wheel and `surface` the circle an idler
    runs on, `root` the bottom of the grooves and `back` how far the smooth back of the belt stands
    under the pitch circle. `width` is the width of the belt, `zb` and `ze` are the heights the
    belt runs between, `zt` the top of the wheel and `height` the top of the whole part; `screw` is
    the diameter of the set screw hole and 0.0 when there is none, and `slot_z`, `slot_w` and
    `slot_x` are the floor, the width and the outside face of the slot that holds its nut, behind
    which a wall of the hub stays. As in `timing_pulley`, a size that is not a number, a profile, a
    flange choice or a tooth count that is unknown and a bore or a bearing that does not fit are
    refused; the rest is clamped to what prints.
    """
    g = _sizes(profile, teeth, clearance, width, flanges, flange, hub, hub_d, hub_h, bore, flat,
               nut, bearing, idler)
    return {key: g[key] for key in INFO_KEYS}


def _cutter(g) -> mf.Manifold:
    """The prisms that cut the grooves of a toothed wheel out of its band.

    One prism per tooth, the loops of `grooves` over the whole height the belt runs between, so the
    faces of the collars are the floors the belt teeth stand on and nothing is cut of the collars
    themselves. Where the wheel carries no collar on a side, the belt runs off the wheel there, and
    over that face the prisms reach `MARGIN` millimetres past it, so the cut leaves no skin standing
    over the grooves.
    """
    loops = _loops(g)
    z0 = g["zb"] - (0.0 if g["near"] else MARGIN)
    z1 = g["ze"] + (MARGIN if not g["far"] else 0.0)
    cs = mf.CrossSection([np.ascontiguousarray(q, dtype=float) for q in loops], mf.FillRule.NonZero)
    return mf.Manifold.extrude(cs, z1 - z0).translate((0.0, 0.0, z0))


def _body(g) -> mf.Manifold:
    """The wheel before it is toothed or bored: the band, the collars and the web under them.

    The collar on the bed is a ring of `FLANGE_T` millimetres that stands straight up out of the
    bed, so its only downward face is the bed itself. The collar in the air is that same ring, but
    since it cannot stand on nothing it rises out of the band on a slope of 45 degrees, which is
    the flattest slope that prints bare; its knife edge is where the band ends, so it is clear of
    the belt that runs between the collars.
    """
    pts = [(g["rf"], 0.0)]
    if g["near"]:
        pts.append((g["rf"], FLANGE_T))
    pts += [(g["rim"], FLANGE_T if g["near"] else 0.0), (g["rim"], g["zt"])]
    m = _revolve(pts)
    if not g["far"]:
        return m
    return m + _revolve([(g["rim"], g["ze"]), (g["rf"], g["zs"]), (g["rf"], g["zt"])])


def _hub(g) -> mf.Manifold:
    """The hub the wheel is built around, empty when the part is an idler or asks for none.

    The hub stands on the highest face of the wheel that is under it and is sunk `SINK` millimetres
    into it, so the two are one piece. A hub that is wider than the face it stands on is carried by
    a cone from the edge of that face up to the wall of the hub; because the hub stands `SINK` into
    the wheel that cone rises steeper than 45 degrees, which prints bare.
    """
    if not g["hub"]:
        return mf.Manifold()
    rh = 0.5 * g["hub_d"]
    if g["climb"] > 1e-9:
        # a hub that is wider than the face it stands on is carried by a cone that rises from the
        # edge of that face to the wall of the hub; standing SINK into the wheel makes that cone
        # steeper than 45 degrees, which prints bare
        pts = [(g["stand"], g["foot"] - SINK), (rh, g["hub_base"]), (rh, g["hub_top"])]
    else:
        pts = [(rh, g["foot"] - SINK), (rh, g["hub_top"])]
    return _revolve(pts)


def _cuts(g) -> list:
    """Everything that is taken out of the part, in the order the cuts are made in.

    The bore goes straight through the part and the `flat` files one side of it flat, so the wheel
    cannot turn on a shaft that is filed to match; that flat is a chord of the bore, so the cut
    stays inside the circle of the hole and the hub keeps its wall all round it, and it reaches
    over the top, so the flat runs the whole length of the hub too. The pocket of a bearing opens
    at the top of the part and is `BEARING_W` millimetres deep with `POCKET_OVER` millimetres of
    room under it, and under its floor the hole widens to `BEARING_RELIEF` millimetres, so the
    inner ring of the bearing and the axle stand clear of the wheel that turns round them. The set
    screw runs in from the outside of the hub straight to the middle, where the axle is, and its nut
    stands in a slot with straight walls that is open at the top of the hub; that slot stops at
    `slot_x`, so the wall of the hub in front of the nut stays and only the hole of the screw
    pierces it: the screw bears on the nut and presses on the axle instead of pushing the nut out.
    """
    g_high = g["height"] + MARGIN
    out = []
    if g["bore"] > 0.0:
        hole = _cyl(g["bore"], -MARGIN, g_high)
        if g["flat"] > 0.0:
            # the flat is a chord of the bore and what stands under it stays: the cut ends on the
            # plane of the flat instead of reaching past the hole, so the hub keeps its wall
            edge = 0.5 * g["bore"] - g["flat"]
            hole = hole - _box(-0.5 * g["bore"] - MARGIN, 0.5 * g["bore"] + MARGIN,
                               -0.5 * g["bore"] - MARGIN, -edge, -MARGIN - 1.0, g_high)
        out.append(hole)
    if g["bearing"]:
        floor = g["zt"] - BEARING_W - POCKET_OVER
        out.append(_cyl(g["pocket"], floor, g["zt"] + MARGIN))
        out.append(_cyl(BEARING_RELIEF, -MARGIN, floor))
    if g["screw"] > 0.0:
        out.append(_radial(g["screw"], 0.0, 0.5 * g["hub_d"] + MARGIN, 0.0, g["screw_z"]))
        out.append(_box(0.0, g["slot_x"], -0.5 * g["slot_w"], 0.5 * g["slot_w"],
                        g["slot_z"], g["height"] + MARGIN))
    return out


def timing_pulley(profile="gt2", teeth=20, clearance=0.1, width=6.0, flanges="both", flange=1.0,
                  hub=True, hub_d=16.0, hub_h=10.0, bore=5.0, flat=0.0,
                  nut=True) -> mf.Manifold:
    """One printable timing pulley: a wheel of teeth that takes a toothed belt.

    The wheel stands on `z = 0` on the collar that is on that side and is centred on the origin;
    `profile` is the belt it is cut for ("gt2", "gt3" or "htd3m"), `teeth` its number of teeth and
    `width` the width of the belt; the wheel is turned to the published outside diameter of that
    profile and tooth count and has a groove of that profile per tooth. `clearance` is the print
    clearance, which is given on the grooves: every groove is cut that much deeper and twice that
    much wider at its mouth, so a moulded belt tooth runs into it, while the outside circle stays
    what the table says. `flanges` keeps a belt on the wheel and is "none", "one" or "both", and
    `flange` is how far a collar stands past the teeth.

    With `hub` the wheel is built around a hub of `hub_d` millimetres that is `hub_h` millimetres
    above it and bored for an axle of `bore` millimetres; `flat` files one side of that bore flat
    `flat` millimetres deep, for a shaft that is filed to match, and `nut` gives the hub an M3 set
    screw hole that runs into the bore and a slot above it in which the nut of that screw is
    captured. That slot is open at the top and closed on the outside by a wall of the hub that only
    the screw hole pierces, so the screw bears on the nut and presses on the axle, and the hub is
    grown until it holds the bore and that nut with a wall around them. A hub that is too short for
    that nut, an idler, and a wheel without a bore are made without the screw. The part prints with
    its collar on the bed and its hub in the air; the faces that look down are the face on the bed,
    the slope of 45 degrees under the collar that is in the air, the cone of a hub that is wider
    than the face it stands on and the roof the wheel makes over the mouth of each groove, and none
    of them looks down by more than 45 degrees from the vertical.

    A profile, flange choice or tooth count that is not one of these, a size that is not a number,
    a tooth count whose grooves would run into one another and a bore that would eat the bottom of
    the grooves are refused with a message; every other size is clamped to what prints.
    """
    g = _sizes(profile, teeth, clearance, width, flanges, flange, hub, hub_d, hub_h, bore, flat,
               nut, False, False)
    m = _body(g) - _cutter(g)
    hub_m = _hub(g)
    if not hub_m.is_empty():
        m = m + hub_m
    for cut in _cuts(g):
        m = m - cut
    return m


def idler(profile="gt2", teeth=20, width=6.0, flange=1.0, bore=5.0, bearing=True,
          flat=0.0) -> mf.Manifold:
    """One printable idler: the smooth wheel a toothed belt runs over.

    The idler stands on `z = 0`, is `width` millimetres wide between its two collars and is turned
    to the circle its belt runs on: its surface stands `back` millimetres under the pitch circle of
    `teeth` teeth of `profile`, which is how far the smooth back of that belt stands under its own
    pitch line, so the cords of the belt run over the idler on the circle they run on over a toothed
    wheel of that tooth count and a belt keeps its path. The collars stand `flange` millimetres
    past that surface and keep the belt on it; the one in the air is supported by a 45 degree slope.

    With `bearing` the middle is bored for a 625 bearing: a pocket of 16.15 millimetres that is 5
    millimetres deep, open at the top, and a bore of `BEARING_BORE` millimetres through it. Under
    the floor of that pocket the hole widens to `BEARING_RELIEF` millimetres, which is more than
    the axle and more than the inner ring of the bearing, so the ring of floor that carries the
    outer ring never touches what stands still and the wheel turns on the bearing. Without
    `bearing` the middle is bored for a bush, of `bore` millimetres, and `flat` can file one side of
    it flat. The part prints on its collar and nothing hangs over by more than 45 degrees.

    An idler too small for that bearing and a bore that leaves too little wheel around it are
    refused with a message; every other size is clamped to what prints.
    """
    g = _sizes(profile, teeth, 0.0, width, "both", flange, False, HUB_MIN, HUBH_MIN, bore, flat,
               False, bearing, True)
    m = _body(g)
    for cut in _cuts(g):
        m = m - cut
    return m

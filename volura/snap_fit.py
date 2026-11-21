"""Snap-fit clips: the computational core for a printable hook and catch that click together.

No Qt in this file. It gives the clip as one closed volume, so it can be tested on its own and
reused in the model and the export.

A **snap-fit clip** is printed as one part with two pieces side by side on the bed: the **hook**
and the **catch**, `PIECE_ROOM` millimetres of bed apart, so they print in one go and click
together afterwards.

The **hook** is a base plate `w` millimetres wide, `BASE_T` millimetres thick and `BASE_L`
millimetres long, with a cantilever beam behind it: `L` millimetres long, `t` millimetres thick
and `w` millimetres wide, standing on the bed and bending inside its own layers. The beam ends
in a **barb**: a wedge that points `undercut` millimetres out on one side. Its slope of
`LEAD_IN` degrees faces the beam tip, so the edge of the slot rides over it and bends the beam
aside while the hook is pushed through; just behind the slope stands the retaining face, square
to the beam (90 degrees) so the hook holds — or, when `releasable`, leaning at `RELEASE_FACE`
degrees, so the hook can be pulled out again.

The **catch** is an upright wall `CATCH_WALL` millimetres thick, standing on the bed and facing
the beam tip, with a foot plate `BASE_L` long and `BASE_T` thick behind it to glue or screw
down. A slot of `t + clearance` wide and `CATCH_FLOOR + w + 2 clearance` high goes clean
through the wall, open on both sides: it is only as wide as the beam, so the barb has to bend
the beam aside by about its own depth to get through; once the barb is past the wall the beam
springs back and its retaining face hooks behind the wall. The wall stands `u + 1.5` millimetres
beside the slot on the side the barb points to — the retaining face needs something to hook on
— and 2 millimetres on the other side; `WALL_ABOVE` millimetres of wall stand over the slot.
The foot is two strips beside the slot, their inner edge 1 mm clear of the slot and of where
the barb swings, so neither the beam nor the barb can hit it. `CATCH_FLOOR` millimetres of wall
under the slot survive only beside the entrance: under the slot itself the wall is open to the
bed, because the beam enters along the bed and must go straight through.

The hook reaches behind the wall only when its retaining face comes `CATCH_WALL` plus the
clearance plus 0.5 mm past the front of the wall, and the lead-in slope eats `undercut / tan
(LEAD_IN)` millimetres of the beam: :func:`min_length` gives the shortest beam that still
reaches, and a shorter one raises — the hook could push in but nothing would ever catch.

When the barb is pushed aside by `undercut`, the outer layer of the beam stretches by
`1.5 * t * undercut / L^2` (see :func:`strain`). Every material takes a certain strain before it
takes permanent damage (:data:`MATERIALS`); a clip above that limit can still be made — the user
may know better — but it will probably break when it clicks, and the dialog says so.
"""
from __future__ import annotations

import math

import manifold3d as mf
import numpy as np

from .i18n import tr
from . import threads

#: the material a clip is printed in and the bending strain it takes before it takes permanent
#: damage, as a fraction of its length; None (TPU) bends as far as the clip allows
MATERIALS = {"PLA": 0.025, "PETG": 0.04, "ABS/ASA": 0.035, "Nylon": 0.06, "TPU": None}
#: the shortest beam that does not strain itself to death on the first click and the longest
#: that is still a beam and not a leaf spring
LENGTH_MIN, LENGTH_MAX = 5.0, 60.0
#: the thinnest beam that survives being handled and the thickest that still springs
THICKNESS_MIN, THICKNESS_MAX = 0.8, 5.0
#: the narrowest clip that still prints true and the widest that is still a clip
WIDTH_MIN, WIDTH_MAX = 3.0, 30.0
#: the shallowest click you can feel and the deepest any of these materials takes
UNDERCUT_MIN, UNDERCUT_MAX = 0.3, 3.0
#: an undercut deeper than this times the beam thickness bends any of these beams past what
#: they spring back from, so the dialog clamps it there
UNDERCUT_MAX_OF_T = 1.5
#: the widest room on the slot: more is not a click, it is a rattle
CLEARANCE_MAX = 1.0
#: mm of base plate: the plate is `BASE_L` long, `BASE_T` thick and as wide as the beam
BASE_L, BASE_T = 10.0, 3.0
#: mm of bed between the hook and the catch, so they break apart without cutting them
PIECE_ROOM = 5.0
#: mm of wall the slot goes through, and mm of wall under the slot line and over the slot
CATCH_WALL, CATCH_FLOOR, WALL_ABOVE = 2.0, 2.0, 2.0
#: mm of wall beside the slot on the side the barb points to, beyond the barb itself: the
#: retaining face needs something to hook on
LIP_BEYOND_BARB = 1.5
#: mm the foot strips of the catch stay clear of the slot and of where the barb swings
FOOT_CLEAR = 1.0
#: mm the retaining face has to come past the back of the wall before it hooks behind it
HOOK_ROOM = 0.5
#: mm the beam root runs into the plate, so the two fuse into one piece
ROOT_ROOM = 1.0
#: degrees of the lead-in slope the edge of the slot rides over
LEAD_IN = 30.0
#: degrees of the retaining face of a clip that can be released
RELEASE_FACE = 45.0
#: mm of room on the slot for a printer that has not proved its own clearance yet
DEFAULT_CLEARANCE = threads.DEFAULT_HOLE_CLEARANCE


def _material_limit(material):
    """What `material` takes as bending strain, None for what may bend as far as it likes."""
    if material not in MATERIALS:
        raise ValueError(tr("the material of a snap-fit clip must be one of: {names}")
                         .format(names=", ".join(MATERIALS)))
    return MATERIALS[material]


def strain(t, undercut, L) -> float:
    """The bending strain of the outer layer of a beam whose tip is pushed `undercut` aside.

    A rectangular cantilever `t` millimetres thick and `L` millimetres long, with its tip
    pushed `undercut` millimetres to the side, stretches at its surface by
    `1.5 * t * undercut / L^2`, as a fraction of its length.
    """
    return 1.5 * float(t) * float(undercut) / float(L) ** 2


def shortest_beam(t, undercut, material) -> float | None:
    """The beam length at which :func:`strain` is exactly the limit of `material`.

    A material with no limit (TPU) has no shortest beam, and neither has a barb with nothing
    to push aside; both give None.
    """
    limit = _material_limit(material)
    y = float(undercut)
    if limit is None or y <= 0.0:
        return None
    return math.sqrt(1.5 * float(t) * y / limit)


def min_length(undercut, clearance=DEFAULT_CLEARANCE) -> float:
    """The shortest beam that still reaches behind the wall of its catch.

    The lead-in slope eats `undercut / tan(LEAD_IN)` millimetres between the beam tip and the
    retaining face, and the retaining face only hooks when it comes `CATCH_WALL` plus the
    clearance plus `HOOK_ROOM` millimetres past the front of the wall.
    """
    lead = float(undercut) / math.tan(math.radians(LEAD_IN))
    return lead + CATCH_WALL + float(clearance) + HOOK_ROOM


def _rect(x0, x1, y0, y1) -> mf.CrossSection:
    """A rectangle from (`x0`, `y0`) to (`x1`, `y1`), in the plane the section is extruded from."""
    return mf.CrossSection([np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=float)])


def snap_fit_clip(material="PLA", L=15.0, t=1.6, w=8.0, undercut=1.0, releasable=False,
                  clearance=DEFAULT_CLEARANCE) -> mf.Manifold:
    """A printable snap-fit clip: a hook and its catch, side by side on the bed.

    The hook is a base plate of `w` millimetres wide, `BASE_T` millimetres thick and `BASE_L`
    millimetres long with a beam standing on the bed behind it: `L` millimetres long, `t`
    millimetres thick and `w` millimetres wide, ending in a barb that points `undercut`
    millimetres out on one side; `releasable` leans the retaining face of the barb over by
    `RELEASE_FACE` degrees, so the hook can be pulled out of the catch instead of holding. The
    catch is a wall of `CATCH_WALL` millimetres with a slot of `t + clearance` by `CATCH_FLOOR +
    w + 2 clearance` clean through it — only as wide as the beam, so the barb has to push the
    beam aside by the undercut to get through and the retaining face hooks behind the wall — on
    a foot of two strips behind it; `clearance` is the room this printer proved.

    Both pieces stand on `z = 0` with the beam centre line along `+x` near `y = 0`: the plate at
    the `x = 0` end and the catch `PIECE_ROOM` millimetres of bed in front of the beam tip. The
    hook prints lying on its side: the beam stands on its `w` edge and springs sideways, inside
    its own layers, which a beam printed standing up would not survive.
    """
    _material_limit(material)
    length = float(L)
    if length < LENGTH_MIN or length > LENGTH_MAX:
        raise ValueError(tr("the beam of a snap-fit clip must be {lo} to {hi} mm long")
                         .format(lo=f"{LENGTH_MIN:g}", hi=f"{LENGTH_MAX:g}"))
    th = float(t)
    if th < THICKNESS_MIN or th > THICKNESS_MAX:
        raise ValueError(tr("the beam of a snap-fit clip must be {lo} to {hi} mm thick")
                         .format(lo=f"{THICKNESS_MIN:g}", hi=f"{THICKNESS_MAX:g}"))
    width = float(w)
    if width < WIDTH_MIN or width > WIDTH_MAX:
        raise ValueError(tr("the width of a snap-fit clip must be {lo} to {hi} mm")
                         .format(lo=f"{WIDTH_MIN:g}", hi=f"{WIDTH_MAX:g}"))
    u = float(undercut)
    if u < UNDERCUT_MIN or u > UNDERCUT_MAX or u > UNDERCUT_MAX_OF_T * th:
        raise ValueError(tr("the undercut of a snap-fit clip must be {lo} to {hi} mm and at "
                            "most {of} times the beam thickness")
                         .format(lo=f"{UNDERCUT_MIN:g}", hi=f"{UNDERCUT_MAX:g}",
                                 of=f"{UNDERCUT_MAX_OF_T:g}"))
    c = float(clearance)
    if c < 0.0 or c > CLEARANCE_MAX:
        raise ValueError(tr("the clearance of a snap-fit clip must be {lo} to {hi} mm")
                         .format(lo="0", hi=f"{CLEARANCE_MAX:g}"))
    need = min_length(u, c)
    if length < need:
        raise ValueError(tr("the beam of a snap-fit clip with a {u} mm undercut must be at "
                            "least {mm} mm long, or the barb never reaches behind the wall of "
                            "the catch")
                         .format(u=f"{u:g}", mm=f"{need:.1f}"))

    # the beam with its barb as one contour seen from above: the axis along +x, the thickness
    # across it, and the wedge of the barb at the tip of the +y side; the slope of LEAD_IN
    # degrees runs down to the beam tip and the retaining face stands just behind it
    root, tip = BASE_L - ROOT_ROOM, BASE_L + length
    lead = u / math.tan(math.radians(LEAD_IN))
    keep = (tip - lead, 0.5 * th) if not releasable else (tip - lead - u, 0.5 * th)
    profile = np.array([(root, -0.5 * th), (tip, -0.5 * th), (tip, 0.5 * th),
                        (tip - lead, 0.5 * th + u), keep, (root, 0.5 * th)], dtype=float)
    # the beam stands on the bed and springs sideways, inside its own layers, as wide in z as
    # the slot stands high in the wall
    hook = (mf.Manifold.extrude(_rect(0.0, BASE_L, -0.5 * width, 0.5 * width), BASE_T)
            + mf.Manifold.extrude(mf.CrossSection([profile]), width))

    # the catch: a wall with the slot clean through it, open on both faces and open to the bed
    # under the slot itself so the beam can enter along the bed; the wall stands beside the
    # slot where the retaining face of the barb can hook behind it, and two foot strips lie on
    # the bed behind the wall, clear of the slot and of where the barb swings
    bx = tip + PIECE_ROOM                      # the front face of the wall
    slot_c = 0.5 * (th + c)                     # the slot, half as wide as the beam plus room
    slot_top = CATCH_FLOOR + width + 2.0 * c
    wall_y0 = -slot_c - WALL_ABOVE
    wall_y1 = slot_c + max(u + LIP_BEYOND_BARB, WALL_ABOVE)
    foot_in = 0.5 * th + max(u, 0.5 * c) + FOOT_CLEAR  # clear of the slot and of the barb
    foot_x = [bx + CATCH_WALL, bx + CATCH_WALL + BASE_L]
    catch = mf.Manifold.cube([CATCH_WALL, wall_y1 - wall_y0, WALL_ABOVE + slot_top]) \
        .translate((bx, wall_y0, 0.0))
    catch -= mf.Manifold.cube([CATCH_WALL + 2.0, 2.0 * slot_c, slot_top + 0.5]) \
        .translate((bx - 1.0, -slot_c, -0.5))
    foot = (mf.Manifold.cube([foot_x[1] - foot_x[0], wall_y1 - foot_in, BASE_T])
            .translate((foot_x[0], foot_in, 0.0))
            + mf.Manifold.cube([foot_x[1] - foot_x[0], -slot_c - FOOT_CLEAR - wall_y0, BASE_T])
            .translate((foot_x[0], wall_y0, 0.0)))

    clip = mf.Manifold.batch_boolean([hook, catch + foot], mf.OpType.Add)
    # the clip stands on the bed and the beam centre line runs along x in the middle of the bed
    bb = np.asarray(clip.bounding_box(), dtype=float)
    return clip.translate((-0.5 * (bb[0] + bb[3]), -0.5 * (bb[1] + bb[4]), -bb[2]))

"""More than one build plate: a big project spread over several beds of the same printer.

No Qt in this file. A model that does not fit on one bed is spread over plates 1, 2, 3, ...
which lie side by side along +X, plate 1 being the real bed of the printer and every plate
after it an identical bed next to it. `volura.ops.arrange_on_plates` lays the parts out over
those beds and `volura.window_print` slices them one plate after the other, because a printer
prints one plate at a time.

**A group's plate is where it stands**: the bed that holds the centre of its bounding box.
That is what the whole file follows from, and it is the one thing that cannot go out of sync -
move a group with Move, Scale, Undo or a script and its plate is right in the same frame,
because there is nothing to keep in step. `g.meta["plate"]` is only a *cache* of that: written
by `sync_meta` so a saved file says what the user saw, read back by `stored_plate` for geometry
that has no bounding box to stand on. A plate number in the file that disagrees with where the
part stands is wrong, and where the part stands wins.

The beds of `Preferences > Print` are the size of every plate, so `bed` (the ``(w, d, h)`` of
those settings) travels with the groups everywhere here. The free room between two beds is
`PLATE_GAP` of their width: wide enough to see that two beds are two beds.
"""
from __future__ import annotations

import math

import numpy as np

from .i18n import tr

#: the room between two beds, as a part of the width of a bed
PLATE_GAP = 0.2
#: the plate choice that says "every plate at once" instead of one of them
ALL = 0
#: the first plate: the real bed of the printer, and what `stored_plate` answers for a group
#: that says nothing about its plate
FIRST = DEFAULT = 1


def plate_number(value, default: int = DEFAULT) -> int:
    """`value` as a plate number: 1 for the first plate, never lower, `default` for junk.

    A number written by hand in a file can be "3", 3.0, "3 plates" or nothing at all; only a
    whole number of one or higher says anything, and everything else is the first plate."""
    try:
        got = int(float(value))
    except (TypeError, ValueError, OverflowError):
        return int(default)
    if not math.isfinite(float(default)) or got < 1:
        return int(default)
    return got


def gap(bed) -> float:
    """The free room between two beds, in millimetres."""
    return PLATE_GAP * _width(bed)


def pitch(bed) -> float:
    """From the corner of a bed to the corner of the next one: a bed wide plus the gap."""
    return _width(bed) + gap(bed)


def _width(bed) -> float:
    """The width of a bed (X); a bed without a width lays its plates on top of each other."""
    try:
        w = float(bed[0])
    except (TypeError, ValueError, IndexError, KeyError):
        return 0.0
    return w if w > 0.0 and math.isfinite(w) else 0.0


def origin(plate, bed):
    """The corner of the bed of `plate`, in model millimetres: plate 1 is (0, 0, 0)."""
    step = pitch(bed)
    return np.array([max(0, plate_number(plate) - 1) * step, 0.0, 0.0], dtype=float)


def plate_of_x(x, bed) -> int:
    """The plate whose bed holds this X.

    Everything left of the second bed - also what stands left of the real bed, which is where
    loose geometry of an imported model usually is - is the first plate."""
    step = pitch(bed)
    if step <= 0.0:
        return FIRST
    try:
        value = float(x)
    except (TypeError, ValueError):
        return FIRST
    if not math.isfinite(value) or value < step:
        return FIRST
    return int(value / step) + 1


def stored_plate(g) -> int:
    """The plate `g.meta` says the group is on; a group that says nothing is on the first."""
    meta = getattr(g, "meta", None)
    if not isinstance(meta, dict) or "plate" not in meta:
        return DEFAULT
    return plate_number(meta.get("plate"))


def plate_of_group(g, bed) -> int:
    """The plate whose bed stands under the centre of `g`.

    Geometry without a bounding box says nothing about where it is, so such a group keeps
    whatever its meta has to say; a group that says nothing stands on the first plate."""
    bb = g.bbox() if hasattr(g, "bbox") else None
    if bb is None:
        return stored_plate(g)
    return plate_of_x(0.5 * (float(bb[0][0]) + float(bb[1][0])), bed)


def on_plate(g, plate, bed) -> bool:
    """Whether `g` is part of `plate`; `ALL` is the plate that holds everything."""
    if plate is None or plate == ALL:
        return True
    return plate_of_group(g, bed) == plate_number(plate)


def plate_choice(value) -> int:
    """The plate a chooser hands back: `ALL` for "all plates", otherwise a plate number.

    "All plates" is 0 and `plate_number` would make the first plate of it, which is not what the
    chooser means; nothing at all says every plate too, because that is what a model of one plate
    shows anyway."""
    if value is None:
        return ALL
    try:
        if int(float(value)) == ALL:
            return ALL
    except (TypeError, ValueError, OverflowError):
        return FIRST
    return plate_number(value)


def sync_meta(doc, bed) -> int:
    """Write the plate every group stands on into its meta; how many groups said otherwise.

    Undo keeps the meta of a state with it, so this is what makes a saved file say the same as
    the screen - and a part that moves back by undo moves its number back with it."""
    changed = 0
    for g in doc.groups:
        if not isinstance(g.meta, dict):
            continue
        want = plate_of_group(g, bed)
        if stored_plate(g) == want and ("plate" in g.meta) == (want != FIRST):
            continue
        if want == FIRST:
            g.meta.pop("plate", None)
        else:
            g.meta["plate"] = want
        changed += 1
    return changed


def plates_used(doc, bed, visible_only=True) -> list:
    """The plates that hold something, smallest first; the first plate is always one of them.

    Loose geometry of the model itself has no group to carry a plate, so it stands on the first
    plate, which is why that plate is always in the list."""
    groups = doc.visible_groups() if visible_only else doc.groups
    used = {plate_of_group(g, bed) for g in groups}
    used.add(FIRST)
    return sorted(used)


def plate_count(doc, bed, visible_only=True) -> int:
    """How many plates the model needs: the highest plate that holds something."""
    return plates_used(doc, bed, visible_only)[-1]


def groups_on_plate(doc, plate, bed, visible_only=True) -> list:
    """The groups that stand on the bed of `plate`."""
    groups = doc.visible_groups() if visible_only else doc.groups
    return [g for g in groups if on_plate(g, plate, bed)]


def label(plate, total) -> str:
    """What one plate of a model is called in a chooser: "Plate 2 of 3"."""
    return tr("Plate {n} of {total}").format(n=plate_number(plate), total=max(1, int(total)))


def file_name(stem: str, plate) -> str:
    """The name of the G-code of one plate: `name_plate2` (no extension)."""
    return f"{stem}_plate{plate_number(plate)}"

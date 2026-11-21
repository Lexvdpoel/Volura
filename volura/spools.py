"""How much filament is left on every spool, so a print is seen to run out before it does.

No Qt in this file: the dialog of Window > Spools… (volura.spools_dialog) and the print
preview (volura.print_preview) ask this module what a spool holds and what is still on it. A
`Spool` is one roll on the shelf: what it is called, what it is made of, which colour it has,
how much a new one holds and how much is left of it.

Every gram in here is net: the plastic of the spool itself is never counted, except in
`tare_g`, which says how heavy an empty one is. That is what makes the last number easy to
keep honest: put the spool on a scale, type what the scale says and Volura subtracts the
empty spool (see `set_left`).

The print preview points every filament of a print at a spool and says whether that spool has
enough for it (`enough`, with a margin, because a print that uses the last of a spool stops
halfway). When a print really starts on a printer through Volura, or the user says so with
`Mark as printed`, the grams come off those spools (`take`).

The settings keep the spools under `spools`, one line of JSON each like the printers of
volura.printer_link, and which spool goes with which filament slot of the print preview under
`spool_for_slot` (volura.app.Settings). A filament slot that remembers a spool which was
removed in the meantime simply has no spool again (`find` gives None for it).
"""
from __future__ import annotations

import json
import math
import re
import uuid
from dataclasses import dataclass, replace

from .i18n import tr

#: what a full spool holds when nothing says otherwise, in net grams
FULL_G = 1000.0
#: how heavy an empty spool is, what a scale reading is corrected with, in grams
TARE_G = 250.0
#: the colour of a spool that does not say what it looks like
DEFAULT_COLOUR = "#8a8f98"
#: the material of a spool that does not say what it is made of
OTHER = "Other"
#: how much more than a print needs to dare to start it anyway (see `enough`)
MARGIN = 0.05
#: the bounds of what a spool can hold, weigh empty and cost, so junk in the settings cannot
#: make a spool of negative weight
FULL_MIN, FULL_MAX = 1.0, 100000.0
TARE_MIN, TARE_MAX = 0.0, 5000.0
PRICE_MIN, PRICE_MAX = 0.0, 100000.0

#: the fields of a spool, in the order they are written
FIELDS = ("id", "name", "material", "colour", "full_g", "left_g", "price", "tare_g")

_COLOUR = re.compile(r"#([0-9a-fA-F]{3}|[0-9a-fA-F]{6})$")


def new_id(taken=()) -> str:
    """A short id for a spool, one that is not in `taken` yet."""
    taken = set(taken)
    made = uuid.uuid4().hex[:8]
    while made in taken:
        made = uuid.uuid4().hex[:8]
    return made


def _weight(value, default) -> float:
    """`value` as a number of grams that can be counted: junk and nonsense become `default`."""
    try:
        got = float(value)
    except (TypeError, ValueError):
        got = float(default)
    return got if math.isfinite(got) else float(default)


def _number(value, default, lo, hi) -> float:
    """`value` as a weight or a price that is kept, within the bounds a spool can have; junk
    becomes `default` and nonsense like the square root of -1 does too."""
    return max(lo, min(hi, _weight(value, default)))


def _text(value, default: str) -> str:
    """`value` as the text of a field; `default` when there is nothing to keep."""
    text = "" if value is None else str(value).strip()
    return text or default


def _colour(value) -> str:
    """`value` as "#rrggbb" in lower case; a short "#rgb" is taken with every digit twice."""
    text = str(value or "").strip()
    if not _COLOUR.fullmatch(text):
        return DEFAULT_COLOUR
    text = text.lower()
    return text if len(text) == 7 else "#" + "".join(digit * 2 for digit in text[1:])


def grams_text(value) -> str:
    """Grams as Volura says them: at most one decimal and no trailing zeros."""
    return f"{round(_weight(value, 0.0), 1):g}"


@dataclass
class Spool:
    """One roll of filament on the shelf.

    `full_g` is what a new spool holds and `left_g` what is still on it, both net of the
    plastic; `tare_g` is what the empty spool weighs, so a scale reading becomes what is left
    with one subtraction (see `set_left`). `price` is what a full spool cost."""

    id: str = ""
    name: str = ""
    material: str = OTHER
    colour: str = DEFAULT_COLOUR
    full_g: float = FULL_G
    left_g: float | None = None          # None: a spool that was never used, still full
    price: float = 0.0
    tare_g: float = TARE_G

    def __post_init__(self):
        self.id = _text(self.id, "") or new_id()
        self.name = _text(self.name, "")
        self.material = _text(self.material, OTHER)
        self.colour = _colour(self.colour)
        self.full_g = _number(self.full_g, FULL_G, FULL_MIN, FULL_MAX)
        left = self.full_g if self.left_g is None else self.left_g
        self.left_g = _number(left, self.full_g, 0.0, self.full_g)
        self.price = _number(self.price, 0.0, PRICE_MIN, PRICE_MAX)
        self.tare_g = _number(self.tare_g, TARE_G, TARE_MIN, TARE_MAX)


def to_dict(spool) -> dict:
    """A spool as the settings keep it: plain values that survive JSON."""
    return {"id": spool.id, "name": spool.name, "material": spool.material,
            "colour": spool.colour, "full_g": round(spool.full_g, 3),
            "left_g": round(spool.left_g, 3), "price": round(spool.price, 2),
            "tare_g": round(spool.tare_g, 1)}


def from_dict(data) -> Spool:
    """The spool that stands in a dict of the settings.

    A field that is missing or unusable takes its default, so a spool written by an older
    Volura still loads, and a field the spool does not know is dropped. What is no dict at
    all becomes a spool with nothing filled in, which is to say: a new empty spool."""
    data = data if isinstance(data, dict) else {}
    return Spool(**{key: data[key] for key in FIELDS if key in data})


def record(value) -> Spool:
    """A spool, out of a Spool or out of the dict the settings keep it in."""
    return value if isinstance(value, Spool) else from_dict(value)


def records(values) -> list:
    """Every spool of a list as a Spool; what is no spool at all is left out."""
    out = []
    for value in values or []:
        if isinstance(value, (Spool, dict)):
            out.append(record(value))
    return out


def to_json(value) -> str:
    """One spool as a line of JSON, for the list of spools in the settings."""
    return json.dumps(to_dict(record(value)))


def from_json(line) -> dict | None:
    """The spool that stands in one line of the settings, as a dict.

    None when the line is no spool at all: what is no JSON object, and an object without an
    id, because a spool of the settings always has one (see `to_dict`)."""
    try:
        data = json.loads(line)
    except (TypeError, ValueError):
        return None
    if isinstance(data, dict):
        return to_dict(from_dict(data)) if _text(data.get("id"), "") else None
    return None


def to_lines(values) -> list:
    """The spools as the lines of JSON the settings keep them in.

    Dicts and Spools both work; what is neither - a stray text or number in the list - is
    left out, so a junk value cannot write a shelf of empty spools."""
    return [to_json(value) for value in (values or []) if isinstance(value, (Spool, dict))]


def from_lines(lines) -> list:
    """The spools of the settings as a list of dicts; a line that is no spool is skipped.

    A spool with one field that is unusable is not skipped: that field takes its default and
    the rest of the spool stays (see `from_dict`)."""
    if isinstance(lines, str):
        lines = [lines]
    out = []
    for line in lines or []:
        if isinstance(line, dict):
            line = json.dumps(line)
        data = from_json(line)
        if data is not None:
            out.append(data)
    return out


def take(spool, grams) -> Spool:
    """A copy of `spool` with `grams` gone off it; nothing ever goes below zero on it.

    What was not there to take stays off the books: a print of negative grams takes nothing."""
    used = max(0.0, _weight(grams, 0.0))
    return replace(spool, left_g=max(0.0, spool.left_g - used))


def set_left(spool, scale_g) -> Spool:
    """A copy of `spool` with what a scale says is left on it.

    `scale_g` is the reading of the whole spool with its filament on it; the weight of the
    empty spool (`tare_g`) is subtracted, so the number that is kept stays net. A reading
    that is no reading at all leaves the spool as it was: nothing weighed, nothing changed."""
    got = _weight(scale_g, spool.tare_g + spool.left_g)
    return replace(spool, left_g=max(0.0, got - spool.tare_g))


def enough(spool, grams, margin=MARGIN) -> bool:
    """True when `spool` holds `grams`, with `margin` part more on top of it.

    The margin is what says the spool may still be trusted with the print: one that holds
    exactly enough prints to the very last gram and stops halfway."""
    return spool.left_g >= _weight(grams, 0.0) * (1.0 + _weight(margin, MARGIN))


def left_text(spool) -> str:
    """What is still on a spool, as the dialog and the print preview say it."""
    return tr("{left} g of {full} g left ({pct} %)").format(
        left=grams_text(spool.left_g), full=grams_text(spool.full_g),
        pct=int(round(100.0 * spool.left_g / spool.full_g)))


def matches(spool, name) -> bool:
    """True when the material of a spool comes back in the name of a filament of a print.

    The preview uses this to put the spools that could be this filament on top of its list."""
    material = str(spool.material or "").strip().casefold()
    return bool(material) and material in str(name or "").casefold()


def find(spools, spool_id):
    """The spool of that id in `spools` (dicts or Spools); None when there is no such spool.

    A spool can be off the list while a filament slot of the print preview still remembers
    its id; a caller that gets None back treats that slot as having no spool."""
    if not spool_id:
        return None
    for value in spools or []:
        spool = record(value)
        if spool.id == str(spool_id):
            return spool
    return None


def slot_ids(value) -> list:
    """Which spool every filament slot of the print preview comes off, in the order of the
    slots; an empty string stands for a slot that has no spool.

    The settings keep them as a list of ids, like the files under Recent (volura.app.Settings),
    and a list of one is stored as one plain text. What is no id at all is no slot."""
    if isinstance(value, str):
        value = [value] if value.strip() else []
    return ["" if item is None else str(item).strip() for item in value or []]

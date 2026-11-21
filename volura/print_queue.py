"""The print queue: prints that wait for a free printer, so a line of prints runs itself.

No Qt in this file: the print preview (volura.print_preview) and Window > Print History…
(volura.print_history_dialog) put a print on the queue, and Window > Print Queue…
(volura.print_queue_dialog) shows the line and sends the prints that can go. A `QueueItem` is
one print that waits: what it is called, where its G-code stands, what the slice worked out
and which printer it is for - "" for any printer, or the name of one of them.

The safety rule of the queue is one line: a printer that just finished still has its last print
on the bed, and starting the next one would crash the nozzle into it. So Volura starts nothing
on a printer unless the user said, with *Bed is clear*, that the bed of that printer is clear
since its last print. That word is worth nothing from the moment the printer says it is printing
or paused (see `keep_clear`), and is taken back the moment Volura starts something on it (see
`store` of volura.print_queue_dialog). A printer that says error or answers nothing is never
used, and nothing at all is sent while the queue stands paused - which is how it starts every
time Volura runs, so a queue that was left behind overnight sends nothing on its own.

`next_job` is the whole of the choosing: the first print of the queue, in the order it waits,
that may go to a printer that is idle, whose bed was said clear and that is the printer the
print asks for (or any printer when it does not care). It is a plain function of what it is
handed, so the rule above is testable without a printer anywhere.

Of every print on the queue the G-code is copied into the folder of the print history
(volura.print_history), because the file a preview works on does not outlive the preview; a
print that leaves the queue loses that copy again, and a file the user saved is never touched.
The settings keep the queue under ``print_queue``, in the order the prints wait
(volura.app.Settings)."""
from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass
from datetime import datetime

from . import print_history, printer_link
from .i18n import tr, tr_noop, trn

#: a print that does not care which printer takes it waits for the first one that is free
ANY = ""
#: what such a print is called in the window
ANY_LABEL = tr_noop("Any printer")
#: the fields of a queued print, in the order they are written
FIELDS = ("id", "name", "gcode", "grams", "minutes", "printer", "added")
#: what a print that says nothing about itself is worth
NO_TIME, NO_GRAMS = 0.0, 0.0
#: the one state a printer has to be in before a print may start on it
FREE = "idle"
#: a printer that says one of these has something on its bed, so its word of before is gone
BUSY = ("printing", "paused")
#: a printer in one of these is never used: it says something is wrong or answers nothing
BROKEN = ("error", "offline")
#: how long a model name or a printer name may be
MAX_TEXT = print_history.MAX_TEXT
#: what a run of spaces, tabs or newlines between two words becomes: the list shows one line
_SPACES = re.compile(r"\s+")


def _text(value, default: str = "") -> str:
    """`value` as the text of a field; `default` when there is nothing to keep."""
    text = "" if value is None else str(value).strip()
    return text or default


def _line(value, default: str = "", limit: int = MAX_TEXT) -> str:
    """`value` as a text that stands on one line of the list, never longer than `limit`.

    A name of a model comes from a file on disk and a printer name from a form, so both can
    hold a tab or a newline; the queue shows one line per print."""
    text = _SPACES.sub(" ", _text(value)).strip()[:max(0, int(limit))]
    return text.strip() or default


def _number(value, default: float) -> float:
    """`value` as grams or minutes that can be counted; junk and nonsense become `default`.

    A print never used a negative amount of filament, so what is below zero is worth nothing."""
    try:
        got = float(value)
    except (TypeError, ValueError):
        got = float(default)
    return got if math.isfinite(got) and got >= 0.0 else float(default)


def _when(value) -> str:
    """The moment a print was put on the queue, as the settings keep it: local time, ISO
    spelled. What is no moment at all is the moment of writing it down."""
    text = str(value or "").strip()
    try:
        datetime.fromisoformat(text)
    except (TypeError, ValueError):
        return datetime.now().isoformat(timespec="seconds")
    return text


def _state(value) -> str:
    """What a printer said it is doing, one of `printer_link.STATES`; what it said wrong is
    said as `offline`, which is the one thing a queue may never start a print on."""
    text = str(value or "").strip().lower()
    return text if text in printer_link.STATES else "offline"


@dataclass
class QueueItem:
    """One print that waits for a printer.

    `gcode` is the G-code it will go to the printer with: a copy in the folder of the print
    history, or a file the user saved - and "" when there is nothing to send. `printer` is the
    name of the connection it is meant for, or "" for any printer that is free. `added` is the
    local moment it was put on the queue, ISO spelled; the queue is worked off in its own
    order, so that moment is what the window shows and not what it sorts by."""

    id: str = ""
    name: str = print_history.UNTITLED
    gcode: str = ""
    grams: float = NO_GRAMS
    minutes: float = NO_TIME
    printer: str = ANY
    added: str = ""

    def __post_init__(self):
        self.id = _text(self.id, "") or print_history.new_id()
        self.name = _line(self.name, tr(print_history.UNTITLED))
        self.gcode = _text(self.gcode)
        self.grams = _number(self.grams, NO_GRAMS)
        self.minutes = _number(self.minutes, NO_TIME)
        self.printer = _line(self.printer, ANY)
        self.added = _when(self.added)


def to_dict(item) -> dict:
    """A queued print as the settings keep it: plain values that survive the store."""
    item = record(item)
    return {"id": item.id, "name": item.name, "gcode": item.gcode,
            "grams": round(item.grams, 3), "minutes": round(item.minutes, 3),
            "printer": item.printer, "added": item.added}


def from_dict(data) -> QueueItem:
    """The queued print that stands in a dict of the settings.

    A field that is missing or unusable takes its default, so an item written by an older
    Volura still loads, and a field the item does not know is dropped. What is no dict at all
    becomes a print with nothing on it, which is what there is to say about junk."""
    data = data if isinstance(data, dict) else {}
    return QueueItem(**{key: data[key] for key in FIELDS if key in data})


def record(value) -> QueueItem:
    """A queued print, out of a QueueItem or out of the dict the settings keep it in."""
    return value if isinstance(value, QueueItem) else from_dict(value)


def records(values) -> list:
    """Every queued print of a list as a QueueItem; what is no print at all is left out."""
    try:
        items = [] if isinstance(values, (str, bytes)) else list(values)
    except TypeError:
        return []
    return [record(value) for value in items if isinstance(value, (QueueItem, dict))]


def from_values(values) -> list:
    """The queue that stands in the settings, as the list of dicts the settings keep it in.

    A print with one field that is unusable is not left out: that field takes its default and
    the rest of the print stays (see `from_dict`). This is what volura.app.Settings keeps under
    `print_queue`, in the order the prints wait."""
    return [to_dict(value) for value in records(values)]


# ------------------------------------------------------------------ the order of the queue


def add(items, item) -> list:
    """`item` at the back of the queue, so the print that has waited longest goes first.

    What is no print at all is left out, and the copy of the print itself is already kept by
    `store`; this only lays the queue out again."""
    out = from_values(items)
    if isinstance(item, (QueueItem, dict)):
        out.append(to_dict(item))
    return out


def move(items, item_id, up) -> list:
    """The queue with the print of that id one place `up` (or down, for False).

    A print at the front does not go further forward and one at the back no further back, and a
    print that is not on the queue leaves the queue as it was."""
    out = from_values(items)
    for at, value in enumerate(out):
        if value["id"] != str(item_id):
            continue
        to = at - 1 if up else at + 1
        if 0 <= to < len(out):
            out[at], out[to] = out[to], out[at]
        return out
    return out


def remove(items, item_id) -> list:
    """The queue without the print of that id; the G-code Volura kept of it goes with it."""
    out = []
    for value in records(items):
        if value.id == str(item_id):
            print_history.forget(value.gcode)
        else:
            out.append(to_dict(value))
    return out


def cleared(items) -> list:
    """An empty queue, with every G-code of Volura itself taken away; what the user saved
    itself stays."""
    for value in records(items):
        print_history.forget(value.gcode)
    return []


def store(settings, item, source="") -> QueueItem:
    """Write `item` on the queue of `settings` and keep its G-code.

    `source` is the G-code the print will go to the printer with: a preview works on a
    temporary file, so a copy of it is kept with the queue and taken away when the print leaves
    it. Returns the item, with its `gcode` filled in - which is empty when there was nothing to
    copy, and then the print has nothing to send."""
    item = record(item)
    if source and not item.gcode:
        item.gcode = print_history.keep_copy(source, item.id)
    if settings is not None and hasattr(settings, "print_queue"):
        settings.print_queue = add(settings.print_queue, item)
        settings.save()
    return item


def gcode_of(item):
    """The G-code of a queued print that is still there, or None when it is gone (or never was)
    - and a print without a G-code has nothing to send anywhere."""
    path = record(item).gcode
    return path if path and os.path.isfile(path) else None


# ------------------------------------------------------------------ what a printer can do


def printer_state(conn, state="offline", bed_clear=False) -> dict:
    """How the queue looks at one printer right now: which printer it is, what it says it is
    doing and whether the user said that its bed is clear since its last print.

    That last part is not something a printer says of itself; it is the word of the user, given
    with *Bed is clear* in Window > Print Queue… and nothing else."""
    return {"conn": conn, "state": _state(state), "bed_clear": bool(bed_clear)}


def keep_clear(state, bed_clear) -> bool:
    """Whether the word that the bed is clear still stands after what the printer just said.

    A printer that says it prints or stands paused has something on its bed - whether Volura
    started it or somebody else did - so from that moment the queue may not start a second
    print on it. Every other state says nothing about what lies on the bed, so the word of the
    user stays."""
    return bool(bed_clear) and _state(state) not in BUSY


def free(state, taken=()) -> bool:
    """True when this printer could start a print: idle, its bed said clear, and not already
    given a print in this round.

    A printer that says error or answers nothing is never used, whatever its bed looks like:
    there is nobody to ask."""
    if not isinstance(state, dict):
        return False
    conn = state.get("conn")
    if conn is None or not getattr(conn, "name", None):
        return False
    return (state.get("state") == FREE and bool(state.get("bed_clear"))
            and str(conn.name) not in set(taken or ()))


def fits(item, state) -> bool:
    """True when the print will go to this printer: one that asks for a printer of its own only
    goes to that one, and one that does not care goes to any that is free."""
    conn = state.get("conn") if isinstance(state, dict) else None
    name = str(getattr(conn, "name", ""))
    if not name:
        return False
    want = record(item).printer
    return want == ANY or want == name


def next_job(items, printers_state, taken=()):
    """The first print of the queue that can start, with the printer it goes to; None when
    nothing can start.

    Prints are looked at in the order they wait and printers in the order they are given, so
    the print that has waited longest takes the first printer that suits it. `taken` names the
    printers that already got a print in this round, so one printer never gets two prints at
    once; a caller that hands out one print at a time passes nothing.

    What a print wants to go to - the G-code of a print that is gone - is not looked at here,
    because this knows only what it is handed; the window leaves such prints out (see
    `gcode_of`)."""
    for item in records(items):
        for state in printers_state or ():
            if free(state, taken) and fits(item, state):
                return item, state["conn"]
    return None


# ------------------------------------------------------------------ how the window says it


def printer_text(item) -> str:
    """The printer a queued print is meant for, as the window says it: "Any printer" for one
    that does not care."""
    item = record(item)
    return tr(ANY_LABEL) if not item.printer else item.printer


def item_text(item) -> str:
    """One line about a print of the queue: what it is, where it goes and what it takes."""
    item = record(item)
    line = f"{item.name} {tr('on {printer}').format(printer=printer_text(item))}"
    if item.grams > 0.0:
        line += " - " + tr("{grams} in {time}").format(grams=print_history.grams_text(item.grams),
                                                       time=print_history.time_text(item.minutes))
    return line


def waiting_text(n) -> str:
    """How many prints the queue still has, as the window and the preview say it."""
    try:
        count = max(0, int(n))
    except (TypeError, ValueError):
        count = 0
    return trn("{n} print waiting", "{n} prints waiting", count).format(n=count)


def added_text(n) -> str:
    """What is said when a print is put on the queue: it waits, with how many before and after
    it."""
    try:
        count = max(0, int(n))
    except (TypeError, ValueError):
        count = 0
    return tr("Added to the queue ({n} waiting)").format(n=count)

"""What you printed, so that the same print can go on the bed again with one click.

No Qt in this file: the print preview (volura.print_preview) writes an entry when it closes,
and the dialog of Window > Print History… (volura.print_history_dialog) shows the list and
hands a stored G-code to a printer again. An `Entry` is one print: when it was made, what
the model it came from was called, where its G-code stands, which printer took it, how much
filament and time it cost, what its filaments were and what became of it.

What is remembered is the strongest thing that happened to a preview: a G-code that was
saved, then sent and then started is recorded as "started", because that is the print that
really ran. Nothing happened means nothing is recorded - a preview that was only looked at
told Volura nothing.

Of every print that went to a printer the G-code is kept too: a preview works on a temporary
file that is taken away when it closes, so without a copy there would be nothing to print
again. Those copies live under ``history/gcode`` in the folder of the program
(volura.errors.log_dir) and go when their entry leaves the history. A file the user saved is
never copied and never taken away - Volura did not make it and may not remove it.

The settings keep the history under ``print_history``, newest first and at most `LIMIT`
entries (volura.app.Settings), so a print of three years ago is still to be found while the
list does not grow without end."""
from __future__ import annotations

import math
import os
import re
import shutil
import uuid
from dataclasses import dataclass, field
from datetime import datetime

from . import printer_link, spools
from .i18n import tr, tr_noop, trn

#: how many prints the history keeps; what goes past that falls off the bottom
LIMIT = 200
#: what the G-code of a print is called in the folder of the program
GCODE_SUFFIX = ".gcode"
#: the subfolders of the folder of the program that keep those G-codes
GCODE_FOLDER = ("history", "gcode")
#: the name of a print that was made from a model that was never saved
UNTITLED = tr_noop("Untitled")

#: what can have happened to a print, from the weakest to the strongest
SAVED, SENT, MARKED, STARTED = "saved", "sent", "marked", "started"
HOWS = (SAVED, SENT, MARKED, STARTED)
#: how far each of them reaches: a preview records the strongest thing that happened to it
STRONGEST = {SAVED: 0, SENT: 1, MARKED: 2, STARTED: 3}
#: what the list calls each of them
HOW_LABELS = {SAVED: tr_noop("Saved"), SENT: tr_noop("Sent"),
              MARKED: tr_noop("Marked as printed"), STARTED: tr_noop("Started")}

#: the fields of an entry, in the order they are written
FIELDS = ("id", "when", "name", "gcode", "printer", "grams", "minutes", "filaments", "how")
#: what a print that says nothing about itself is worth
NO_TIME, NO_GRAMS = 0.0, 0.0
#: an id of a copy is made of what a file name can hold, so junk cannot point outside the folder
_SAFE = re.compile(r"[^A-Za-z0-9_.-]")
#: what a run of spaces, tabs or newlines between two words becomes: the list shows one line
_SPACES = re.compile(r"\s+")
#: how long a model name or a printer name may be; a file name on disk is shorter than this
MAX_TEXT = 300


def new_id(taken=()) -> str:
    """A short id for a print, one that is not in `taken` yet."""
    taken = set(taken)
    made = uuid.uuid4().hex[:8]
    while made in taken:
        made = uuid.uuid4().hex[:8]
    return made


def _text(value, default: str = "") -> str:
    """`value` as the text of a field; `default` when there is nothing to keep."""
    text = "" if value is None else str(value).strip()
    return text or default


def _line(value, default: str = "", limit: int = MAX_TEXT) -> str:
    """`value` as a text that stands on one line of the list, never longer than `limit`.

    A name of a model comes from a file on disk and a printer name from a form, so both can
    hold a tab or a newline; the list shows one line per print, so every run of white space
    between two words becomes a single space there."""
    text = _SPACES.sub(" ", _text(value)).strip()[:max(0, int(limit))]
    return text.strip() or default


def _number(value, default: float) -> float:
    """`value` as grams or minutes that can be counted; junk and nonsense become `default`.

    A print never used a negative amount of filament, so what is below zero is worth
    nothing."""
    try:
        got = float(value)
    except (TypeError, ValueError):
        got = float(default)
    return got if math.isfinite(got) and got >= 0.0 else float(default)


def _when(value) -> str:
    """The moment a print was made, as the settings keep it: local time, ISO spelled.

    What is no moment at all is the moment of writing it down, so an entry always says when
    it was made."""
    text = str(value or "").strip()
    try:
        datetime.fromisoformat(text)
    except (TypeError, ValueError):
        return datetime.now().isoformat(timespec="seconds")
    return text


def _filaments(value) -> list:
    """Every filament of a print as ``[name, colour]``, in the order the print used them.

    What is no name at all is no filament: a stray number in the list is left out, so junk
    cannot write a print with a filament that has no name."""
    out = []
    for item in value or []:
        try:
            pair = [item] if isinstance(item, str) else list(item)
        except TypeError:
            continue
        name = _line(pair[0] if pair else "")
        if not name:
            continue
        out.append([name, _text(pair[1] if len(pair) > 1 else "")])
    return out


def _how(value) -> str:
    """What became of a print, one of `HOWS`; a print that does not say was saved."""
    text = str(value or "").strip().lower()
    return text if text in HOWS else SAVED


@dataclass
class Entry:
    """One print of the history.

    `when` is the local moment it was made, ISO spelled, and `name` the model it was sliced
    from ("Untitled" for one that was never saved). `gcode` is the G-code it ended up with:
    the file the user saved, or a copy Volura kept when the print only went to a printer -
    and "" when there is nothing to print again. `printer` is the name of the connection it
    went to, `grams` and `minutes` what the slice worked out and `filaments` its filaments,
    every one of them ``[name, colour]``."""

    id: str = ""
    when: str = ""
    name: str = UNTITLED
    gcode: str = ""
    printer: str = ""
    grams: float = NO_GRAMS
    minutes: float = NO_TIME
    filaments: list = field(default_factory=list)
    how: str = SAVED

    def __post_init__(self):
        self.id = _text(self.id, "") or new_id()
        self.when = _when(self.when)
        self.name = _line(self.name, tr(UNTITLED))
        self.gcode = _text(self.gcode)
        self.printer = _line(self.printer)
        self.grams = _number(self.grams, NO_GRAMS)
        self.minutes = _number(self.minutes, NO_TIME)
        self.filaments = _filaments(self.filaments)
        self.how = _how(self.how)


def to_dict(entry) -> dict:
    """An entry as the settings keep it: plain values that survive the store."""
    entry = record(entry)
    return {"id": entry.id, "when": entry.when, "name": entry.name, "gcode": entry.gcode,
            "printer": entry.printer, "grams": round(entry.grams, 3),
            "minutes": round(entry.minutes, 3),
            "filaments": [list(pair) for pair in entry.filaments], "how": entry.how}


def from_dict(data) -> Entry:
    """The print that stands in a dict of the settings.

    A field that is missing or unusable takes its default, so an entry written by an older
    Volura still loads, and a field the entry does not know is dropped. What is no dict at
    all becomes a print of nothing at all, which is what there is to say about junk."""
    data = data if isinstance(data, dict) else {}
    return Entry(**{key: data[key] for key in FIELDS if key in data})


def record(value) -> Entry:
    """An entry, out of an Entry or out of the dict the settings keep it in."""
    return value if isinstance(value, Entry) else from_dict(value)


def records(values) -> list:
    """Every print of a list as an Entry; what is no print at all is left out."""
    try:
        items = [] if isinstance(values, (str, bytes)) else list(values)
    except TypeError:
        return []
    return [record(value) for value in items if isinstance(value, (Entry, dict))]


def from_values(values) -> list:
    """The prints that stand in the settings, as the list of dicts the settings keep them in.

    What is no print at all is left out, and a print with one field that is unusable is not
    left out: that field takes its default and the rest of the print stays (see `from_dict`).
    This is what volura.app.Settings keeps under `print_history`, newest first."""
    return [to_dict(value) for value in records(values)]


def add(entries, entry, limit=LIMIT) -> list:
    """`entry` on top of the history, newest first; what goes past `limit` drops off.

    A print that drops off also loses the copy Volura kept of its G-code, so the folder does
    not grow without end; a file the user saved never goes with it (see `is_copy`). A limit
    that is no limit is the usual one, and the print that was just made is never the one that
    drops off."""
    try:
        kept = max(1, int(limit))
    except (TypeError, ValueError):
        kept = LIMIT
    out = [to_dict(entry)]
    for value in records(entries):
        if len(out) < kept:
            out.append(to_dict(value))
        else:
            forget(value.gcode)
    return out


def remove(entries, entry_id) -> list:
    """The history without the print of that id; its copy of the G-code goes with it."""
    out = []
    for value in records(entries):
        if value.id == str(entry_id):
            forget(value.gcode)
        else:
            out.append(to_dict(value))
    return out


def cleared(entries) -> list:
    """An empty history, with every copy of a G-code taken away; the files the user saved
    itself stay."""
    for value in records(entries):
        forget(value.gcode)
    return []


def strongest(hows) -> str:
    """The strongest thing that happened to a print, out of the ones that happened.

    "" when nothing did: a preview that was only looked at has nothing to remember."""
    best = ""
    for how in hows or ():
        if how in STRONGEST and STRONGEST[how] > STRONGEST.get(best, -1):
            best = how
    return best


def printed(entry) -> bool:
    """True when this print used filament: it started on a printer, or the user said it did.

    A G-code on an SD card has printed nothing yet, and neither has one that only stands on
    the printer (see volura.spools, which is weighed the same way)."""
    return record(entry).how in (STARTED, MARKED)


def totals(entries) -> tuple:
    """How many prints really ran, with the grams and the minutes they took."""
    count, grams, minutes = 0, 0.0, 0.0
    for value in records(entries):
        if printed(value):
            count += 1
            grams += value.grams
            minutes += value.minutes
    return count, grams, minutes


# ------------------------------------------------------------------ the G-codes of the history


def gcode_dir(create=True) -> str:
    """The folder that keeps the G-codes of prints that went to a printer.

    It lives in the folder of the program, next to the log and the autosave; "" when there
    is no folder to write to."""
    from . import errors               # here and not at the top: this module stays without Qt
    folder = os.path.join(errors.log_dir(), *GCODE_FOLDER)
    if create:
        try:
            os.makedirs(folder, exist_ok=True)
        except OSError:
            return ""
    return folder


def is_copy(path) -> bool:
    """True when `path` is a G-code that Volura kept itself, in its own folder.

    Only those are Volura's to take away; what the user saved elsewhere is nothing to Volura,
    also not when the entry that points at it leaves the history."""
    folder, path = gcode_dir(create=False), str(path or "").strip()
    if not folder or not path:
        return False
    folder = os.path.normcase(os.path.abspath(folder)) + os.sep
    return os.path.normcase(os.path.abspath(path)).startswith(folder)


def copy_path(entry_id) -> str:
    """The file the G-code of a print of that id is kept in; "" when there is no folder."""
    folder = gcode_dir()
    if not folder:
        return ""
    return os.path.join(folder, _SAFE.sub("_", str(entry_id or "")) + GCODE_SUFFIX)


def keep_copy(path, entry_id) -> str:
    """The G-code a print went to the printer with, copied into the folder of the history.

    That file is a temporary one: the preview that showed the print takes it away when it
    closes, so without this copy there is nothing to print again. "" when there was nothing
    to copy, which leaves the print without a G-code of its own."""
    source = str(path or "").strip()
    if not source or not os.path.isfile(source):
        return ""
    target = copy_path(entry_id)
    if not target:
        return ""
    try:
        shutil.copyfile(source, target)
    except OSError:
        return ""
    return target


def forget(path):
    """Take away a G-code of the history itself; what the user saved is never touched."""
    if not is_copy(path):
        return
    try:
        os.remove(path)
    except OSError:
        pass


def gcode_of(entry):
    """The G-code of a print that is still there, or None when it is gone (or never was)."""
    path = record(entry).gcode
    return path if path and os.path.isfile(path) else None


def store(settings, entry, source="", limit=LIMIT):
    """Write `entry` on top of the history of `settings` and keep it.

    `source` is the G-code the print was made with, for a print that went to a printer: that
    file does not outlive the preview, so a copy of it is kept with the history. Returns the
    entry, with its `gcode` filled in."""
    entry = record(entry)
    if source and entry.how in (SENT, STARTED) and not entry.gcode:
        entry.gcode = keep_copy(source, entry.id) or source
    if settings is not None and hasattr(settings, "print_history"):
        settings.print_history = add(settings.print_history, entry, limit)
        settings.save()
    return entry


# ------------------------------------------------------------------ how the list says it


def how_text(how) -> str:
    """What the list calls what became of a print, as "Started"; "Saved" for anything else."""
    return tr(HOW_LABELS.get(_how(how), HOW_LABELS[SAVED]))


def when_text(when) -> str:
    """The moment a print was made, as the list says it; what is no moment stands as it was
    written."""
    text = str(when or "").strip()
    try:
        return datetime.fromisoformat(text).strftime("%Y-%m-%d %H:%M")
    except (TypeError, ValueError):
        return text


def grams_text(grams) -> str:
    """What a print used, in the grams the rest of Volura says (see volura.spools)."""
    return spools.grams_text(grams) + " g"


def time_text(minutes) -> str:
    """How long a print takes, from the minutes the slice worked out."""
    return printer_link.format_duration(_number(minutes, NO_TIME) * 60.0)


def entry_text(entry) -> str:
    """One short line about a print: what it was, what became of it and what it used."""
    entry = record(entry)
    line = tr("{how} {when}: {name}").format(how=how_text(entry.how), when=when_text(entry.when),
                                             name=entry.name)
    if entry.printer:
        line += " " + tr("on {printer}").format(printer=entry.printer)
    if entry.grams > 0.0:
        line += " - " + tr("{grams} in {time}").format(grams=grams_text(entry.grams),
                                                       time=time_text(entry.minutes))
    return line


def totals_text(entries) -> str:
    """The line under the list: how much has been printed with what Volura remembers."""
    count, grams, minutes = totals(entries)
    return trn("{n} print, {kg} kg filament, {h} h printing",
               "{n} prints, {kg} kg filament, {h} h printing", count).format(
        n=count, kg=round(grams / 1000.0, 2), h=round(minutes / 60.0, 1))

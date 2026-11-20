"""Presets of the user for File > Slice: a printer and what to print with, under a name.

No Qt in this file. A preset is what the slice dialog's `choice()` gives - the printer with
its nozzle, material and quality, and the infill, support, bed adhesion and extras that go
on top of the profile (volura.slice_dialog) - with a name in front of it. Printing the same
few combinations again and again, "PETG functional" or "PLA quick draft", then means one
pick from a list instead of eleven fields again.

What a print does with the colours of the model is not part of a preset: which filament
prints which colour, how deep the skins of painted faces go, the heights of the manual
colour changes and the command that asks for them belong to the model, not to how it prints.

The settings keep the presets as `slice_presets`, one line of JSON each like the printers of
volura.printer_link, and the name of the one chosen last as `slice_preset` (volura.app.Settings).
A preset that does not know a key yet - an older one without "walls" - loads and leaves that
one field in the dialog as it is; a preset of a printer that is no longer in the profile
library loads too, and the dialog says so instead of applying it.
"""
from __future__ import annotations

import json

#: what a preset holds: the keys of the dict `SliceDialog.choice()` returns that say how a
#: print is made; the colours of the model and the filament changes are not among them
PRESET_KEYS = ("printer", "variant", "material", "quality", "infill", "support", "adhesion",
               "infill_pattern", "adaptive", "ironing", "fuzzy", "fuzz", "walls", "seam", "vase",
               "thumbnail")


def to_json(preset) -> str:
    """One preset as a line of JSON, for the list of presets in the settings."""
    out = {"name": str(preset.get("name", ""))}
    out.update({key: preset[key] for key in PRESET_KEYS if key in preset})
    return json.dumps(out)


def from_json(text):
    """The preset that stands in that line of JSON; None when it is not one.

    A key that does not belong to a preset is dropped, and a key that is missing stays
    missing: an older preset that knows nothing about the number of walls still loads, and
    the dialog leaves that field alone. So does a preset of a printer that is no longer in
    the profile library - only the dialog knows that library, so only it can say so."""
    try:
        data = json.loads(text)
    except (TypeError, ValueError):
        return None
    if not isinstance(data, dict) or not str(data.get("name", "")).strip():
        return None
    out = {"name": str(data["name"]).strip()}
    out.update({key: data[key] for key in PRESET_KEYS if key in data})
    return out


def to_lines(presets) -> list:
    """The presets as the lines of JSON that the settings keep them in."""
    return [to_json(preset) for preset in presets]


def from_lines(lines) -> list:
    """The presets in the lines of the settings; a line that is not one is skipped."""
    if isinstance(lines, str):
        lines = [lines]
    out = []
    for line in lines or []:
        preset = from_json(line)
        if preset is not None:
            out.append(preset)
    return out


def find(presets, name):
    """The preset of that name; None when there is no such preset."""
    for preset in presets:
        if str(preset.get("name", "")) == str(name):
            return preset
    return None


def same(one, other) -> bool:
    """Whether two values of a choice are the same thing.

    JSON gives back 4 where a spin box has 4.0, and a tick is not the number 1."""
    if isinstance(one, bool) != isinstance(other, bool):
        return False
    try:
        return abs(float(one) - float(other)) < 1e-6
    except (TypeError, ValueError):
        return str(one) == str(other)


def matches(preset, choice) -> bool:
    """Whether a choice (what `SliceDialog.choice()` gives) is still what a preset holds.

    Only the keys the preset holds count, so an older preset that knows nothing about the
    number of walls still matches the dialog, whatever the number of walls is there."""
    return all(key in choice and same(preset[key], choice[key])
               for key in preset if key in PRESET_KEYS)

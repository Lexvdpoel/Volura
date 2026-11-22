"""What the calibration prints taught you about one filament on one printer.

No Qt in this file: the wizard of 3D Printing > Calibrate Filament… (volura.calibration_wizard)
asks this module what a calibration may hold and how it stands in the settings, and File > Slice
(volura.slice_dialog) asks it which Cura settings and which command line a calibration gives. A
calibration is keyed by the printer and the material it was found with - the same two the slice
dialog picks - because a temperature that is right for one machine is wrong for the next, and the
flow that suits one filament does not suit the other.

Five test prints give six numbers: the nozzle temperature, the material flow, the pressure advance
with the firmware that sets it, the retraction length and the highest volumetric flow the hotend
still melts. Every one of them is on its own: a test you skipped is simply not there, and what is
not there is not applied either. The settings keep the whole lot under `calibrations`, one line of
JSON each like the spools of volura.spools, and because a settings file is also a file a person can
edit, `clean` takes every field that is no usable number out before anything is applied.

A maximum flow is no setting of its own: a line of `layer_height` by `line_width` millimetres needs
`max_flow / (layer * width)` millimetres per second at most, so it comes back as the print speeds
brought down to that - brought down and never up, because a calibration is what you found, not what
you hope. Pressure advance is no Cura setting either: it goes into the G-code as the command line of
its firmware, the same one a Pressure Advance Test asks for (volura.pressure_advance_test).
"""
from __future__ import annotations

import json
import math

from .i18n import tr

#: the separator between the printer and the material in the key of a calibration
SEPARATOR = "|"
#: the numbers a calibration keeps, in the order they are written; `firmware` belongs to `advance`
FIELDS = ("temperature", "flow", "advance", "firmware", "retraction", "max_flow")
#: °C the coolest and the hottest a nozzle prints at, as in the temperature tower
TEMP_MIN, TEMP_MAX = 150.0, 300.0
#: % the leanest and the richest flow a flow rate test walks through
FLOW_MIN, FLOW_MAX = 50.0, 150.0
#: the lowest and the highest pressure advance value a firmware takes
ADVANCE_MIN, ADVANCE_MAX = 0.0, 5.0
#: mm the shortest and the longest retraction; 0 mm says this filament wants none at all
LENGTH_MIN, LENGTH_MAX = 0.0, 10.0
#: mm³/s the fewest and the most plastic a hotend melts per second
MELT_MIN, MELT_MAX = 1.0, 60.0
#: the bounds of every number, and how many decimals it keeps, so a value is written as it is read
BOUNDS = {"temperature": (TEMP_MIN, TEMP_MAX), "flow": (FLOW_MIN, FLOW_MAX),
          "advance": (ADVANCE_MIN, ADVANCE_MAX), "retraction": (LENGTH_MIN, LENGTH_MAX),
          "max_flow": (MELT_MIN, MELT_MAX)}
DIGITS = {"temperature": 0, "flow": 1, "advance": 4, "retraction": 2, "max_flow": 1}
#: the Cura setting of every field that is one; a pressure advance is no setting and a maximum flow
#: is no setting of its own, and the temperature of the first layer stays what the profile has
SETTING = {"temperature": "material_print_temperature", "flow": "material_flow",
           "retraction": "retraction_amount"}
#: the print speeds that lay plastic, and are brought down by a maximum flow
SPEED_KEYS = ("speed_print", "speed_infill", "speed_wall", "speed_wall_0", "speed_wall_x")
#: the decimals a lowered speed is rounded to, and always downwards, so a cap never speeds a print up
SPEED_DIGITS = 1

#: the materials a temperature preset is known for (volura.temperature_tower)
KINDS = ("PLA", "PETG", "ABS", "ASA", "TPU")


def key(printer_id, material_id) -> str:
    """The key a calibration stands under: the printer and the material it was found with."""
    return f"{printer_id}{SEPARATOR}{material_id}"


def parts(name) -> tuple | None:
    """The ``(printer, material)`` of a key; None when it is no key at all."""
    text = str(name or "")
    if SEPARATOR not in text:
        return None
    printer, _sep, material = text.partition(SEPARATOR)
    printer, material = printer.strip(), material.strip()
    if not printer or not material or SEPARATOR in material:
        return None
    return printer, material


def number_text(value) -> str:
    """A number of a calibration the way a summary and a command write it: 97.5, not
    97.50000000000001; one that is no number is written as it stands."""
    try:
        made = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isfinite(made) and abs(made) < 1e-9:
        made = 0.0                            # -0.0 and -1e-17 are the same value here
    return f"{made:g}"


def _number(value, lo, hi):
    """`value` as a number between `lo` and `hi`, or None when there is nothing to keep.

    Nothing at all, a text that is no number, a tick and a number that is not a number (the square
    root of -1, infinity) all give None: a field that is not there is skipped, not guessed at. What
    is there but past what a printer does is taken to the nearest bound, which is what the field of
    the wizard that typed it would have given anyway."""
    if value is None or isinstance(value, bool):
        return None
    try:
        made = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(made):
        return None
    return max(lo, min(hi, made))


def _firmware(value) -> str:
    """The firmware a pressure advance value is set with; one Volura does not know - or none at
    all - is the one most printers run, because the value is worth applying either way."""
    from . import pressure_advance_test          # only the command of a firmware is needed
    text = str(value or "").strip().lower()
    return text if text in pressure_advance_test.FIRMWARES else (
        pressure_advance_test.FIRMWARE_DEFAULT)


def clean(data) -> dict:
    """What is kept of one calibration: every field that is a usable number, in the order they are
    written, and nothing else.

    A calibration the wizard writes is whole, but the settings are also a file a person edits, and
    an older Volura may not know a field yet, so nothing here may break on what it finds. A
    pressure advance value always gets a firmware of its own, because without one there is no
    command line to write; a field that is junk, a text that is no number or a whole calibration
    that is no list of numbers at all simply has fewer numbers in the end - and nothing found is an
    empty dict, which is no calibration (`find` gives None for it)."""
    if not isinstance(data, dict):
        return {}
    out = {}
    for name in FIELDS:
        if name == "firmware":
            continue
        value = _number(data.get(name), *BOUNDS[name])
        if value is not None:
            out[name] = round(value, DIGITS[name])
    if "advance" in out:
        out["firmware"] = _firmware(data.get("firmware"))
    return out


def find(kept, printer_id, material_id) -> dict | None:
    """What is kept for this printer and this filament; None when there is nothing to apply.

    A printer or a material that is no text is no key, and an entry that lost every number of it -
    by hand, or because the wizard that wrote it was cancelled - is no calibration either."""
    if not printer_id or not material_id:
        return None
    got = (kept or {}).get(key(printer_id, material_id)) if isinstance(kept, dict) else None
    return clean(got) or None


def to_json(printer_id, material_id, found) -> str:
    """One calibration as a line of JSON: the printer and the filament it is for, and what was
    found for them."""
    made = {"printer": str(printer_id), "material": str(material_id)}
    made.update(clean(found))
    return json.dumps(made)


def from_json(line) -> tuple:
    """The ``(key, numbers)`` of one line of the settings, or None when the line is no calibration:
    no JSON object, no printer on it, no material to go with it or nothing found in it."""
    if isinstance(line, dict):
        data = line
    else:
        try:
            data = json.loads(line)
        except (TypeError, ValueError):
            return None
    if not isinstance(data, dict):
        return None
    printer = str(data.get("printer") or "").strip()
    material = str(data.get("material") or "").strip()
    if not printer or not material:
        return None
    made = {name: data[name] for name in FIELDS if name in data}
    numbers = clean(made)
    return (key(printer, material), numbers) if numbers else None


def to_lines(kept) -> list:
    """The calibrations as the lines of JSON the settings keep them in.

    A key that is no printer and filament, and an entry with nothing found in it, are left out:
    junk in the settings is not worth a line, and an empty calibration would only say that nothing
    was found."""
    out = []
    for name, found in (kept.items() if isinstance(kept, dict) else []):
        pair, numbers = parts(name), clean(found)
        if pair and numbers:
            out.append(to_json(pair[0], pair[1], numbers))
    return out


def from_lines(lines) -> dict:
    """The calibrations of the settings as a dict of key -> numbers; a line that is no calibration
    is left out, as is one whose printer or filament is gone.

    A dict comes back as it went in - through the lines, so every field is checked again - and a
    value of the settings that is no list at all is no shelf of calibrations."""
    if isinstance(lines, dict):
        lines = to_lines(lines)
    if isinstance(lines, str):
        lines = [lines]
    if not isinstance(lines, (list, tuple)):
        return {}
    out = {}
    for line in lines:
        made = from_json(line)
        if made:
            out[made[0]] = made[1]
    return out


def material_type(name, default="PLA") -> str:
    """Which kind of filament a material name describes: the one the temperature presets know.

    The profiles say what a filament is - "Generic PETG", "HIPS" - so the first kind above that
    comes back in the name wins. A name that names none of them - a nylon, nothing at all, no text
    - takes `default`, which a caller takes from what the user filled in under Window > Preferences
    > Filament; a default that is no kind either is PLA, which is what most people print."""
    text = str(name or "").upper()
    for kind in KINDS:
        if kind in text:
            return kind
    return default if default in KINDS else KINDS[0]


def values(found) -> dict:
    """The Cura settings a calibration gives: the temperature of the print, its flow and its
    retraction. What was skipped is not there, and so is not asked of the profile; the temperature
    of the first layer stays what the profile has, because that one is about the bed, not about the
    filament."""
    out = {}
    for name, setting in SETTING.items():
        if found and name in found:
            out[setting] = float(found[name])
    return out


def fill_in(made, found) -> dict:
    """`made` with every setting of `found` that it does not hold yet.

    Your own choice in the slice dialog goes above a calibration: a calibration fills in what the
    dialog leaves to the profile, it does not overrule what was chosen there."""
    out = dict(made or {})
    for name, value in values(found).items():
        out.setdefault(name, value)
    return out


def advance_command(found):
    """The command line with which the firmware of this calibration sets its pressure advance, or
    None when there is no pressure advance to set.

    The same syntax a Pressure Advance Test writes, so a printer that takes the one takes the
    other; a firmware a hand-written settings file made up is no command either."""
    from . import pressure_advance_test          # the command belongs to the firmware
    if not found or "advance" not in found:
        return None
    try:
        return pressure_advance_test.command(found.get("firmware"), found["advance"])
    except ValueError:
        return None


def speed_cap(max_flow, layer_height, line_width, current) -> dict:
    """The speeds in `current` that a hotend melting `max_flow` mm³/s cannot keep up with.

    A line of `layer_height` by `line_width` millimetres lays `speed * layer * width` cubic
    millimetres per second, so every speed above `max_flow / (layer * width)` comes back lowered to
    that number - rounded down, so a cap never makes a print faster than the flow it was found
    with. A speed that already stays under it is left out, and so is one that is not there: what the
    profile does not set, the cap does not touch. No flow, no layer height, no line width, or one of
    them no number or no positive measure, and nothing is lowered."""
    flow, layer, wide = (_number(max_flow, 0.0, 1e6), _number(layer_height, 0.0, 1e6),
                         _number(line_width, 0.0, 1e6))
    if not flow or not layer or not wide:
        return {}
    # the quotient of a 0.2 mm layer of 0.4 mm wide carries the noise of those numbers; it is
    # rounded off before it is floored, so the cap is the number the flow works out to
    limit = math.floor(round(flow / (layer * wide), 6) * 10 ** SPEED_DIGITS
                       + 1e-9) / 10 ** SPEED_DIGITS
    out = {}
    for name in SPEED_KEYS:
        speed = _number((current or {}).get(name), 0.0, 1e6)
        if speed is not None and speed > limit + 1e-9:
            out[name] = limit
    return out


def text_of(found) -> str:
    """What was found, as one line: "215 °C, flow 97.5 %, retraction 0.8 mm, PA 0.035,
    max 15 mm³/s".

    Only what is there is named, so a filament that only has a temperature says so and nothing more;
    nothing found is an empty line, which is how a dialog hides the whole row. The temperatures of
    the nozzle come first, then what the extruder does with the plastic."""
    if not found:
        return ""
    bits = []
    if "temperature" in found:
        bits.append(tr("{temp} °C").format(temp=number_text(found["temperature"])))
    if "flow" in found:
        bits.append(tr("flow {flow} %").format(flow=number_text(found["flow"])))
    if "retraction" in found:
        bits.append(tr("retraction {length} mm").format(length=number_text(found["retraction"])))
    if "advance" in found:
        bits.append(tr("PA {value}").format(value=number_text(found["advance"])))
    if "max_flow" in found:
        bits.append(tr("max {flow} mm³/s").format(flow=number_text(found["max_flow"])))
    return ", ".join(bits)


def lowered_text(lowered) -> str:
    """What a maximum flow did to the print speeds, in the words of the dialog: every speed it had
    to bring down is brought down to the same number, so one number says it all."""
    if not lowered:
        return ""
    return tr("speed lowered to {speed} mm/s").format(speed=number_text(min(lowered.values())))

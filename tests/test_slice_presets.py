"""Presets of the user for File > Slice: a printer and what to print with, under a name.

`volura.slice_presets` keeps such a preset as one line of JSON in the settings and reads it
back, without Qt in between. These tests check that round trip and what an older or a broken
line does with it: a key that a preset does not hold stays out and is not made up, a line that
is not a preset is skipped, a preset of a printer that is no longer in the library loads (the
dialog is the one that knows the library), and `matches` says whether a dialog still holds what
a preset holds.

Run with:  python -m pytest tests/test_slice_presets.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from volura import slice_presets  # noqa: E402


def choice(**over):
    """A choice as `SliceDialog.choice()` gives it, with these fields different."""
    got = {"printer": "creality_ender3", "variant": "0.4mm Nozzle",
           "material": "generic_pla_175", "quality": "standard",
           "infill": 20, "support": 1, "adhesion": "brim", "infill_pattern": "gyroid",
           "paint": 1.0, "changes": "5", "change_command": "M600",
           "adaptive": True, "ironing": False, "fuzzy": True, "fuzz": 0.35, "walls": 3}
    got.update(over)
    return got


def preset(name="PETG functional", **over):
    """A preset: that choice with a name in front of it."""
    return {"name": name, **choice(), **over}


def test_every_field_of_a_preset_comes_back():
    line = slice_presets.to_json(preset())
    got = slice_presets.from_json(line)
    assert got["name"] == "PETG functional"
    for key in slice_presets.PRESET_KEYS:
        assert got[key] == choice()[key], key
    assert json.loads(line)["infill"] == 20


def test_the_colours_of_the_model_stay_out_of_a_preset():
    """The filament plan, the colour changes and the command that asks for them belong to the
    model, so a preset does not carry them and cannot put them back in another model."""
    data = json.loads(slice_presets.to_json(preset()))
    assert set(data) == {"name", *slice_presets.PRESET_KEYS}
    for key in ("paint", "changes", "change_command"):
        assert key not in data
    assert slice_presets.from_json(json.dumps(choice())) is None, "a choice without a name"


def test_a_key_a_preset_does_not_hold_stays_missing():
    """An older preset that knows nothing about the number of walls still loads, and nothing
    is made up for it; the dialog is the one that leaves that field alone."""
    old = slice_presets.from_json(json.dumps({"name": "old", "printer": "testprinter",
                                              "infill": 20, "walls_of_the_future": 4}))
    assert old == {"name": "old", "printer": "testprinter", "infill": 20}


def test_a_preset_of_a_printer_that_is_no_longer_there_still_loads():
    """Only the dialog knows the profile library, so the preset of a printer that is gone
    comes back whole; the dialog says so instead of applying it."""
    got = slice_presets.from_json(slice_presets.to_json(preset(printer="gone_printer")))
    assert got["printer"] == "gone_printer"


def test_a_line_that_is_not_a_preset_is_no_preset():
    for line in ("", None, 5, "not json", "[]", "{}", '{"infill": 20}', '{"name": ""}',
                 '{"name": "   "}'):
        assert slice_presets.from_json(line) is None, line


def test_a_broken_line_in_the_settings_is_skipped():
    kept = [preset("first"), preset("second", infill=40)]
    lines = slice_presets.to_lines(kept)
    assert len(lines) == 2
    assert [p["name"] for p in slice_presets.from_lines(lines)] == ["first", "second"]
    mixed = [lines[0], "{ not json", "", "[]", '{"name": ""}', lines[1]]
    assert [p["name"] for p in slice_presets.from_lines(mixed)] == ["first", "second"]
    assert [p["name"] for p in slice_presets.from_lines(lines[0])] == ["first"], \
        "one line is also a settings list"
    assert slice_presets.from_lines([]) == [] and slice_presets.from_lines(None) == []


def test_find_gives_the_preset_of_that_name():
    kept = [preset("first"), preset("second")]
    assert slice_presets.find(kept, "second")["infill"] == 20
    assert slice_presets.find(kept, "third") is None
    assert slice_presets.find([], "first") is None


def test_matches_says_whether_the_dialog_still_holds_the_preset():
    kept = preset()
    assert slice_presets.matches(kept, choice())
    assert not slice_presets.matches(kept, choice(infill=40))
    assert not slice_presets.matches(kept, choice(adhesion=""))
    assert not slice_presets.matches(kept, choice(quality="draft")), "another quality"
    assert not slice_presets.matches(kept, choice(walls=0)), "0 leaves it to the profile"
    assert not slice_presets.matches(kept, {}), "an empty dialog holds nothing"


def test_matches_only_counts_the_keys_the_preset_holds():
    """An older preset does not know the number of walls, so it still matches whatever the
    number of walls is - but not another setting it does know."""
    old = slice_presets.from_json(json.dumps({"name": "old", "printer": "testprinter",
                                              "infill": 20}))
    assert slice_presets.matches(old, choice(printer="testprinter", walls=9))
    assert not slice_presets.matches(old, choice(printer="testprinter", walls=9, infill=21))


def test_the_same_value_in_the_json_of_a_preset_and_in_the_dialog():
    assert slice_presets.same(4, 4.0), "a spin box has 4.0 where JSON has 4"
    assert slice_presets.same(0.3, 0.30000000001)
    assert slice_presets.same("brim", "brim") and not slice_presets.same("brim", "raft")
    assert not slice_presets.same(True, 1) and not slice_presets.same(1, True), "a tick is no 1"
    assert slice_presets.same(False, False) and not slice_presets.same(False, True)
    assert not slice_presets.same(20, None) and not slice_presets.same(None, "x")

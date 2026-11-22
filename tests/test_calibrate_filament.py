"""Calibrate Filament…: the five calibration prints of one filament, walked in one wizard.

`volura.calibrations` is the store and the arithmetic behind it - no Qt: what a calibration may hold
(`clean` throws away everything that is no usable number, because the settings are also a file a
person edits), how it is keyed by the printer and the filament it was found for and stands in the
settings as one line of JSON each, which Cura settings it gives (`values` and `fill_in`: what you
chose in the slice dialog wins over it), what one maximum flow does to the print speeds (`speed_cap`:
a line of `layer_height` by `line_width` millimetres lays `speed * layer * width` cubic millimetres
per second, so a speed above `max_flow / (layer * width)` comes back lowered to that number and never
raised) and which command line sets its pressure advance (`advance_command`, the syntax of the
firmware that belongs to the value).

`volura.calibration_wizard` is the walk: a first page with the printer and the filament those numbers
belong to - the same cascade as File > Slice, and the same two that make the key - then one page per
test with two sentences about the print, a button that puts the test on the free spot of the bed
through the very methods the five dialogs use, and the number read off the print or a Skip. A page on
Skip is not in the result and is applied to nothing, which is what makes a second walk fill in what
is missing and leave the rest alone; a walk that skipped everything saves nothing at all. `File >
Slice` shows what was found, prints with it when Use my calibration is ticked, lowers the speeds to
the flow that was found and hands the pressure advance to the slice as one command line.

These tests (Qt offscreen) check the store and everything junk can look like in a settings file, the
walk and its pages, the five bed buttons, what the slice dialog does with a calibration and what it
does without one, the commands the window merges into the slice, and, when CuraEngine is there, that
the numbers really reach the G-code.

Run with:  python -m pytest tests/test_calibrate_filament.py
"""
import json
import os
import statistics
import subprocess
import sys
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import manifold3d as mf  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from PySide6.QtWidgets import QApplication, QDialog, QWizardPage  # noqa: E402

from test_cura_profiles import (DUAL, EXTRUDER, FDMPRINTER, FDMEXTRUDER, PRINTER, _inst,  # noqa: E402
                                _material)
from volura import (app as app_module, calibration_wizard, calibrations, cura_profiles,  # noqa: E402
                    flow_rate_test, max_speed_test, ops, pressure_advance_test, retraction_test,
                    slice_dialog as sd, slice_presets, slicer, temperature_tower, window_print)
from volura.settings import qsettings  # noqa: E402

_app = QApplication.instance() or QApplication(sys.argv[:1])   # the wizard needs fonts and Qt

ENGINE, RESOURCES = slicer.find_engine(), slicer.find_resources()
needs_engine = pytest.mark.skipif(
    ENGINE is None or RESOURCES is None or slicer.engine_problem(ENGINE) is not None,
    reason="no CuraEngine that runs here (tools/fetch_curaengine.py; Linux: glibc 2.38+)")

#: what the filament of this test library was calibrated to: all five numbers, on Klipper
FOUND = {"temperature": 205.0, "flow": 97.5, "advance": 0.035, "firmware": "klipper",
         "retraction": 0.8, "max_flow": 14.0}

#: the five methods of the window that put a test on the bed
APPLY = ("_apply_temperature_tower", "_apply_flow_rate_test", "_apply_pressure_advance_test",
         "_apply_retraction_test", "_apply_max_speed_test")


class FakeBox:
    """Stands in for QMessageBox: remembers what the window asked."""

    def __init__(self):
        self.asked = []

    def warning(self, *args, **kwargs):
        self.asked.append(args[-1])

    def information(self, *args, **kwargs):
        self.asked.append(args[-1])

    def critical(self, *args, **kwargs):
        self.asked.append(args[-1])


@pytest.fixture()
def win(monkeypatch, tmp_path):
    """A window in which nothing has been calibrated yet, so every test puts in its own numbers."""
    monkeypatch.setattr(app_module.errors, "log_dir", lambda: str(tmp_path))
    monkeypatch.setattr(app_module, "autosave_path", lambda: str(tmp_path / "autosave.volura"))
    monkeypatch.setattr(app_module, "QMessageBox", FakeBox())
    w = app_module.MainWindow()
    for key, value in (("filament_material", "PLA"), ("pressure_advance_firmware", "marlin"),
                       ("slice_printer", "testprinter"), ("slice_variant", "0.4mm Nozzle"),
                       ("slice_material", "generic_pla_175"), ("slice_quality", "standard"),
                       ("slice_calibration", True),
                       ("temperature_tower_preset", "PLA"), ("temperature_tower_start", 220.0),
                       ("temperature_tower_end", 190.0), ("temperature_tower_step", 5.0),
                       ("flow_rate_test_start", 90.0), ("flow_rate_test_end", 110.0),
                       ("flow_rate_test_step", 2.5)):
        setattr(w.settings, key, value)
    w.settings.calibrations = {}
    w.settings.save()                 # the settings file of this run holds no calibration either
    w.doc.dirty = False
    yield w
    w.doc.dirty = False
    w.close()


@pytest.fixture()
def lib(tmp_path):
    """A little Cura library of two printers, with PLA and PETG on 1.75 mm and the qualities the
    slice dialog needs (the same little world tests/test_cura_profiles.py builds)."""
    d, e = tmp_path / "definitions", tmp_path / "extruders"
    d.mkdir()
    e.mkdir()
    for name, doc in (("fdmprinter", FDMPRINTER), ("fdmextruder", FDMEXTRUDER),
                      ("testprinter", PRINTER), ("dual", DUAL)):
        (d / f"{name}.def.json").write_text(json.dumps(doc), encoding="utf-8")
    (e / "testprinter_extruder_0.def.json").write_text(json.dumps(EXTRUDER), encoding="utf-8")
    for i in (0, 1):
        doc = dict(EXTRUDER, metadata={"machine": "dual", "position": str(i)},
                   overrides={"extruder_nr": {"default_value": i}})
        (e / f"dual_extruder_{i}.def.json").write_text(json.dumps(doc), encoding="utf-8")
    for size in ("0.4", "0.6"):
        _inst(tmp_path / "variants" / "test" / f"testprinter_{size}.inst.cfg",
              {"definition": "testprinter", "name": f"{size}mm Nozzle", "version": 4},
              {"hardware_type": "nozzle", "type": "variant"}, {"machine_nozzle_size": size})
    q = tmp_path / "quality" / "test"
    for qt, layer in (("standard", "0.2"), ("draft", "0.3")):
        _inst(q / f"global_{qt}.inst.cfg",
              {"definition": "testprinter", "name": f"{qt.capitalize()} Quality", "version": 4},
              {"global_quality": "True", "quality_type": qt, "type": "quality"},
              {"layer_height": layer})
    for qt in ("standard", "draft"):
        _inst(q / f"pla_{qt}_0.4.inst.cfg",
              {"definition": "testprinter", "name": qt, "version": 4},
              {"quality_type": qt, "material": "generic_pla_175", "variant": "0.4mm Nozzle",
               "type": "quality"}, {"speed_print": "40"})
    m = tmp_path / "materials"
    m.mkdir()
    _material(m / "generic_pla_175.xml.fdm_material", "generic_pla_175", "PLA", 1.75, 200)
    _material(m / "generic_petg_175.xml.fdm_material", "generic_petg_175", "PETG", 1.75, 240)
    return cura_profiles.Library(str(tmp_path))


def pump_until(predicate, timeout_ms=60000):
    deadline = time.monotonic() + timeout_ms / 1000.0
    while not predicate() and time.monotonic() < deadline:
        _app.processEvents()
        time.sleep(0.005)
    return predicate()


def wizard(win, lib):
    return calibration_wizard.CalibrationWizard(win, lib)


def found(win, printer="testprinter", material="generic_pla_175", **over):
    """Keep the five numbers for a printer and a filament, as the wizard's Save would."""
    made = calibrations.clean(dict(FOUND, **over))
    win.settings.calibrations[calibrations.key(printer, material)] = made
    win.settings.save()
    return made


def fill(page, value):
    """A page's way of saying "this is what I read off the print"."""
    page.skip.setChecked(False)
    page.value.setValue(value)


# ------------------------------------------------------------------ the store

def test_a_key_holds_the_printer_and_the_filament_it_was_found_for():
    assert calibrations.key("pr", "ma") == "pr|ma"
    assert calibrations.parts("pr|ma") == ("pr", "ma")


@pytest.mark.parametrize("name", ("", "nothing", "pr|", "|ma", "a|b|c", None, 7))
def test_a_key_that_is_no_key_is_no_key(name):
    assert calibrations.parts(name) is None


def test_every_field_keeps_only_what_is_a_usable_number():
    assert calibrations.clean(FOUND) == {"temperature": 205.0, "flow": 97.5, "advance": 0.035,
                                         "retraction": 0.8, "max_flow": 14.0,
                                         "firmware": "klipper"}


@pytest.mark.parametrize("junk", (None, "warm", True, [1], float("nan"), float("inf"), {}))
def test_a_field_of_junk_is_left_out_rather_than_guessed_at(junk):
    assert calibrations.clean({"temperature": junk, "flow": 97.5}) == {"flow": 97.5}


@pytest.mark.parametrize("junk", (None, "everything", 7, ["temperature"],
                                  {"temperature": "warm"}, 3.5))
def test_a_whole_calibration_that_is_no_list_of_numbers_is_empty(junk):
    assert calibrations.clean(junk) == {}


def test_a_number_past_what_a_printer_does_is_taken_to_the_nearest_bound():
    made = calibrations.clean({"temperature": 900, "flow": 4, "advance": -3, "retraction": 999,
                               "max_flow": 0})
    assert made["temperature"] == calibrations.TEMP_MAX
    assert made["flow"] == calibrations.FLOW_MIN
    assert made["advance"] == calibrations.ADVANCE_MIN
    assert made["retraction"] == calibrations.LENGTH_MAX
    assert made["max_flow"] == calibrations.MELT_MIN


def test_a_huge_flow_is_held_to_what_a_hotend_melts():
    """A settings file that says 100000 mm³/s is a mistake, not a licence to print without speed."""
    made = calibrations.clean({"max_flow": 1e9, "temperature": 1e9})
    assert made["max_flow"] == calibrations.MELT_MAX
    assert made["temperature"] == calibrations.TEMP_MAX


def test_a_pressure_advance_value_always_gets_a_firmware_of_its_own():
    assert calibrations.clean({"advance": 0.04}) == {"advance": 0.04, "firmware": "marlin"}
    assert calibrations.clean({"advance": 0.04, "firmware": "toaster"})["firmware"] == "marlin"
    assert "firmware" not in calibrations.clean({"retraction": 0.8, "firmware": "klipper"})


def test_the_order_of_the_fields_is_the_order_of_the_walk():
    assert [one.key for one in calibration_wizard.TESTS] == ["temperature", "flow", "advance",
                                                             "retraction", "max_flow"]
    assert list(calibrations.FIELDS) == ["temperature", "flow", "advance", "firmware", "retraction",
                                         "max_flow"]


def test_nothing_found_is_no_calibration():
    assert calibrations.find({"a|b": {"flow": 97.5}}, "a", "b") == {"flow": 97.5}
    assert calibrations.find({"a|b": FOUND}, "a", "other") is None
    assert calibrations.find({"a|b": {"temperature": "warm"}}, "a", "b") is None, "junk only"
    assert calibrations.find({}, "a", "b") is None
    assert calibrations.find(None, "a", "b") is None
    assert calibrations.find("nonsense", "a", "b") is None


def test_no_printer_or_no_filament_is_no_key_to_look_under():
    for printer, material in (("", "b"), ("a", ""), (None, None), (0, "b")):
        assert calibrations.find({"|": FOUND, "a|b": FOUND}, printer, material) is None


def test_one_calibration_is_one_line_of_json_like_the_spools():
    line = calibrations.to_json("pr", "ma", FOUND)
    assert json.loads(line)["printer"] == "pr"
    assert calibrations.from_json(line) == ("pr|ma", calibrations.clean(FOUND))


@pytest.mark.parametrize("line", ("", "junk", "[", "[]", "{}", None, 7, ["a"],
                                  json.dumps({"printer": "", "material": "m", "flow": 97.5}),
                                  json.dumps({"printer": "p", "material": "", "flow": 97.5}),
                                  json.dumps({"printer": "p", "material": "m"}),
                                  json.dumps({"printer": "p", "material": "m", "flow": "rich"})))
def test_a_line_that_is_no_calibration_is_left_out_of_the_settings(line):
    assert calibrations.from_json(line) is None


def test_the_whole_shelf_round_trips_through_the_lines_of_the_settings():
    kept = {"pr|ma": FOUND, "pr|mb": {"retraction": 0.8}}
    lines = calibrations.to_lines(kept)
    assert len(lines) == 2
    assert calibrations.from_lines(lines) == {"pr|ma": calibrations.clean(FOUND),
                                              "pr|mb": {"retraction": 0.8}}
    again = calibrations.from_lines(calibrations.to_lines(calibrations.from_lines(lines)))
    assert again == calibrations.from_lines(lines)


@pytest.mark.parametrize("junk", (None, "one line", 7, {"x": "y"}, [], ["junk"], [7],
                                  ["junk", 5], [{"printer": "p"}]))
def test_the_shelf_tolerates_everything_a_settings_file_can_hold(junk):
    for name, numbers in calibrations.from_lines(junk).items():
        assert calibrations.parts(name) and numbers, "an entry that is no calibration came back"


def test_junk_lines_and_entries_that_are_no_calibrations_leave_no_trace():
    assert calibrations.from_lines(["junk"]) == {}
    assert calibrations.to_lines("junk") == []
    assert calibrations.to_lines({"junk key": FOUND}) == []
    assert calibrations.to_lines({"a|b": {"temperature": "warm"}}) == []


def test_a_dict_comes_back_through_the_lines_so_every_field_is_checked_again():
    assert calibrations.from_lines({"a|b": {"flow": 97.5, "temperature": "warm"}}) == \
        {"a|b": {"flow": 97.5}}


@pytest.mark.parametrize("name, kind", (("Generic PETG", "PETG"), ("pla+", "PLA"),
                                        ("Generic ABS", "ABS"), ("ASA filaments", "ASA"),
                                        ("TPU 95A", "TPU"), ("Generic TPU", "TPU")))
def test_which_kind_of_filament_a_name_describes(name, kind):
    assert calibrations.material_type(name) == kind


@pytest.mark.parametrize("name", ("", None, "HIPS", "Nylon", "PC", 7))
def test_a_name_that_names_no_known_kind_takes_the_preference(name):
    """The filament of Window > Preferences > Filament, so the tower prints the range the user's
    ordinary filament is known for."""
    assert calibrations.material_type(name, "PETG") == "PETG"
    assert calibrations.material_type(name, "nonsense") == "PLA"
    assert calibrations.material_type(name, None) == "PLA"


def test_a_calibration_gives_the_settings_of_only_the_numbers_it_holds():
    assert calibrations.values(FOUND) == {"material_print_temperature": 205.0,
                                          "material_flow": 97.5, "retraction_amount": 0.8}
    assert calibrations.values({"advance": 0.035, "max_flow": 14.0}) == {}
    assert calibrations.values({"temperature": 205.0}) == {"material_print_temperature": 205.0}
    assert calibrations.values(None) == {}


def test_the_temperature_of_the_first_layer_is_not_one_of_them():
    """That one is about the bed, not about the filament, and the profile knows it best."""
    assert "material_print_temperature_layer_0" not in calibrations.values(FOUND)


def test_what_was_chosen_in_the_dialog_wins_over_the_calibration():
    made = calibrations.fill_in({"material_flow": 120.0}, FOUND)
    assert made == {"material_flow": 120.0, "material_print_temperature": 205.0,
                    "retraction_amount": 0.8}
    assert calibrations.fill_in({}, FOUND) == calibrations.values(FOUND)
    assert calibrations.fill_in({"infill_sparse_density": 20.0}, None) == \
        {"infill_sparse_density": 20.0}
    assert calibrations.fill_in(None, FOUND) == calibrations.values(FOUND)


@pytest.mark.parametrize("firmware, want", (("marlin", "M900 K0.035"),
                                            ("klipper", "SET_PRESSURE_ADVANCE ADVANCE=0.035"),
                                            ("rrf", "M572 D0 S0.035")))
def test_the_pressure_advance_rides_as_the_command_of_its_own_firmware(firmware, want):
    assert calibrations.advance_command(dict(FOUND, firmware=firmware)) == want


def test_no_pressure_advance_is_no_command():
    assert calibrations.advance_command(None) is None
    assert calibrations.advance_command({"temperature": 205.0}) is None
    assert calibrations.advance_command({"advance": 0.04, "firmware": "toaster"}) is None


def test_a_maximum_flow_lowers_every_speed_that_lays_too_much_plastic():
    current = {"speed_print": 200.0, "speed_infill": 200, "speed_wall": 100, "speed_wall_0": 90,
               "speed_wall_x": 180}
    assert calibrations.speed_cap(14.0, 0.2, 0.4, current) == {"speed_print": 175.0,
                                                               "speed_infill": 175.0,
                                                               "speed_wall_x": 175.0}


def test_a_speed_that_already_stays_under_the_flow_is_left_alone():
    """A calibration lowers a print; it never speeds it up."""
    assert calibrations.speed_cap(14.0, 0.2, 0.4, {"speed_print": 175.0, "speed_wall": 25.0}) == {}


def test_a_lowered_speed_is_rounded_down_so_that_the_cap_holds():
    assert calibrations.speed_cap(15.0, 0.28, 0.45, {"speed_print": 119.047}) == \
        {"speed_print": 119.0}


@pytest.mark.parametrize("flow, layer, wide", ((None, 0.2, 0.4), (14.0, None, 0.4),
                                               (14.0, 0.2, None), (0, 0.2, 0.4), (14.0, 0, 0.4),
                                               (14.0, 0.2, 0), ("much", 0.2, 0.4),
                                               (float("nan"), 0.2, 0.4)))
def test_without_one_of_the_three_nothing_is_lowered(flow, layer, wide):
    assert calibrations.speed_cap(flow, layer, wide, {"speed_print": 200.0}) == {}


def test_a_speed_of_no_number_is_left_to_the_profile():
    assert calibrations.speed_cap(2.0, 0.2, 0.4, {"speed_print": None, "speed_wall": "fast"}) == {}
    assert calibrations.speed_cap(2.0, 0.2, 0.4, None) == {}


def test_a_travel_speed_is_not_capped():
    """A travel extrudes nothing, so the melt limit has nothing to say about it."""
    lowered = calibrations.speed_cap(2.0, 0.2, 0.4, {"speed_travel": 400.0, "speed_print": 200.0})
    assert lowered == {"speed_print": 25.0}


def test_what_was_found_is_named_on_one_line():
    assert calibrations.text_of(FOUND) == ("205 °C, flow 97.5 %, retraction 0.8 mm, PA 0.035, "
                                           "max 14 mm³/s")
    assert calibrations.text_of({"temperature": 215.0}) == "215 °C"
    assert calibrations.text_of({"advance": 0.035, "firmware": "klipper"}) == "PA 0.035"
    assert calibrations.text_of({}) == ""
    assert calibrations.text_of(None) == ""


def test_a_number_is_written_as_it_is_read():
    assert calibrations.number_text(97.5) == "97.5"
    assert calibrations.number_text(100.0) == "100"
    assert calibrations.number_text(-1e-17) == "0"
    assert calibrations.number_text("warm") == "warm"


def test_what_a_flow_did_to_the_speeds_is_one_number():
    assert calibrations.lowered_text({"speed_print": 25.0, "speed_infill": 25.0}) == \
        "speed lowered to 25 mm/s"
    assert calibrations.lowered_text({}) == ""
    assert calibrations.lowered_text(None) == ""


def test_the_settings_take_only_usable_calibrations_out_of_the_file():
    q = qsettings()
    saved = q.value("calibrations")
    try:
        q.setValue("calibrations", ["junk",
                                    json.dumps({"printer": "p", "material": "m",
                                                "temperature": "warm"}),
                                    json.dumps({"printer": "p", "material": "m", "flow": 97.5}),
                                    json.dumps({"printer": "", "material": "m", "flow": 97.5})])
        q.sync()
        assert app_module.Settings().calibrations == {"p|m": {"flow": 97.5}}
    finally:
        q.remove("calibrations") if saved is None else q.setValue("calibrations", saved)
        q.sync()


def test_the_settings_write_the_calibrations_back_as_lines(win):
    made = found(win)
    win.settings.save()
    assert app_module.Settings().calibrations == {calibrations.key("testprinter",
                                                                  "generic_pla_175"): made}


# --------------------------------------------------------------------- the walk

@pytest.mark.parametrize("spec", calibration_wizard.TESTS, ids=lambda one: one.key)
def test_every_test_has_two_sentences_and_a_field_of_its_own(spec):
    """What the print is, and what to look for on it: one sentence each."""
    assert spec.title and spec.label
    assert spec.what.endswith(".") and spec.look.endswith(".")
    assert spec.unit == "" or spec.unit.startswith(" ")
    assert spec.step > 0 and 0 <= spec.decimals <= 4


def test_the_walk_asks_the_five_tests_in_their_order(win, lib):
    wiz = wizard(win, lib)
    titles = [wiz.page(i).title() for i in wiz.pageIds()]
    assert titles == ["Printer and filament", "Nozzle Temperature", "Material Flow",
                      "Pressure Advance", "Retraction", "Max Flow", "What you found"]
    assert list(wiz.pages) == ["temperature", "flow", "advance", "retraction", "max_flow"]
    assert all(isinstance(wiz.page(i), QWizardPage) for i in wiz.pageIds())
    assert wiz.summary.isFinalPage() is True
    assert wiz.windowTitle() == "Calibrate Filament"


def test_the_first_page_is_the_printer_and_the_filament_of_the_last_slice(win, lib):
    wiz = wizard(win, lib)
    assert wiz.choose.printer_id() == "testprinter"
    assert wiz.choose.material_id() == "generic_pla_175"
    assert wiz.key() == "testprinter|generic_pla_175"
    assert wiz.choose.isComplete() is True
    assert wiz.choose.printer_name() == "Test Printer"
    assert wiz.choose.material_name() == "Generic PLA"


def test_the_walk_cannot_start_without_a_printer_and_a_filament(win, lib):
    wiz = wizard(win, lib)
    wiz.choose.material.setCurrentIndex(-1)
    assert wiz.choose.isComplete() is False
    assert wiz.key() == ""


def test_every_page_names_its_test_and_tells_two_sentences_about_it(win, lib):
    wiz = wizard(win, lib)
    for spec in calibration_wizard.TESTS:
        page = wiz.pages[spec.key]
        assert page.title() == spec.title
        assert page.intro.text() == f"{spec.what} {spec.look}"
        assert page.button.text() == "Put the test on the bed"
        assert page.skip.text() == "Skip this test"
        assert page.note.text(), "the page says what its button puts on the bed"


def test_the_result_field_asks_for_the_number_of_its_own_test(win, lib):
    wiz = wizard(win, lib)
    for spec in calibration_wizard.TESTS:
        page = wiz.pages[spec.key]
        lo, hi = calibrations.BOUNDS[spec.key]
        assert (page.value.minimum(), page.value.maximum()) == (lo, hi)
        assert page.value.suffix() == spec.unit
        assert page.value.decimals() == spec.decimals
    assert wiz.pages["temperature"].value.suffix() == " °C"
    assert wiz.pages["max_flow"].value.suffix() == " mm³/s"


def test_a_page_starts_on_skip_so_nothing_counts_that_was_not_read_off_a_print(win, lib):
    wiz = wizard(win, lib)
    assert wiz.results() == {}
    for page in wiz.pages.values():
        assert page.skip.isChecked() is True
        assert page.value.isEnabled() is False, "a field that is not in use stays out of the way"


def test_a_number_only_counts_when_its_test_was_printed(win, lib):
    wiz = wizard(win, lib)
    fill(wiz.pages["temperature"], 205.0)
    assert wiz.results() == {"temperature": 205.0}
    wiz.pages["temperature"].skip.setChecked(True)
    assert wiz.results() == {}
    assert wiz.pages["temperature"].value.value() == 205.0, "the number stays for a second look"


def test_the_number_a_page_starts_with_suits_this_filament(win, lib):
    wiz = wizard(win, lib)
    assert wiz.pages["temperature"].value.value() == 220.0, "the PLA preset starts at 220 °C"
    assert wiz.pages["flow"].value.value() == 100.0
    assert wiz.pages["advance"].value.value() == 0.05
    assert wiz.pages["retraction"].value.value() == 1.75
    assert wiz.pages["max_flow"].value.value() == 12.5


def test_the_pressure_advance_page_asks_which_firmware_sets_it(win, lib):
    wiz = wizard(win, lib)
    page = wiz.pages["advance"]
    assert page.firmware is not None
    assert [page.firmware.itemData(i) for i in range(page.firmware.count())] == \
        list(pressure_advance_test.FIRMWARES)
    assert page.firmware.currentData() == "marlin", "the firmware of the last Pressure Advance Test"
    assert wiz.pages["flow"].firmware is None


def test_a_firmware_that_stands_in_the_settings_is_the_one_the_page_starts_with(win, lib):
    win.settings.pressure_advance_firmware = "rrf"
    wiz = wizard(win, lib)
    assert wiz.pages["advance"].firmware.currentData() == "rrf"
    assert "RepRapFirmware" in wiz.pages["advance"].note.text()


@pytest.mark.parametrize("firmware, note", (("marlin", "0 to 0.1 with Marlin, in steps of 0.005"),
                                            ("klipper", "0 to 0.1 with Klipper, in steps of 0.005"),
                                            ("rrf", "0 to 0.2 with RepRapFirmware, "
                                                   "in steps of 0.01")))
def test_another_firmware_another_range_to_walk_through(win, lib, firmware, note):
    """The preset of the extruder that belongs to that firmware (pressure_advance_test)."""
    win.settings.pressure_advance_firmware = firmware
    wiz = wizard(win, lib)
    assert wiz.pages["advance"].note.text() == f"The tower prints {note}."


def test_every_page_says_what_its_button_puts_on_the_bed(win, lib):
    wiz = wizard(win, lib)
    assert wiz.pages["temperature"].note.text() == \
        "The tower prints 220 to 190 °C in steps of 5."
    assert wiz.pages["flow"].note.text() == "The pads print 90 to 110 % in steps of 2.5."
    assert wiz.pages["retraction"].note.text() == \
        "The towers travel with 0.5 to 3 mm in steps of 0.5."
    assert wiz.pages["max_flow"].note.text() == "The tube prints 5 to 20 mm³/s in steps of 2.5."


def test_another_filament_another_temperature_range(win, lib):
    """The tower prints the range the maker of that material says it prints well over."""
    wiz = wizard(win, lib)
    assert "220 to 190 °C" in wiz.pages["temperature"].note.text()
    wiz.choose.material.setCurrentIndex(wiz.choose.material.findData("generic_petg_175"))
    assert wiz.filament_kind == "PETG"
    assert "250 to 220 °C" in wiz.pages["temperature"].note.text()
    assert wiz.pages["temperature"].value.value() == 250.0


def test_a_preference_is_enough_for_a_material_that_names_no_kind(win, lib):
    win.settings.filament_material = "PETG"
    wiz = wizard(win, lib)
    wiz.choose.material.setCurrentIndex(wiz.choose.material.findData("generic_pla_175"))
    assert wiz.filament_kind == "PLA", "the material itself names its kind, so the preference is by"
    wiz.pages["temperature"].skip.setChecked(True)     # untouched pages only


def test_a_button_of_a_page_that_is_skipped_still_builds_its_test(win, lib, monkeypatch):
    """Skip is about the number, not about the print: a user who only wants the tower on the bed can
    still have it, without the wizard writing anything down."""
    seen = []
    monkeypatch.setattr(win, "_apply_temperature_tower", lambda *args: seen.append(args))
    wiz = wizard(win, lib)
    assert wiz.pages["temperature"].skip.isChecked() is True
    wiz.pages["temperature"].button.click()
    assert seen and seen[0][0] == 220.0


def test_the_button_of_every_page_puts_its_own_test_on_the_bed(win, lib, monkeypatch):
    """Every page goes through the method of its own test, with a plan worked out for this filament -
    the very methods the five dialogs call, so the bed, the undo step and the slice behave the same
    as if the test had been put there by its own dialog."""
    seen = {}

    def spy(name):
        def take(*args):
            seen[name] = args
        return take

    for name in APPLY:
        monkeypatch.setattr(win, name, spy(name))
    wiz = wizard(win, lib)
    want = {"_apply_temperature_tower": (220.0, 190.0, 5.0, temperature_tower.steps(220.0, 190.0,
                                                                                   5.0)),
            "_apply_flow_rate_test": (90.0, 110.0, 2.5, flow_rate_test.flows(90.0, 110.0, 2.5)),
            "_apply_pressure_advance_test": ("marlin", 0.0, 0.1, 0.005,
                                             pressure_advance_test.steps(0.0, 0.1, 0.005)),
            "_apply_retraction_test": (0.5, 3.0, 0.5, retraction_test.bands(0.5, 3.0, 0.5)),
            "_apply_max_speed_test": (5.0, 20.0, 2.5, max_speed_test.steps(5.0, 20.0, 2.5))}
    for key, method in (("temperature", APPLY[0]), ("flow", APPLY[1]), ("advance", APPLY[2]),
                        ("retraction", APPLY[3]), ("max_flow", APPLY[4])):
        wiz.pages[key].button.click()
        assert seen[method] == want[method], f"{key} put something else on the bed"
    assert set(seen) == set(APPLY)


def test_the_pressure_advance_button_uses_the_firmware_that_stands_on_the_page(win, lib,
                                                                              monkeypatch):
    seen = {}
    monkeypatch.setattr(win, "_apply_pressure_advance_test",
                        lambda *args: seen.setdefault("args", args))
    wiz = wizard(win, lib)
    page = wiz.pages["advance"]
    page.firmware.setCurrentIndex(page.firmware.findData("klipper"))
    assert "Klipper" in page.note.text()
    page.button.click()
    assert seen["args"][0] == "klipper"


def test_a_button_really_builds_a_test_on_the_bed(win, lib):
    """One of the five for real: the same code path the Flow Rate Test dialog walks."""
    before = len(win.doc.groups)
    wiz = wizard(win, lib)
    wiz.pages["flow"].button.click()
    pads = len(flow_rate_test.flows(90.0, 110.0, 2.5))
    assert pump_until(lambda: len(win.doc.groups) == before + pads), win.hint.text()
    assert pump_until(lambda: "pad" in win.hint.text().lower()), win.hint.text()
    assert any(g.name == "Flow 90 %" for g in win.doc.groups)
    win.doc.undo()
    assert len(win.doc.groups) == before, "one undo does not take the pads away again"


def test_a_walk_that_found_nothing_saves_nothing(win, lib):
    wiz = wizard(win, lib)
    assert wiz.save() == {}
    assert win.settings.calibrations == {}
    assert win.hint.text() == "Nothing was found, so nothing will be saved."


def test_a_walk_that_skips_everything_leaves_what_was_kept_alone(win, lib):
    kept = found(win)
    wiz = wizard(win, lib)
    for page in wiz.pages.values():
        page.skip.setChecked(True)
    assert wiz.save() == {}
    assert win.settings.calibrations[calibrations.key("testprinter", "generic_pla_175")] == kept
    assert win.hint.text() == "Nothing was found, so nothing will be saved."
    assert app_module.Settings().calibrations == \
        {calibrations.key("testprinter", "generic_pla_175"): kept}


def test_a_walk_without_a_printer_or_a_filament_says_so(win, lib):
    wiz = wizard(win, lib)
    fill(wiz.pages["flow"], 97.5)
    wiz.choose.material.setCurrentIndex(-1)
    assert wiz.save() == {}
    assert win.hint.text() == "Choose the printer and the filament first."
    assert win.settings.calibrations == {}


def test_saving_keeps_the_numbers_under_this_printer_and_filament(win, lib):
    wiz = wizard(win, lib)
    for key, value in FOUND.items():
        if key != "firmware":
            fill(wiz.pages[key], value)
    advance = wiz.pages["advance"]
    advance.firmware.setCurrentIndex(advance.firmware.findData(FOUND["firmware"]))
    made = wiz.save()
    assert made == calibrations.clean(FOUND)
    key = calibrations.key("testprinter", "generic_pla_175")
    assert win.settings.calibrations == {key: made}
    assert app_module.Settings().calibrations[key] == made, "it is on disk for the next start"
    assert win.hint.text() == ("Calibration saved for Test Printer with Generic PLA: "
                               "205 °C, flow 97.5 %, retraction 0.8 mm, PA 0.035, max 14 mm³/s")


def test_a_second_walk_goes_on_where_the_first_one_left_off(win, lib):
    found(win)
    wiz = wizard(win, lib)
    assert wiz.choose.note.text() == \
        "Calibrated: 205 °C, flow 97.5 %, retraction 0.8 mm, PA 0.035, max 14 mm³/s"
    assert wiz.choose.note.isVisibleTo(wiz.choose)
    for key, value in FOUND.items():
        if key == "firmware":
            continue
        page = wiz.pages[key]
        assert page.skip.isChecked() is False, f"{key} starts on with its own number"
        assert page.value.isEnabled() is True
    assert wiz.pages["advance"].value.value() == 0.035
    assert wiz.pages["advance"].firmware.currentData() == "klipper"
    assert wiz.results() == calibrations.clean(FOUND)


def test_filling_in_a_second_number_keeps_the_first_one(win, lib):
    win.settings.calibrations[calibrations.key("testprinter", "generic_pla_175")] = \
        {"temperature": 205.0}
    wiz = wizard(win, lib)
    assert wiz.pages["temperature"].skip.isChecked() is False
    assert wiz.pages["flow"].skip.isChecked() is True, "a test without a number is still a skip"
    fill(wiz.pages["flow"], 97.5)
    assert wiz.save() == {"temperature": 205.0, "flow": 97.5}


def test_junk_in_the_settings_is_nothing_to_start_from(win, lib):
    win.settings.calibrations[calibrations.key("testprinter", "generic_pla_175")] = \
        {"temperature": "warm", "flow": 97.5}
    wiz = wizard(win, lib)
    assert wiz.pages["temperature"].skip.isChecked() is True
    assert wiz.pages["flow"].skip.isChecked() is False
    assert wiz.results() == {"flow": 97.5}


def test_another_printer_starts_empty(win, lib):
    """The numbers found for one printer say nothing about the next; the filament id is even the
    same here, so only the printer keeps them apart."""
    kept = found(win)
    wiz = wizard(win, lib)
    assert wiz.choose.note.text()
    wiz.choose.printer_box.setCurrentIndex(wiz.choose.printer_box.findData("dual"))
    assert wiz.choose.printer_id() == "dual"
    assert wiz.key() == "dual|generic_pla_175"
    assert wiz.choose.note.text() == ""
    assert wiz.choose.note.isVisibleTo(wiz.choose) is False
    assert wiz.results() == {}
    assert all(page.skip.isChecked() for page in wiz.pages.values())
    assert win.settings.calibrations == {calibrations.key("testprinter", "generic_pla_175"): kept}


def test_numbers_kept_for_another_printer_are_the_right_ones_to_find(win, lib):
    kept = found(win, printer="dual")
    wiz = wizard(win, lib)
    assert wiz.found() is None
    wiz.choose.printer_box.setCurrentIndex(wiz.choose.printer_box.findData("dual"))
    assert wiz.found() == kept
    assert wiz.results() == kept


def test_a_number_that_is_not_a_number_in_a_page_is_no_result(win, lib):
    """A field always holds a number, but a page on Skip is dropped before that: an untouched walk is
    empty whatever the fields hold."""
    wiz = wizard(win, lib)
    for page in wiz.pages.values():
        page.value.setValue(page.value.minimum())
    assert wiz.results() == {}


def test_the_summary_lists_the_numbers_and_the_skips(win, lib):
    wiz = wizard(win, lib)
    fill(wiz.pages["temperature"], 205.0)
    fill(wiz.pages["advance"], 0.035)
    wiz.pages["advance"].firmware.setCurrentIndex(wiz.pages["advance"].firmware.findData("rrf"))
    wiz.summary.initializePage()
    lines = wiz.summary.found.text().splitlines()
    assert lines[0] == "Best temperature: 205 °C"
    assert lines[1] == "Best value: 0.035 (RepRapFirmware)"
    assert "Skipped: Material Flow, Retraction, Max Flow" in lines[-1]
    assert "File > Slice prints with these numbers" in wiz.summary.where.text()


def test_a_summary_without_a_single_number_says_so(win, lib):
    wiz = wizard(win, lib)
    assert wiz.findings() == "Nothing was found, so nothing will be saved."
    wiz.summary.initializePage()
    assert wiz.summary.found.text() == "Nothing was found, so nothing will be saved."
    assert "Skipped" not in wiz.findings()


# ---------------------------------------------------------- File > Slice with it

def dialog(win, lib):
    return sd.SliceDialog(win, lib, colors=(), parts=1)


def test_slice_shows_what_was_found_and_prints_with_it(win, lib):
    found(win)
    dlg = dialog(win, lib)
    assert dlg.calibration_line.text() == \
        "Calibrated: 205 °C, flow 97.5 %, retraction 0.8 mm, PA 0.035, max 14 mm³/s"
    assert dlg.calibration_line.isVisibleTo(dlg)
    assert dlg.use_calibration.isChecked() is True
    assert dlg.calibration() == calibrations.clean(FOUND)
    values = dlg.user_values()
    assert values["material_print_temperature"] == 205.0
    assert values["material_flow"] == 97.5
    assert values["retraction_amount"] == 0.8
    assert values["infill_sparse_density"] == 20.0, "the rest of the choice stays as it was"
    assert "material_print_temperature_layer_0" not in values


def test_the_temperature_and_the_flow_reach_the_stacks(win, lib):
    found(win)
    dlg = dialog(win, lib)
    st = dlg.stacks()
    assert st.glob.layers[0]["material_print_temperature"] == 205.0
    assert st.glob.layers[0]["material_flow"] == 97.5
    assert st.glob.get("material_print_temperature") == 205.0, "the engine reads the worked-out value"
    assert "material_print_temperature_layer_0" not in st.glob.layers[0]


def test_unticking_the_calibration_prints_this_one_without_it(win, lib):
    found(win)
    dlg = dialog(win, lib)
    dlg.use_calibration.setChecked(False)
    assert dlg.calibration_line.text(), "the line stays: the numbers are still kept"
    assert dlg.calibration() is None
    values = dlg.user_values()
    assert not set(calibrations.values(FOUND)) & set(values), "nothing of it is asked of the profile"
    assert values["infill_sparse_density"] == 20.0, "the dialog still says its own piece"
    assert dlg.calibration_command() is None
    assert "material_print_temperature" not in dlg.stacks().glob.layers[0]


def test_the_tick_is_remembered_for_the_next_slice(win, lib):
    found(win)
    dlg = dialog(win, lib)
    dlg.use_calibration.setChecked(False)
    assert dlg.choice()["calibration"] is False
    assert dialog(win, lib).use_calibration.isChecked() is True, "the window has not saved yet"
    for key, value in dlg.choice().items():
        setattr(win.settings, "slice_" + key, value)
    win.settings.save()
    assert app_module.Settings().slice_calibration is False
    assert dialog(win, lib).use_calibration.isChecked() is False


def test_a_preset_never_carries_the_tick(win, lib):
    """A preset is a printer and how to print with it; whether to use what was found is about the
    filament, and a preset that would carry it overrules what the user ticks."""
    dlg = dialog(win, lib)
    kept = slice_presets.from_json(slice_presets.to_json(dict(dlg.choice(), name="Mijn")))
    assert "calibration" not in kept
    assert "calibration" not in slice_presets.PRESET_KEYS
    assert slice_presets.matches(kept, dlg.choice())


def test_a_filament_that_was_never_calibrated_shows_nothing_at_all(win, lib):
    dlg = dialog(win, lib)
    assert dlg.calibration_line.text() == ""
    assert dlg.calibration_line.isVisibleTo(dlg) is False
    assert dlg.use_calibration.isVisibleTo(dlg) is False
    assert dlg.calibration() is None
    assert dlg.user_values()["infill_sparse_density"] == 20.0, "the dialog still says its own piece"


def test_the_line_says_what_a_maximum_flow_did_to_the_speeds(win, lib):
    found(win, max_flow=2.0)
    dlg = dialog(win, lib)
    st = dlg.stacks()
    assert st.glob.get("speed_print") == 25.0, "2 mm³/s through 0.2 by 0.4 mm is 25 mm/s"
    assert st.glob.get("speed_wall") == 12.5, "a wall that follows the print speed follows it down"
    assert "speed lowered to 25 mm/s" in dlg.calibration_line.text()


def test_a_flow_the_profile_already_stays_under_lowers_nothing(win, lib):
    found(win, max_flow=14.0)
    dlg = dialog(win, lib)
    st = dlg.stacks()
    assert st.glob.get("speed_print") == 50.0, "the printer definition's own speed"
    assert st.extruders[0].get("speed_print") == 40.0, "the quality's own speed"
    assert dlg._capped == {}
    assert "speed lowered" not in dlg.calibration_line.text()


def test_a_cap_never_speeds_a_print_up(win, lib):
    found(win, max_flow=calibrations.MELT_MAX)
    dlg = dialog(win, lib)
    assert dlg.stacks().glob.get("speed_print") == 50.0
    assert dlg._capped == {}


def test_unticked_the_speeds_stay_what_the_profile_has(win, lib):
    found(win, max_flow=2.0)
    dlg = dialog(win, lib)
    dlg.use_calibration.setChecked(False)
    assert dlg.stacks().glob.get("speed_print") == 50.0
    assert "speed lowered" not in dlg.calibration_line.text()


def test_the_cap_follows_the_layer_and_line_width_of_the_choice(win, lib):
    """The same flow is worth another speed on thick layers than on thin ones."""
    found(win, max_flow=2.0)
    dlg = dialog(win, lib)
    st = dlg.stacks()
    assert st.extruders[0].get("line_width") == pytest.approx(0.4)
    assert st.extruders[0].get("speed_print") == 25.0, "0.2 mm layers of 0.4 mm wide"
    assert "speed lowered to 25 mm/s" in dlg.calibration_line.text()
    for nozzle in range(dlg.nozzle.count()):
        if dlg.nozzle.itemText(nozzle).startswith("0.6"):
            dlg.nozzle.setCurrentIndex(nozzle)
    assert dlg.nozzle.currentText().startswith("0.6")
    st = dlg.stacks()
    assert st.extruders[0].get("line_width") == pytest.approx(0.6)
    assert st.extruders[0].get("speed_print") == pytest.approx(16.6)
    assert dlg._capped["speed_print"] == pytest.approx(16.6)
    assert "speed lowered to 16.6 mm/s" in dlg.calibration_line.text()


def test_the_pressure_advance_of_a_calibration_is_one_command_for_the_first_layer(win, lib):
    found(win)
    dlg = dialog(win, lib)
    assert dlg.calibration_command() == (0.0, "SET_PRESSURE_ADVANCE ADVANCE=0.035")


def test_a_calibration_without_a_pressure_advance_is_no_command(win, lib):
    found(win, advance=None, firmware=None)
    dlg = dialog(win, lib)
    assert dlg.calibration_command() is None
    assert dlg.calibration_line.text() == \
        "Calibrated: 205 °C, flow 97.5 %, retraction 0.8 mm, max 14 mm³/s"


def test_another_filament_another_calibration(win, lib):
    found(win)
    dlg = dialog(win, lib)
    dlg.material.setCurrentIndex(dlg.material.findData("generic_petg_175"))
    assert dlg.calibration_line.text() == ""
    assert dlg.calibration() is None
    found(win, material="generic_petg_175", temperature=240.0)
    again = dialog(win, lib)
    again.material.setCurrentIndex(again.material.findData("generic_petg_175"))
    assert again.calibration()["temperature"] == 240.0
    assert "240 °C" in again.calibration_line.text()


# ---------------------------------------------------- the window around the walk

def test_the_action_is_the_last_of_the_3d_printing_menu(win):
    labels = [a.text().replace("&", "").replace("…", "").strip()
              for a in win.flat_actions(win.print_tools_menu)]
    assert labels[-1] == "Calibrate Filament"
    assert labels[-2] == "Retraction Test", "right after the five tests it walks"


def test_the_command_walks_the_wizard_and_saves_only_an_accepted_walk(win, monkeypatch):
    """The profiles come in through the background task, so the window keeps answering while the
    first read takes its minutes (volura/background.py)."""
    walked = []

    class Fake:
        answer = QDialog.Accepted

        def __init__(self, parent, profiles):
            self.profiles, self.saved = profiles, False
            walked.append(self)

        def exec(self):
            return Fake.answer

        def save(self):
            self.saved = True

    monkeypatch.setattr(calibration_wizard, "CalibrationWizard", Fake)
    monkeypatch.setattr(window_print.slicer, "find_resources", lambda: "resources")
    monkeypatch.setattr(sd, "is_loaded", lambda resources: True)
    monkeypatch.setattr(sd, "library", lambda resources: "the profiles")
    win.calibrate_filament_dialog()
    assert pump_until(lambda: walked), "the wizard was never walked"
    assert pump_until(lambda: walked[0].saved), "an accepted walk was not saved"
    assert walked[0].profiles == "the profiles"
    Fake.answer = QDialog.Rejected
    win.calibrate_filament_dialog()
    assert pump_until(lambda: len(walked) == 2)
    assert walked[1].saved is False, "a walk that was cancelled has nothing to say"


def test_the_command_needs_the_printers_profiles_but_not_the_engine(win, monkeypatch):
    """Without the Cura profiles there is no printer to calibrate for, and the wizard says so."""
    box = FakeBox()
    monkeypatch.setattr(window_print, "QMessageBox", box)
    monkeypatch.setattr(window_print.slicer, "find_resources", lambda: None)
    monkeypatch.setattr(window_print.slicer, "find_engine", lambda: None)
    win.calibrate_filament_dialog()
    assert box.asked, "nothing was said about the missing profiles"
    assert "releases" in box.asked[0]


def test_the_wizard_reports_a_failure_of_its_own(win, monkeypatch):
    box = FakeBox()
    monkeypatch.setattr(window_print, "QMessageBox", box)
    monkeypatch.setattr(window_print.slicer, "find_resources", lambda: "resources")
    monkeypatch.setattr(sd, "is_loaded", lambda resources: True)

    def boom(report):
        raise OSError("no profiles here")

    monkeypatch.setattr(sd, "library", lambda resources: boom(None))
    win.calibrate_filament_dialog()
    assert pump_until(lambda: box.asked), win.hint.text()
    assert "no profiles here" in box.asked[0]


def test_a_calibration_rides_into_the_slice_as_a_command_of_its_own(win):
    made = win._bed_height_commands(only=False, stacks=None, calibration=(0.0, "M900 K0.04"))
    assert made == (None, [(0.0, "M900 K0.04")])


def test_a_pressure_advance_test_on_the_bed_keeps_its_own_values(win):
    """At one and the same height the last line in the G-code is the one the printer ends up with, so
    the values a Pressure Advance Test walks through come after the one the calibration set."""
    win._apply_pressure_advance_test("marlin", 0.0, 0.03, 0.01,
                                    pressure_advance_test.steps(0.0, 0.03, 0.01))
    assert pump_until(lambda: len(win.doc.groups) == 1)
    _temp, changes = win._bed_height_commands(only=False, stacks=None,
                                             calibration=(0.0, "M900 K0.02"))
    assert [line for _z, line in changes][:2] == ["M900 K0.02", "M900 K0"]
    assert changes == sorted(changes, key=lambda one: one[0])


def test_a_temperature_tower_on_the_bed_keeps_its_temperatures(win):
    """The tower says what the nozzle heats to at its own heights; a calibrated temperature only
    fills in the print itself, which is what the slice reads out of the stacks."""
    win._apply_temperature_tower(220.0, 190.0, 5.0, temperature_tower.steps(220.0, 190.0, 5.0))
    assert pump_until(lambda: len(win.doc.groups) == 1)
    temp, changes = win._bed_height_commands(only=False, stacks=None,
                                            calibration=(0.0, "M900 K0.02"))
    assert temp == 220
    assert changes[0] == (0.0, "M900 K0.02"), "the pressure advance still comes first"
    assert any(line.startswith("M104 S190") for _z, line in changes)


def test_no_calibration_and_no_test_on_the_bed_is_nothing_to_ask(win):
    assert win._bed_height_commands(only=False, stacks=None) is None
    assert win._bed_height_commands(only=False, stacks=None, calibration=None) is None


@pytest.mark.parametrize("junk", ((), "M900 K0.04", (0.0,), ("warm", "M900 K0.04"),
                                  {"niet": "dit"}, [0, 1, 2], None))
def test_a_calibration_that_is_no_height_and_a_command_is_ignored(win, junk):
    assert win._bed_height_commands(only=False, stacks=None, calibration=junk) is None


# ------------------------------------------------------------- with CuraEngine

def _cube(size=20.0, at=(40.0, 40.0, 0.0)):
    mesh = mf.Manifold.cube((size, size, size)).translate(at).to_mesh()
    return (np.asarray(mesh.vert_properties)[:, :3], np.asarray(mesh.tri_verts))


def _what_the_gcode_lays(text) -> dict:
    """The temperature lines and the median plastic per millimetre of the moves inside the layers."""
    temps, ratios = [], []
    inside = e = px = py = None
    relative = False
    for raw in text.splitlines():
        words = raw.split(";", 1)[0].split()
        if raw.startswith(";LAYER:"):
            inside = True
            continue
        if raw.startswith(";TIME_ELAPSED:"):
            inside = False
            continue
        if not words:
            continue
        head = words[0].upper()
        if head in ("M104", "M109"):
            temps.append(raw.strip())
        elif head in ("M82", "M83"):
            relative, e = head == "M83", None
        elif head == "G92":
            e = 0.0
        elif head in ("G0", "G1", "G00", "G01"):
            x = y = value = None
            for word in words[1:]:
                letter, digits = word[0].upper(), word[1:]
                try:
                    number = float(digits)
                except ValueError:
                    continue
                if letter == "X":
                    x = number
                elif letter == "Y":
                    y = number
                elif letter == "E":
                    value = number
            before = (px, py)
            if x is not None:
                px = x
            if y is not None:
                py = y
            if value is None:
                continue
            delta = value if relative else (None if e is None else value - e)
            e = value
            step = 0.0
            if before[0] is not None and before[1] is not None and px is not None and py is not None:
                step = ((px - before[0]) ** 2 + (py - before[1]) ** 2) ** 0.5
            if inside and delta is not None and delta > 1e-9 and step > 1e-9:
                ratios.append(delta / step)
    return {"temps": temps, "e_per_mm": statistics.median(ratios) if ratios else 0.0}


class EngineSettings:
    """`subprocess` with its `Popen` wrapped, so the test can read the settings CuraEngine really got."""

    def __init__(self, docs):
        self.docs = docs

    def Popen(self, cmd, *args, **kwargs):
        with open(cmd[cmd.index("-r") + 1], encoding="utf-8") as fh:
            self.docs.append(json.load(fh))
        return subprocess.Popen(cmd, *args, **kwargs)

    def __getattr__(self, name):
        return getattr(subprocess, name)


@needs_engine
def test_the_calibrated_numbers_reach_the_slice(tmp_path, monkeypatch):
    """One cube on the real Ender-3 profiles, sliced twice: once as the profiles have it and once with
    the numbers a calibration would hand over. The temperature and the flow say so in the G-code, the
    retraction length is written in the settings CuraEngine was given, and the pressure advance line
    of the firmware lands in the first layer."""
    docs = []
    monkeypatch.setattr(slicer, "subprocess", EngineSettings(docs))
    lib = sd.library(RESOURCES)
    p = lib.printer("creality_ender3")
    v = p.default_variant()
    m = p.default_material(v)
    q = p.default_quality(v, m)

    def slice_with(user):
        out = tmp_path / ("cal.gcode" if user else "plain.gcode")
        st = p.stacks(v, m, q, dict({"infill_sparse_density": 0.0}, **user))
        made = slicer.slice_meshes(st, [_cube() + (0,)], str(out))
        assert made.layers > 10, made
        return out.read_text(encoding="utf-8")

    plain = slice_with({})
    cal = slice_with({"material_print_temperature": 205.0, "material_flow": 95.0,
                      "retraction_amount": 1.0})
    command = calibrations.advance_command({"advance": 0.035, "firmware": "marlin"})
    assert command == "M900 K0.035"
    cal = slicer.with_commands(cal, [(0.0, command)])

    first, second = docs[0]["global"], docs[1]["global"]
    assert first["material_flow"] == pytest.approx(100.0)
    assert second["material_print_temperature"] == pytest.approx(205.0)
    assert second["material_flow"] == pytest.approx(95.0)
    assert second["retraction_amount"] == pytest.approx(1.0)

    assert "M109 S205" in cal and "M109 S205" not in plain, "the calibrated temperature"
    assert command not in plain and cal.count(command) == 1, "the pressure advance line"
    first_layer = cal.split(";LAYER:1", 1)[0]
    assert ";LAYER:0" in first_layer and command in first_layer[first_layer.index(";LAYER:0"):], \
        "the line is not in the first layer"

    a, b = _what_the_gcode_lays(plain), _what_the_gcode_lays(cal)
    assert a["e_per_mm"] > 0 and b["e_per_mm"] > 0
    assert 0.93 < b["e_per_mm"] / a["e_per_mm"] < 0.97, "95 % of the plastic per millimetre"

"""The filament spool tracker: volura/spools.py, Window > Spools… and the spools of the print.

A spool knows what a full one holds and what is still on it, weighed net of the empty spool.
The dialog keeps that list honest with a scale, and the print preview points every filament of
a print at a spool, says beforehand whether it holds enough and takes its grams off it when the
print really starts - once, never twice, and nothing at all when only the G-code was saved.

Run with:  python -m pytest tests/test_spools.py
"""
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest  # noqa: E402
from PySide6.QtWidgets import QApplication, QDialog, QLabel  # noqa: E402

from volura import gcode_preview as gp, printer_link, spools, theme  # noqa: E402
from volura.settings import qsettings  # noqa: E402

_app = QApplication.instance() or QApplication(sys.argv[:1])

#: a print that changes filament three times: T0, T1 and a manual change after that
MULTI = """\
;FLAVOR:Marlin
;LAYER:0
G0 F6000 X10 Y10 Z0.2
T0
;TYPE:WALL-OUTER
G1 F1200 X20 Y10 E1.0
G1 X20 Y20 E2.0
;LAYER:1
G0 F6000 X10 Y10 Z0.4
T1
;TYPE:WALL-OUTER
G1 F1200 X20 Y10 E3.0
;LAYER:2
G0 F6000 X10 Y10 Z0.6
M600
;TYPE:FILL
G1 F1200 X20 Y10 E4.0
G1 X20 Y20 E5.0
;TIME_ELAPSED:30.0
"""

#: a print that never changes filament
SINGLE = """\
;FLAVOR:Marlin
;LAYER:0
G0 F6000 X10 Y10 Z0.2
;TYPE:WALL-OUTER
G1 F1200 X20 Y10 E1.0
G1 X20 Y20 E2.0
;TIME_ELAPSED:6.0
"""

PLA_RED = spools.Spool("pla1", "Red PLA", "PLA", "#d23b3b", 1000.0, 900.0)


@pytest.fixture(scope="module")
def win():
    """The window the dialogs below sit in; its shelf is empty again once they all have been."""
    from volura.app import MainWindow
    w = MainWindow()
    yield w
    w.settings.spools = []
    w.settings.spool_for_slot = []
    w.settings.printers = []
    w.settings.save()
    w.doc.dirty = False
    w.close()


def shelf(win, *records, slots=()):
    """Put the spools on the shelf of the settings and say which filament slot came off what."""
    win.settings.spools = [spools.to_dict(spool) for spool in records]
    win.settings.spool_for_slot = list(slots)
    win.settings.save()
    return win.settings.spools


def left_of(records):
    """The grams every spool on the shelf still holds, by its id."""
    return {record["id"]: record["left_g"] for record in records}


def preview(win, tmp_path, filaments, gcode=MULTI):
    """The print preview of a small slice, with the filaments the slice says it prints with."""
    from volura import print_preview as pp
    path = tmp_path / "box.gcode"
    path.write_text(gcode, encoding="utf-8")
    return pp.PrintPreviewDialog(win, str(path), gp.parse(gcode.splitlines()),
                                 (0.0, 0.0, 100.0, 100.0), "", None, filaments)


def press(dialog):
    """Stands in for a user who presses Send and starts printing right away."""
    from PySide6.QtWidgets import QDialog
    dialog.start.setChecked(True)
    dialog.accept()
    return QDialog.Accepted


class Answer:
    """Stands in for QMessageBox: answers the question about a spool without a dialog."""

    Yes, No = 1, 0

    def __init__(self, answer=1):
        self.answer = answer
        self.asked = []

    def question(self, *args, **kwargs):
        self.asked.append(args)
        return self.answer


class Scale:
    """Stands in for QInputDialog: what the scale says, or nobody who cancelled the weighing."""

    def __init__(self, grams=None):
        self.grams = grams
        self.asked = []

    def getDouble(self, *args, **kwargs):
        self.asked.append(args)
        return (self.grams, True) if self.grams is not None else (0.0, False)


# ------------------------------------------------------------------ a spool on its own

def test_a_spool_comes_back_out_of_the_settings_as_it_went_in():
    spool = spools.Spool("abc123", "Galaxy black PETG", "PETG", "#1234AB", 750.0, 300.0,
                         21.0, 230.0)
    assert spools.from_dict(spools.to_dict(spool)) == spool
    assert spools.from_lines(spools.to_lines([spools.to_dict(spool)])) == [spools.to_dict(spool)]
    assert spools.find([spools.to_dict(spool)], "abc123") == spool
    assert spools.to_dict(spool)["left_g"] == 300.0


def test_a_spool_with_junk_in_it_falls_back_to_what_is_known():
    spool = spools.from_dict({"id": None, "name": 5, "material": "   ", "colour": "green",
                              "full_g": "a lot", "left_g": -5, "price": None, "tare_g": [1],
                              "where_it_was_bought": "the fair"})
    assert spool.id and spool.name == "5" and spool.material == spools.OTHER
    assert spool.colour == spools.DEFAULT_COLOUR
    assert spool.full_g == spools.FULL_G and spool.left_g == 0.0
    assert spool.price == 0.0 and spool.tare_g == spools.TARE_G
    assert "where_it_was_bought" not in spools.to_dict(spool)
    # a new spool is full, a short colour is taken with every digit twice and a spool that
    # holds more than is left on it is clamped to what it holds
    assert spools.Spool("x").left_g == spools.FULL_G
    assert spools.from_dict({"id": "x", "colour": "#abc"}).colour == "#aabbcc"
    assert spools.from_dict({"id": "x", "full_g": 100, "left_g": 900}).left_g == 100.0
    assert spools.from_dict("not a spool").name == ""
    assert spools.from_dict({"id": "nan", "full_g": "nan"}) == spools.from_dict({"id": "nan"})


def test_weights_that_are_extreme_stay_within_what_a_spool_can_be():
    thin = spools.Spool("thin", "A thread", "PLA", "#ffffff", 0.0)
    assert thin.full_g == spools.FULL_MIN and thin.left_g == spools.FULL_MIN
    assert spools.left_text(thin) == "1 g of 1 g left (100 %)", "nothing is divided by zero"
    assert spools.enough(thin, 0.0) and not spools.enough(thin, 1.2)
    assert spools.take(thin, 1e9).left_g == 0.0 and spools.take(thin, "junk").left_g == 1.0
    big = spools.from_dict({"id": "big", "full_g": 1e12, "left_g": 1e12, "tare_g": -5.0,
                            "price": 1e9})
    assert big.full_g == spools.FULL_MAX and big.left_g == spools.FULL_MAX
    assert big.tare_g == spools.TARE_MIN and big.price == spools.PRICE_MAX
    assert spools.left_text(big) == "100000 g of 100000 g left (100 %)"


def test_the_settings_list_of_spools_survives_every_kind_of_junk():
    good = spools.to_dict(PLA_RED)
    assert spools.from_lines([good]) == [good]
    assert spools.from_lines(["nonsense", None, "", 7, "{}", '{"id": ""}']) == []
    assert spools.from_lines(json_line(PLA_RED)) == [good]
    assert spools.from_lines("nonsense") == []
    assert spools.to_lines(["junk", 3, None]) == []
    assert spools.records([good, PLA_RED, "junk", None]) == [PLA_RED, PLA_RED]


def json_line(spool):
    """One spool as the line of JSON the settings keep it in."""
    return spools.to_lines([spool])[0]


def test_taking_leaves_the_spool_lighter_and_never_empty_in_the_red():
    spool = spools.Spool("s1", left_g=20.0)
    assert spools.take(spool, 5.0).left_g == 15.0
    assert spools.take(spool, 20.0).left_g == 0.0
    assert spools.take(spool, 900.0).left_g == 0.0
    assert spools.take(spool, -5.0).left_g == 20.0
    assert spools.take(spool, "a spool of grams").left_g == 20.0
    assert spool.left_g == 20.0, "the spool itself is not touched by taking"
    assert spools.take(spool, 5.0).id == spool.id, "a taken spool is the same spool"


def test_enough_asks_for_the_margin_on_top_of_what_a_print_needs():
    full = spools.Spool("s1", left_g=105.0)
    exact = spools.Spool("s2", left_g=100.0)
    assert spools.enough(full, 100.0), "5 % more than needed is the usual spare"
    assert not spools.enough(exact, 100.0), "a spool without spare stops halfway"
    assert spools.enough(exact, 100.0, margin=0.0)
    assert spools.enough(exact, 100.0, margin=-0.1)
    assert not spools.enough(spools.Spool("s3", left_g=104.9), 100.0)
    assert spools.enough(full, "no idea"), "a print that names no grams needs nothing"
    assert spools.enough(spools.Spool("s4", left_g=0.0), 0.0)


def test_a_spool_says_what_is_left_on_it():
    assert spools.left_text(spools.Spool("s", left_g=300.0)) == "300 g of 1000 g left (30 %)"
    assert spools.left_text(spools.Spool("s", full_g=750, left_g=20)) == \
        "20 g of 750 g left (3 %)"
    assert spools.left_text(spools.Spool("s", left_g=333.33)) == \
        "333.3 g of 1000 g left (33 %)"
    assert spools.grams_text("junk") == "0"


def test_the_scale_reading_becomes_what_is_left_on_the_spool():
    spool = spools.Spool("s", "Blue PLA", "PLA", "#223344", 1000.0, 800.0, 0.0, 250.0)
    assert spools.set_left(spool, 900.0).left_g == 650.0
    assert spools.set_left(spool, 250.0).left_g == 0.0, "the empty spool alone is nothing"
    assert spools.set_left(spool, 20.0).left_g == 0.0, "less than the core alone is nothing"
    assert spools.set_left(spool, "the cat sat on it").left_g == 800.0, "nothing was weighed"
    assert spools.set_left(spool, 3000.0).left_g == 1000.0, "never more than a full spool"


def test_a_spool_of_the_right_material_comes_first_for_a_filament():
    petg = spools.Spool("p", "Clear PETG", "PETG")
    pla = spools.Spool("l", "Black PLA", "PLA")
    other = spools.Spool("o", "Mystery roll")
    assert spools.matches(petg, "T0 PETG clear") and not spools.matches(petg, "T1 PLA black")
    assert not spools.matches(other, "T0 PETG clear"), "Other says nothing about a filament"
    assert spools.matches(pla, "") is False and spools.matches(pla, None) is False
    assert spools.find([petg, pla], "") is None
    assert spools.find([petg, pla], "gone") is None


def test_a_slot_remembers_its_spool_and_forgets_one_that_is_gone():
    assert spools.slot_ids(["pla1", "", "petg1"]) == ["pla1", "", "petg1"]
    assert spools.slot_ids([]) == [] and spools.slot_ids(None) == []
    assert spools.slot_ids("pla1") == ["pla1"], "one slot is stored as one plain text"
    assert spools.slot_ids("") == [] and spools.slot_ids("   ") == []
    assert spools.slot_ids([" petg1 ", None, 7]) == ["petg1", "", "7"]
    assert spools.slot_ids(()) == [] and spools.slot_ids(0) == []
    gone = spools.slot_ids("gone")[0]
    assert spools.find([PLA_RED], gone) is None, "a spool off the shelf is no spool"


# ------------------------------------------------------------------ in the settings

def test_the_settings_remember_the_shelf_and_which_slot_came_off_what(win):
    from volura.app import Settings
    petg = spools.Spool("aaa111", "Galaxy black PETG", "PETG", "#101010", 750.0, 400.0, 21.0)
    shelf(win, petg, PLA_RED, slots=["aaa111", ""])
    back = Settings()
    assert back.spools == [spools.to_dict(petg), spools.to_dict(PLA_RED)]
    assert back.spool_for_slot == ["aaa111", ""]
    # what the settings hold for junk is no shelf, and a slot of no spool
    qsettings().setValue("spools", "nonsense")
    qsettings().setValue("spool_for_slot", "nonsense")
    qsettings().sync()
    junked = Settings()
    assert junked.spools == []
    assert spools.find(junked.spools, junked.spool_for_slot[0]) is None, "nothing to come back to"
    qsettings().remove("spools")
    qsettings().remove("spool_for_slot")
    qsettings().sync()
    assert Settings().spools == []


# ------------------------------------------------------------------ Window > Spools…

def test_the_spools_dialog_adds_edits_and_removes(win, monkeypatch):
    from volura import spools_dialog as sd
    shelf(win)

    def fill(dlg):
        dlg.name.setText("Galaxy black PETG")
        dlg.material.setCurrentIndex(dlg.material.findData("PETG"))
        dlg.full.setValue(750)
        dlg.left.setValue(700)
        dlg.tare.setValue(230)
        return QDialog.Accepted

    monkeypatch.setattr(sd.SpoolEditDialog, "exec", fill)
    dlg = sd.SpoolsDialog(win)
    try:
        assert dlg.table.rowCount() == 0 and not dlg.b_edit.isEnabled()
        dlg._add_spool()
        kept = win.settings.spools[0]
        assert dlg.table.rowCount() == 1 and dlg.table.currentRow() == 0
        assert (kept["name"], kept["material"]) == ("Galaxy black PETG", "PETG")
        assert (kept["full_g"], kept["left_g"], kept["tare_g"]) == (750.0, 700.0, 230.0)
        assert dlg.table.item(0, 3).text() == "700 g of 750 g left (93 %)"
        assert dlg.table.item(0, 0).icon().isNull() is False, "the row has no colour swatch"

        def change(edit):
            assert edit.name.text() == "Galaxy black PETG" and edit.tare.value() == 230
            assert edit.material.currentData() == "PETG"
            edit.name.setText("Galaxy black on the spare reel")
            edit.left.setValue(360)
            return QDialog.Accepted

        monkeypatch.setattr(sd.SpoolEditDialog, "exec", change)
        dlg.table.selectRow(0)
        dlg._edit_spool()
        assert win.settings.spools[0]["name"] == "Galaxy black on the spare reel"
        assert win.settings.spools[0]["left_g"] == 360.0
        assert win.settings.spools[0]["id"] == kept["id"], "editing gives a spool another id"
        assert dlg.table.item(0, 3).text() == "360 g of 750 g left (48 %)"

        box = Answer(Answer.No)
        monkeypatch.setattr(sd, "QMessageBox", box)
        dlg._remove_spool()
        assert box.asked, "the spool went off the shelf without asking"
        assert win.settings.spools, "no said no, so the spool stays"
        box.answer = Answer.Yes
        dlg._remove_spool()
        assert win.settings.spools == [] and dlg.table.rowCount() == 0
        assert not dlg.b_remove.isEnabled()
    finally:
        dlg.close()


def test_the_spools_dialog_keeps_a_spool_of_an_unusual_material(win, monkeypatch):
    """An older spool does not quietly become something else while nothing is edited in it."""
    from volura import spools_dialog as sd
    shelf(win, spools.Spool("pp1", "Hinge filaments", "PP", "#ff00ff"))

    def touch(dlg):
        assert dlg.material.currentData() == "PP"
        return QDialog.Accepted

    monkeypatch.setattr(sd.SpoolEditDialog, "exec", touch)
    dlg = sd.SpoolsDialog(win)
    try:
        dlg.table.selectRow(0)
        dlg._edit_spool()
        assert win.settings.spools[0]["material"] == "PP"
        assert dlg.table.item(0, 2).text() == "PP"
    finally:
        dlg.close()


def test_set_left_weighs_the_spool_and_subtracts_the_empty_one(win, monkeypatch):
    from volura import spools_dialog as sd
    spool = spools.Spool("s1", "Blue PLA", "PLA", "#223344", 1000.0, 800.0, 0.0, 250.0)
    shelf(win, spool)
    monkeypatch.setattr(sd, "QInputDialog", scale := Scale(900.0))
    dlg = sd.SpoolsDialog(win)
    try:
        dlg.table.selectRow(0)
        assert dlg.b_left.toolTip().startswith("Weigh the spool"), "the button must say how"
        dlg._set_left()
        assert scale.asked[0][1] == "Set left"
        assert scale.asked[0][3] == pytest.approx(1050.0), \
            "the field starts at what the scale would have to say for what is recorded"
        assert "250 g" in scale.asked[0][2]
        assert win.settings.spools[0]["left_g"] == pytest.approx(650.0)
        assert dlg.table.item(0, 3).text() == "650 g of 1000 g left (65 %)"
        assert dlg.table.cellWidget(0, 4).value() == 65
        # a reading below the empty spool is nothing on the spool, and cancelling keeps it
        dlg._set_left()                        # 900 again: 650 is still what is left
        monkeypatch.setattr(sd, "QInputDialog", Scale(None))
        dlg._set_left()
        assert win.settings.spools[0]["left_g"] == pytest.approx(650.0)
        monkeypatch.setattr(sd, "QInputDialog", Scale(200.0))
        dlg._set_left()
        assert win.settings.spools[0]["left_g"] == 0.0
    finally:
        dlg.close()


# ------------------------------------------------------------------ in the print preview

TWO = [("T0 PETG clear", "#eeeeee", 1000.0, 20.0), ("T1 PLA black", "#111111", 2000.0, 40.0)]


def test_the_preview_offers_the_spools_of_the_right_material_first(win, tmp_path):
    petg = spools.Spool("petg1", "Clear PETG", "PETG", "#eeeeee", 1000.0, 900.0)
    pla = spools.Spool("pla1", "Black PLA", "PLA", "#111111", 1000.0, 30.0)
    shelf(win, pla, petg)
    dlg = preview(win, tmp_path, TWO)
    try:
        assert len(dlg.spool_combos) == 3, "T0, T1 and the filament after the M600"
        assert [dlg.spool_combos[0].itemData(i) for i in range(3)] == ["", "petg1", "pla1"]
        assert [dlg.spool_combos[1].itemData(i) for i in range(3)] == ["", "pla1", "petg1"]
        assert dlg.spool_combos[0].itemText(0) == "No spool"
        assert dlg.spool_notes[0].text() == "", "nothing is chosen, so nothing is promised"
    finally:
        dlg.close()


def test_the_preview_says_whether_the_spool_holds_enough(win, tmp_path):
    petg = spools.Spool("petg1", "Clear PETG", "PETG", "#eeeeee", 1000.0, 900.0)
    pla = spools.Spool("pla1", "Black PLA", "PLA", "#111111", 1000.0, 30.0)
    shelf(win, pla, petg)
    dlg = preview(win, tmp_path, TWO)
    try:
        combo = dlg.spool_combos[0]
        combo.setCurrentIndex(combo.findData("petg1"))
        assert dlg.spool_notes[0].text() == "20 g needed, 900 g left"
        assert theme.color("warn") not in dlg.spool_notes[0].styleSheet()
        second = dlg.spool_combos[1]
        second.setCurrentIndex(second.findData("pla1"))
        assert "40 g needed, 30 g left" in dlg.spool_notes[1].text()
        assert "Not enough filament on this spool" in dlg.spool_notes[1].text()
        assert theme.color("warn") in dlg.spool_notes[1].styleSheet()
        assert win.settings.spool_for_slot == ["petg1", "pla1", ""]
    finally:
        again = preview(win, tmp_path, TWO)
        try:
            assert again.spool_combos[0].currentData() == "petg1", "the slot lost its spool"
            assert again.spool_combos[2].currentData() == ""
        finally:
            again.close()
        dlg.close()


def test_a_slot_that_remembers_a_removed_spool_simply_has_no_spool(win, tmp_path):
    petg = spools.Spool("petg1", "Clear PETG", "PETG", "#eeeeee", 1000.0, 900.0)
    shelf(win, petg, slots=["pla1", "", "petg1"])
    dlg = preview(win, tmp_path, TWO)
    try:
        assert dlg.spool_combos[0].currentIndex() == 0, "a spool that is gone is no spool"
        assert dlg.spool_notes[0].text() == ""
        assert dlg.spool_combos[2].currentData() == "petg1"
        assert dlg.spool_notes[2].text() == "900 g of 1000 g left (90 %)", \
            "the print does not know what that filament wants, so it only says what is left"
    finally:
        dlg.close()


def test_a_print_of_one_filament_still_comes_off_a_spool(win, tmp_path):
    """Most prints run on one filament; that is exactly the print a spool has to last for."""
    red = spools.Spool("pla1", "Spool of the red one", "PLA", "#ff0000", 1000.0, 100.0)
    shelf(win, red, slots=["pla1"])
    dlg = preview(win, tmp_path, [("Red PLA", "#ff0000", 1000.0, 10.0)], gcode=SINGLE)
    try:
        assert not dlg.by_filament and len(dlg.spool_combos) == 1
        assert dlg.spool_combos[0].currentData() == "pla1", "the filament remembers its spool"
        assert dlg.spool_notes[0].text() == "10 g needed, 100 g left"
        assert theme.color("warn") not in dlg.spool_notes[0].styleSheet()
        text = " ".join(label.text() for label in dlg.findChildren(QLabel))
        assert "Outer wall" in text, "the colours of the kinds of line keep their legend"
        dlg._mark_printed()
        assert left_of(win.settings.spools) == {"pla1": 90.0}
        assert dlg.taken_note.text() == "Taken from the spools" and not dlg.mark.isEnabled()
    finally:
        dlg.close()


def test_a_print_of_nothing_at_all_has_no_spool_to_point_at(win, tmp_path):
    """A slice that says nothing about its filaments has nothing to weigh or to take."""
    shelf(win, PLA_RED, slots=["pla1"])
    dlg = preview(win, tmp_path, [], gcode=SINGLE)
    try:
        assert dlg.spool_combos == [] and dlg.spool_notes == [] and dlg.mark is None
        assert dlg.taken_note.text() == ""
        dlg._mark_printed()
        assert left_of(win.settings.spools) == {"pla1": 900.0}
    finally:
        dlg.close()


def test_a_print_that_starts_on_the_printer_takes_the_grams_once(win, tmp_path, monkeypatch):
    from volura import print_preview as pp
    petg = spools.Spool("petg1", "Clear PETG", "PETG", "#eeeeee", 1000.0, 900.0)
    pla = spools.Spool("pla1", "Black PLA", "PLA", "#111111", 1000.0, 500.0)
    shelf(win, petg, pla, slots=["petg1", "pla1", ""])
    win.settings.printers = [printer_link.Connection("Octo", "octoprint", "192.168.1.20", "k")]
    win.settings.save()
    calls = []

    def fake_upload(conn, path, name, start=False, on_progress=None):
        calls.append((conn.name, name, bool(start)))

    monkeypatch.setattr(pp.printer_link, "upload", fake_upload)
    monkeypatch.setattr(pp.SendDialog, "exec", press)
    dlg = preview(win, tmp_path, TWO)
    try:
        dlg._send()
        assert dlg._job.wait(20000), dlg._job.error
        assert [call[0] for call in calls] == ["Octo"] and calls[0][2] is True
        assert left_of(win.settings.spools) == {"petg1": 880.0, "pla1": 460.0}
        assert dlg.spool_notes[0].text() == "20 g needed, 880 g left"
        assert dlg.taken_note.text() == "Taken from the spools"
        assert not dlg.mark.isEnabled(), "the grams are off the spools, so Mark may not again"
        dlg._mark_printed()
        assert left_of(win.settings.spools) == {"petg1": 880.0, "pla1": 460.0}
        assert left_of(type(win.settings)().spools) == {"petg1": 880.0, "pla1": 460.0}, \
            "the grams went off the settings of this computer"
    finally:
        win.settings.printers = []
        win.settings.save()
        dlg.close()


def test_a_print_that_never_started_takes_nothing_until_it_is_said(win, tmp_path, monkeypatch):
    """A file that only went to an SD card printed nothing yet: saving is not printing."""
    from volura import print_preview as pp
    petg = spools.Spool("petg1", "Clear PETG", "PETG", "#eeeeee", 1000.0, 900.0)
    shelf(win, petg, slots=["petg1", "", ""])
    dlg = preview(win, tmp_path, TWO)
    try:
        target = str(tmp_path / "saved.gcode")
        dlg.ask_path = lambda: target
        dlg._save()
        assert dlg.saved_to == target
        assert left_of(win.settings.spools) == {"petg1": 900.0}
        assert dlg.taken_note.text() == "" and dlg.mark.isEnabled()
        # and marking it printed takes them once, then the button is done
        dlg._mark_printed()
        assert left_of(win.settings.spools) == {"petg1": 880.0}
        assert dlg.taken_note.text() == "Taken from the spools"
        assert not dlg.mark.isEnabled()
        dlg._mark_printed()
        assert left_of(win.settings.spools) == {"petg1": 880.0}
    finally:
        dlg.close()


def test_marking_printed_without_a_spool_chosen_takes_nothing(win, tmp_path):
    petg = spools.Spool("petg1", "Clear PETG", "PETG", "#eeeeee", 1000.0, 900.0)
    shelf(win, petg)
    dlg = preview(win, tmp_path, TWO)
    try:
        dlg._mark_printed()
        assert left_of(win.settings.spools) == {"petg1": 900.0}
        assert dlg.taken_note.text() == ""
        assert dlg.mark.isEnabled(), "nothing was taken, so there is still something to mark"
    finally:
        dlg.close()


def test_a_spool_that_left_the_shelf_since_the_preview_opened_is_skipped(win, tmp_path):
    """What is not on the shelf cannot give anything, and does not come back on it because a
    preview had it on its list."""
    petg = spools.Spool("petg1", "Clear PETG", "PETG", "#eeeeee", 1000.0, 900.0)
    pla = spools.Spool("pla1", "Black PLA", "PLA", "#111111", 1000.0, 500.0)
    shelf(win, petg, pla, slots=["petg1", "pla1", ""])
    dlg = preview(win, tmp_path, TWO)
    try:
        shelf(win, petg, slots=["petg1", "pla1", ""])   # the PLA spool went out the door
        dlg._mark_printed()
        assert left_of(win.settings.spools) == {"petg1": 880.0}
        assert dlg.spool_notes[1].text() == "", "the spool that is gone promises nothing"
    finally:
        dlg.close()


def test_an_empty_shelf_offers_no_spool_and_says_nothing(win, tmp_path):
    shelf(win)
    dlg = preview(win, tmp_path, TWO)
    try:
        assert len(dlg.spool_combos) == len(dlg.spool_notes) == 3
        for combo, amount in zip(dlg.spool_combos, dlg.spool_notes):
            assert combo.count() == 1 and combo.currentData() == "", "only No spool to pick"
            assert amount.text() == "" and amount.styleSheet() == ""
        dlg._mark_printed()
        assert win.settings.spools == [] and dlg.taken_note.text() == ""
    finally:
        dlg.close()


def test_a_filament_the_print_cannot_weigh_promises_nothing(win, tmp_path):
    """The filament after the M600 has no grams of its own: there is nothing to check and
    nothing to take, so its spool only says what it still holds."""
    petg = spools.Spool("petg1", "Clear PETG", "PETG", "#eeeeee", 1000.0, 900.0)
    shelf(win, petg, slots=["petg1", "", "petg1"])
    dlg = preview(win, tmp_path, TWO)
    try:
        assert dlg.spool_notes[0].text() == "20 g needed, 900 g left"
        assert dlg.spool_notes[2].text() == "900 g of 1000 g left (90 %)"
        assert dlg.spool_notes[2].styleSheet() == "", "nothing known, so nothing warned"
        dlg._mark_printed()
        assert left_of(win.settings.spools) == {"petg1": 880.0}, "one spool, only the 20 g of T0"
        assert dlg.spool_notes[2].text() == "880 g of 1000 g left (88 %)"
    finally:
        dlg.close()

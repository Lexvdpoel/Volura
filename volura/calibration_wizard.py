"""3D Printing > Calibrate Filament…: the five calibration prints of one filament, in one walk.

The temperature tower, the flow rate test, the pressure advance test, the retraction test and the
max speed test each tell one thing about a filament on a printer (volura.temperature_tower and the
four modules after it). This wizard walks those five in that order: every page tells what the test
is and what to look for on it, puts it on the free spot of the bed, and asks for the number that
came out of it. What was found is kept per printer and filament (volura.calibrations), and File >
Slice prints with those numbers from then on, until the filament is calibrated again.

Every page stands on its own, because that is how these prints are really done: one Saturday the
temperature, a month later the retraction. A test that is put on Skip is worth nothing later on -
it is simply not there - and a page that has a number kept for it starts with that number in it.
So a second walk through the wizard fills in what is missing and leaves the rest as it is, and a
walk with every page on Skip changes nothing at all.
"""
from __future__ import annotations

from collections import namedtuple

from PySide6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout, QLabel,
                               QPushButton, QWizard, QWizardPage)

from . import (calibrations, flow_rate_test, max_speed_test, pressure_advance_test,
               retraction_test, temperature_tower)
from .i18n import tr, tr_noop

#: one page of the walk: which number of the calibration it fills in, what the page is called,
#: two sentences about the print, how the field is named and how its number is written
Test = namedtuple("Test", "key title what look label unit decimals step")

#: the five tests, in the order they are walked and in the order the settings write their numbers
TESTS = (
    Test("temperature", tr_noop("Nozzle Temperature"),
         tr_noop("The temperature tower is a stack of segments, and every segment prints at "
                 "another temperature."),
         tr_noop("The coolest segment whose tops are smooth, whose bridges hang straight and "
                 "whose corners come out clean was printed at the temperature to keep."),
         tr_noop("Best temperature"), " °C", 0, 5.0),
    Test("flow", tr_noop("Material Flow"),
         tr_noop("The flow rate test is a row of pads, and every pad prints at another flow."),
         tr_noop("Read the tops: pick the pad with a smooth, closed top without ridges."),
         tr_noop("Best flow"), " %", 1, 0.5),
    Test("advance", tr_noop("Pressure Advance"),
         tr_noop("The pressure advance test is a hollow tower, and every segment prints with "
                 "another pressure advance value."),
         tr_noop("Read the corners: the printer changes value at every segment, on the whole "
                 "bed."),
         tr_noop("Best value"), "", 3, 0.005),
    Test("retraction", tr_noop("Retraction"),
         tr_noop("The retraction test is two thin towers, and every band above them travels with "
                 "a longer retraction."),
         tr_noop("The band whose towers are free of strings and whose tops stay clean travelled "
                 "with the retraction to keep."),
         tr_noop("Best length"), " mm", 2, 0.1),
    Test("max_flow", tr_noop("Max Flow"),
         tr_noop("The max speed test is a tube, and every segment of it melts more plastic per "
                 "second than the one below."),
         tr_noop("Print it and count the segments by the ribs: the highest segment whose wall "
                 "stays smooth and full is the flow this printer can melt."),
         tr_noop("Highest flow"), " mm³/s", 1, 0.5))


def numbers(win, name, kind, firmware=None) -> dict:
    """The ``start``, ``end``, ``step`` and ``firmware`` the button of a page puts on the bed for a
    filament of `kind`, with the ``text`` that says what that test prints.

    The temperatures follow the preset of the material, because that is the range its maker says it
    prints well over. The pressure advance follows the firmware - the one picked on the page, or,
    with nothing picked, the one the printers on the network run (volura.printer_link) - and the
    preset of the extruder that belongs to that firmware. The other three take the first preset of
    their own test, which is the ordinary one for an ordinary machine."""
    if name == "temperature":
        # the preset of the material, or the ordinary PLA one for a filament none of them knows
        start, end, step = temperature_tower.PRESETS.get(
            kind, temperature_tower.PRESETS[temperature_tower.PRESET_DEFAULT])
        text = tr("The tower prints {start} to {end} °C in steps of {step}.")
    elif name == "flow":
        start, end, step = (flow_rate_test.FLOW_START, flow_rate_test.FLOW_END,
                            flow_rate_test.FLOW_STEP)
        text = tr("The pads print {start} to {end} % in steps of {step}.")
    elif name == "advance":
        firm = firmware or pressure_advance_test.default_firmware(win.settings.printers)
        _firm, start, end, step = pressure_advance_test.PRESETS[
            pressure_advance_test.default_preset(firm)]
        text = tr("The tower prints {start} to {end} with {firmware}, in steps of {step}.")
        return {"start": start, "end": end, "step": step, "firmware": firm,
                "text": text.format(start=calibrations.number_text(start),
                                    end=calibrations.number_text(end),
                                    firmware=pressure_advance_test.firmware_label(firm),
                                    step=calibrations.number_text(step))}
    elif name == "retraction":
        start, end, step = retraction_test.PRESETS[retraction_test.default_preset()]
        text = tr("The towers travel with {start} to {end} mm in steps of {step}.")
    else:
        start, end, step = max_speed_test.PRESETS[max_speed_test.default_preset()]
        text = tr("The tube prints {start} to {end} mm³/s in steps of {step}.")
    return {"start": start, "end": end, "step": step, "firmware": None,
            "text": text.format(start=calibrations.number_text(start),
                                end=calibrations.number_text(end),
                                step=calibrations.number_text(step))}


def default_number(win, name, kind, firmware=None) -> float:
    """The number a page starts with when nothing is kept for this filament: the temperature a print
    starts at, which is the hottest of the tower, and the middle of what the other tests walk
    through, which is where a number read off such a test usually falls."""
    made = numbers(win, name, kind, firmware)
    if name == "temperature":
        return float(made["start"])
    return round(0.5 * (float(made["start"]) + float(made["end"])), 4)


def put(win, name, made):
    """The test of a page on the bed, through the very methods the dialogs of the five tests call
    (volura.window_print): every one of them builds in the background, lands on the free spot right
    of the model and is one undo step, and the number that belongs to it is remembered for the
    slice that follows."""
    start, end, step = float(made["start"]), float(made["end"]), float(made["step"])
    if name == "temperature":
        win._apply_temperature_tower(start, end, step, temperature_tower.steps(start, end, step))
    elif name == "flow":
        win._apply_flow_rate_test(start, end, step, flow_rate_test.flows(start, end, step))
    elif name == "advance":
        plan = pressure_advance_test.steps(start, end, step)
        firm = made.get("firmware") or pressure_advance_test.FIRMWARE_DEFAULT
        win._apply_pressure_advance_test(firm, start, end, step, plan)
    elif name == "retraction":
        win._apply_retraction_test(start, end, step, retraction_test.bands(start, end, step))
    else:
        win._apply_max_speed_test(start, end, step, max_speed_test.steps(start, end, step))


class ChoosePage(QWizardPage):
    """The printer and the filament the numbers are for: those two make the key of a calibration.

    The same cascade as File > Slice (volura.slice_dialog), and it starts where the last slice
    started, because the filament that is in the machine now is the one that needs calibrating."""

    def __init__(self, wiz, lib):
        super().__init__(wiz)
        self.wiz = wiz
        self.lib = lib
        self.setTitle(tr("Printer and filament"))
        lay = QFormLayout(self)
        intro = QLabel(tr("Every number you find here belongs to this printer and this filament: "
                          "another printer wants another temperature, another filament another "
                          "flow."))
        intro.setWordWrap(True)
        self.maker = QComboBox()
        self.printer_box = QComboBox()
        self.nozzle = QComboBox()
        self.brand = QComboBox()
        self.material = QComboBox()
        self.note = QLabel()
        self.note.setWordWrap(True)
        lay.addRow(intro)
        lay.addRow(tr("Maker"), self.maker)
        lay.addRow(tr("Printer"), self.printer_box)
        lay.addRow(tr("Nozzle"), self.nozzle)
        lay.addRow(tr("Filament brand"), self.brand)
        lay.addRow(tr("Material"), self.material)
        lay.addRow("", self.note)
        s = wiz.win.settings
        self.machines = list(getattr(lib, "machines", None) or [])
        self.maker.addItems(sorted({one[2] for one in self.machines}, key=str.lower))
        # what the last slice used, so the wizard opens on the filament that is in the machine
        self._want = {"printer": str(getattr(s, "slice_printer", "") or ""),
                      "variant": str(getattr(s, "slice_variant", "") or ""),
                      "material": str(getattr(s, "slice_material", "") or "")}
        mine = next((one for one in self.machines
                     if one[0] == str(getattr(s, "slice_printer", "") or "")), None)
        if mine is not None:
            self.maker.setCurrentText(mine[2])
        self.maker.currentIndexChanged.connect(self._maker_changed)
        self.printer_box.currentIndexChanged.connect(self._printer_changed)
        self.nozzle.currentIndexChanged.connect(self._nozzle_changed)
        self.brand.currentIndexChanged.connect(self._brand_changed)
        self.material.currentIndexChanged.connect(self._picked)
        self._maker_changed()

    # ---- the cascade of File > Slice: maker, printer, nozzle, brand, material
    def _maker_changed(self, *_):
        maker = self.maker.currentText()
        self.printer_box.blockSignals(True)
        self.printer_box.clear()
        for did, name, m in self.machines:
            if m == maker:
                self.printer_box.addItem(name, did)
        self.printer_box.blockSignals(False)
        i = self.printer_box.findData(self._want["printer"])
        self.printer_box.setCurrentIndex(max(0, i))
        self._printer_changed()

    def _printer_changed(self, *_):
        did = self.printer_box.currentData()
        self.printer = self.lib.printer(did) if did else None
        self.nozzle.blockSignals(True)
        self.nozzle.clear()
        if self.printer is not None:
            for variant in self.printer.variants:
                self.nozzle.addItem(variant.name, variant.id)
            default = self.printer.default_variant()
            wanted = self.nozzle.findText(self._want["variant"])
            if wanted < 0 and default is not None:
                wanted = self.nozzle.findText(default.name)
            self.nozzle.setCurrentIndex(max(0, wanted))
        self.nozzle.blockSignals(False)
        self._nozzle_changed()

    def _nozzle_changed(self, *_):
        self.brand.blockSignals(True)
        self.material.blockSignals(True)
        self.brand.clear()
        self.material.clear()
        if self.printer is not None:
            materials = self.printer.materials(self._variant())
            self.brand.addItems(sorted({one.metadata["brand"] for one in materials},
                                       key=lambda brand: (brand != "Generic", brand.lower())))
            wanted = next((one for one in materials if one.id == self._want["material"]),
                          self.printer.default_material(self._variant()))
            if wanted is not None:
                self.brand.setCurrentText(wanted.metadata["brand"])
            self._fill_materials(wanted.id if wanted is not None else None)
        self.material.blockSignals(False)
        self.brand.blockSignals(False)
        self._picked()

    def _fill_materials(self, want=None):
        """The materials of the brand that stands in the combo box, with `want` picked back up."""
        brand = self.brand.currentText()
        self.material.clear()
        for one in self.printer.materials(self._variant()):
            if one.metadata["brand"] == brand:
                label = one.name[len(brand):].strip() if one.name.startswith(brand) else one.name
                self.material.addItem(label or one.name, one.id)
        i = self.material.findData(want) if want else -1
        if i < 0:
            i = self.material.findData(self._want["material"])
        self.material.setCurrentIndex(max(0, i))

    def _brand_changed(self, *_, want=None):
        if self.printer is None:
            return
        self.material.blockSignals(True)
        self._fill_materials(want)
        self.material.blockSignals(False)
        self._picked()

    def _variant(self):
        if self.printer is None:
            return None
        vid = self.nozzle.currentData()
        return next((one for one in self.printer.variants if one.id == vid), None)

    # ---- what the two together mean for the rest of the wizard
    def show_found(self):
        """The line that says what is kept for this printer and this filament, shown only when
        there is something to say about it; the numbers themselves are what it returns."""
        made = self.wiz.found()
        self.note.setText(tr("Calibrated: {found}").format(found=calibrations.text_of(made))
                          if made else "")
        self.note.setVisible(bool(made))
        return made

    def _picked(self, *_):
        """The printer, the brand or the filament changed: what the tests print follows the
        filament, and what the pages hold follows what is kept for this printer and filament."""
        self.wiz.filament_kind = calibrations.material_type(
            " ".join((self.brand.currentText(), self.material.currentText(),
                      self.material_id())),
            str(getattr(self.wiz.win.settings, "filament_material", "") or ""))
        made = self.show_found()
        if self.wiz.pages:
            self.wiz.start_from(made)
        self.completeChanged.emit()

    def printer_id(self) -> str:
        """The printer of the choice, by the name the settings and File > Slice know it under."""
        return str(self.printer_box.currentData() or "")

    def material_id(self) -> str:
        """The filament of the choice, by the id of its profile."""
        return str(self.material.currentData() or "")

    def printer_name(self) -> str:
        """The printer of the choice as it stands in the combo box, for a line about the choice."""
        return str(self.printer_box.currentText() or "")

    def material_name(self) -> str:
        """The filament with its brand on front, the way the profiles name it."""
        return f"{self.brand.currentText()} {self.material.currentText()}".strip()

    def isComplete(self) -> bool:
        """The walk can only start once there is a printer and a filament to calibrate for."""
        return bool(self.printer_id() and self.material_id())


class TestPage(QWizardPage):
    """One calibration print: what it is, what to look for, the button that puts it on the bed, and
    the number the user reads off it - or the Skip that says this one is not worth a number."""

    def __init__(self, wiz, spec: Test):
        super().__init__(wiz)
        self.wiz, self.spec = wiz, spec
        self.setTitle(tr(spec.title))
        lay = QFormLayout(self)
        self.intro = QLabel(f"{tr(spec.what)} {tr(spec.look)}")
        self.intro.setWordWrap(True)
        self.note = QLabel()
        self.note.setWordWrap(True)
        self.button = QPushButton(tr("Put the test on the bed"))
        self.button.setToolTip(tr("The test lands on the free spot of the bed, right of the "
                                  "model, as one undo step"))
        self.firmware = None
        if spec.key == "advance":
            # without a firmware a pressure advance value is nothing: it is a command line, and
            # every firmware writes that line differently (volura.pressure_advance_test)
            self.firmware = QComboBox()
            for name in pressure_advance_test.FIRMWARES:
                self.firmware.addItem(pressure_advance_test.firmware_label(name), name)
            self.firmware.setToolTip(tr("The firmware your printer runs: every firmware sets the "
                                        "pressure advance with another command"))
            # the firmware of the last Pressure Advance Test, or what the printers on the network
            # run, or Marlin with nothing on the network: the ordinary case, and the user can pick
            # another one here (volura.pressure_advance_test)
            want = str(getattr(wiz.win.settings, "pressure_advance_firmware", "") or "")
            if want not in pressure_advance_test.FIRMWARES:
                want = pressure_advance_test.default_firmware(wiz.win.settings.printers)
            i = self.firmware.findData(want)
            self.firmware.setCurrentIndex(max(0, i))
        lo, hi = calibrations.BOUNDS[spec.key]
        self.value = QDoubleSpinBox()
        self.value.setRange(lo, hi)
        self.value.setDecimals(spec.decimals)
        self.value.setSingleStep(spec.step)
        self.value.setSuffix(spec.unit)
        self.skip = QCheckBox(tr("Skip this test"))
        self.skip.setToolTip(tr("Nothing is remembered of this test, so nothing of it is applied "
                                "when you slice"))
        lay.addRow(self.intro)
        lay.addRow(self.note)
        lay.addRow("", self.button)
        if self.firmware is not None:
            lay.addRow(tr("Firmware"), self.firmware)
        lay.addRow(tr(spec.label), self.value)
        lay.addRow("", self.skip)
        self.button.clicked.connect(self._on_bed)
        self.skip.toggled.connect(self._use)
        if self.firmware is not None:
            self.firmware.currentIndexChanged.connect(self.show_numbers)
        self.value.setValue(default_number(wiz.win, spec.key, wiz.filament_kind,
                                           self.firmware_choice()))
        self.skip.setChecked(True)     # a number is only worth having once it was read off a print
        self._use(False)
        self.show_numbers()

    def _use(self, on):
        """The field only follows what is typed in when the test is really printed."""
        self.value.setEnabled(bool(on))

    def firmware_choice(self):
        """The firmware the pressure advance of this page is set with; None on every other page."""
        return self.firmware.currentData() if self.firmware is not None else None

    def show_numbers(self):
        """The line that says what the button puts on the bed for a filament of this kind."""
        made = numbers(self.wiz.win, self.spec.key, self.wiz.filament_kind,
                       self.firmware_choice())
        self.note.setText(made["text"])

    def _on_bed(self):
        """The button: the test of this page on the bed, with the numbers the note just said."""
        made = numbers(self.wiz.win, self.spec.key, self.wiz.filament_kind,
                       self.firmware_choice())
        self.note.setText(made["text"])
        put(self.wiz.win, self.spec.key, made)

    def set_kept(self, value, firmware=None, reset=False):
        """The field to a number that is kept for this printer and filament; a test that has nothing
        kept is put back on Skip, so walking the wizard again leaves it alone. A page that has
        nothing kept and `reset` starts again on the number that suits the filament of the page, so
        another filament does not leave the old one's starting point standing in the field."""
        try:
            number = None if value is None else float(value)
        except (TypeError, ValueError):
            number = None
        boxes = [self.skip, self.value] + ([self.firmware] if self.firmware is not None else [])
        for box in boxes:
            box.blockSignals(True)
        self.skip.setChecked(number is None)
        if number is not None:
            self.value.setValue(number)
        if self.firmware is not None and firmware is not None:
            i = self.firmware.findData(firmware)
            if i >= 0:
                self.firmware.setCurrentIndex(i)
        if number is None and reset:
            self.value.setValue(default_number(self.wiz.win, self.spec.key, self.wiz.filament_kind,
                                               self.firmware_choice()))
        for box in boxes:
            box.blockSignals(False)
        self._use(number is not None)

    def value_text(self) -> str:
        """The number of this page as the summary writes it: with its unit, and for a pressure
        advance with the firmware that sets it."""
        text = f"{calibrations.number_text(self.value.value())}{self.spec.unit}"
        if self.firmware is not None:
            text += f" ({pressure_advance_test.firmware_label(self.firmware_choice())})"
        return text


class SummaryPage(QWizardPage):
    """What the walk found, and where those numbers go from here."""

    def __init__(self, wiz):
        super().__init__(wiz)
        self.wiz = wiz
        self.setTitle(tr("What you found"))
        self.setFinalPage(True)
        lay = QFormLayout(self)
        self.found = QLabel()
        self.found.setWordWrap(True)
        self.where = QLabel(tr("File > Slice prints with these numbers from now on, for this "
                               "printer and this filament; untick Use my calibration there for one "
                               "print without them."))
        self.where.setWordWrap(True)
        lay.addRow(self.found)
        lay.addRow(self.where)

    def initializePage(self):
        """The numbers are read as the page is reached, so they are what the pages hold by then."""
        self.found.setText(self.wiz.findings())


class CalibrationWizard(QWizard):
    """3D Printing > Calibrate Filament…: the five calibration prints of one filament, and what the
    user read off them.

    The first page is the printer and the filament those numbers belong to, then the five tests in
    the order of the walks, and a page that says what was found. `exec()` asks the questions; the
    window calls `save()` after an accepted walk, which keeps what was found under that printer and
    filament. `pages` holds every test page by the field it fills in, so the field of one test is
    close at hand - and the walk itself is built out of `TESTS`, which is what the order and the
    fields of the pages really are."""

    def __init__(self, win, lib):
        super().__init__(win)
        self.win = win
        self.setWindowTitle(tr("Calibrate Filament"))
        self.setOption(QWizard.NoBackButtonOnStartPage, True)
        self.filament_kind = calibrations.material_type(
            str(getattr(win.settings, "filament_material", "") or ""), "PLA")
        # the first page asks around while it is built, and the pages of the five tests are not
        # there yet then: no first page and an empty dict are what it reads as "nothing kept yet"
        self.pages = {}
        self.choose = None
        self.choose = ChoosePage(self, lib)
        self.addPage(self.choose)
        for spec in TESTS:
            page = TestPage(self, spec)
            self.pages[spec.key] = page
            self.addPage(page)
        self.summary = SummaryPage(self)
        self.addPage(self.summary)
        self.start_from(self.choose.show_found())

    # ---- what stands in the settings for what the first page holds
    def pair(self) -> tuple:
        """The printer and the filament of the first page; two empty texts while that page is still
        being built, which is no printer and no filament, and so nothing kept and nothing to keep."""
        if self.choose is None:
            return "", ""
        return self.choose.printer_id(), self.choose.material_id()

    def key(self) -> str:
        """The printer and the filament of the first page, as the settings keep a calibration under
        them; an empty text when that page holds no printer or no filament, which is no key at all."""
        printer, material = self.pair()
        return calibrations.key(printer, material) if printer and material else ""

    def found(self):
        """What is kept for the printer and filament of the first page; None when there is nothing
        to go on, which is what a wizard for a new filament starts with."""
        printer, material = self.pair()
        return calibrations.find(getattr(self.win.settings, "calibrations", None),
                                 printer, material)

    def start_from(self, found):
        """Every page to what is kept for the chosen printer and filament: the pages of those
        numbers on - with the firmware of the pressure advance among them - and the tests that were
        never walked back on Skip.

        Called whenever the printer or the filament changes, so the wizard goes on where the last
        walk of this filament left off; a filament that was never calibrated starts on Skip on every
        page, which is the honest place to start."""
        kept = found or {}
        for spec in TESTS:
            page = self.pages[spec.key]
            page.set_kept(kept.get(spec.key), kept.get("firmware"), reset=True)
            page.show_numbers()

    # ---- what the user made of the prints
    def results(self) -> dict:
        """What the pages hold, in the shape the settings keep: a page on Skip is not there, and so
        is nothing else that is no number; the firmware of the pressure advance comes along with it,
        because without a firmware there is no command to set it with."""
        made = {}
        for spec in TESTS:
            page = self.pages[spec.key]
            if not page.skip.isChecked():
                made[spec.key] = page.value.value()
                if page.firmware is not None:
                    made["firmware"] = page.firmware_choice()
        return calibrations.clean(made)

    def findings(self) -> str:
        """What was found, one number to a line, with the tests that were skipped under it."""
        made = self.results()
        if not made:
            return tr("Nothing was found, so nothing will be saved.")
        lines = [tr("{name}: {value}").format(name=tr(spec.label),
                                              value=self.pages[spec.key].value_text())
                 for spec in TESTS if spec.key in made]
        skipped = [tr(spec.title) for spec in TESTS if spec.key not in made]
        if skipped:
            lines.append(tr("Skipped: {names}").format(names=", ".join(skipped)))
        return "\n".join(lines)

    def save(self) -> dict:
        """Keep what was found for this printer and this filament, and say what came of it.

        A walk with every page on Skip leaves the settings alone on purpose: it has nothing to say
        about this filament, and the numbers of the last walk are worth keeping better than they are
        worth clearing. So does a first page without a printer or a filament: there is no key to
        keep anything under, and the wizard says so instead of dropping the numbers on nothing."""
        name = self.key()
        made = self.results()
        if not name:
            self.win.view.set_status(tr("Choose the printer and the filament first."))
            return {}
        if not made:
            self.win.view.set_status(tr("Nothing was found, so nothing will be saved."))
            return {}
        self.win.settings.calibrations[name] = made
        self.win.settings.save()
        self.win.view.set_status(tr("Calibration saved for {printer} with {material}: {found}")
                                .format(printer=self.choose.printer_name(),
                                        material=self.choose.material_name(),
                                        found=calibrations.text_of(made)))
        return made

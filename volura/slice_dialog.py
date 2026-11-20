"""The "Slice…" dialog: the printer, nozzle, material and quality to slice with.

The printers come from the Cura profile library that Volura brings along (see
`volura.cura_profiles`): first the maker, then the printer, then what that printer offers. A few
settings that you change on almost every print sit underneath - infill and its pattern, support and
how the print sticks to the bed - and an Extras group with adaptive layers, ironing, a fuzzy skin,
the number of walls, where the seam sits and a vase mode that prints one wall spiralling up,
besides the thumbnail for the printer's screen; everything else comes from the profile. The
choice is remembered in Settings, and a choice you come back to is saved under a name above the
Maker row (see `volura.slice_presets`).

Below that sits what the print does with the colors of the model (see `volura.paint_split` and
`volura.fileio`): which filament prints which color, how deep the skins of painted faces go,
whether a printer with one nozzle has a filament changer in front of it and at which heights a
single-extruder printer changes its filament by hand. `multicolor()` gives all of that, and a
slice follows it.
"""
from __future__ import annotations

import os

from PySide6.QtCore import QSize, QStandardPaths, Qt
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox,
                               QFormLayout, QGroupBox, QHBoxLayout, QHeaderView, QInputDialog,
                               QLabel, QLineEdit, QListWidget, QListWidgetItem, QMessageBox,
                               QPushButton, QSpinBox, QTableWidget, QTableWidgetItem, QWidget)

from . import __version__, cura_profiles, gcode_thumbnail, slice_presets, slicer
from .i18n import tr, tr_noop
from .panels import swatch

#: one library per resources folder, kept for the session: reading it takes a while
_LIBRARIES: dict[str, cura_profiles.Library] = {}

#: the support choices: (label, support_enable, support_type)
SUPPORTS = [(tr_noop("None"), False, "buildplate"),
            (tr_noop("Touching the bed"), True, "buildplate"),
            (tr_noop("Everywhere"), True, "everywhere")]
#: the bed adhesion choices: (label, adhesion_type or "" for the printer's own)
ADHESION = [(tr_noop("As the printer has it"), ""), (tr_noop("Skirt"), "skirt"),
            (tr_noop("Brim"), "brim"), (tr_noop("Raft"), "raft"), (tr_noop("None"), "none")]
#: the infill patterns: (label, infill_pattern or "" for the profile's own)
INFILL_PATTERNS = [(tr_noop("As the profile has it"), ""), (tr_noop("Grid"), "grid"),
                   (tr_noop("Lines"), "lines"), (tr_noop("Triangles"), "triangles"),
                   (tr_noop("Tri-Hexagon"), "trihexagon"), (tr_noop("Cubic"), "cubic"),
                   (tr_noop("Gyroid"), "gyroid"), (tr_noop("Honeycomb"), "honeycomb"),
                   (tr_noop("Concentric"), "concentric"), (tr_noop("Zig Zag"), "zigzag"),
                   (tr_noop("Cross"), "cross"), (tr_noop("Lightning"), "lightning")]
#: where the seam of the print sits: (label, z_seam_type or "" for the profile's own, the settings
#: that go with it; the names and the values are those of Cura's fdmprinter definitions)
SEAMS = [(tr_noop("Profile"), "", {}),
         (tr_noop("Sharpest corner"), "sharpest_corner",
          {"z_seam_corner": "z_seam_corner_weighted"}),
         (tr_noop("At the back"), "back", {"z_seam_position": "back"}),
         (tr_noop("Random"), "random", {}),
         (tr_noop("Shortest"), "shortest", {})]


#: a filament changer in front of one nozzle: from two filaments up to what the largest
#: MMU-style changer holds
FILAMENTS_MIN, FILAMENTS_MAX = 2, 16
#: how deep the skin of a painted face goes (mm): about what PrusaSlicer and OrcaSlicer use to
#: segment a painted model for their own multi-material print
PAINT_MIN, PAINT_MAX, PAINT_DEFAULT = 0.4, 5.0, 1.0
#: the prime tower that wipes the nozzle of an extruder that is not printing: a square of this
#: part of the smallest side of the bed, never smaller and never larger than these bounds
TOWER_PART, TOWER_MIN, TOWER_MAX = 0.15, 15.0, 40.0
#: how thick the bumps of the fuzzy skin are (mm), and the number of wall lines the dialog
#: offers - 0 walls says "Profile": leave it to the profile
FUZZ_MIN, FUZZ_MAX, FUZZ_DEFAULT = 0.1, 1.0, 0.3
WALLS_MIN, WALLS_MAX = 0, 10


def parse_heights(text) -> list:
    """The heights of a settings line ("5,12.5") as a sorted list of millimetres."""
    out = set()
    for part in str(text or "").replace(";", ",").split(","):
        try:
            h = float(part)
        except ValueError:
            continue
        if h > 0:
            out.add(round(h, 3))
    return sorted(out)


def heights_text(heights) -> str:
    """Heights in millimetres as the one settings line `parse_heights` reads back."""
    return ",".join(f"{h:g}" for h in sorted({round(float(h), 3) for h in heights}))


def _number(value, current, lo, hi, whole=True):
    """The value a choice gives for a number field: `current` when it is not a number, and
    never outside the range of the field."""
    try:
        value = int(float(value)) if whole else float(value)
    except (TypeError, ValueError):
        return current
    return max(lo, min(hi, value))


def _index_of(choices, want, current):
    """The row of a combo box of (label, value, ...) pairs that `want` names, or `current` when
    the choice does not name one of them - an older preset may not know the setting yet."""
    if want is None:
        return current
    kinds = [value for _label, value, *_ in choices]
    want = str(want)
    return kinds.index(want) if want in kinds else current


class Multicolor:
    """What a slice does with the colors of the model: which filament prints what.

    `extruders` is how many the print uses, `mapping` says per model color (lower case) which
    of them it prints with, `depth` is how deep the skins of painted faces go and `changes` are
    the heights at which a printer with one nozzle changes its filament by hand, asking for it
    with `command`. `tower` and `ooze` are the two helpers of a print with more than one
    extruder: the block the nozzle wipes itself clean on, and the shield the ooze runs down.

    `colors` are the colors of the model as `volura.fileio.model_colors` gives them:
    ``[(name, color, filament number)]``, in the order the multicolor 3MF numbers them.
    """

    def __init__(self, extruders=1, mapping=None, depth=PAINT_DEFAULT, tower=True, ooze=False,
                 changes=(), command=slicer.FILAMENT_CHANGE, colors=()):
        self.extruders = max(1, int(extruders))
        self.mapping = dict(mapping or {})
        self.depth = float(depth)
        self.tower = bool(tower)
        self.ooze = bool(ooze)
        self.changes = [float(h) for h in changes]
        self.command = command or slicer.FILAMENT_CHANGE
        self.colors = list(colors)
        #: the filament number every color of the model has in the multicolor 3MF
        self.numbers = {str(color).lower(): number for _n, color, number in self.colors}

    def extruder(self, color) -> int:
        """Which extruder a model color prints with (0 is the first).

        Without a row in the color table that is the filament number of the multicolor 3MF, so
        a print follows the order of the Materials panel on its own.
        """
        key = str(color).lower()
        number = self.mapping[key] + 1 if key in self.mapping else self.numbers.get(key, 1)
        return (number - 1) % self.extruders

    def filaments(self) -> list:
        """The color and the name of every extruder's filament, first to last.

        The colors come from the model, so an extruder that several of them print with takes the
        first one; an extruder with nothing mapped to it comes back as an empty color.
        """
        out = [("", "")] * self.extruders
        for name, color, _number in sorted(self.colors, key=lambda item: item[2]):
            index = self.extruder(color)
            if not out[index][0]:
                out[index] = (str(color).lower(), name)
        return out

    def values(self, bed_w=0.0, bed_d=0.0) -> dict:
        """The Cura settings that go with these choices."""
        if self.extruders <= 1:
            return {}
        out = {"prime_tower_enable": self.tower, "ooze_shield_enabled": self.ooze}
        if self.tower:
            side = min(bed_w or 0.0, bed_d or 0.0) or 100.0
            out["prime_tower_size"] = round(max(TOWER_MIN, min(TOWER_MAX, TOWER_PART * side)), 1)
        return out


def is_loaded(resources: str) -> bool:
    """Whether the profiles of `resources` were read already in this session."""
    return resources in _LIBRARIES


def cache_path() -> str:
    """Where the profiles read once are kept for the next time (see `Library.load`)."""
    folder = QStandardPaths.writableLocation(QStandardPaths.CacheLocation)
    return os.path.join(folder, "cura-profiles.cache") if folder else ""


def library(resources: str) -> cura_profiles.Library:
    """The profile library of `resources`, read completely the first time it is asked for:
    from the cache when it is still good, otherwise from the files and into the cache."""
    lib = _LIBRARIES.get(resources)
    if lib is None:
        lib = cura_profiles.Library(resources)
        lib.load(cache_path() or None, __version__)
        _LIBRARIES[resources] = lib
    return lib


class SliceDialog(QDialog):
    """File > Slice…: choose what to slice with; `choice()` gives it, `stacks()` the settings.

    `colors` are the colors the model prints with, as `volura.fileio.model_colors` gives them;
    the dialog turns them into the filament plan of `multicolor()`. `parts` is how many parts the
    print has, which is of interest to vase mode alone: Cura spirals every one of them.

    The last choice is restored when the dialog opens and a choice you come back to is saved
    under a name above the Maker row (see `volura.slice_presets`); both go over the dialog
    through `apply_choice()`.
    """

    #: the filament changer belongs to the printer, so it is remembered per printer
    CHANGER = "slice_changer_"
    #: in Settings: the presets of the user, and the name of the one chosen last
    PRESETS, PRESET = "slice_presets", "slice_preset"

    def __init__(self, win, lib: cura_profiles.Library, colors=(), parts=1):
        super().__init__(win)
        self.lib = lib
        self.colors = list(colors)
        self.parts = max(1, int(parts))
        self.went_3mf = False
        self._boxes = []                        # the extruder box of every row of the table
        s = win.settings
        self.settings = s
        self.setWindowTitle(tr("Slice"))
        lay = QFormLayout(self)
        # a whole choice, saved under a name to come back to (see `volura.slice_presets`)
        self.preset_row = QWidget()
        line = QHBoxLayout(self.preset_row)
        line.setContentsMargins(0, 0, 0, 0)
        self.preset = QComboBox()
        self.preset.setToolTip(tr("A printer and everything to print with, kept under a name"))
        self.b_preset_save = QPushButton(tr("Save…"))
        self.b_preset_save.setToolTip(tr("Keep this choice under a name of your own"))
        self.b_preset_delete = QPushButton(tr("Delete"))
        self.b_preset_delete.setToolTip(tr("Forget the preset that is chosen now"))
        self.b_preset_delete.setEnabled(False)
        line.addWidget(self.preset, 1)
        line.addWidget(self.b_preset_save)
        line.addWidget(self.b_preset_delete)
        self.maker = QComboBox()
        self.printer_box = QComboBox()
        self.nozzle = QComboBox()
        self.brand = QComboBox()
        self.material = QComboBox()
        self.quality = QComboBox()
        self.infill = QSpinBox()
        self.infill.setRange(0, 100)
        self.infill.setSuffix(" %")
        self.infill.setToolTip(tr("How full the inside of the print is"))
        self.pattern = QComboBox()
        for label, _kind in INFILL_PATTERNS:
            self.pattern.addItem(tr(label))
        self.pattern.setToolTip(tr("The shape the inside of the print is filled with: gyroid and "
                                   "cubic are strong in every direction, lines and zig zag print "
                                   "fastest, lightning only holds up the top"))
        self.support = QComboBox()
        for label, _on, _kind in SUPPORTS:
            self.support.addItem(tr(label))
        self.adhesion = QComboBox()
        for label, _kind in ADHESION:
            self.adhesion.addItem(tr(label))
        # the extras on top of the profile, with Cura's own setting names (checked against the
        # fdmprinter definitions of the Cura the profiles come from); off or 0 leaves it to the profile
        self.extras = QGroupBox(tr("Extras"))
        extras = QFormLayout(self.extras)
        self.adaptive = QCheckBox(tr("Adaptive layer height"))
        self.adaptive.setToolTip(tr("Thinner layers where the surface slopes, thicker where it "
                                    "is steep; faster with smoother curves"))
        self.ironing = QCheckBox(tr("Ironing"))
        self.ironing.setToolTip(tr("The nozzle goes over the top surface once more with very "
                                   "little filament, so a flat top comes out smooth"))
        self.fuzzy = QCheckBox(tr("Fuzzy skin"))
        self.fuzzy.setToolTip(tr("A rough, textured outer wall that hides layer lines and "
                                 "grips well"))
        self.fuzz = QDoubleSpinBox()
        self.fuzz.setRange(FUZZ_MIN, FUZZ_MAX)
        self.fuzz.setDecimals(2)
        self.fuzz.setSingleStep(0.05)
        self.fuzz.setSuffix(" mm")
        self.fuzz.setToolTip(tr("How thick the bumps of the textured outer wall are"))
        self.walls = QSpinBox()
        self.walls.setRange(WALLS_MIN, WALLS_MAX)
        self.walls.setSpecialValueText(tr("Profile"))
        self.walls.setToolTip(tr("How many nozzle lines thick the walls of the print are; "
                                 "Profile leaves it to the profile"))
        self.seam = QComboBox()
        for label, _kind, _extra in SEAMS:
            self.seam.addItem(tr(label))
        self.seam.setToolTip(tr("Where the thin vertical line of the print sits: at a corner, "
                                "at the back, moved around, or wherever the nozzle has the "
                                "shortest way"))
        self.vase = QCheckBox(tr("Vase mode (one spiral wall)"))
        self.vase.setToolTip(tr("One wall that spirals up in one continuous line, with a bottom "
                               "under it: a vase, without infill or a top"))
        self.thumbnail = QComboBox()
        for label, kind, _sizes in gcode_thumbnail.THUMBNAILS:
            self.thumbnail.addItem(tr(label), kind)
        self.thumbnail.setToolTip(tr("A picture of the print in the G-code, for the screen of "
                                     "the printer"))
        extras.addRow("", self.adaptive)
        extras.addRow("", self.ironing)
        extras.addRow("", self.fuzzy)
        extras.addRow(tr("Fuzz"), self.fuzz)
        extras.addRow(tr("Walls"), self.walls)
        extras.addRow(tr("Seam"), self.seam)
        extras.addRow("", self.vase)
        extras.addRow(tr("Thumbnail"), self.thumbnail)
        # what the print does with the colors of the model
        self.paint_depth = QDoubleSpinBox()
        self.paint_depth.setRange(PAINT_MIN, PAINT_MAX)
        self.paint_depth.setSingleStep(0.1)
        self.paint_depth.setSuffix(" mm")
        self.paint_depth.setValue(float(getattr(s, "slice_paint", PAINT_DEFAULT)))
        self.paint_depth.setToolTip(tr("How far into the part a face painted with the Paint "
                                       "Bucket prints with the filament of its own color"))
        self.changer = QCheckBox(tr("Filament changer on one nozzle"))
        self.changer.setToolTip(tr("A Prusa MMU, an ERCF or another changer that feeds several "
                                   "filaments through the one nozzle: the print asks for every "
                                   "filament with T0, T1, ... like the changer's firmware wants"))
        self.filaments = QSpinBox()
        self.filaments.setRange(FILAMENTS_MIN, FILAMENTS_MAX)
        self.filaments.setToolTip(tr("How many filaments the changer feeds through the nozzle"))
        self.table = QTableWidget(0, 2)
        self.table.setAccessibleName(tr("Colour -> extruder"))
        self.table.setHorizontalHeaderLabels([tr("Colour"), tr("Extruder")])
        self.table.verticalHeader().setVisible(False)
        self.table.setIconSize(QSize(16, 16))
        self.table.setColumnWidth(0, 180)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.table.setMaximumHeight(132)
        self.warn = QLabel()
        self.warn.setWordWrap(True)
        self.tower = QCheckBox(tr("Prime tower"))
        self.tower.setToolTip(tr("A block next to the print that the nozzle wipes itself clean "
                                 "on every time it goes over to another extruder"))
        self.ooze = QCheckBox(tr("Ooze shield"))
        self.ooze.setToolTip(tr("A thin wall around the print that catches what drips from the "
                                "nozzle while it is not printing"))
        self.single = QWidget()               # for a printer that has no second filament at all
        line = QHBoxLayout(self.single)
        line.setContentsMargins(0, 0, 0, 0)
        self.single_note = QLabel(tr("This printer has one extruder: the parts print in one "
                                     "colour. Use a colour change by height or export a "
                                     "multicolour 3MF."))
        self.single_note.setWordWrap(True)
        self.b_3mf = QPushButton(tr("Export Multicolor 3MF…"))
        line.addWidget(self.single_note, 1)
        line.addWidget(self.b_3mf)
        self.changes = QListWidget()
        self.changes.setMaximumHeight(76)
        self.changes.setToolTip(tr("The heights at which the printer stops to change its "
                                   "filament by hand"))
        self.change_row = QWidget()
        line = QHBoxLayout(self.change_row)
        line.setContentsMargins(0, 0, 0, 0)
        self.change_at = QDoubleSpinBox()
        self.change_at.setRange(0.05, 1000.0)
        self.change_at.setDecimals(2)
        self.change_at.setSingleStep(1.0)
        self.change_at.setSuffix(" mm")
        self.change_at.setToolTip(tr("Add a filament change at this height above the bed"))
        self.b_add = QPushButton(tr("Add"))
        self.b_remove = QPushButton(tr("Remove"))
        line.addWidget(self.change_at)
        line.addWidget(self.b_add)
        line.addWidget(self.b_remove)
        line.addStretch(1)
        self.change_cmd = QLineEdit(slicer.FILAMENT_CHANGE)
        self.change_cmd.setToolTip(tr("The G-code that asks the printer to change its filament"))
        self.info = QLabel()
        self.info.setWordWrap(True)
        lay.addRow(tr("Preset"), self.preset_row)
        lay.addRow(tr("Maker"), self.maker)
        lay.addRow(tr("Printer"), self.printer_box)
        lay.addRow(tr("Nozzle"), self.nozzle)
        lay.addRow(tr("Filament brand"), self.brand)
        lay.addRow(tr("Material"), self.material)
        lay.addRow(tr("Quality"), self.quality)
        lay.addRow(tr("Infill"), self.infill)
        lay.addRow(tr("Infill pattern"), self.pattern)
        lay.addRow(tr("Support"), self.support)
        lay.addRow(tr("Bed adhesion"), self.adhesion)
        lay.addRow(self.extras)
        lay.addRow(tr("Colour depth of painted faces"), self.paint_depth)
        lay.addRow("", self.changer)
        lay.addRow(tr("Filaments"), self.filaments)
        lay.addRow(tr("Colour -> extruder"), self.table)
        lay.addRow("", self.warn)
        lay.addRow("", self.tower)
        lay.addRow("", self.ooze)
        lay.addRow("", self.single)
        lay.addRow(tr("Change filament at"), self.changes)
        lay.addRow(tr("Change height"), self.change_row)
        lay.addRow(tr("Filament change command"), self.change_cmd)
        lay.addRow("", self.info)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.button(QDialogButtonBox.Ok).setText(tr("Slice…"))
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addRow(bb)
        self._form = lay
        self.printer = None
        self._remembered = {}         # what the cascade picks back up of the choice applied
        self._preset = None           # the preset that stands in the combo box, None on "—"
        self.preset_note = ""         # what that preset has to say about itself
        self._applying = False        # the dialog is setting its own fields

        self.machines = lib.machines
        self.maker.addItems(sorted({m[2] for m in self.machines}, key=str.lower))
        self.maker.currentIndexChanged.connect(self._maker_changed)
        self.printer_box.currentIndexChanged.connect(self._printer_changed)
        self.nozzle.currentIndexChanged.connect(self._nozzle_changed)
        self.brand.currentIndexChanged.connect(self._brand_changed)
        self.material.currentIndexChanged.connect(self._material_changed)
        self.quality.currentIndexChanged.connect(self._show_info)
        self.changer.toggled.connect(self._changer_changed)
        self.filaments.valueChanged.connect(self._changer_changed)
        self.fuzzy.toggled.connect(self._fuzzy_changed)
        self._fuzzy_changed(self.fuzzy.isChecked())
        self.vase.toggled.connect(self._vase_changed)
        self._vase_changed(self.vase.isChecked())
        self.b_add.clicked.connect(self._add_change)
        self.b_remove.clicked.connect(self._remove_change)
        self.b_3mf.clicked.connect(self._to_3mf)
        # a field that says something else than the chosen preset puts the combo box back on "—"
        for box in (self.maker, self.printer_box, self.nozzle, self.brand, self.material,
                    self.quality, self.pattern, self.support, self.adhesion, self.seam,
                    self.thumbnail):
            box.currentIndexChanged.connect(self._field_changed)
        for box in (self.infill, self.fuzz, self.walls):
            box.valueChanged.connect(self._field_changed)
        for box in (self.adaptive, self.ironing, self.fuzzy, self.vase):
            box.toggled.connect(self._field_changed)
        self.preset.currentIndexChanged.connect(self._preset_changed)
        self.b_preset_save.clicked.connect(self._save_preset)
        self.b_preset_delete.clicked.connect(self._delete_preset)
        self.change_cmd.setText(str(getattr(s, "slice_change_command", "")
                                    or slicer.FILAMENT_CHANGE))
        self._set_changes(parse_heights(getattr(s, "slice_changes", "")))
        self.apply_choice(self._last_choice())
        self._fill_presets(self._matching_preset())

    # ---- the whole choice: the one of the last slice, and presets of the user
    def _last_choice(self) -> dict:
        """What the last slice used, as it stands in Settings; `choice()` gives it back."""
        s = self.settings
        return {"printer": str(getattr(s, "slice_printer", "") or ""),
                "variant": str(getattr(s, "slice_variant", "") or ""),
                "material": str(getattr(s, "slice_material", "") or ""),
                "quality": str(getattr(s, "slice_quality", "") or ""),
                # the defaults are those of Settings, for a window that has nothing to remember
                "infill": getattr(s, "slice_infill", 20),
                "support": getattr(s, "slice_support", 0),
                "adhesion": str(getattr(s, "slice_adhesion", "") or ""),
                "infill_pattern": str(getattr(s, "slice_infill_pattern", "") or ""),
                "adaptive": getattr(s, "slice_adaptive", False),
                "ironing": getattr(s, "slice_ironing", False),
                "fuzzy": getattr(s, "slice_fuzzy", False),
                "fuzz": getattr(s, "slice_fuzz", FUZZ_DEFAULT),
                "walls": getattr(s, "slice_walls", WALLS_MIN),
                "seam": str(getattr(s, "slice_seam", "") or ""),
                "vase": getattr(s, "slice_vase", False),
                "thumbnail": (str(getattr(s, "slice_thumbnail", "") or "")
                              or gcode_thumbnail.DEFAULT)}

    def apply_choice(self, choice: dict):
        """Put a choice over the dialog: the one of the last slice when the dialog opens, a
        preset when one is chosen; one code path for both.

        A key that `choice` does not hold leaves its field as it is, so a preset of an older
        version that knows nothing about the number of walls keeps the number that stands
        there now. What the print does with the colours of the model is not part of a choice:
        those belong to the model and stay as the dialog has them."""
        self._applying = True
        # what the cascade of the printer picks back up once it is built again
        self._remembered = {"variant": self.nozzle.currentText(),
                            "material": str(self.material.currentData() or ""),
                            "quality": str(self.quality.currentData() or "")}
        for key in ("variant", "material", "quality"):
            if key in choice:
                self._remembered[key] = str(choice[key] or "")
        self.infill.setValue(_number(choice.get("infill"), self.infill.value(), 0, 100))
        self.support.setCurrentIndex(_number(choice.get("support"), self.support.currentIndex(),
                                             0, len(SUPPORTS) - 1))
        self.adhesion.setCurrentIndex(_index_of(ADHESION, choice.get("adhesion"),
                                                self.adhesion.currentIndex()))
        self.pattern.setCurrentIndex(_index_of(INFILL_PATTERNS, choice.get("infill_pattern"),
                                               self.pattern.currentIndex()))
        if "adaptive" in choice:
            self.adaptive.setChecked(bool(choice["adaptive"]))
        if "ironing" in choice:
            self.ironing.setChecked(bool(choice["ironing"]))
        if "fuzzy" in choice:
            self.fuzzy.setChecked(bool(choice["fuzzy"]))
        self.fuzz.setValue(_number(choice.get("fuzz"), self.fuzz.value(), FUZZ_MIN, FUZZ_MAX,
                                   whole=False))
        self.walls.setValue(_number(choice.get("walls"), self.walls.value(), WALLS_MIN,
                                    WALLS_MAX))
        self.seam.setCurrentIndex(_index_of(SEAMS, choice.get("seam"), self.seam.currentIndex()))
        if "vase" in choice:
            self.vase.setChecked(bool(choice["vase"]))
        self.thumbnail.setCurrentIndex(_index_of(gcode_thumbnail.THUMBNAILS,
                                                 choice.get("thumbnail"),
                                                 self.thumbnail.currentIndex()))
        wanted = next((m for m in self.machines if m[0] == str(choice.get("printer") or "")),
                      None)
        if wanted is not None:
            self.maker.blockSignals(True)
            self.maker.setCurrentText(wanted[2])
            self.maker.blockSignals(False)
        self._maker_changed()             # the cascade: printer, nozzle, material, quality
        if wanted is not None:
            i = self.printer_box.findData(wanted[0])
            if i >= 0:
                self.printer_box.setCurrentIndex(i)
        self._applying = False

    def _presets(self) -> list:
        """The presets of the user, as they stand in Settings (volura.slice_presets)."""
        return slice_presets.from_lines(self.settings.q.value(self.PRESETS, []) or [])

    def _store_presets(self, presets):
        """Keep the presets of the user, also for what the window saves before it slices."""
        self.settings.slice_presets = presets
        self.settings.q.setValue(self.PRESETS, slice_presets.to_lines(presets))
        self.settings.q.sync()

    def preset_name(self):
        """The name of the preset that stands in the combo box; None on "—" (no preset)."""
        i = self.preset.currentIndex()
        return self.preset.itemText(i) if i > 0 else None

    def _fill_presets(self, want=""):
        """The combo box of presets again: "—" and every name that is kept, `want` chosen when
        it is one of them; what the dialog holds is left alone."""
        self._applying = True
        self.preset.clear()
        self.preset.addItem(tr("—"))                          # nothing chosen
        for preset in self._presets():
            self.preset.addItem(str(preset["name"]))
        self.preset.setCurrentIndex(max(0, self.preset.findText(want, Qt.MatchExactly))
                                    if want else 0)
        self._applying = False
        name = self.preset_name()
        self._preset = slice_presets.find(self._presets(), name) if name else None
        self.b_preset_delete.setEnabled(self._preset is not None)

    def _show_preset(self, name):
        """Put the preset `name` (None for "—") in the combo box and remember it as the choice
        of the last time; the fields of the dialog stay as they are."""
        self._fill_presets(name or "")
        self._remember_preset(name if self._preset is not None else "")

    def _remember_preset(self, name):
        """Which preset was chosen last, so that the next dialog can preselect it again."""
        self.settings.slice_preset = name
        self.settings.q.setValue(self.PRESET, name)
        self.settings.q.sync()

    def _matching_preset(self) -> str:
        """The name of the preset chosen last, but only when the dialog now holds what that
        preset holds; otherwise the combo box stays on "—"."""
        name = str(getattr(self.settings, "slice_preset", "") or "")
        preset = slice_presets.find(self._presets(), name) if name else None
        if preset is None or not slice_presets.matches(preset, self.choice()):
            return ""
        return name

    def _printer_gone(self, preset) -> bool:
        """Whether a preset asks for a printer that is no longer in the profile library."""
        did = str(preset.get("printer") or "")
        return bool(did) and not any(m[0] == did for m in self.machines)

    def _preset_changed(self, *_):
        """A preset is chosen from the combo box: everything it holds goes over the dialog."""
        if self._applying:
            return
        name = self.preset_name()
        preset = slice_presets.find(self._presets(), name) if name else None
        self.preset_note = ""
        self._remember_preset(name or "")
        self.b_preset_delete.setEnabled(preset is not None)
        if preset is None:
            self._show_info()
            return
        if self._printer_gone(preset):
            self._preset = None         # the fields stay as they are: the printer is gone
            self.preset_note = tr("This preset is for a printer that is no longer in the "
                                  "list, so nothing changed.")
            self._show_info()
            return
        self._preset = preset
        self.apply_choice(preset)

    def _field_changed(self, *_):
        """A field moved, and with it possibly away from the chosen preset. The preset itself
        never changes, so the combo box goes back to "—" - only when the fields really say
        something else than the preset does."""
        if self._applying or self._preset is None:
            return
        if not slice_presets.matches(self._preset, self.choice()):
            self._show_preset(None)

    def _save_preset(self):
        """Keep this choice under a name of your own; a name that is taken is asked about."""
        text, ok = QInputDialog.getText(self, tr("Save Preset"), tr("Name of this preset"))
        name = str(text or "").strip()
        if not ok or not name:
            return
        kept = self._presets()
        if slice_presets.find(kept, name) is not None:
            answer = QMessageBox.question(self, tr("Save Preset"),
                                          tr("There is already a preset called {name}. "
                                             "Replace it?").format(name=name))
            if answer != QMessageBox.Yes:
                return
        got = self.choice()
        preset = {"name": name}
        preset.update({key: got[key] for key in slice_presets.PRESET_KEYS})
        self._store_presets([item for item in kept if item["name"] != name] + [preset])
        self._show_preset(name)

    def _delete_preset(self):
        """Forget the chosen preset, asking first; what the dialog holds stays on the screen."""
        name = self.preset_name()
        if name is None:
            return
        answer = QMessageBox.question(self, tr("Delete Preset"),
                                      tr("Remove the preset {name}?").format(name=name))
        if answer != QMessageBox.Yes:
            return
        self._store_presets([item for item in self._presets() if item["name"] != name])
        self._show_preset(None)

    # ---- the cascade: maker -> printer -> nozzle -> material -> quality
    def _maker_changed(self, *_):
        maker = self.maker.currentText()
        self.printer_box.blockSignals(True)
        self.printer_box.clear()
        for did, name, m in self.machines:
            if m == maker:
                self.printer_box.addItem(name, did)
        self.printer_box.blockSignals(False)
        self._printer_changed()

    def _printer_changed(self, *_):
        did = self.printer_box.currentData()
        self.printer = self.lib.printer(did) if did else None
        self._load_changer()                # what this printer does with more than one filament
        self.nozzle.blockSignals(True)
        self.nozzle.clear()
        if self.printer is not None:
            for v in self.printer.variants:
                self.nozzle.addItem(v.name, v.id)
            default = self.printer.default_variant()
            name = self._remembered.get("variant") or (default.name if default else "")
            i = self.nozzle.findText(name)
            if i < 0 and default is not None:
                i = self.nozzle.findText(default.name)
            self.nozzle.setCurrentIndex(max(0, i))
        self.nozzle.blockSignals(False)
        self._form.setRowVisible(self.nozzle, self.nozzle.count() > 0)
        self._nozzle_changed()
        self._colors_changed()              # what this printer can do with the colors

    def _variant(self):
        if self.printer is None:
            return None
        vid = self.nozzle.currentData()
        return next((v for v in self.printer.variants if v.id == vid), None)

    def _nozzle_changed(self, *_):
        """The brands of the materials this nozzle takes; the remembered or default one chosen."""
        self.brand.blockSignals(True)
        self.brand.clear()
        wanted = None
        if self.printer is not None:
            variant = self._variant()
            mats = self.printer.materials(variant)
            brands = sorted({m.metadata["brand"] for m in mats},
                            key=lambda b: (b != "Generic", b.lower()))
            self.brand.addItems(brands)
            wanted = next((m for m in mats if m.id == self._remembered.get("material")), None)
            if wanted is None:
                wanted = self.printer.default_material(variant)
            if wanted is not None:
                self.brand.setCurrentText(wanted.metadata["brand"])
        self.brand.blockSignals(False)
        self._form.setRowVisible(self.brand, self.brand.count() > 0)
        self._brand_changed(want=wanted.id if wanted is not None else None)

    def _brand_changed(self, *_, want=None):
        """The materials of the chosen brand."""
        self.material.blockSignals(True)
        self.material.clear()
        if self.printer is not None:
            brand = self.brand.currentText()
            for m in self.printer.materials(self._variant()):
                if m.metadata["brand"] == brand:
                    label = m.name[len(brand):].strip() if m.name.startswith(brand) else m.name
                    self.material.addItem(label or m.name, m.id)
            i = self.material.findData(want) if want else -1
            if i < 0:
                i = self.material.findData(self._remembered.get("material"))
            self.material.setCurrentIndex(max(0, i))
        self.material.blockSignals(False)
        self._form.setRowVisible(self.material, self.material.count() > 0)
        self._material_changed()

    def _material(self):
        if self.printer is None:
            return None
        mid = self.material.currentData()
        return next((m for m in self.printer.materials(self._variant()) if m.id == mid), None)

    def _material_changed(self, *_):
        self.quality.blockSignals(True)
        self.quality.clear()
        if self.printer is not None:
            variant, material = self._variant(), self._material()
            for qt, name, lh in self.printer.qualities(variant, material):
                label = name if lh != lh else tr("{name} ({lh:.2f} mm)").format(name=name, lh=lh)
                self.quality.addItem(label, qt)
            i = self.quality.findData(self._remembered.get("quality"))
            if i < 0:
                i = self.quality.findData(self.printer.default_quality(variant, material))
            self.quality.setCurrentIndex(max(0, i))
        self.quality.blockSignals(False)
        self._show_info()

    def _show_info(self, *_):
        """What the printer of the choice can do, with the note of a preset and the one of vase
        mode on top of it."""
        note = "\n".join(part for part in (self.preset_note, self._vase_note()) if part)
        if self.printer is None:
            self.info.setText(note)
            return
        st = self.stacks()
        g = st.glob
        height = float(g.get("machine_height") or 0)
        if height > 0:
            self.change_at.setMaximum(height)
        text = tr("Build volume {w:g} × {d:g} × {h:g} mm, {f:g} mm filament").format(
            w=float(g.get("machine_width") or 0),
            d=float(g.get("machine_depth") or 0),
            h=float(g.get("machine_height") or 0),
            f=float(st.extruders[0].get("material_diameter") or 0) if st.extruders else 0)
        self.info.setText(f"{note}\n{text}" if note else text)

    # ---- the colors of the model
    def extruder_count(self) -> int:
        """How many extruders this prints with: the printer's own, or the filaments of a
        changer in front of its one nozzle."""
        if self.printer is None:
            return 1
        own = self.printer.extruder_count
        return self.filaments.value() if own == 1 and self.changer.isChecked() else own

    def _load_changer(self):
        """The filament changer of the printer chosen now; see `CHANGER`."""
        want = 0
        if self.printer is not None:
            try:
                want = int(float(self.settings.q.value(self.CHANGER + self.printer.id, 0)))
            except (TypeError, ValueError):
                want = 0
        self.changer.blockSignals(True)
        self.filaments.blockSignals(True)
        self.filaments.setValue(max(FILAMENTS_MIN, min(FILAMENTS_MAX, want)))
        self.changer.setChecked(self.printer is not None and self.printer.extruder_count == 1
                                and want >= FILAMENTS_MIN)
        self.changer.blockSignals(False)
        self.filaments.blockSignals(False)

    def _remember_changer(self):
        """How many filaments hang in front of this printer's nozzle, for the next time."""
        if self.printer is None:
            return
        self.settings.q.setValue(self.CHANGER + self.printer.id,
                                 self.filaments.value() if self.changer.isChecked() else 0)
        self.settings.q.sync()

    def _changer_changed(self, *_):
        self._colors_changed()
        self._show_info()
        self._remember_changer()

    def _fuzzy_changed(self, on):
        """The thickness of the fuzz is of interest only when there is a fuzzy skin."""
        self.fuzz.setEnabled(bool(on))

    def _vase_changed(self, on):
        """A vase is one wall that spirals up with a bottom under it: what goes inside the print,
        how many lines its walls take, ironing and support have nothing to say about it. Their
        values stay as they are, so a vase that goes off prints what it printed before."""
        for field in (self.infill, self.support, self.ironing, self.walls):
            field.setEnabled(not on)
        self._show_info()

    def _vase_note(self) -> str:
        """What vase mode has to say about the print: what it prints, and that Cura spirals every
        part of a print of more than one part on its own - which is allowed, only worth knowing."""
        if not self.vase.isChecked():
            return ""
        note = tr("Vase mode prints one wall and a bottom, without infill or a top.")
        if self.parts > 1:
            note += " " + tr("There are {n} parts, so each of them spiralizes on its own.").format(
                n=self.parts)
        return note

    def _colors_changed(self):
        """The color table and the filament choices, as the printer chosen now can use them."""
        n, one = self.extruder_count(), self.printer is not None
        one = one and self.printer.extruder_count == 1
        many = len(self.colors) > 1
        self._form.setRowVisible(self.changer, one)
        self._form.setRowVisible(self.filaments, one and self.changer.isChecked())
        self._form.setRowVisible(self.table, many and n > 1)
        self._form.setRowVisible(self.tower, n > 1)
        self._form.setRowVisible(self.ooze, n > 1)
        self._form.setRowVisible(self.single, many and n == 1)
        self._form.setRowVisible(self.changes, n == 1)
        self._form.setRowVisible(self.change_row, n == 1)
        self._form.setRowVisible(self.change_cmd, n == 1)
        self.tower.setChecked(n > 1)
        self._fill_colors()

    def _fill_colors(self):
        """One row per color of the model: the swatch, its name and the extruder that prints it."""
        n = self.extruder_count()
        self.table.clearContents()
        self.table.setRowCount(len(self.colors))
        self._boxes = []
        for row, (name, color, number) in enumerate(self.colors):
            item = QTableWidgetItem(name)
            item.setIcon(swatch(str(color)))     # an unknown color shows as an empty square
            item.setToolTip(tr("Which extruder prints the parts and faces of this color"))
            self.table.setItem(row, 0, item)
            box = QComboBox()
            box.addItems([str(e) for e in range(1, n + 1)])
            box.setCurrentIndex((number - 1) % n)     # the filament order of the multicolor 3MF
            box.setToolTip(tr("Which extruder prints the parts and faces of this color"))
            self.table.setCellWidget(row, 1, box)
            self._boxes.append(box)
        wrap = n > 1 and len(self.colors) > n
        self.warn.setText(tr("{n} colors on {e} extruders: the colors that do not fit wrap "
                             "around and print with the same filament.").format(
                                 n=len(self.colors), e=n) if wrap else "")
        self._form.setRowVisible(self.warn, wrap)

    def _set_changes(self, heights):
        """The list of heights of a colour change by height, shortest first."""
        self.changes.clear()
        for h in sorted(heights):
            item = QListWidgetItem(f"{float(h):g} mm")
            item.setData(Qt.UserRole, float(h))
            self.changes.addItem(item)

    def _listed_changes(self) -> list:
        """Every height in the list, whether or not this printer can use it."""
        return [float(self.changes.item(i).data(Qt.UserRole)) for i in range(self.changes.count())]

    def _add_change(self):
        self._set_changes(self._listed_changes() + [self.change_at.value()])

    def _remove_change(self):
        rows = self._listed_changes()
        row = self.changes.currentRow()
        if 0 <= row < len(rows):
            del rows[row]
        self._set_changes(rows)

    def _to_3mf(self):
        """A printer with one extruder prints the colors as a multicolor 3MF instead."""
        self.went_3mf = True
        self.reject()

    def change_heights(self) -> list:
        """The heights of a colour change by height (mm); nothing when the printer changes its
        filament by itself."""
        return [] if self.extruder_count() > 1 else self._listed_changes()

    def multicolor(self) -> Multicolor:
        """What this slice does with the colors of the model, as they are chosen here."""
        mapping = {}
        for row, (_name, color, _number) in enumerate(self.colors):
            box = self._boxes[row] if row < len(self._boxes) else None
            mapping[str(color).lower()] = box.currentIndex() if box is not None else 0
        return Multicolor(self.extruder_count(), mapping, self.paint_depth.value(),
                          self.tower.isChecked(), self.ooze.isChecked(), self.change_heights(),
                          self.change_cmd.text().strip(), self.colors)

    # ---- the result
    def user_values(self) -> dict:
        """Your own settings on top of the profile."""
        _label, support, kind = SUPPORTS[self.support.currentIndex()]
        # a vase is one wall spiralling up with a bottom under it: the infill, the walls, ironing
        # and support say nothing about it, so those fields stand aside and stay with the profile
        vase = self.vase.isChecked()
        values = ({"magic_spiralize": True} if vase else
                  {"infill_sparse_density": float(self.infill.value()),
                   "support_enable": support, "support_type": kind})
        adhesion = ADHESION[self.adhesion.currentIndex()][1]
        if adhesion:
            values["adhesion_type"] = adhesion
        pattern = INFILL_PATTERNS[self.pattern.currentIndex()][1]
        if pattern and not vase:
            values["infill_pattern"] = pattern
        if self.adaptive.isChecked():
            values["adaptive_layer_height_enabled"] = True
        if self.ironing.isChecked() and not vase:
            values["ironing_enabled"] = True
            # the smooth top of a part, not every flat step in it
            values["ironing_only_highest_layer"] = True
        if self.fuzzy.isChecked():
            values["magic_fuzzy_skin_enabled"] = True
            values["magic_fuzzy_skin_thickness"] = float(self.fuzz.value())
        if self.walls.value() and not vase:
            values["wall_line_count"] = int(self.walls.value())
        _label, seam, extra = SEAMS[self.seam.currentIndex()]
        if seam:
            values["z_seam_type"] = seam
            values.update(extra)
        return values

    def choice(self) -> dict:
        """What to remember: the printer and what was chosen for it."""
        variant = self._variant()
        return {"printer": self.printer.id if self.printer else "",
                "variant": variant.name if variant else "",
                "material": self.material.currentData() or "",
                "quality": self.quality.currentData() or "",
                "infill": self.infill.value(), "support": self.support.currentIndex(),
                "adhesion": ADHESION[self.adhesion.currentIndex()][1],
                "infill_pattern": INFILL_PATTERNS[self.pattern.currentIndex()][1],
                "paint": self.paint_depth.value(),
                "changes": heights_text(self._listed_changes()),
                "change_command": self.change_cmd.text().strip(),
                "adaptive": self.adaptive.isChecked(), "ironing": self.ironing.isChecked(),
                "fuzzy": self.fuzzy.isChecked(), "fuzz": self.fuzz.value(),
                "walls": self.walls.value(),
                "seam": SEAMS[self.seam.currentIndex()][1], "vase": self.vase.isChecked(),
                "thumbnail": gcode_thumbnail.THUMBNAILS[self.thumbnail.currentIndex()][1]}

    def stacks(self) -> cura_profiles.Stacks:
        """The settings to slice with, following the colors of the model."""
        st = self.printer.stacks(self._variant(), self._material(),
                                 self.quality.currentData(), self.user_values())
        if self.changer.isChecked():
            cura_profiles.filament_changer(st, self.filaments.value())
        g = st.glob
        st.glob.layers[0].update(self.multicolor().values(
            float(g.get("machine_width") or 0.0), float(g.get("machine_depth") or 0.0)))
        return st

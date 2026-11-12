"""The "Slice…" dialog: the printer, nozzle, material and quality to slice with.

The printers come from the Cura profile library that Volura brings along (see
`volura.cura_profiles`): first the maker, then the printer, then what that printer offers. A few
settings that you change on almost every print sit underneath - infill, support and how the print
sticks to the bed; everything else comes from the profile. The choice is remembered in Settings.
"""
from __future__ import annotations

from PySide6.QtWidgets import (QComboBox, QDialog, QDialogButtonBox, QFormLayout, QLabel,
                               QSpinBox)

from . import cura_profiles
from .i18n import tr, tr_noop

#: one library per resources folder, kept for the session: reading it takes a while
_LIBRARIES: dict[str, cura_profiles.Library] = {}

#: the support choices: (label, support_enable, support_type)
SUPPORTS = [(tr_noop("None"), False, "buildplate"),
            (tr_noop("Touching the bed"), True, "buildplate"),
            (tr_noop("Everywhere"), True, "everywhere")]
#: the bed adhesion choices: (label, adhesion_type or "" for the printer's own)
ADHESION = [(tr_noop("As the printer has it"), ""), (tr_noop("Skirt"), "skirt"),
            (tr_noop("Brim"), "brim"), (tr_noop("Raft"), "raft"), (tr_noop("None"), "none")]


def is_loaded(resources: str) -> bool:
    """Whether the profiles of `resources` were read already in this session."""
    return resources in _LIBRARIES


def library(resources: str) -> cura_profiles.Library:
    """The profile library of `resources`, read completely the first time it is asked for."""
    lib = _LIBRARIES.get(resources)
    if lib is None:
        lib = cura_profiles.Library(resources)
        lib.machines, lib.qualities, lib.variants, lib.materials     # read it all now
        _LIBRARIES[resources] = lib
    return lib


class SliceDialog(QDialog):
    """File > Slice…: choose what to slice with; `choice()` gives it, `stacks()` the settings."""

    def __init__(self, win, lib: cura_profiles.Library):
        super().__init__(win)
        self.lib = lib
        s = win.settings
        self.setWindowTitle(tr("Slice"))
        lay = QFormLayout(self)
        self.maker = QComboBox()
        self.printer_box = QComboBox()
        self.nozzle = QComboBox()
        self.brand = QComboBox()
        self.material = QComboBox()
        self.quality = QComboBox()
        self.infill = QSpinBox()
        self.infill.setRange(0, 100)
        self.infill.setSuffix(" %")
        self.infill.setValue(int(s.slice_infill))
        self.infill.setToolTip(tr("How full the inside of the print is"))
        self.support = QComboBox()
        for label, _on, _kind in SUPPORTS:
            self.support.addItem(tr(label))
        self.support.setCurrentIndex(max(0, min(len(SUPPORTS) - 1, int(s.slice_support))))
        self.adhesion = QComboBox()
        for label, _kind in ADHESION:
            self.adhesion.addItem(tr(label))
        kinds = [k for _label, k in ADHESION]
        self.adhesion.setCurrentIndex(kinds.index(s.slice_adhesion)
                                      if s.slice_adhesion in kinds else 0)
        self.info = QLabel()
        self.info.setWordWrap(True)
        lay.addRow(tr("Maker"), self.maker)
        lay.addRow(tr("Printer"), self.printer_box)
        lay.addRow(tr("Nozzle"), self.nozzle)
        lay.addRow(tr("Filament brand"), self.brand)
        lay.addRow(tr("Material"), self.material)
        lay.addRow(tr("Quality"), self.quality)
        lay.addRow(tr("Infill"), self.infill)
        lay.addRow(tr("Support"), self.support)
        lay.addRow(tr("Bed adhesion"), self.adhesion)
        lay.addRow("", self.info)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.button(QDialogButtonBox.Ok).setText(tr("Slice…"))
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addRow(bb)
        self._form = lay
        self.printer = None

        self.machines = lib.machines
        makers = sorted({m[2] for m in self.machines}, key=str.lower)
        self.maker.addItems(makers)
        wanted = next((m for m in self.machines if m[0] == s.slice_printer), None)
        if wanted is not None:
            self.maker.setCurrentText(wanted[2])
        self.maker.currentIndexChanged.connect(self._maker_changed)
        self.printer_box.currentIndexChanged.connect(self._printer_changed)
        self.nozzle.currentIndexChanged.connect(self._nozzle_changed)
        self.brand.currentIndexChanged.connect(self._brand_changed)
        self.material.currentIndexChanged.connect(self._material_changed)
        self.quality.currentIndexChanged.connect(self._show_info)
        self._remembered = {"variant": s.slice_variant, "material": s.slice_material,
                            "quality": s.slice_quality}
        self._maker_changed()
        if wanted is not None:
            i = self.printer_box.findData(wanted[0])
            if i >= 0:
                self.printer_box.setCurrentIndex(i)

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
        if self.printer is None:
            self.info.setText("")
            return
        st = self.stacks()
        g = st.glob
        self.info.setText(tr("Build volume {w:g} × {d:g} × {h:g} mm, {f:g} mm filament")
                          .format(w=float(g.get("machine_width") or 0),
                                  d=float(g.get("machine_depth") or 0),
                                  h=float(g.get("machine_height") or 0),
                                  f=float(st.extruders[0].get("material_diameter") or 0)
                                  if st.extruders else 0))

    # ---- the result
    def user_values(self) -> dict:
        """Your own settings on top of the profile."""
        _label, support, kind = SUPPORTS[self.support.currentIndex()]
        values = {"infill_sparse_density": float(self.infill.value()),
                  "support_enable": support, "support_type": kind}
        adhesion = ADHESION[self.adhesion.currentIndex()][1]
        if adhesion:
            values["adhesion_type"] = adhesion
        return values

    def choice(self) -> dict:
        """What to remember: the printer and what was chosen for it."""
        variant = self._variant()
        return {"printer": self.printer.id if self.printer else "",
                "variant": variant.name if variant else "",
                "material": self.material.currentData() or "",
                "quality": self.quality.currentData() or "",
                "infill": self.infill.value(), "support": self.support.currentIndex(),
                "adhesion": ADHESION[self.adhesion.currentIndex()][1]}

    def stacks(self) -> cura_profiles.Stacks:
        """The settings to slice with."""
        return self.printer.stacks(self._variant(), self._material(),
                                   self.quality.currentData(), self.user_values())

"""The "Slice for Resin…" dialog: the resin printer, the layers, the light and the supports.

The printers come from the resin profiles that Volura brings along (see `volura.sla_profiles`):
first the maker, then the printer. The layer height picks the print profile of that height, and
Light, Medium or Heavy its supports. The exposure depends on the resin, so it is asked here, with
the defaults of a standard resin on a mono screen; the resin's bottle or its maker's site gives
the right times. The choice is remembered in Settings.
"""
from __future__ import annotations

from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox,
                               QFormLayout, QLabel, QSpinBox)

from .i18n import tr, tr_noop
from .sla_profiles import ResinLibrary, ResinSettings

#: the support levels of the print profiles, with their names on screen
LEVELS = (("Light", tr_noop("Light")), ("Medium", tr_noop("Medium")), ("Heavy", tr_noop("Heavy")))


class ResinDialog(QDialog):
    """File > Slice for Resin…: `printer`, `profile()` and `settings()` give the choice."""

    def __init__(self, win, lib: ResinLibrary):
        super().__init__(win)
        self.lib = lib
        s = win.settings
        self.setWindowTitle(tr("Slice for Resin"))
        lay = QFormLayout(self)
        self.maker = QComboBox()
        self.maker.addItems(lib.makers)
        self.printer_box = QComboBox()
        self.layer = QComboBox()
        for h in lib.layer_heights():
            self.layer.addItem(f"{h:g} mm", h)
        i = min(range(self.layer.count()), key=lambda k: abs(self.layer.itemData(k)
                                                              - s.resin_layer), default=0)
        self.layer.setCurrentIndex(i)

        def spin(lo, hi, value, step, suffix, decimals=1):
            sp = QDoubleSpinBox()
            sp.setRange(lo, hi)
            sp.setDecimals(decimals)
            sp.setSingleStep(step)
            sp.setSuffix(suffix)
            sp.setValue(float(value))
            return sp

        self.exposure = spin(0.1, 120.0, s.resin_exposure, 0.1, " s")
        self.exposure.setToolTip(tr("Light per layer; the bottle of the resin says how long"))
        self.bottom_exposure = spin(1.0, 300.0, s.resin_bottom_exposure, 1.0, " s")
        self.bottom_exposure.setToolTip(tr("Light for the first layers, so the print sticks to "
                                           "the build plate"))
        self.bottom_layers = QSpinBox()
        self.bottom_layers.setRange(1, 20)
        self.bottom_layers.setValue(int(s.resin_bottom_layers))
        self.supports = QCheckBox(tr("Make supports"))
        self.supports.setChecked(bool(s.resin_supports))
        self.level = QComboBox()
        for key, label in LEVELS:
            self.level.addItem(tr(label), key)
        self.level.setCurrentIndex(max(0, self.level.findData(s.resin_level)))
        self.level.setToolTip(tr("How thick the tips and pillars of the supports are"))
        self.density = spin(25.0, 400.0, s.resin_density, 25.0, " %", 0)
        self.density.setToolTip(tr("More support points: more for delicate prints, fewer for "
                                   "easier clean-up"))
        self.elevation = spin(0.0, 30.0, s.resin_elevation, 1.0, " mm")
        self.elevation.setToolTip(tr("How high the part stands above the pad on its supports"))
        self.pad = QCheckBox(tr("Pad under the print"))
        self.pad.setChecked(bool(s.resin_pad))
        self.info = QLabel()
        self.info.setWordWrap(True)
        lay.addRow(tr("Maker"), self.maker)
        lay.addRow(tr("Printer"), self.printer_box)
        lay.addRow(tr("Layer height"), self.layer)
        lay.addRow(tr("Exposure"), self.exposure)
        lay.addRow(tr("Bottom exposure"), self.bottom_exposure)
        lay.addRow(tr("Bottom layers"), self.bottom_layers)
        lay.addRow("", self.supports)
        lay.addRow(tr("Supports"), self.level)
        lay.addRow(tr("Support density"), self.density)
        lay.addRow(tr("Lift off the pad"), self.elevation)
        lay.addRow("", self.pad)
        lay.addRow("", self.info)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.button(QDialogButtonBox.Ok).setText(tr("Supports…"))
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addRow(bb)
        self.printer = None
        wanted = lib.printer(s.resin_printer)
        if wanted is not None:
            self.maker.setCurrentText(wanted.maker)
        self.maker.currentIndexChanged.connect(self._maker_changed)
        self.printer_box.currentIndexChanged.connect(self._printer_changed)
        self.supports.toggled.connect(self._supports_toggled)
        self._maker_changed()
        if wanted is not None:
            self.printer_box.setCurrentIndex(max(0, self.printer_box.findData(wanted.id)))
        self._supports_toggled(self.supports.isChecked())

    def _maker_changed(self, *_):
        maker = self.maker.currentText()
        self.printer_box.blockSignals(True)
        self.printer_box.clear()
        for p in self.lib.printers:
            if p.maker == maker:
                self.printer_box.addItem(p.name, p.id)
        self.printer_box.blockSignals(False)
        self._printer_changed()

    def _printer_changed(self, *_):
        self.printer = self.lib.printer(self.printer_box.currentData())
        p = self.printer
        if p is None:
            self.info.setText("")
            return
        self.info.setText(tr("Build volume {w:g} × {d:g} × {h:g} mm, screen {px} × {py} pixels, "
                             "file .{ext}").format(w=p.width, d=p.depth, h=p.height,
                                                   px=p.pixels_x, py=p.pixels_y, ext=p.extension))

    def _supports_toggled(self, on):
        for w in (self.level, self.density, self.elevation):
            w.setEnabled(on)

    # ---- the result
    def settings(self) -> ResinSettings:
        return ResinSettings(layer_height=float(self.layer.currentData() or 0.05),
                             exposure=self.exposure.value(),
                             bottom_exposure=self.bottom_exposure.value(),
                             bottom_layers=self.bottom_layers.value(),
                             supports=self.supports.isChecked(),
                             support_level=self.level.currentData(),
                             density=self.density.value(), elevation=self.elevation.value(),
                             pad=self.pad.isChecked())

    def profile(self):
        st = self.settings()
        return self.lib.print_profile(st.layer_height, st.support_level)

    def choice(self) -> dict:
        """What to remember, as the resin_* settings."""
        st = self.settings()
        return {"printer": self.printer.id if self.printer else "",
                "level": st.support_level, "layer": st.layer_height,
                "exposure": st.exposure, "bottom_exposure": st.bottom_exposure,
                "bottom_layers": st.bottom_layers, "density": st.density,
                "elevation": st.elevation, "supports": st.supports, "pad": st.pad}

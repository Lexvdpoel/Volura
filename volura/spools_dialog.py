"""Window > Spools…: the rolls of filament on the shelf and how much is still on each.

The list is what the print preview (volura.print_preview) takes its grams from, so a print
that would run out halfway is seen before it starts. One row is one spool: the colour it has
on the spool, the name it is given, what it is made of and how much is still on it, with a
little bar for how full it is.

*Set left…* is how that last number is kept honest. Weigh the spool with its filament on it
and type what the scale says; Volura subtracts the weight of the empty spool, which stands on
the spool itself because a cardboard core weighs something else than a plastic one. The rest
of the fields say what a new spool holds, what it cost and what it is, and are what the
preview uses to put the right spools on top of the list of a filament.

What is put here is kept right away, under `spools` in the settings (volura.spools), the same
way the printers of Window > Preferences > Printers are kept.
"""
from __future__ import annotations

from PySide6.QtCore import QSize
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (QAbstractItemView, QColorDialog, QComboBox, QDialog,
                               QDialogButtonBox, QDoubleSpinBox, QFormLayout, QHBoxLayout,
                               QHeaderView, QLineEdit, QInputDialog, QMessageBox, QProgressBar,
                               QPushButton, QTableWidget, QTableWidgetItem, QVBoxLayout)

from . import spools
from .app_dialogs import FILAMENT_MATERIALS
from .i18n import tr
from .panels import note, swatch


class SpoolEditDialog(QDialog):
    """One spool: what it is called, what it is made of, its colour and what is left on it.

    The weights are net, so the spool's own plastic is never counted; the empty spool weight
    says how heavy an empty one is, which is what *Set left…* subtracts from a scale reading.
    A material that is not one of the usual ones is kept as it was written, so an older spool
    does not quietly become something else.
    """

    def __init__(self, parent, spool=None):
        super().__init__(parent)
        was = spools.record(spool) if spool is not None else spools.Spool()
        self._id = was.id
        self._colour = was.colour
        self.setWindowTitle(tr("Filament spool"))
        lay = QFormLayout(self)

        def grams(value, lo, hi):
            spin = QDoubleSpinBox()
            spin.setRange(lo, hi)
            spin.setDecimals(1)
            spin.setSingleStep(10.0)
            spin.setSuffix(" g")
            spin.setValue(float(value))
            return spin

        self.name = QLineEdit(was.name)
        self.name.setToolTip(tr("The name you know this spool by"))
        self.material = QComboBox()
        for name, _density in FILAMENT_MATERIALS:
            self.material.addItem(name, name)
        self.material.addItem(spools.OTHER, spools.OTHER)
        if self.material.findData(was.material) < 0:
            self.material.addItem(was.material, was.material)
        self.material.setCurrentIndex(max(0, self.material.findData(was.material)))
        self.material.setToolTip(tr("What the filament is made of; the print preview puts the "
                                    "spools of the same material first on a filament"))
        self.colour = QPushButton(self._colour)
        self.colour.setAccessibleName(tr("Colour of the spool"))
        self.colour.setToolTip(tr("The colour this filament has on the spool"))
        self.colour.clicked.connect(self._pick_colour)
        self.full = grams(was.full_g, spools.FULL_MIN, spools.FULL_MAX)
        self.full.setToolTip(tr("What a full spool holds, without the empty spool"))
        self.left = grams(was.left_g, 0.0, spools.FULL_MAX)
        self.left.setToolTip(tr("What is still on it, without the empty spool; weigh it with "
                                "Set left… in the list"))
        self.tare = grams(was.tare_g, spools.TARE_MIN, spools.TARE_MAX)
        self.tare.setSingleStep(10.0)
        self.tare.setToolTip(tr("What the empty spool weighs: Set left… in the list subtracts "
                                "this from what your scale says"))
        self.price = QDoubleSpinBox()
        self.price.setRange(spools.PRICE_MIN, spools.PRICE_MAX)
        self.price.setDecimals(2)
        self.price.setValue(float(was.price))
        self.price.setToolTip(tr("What a full spool cost, so the cost of a print can be worked "
                                 "out from what it used"))
        lay.addRow(tr("Name"), self.name)
        lay.addRow(tr("Material"), self.material)
        lay.addRow(tr("Colour"), self.colour)
        lay.addRow(tr("Full spool"), self.full)
        lay.addRow(tr("What is left"), self.left)
        lay.addRow(tr("Empty spool weight"), self.tare)
        lay.addRow(tr("Price"), self.price)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addRow(bb)
        self._show_colour()

    def _pick_colour(self):
        got = QColorDialog.getColor(QColor(self._colour), self, tr("Colour of the spool"))
        if got.isValid():
            self._colour = got.name()
            self._show_colour()

    def _show_colour(self):
        """The button is its own swatch: its background is the colour it names."""
        self.colour.setText(self._colour)
        ink = "#f2f2f2" if QColor(self._colour).lightness() < 128 else "#1f1f1f"
        self.colour.setStyleSheet(f"background: {self._colour}; color: {ink};")

    def spool(self):
        """What stands in the fields, as a spool."""
        return spools.Spool(id=self._id, name=self.name.text(),
                            material=str(self.material.currentData() or spools.OTHER),
                            colour=self._colour, full_g=self.full.value(),
                            left_g=self.left.value(), price=self.price.value(),
                            tare_g=self.tare.value())


class SpoolsDialog(QDialog):
    """Window > Spools…: every roll of filament on the shelf, with Add, Edit and Set left…

    Changes are kept right away, like the layers panel of the Default Tray keeps its own,
    because a spool is weighed between prints and there is nothing to undo."""

    WIDTHS = (26, 190, 90, 185, 105)     # colour, name, material, what is left, how full

    def __init__(self, win):
        super().__init__(win)
        self.win = win
        self.setWindowTitle(tr("Spools"))
        self.resize(640, 420)
        self.spools = spools.records(getattr(getattr(win, "settings", None), "spools", None))
        lay = QVBoxLayout(self)
        row = QHBoxLayout()
        self.table = QTableWidget(0, len(self.WIDTHS))
        self.table.setAccessibleName(tr("Filament spools"))
        self.table.setHorizontalHeaderLabels(
            ("", tr("Name"), tr("Material"), tr("What is left"), ""))
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setIconSize(QSize(16, 16))
        for col, width in enumerate(self.WIDTHS):
            self.table.setColumnWidth(col, width)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.table.itemSelectionChanged.connect(self._show_buttons)
        self.table.cellDoubleClicked.connect(lambda *_row: self._edit_spool())
        row.addWidget(self.table, 1)
        side = QVBoxLayout()
        self.b_add = QPushButton(tr("Add"))
        self.b_add.setToolTip(tr("Put a new spool on the shelf"))
        self.b_add.clicked.connect(self._add_spool)
        self.b_edit = QPushButton(tr("Edit"))
        self.b_edit.setToolTip(tr("Change the name, the material, the colour or the weights of "
                                  "the chosen spool"))
        self.b_edit.clicked.connect(self._edit_spool)
        self.b_left = QPushButton(tr("Set left…"))
        self.b_left.setToolTip(tr("Weigh the spool with its filament on it and type what the "
                                  "scale says; Volura subtracts the weight of the empty spool"))
        self.b_left.clicked.connect(self._set_left)
        self.b_remove = QPushButton(tr("Remove"))
        self.b_remove.setToolTip(tr("Take the chosen spool off the shelf"))
        self.b_remove.clicked.connect(self._remove_spool)
        for button in (self.b_add, self.b_edit, self.b_left, self.b_remove):
            side.addWidget(button)
        side.addStretch(1)
        row.addLayout(side)
        lay.addLayout(row, 1)
        lay.addWidget(note(tr("The print preview takes what a print uses off the spool you "
                              "point every filament at, and says beforehand whether that spool "
                              "still has enough. Weigh a spool with Set left… and the weight of "
                              "the empty spool is subtracted for you; what you change here is "
                              "kept right away.")))
        bb = QDialogButtonBox(QDialogButtonBox.Close)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)
        self._reload(0)

    # ------------------------------------------------------------------ the rows
    def _bar(self, spool) -> QProgressBar:
        """The little bar on a row: how full this spool still is."""
        bar = QProgressBar()
        bar.setRange(0, 100)
        bar.setFormat("%p %")
        bar.setValue(round(100.0 * spool.left_g / spool.full_g))
        bar.setFixedHeight(16)
        bar.setAccessibleName(tr("How much is left on this spool"))
        bar.setToolTip(spools.left_text(spool))
        return bar

    def _reload(self, row):
        """Fill the table again and show the spool at `row`."""
        row = max(-1, min(int(row), len(self.spools) - 1))
        self.table.blockSignals(True)
        self.table.setCurrentCell(-1, -1)
        self.table.clearSelection()
        self.table.setRowCount(0)
        for spool in self.spools:
            r = self.table.rowCount()
            self.table.insertRow(r)
            self.table.setItem(r, 0, QTableWidgetItem(swatch(spool.colour), ""))
            self.table.setItem(r, 1, QTableWidgetItem(spool.name))
            self.table.setItem(r, 2, QTableWidgetItem(spool.material))
            self.table.setItem(r, 3, QTableWidgetItem(spools.left_text(spool)))
            self.table.setCellWidget(r, 4, self._bar(spool))
        if 0 <= row < self.table.rowCount():
            self.table.selectRow(row)
        self.table.blockSignals(False)
        self._show_buttons()

    def _chosen(self) -> int:
        """The row of the chosen spool, or -1 when there is none."""
        row = self.table.currentRow()
        return row if 0 <= row < len(self.spools) else -1

    def _show_buttons(self):
        """There is something to edit, weigh or remove only with a spool chosen."""
        some = 0 <= self._chosen() < len(self.spools)
        for button in (self.b_edit, self.b_left, self.b_remove):
            button.setEnabled(some)

    # ------------------------------------------------------------------ the buttons
    def _add_spool(self):
        """Add: a new spool on the shelf, ready to fill in; nothing added when cancelled."""
        dlg = SpoolEditDialog(self, spools.Spool(name=tr("New spool")))
        if dlg.exec() != QDialog.Accepted:
            return
        self.spools.append(dlg.spool())
        self._save()
        self._reload(len(self.spools) - 1)

    def _edit_spool(self):
        """Edit: the chosen spool in the fields, back on the row what the fields say."""
        row = self._chosen()
        if row < 0:
            return
        dlg = SpoolEditDialog(self, self.spools[row])
        if dlg.exec() != QDialog.Accepted:
            return
        self.spools[row] = dlg.spool()
        self._save()
        self._reload(row)

    def _remove_spool(self):
        """Remove: the chosen spool off the shelf, asking first; a filament slot that
        remembers it simply has no spool again (volura.spools.find)."""
        row = self._chosen()
        if row < 0:
            return
        name = self.spools[row].name or self.spools[row].id
        answer = QMessageBox.question(self, tr("Remove spool"),
                                      tr("Remove {name} from the spools?").format(name=name))
        if answer != QMessageBox.Yes:
            return
        del self.spools[row]
        self._save()
        self._reload(row)

    def _set_left(self):
        """Set left…: the scale reading of the whole spool in, what is left on it out."""
        row = self._chosen()
        if row < 0:
            return
        spool = self.spools[row]
        question = tr("What does the scale say for {name}? The weight of the empty spool "
                      "({tare} g) is subtracted.").format(name=spool.name or spool.id,
                                                          tare=spools.grams_text(spool.tare_g))
        got, ok = QInputDialog.getDouble(self, tr("Set left"), question,
                                         spool.left_g + spool.tare_g, 0.0, spools.FULL_MAX, 1)
        if not ok:
            return
        self.spools[row] = spools.set_left(spool, got)
        self._save()
        self._reload(row)

    # ------------------------------------------------------------------ keeping it
    def _save(self):
        """The shelf as the settings keep it; the print preview reads it from there."""
        settings = getattr(self.win, "settings", None)
        if settings is None:
            return
        settings.spools = [spools.to_dict(spool) for spool in self.spools]
        settings.save()

"""Window > Print History…: what you printed, and the same print back on the bed again.

One row is one print, newest first: when it was made, what the model it came from was called,
which printer took it, what became of it and what it took in filament and time. Under the list
stands what every print that really ran together cost; a G-code that was only saved has printed
nothing yet and counts with nothing (the same difference the spools of volura.spools make).

*Print Again…* hands the G-code of the chosen print to a printer, with the very flow the print
preview uses (volura.print_preview.send_to_printer), so nothing has to be sliced over and the
printer is asked again the way the preview asks - with the printer and the answer of last time
already filled in. What then becomes of the print goes into the history like everything else.
Of every print that went to a printer Volura kept its G-code itself, because the file the
preview works on does not outlive it; a print of which that file is gone stands grey and has
nothing to print again. *Show File* opens the folder that holds it. *Remove* takes one print off
the list and *Clear All* the whole list, with the G-codes Volura kept itself; what you saved
yourself is never touched.
"""
from __future__ import annotations

import os

from PySide6.QtCore import QUrl
from PySide6.QtGui import QBrush, QDesktopServices, QPalette
from PySide6.QtWidgets import (QAbstractItemView, QDialog, QDialogButtonBox, QHBoxLayout,
                               QHeaderView, QLabel, QMessageBox, QPushButton, QTableWidget,
                               QTableWidgetItem, QVBoxLayout)

from . import print_history, print_preview, printer_link
from .i18n import tr
from .panels import note


class PrintHistoryDialog(QDialog):
    """The prints of volura.print_history, newest first, with Print Again… beside them.

    The list is what the print preview writes when it closes, so a row here is a memory:
    taking one off the list takes nothing of the model and nothing of the files the user
    saved itself. A print can go to a printer again as long as its G-code is still there and
    a printer stands under Window > Preferences > Printers."""

    WIDTHS = (125, 200, 110, 90, 90, 100)       # when, model, printer, how, grams, time

    def __init__(self, win):
        super().__init__(win)
        self.win = win
        self.settings = getattr(win, "settings", None)
        self.printers = list(getattr(self.settings, "printers", None) or [])
        self.entries = self._stored()
        self._job = None                        # the send that is running, if any
        self.setWindowTitle(tr("Print History"))
        self.resize(760, 460)
        lay = QVBoxLayout(self)
        row = QHBoxLayout()
        self.table = QTableWidget(0, len(self.WIDTHS))
        self.table.setAccessibleName(tr("Prints"))
        self.table.setHorizontalHeaderLabels(
            (tr("When"), tr("Model"), tr("Printer"), tr("How"), tr("Filament"), tr("Time")))
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        for col, width in enumerate(self.WIDTHS):
            self.table.setColumnWidth(col, width)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.table.itemSelectionChanged.connect(self._show_buttons)
        self.table.cellDoubleClicked.connect(lambda *_row: self._print_again())
        row.addWidget(self.table, 1)
        side = QVBoxLayout()
        self.b_again = QPushButton(tr("Print Again…"))
        self.b_again.setToolTip(tr("Send the G-code of this print to a printer, the way the "
                                   "print preview does"))
        self.b_again.clicked.connect(self._print_again)
        self.b_show = QPushButton(tr("Show File"))
        self.b_show.setToolTip(tr("Open the folder that holds the G-code of this print"))
        self.b_show.clicked.connect(self._show_file)
        self.b_remove = QPushButton(tr("Remove"))
        self.b_remove.setToolTip(tr("Take this print off the history"))
        self.b_remove.clicked.connect(self._remove_entry)
        self.b_clear = QPushButton(tr("Clear All"))
        self.b_clear.setToolTip(tr("Take every print off the history, with the G-codes Volura "
                                   "kept of them"))
        self.b_clear.clicked.connect(self._clear_all)
        for button in (self.b_again, self.b_show, self.b_remove, self.b_clear):
            side.addWidget(button)
        side.addStretch(1)
        row.addLayout(side)
        lay.addLayout(row, 1)
        self.total = QLabel()
        self.total.setEnabled(False)
        lay.addWidget(self.total)
        lay.addWidget(note(tr("A print that was only saved has printed nothing, so it counts "
                              "with nothing. Of a print that went to a printer Volura keeps the "
                              "G-code, so a good print can go on the bed again without slicing "
                              "it over; what you saved yourself stays where you put it.")))
        bb = QDialogButtonBox(QDialogButtonBox.Close)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)
        self._reload(-1)

    # ------------------------------------------------------------------ the rows
    def _stored(self) -> list:
        """What the settings remember now: another preview may have written a print since."""
        return print_history.records(getattr(self.settings, "print_history", None))

    def _reload(self, row):
        """Fill the list again, newest first, grey of colour where there is nothing to print
        again, and show the print at `row`."""
        row = max(-1, min(int(row), len(self.entries) - 1))
        self.table.blockSignals(True)
        self.table.setCurrentCell(-1, -1)
        self.table.clearSelection()
        self.table.setRowCount(0)
        dim = QBrush(self.palette().color(QPalette.Disabled, QPalette.Text))
        for entry in self.entries:
            r = self.table.rowCount()
            self.table.insertRow(r)
            gone = print_history.gcode_of(entry) is None
            tip = print_history.entry_text(entry)
            if gone:
                tip = f"{tip}\n{tr('The G-code is no longer there')}"
            texts = (print_history.when_text(entry.when), entry.name, entry.printer,
                     print_history.how_text(entry.how),
                     print_history.grams_text(entry.grams) if entry.grams else "",
                     print_history.time_text(entry.minutes) if entry.minutes else "")
            for col, text in enumerate(texts):
                item = QTableWidgetItem(text)
                item.setToolTip(tip)
                if gone:
                    item.setForeground(dim)
                self.table.setItem(r, col, item)
        if 0 <= row < self.table.rowCount():
            self.table.selectRow(row)
        self.table.blockSignals(False)
        self.total.setText(print_history.totals_text(self.entries))
        self._show_buttons()

    def _chosen(self) -> int:
        """The row of the chosen print, or -1 when there is none."""
        row = self.table.currentRow()
        return row if 0 <= row < len(self.entries) else -1

    def _chosen_entry(self):
        """The print on the chosen row; None when there is none."""
        row = self._chosen()
        return self.entries[row] if 0 <= row < len(self.entries) else None

    def _show_buttons(self):
        """There is something to print again only with a G-code that is still there and a
        printer under Preferences; there is something to show or remove with any print."""
        entry = self._chosen_entry()
        where = print_history.gcode_of(entry) if entry is not None else None
        self.b_again.setEnabled(bool(where) and bool(self.printers) and self._job is None)
        self.b_show.setEnabled(where is not None)
        self.b_remove.setEnabled(entry is not None)
        self.b_clear.setEnabled(bool(self.entries))
        self.b_again.setToolTip(self._wait_for(entry, where))

    def _wait_for(self, entry, where) -> str:
        """What Print Again… is waiting for, so the button can say it: a chosen print, a
        G-code that is still there and a printer under Preferences."""
        if entry is None:
            return tr("Choose the print that has to go on the bed again")
        if not entry.gcode:
            return tr("This print has no G-code of its own")
        if where is None:
            return tr("The G-code is no longer there")
        if not self.printers:
            return tr("Add a printer under Window > Preferences > Printers, then send the "
                      "G-code straight from here.")
        return tr("Send the G-code of this print to a printer, the way the print preview does")

    # ------------------------------------------------------------------ the buttons
    def _print_again(self):
        """Print Again…: the G-code of this print to a printer, with the flow of the preview."""
        entry = self._chosen_entry()
        path = print_history.gcode_of(entry) if entry is not None else None
        if entry is None or not path or not self.printers:
            return

        def sent(conn, name, start):
            self._remember(entry, conn.name, start)

        self._job = print_preview.send_to_printer(self, self.printers, path,
                                                 printer_link.file_name(entry.name),
                                                 busy=self._busy, on_sent=sent)

    def _busy(self, running):
        """While a print is on its way to the printer, nothing else goes from here."""
        if not running:
            self._job = None
        self._show_buttons()

    def _remember(self, was, printer: str, start: bool):
        """The print that went to the printer again stands in the history as a print of its
        own, with a copy of the G-code that made it."""
        entry = print_history.Entry(name=was.name,
                                    how=print_history.STARTED if start else print_history.SENT,
                                    printer=printer, grams=was.grams, minutes=was.minutes,
                                    filaments=was.filaments)
        entry = print_history.store(self.settings, entry, source=was.gcode)
        self.entries = self._stored() if self.settings is not None else [entry] + self.entries
        self._reload(0)

    def _show_file(self):
        """Show File: the folder that holds this G-code, in the file manager of the computer."""
        path = print_history.gcode_of(self._chosen_entry())
        if path:
            QDesktopServices.openUrl(QUrl.fromLocalFile(os.path.dirname(os.path.abspath(path))))

    def _remove_entry(self):
        """Remove: this print off the history, asking first; the G-code Volura kept of it goes
        with it, what the user saved itself stays where it is."""
        entry = self._chosen_entry()
        if entry is None:
            return
        row = self._chosen()
        answer = QMessageBox.question(
            self, tr("Remove print"),
            tr("Remove {name} from the print history? The G-code Volura kept of it goes with "
               "it.").format(name=entry.name))
        if answer != QMessageBox.Yes:
            return
        self.entries = print_history.records(print_history.remove(self.entries, entry.id))
        self._keep()
        self._reload(row - 1)

    def _clear_all(self):
        """Clear All: the whole history, asking first, with every G-code of Volura itself."""
        if not self.entries:
            return
        answer = QMessageBox.question(
            self, tr("Clear print history"),
            tr("Clear the print history? The G-codes Volura kept itself go with it; the files "
               "you saved yourself stay."))
        if answer != QMessageBox.Yes:
            return
        print_history.cleared(self.entries)        # takes away the G-codes of Volura itself
        self.entries = []
        self._keep()
        self._reload(-1)

    # ------------------------------------------------------------------ keeping it
    def _keep(self):
        """The history as the settings keep it; the print preview writes onto it there."""
        if self.settings is None or not hasattr(self.settings, "print_history"):
            return
        self.settings.print_history = print_history.from_values(self.entries)
        self.settings.save()

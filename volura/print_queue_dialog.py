"""Window > Print Queue…: the prints that wait for a printer, lined up and started in turn.

The prints you put here wait for the first printer that is free (volura.print_queue): the list
shows them in the order they wait, with *Move Up*, *Move Down* and *Remove*, and under it stand
the printers with what they say they are doing. Every five seconds every printer is asked again
and the queue is looked over; when a printer can take a print, its G-code goes out in the
background and the print starts on it.

**The safety rule of this window is the rule of the whole queue.** A printer that just finished
still has its last print on the bed, and starting the next one would crash the nozzle into it,
so Volura starts nothing on a printer before you pressed *Bed is clear* for that printer. That
button is the only thing that says a bed is clear - a printer never says it of itself - and the
word is taken back the moment Volura starts a print on it, and whenever the printer says it is
*printing* or *paused*: then somebody started something and there is something on the bed. A
printer in *error* or *offline* is never used, and while the queue stands paused nothing is sent
at all - which is how it stands when you open this window, and how it stands again after you
closed it.

The queue works while this window exists: resumed, it keeps asking the printers and starting
prints also when it stands behind another window, because a print that is waiting should not
need your attention. Closing this window stops the queue - no printer is asked and nothing is
started - and puts it back paused, so a queue that is opened again never does something behind
your back. A G-code that is already on its way to a printer is finished off: stopping that
halfway is exactly what leaves a printer printing without the queue knowing it.
"""
from __future__ import annotations

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QBrush, QPalette
from PySide6.QtWidgets import (QAbstractItemView, QFrame, QGridLayout, QHBoxLayout, QHeaderView,
                               QLabel, QMessageBox, QPushButton, QScrollArea, QTableWidget,
                               QTableWidgetItem, QVBoxLayout, QWidget)

from . import background, print_history, print_queue, printer_link, printer_status, theme
from .i18n import tr
from .panels import note

#: how often every printer is asked again and the queue is looked over (seconds)
POLL_SECONDS = printer_status.POLL_SECONDS
#: the colour of every state here, the same as in Window > Printer Status
STATE_COLOUR = printer_status.STATE_COLOUR
#: what the window says when there is no printer to ask
NO_PRINTERS = printer_status.NO_PRINTERS


class QueuePrinterRow(QWidget):
    """One printer of the queue: what it says it is doing, and whether its bed is clear.

    `bed_clear` is the word of the user and of nobody else: only the *Bed is clear* button
    gives it, and what the printer says of itself takes it back (see
    `print_queue.keep_clear`). It starts out not given, because a printer whose bed nobody
    looked at is a printer nothing is started on.
    """

    def __init__(self, parent, conn):
        super().__init__(parent)
        self.conn = conn
        self.job = None                       # the asking on its way, so none is asked twice
        self.seen = "offline"                 # what the printer last said it was doing
        self.bed_clear = False                # the word of the user, never of the printer
        # what the last sending of a print to this printer said; an answer about the state of
        # the printer that comes in after it does not wipe it away
        self.upload_note = ""
        self.name = QLabel(conn.name)
        bold = self.name.font()
        bold.setBold(True)
        self.name.setFont(bold)
        self.name.setToolTip(printer_link.normalise_url(conn.url))
        self.state = QLabel()
        self.bed = QLabel()
        self.bed.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.error = QLabel()
        self.error.setWordWrap(True)
        self.error.setStyleSheet(f"color: {theme.color('muted')};")
        self.error.hide()
        self.clear = QPushButton(tr("Bed is clear"))
        self.clear.setToolTip(tr("Say that nothing of the last print is left on the bed of "
                                 "this printer; only this button says it"))
        self.clear.clicked.connect(self._said_clear)
        lay = QGridLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.name, 0, 0)
        lay.addWidget(self.clear, 0, 1)
        lay.addWidget(self.bed, 1, 0)
        lay.addWidget(self.state, 1, 1)
        lay.addWidget(self.error, 2, 0, 1, 2)
        lay.setColumnStretch(0, 1)
        self._say_state("offline")            # nothing was asked yet, so nothing is known
        self._say_bed()

    # ------------------------------------------------------------------ the asking
    def ask(self):
        """Ask this printer what it is doing.

        A printer whose *answer* is not handed over yet is not asked again - not one that is
        still answering, and not one that answered while the window was busy elsewhere: an older
        answer may not say over a newer word of the same printer."""
        if self.job is not None:
            return
        self.job = printer_status.ask_status(self.conn, self.answered, self.failed)

    def stop(self):
        """What is on its way may come back to nothing: the row is out of use."""
        if self.job is not None and not self.job.done:
            self.job.cancel()
        self.job = None

    def answered(self, status):
        """What the printer says it is doing, on the row - and what that says about its bed."""
        if self.job is None:
            return                       # stopped while it was on its way: nothing to show
        self.job = None
        self._say_state(status.state)
        self.set_bed(print_queue.keep_clear(status.state, self.bed_clear))
        self.show_error(self.upload_note)

    def failed(self, exc):
        """What went wrong stands under the row; the next asking tries again."""
        if self.job is None:
            return
        self.job = None
        self._say_state("offline")
        self.show_error(str(exc))

    def show_error(self, text):
        """What went wrong with this printer, muted under its row like everything else; an
        empty line has nothing to say, so it is not shown."""
        self.error.setText(str(text or ""))
        self.error.setVisible(bool(text))

    def _say_state(self, state):
        """What the printer last said it is doing, in the colour that belongs to it."""
        self.seen = state
        self.state.setText(printer_link.state_label(state))
        self.state.setStyleSheet(f"color: {theme.color(STATE_COLOUR.get(state, 'muted'))};")

    # ------------------------------------------------------------------------ the bed
    def _said_clear(self):
        """*Bed is clear*: the only word that says a bed is clear, and the user's to give."""
        self.set_bed(True)

    def set_bed(self, clear):
        """Whether the bed of this printer counts as clear, on the row and in the queue."""
        self.bed_clear = bool(clear)
        self._say_bed()

    def _say_bed(self):
        """The word about the bed, in the colour that belongs to it."""
        text = tr("Bed clear") if self.bed_clear else tr("Bed is not clear")
        self.bed.setText(text)
        self.bed.setStyleSheet(f"color: {theme.color('ok' if self.bed_clear else 'muted')};")


class PrintQueueWindow(QWidget):
    """Window > Print Queue…: the prints that wait, the printers they wait for, and the switch
    between doing something and standing still.

    Modeless, like Window > Printer Status: a line of prints that runs itself may not stop you
    from working on the model. Volura keeps one of these; calling it again brings the window
    that is already there to the front. The queue is paused from the first moment, so a print
    that was left on it yesterday starts nothing today.
    """

    WIDTHS = (230, 130, 90, 100)          # model, printer, grams, time

    def __init__(self, win):
        super().__init__(win, Qt.Window)
        self.win = win
        self.settings = getattr(win, "settings", None)
        self.items = self._stored()
        self.conns = []
        self.rows = []
        self._job = None                  # the G-code on its way to a printer, never two at once
        self._going = None                # the print and the printer that job is for
        self.paused = True                # nothing is sent before the user says Resume
        self.setWindowTitle(tr("Print Queue"))
        self.resize(740, 560)
        lay = QVBoxLayout(self)

        row = QHBoxLayout()
        self.table = QTableWidget(0, len(self.WIDTHS))
        self.table.setAccessibleName(tr("Prints"))
        self.table.setHorizontalHeaderLabels(
            (tr("Model"), tr("Printer"), tr("Filament"), tr("Time")))
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        for col, width in enumerate(self.WIDTHS):
            self.table.setColumnWidth(col, width)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.table.itemSelectionChanged.connect(self._show_buttons)
        row.addWidget(self.table, 1)
        side = QVBoxLayout()
        self.b_up = QPushButton(tr("Move Up"))
        self.b_up.setToolTip(tr("Let this print wait no longer than the one before it"))
        self.b_up.clicked.connect(lambda: self._move(True))
        self.b_down = QPushButton(tr("Move Down"))
        self.b_down.setToolTip(tr("Let the print below this one go first"))
        self.b_down.clicked.connect(lambda: self._move(False))
        self.b_remove = QPushButton(tr("Remove"))
        self.b_remove.setToolTip(tr("Take this print off the queue; the G-code Volura kept of "
                                    "it goes with it"))
        self.b_remove.clicked.connect(self._remove_item)
        for button in (self.b_up, self.b_down, self.b_remove):
            side.addWidget(button)
        side.addStretch(1)
        row.addLayout(side)
        lay.addLayout(row, 1)

        printers = QVBoxLayout()
        title = QLabel(tr("Printers"))
        bold = title.font()
        bold.setBold(True)
        title.setFont(bold)
        printers.addWidget(title)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        body = QWidget()
        self._lines = QVBoxLayout(body)
        self._lines.setSpacing(12)
        self._lines.addStretch(1)
        scroll.setWidget(body)
        printers.addWidget(scroll, 1)
        self.hint = QLabel()
        self.hint.setWordWrap(True)
        self.hint.setEnabled(False)
        printers.addWidget(self.hint)
        lay.addLayout(printers, 1)

        controls = QHBoxLayout()
        self.pause = QPushButton(tr("Resume"))
        self.pause.setCheckable(True)
        self.pause.setToolTip(tr("Start the queue, or let it stand still; it stands still from "
                                 "the first moment"))
        controls.addWidget(self.pause)
        self.line = QLabel()
        self.line.setEnabled(False)
        controls.addWidget(self.line, 1)
        lay.addLayout(controls)
        self.safety = note(tr(
            "Volura only starts a print on a printer after you pressed Bed is clear for it, and "
            "takes that word back as soon as it starts one there or the printer says it is "
            "printing or paused - a printer that just finished still has its last print on the "
            "bed. Nothing is sent while the queue stands paused. The queue works while this "
            "window is open, also when it stands behind another window; closing this window "
            "stops it and puts it back paused."))
        lay.addWidget(self.safety)
        self.timer = QTimer(self)
        self.timer.setInterval(POLL_SECONDS * 1000)
        self.timer.timeout.connect(self.poll)
        self.pause.toggled.connect(self._toggled)      # only now: the queue of before is no user
        self.reload()
        self._reload(-1)

    # ------------------------------------------------------------------ what is queued
    def _stored(self) -> list:
        """What the settings remember now: another preview may have added a print since."""
        return print_queue.records(getattr(self.settings, "print_queue", None))

    def _keep(self):
        """The queue as the settings keep it; a preview writes onto it there."""
        if self.settings is None or not hasattr(self.settings, "print_queue"):
            return
        self.settings.print_queue = print_queue.from_values(self.items)
        self.settings.save()

    def _reload(self, row):
        """Fill the list again, in the order the prints wait, grey of colour where there is
        nothing to send, and show the print at `row`."""
        row = max(-1, min(int(row), len(self.items) - 1))
        self.table.blockSignals(True)
        self.table.setCurrentCell(-1, -1)
        self.table.clearSelection()
        self.table.setRowCount(0)
        dim = self.palette().color(QPalette.Disabled, QPalette.Text)
        for item in self.items:
            r = self.table.rowCount()
            self.table.insertRow(r)
            gone = print_queue.gcode_of(item) is None
            tip = print_queue.item_text(item)
            if gone:
                tip = f"{tip}\n{tr('The G-code is no longer there')}"
            texts = (item.name, print_queue.printer_text(item),
                     print_history.grams_text(item.grams) if item.grams else "",
                     print_history.time_text(item.minutes) if item.minutes else "")
            for col, text in enumerate(texts):
                cell = QTableWidgetItem(text)
                cell.setToolTip(tip)
                if gone:
                    cell.setForeground(QBrush(dim))
                self.table.setItem(r, col, cell)
        if 0 <= row < self.table.rowCount():
            self.table.selectRow(row)
        self.table.blockSignals(False)
        self._show_buttons()

    def _chosen(self) -> int:
        """The row of the chosen print, or -1 when there is none."""
        row = self.table.currentRow()
        return row if 0 <= row < len(self.items) else -1

    def _chosen_item(self):
        """The print on the chosen row; None when there is none."""
        row = self._chosen()
        return self.items[row] if 0 <= row < len(self.items) else None

    def _show_buttons(self):
        """There is something to move only between two prints, and something to remove only
        with a print chosen."""
        row = self._chosen()
        self.b_up.setEnabled(row > 0)
        self.b_down.setEnabled(0 <= row < len(self.items) - 1)
        self.b_remove.setEnabled(self._chosen_item() is not None)
        self._show_line()

    def _show_line(self):
        """Under the buttons: whether the queue runs, and how many prints it still has."""
        if self.paused:
            self.line.setText(tr("The queue stands paused, so nothing is sent"))
        else:
            self.line.setText(print_queue.waiting_text(len(self.items)))

    # ------------------------------------------------------------------ the printers
    def reload(self):
        """The printers under Preferences can have changed: the rows go along with them."""
        conns = list(getattr(self.settings, "printers", None) or [])
        if conns != self.conns:
            self.conns = conns
            for row in self.rows:
                row.stop()
                self._lines.removeWidget(row)
                row.deleteLater()
            self.rows = []
            for conn in conns:
                row = QueuePrinterRow(self, conn)
                self.rows.append(row)
                self._lines.insertWidget(self._lines.count() - 1, row)
                row.show()                # a widget that goes into a layout starts out hidden
        self.hint.setText("" if conns else NO_PRINTERS)

    def _row_of(self, name):
        """The row of the printer of that name; None when it left Preferences meanwhile."""
        for row in self.rows:
            if row.conn.name == str(name):
                return row
        return None

    def _states(self) -> list:
        """Every printer as the queue looks at it: what it said it is doing, and what the user
        said about its bed."""
        return [print_queue.printer_state(row.conn, row.seen, row.bed_clear) for row in self.rows]

    # ------------------------------------------------------------------ the asking and the sending
    def follow(self):
        """The queue of the settings, on the screen.

        A print preview or Window > Print History… can put a print on the queue at any moment,
        also while this window stands behind another window, and a print that waits should not
        need a new trip to Window > Print Queue… first. Nothing is laid out again when nothing
        changed: a list that rebuilds itself every five seconds moves the chosen print out from
        under the user's hand."""
        stored = self._stored()
        if print_queue.from_values(stored) != print_queue.from_values(self.items):
            self.items = stored
            self._reload(self._chosen())
        self.reload()

    def poll(self):
        """Ask every printer what it is doing and start what can start; a printer that is still
        answering waits its turn, and a paused queue sends nothing."""
        self.follow()
        for row in self.rows:
            row.ask()
        self._dispatch()

    def _ready(self) -> list:
        """The prints that have something to send: a G-code that went missing in the meantime
        has nothing to give a printer, so it waits instead of eating the queue."""
        return [item for item in self.items if print_queue.gcode_of(item)]

    def _dispatch(self):
        """One print of the queue to one free printer; while a G-code is on its way no second
        one goes, and while the queue is paused nothing goes at all."""
        if self.paused or self._job is not None:
            return
        got = print_queue.next_job(self._ready(), self._states())
        if got is None:
            return
        item, conn = got
        row = self._row_of(conn.name)
        if row is None:
            return
        self._send(item, row)

    def _send(self, item, row):
        """The G-code of this print to this printer, started right away, in the background."""
        name = printer_link.file_name(item.name)
        conn, path = row.conn, item.gcode
        self._going = (item, row)
        # the word about this bed goes the moment the file goes, not once it has arrived: an
        # upload that fails halfway may still have started the print on the printer
        row.set_bed(False)
        row.upload_note = tr("Sending {file}…").format(file=name)
        row.show_error(row.upload_note)

        def work(report):
            printer_link.upload(conn, path, name, start=True,
                                on_progress=lambda fraction: report(fraction))
            return name

        self._job = background.run(self, work, can_cancel=False,
                                   text=tr("Sending {file} to {printer}…")
                                   .format(file=name, printer=conn.name),
                                   show_dialog=False,     # the queue works while you do
                                   on_done=self._started, on_error=self._wrong)

    def _started(self, name):
        """The print is on its way to the printer and started: it leaves the queue, goes into
        the history and takes the word about that bed with it.

        The bed is not clear any more - there is a print on it now - and the user has to say so
        again before anything else can start there."""
        self._job = None
        going, self._going = self._going, None
        if going is None:
            return
        item, row = going
        entry = print_history.Entry(name=item.name, how=print_history.STARTED,
                                    printer=row.conn.name, grams=item.grams,
                                    minutes=item.minutes)
        print_history.store(self.settings, entry, source=item.gcode)
        # off the queue of the settings, not off the list of a moment ago: a preview or
        # Window > Print History… can have put prints on it while this one was on its way
        self.items = print_queue.records(print_queue.remove(self._stored(), item.id))
        self._keep()
        # the row can be gone - Preferences can have dropped that printer while this G-code was on
        # its way - and so can this window, which is closed and put back paused while the file is
        # still on its way; the print itself has nothing to do with the row it went out on
        if background.alive(row):
            row.set_bed(False)
            row.upload_note = ""
            row.show_error("")
        if background.alive(self):
            self._reload(self._chosen())

    def _wrong(self, exc):
        """What went wrong stands under the printer, the print stays on the queue and the queue
        stands still: a printer that is wrong may not eat the line behind it."""
        self._job = None
        going, self._going = self._going, None
        if going is not None and background.alive(going[1]):
            going[1].upload_note = str(exc)
            going[1].show_error(going[1].upload_note)
        if background.alive(self):
            self._set_paused(True)

    # ------------------------------------------------------------------ paused or running
    def _toggled(self, on):
        """Resume or Pause: the switch of the whole queue, and the only way it starts."""
        self._set_paused(not on)
        if on:
            self.poll()          # resumed: the printers are asked at once, not in five seconds

    def _set_paused(self, paused):
        """The queue stands still or runs; the button says which of the two pressing it does,
        and the line under it says what the queue is doing."""
        self.paused = bool(paused)
        self.pause.blockSignals(True)
        self.pause.setChecked(not self.paused)
        self.pause.setText(tr("Pause") if not self.paused else tr("Resume"))
        self.pause.blockSignals(False)
        self._show_line()

    # ------------------------------------------------------------------ the buttons of the list
    def _move(self, up):
        """Move Up / Move Down: one place in the order of waiting, and the print with it."""
        item = self._chosen_item()
        if item is None:
            return
        self.items = print_queue.records(print_queue.move(self.items, item.id, up))
        self._keep()
        self._reload(self.table.currentRow() + (-1 if up else 1))

    def _remove_item(self):
        """Remove: this print off the queue, asking first; the G-code Volura kept of it goes
        with it, what the user saved itself stays where it is."""
        item = self._chosen_item()
        if item is None:
            return
        row = self._chosen()
        answer = QMessageBox.question(
            self, tr("Remove print"),
            tr("Take {name} off the print queue? The G-code Volura kept of it goes with "
               "it.").format(name=item.name))
        if answer != QMessageBox.Yes:
            return
        self.items = print_queue.records(print_queue.remove(self.items, item.id))
        self._keep()
        self._reload(row - 1)

    # --------------------------------------------------------------- on and off the screen
    def showEvent(self, event):
        """Back on the screen: the asking starts again, and `poll` takes the queue and the
        printers of the settings along; the queue stays paused until the user presses Resume."""
        super().showEvent(event)
        self.poll()
        self.timer.start()

    def closeEvent(self, event):
        """Closed: this window stops the queue. Nothing is asked and nothing is started, and the
        queue is put back paused, so opening it again starts nothing either.

        A G-code that is already on its way to a printer is left alone: stopping that halfway
        can leave a printer printing without the queue having seen it, and that is the one
        thing all of this is meant to prevent."""
        self.timer.stop()
        for row in self.rows:
            row.stop()
        self._set_paused(True)
        super().closeEvent(event)
        if self.win is not None:
            keep = getattr(self.win, "_print_queue", None)
            if keep is self:
                self.win._print_queue = None   # this window is gone: the next one is a new one

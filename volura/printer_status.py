"""Window > Printer Status: what every printer on the network is doing right now.

One line per printer under Window > Preferences > Printers: whether it prints, waits, is
done or answers nothing, which G-code it has, how far it got, how long it has run and
still needs, and how hot the nozzle and the bed are. `volura/printer_link.py` asks the
printers; this window only lays out what comes back.

Every `POLL_SECONDS` each printer is asked again, and only while the window is on the
screen: hidden or closed it asks nothing at all, and a printer whose last asking has not
come back yet is not asked a second time. The asking runs in the background
(volura/background.py) and without a progress dialog, so a printer that is switched off
costs a window that simply stays responsive; what it says wrong stands under its own
line, muted, and the next asking tries again.
"""
from __future__ import annotations

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (QFrame, QGridLayout, QLabel, QProgressBar, QScrollArea,
                               QVBoxLayout, QWidget)

from . import background, printer_link, theme
from .i18n import tr

#: how often every printer is asked again (seconds)
POLL_SECONDS = 5
#: what stands where a printer said nothing
NOTHING = "—"
#: the colour of every state here: what prints is green, what went wrong is red
STATE_COLOUR = {"printing": "ok", "paused": "warn", "error": "error", "idle": "muted",
                "offline": "muted"}
#: what the window says when there is no printer to ask
NO_PRINTERS = tr("There are no printers yet. Add one under Window > Preferences > Printers "
                 "and what it is doing shows up here.")


def _degrees(value) -> str:
    """A temperature as this window says it: whole degrees, a dash when it is not known."""
    return NOTHING if value is None else f"{value:.0f}"


def _times(status) -> str:
    """How long the print has run and how long it still needs: "12 min / 1 h 05 min".

    A printer that is not printing knows neither, and then the line stays empty; one that
    knows only the one shows a dash for the rest."""
    if status.elapsed is None and status.left is None:
        return ""
    went = printer_link.format_duration(status.elapsed) or NOTHING
    left = printer_link.format_duration(status.left) or NOTHING
    return f"{went} / {left}"


def _temps(status) -> str:
    """How hot the nozzle and the bed are and what they should be, as "nozzle 205 / 210 °C"."""
    said = []
    for name, now, want in ((tr("nozzle"), status.nozzle, status.nozzle_target),
                            (tr("bed"), status.bed, status.bed_target)):
        if now is None and want is None:
            continue
        said.append(f"{name} {_degrees(now)} / {_degrees(want)} °C")
    return "   ·   ".join(said)


class PrinterRow(QWidget):
    """One printer on the list: its name, and what the asking in the background came back with."""

    def __init__(self, parent, conn):
        super().__init__(parent)
        self.conn = conn
        self.job = None
        self.name = QLabel(conn.name)
        bold = self.name.font()
        bold.setBold(True)
        self.name.setFont(bold)
        self.name.setToolTip(printer_link.normalise_url(conn.url))
        self.state = QLabel()
        self.state.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.file = QLabel(NOTHING)
        self.bar = QProgressBar()
        self.bar.setRange(0, 100)
        self.bar.setFormat("%p %")
        self.bar.setAccessibleName(tr("How far the print is"))
        self.times = QLabel(NOTHING)
        self.times.setToolTip(tr("how long the print has run / how long it still needs"))
        self.temps = QLabel()
        self.temps.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.error = QLabel()
        self.error.setWordWrap(True)
        self.error.setStyleSheet(f"color: {theme.color('muted')};")
        self.error.hide()
        lay = QGridLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.name, 0, 0)
        lay.addWidget(self.state, 0, 1)
        lay.addWidget(self.file, 1, 0, 1, 2)
        lay.addWidget(self.bar, 2, 0, 1, 2)
        lay.addWidget(self.times, 3, 0)
        lay.addWidget(self.temps, 3, 1)
        lay.addWidget(self.error, 4, 0, 1, 2)
        self._say_state("offline")               # nothing was asked yet, so nothing is known

    # ------------------------------------------------------------------ the asking
    def ask(self):
        """Ask this printer what it is doing; one that is still answering is not asked again."""
        if self.job is not None and not self.job.done:
            return
        conn = self.conn

        def work(report):
            return printer_link.status(conn)

        # no window of its own here: a printer that is slow may not hold up the window it
        # stands in, and a row can go away while its answer is still on its way (see `stop`)
        self.job = background.run(None, work, text=tr("Speaking to the printer…"),
                                  can_cancel=False, show_dialog=False,
                                  on_done=self.answered, on_error=self.failed)

    def stop(self):
        """What is on its way may come back to nothing: the row is out of use."""
        if self.job is not None and not self.job.done:
            self.job.cancel()
        self.job = None

    def answered(self, status):
        """What the printer says it is doing, on the row."""
        if self.job is None:
            return                       # stopped while it was on its way: nothing to show
        self.job = None
        self._say_state(status.state)
        self.error.setText("")
        self.error.hide()
        self.file.setText(status.file or NOTHING)
        self.bar.setValue(0 if status.progress is None else round(status.progress * 100))
        self.times.setText(_times(status))
        self.temps.setText(_temps(status))

    def failed(self, exc):
        """What went wrong stands under the row; the next asking tries again."""
        if self.job is None:
            return
        self.job = None
        self._say_state("offline")
        self.file.setText(NOTHING)
        self.bar.setValue(0)
        self.times.setText("")
        self.temps.setText("")
        self.error.setText(str(exc))
        self.error.show()

    def _say_state(self, state):
        """The state of the printer, in the colour that belongs to it."""
        self.state.setText(printer_link.state_label(state))
        self.state.setStyleSheet(f"color: {theme.color(STATE_COLOUR.get(state, 'muted'))};")


class PrinterStatusWindow(QWidget):
    """Window > Printer Status: a row per printer under Preferences, asked every five seconds.

    Modeless, like the 3D mill dialog: a print that runs next to the model may not stop
    you from working on it. Volura keeps one of these; calling it again brings the window
    that is already there to the front.
    """

    def __init__(self, win):
        super().__init__(win, Qt.Window)
        self.win = win
        self.conns = []
        self.rows = []
        self.setWindowTitle(tr("Printer Status"))
        self.resize(560, 320)
        lay = QVBoxLayout(self)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        body = QWidget()
        self._lines = QVBoxLayout(body)
        self._lines.setSpacing(14)
        self._lines.addStretch(1)
        scroll.setWidget(body)
        lay.addWidget(scroll, 1)
        self.hint = QLabel()
        self.hint.setWordWrap(True)
        self.hint.setEnabled(False)
        lay.addWidget(self.hint)
        self.timer = QTimer(self)
        self.timer.setInterval(POLL_SECONDS * 1000)
        self.timer.timeout.connect(self.poll)
        self.reload()

    def reload(self):
        """The printers under Preferences can have changed: the rows go along with them."""
        conns = list(self.win.settings.printers)
        if conns != self.conns:
            self.conns = conns
            for row in self.rows:
                row.stop()
                self._lines.removeWidget(row)
                row.deleteLater()
            self.rows = []
            for conn in conns:
                row = PrinterRow(self, conn)
                self.rows.append(row)
                self._lines.insertWidget(self._lines.count() - 1, row)
        self.hint.setText("" if conns else NO_PRINTERS)

    def poll(self):
        """Ask every printer what it is doing; one that is still answering waits its turn."""
        for row in self.rows:
            row.ask()

    # --------------------------------------------------------------- on and off screen
    def showEvent(self, event):
        """Back on the screen: the rows follow Preferences and the asking starts again."""
        super().showEvent(event)
        self.reload()
        self.poll()
        self.timer.start()

    def hideEvent(self, event):
        """Off the screen, closed or hidden: nothing asks a printer anything any more."""
        self.timer.stop()
        for row in self.rows:
            row.stop()
        super().hideEvent(event)

"""The laser controller: the board on the cable, driven from the screen.

Laser and Mill > Laser Controller… opens the window. What a laser program has next to its
drawing, Volura has here: the port the board hangs on, where the head is, a pad to jog
with, the origin of the work, the overrides of feed and power while a job runs, and the
job of the worksheet with its progress. Volura makes the G-code (`volura/gcode.py`) and
knows the protocol (`volura/grbl.py`); this window is what turns that into a machine you
can point at a piece of plywood.

The conversation itself lives in `volura/laser.py`: a session and a thread that reads the
cable. This file only shows it, on a timer of `laser.UPDATE_SECONDS` - the thread may not
touch a widget, so the window comes and looks instead of being told. Everything the window
sends goes into the session, which hands it to the board as fast as the board has room.

Closing the window hides it and lets a running job run on; the same menu item brings the
window back. The cable is only let go when the program closes (`shutdown`).
"""
from __future__ import annotations

import os

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import (QComboBox, QDialog, QFileDialog, QFormLayout, QGridLayout,
                               QGroupBox, QHBoxLayout, QLabel, QMessageBox, QPlainTextEdit,
                               QProgressBar, QPushButton, QVBoxLayout)

from . import gcode, grbl, laser, panels
from .i18n import tr
from .settings import qsettings

#: how many lines of what the board said the window keeps
LOG_LINES = 400

#: what the window can send: the same kinds as the export of the worksheet
GCODE_FILTER = "G-code (*.gcode *.nc *.gc *.tap);;All files (*)"

#: the arrows of the jog pad: text, which way it moves, and what to tell a screen reader
PAD = (
    ("↑", 0.0, 1.0, 0.0, tr("Jog one step up the sheet (+Y)")),
    ("↓", 0.0, -1.0, 0.0, tr("Jog one step down the sheet (-Y)")),
    ("←", -1.0, 0.0, 0.0, tr("Jog one step to the left (-X)")),
    ("→", 1.0, 0.0, 0.0, tr("Jog one step to the right (+X)")),
    ("Z ↑", 0.0, 0.0, 1.0, tr("Lift the head (+Z)")),
    ("Z ↓", 0.0, 0.0, -1.0, tr("Lower the head (-Z)")),
)

#: where each arrow stands in the pad: row, column
PAD_PLACE = {"↑": (0, 1), "↓": (2, 1), "←": (1, 0), "→": (1, 2), "Z ↑": (0, 2), "Z ↓": (2, 2)}


def _pos(where) -> str:
    """A position of the board as one line of numbers; a dash when the board has not said."""
    if where is None:
        return "—"
    return "  ".join(f"{name} {v:.3f}" for name, v in zip("XYZ", where))


class LaserWindow(QDialog):
    """The window that drives a GRBL laser: cable, board, jog, origin, overrides, job."""

    def __init__(self, win):
        super().__init__(win)
        self.win = win
        self.setWindowTitle(tr("Laser Controller"))
        self.session = laser.Session()
        self.worker = None
        self.job: list[str] = []
        self.job_name = ""
        self._boot_said = False
        self._messages = 0
        self._errors = 0

        lay = QVBoxLayout(self)
        columns = QHBoxLayout()
        left = QVBoxLayout()
        left.addWidget(self._connection_box())
        left.addWidget(self._board_box())
        left.addWidget(self._jog_box())
        right = QVBoxLayout()
        right.addWidget(self._origin_box())
        right.addWidget(self._override_box())
        right.addWidget(self._job_box())
        columns.addLayout(left)
        columns.addLayout(right)
        lay.addLayout(columns)
        lay.addWidget(self._log_box(), 1)

        self.timer = QTimer(self)
        self.timer.setInterval(int(laser.UPDATE_SECONDS * 1000))
        self.timer.timeout.connect(self.refresh)
        self.timer.start()
        self.refresh_ports()
        self.refresh()

    # ------------------------------------------------------------------ the cable
    def _connection_box(self):
        box = QGroupBox(tr("Connection"))
        form = QFormLayout(box)
        self.port_box = QComboBox()
        self.port_box.setToolTip(tr("The serial port the board hangs on"))
        self.b_refresh = QPushButton(tr("Refresh"))
        self.b_refresh.setToolTip(tr("Look again for the ports of the computer"))
        ports = QHBoxLayout()
        ports.addWidget(self.port_box, 1)
        ports.addWidget(self.b_refresh)
        self.baud_box = QComboBox()
        for rate in grbl.BAUD_RATES:
            self.baud_box.addItem(str(rate), rate)
        self.baud_box.setToolTip(tr("How fast the board talks; 115200 is the default of GRBL 1.1"))
        self.b_connect = QPushButton(tr("Connect"))
        self.b_connect.setToolTip(tr("Open the port and ask the board who it is"))
        self.port_hint = panels.note("")
        form.addRow(tr("Port"), ports)
        form.addRow(tr("Speed"), self.baud_box)
        form.addRow(self.b_connect)
        form.addRow(self.port_hint)
        self.b_refresh.clicked.connect(self.refresh_ports)
        self.b_connect.clicked.connect(self.toggle_connection)
        return box

    def refresh_ports(self):
        """Fill the list of ports again, keeping the one that was chosen before."""
        want = str(self.port_box.currentData()
                   or qsettings().value(laser.PORT_KEY, "") or "")
        self.port_box.clear()
        for device, description in laser.ports():
            self.port_box.addItem(f"{device}  {description}", device)
        i = self.port_box.findData(want)
        if i >= 0:
            self.port_box.setCurrentIndex(i)
        if self.worker is not None:
            return
        if not laser.has_serial():
            self.port_hint.setText(tr("Volura needs the pyserial package to talk to a laser: "
                                      "pip install pyserial"))
        elif self.port_box.count() == 0:
            self.port_hint.setText(tr("No serial port. Is the board on, and is its cable in "
                                      "the computer?"))
        else:
            self.port_hint.setText("")

    @property
    def connected(self) -> bool:
        return self.worker is not None

    def toggle_connection(self):
        if self.connected:
            self.disconnect_board()
        else:
            self.connect_board()

    def connect_board(self):
        """Open the port and hand the cable to the thread.

        A board that was left in the alarm state answers nothing until it is unlocked,
        and nobody knows why it got there, so opening the cable is also the moment to
        reset the board, ask what it is set to and unlock it.
        """
        device = self.port_box.currentData()
        if not device:
            QMessageBox.information(self, tr("Laser Controller"), tr("Choose a port first."))
            return
        try:
            port = laser.open_port(device, self.baud_box.currentData())
        except laser.CableError as e:
            QMessageBox.warning(self, tr("Laser Controller"), str(e))
            return
        qsettings().setValue(laser.PORT_KEY, str(device))
        qsettings().setValue(laser.BAUD_KEY, int(self.baud_box.currentData()))
        self.worker = laser.Worker(port, self.session, poll=laser.UPDATE_SECONDS)
        self.worker.start()
        self.session.realtime(grbl.RESET)
        self.session.send(grbl.SETTINGS)
        self.session.send(grbl.UNLOCK)
        self.refresh()

    def disconnect_board(self):
        """Let go of the cable; a job that is running is stopped first."""
        if self.session.running:
            self.session.stop()
        if self.worker is not None:
            self.worker.shutdown()
            self.worker = None
        self.b_check.setChecked(False)        # check mode ends with the cable too
        self.refresh()

    def shutdown(self):
        """The program is closing: stop the job and let go of the cable."""
        self.timer.stop()
        self.disconnect_board()

    # ------------------------------------------------------------------ the board
    def _board_box(self):
        box = QGroupBox(tr("The board"))
        form = QFormLayout(box)
        self.state_label = QLabel("—")
        self.mpos_label = QLabel("—")
        self.wpos_label = QLabel("—")
        self.feed_label = QLabel("—")
        form.addRow(tr("State"), self.state_label)
        form.addRow(tr("Machine"), self.mpos_label)
        form.addRow(tr("Work"), self.wpos_label)
        form.addRow(tr("Feed"), self.feed_label)
        self.b_unlock = QPushButton(tr("Unlock"))
        self.b_unlock.setToolTip(tr("$X: tell the board it may move again"))
        self.b_home = QPushButton(tr("Home"))
        self.b_home.setToolTip(tr("$H: find the origin of the machine"))
        self.b_check = QPushButton(tr("Check"))
        self.b_check.setCheckable(True)
        self.b_check.setToolTip(tr("$C: plan the job and move nothing; again to stop"))
        self.b_reset = QPushButton(tr("Reset"))
        self.b_reset.setToolTip(tr("Empty what the board has queued; it lands in the alarm state"))
        self.b_settings = QPushButton(tr("Settings"))
        self.b_settings.setToolTip(tr("$$: ask the board what it is set to"))
        row = QHBoxLayout()
        for b in (self.b_unlock, self.b_home, self.b_check, self.b_reset, self.b_settings):
            row.addWidget(b)
        form.addRow(row)
        self.b_unlock.clicked.connect(lambda checked=False: self.session.send(grbl.UNLOCK))
        self.b_home.clicked.connect(lambda checked=False: self.session.send(grbl.HOME))
        self.b_check.clicked.connect(self.toggle_check)
        self.b_reset.clicked.connect(lambda checked=False: self.session.realtime(grbl.RESET))
        self.b_settings.clicked.connect(lambda checked=False: self.session.send(grbl.SETTINGS))
        return box

    def toggle_check(self):
        """Check mode is a switch on the board: the same block turns it on and off."""
        self.session.send(grbl.CHECK)

    # ------------------------------------------------------------------ the pad
    def _jog_box(self):
        box = QGroupBox(tr("Jog"))
        lay = QVBoxLayout(box)
        fields = QFormLayout()
        self.step_box = QComboBox()
        for mm in laser.JOG_STEPS:
            self.step_box.addItem(f"{mm:g} mm", mm)
        self.step_box.setToolTip(tr("How far one click of the pad moves the head"))
        self.feed_box = QComboBox()
        for f in laser.JOG_FEEDS:
            self.feed_box.addItem(tr("{n} mm/min").format(n=f), f)
        self.feed_box.setToolTip(tr("How fast the head moves while jogging"))
        self._remember(self.step_box, laser.STEP_KEY, laser.JOG_STEPS[0])
        self._remember(self.feed_box, laser.FEED_KEY, laser.JOG_FEEDS[0])
        fields.addRow(tr("Step"), self.step_box)
        fields.addRow(tr("Speed"), self.feed_box)
        lay.addLayout(fields)

        pad = QGridLayout()
        self.pad_buttons = {}
        for text, dx, dy, dz, said in PAD:
            b = QPushButton(text)
            b.setAccessibleName(said)
            b.setToolTip(said)
            b.clicked.connect(lambda checked=False, x=dx, y=dy, z=dz: self.jog(x, y, z))
            r, c = PAD_PLACE[text]
            pad.addWidget(b, r, c)
            self.pad_buttons[text] = b
        self.b_home_origin = QPushButton("⌂")
        self.b_home_origin.setAccessibleName(tr("Go to the origin of the work"))
        self.b_home_origin.setToolTip(tr("Rapid to X0 Y0 of the work"))
        self.b_home_origin.clicked.connect(self.go_to_origin)
        pad.addWidget(self.b_home_origin, 1, 1)
        lay.addLayout(pad)
        return box

    def _remember(self, box, key, default):
        """Put the saved choice of a combo box back, and keep it for the next time."""
        try:
            i = [box.itemData(n) for n in range(box.count())].index(float(qsettings()
                                                                         .value(key, default)))
        except (TypeError, ValueError):
            i = 0
        box.setCurrentIndex(max(0, i))
        box.currentIndexChanged.connect(
            lambda _n, b=box, k=key: qsettings().setValue(k, float(b.currentData())))

    def jog(self, dx, dy, dz):
        """One step of the pad: a relative move the board makes in jog mode."""
        step = float(self.step_box.currentData() or 0.0)
        feed = float(self.feed_box.currentData() or 0.0)
        block = grbl.jog(dx * step, dy * step, dz * step, feed)
        if block:
            self.session.send(block)

    def go_to_origin(self):
        self.session.send(grbl.move(x=0.0, y=0.0))

    # ------------------------------------------------------------------ the work origin
    def _origin_box(self):
        box = QGroupBox(tr("Work origin"))
        lay = QVBoxLayout(box)
        row = QHBoxLayout()
        self.b_zero_xy = QPushButton(tr("Zero here"))
        self.b_zero_xy.setToolTip(tr("Put the origin of the work under the head"))
        self.b_zero_x = QPushButton(tr("Zero X"))
        self.b_zero_x.setToolTip(tr("Only the X of the work moves to the head"))
        self.b_zero_y = QPushButton(tr("Zero Y"))
        self.b_zero_y.setToolTip(tr("Only the Y of the work moves to the head"))
        self.b_zero_z = QPushButton(tr("Zero Z"))
        self.b_zero_z.setToolTip(tr("The Z of the work starts at the head"))
        for b in (self.b_zero_xy, self.b_zero_x, self.b_zero_y, self.b_zero_z):
            row.addWidget(b)
        lay.addLayout(row)
        lay.addWidget(panels.note(tr("The head is where the work starts; the job is cut "
                                     "around that point.")))
        self.b_zero_xy.clicked.connect(lambda checked=False: self.zero(x=True, y=True))
        self.b_zero_x.clicked.connect(lambda checked=False: self.zero(x=True, y=False))
        self.b_zero_y.clicked.connect(lambda checked=False: self.zero(x=False, y=True))
        self.b_zero_z.clicked.connect(lambda checked=False: self.zero(x=False, y=False, z=True))
        return box

    def zero(self, x=True, y=True, z=False):
        self.session.send(grbl.zero_here(x=x, y=y, z=z))

    # ------------------------------------------------------------------ overrides
    def _override_box(self):
        box = QGroupBox(tr("Overrides"))
        form = QFormLayout(box)
        self.feed_override_label = QLabel("100%")
        self.power_override_label = QLabel("100%")
        self.rapid_override_label = QLabel("100%")
        form.addRow(tr("Feed"), self._override_row("feed", self.feed_override_label,
                                                   tr("feed rate")))
        form.addRow(tr("Power"), self._override_row("power", self.power_override_label,
                                                    tr("laser power")))
        form.addRow(tr("Rapid"), self._rapid_row())
        return box

    def _override_row(self, kind, shown, what):
        """The three buttons that change one override while the board is moving."""
        row = QHBoxLayout()
        row.addWidget(shown, 1)
        down = QPushButton("-10%")
        down.setAccessibleName(tr("Less {what}").format(what=what))
        up = QPushButton("+10%")
        up.setAccessibleName(tr("More {what}").format(what=what))
        full = QPushButton("100%")
        full.setAccessibleName(tr("{what} as the job says").format(what=what))
        down.clicked.connect(lambda checked=False, k=kind: self.session.override(k, "down10"))
        up.clicked.connect(lambda checked=False, k=kind: self.session.override(k, "up10"))
        full.clicked.connect(lambda checked=False, k=kind: self.session.override(k, "set"))
        for b in (down, up, full):
            row.addWidget(b)
        return row

    def _rapid_row(self):
        row = QHBoxLayout()
        row.addWidget(self.rapid_override_label, 1)
        for name, action in (("50%", "half"), ("25%", "quarter"), ("100%", "set")):
            b = QPushButton(name)
            b.setAccessibleName(tr("Rapid moves at {part}").format(part=name))
            b.clicked.connect(lambda checked=False, a=action: self.session.override("rapid", a))
            row.addWidget(b)
        return row

    # ------------------------------------------------------------------ the job
    def _job_box(self):
        box = QGroupBox(tr("Job"))
        lay = QVBoxLayout(box)
        self.job_label = panels.note(tr("There is no job to send yet."))
        lay.addWidget(self.job_label)
        row = QHBoxLayout()
        self.b_load = QPushButton(tr("From the worksheet…"))
        self.b_load.setToolTip(tr("Make G-code of the layers the laser follows"))
        self.b_open = QPushButton(tr("Open G-code…"))
        self.b_open.setToolTip(tr("Send a G-code file that is already there"))
        self.b_start = QPushButton(tr("Start"))
        self.b_start.setToolTip(tr("Send the job to the board"))
        self.b_pause = QPushButton(tr("Pause"))
        self.b_pause.setToolTip(tr("Feed hold: the head stops and the beam goes off"))
        self.b_stop = QPushButton(tr("Stop"))
        self.b_stop.setToolTip(tr("End the job and empty what the board has queued"))
        for b in (self.b_load, self.b_open, self.b_start, self.b_pause, self.b_stop):
            row.addWidget(b)
        lay.addLayout(row)
        self.bar = QProgressBar()
        self.bar.setRange(0, 100)
        self.bar.setTextVisible(False)
        self.bar.setAccessibleName(tr("How far the job is"))
        lay.addWidget(self.bar)
        self.b_load.clicked.connect(self.load_job)
        self.b_open.clicked.connect(self.open_job)
        self.b_start.clicked.connect(self.start_job)
        self.b_pause.clicked.connect(self.toggle_pause)
        self.b_stop.clicked.connect(lambda checked=False: self.session.stop())
        return box

    def load_job(self):
        """The worksheet as a job, with the layers and the power of the export dialog."""
        options = self.win.sheet_export_options("gcode")
        if not options:
            return
        try:
            drawing = self.win.sheet_export_drawing(options)
            text = gcode.grbl_laser(drawing, s_max=int(options.get("s_max") or 1000),
                                    dynamic=bool(options.get("dynamic", True)))
        except Exception as e:
            QMessageBox.critical(self, tr("Laser Controller"), str(e))
            return
        self.set_job(grbl.job_lines(text), tr("the worksheet"))

    def open_job(self):
        """A G-code file from elsewhere, sent as it stands."""
        path, _ = QFileDialog.getOpenFileName(
            self, tr("Open G-code"), "", GCODE_FILTER)
        if not path:
            return
        try:
            with open(path, encoding="utf-8", errors="replace") as fh:
                text = fh.read()
        except OSError as e:
            QMessageBox.critical(self, tr("Laser Controller"), str(e))
            return
        self.set_job(grbl.job_lines(text), os.path.basename(path))

    def set_job(self, lines, name):
        """The blocks to send, and where they came from."""
        self.job = [str(line) for line in lines]
        self.job_name = str(name)
        self.refresh()

    def start_job(self):
        """Begin the job: the thread sends it as fast as the board has room for it."""
        if not self.job:
            return
        if not self.connected:
            QMessageBox.information(self, tr("Laser Controller"),
                                    tr("Connect to the board first."))
            return
        self.session.start(self.job)

    def toggle_pause(self):
        if self.session.paused:
            self.session.resume()
            self.b_pause.setText(tr("Pause"))
        else:
            self.session.pause()
            self.b_pause.setText(tr("Resume"))

    # ------------------------------------------------------------------ what it says
    def _log_box(self):
        box = QGroupBox(tr("What the board said"))
        lay = QVBoxLayout(box)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setAccessibleName(tr("What the board said"))
        self.log.setMaximumHeight(110)
        self.log.document().setMaximumBlockCount(LOG_LINES)
        lay.addWidget(self.log)
        return box

    def say(self, text):
        self.log.appendPlainText(str(text))

    def _listen(self):
        """Everything the session heard since the window last looked."""
        if self.session.boot and not self._boot_said:
            self._boot_said = True
            self.say(self.session.boot)
        while self._messages < len(self.session.messages):
            self.say(self.session.messages[self._messages])
            self._messages += 1
        while self._errors < len(self.session.errors):
            self.say(tr("Error: {why}").format(why=self.session.errors[self._errors]))
            self._errors += 1
        if self.worker is not None and self.worker.problem:
            self.say(self.worker.problem)
            self.disconnect_board()

    # ------------------------------------------------------------------ every tick
    def refresh(self):
        """Show the session as it is now; the thread is the only one that changes it."""
        s = self.session
        self._listen()
        self.state_label.setText(s.state_label or "—")
        self.mpos_label.setText(_pos(s.mpos))
        self.wpos_label.setText(_pos(s.wpos))
        self.feed_label.setText(tr("{n} mm/min").format(n=int(round(s.feed_rate))))
        self.feed_override_label.setText(f"{int(round(s.feed_override))}%")
        self.power_override_label.setText(f"{int(round(s.power_override))}%")
        self.rapid_override_label.setText(f"{int(round(s.rapid_override))}%")
        self.job_label.setText(tr("{n} blocks from {where}").format(n=len(self.job),
                                                                   where=self.job_name)
                               if self.job else tr("There is no job to send yet."))
        self.bar.setValue(int(round(s.fraction * 100.0)))
        self.b_connect.setText(tr("Disconnect") if self.connected else tr("Connect"))
        self.port_box.setEnabled(not self.connected)
        self.b_refresh.setEnabled(not self.connected)
        self.baud_box.setEnabled(not self.connected)
        free = self.connected and not s.running
        for b in list(self.pad_buttons.values()) + [self.b_home_origin, self.b_zero_xy,
                                                    self.b_zero_x, self.b_zero_y,
                                                    self.b_zero_z, self.b_home,
                                                    self.b_unlock, self.b_settings]:
            b.setEnabled(free)
        self.b_check.setEnabled(self.connected)
        self.b_reset.setEnabled(self.connected)
        self.b_start.setEnabled(self.connected and not s.running and bool(self.job))
        self.b_pause.setEnabled(self.connected and s.running)
        self.b_stop.setEnabled(self.connected and s.running)

    def closeEvent(self, e):
        """Closing the window hides it: a job that runs must not stop because of that."""
        e.ignore()
        self.hide()

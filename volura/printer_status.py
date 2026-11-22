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

What every row last showed stands on that row (`PrinterRow.status`), so the desktop
messages of `volura/print_watch.py` - which watch the same printers also when this window
is closed - can take those answers instead of asking every printer a second time.

A printer whose camera has a still picture to offer has a **Camera** button under its
line, and turning that on puts the picture under the line too, asked with the same
timer. Which of those buttons were on comes back next time the window opens
(`printer_cameras_on` in the settings).
"""
from __future__ import annotations

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (QFrame, QGridLayout, QLabel, QPushButton, QProgressBar,
                               QScrollArea, QVBoxLayout, QWidget)

from . import background, printer_link, theme
from .i18n import tr

#: how often every printer is asked again (seconds)
POLL_SECONDS = 5
#: how wide at most a picture from a printer camera is shown (pixels); its height follows
CAMERA_WIDTH = 320
#: what stands where a printer said nothing
NOTHING = "—"
#: the colour of every state here: what prints is green, what went wrong is red
STATE_COLOUR = {"printing": "ok", "paused": "warn", "error": "error", "idle": "muted",
                "offline": "muted"}
#: what the window says when there is no printer to ask
NO_PRINTERS = tr("There are no printers yet. Add one under Window > Preferences > Printers "
                 "and what it is doing shows up here.")


def key_of(conn) -> tuple:
    """What a printer's camera belongs to in this window: one printer, one camera."""
    return (conn.name, conn.kind, printer_link.normalise_url(conn.url))


def ask_status(conn, on_done, on_error):
    """Ask one printer what it is doing right now, in the background and without a window of
    its own.

    A printer that is slow may not hold up the window it stands in, and the widget that asked
    can be out of use before the answer comes; `on_done(status)` and `on_error(exc)` reach the
    GUI thread. The job comes back, so the caller knows whether an asking is still on its way.
    Window > Printer Status asks this way for every row, and so does Window > Print Queue…
    (volura.print_queue_dialog)."""
    def work(report):
        return printer_link.status(conn)

    return background.run(None, work, text=tr("Speaking to the printer…"),
                          can_cancel=False, show_dialog=False,
                          on_done=on_done, on_error=on_error)


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
    """One printer on the list: its name, and what the asking in the background came back with.

    A printer whose camera has a picture to offer gets a **Camera** button under its line,
    put there by the window as soon as it knows (`set_camera`). Turning that button on shows
    the picture under the line and asks for a new one with every asking of the printer;
    turning it off stops the camera and takes the picture away."""

    def __init__(self, parent, conn, camera=False):
        super().__init__(parent)
        self.conn = conn
        self.job = None
        self.status = None                  # what the printer last said it is doing, so the
                                            # desktop watching can share it (print_watch)
        self.cam = ""                       # where this printer's camera has a picture, if ever
        self.cam_job = None                 # the picture on its way, so none is asked twice
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
        self.camera = QPushButton(tr("Camera"))
        self.camera.setCheckable(True)
        self.camera.setChecked(bool(camera))          # what the settings remembered
        self.camera.setToolTip(tr("Watch the camera of this printer"))
        self.camera.hide()                  # shown as soon as the window knows there is one
        self.picture = QLabel()
        self.picture.hide()
        self.cam_error = QLabel()
        self.cam_error.setWordWrap(True)
        self.cam_error.setStyleSheet(f"color: {theme.color('muted')};")
        self.cam_error.hide()
        lay = QGridLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.name, 0, 0)
        lay.addWidget(self.state, 0, 1)
        lay.addWidget(self.file, 1, 0, 1, 2)
        lay.addWidget(self.bar, 2, 0, 1, 2)
        lay.addWidget(self.times, 3, 0)
        lay.addWidget(self.temps, 3, 1)
        lay.addWidget(self.error, 4, 0, 1, 2)
        lay.addWidget(self.camera, 5, 0)
        lay.addWidget(self.picture, 6, 0, 1, 2)
        lay.addWidget(self.cam_error, 7, 0, 1, 2)
        self._say_state("offline")               # nothing was asked yet, so nothing is known
        self.camera.toggled.connect(self._toggled)   # only now: a restored button is no user

    # ------------------------------------------------------------------ the asking
    def ask(self):
        """Ask this printer what it is doing; one that is still answering is not asked again."""
        if self.job is not None and not self.job.done:
            return
        # no window of its own here: a printer that is slow may not hold up the window it
        # stands in, and a row can go away while its answer is still on its way (see `stop`)
        self.job = ask_status(self.conn, self.answered, self.failed)

    def stop(self):
        """What is on its way may come back to nothing: the row is out of use."""
        for job in (self.job, self.cam_job):
            if job is not None and not job.done:
                job.cancel()
        self.job = None
        self.cam_job = None

    def answered(self, status):
        """What the printer says it is doing, on the row."""
        if self.job is None:
            return                       # stopped while it was on its way: nothing to show
        self.job = None
        self.status = status
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
        self.status = printer_link.Status("offline")
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

    # -------------------------------------------------------------------- the camera
    def set_camera(self, url):
        """What the window found out about the camera of this printer: the button follows.

        A printer with nothing to ask has no button, and one that was being watched stops
        being watched. A row that is already watching asks for its first picture at once, so
        that picture does not wait for the next asking of the printer."""
        self.cam = url or ""
        self.camera.setVisible(bool(self.cam))
        if not self.cam:
            self.camera.setChecked(False)
        elif self.camera.isChecked():
            self.ask_picture()

    def ask_picture(self):
        """Ask the camera of this printer a picture; one that is on its way is not asked again.

        Half of the cameras on a home network stand behind the printer they hang on, so this
        asking carries the same key as every other asking of that printer."""
        if not self.cam or not self.camera.isChecked() or not self.window().isVisible():
            return                        # nothing to ask, not watched, or off the screen
        if self.cam_job is not None and not self.cam_job.done:
            return
        conn, url = self.conn, self.cam

        def work(report):
            return printer_link.snapshot(conn, url)

        self.cam_job = background.run(None, work, text=tr("Speaking to the printer…"),
                                      can_cancel=False, show_dialog=False,
                                      on_done=self.pictured, on_error=self.no_picture)

    def pictured(self, data):
        """What the camera saw, under the row; the next asking of the printer replaces it."""
        if self.cam_job is None:
            return                       # stopped while it was on its way: nothing to show
        self.cam_job = None
        picture = QPixmap()
        picture.loadFromData(data)
        if picture.isNull():             # the camera called it a picture and sent nonsense
            self._camera_wrong(tr("The camera sent nothing that is a picture."))
            return
        if picture.width() > CAMERA_WIDTH:
            picture = picture.scaledToWidth(CAMERA_WIDTH, Qt.SmoothTransformation)
        self.cam_error.setText("")
        self.cam_error.hide()
        self.picture.setPixmap(picture)
        self.picture.show()

    def no_picture(self, exc):
        """What went wrong with the picture stands under the row; the next asking tries again."""
        if self.cam_job is None:
            return
        self.cam_job = None
        self._camera_wrong(str(exc))

    def _camera_wrong(self, text):
        """What the camera got wrong, muted under the row like everything else that went wrong."""
        self.picture.clear()
        self.picture.hide()
        self.cam_error.setText(text)
        self.cam_error.show()

    def _toggled(self, on):
        """Watch the camera or not: the button says which of the two this row does."""
        if on:
            self.ask_picture()
        else:
            self._stop_camera()

    def _stop_camera(self):
        """Nothing is asked of the camera any more, and nothing of it is shown."""
        if self.cam_job is not None and not self.cam_job.done:
            self.cam_job.cancel()
        self.cam_job = None
        self.picture.clear()
        self.picture.hide()
        self.cam_error.setText("")
        self.cam_error.hide()


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
        self._cam_urls = {}               # what every printer said about its camera, kept a while
        self._cam_jobs = {}               # the printers that are answering that right now
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
                row = PrinterRow(self, conn, conn.name in self.watched_cameras())
                row.camera.toggled.connect(self._remember_cameras)
                self.rows.append(row)
                self._lines.insertWidget(self._lines.count() - 1, row)
                row.show()                # a widget that goes into a layout starts out hidden
                if key_of(conn) in self._cam_urls:
                    row.set_camera(self._cam_urls[key_of(conn)])
            if self.isVisible():
                self.look_cameras()       # the printers of Preferences may have brought one more
        self.hint.setText("" if conns else NO_PRINTERS)

    def watched_cameras(self) -> list:
        """The names of the printers whose camera was on when the window was last closed."""
        return list(self.win.settings.printer_cameras_on)

    def poll(self):
        """Ask every printer what it is doing and, when it is watched, what its camera sees;
        one that is still answering waits its turn."""
        for row in self.rows:
            row.ask()
            row.ask_picture()

    # ------------------------------------------------------------------------ cameras
    def look_cameras(self):
        """Ask every printer on the rows where its camera has a picture - each one only once.

        What a printer said is kept for as long as this window lives, so the asking of the
        printers never brings that question back and a printer that moves in and out of sight
        is not asked again. A printer that has nothing to watch, or answered nothing at all,
        is asked again the next time the window is on the screen."""
        for row in self.rows:
            key = key_of(row.conn)
            if key not in self._cam_urls and key not in self._cam_jobs:
                self._look_camera(row.conn)

    def _look_camera(self, conn):
        """Ask one printer where its camera has a picture; what it says belongs to the window."""
        key = key_of(conn)

        def work(report):
            return printer_link.snapshot_url(conn)

        def answered(url):
            if self._cam_jobs.pop(key, None) is None:
                return                    # stopped on its way: nothing is known of it
            if url:
                self._cam_urls[key] = url
            else:                         # nothing to watch: the next time on screen asks again
                self._cam_urls.pop(key, None)
            for row in self.rows:
                if key_of(row.conn) == key:
                    row.set_camera(url)

        def failed(exc):                  # nothing answered: the next time on screen asks again
            self._cam_jobs.pop(key, None)

        self._cam_jobs[key] = background.run(None, work, text=tr("Speaking to the printer…"),
                                             can_cancel=False, show_dialog=False,
                                             on_done=answered, on_error=failed)

    def _remember_cameras(self, checked=False):
        """Which cameras are being watched, named after their printer: back in the settings."""
        self.win.settings.printer_cameras_on = [row.conn.name for row in self.rows
                                                if row.camera.isChecked()]
        self.win.settings.save()

    # --------------------------------------------------------------- on and off screen
    def showEvent(self, event):
        """Back on the screen: the rows follow Preferences and the asking starts again."""
        super().showEvent(event)
        self.reload()
        self.look_cameras()
        self.poll()
        self.timer.start()

    def hideEvent(self, event):
        """Off the screen, closed or hidden: nothing asks a printer anything any more."""
        self.timer.stop()
        for job in self._cam_jobs.values():
            if not job.done:
                job.cancel()
        self._cam_jobs.clear()
        for row in self.rows:
            row.stop()
        super().hideEvent(event)

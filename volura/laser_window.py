"""The laser controller: the board on the cable, driven from the screen.

Laser and Mill > Laser Controller… opens the window. What a laser program has next to its
drawing, Volura has here: the port the board hangs on, where the head is, a pad to jog
with, the origin of the work, the overrides of feed and power while a job runs, and the
job of the worksheet with its progress. Volura makes the G-code (`volura/gcode.py` for a
laser and `volura/gcode_mill.py` for a mill) and knows the protocol (`volura/grbl.py`);
this window is what turns that into a machine you can point at a piece of plywood.

The conversation itself lives in `volura/laser.py`: a session and a thread that reads the
cable. This file only shows it, on a timer of `laser.UPDATE_SECONDS` - the thread may not
touch a widget, so the window comes and looks instead of being told. Everything the window
sends goes into the session, which hands it to the board as fast as the board has room.

Closing the window hides it and lets a running job run on; the same menu item brings the
window back. The cable is only let go when the program closes (`shutdown`).
"""
from __future__ import annotations

import math
import os

import numpy as np

from PySide6.QtCore import QTimer
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage
from PySide6.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox, QDialog,
                               QDialogButtonBox, QDoubleSpinBox, QFileDialog, QFormLayout,
                               QFrame, QGridLayout, QScrollArea, QWidget,
                               QGroupBox, QHBoxLayout, QHeaderView, QLabel, QLineEdit,
                               QListWidget, QMessageBox, QPlainTextEdit, QProgressBar,
                               QPushButton, QSpinBox, QTableWidget, QTableWidgetItem,
                               QVBoxLayout)

from . import gcode, gcode_mill, grbl, laser, laser_image, panels, sheet_image
from . import laser_profiles as profiles
from .i18n import tr
from .settings import qsettings
from .sheet_dialogs import (GCODE_DYNAMIC_KEY, GCODE_S_MAX_KEY, fmt_time,
                           gcode_export_defaults)

#: how many lines of what the board said the window keeps
LOG_LINES = 400

#: how many ticks of the window the blocks of connecting wait for the board to say who it
#: is: an Arduino restarts when its port opens and hears nothing for a second or two
BOOT_TICKS = int(round(3.0 / laser.UPDATE_SECONDS))

#: how wide the last picture was engraved, mm
IMAGE_WIDTH_KEY = "laser/image_width"


def _grid(buttons, columns=3) -> QGridLayout:
    """Buttons in rows of `columns`: the controller stands in a narrow column next to the
    view, and one long row of buttons would be cut off at its edge."""
    grid = QGridLayout()
    grid.setHorizontalSpacing(4)
    grid.setVerticalSpacing(4)
    for i, b in enumerate(buttons):
        grid.addWidget(b, i // columns, i % columns)
    return grid


def _float(value, default) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


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


class PresetsDialog(QDialog):
    """The presets of the user: everything a board answered to `$$`, saved under a name.

    A board knows more about itself than the list of machines does: the feed it allows,
    the acceleration it was built for, the travel it was measured with. So a preset keeps
    every number the board answered, and gives them back to a board of the same machine.

    The numbers of the calibration (`$100` and further) and of the limits are kept too -
    nothing is thrown away - but they do not go to a board unless the user ticks them,
    because those are the numbers that belong to one machine and to no other.
    """

    def __init__(self, win):
        super().__init__(win)
        self.controller = win
        self.kind = profiles.CNC if getattr(win, "is_cnc", False) else profiles.LASER
        self.setWindowTitle(tr("CNC presets") if self.kind == profiles.CNC
                            else tr("Laser presets"))
        self.resize(560, 420)
        lay = QHBoxLayout(self)

        left = QVBoxLayout()
        left.addWidget(QLabel(tr("Saved presets")))
        self.list = QListWidget()
        self.list.setAccessibleName(tr("Saved presets"))
        self.list.currentItemChanged.connect(self._chosen)
        left.addWidget(self.list, 1)
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText(tr("Name of this machine"))
        self.name_edit.setToolTip(tr("Under what name the settings of the board are kept"))
        left.addWidget(self.name_edit)
        self.b_save = QPushButton(tr("Save"))
        self.b_save.setToolTip(tr("Keep the settings of the board under that name"))
        self.b_delete = QPushButton(tr("Forget"))
        self.b_delete.setToolTip(tr("Remove the chosen preset"))
        row = QHBoxLayout()
        row.addWidget(self.b_save)
        row.addWidget(self.b_delete)
        left.addLayout(row)
        lay.addLayout(left, 1)

        right = QVBoxLayout()
        right.addWidget(QLabel(tr("What the preset holds")))
        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels([tr("Send"), tr("Setting"), tr("Value")])
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.DoubleClicked | QAbstractItemView.SelectedClicked)
        self.table.setAccessibleName(tr("What the preset holds"))
        right.addWidget(self.table, 1)
        self.hint = panels.note(tr("The steps per millimetre and the limits belong to one "
                                   "machine: only send them to the machine they came from."))
        right.addWidget(self.hint)
        self.b_apply = QPushButton(tr("Send this preset to the board"))
        self.b_apply.clicked.connect(self.apply_preset)
        right.addWidget(self.b_apply)
        self.b_export = QPushButton(tr("Export…"))
        self.b_export.setToolTip(tr("Write the presets to a file, to give to someone else"))
        self.b_export.clicked.connect(self.export_presets)
        self.b_import = QPushButton(tr("Import…"))
        self.b_import.setToolTip(tr("Read presets from a file"))
        self.b_import.clicked.connect(self.import_presets)
        row = QHBoxLayout()
        row.addWidget(self.b_export)
        row.addWidget(self.b_import)
        row.addStretch(1)
        right.addLayout(row)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        buttons.accepted.connect(self.accept)
        right.addWidget(buttons)
        lay.addLayout(right, 2)

        self.b_save.clicked.connect(self.save_preset)
        self.b_delete.clicked.connect(self.delete_preset)
        key = str(self.controller.machine_box.currentData() or "")
        self.reload(profiles.presets(self.kind), key[len(profiles.USER_PREFIX):]
                    if profiles.is_user_key(key) else None)
        self.b_apply.setEnabled(self.controller.connected)

    # ------------------------------------------------------------------ what is on the screen
    def reload(self, items, want=None):
        """Fill the list again and show one of the presets."""
        self.list.blockSignals(True)
        self.list.clear()
        for item in items:
            self.list.addItem(str(item.get("name", "")))
        self.list.blockSignals(False)
        i = self.list.findItems(str(want or ""), Qt.MatchExactly)
        if i:
            self.list.setCurrentItem(i[0])
        elif self.list.count():
            self.list.setCurrentRow(0)
        else:
            self.show_settings({})
        self._refresh_buttons()

    def chosen_name(self) -> str:
        item = self.list.currentItem()
        return item.text() if item is not None else ""

    def _chosen(self, *_):
        preset = profiles.find_preset(self.chosen_name(), self.kind)
        self.name_edit.setText(self.chosen_name())
        self.show_settings(preset or {})
        self._refresh_buttons()

    def show_settings(self, preset: dict):
        """One row for every setting the preset holds, with the board's own on top."""
        settings = {int(n): float(v) for n, v in (preset.get("settings") or {}).items()}
        for n, v in (self.controller.session.settings or {}).items():
            settings.setdefault(int(n), float(v))
        send = {int(n) for n in (preset.get("send") or [])}
        self.table.setRowCount(len(settings))
        for row, n in enumerate(sorted(settings)):
            check = QTableWidgetItem()
            check.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
            check.setCheckState(Qt.Checked if n in send else Qt.Unchecked)
            check.setToolTip(tr("Send this setting to the board"))
            self.table.setItem(row, 0, check)
            what = QTableWidgetItem(profiles.setting_name(n))
            what.setData(Qt.UserRole, n)
            if profiles.is_risky(n):
                what.setToolTip(tr("This belongs to one machine: the steps per millimetre "
                                   "and the limits"))
            self.table.setItem(row, 1, what)
            value = QTableWidgetItem(f"{settings[n]:g}")
            self.table.setItem(row, 2, value)
        self.table.setColumnWidth(0, 54)

    def _rows(self) -> list:
        """The rows of the table as ``(number, value, send)``."""
        out = []
        for row in range(self.table.rowCount()):
            out.append((self._number(row), self.table.item(row, 2).text(),
                        self.table.item(row, 0).checkState() == Qt.Checked))
        return out

    def _number(self, row) -> int:
        """The number of the setting of a row; the row keeps it with itself."""
        return int(self.table.item(row, 1).data(Qt.UserRole))

    def _refresh_buttons(self):
        has = bool(self.chosen_name())
        self.b_delete.setEnabled(has)
        self.b_apply.setEnabled(self.controller.connected and has)

    # ------------------------------------------------------------------ what the buttons do
    def save_preset(self):
        """Keep what is on the screen under the name that stands above it."""
        name = self.name_edit.text().strip()
        if not name:
            QMessageBox.information(self, self.windowTitle(), tr("A name first."))
            return
        settings, send = {}, []
        for number, value, on in self._rows():
            try:
                settings[str(number)] = float(str(value).replace(",", "."))
            except ValueError:
                QMessageBox.warning(self, self.windowTitle(),
                                    tr("{what} is not a number.").format(
                                        what=profiles.setting_name(number)))
                return
            if on:
                send.append(number)
        x, y = self.controller.travel_size()
        machine = self.controller.machine
        profiles.add_preset(profiles.new_preset(
            name, settings, x, y, s_max=profiles.s_max_of(
                {int(n): v for n, v in settings.items()}, machine.s_max),
            dynamic=machine.dynamic, jog=machine.jog, rapid=machine.rapid,
            baud=int(self.controller.baud_box.currentData() or 115200), send=send),
            self.kind)
        self.controller.refresh_machines()
        self.reload(profiles.presets(self.kind), name)

    def delete_preset(self):
        name = self.chosen_name()
        if not name:
            return
        answer = QMessageBox.question(
            self, self.windowTitle(), tr("Forget {name}?").format(name=name),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No)
        if answer == QMessageBox.StandardButton.Yes:
            profiles.remove_preset(name, self.kind)
            self.controller.refresh_machines()
            self.reload(profiles.presets(self.kind))

    def apply_preset(self):
        """Every setting that is ticked, to the board."""
        preset = profiles.find_preset(self.chosen_name(), self.kind)
        if preset is None:
            return
        lines = profiles.preset_lines(preset, [n for n, _v, on in self._rows() if on],
                                      kind=self.kind)
        if not lines:
            return
        answer = QMessageBox.question(
            self, self.windowTitle(),
            tr("Set the board to this?\n\n{lines}").format(lines="\n".join(lines)),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes)
        if answer != QMessageBox.StandardButton.Yes:
            return
        self.controller.send_settings(lines)

    def export_presets(self):
        path, _ = QFileDialog.getSaveFileName(self, tr("Export presets"), f"volura-{self.kind}.json",
                                              "JSON (*.json)")
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(profiles.presets_to_json(profiles.presets(self.kind)))
        except OSError as e:
            QMessageBox.critical(self, self.windowTitle(), str(e))

    def import_presets(self):
        path, _ = QFileDialog.getOpenFileName(self, tr("Import presets"), "", "JSON (*.json)")
        if not path:
            return
        try:
            with open(path, encoding="utf-8", errors="replace") as fh:
                items = profiles.presets_from_json(fh.read())
        except OSError as e:
            QMessageBox.critical(self, self.windowTitle(), str(e))
            return
        for item in items:
            profiles.add_preset(item, self.kind)
        self.controller.refresh_machines()
        self.reload(profiles.presets(self.kind))


def image_gray(path, max_pixels=4000000):
    """The grey values of a picture on disk, with what is transparent laid over white.

    QImage reads the file - it knows PNG, JPEG, BMP, GIF, TIFF and the rest without
    another package in the build - and the pixels go to numpy from there. A picture of
    forty million dots engraves for a day and needs more blocks than a board can hold in
    its head, so a bigger one is made smaller first, keeping its shape.
    """
    image = QImage(str(path))
    if image.isNull():
        raise ValueError(tr("{file} is not a picture Volura can read.").format(
            file=os.path.basename(str(path))))
    if image.width() * image.height() > max_pixels:
        scale = math.sqrt(max_pixels / float(image.width() * image.height()))
        image = image.scaled(max(1, int(image.width() * scale)),
                             max(1, int(image.height() * scale)),
                             Qt.KeepAspectRatio, Qt.SmoothTransformation)
    has_alpha = image.hasAlphaChannel()
    image = image.convertToFormat(QImage.Format_RGBA8888)
    width, height, line = image.width(), image.height(), image.bytesPerLine()
    data = np.frombuffer(image.constBits(), dtype=np.uint8,
                         count=line * height).reshape(height, line)[:, :width * 4]
    data = data.reshape(height, width, 4).astype(float)
    if has_alpha:
        alpha = data[:, :, 3:4] / 255.0
        data = data[:, :, :3] * alpha + 255.0 * (1.0 - alpha)
    return laser_image.to_gray(data)



class LaserWindow(QWidget):
    """The machine on the cable, next to the view: cable, board, jog, origin, job.

    It stands in the Laser and in the CNC tab under the viewer (`volura/workspaces.py`),
    one of each. `kind` says which: a laser has the presets of the known lasers and the
    engraving of a picture; a mill has neither, and its power is the spindle. Both keep
    their own port and settings, so a laser and a mill can hang on the same computer.
    """

    def __init__(self, win, kind="laser"):
        super().__init__(win)
        self.win = win
        self.kind = "cnc" if str(kind) == "cnc" else "laser"
        self.setWindowTitle(tr("CNC Controller") if self.is_cnc else tr("Laser Controller"))
        self.session = laser.Session()
        self.worker = None
        self.job: list[str] = []
        self.job_name = ""
        self.image = None
        self.image_name = ""
        self._dots = (None, None)       # (what they were made of, the dots of the picture)
        self._boot_said = False
        self._boot_wait = 0             # ticks left before the held blocks go anyway
        self._messages = 0
        self._errors = 0

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setAccessibleName(self.windowTitle())
        inner = QWidget()
        lay = QVBoxLayout(inner)
        lay.setContentsMargins(6, 6, 6, 6)
        title = QLabel(f"<b>{self.windowTitle()}</b>")
        lay.addWidget(title)
        lay.addWidget(self._connection_box())
        lay.addWidget(self._board_box())
        lay.addWidget(self._jog_box())
        lay.addWidget(self._origin_box())
        lay.addWidget(self._job_box())
        lay.addWidget(self._override_box())
        image = self._image_box()
        lay.addWidget(image)
        lay.addWidget(self._log_box())
        lay.addStretch(1)
        scroll.setWidget(inner)
        outer.addWidget(scroll)
        if self.is_cnc:
            # a mill: its own list of machines and no picture, but the job comes from
            # the worksheet just as well
            image.hide()

        self.timer = QTimer(self)
        self.timer.setInterval(int(laser.UPDATE_SECONDS * 1000))
        self.timer.timeout.connect(self.refresh)
        self.timer.start()
        self.refresh_ports()
        self.refresh()

    @property
    def is_cnc(self) -> bool:
        return self.kind == "cnc"

    def _k(self, key) -> str:
        """The settings key of this machine: a mill keeps its own beside the laser."""
        key = str(key)
        return "cnc/" + key.split("/", 1)[-1] if self.is_cnc else key

    # ------------------------------------------------------------------ the cable
    def _connection_box(self):
        box = QGroupBox(tr("Connection"))
        form = QFormLayout(box)
        self._connection_form = form
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
        try:
            saved = int(qsettings().value(self._k(laser.BAUD_KEY), grbl.DEFAULT_BAUD))
        except (TypeError, ValueError):
            saved = grbl.DEFAULT_BAUD
        self.baud_box.setCurrentIndex(max(0, self.baud_box.findData(saved)))
        self.b_connect = QPushButton(tr("Connect"))
        self.b_connect.setToolTip(tr("Open the port and ask the board who it is"))
        self.port_hint = panels.note("")
        self.machine_box = QComboBox()
        for key, label in profiles.choices(self.kind):
            self.machine_box.addItem(label, key)
        self.machine_box.setToolTip(
            tr("A mill of the list, or one you saved: its work area and what its board is "
               "set to") if self.is_cnc else
            tr("A laser of the list, or one you saved: its work area and what its board is "
               "set to"))
        self.travel_x = QLineEdit()
        self.travel_x.setPlaceholderText("X")
        self.travel_x.setToolTip(tr("How far the head goes in X, in millimetres"))
        self.travel_y = QLineEdit()
        self.travel_y.setPlaceholderText("Y")
        self.travel_y.setToolTip(tr("How far the head goes in Y, in millimetres"))
        self.travel_z = QLineEdit()
        self.travel_z.setPlaceholderText("Z")
        self.travel_z.setToolTip(tr("How far the spindle goes up and down in Z, in "
                                    "millimetres"))
        travel = QHBoxLayout()
        travel.addWidget(self.travel_x)
        travel.addWidget(QLabel("×"))
        travel.addWidget(self.travel_y)
        times_z = QLabel("×")
        travel.addWidget(times_z)
        travel.addWidget(self.travel_z)
        travel.addWidget(QLabel("mm"))
        if not self.is_cnc:
            # a laser moves flat: the height of its head is set by hand, not by the board
            times_z.hide()
            self.travel_z.hide()
        self.clamp_check = QCheckBox(tr("Stay on the machine"))
        self.clamp_check.setToolTip(tr("Refuse a jog that would move the head past the "
                                       "end of the machine. Only after homing ($H) does the "
                                       "board know where its ends are."))
        self.clamp_check.setChecked(qsettings().value(self._k(profiles.CLAMP_KEY), False)
                                    in (True, "true", "1", 1))
        self.b_apply = QPushButton(tr("Settings to the board…"))
        self.b_apply.setToolTip(
            tr("Switch laser mode off and set the travel of the board to what this mill is")
            if self.is_cnc else
            tr("Set laser mode, the S maximum and the travel of the board to what this "
               "machine is"))
        self.b_take = QPushButton(tr("Take from the board"))
        self.b_take.setToolTip(tr("Read the work area and the S maximum out of what the "
                                  "board answered to $$"))
        self.b_presets = QPushButton(tr("Presets…"))
        self.b_presets.setToolTip(tr("Save all the settings of the board under a name, and "
                                     "give them back to it later"))
        buttons = _grid((self.b_apply, self.b_take, self.b_presets), 2)
        self.machine_hint = panels.note("")
        form.addRow(tr("Port"), ports)
        form.addRow(tr("Speed"), self.baud_box)
        form.addRow(self.b_connect)
        form.addRow(self.port_hint)
        form.addRow(tr("Machine"), self.machine_box)
        form.addRow(tr("Work area"), travel)
        form.addRow(self.clamp_check)
        form.addRow(buttons)
        form.addRow(self.machine_hint)
        self.b_refresh.clicked.connect(self.refresh_ports)
        self.b_connect.clicked.connect(self.toggle_connection)
        self.b_apply.clicked.connect(self.apply_settings)
        self.b_take.clicked.connect(self.take_from_board)
        self.b_presets.clicked.connect(self.open_presets)
        self.machine_box.currentIndexChanged.connect(self.choose_machine)
        self.clamp_check.toggled.connect(
            lambda on: (qsettings().setValue(self._k(profiles.CLAMP_KEY), bool(on)), qsettings().sync()))
        for edit, key in ((self.travel_x, self._k(profiles.TRAVEL_X_KEY)),
                          (self.travel_y, self._k(profiles.TRAVEL_Y_KEY)),
                          (self.travel_z, self._k(profiles.TRAVEL_Z_KEY))):
            edit.editingFinished.connect(
                lambda e=edit, k=key: (qsettings().setValue(k, e.text().strip()),
                                       self._say_machine()))
        self._restore_machine()
        return box

    def refresh_ports(self):
        """Fill the list of ports again, keeping the one that was chosen before."""
        want = str(self.port_box.currentData()
                   or qsettings().value(self._k(laser.PORT_KEY), "") or "")
        self.port_box.clear()
        for device, description in laser.ports():
            self.port_box.addItem(f"{device}  {description}", device)
        i = self.port_box.findData(want)
        if i >= 0:
            self.port_box.setCurrentIndex(i)
        if self.worker is not None:
            return
        if not laser.has_serial():
            self.port_hint.setText(tr("Volura needs the pyserial package to talk to a board: "
                                      "pip install pyserial"))
        elif self.port_box.count() == 0:
            self.port_hint.setText(tr("No serial port. Is the board on, and is its cable in "
                                      "the computer?"))
        else:
            self.port_hint.setText("")

    # ------------------------------------------------------------------ the machine
    @property
    def machine(self) -> profiles.Machine:
        """The preset that is chosen in the combo box."""
        return profiles.find(self.machine_box.currentData(), self.kind)

    def travel_size(self) -> tuple:
        """The work area on the screen, in millimetres; zero where nothing stands."""
        out = []
        for edit in (self.travel_x, self.travel_y):
            try:
                out.append(max(0.0, float(edit.text().replace(",", "."))))
            except ValueError:
                out.append(0.0)
        return tuple(out)

    def travel_height(self) -> float:
        """How far Z goes on the screen, in millimetres; zero when nothing stands (a laser)."""
        if not self.is_cnc:
            return 0.0
        try:
            return max(0.0, float(self.travel_z.text().replace(",", ".")))
        except ValueError:
            return 0.0

    def _restore_machine(self):
        """The machine of the last time, with the work area as it was left."""
        self._machine_rows()
        default = "grbl_mill" if self.is_cnc else "grbl32"
        key = str(qsettings().value(self._k(profiles.MACHINE_KEY), default))
        i = self.machine_box.findData(key)
        if i < 0:
            i = max(0, self.machine_box.findData(default))
        self.machine_box.blockSignals(True)
        self.machine_box.setCurrentIndex(i)
        self.machine_box.blockSignals(False)
        machine = self.machine
        for edit, key2, value in ((self.travel_x, self._k(profiles.TRAVEL_X_KEY), machine.x),
                                  (self.travel_y, self._k(profiles.TRAVEL_Y_KEY), machine.y),
                                  (self.travel_z, self._k(profiles.TRAVEL_Z_KEY), machine.z)):
            was = str(qsettings().value(key2, "") or "").strip()
            edit.setText(was or (f"{float(value):g}" if float(value) > 0 else ""))
        self._say_machine()

    def choose_machine(self, *_):
        """A preset is chosen: its work area onto the screen and its S word into the G-code."""
        machine = self.machine
        qsettings().setValue(self._k(profiles.MACHINE_KEY), machine.key)
        if machine.known:
            self.travel_x.setText(f"{machine.x:g}")
            self.travel_y.setText(f"{machine.y:g}")
            qsettings().setValue(self._k(profiles.TRAVEL_X_KEY), self.travel_x.text())
            qsettings().setValue(self._k(profiles.TRAVEL_Y_KEY), self.travel_y.text())
        if float(machine.z) > 0.0:
            self.travel_z.setText(f"{machine.z:g}")
            qsettings().setValue(self._k(profiles.TRAVEL_Z_KEY), self.travel_z.text())
        self._set_jog_feed(machine.jog)
        q = qsettings()
        if not self.is_cnc:
            # the S word and M4 of the laser export; a mill leaves them alone
            q.setValue(GCODE_S_MAX_KEY, int(machine.s_max))
            q.setValue(GCODE_DYNAMIC_KEY, bool(machine.dynamic))
        q.sync()
        self._say_machine()

    def _set_jog_feed(self, feed):
        """The speed of the pad nearest to what this machine moves at."""
        feeds = [self.feed_box.itemData(n) for n in range(self.feed_box.count())]
        if feeds:
            self.feed_box.setCurrentIndex(min(range(len(feeds)), key=lambda n: abs(feeds[n] - feed)))

    def _say_machine(self):
        """Under the combo box: what the chosen machine is and what the board says about it."""
        machine = self.machine
        bits = [machine.note] if machine.note else []
        x, y = self.travel_size()
        z = self.travel_height()
        if x and y and z:
            bits.append(tr("work area {x} × {y} × {z} mm").format(
                x=f"{x:g}", y=f"{y:g}", z=f"{z:g}"))
        elif x and y:
            bits.append(tr("work area {x} × {y} mm").format(x=f"{x:g}", y=f"{y:g}"))
        else:
            bits.append(tr("the work area is unknown"))
        if self.is_cnc:
            if int(machine.axes) > 3:
                bits.append(tr("{n} axes").format(n=int(machine.axes)))
        mode = profiles.laser_mode(self.session.settings)
        if self.is_cnc:
            if mode is True:
                # a spindle in laser mode does not stop for a corner and G0 switches it off
                bits.append(tr("the board says laser mode is on; a spindle wants $32=0"))
        else:
            bits.append(tr("S up to {n}").format(n=machine.s_max))
            if mode is False:
                bits.append(tr("the board says laser mode is off"))
        self.machine_hint.setText(" · ".join(bits))

    def apply_settings(self):
        """Set the board to what the machine is; every block is shown before it goes out."""
        if not self.connected:
            QMessageBox.information(self, self.windowTitle(), tr("Connect to the board first."))
            return
        if self.session.running:
            QMessageBox.information(self, self.windowTitle(),
                                    tr("A job is running; a board takes new settings between jobs."))
            return
        lines = self.machine.settings(self.travel_size() + (self.travel_height(),))
        answer = QMessageBox.question(
            self, self.windowTitle(),
            tr("Set the board to this?\n\n{lines}").format(lines="\n".join(lines)),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes)
        if answer != QMessageBox.StandardButton.Yes:
            return
        self.send_settings(lines)

    def take_from_board(self):
        """The work area and the S maximum out of what the board answered to `$$`."""
        if not self.session.settings:
            QMessageBox.information(self, self.windowTitle(),
                                    tr("The board has not answered yet. Disconnect and "
                                       "connect again to ask it."))
            return
        x, y = profiles.travel_of(self.session.settings)
        s_max = profiles.s_max_of(self.session.settings, self.machine.s_max)
        if x and y:
            self.travel_x.setText(f"{x:g}")
            self.travel_y.setText(f"{y:g}")
            qsettings().setValue(self._k(profiles.TRAVEL_X_KEY), self.travel_x.text())
            qsettings().setValue(self._k(profiles.TRAVEL_Y_KEY), self.travel_y.text())
        if self.is_cnc:
            # a mill has no S maximum for the laser export, but a height
            z = profiles.height_of(self.session.settings)
            self._choose_matching(x, y)
            if z:
                self.travel_z.setText(f"{z:g}")
                qsettings().setValue(self._k(profiles.TRAVEL_Z_KEY), self.travel_z.text())
            self.say(tr("the board: work area {x} × {y} × {z} mm").format(
                x=f"{x:g}" if x else "?", y=f"{y:g}" if y else "?", z=f"{z:g}" if z else "?"))
            self._say_machine()
            return
        q = qsettings()
        q.setValue(GCODE_S_MAX_KEY, int(s_max))
        q.sync()
        self._choose_matching(x, y, s_max)
        self.say(tr("the board: S up to {s}, work area {x} × {y} mm").format(
            s=s_max, x=f"{x:g}" if x else "?", y=f"{y:g}" if y else "?"))
        self._say_machine()

    def _choose_matching(self, x, y, s_max=None):
        """The machine of the list the board's numbers belong to; when none, the list
        stays as it is and the log says how to keep this board as a preset."""
        same = profiles.match(x, y, s_max, self.kind) if x and y else None
        if same is None:
            self.say(tr("This board is not in the list; keep it with Presets… under a name "
                        "of its own."))
            return
        i = self.machine_box.findData(same.key)
        if i >= 0:
            self.machine_box.blockSignals(True)
            self.machine_box.setCurrentIndex(i)
            self.machine_box.blockSignals(False)
            qsettings().setValue(self._k(profiles.MACHINE_KEY), same.key)

    def _machine_rows(self):
        """Fill the combo box again, keeping what is chosen."""
        want = self.machine_box.currentData()
        self.machine_box.blockSignals(True)
        self.machine_box.clear()
        for key, label in profiles.choices(self.kind):
            self.machine_box.addItem(label, key)
        i = self.machine_box.findData(want)
        self.machine_box.setCurrentIndex(max(0, i))
        self.machine_box.blockSignals(False)

    def refresh_machines(self):
        """The presets may have changed: fill the list again and say what stands there."""
        self._machine_rows()
        self._say_machine()

    def open_presets(self):
        """The window that saves and brings back the settings of a board."""
        dialog = PresetsDialog(self)
        dialog.exec()
        self.refresh_machines()
        self._say_machine()

    def send_settings(self, lines):
        """Blocks to the board, one by one, written in the log as they go."""
        if self.session.running:
            QMessageBox.information(self, self.windowTitle(),
                                    tr("A job is running; a board takes new settings between jobs."))
            return
        for line in lines:
            self.session.send(line)
            self.say("> " + line)
        self.session.send(grbl.SETTINGS)

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
            QMessageBox.information(self, self.windowTitle(), tr("Choose a port first."))
            return
        try:
            port = laser.open_port(device, self.baud_box.currentData())
        except laser.CableError as e:
            QMessageBox.warning(self, self.windowTitle(), str(e))
            return
        qsettings().setValue(self._k(laser.PORT_KEY), str(device))
        qsettings().setValue(self._k(laser.BAUD_KEY), int(self.baud_box.currentData()))
        self.worker = laser.Worker(port, self.session, poll=laser.UPDATE_SECONDS)
        self._boot_wait = BOOT_TICKS
        self.session.send_after_boot(grbl.SETTINGS)
        self.session.send_after_boot(grbl.UNLOCK)
        self.worker.start()
        self.session.realtime(grbl.RESET)
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
        form.addRow(_grid((self.b_unlock, self.b_home, self.b_check, self.b_reset,
                           self.b_settings)))
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
        self._remember(self.step_box, self._k(laser.STEP_KEY), laser.JOG_STEPS[0])
        self._remember(self.feed_box, self._k(laser.FEED_KEY), laser.JOG_FEEDS[0])
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
        if self.clamp_check.isChecked() and self.session.mpos is not None:
            x = float(self.session.mpos[0]) + dx * step
            y = float(self.session.mpos[1]) + dy * step
            z = float(self.session.mpos[2]) + dz * step if len(self.session.mpos) > 2 else 0.0
            if not self._on_machine(x, y) or (dz and not self._in_height(z)):
                self.say(tr("that would move the head past the end of the machine"))
                return
        block = grbl.jog(dx * step, dy * step, dz * step, feed)
        if block:
            self.session.send(block)

    def _on_machine(self, x, y) -> bool:
        """Whether a machine position is on the work area of the machine."""
        w, h = self.travel_size()
        if not (w and h):
            return True
        # a board that homes to the far corner counts its machine from 0 down to -travel,
        # one that homes to the near corner from 0 up; the head now says which it is
        here = self.session.mpos or (0.0, 0.0, 0.0)

        def inside(v, now, size):
            lo, hi = (-size, 0.0) if now < -1e-6 else (0.0, size)
            return lo - 1e-6 <= float(v) <= hi + 1e-6
        return inside(x, float(here[0]), w) and inside(y, float(here[1]), h)

    def _in_height(self, z) -> bool:
        """Whether a machine Z is within the height of a mill. Z homes at the top, so a
        board counts it from 0 down to -height; a board that counts up is believed too."""
        size = self.travel_height()
        if not size or self.session.mpos is None or len(self.session.mpos) < 3:
            return True
        now = float(self.session.mpos[2])
        lo, hi = (-size, 0.0) if now <= 1e-6 else (0.0, size)
        return lo - 1e-6 <= float(z) <= hi + 1e-6

    def go_to_origin(self):
        self.session.send(grbl.move(x=0.0, y=0.0))

    # ------------------------------------------------------------------ the work origin
    def _origin_box(self):
        box = QGroupBox(tr("Work origin"))
        lay = QVBoxLayout(box)
        self.b_zero_xy = QPushButton(tr("Zero here"))
        self.b_zero_xy.setToolTip(tr("Put the origin of the work under the head"))
        self.b_zero_x = QPushButton(tr("Zero X"))
        self.b_zero_x.setToolTip(tr("Only the X of the work moves to the head"))
        self.b_zero_y = QPushButton(tr("Zero Y"))
        self.b_zero_y.setToolTip(tr("Only the Y of the work moves to the head"))
        self.b_zero_z = QPushButton(tr("Zero Z"))
        self.b_zero_z.setToolTip(tr("The Z of the work starts at the head"))
        lay.addLayout(_grid((self.b_zero_xy, self.b_zero_x, self.b_zero_y, self.b_zero_z), 2))
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
        if self.is_cnc:
            form.addRow(tr("Spindle"), self._override_row("power", self.power_override_label,
                                                          tr("spindle speed")))
        else:
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


    # ------------------------------------------------------------------ a picture to engrave
    def _image_box(self):
        """A photograph or a logo, engraved as a raster: the modes of LightBurn."""
        box = QGroupBox(tr("Picture"))
        form = QFormLayout(box)
        self.b_image = QPushButton(tr("Picture…"))
        self.b_image.setToolTip(tr("A quick test: engrave a picture at the origin of the "
                                   "work. To place it, use File > Import > Picture… on the "
                                   "worksheet"))
        self.mode_box = QComboBox()
        for key, label in laser_image.MODES:
            self.mode_box.addItem(label, key)
        self.mode_box.setToolTip(tr("What a pixel of the picture becomes on the sheet"))
        self.style_box = QComboBox()
        self.style_box.setToolTip(tr("The matrix that makes the dots"))
        self.dot_box = QComboBox()
        for mm in laser.DOT_SIZES:
            self.dot_box.addItem(tr("{n} mm").format(n=f"{mm:g}"), mm)
        self.dot_box.setToolTip(tr("The distance between two lines of the engraving: how "
                                   "big one dot becomes on the sheet"))
        self.width_spin = QDoubleSpinBox()
        self.width_spin.setRange(1.0, 2000.0)
        self.width_spin.setDecimals(1)
        self.width_spin.setSuffix(" mm")
        self.width_spin.setValue(_float(qsettings().value(IMAGE_WIDTH_KEY, 100.0), 100.0))
        self.width_spin.setToolTip(tr("How wide the picture becomes on the sheet; the height "
                                      "follows the picture"))
        self.raster_feed_box = QComboBox()
        for f in laser.RASTER_FEEDS:
            self.raster_feed_box.addItem(tr("{n} mm/min").format(n=f), f)
        self.raster_feed_box.setToolTip(tr("How fast the head moves while it engraves"))
        self.power_spin = QSpinBox()
        self.power_spin.setRange(1, 100)
        self.power_spin.setValue(int(qsettings().value(laser.IMAGE_POWER_KEY, 30)))
        self.power_spin.setSuffix(" %")
        self.power_spin.setToolTip(tr("The power of the beam on the darkest pixel of the picture"))
        self.passes_spin = QSpinBox()
        self.passes_spin.setRange(1, 20)
        self.passes_spin.setValue(int(qsettings().value(laser.IMAGE_PASSES_KEY, 1)))
        self.passes_spin.setToolTip(tr("How often the whole picture is burned in"))
        self.invert_check = QCheckBox(tr("Invert"))
        self.invert_check.setToolTip(tr("Burn the light pixels instead of the dark ones, "
                                        "for a dark material that the beam makes lighter"))
        self.one_way_check = QCheckBox(tr("One way"))
        self.one_way_check.setToolTip(tr("Come back over the row with the beam off instead of "
                                         "turning around: slower, but no burn-in on the return"))
        self.b_image_job = QPushButton(tr("Engrave the picture"))
        self.b_image_job.setToolTip(tr("Make the raster of the picture and put it in the job"))
        self.image_hint = panels.note(tr("There is no picture to engrave yet."))
        form.addRow(self.b_image)
        form.addRow(tr("Mode"), self.mode_box)
        form.addRow(tr("Matrix"), self.style_box)
        form.addRow(tr("Width"), self.width_spin)
        form.addRow(tr("Dot"), self.dot_box)
        form.addRow(tr("Speed"), self.raster_feed_box)
        form.addRow(tr("Power"), self.power_spin)
        form.addRow(tr("Passes"), self.passes_spin)
        form.addRow(self.invert_check)
        form.addRow(self.one_way_check)
        form.addRow(self.b_image_job)
        form.addRow(self.image_hint)
        self.b_image.clicked.connect(self.load_image)
        self.b_image_job.clicked.connect(self.image_job)
        self.mode_box.currentIndexChanged.connect(self._mode_changed)
        self.invert_check.setChecked(qsettings().value(laser.IMAGE_INVERT_KEY, False)
                                     in (True, "true", "1", 1))
        self.one_way_check.setChecked(qsettings().value(laser.IMAGE_ONEWAY_KEY, False)
                                      in (True, "true", "1", 1))
        self._remember(self.dot_box, laser.DOT_KEY, laser.DOT_SIZES[3])
        self._remember(self.raster_feed_box, laser.RASTER_FEED_KEY, 3000)
        self._pick(self.mode_box, laser.IMAGE_MODE_KEY, "dither")
        self._fill_styles()
        self.power_spin.valueChanged.connect(
            lambda v: (qsettings().setValue(laser.IMAGE_POWER_KEY, int(v)), self.refresh_image()))
        self.passes_spin.valueChanged.connect(
            lambda v: (qsettings().setValue(laser.IMAGE_PASSES_KEY, int(v)), self.refresh_image()))
        self.invert_check.toggled.connect(
            lambda on: (qsettings().setValue(laser.IMAGE_INVERT_KEY, bool(on)),
                        self.refresh_image()))
        self.one_way_check.toggled.connect(
            lambda on: (qsettings().setValue(laser.IMAGE_ONEWAY_KEY, bool(on)),
                        self.refresh_image()))
        self.dot_box.currentIndexChanged.connect(self.refresh_image)
        self.width_spin.editingFinished.connect(
            lambda: (qsettings().setValue(IMAGE_WIDTH_KEY, float(self.width_spin.value())),
                     self.refresh_image()))
        self.raster_feed_box.currentIndexChanged.connect(self.refresh_image)
        self.style_box.currentIndexChanged.connect(self.refresh_image)
        return box

    def _pick(self, box, key, default):
        """Put the saved choice of a combo box back, and keep it for the next time."""
        data = [box.itemData(n) for n in range(box.count())]
        want = str(qsettings().value(key, default))
        box.blockSignals(True)
        box.setCurrentIndex(data.index(want) if want in data else 0)
        box.blockSignals(False)
        box.currentIndexChanged.connect(
            lambda _n, b=box, k=key: qsettings().setValue(k, b.currentData()))

    def _mode_changed(self, *_):
        """Another mode: the matrices that belong to it, and the picture again."""
        self._fill_styles()
        self.refresh_image()

    def _fill_styles(self):
        """The matrices of the chosen mode: the names of the dithering, or 4x4 and 8x8."""
        mode = self.mode_box.currentData()
        self.style_box.blockSignals(True)
        self.style_box.clear()
        if mode == "dither":
            for key, (label, _taps, _total) in laser_image.DITHERS.items():
                self.style_box.addItem(label, key)
        elif mode == "halftone":
            for key in laser_image.HALFTONES:
                self.style_box.addItem(tr("{n} by {n}").format(n=key), key)
        self.style_box.blockSignals(False)
        self.style_box.setEnabled(self.style_box.count() > 0)

    def load_image(self):
        """A picture from disk, as grey values."""
        path, _ = QFileDialog.getOpenFileName(
            self, tr("Picture to engrave"), "",
            "Images (*.png *.jpg *.jpeg *.bmp *.gif *.webp *.tif *.tiff);;All files (*)")
        if not path:
            return
        try:
            self.image = image_gray(path)
        except ValueError as e:
            QMessageBox.warning(self, self.windowTitle(), str(e))
            return
        self.image_name = os.path.basename(path)
        self._dots = (None, None)
        self.refresh_image()

    def image_dots(self):
        """The picture as the machine should burn it; nothing when there is no picture.

        The picture is first brought to one pixel per dot of the engraving, at the width
        that was asked for, and only then made into dots: a dot of the dithering is then
        exactly a dot of the beam. Dithering a large picture takes a moment, so the dots
        are kept until something they were made of changes; the power, the speed and the
        passes do not touch them.
        """
        if self.image is None:
            return None
        style = self.style_box.currentData()
        dot = float(self.dot_box.currentData() or 0.1)
        width = float(self.width_spin.value())
        made_of = (id(self.image), self.mode_box.currentData(), style,
                   self.invert_check.isChecked(), width, dot)
        if self._dots[0] == made_of:
            return self._dots[1]
        gray = laser_image.resample(self.image, width, dot)
        gray = laser_image.adjust(gray, invert=self.invert_check.isChecked())
        dots = laser_image.bitmap(gray, self.mode_box.currentData(),
                                  style or "floyd", style or "8x8")
        self._dots = (made_of, dots)
        return dots

    def image_job(self):
        """The raster of the picture as the job, at the origin of the work."""
        dots = self.image_dots()
        if dots is None:
            QMessageBox.information(self, self.windowTitle(),
                                    tr("Choose a picture first."))
            return
        dot = float(self.dot_box.currentData() or 0.1)
        feed = float(self.raster_feed_box.currentData() or 3000)
        text = laser_image.engrave(dots, dot=dot, feed=feed,
                                   power=float(self.power_spin.value()),
                                   s_max=int(gcode_export_defaults()["s_max"]),
                                   one_way=self.one_way_check.isChecked(),
                                   passes=self.passes_spin.value(),
                                   title=self.image_name)
        self.set_job(grbl.job_lines(text), self.image_name)
        self.say(tr("the picture is {w} × {h} mm and takes about {t}").format(
            w=f"{dots.shape[1] * dot:g}", h=f"{dots.shape[0] * dot:g}",
            t=fmt_time(self.image_seconds())))

    def image_seconds(self) -> float:
        """How long the engraving of the picture takes, as the estimate says."""
        dots = self.image_dots()
        if dots is None:
            return 0.0
        return laser_image.estimate(
            dots, float(self.dot_box.currentData() or 0.1),
            float(self.raster_feed_box.currentData() or 3000),
            self.one_way_check.isChecked(), self.passes_spin.value())["seconds"]

    def refresh_image(self):
        """Under the buttons: how big the picture becomes and what it costs."""
        if self.image is None:
            self.image_hint.setText(tr("There is no picture to engrave yet."))
            return
        dot = float(self.dot_box.currentData() or 0.1)
        info = laser_image.estimate(self.image_dots(), dot,
                                    float(self.raster_feed_box.currentData() or 3000),
                                    self.one_way_check.isChecked(), self.passes_spin.value())
        self.image_hint.setText(tr("{w} × {h} mm, {rows} rows, about {t}").format(
            w=f"{info['width']:g}", h=f"{info['height']:g}", rows=info["rows"],
            t=fmt_time(info["seconds"])))


    # ------------------------------------------------------------------ the job
    def _job_box(self):
        box = QGroupBox(tr("Job"))
        lay = QVBoxLayout(box)
        self.job_label = panels.note(tr("There is no job to send yet."))
        lay.addWidget(self.job_label)
        self.b_load = QPushButton(tr("From the worksheet…"))
        self.b_load.setToolTip(tr("Make G-code of the layers the mill follows") if self.is_cnc
                               else tr("Make G-code of the layers the laser follows"))
        self.b_open = QPushButton(tr("Open G-code…"))
        self.b_open.setToolTip(tr("Send a G-code file that is already there"))
        self.b_start = QPushButton(tr("Start"))
        self.b_start.setToolTip(tr("Send the job to the board"))
        self.b_pause = QPushButton(tr("Pause"))
        self.b_pause.setToolTip(tr("Feed hold: the head stops and the beam goes off"))
        self.b_stop = QPushButton(tr("Stop"))
        self.b_stop.setToolTip(tr("End the job and empty what the board has queued"))
        lay.addLayout(_grid((self.b_load, self.b_open), 2))
        lay.addLayout(_grid((self.b_start, self.b_pause, self.b_stop)))
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
        options = self.win.sheet_export_options("mill" if self.is_cnc else "gcode")
        if not options:
            return
        try:
            drawing = self.win.sheet_export_drawing(options)
            if self.is_cnc:
                # the drawing carries the compensation the dialog asked for already
                text = gcode_mill.grbl_mill(drawing, compensate=False)
            else:
                text = gcode.grbl_laser(drawing, s_max=int(options.get("s_max") or 1000),
                                        dynamic=bool(options.get("dynamic", True)),
                                        pictures=sheet_image.pictures_of(drawing))
        except Exception as e:
            QMessageBox.critical(self, self.windowTitle(), str(e))
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
            QMessageBox.critical(self, self.windowTitle(), str(e))
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
            QMessageBox.information(self, self.windowTitle(),
                                    tr("Connect to the board first."))
            return
        self.session.start(self.job)

    def toggle_pause(self):
        if self.session.paused:
            self.session.resume()
        else:
            self.session.pause()
        self.refresh()

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
        if self._boot_wait > 0 and self.worker is not None:
            self._boot_wait -= 1
            if self._boot_wait == 0:
                s.release_held()      # a board that greets nobody still gets its blocks
        self._listen()
        self._say_machine()
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
        self.b_check.setEnabled(free)
        self.b_reset.setEnabled(self.connected)
        self.b_apply.setEnabled(self.connected)
        self.b_take.setEnabled(bool(s.settings))
        self.b_start.setEnabled(self.connected and not s.running and bool(self.job))
        self.b_pause.setEnabled(self.connected and s.running)
        self.b_stop.setEnabled(self.connected and s.running)
        self.b_pause.setText(tr("Resume") if s.paused and s.running else tr("Pause"))


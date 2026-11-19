"""The print preview: the sliced G-code in 3D, layer by layer, with a virtual printer that prints it.

After File > Slice the G-code goes here first instead of straight to a file. The view shows the
print in 3D on the bed: every layer up to the chosen one, its lines as wide and high as they print
and in the colour of what they print (outer wall, inner wall, top and bottom, infill, support,
skirt), the layers below it dimmed, the travel moves of the chosen layer as thin lines when you
want them, and the nozzle where it is. A print that goes over more than one filament colours its
lines with those filaments instead, and the legend under the view names every filament with how
much of it the print used. Drag with
the left mouse button to turn the view, with the right or middle button (or Shift and the left) to
move it; the wheel zooms towards the mouse and a double-click shows the whole print again. Top and
3D switch the view; Only this layer hides the layers below. Play moves the nozzle through the layer
at the speed of the real print times a factor and goes on to the next layer; the slider and the
arrow keys pick a layer. Save G-code… writes the file, and Send to printer… hands it to
OctoPrint, Klipper (Moonraker) or PrusaLink over the network (see volura/printer_link.py).
"""
from __future__ import annotations

import os
import shutil

import numpy as np
from OpenGL import GL as gl
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QColor
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFormLayout,
                               QHBoxLayout, QLabel, QLineEdit, QMessageBox, QPushButton,
                               QSlider, QVBoxLayout)

from . import background, errors, printer_link
from . import gcode_preview as gp
from .camera import Camera
from .i18n import tr, tr_noop
from .settings import flag, qsettings
from .viewcube import ViewCube

#: the colour and the name of every kind of line, in the order of gcode_preview.KIND_NAMES
KIND_STYLE = [("#9aa4b1", tr_noop("Travel")), ("#d9480f", tr_noop("Outer wall")),
              ("#f08c00", tr_noop("Inner wall")), ("#e8590c", tr_noop("Top and bottom")),
              ("#2f9e44", tr_noop("Infill")), ("#1c7ed6", tr_noop("Support")),
              ("#4dabf7", tr_noop("Support interface")), ("#7048e8", tr_noop("Skirt and brim")),
              ("#ae3ec9", tr_noop("Prime tower")), ("#868e96", tr_noop("Other"))]
#: the same kinds in the Okabe-Ito palette, which people with red-green colour blindness
#: can tell apart (Window > Preferences > Display)
KIND_STYLE_CB = [("#999999", tr_noop("Travel")), ("#E69F00", tr_noop("Outer wall")),
                 ("#56B4E9", tr_noop("Inner wall")), ("#009E73", tr_noop("Top and bottom")),
                 ("#F0E442", tr_noop("Infill")), ("#0072B2", tr_noop("Support")),
                 ("#8FD3FF", tr_noop("Support interface")), ("#CC79A7", tr_noop("Skirt and brim")),
                 ("#D55E00", tr_noop("Prime tower")), ("#BBBBBB", tr_noop("Other"))]
#: the colours a print that changes filament by height shows its filaments in, in the order
#: the printer loads them; a slice that knows its filaments brings its own colours
FILAMENT_STYLE = ("#d23b3b", "#2f6fd6", "#3aa655", "#e8a33d", "#8e5bd0", "#2abfbf",
                  "#e05fa0", "#7a7a7a")
#: how much faster than the real print the virtual printer plays
SPEEDS = (1, 5, 20, 100)
#: frames a second of the playback
FPS = 30
#: how bright the layers below the chosen one are, against its own colours
DIM = 0.5
#: the cross-section of a printed line, from one bottom edge over the top to the other: for
#: every point how far across (in half widths), how far down (in layer heights) and how bright
#: against the colour of the line, so a line looks round on top and lines lying side by side
#: stand apart
PROFILE = ((-1.0, 1.0, 0.45), (-1.0, 0.35, 0.72), (0.0, 0.0, 1.0), (1.0, 0.35, 0.72),
           (1.0, 1.0, 0.45))
#: vertices of one printed line (see `_beads`)
VERTS = 2 * len(PROFILE)
#: the colours of the background, the bed and its grid
BACKGROUND, BED, GRID = "#20242a", "#2b3038", "#3a404a"
#: the settings keys of the printer the last slice went to and whether it started there
LAST_PRINTER, LAST_START = "send_printer", "send_start"


def kind_style(colour_blind: bool):
    """The colours and names of every kind of line: the standard or the colour-blind palette."""
    return KIND_STYLE_CB if colour_blind else KIND_STYLE


def filament_color(filaments, i: int) -> str:
    """The colour of filament `i` of a print: what the slice says, or one of the palette.

    `filaments` is what `PrintPreviewDialog` gets: `(name, colour, mm, grams)` per filament.
    """
    color = filaments[i][1] if i < len(filaments) else ""
    return color or FILAMENT_STYLE[i % len(FILAMENT_STYLE)]


def _hms(seconds: float) -> str:
    s = int(round(seconds))
    h, rest = divmod(s, 3600)
    m, s = divmod(rest, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def _rgb(hex_color) -> tuple:
    c = QColor(hex_color)
    return c.redF(), c.greenF(), c.blueF()


def _beads(seg, width, z, height):
    """The vertices of printed lines as bars, `VERTS` per line, for a GL_QUAD_STRIP each.

    Every line is as wide as it prints and as high as its layer and runs half its width past both
    ends, so lines that meet at a corner close it. Each strip runs along the line through the
    points of `PROFILE`, from one bottom edge over the top to the other.
    """
    p0, p1 = seg[:, :2].astype(np.float64), seg[:, 2:].astype(np.float64)
    d = p1 - p0
    length = np.maximum(np.hypot(d[:, 0], d[:, 1]), 1e-9)[:, None]
    u = d / length * (np.asarray(width, dtype=np.float64) / 2.0)[:, None]
    n = np.column_stack([-u[:, 1], u[:, 0]])          # across the line, half a width long
    a, b = p0 - u, p1 + u
    out = np.empty((len(seg), VERTS, 3), dtype=np.float32)
    for k, (side, down, _light) in enumerate(PROFILE):
        out[:, 2 * k, :2] = a + side * n
        out[:, 2 * k + 1, :2] = b + side * n
        out[:, 2 * k:2 * k + 2, 2] = z - down * height
    return out.reshape(-1, 3)


class _Lines:
    """Every printed line of the print as one array of vertices and colours, layer after layer.

    Each line is a flat bar of `VERTS` vertices (see `_beads`) as wide as it prints, its top in
    the colour of its kind and its sides darker towards the bottom, so the lines of a layer lie
    against each other as they do on the bed and still stand apart. `start[i]` is the first line
    of layer `i` (and `start[-1]` the end); `done[i]` counts, for every move of layer `i`, the
    printed lines up to and including it, so a layer that is half printed draws only those.

    `filaments` is what the print prints with (see `filament_color`): with more than one the
    lines are drawn in the colour of the filament they print with, instead of the kind of line.
    """

    def __init__(self, preview: gp.Preview, colour_blind=False, filaments=()):
        style = kind_style(colour_blind)
        colours = np.array([_rgb(c) for c, _name in style], dtype=np.float32)
        by_filament = len(filaments) > 1 and preview.materials > 1
        base = 0
        if by_filament:                     # after the kinds come the filaments
            extra = [_rgb(filament_color(filaments, i)) for i in range(preview.materials)]
            colours = np.vstack([colours, np.array(extra, dtype=np.float32)])
            base = len(style)
        shade = np.repeat(np.array([p[2] for p in PROFILE], dtype=np.float32), 2)[:, None]
        pos, col, travel, start, tstart, done = [], [], [], [0], [0], []
        for layer in preview.layers:
            printed = layer.kind != gp.TRAVEL
            seg = layer.seg[printed]
            width = (layer.width[printed] if layer.width is not None
                     else np.full(len(seg), 0.4, dtype=np.float32))
            pos.append(_beads(seg, width, layer.z, layer.height))
            style_of = np.asarray(layer.kind[printed], dtype=np.int64)
            if by_filament and layer.material is not None:
                style_of = np.asarray(layer.material[printed], dtype=np.int64)
            col.append((colours[base + style_of][:, None, :] * shade).reshape(-1, 3))
            trav = layer.seg[~printed]
            zt = np.full((len(trav), 1), layer.z + 0.02, dtype=np.float32)   # above the tops
            travel.append(np.hstack([trav[:, :2], zt, trav[:, 2:], zt]).reshape(-1, 3))
            start.append(start[-1] + len(seg))
            tstart.append(tstart[-1] + 2 * len(trav))
            done.append(np.cumsum(printed))
        empty = np.zeros((0, 3), dtype=np.float32)
        self.pos = np.ascontiguousarray(np.vstack(pos) if pos else empty, dtype=np.float32)
        col = np.vstack(col) if col else empty
        self.col = np.ascontiguousarray(np.round(col * 255), dtype=np.uint8)
        self.dim = np.ascontiguousarray(np.round(col * DIM * 255), dtype=np.uint8)
        self.travel = np.ascontiguousarray(np.vstack(travel) if travel else empty,
                                           dtype=np.float32)
        self.start, self.tstart, self.done = start, tstart, done
        self.first = np.arange(0, start[-1] * VERTS, VERTS, dtype=np.int32)
        self.count = np.full(start[-1], VERTS, dtype=np.int32)

    def draw(self, lo: int, hi: int):
        """Draw lines `lo` up to `hi`, with the vertex and colour pointers set already."""
        if hi > lo:
            gl.glMultiDrawArrays(gl.GL_QUAD_STRIP, self.first[lo:hi], self.count[lo:hi], hi - lo)


class LayerView(QOpenGLWidget):
    """The print in 3D on the bed, up to one layer, with the nozzle on it.

    `bed` is (x0, y0, x1, y1): the bed in the coordinates of the G-code. `layer` is the chosen
    layer, `at` the seconds into it the virtual printer is (None: the whole layer). `filaments`
    is what the print prints with: with more than one, the lines show their filament.
    """

    def __init__(self, preview: gp.Preview, bed, colour_blind=False, filaments=(), parent=None):
        super().__init__(parent)
        self.preview, self.bed = preview, bed
        self.colour_blind = bool(colour_blind)
        self.filaments = list(filaments)
        self.by_filament = len(self.filaments) > 1 and preview.materials > 1
        self.lines = _Lines(preview, self.colour_blind, self.filaments if self.by_filament else ())
        self.layer = 0
        self.at = None
        self.show_travel = False
        self.only_layer = False
        self.cam = Camera()
        self._drag, self._last = None, None
        self._fitted = False                    # fitted once the view has its real size
        self.setMinimumSize(420, 420)
        self.setFocusPolicy(Qt.StrongFocus)
        self.view_cube = ViewCube(self, "print")      # the cube in the top right corner
        self.fit("iso")

    # ---- the camera
    def _box(self):
        """The print, or the bed when nothing is printed, as (lo, hi)."""
        layers = self.preview.layers
        top = max((layer.z for layer in layers), default=0.0)
        if self.preview.hi != self.preview.lo:
            lo = np.array([*self.preview.lo, 0.0])
            hi = np.array([*self.preview.hi, top])
        else:
            x0, y0, x1, y1 = self.bed
            lo, hi = np.array([x0, y0, 0.0]), np.array([x1, y1, top])
        return lo, hi

    def fit(self, view=None):
        """Show the whole print, from `view` ("iso", "top", ...) or from where the camera is."""
        if view:
            self.cam.set_view(view)
        self.cam.width, self.cam.height = max(1, self.width()), max(1, self.height())
        self.cam.fit(self._box())
        self.update()

    # ---- drawing
    def initializeGL(self):
        errors.clear_gl_errors()
        gl.glClearColor(*_rgb(BACKGROUND), 1.0)

    def resizeGL(self, w, h):
        self.cam.width, self.cam.height = max(1, w), max(1, h)
        if not self._fitted and w > 50 and h > 50:
            self._fitted = True
            self.cam.fit(self._box())

    @errors.safe_handler
    def paintGL(self):
        self.cam.width, self.cam.height = max(1, self.width()), max(1, self.height())
        dpr = self.devicePixelRatioF()
        gl.glViewport(0, 0, int(self.width() * dpr), int(self.height() * dpr))
        gl.glClearColor(*_rgb(BACKGROUND), 1.0)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        if bool(gl.glUseProgram):              # not there on an old or software OpenGL
            gl.glUseProgram(0)
        gl.glDisable(gl.GL_LIGHTING)
        gl.glDisable(gl.GL_CULL_FACE)
        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glLoadMatrixd(self.cam.proj_matrix().T.ravel())
        gl.glMatrixMode(gl.GL_MODELVIEW)
        gl.glLoadMatrixd(self.cam.view_matrix().T.ravel())
        gl.glEnable(gl.GL_DEPTH_TEST)
        self._draw_bed()
        if self.preview.layers:
            self._draw_print()
        self.view_cube.sync()              # the cube in the corner follows the camera

    def _draw_bed(self):
        x0, y0, x1, y1 = self.bed
        gl.glDepthMask(gl.GL_FALSE)
        gl.glColor3f(*_rgb(BED))
        gl.glBegin(gl.GL_QUADS)
        for x, y in ((x0, y0), (x1, y0), (x1, y1), (x0, y1)):
            gl.glVertex3f(x, y, -0.01)
        gl.glEnd()
        gl.glLineWidth(1.0)
        gl.glColor3f(*_rgb(GRID))
        gl.glBegin(gl.GL_LINES)
        for x in np.arange(x0, x1 + 1e-6, 10.0):
            gl.glVertex3f(x, y0, 0.0)
            gl.glVertex3f(x, y1, 0.0)
        for y in np.arange(y0, y1 + 1e-6, 10.0):
            gl.glVertex3f(x0, y, 0.0)
            gl.glVertex3f(x1, y, 0.0)
        gl.glEnd()
        gl.glDepthMask(gl.GL_TRUE)

    def _draw_print(self):
        ln, i = self.lines, self.layer
        first, end = ln.start[i], ln.start[i + 1]
        if self.at is not None:
            layer = self.preview.layers[i]
            moves = min(len(layer.t), int(np.searchsorted(layer.t, self.at, side="right")) + 1)
            end = first + (int(ln.done[i][moves - 1]) if moves else 0)
        gl.glEnableClientState(gl.GL_VERTEX_ARRAY)
        gl.glEnableClientState(gl.GL_COLOR_ARRAY)
        gl.glVertexPointer(3, gl.GL_FLOAT, 0, ln.pos)
        if first > 0 and not self.only_layer:
            gl.glColorPointer(3, gl.GL_UNSIGNED_BYTE, 0, ln.dim)
            ln.draw(0, first)
        gl.glColorPointer(3, gl.GL_UNSIGNED_BYTE, 0, ln.col)
        ln.draw(first, end)
        gl.glDisableClientState(gl.GL_COLOR_ARRAY)
        if self.show_travel and ln.tstart[i + 1] > ln.tstart[i]:
            gl.glLineWidth(1.0)
            gl.glColor3f(*_rgb(kind_style(self.colour_blind)[gp.TRAVEL][0]))
            gl.glVertexPointer(3, gl.GL_FLOAT, 0, ln.travel)
            gl.glDrawArrays(gl.GL_LINES, ln.tstart[i], ln.tstart[i + 1] - ln.tstart[i])
        gl.glDisableClientState(gl.GL_VERTEX_ARRAY)
        if self.at is not None:
            layer = self.preview.layers[i]
            pos, _done = layer.position(self.at)
            if pos is not None:
                x, y, z = pos[0], pos[1], layer.z
                gl.glDisable(gl.GL_DEPTH_TEST)
                gl.glLineWidth(3.0)
                gl.glColor3f(0.85, 0.85, 0.85)
                gl.glBegin(gl.GL_LINES)             # the nozzle, coming down from above
                gl.glVertex3f(x, y, z)
                gl.glVertex3f(x, y, z + 8.0)
                gl.glEnd()
                gl.glPointSize(9.0)
                gl.glColor3f(1.0, 0.42, 0.42)
                gl.glBegin(gl.GL_POINTS)
                gl.glVertex3f(x, y, z)
                gl.glEnd()
                gl.glEnable(gl.GL_DEPTH_TEST)

    # ---- turn, move and zoom
    def mousePressEvent(self, event):
        self.setFocus()
        self._last = event.position()
        shift = bool(event.modifiers() & Qt.ShiftModifier)
        if event.button() == Qt.LeftButton:
            self._drag = "pan" if shift else "orbit"
        elif event.button() in (Qt.RightButton, Qt.MiddleButton):
            self._drag = "pan"

    def mouseMoveEvent(self, event):
        if self._drag is None or self._last is None:
            return
        pos = event.position()
        dx, dy = pos.x() - self._last.x(), pos.y() - self._last.y()
        self._last = pos
        if self._drag == "orbit":
            self.cam.orbit(dx, dy)
        else:
            self.cam.pan(dx, dy)
        self.update()

    def mouseReleaseEvent(self, _event):
        self._drag = None

    def mouseDoubleClickEvent(self, _event):
        self.fit()

    def wheelEvent(self, event):
        steps = event.angleDelta().y() / 120.0
        if not steps:
            return
        # towards the point under the mouse on the height of the chosen layer
        o, d = self.cam.ray(event.position().x(), event.position().y())
        z = self.preview.layers[self.layer].z if self.preview.layers else 0.0
        about = None
        if abs(d[2]) > 1e-6:
            t = (z - o[2]) / d[2]
            if 0 < t < self.cam.distance * 5:
                about = o + t * d
        self.cam.zoom(0.85 ** steps, about)
        self.update()


class SendDialog(QDialog):
    """Send to printer…: which printer the G-code goes to, under what name, and whether it
    starts printing right away. The printer and that last yes or no are remembered.
    """

    def __init__(self, parent, printers, name: str):
        super().__init__(parent)
        self.setWindowTitle(tr("Send to printer"))
        lay = QVBoxLayout(self)
        form = QFormLayout()
        self.printer = QComboBox()
        self.printer.setToolTip(tr("The printers under Window > Preferences > Printers"))
        for conn in printers:
            self.printer.addItem(conn.name, conn)
            self.printer.setItemData(self.printer.count() - 1,
                                     printer_link.normalise_url(conn.url), Qt.ToolTipRole)
        self.name = QLineEdit(name)
        self.name.setToolTip(tr("The name the file gets on the printer"))
        self.start = QCheckBox(tr("Start printing right away"))
        self.start.setChecked(flag(LAST_START, False))
        form.addRow(tr("Printer"), self.printer)
        form.addRow(tr("File name"), self.name)
        form.addRow(self.start)
        lay.addLayout(form)
        bb = QDialogButtonBox(QDialogButtonBox.Cancel)
        send = bb.addButton(tr("Send"), QDialogButtonBox.AcceptRole)
        send.setDefault(True)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)
        self._last_used(str(qsettings().value(LAST_PRINTER, "") or ""))

    def _last_used(self, name):
        """Show the printer the last slice went to, when it is still there."""
        for i in range(self.printer.count()):
            if self.printer.itemData(i).name == name:
                self.printer.setCurrentIndex(i)
                return

    def connection(self):
        """The printer this G-code goes to."""
        return self.printer.currentData()

    def file_name(self) -> str:
        """The name the G-code gets on the printer, made safe."""
        return printer_link.file_name(self.name.text())

    def accept(self):
        """Send: keep what was chosen for the next slice."""
        qsettings().setValue(LAST_PRINTER, self.connection().name)
        qsettings().setValue(LAST_START, self.start.isChecked())
        qsettings().sync()
        super().accept()


class PrintPreviewDialog(QDialog):
    """The G-code of a slice, layer by layer; `Save G-code…` hands it on (see `save_to`)
    and `Send to printer…` sends it to a printer on the network (see `printer_link`).

    `filaments` is what the print prints with, `(name, colour, mm, grams)` per filament
    (see `filament_color`): with more than one the view colours every line with its filament
    and the legend names them, instead of the kinds of line.
    """

    def __init__(self, win, gcode_path: str, preview: gp.Preview, bed, summary: str,
                 ask_path=None, filaments=()):
        super().__init__(win)
        self.win = win
        self.gcode_path, self.preview = gcode_path, preview
        self.ask_path = ask_path
        self.saved_to = None
        self.sent = None                      # what stands on the printer, once it is there
        self._job = None
        self.filaments = list(filaments)
        self.by_filament = len(self.filaments) > 1 and preview.materials > 1
        settings = getattr(win, "settings", None)
        self.colour_blind = bool(getattr(settings, "colour_blind", False))
        self.printers = list(getattr(settings, "printers", None) or [])
        self.setWindowTitle(tr("Print Preview"))
        self.resize(980, 780)
        lay = QVBoxLayout(self)
        self.summary = QLabel(summary)
        self.summary.setWordWrap(True)
        lay.addWidget(self.summary)
        row = QHBoxLayout()
        self.view = LayerView(preview, bed, self.colour_blind, self.filaments)
        row.addWidget(self.view, 1)
        self.slider = QSlider(Qt.Vertical)
        self.slider.setRange(0, max(0, len(preview.layers) - 1))
        self.slider.setValue(max(0, len(preview.layers) - 1))
        row.addWidget(self.slider)
        lay.addLayout(row, 1)

        controls = QHBoxLayout()
        self.play = QPushButton(tr("Play"))
        self.play.setCheckable(True)
        self.speed = QComboBox()
        for f in SPEEDS:
            self.speed.addItem(f"{f}×", f)
        self.speed.setCurrentIndex(2)
        self.travel = QCheckBox(tr("Travel moves"))
        self.only = QCheckBox(tr("Only this layer"))
        self.top = QPushButton(tr("Top"))
        self.iso = QPushButton(tr("3D"))
        self.top.setToolTip(tr("Look straight down on the bed"))
        self.iso.setToolTip(tr("Look at the print from the side, at an angle"))
        self.info = QLabel()
        for w in (self.play, self.speed, self.travel, self.only, self.top, self.iso):
            controls.addWidget(w)
        controls.addStretch(1)
        controls.addWidget(self.info)
        lay.addLayout(controls)

        legend = QHBoxLayout()
        if self.by_filament:
            for i in range(max(preview.materials, len(self.filaments))):
                legend.addWidget(QLabel("<span style='color:{c}'>■</span> {t}".format(
                    c=filament_color(self.filaments, i), t=self._filament_legend(i))))
        else:
            used = set()
            for layer in preview.layers:
                used.update(int(k) for k in np.unique(layer.kind))
            for k, (color, name) in enumerate(kind_style(self.colour_blind)):
                if k in used and k != gp.TRAVEL:
                    legend.addWidget(QLabel(f"<span style='color:{color}'>■</span> {tr(name)}"))
        legend.addStretch(1)
        hint = QLabel(tr("Drag to turn, right-drag to move, wheel to zoom"))
        hint.setEnabled(False)
        legend.addWidget(hint)
        lay.addLayout(legend)

        bb = QDialogButtonBox()
        self.save_button = bb.addButton(tr("Save G-code…"), QDialogButtonBox.AcceptRole)
        self.send_button = bb.addButton(tr("Send to printer…"), QDialogButtonBox.ActionRole)
        bb.addButton(QDialogButtonBox.Close)
        bb.accepted.connect(self._save)
        bb.rejected.connect(self.reject)
        self.send_button.clicked.connect(self._send)
        self._show_send()
        lay.addWidget(bb)

        self.timer = QTimer(self)
        self.timer.setInterval(int(1000 / FPS))
        self.timer.timeout.connect(self._tick)
        self.slider.valueChanged.connect(self._layer_changed)
        self.play.toggled.connect(self._play_toggled)
        self.travel.toggled.connect(self._travel_toggled)
        self.only.toggled.connect(self._only_toggled)
        self.top.clicked.connect(lambda: self.view.fit("top"))
        self.iso.clicked.connect(lambda: self.view.fit("iso"))
        self._layer_changed(self.slider.value())

    # ---- layers and playback
    def _filament_legend(self, i: int) -> str:
        """One line of the legend: which filament this is and how much of it the print used."""
        if i >= len(self.filaments):
            name, mm, grams = tr("Filament {n}").format(n=i + 1), 0.0, 0.0
        else:
            name = self.filaments[i][0]
            mm, grams = float(self.filaments[i][2]), float(self.filaments[i][3])
        if grams:
            return tr("{name}: {g:.1f} g, {m:.2f} m").format(name=name, g=grams, m=mm / 1000.0)
        return tr("{name}: {m:.2f} m").format(name=name, m=mm / 1000.0)

    def _layer_changed(self, value):
        self.view.layer = int(value)
        if not self.play.isChecked():
            self.view.at = None
        self._show_info()
        self.view.update()

    def _show_info(self):
        layers = self.preview.layers
        if not layers:
            self.info.setText(tr("No layers"))
            return
        layer = layers[self.view.layer]
        self.info.setText(tr("Layer {n} of {total}, z {z:.2f} mm, {t}")
                          .format(n=self.view.layer + 1, total=len(layers), z=layer.z,
                                  t=_hms(layer.seconds)))

    def _play_toggled(self, on):
        self.play.setText(tr("Pause") if on else tr("Play"))
        if on:
            if self.view.at is None:
                if self.view.layer == len(self.preview.layers) - 1:
                    self.slider.setValue(0)          # the print is done: start it again
                self.view.at = 0.0
            self.timer.start()
        else:
            self.timer.stop()

    def _tick(self):
        layers = self.preview.layers
        if not layers:
            self.play.setChecked(False)
            return
        self.view.at = (self.view.at or 0.0) + self.speed.currentData() / FPS
        layer = layers[self.view.layer]
        if self.view.at >= layer.seconds:
            if self.view.layer + 1 < len(layers):
                self.view.at = 0.0
                self.slider.setValue(self.view.layer + 1)
            else:
                self.view.at = None
                self.play.setChecked(False)
        self.view.update()

    def _travel_toggled(self, on):
        self.view.show_travel = on
        self.view.update()

    def _only_toggled(self, on):
        self.view.only_layer = on
        self.view.update()

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key_Up, Qt.Key_PageUp):
            self.slider.setValue(self.slider.value() + 1)
        elif event.key() in (Qt.Key_Down, Qt.Key_PageDown):
            self.slider.setValue(self.slider.value() - 1)
        elif event.key() == Qt.Key_Space:
            self.play.toggle()
        else:
            super().keyPressEvent(event)

    # ---- saving
    def save_to(self, path: str) -> str:
        shutil.copyfile(self.gcode_path, path)
        self.saved_to = path
        return path

    def _save(self):
        path = self.ask_path() if self.ask_path else None
        if path:
            self.save_to(path)
            self.accept()

    # ---- sending
    def _show_send(self):
        """Send to printer… only works once there is a printer under Preferences."""
        self.send_button.setEnabled(bool(self.printers))
        self.send_button.setToolTip("" if self.printers else
                                    tr("Add a printer under Window > Preferences > Printers, "
                                       "then send the G-code straight from here."))

    def _send_name(self) -> str:
        """What the G-code is called on the printer: the name of the model, or "volura"."""
        path = getattr(getattr(self.win, "doc", None), "path", None)
        stem = os.path.splitext(os.path.basename(path))[0] if path else None
        return printer_link.file_name(stem)

    def _send(self):
        """Send to printer…: exactly the G-code of this preview, straight to the printer."""
        dlg = SendDialog(self, self.printers, self._send_name())
        if dlg.exec() != QDialog.Accepted:
            return
        conn, name, start = dlg.connection(), dlg.file_name(), dlg.start.isChecked()
        self.send_button.setEnabled(False)

        def work(report):
            printer_link.upload(conn, self.gcode_path, name, start=start,
                                on_progress=lambda fraction: report(fraction))
            return name

        def sent(what):
            self.send_button.setEnabled(True)
            started = tr("Sent {file} to {printer} and started printing")
            plain = tr("Sent {file} to {printer}")
            self.sent = (started if start else plain).format(file=what, printer=conn.name)
            hint = getattr(self.win, "hint", None)
            if hint is not None:
                hint.setText(self.sent)

        def failed(exc):
            self.send_button.setEnabled(True)
            QMessageBox.critical(self, tr("Send to printer"), str(exc))

        self._job = background.run(self, work, can_cancel=True,
                                   text=tr("Sending {file} to {printer}…")
                                   .format(file=name, printer=conn.name),
                                   on_done=sent, on_error=failed,
                                   on_cancel=lambda: self.send_button.setEnabled(True))

    def done(self, result):
        self.timer.stop()
        super().done(result)

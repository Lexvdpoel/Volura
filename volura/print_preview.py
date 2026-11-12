"""The print preview: the sliced G-code in 3D, layer by layer, with a virtual printer that prints it.

After File > Slice the G-code goes here first instead of straight to a file. The view shows the
print in 3D on the bed: every layer up to the chosen one, in the colour of what it prints (outer
wall, inner wall, top and bottom, infill, support, skirt), the layers below it dimmed, the travel
moves of the chosen layer as thin lines when you want them, and the nozzle where it is. Drag with
the left mouse button to turn the view, with the right or middle button (or Shift and the left) to
move it; the wheel zooms towards the mouse and a double-click shows the whole print again. Top and
3D switch the view; Only this layer hides the layers below. Play moves the nozzle through the layer
at the speed of the real print times a factor and goes on to the next layer; the slider and the
arrow keys pick a layer. Save G-code… writes the file.
"""
from __future__ import annotations

import shutil

import numpy as np
from OpenGL import GL as gl
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QColor
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDialogButtonBox, QHBoxLayout,
                               QLabel, QPushButton, QSlider, QVBoxLayout)

from . import errors
from . import gcode_preview as gp
from .camera import Camera
from .i18n import tr, tr_noop

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
#: how much faster than the real print the virtual printer plays
SPEEDS = (1, 5, 20, 100)
#: frames a second of the playback
FPS = 30
#: how bright the layers below the chosen one are, against its own colours
DIM = 0.5
#: the colours of the background, the bed and its grid
BACKGROUND, BED, GRID = "#20242a", "#2b3038", "#3a404a"


def kind_style(colour_blind: bool):
    """The colours and names of every kind of line: the standard or the colour-blind palette."""
    return KIND_STYLE_CB if colour_blind else KIND_STYLE


def _hms(seconds: float) -> str:
    s = int(round(seconds))
    h, rest = divmod(s, 3600)
    m, s = divmod(rest, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def _rgb(hex_color) -> tuple:
    c = QColor(hex_color)
    return c.redF(), c.greenF(), c.blueF()


class _Lines:
    """Every printed line of the print as one array of vertices and colours, layer after layer.

    `start[i]` is the first vertex of layer `i` (and `start[-1]` the end); `done[i]` counts, for
    every move of layer `i`, the printed lines up to and including it, so a layer that is half
    printed draws only those.
    """

    def __init__(self, preview: gp.Preview, colour_blind=False):
        colours = np.array([_rgb(c) for c, _name in kind_style(colour_blind)], dtype=np.float32)
        pos, col, travel, start, tstart, done = [], [], [], [0], [0], []
        for layer in preview.layers:
            printed = layer.kind != gp.TRAVEL
            seg = layer.seg[printed]
            z = np.full((len(seg), 1), layer.z, dtype=np.float32)
            pos.append(np.hstack([seg[:, :2], z, seg[:, 2:], z]).reshape(-1, 3))
            col.append(np.repeat(colours[layer.kind[printed]], 2, axis=0))
            trav = layer.seg[~printed]
            zt = np.full((len(trav), 1), layer.z, dtype=np.float32)
            travel.append(np.hstack([trav[:, :2], zt, trav[:, 2:], zt]).reshape(-1, 3))
            start.append(start[-1] + 2 * len(seg))
            tstart.append(tstart[-1] + 2 * len(trav))
            done.append(np.cumsum(printed))
        empty = np.zeros((0, 3), dtype=np.float32)
        self.pos = np.ascontiguousarray(np.vstack(pos) if pos else empty, dtype=np.float32)
        self.col = np.ascontiguousarray(np.vstack(col) if col else empty, dtype=np.float32)
        self.dim = np.ascontiguousarray(self.col * DIM, dtype=np.float32)
        self.travel = np.ascontiguousarray(np.vstack(travel) if travel else empty,
                                           dtype=np.float32)
        self.start, self.tstart, self.done = start, tstart, done


class LayerView(QOpenGLWidget):
    """The print in 3D on the bed, up to one layer, with the nozzle on it.

    `bed` is (x0, y0, x1, y1): the bed in the coordinates of the G-code. `layer` is the chosen
    layer, `at` the seconds into it the virtual printer is (None: the whole layer).
    """

    def __init__(self, preview: gp.Preview, bed, colour_blind=False, parent=None):
        super().__init__(parent)
        self.preview, self.bed = preview, bed
        self.colour_blind = bool(colour_blind)
        self.lines = _Lines(preview, self.colour_blind)
        self.layer = 0
        self.at = None
        self.show_travel = False
        self.only_layer = False
        self.cam = Camera()
        self._drag, self._last = None, None
        self._fitted = False                    # fitted once the view has its real size
        self.setMinimumSize(420, 420)
        self.setFocusPolicy(Qt.StrongFocus)
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
            end = first + 2 * (int(ln.done[i][moves - 1]) if moves else 0)
        gl.glEnableClientState(gl.GL_VERTEX_ARRAY)
        gl.glEnableClientState(gl.GL_COLOR_ARRAY)
        gl.glVertexPointer(3, gl.GL_FLOAT, 0, ln.pos)
        gl.glLineWidth(1.0)
        if first > 0 and not self.only_layer:
            gl.glColorPointer(3, gl.GL_FLOAT, 0, ln.dim)
            gl.glDrawArrays(gl.GL_LINES, 0, first)
        gl.glLineWidth(2.0)
        gl.glColorPointer(3, gl.GL_FLOAT, 0, ln.col)
        if end > first:
            gl.glDrawArrays(gl.GL_LINES, first, end - first)
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


class PrintPreviewDialog(QDialog):
    """The G-code of a slice, layer by layer; `Save G-code…` hands it on (see `save_to`)."""

    def __init__(self, win, gcode_path: str, preview: gp.Preview, bed, summary: str,
                 ask_path=None):
        super().__init__(win)
        self.gcode_path, self.preview = gcode_path, preview
        self.ask_path = ask_path
        self.saved_to = None
        settings = getattr(win, "settings", None)
        self.colour_blind = bool(getattr(settings, "colour_blind", False))
        self.setWindowTitle(tr("Print Preview"))
        self.resize(980, 780)
        lay = QVBoxLayout(self)
        self.summary = QLabel(summary)
        self.summary.setWordWrap(True)
        lay.addWidget(self.summary)
        row = QHBoxLayout()
        self.view = LayerView(preview, bed, self.colour_blind)
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
        bb.addButton(QDialogButtonBox.Close)
        bb.accepted.connect(self._save)
        bb.rejected.connect(self.reject)
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

    def done(self, result):
        self.timer.stop()
        super().done(result)

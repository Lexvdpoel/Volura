"""The resin print preview: the print on its supports in 3D, and the layer images it exposes.

After File > Slice for Resin the supports are made first and shown here before anything is
written: the model, the supports and the pad in 3D on the build plate, each in its own colour.
Drag with the left mouse button to turn the view, with the right or middle button (or Shift and
the left) to move it; the wheel zooms towards the mouse and a double-click shows the whole print.
The Layers tab shows the image of one layer, as the printer's screen will show it. Save… slices
and writes the printer's file.
"""
from __future__ import annotations

import math

import manifold3d as mf
import numpy as np
from OpenGL import GL as gl
from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import (QBrush, QColor, QImage, QPainter, QPainterPath, QPixmap, QPolygonF,
                           QTransform)
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtWidgets import (QDialog, QDialogButtonBox, QHBoxLayout, QLabel, QSlider,
                               QTabWidget, QVBoxLayout, QWidget)

from . import errors
from .camera import Camera
from .i18n import tr

#: the colours of what the preview shows
MODEL, SUPPORT, PAD, PLATE, BACKGROUND = "#c9ced6", "#4dabf7", "#74c0fc", "#2b3038", "#20242a"


def _rgb(hex_color) -> tuple:
    c = QColor(hex_color)
    return c.redF(), c.greenF(), c.blueF()


def mesh_arrays(manifold):
    """Flat-shaded triangles of a manifold: vertices and normals, three per triangle."""
    m = manifold.to_mesh()
    v = np.asarray(m.vert_properties, dtype=np.float32)[:, :3]
    t = np.asarray(m.tri_verts, dtype=np.int64)
    tri = v[t]                                           # (n, 3, 3)
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    ln = np.linalg.norm(n, axis=1)
    n = n / np.where(ln > 0, ln, 1.0)[:, None]
    return (np.ascontiguousarray(tri.reshape(-1, 3), dtype=np.float32),
            np.ascontiguousarray(np.repeat(n, 3, axis=0), dtype=np.float32))


class MeshView(QOpenGLWidget):
    """Shaded solids on the build plate, with an orbit camera.

    `items` is a list of (manifold, colour); `plate` is (width, depth) of the build plate.
    """

    def __init__(self, plate=(100.0, 100.0), parent=None):
        super().__init__(parent)
        self.plate = plate
        self.items = []                 # (vertices, normals, rgb)
        self.cam = Camera()
        self._drag = self._last = None
        self._fitted = False
        self.setMinimumSize(420, 380)
        self.setFocusPolicy(Qt.StrongFocus)

    def set_items(self, items):
        self.items = [(*mesh_arrays(m), _rgb(c)) for m, c in items
                      if m is not None and not m.is_empty()]
        self._fitted = False
        self.fit()

    def _box(self):
        boxes = [(v.min(axis=0), v.max(axis=0)) for v, _n, _c in self.items if len(v)]
        if not boxes:
            return np.zeros(3), np.array([*self.plate, 10.0])
        return np.min([b[0] for b in boxes], axis=0), np.max([b[1] for b in boxes], axis=0)

    def fit(self, view="iso"):
        if view:
            self.cam.set_view(view)
        self.cam.width, self.cam.height = max(1, self.width()), max(1, self.height())
        self.cam.fit(self._box())
        self.update()

    def initializeGL(self):
        errors.clear_gl_errors()

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
        gl.glUseProgram(0)
        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glLoadMatrixd(self.cam.proj_matrix().T.ravel())
        gl.glMatrixMode(gl.GL_MODELVIEW)
        gl.glLoadIdentity()
        gl.glEnable(gl.GL_LIGHTING)
        gl.glEnable(gl.GL_LIGHT0)
        gl.glEnable(gl.GL_LIGHT1)
        gl.glLightfv(gl.GL_LIGHT0, gl.GL_POSITION, (0.35, 0.55, 1.0, 0.0))
        gl.glLightfv(gl.GL_LIGHT0, gl.GL_DIFFUSE, (0.75, 0.75, 0.75, 1.0))
        gl.glLightfv(gl.GL_LIGHT1, gl.GL_POSITION, (-0.5, -0.3, 0.4, 0.0))
        gl.glLightfv(gl.GL_LIGHT1, gl.GL_DIFFUSE, (0.35, 0.35, 0.35, 1.0))
        gl.glLightModelfv(gl.GL_LIGHT_MODEL_AMBIENT, (0.3, 0.3, 0.3, 1.0))
        gl.glLightModeli(gl.GL_LIGHT_MODEL_TWO_SIDE, gl.GL_TRUE)
        gl.glEnable(gl.GL_COLOR_MATERIAL)
        gl.glColorMaterial(gl.GL_FRONT_AND_BACK, gl.GL_AMBIENT_AND_DIFFUSE)
        gl.glLoadMatrixd(self.cam.view_matrix().T.ravel())
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glDisable(gl.GL_CULL_FACE)
        # the plate
        gl.glDisable(gl.GL_LIGHTING)
        w, d = self.plate
        gl.glColor3f(*_rgb(PLATE))
        gl.glBegin(gl.GL_QUADS)
        for x, y in ((0, 0), (w, 0), (w, d), (0, d)):
            gl.glVertex3f(x, y, -0.02)
        gl.glEnd()
        gl.glEnable(gl.GL_LIGHTING)
        gl.glEnableClientState(gl.GL_VERTEX_ARRAY)
        gl.glEnableClientState(gl.GL_NORMAL_ARRAY)
        for v, n, rgb in self.items:
            gl.glColor3f(*rgb)
            gl.glVertexPointer(3, gl.GL_FLOAT, 0, v)
            gl.glNormalPointer(gl.GL_FLOAT, 0, n)
            gl.glDrawArrays(gl.GL_TRIANGLES, 0, len(v))
        gl.glDisableClientState(gl.GL_NORMAL_ARRAY)
        gl.glDisableClientState(gl.GL_VERTEX_ARRAY)
        gl.glDisable(gl.GL_LIGHTING)

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
        self.fit(None)

    def wheelEvent(self, event):
        steps = event.angleDelta().y() / 120.0
        if steps:
            self.cam.zoom(0.85 ** steps)
            self.update()


class LayerImage(QLabel):
    """One layer of the print, white on black, as seen from above, scaled to fit."""

    def __init__(self, solid, plate, parent=None):
        super().__init__(parent)
        self.solid, self.plate = solid, plate
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(420, 300)
        self.setStyleSheet(f"background: {BACKGROUND};")
        self.z = 0.0

    def show_layer(self, z):
        self.z = z
        self._draw()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._draw()

    def _draw(self):
        w, d = self.plate
        size = self.size()
        scale = max(0.1, min((size.width() - 8) / w, (size.height() - 8) / d))
        img = QImage(max(1, int(w * scale)), max(1, int(d * scale)), QImage.Format_Grayscale8)
        img.fill(0)
        polys = self.solid.slice(self.z).to_polygons() if self.solid is not None else []
        if polys:
            path = QPainterPath()
            path.setFillRule(Qt.OddEvenFill)
            for poly in polys:
                path.addPolygon(QPolygonF([QPointF(float(x), float(y)) for x, y in poly]))
                path.closeSubpath()
            p = QPainter(img)
            p.setRenderHint(QPainter.Antialiasing)
            # from above: y up on the screen
            p.setTransform(QTransform.fromScale(scale, -scale)
                           * QTransform.fromTranslate(0, d * scale))
            p.setPen(Qt.NoPen)
            p.setBrush(QBrush(Qt.white))
            p.drawPath(path)
            p.end()
        self.setPixmap(QPixmap.fromImage(img))


class ResinPreviewDialog(QDialog):
    """The print on its supports in 3D and layer by layer; Save… hands it on to slicing."""

    def __init__(self, win, supported, plate, layer_height, summary, ask_path=None):
        super().__init__(win)
        self.supported, self.ask_path = supported, ask_path
        self.path = None
        self.setWindowTitle(tr("Resin Print Preview"))
        self.resize(980, 780)
        lay = QVBoxLayout(self)
        self.summary = QLabel(summary)
        self.summary.setWordWrap(True)
        lay.addWidget(self.summary)
        self.tabs = QTabWidget()
        self.view = MeshView(plate)
        self.view.set_items([(p, MODEL) for p in supported.parts]
                            + [(supported.supports, SUPPORT), (supported.pad, PAD)])
        self.tabs.addTab(self.view, tr("3D"))
        parts = supported.all
        solid = mf.Manifold.batch_boolean(parts, mf.OpType.Add) if len(parts) > 1 else parts[0]
        top = float(solid.bounding_box()[5])
        self.layer_height = layer_height
        self.count = max(1, int(math.ceil(top / layer_height - 1e-9)))
        page = QWidget()
        row = QHBoxLayout(page)
        self.image = LayerImage(solid, plate)
        row.addWidget(self.image, 1)
        self.slider = QSlider(Qt.Vertical)
        self.slider.setRange(0, self.count - 1)
        row.addWidget(self.slider)
        self.tabs.addTab(page, tr("Layers"))
        lay.addWidget(self.tabs, 1)
        self.info = QLabel()
        lay.addWidget(self.info)
        hint = QLabel(tr("Drag to turn, right-drag to move, wheel to zoom"))
        hint.setEnabled(False)
        lay.addWidget(hint)
        bb = QDialogButtonBox()
        self.save_button = bb.addButton(tr("Slice and Save…"), QDialogButtonBox.AcceptRole)
        bb.addButton(QDialogButtonBox.Close)
        bb.accepted.connect(self._save)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)
        self.slider.valueChanged.connect(self._layer_changed)
        self.slider.setValue(self.count // 2)
        self._layer_changed(self.slider.value())

    def _layer_changed(self, value):
        z = (int(value) + 0.5) * self.layer_height
        self.image.show_layer(z)
        self.info.setText(tr("Layer {n} of {total}, z {z:.2f} mm").format(
            n=int(value) + 1, total=self.count, z=z))

    def _save(self):
        path = self.ask_path() if self.ask_path else None
        if path:
            self.path = path
            self.accept()

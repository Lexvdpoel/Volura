"""The print preview: the sliced G-code layer by layer, with a virtual printer that prints it.

After File > Slice the G-code goes here first instead of straight to a file. The canvas shows the
bed from above with one layer: what it prints in the colour of its kind (outer wall, inner wall,
top and bottom, infill, support, skirt), the layer under it faint, the travel moves as thin lines
when you want them, and the nozzle where it is. Play moves the nozzle through the layer at the
speed of the real print times a factor, and goes on to the next layer; the slider and the arrow
keys pick a layer, the mouse wheel zooms and dragging moves the view. Save G-code… writes the file.
"""
from __future__ import annotations

import shutil

import numpy as np
from PySide6.QtCore import QLineF, QPointF, QRectF, Qt, QTimer
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDialogButtonBox, QHBoxLayout,
                               QLabel, QPushButton, QSlider, QVBoxLayout, QWidget)

from . import gcode_preview as gp
from .i18n import tr, tr_noop

#: the colour and the name of every kind of line, in the order of gcode_preview.KIND_NAMES
KIND_STYLE = [("#9aa4b1", tr_noop("Travel")), ("#d9480f", tr_noop("Outer wall")),
              ("#f08c00", tr_noop("Inner wall")), ("#e8590c", tr_noop("Top and bottom")),
              ("#2f9e44", tr_noop("Infill")), ("#1c7ed6", tr_noop("Support")),
              ("#4dabf7", tr_noop("Support interface")), ("#7048e8", tr_noop("Skirt and brim")),
              ("#ae3ec9", tr_noop("Prime tower")), ("#868e96", tr_noop("Other"))]
#: how much faster than the real print the virtual printer plays
SPEEDS = (1, 5, 20, 100)
#: frames a second of the playback
FPS = 30


def _hms(seconds: float) -> str:
    s = int(round(seconds))
    h, rest = divmod(s, 3600)
    m, s = divmod(rest, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


class LayerView(QWidget):
    """The bed from above with one layer of the print, and the nozzle on it.

    `bed` is (x0, y0, x1, y1): the bed in the coordinates of the G-code.
    """

    def __init__(self, preview: gp.Preview, bed, parent=None):
        super().__init__(parent)
        self.preview, self.bed = preview, bed
        self.layer = 0
        self.at = None                  # seconds into the layer; None: the whole layer
        self.show_travel = False
        self.zoom, self.pan = 1.0, QPointF(0.0, 0.0)
        self._drag = None
        self.setMinimumSize(420, 420)
        self.setFocusPolicy(Qt.StrongFocus)

    # ---- the mapping from the bed to the widget
    def _scale(self):
        x0, y0, x1, y1 = self.bed
        return self.zoom * 0.92 * min(self.width() / max(x1 - x0, 1.0),
                                      self.height() / max(y1 - y0, 1.0))

    def _to_view(self, x, y):
        x0, y0, x1, y1 = self.bed
        s = self._scale()
        cx = 0.5 * self.width() + self.pan.x()
        cy = 0.5 * self.height() + self.pan.y()
        return QPointF(cx + (x - 0.5 * (x0 + x1)) * s, cy - (y - 0.5 * (y0 + y1)) * s)

    def paintEvent(self, _event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.fillRect(self.rect(), QColor("#20242a"))
        x0, y0, x1, y1 = self.bed
        p.setPen(QPen(QColor("#5c6370"), 1.0))
        p.setBrush(QColor("#2b3038"))
        p.drawRect(QRectF(self._to_view(x0, y1), self._to_view(x1, y0)))
        layers = self.preview.layers
        if not layers:
            return
        s = self._scale()
        width = max(1.0, 0.4 * s)
        if self.layer > 0:
            self._draw(p, layers[self.layer - 1], None, QColor(255, 255, 255, 40), width, False)
        layer = layers[self.layer]
        self._draw(p, layer, self.at, None, width, self.show_travel)
        if self.at is not None:
            pos, _done = layer.position(self.at)
            if pos is not None:
                c = self._to_view(*pos)
                p.setPen(QPen(QColor("#ffffff"), 2.0))
                p.setBrush(QColor("#ff6b6b"))
                p.drawEllipse(c, 5.0, 5.0)

    def _draw(self, p, layer, at, color, width, travel):
        count = len(layer.seg)
        if at is not None:
            count = min(count, int(np.searchsorted(layer.t, at, side="right")) + 1)
        seg, kinds = layer.seg[:count], layer.kind[:count]
        for k in np.unique(kinds):
            if k == gp.TRAVEL and not travel:
                continue
            rows = seg[kinds == k]
            pen = QPen(color if color is not None else QColor(KIND_STYLE[int(k)][0]),
                       1.0 if k == gp.TRAVEL else width)
            pen.setCapStyle(Qt.RoundCap)
            p.setPen(pen)
            p.drawLines([QLineF(self._to_view(r[0], r[1]), self._to_view(r[2], r[3]))
                         for r in rows])

    # ---- zoom and pan
    def wheelEvent(self, event):
        factor = 1.2 if event.angleDelta().y() > 0 else 1.0 / 1.2
        self.zoom = max(0.5, min(40.0, self.zoom * factor))
        self.update()

    def mousePressEvent(self, event):
        self._drag = event.position()

    def mouseMoveEvent(self, event):
        if self._drag is not None:
            self.pan += event.position() - self._drag
            self._drag = event.position()
            self.update()

    def mouseReleaseEvent(self, _event):
        self._drag = None

    def mouseDoubleClickEvent(self, _event):
        self.zoom, self.pan = 1.0, QPointF(0.0, 0.0)
        self.update()


class PrintPreviewDialog(QDialog):
    """The G-code of a slice, layer by layer; `Save G-code…` hands it on (see `save_to`)."""

    def __init__(self, win, gcode_path: str, preview: gp.Preview, bed, summary: str,
                 ask_path=None):
        super().__init__(win)
        self.gcode_path, self.preview = gcode_path, preview
        self.ask_path = ask_path
        self.saved_to = None
        self.setWindowTitle(tr("Print Preview"))
        self.resize(900, 760)
        lay = QVBoxLayout(self)
        self.summary = QLabel(summary)
        self.summary.setWordWrap(True)
        lay.addWidget(self.summary)
        row = QHBoxLayout()
        self.view = LayerView(preview, bed)
        row.addWidget(self.view, 1)
        self.slider = QSlider(Qt.Vertical)
        self.slider.setRange(0, max(0, len(preview.layers) - 1))
        self.slider.setValue(0)
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
        self.info = QLabel()
        controls.addWidget(self.play)
        controls.addWidget(self.speed)
        controls.addWidget(self.travel)
        controls.addStretch(1)
        controls.addWidget(self.info)
        lay.addLayout(controls)

        legend = QHBoxLayout()
        used = set()
        for layer in preview.layers:
            used.update(int(k) for k in np.unique(layer.kind))
        for k, (color, name) in enumerate(KIND_STYLE):
            if k in used and k != gp.TRAVEL:
                swatch = QLabel(f"<span style='color:{color}'>■</span> {tr(name)}")
                legend.addWidget(swatch)
        legend.addStretch(1)
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
        self._layer_changed(0)

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

"""Main window of Volura."""
from __future__ import annotations

import math
import os
import sys
import threading
import time
from contextlib import contextmanager
from dataclasses import replace

import numpy as np
import manifold3d as mf
from PySide6.QtCore import (QObject, QPointF, QRectF, QSettings, QSize, Qt, QTimer, QUrl,
                            QVariantAnimation, Signal)
from PySide6.QtGui import (QAction, QActionGroup, QColor, QFont, QIcon, QKeySequence, QPainter,
                           QPainterPath, QPen, QPixmap, QDesktopServices)
from PySide6.QtWidgets import (QAbstractItemView, QApplication, QCheckBox, QColorDialog, QComboBox,
                               QDialog, QDialogButtonBox, QDoubleSpinBox, QFileDialog,
                               QFontComboBox, QFormLayout, QGroupBox, QHBoxLayout, QHeaderView,
                               QInputDialog,
                               QLabel, QLineEdit, QKeySequenceEdit, QListWidget, QListWidgetItem,
                               QMainWindow, QMenu, QMessageBox, QProgressDialog, QPushButton, QRadioButton,
                               QSizePolicy, QSpinBox, QStackedWidget, QTableWidget,
                               QTableWidgetItem, QTreeWidget, QTreeWidgetItem, QTextBrowser,
                               QToolBar, QToolButton, QVBoxLayout, QWidget)

from . import (__version__, boxgen, cam, dxf, errors, fileio, joints, nesting, ops, sheet3d, svg,
               threads, updater, vector_export, vector_import)
from .brep import Entities
from .geom import Plane
from .icons import icon
from .model import ROOT, LOCKED_MSG, Document
from .sheetview import SheetView
from .tools import TOOL_CLASSES, fmt_deg, format_length, parse_length, set_length_precision
from .tray import TRAY_WIDTH, Tray
from .vector2d import (CORNER_LABELS, CORNER_STYLES, SHEET_SPACING, Drawing, LASER_OPS,
                       LASER_PARAMS, MILL_OPS, MILL_PARAMS, OP_LABELS, OPS, PARAM_LABELS,
                       VPath, _transform_meta, layer_param)
from .viewport import (FACE_STYLES, HIDDENLINE, MONOCHROME, SHADED, WIREFRAME, Viewport)

PRINTERS = [
    ("Bambu Lab X1 / P1 / A1", (256, 256, 256)),
    ("Bambu Lab A1 mini", (180, 180, 180)),
    ("Prusa MK4 / MK3S", (250, 210, 220)),
    ("Prusa MINI", (180, 180, 180)),
    ("Prusa XL", (360, 360, 360)),
    ("Creality Ender 3", (220, 220, 250)),
    ("Creality K1", (220, 220, 250)),
    ("Voron 2.4 (350)", (350, 350, 340)),
]

TOOL_INFO = [
    # name, label, icon, shortcut
    ("select", "Selecteren", "select", "Space"),
    ("eraser", "Gum", "eraser", "E"),
    ("paint", "Verfemmer", "paint", "B"),
    ("line", "Lijn", "line", "L"),
    ("rectangle", "Rechthoek", "rectangle", "R"),
    ("circle", "Cirkel", "circle", "C"),
    ("polygon", "Veelhoek", "polygon", None),
    ("text", "3D-tekst", "text", "Shift+T"),
    ("pushpull", "Duwen/trekken", "pushpull", "P"),
    ("offset", "Verschuiving", "offset", "F"),
    ("move", "Verplaatsen", "move", "M"),
    ("rotate", "Roteren", "rotate", "Q"),
    ("scale", "Schalen", "scale", "S"),
    ("tape", "Meetlint", "tape", "T"),
    ("orbit", "Draaien", "orbit", "O"),
    ("pan", "Pannen", "pan", "H"),
    ("zoom", "Zoomen", "zoom", "Z"),
]

# De gereedschappen van het 2D-werkblad; de toetsen doen het alleen in
# 2D, want in 3D horen M, Q, S en spatie toe aan de 3D-gereedschappen. De letters van de
# tekengereedschappen zijn dezelfde als in 3D (L, R, C, A, F), maar zitten op het werkblad
# zelf, zodat er geen dubbele sneltoetsen in het venster ontstaan.
class ModeSwitch(QToolButton):
    """De schuifknop 3D | 2D rechtsonder: de helft van de huidige modus is blauw.

    Niet ingedrukt is 3D, ingedrukt is 2D; bij het wisselen schuift het blauwe blokje
    naar de andere kant. De tekst blijft "3D | 2D" voor schermlezers en sneltoetsen.
    """
    ON = QColor("#2f6fd6")
    TRACK = QColor("#e6eaf0")
    BORDER = QColor("#b9c3d0")
    TEXT = QColor("#4b5563")

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("modeSwitch")
        self.setCheckable(True)
        self.setText("3D | 2D")
        self.setCursor(Qt.PointingHandCursor)
        self._pos = 0.0                     # 0 = blokje op 3D, 1 = op 2D
        self._anim = QVariantAnimation(self)
        self._anim.setDuration(140)
        self._anim.valueChanged.connect(self._moved)
        self.toggled.connect(self._slide)

    def sizeHint(self):
        fm = self.fontMetrics()
        return QSize(2 * (fm.horizontalAdvance("3D") + 22), max(22, fm.height() + 8))

    def minimumSizeHint(self):
        return self.sizeHint()

    def _moved(self, value):
        self._pos = float(value)
        self.update()

    def _slide(self, checked):
        target = 1.0 if checked else 0.0
        self._anim.stop()
        if not self.isVisible():
            self._moved(target)
            return
        self._anim.setStartValue(self._pos)
        self._anim.setEndValue(target)
        self._anim.start()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        r = QRectF(self.rect()).adjusted(1.0, 2.0, -1.0, -2.0)
        radius = r.height() / 2.0
        p.setPen(QPen(self.BORDER, 1.0))
        p.setBrush(self.TRACK.darker(104) if self.underMouse() else self.TRACK)
        p.drawRoundedRect(r, radius, radius)
        half = r.width() / 2.0
        knob = QRectF(r.left() + 1.5 + self._pos * (half - 1.5), r.top() + 1.5,
                      half - 1.5, r.height() - 3.0)
        p.setPen(Qt.NoPen)
        p.setBrush(self.ON)
        p.drawRoundedRect(knob, knob.height() / 2.0, knob.height() / 2.0)
        font = self.font()
        font.setBold(True)
        p.setFont(font)
        for i, label in enumerate(("3D", "2D")):
            part = QRectF(r.left() + i * half, r.top(), half, r.height())
            active = abs(self._pos - i) < 0.5
            p.setPen(QColor("#ffffff") if active else self.TEXT)
            p.drawText(part, Qt.AlignCenter, label)
        p.end()


SHEET_TOOL_INFO = [
    # naam in volura.sheettools, label, hulptekst
    ("select", "Selecteren", "Klik een lijn of sleep een kader (spatie)"),
    ("move", "Verplaatsen", "Basispunt en doel klikken, of een afstand typen (M)"),
    ("rotate", "Draaien", "Middelpunt klikken en een hoek in graden typen (Q)"),
    ("scale", "Schalen", "Sleep vanaf het middelpunt of typ een factor (S)"),
    ("line", "Lijn", "Punten klikken; klik op het startpunt om te sluiten (L)"),
    ("rectangle", "Rechthoek", "Twee hoeken klikken of 30;20 typen (R)"),
    ("roundrect", "Afgeronde rechthoek", "Rechthoek met afgeronde hoeken, straal typen met r (N)"),
    ("circle", "Cirkel", "Middelpunt en straal klikken of typen (C)"),
    ("ellipse", "Ellips", "Middelpunt klikken en de twee halvematen slepen (E)"),
    ("arc", "Boog", "Boog door drie punten (A)"),
    ("centerarc", "Boog om middelpunt", "Middelpunt, straal en hoek van de boog (B)"),
    ("polygon", "Veelhoek", "Middelpunt en straal; aantal zijden typen zoals 6s (P)"),
    ("text", "Tekst", "De contouren van tekst op het werkblad plaatsen (T)"),
    ("offset", "Verschuiving", "Een gesloten contour naar binnen of buiten schuiven (F)"),
]


class Settings:
    def __init__(self):
        self.q = QSettings("Volura", "Volura")
        self.bed_size = tuple(float(v) for v in self.q.value("bed_size", [256, 256, 256]))
        self.show_grid = self._bool("show_grid", True)
        self.show_axes = self._bool("show_axes", True)
        self.show_edges = self._bool("show_edges", True)
        self.xray = self._bool("xray", False)
        self.face_style = str(self.q.value("face_style", SHADED))
        if self.face_style not in FACE_STYLES:
            self.face_style = SHADED
        self.back_edges = self._bool("back_edges", False)
        self.show_hidden = self._bool("show_hidden", False)
        self.show_guides = self._bool("show_guides", True)
        self.hide_rest = self._bool("hide_rest", False)
        self.circle_segments = int(self.q.value("circle_segments", 64))
        self.length_snap = float(self.q.value("length_snap", 1.0))
        try:
            self.length_precision = int(self.q.value("length_precision", 2))
        except (TypeError, ValueError):
            self.length_precision = 2
        if self.length_precision not in (0, 1, 2, 3):
            self.length_precision = 2
        self.current_color = str(self.q.value("current_color", "#d23b3b"))
        self.check_updates = self._bool("check_updates", True)
        self.recent = list(self.q.value("recent", []) or [])
        if isinstance(self.recent, str):
            self.recent = [self.recent]

    def _bool(self, key, default):
        v = self.q.value(key, default)
        return v in (True, "true", "1", 1)

    def save(self):
        self.q.setValue("bed_size", list(self.bed_size))
        self.q.setValue("show_grid", self.show_grid)
        self.q.setValue("show_axes", self.show_axes)
        self.q.setValue("show_edges", self.show_edges)
        self.q.setValue("xray", self.xray)
        self.q.setValue("check_updates", self.check_updates)
        self.q.setValue("face_style", self.face_style)
        self.q.setValue("back_edges", self.back_edges)
        self.q.setValue("show_hidden", self.show_hidden)
        self.q.setValue("show_guides", self.show_guides)
        self.q.setValue("hide_rest", self.hide_rest)
        self.q.setValue("circle_segments", self.circle_segments)
        self.q.setValue("length_snap", self.length_snap)
        self.q.setValue("length_precision", self.length_precision)
        self.q.setValue("current_color", self.current_color)
        self.q.setValue("recent", self.recent[:8])


def swatch(color, size=16):
    pm = QPixmap(size, size)
    pm.fill(QColor(color))
    return QIcon(pm)


# ---------------------------------------------------------------- panels

class MaterialsPanel(QWidget):
    def __init__(self, win):
        super().__init__()
        self.win = win
        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        lay.addWidget(QLabel("Filamenten (volgorde = extrudernummer):"))
        self.list = QListWidget()
        self.list.setIconSize(QSize(22, 22))
        self.list.itemClicked.connect(self.pick)
        self.list.itemDoubleClicked.connect(lambda it: self.edit())
        lay.addWidget(self.list)
        row = QHBoxLayout()
        for text, fn in (("Nieuw", self.add), ("Wijzig", self.edit), ("Verwijder", self.remove)):
            b = QPushButton(text)
            b.clicked.connect(fn)
            row.addWidget(b)
        lay.addLayout(row)
        self.refresh()

    def refresh(self):
        doc = self.win.doc
        self.list.clear()
        cur = self.win.settings.current_color.lower()
        for i, (name, color) in enumerate(doc.materials, 1):
            it = QListWidgetItem(swatch(color, 22), f"{i}: {name}")
            it.setData(Qt.UserRole, color)
            self.list.addItem(it)
            if color.lower() == cur:
                self.list.setCurrentItem(it)

    def pick(self, it):
        self.win.settings.current_color = it.data(Qt.UserRole)
        self.win.set_tool("paint")

    def add(self):
        c = QColorDialog.getColor(QColor("#888888"), self, "Nieuwe filamentkleur")
        if not c.isValid():
            return
        doc = self.win.doc
        doc.checkpoint()
        doc.materials.append([f"Kleur {len(doc.materials) + 1}", c.name()])
        self.win.settings.current_color = c.name()
        doc.notify()

    def edit(self):
        row = self.list.currentRow()
        doc = self.win.doc
        if row < 0:
            return
        name, color = doc.materials[row]
        dlg = QDialog(self)
        dlg.setWindowTitle("Filament wijzigen")
        form = QFormLayout(dlg)
        ed = QLineEdit(name)
        btn = QPushButton()
        state = {"color": color}
        btn.setIcon(swatch(color, 20))
        btn.setText(color)

        def choose():
            c = QColorDialog.getColor(QColor(state["color"]), dlg)
            if c.isValid():
                state["color"] = c.name()
                btn.setIcon(swatch(c.name(), 20))
                btn.setText(c.name())
        btn.clicked.connect(choose)
        form.addRow("Naam", ed)
        form.addRow("Kleur", btn)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(dlg.accept)
        bb.rejected.connect(dlg.reject)
        form.addRow(bb)
        if dlg.exec() != QDialog.Accepted:
            return
        doc.checkpoint()
        new = state["color"]
        doc.materials[row] = [ed.text() or name, new]
        if new.lower() != color.lower():
            for g in doc.groups:
                if g.color.lower() == color.lower():
                    g.color = new
            for ents in [doc.root] + [g.ents for g in doc.groups]:
                for f, face in list(ents.F.items()):
                    if face.color and face.color.lower() == color.lower():
                        ents.set_face_color(f, new)
            if self.win.settings.current_color.lower() == color.lower():
                self.win.settings.current_color = new
        doc.notify()

    def remove(self):
        row = self.list.currentRow()
        doc = self.win.doc
        if row < 0 or len(doc.materials) <= 1:
            return
        doc.checkpoint()
        doc.materials.pop(row)
        doc.notify()


class EntityInfoPanel(QWidget):
    def __init__(self, win):
        super().__init__()
        self.win = win
        self.lay = QVBoxLayout(self)
        self.lay.setContentsMargins(6, 6, 6, 6)
        self.lay.addStretch(1)
        self.body = None

    def refresh(self):
        if self.body is not None:
            self.lay.removeWidget(self.body)
            self.body.hide()
            self.body.deleteLater()
        self.body = QWidget()
        form = QFormLayout(self.body)
        form.setContentsMargins(0, 0, 0, 0)
        doc = self.win.doc
        if getattr(self.win, "mode", "3d") == "2d":
            self._sheet_form(form)
            self.lay.insertWidget(0, self.body)
            return
        sel = doc.selection
        groups = doc.selected_groups()
        faces = [(c, i) for k, c, i in sel if k == "f"]
        edges = [(c, i) for k, c, i in sel if k == "e"]
        guides = [(c, i) for k, c, i in sel if k == "guide"]
        if doc.active != ROOT and doc.group(doc.active):
            form.addRow(QLabel(f"<i>Bewerken: {doc.group(doc.active).name}</i>"))
        if not sel:
            form.addRow(QLabel("<b>Geen selectie</b>"))
            form.addRow("Groepen", QLabel(str(len(doc.groups))))
            form.addRow("Losse vlakken", QLabel(str(len(doc.root.F))))
            form.addRow("Losse randen", QLabel(str(len(doc.root.E))))
            form.addRow("Hulplijnen", QLabel(str(len(doc.guides))))
        elif len(sel) == 1 and groups:
            self._group_form(form, groups[0])
        elif len(sel) == 1 and guides:
            g = doc.guide(guides[0][1])
            if g is not None:
                form.addRow(QLabel("<b>Hulplijn</b>" if g.is_line else "<b>Hulppunt</b>"))
                form.addRow("Punt", QLabel(f"{g.point[0]:.2f}, {g.point[1]:.2f}, {g.point[2]:.2f} mm"))
                if g.is_line:
                    d = g.direction
                    form.addRow("Richting", QLabel(f"{d[0]:.3f}, {d[1]:.3f}, {d[2]:.3f}"))
        elif len(sel) == 1 and faces:
            c, f = faces[0]
            ents = doc.ents(c)
            tri, _ = ents.face_triangles(f)
            area = float(np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1).sum() / 2)
            form.addRow(QLabel("<b>Vlak</b>"))
            form.addRow("Oppervlakte", QLabel(f"{area:.2f} mm²"))
            form.addRow("Kleur", QLabel(ents.F[f].color or "standaard"))
        elif len(sel) == 1 and edges:
            c, e = edges[0]
            ents = doc.ents(c)
            A, B = ents.edge_points(e)
            form.addRow(QLabel("<b>Rand</b>"))
            form.addRow("Lengte", QLabel(f"{np.linalg.norm(B - A):.3f} mm"))
            form.addRow("Vlakken", QLabel(str(len(ents.e2f.get(e, ())))))
            form.addRow("Verzacht", QLabel("ja" if e in ents.soft else "nee"))
        else:
            form.addRow(QLabel(f"<b>{len(sel)} items geselecteerd</b>"))
            form.addRow("Groepen", QLabel(str(len(groups))))
            form.addRow("Vlakken", QLabel(str(len(faces))))
            form.addRow("Randen", QLabel(str(len(edges))))
            if guides:
                form.addRow("Hulplijnen", QLabel(str(len(guides))))
            bb = ops.selection_bbox(doc)
            if bb is not None:
                d = bb[1] - bb[0]
                form.addRow("Afmetingen", QLabel(f"{d[0]:.2f} × {d[1]:.2f} × {d[2]:.2f} mm"))
        self.lay.insertWidget(0, self.body)

    def _sheet_form(self, form):
        """Entiteitsinfo op het werkblad: de geselecteerde contouren en de plaat."""
        sheet = self.win.doc.sheet
        sel = self.win.sheet_view.selected_paths()
        paths = sel or sheet.paths
        if sel:
            form.addRow(QLabel(f"<b>{len(sel)} contour{'en' if len(sel) != 1 else ''} "
                               "geselecteerd</b>"))
        else:
            form.addRow(QLabel("<b>Geen selectie</b>"))
            form.addRow("Contouren", QLabel(str(len(sheet.paths))))
        layers = sorted({p.layer for p in paths})
        if sel and len(layers) == 1:
            lay = sheet.layers.get(layers[0])
            op = OP_LABELS.get(lay.op, lay.op) if lay else ""
            form.addRow("Laag", QLabel(f"{layers[0]} ({op})" if op else layers[0]))
        elif layers:
            form.addRow("Lagen", QLabel(str(len(layers))))
        if sel:
            closed = sum(1 for p in sel if p.closed)
            form.addRow("Gesloten", QLabel(f"{closed} van {len(sel)}"))
            pts = [p.pts for p in sel if len(p.pts)]
            if pts:
                P = np.vstack(pts)
                lo, hi = P.min(axis=0), P.max(axis=0)
                form.addRow("Afmetingen", QLabel(f"{hi[0] - lo[0]:.2f} × {hi[1] - lo[1]:.2f} mm"))
                form.addRow("Positie", QLabel(f"{lo[0]:.2f}, {lo[1]:.2f} mm"))
        form.addRow(QLabel("<b>Plaat</b>"))
        if sheet.sheet_w and sheet.sheet_h:
            form.addRow("Formaat", QLabel(f"{sheet.sheet_w:g} × {sheet.sheet_h:g} mm"))
            form.addRow("Aantal", QLabel(str(int(sheet.sheet_count or 1))))
        else:
            form.addRow("Formaat", QLabel("niet ingesteld"))
        form.addRow("Dikte", QLabel(f"{sheet.sheet_thickness:g} mm" if sheet.sheet_thickness
                                    else "niet ingesteld"))

    def _group_form(self, form, g):
        doc = self.win.doc
        form.addRow(QLabel("<b>Groep (printbaar onderdeel)</b>"))
        name = QLineEdit(g.name)

        def rename():
            if name.text() and name.text() != g.name:
                self._apply(lambda: setattr(doc.group(g.id), "name", name.text()))
        name.editingFinished.connect(rename)
        form.addRow("Naam", name)
        combo = QComboBox()
        idx = -1
        for i, (mname, color) in enumerate(doc.materials):
            combo.addItem(swatch(color), f"{i + 1}: {mname}", color)
            if color.lower() == g.color.lower():
                idx = i
        if idx < 0:
            combo.addItem(swatch(g.color), g.color, g.color)
            idx = combo.count() - 1
        combo.setCurrentIndex(idx)

        def recolor(i):
            c = combo.itemData(i)
            if c and c.lower() != g.color.lower():
                self._apply(lambda: setattr(doc.group(g.id), "color", c))
        combo.currentIndexChanged.connect(recolor)
        form.addRow("Filament", combo)
        bb = g.bbox()
        if bb is None:
            return
        lo, hi = bb
        size = hi - lo
        spins = []
        for label, v in zip(("Breedte (X)", "Diepte (Y)", "Hoogte (Z)"), size):
            sp = QDoubleSpinBox()
            sp.setRange(0.01, 100000)
            sp.setDecimals(2)
            sp.setSuffix(" mm")
            sp.setValue(v)
            spins.append(sp)
            form.addRow(label, sp)
        lock = QCheckBox("Verhoudingen behouden")
        lock.setChecked(True)
        form.addRow(lock)

        def resize(changed):
            new = np.array([s.value() for s in spins])
            if np.allclose(new, size, atol=1e-3):
                return
            if lock.isChecked() and size[changed] > 1e-9:
                new = size * (new[changed] / size[changed])
            self._apply(lambda: ops.resize_group(doc, doc.group(g.id), new))
        for i, sp in enumerate(spins):
            sp.editingFinished.connect(lambda i=i: resize(i))
        form.addRow("Positie", QLabel(", ".join(f"{round(v, 2) + 0.0:.2f}" for v in lo) + " mm"))
        part = ops._part(g.name, g.color, g.ents, list(g.ents.F))
        if part.manifold is not None:
            m = part.manifold
            form.addRow("Volume", QLabel(f"{m.volume() / 1000:.3f} cm³"))
            form.addRow("Oppervlak", QLabel(f"{m.surface_area() / 100:.2f} cm²"))
            ok = QLabel("✔ gesloten volume (solid)")
            ok.setStyleSheet("color: #2e7d32; font-weight: bold;")
        else:
            ok = QLabel(f"✖ {part.problem}")
            ok.setStyleSheet("color: #c62828; font-weight: bold;")
            ok.setWordWrap(True)
        form.addRow("Printbaar", ok)

    def _apply(self, fn):
        doc = self.win.doc
        doc.checkpoint()
        fn()
        QTimer.singleShot(0, doc.notify)


class OutlinerPanel(QWidget):
    def __init__(self, win):
        super().__init__()
        self.win = win
        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        self.list = QListWidget()
        self.list.setSelectionMode(QListWidget.ExtendedSelection)
        self.list.itemSelectionChanged.connect(self.on_select)
        self.list.itemChanged.connect(self.on_changed)
        self.list.itemDoubleClicked.connect(self.on_double)
        lay.addWidget(self.list)
        self._updating = False

    def refresh(self):
        self._updating = True
        doc = self.win.doc
        self.list.clear()
        sel = {i for k, _, i in doc.selection if k == "g"}
        for g in doc.groups:
            it = QListWidgetItem(swatch(g.color), g.name)
            it.setData(Qt.UserRole, g.id)
            it.setFlags(it.flags() | Qt.ItemIsUserCheckable | Qt.ItemIsEditable)
            it.setCheckState(Qt.Checked if g.visible else Qt.Unchecked)
            self.list.addItem(it)
            it.setSelected(g.id in sel)
        if doc.root.F or doc.root.E:
            it = QListWidgetItem(f"Losse geometrie ({len(doc.root.F)} vlakken)")
            it.setData(Qt.UserRole, None)
            it.setFlags(it.flags() & ~Qt.ItemIsSelectable)
            self.list.addItem(it)
        self._updating = False

    def on_select(self):
        if self._updating:
            return
        doc = self.win.doc
        if doc.active != ROOT:
            return
        ids = [it.data(Qt.UserRole) for it in self.list.selectedItems() if it.data(Qt.UserRole)]
        keep = [s for s in doc.selection if s[0] != "g"]
        doc.set_selection(keep + [("g", i, i) for i in ids])

    def on_double(self, it):
        gid = it.data(Qt.UserRole)
        if gid and not self.win.doc.enter_group(gid):
            self.win.warn_locked()      # a locked group does not open

    def on_changed(self, it):
        if self._updating:
            return
        doc = self.win.doc
        g = doc.group(it.data(Qt.UserRole)) if it.data(Qt.UserRole) else None
        if g is None:
            return
        vis = it.checkState() == Qt.Checked
        if vis != g.visible or it.text() != g.name:
            doc.checkpoint()
            g.visible = vis
            g.name = it.text() or g.name
            QTimer.singleShot(0, doc.notify)


class PrintCheckPanel(QWidget):
    """Tray paneel "Printcontrole": waarom iets geen geldig volume is (Solid Inspector).

    One row per part with its problems underneath. Clicking a problem selects the
    geometry involved and marks it red in the viewport; double clicking zooms on it."""

    def __init__(self, win):
        super().__init__()
        self.win = win
        self.problems = []
        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        lay.setSpacing(4)
        self.tree = QTreeWidget()
        self.tree.setHeaderHidden(True)
        self.tree.setUniformRowHeights(True)
        self.tree.currentItemChanged.connect(self.on_current)
        self.tree.itemDoubleClicked.connect(self.on_double)
        lay.addWidget(self.tree, 1)
        row = QHBoxLayout()
        self.b_check = QPushButton("Controleren")
        self.b_check.clicked.connect(self.refresh)
        self.b_fix = QPushButton("Alles repareren")
        self.b_fix.clicked.connect(self.repair)
        row.addWidget(self.b_check)
        row.addWidget(self.b_fix)
        lay.addLayout(row)
        self.summary = QLabel("Klik op Controleren om elk onderdeel op een gesloten "
                              "volume te testen.")
        self.summary.setWordWrap(True)
        lay.addWidget(self.summary)

    # ------------------------------------------------------------ checking
    def refresh(self, checked=False):
        self.problems = ops.inspect(self.win.doc)
        by_part, order = {}, []
        for n, p in enumerate(self.problems):
            key = (p.ctx, p.part)
            if key not in by_part:
                by_part[key] = []
                order.append(key)
            by_part[key].append(n)
        self.tree.blockSignals(True)
        self.tree.clear()
        for key in order:
            head = QTreeWidgetItem([f"{key[1]}  ({len(by_part[key])})"])
            head.setForeground(0, QColor("#c62828"))
            self.tree.addTopLevelItem(head)
            for n in by_part[key]:
                p = self.problems[n]
                it = QTreeWidgetItem([p.text()])
                it.setData(0, Qt.UserRole, n)
                it.setToolTip(0, f"{p.kind}: {p.detail}\n{len(p.faces)} vlakken, {len(p.edges)} randen")
                head.addChild(it)
            head.setExpanded(True)
        if not order:
            ok = QTreeWidgetItem(["✔ elk onderdeel is een gesloten volume"])
            ok.setForeground(0, QColor("#2e7d32"))
            self.tree.addTopLevelItem(ok)
        self.tree.blockSignals(False)
        if order:
            bad = len({p.part for p in self.problems})
            self.summary.setText(f"{bad} onderdeel(en) met problemen. Klik op een probleem om het "
                                 "te zien; dubbelklik zoomt erop.")
        else:
            self.summary.setText("Alle onderdelen zijn gesloten volumes en kunnen geprint worden.")
        self.win.view.problem_items = []
        self.win.view.update()

    def repair(self, checked=False):
        """Alles repareren: één undo-stap die de onderdelen sluit en oriënteert."""
        fixed = ops.repair_all(self.win.doc)
        self.refresh()
        self.win.hint.setText("Printcontrole: onderdelen gerepareerd (Ongedaan maken maakt het terug)."
                              if fixed else "Printcontrole: er was niets te repareren.")

    # ------------------------------------------------------------ picking
    def _problem(self, it):
        n = None if it is None else it.data(0, Qt.UserRole)
        if n is None or n >= len(self.problems):
            return None
        return self.problems[n]

    def on_current(self, cur, prev=None):
        p = self._problem(cur)
        if p is None:
            self.win.view.problem_items = []
            self.win.view.update()
            return
        items = p.items()
        self.win.view.problem_items = items
        self.win.doc.set_selection(items)
        self.win.view.update()

    def on_double(self, it, column=0):
        p = self._problem(it)
        if p is None:
            return
        items = p.items()
        self.win.view.problem_items = items
        self.win.doc.set_selection(items)
        self.win.view.zoom_items(items)

    def clear_marks(self):
        self.win.view.problem_items = []


# ---------------------------------------------------------------- lagen van het 2D-werkblad

class LayerParamsDialog(QDialog):
    """Dialoog "Instellingen van een laag": wat de machine met die laag moet doen.

    Een laserlaag krijgt vermogen, snelheid, aantal doorgangen en de kerf; een
    freeslaag de freesdiameter, de diepte, de stapdiepte, de stapover en de
    hoekuitsparing voor de binnenhoeken. De waarden komen in `Layer.params` te staan
    en gaan zo mee in het bestand en in de SVG-export.
    """

    SUFFIX = {"power": " %", "stepover": " %", "speed": " mm/s"}

    def __init__(self, layer, parent=None):
        super().__init__(parent)
        self.layer = layer
        self.op = getattr(layer, "op", "cut")
        self.fields = {}
        self.setWindowTitle(f"Instellingen van laag {layer.name}")
        form = QFormLayout(self)
        form.addRow(QLabel(f"Bewerking: {OP_LABELS.get(self.op, self.op)}"))
        keys = (LASER_PARAMS if self.op in LASER_OPS
                else MILL_PARAMS if self.op in MILL_OPS else ())
        for key, default in keys:
            value = layer_param(layer, key, default)
            if key == "passes":
                w = QSpinBox()
                w.setRange(1, 50)
                w.setSuffix("×")
                w.setValue(int(value))
            elif key == "corner":
                w = QComboBox()
                for style in CORNER_STYLES:
                    w.addItem(CORNER_LABELS[style], style)
                w.setCurrentIndex(max(0, list(CORNER_STYLES).index(
                    value if value in CORNER_STYLES else "none")))
            else:
                w = QDoubleSpinBox()
                w.setDecimals(3)
                w.setRange(0.0, 100000.0)
                w.setSingleStep(0.05 if key in ("kerf", "tool_d", "step_depth", "depth") else 1.0)
                w.setSuffix(self.SUFFIX.get(key, " mm"))
                w.setValue(float(value))
            w.setToolTip(PARAM_LABELS.get(key, key))
            form.addRow(PARAM_LABELS.get(key, key), w)
            self.fields[key] = w
        if not keys:
            form.addRow(note("Deze laag is alleen ter informatie en wordt niet geëxporteerd."))
        elif self.op in MILL_OPS:
            form.addRow(note("Bij de correctie schuift de contour met de halve freesdiameter opzij. "
                             "Uitkameren rijdt concentrische paden met de stapover, tot het gebied "
                             "op is. Een hoekuitsparing geeft de frees ruimte in de binnenhoeken."))
        else:
            form.addRow(note("De kerfcorrectie schuift buitenranden de halve kerf naar buiten en "
                             "maakt gaten evenveel kleiner, zodat het onderdeel na het snijden de "
                             "getekende maat heeft."))
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        form.addRow(bb)

    def values(self) -> dict:
        out = {}
        for key, w in self.fields.items():
            if isinstance(w, QSpinBox):
                out[key] = int(w.value())
            elif isinstance(w, QComboBox):
                out[key] = str(w.currentData())
            else:
                out[key] = float(w.value())
        return out


class LayersPanel(QWidget):
    """Tray paneel "Lagen": kleur, naam, bewerking, zichtbaarheid en instellingen per laag.

    Eén rij per laag van het 2D-werkblad. Klik het kleurvak om een kleur te kiezen
    (rood snijdt, blauw graveert, zoals lasersoftware het verwacht), dubbelklik de naam
    om te hernoemen, kies de bewerking in de keuzelijst, en zet een laag met het oogje
    uit of met het slotje op slot. Nieuwe paden komen op de actieve laag; die kies je
    door op een rij te klikken.
    """

    COLS = 6
    WIDTHS = (26, 26, 30, 90, 116, 36)

    def __init__(self, win):
        super().__init__()
        self.win = win
        self.names: list[str] = []
        self._busy = False
        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        lay.setSpacing(4)
        lay.addWidget(QLabel("Lagen van het 2D-werkblad (klik een rij voor de actieve laag):"))
        self.table = QTableWidget(0, self.COLS)
        self.table.setHorizontalHeaderLabels(["", "", "", "Laag", "Bewerking", ""])
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QTableWidget.DoubleClicked | QTableWidget.EditKeyPressed)
        self.table.setIconSize(QSize(16, 16))
        for col, width in enumerate(self.WIDTHS):
            self.table.setColumnWidth(col, width)
        self.table.horizontalHeader().setSectionResizeMode(3, QHeaderView.Stretch)
        self.table.itemSelectionChanged.connect(self.on_row)
        self.table.itemChanged.connect(self.on_rename)
        lay.addWidget(self.table, 1)
        row = QHBoxLayout()
        self.b_add = QPushButton("Toevoegen")
        self.b_add.clicked.connect(self.add_layer)
        self.b_remove = QPushButton("Verwijderen")
        self.b_remove.clicked.connect(self.remove_layer)
        row.addWidget(self.b_add)
        row.addWidget(self.b_remove)
        lay.addLayout(row)
        row = QHBoxLayout()
        self.b_active = QPushButton("Actieve laag")
        self.b_active.clicked.connect(self.pick_active)
        self.b_params = QPushButton("Instellingen…")
        self.b_params.clicked.connect(self.edit_params)
        row.addWidget(self.b_active)
        row.addWidget(self.b_params)
        lay.addLayout(row)
        self.info = QLabel("")
        self.info.setWordWrap(True)
        lay.addWidget(self.info)
        self.refresh()

    # ------------------------------------------------------------ opbouw

    @contextmanager
    def _quiet(self):
        """Geen reactie op de eigen wijzigingen in de tabel (ze komen van het model)."""
        saved, self._busy = self._busy, True
        try:
            yield
        finally:
            self._busy = saved

    def refresh(self, checked=False):
        sheet = self.win.doc.sheet
        active = self.win.sheet_view.current_layer
        with self._quiet():
            self.names = list(sheet.layers)
            self.table.clearContents()
            self.table.setRowCount(len(self.names))
            for row, name in enumerate(self.names):
                lay = sheet.layers[name]
                it = QTableWidgetItem(name)
                if name == active:
                    font = QFont()
                    font.setBold(True)
                    it.setFont(font)
                it.setToolTip("Dubbelklik om te hernoemen")
                self.table.setItem(row, 3, it)
                self.table.setCellWidget(row, 0, self._button(
                    row, "eye" if lay.visible else "eye_off",
                    "Laag verbergen (wordt niet geëxporteerd)" if lay.visible
                    else "Laag tonen", self.toggle_visible))
                self.table.setCellWidget(row, 1, self._button(
                    row, "lock" if lay.locked else "unlock",
                    "Laag vergrendelen (kun je niet meer selecteren)" if not lay.locked
                    else "Laag ontgrendelen", self.toggle_locked))
                self.table.setCellWidget(row, 2, self._button(
                    row, "paint", f"Kleur van de laag ({lay.color})", self.pick_color))
                self.table.setCellWidget(row, 4, self._op_box(row, lay))
                self.table.setCellWidget(row, 5, self._button(
                    row, "check", "Instellingen van deze laag", self.edit_params))
            self.table.selectRow(max(0, self.names.index(active)) if active in self.names else 0)
        self.info.setText(self._summary(sheet, active))

    def _summary(self, sheet, active) -> str:
        lay = sheet.layers.get(active)
        if lay is None:
            return ""
        n = sum(1 for p in sheet.paths if p.layer == active)
        if lay.op in MILL_OPS:
            details = (f"frees Ø{layer_param(lay, 'tool_d'):g} mm, "
                       f"diepte {layer_param(lay, 'depth'):g} mm")
        elif lay.op in LASER_OPS:
            details = (f"{layer_param(lay, 'power'):g} %, {layer_param(lay, 'speed'):g} mm/s, "
                       f"{layer_param(lay, 'passes')}×, kerf {layer_param(lay, 'kerf'):g} mm")
        else:
            details = "wordt niet geëxporteerd"
        return (f"Actieve laag: {active} — {OP_LABELS.get(lay.op, lay.op)}; {n} contouren; "
                f"{details}.")

    def _button(self, row, name, tip, slot):
        b = QToolButton()
        b.setIcon(icon(name, 16))
        b.setIconSize(QSize(16, 16))
        b.setAutoRaise(True)
        b.setToolTip(tip)
        b.clicked.connect(lambda checked=False, r=row: slot(r))
        return b

    def _op_box(self, row, lay):
        box = QComboBox()
        for op in OPS:
            box.addItem(OP_LABELS[op], op)
        box.setCurrentIndex(max(0, OPS.index(lay.op) if lay.op in OPS else 0))
        box.setToolTip("Wat de machine met deze laag doet")
        box.currentIndexChanged.connect(lambda _=0, r=row, b=box: self.set_op(r, b))
        return box

    def _layer(self, row):
        try:
            return self.win.doc.sheet.layers[self.names[row]]
        except (IndexError, KeyError):
            return None

    def _edit(self, change, message):
        """Eén undo-stap op het werkblad, daarna alles bijwerken."""
        self.win.doc.checkpoint()
        change()
        self.win.doc.notify()
        self.refresh()
        self.win.hint.setText(message)

    # ------------------------------------------------------------ knoppen

    def on_row(self):
        if self._busy:
            return
        row = self.table.currentRow()
        if 0 <= row < len(self.names):
            self.win.sheet_view.set_current_layer(self.names[row])
            self.refresh()

    def pick_active(self):
        row = self.table.currentRow()
        lay = self._layer(row)
        if lay is None:
            return
        if lay.locked:
            self.win.hint.setText(f"Laag {lay.name} staat op slot; dat kan nu niet als actieve laag.")
            return
        self.win.sheet_view.set_current_layer(lay.name)
        self.refresh()
        self.win.hint.setText(f"Nieuwe contouren gaan naar laag {lay.name}.")

    def on_rename(self, item):
        if self._busy or item is None or item.column() != 3:
            return
        sheet = self.win.doc.sheet
        old = self.names[item.row()] if item.row() < len(self.names) else None
        new = item.text().strip()
        if old is None or not new or new == old:
            self.refresh()
            return
        if new in sheet.layers:
            self.win.hint.setText(f"Er is al een laag met de naam {new}.")
            self.refresh()
            return
        with self._quiet():
            self.win.doc.checkpoint()
            sheet.rename_layer(old, new)
            if self.win.sheet_view.current_layer == old:
                self.win.sheet_view.current_layer = new
            self.win.doc.notify()
        self.refresh()
        self.win.hint.setText(f"Laag {old} heet nu {new}.")

    def add_layer(self):
        sheet = self.win.doc.sheet
        name = sheet.free_layer_name()
        color = ["#00a0a0", "#a000a0", "#0000a0", "#a0a000", "#00a000"][len(sheet.layers) % 5]

        def change():
            sheet.ensure_layer(name, color, "cut")
        self._edit(change, f"Laag {name} toegevoegd; teken of verplaats er contouren naartoe.")
        if name in self.names:
            self.table.selectRow(self.names.index(name))
            self.win.sheet_view.set_current_layer(name)
            self.refresh()

    def remove_layer(self):
        row = self.table.currentRow()
        lay = self._layer(row)
        if lay is None:
            return
        if lay.name == "0":
            self.win.hint.setText("Laag 0 is de basislaag en kan er niet uit.")
            return
        sheet = self.win.doc.sheet
        n = sum(1 for p in sheet.paths if p.layer == lay.name)
        if n:
            box = QMessageBox(self)
            box.setIcon(QMessageBox.Question)
            box.setWindowTitle(f"Laag {lay.name} verwijderen")
            box.setText(f"Er staan {n} contouren op laag {lay.name}.")
            b_move = box.addButton("Paden naar laag 0", QMessageBox.AcceptRole)
            b_del = box.addButton("Paden ook verwijderen", QMessageBox.DestructiveRole)
            box.addButton(QMessageBox.Cancel)
            box.exec()
            clicked = box.clickedButton()
            if clicked not in (b_move, b_del):
                return
            move = clicked is b_move
        else:
            move = True
        active = self.win.sheet_view.current_layer

        def change():
            sheet.remove_layer(lay.name, move_paths=move)
            if active == lay.name:
                self.win.sheet_view.current_layer = "0"
        self._edit(change, f"Laag {lay.name} verwijderd"
                   + (f" ({n} contouren naar laag 0)" if move and n else
                      f" met {n} contouren" if n else "") + ".")

    def toggle_visible(self, row):
        lay = self._layer(row)
        if lay is None:
            return
        new = not lay.visible

        def change():
            lay.visible = new
        self._edit(change, f"Laag {lay.name} "
                   + ("weer zichtbaar." if new
                      else "verborgen; die gaat niet mee naar de machine."))

    def toggle_locked(self, row):
        lay = self._layer(row)
        if lay is None:
            return
        new = not lay.locked

        def change():
            lay.locked = new
        self._edit(change, f"Laag {lay.name} "
                   + ("op slot: je kunt de contouren nu niet meer aanklikken." if new
                      else "ontgrendeld."))

    def pick_color(self, row):
        lay = self._layer(row)
        if lay is None:
            return
        color = QColorDialog.getColor(QColor(lay.color or "#000000"), self,
                                      f"Kleur van laag {lay.name}")
        if not color.isValid():
            return

        new = color.name().lower()

        def change():
            lay.color = new
        self._edit(change, f"Laag {lay.name} is nu {new}.")

    def set_op(self, row, box):
        lay = self._layer(row)
        if lay is None or self._busy:
            return
        op = box.currentData()
        if op == lay.op:
            return

        def change():
            lay.op = op
        self._edit(change, f"Laag {lay.name}: {OP_LABELS.get(op, op)}.")

    def edit_params(self, row=None):
        if row is None:
            row = self.table.currentRow()
        lay = self._layer(row)
        if lay is None:
            return
        dlg = LayerParamsDialog(lay, self)
        if dlg.exec() != QDialog.Accepted:
            self.refresh()
            return
        values = dlg.values()

        def change():
            lay.params = {**dict(lay.params or {}), **values}
        self._edit(change, f"Instellingen van laag {lay.name} aangepast.")


# ---------------------------------------------------------------- dialogs

SHORTCUTS_KEY = "shortcuts/"
SHOW_WELCOME_KEY = "show_welcome"
OTHER_COMMANDS = "Overig"     # the group of the commands that are only in the right-click menu


def qsettings():
    """The application settings, the same store as Settings.q."""
    return QSettings("Volura", "Volura")


def clean_text(text):
    """Menu text without the & accelerator and without a trailing ellipsis."""
    return text.replace("&", "").replace("…", "").replace("...", "").strip()


def welcome_enabled():
    """Help > Welkom bij Volura at startup, unless the user unchecked the box."""
    return qsettings().value(SHOW_WELCOME_KEY, True) in (True, "true", "1", 1)


UPDATE_DELAY_MS = 4000             # niet meteen bij het opstarten: eerst het venster
UPDATE_INTERVAL = 24 * 3600        # automatisch hooguit één keer per dag
UPDATE_LAST_KEY = "update/last_check"
UPDATE_SKIP_KEY = "update/skip"


class _Bridge(QObject):
    """Brengt de uitkomst van een achtergronddraad terug naar de hoofddraad."""

    done = Signal(object, object)          # resultaat, fout
    progress = Signal(int, int)


def run_in_background(work, on_done, on_progress=None):
    """`work(progress)` in een draad; `on_done(resultaat, fout)` in de hoofddraad."""
    bridge = _Bridge()
    bridge.done.connect(on_done)
    if on_progress is not None:
        bridge.progress.connect(on_progress)

    def run():
        try:
            result = work(bridge.progress.emit)
        except Exception as e:               # netwerkfouten horen in de melding, niet in een crash
            bridge.done.emit(None, e)
        else:
            bridge.done.emit(result, None)
    threading.Thread(target=run, daemon=True).start()
    return bridge


def is_offscreen():
    """True when Qt runs without a display, which is how the tests run."""
    if os.environ.get("QT_QPA_PLATFORM", "").lower() == "offscreen":
        return True
    app = QApplication.instance()
    return app is not None and app.platformName() == "offscreen"


AUTOSAVE_NAME = "autosave.volura"
AUTOSAVE_INTERVAL_MS = 5 * 60 * 1000      # automatisch opslaan, zoals SketchUp


def autosave_path():
    """<AppDataLocation>/autosave.volura: het model van de laatste sessie."""
    return os.path.join(errors.log_dir(), AUTOSAVE_NAME)


def remove_autosave():
    try:
        os.remove(autosave_path())
    except OSError:
        pass


def note(text):
    """An explanatory line under the fields of a dialog page."""
    label = QLabel(text)
    label.setWordWrap(True)
    return label


class VectorPreview(QWidget):
    """A small preview of a 2D drawing (millimetres, Y up) for the export dialog."""

    MARGIN = 8

    def __init__(self, parent=None):
        super().__init__(parent)
        self.drawing = None
        self.setMinimumSize(220, 170)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def set_drawing(self, drawing):
        self.drawing = drawing
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.fillRect(self.rect(), QColor("#ffffff"))
        box = self.rect().adjusted(self.MARGIN, self.MARGIN, -self.MARGIN, -self.MARGIN)
        p.setPen(QPen(QColor("#c8c8c8")))
        p.drawRect(box)
        bounds = self.drawing.bounds() if self.drawing is not None else None
        if bounds is None or box.width() <= 0 or box.height() <= 0:
            p.setPen(QColor("#808080"))
            p.drawText(box, Qt.AlignCenter, "geen contouren")
            return
        w = max(bounds[2] - bounds[0], 1e-6)
        h = max(bounds[3] - bounds[1], 1e-6)
        scale = min(box.width() / w, box.height() / h)
        ox = box.x() + 0.5 * (box.width() - w * scale)
        oy = box.y() + 0.5 * (box.height() - h * scale)

        def point(xy):
            return QPointF(ox + (float(xy[0]) - bounds[0]) * scale,
                           box.bottom() - (float(xy[1]) - bounds[1]) * scale)

        for path in self.drawing.paths:
            if len(path.pts) < 2:
                continue
            layer = self.drawing.layers.get(path.layer)
            pen = QPen(QColor((layer.color if layer else "") or "#000000"))
            pen.setWidthF(1.2)
            p.setPen(pen)
            if len(path.pts) == 2 and not path.closed:
                p.drawLine(point(path.pts[0]), point(path.pts[1]))
                continue
            line = QPainterPath(point(path.pts[0]))
            for xy in path.pts[1:]:
                line.lineTo(point(xy))
            if path.closed:
                line.closeSubpath()
            p.drawPath(line)


class VectorExportDialog(QDialog):
    """Dialoog '2D-vector exporteren': wat er weg moet en een klein voorbeeld.

    Te kiezen: de geselecteerde vlakken (alleen als er vlakken zijn), het
    bovenaanzicht of een doorsnede op hoogte Z -- van de selectie of van alles.
    De hoogte is een getal met eenheid zoals in het meetvak (12 of 0,5 in) en staat
    halverwege het model. Het voorbeeld tekent mee wat er weggeschreven wordt.
    """

    def __init__(self, win):
        super().__init__(win)
        self.win = win
        self.setWindowTitle("2D-vector exporteren")
        form = QFormLayout(self)
        faces = [i for i in win.doc.selection if i[0] == "f"]

        self.b_faces = QRadioButton(f"Geselecteerde vlakken ({len(faces)})")
        self.b_faces.setEnabled(bool(faces))
        self.b_faces.setToolTip("Elk vlak op ware grootte, in zijn eigen vlak")
        self.b_top = QRadioButton("Bovenaanzicht (de omtrek van bovenaf)")
        self.b_section = QRadioButton("Doorsnede op hoogte Z")
        (self.b_faces if faces else self.b_top).setChecked(True)
        box = QGroupBox("Wat te exporteren")
        col = QVBoxLayout(box)
        for b in (self.b_faces, self.b_top, self.b_section):
            col.addWidget(b)

        self.z_edit = QLineEdit(format_length(vector_export.section_z_default(win.doc)))
        self.z_edit.setToolTip("Hoogte van de doorsnede, bijvoorbeeld 12 of 0,5 in")
        self.z_edit.setFixedWidth(90)

        self.only = QCheckBox("Alleen de selectie")
        self.only.setToolTip("Alleen de geselecteerde groepen; anders het hele model")

        self.preview = VectorPreview()

        form.addRow(box)
        form.addRow("Hoogte Z", self.z_edit)
        form.addRow("", self.only)
        form.addRow("Voorbeeld", self.preview)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        form.addRow(bb)

        for b in (self.b_faces, self.b_top, self.b_section):
            b.toggled.connect(self.refresh)
        self.only.toggled.connect(self.refresh)
        self.z_edit.textEdited.connect(self.refresh)
        self.refresh()

    def options(self):
        """De keuzes als dict: bron, alleen de selectie en de hoogte van de doorsnede."""
        source = ("faces" if self.b_faces.isChecked()
                  else "section" if self.b_section.isChecked() else "top")
        try:
            z = parse_length(self.z_edit.text())
        except ValueError:
            z = vector_export.section_z_default(self.win.doc)
        return {"source": source, "only_selected": self.only.isChecked(), "z": z}

    def refresh(self, *_):
        faces = self.b_faces.isChecked()
        self.z_edit.setEnabled(self.b_section.isChecked())
        self.only.setEnabled(not faces and bool(self.win.doc.selected_groups()))
        try:
            parse_length(self.z_edit.text())
            self.z_edit.setStyleSheet("")
        except ValueError:
            self.z_edit.setStyleSheet("border: 1px solid #c0392b;")
        self.preview.set_drawing(self.win.vector_export_drawing(self.options()))


class SheetExportDialog(QDialog):
    """Dialoog "Exporteren voor laser/frees": welke lagen er weg mogen en welke correctie.

    Standaard gaan de zichtbare lagen mee die iets voor de machine betekenen (behalve
    "Geen", die is alleen ter informatie). De kerf-/freescorrectie staat aan zodra er
    een freeslaag meegaat — een frees heeft een eigen diameter — en uit bij laser, want
    lasersoftware doet die correctie vaak zelf. Met "per laag een apart bestand" krijgt
    elke laag zijn eigen bestand, zoals sommige snijsoftware graag ziet. Het voorbeeld
    tekent mee wat er weggeschreven wordt.
    """

    def __init__(self, win, kind="svg"):
        super().__init__(win)
        self.win = win
        self.kind = "dxf" if str(kind).lower() == "dxf" else "svg"
        self._busy = False
        self.setWindowTitle("Exporteren voor laser/frees")
        self.resize(430, 520)
        lay = QVBoxLayout(self)
        lay.addWidget(QLabel("Lagen die naar de machine gaan:"))
        self.list = QListWidget()
        self.list.setIconSize(QSize(16, 16))
        lay.addWidget(self.list, 1)
        self.compensate = QCheckBox("Kerf-/freescorrectie toepassen")
        self.compensate.setToolTip("Contouren opzij zetten met de halve kerf of de freesradius, "
                                   "zodat het werkstuk de getekende maat wordt")
        self.per_layer = QCheckBox("Per laag een apart bestand")
        self.per_layer.setToolTip(f"Elke laag krijgt zijn eigen bestand: naam_laagnaam.{self.kind}")
        self.per_sheet = QCheckBox("Per plaat een apart bestand")
        self.per_sheet.setToolTip("Elke plaat wordt een eigen bestand met die plaat als pagina "
                                  "en zijn oorsprong als nulpunt: naam_plaat1.svg ...")
        lay.addWidget(self.compensate)
        lay.addWidget(self.per_layer)
        lay.addWidget(self.per_sheet)
        self.warning = QLabel("")
        self.warning.setWordWrap(True)
        self.warning.setStyleSheet("color: #c0392b;")
        lay.addWidget(self.warning)
        self.preview = VectorPreview()
        lay.addWidget(QLabel("Voorbeeld van wat er weggaat:"))
        lay.addWidget(self.preview, 1)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)

        sheet = win.doc.sheet
        default = win.default_sheet_export_options()
        self._busy = True
        for name, layer in sheet.layers.items():
            it = QListWidgetItem(swatch(layer.color or "#000000", 16),
                                 f"{name}   —   {OP_LABELS.get(layer.op, layer.op)}")
            it.setData(Qt.UserRole, name)
            it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
            it.setCheckState(Qt.Checked if name in default["layers"] else Qt.Unchecked)
            if layer.op == "none":
                it.setToolTip("Deze laag is alleen ter informatie en gaat normaal niet mee")
            self.list.addItem(it)
        self._busy = False
        self.compensate.setChecked(bool(default["compensate"]))
        self.per_layer.setChecked(bool(default["per_layer"]))
        self.per_sheet.setChecked(bool(default.get("per_sheet", False)))
        self.list.itemChanged.connect(self.refresh)
        self.compensate.toggled.connect(self.refresh)
        self.refresh()

    def options(self) -> dict:
        names = []
        for i in range(self.list.count()):
            it = self.list.item(i)
            if it.checkState() == Qt.Checked:
                names.append(str(it.data(Qt.UserRole)))
        return {"layers": names, "compensate": self.compensate.isChecked(),
                "per_layer": self.per_layer.isChecked(), "per_sheet": self.per_sheet.isChecked()}

    def refresh(self, *_):
        if self._busy:
            return
        try:
            self.preview.set_drawing(self.win.sheet_export_drawing(self.options()))
        except Exception:                       # het voorbeeld mag nooit een dialoog breken
            self.preview.set_drawing(None)
        try:
            n = len(nesting.outside_paths(self.win.doc.sheet))
        except Exception:
            n = 0
        self.warning.setText("" if not n else
                             f"Let op: {n} paden liggen buiten de plaat of in de marge en gaan "
                             "zo mee naar de machine.")


class SheetSettingsDialog(QDialog):
    """Dialoog "Plaat…": voorgegeven maten of vrij, plus dikte, marge, ruimte en aantal."""

    PRESETS = [("A4 — 297 × 210", 297.0, 210.0),
               ("A3 — 420 × 297", 420.0, 297.0),
               ("300 × 200", 300.0, 200.0),
               ("400 × 300", 400.0, 300.0),
               ("600 × 400", 600.0, 400.0),
               ("1220 × 610 (halve plaat)", 1220.0, 610.0),
               ("2440 × 1220 (plaat)", 2440.0, 1220.0),
               ("Vrij formaat", None, None)]

    def __init__(self, win):
        super().__init__(win)
        self.win = win
        self.setWindowTitle("Plaat")
        sheet = win.doc.sheet
        lay = QFormLayout(self)
        self.preset = QComboBox()
        for label, *_ in self.PRESETS:
            self.preset.addItem(label)
        self.w = QDoubleSpinBox()
        self.h = QDoubleSpinBox()
        for spin in (self.w, self.h):
            spin.setRange(1.0, 100000.0)
            spin.setDecimals(2)
            spin.setSuffix(" mm")
        self.thickness = QDoubleSpinBox()
        self.thickness.setRange(0.0, 500.0)
        self.thickness.setDecimals(2)
        self.thickness.setSuffix(" mm")
        self.thickness.setToolTip("Dikte van het materiaal (gebruikt door de freespaden)")
        self.margin = QDoubleSpinBox()
        self.margin.setRange(0.0, 1000.0)
        self.margin.setDecimals(2)
        self.margin.setSuffix(" mm")
        self.margin.setToolTip("Afstand die langs de rand van de plaat vrij blijft")
        self.gap = QDoubleSpinBox()
        self.gap.setRange(0.0, 1000.0)
        self.gap.setDecimals(2)
        self.gap.setSuffix(" mm")
        self.gap.setToolTip("Minimale afstand tussen twee onderdelen")
        self.count = QSpinBox()
        self.count.setRange(1, 99)
        self.count.setToolTip("Aantal platen naast elkaar op het werkblad")
        thick_row = QHBoxLayout()
        self.thick_off = QCheckBox("geen dikte")
        self.thick_off.toggled.connect(self.thickness.setDisabled)
        thick_row.addWidget(self.thickness)
        thick_row.addWidget(self.thick_off)
        lay.addRow("Voorinstelling", self.preset)
        lay.addRow("Breedte", self.w)
        lay.addRow("Hoogte", self.h)
        lay.addRow("Materiaaldikte", thick_row)
        lay.addRow("Marge langs de rand", self.margin)
        lay.addRow("Tussenruimte", self.gap)
        lay.addRow("Aantal platen", self.count)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addRow(bb)
        self._busy = True
        self.w.setValue(float(sheet.sheet_w or 300.0))
        self.h.setValue(float(sheet.sheet_h or 200.0))
        self.thickness.setValue(float(sheet.sheet_thickness or 0.0))
        self.thick_off.setChecked(sheet.sheet_thickness is None)
        self.thickness.setDisabled(self.thick_off.isChecked())
        self.margin.setValue(float(sheet.sheet_margin))
        self.gap.setValue(float(sheet.sheet_gap))
        self.count.setValue(int(sheet.sheet_count or 1))
        idx = next((i for i, (_l, pw, ph) in enumerate(self.PRESETS)
                    if pw and abs(pw - self.w.value()) < 1e-6 and abs(ph - self.h.value()) < 1e-6),
                   len(self.PRESETS) - 1)
        self.preset.setCurrentIndex(idx)
        self._busy = False
        self.preset.currentIndexChanged.connect(self._pick_preset)

    def _pick_preset(self, index):
        if self._busy:
            return
        _label, pw, ph = self.PRESETS[index]
        if pw is None:
            return
        self._busy = True
        self.w.setValue(pw)
        self.h.setValue(ph)
        self._busy = False

    def values(self) -> dict:
        return {"w": round(self.w.value(), 6), "h": round(self.h.value(), 6),
                "thickness": None if self.thick_off.isChecked() else round(self.thickness.value(), 6),
                "margin": round(self.margin.value(), 6), "gap": round(self.gap.value(), 6),
                "count": int(self.count.value())}


class AutoArrangeDialog(QDialog):
    """Dialoog "Automatisch schikken": draaien, tussenruimte, marge en nieuwe platen."""

    def __init__(self, win):
        super().__init__(win)
        self.win = win
        sheet = win.doc.sheet
        self.setWindowTitle("Automatisch schikken")
        lay = QFormLayout(self)
        self.rot_all = QRadioButton("0°, 90°, 180° en 270°")
        self.rot_none = QRadioButton("alleen 0° (niet draaien)")
        self.rot_all.setChecked(True)
        rot_row = QHBoxLayout()
        rot_row.addWidget(self.rot_all)
        rot_row.addWidget(self.rot_none)
        lay.addRow("Draaien toestaan", rot_row)
        self.gap = QDoubleSpinBox()
        self.gap.setRange(0.0, 1000.0)
        self.gap.setDecimals(2)
        self.gap.setSuffix(" mm")
        self.gap.setValue(float(sheet.sheet_gap))
        self.margin = QDoubleSpinBox()
        self.margin.setRange(0.0, 1000.0)
        self.margin.setDecimals(2)
        self.margin.setSuffix(" mm")
        self.margin.setValue(float(sheet.sheet_margin))
        self.add_sheets = QCheckBox("nieuwe platen toevoegen als het niet past")
        self.add_sheets.setChecked(True)
        lay.addRow("Tussenruimte", self.gap)
        lay.addRow("Marge", self.margin)
        lay.addRow("Platen", self.add_sheets)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addRow(bb)

    def options(self) -> dict:
        return {"rotations": (0, 90, 180, 270) if self.rot_all.isChecked() else (0,),
                "gap": round(self.gap.value(), 6), "margin": round(self.margin.value(), 6),
                "add_sheets": self.add_sheets.isChecked()}


class BoxJointDialog(QDialog):
    """Dialoog "Doos met vingerlassen…": maten, materiaal, vingers, kerf en deksel.

    De twee knoppen onderaan doen allebei iets met de maten: "Op werkblad" zet de
    snijcontouren op het 2D-werkblad en schikt ze op de plaat, "Voorbeeld in 3D" zet de
    panelen geëxtrudeerd in het model. De dikte komt uit de plaatinstellingen en de kerf
    uit de laag "Snijden"; de vingerbreedte volgt de dikte (2,5 ×) zolang die niet zelf
    gekozen is.
    """

    SHEET = "sheet"
    MODEL = "model"

    def __init__(self, win):
        super().__init__(win)
        self.win = win
        self.mode = self.SHEET
        self._sync = False
        self._finger_fixed = False
        self.setWindowTitle("Doos met vingerlassen")
        sheet = win.doc.sheet
        t = float(sheet.sheet_thickness or 0.0) or boxgen.DEFAULT_THICKNESS
        kerf = float(layer_param(sheet.layers.get("Snijden"), "kerf") or 0.0)
        lay = QFormLayout(self)

        def spin(lo, hi, value, step=0.5):
            sp = QDoubleSpinBox()
            sp.setRange(lo, hi)
            sp.setDecimals(2)
            sp.setSingleStep(step)
            sp.setSuffix(" mm")
            sp.setValue(float(value))
            return sp

        self.w = spin(1.0, 5000.0, 100.0)
        self.d = spin(1.0, 5000.0, 60.0)
        self.h = spin(1.0, 5000.0, 40.0)
        self.kind = QComboBox()
        self.kind.addItem("binnenmaten (de ruimte binnen de doos)", True)
        self.kind.addItem("buitenmaten (de afmetingen van de doos)", False)
        self.t = spin(0.1, 100.0, t, 0.1)
        self.t.setToolTip("Dikte van het paneel; de vingers zijn even diep als de dikte")
        self.finger = spin(0.5, 200.0, boxgen.FINGER_FACTOR * t, 0.1)
        self.finger.setToolTip("Gewenste vingerbreedte; het aantal vingers wordt er het "
                               "grootste oneven getal van dat past")
        self.kerf = spin(0.0, 5.0, kerf, 0.05)
        self.kerf.setToolTip("Breedte van de snedespleet: de vingers worden er breder en "
                             "de uitsparingen smaller door, zodat de las strak past")
        self.lid = QComboBox()
        for label, key in (("geen deksel (open doos)", "none"), ("vast deksel", "closed")):
            self.lid.addItem(label, key)
        self.labels = QCheckBox("paneelnaam klein graveren in het midden")
        self.labels.setChecked(True)
        self.info = QLabel()
        self.info.setWordWrap(True)
        lay.addRow("Breedte", self.w)
        lay.addRow("Diepte", self.d)
        lay.addRow("Hoogte", self.h)
        lay.addRow("Maten", self.kind)
        lay.addRow("Materiaaldikte", self.t)
        lay.addRow("Vingerbreedte", self.finger)
        lay.addRow("Kerf", self.kerf)
        lay.addRow("Deksel", self.lid)
        lay.addRow("", self.labels)
        lay.addRow(self.info)
        bb = QDialogButtonBox(QDialogButtonBox.Cancel)
        to_sheet = QPushButton("Op werkblad")
        to_sheet.setToolTip("De panelen snijden op het werkblad en automatisch schikken op de plaat")
        to_model = QPushButton("Voorbeeld in 3D")
        to_model.setToolTip("De panelen geëxtrudeerd in het model zetten, zoals de gelijmde doos eruitziet")
        bb.addButton(to_sheet, QDialogButtonBox.AcceptRole)
        bb.addButton(to_model, QDialogButtonBox.AcceptRole)
        bb.rejected.connect(self.reject)
        to_sheet.clicked.connect(lambda checked=False: self._choose(self.SHEET))
        to_model.clicked.connect(lambda checked=False: self._choose(self.MODEL))
        lay.addRow(bb)
        self.t.valueChanged.connect(self._follow_thickness)
        self.finger.valueChanged.connect(self._finger_chosen)
        for sp in (self.w, self.d, self.h, self.t, self.finger, self.kerf):
            sp.valueChanged.connect(self._refresh)
        self.kind.currentIndexChanged.connect(self._refresh)
        self.lid.currentIndexChanged.connect(self._refresh)
        self._refresh()

    def _choose(self, mode):
        self.mode = mode
        self.accept()

    def _finger_chosen(self, _value):
        if not self._sync:
            self._finger_fixed = True

    def _follow_thickness(self, value):
        """De vingerbreedte volgt 2,5 × de dikte, tenzij de gebruiker die zelf koos."""
        if self._sync or self._finger_fixed:
            return
        self._sync = True
        self.finger.setValue(boxgen.FINGER_FACTOR * float(value))
        self._sync = False

    def _refresh(self, *_args):
        o = self.options()
        try:
            W, D, H = boxgen.box_size(o["w"], o["d"], o["h"], o["t"], o["inner"], o["lid"])
            n = boxgen.finger_count(max(o["w"], o["d"], o["h"]), o["finger_w"])
            step = max(o["w"], o["d"], o["h"]) / n
            self.info.setText(f"Doos van {W:g} × {D:g} × {H:g} mm, vingers van "
                              f"{step:.2f} mm ({n} over de langste rand).")
        except (ValueError, ZeroDivisionError) as e:
            self.info.setText(str(e))

    def options(self) -> dict:
        return {"w": round(self.w.value(), 6), "d": round(self.d.value(), 6),
                "h": round(self.h.value(), 6), "inner": bool(self.kind.currentData()),
                "t": round(self.t.value(), 6), "finger_w": round(self.finger.value(), 6),
                "kerf": round(self.kerf.value(), 6), "lid": str(self.lid.currentData()),
                "labels": self.labels.isChecked(), "mode": self.mode}


class PageDialog(QDialog):
    """A dialog with the pages on the left and their content on the right, like SketchUp."""
    SIZE = (640, 440)

    def __init__(self, win, title, pages, page=0):
        super().__init__(win)
        self.win = win
        self.page_titles = list(pages)
        self.setWindowTitle(title)
        self.resize(*self.SIZE)
        lay = QVBoxLayout(self)
        body = QHBoxLayout()
        self.pages_list = QListWidget()
        self.pages_list.setFixedWidth(140)
        for name in pages:
            QListWidgetItem(name, self.pages_list)
        self.stack = QStackedWidget()
        body.addWidget(self.pages_list)
        body.addWidget(self.stack, 1)
        lay.addLayout(body)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)
        self.pages_list.currentRowChanged.connect(self.stack.setCurrentIndex)
        self.select_page(page)

    def add_page(self, widget):
        self.stack.addWidget(widget)
        return widget

    def select_page(self, page):
        if isinstance(page, str):
            page = self.page_titles.index(page)
        page = max(0, min(len(self.page_titles) - 1, int(page)))
        self.pages_list.setCurrentRow(page)
        self.stack.setCurrentIndex(page)

    def accept(self):
        self.apply()
        super().accept()

    def apply(self):
        """Subclasses store what is on their pages here."""


class ModelInfoDialog(PageDialog):
    """Venster > Modelinfo: Eenheden, Statistieken en Printbed (SketchUp 2017)."""

    PAGES = ("Eenheden", "Statistieken", "Printbed")
    SIZE = (660, 430)

    def __init__(self, win, page=0):
        super().__init__(win, "Modelinfo", self.PAGES, page)
        self._build_units()
        self._build_statistics()
        self._build_bed()
        self.refresh_statistics()
        self.pages_list.currentRowChanged.connect(self._page_shown)
        self._page_shown(self.pages_list.currentRow())

    def _page_shown(self, row):
        if row == 1:
            self.refresh_statistics()

    def _build_units(self):
        page = QWidget()
        form = QFormLayout(page)
        snap = self.win.settings.length_snap
        box = QGroupBox("Lengte")
        inner = QFormLayout(box)
        self.snap_on = QCheckBox("Lengte-afronding inschakelen")
        self.snap_on.setChecked(snap > 0)
        self.snap = QDoubleSpinBox()
        self.snap.setRange(0.01, 100)
        self.snap.setDecimals(2)
        self.snap.setSingleStep(0.5)
        self.snap.setSuffix(" mm")
        self.snap.setValue(snap or 1.0)
        self.snap.setEnabled(snap > 0)
        self.snap_on.toggled.connect(self.snap.setEnabled)
        self.snap.setToolTip("Lengtes die niet op een punt of rand snappen worden hierop afgerond, "
                             "zoals 'Lengte-uitlijning' in SketchUp.")
        inner.addRow(self.snap_on)
        inner.addRow("Afronden op", self.snap)
        self.precision = QComboBox()
        self.precision.addItems(["0", "0.0", "0.00"])
        self.precision.setCurrentIndex(min(self.win.settings.length_precision, 2))
        self.sample = QLabel()
        self.precision.currentIndexChanged.connect(self._show_sample)
        self._show_sample()
        inner.addRow("Weergaveprecisie", self.precision)
        inner.addRow("Voorbeeld", self.sample)
        form.addRow(box)
        form.addRow(note("De weergaveprecisie is het aantal decimalen van de maatinvoer en van de "
                         "afmetingen in het tekenvenster, zoals 'Lengteprecisie' in SketchUp."))
        form.addRow(note("Het printbed en de printer staan op de pagina Printbed."))
        self.add_page(page)

    def _show_sample(self):
        self.sample.setText(format_length(123.456, self.precision.currentIndex()))

    def _build_statistics(self):
        page = QWidget()
        form = QFormLayout(page)
        self.stats = {}
        for label in ("Groepen", "Losse vlakken", "Losse randen", "Volume in totaal"):
            value = QLabel()
            form.addRow(label, value)
            self.stats[label] = value
        form.addRow(note("Het volume is de som van alle gesloten onderdelen: elke groep en elke "
                         "losse vorm die een gesloten volume vormt."))
        self.add_page(page)

    def refresh_statistics(self):
        doc = self.win.doc
        self.stats["Groepen"].setText(str(len(doc.groups)))
        self.stats["Losse vlakken"].setText(str(len(doc.root.F)))
        self.stats["Losse randen"].setText(str(len(doc.root.E)))
        volume = sum(p.manifold.volume() for p in ops.parts(doc) if p.manifold is not None)
        self.stats["Volume in totaal"].setText(f"{volume / 1000:.3f} cm³")

    def _build_bed(self):
        page = QWidget()
        form = QFormLayout(page)
        self.printer = QComboBox()
        self.printer.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.printer.addItem("Aangepast", None)
        for name, size in PRINTERS:
            self.printer.addItem(f"{name}  ({size[0]}×{size[1]}×{size[2]})", size)
        self.bed_spins = []
        for label, v in zip(("Bed X", "Bed Y", "Hoogte Z"), self.win.settings.bed_size):
            sp = QDoubleSpinBox()
            sp.setRange(10, 2000)
            sp.setSuffix(" mm")
            sp.setValue(v)
            self.bed_spins.append(sp)
            form.addRow(label, sp)

        def preset(i):
            size = self.printer.itemData(i)
            if size:
                for sp, v in zip(self.bed_spins, size):
                    sp.setValue(v)
        self.printer.currentIndexChanged.connect(preset)
        form.insertRow(0, "Printer", self.printer)
        form.addRow(note("Het bed bepaalt het raster van het tekenvlak en de grenzen van de "
                         "printcontrole."))
        self.add_page(page)

    def apply(self):
        s = self.win.settings
        s.length_snap = self.snap.value() if self.snap_on.isChecked() else 0.0
        s.length_precision = self.precision.currentIndex()
        s.bed_size = tuple(sp.value() for sp in self.bed_spins)
        s.save()
        set_length_precision(s.length_precision)
        self.win.view.update()


class PreferencesDialog(PageDialog):
    """Venster > Voorkeuren: Algemeen, Tekenen en Sneltoetsen (SketchUp 2017)."""

    PAGES = ("Algemeen", "Tekenen", "Sneltoetsen")
    SIZE = (680, 470)

    def __init__(self, win, page=0):
        super().__init__(win, "Voorkeuren", self.PAGES, page)
        self._build_general()
        self._build_drawing()
        self.shortcuts = ShortcutsPanel(win)
        self.add_page(self.shortcuts)

    def _build_general(self):
        page = QWidget()
        form = QFormLayout(page)
        self.segments = QSpinBox()
        self.segments.setRange(8, 512)
        self.segments.setValue(self.win.settings.circle_segments)
        self.segments.setToolTip("Aantal lijnstukken van een cirkel, boog of veelhoek. Typ tijdens het "
                                 "tekenen bijvoorbeeld 32s om het voor die vorm te veranderen.")
        form.addRow("Cirkelsegmenten", self.segments)
        form.addRow(note("Hoe meer segmenten, hoe ronder een cirkel eruitziet en hoe langer "
                         "tekenen en exporteren duurt."))
        self.updates = QCheckBox("Bij het opstarten op updates controleren")
        self.updates.setChecked(self.win.settings.check_updates)
        self.updates.setToolTip("Eén keer per dag kijkt Volura op GitHub of er een nieuwere versie is.")
        form.addRow(self.updates)
        self.add_page(page)

    def _build_drawing(self):
        page = QWidget()
        form = QFormLayout(page)
        self.split_lines = QCheckBox("Lijnen automatisch splitsen")
        self.split_lines.setChecked(True)
        self.split_lines.setEnabled(False)
        self.split_lines.setToolTip("Zoals in SketchUp: randen die elkaar kruisen splitsen elkaar altijd, "
                                    "en een lijn over een vlak splitst dat vlak.")
        form.addRow(self.split_lines)
        form.addRow(note("Dit doet Volura altijd, net als SketchUp; het is hier alleen te zien."))
        self.add_page(page)

    def apply(self):
        s = self.win.settings
        s.circle_segments = self.segments.value()
        s.check_updates = self.updates.isChecked()
        s.save()
        self.win.view.tools["circle"].segments = s.circle_segments


class ShortcutsPanel(QWidget):
    """Voorkeuren > Sneltoetsen: filter, de opdrachten per menu, en sneltoets toevoegen/afhalen."""

    def __init__(self, win):
        super().__init__()
        self.win = win
        self.items = {}
        lay = QVBoxLayout(self)
        top = QHBoxLayout()
        top.addWidget(QLabel("Filter"))
        self.filter = QLineEdit()
        self.filter.setPlaceholderText("Typ om te zoeken")
        self.filter.textChanged.connect(self._filter)
        top.addWidget(self.filter, 1)
        lay.addLayout(top)
        self.tree = QTreeWidget()
        self.tree.setColumnCount(2)
        self.tree.setHeaderLabels(["Opdracht", "Sneltoets"])
        self.tree.header().setSectionResizeMode(0, QHeaderView.Stretch)
        self.tree.setColumnWidth(1, 150)
        lay.addWidget(self.tree, 1)
        row = QHBoxLayout()
        row.addWidget(QLabel("Sneltoets toevoegen"))
        self.key_edit = QKeySequenceEdit()
        row.addWidget(self.key_edit, 1)
        self.add_button = QPushButton("+")
        self.add_button.setToolTip("Zet de ingetypte sneltoets op de gekozen opdracht")
        self.add_button.clicked.connect(self.add_shortcut)
        self.remove_button = QPushButton("−")
        self.remove_button.setToolTip("Haalt de sneltoets van de gekozen opdracht af")
        self.remove_button.clicked.connect(self.remove_shortcut)
        self.restore_button = QPushButton("Alles herstellen")
        self.restore_button.setToolTip("Zet alle sneltoetsen terug op de standaardwaarden")
        self.restore_button.clicked.connect(self.restore_all)
        for b in (self.add_button, self.remove_button, self.restore_button):
            b.setMinimumWidth(40)
            row.addWidget(b)
        lay.addLayout(row)
        self.reload()

    def reload(self, keep=None):
        """Every command of the menus, grouped per menu, with the shortcut it has now."""
        keep = keep or self.selected_path()
        self.tree.blockSignals(True)
        self.tree.clear()
        self.items = {}
        groups = {}
        for path, action in self.win.command_paths().items():
            menu, _, name = path.partition("/")
            groups.setdefault(menu, []).append((name, path, action))
        for menu, rows in groups.items():
            head = QTreeWidgetItem([menu, ""])
            head.setFlags(head.flags() & ~Qt.ItemIsSelectable)
            self.tree.addTopLevelItem(head)
            for name, path, action in sorted(rows, key=lambda r: r[0].lower()):
                it = QTreeWidgetItem([name, self.win.shortcut_text(action)])
                it.setData(0, Qt.UserRole, path)
                head.addChild(it)
                self.items[path] = it
            head.setExpanded(True)
        self.tree.blockSignals(False)
        if keep in self.items:
            self.tree.setCurrentItem(self.items[keep])

    def selected_path(self):
        it = self.tree.currentItem()
        if it is None:
            return None
        return it.data(0, Qt.UserRole) or None

    def select_command(self, path):
        it = self.items.get(path)
        if it is None:
            return False
        self.tree.setCurrentItem(it)
        return True

    def add_shortcut(self, checked=False):
        """The "+" button: the typed shortcut goes to the chosen command."""
        path = self.selected_path()
        seq = QKeySequence(self.key_edit.keySequence())
        if not path or seq.isEmpty() or not self.win.assign_shortcut(path, seq):
            return False
        self.key_edit.clear()
        self._show(path)
        return True

    def remove_shortcut(self, checked=False):
        """The "−" button: the typed shortcut, or all shortcuts, off the chosen command."""
        path = self.selected_path()
        if not path:
            return False
        seq = QKeySequence(self.key_edit.keySequence())
        if not self.win.remove_shortcut(path, None if seq.isEmpty() else seq):
            return False
        self._show(path)
        return True

    def restore_all(self, checked=False):
        """Alles herstellen: back to the shortcuts Volura ships with."""
        self.win.restore_shortcuts()
        self.reload()

    def _show(self, path):
        action = self.win.command_paths().get(path)
        it = self.items.get(path)
        if action is not None and it is not None:
            it.setText(1, self.win.shortcut_text(action))

    def _filter(self, text):
        text = text.strip().lower()
        for path, it in self.items.items():
            it.setHidden(bool(text) and text not in path.lower())
        for i in range(self.tree.topLevelItemCount()):
            head = self.tree.topLevelItem(i)
            found = any(not head.child(j).isHidden() for j in range(head.childCount()))
            head.setHidden(bool(text) and not found)


class WelcomeDialog(QDialog):
    """Help > Welkom bij Volura, the welcome screen of SketchUp 2017."""

    STEPS = (("Teken een vorm", "Lijn (L), Rechthoek (R), Cirkel (C) of Veelhoek: van elke gesloten "
                                "vorm maakt Volura een vlak."),
             ("Duwen/trekken", "Kies Duwen/trekken (P), klik het vlak en typ een maat: er ontstaat een "
                               "volume. Omlaag maakt een uitsparing of gat."),
             ("Exporteer naar je slicer", "Bestand > Exporteren als 3MF (meerkleuren) of STL, en open "
                                          "het bestand in PrusaSlicer, OrcaSlicer of Bambu Studio."))

    def __init__(self, win):
        super().__init__(win)
        self.setWindowTitle("Welkom bij Volura")
        self.resize(700, 430)
        lay = QVBoxLayout(self)
        title = QLabel("Volura")
        title.setStyleSheet("font-size: 34px; font-weight: bold; color: #d23b3b;")
        lay.addWidget(title)
        lay.addWidget(QLabel("<b>3D-modelleren voor 3D-printen, zoals in SketchUp 2017.</b>"))
        lay.addSpacing(10)
        for i, (head, text) in enumerate(self.STEPS, 1):
            step = QLabel(f"<b>{i}. {head}</b><br><span style='color:#555'>{text}</span>")
            step.setWordWrap(True)
            step.setTextFormat(Qt.RichText)
            lay.addWidget(step)
        lay.addStretch(1)
        self.show_at_start = QCheckBox("Altijd tonen bij opstarten")
        self.show_at_start.setChecked(welcome_enabled())
        self.show_at_start.toggled.connect(self.save)
        lay.addWidget(self.show_at_start)
        row = QHBoxLayout()
        row.addStretch(1)
        start = QPushButton("Begin met Volura")
        start.setDefault(True)
        start.clicked.connect(self.accept)
        row.addWidget(start)
        lay.addLayout(row)

    def save(self, on=None):
        q = qsettings()
        q.setValue(SHOW_WELCOME_KEY, bool(self.show_at_start.isChecked() if on is None else on))
        q.sync()

    def accept(self):
        self.save()
        super().accept()


# ---------------------------------------------------------------- window

class MainWindow(QMainWindow):
    #: de actieve omgeving: "3d" of "2d" (het 2D-werkblad); hoort niet bij het bestand
    mode = "3d"

    def __init__(self):
        super().__init__()
        self.settings = Settings()
        set_length_precision(self.settings.length_precision)
        self.doc = Document()
        self.view = Viewport(self.doc, self.settings, self)
        self.view.tools = {cls.name: cls(self.view) for cls in TOOL_CLASSES}
        self.sheet_view = SheetView(self.doc, self.settings, self)
        self.central = QStackedWidget(self)
        self.central.addWidget(self.view)
        self.central.addWidget(self.sheet_view)
        self.setCentralWidget(self.central)
        self.setWindowIcon(icon("app"))
        self.setAcceptDrops(True)
        self.clipboard = []
        self.tool_actions = {}
        self.sheet_tool_actions = {}
        self.resize(1400, 900)

        self._build_statusbar()
        self._build_actions()
        self._build_menus()
        self._build_toolbars()
        self._build_docks()
        self.load_shortcuts()

        self.doc.changed.connect(self.on_doc_changed)
        self.doc.selection_changed.connect(self.on_selection_changed)
        self.view.status_changed.connect(self.hint.setText)
        self.view.vcb_changed.connect(self.on_vcb_changed)
        self.view.tool_changed.connect(self.on_tool_changed)
        self.view.context_menu.connect(self.show_context_menu)
        self.view.cam_history_changed.connect(self.on_cam_history)
        self.sheet_view.status_changed.connect(self.hint.setText)
        self.sheet_view.vcb_changed.connect(self.on_vcb_changed)
        self.sheet_view.tool_changed.connect(self.on_sheet_tool_changed)
        self.sheet_view.coords_changed.connect(self.coords.setText)
        self.sheet_view.sheet_settings_requested.connect(self.open_sheet_settings)
        self.sheet_view.auto_arrange_requested.connect(self.auto_arrange)
        self.sheet_view.to_model_requested.connect(self.sheet_to_model)
        self.sheet_view.selection_changed.connect(self.info.refresh)
        self.set_tool("select")
        self.update_title()
        QTimer.singleShot(0, self.initial_view)
        QTimer.singleShot(0, self.show_welcome_on_startup)
        QTimer.singleShot(0, self.offer_autosave_restore)
        QTimer.singleShot(UPDATE_DELAY_MS, self.check_for_updates_on_startup)
        self.autosave_timer = QTimer(self)
        self.autosave_timer.setInterval(AUTOSAVE_INTERVAL_MS)
        self.autosave_timer.timeout.connect(self.autosave)
        self.autosave_timer.start()

    # ------------------------------------------------------------ autosave
    def autosave(self):
        """Automatisch opslaan: elke 5 minuten, alleen als er iets is gewijzigd."""
        if not self.doc.dirty:
            return False
        try:
            fileio.save_document(self.doc, autosave_path())
        except Exception as e:
            errors.log_exception(note=f"automatisch opslaan mislukt: {e}")
            return False
        return True

    def offer_autosave_restore(self):
        """Bij opstarten vragen of het automatisch opgeslagen model hersteld moet worden."""
        if is_offscreen() or not os.path.exists(autosave_path()):
            return False
        r = QMessageBox.question(self, "Automatisch opgeslagen model",
                                 "Er is een automatisch opgeslagen model van een vorige sessie.\n\n"
                                 "Wilt u dat model herstellen?",
                                 QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
        if r != QMessageBox.Yes:
            remove_autosave()
            return False
        return self.restore_autosave()

    def restore_autosave(self):
        """Het automatisch opgeslagen model in laden; het document blijft zijn eigen naam houden."""
        path = self.doc.path
        try:
            self.view.tool.reset()
            fileio.load_document(self.doc, autosave_path())
        except Exception as e:
            QMessageBox.critical(self, "Herstellen mislukt", str(e))
            return False
        self.doc.path = path
        self.doc.dirty = True
        self.doc.notify()
        self.update_title()
        self.view.zoom_extents()
        self.hint.setText("Automatisch opgeslagen model hersteld.")
        return True

    def initial_view(self):
        self.view.set_view("iso")
        self.view.zoom_extents()

    # ------------------------------------------------------------ building
    def _build_statusbar(self):
        sb = self.statusBar()
        self.hint = QLabel()
        sb.addWidget(self.hint, 1)
        self.coords = QLabel()
        self.coords.setObjectName("sheetCoords")
        self.coords.setToolTip("Plaats van de cursor op het 2D-werkblad")
        self.vcb_label = QLabel("Afmetingen")
        self.vcb = QLineEdit()
        self.vcb.setFixedWidth(180)
        self.vcb.returnPressed.connect(self.on_vcb_return)
        sb.addPermanentWidget(self.coords)
        sb.addPermanentWidget(self.vcb_label)
        sb.addPermanentWidget(self.vcb)
        self.mode_button = ModeSwitch()
        sb.addPermanentWidget(self.mode_button)
        self.mode_button.clicked.connect(lambda checked=False: self.toggle_sheet_mode())
        self._update_mode_button()
        self.coords.hide()

    def _update_mode_button(self):
        """De schuifknop '3D | 2D' rechtsonder: de helft van de modus van nu is blauw."""
        two = self.mode == "2d"
        self.mode_button.setText("3D | 2D")
        self.mode_button.setChecked(two)
        self.mode_button.setToolTip(
            f"Nu: {'2D-werkblad' if two else '3D-model'} — klik voor het "
            f"{'3D-model' if two else '2D-werkblad'} (Ctrl+Shift+2)")

    # ------------------------------------------------------------ 3D of 2D
    def set_mode(self, mode) -> bool:
        """Naar het 3D-model of het 2D-werkblad; de modus hoort niet bij het bestand."""
        want = "2d" if str(mode).lower().startswith("2") else "3d"
        changed = want != self.mode
        self.mode = want
        self.central.setCurrentWidget(self.sheet_view if want == "2d" else self.view)
        if self.a_sheet.isChecked() != (want == "2d"):
            self.a_sheet.setChecked(want == "2d")
        self._update_mode_button()
        self.coords.setVisible(want == "2d")
        self._show_mode_toolbars(changed)
        self._enable_3d_tools(want == "3d")
        if want == "2d":
            self.sheet_view.set_tool("select")
            self.sheet_view.setFocus()
            self.sheet_view.announce()
        else:
            self.view.setFocus()
            self.view.announce()
        self._show_layers_panel(want == "2d")
        return changed

    #: De panelen van het standaardvenster die bij elke modus horen.
    MODE_PANELS = {"3d": ("Entiteitsinfo", "Materialen", "Overzicht", "Printcontrole"),
                   "2d": ("Entiteitsinfo", "Lagen")}

    def _show_layers_panel(self, two_d: bool):
        """Alleen de panelen die bij de modus horen: filamenten, overzicht en printcontrole
        bij het model, de lagen bij het werkblad. Entiteitsinfo toont wat bij de modus hoort."""
        if not hasattr(self, "layers_panel"):
            return
        mode = "2d" if two_d else "3d"
        for title in list(self.tray.panel_titles) + list(self.tray.extra_titles):
            shown = title in self.MODE_PANELS[mode]
            self.tray.set_panel_shown(title, shown)
            a = self.tray_actions.get(title)
            if a is not None:
                a.setVisible(shown)
        if two_d and self.tray.is_open("Printcontrole"):
            self.print_panel.clear_marks()
        self.layers_panel.refresh()
        self.tray.set_panel_open("Lagen", bool(two_d))
        self.info.refresh()

    def toggle_sheet_mode(self, checked=None):
        """Beeld > 2D-werkblad en de knop rechtsonder in de statusbalk."""
        want = "2d" if (self.mode == "3d" if checked is None else bool(checked)) else "3d"
        return self.set_mode(want)

    def zoom_extents(self):
        """Zoomen tot alles: in de 3D-weergave het model, op het werkblad de tekening."""
        (self.sheet_view if self.mode == "2d" else self.view).zoom_extents()

    def active_view(self):
        """Het venster dat nu bij de modus hoort: de 3D-weergave of het werkblad."""
        return self.sheet_view if self.mode == "2d" else self.view

    def _show_mode_toolbars(self, changed=True):
        """In 2D wijken de 3D-werkbalken voor die van het werkblad; terug herstelt alles.

        Wat zichtbaar was wordt alleen bij het binnengaan van 2D onthouden: een tweede
        set_mode("2d") (bijvoorbeeld door Plaat… of de doosgenerator) zou anders de
        verborgen stand onthouden en de werkbalken bij terugkeer weg laten.
        """
        if not hasattr(self, "tb_sheet"):
            return
        if self.mode == "2d":
            if changed or not getattr(self, "_hidden_3d_toolbars", None):
                self._hidden_3d_toolbars = {bar: not bar.isHidden()
                                            for bar in self._toolbars_3d()}
            for bar, shown in self._hidden_3d_toolbars.items():
                if shown:
                    bar.hide()
            self.tb_sheet.show()
        else:
            self.tb_sheet.hide()
            for bar, shown in getattr(self, "_hidden_3d_toolbars", {}).items():
                if shown:
                    bar.show()
            self._hidden_3d_toolbars = {}

    def _toolbars_3d(self):
        return [bar for bar in (getattr(self, "tb_main", None), getattr(self, "tb_tools", None),
                                getattr(self, "tb_print", None)) if bar is not None]

    def _enable_3d_tools(self, enabled: bool):
        """De 3D-gereedschappen en wat erbij hoort zijn in 2D buiten bereik gezet."""
        for a in list(self.tool_actions.values()) + [self.a_delete, self.a_sel_all,
                                                     self.a_sel_none, self.a_hide, self.a_lock]:
            a.setEnabled(enabled)
        if not enabled:
            for a in self.tool_actions.values():
                a.setChecked(False)
        else:
            self.on_tool_changed(self.view.tool.name if self.view.tool else "select")

    def act(self, text, slot, shortcut=None, icon_name=None, checkable=False, tip=None):
        a = QAction(text, self)
        if icon_name:
            a.setIcon(icon(icon_name))
        keys = [shortcut] if isinstance(shortcut, str) else list(shortcut or [])
        if keys:
            a.setShortcuts([QKeySequence(k) for k in keys])
        if checkable:
            a.setCheckable(True)
        a.setToolTip(f"{tip or text}" + (f" ({', '.join(keys)})" if keys else ""))
        a.triggered.connect(slot)
        self.addAction(a)
        return a

    def stub(self, text, shortcut=None, icon_name=None):
        """Disabled placeholder action for a SketchUp feature Volura does not have yet."""
        a = self.act(text, lambda checked=False: None, shortcut, icon_name)
        a.setToolTip("Nog niet beschikbaar")
        a.setStatusTip("Nog niet beschikbaar")
        a.setEnabled(False)
        return a

    def tool_act(self, text, name, shortcut=None, icon_name=None):
        """Action that selects a drawing tool; also reachable as self.tool_actions[name]."""
        a = self.act(text, lambda checked=False, n=name: self.set_tool(n), shortcut, icon_name,
                     checkable=True)
        self.tool_actions[name] = a
        return a

    def face_style_act(self, text, value):
        """One choice of the exclusive Beeld > Vlakstijl group."""
        a = self.act(text, lambda checked=False, v=value: self.set_face_style(v), None, None,
                     checkable=True)
        a.setChecked(self.settings.face_style == value)
        self.face_style_group.addAction(a)
        self.face_style_actions[value] = a
        return a

    @staticmethod
    def fill(menu, items):
        for a in items:
            if a is None:
                menu.addSeparator()
            elif isinstance(a, QMenu):
                menu.addMenu(a)
            else:
                menu.addAction(a)

    def _build_actions(self):
        A = self.act
        self.a_new = A("&Nieuw", self.file_new, "Ctrl+N", "new")
        self.a_open = A("&Openen", self.file_open, "Ctrl+O", "open")
        self.a_save = A("Op&slaan", self.file_save, "Ctrl+S", "save")
        self.a_save_as = A("Opslaan &als", self.file_save_as)
        self.a_save_copy = A("Kopie opslaan al&s", self.file_save_copy_as)
        self.a_import = A("&Importeren…", self.file_import)
        self.a_exp_stl = A("&3D-model (STL)…", self.export_stl, None, "stl")
        self.a_exp_3mf = A("&Meerkleuren 3MF…", self.export_3mf, None, "3mf")
        self.a_exp_colors = A("STL per &kleur…", self.export_stl_colors)
        self.a_exp_svg2d = A("2D-vector (&SVG)…", lambda checked=False: self.export_vector("svg"))
        self.a_exp_dxf2d = A("2D-vector (&DXF)…", lambda checked=False: self.export_vector("dxf"))
        self.a_print_setup = A("&Printer instellen…", lambda checked=False: self.print_setup())
        self.a_check = A("Print&controle", self.print_check, "F5", "check")
        self.a_print = A("&Afdrukken", self.export_3mf, "Ctrl+P", "3mf")
        self.a_quit = A("&Afsluiten", self.close, "Ctrl+Q")

        self.a_undo = A("&Ongedaan maken", self.undo, ["Alt+Backspace", "Ctrl+Z"], "undo")
        self.a_redo = A("Opnie&uw", self.redo, "Ctrl+Y", "redo")
        self.a_cut = A("&Knippen", self.cut, ["Shift+Del", "Ctrl+X"])
        self.a_copy = A("&Kopiëren", self.copy, "Ctrl+C")
        self.a_paste = A("&Plakken", self.paste, "Ctrl+V")
        self.a_paste_loc = A("Plakken op l&ocatie", self.paste_at_location)
        self.a_delete = A("&Wissen", self.delete, "Del")
        self.a_erase_guides = A("Hulplijnen &wissen", self.erase_guides)
        self.a_sel_all = A("&Alles selecteren", self.select_all, "Ctrl+A")
        self.a_sel_none = A("Niets selecteren", self.select_none, "Ctrl+T")
        self.a_hide = A("&Verbergen", self.hide_selected)
        self.a_unhide_sel = A("&Geselecteerd", self.unhide_selected)
        self.a_unhide_last = A("&Laatste", self.unhide_last)
        self.a_unhide = A("&Alles", self.unhide_all)
        self.a_lock = A("&Vergrendelen", self.lock_selected)
        self.a_unlock = A("&Ontgrendelen", self.unlock_selected)
        self.a_unlock_sel = A("&Geselecteerd", self.unlock_selected)
        self.a_unlock_all = A("&Alles", self.unlock_all)
        self.a_cut_face_model = A("&Met model", lambda checked=False: self.intersect_faces(True),
                                  None, "cut_faces", False,
                                  "Snijlijnen van de selectie met de rest van het model")
        self.a_cut_face_sel = A("&Met selectie", lambda checked=False: self.intersect_faces(False),
                                None, "cut_faces", False,
                                "Snijlijnen tussen de geselecteerde items")
        self.a_component = A("&Component maken", self.make_group, "G")
        self.a_group = A("&Groep maken", self.make_group)
        self.a_close_group = A("Groep/component &sluiten", self.close_group)
        self.a_explode = A("Groep e&xploderen", self.explode_group)
        self.a_edit_group = A("Groep &bewerken", self.edit_group)

        for name, label, ic, sc in TOOL_INFO:
            a = A(label, lambda checked=False, n=name: self.set_tool(n), sc, ic, checkable=True)
            self.tool_actions[name] = a

        self.a_freehand = self.tool_act("&Uit de vrije hand", "freehand", None, "freehand")
        self.a_arc = self.tool_act("&Boog", "arc", None, "arc")
        self.a_arc2 = self.tool_act("Boog door &2 punten", "arc2", "A", "arc2")
        self.a_arc3 = self.tool_act("Boog door &3 punten", "arc3", None, "arc3")
        self.a_pie = self.tool_act("&Taartpunt", "pie", None, "pie")
        self.a_rotrect = self.tool_act("&Gedraaide rechthoek", "rotrect", None, "rotrect")
        self.a_box = A("&Kubus / balk…", lambda checked=False: self.insert_primitive("box"))
        self.a_cylinder = A("&Cilinder…", lambda checked=False: self.insert_primitive("cylinder"))
        self.a_cone = A("&Kegel…", lambda checked=False: self.insert_primitive("cone"))
        self.a_sphere = A("&Bol…", lambda checked=False: self.insert_primitive("sphere"))
        self.a_torus = A("&Torus…", lambda checked=False: self.insert_primitive("torus"))

        self.a_extents = A("Zoomen tot alles", self.zoom_extents, ["Ctrl+Shift+E", "Shift+Z"],
                           "zoom_extents")
        self.views = {}
        for key, label, ic in (("top", "Boven", "view_top"), ("bottom", "Onder", "view_top"),
                               ("front", "Voor", "view_front"), ("back", "Achter", "view_back"),
                               ("left", "Links", "view_left"), ("right", "Rechts", "view_right"),
                               ("iso", "Iso", "view_iso")):
            self.views[key] = A(label, lambda checked=False, k=key: self.view.set_view(k), None, ic)
        self.a_prev_view = A("&Vorige", self.camera_prev)
        self.a_next_view = A("&Volgende", self.camera_next)
        self.a_prev_view.setEnabled(self.view.can_prev_camera())
        self.a_next_view.setEnabled(self.view.can_next_camera())
        self.a_ortho = A("Parallelle projectie", lambda checked=False: self.apply_projection(),
                         None, "ortho", checkable=True)
        self.a_persp = A("&Perspectief", lambda checked=False: self.apply_projection(), None, None,
                         checkable=True)
        proj = QActionGroup(self)
        proj.setExclusive(True)
        proj.addAction(self.a_ortho)
        proj.addAction(self.a_persp)
        self.a_persp.setChecked(True)
        self.a_fov = A("&Gezichtsveld", self.camera_fov)
        self.a_zoom_window = A("&Zoomvenster", self.camera_zoom_window, "Ctrl+Shift+W")

        self.a_toolbars = A("&Werkbalken", self.show_toolbar_dialog)
        self.a_hidden_geometry = A("Verborgen &geometrie", self.toggle_show_hidden, None, None,
                                   checkable=True)
        self.a_hidden_geometry.setChecked(self.settings.show_hidden)
        self.a_guides = A("Hulpl&ijnen", self.toggle_guides, None, None, checkable=True)
        self.a_guides.setChecked(self.settings.show_guides)
        self.a_grid = A("Printbed", self.toggle_grid, None, None, checkable=True)
        self.a_grid.setChecked(self.settings.show_grid)
        self.a_axes = A("&Assen", self.toggle_axes, None, None, checkable=True)
        self.a_axes.setChecked(self.settings.show_axes)
        self.a_edges = A("&Randen", self.toggle_edges, None, None, checkable=True)
        self.a_edges.setChecked(self.settings.show_edges)
        self.a_back_edges = A("&Achterliggende randen", self.toggle_back_edges, "K", None,
                              checkable=True)
        self.a_back_edges.setChecked(self.settings.back_edges)
        self.a_xray = A("&Röntgen", self.toggle_xray, "Alt+X", "xray", checkable=True)
        self.a_xray.setChecked(self.settings.xray)
        self.face_style_group = QActionGroup(self)
        self.face_style_group.setExclusive(True)
        self.face_style_actions = {}
        self.a_wireframe = self.face_style_act("&Draadmodel", WIREFRAME)
        self.a_hidden_lines = self.face_style_act("Verborgen &lijnen", HIDDENLINE)
        self.a_shaded = self.face_style_act("&Gearceerd", SHADED)
        self.a_monochrome = self.face_style_act("&Monochroom", MONOCHROME)
        self.a_hide_rest = A("&Rest van model verbergen", self.toggle_hide_rest, None, None,
                             checkable=True)
        self.a_hide_rest.setChecked(self.settings.hide_rest)

        self.a_union = A("Samenvoegen", lambda checked=False: self.solid_op("union"), None, "union")
        self.a_subtract = A("Aftrekken", lambda checked=False: self.solid_op("subtract"), None, "subtract")
        self.a_intersect = A("Snijpunt", lambda checked=False: self.solid_op("intersect"), None, "intersect")
        self.a_trim = A("&Bijsnijden", lambda checked=False: self.trim_solids(), None, "trim",
                        False, "De tweede groep met de eerste bijsnijden (B = B - A)")
        self.a_split = A("Splitsen", self.split_parts)
        self.a_follow_me = self.tool_act("&Volg mij", "followme", None, "followme")
        self.a_shell = A("&Buitenschil", self.outer_shell, None, "shell",
                         False, "Alleen de buitenkant van de geselecteerde groepen overhouden")
        self.a_protractor = self.tool_act("&Gradenboog", "protractor")
        self.a_drop = A("Op bed plaatsen", self.drop_to_bed, "Ctrl+B", "bed")
        self.a_center = A("Centreren op bed", self.center_on_bed)

        self.a_thread = A("Schroefdraad...", lambda checked=False: self.thread_dialog(), None, "thread",
                          False, "Schroefdraad zetten of een bout of moer maken (ISO metrisch)")
        self.a_thread_ext = A("Buitendraad op cilinder...",
                              lambda checked=False: self.thread_dialog("external"), None, "thread_ext")
        self.a_thread_int = A("Binnendraad in gat...",
                              lambda checked=False: self.thread_dialog("internal"), None, "thread_int")
        self.a_thread_bolt = A("Nieuwe bout...",
                               lambda checked=False: self.thread_dialog("bolt"), None, "bolt")
        self.a_thread_nut = A("Nieuwe moer of bus...",
                              lambda checked=False: self.thread_dialog("nut"), None, "nut")

        self.a_cutplane = self.tool_act("Snijden", "cutplane", None, "cutplane")
        self.a_cut_join = A("Snijden en verbinden...", lambda checked=False: self.cut_join_dialog(),
                            None, "cut_join", False,
                            "Knip het model op een vlak en maak pennen, gaten of een zwaluwstaart")
        self.a_fillet = self.tool_act("Afronden", "fillet", None, "fillet")
        self.a_fillet.setToolTip("Klik randen aan en typ een straal: een bolle of holle afronding")
        self.a_chamfer = self.tool_act("Afschuinen", "chamfer", None, "chamfer")
        self.a_chamfer.setToolTip("Klik randen aan en typ een afstand: een schuine kant langs de rand")

        self.a_prefs = A("&Voorkeuren", self.preferences)
        self.a_model_info = A("&Modelinfo", lambda checked=False: self.model_info())
        self.a_welcome = A("&Welkom bij Volura", self.show_welcome)
        self.a_updates = A("Controleren op &updates…", lambda checked=False: self.check_for_updates())
        self.a_help = A("&Sneltoetsen", self.show_help, "F1")
        self.a_about = A("Over &Volura", self.about)

        # ---- het 2D-werkblad
        self.a_sheet = A("2D-werk&blad", self.toggle_sheet_mode, "Ctrl+Shift+2", None,
                         checkable=True, tip="Tekenen in 2D voor lasersnijder en frees")
        self.a_to_sheet = A("Naar 2D-werkblad &kopiëren…", self.copy_to_sheet, None, None, False,
                            "Geselecteerde vlakken, bovenaanzicht of doorsnede naar het werkblad")
        self.a_to_laser = A("Naar &laser of frees…", self.send_selection_to_sheet, None, "to_laser",
                            False, "De geselecteerde vlakken of platen op een laag van het werkblad "
                                   "zetten om te snijden, graveren of frezen")
        self.a_sheet_to_model = A("Onderdelen naar &3D-model", self.sheet_to_model, None, "to_model",
                                  False, "Wat gesneden wordt als plaat met de plaatdikte in het "
                                         "3D-model zetten")
        for name, label, tip in SHEET_TOOL_INFO:
            self.sheet_tool_act(label, name, tip)
        self.a_mirror_h = A("Spiegelen &horizontaal",
                            lambda checked=False: self.sheet_view.mirror(True), None, "mirror_h",
                            False, "Links en rechts omwisselen rond het midden van de selectie")
        self.a_mirror_v = A("Spiegelen &verticaal",
                            lambda checked=False: self.sheet_view.mirror(False), None, "mirror_v",
                            False, "Boven en onder omwisselen rond het midden van de selectie")
        # ---- bewerkingen op de selectie van het 2D-werkblad
        self.a_sheet_union = A("Contouren samenvoegen",
                               lambda checked=False: self.sheet_op("union"), None, "union", False,
                               "De geselecteerde gesloten contouren als één vorm")
        self.a_sheet_subtract = A("Contouren aftrekken",
                                  lambda checked=False: self.sheet_op("subtract"), None, "subtract",
                                  False, "Het eerste geselecteerde pad minus de rest")
        self.a_sheet_intersect = A("Snijpunt van contouren",
                                   lambda checked=False: self.sheet_op("intersect"), None,
                                   "intersect", False, "Alleen wat alle contouren gemeen hebben")
        self.a_sheet_join = A("Lijnen aaneenrijgen",
                              lambda checked=False: self.sheet_view.join_selected(), None, "join",
                              False, "Losse lijnstukken tot polylijnen en vormen maken")
        self.a_sheet_split = A("Opsplitsen in lijnstukken",
                               lambda checked=False: self.sheet_view.split_selected(), None, "split",
                               False, "Elk lijnstuk van de selectie apart zetten")
        self.a_sheet_align = A("Uitlijnen…", lambda checked=False: self.sheet_view.align_selected("left"),
                               None, "align", False,
                               "De selectie links, midden, rechts, boven of onder op één lijn")
        self.sheet_align_actions = {}
        for mode, label in (("left", "Links uitlijnen"), ("center", "Midden uitlijnen"),
                            ("right", "Rechts uitlijnen"), ("top", "Boven uitlijnen"),
                            ("bottom", "Onder uitlijnen")):
            self.sheet_align_actions[mode] = A(
                label, lambda checked=False, m=mode: self.sheet_view.align_selected(m), None,
                None, False, "De selectie " + label.lower().replace(" uitlijnen", "")
                                    .replace("midden", "in het midden") + " zetten")
        self.a_sheet_distribute = A("Gelijk verdelen…",
                                    lambda checked=False: self.sheet_view.distribute_selected("x"),
                                    None, "distribute", False,
                                    "Over de selectie steeds dezelfde afstand zetten")
        self.sheet_distribute_actions = {
            "x": A("Horizontaal verdelen",
                   lambda checked=False: self.sheet_view.distribute_selected("x"), None, None,
                   False, "De middenpunten horizontaal op gelijke afstand zetten"),
            "y": A("Verticaal verdelen",
                   lambda checked=False: self.sheet_view.distribute_selected("y"), None, None,
                   False, "De middenpunten verticaal op gelijke afstand zetten"),
        }
        self.a_sheet_duplicate = A("Dupliceren in rij of raster…",
                                   lambda checked=False: self.sheet_view.duplicate_dialog(), None,
                                   "duplicate", False,
                                   "Kopieën van de selectie zetten met een vaste afstand")
        self.a_sheet_grid = A("Raster vastzetten", self.set_sheet_grid_snap, None, "grid",
                              checkable=True, tip="De cursor naar rasterkruisingen trekken")
        self.a_sheet_grid.setChecked(True)
        self.a_sheet_round = A("Afgeronde hoeken bij verschuiving", self.set_sheet_round_join, None,
                               "round_join", checkable=True,
                               tip="Verschuivingen met ronde hoeken (uit) of scherpe hoeken (aan)")
        self.a_sheet_round.setChecked(True)
        # ---- lagen van het werkblad
        self.a_sheet_to_layer = A("Naar &laag", lambda checked=False: self.move_selection_to_layer(),
                                  None, "layers", False,
                                  "De selectie naar de actieve laag zetten; het pijltje kiest een andere")
        self.a_sheet_layers = A("Lagen van het &werkblad", self.open_layers_panel, None, None, False,
                                "Het lagenpaneel openen: kleur, bewerking en instellingen per laag")
        # ---- plaatformaat en automatisch nesten
        self.a_sheet_plaat = A("&Plaat…", lambda checked=False: self.open_sheet_settings(), None,
                               "sheet", False,
                               "Plaatformaat, materiaaldikte, marge, tussenruimte en aantal platen")
        self.a_sheet_nest = A("&Automatisch schikken…", lambda checked=False: self.auto_arrange(),
                              None, "nest", False,
                              "Onderdelen zo zuinig mogelijk over de platen schikken")
        # ---- doos met vingerlassen
        self.a_box_joint = A("Doos met &vingerlassen…",
                             lambda checked=False: self.box_joint_dialog(), None, "boxjoint",
                             False,
                             "Doos met vingerlassen voor de lasersnijder, op het werkblad "
                             "of als voorbeeld in 3D")

    def sheet_tool_act(self, label, name, tip=None):
        """Knop voor een 2D-gereedschap; de toets werkt op het werkblad (M, Q, S, spatie)."""
        a = self.act(label, lambda checked=False, n=name: self.sheet_view.set_tool(n), None,
                     name, checkable=True, tip=tip)
        self.sheet_tool_actions[name] = a
        return a

    def _build_menus(self):
        mb = self.menuBar()

        m = mb.addMenu("&Bestand")
        export = QMenu("&Exporteren", m)
        for a in (self.a_exp_stl, self.a_exp_3mf, self.a_exp_colors, None,
                  self.a_exp_svg2d, self.a_exp_dxf2d):
            if a is None:
                export.addSeparator()
            else:
                export.addAction(a)
        self.recent_menu = QMenu("&Recente bestanden", m)
        self.update_recent_menu()
        self.fill(m, (self.a_new, self.a_open, self.recent_menu, None,
                      self.a_save, self.a_save_as, self.a_save_copy, None,
                      self.a_import, export, None,
                      self.a_print_setup, self.a_check, self.a_print, None,
                      self.a_quit))

        m = mb.addMenu("Be&werken")
        unhide = QMenu("Zichtbaar &maken", m)
        self.fill(unhide, (self.a_unhide_sel, self.a_unhide_last, self.a_unhide))
        unlock = QMenu("&Ontgrendelen", m)
        self.fill(unlock, (self.a_unlock_sel, self.a_unlock_all))
        cut_faces = QMenu("&Vlakken snijden", m)
        self.fill(cut_faces, (self.a_cut_face_model, self.a_cut_face_sel))
        self.fill(m, (self.a_undo, self.a_redo, None,
                      self.a_cut, self.a_copy, self.a_paste, self.a_paste_loc, self.a_to_sheet,
                      self.a_to_laser, self.a_delete,
                      self.a_erase_guides, None,
                      self.a_sel_all, self.a_sel_none, None,
                      self.a_hide, unhide, None,
                      self.a_lock, unlock, None,
                      self.a_component, self.a_group, self.a_close_group, None,
                      cut_faces))

        m = mb.addMenu("Beel&d")
        edge_style = QMenu("&Randstijl", m)
        self.fill(edge_style, (self.a_edges, self.a_back_edges))
        face_style = QMenu("&Vlakstijl", m)
        self.fill(face_style, (self.a_xray, None, self.a_wireframe, self.a_hidden_lines,
                               self.a_shaded, self.a_monochrome))
        comp_edit = QMenu("Component&bewerking", m)
        self.fill(comp_edit, (self.a_hide_rest,))
        self.fill(m, (self.a_toolbars, self.a_sheet, None,
                      self.a_sheet_plaat, self.a_sheet_nest, self.a_box_joint,
                      self.a_sheet_to_model, None,
                      self.a_hidden_geometry, self.a_axes, self.a_guides, self.a_grid, None,
                      edge_style, face_style, comp_edit))

        m = mb.addMenu("&Camera")
        std_views = QMenu("Standaard&weergaven", m)
        self.fill(std_views, tuple(self.views[k] for k in
                                   ("top", "bottom", "front", "back", "left", "right", "iso")))
        self.fill(m, (self.a_prev_view, self.a_next_view, std_views, None,
                      self.a_ortho, self.a_persp, None,
                      self.tool_actions["orbit"], self.tool_actions["pan"], self.tool_actions["zoom"],
                      self.a_fov, self.a_zoom_window, self.a_extents))

        m = mb.addMenu("&Tekenen")
        lines = QMenu("&Lijnen", m)
        self.fill(lines, (self.tool_actions["line"], self.a_freehand))
        arcs = QMenu("&Bogen", m)
        self.fill(arcs, (self.a_arc, self.a_arc2, self.a_arc3, self.a_pie))
        shapes = QMenu("&Vormen", m)
        self.fill(shapes, (self.tool_actions["rectangle"], self.a_rotrect, self.tool_actions["circle"],
                           self.tool_actions["polygon"]))
        primitives = QMenu("&Basisvormen", m)
        self.fill(primitives, (self.a_box, self.a_cylinder, self.a_cone, self.a_sphere, self.a_torus))
        self.fill(m, (lines, arcs, shapes, None, primitives))

        m = mb.addMenu("&Gereedschap")
        solids = QMenu("&Volumegereedschap", m)
        self.fill(solids, (self.a_intersect, self.a_union, self.a_subtract, self.a_trim, self.a_split))
        self.print_tools_menu = QMenu("&Printhulpmiddelen", m)
        self.fill(self.print_tools_menu, (self.a_thread, self.a_thread_ext, self.a_thread_int,
                                          self.a_thread_bolt, self.a_thread_nut, None,
                                          self.a_cutplane, self.a_cut_join, None,
                                          self.a_fillet, self.a_chamfer))
        self.fill(m, (self.tool_actions["select"], self.tool_actions["eraser"], self.tool_actions["paint"],
                      None,
                      self.tool_actions["move"], self.tool_actions["rotate"], self.tool_actions["scale"],
                      None,
                      self.tool_actions["pushpull"], self.a_follow_me, self.tool_actions["offset"], None,
                      self.a_shell, solids, None,
                      self.tool_actions["tape"], self.a_protractor, None,
                      self.tool_actions["text"], None,
                      self.print_tools_menu))

        m = mb.addMenu("&Venster")
        self.standard_menu = QMenu("Standaard&venster", m)
        self.fill(m, (self.standard_menu, None, self.a_model_info, self.a_prefs))

        m = mb.addMenu("&Help")
        self.fill(m, (self.a_welcome, self.a_help, None, self.a_updates, self.a_about))

    def _build_toolbars(self):
        self._dropdown_buttons = {}
        self._toolbar_tool_icons = {}
        T = self.tool_actions

        tb = QToolBar("Aan de slag")
        tb.setObjectName("gettingstarted")
        tb.setIconSize(QSize(24, 24))
        self.tb_main = tb
        for a in (T["select"], T["eraser"], T["line"], self.a_arc2, T["rectangle"],
                  T["pushpull"], T["offset"], T["move"], T["rotate"], T["scale"], None,
                  T["tape"], T["text"], T["paint"], None,
                  T["orbit"], T["pan"], T["zoom"], self.a_extents, None,
                  self.a_check, self.a_print):
            tb.addSeparator() if a is None else tb.addAction(a)
        self.addToolBar(Qt.TopToolBarArea, tb)
        self._add_dropdown(tb, T["line"], (T["line"], self.a_freehand))
        self._add_dropdown(tb, self.a_arc2, (self.a_arc, self.a_arc2, self.a_arc3, self.a_pie))
        self._add_dropdown(tb, T["rectangle"], (T["rectangle"], self.a_rotrect, T["circle"], T["polygon"]))

        tools = QToolBar("Grote gereedschapsset")
        tools.setObjectName("tools")
        tools.setIconSize(QSize(28, 28))
        self.tb_tools = tools
        for n in ("select", "eraser", "paint", None, "line", "rectangle", "circle", "polygon",
                  "text", None, "pushpull", "offset", "move", "rotate", "scale", None, "tape",
                  None, "orbit", "pan", "zoom"):
            tools.addSeparator() if n is None else tools.addAction(self.tool_actions[n])
        tools.addAction(self.a_extents)
        self.addToolBar(Qt.LeftToolBarArea, tools)
        tools.hide()

        self._build_print_tools_toolbar()
        self._build_sheet_toolbar()

    def _build_sheet_toolbar(self):
        """Werkbalk '2D-werkblad' links; alleen zichtbaar in de 2D-modus."""
        bar = self.findChild(QToolBar, "SheetToolbar")
        if bar is None:
            bar = QToolBar("2D-werkblad")
            bar.setObjectName("SheetToolbar")
            bar.setIconSize(QSize(24, 24))
            self.addToolBar(Qt.LeftToolBarArea, bar)
        self.tb_sheet = bar
        if not bar.actions():
            T = self.sheet_tool_actions
            for a in (T["select"], T["move"], T["rotate"], T["scale"], None,
                      T["line"], T["rectangle"], T["roundrect"], T["circle"], T["ellipse"],
                      T["arc"], T["centerarc"], T["polygon"], T["text"], T["offset"], None,
                      self.a_sheet_union, self.a_sheet_subtract, self.a_sheet_intersect,
                      self.a_sheet_join, self.a_sheet_split, None,
                      self.a_mirror_h, self.a_mirror_v, self.a_sheet_align,
                      self.a_sheet_distribute, self.a_sheet_duplicate, None,
                      self.a_sheet_grid, self.a_sheet_round, self.a_sheet_to_layer,
                      None, self.a_sheet_plaat, self.a_sheet_nest, self.a_box_joint,
                      self.a_sheet_to_model, None, self.a_extents):
                bar.addSeparator() if a is None else bar.addAction(a)
            self._add_dropdown(bar, self.a_sheet_align,
                               tuple(self.sheet_align_actions.values()))
            self._add_dropdown(bar, self.a_sheet_distribute,
                               tuple(self.sheet_distribute_actions.values()))
            self._add_layer_dropdown(bar, self.a_sheet_to_layer)
        bar.hide()          # SketchUp toont standaard geen gereedschapsbalk links

    def _add_layer_dropdown(self, tb, action):
        """Knop met een uitklapmenu van de lagen; het menu wordt bij elk openen gevuld."""
        btn = tb.widgetForAction(action)
        if not isinstance(btn, QToolButton):
            return
        menu = QMenu(btn)
        menu.aboutToShow.connect(lambda m=menu: self._fill_layer_menu(m))
        btn.setMenu(menu)
        btn.setPopupMode(QToolButton.MenuButtonPopup)
        self._dropdown_buttons[action] = btn

    def _fill_layer_menu(self, menu):
        """Het menu "Naar laag": één regel per laag van het werkblad."""
        menu.clear()
        for name, lay in self.doc.sheet.layers.items():
            a = menu.addAction(f"{name}   —   {OP_LABELS.get(lay.op, lay.op)}")
            a.setIcon(swatch(lay.color or "#000000", 14))
            if lay.locked:
                a.setEnabled(False)
                a.setToolTip("Deze laag staat op slot")
            a.triggered.connect(lambda checked=False, n=name: self.move_selection_to_layer(n))
        if not self.doc.sheet.layers:
            menu.addAction("(geen lagen)").setEnabled(False)

    def sheet_op(self, op):
        """Bewerken op de selectie: Samenvoegen, Aftrekken of Snijpunt."""
        self.set_mode("2d")
        return self.sheet_view.sheet_boolean(op)

    def set_sheet_grid_snap(self, checked):
        self.sheet_view.set_grid_snap(checked)

    def set_sheet_round_join(self, checked):
        self.sheet_view.set_offset_round(checked)

    def move_selection_to_layer(self, name=None) -> bool:
        """"Naar laag": de selectie naar de actieve laag, of naar de gekozen laag."""
        self.set_mode("2d")
        sel = self.sheet_view.selected_paths()
        if not sel:
            self.hint.setText("Selecteer eerst de contouren die naar een laag moeten.")
            return False
        if name is None:
            name = self.sheet_view.draw_layer()
            if all(p.layer == name for p in sel):
                self.hint.setText(f"Die contouren staan al op laag {name}; kies met het pijltje "
                                  "een andere laag.")
                return False
        return self.sheet_view.move_to_layer(name)

    def open_layers_panel(self):
        """Venster > Standaardvenster > Lagen: kleur, bewerking en instellingen per laag."""
        self.tray.show()
        self.tray.raise_()
        self.tray.open_panel("Lagen")
        self.layers_panel.refresh()

    # ------------------------------------------------------------ plaat en nesten
    def open_sheet_settings(self):
        """Beeld > Plaat…: formaat, dikte, marge, tussenruimte en aantal platen."""
        self.set_mode("2d")
        dlg = SheetSettingsDialog(self)
        if dlg.exec() != QDialog.Accepted:
            self.setFocus()
            return
        values = dlg.values()
        sheet = self.doc.sheet

        def change():
            sheet.sheet_w = values["w"]
            sheet.sheet_h = values["h"]
            sheet.sheet_thickness = values["thickness"]
            sheet.sheet_margin = values["margin"]
            sheet.sheet_gap = values["gap"]
            sheet.sheet_count = values["count"]
        self.doc.checkpoint()
        change()
        self.doc.notify()
        self.sheet_view.zoom_extents()
        self.sheet_view._refresh_outside(announce=False)
        self.hint.setText(f"Plaat {values['w']:g} × {values['h']:g} mm, {values['count']} "
                          f"plaat{'s' if values['count'] != 1 else ''}, marge "
                          f"{values['margin']:g} mm, tussenruimte {values['gap']:g} mm")
        self.setFocus()

    def auto_arrange(self, options=None):
        """Beeld > Automatisch schikken…: onderdelen zuinig over de platen schikken."""
        self.set_mode("2d")
        if not self.doc.sheet.paths:
            self.hint.setText("Er staat niets op het werkblad om te schikken.")
            return
        if options is None:
            dlg = AutoArrangeDialog(self)
            if dlg.exec() != QDialog.Accepted:
                self.setFocus()
                return
            options = dlg.options()
        self.sheet_view.auto_arrange(options)
        self.setFocus()

    # ------------------------------------------------------------ doos met vingerlassen
    def box_joint_dialog(self):
        """Beeld > Doos met vingerlassen…: de panelen op het werkblad of als 3D-voorbeeld."""
        dlg = BoxJointDialog(self)
        if dlg.exec() != QDialog.Accepted:
            self.setFocus()
            return None
        options = dlg.options()
        if options["mode"] == BoxJointDialog.MODEL:
            made = self.make_box_model(options)
        else:
            made = self.make_box_sheet(options)
        self.setFocus()
        return made

    def box_panels_from(self, options):
        """De panelen van de doos uit een opties-dict; `ValueError` bij onmogelijke maten."""
        o = dict(options or {})
        sheet = self.doc.sheet
        t = float(o.get("t") or sheet.sheet_thickness or boxgen.DEFAULT_THICKNESS)
        return boxgen.box_panels(float(o.get("w", 100.0)), float(o.get("d", 60.0)),
                                 float(o.get("h", 40.0)), t, o.get("finger_w"),
                                 bool(o.get("inner", True)), str(o.get("lid", "none")),
                                 float(o.get("kerf", 0.0) or 0.0))

    def _box_failure(self, error, note):
        """Een mislukte doos: een duidelijke tekst in de statusbalk, nooit een crash."""
        if isinstance(error, ValueError):
            self.hint.setText(str(error))
        else:
            errors.log_exception(note=f"{note}: {error}")
            self.hint.setText(f"{note.capitalize()} lukt niet: {error}")
        return None

    def make_box_sheet(self, options):
        """"Op werkblad": de snijcontouren op het werkblad, automatisch op de plaat geschikt.

        Eén undo-stap: de panelen gaan er in dezelfde stap op en hun plek komt meteen uit
        het automatisch schikken. Zonder plaatformaat blijven ze naast elkaar liggen.
        """
        self.set_mode("2d")
        sheet = self.doc.sheet
        try:
            panels = self.box_panels_from(options)
            drawing = boxgen.panels_to_drawing(panels, labels=bool(options.get("labels", True)))
        except Exception as e:                      # de generator mag nooit crashen
            return self._box_failure(e, "de doos")
        note = ""
        used = 0
        if sheet.sheet_w and sheet.sheet_h:
            try:
                placements = nesting.nest(nesting.parts_of(drawing), float(sheet.sheet_w),
                                          float(sheet.sheet_h), float(sheet.sheet_margin),
                                          float(sheet.sheet_gap))
            except ValueError as e:
                note = f" ({e})"
            except Exception as e:
                return self._box_failure(e, "het schikken")
            else:
                drawing = self._nest_box_drawing(drawing, placements, sheet)
                used = max((si for (_p, si, *_r) in placements), default=0) + 1
        else:
            note = " (geen plaatformaat: kies Beeld > Plaat… om te schikken)"
        self.doc.checkpoint()
        added = sheet.merge(drawing)
        if used:
            sheet.sheet_count = max(int(sheet.sheet_count or 1), used)
        self.doc.notify()
        self.sheet_view.set_selection(added)
        self.sheet_view.zoom_extents()
        self.sheet_view._refresh_outside(announce=False)
        W, D, H = boxgen.box_size(float(options.get("w", 100.0)), float(options.get("d", 60.0)),
                                  float(options.get("h", 40.0)),
                                  float(options.get("t") or sheet.sheet_thickness
                                          or boxgen.DEFAULT_THICKNESS),
                                  bool(options.get("inner", True)),
                                  str(options.get("lid", "none")))
        self.hint.setText(f"{len(panels)} panelen van de doos op het werkblad "
                          f"({W:g} × {D:g} × {H:g} mm)"
                          + (f", geschikt op {used} plaat{'s' if used != 1 else ''}" if used else "")
                          + note)
        return added

    def _nest_box_drawing(self, drawing, placements, sheet):
        """De geschikte plekken op de tekening zetten: elk paneel met zijn gravure mee."""
        spacing = float(sheet.sheet_w) + SHEET_SPACING
        out = Drawing(layers={n: replace(l, params=dict(l.params))
                              for n, l in drawing.layers.items()},
                      **sheet._sheet_kwargs())
        for (part, si, angle, dx, dy) in placements:
            M = nesting.place_transform(part, angle, dx, dy, origin=(si * spacing, 0.0))
            lin, off = M[:2, :2], M[:2, 2]
            for p in part.paths:
                q = np.asarray(p.pts, dtype=float) @ lin.T + off
                out.paths.append(VPath(np.ascontiguousarray(q), p.closed, p.layer, p.filled,
                                       p.rule, p.src, _transform_meta(p.meta, lin, off)))
                out.ensure_layer(p.layer)
        return out

    def make_box_model(self, options):
        """"Voorbeeld in 3D": elk paneel als eigen groep, op zijn plek in de doos."""
        self.set_mode("3d")
        try:
            panels = self.box_panels_from(options)
            solids = boxgen.panels_to_manifolds(panels)
        except Exception as e:                      # de generator mag nooit crashen
            return self._box_failure(e, "de doos")
        t = float(options.get("t") or self.doc.sheet.sheet_thickness
                  or boxgen.DEFAULT_THICKNESS)
        W, D, H = boxgen.box_size(float(options.get("w", 100.0)), float(options.get("d", 60.0)),
                                  float(options.get("h", 40.0)), t,
                                  bool(options.get("inner", True)),
                                  str(options.get("lid", "none")))
        bx, by, _bz = self.settings.bed_size
        shift = (bx / 2.0 - W / 2.0, by / 2.0 - D / 2.0, 0.0)
        self.doc.checkpoint()
        groups = []
        for p, man in zip(panels, solids):
            g = ops.group_from_manifold(self.doc, man.translate(shift),
                                        self.settings.current_color, f"Doos - {p.name}")
            groups.append(g)
        self.doc.set_selection([("g", g.id, g.id) for g in groups])
        self.doc.notify()
        self.view.zoom_extents()
        self.hint.setText(f"{len(groups)} panelen van de doos in het model gezet "
                          f"({W:g} × {D:g} × {H:g} mm)")
        return groups

    def ask_sheet_text(self):
        """Dialoog bij het 2D-tekengereedschap: tekst, lettertype, hoogte en vet."""
        from .fonts import default_family, PREFERRED
        tool = self.sheet_view.tools["text"]
        dlg = QDialog(self)
        dlg.setWindowTitle("Tekst op het werkblad")
        form = QFormLayout(dlg)
        ed = QLineEdit(tool.text or getattr(self, "_last_sheet_text", "Volura"))
        family = QComboBox()
        family.addItems(PREFERRED)
        if tool.family:
            family.setCurrentText(tool.family)
        elif default_family() in PREFERRED:
            family.setCurrentText(default_family())
        bold = QCheckBox("Vet")
        bold.setChecked(bool(tool.bold))
        h = QDoubleSpinBox()
        h.setRange(0.5, 2000)
        h.setDecimals(2)
        h.setSuffix(" mm")
        h.setValue(float(tool.height) if tool.height else 10.0)
        form.addRow("Tekst", ed)
        form.addRow("Lettertype", family)
        form.addRow("", bold)
        form.addRow("Hoogte hoofdletter", h)
        form.addRow(QLabel("De letters worden als gesloten contouren neergezet, met de gaten\n"
                           "van o, a en e erin. Typ een hoogte in het meetvak om te wijzigen."))
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(dlg.accept)
        bb.rejected.connect(dlg.reject)
        form.addRow(bb)
        if dlg.exec() != QDialog.Accepted or not ed.text().strip():
            return False
        self._last_sheet_text = ed.text()
        if not tool.set_text(ed.text(), family.currentText(), h.value(), bold.isChecked()):
            self.hint.setText("Van deze tekst zijn geen contouren gekomen; kies een ander lettertype.")
            return False
        self.sheet_view.set_tool("text")
        self.sheet_view.setFocus()
        return True

    def _build_print_tools_toolbar(self):
        """Werkbalk 'Printhulpmiddelen' rechts; ook andere functies kunnen er knoppen aan toevoegen."""
        bar = self.findChild(QToolBar, "PrintToolsToolbar")
        if bar is None:
            bar = QToolBar("Printhulpmiddelen")
            bar.setObjectName("PrintToolsToolbar")
            bar.setIconSize(QSize(24, 24))
            self.addToolBar(Qt.RightToolBarArea, bar)
        self.tb_print = bar
        have = {id(a) for a in bar.actions()}
        todo = [a for a in (self.a_thread, self.a_thread_ext, self.a_thread_int,
                            self.a_thread_bolt, self.a_thread_nut, self.a_cutplane,
                            self.a_cut_join, self.a_fillet, self.a_chamfer)
                if id(a) not in have]
        if todo and bar.actions():
            bar.addSeparator()
        for a in todo:
            bar.addAction(a)

    def _add_dropdown(self, tb, action, items):
        """Give a toolbar button a SketchUp-style drop-down with the related drawing tools."""
        btn = tb.widgetForAction(action)
        if not isinstance(btn, QToolButton):
            return
        menu = QMenu(btn)
        for a in items:
            menu.addAction(a)
            if a.isEnabled():
                a.triggered.connect(
                    lambda checked=False, ic=a.icon(), act=action, b=btn: self._remember_tool_icon(act, b, ic))
        btn.setMenu(menu)
        btn.setPopupMode(QToolButton.MenuButtonPopup)
        self._dropdown_buttons[action] = btn

    def _remember_tool_icon(self, action, btn, ic):
        self._toolbar_tool_icons[action] = ic
        btn.setIcon(ic)

    def show_toolbar_dialog(self):
        dlg = QDialog(self)
        dlg.setWindowTitle("Werkbalken")
        lay = QVBoxLayout(dlg)
        lay.addWidget(QLabel("Toon of verberg de werkbalken:"))
        for bar, label in ((self.tb_main, "Aan de slag"), (self.tb_tools, "Grote gereedschapsset"),
                           (getattr(self, "tb_print", None), "Printhulpmiddelen"),
                           (getattr(self, "tb_sheet", None), "2D-werkblad")):
            if bar is None:
                continue
            cb = QCheckBox(label)
            cb.setChecked(not bar.isHidden())
            cb.toggled.connect(bar.setVisible)
            lay.addWidget(cb)
        bb = QDialogButtonBox(QDialogButtonBox.Ok)
        bb.accepted.connect(dlg.accept)
        lay.addWidget(bb)
        dlg.exec()

    def _build_docks(self):
        """One dock on the right: the Default Tray with the collapsible panels."""
        self.materials = MaterialsPanel(self)
        self.info = EntityInfoPanel(self)
        self.outliner = OutlinerPanel(self)
        self.print_panel = PrintCheckPanel(self)
        self.layers_panel = LayersPanel(self)
        self.tray = Tray((("Entiteitsinfo", self.info), ("Materialen", self.materials),
                          ("Overzicht", self.outliner)),
                         open_panels=("Entiteitsinfo", "Materialen"), parent=self,
                         extra_panels=(("Printcontrole", self.print_panel),
                                       ("Lagen", self.layers_panel)))
        self.addDockWidget(Qt.RightDockWidgetArea, self.tray)
        self.resizeDocks([self.tray], [TRAY_WIDTH], Qt.Horizontal)
        self._build_tray_menu()
        self._show_layers_panel(False)
        self.info.refresh()
        self.outliner.refresh()

    def _build_tray_menu(self):
        """Venster > Standaardvenster: an item per panel that opens/closes that panel."""
        self.tray_actions = {}
        for title in list(self.tray.panel_titles) + list(self.tray.extra_titles):
            a = self.act(f"&{title}", lambda checked=False, t=title: self.tray.toggle_panel(t),
                         checkable=True)
            a.setChecked(self.tray.is_open(title))
            self.tray_actions[title] = a
            self.standard_menu.addAction(a)
        self.tray.panel_toggled.connect(self.on_tray_panel_toggled)
        show = self.tray.toggleViewAction()
        show.setText("&Standaardvenster tonen")
        self.standard_menu.addAction(show)

    def on_tray_panel_toggled(self, title, is_open):
        a = self.tray_actions.get(title)
        if a is not None and a.isChecked() != is_open:
            a.setChecked(is_open)
        if title == "Printcontrole":
            if is_open:
                self.print_panel.refresh()
            else:
                self.print_panel.clear_marks()
        if title == "Lagen" and is_open:
            self.layers_panel.refresh()

    def open_print_check(self):
        """Venster > Standaardvenster > Printcontrole: open the panel and check the model."""
        self.tray.show()
        self.tray.raise_()
        if self.tray.is_open("Printcontrole"):
            self.print_panel.refresh()
        else:
            self.tray.open_panel("Printcontrole")   # opening runs the check through the toggle
        self.view.update()

    # ------------------------------------------------------------ shortcuts
    def command_paths(self):
        """{'Bestand/Nieuw': QAction} for every command in the menus; the first menu wins."""
        out = {}

        def walk(menu, prefix):
            for a in menu.actions():
                if a.isSeparator():
                    continue
                sub = a.menu()
                if sub is not None:
                    if sub is not getattr(self, "recent_menu", None):
                        walk(sub, f"{prefix}/{clean_text(a.text())}")
                    continue
                name = clean_text(a.text())
                if name:
                    out.setdefault(f"{prefix}/{name}", a)

        for a in self.menuBar().actions():
            menu = a.menu()
            if menu is not None:
                walk(menu, clean_text(a.text()))
        seen = {id(a) for a in out.values()}
        for a in self.actions():          # the commands that only live in the right-click menu
            name = clean_text(a.text())
            if name and id(a) not in seen:
                out[f"{OTHER_COMMANDS}/{name}"] = a
        return out

    def path_of(self, action):
        """The menu path of an action, the key its shortcuts are stored under."""
        for path, a in self.command_paths().items():
            if a is action:
                return path
        return None

    @staticmethod
    def shortcut_text(action):
        """The shortcuts of an action as shown in the list, e.g. 'Ctrl+Shift+E'."""
        return ", ".join(s.toString(QKeySequence.NativeText) for s in action.shortcuts())

    @staticmethod
    def sequence_texts(action):
        return [s.toString(QKeySequence.PortableText) for s in action.shortcuts()]

    def action_using(self, text, skip=None):
        """The command that now has this shortcut, if any."""
        for a in self.command_paths().values():
            if a is not skip and text in self.sequence_texts(a):
                return a
        return None

    @staticmethod
    def read_shortcut_value(value):
        if value is None:
            return []
        if isinstance(value, str):
            value = [value]
        return [str(v) for v in value if str(v).strip()]

    def load_shortcuts(self):
        """Remember the defaults, then put the shortcuts from QSettings on the actions."""
        paths = self.command_paths()
        self._default_shortcuts = {p: self.sequence_texts(a) for p, a in paths.items()}
        taken = {}
        for a in paths.values():
            for text in self.sequence_texts(a):
                taken.setdefault(text, a)
        stored = {}
        q = qsettings()
        for key in q.allKeys():
            if key.startswith(SHORTCUTS_KEY):
                path = key[len(SHORTCUTS_KEY):]
                if path in paths:
                    stored[path] = self.read_shortcut_value(q.value(key))
        for path in stored:                   # first let go of what those commands use now
            a = paths[path]
            for text in self.sequence_texts(a):
                if taken.get(text) is a:
                    del taken[text]
            a.setShortcuts([])
        for path, texts in stored.items():    # then hand out the stored ones
            a = paths[path]
            keep = [t for t in texts if taken.get(t) in (None, a)]
            for t in keep:
                taken[t] = a
            a.setShortcuts([QKeySequence(t) for t in keep])

    def store_shortcuts(self, action, texts):
        """Keep the user's shortcut list in QSettings; a default list is not stored."""
        path = self.path_of(action)
        if path is None:
            return
        q = qsettings()
        if texts == list(getattr(self, "_default_shortcuts", {}).get(path, [])):
            q.remove(SHORTCUTS_KEY + path)
        else:
            q.setValue(SHORTCUTS_KEY + path, texts)
        q.sync()

    def change_shortcuts(self, action, texts):
        action.setShortcuts([QKeySequence(t) for t in texts])
        self.store_shortcuts(action, texts)

    def assign_shortcut(self, path, sequence, ask=True):
        """Voorkeuren > Sneltoetsen > '+': the shortcut goes to this command."""
        action = self.command_paths().get(path)
        seq = QKeySequence(sequence)
        if action is None or seq.isEmpty():
            return False
        text = seq.toString(QKeySequence.PortableText)
        current = self.sequence_texts(action)
        if text in current:
            return True
        other = self.action_using(text, skip=action)
        if other is not None:
            if ask and QMessageBox.question(
                    self, "Sneltoets in gebruik",
                    f"{seq.toString(QKeySequence.NativeText)} is nu de sneltoets van "
                    f"'{clean_text(other.text())}'.\n\nWilt u die sneltoets hiernaartoe halen?"
            ) != QMessageBox.Yes:
                return False
            self.change_shortcuts(other, [t for t in self.sequence_texts(other) if t != text])
        self.change_shortcuts(action, current + [text])
        return True

    def remove_shortcut(self, path, sequence=None):
        """Voorkeuren > Sneltoetsen > '−': this shortcut, or all of them, off the command."""
        action = self.command_paths().get(path)
        if action is None:
            return False
        current = self.sequence_texts(action)
        if sequence is None:
            if not current:
                return False
            texts = []
        else:
            text = QKeySequence(sequence).toString(QKeySequence.PortableText)
            if text not in current:
                return False
            texts = [t for t in current if t != text]
        self.change_shortcuts(action, texts)
        return True

    def restore_shortcuts(self):
        """Voorkeuren > Sneltoetsen > Alles herstellen: back to the shortcuts Volura ships with."""
        q = qsettings()
        for key in q.allKeys():
            if key.startswith(SHORTCUTS_KEY):
                q.remove(key)
        q.sync()
        defaults = getattr(self, "_default_shortcuts", {})
        for path, a in self.command_paths().items():
            a.setShortcuts([QKeySequence(t) for t in defaults.get(path, [])])

    # ------------------------------------------------------------ dialogs
    def model_info(self, page=0):
        """Venster > Modelinfo; Bestand > Printer instellen komt op de pagina Printbed uit."""
        dlg = ModelInfoDialog(self, page)
        dlg.exec()
        return dlg

    def print_setup(self):
        """Bestand > Printer instellen: Modelinfo, pagina Printbed (SketchUp)."""
        return self.model_info("Printbed")

    def preferences(self, checked=False):
        """Venster > Voorkeuren: Algemeen, Tekenen en Sneltoetsen."""
        dlg = PreferencesDialog(self)
        dlg.exec()
        return dlg

    def show_welcome(self, checked=False):
        """Help > Welkom bij Volura: the welcome screen."""
        dlg = WelcomeDialog(self)
        dlg.exec()
        return dlg

    def show_welcome_on_startup(self):
        """Show the welcome screen when the user wants it; never in a test (offscreen)."""
        if is_offscreen() or not welcome_enabled():
            return
        self.show_welcome()

    # ------------------------------------------------------------ status
    def set_tool(self, name):
        self.view.set_tool(name)
        if name == "text":
            QTimer.singleShot(0, self.ask_text)

    def on_tool_changed(self, name):
        for n, a in self.tool_actions.items():
            a.setChecked(n == name)
        icons = getattr(self, "_toolbar_tool_icons", {})
        for act, btn in getattr(self, "_dropdown_buttons", {}).items():
            ic = icons.get(act)
            if ic is not None:
                btn.setIcon(ic)

    def on_sheet_tool_changed(self, name):
        for n, a in self.sheet_tool_actions.items():
            a.setChecked(n == name)

    def on_vcb_changed(self, label, value):
        self.vcb_label.setText(label)
        if not self.vcb.hasFocus():
            self.vcb.setText(value)

    def on_vcb_return(self):
        text = self.vcb.text()
        view = self.sheet_view if self.mode == "2d" else self.view
        view.setFocus()
        view.submit_vcb(text)

    def on_doc_changed(self):
        self.update_title()
        self.materials.refresh()
        self.outliner.refresh()
        self.info.refresh()
        if hasattr(self, "layers_panel"):
            self.layers_panel.refresh()
        if self.view.problem_items:
            self.print_panel.clear_marks()      # the marked geometry may be gone now

    def on_selection_changed(self):
        self.outliner.refresh()
        self.info.refresh()

    def update_title(self):
        name = os.path.basename(self.doc.path) if self.doc.path else "Naamloos"
        star = "*" if self.doc.dirty else ""
        self.setWindowTitle(f"{name}{star} - Volura")

    def show_entity_info(self):
        self.tray.show()
        self.tray.raise_()
        self.tray.open_panel("Entiteitsinfo")

    def select_color(self, color):
        self.settings.current_color = color
        if not any(c.lower() == color.lower() for _, c in self.doc.materials):
            self.doc.checkpoint()
            self.doc.materials.append([f"Kleur {len(self.doc.materials) + 1}", color])
            self.doc.notify()
        self.materials.refresh()
        self.view.set_status(f"Huidige kleur: {color}")

    # ------------------------------------------------------------ files
    def confirm_discard(self):
        if not self.doc.dirty or (not self.doc.groups and not self.doc.root.E
                                  and not self.doc.sheet.paths):
            return True
        r = QMessageBox.question(self, "Volura", "Wijzigingen in het huidige model opslaan?",
                                 QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel)
        if r == QMessageBox.Save:
            return self.file_save()
        return r == QMessageBox.Discard

    def file_new(self):
        if not self.confirm_discard():
            return
        self.view.tool.reset()
        self.doc.reset()
        self.doc.notify()
        self.update_title()
        self.view.zoom_extents()

    def file_open(self, path=None):
        if not self.confirm_discard():
            return
        if not path:
            path, _ = QFileDialog.getOpenFileName(self, "Model openen", self.last_dir(),
                                                  "Volura-model (*.volura);;STL (*.stl)")
        if not path:
            return
        if path.lower().endswith(".stl"):
            self.doc.reset()
            self.import_stl(path)
            return
        try:
            self.view.tool.reset()
            fileio.load_document(self.doc, path)
        except Exception as e:
            QMessageBox.critical(self, "Openen mislukt", str(e))
            return
        self.add_recent(path)
        self.doc.notify()
        self.update_title()
        self.view.zoom_extents()

    def file_save(self):
        if not self.doc.path:
            return self.file_save_as()
        try:
            fileio.save_document(self.doc, self.doc.path)
        except Exception as e:
            QMessageBox.critical(self, "Opslaan mislukt", str(e))
            return False
        self.doc.dirty = False
        self.update_title()
        self.hint.setText(f"Opgeslagen: {self.doc.path}")
        return True

    def file_save_as(self):
        path, _ = QFileDialog.getSaveFileName(self, "Opslaan als", self.last_dir("model.volura"),
                                              "Volura-model (*.volura)")
        if not path:
            return False
        if not path.lower().endswith(".volura"):
            path += ".volura"
        self.doc.path = path
        self.add_recent(path)
        return self.file_save()

    def file_save_copy_as(self):
        """Save a copy under another name; the document keeps its own path."""
        name = os.path.basename(self.doc.path) if self.doc.path else "model.volura"
        path, _ = QFileDialog.getSaveFileName(self, "Kopie opslaan als", self.last_dir(name),
                                              "Volura-model (*.volura)")
        if not path:
            return False
        if not path.lower().endswith(".volura"):
            path += ".volura"
        try:
            fileio.save_document(self.doc, path)
        except Exception as e:
            QMessageBox.critical(self, "Kopie opslaan mislukt", str(e))
            return False
        self.add_recent(path)
        self.hint.setText(f"Kopie opgeslagen: {path}")
        return True

    def file_import(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Importeren", self.last_dir(),
            "Alle ondersteunde bestanden (*.stl *.svg *.dxf);;STL (*.stl);;"
            "SVG (*.svg);;DXF (*.dxf)")
        if not path:
            return
        if vector_import.is_vector(path):
            if self.mode == "2d":
                self.import_vector_to_sheet(path)     # op het werkblad blijft het plat
            else:
                self.import_vector_interactive(path)
        else:
            self.import_stl(path)

    def import_stl(self, path):
        try:
            m = fileio.import_stl(path)
        except Exception as e:
            QMessageBox.critical(self, "Importeren mislukt", str(e))
            return
        self.doc.checkpoint()
        g = ops.group_from_manifold(self.doc, m, self.settings.current_color,
                                    os.path.splitext(os.path.basename(path))[0])
        self.doc.active = ROOT
        self.doc.set_selection([("g", g.id, g.id)])
        self.doc.notify()
        self.view.zoom_extents()

    # ------------------------------------------------------------ vector import
    def ask_vector_options(self, drawing, name="Vector"):
        """Dialoog 'Vectorbestand importeren'; None bij annuleren, anders een opties-dict."""
        b = drawing.bounds()
        w = (b[2] - b[0]) if b else 0.0
        h = (b[3] - b[1]) if b else 0.0
        dlg = QDialog(self)
        dlg.setWindowTitle("Vectorbestand importeren")
        form = QFormLayout(dlg)
        form.addRow("Afmetingen", QLabel(f"{w:.1f} x {h:.1f} mm"))
        scale = QDoubleSpinBox()
        scale.setRange(0.001, 10000.0)
        scale.setDecimals(4)
        scale.setValue(1.0)
        scale.setSuffix(" x")
        width = QDoubleSpinBox()
        width.setRange(0.01, 100000.0)
        width.setDecimals(3)
        width.setSuffix(" mm")
        width.setValue(w if w > 0 else 1.0)
        width.setEnabled(w > 0)
        anchor = QComboBox()
        anchor.addItem("Midden", "center")
        anchor.addItem("Linksonder", "lower_left")
        open_edges = QCheckBox("Open lijnen als randen importeren")
        open_edges.setChecked(True)
        extrude = QCheckBox("Extruderen tot volume")
        height = QDoubleSpinBox()
        height.setRange(-1000.0, 1000.0)
        height.setDecimals(3)
        height.setValue(3.0)
        height.setSuffix(" mm")
        height.setEnabled(False)
        color_state = {"color": self.settings.current_color}
        cbtn = QPushButton()
        cbtn.setIcon(swatch(color_state["color"], 20))
        cbtn.setText(color_state["color"])

        def choose_color():
            c = QColorDialog.getColor(QColor(color_state["color"]), dlg)
            if c.isValid():
                color_state["color"] = c.name()
                cbtn.setIcon(swatch(c.name(), 20))
                cbtn.setText(c.name())
        cbtn.clicked.connect(choose_color)

        updating = {"lock": False}

        def from_scale(val):
            if updating["lock"] or w <= 0:
                return
            updating["lock"] = True
            width.setValue(max(width.minimum(), w * val))
            updating["lock"] = False

        def from_width(val):
            if updating["lock"] or w <= 0:
                return
            updating["lock"] = True
            scale.setValue(max(scale.minimum(), val / w))
            updating["lock"] = False

        scale.valueChanged.connect(from_scale)
        width.valueChanged.connect(from_width)
        extrude.toggled.connect(height.setEnabled)

        form.addRow("Schaal", scale)
        form.addRow("Doelbreedte", width)
        form.addRow("Ankerpunt", anchor)
        form.addRow("", open_edges)
        form.addRow("", extrude)
        form.addRow("Hoogte", height)
        form.addRow("Kleur", cbtn)
        if drawing.warnings:
            form.addRow(QLabel("\n".join(drawing.warnings)))
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(dlg.accept)
        bb.rejected.connect(dlg.reject)
        form.addRow(bb)
        if dlg.exec() != QDialog.Accepted:
            return None
        return {"scale": scale.value(),
                "target_width": (width.value() if w > 0 else None),
                "anchor": anchor.currentData(),
                "open_edges": open_edges.isChecked(),
                "extrude": extrude.isChecked(),
                "height": height.value(),
                "color": color_state["color"],
                "name": name}

    def place_vector(self, drawing, options, plane=None, point=None):
        """Plaats een tekening direct (zonder dialoog of gereedschap) als één undo-stap."""
        plane = plane or Plane((0, 0, 0), (0, 0, 1))
        point = np.zeros(3) if point is None else np.asarray(point, dtype=float)
        regios, open_paden = vector_import.prepare(
            drawing, scale=options.get("scale", 1.0),
            target_width=options.get("target_width"),
            anchor=options.get("anchor", "center"))
        if not regios and not open_paden:
            self.view.set_status("Er viel niets te importeren uit het vectorbestand.")
            return None
        if not options.get("open_edges", True):
            open_paden = []
        self.doc.checkpoint()
        offset = plane.to_local(point)
        group = None
        if options.get("extrude"):
            group = vector_import.place_extruded(
                self.doc, plane, regios, offset, options.get("height", 3.0),
                options.get("color", self.settings.current_color),
                options.get("name", "Vector"))
            if options.get("open_edges", True) and open_paden:
                vector_import.place_faces(self.doc.active_ents(), plane, [], open_paden, offset)
        else:
            vector_import.place_faces(self.doc.active_ents(), plane, regios, open_paden, offset)
        self.doc.active = ROOT
        if group is not None:
            self.doc.set_selection([("g", group.id, group.id)])
        self.doc.notify()
        self.view.zoom_extents()
        return group

    def import_vector(self, path, options=None):
        """Importeer een SVG/DXF naar de grond; de dialoog is over te slaan met `options`."""
        try:
            drawing = vector_import.load_vector(path)
        except Exception as e:
            QMessageBox.critical(self, "Importeren mislukt", str(e))
            return None
        if options is None:
            options = self.ask_vector_options(drawing,
                                              os.path.splitext(os.path.basename(path))[0])
            if options is None:
                return None
        return self.place_vector(drawing, options)

    def import_vector_interactive(self, path, drawing=None, name=None):
        """Kies opties en plaats de tekening met het plaatsingsgereedschap achter de cursor."""
        if drawing is None:
            try:
                drawing = vector_import.load_vector(path)
            except Exception as e:
                QMessageBox.critical(self, "Importeren mislukt", str(e))
                return
        if name is None:
            name = os.path.splitext(os.path.basename(path))[0] if path else "Vector"
        options = self.ask_vector_options(drawing, name)
        if options is None:
            return
        regios, open_paden = vector_import.prepare(
            drawing, scale=options["scale"], target_width=options.get("target_width"),
            anchor=options.get("anchor", "center"))
        if not regios and not open_paden:
            self.view.set_status("Er viel niets te importeren uit het vectorbestand.")
            return
        options["name"] = name
        self.view.tools["vectorplace"].set_drawing(regios, open_paden, options)
        self.set_tool("vectorplace")

    def add_drawing_to_sheet(self, drawing, message="contouren aan het werkblad toegevoegd"):
        """Een tekening als één undo-stap op het werkblad zetten en de nieuwe paden kiezen."""
        if drawing is None or not drawing.paths:
            self.active_view().set_status("Er viel niets naar het werkblad toe te voegen.")
            self.hint.setText("Er is niets toegevoegd.")
            return None
        self.doc.checkpoint()
        added = self.doc.sheet.merge(drawing)
        self.doc.notify()
        self.set_mode("2d")
        self.sheet_view.set_selection(added)
        self.sheet_view.zoom_extents()
        self.hint.setText(f"{len(added)} {message}")
        return added

    def import_vector_to_sheet(self, path, drawing=None):
        """Bestand > Importeren in 2D: de SVG of DXF op het werkblad, op zijn eigen plek.

        De posities uit het bestand blijven zoals ze zijn en lagen met dezelfde naam
        worden samengevoegd; de nieuwe paden zijn meteen geselecteerd.
        """
        name = os.path.splitext(os.path.basename(path))[0] if path else "Vector"
        if drawing is None:
            try:
                drawing = vector_import.load_vector(path)
            except Exception as e:
                QMessageBox.critical(self, "Importeren mislukt", str(e))
                return None
        added = self.add_drawing_to_sheet(drawing, f"contouren uit {name} toegevoegd")
        if added is not None and drawing.warnings:
            self.hint.setText(f"{len(added)} contouren uit {name} toegevoegd "
                              f"({'; '.join(drawing.warnings)})")
        return added

    def copy_to_sheet(self, options=None):
        """Bewerken > Naar 2D-werkblad kopiëren: dezelfde keuzes als de 2D-export."""
        opts = self.vector_export_options() if options is None else dict(options)
        if opts is None:
            return None
        try:
            drawing = self.vector_export_drawing(opts)
        except Exception as e:
            QMessageBox.critical(self, "Kopiëren mislukt", str(e))
            return None
        if not drawing.paths:
            msg = self.vector_export_empty_message(opts, drawing)
            if drawing.warnings:
                msg += "\n\n" + "\n".join(drawing.warnings)
            QMessageBox.information(self, "Naar 2D-werkblad kopiëren", msg)
            self.hint.setText("Er is niets gekopieerd.")
            return None
        return self.add_drawing_to_sheet(drawing, "contouren naar het werkblad gekopieerd")

    # ------------------------------------------------------------ 3D en werkblad samen
    def _laser_layers(self):
        """De lagen van het werkblad waar een bewerking aan hangt (niet de hulplijnen)."""
        return [(n, lay) for n, lay in self.doc.sheet.layers.items()
                if lay.op != "none" and not lay.locked]

    def _add_laser_menu(self, menu):
        """Rechtsklik in 3D: "Laseren / frezen" met een regel per laag van het werkblad."""
        sm = menu.addMenu(icon("to_laser"), "Laseren / frezen")
        face = getattr(self, "_context_face", None)
        for name, lay in self._laser_layers():
            a = sm.addAction(swatch(lay.color or "#000000", 14),
                             f"{name}   —   {OP_LABELS.get(lay.op, lay.op)}")
            a.triggered.connect(lambda checked=False, n=name, f=face:
                                self.send_selection_to_sheet(n, f))
        return sm

    def send_selection_to_sheet(self, layer=None, face=None):
        """De selectie in 3D als bewerking op het werkblad: vlakken op ware grootte,
        geselecteerde groepen (platen) als bovenaanzicht.

        `face` is het vlak (ctx, fid) waarop met rechts geklikt is, ook als dat in een
        groep ligt: dan gaat dat vlak, niet het bovenaanzicht van de hele groep (tenzij
        het bij een selectie van meer vlakken hoort). Zonder `layer` wordt de laag gevraagd. De contouren komen rechts naast wat er al
        op het werkblad staat. Is er nog geen plaatdikte, dan neemt het werkblad de
        dikte van de geselecteerde platen over als die allemaal even dik zijn.
        """
        if layer is None or isinstance(layer, bool):
            names = [f"{n}   —   {OP_LABELS.get(l.op, l.op)}" for n, l in self._laser_layers()]
            if not names:
                self.hint.setText("Het werkblad heeft geen laag met een bewerking.")
                return None
            choice, ok = QInputDialog.getItem(self, "Naar laser of frees",
                                              "Op welke laag (bewerking)?", names, 0, False)
            if not ok:
                return None
            layer = self._laser_layers()[names.index(choice)][0]
        groups = self.doc.selected_groups()
        faces = self._selected_faces()
        if face is not None and tuple(face) not in faces:
            faces = [tuple(face)]
        try:
            if faces:
                drawing = vector_export.drawing_from_faces(
                    self.doc, [("f", c, i) for c, i in faces])
            elif groups:
                drawing = vector_export.drawing_top(self.doc, True)
            else:
                drawing = None
        except Exception as e:
            QMessageBox.critical(self, "Naar laser of frees", str(e))
            return None
        if drawing is None or not drawing.paths:
            self.hint.setText("Selecteer eerst vlakken of platen (groepen) in het model.")
            return None
        sheet = self.doc.sheet
        lay = sheet.layers.get(layer)
        drawing = sheet3d.on_layer(drawing, layer, lay.color if lay else None,
                                   lay.op if lay else None)
        self._place_beside_sheet(drawing)
        if faces:
            thickness = self._face_plate_thickness(faces)
        else:
            thickness = sheet3d.solid_thickness(groups) if groups else None
        added = self.add_drawing_to_sheet(drawing, f"contouren op laag {layer} gezet")
        if added and thickness and not sheet.sheet_thickness:
            sheet.sheet_thickness = thickness
            self.hint.setText(f"{len(added)} contouren op laag {layer} gezet; "
                              f"plaatdikte {thickness:g} mm overgenomen")
        return added

    def _face_plate_thickness(self, faces):
        """De plaatdikte bij liggende vlakken in een groep: de hoogte van die groep."""
        groups = []
        for c, i in faces:
            g = next((g for g in self.doc.groups if g.id == c), None)
            if g is None or abs(g.ents.F[i].normal[2]) < 0.999:
                return None
            groups.append(g)
        return sheet3d.solid_thickness(groups)

    def _place_beside_sheet(self, drawing):
        """Schuif een nieuwe tekening rechts naast de inhoud van het werkblad (of in de
        hoek van de plaat als die nog leeg is), zodat niets over elkaar komt."""
        pts = [p.pts for p in drawing.paths if len(p.pts)]
        if not pts:
            return
        P = np.vstack(pts)
        lo = P.min(axis=0)
        sheet = self.doc.sheet
        have = [p.pts for p in sheet.paths if len(p.pts)]
        if have:
            Q = np.vstack(have)
            target = np.array([Q[:, 0].max() + float(sheet.sheet_gap) + 5.0, Q[:, 1].min()])
        else:
            m = float(sheet.sheet_margin) if sheet.sheet_w else 0.0
            target = np.array([m, m])
        shift = target - lo
        for p in drawing.paths:
            p.pts = np.ascontiguousarray(p.pts + shift)

    def sheet_to_model(self, checked=False, thickness=None):
        """Werkblad > Onderdelen naar 3D-model: alles wat gesneden wordt als plaat.

        Elk onderdeel wordt een eigen groep met de plaatdikte, op dezelfde X/Y als op
        het werkblad. Een eerdere keer naar het model gezette onderdelen worden
        vervangen; met een selectie op het werkblad gaan alleen die onderdelen.
        """
        sheet = self.doc.sheet
        t = thickness or sheet.sheet_thickness
        if not t:
            t, ok = QInputDialog.getDouble(self, "Plaatdikte", "Dikte van de plaat (mm):",
                                           sheet3d.DEFAULT_THICKNESS, 0.1, 500.0, 2)
            if not ok:
                return None
        selected = self.sheet_view.selected_paths() if self.mode == "2d" else []
        try:
            solids = sheet3d.part_solids(sheet, t, selected or None)
        except Exception as e:
            QMessageBox.critical(self, "Onderdelen naar 3D-model", str(e))
            return None
        if not solids:
            self.hint.setText("Er staan geen gesloten snijcontouren op het werkblad "
                              "(laag Snijden, Frezen of Boren).")
            return None
        self.doc.checkpoint()
        if not sheet.sheet_thickness:
            sheet.sheet_thickness = float(t)
        live = {g.id for g in self.doc.groups}
        if not selected:
            for gid in [g for g in getattr(self.doc, "sheet_part_groups", ()) if g in live]:
                self.doc.remove_group(gid)
            live = {g.id for g in self.doc.groups}
        groups = [ops.group_from_manifold(self.doc, man, self.settings.current_color, name)
                  for name, man in solids]
        self.doc.sheet_part_groups = ([] if not selected else
                                      [g for g in getattr(self.doc, "sheet_part_groups", ())
                                       if g in live]) + [g.id for g in groups]
        self.doc.set_selection([("g", g.id, g.id) for g in groups])
        self.doc.notify()
        self.set_mode("3d")
        self.view.zoom_extents()
        self.hint.setText(f"{len(groups)} onderdel{'en' if len(groups) != 1 else ''} van "
                          f"{float(t):g} mm in het model gezet")
        return groups

    # ------------------------------------------------------------ export voor laser en frees
    def sheet_layers_for_export(self) -> list:
        """De lagen van het werkblad die iets voor de machine betekenen."""
        return [n for n, lay in self.doc.sheet.layers.items()
                if lay.visible and lay.op != "none"]

    def default_sheet_export_options(self) -> dict:
        """De standaardkeuzes: alle zichtbare werklagen, correctie alleen bij frezen."""
        names = self.sheet_layers_for_export()
        mill = any(self.doc.sheet.layers[n].op in MILL_OPS
                   and any(p.layer == n for p in self.doc.sheet.paths) for n in names)
        return {"layers": names, "compensate": mill, "per_layer": False, "per_sheet": False}

    def sheet_export_options(self, kind="svg"):
        """Dialoog "Exporteren voor laser/frees"; None bij annuleren, anders een opties-dict."""
        dlg = SheetExportDialog(self, kind)
        if dlg.exec() != QDialog.Accepted:
            return None
        return dlg.options()

    def sheet_export_drawing(self, options) -> "Drawing":
        """De tekening van de gekozen lagen, met de correctie erbij als die aan staat."""
        sheet = self.doc.sheet
        names = [n for n in (options.get("layers") or ()) if n in sheet.layers]
        out = Drawing(layers={n: replace(sheet.layers[n], params=dict(sheet.layers[n].params))
                              for n in names},
                      **sheet._sheet_kwargs())
        for p in sheet.paths:
            if p.layer in names:
                out.paths.append(VPath(np.asarray(p.pts, dtype=float).copy(), p.closed, p.layer,
                                       p.filled, p.rule, p.src, dict(p.meta)))
        for n in names:
            out.ensure_layer(n)
        if options.get("compensate"):
            out = cam.compensate(out)
        return out

    @staticmethod
    def layer_file_name(path, name, kind) -> str:
        """`bestandsnaam_laagnaam.svg`: de laagnaam netjes voor in een bestandsnaam."""
        base = os.path.splitext(path)[0]
        safe = "".join(ch for ch in str(name) if ch not in '<>:"/\\|?*\n').strip(" .") or "laag"
        return f"{base}_{safe}.{kind}"

    @staticmethod
    def sheet_file_name(path, index, kind) -> str:
        """`bestandsnaam_plaat1.svg`: het bestand van één plaat."""
        base = os.path.splitext(path)[0]
        return f"{base}_plaat{int(index)}.{kind}"

    def export_sheet(self, kind="svg", path=None, options=None):
        """Exporteren > 2D-vector op het werkblad: "Exporteren voor laser/frees".

        Zonder `path` vraagt de dialoog welke lagen er weg mogen, of de kerf- en
        freescorrectie toegepast moet worden en of elke laag een eigen bestand krijgt;
        met een `path` gaan die keuzes standaard. Rasterlagen gaan gevuld weg, de rest
        als haarlijn, en de bewerking en de instellingen van elke laag komen in de SVG
        zodat Volura ze bij inlezen terugvindt.
        """
        kind = "dxf" if str(kind).lower() == "dxf" else "svg"
        if options is None:
            options = (self.sheet_export_options(kind) if path is None
                       else self.default_sheet_export_options())
            if options is None:
                return None
        names = [n for n in (options.get("layers") or ()) if n in self.doc.sheet.layers]
        if not names:
            QMessageBox.information(self, "Exporteren voor laser/frees",
                                    "Kies minstens één laag met een bewerking. Een laag op "
                                    "\"Geen\" of een verborgen laag gaat niet mee.")
            self.hint.setText("Er is niets geëxporteerd.")
            return None
        try:
            drawing = self.sheet_export_drawing(options)
        except Exception as e:
            QMessageBox.critical(self, "Exporteren mislukt", str(e))
            return None
        if not drawing.paths:
            QMessageBox.information(self, "Exporteren voor laser/frees",
                                    "Er staan geen contouren op de gekozen lagen. Teken iets, "
                                    "of importeer een SVG- of DXF-bestand.")
            self.hint.setText("Er is niets geëxporteerd.")
            return None
        if path is None:
            path, _ = QFileDialog.getSaveFileName(
                self, "Exporteren voor laser/frees", self.export_name("." + kind),
                "SVG (*.svg)" if kind == "svg" else "DXF (*.dxf)")
            if not path:
                return None
            if not os.path.splitext(path)[1]:
                path += "." + kind
        filled = [n for n in names
                  if drawing.layers.get(n) is not None and drawing.layers[n].op == "raster"]
        per_sheet = bool(options.get("per_sheet")) and bool(self.doc.sheet.sheet_w) \
            and bool(self.doc.sheet.sheet_h)
        files = []
        try:
            if per_sheet:
                n = max(1, int(self.doc.sheet.sheet_count or 1))
                for si in range(n):
                    sub = nesting.sheet_subdrawing(drawing, si)
                    target = self.sheet_file_name(path, si + 1, kind)
                    if options.get("per_layer"):
                        for nm in names:
                            one = self.layer_file_name(target, nm, kind)
                            self._write_vector(sub, one, kind, [nm],
                                               [nm] if nm in filled else [])
                            files.append(one)
                    else:
                        self._write_vector(sub, target, kind, names, filled)
                        files.append(target)
            elif options.get("per_layer"):
                for n in names:
                    one = self.layer_file_name(path, n, kind)
                    self._write_vector(drawing, one, kind, [n], [n] if n in filled else [])
                    files.append(one)
            else:
                self._write_vector(drawing, path, kind, names, filled)
                files.append(path)
        except Exception as e:
            QMessageBox.critical(self, "Exporteren mislukt", str(e))
            return None
        where = files[0] if len(files) == 1 else f"{len(files)} bestanden ({files[0]})"
        self.hint.setText(f"{len(drawing.paths)} contouren geëxporteerd naar {where}"
                          + (" met kerf-/freescorrectie" if options.get("compensate") else ""))
        return files[0]

    @staticmethod
    def _write_vector(drawing, path, kind, layers, fill_layers=()):
        """Eén tekening naar een SVG- of DXF-bestand schrijven, met de gekozen lagen."""
        if kind == "dxf":
            dxf.write_dxf(drawing, path, layers=layers)
        else:
            svg.write_svg(drawing, path, layers=layers, fill_layers=fill_layers)

    def import_dropped_file(self, path):
        """Een gesleept bestand: .stl of .svg/.dxf importeren."""
        ext = os.path.splitext(path)[1].lower()
        if vector_import.is_vector(path):
            if self.mode == "2d":
                self.import_vector_to_sheet(path)
            else:
                self.import_vector_interactive(path)
        elif ext == ".stl":
            self.import_stl(path)

    def dragEnterEvent(self, e):
        md = e.mimeData()
        if md.hasUrls() and any(vector_import.is_vector(u.toLocalFile()) or
                                u.toLocalFile().lower().endswith(".stl") for u in md.urls()):
            e.acceptProposedAction()

    def dragMoveEvent(self, e):
        self.dragEnterEvent(e)

    def dropEvent(self, e):
        md = e.mimeData()
        if not md.hasUrls():
            return
        paths = [u.toLocalFile() for u in md.urls() if u.isLocalFile()]
        supported = [p for p in paths if vector_import.is_vector(p) or
                     p.lower().endswith(".stl")]
        if not supported:
            return
        e.acceptProposedAction()
        for p in supported:
            self.import_dropped_file(p)

    def last_dir(self, name=""):
        base = os.path.dirname(self.doc.path) if self.doc.path else (
            os.path.dirname(self.settings.recent[0]) if self.settings.recent else os.path.expanduser("~"))
        return os.path.join(base, name) if name else base

    def add_recent(self, path):
        r = [p for p in self.settings.recent if p != path]
        self.settings.recent = [path] + r
        self.settings.save()
        self.update_recent_menu()

    def update_recent_menu(self):
        self.recent_menu.clear()
        for p in self.settings.recent[:8]:
            a = self.recent_menu.addAction(os.path.basename(p))
            a.setToolTip(p)
            a.triggered.connect(lambda checked=False, p=p: self.file_open(p))
        self.recent_menu.setEnabled(bool(self.settings.recent))

    def export_objects(self):
        """Printable parts for export; refuses when something is not a closed volume."""
        only = bool(self.doc.selected_groups())
        part_list = ops.parts(self.doc, only_selected=only)
        if not part_list:
            QMessageBox.information(self, "Exporteren", "Er zijn geen volumes om te exporteren. "
                                    "Teken een vlak en gebruik Duwen/trekken (P).")
            return None
        bad = [p for p in part_list if p.problem]
        if bad:
            text = "\n".join(f"• {p.name}: {p.problem}" for p in bad)
            if not [p for p in part_list if p.manifold is not None]:
                self.ask_export("Geen enkel onderdeel is een gesloten volume:\n\n" + text)
                return None
            if not self.ask_export("Deze onderdelen zijn niet gesloten en worden overgeslagen:\n\n"
                                   + text + "\n\nToch exporteren?"):
                return None
        return part_list

    def ask_export(self, text):
        """The export warning, with a button that opens the Printcontrole panel."""
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Warning)
        box.setWindowTitle("Exporteren")
        box.setText(text)
        b_open = box.addButton("Printcontrole openen", QMessageBox.ActionRole)
        b_no = box.addButton(QMessageBox.No)
        b_yes = box.addButton(QMessageBox.Yes)
        box.setDefaultButton(b_no)
        box.exec()
        clicked = box.clickedButton()
        if clicked is b_open:
            self.open_print_check()
            return False
        return clicked is b_yes

    def export_name(self, ext):
        base = os.path.splitext(self.doc.path)[0] if self.doc.path else os.path.join(self.last_dir(), "model")
        return base + ext

    def export_stl(self):
        objs = self.export_objects()
        if not objs:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Exporteren als STL", self.export_name(".stl"), "STL (*.stl)")
        if not path:
            return
        try:
            n = fileio.export_stl(path, objs)
        except Exception as e:
            QMessageBox.critical(self, "Export mislukt", str(e))
            return
        self.hint.setText(f"STL geëxporteerd ({n} driehoeken, waterdicht): {path}")

    def export_3mf(self):
        objs = self.export_objects()
        if not objs:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Exporteren als 3MF", self.export_name(".3mf"), "3MF (*.3mf)")
        if not path:
            return
        try:
            used = fileio.export_3mf(path, objs, self.doc.materials,
                                     os.path.splitext(os.path.basename(path))[0])
        except Exception as e:
            QMessageBox.critical(self, "Export mislukt", str(e))
            return
        lines = "\n".join(f"  extruder {e}: {n}" for n, e in used)
        QMessageBox.information(self, "3MF geëxporteerd",
                                f"{path}\n\nOnderdelen per filament:\n{lines}\n\n"
                                "Open het bestand in PrusaSlicer, OrcaSlicer of Bambu Studio; "
                                "elk kleurdeel staat al op de juiste extruder.")

    def export_stl_colors(self):
        objs = self.export_objects()
        if not objs:
            return
        path, _ = QFileDialog.getSaveFileName(self, "STL per kleur (basisnaam)", self.export_name(".stl"), "STL (*.stl)")
        if not path:
            return
        try:
            files = fileio.export_stl_per_color(path, objs, self.doc.materials)
        except Exception as e:
            QMessageBox.critical(self, "Export mislukt", str(e))
            return
        QMessageBox.information(self, "STL per kleur", "Geschreven:\n" + "\n".join(files) +
                                "\n\nLaad ze samen in je slicer als één object met meerdere onderdelen.")

    # ------------------------------------------------------------ 2D-vector export
    def vector_export_options(self):
        """Dialoog '2D-vector exporteren'; None bij annuleren, anders een opties-dict."""
        dlg = VectorExportDialog(self)
        if dlg.exec() != QDialog.Accepted:
            return None
        return dlg.options()

    def vector_export_drawing(self, options):
        """De tekening bij deze keuzes: geselecteerde vlakken, bovenaanzicht of doorsnede."""
        doc = self.doc
        source = options.get("source", "top")
        if source == "faces":
            return vector_export.drawing_from_faces(doc)
        only = bool(options.get("only_selected"))
        if source == "section":
            return vector_export.drawing_section(doc, float(options.get("z", 0.0)), only)
        return vector_export.drawing_top(doc, only)

    def vector_export_empty_message(self, options, drawing):
        """Waarom er niets weg te schrijven is, in woorden die de gebruiker helpt."""
        source = options.get("source", "top")
        if source == "faces":
            return ("Er is geen vlak geselecteerd. Klik een vlak aan (of zet er een "
                    "rechthoek neer) en probeer het opnieuw.")
        only = bool(options.get("only_selected"))
        if source == "section" and ops.parts(self.doc, only_selected=only):
            return (f"Op hoogte {format_length(float(options.get('z', 0.0)))} snijdt het model "
                    "niets: er is daar geen volume.")
        return ("Er zijn geen gesloten volumes om te exporteren. Teken een vlak en gebruik "
                "Duwen/trekken (P), of maak de groep dicht met Printcontrole.")

    def export_vector(self, kind="svg", path=None, options=None):
        """Bestand > Exporteren > 2D-vector: de tekening naar SVG of DXF schrijven.

        Zonder `options` vraagt de dialoog wat er weg moet; zonder `path` vraagt Qt
        een bestandsnaam. Op het 2D-werkblad gaat de dialoog over het werkblad zelf
        en worden alleen de zichtbare lagen weggeschreven. Geeft het geschreven pad
        terug, of None.
        """
        kind = "dxf" if str(kind).lower() == "dxf" else "svg"
        if self.mode == "2d":
            return self.export_sheet(kind, path)
        opts = self.vector_export_options() if options is None else dict(options)
        if opts is None:
            return None
        try:
            drawing = self.vector_export_drawing(opts)
        except Exception as e:
            QMessageBox.critical(self, "Exporteren mislukt", str(e))
            return None
        if not drawing.paths:
            msg = self.vector_export_empty_message(opts, drawing)
            if drawing.warnings:
                msg += "\n\n" + "\n".join(drawing.warnings)
            QMessageBox.information(self, "2D-vector exporteren", msg)
            self.hint.setText("Er is niets geëxporteerd.")
            return None
        if path is None:
            path, _ = QFileDialog.getSaveFileName(
                self, "2D-vector exporteren", self.export_name("." + kind),
                "SVG (*.svg)" if kind == "svg" else "DXF (*.dxf)")
            if not path:
                return None
            if not os.path.splitext(path)[1]:
                path += "." + kind
        try:
            (dxf.write_dxf if kind == "dxf" else svg.write_svg)(drawing, path)
        except Exception as e:
            QMessageBox.critical(self, "Exporteren mislukt", str(e))
            return None
        self.hint.setText(f"{len(drawing.paths)} contouren geëxporteerd naar {path}")
        return path

    # ------------------------------------------------------------ edit
    def undo(self):
        self.view.tool.reset()
        self.sheet_view.tool.reset()
        self.view.preview_transform = None
        self.doc.undo()

    def redo(self):
        self.view.tool.reset()
        self.sheet_view.tool.reset()
        self.doc.redo()

    def warn_locked(self):
        """Bewerken > Vergrendelen: say in the status bar why nothing happened."""
        self.view.set_status(LOCKED_MSG)

    def unlocked(self, groups):
        """The groups that may be changed; warns about the locked ones."""
        locked = [g for g in groups if g.locked]
        if locked:
            self.warn_locked()
        return [g for g in groups if not g.locked]

    def delete(self):
        sel = list(self.doc.selection)
        if not sel:
            return
        locked = {g.id for g in self.doc.selected_groups() if g.locked}
        if locked:
            self.warn_locked()
            sel = [it for it in sel if not (it[0] == "g" and it[2] in locked)]
            if not sel:
                return
        self.doc.checkpoint()
        for k, c, i in sel:
            if k == "g":
                self.doc.remove_group(i)
            elif k == "guide":
                self.doc.remove_guides([i])
        by_ctx = {}
        for k, c, i in sel:
            if k in ("f", "e"):
                by_ctx.setdefault(c, ([], []))[0 if k == "f" else 1].append(i)
        for c, (fs, es) in by_ctx.items():
            ents = self.doc.ents(c)
            if ents is None:
                continue
            ents.erase_faces(fs)
            ents.erase_edges(es)
        self.doc.selection = []
        self.doc.notify()

    def select_all(self):
        ctx = self.doc.active
        ents = self.doc.active_ents()
        if self.settings.show_hidden:
            faces, edges = ents.F, ents.E
        else:
            faces, edges = ents.visible_faces(), ents.visible_edges()
        items = [("f", ctx, f) for f in faces] + [("e", ctx, e) for e in edges]
        if ctx == ROOT:
            items += [("g", g.id, g.id) for g in self.doc.visible_groups()]
        self.doc.set_selection(items)

    def select_none(self):
        self.doc.set_selection([])

    def erase_guides(self):
        """Bewerken > Hulplijnen wissen: every guide of the document, in one undo step."""
        if not self.doc.guides:
            return
        self.doc.checkpoint()
        self.doc.clear_guides()
        self.doc.notify()

    # ------------------------------------------------------------ hide / lock
    def _hidden_items(self, items):
        """The subset of the given selection items that is now hidden."""
        out = []
        for k, ctx, i in items:
            if k == "g":
                g = self.doc.group(i)
                if g is not None and not g.visible:
                    out.append(("g", ctx, i))
                continue
            ents = self.doc.ents(ctx)
            if ents is not None and i in (ents.hidden_f if k == "f" else ents.hidden_e):
                out.append((k, ctx, i))
        return out

    def _show_items(self, items):
        """Make the given items visible again (groups through `visible`, the rest through hidden_*)."""
        by_ctx = {}
        for k, ctx, i in items:
            if k == "g":
                g = self.doc.group(i)
                if g is not None:
                    g.visible = True
                continue
            by_ctx.setdefault(ctx, ([], []))[0 if k == "f" else 1].append(i)
        for ctx, (fs, es) in by_ctx.items():
            ents = self.doc.ents(ctx)
            if ents is not None:
                ents.unhide(fs, es)

    def hide_selected(self):
        """Bewerken > Verbergen: hide the selection and remember what it was."""
        items = [("g", g.id, g.id) for g in self.doc.selected_groups() if g.visible]
        items += [it for it in self.doc.selection if it[0] in ("f", "e")]
        if not items:
            return
        self.doc.checkpoint()
        by_ctx = {}
        for k, ctx, i in items:
            if k == "g":
                self.doc.group(i).visible = False
            else:
                by_ctx.setdefault(ctx, ([], []))[0 if k == "f" else 1].append(i)
        for ctx, (fs, es) in by_ctx.items():
            ents = self.doc.ents(ctx)
            if ents is not None:
                ents.hide(fs, es)
        self.doc.last_hidden = items
        self.doc.selection = []
        self.doc.notify()

    def unhide_selected(self):
        """Bewerken > Zichtbaar maken > Geselecteerd (useful with Verborgen geometrie on)."""
        items = self._hidden_items(self.doc.selection)
        if not items:
            return
        self.doc.checkpoint()
        self._show_items(items)
        self.doc.notify()

    def unhide_last(self):
        """Bewerken > Zichtbaar maken > Laatste: what Bewerken > Verbergen hid last."""
        items = self._hidden_items(self.doc.last_hidden)
        if not items:
            return
        self.doc.checkpoint()
        self._show_items(items)
        self.doc.last_hidden = []
        self.doc.notify()

    def unhide_all(self):
        """Bewerken > Zichtbaar maken > Alles: every hidden face, edge and group."""
        hidden = any(not g.visible for g in self.doc.groups) or \
            any(e.hidden_f or e.hidden_e for e in self.doc.all_ents())
        if not hidden:
            return
        self.doc.checkpoint()
        for g in self.doc.groups:
            g.visible = True
        for ents in self.doc.all_ents():
            ents.unhide_all()
        self.doc.last_hidden = []
        self.doc.notify()

    def lock_selected(self):
        """Bewerken > Vergrendelen: the selected groups can no longer be changed."""
        groups = [g for g in self.doc.selected_groups() if not g.locked]
        if not groups:
            return
        self.doc.checkpoint()
        for g in groups:
            g.locked = True
        self.doc.notify()

    def unlock_selected(self):
        """Bewerken > Ontgrendelen > Geselecteerd."""
        groups = [g for g in self.doc.selected_groups() if g.locked]
        if not groups:
            return
        self.doc.checkpoint()
        for g in groups:
            g.locked = False
        self.doc.notify()

    def unlock_all(self):
        """Bewerken > Ontgrendelen > Alles."""
        groups = [g for g in self.doc.groups if g.locked]
        if not groups:
            return
        self.doc.checkpoint()
        for g in groups:
            g.locked = False
        self.doc.notify()

    def copy(self):
        self.clipboard = list(self.doc.selection)
        self.hint.setText(f"{len(self.clipboard)} item(s) gekopieerd.")

    def cut(self):
        if not self.doc.selection:
            return
        self.copy()
        self.delete()

    def paste(self):
        if self._paste_svg_from_clipboard():
            return
        self._paste(np.array([10.0, 10.0, 0.0]), move_tool=True)

    def _paste_svg_from_clipboard(self):
        """Bewerken > Plakken: SVG op het klembord (bijv. uit Inkscape) importeren."""
        md = QApplication.clipboard().mimeData()
        if md is None or not md.hasFormat("image/svg+xml"):
            return False
        raw = bytes(md.data("image/svg+xml"))
        try:
            drawing = vector_import.load_svg_text(raw.decode("utf-8", "replace"))
        except Exception as e:
            QMessageBox.critical(self, "Plakken mislukt", str(e))
            return True
        self.import_vector_interactive(None, drawing=drawing, name="Geplakte SVG")
        return True

    def paste_at_location(self):
        self._paste(np.zeros(3), move_tool=False)

    def _paste(self, offset, move_tool):
        self.doc.validate_selection()
        items = [it for it in self.clipboard if it[0] == "g" and self.doc.group(it[1])] + \
                [it for it in self.clipboard if it[0] != "g" and self.doc.ents(it[1]) is not None
                 and (it[2] in self.doc.ents(it[1]).F or it[2] in self.doc.ents(it[1]).E)]
        if not items:
            return
        self.doc.checkpoint()
        new = ops.transform_selection(self.doc, np.eye(3), offset, copy=True, items=items)
        self.doc.set_selection(new)
        self.doc.notify()
        if move_tool:
            self.set_tool("move")

    def make_group(self):
        if not any(k in ("f", "e") for k, _, _ in self.doc.selection):
            return
        self.doc.checkpoint()
        g = ops.make_group(self.doc)
        self.doc.active = ROOT
        self.doc.set_selection([("g", g.id, g.id)] if g else [])
        self.doc.notify()

    def explode_group(self):
        groups = self.unlocked(self.doc.selected_groups())
        if not groups:
            return
        self.doc.checkpoint()
        for g in groups:
            ops.explode(self.doc, g)
        self.doc.selection = []
        self.doc.notify()

    def edit_group(self):
        groups = self.doc.selected_groups()
        for g in groups:
            if self.doc.enter_group(g.id):
                return
        if groups:
            self.warn_locked()      # a locked group does not open

    def close_group(self):
        self.doc.exit_group()

    # ------------------------------------------------------------ view
    def on_cam_history(self, can_prev, can_next):
        self.a_prev_view.setEnabled(can_prev)
        self.a_next_view.setEnabled(can_next)

    def camera_prev(self, checked=False):
        self.view.prev_camera()

    def camera_next(self, checked=False):
        self.view.next_camera()

    def camera_fov(self, checked=False):
        """Camera > Gezichtsveld: the Zoom tool with the field of view in the Measurements box."""
        self.set_tool("zoom")
        self.vcb.setText(fmt_deg(self.view.cam.fov))
        self.vcb.setFocus()
        self.vcb.selectAll()

    def camera_zoom_window(self, checked=False):
        self.set_tool("zoomwindow")

    def apply_projection(self):
        self.view.set_projection(self.a_ortho.isChecked())

    def toggle_grid(self, on):
        self.settings.show_grid = bool(on)
        self.settings.save()
        self.view.update()

    def toggle_axes(self, on):
        self.settings.show_axes = bool(on)
        self.settings.save()
        self.view.update()

    def toggle_edges(self, on):
        self.settings.show_edges = bool(on)
        self.settings.save()
        self.view.update()

    def toggle_xray(self, on):
        self.settings.xray = bool(on)
        self.settings.save()
        self.view.update()

    def toggle_back_edges(self, on):
        self.settings.back_edges = bool(on)
        self.settings.save()
        self.view.update()

    def toggle_show_hidden(self, on):
        self.settings.show_hidden = bool(on)
        self.settings.save()
        self.view.update()

    def toggle_guides(self, on):
        """Beeld > Hulplijnen: show or hide the construction guides."""
        self.settings.show_guides = bool(on)
        self.settings.save()
        self.view.update()

    def toggle_hide_rest(self, on):
        self.settings.hide_rest = bool(on)
        self.settings.save()
        self.view.update()

    def set_face_style(self, value):
        self.settings.face_style = value
        self.settings.save()
        self.view.update()

    # ------------------------------------------------------------ solids
    def _sel_groups(self, minimum=1):
        groups = self.doc.selected_groups()
        if len(groups) < minimum:
            QMessageBox.information(self, "Volura", f"Selecteer minimaal {minimum} groep(en). "
                                    "Maak eerst een groep van losse geometrie (G).")
            return None
        return groups

    def _in_selection_order(self, groups):
        """Groups in the order in which they were selected (SketchUp solid tools)."""
        order = {it: k for k, it in enumerate(self.doc.selection)}
        return sorted(groups, key=lambda g: order.get(("g", g.id, g.id), 0))

    def _failed(self, e):
        """Undo the checkpoint of an operation that turned out to be impossible."""
        self.doc.undo()
        self.doc.redo_stack.clear()
        QMessageBox.warning(self, "Volura", str(e))

    def solid_op(self, op):
        groups = self._sel_groups(2)
        if not groups:
            return
        groups = self.unlocked(groups)
        if len(groups) < 2:
            return
        groups = self._in_selection_order(groups)
        self.doc.checkpoint()
        try:
            res = ops.boolean(self.doc, groups, op)
        except ValueError as e:
            self._failed(e)
            return
        self.doc.set_selection([("g", res.id, res.id)] if res else [])
        self.doc.notify()

    def trim_solids(self):
        """Gereedschap > Volumegereedschap > Bijsnijden: the second group minus the first."""
        groups = self._sel_groups(2)
        if not groups:
            return
        if len(groups) != 2:
            QMessageBox.information(self, "Volura", "Selecteer precies twee groepen: de tweede "
                                    "wordt met de eerste bijgesneden.")
            return
        groups = self._in_selection_order(self.unlocked(groups))
        if len(groups) != 2:
            return
        self.doc.checkpoint()
        try:
            res = ops.trim(self.doc, groups[0], groups[1])
        except ValueError as e:
            self._failed(e)
            return
        keep = res or groups[0]
        self.doc.set_selection([("g", keep.id, keep.id)])
        self.doc.notify()

    def outer_shell(self):
        """Gereedschap > Buitenschil: only the outside of the selected groups."""
        groups = self.unlocked(self._sel_groups() or [])
        if not groups:
            return
        self.doc.checkpoint()
        try:
            res = ops.outer_shell(self.doc, groups)
        except ValueError as e:
            self._failed(e)
            return
        self.doc.set_selection([("g", res.id, res.id)] if res else [])
        self.doc.notify()

    def intersect_faces(self, with_model):
        """Bewerken > Vlakken snijden: the cut lines of the selection go into the model.

        "Met model" cuts with everything else in the model, "Met selectie" only with
        the other selected items. The edges are added to the active context."""
        g = self.doc.group(self.doc.active) if self.doc.active != ROOT else None
        if g is not None and g.locked:
            self.warn_locked()
            return
        sel = [it for it in self.doc.selection if it[0] in ("f", "g")]
        if not sel:
            QMessageBox.information(self, "Volura", "Selecteer eerst de vlakken of groepen "
                                    "waarvan je de snijlijnen wilt.")
            return
        if with_model:
            items_a, items_b = sel, [it for it in ops.whole_items(self.doc) if it not in set(sel)]
        else:
            items_a = items_b = sel
        self.doc.checkpoint()
        new = ops.intersect_faces(self.doc, items_a, items_b)
        if not new:
            self.doc.undo()
            self.doc.redo_stack.clear()
            self.view.set_status("Geen snijlijnen gevonden.")
            return
        self.doc.notify()

    def split_parts(self):
        groups = self._sel_groups()
        if not groups:
            return
        groups = self.unlocked(groups)
        if not groups:
            return
        self.doc.checkpoint()
        out = []
        for g in groups:
            out += [("g", p.id, p.id) for p in ops.split_parts(self.doc, g)]
        self.doc.set_selection(out)
        self.doc.notify()

    def _target_items(self):
        return list(self.doc.selection) or ops.whole_items(self.doc)

    def drop_to_bed(self):
        items = self._target_items()
        if items:
            self.doc.checkpoint()
            ops.drop_to_bed(self.doc, items)
            self.doc.notify()

    def center_on_bed(self):
        items = self._target_items()
        if items:
            self.doc.checkpoint()
            ops.center_on_bed(self.doc, items, self.settings.bed_size)
            self.doc.notify()

    def mirror(self, axis):
        items = list(self.doc.selection)
        if items:
            self.doc.checkpoint()
            ops.mirror(self.doc, items, axis)
            self.doc.notify()

    def insert_primitive(self, kind):
        fields = {
            "box": [("Breedte (X)", 20), ("Diepte (Y)", 20), ("Hoogte (Z)", 20)],
            "cylinder": [("Straal", 10), ("Hoogte", 20)],
            "cone": [("Straal onder", 10), ("Straal boven", 0), ("Hoogte", 20)],
            "sphere": [("Straal", 10)],
            "torus": [("Hoofdstraal", 15), ("Buisstraal", 4)],
        }[kind]
        dlg = QDialog(self)
        dlg.setWindowTitle("Invoegen")
        form = QFormLayout(dlg)
        spins = []
        for label, v in fields:
            sp = QDoubleSpinBox()
            sp.setRange(0 if "boven" in label else 0.01, 10000)
            sp.setDecimals(2)
            sp.setSuffix(" mm")
            sp.setValue(v)
            form.addRow(label, sp)
            spins.append(sp)
        seg = QSpinBox()
        seg.setRange(3, 512)
        seg.setValue(self.settings.circle_segments)
        if kind != "box":
            form.addRow("Segmenten", seg)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(dlg.accept)
        bb.rejected.connect(dlg.reject)
        form.addRow(bb)
        if dlg.exec() != QDialog.Accepted:
            return
        vals = [s.value() for s in spins]
        if kind == "cone":
            vals = [vals[0], vals[1], vals[2]]
        try:
            m = ops.primitive(kind, *vals, segments=seg.value())
        except Exception as e:
            QMessageBox.critical(self, "Invoegen", str(e))
            return
        bx, by, _ = self.settings.bed_size
        lo, hi = np.asarray(m.bounding_box()[:3]), np.asarray(m.bounding_box()[3:])
        m = m.translate((bx / 2 - (lo[0] + hi[0]) / 2, by / 2 - (lo[1] + hi[1]) / 2, -lo[2]))
        names = {"box": "Balk", "cylinder": "Cilinder", "cone": "Kegel", "sphere": "Bol", "torus": "Torus"}
        self.doc.checkpoint()
        g = ops.group_from_manifold(self.doc, m, self.settings.current_color,
                                    f"{names[kind]} {self.doc.group_counter}")
        self.doc.group_counter += 1
        self.doc.active = ROOT
        self.doc.set_selection([("g", g.id, g.id)])
        self.doc.notify()
        self.set_tool("move")

    # ------------------------------------------------------------ schroefdraad
    THREAD_KINDS = [("Buitendraad op cilinder", "external"), ("Binnendraad in gat", "internal"),
                    ("Nieuwe bout", "bolt"), ("Nieuwe moer of bus", "nut")]

    def _thread_target(self, kind):
        """(ents, groep, center, as, straal, lengte) uit de selectie, of None.

        Geldig: een geselecteerd cirkelvormig eindvlak (buitendraad) of opening
        van een gat (binnendraad), of een groep die een cilinder/bus is."""
        want = "outer" if kind == "external" else "hole"
        doc = self.doc
        for k, ctx, i in list(doc.selection):
            if k != "f":
                continue
            ents = doc.ents(ctx)
            if ents is None:
                continue
            tc = threads.face_thread_context(ents, i, want)
            if tc is not None:
                return (ents, doc.group(ctx)) + tc
        for g in doc.selected_groups():
            tc = threads.group_thread_context(g.ents, want)
            if tc is not None:
                return (g.ents, g) + tc
        return None

    def thread_dialog(self, preset=None):
        kinds = self.THREAD_KINDS
        idx = next((i for i, (_, k) in enumerate(kinds) if k == preset), 0)
        dlg = QDialog(self)
        dlg.setWindowTitle("Schroefdraad")
        form = QFormLayout(dlg)
        kind_cb = QComboBox()
        for label, _ in kinds:
            kind_cb.addItem(label)
        kind_cb.setCurrentIndex(idx)
        dia = QDoubleSpinBox()
        dia.setRange(1.0, 500.0)
        dia.setDecimals(2)
        dia.setSuffix(" mm")
        dia.setValue(8.0)
        pitch_cb = QComboBox()
        pitch_cb.setEditable(True)
        for p in sorted(threads.ISO_PITCH.values()):
            pitch_cb.addItem(f"{p:g}")
        len_sp = QDoubleSpinBox()
        len_sp.setRange(0.5, 2000.0)
        len_sp.setDecimals(2)
        len_sp.setSuffix(" mm")
        len_sp.setValue(20.0)
        clr_sp = QDoubleSpinBox()
        clr_sp.setRange(0.0, 2.0)
        clr_sp.setDecimals(2)
        clr_sp.setSingleStep(0.05)
        clr_sp.setSuffix(" mm")
        clr_sp.setValue(threads.DEFAULT_CLEARANCE)
        starts_sp = QSpinBox()
        starts_sp.setRange(1, 4)
        left_cb = QCheckBox("Linksdraad")
        outer_sp = QDoubleSpinBox()
        outer_sp.setRange(2.0, 500.0)
        outer_sp.setDecimals(1)
        outer_sp.setSuffix(" mm")
        outer_sp.setValue(13.6)
        hex_cb = QCheckBox("Zeskant (uitvinken voor een ronde bus)")
        hex_cb.setChecked(True)
        mate_cb = QCheckBox("Bijpassende moer/bout ook maken")
        form.addRow("Soort", kind_cb)
        form.addRow("Diameter", dia)
        form.addRow("Spoed (ISO grof, vrij in te vullen)", pitch_cb)
        form.addRow("Lengte / hoogte", len_sp)
        form.addRow("Speling", clr_sp)
        form.addRow("Aantal gangen", starts_sp)
        form.addRow("", left_cb)
        form.addRow("Buitenmaat moer/bus", outer_sp)
        form.addRow("", hex_cb)
        form.addRow("", mate_cb)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(dlg.accept)
        bb.rejected.connect(dlg.reject)
        form.addRow(bb)

        def defaults(*_):
            kind = kinds[kind_cb.currentIndex()][1]
            tc = self._thread_target(kind) if kind in ("external", "internal") else None
            if tc is not None:
                dia.setValue(max(dia.minimum(), min(dia.maximum(), 2 * tc[4])))
                len_sp.setValue(max(len_sp.minimum(), min(len_sp.maximum(), tc[5])))
            pitch_cb.setEditText(f"{threads.iso_pitch(dia.value()):g}")
            outer_sp.setValue(round(min(outer_sp.maximum(), dia.value() * 1.7), 1))

        kind_cb.currentIndexChanged.connect(defaults)
        dia.valueChanged.connect(lambda v: outer_sp.setValue(
            round(min(outer_sp.maximum(), v * 1.7), 1)))
        defaults()
        if dlg.exec() != QDialog.Accepted:
            return
        kind = kinds[kind_cb.currentIndex()][1]
        try:
            pitch = float(pitch_cb.currentText().replace(",", "."))
        except ValueError:
            pitch = 0.0
        if pitch <= 0:
            pitch = threads.iso_pitch(dia.value())
        try:
            self._apply_thread(kind, dia.value(), pitch, len_sp.value(), clr_sp.value(),
                               starts_sp.value(), left_cb.isChecked(), outer_sp.value(),
                               hex_cb.isChecked(), mate_cb.isChecked())
        except ValueError as e:
            self.view.set_status(f"Schroefdraad mislukt: {e}")

    def _mate_group(self, man, label, near_bb, dx):
        """Plaats een bijpassend onderdeel naast het bewerkte deel en maak er een groep van."""
        lo, hi = np.asarray(near_bb[:3]), np.asarray(near_bb[3:])
        man = man.translate((hi[0] + dx, (lo[1] + hi[1]) / 2.0, lo[2]))
        return ops.group_from_manifold(self.doc, man, self.settings.current_color, label)

    def _apply_thread(self, kind, d, pitch, length, clearance, starts, left, outer_d,
                      hexagon, mate):
        doc = self.doc
        if kind in ("external", "internal"):
            t = self._thread_target(kind)
            if t is None:
                self.view.set_status(
                    "Selecteer eerst een groep (of het cirkelvormige vlak ervan) "
                    "met een cilinder." if kind == "external" else
                    "Selecteer eerst een groep (of het vlak met de opening) met een gat.")
                return
            ents, g, center, axis, radius, depth = t
            if g is not None and g.locked:
                self.warn_locked()
                return
            m = ents.to_manifold()
            if m.status() != mf.Error.NoError or m.is_empty():
                self.view.set_status("Dit onderdeel is geen gesloten volume.")
                return
            if kind == "external":
                res = threads.external_on(m, center, axis, radius, length, pitch=pitch,
                                          starts=starts, left=left)
                label, near = "Buitendraad", (g.name if g else "Losse geometrie")
            else:
                res = threads.internal_in(m, center, axis, radius, length, clearance,
                                          pitch=pitch, starts=starts, left=left)
                label, near = "Binnendraad", (g.name if g else "Losse geometrie")
            if res.status() != mf.Error.NoError or res.is_empty():
                self.view.set_status("Schroefdraad mislukt: het resultaat is geen geldig volume.")
                return
            doc.checkpoint()
            new_ents = Entities.from_manifold(res)
            if g is not None:
                g.ents = new_ents
            else:
                doc.root = new_ents
            sel = [] if g is None else [("g", g.id, g.id)]
            if mate:
                bb = res.bounding_box()
                if kind == "external":
                    nm = threads.nut(d, pitch, max(0.8 * d, 4 * pitch), outer_d, clearance,
                                     hex=hexagon, starts=starts, left=left)
                    mg = self._mate_group(nm, f"Moer M{d:g}x{pitch:g}", bb, outer_d / 2 + 5)
                else:
                    nm = threads.thread_solid(d, pitch, max(length, 2 * d), starts=starts,
                                              left=left)
                    mg = self._mate_group(nm, f"Bout M{d:g}x{pitch:g}", bb, d + 5)
                if mg is not None:
                    sel.append(("g", mg.id, mg.id))
            doc.set_selection(sel)
            doc.notify()
            self.view.set_status(f"{label} M{d:g}x{pitch:g} gezet op {near}"
                                 + (" inclusief bijpassend onderdeel." if mate else "."))
            return

        if kind == "bolt":
            man = threads.thread_solid(d, pitch, length, starts=starts, left=left)
            label = f"Bout M{d:g}x{pitch:g}"
        else:
            man = threads.nut(d, pitch, length, outer_d, clearance, hex=hexagon,
                              starts=starts, left=left)
            label = ("Moer " if hexagon else "Bus ") + f"M{d:g}x{pitch:g}"
        if man.status() != mf.Error.NoError or man.is_empty():
            self.view.set_status("Schroefdraad mislukt: ongeldige afmetingen.")
            return
        doc.checkpoint()
        bx, by, _ = self.settings.bed_size
        lo, hi = np.asarray(man.bounding_box()[:3]), np.asarray(man.bounding_box()[3:])
        man = man.translate((bx / 2 - (lo[0] + hi[0]) / 2, by / 2 - (lo[1] + hi[1]) / 2, -lo[2]))
        g = ops.group_from_manifold(doc, man, self.settings.current_color, label)
        sel = [("g", g.id, g.id)]
        if mate:
            if kind == "bolt":
                nm = threads.nut(d, pitch, max(0.8 * d, 4 * pitch), outer_d, clearance,
                                 hex=hexagon, starts=starts, left=left)
                mg = self._mate_group(nm, ("Moer " if hexagon else "Bus ") + f"M{d:g}x{pitch:g}",
                                      man.bounding_box(), outer_d / 2 + 5)
            else:
                nm = threads.thread_solid(d, pitch, max(2 * d, length), starts=starts, left=left)
                mg = self._mate_group(nm, f"Bout M{d:g}x{pitch:g}", man.bounding_box(), d + 5)
            if mg is not None:
                sel.append(("g", mg.id, mg.id))
        doc.active = ROOT
        doc.set_selection(sel)
        doc.notify()
        self.view.set_status(f"{label} geplaatst"
                             + (" met bijpassend onderdeel." if mate else "."))

    # ------------------------------------------------------------ snijden & verbinden
    CUT_JOINTS = [("Geen verbinding (alleen snijden)", "geen"),
                  ("Pennen (aan het bovenste deel)", "pennen"),
                  ("Gaten voor losse pennen", "losse pennen"),
                  ("Zwaluwstaart", "zwaluwstaart")]

    def _cut_model(self, groups):
        """One solid manifold of the selected groups, or None when they are not solid."""
        ms = []
        for g in groups:
            m = g.ents.to_manifold()
            if m.status() != mf.Error.NoError or m.is_empty():
                return None
            ms.append(m)
        return ops.union_all(ms) if len(ms) > 1 else ms[0]

    def _spin(self, lo, hi, value, decimals=2, suffix=" mm", step=None):
        sp = QDoubleSpinBox()
        sp.setRange(lo, hi)
        sp.setDecimals(decimals)
        sp.setSuffix(suffix)
        if step is not None:
            sp.setSingleStep(step)
        sp.setValue(value)
        return sp

    def cut_join_dialog(self, origin=None, normal=None):
        groups = self._sel_groups()
        if not groups:
            return
        groups = self.unlocked(groups)
        if not groups:
            return
        model = self._cut_model(groups)
        if model is None:
            self.view.set_status("Snijden: de selectie is geen gesloten volume.")
            return
        if origin is None:
            bb = np.asarray(model.bounding_box())
            origin = tuple((bb[:3] + bb[3:]) / 2)
            normal = (0.0, 0.0, 1.0)

        d = joints.DEFAULTS
        dlg = QDialog(self)
        dlg.setWindowTitle("Snijden en verbinden")
        form = QFormLayout(dlg)
        kind_cb = QComboBox()
        for label, _ in self.CUT_JOINTS:
            kind_cb.addItem(label)
        dia = self._spin(1.0, 100.0, d["pin_diameter"])
        length = self._spin(1.0, 200.0, d["pin_length"])
        count_cb = QComboBox()
        count_cb.setEditable(True)
        for c in ("auto", "1", "2", "3", "4"):
            count_cb.addItem(c)
        wall = self._spin(0.5, 20.0, d["wall"])
        tw = self._spin(1.0, 200.0, d["dovetail_width"])
        td = self._spin(0.5, 100.0, d["dovetail_depth"])
        ta = self._spin(1.0, 60.0, d["dovetail_angle"], 1, "°")
        clr = self._spin(0.0, 2.0, d["clearance"], 2, " mm", 0.05)
        sep = QCheckBox("Delen uit elkaar leggen (10 mm uit elkaar)")
        flat = QCheckBox("Delen plat op het bed leggen")
        form.addRow("Verbinding", kind_cb)
        form.addRow("Pen-diameter", dia)
        form.addRow("Pen-lengte", length)
        form.addRow("Aantal pennen", count_cb)
        form.addRow("Wanddikte rond gat", wall)
        form.addRow("Staartbreedte", tw)
        form.addRow("Staartdiepte", td)
        form.addRow("Staarthoek", ta)
        form.addRow("Speling", clr)
        form.addRow("", sep)
        form.addRow("", flat)

        def update_enabled():
            kind = self.CUT_JOINTS[kind_cb.currentIndex()][1]
            pin = kind in ("pennen", "losse pennen")
            dovetail = kind == "zwaluwstaart"
            for w in (dia, length, count_cb, wall):
                w.setEnabled(pin)
            for w in (tw, td, ta):
                w.setEnabled(dovetail)
            clr.setEnabled(pin or dovetail)
        kind_cb.currentIndexChanged.connect(update_enabled)
        update_enabled()
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(dlg.accept)
        bb.rejected.connect(dlg.reject)
        form.addRow(bb)
        if dlg.exec() != QDialog.Accepted:
            return
        kind = self.CUT_JOINTS[kind_cb.currentIndex()][1]
        txt = count_cb.currentText().strip().lower()
        count = int(txt) if txt.isdigit() else "auto"
        params = {"pin_diameter": dia.value(), "pin_length": length.value(), "count": count,
                  "wall": wall.value(), "dovetail_width": tw.value(), "dovetail_depth": td.value(),
                  "dovetail_angle": ta.value(), "clearance": clr.value(),
                  "segments": self.settings.circle_segments}
        self._apply_cut_join(groups, model, origin, normal, kind, params,
                             sep.isChecked(), flat.isChecked())

    def _apply_cut_join(self, groups, model, origin, normal, kind, params, separate, flat):
        try:
            a, b, extras = joints.split_with_joint(model, origin, normal, kind, params)
        except Exception as e:
            self.view.set_status(f"Snijden mislukt: {e}")
            return
        if (a.is_empty() or b.is_empty() or a.status() != mf.Error.NoError
                or b.status() != mf.Error.NoError):
            self.view.set_status("Snijden gaf geen twee geldige delen.")
            return
        doc = self.doc
        color, name = groups[0].color, groups[0].name
        doc.checkpoint()
        first = groups[0]
        first.ents = Entities.from_manifold(a)
        for g in groups[1:]:
            doc.remove_group(g.id)
        part_b = doc.add_group(Entities.from_manifold(b), color, f"{name} (deel B)")
        sel = [("g", first.id, first.id), ("g", part_b.id, part_b.id)]
        for i, pin in enumerate(extras):
            pg = ops.group_from_manifold(doc, pin, self.settings.current_color, f"Pen {i + 1}")
            sel.append(("g", pg.id, pg.id))
        a_item, b_item = [("g", first.id, first.id)], [("g", part_b.id, part_b.id)]
        if flat:
            self._lay_part_flat(first, normal, +1)
            self._lay_part_flat(part_b, normal, -1)
            self._separate_on_bed(first, part_b)
        elif separate:
            n = joints.normalize(np.asarray(normal, dtype=float))
            ops.transform_selection(doc, np.eye(3), n * 10.0, items=a_item)
            ops.transform_selection(doc, np.eye(3), -n * 10.0, items=b_item)
        doc.active = ROOT
        doc.set_selection(sel)
        doc.notify()
        label = dict((k, l) for l, k in self.CUT_JOINTS)[kind].lower()
        self.view.set_status(f"Model gesneden met {label}"
                             + (f" en {len(extras)} losse pen(nen)." if extras else "."))

    def _lay_part_flat(self, g, normal, sign):
        """Rotate a part about its centre so the cut face lies down, then drop it to the bed."""
        bb = g.bbox()
        if bb is None:
            return
        c = 0.5 * (np.asarray(bb[0], dtype=float) + np.asarray(bb[1], dtype=float))
        R = joints.align_rotation(np.asarray(normal, dtype=float), np.array([0.0, 0.0, float(sign)]))
        item = [("g", g.id, g.id)]
        ops.transform_selection(self.doc, R, c - R @ c, items=item)
        ops.drop_to_bed(self.doc, item)

    def _separate_on_bed(self, ga, gb):
        """Slide part B in +X next to part A so the two do not overlap on the bed."""
        ba = ga.bbox()
        if ba is None:
            return
        gap = float(ba[1][0] - ba[0][0]) + 10.0
        ops.transform_selection(self.doc, np.eye(3), np.array([gap, 0.0, 0.0]),
                                items=[("g", gb.id, gb.id)])

    def ask_text(self):
        dlg = QDialog(self)
        dlg.setWindowTitle("3D-tekst")
        form = QFormLayout(dlg)
        ed = QLineEdit(getattr(self, "_last_text", "Volura"))
        font = QFontComboBox()
        font.setCurrentFont(self.font())
        bold = QCheckBox("Vet")
        bold.setChecked(True)
        h = QDoubleSpinBox()
        h.setRange(0.5, 1000)
        h.setValue(10)
        h.setSuffix(" mm")
        form.addRow("Tekst", ed)
        form.addRow("Lettertype", font)
        form.addRow("", bold)
        form.addRow("Hoogte hoofdletter", h)
        form.addRow(QLabel("Na het plaatsen: Duwen/trekken. Alt ingedrukt = apart onderdeel\n"
                           "(bijv. ingelegde tekst in een andere kleur)."))
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(dlg.accept)
        bb.rejected.connect(dlg.reject)
        form.addRow(bb)
        if dlg.exec() != QDialog.Accepted or not ed.text().strip():
            if not self.view.tools["text"].loops:
                self.set_tool("select")
            return
        self._last_text = ed.text()
        self.view.tools["text"].set_text(ed.text(), font.currentFont().family(), h.value(), bold.isChecked())
        if self.view.tool is not self.view.tools["text"]:
            self.view.set_tool("text")

    # ------------------------------------------------------------ printing
    def print_check(self):
        part_list = ops.parts(self.doc)
        msgs = ops.check_parts(part_list, self.settings.bed_size)
        if not part_list:
            msgs = [("info", "Het model bevat nog geen volumes.")]
        loose = len(self.doc.root.loose_edges())
        if loose:
            msgs.append(("info", f"{loose} losse rand(en) zonder vlak worden niet geprint."))
        icons = {"ok": "✔", "info": "ℹ", "waarschuwing": "⚠", "fout": "✖"}
        colors = {"ok": "#2e7d32", "info": "#1565c0", "waarschuwing": "#e65100", "fout": "#c62828"}
        html = "<h3>Printcontrole</h3><p>Elk onderdeel (groep of losse vorm) moet een gesloten volume zijn: " \
               "geen gaten, geen dubbele of losse vlakken. Alleen gesloten volumes worden geëxporteerd.</p><ul>"
        for level, text in msgs:
            html += f'<li style="color:{colors[level]}">{icons[level]} {text}</li>'
        html += "</ul>"
        dlg = QDialog(self)
        dlg.setWindowTitle("Printcontrole")
        dlg.resize(620, 420)
        lay = QVBoxLayout(dlg)
        tb = QTextBrowser()
        tb.setHtml(html)
        lay.addWidget(tb)
        row = QHBoxLayout()
        b1 = QPushButton("Op bed plaatsen")
        b1.clicked.connect(lambda: (self.drop_to_bed(), dlg.accept()))
        b2 = QPushButton("Sluiten")
        b2.clicked.connect(dlg.accept)
        row.addWidget(b1)
        row.addStretch(1)
        row.addWidget(b2)
        lay.addLayout(row)
        dlg.exec()

    def show_help(self):
        rows = "".join(f"<tr><td><b>{sc}</b></td><td>{label}</td></tr>" for _, label, _, sc in TOOL_INFO if sc)
        html = f"""
<h2>Volura - werkwijze</h2>
<p>Volura werkt zoals SketchUp: teken een plat vlak en maak er met <b>Duwen/trekken</b> een volume van.</p>
<ol>
<li>Teken randen met Lijn (L), Rechthoek (R), Cirkel (C) of Veelhoek. Een gesloten vorm wordt een vlak.</li>
<li>Randen splitsen elkaar, een lijn over een vlak splitst het vlak; met de Gum (E) wis je randen
(of delen ervan) en worden vlakken weer samengevoegd.</li>
<li>Duwen/trekken (P) werkt op elk vlak of deel van een vlak: omhoog = erbij, omlaag = uitsparing of gat.
Typ een maat en Enter; de richting volgt je muis.</li>
<li>Selecteer een vorm (driemaal klikken) en maak er een <b>groep</b> van (G): elke groep is een apart
printbaar onderdeel met een eigen kleur. Dubbelklik op een groep om hem te bewerken.</li>
<li><b>Meerkleuren</b>: houd <b>Alt</b> vast bij duwen/trekken: naar binnen maakt een inleg (uitsparing +
vulling als aparte groep), naar buiten een apart onderdeel. Geef het een kleur met de verfemmer (B) en
exporteer als 3MF.</li>
</ol>
<h3>Sneltoetsen</h3>
<table cellpadding="3">{rows}
<tr><td><b>A</b></td><td>Boog door 2 punten (in Tekenen &gt; Bogen ook Boog, Boog door 3 punten en Taartpunt)</td></tr>
<tr><td><b>Shift+Z</b></td><td>Zoomen tot alles</td></tr>
<tr><td><b>Middelste muisknop</b></td><td>Draaien, met Shift: pannen. Scrollwiel: zoomen</td></tr>
<tr><td><b>Pijltjes</b></td><td>As vergrendelen (rechts = rood/X, links = groen/Y, omhoog = blauw/Z)</td></tr>
<tr><td><b>Shift</b> (vasthouden)</td><td>Huidige richting vergrendelen</td></tr>
<tr><td><b>Ctrl</b></td><td>Verplaatsen/roteren: kopie. Duwen/trekken: nieuw beginvlak</td></tr>
<tr><td><b>Alt</b></td><td>Duwen/trekken: apart onderdeel / inleg</td></tr>
<tr><td><b>G</b></td><td>Component of groep maken van de selectie</td></tr>
<tr><td><b>Esc</b></td><td>Huidige bewerking annuleren</td></tr>
<tr><td><b>Ctrl+Z / Ctrl+Y</b></td><td>Ongedaan maken / opnieuw</td></tr>
</table>
<h3>Maatinvoer (rechtsonder)</h3>
<p>Begin gewoon te typen tijdens een bewerking. Voorbeelden: <code>25</code>, <code>2,5cm</code>,
<code>40;20</code> (rechthoek), <code>64s</code> (segmenten), <code>10;0;5</code> (verplaatsing),
<code>5x</code> of <code>/5</code> na een kopie (reeks), <code>45</code> (graden bij roteren).</p>
"""
        dlg = QDialog(self)
        dlg.setWindowTitle("Volura help")
        dlg.resize(700, 640)
        lay = QVBoxLayout(dlg)
        tb = QTextBrowser()
        tb.setHtml(html)
        lay.addWidget(tb)
        dlg.exec()

    # ------------------------------------------------------------ updates
    def check_for_updates_on_startup(self):
        """Eén keer per dag stil kijken; alleen in de zelfstandige versie, nooit in een test."""
        if is_offscreen() or not self.settings.check_updates or not getattr(sys, "frozen", False):
            return None
        q = qsettings()
        try:
            last = float(q.value(UPDATE_LAST_KEY, 0) or 0)
        except (TypeError, ValueError):
            last = 0.0
        if time.time() - last < UPDATE_INTERVAL:
            return None
        q.setValue(UPDATE_LAST_KEY, time.time())
        return self.check_for_updates(automatic=True)

    def check_for_updates(self, automatic=False):
        """Help > Controleren op updates…: de nieuwste release op GitHub ophalen."""
        if not automatic:
            self.hint.setText("Zoeken naar een nieuwere versie…")
        self._update_bridge = run_in_background(
            lambda progress: updater.latest_release(),
            lambda rel, err: self.on_update_result(rel, err, automatic))
        return self._update_bridge

    def on_update_result(self, release, error, automatic=False):
        if error is not None or release is None:
            if not automatic:
                self.hint.setText("")
                detail = f"\n\n{error}" if error is not None else ""
                QMessageBox.information(self, "Updates", "Volura kon GitHub niet bereiken of vond "
                                        "geen release." + detail)
            return None
        if not updater.is_newer(release.version):
            if not automatic:
                self.hint.setText("")
                QMessageBox.information(self, "Updates", f"Je hebt de nieuwste versie ({__version__}).")
            return None
        if automatic and qsettings().value(UPDATE_SKIP_KEY, "") == release.version:
            return None
        self.hint.setText(f"Volura {release.version} is beschikbaar.")
        return self.offer_update(release)

    def offer_update(self, release):
        """Vraag of de nieuwe versie er nu op moet; de installatie doet de rest."""
        box = QMessageBox(self)
        box.setWindowTitle("Nieuwe versie van Volura")
        box.setIcon(QMessageBox.Information)
        box.setText(f"<b>Volura {release.version}</b> is beschikbaar. Je hebt nu {__version__}.")
        if release.notes.strip():
            box.setDetailedText(release.notes.strip())
        self_install = updater.can_self_install() and bool(release.setup_url)
        if self_install:
            box.setInformativeText("Volura downloadt de nieuwe versie, sluit af, installeert en "
                                   "start daarna opnieuw. Je model wordt eerst bewaard als je dat wilt.")
            go = box.addButton("Nu bijwerken", QMessageBox.AcceptRole)
        else:
            box.setInformativeText("Download de nieuwe versie van de releasepagina op GitHub.")
            go = box.addButton("Downloadpagina openen", QMessageBox.AcceptRole)
        box.addButton("Later", QMessageBox.RejectRole)
        skip = box.addButton("Deze versie overslaan", QMessageBox.DestructiveRole)
        box.exec()
        clicked = box.clickedButton()
        if clicked is skip:
            qsettings().setValue(UPDATE_SKIP_KEY, release.version)
        elif clicked is go:
            if self_install:
                return self.install_update(release)
            QDesktopServices.openUrl(QUrl(release.page))
        return None

    def install_update(self, release):
        """Bewaar het model, download de installatie en laat die Volura vervangen."""
        if not self.confirm_discard():
            return None
        dlg = QProgressDialog(f"Volura {release.version} downloaden…", "Annuleren", 0, 100, self)
        dlg.setWindowTitle("Bijwerken")
        dlg.setWindowModality(Qt.WindowModal)
        dlg.setMinimumDuration(0)
        dlg.setValue(0)
        stop = threading.Event()
        dlg.canceled.connect(stop.set)

        def progress(done, total):
            if total:
                dlg.setValue(min(99, int(100 * done / total)))

        def finished(path, error):
            dlg.reset()
            if error is not None:
                if not isinstance(error, InterruptedError):
                    QMessageBox.warning(self, "Bijwerken", f"Downloaden is niet gelukt:\n{error}")
                return
            try:
                updater.run_installer(path)
            except OSError as e:
                QMessageBox.warning(self, "Bijwerken", f"De installatie kon niet starten:\n{e}")
                return
            self.doc.dirty = False             # al bewaard of bewust weggegooid
            self.close()

        self._update_bridge = run_in_background(
            lambda report: updater.download(release, report, stop.is_set), finished, progress)
        return dlg

    def about(self):
        QMessageBox.about(self, "Over Volura",
                          f"<h3>Volura {__version__}</h3>"
                          "<p>Open source 3D-modelleerprogramma voor 3D-printen, met de werkwijze van "
                          "SketchUp 2017.</p><p>Bij het exporteren wordt elk onderdeel gecontroleerd op een "
                          "gesloten volume en foutloos geschreven als STL of meerkleuren-3MF.</p>"
                          "<p>© 2026 L. van der Poel · Licentie: MIT</p>"
                          "<p>Volura gebruikt open source software van derden:</p><ul>"
                          "<li>Qt 6 en PySide6 (The Qt Company), LGPL v3. De Qt-bibliotheken staan als "
                          "losse bestanden in de programmamap en mogen vervangen worden; broncode op "
                          "<a href='https://code.qt.io'>code.qt.io</a>.</li>"
                          "<li>Manifold (manifold3d), Apache 2.0</li>"
                          "<li>NumPy, BSD 3-clause</li>"
                          "<li>PyOpenGL, BSD-stijl</li>"
                          "<li>Python, PSF-licentie</li></ul>"
                          "<p>De volledige licentieteksten staan in THIRD_PARTY_LICENSES.txt "
                          "naast het programma.</p>")

    # ------------------------------------------------------------ context menu
    # ------------------------------------------------------------ context menu
    def _selected_faces(self):
        """(ctx, fid) of every selected face, the clicked one first."""
        out = []
        for k, c, i in self.doc.selection:
            ents = self.doc.ents(c)
            if k == "f" and ents is not None and i in ents.F:
                out.append((c, i))
        return out

    def _clicked_face(self):
        faces = self._selected_faces()
        return faces[0] if faces else None

    def face_area(self, ents, fid):
        tri, _ = ents.face_triangles(fid)
        if not len(tri):
            return 0.0
        return float(np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]),
                                    axis=1).sum() / 2)

    def select_face_edges(self):
        items = []
        for c, i in self._selected_faces():
            ents = self.doc.ents(c)
            items += [("e", c, e) for e in ents.face_edges(ents.F[i])]
        self.doc.set_selection(items)

    def select_connected_faces(self):
        by_ctx = {}
        for c, i in self._selected_faces():
            by_ctx.setdefault(c, set()).add(i)
        items = []
        for c, fs in by_ctx.items():
            ents = self.doc.ents(c)
            nb = set(fs)
            for f in fs:
                for e in ents.face_edges(ents.F[f]):
                    nb |= ents.e2f.get(e, set())
            items += [("f", c, f) for f in sorted(nb)]
        self.doc.set_selection(items)

    def select_all_connected(self):
        by_ctx = {}
        for k, c, i in self.doc.selection:
            if k in ("f", "e"):
                by_ctx.setdefault(c, ([], []))[0 if k == "f" else 1].append(i)
        items = []
        for c, (fs, es) in by_ctx.items():
            ents = self.doc.ents(c)
            if ents is None:
                continue
            f2, e2 = ents.connected(fs, es)
            items += [("f", c, f) for f in sorted(f2)] + [("e", c, e) for e in sorted(e2)]
        self.doc.set_selection(items)

    def select_same_material(self):
        ref = self._clicked_face()
        if ref is None:
            return
        c, fid = ref
        ents = self.doc.ents(c)
        color = ents.F[fid].color
        self.doc.set_selection([("f", c, f) for f in sorted(ents.F) if ents.F[f].color == color])

    def show_area_selection(self):
        total = sum(self.face_area(self.doc.ents(c), i) for c, i in self._selected_faces())
        QMessageBox.information(self, "Oppervlakte",
                                f"Totale oppervlakte van de selectie: {total:.2f} mm²")

    def show_area_material(self):
        ref = self._clicked_face()
        if ref is None:
            return
        c, fid = ref
        ents = self.doc.ents(c)
        color = ents.F[fid].color
        total = sum(self.face_area(ents, f) for f in ents.F if ents.F[f].color == color)
        QMessageBox.information(self, "Oppervlakte",
                                f"Totale oppervlakte van dit materiaal: {total:.2f} mm²")

    def align_view_to_face(self):
        """Weergave uitlijnen: look straight at the face (view direction = -normal)."""
        ref = self._clicked_face()
        if ref is None:
            return
        c, fid = ref
        ents = self.doc.ents(c)
        pts = np.concatenate(ents.face_points(fid))
        n = np.asarray(ents.F[fid].normal, dtype=float)
        n = n / max(float(np.linalg.norm(n)), 1e-12)
        cam = self.view.cam
        cam.pitch = max(-89.99, min(89.99, math.degrees(math.asin(float(np.clip(n[2], -1.0, 1.0))))))
        cam.yaw = math.degrees(math.atan2(n[1], n[0]))
        cam.fit((pts.min(axis=0), pts.max(axis=0)))
        cam.target = pts.mean(axis=0)
        self.view.commit_camera()
        self.view.update()

    def reverse_selected_faces(self):
        by_ctx = {}
        for c, i in self._selected_faces():
            by_ctx.setdefault(c, []).append(i)
        if not by_ctx:
            return
        self.doc.checkpoint()
        for c, fs in by_ctx.items():
            self.doc.ents(c).reverse_faces(fs)
        self.doc.notify()

    def orient_selected_faces(self):
        ref = self._clicked_face()
        if ref is None:
            return
        c, fid = ref
        self.doc.checkpoint()
        self.doc.ents(c).orient_faces(fid)
        self.doc.notify()

    def zoom_selection(self):
        bb = ops.selection_bbox(self.doc)
        if bb is None:
            return
        self.view.cam.fit((bb[0], bb[1]))
        self.view.commit_camera()
        self.view.update()

    def build_context_menu(self):
        """The right-click menu for the current selection (SketchUp 2017 order)."""
        m = QMenu(self)
        sel = self.doc.selection
        if sel:
            m.addAction("Entiteitsinfo", self.show_entity_info)
            m.addAction(self.a_delete)
            m.addAction(self.a_hide)
            sel_groups = self.doc.selected_groups()
            if sel_groups:
                # SketchUp: right click > Lock / Unlock
                m.addAction(self.a_unlock if all(g.locked for g in sel_groups) else self.a_lock)
            if self._selected_faces():
                m.addSeparator()
                sm = m.addMenu("Selecteren")
                sm.addAction("Begrenzende randen", self.select_face_edges)
                sm.addAction("Verbonden vlakken", self.select_connected_faces)
                sm.addAction("Alles wat verbonden is", self.select_all_connected)
                sm.addAction("Alles met hetzelfde materiaal", self.select_same_material)
                om = m.addMenu("Oppervlakte")
                om.addAction("Selectie", self.show_area_selection)
                om.addAction("Materiaal", self.show_area_material)
                fm = m.addMenu("Vlakken snijden")
                fm.addAction(self.a_cut_face_model)
                fm.addAction(self.a_cut_face_sel)
                m.addAction("Weergave uitlijnen", self.align_view_to_face)
                m.addAction("Vlakken omkeren", self.reverse_selected_faces)
                m.addAction("Vlakken oriënteren", self.orient_selected_faces)
                m.addAction("Selectie zoomen", self.zoom_selection)
            if self._selected_faces() or self.doc.selected_groups():
                m.addSeparator()
                self._add_laser_menu(m)
            m.addSeparator()
            if any(k in ("f", "g") for k, _, _ in sel):
                cm = m.addMenu("Filamentkleur")
                for i, (name, color) in enumerate(self.doc.materials, 1):
                    a = cm.addAction(swatch(color), f"{i}: {name}")
                    a.triggered.connect(lambda checked=False, c=color: self.paint_selection(c))
            if any(k in ("f", "e") for k, _, _ in sel):
                m.addAction(self.a_group)
            groups = self.doc.selected_groups()
            if groups:
                m.addAction(self.a_edit_group)
                m.addAction(self.a_explode)
                if len(groups) >= 2:
                    for a in (self.a_union, self.a_subtract, self.a_intersect):
                        m.addAction(a)
                if len(groups) == 2:
                    m.addAction(self.a_trim)
                m.addAction(self.a_shell)
                m.addAction(self.a_split)
            mm = m.addMenu("Spiegelen langs")
            for i, naam in enumerate(("Rode as", "Groene as", "Blauwe as")):
                mm.addAction(naam, lambda checked=False, ax=i: self.mirror(ax))
            m.addAction(self.a_drop)
            m.addAction(self.a_center)
        else:
            m.addAction(self.a_sel_all)
            m.addAction("Alles zichtbaar maken", self.unhide_all)
        if self.doc.active != ROOT:
            m.addSeparator()
            m.addAction(self.a_close_group)
        m.addSeparator()
        m.addAction(self.a_extents)
        return m

    def show_context_menu(self, gpos):
        local = self.view.mapFromGlobal(gpos)
        pick = self.view.engine.pick_entity(local.x(), local.y())
        from .tools import pick_item
        item = pick_item(pick) if pick[0] else None
        if item is not None and item not in self.doc.selection:
            if item[0] == "f":
                ents = self.doc.ents(item[1])
                self.doc.set_selection([item] + [("e", item[1], e) for e in ents.face_edges(ents.F[item[2]])])
            else:
                self.doc.set_selection([item])
        # het vlak onder de muis, ook in een groep: "Laseren / frezen" neemt dát vlak
        o, d = self.view.cam.ray(local.x(), local.y())
        fh = self.view.engine.pick_face(o, d) if pick[0] in ("face", "group") else None
        self._context_face = (fh[1], fh[2]) if fh else None
        try:
            self.build_context_menu().exec(gpos)
        finally:
            self._context_face = None

    def paint_selection(self, color):
        items = [it for it in self.doc.selection if it[0] in ("f", "g")]
        locked = [it for it in items if it[0] == "g" and getattr(self.doc.group(it[2]), "locked", False)]
        if locked:
            self.warn_locked()
            items = [it for it in items if it not in locked]
        if not items:
            return
        self.doc.checkpoint()
        for k, c, i in items:
            if k == "g":
                self.doc.group(i).color = color
            else:
                self.doc.ents(c).set_face_color(i, color)
        self.doc.notify()

    def closeEvent(self, e):
        if not self.confirm_discard():
            e.ignore()
            return
        self.autosave_timer.stop()
        self.settings.save()
        remove_autosave()       # netjes afgesloten: er is niets te herstellen
        e.accept()


def excepthook(exc_type, exc, tb):
    """Een fout buiten de event-handlers: dezelfde log als daar, plus een melding."""
    text = errors.log_exception(exc_type, exc, tb, note="onafgehandelde fout")
    if is_offscreen():
        return
    app = QApplication.instance()
    if app is not None:
        QMessageBox.critical(None, "Volura - onverwachte fout", text[-3000:])


def main(argv=None):
    argv = sys.argv if argv is None else argv
    if sys.platform.startswith("linux") and os.environ.get("WAYLAND_DISPLAY") \
            and not os.environ.get("QT_QPA_PLATFORM"):
        os.environ["QT_QPA_PLATFORM"] = "xcb"  # PyOpenGL needs GLX
    QApplication.setAttribute(Qt.AA_ShareOpenGLContexts)
    app = QApplication(argv)
    app.setApplicationName("Volura")
    app.setOrganizationName("Volura")
    app.setWindowIcon(icon("app"))
    sys.excepthook = excepthook
    win = MainWindow()
    win.show()
    if len(argv) > 1 and os.path.exists(argv[1]):
        win.file_open(argv[1])
    return app.exec()

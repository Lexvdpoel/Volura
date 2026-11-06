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
                               QSizePolicy, QSlider, QSpinBox, QStackedWidget, QTableWidget,
                               QTableWidgetItem, QTreeWidget, QTreeWidgetItem, QTextBrowser,
                               QToolBar, QToolButton, QVBoxLayout, QWidget)

from . import (__version__, background, boxgen, cam, dxf, errors, fileio, i18n, joints, nesting,
               ops, sheet3d, svg, spacemouse, theme, threads, updater, vector_export, vector_import,
               workers)
from .brep import Entities
from .inference import GeoCache
from .mesh_formats import MESH_EXTENSIONS, is_mesh
from . import repair
from .geom import Plane
from .i18n import tr, tr_noop
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
    # name, label (English, translated at use), icon, shortcut
    ("select", tr_noop("Select"), "select", "Space"),
    ("eraser", tr_noop("Eraser"), "eraser", "E"),
    ("paint", tr_noop("Paint Bucket"), "paint", "B"),
    ("line", tr_noop("Line"), "line", "L"),
    ("rectangle", tr_noop("Rectangle"), "rectangle", "R"),
    ("circle", tr_noop("Circle"), "circle", "C"),
    ("polygon", tr_noop("Polygon"), "polygon", None),
    ("text", tr_noop("3D Text"), "text", "Shift+T"),
    ("pushpull", tr_noop("Push/Pull"), "pushpull", "P"),
    ("offset", tr_noop("Offset"), "offset", "F"),
    ("move", tr_noop("Move"), "move", "M"),
    ("rotate", tr_noop("Rotate"), "rotate", "Q"),
    ("scale", tr_noop("Scale"), "scale", "S"),
    ("tape", tr_noop("Tape Measure"), "tape", "T"),
    ("orbit", tr_noop("Orbit"), "orbit", "O"),
    ("pan", tr_noop("Pan"), "pan", "H"),
    ("zoom", tr_noop("Zoom"), "zoom", "Z"),
]

# The tools of the 2D worksheet; their keys only work in 2D, because in 3D M, Q, S and
# Space belong to the 3D tools. The letters of the drawing tools are the same as in 3D
# (L, R, C, A, F), but they live on the worksheet itself, so the window gets no duplicate
# shortcuts.
class ModeSwitch(QToolButton):
    """The 3D | 2D slide switch at the bottom right: the half of the current mode is blue.

    Unchecked is 3D, checked is 2D; when switching, the blue block slides to the other
    side. The text stays "3D | 2D" for screen readers and shortcuts.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("modeSwitch")
        self.setCheckable(True)
        self.setText("3D | 2D")
        self.setCursor(Qt.PointingHandCursor)
        self._pos = 0.0                     # 0 = block on 3D, 1 = on 2D
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
        p.setPen(QPen(theme.qcolor("switch_border"), 1.0))
        p.setBrush(theme.qcolor("switch_hover" if self.underMouse() else "switch_track"))
        p.drawRoundedRect(r, radius, radius)
        half = r.width() / 2.0
        knob = QRectF(r.left() + 1.5 + self._pos * (half - 1.5), r.top() + 1.5,
                      half - 1.5, r.height() - 3.0)
        p.setPen(Qt.NoPen)
        p.setBrush(theme.qcolor("switch_on"))
        p.drawRoundedRect(knob, knob.height() / 2.0, knob.height() / 2.0)
        font = self.font()
        font.setBold(True)
        p.setFont(font)
        for i, label in enumerate(("3D", "2D")):
            part = QRectF(r.left() + i * half, r.top(), half, r.height())
            active = abs(self._pos - i) < 0.5
            p.setPen(theme.qcolor("switch_on_text" if active else "switch_text"))
            p.drawText(part, Qt.AlignCenter, label)
        p.end()


SHEET_TOOL_INFO = [
    # name in volura.sheettools, label (English, translated at use), help text
    ("select", tr_noop("Select"), tr("Click a line or drag a box (Space)")),
    ("move", tr_noop("Move"), tr("Click a base point and a target, or type a distance (M)")),
    ("rotate", tr_noop("Turn"), tr("Click a center and type an angle in degrees (Q)")),
    ("scale", tr_noop("Scale"), tr("Drag from the center or type a factor (S)")),
    ("line", tr_noop("Line"), tr("Click points; click the start point to close (L)")),
    ("rectangle", tr_noop("Rectangle"), tr("Click two corners or type 30;20 (R)")),
    ("roundrect", tr_noop("Rounded Rectangle"),
     tr("Rectangle with rounded corners, type the radius with r (N)")),
    ("circle", tr_noop("Circle"), tr("Click or type the center and radius (C)")),
    ("ellipse", tr_noop("Ellipse"), tr("Click the center and drag the two semi-axes (E)")),
    ("arc", tr_noop("Arc"), tr("Arc through three points (A)")),
    ("centerarc", tr_noop("Arc Around Center"), tr("Center, radius and angle of the arc (B)")),
    ("polygon", tr_noop("Polygon"), tr("Center and radius; type the number of sides like 6s (P)")),
    ("text", tr_noop("Text"), tr("Place the outlines of text on the worksheet (T)")),
    ("offset", tr_noop("Offset"), tr("Move a closed outline inward or outward (F)")),
]

# The panels of the Default Tray. The Tray knows a panel by its translated title (that is
# what it shows); these English names are the stable keys in the code, used through tr().
PANEL_INFO = tr_noop("Entity Info")
PANEL_MATERIALS = tr_noop("Materials")
PANEL_OUTLINER = tr_noop("Outliner")
PANEL_PRINT_CHECK = tr_noop("Print Check")
PANEL_LAYERS = tr_noop("Layers")
PANELS = (PANEL_INFO, PANEL_MATERIALS, PANEL_OUTLINER, PANEL_PRINT_CHECK, PANEL_LAYERS)


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
        self.ask_simplify_import = self._bool("ask_simplify_import", True)
        self.theme = str(self.q.value("theme", theme.SYSTEM))
        if self.theme not in dict(theme.CHOICES):
            self.theme = theme.SYSTEM
        self.spacemouse = self._bool("spacemouse", True)
        try:
            self.spacemouse_speed = float(self.q.value("spacemouse_speed", 1.0))
        except (TypeError, ValueError):
            self.spacemouse_speed = 1.0
        self.spacemouse_speed = max(0.05, min(2.0, self.spacemouse_speed))
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
        self.q.setValue("ask_simplify_import", self.ask_simplify_import)
        self.q.setValue("theme", self.theme)
        self.q.setValue("spacemouse", self.spacemouse)
        self.q.setValue("spacemouse_speed", self.spacemouse_speed)
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
        self.q.sync()   # now on disk, so the next Settings (or the next start) reads these


#: one file-dialog filter per 3D format Volura imports
MESH_FILTERS = ["STL (*.stl)", "OBJ (*.obj)", "3MF (*.3mf)", "PLY (*.ply)", "OFF (*.off)",
                "AMF (*.amf)", "glTF (*.glb *.gltf)", "COLLADA (*.dae)"]


def mesh_filter():
    """The file-dialog filter for every 3D format Volura imports."""
    return tr("3D models") + " (" + " ".join(f"*{e}" for e in MESH_EXTENSIONS) + ")"


def swatch(color, size=16):
    """A color square with a gray border: black and white stay visible in both themes."""
    pm = QPixmap(size, size)
    pm.fill(QColor(color))
    p = QPainter(pm)
    p.setPen(QPen(QColor("#808080"), 1.0))
    p.drawRect(0, 0, size - 1, size - 1)
    p.end()
    return QIcon(pm)


# ---------------------------------------------------------------- panels

class MaterialsPanel(QWidget):
    def __init__(self, win):
        super().__init__()
        self.win = win
        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        head = QLabel(tr("Filaments (the colors you use get filament 1, 2, 3… in this order):"))
        head.setWordWrap(True)
        lay.addWidget(head)
        self.list = QListWidget()
        self.list.setIconSize(QSize(22, 22))
        self.list.itemClicked.connect(self.pick)
        self.list.itemDoubleClicked.connect(lambda it: self.edit())
        lay.addWidget(self.list)
        row = QHBoxLayout()
        for text, fn in ((tr("New"), self.add), (tr("Edit"), self.edit), (tr("Remove"), self.remove)):
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
        c = QColorDialog.getColor(QColor("#888888"), self, tr("New filament color"))
        if not c.isValid():
            return
        doc = self.win.doc
        doc.checkpoint()
        doc.materials.append([tr("Color {n}").format(n=len(doc.materials) + 1), c.name()])
        self.win.settings.current_color = c.name()
        doc.notify()

    def edit(self):
        row = self.list.currentRow()
        doc = self.win.doc
        if row < 0:
            return
        name, color = doc.materials[row]
        dlg = QDialog(self)
        dlg.setWindowTitle(tr("Edit Filament"))
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
        form.addRow(tr("Name"), ed)
        form.addRow(tr("Color"), btn)
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


# Entity Info checks a group with fewer faces than this right away; a bigger one (an
# imported scan) in the background, so selecting it does not freeze the window.
PART_INFO_BACKGROUND_FACES = 5000


class EntityInfoPanel(QWidget):
    def __init__(self, win):
        super().__init__()
        self.win = win
        self.lay = QVBoxLayout(self)
        self.lay.setContentsMargins(6, 6, 6, 6)
        self.lay.addStretch(1)
        self.body = None
        self._part_cache = {}           # id(ents) -> (ents, version, (name, color), summary)
        self._part_pending = set()      # (id(ents), version, (name, color)) being checked

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
            form.addRow(QLabel(tr("<i>Editing: {name}</i>").format(name=doc.group(doc.active).name)))
        if not sel:
            form.addRow(QLabel(tr("<b>No selection</b>")))
            form.addRow(tr("Groups"), QLabel(str(len(doc.groups))))
            form.addRow(tr("Loose faces"), QLabel(str(len(doc.root.F))))
            form.addRow(tr("Loose edges"), QLabel(str(len(doc.root.E))))
            form.addRow(tr("Guides"), QLabel(str(len(doc.guides))))
        elif len(sel) == 1 and groups:
            self._group_form(form, groups[0])
        elif len(sel) == 1 and guides:
            g = doc.guide(guides[0][1])
            if g is not None:
                form.addRow(QLabel(tr("<b>Guide</b>") if g.is_line else tr("<b>Guide Point</b>")))
                form.addRow(tr("Point"), QLabel(f"{g.point[0]:.2f}, {g.point[1]:.2f}, {g.point[2]:.2f} mm"))
                if g.is_line:
                    d = g.direction
                    form.addRow(tr("Direction"), QLabel(f"{d[0]:.3f}, {d[1]:.3f}, {d[2]:.3f}"))
        elif len(sel) == 1 and faces:
            c, f = faces[0]
            ents = doc.ents(c)
            tri, _ = ents.face_triangles(f)
            area = float(np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1).sum() / 2)
            form.addRow(QLabel(tr("<b>Face</b>")))
            form.addRow(tr("Area"), QLabel(f"{area:.2f} mm²"))
            form.addRow(tr("Color"), QLabel(ents.F[f].color or tr("default")))
        elif len(sel) == 1 and edges:
            c, e = edges[0]
            ents = doc.ents(c)
            A, B = ents.edge_points(e)
            form.addRow(QLabel(tr("<b>Edge</b>")))
            form.addRow(tr("Length"), QLabel(f"{np.linalg.norm(B - A):.3f} mm"))
            form.addRow(tr("Faces"), QLabel(str(len(ents.e2f.get(e, ())))))
            form.addRow(tr("Softened"), QLabel(tr("yes") if e in ents.soft else tr("no")))
        else:
            form.addRow(QLabel(tr("<b>{count} items selected</b>").format(count=len(sel))))
            form.addRow(tr("Groups"), QLabel(str(len(groups))))
            form.addRow(tr("Faces"), QLabel(str(len(faces))))
            form.addRow(tr("Edges"), QLabel(str(len(edges))))
            if guides:
                form.addRow(tr("Guides"), QLabel(str(len(guides))))
            bb = ops.selection_bbox(doc)
            if bb is not None:
                d = bb[1] - bb[0]
                form.addRow(tr("Dimensions"), QLabel(f"{d[0]:.2f} × {d[1]:.2f} × {d[2]:.2f} mm"))
        self.lay.insertWidget(0, self.body)

    def _sheet_form(self, form):
        """Entity Info on the worksheet: the selected outlines and the sheet."""
        sheet = self.win.doc.sheet
        sel = self.win.sheet_view.selected_paths()
        paths = sel or sheet.paths
        if sel:
            form.addRow(QLabel(tr("<b>1 outline selected</b>") if len(sel) == 1
                               else tr("<b>{count} outlines selected</b>").format(count=len(sel))))
        else:
            form.addRow(QLabel(tr("<b>No selection</b>")))
            form.addRow(tr("Outlines"), QLabel(str(len(sheet.paths))))
        layers = sorted({p.layer for p in paths})
        if sel and len(layers) == 1:
            lay = sheet.layers.get(layers[0])
            op = OP_LABELS.get(lay.op, lay.op) if lay else ""
            form.addRow(tr("Layer"), QLabel(f"{layers[0]} ({op})" if op else layers[0]))
        elif layers:
            form.addRow(tr("Layers"), QLabel(str(len(layers))))
        if sel:
            closed = sum(1 for p in sel if p.closed)
            form.addRow(tr("Closed"), QLabel(tr("{closed} of {total}").format(closed=closed,
                                                                            total=len(sel))))
            pts = [p.pts for p in sel if len(p.pts)]
            if pts:
                P = np.vstack(pts)
                lo, hi = P.min(axis=0), P.max(axis=0)
                form.addRow(tr("Dimensions"), QLabel(f"{hi[0] - lo[0]:.2f} × {hi[1] - lo[1]:.2f} mm"))
                form.addRow(tr("Position"), QLabel(f"{lo[0]:.2f}, {lo[1]:.2f} mm"))
        form.addRow(QLabel(tr("<b>Sheet</b>")))
        if sheet.sheet_w and sheet.sheet_h:
            form.addRow(tr("Size"), QLabel(f"{sheet.sheet_w:g} × {sheet.sheet_h:g} mm"))
            form.addRow(tr("Count"), QLabel(str(int(sheet.sheet_count or 1))))
        else:
            form.addRow(tr("Size"), QLabel(tr("not set")))
        form.addRow(tr("Thickness"), QLabel(f"{sheet.sheet_thickness:g} mm" if sheet.sheet_thickness
                                            else tr("not set")))

    def _group_form(self, form, g):
        doc = self.win.doc
        form.addRow(QLabel(tr("<b>Group (printable part)</b>")))
        name = QLineEdit(g.name)

        def rename():
            if name.text() and name.text() != g.name:
                self._apply(lambda: setattr(doc.group(g.id), "name", name.text()))
        name.editingFinished.connect(rename)
        form.addRow(tr("Name"), name)
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
        form.addRow(tr("Filament"), combo)
        bb = g.bbox()
        if bb is None:
            return
        lo, hi = bb
        size = hi - lo
        spins = []
        for label, v in zip((tr("Width (X)"), tr("Depth (Y)"), tr("Height (Z)")), size):
            sp = QDoubleSpinBox()
            sp.setRange(0.01, 100000)
            sp.setDecimals(2)
            sp.setSuffix(" mm")
            sp.setValue(v)
            spins.append(sp)
            form.addRow(label, sp)
        lock = QCheckBox(tr("Keep proportions"))
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
        form.addRow(tr("Position"), QLabel(", ".join(f"{round(v, 2) + 0.0:.2f}" for v in lo) + " mm"))
        ents, version, key = g.ents, g.ents.version, (g.name, g.color)
        cached = self._part_cache.get(id(ents))
        if cached is not None and cached[0] is ents and cached[1:3] == (version, key):
            self._part_rows(form, cached[3])
            return
        if len(ents.F) < PART_INFO_BACKGROUND_FACES:
            summary = ops.part_summary(g.name, g.color, ents)
            self._remember_part(ents, version, key, summary)
            self._part_rows(form, summary)
            return
        # a big group: the check triangulates every face, seconds on an imported scan, so
        # it runs in a worker process and the panel is refreshed when it is done
        form.addRow(tr("Printable"), QLabel(tr("Calculating…")))
        if (id(ents), version, key) in self._part_pending:
            return                       # already on its way
        self._part_pending.add((id(ents), version, key))
        body, name, color = self.body, g.name, g.color

        def task(report):
            return ops.part_summary(name, color, ents, pool=True)

        def finish():
            self._part_pending.discard((id(ents), version, key))

        def done(summary):
            finish()
            if ents.version != version:
                return                   # edited meanwhile: a newer refresh follows
            self._remember_part(ents, version, key, summary)
            if self.body is not None and not self.body.isHidden():
                self.refresh()           # shows it now, if this group is still selected

        background.run(None, task, text=tr("Calculating…"), can_cancel=False,
                       show_dialog=False, on_done=done, on_error=lambda e: finish())

    def remember_part(self, g, summary):
        """The Entity Info check of `g` as a background task already made it (an import,
        a simplify), so selecting the group needs no second check."""
        self._remember_part(g.ents, g.ents.version, (g.name, g.color), summary)

    def _remember_part(self, ents, version, key, summary):
        self._part_cache[id(ents)] = (ents, version, key, summary)
        while len(self._part_cache) > 16:          # it holds the geometry: only a few
            del self._part_cache[next(iter(self._part_cache))]

    @staticmethod
    def _part_rows(form, summary):
        """Volume, surface and whether the group prints, as `summary` gives them."""
        vol, area, problem = summary
        if problem is None:
            form.addRow(tr("Volume"), QLabel(f"{vol / 1000:.3f} cm³"))
            form.addRow(tr("Surface"), QLabel(f"{area / 100:.2f} cm²"))
            ok = QLabel(tr("✔ closed volume (solid)"))
            ok.setStyleSheet(f"color: {theme.color('ok')}; font-weight: bold;")
        else:
            ok = QLabel(f"✖ {problem}")
            ok.setStyleSheet(f"color: {theme.color('error')}; font-weight: bold;")
            ok.setWordWrap(True)
        form.addRow(tr("Printable"), ok)

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
            it = QListWidgetItem(tr("Loose geometry ({n} faces)").format(n=len(doc.root.F)))
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
    """Tray panel "Print Check": why something is not a valid volume (Solid Inspector).

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
        self.b_check = QPushButton(tr("Check"))
        self.b_check.clicked.connect(self.refresh)
        self.b_fix = QPushButton(tr("Repair All"))
        self.b_fix.clicked.connect(self.repair)
        row.addWidget(self.b_check)
        row.addWidget(self.b_fix)
        lay.addLayout(row)
        self.summary = QLabel(tr("Click Check to test every part for a closed volume."))
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
            head.setForeground(0, theme.qcolor("error"))
            self.tree.addTopLevelItem(head)
            for n in by_part[key]:
                p = self.problems[n]
                it = QTreeWidgetItem([p.text()])
                it.setData(0, Qt.UserRole, n)
                it.setToolTip(0, tr("{kind}: {detail}\n{faces} faces, {edges} edges").format(
                    kind=p.kind, detail=p.detail, faces=len(p.faces), edges=len(p.edges)))
                head.addChild(it)
            head.setExpanded(True)
        if not order:
            ok = QTreeWidgetItem([tr("✔ every part is a closed volume")])
            ok.setForeground(0, theme.qcolor("ok"))
            self.tree.addTopLevelItem(ok)
        self.tree.blockSignals(False)
        if order:
            bad = len({p.part for p in self.problems})
            self.summary.setText(tr("{count} part(s) with problems. Click a problem to see it; "
                                    "double-click zooms to it.").format(count=bad))
        else:
            self.summary.setText(tr("All parts are closed volumes and can be printed."))
        self.win.view.problem_items = []
        self.win.view.update()

    def repair(self, checked=False):
        """Repair All: one undo step that closes and orients the parts."""
        fixed = ops.repair_all(self.win.doc)
        self.refresh()
        self.win.hint.setText(tr("Print Check: parts repaired (Undo reverts it).")
                              if fixed else tr("Print Check: there was nothing to repair."))

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


# ---------------------------------------------------------------- layers of the 2D worksheet

class LayerParamsDialog(QDialog):
    """The "Layer settings" dialog: what the machine should do with that layer.

    A laser layer gets power, speed, number of passes and the kerf; a mill layer the
    tool diameter, the depth, the step depth, the stepover and the corner relief for
    the inside corners. The values are stored in `Layer.params` and so travel along in
    the file and in the SVG export.
    """

    SUFFIX = {"power": " %", "stepover": " %", "speed": " mm/s"}

    def __init__(self, layer, parent=None):
        super().__init__(parent)
        self.layer = layer
        self.op = getattr(layer, "op", "cut")
        self.fields = {}
        self.setWindowTitle(tr("Settings of Layer {name}").format(name=layer.name))
        form = QFormLayout(self)
        form.addRow(QLabel(tr("Operation: {op}").format(op=OP_LABELS.get(self.op, self.op))))
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
            form.addRow(note(tr("This layer is for information only and is not exported.")))
        elif self.op in MILL_OPS:
            form.addRow(note(tr("The compensation shifts the outline sideways by half the tool "
                                "diameter. A pocket runs concentric paths with the stepover until "
                                "the area is cleared. A corner relief gives the cutter room in the "
                                "inside corners.")))
        else:
            form.addRow(note(tr("The kerf compensation moves outer edges out by half the kerf and "
                                "makes holes smaller by the same amount, so the part has the drawn "
                                "size after cutting.")))
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
    """Tray panel "Layers": color, name, operation, visibility and settings per layer.

    One row per layer of the 2D worksheet. Click the color box to choose a color (red
    cuts, blue engraves, as laser software expects), double-click the name to rename,
    choose the operation in the combo box, and turn a layer off with the eye or lock it
    with the padlock. New paths go on the active layer; you choose it by clicking a row.
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
        lay.addWidget(QLabel(tr("Layers of the 2D worksheet (click a row for the active layer):")))
        self.table = QTableWidget(0, self.COLS)
        self.table.setHorizontalHeaderLabels(["", "", "", tr("Layer"), tr("Operation"), ""])
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
        self.b_add = QPushButton(tr("Add"))
        self.b_add.clicked.connect(self.add_layer)
        self.b_remove = QPushButton(tr("Delete Layer"))
        self.b_remove.clicked.connect(self.remove_layer)
        row.addWidget(self.b_add)
        row.addWidget(self.b_remove)
        lay.addLayout(row)
        row = QHBoxLayout()
        self.b_active = QPushButton(tr("Active Layer"))
        self.b_active.clicked.connect(self.pick_active)
        self.b_params = QPushButton(tr("Settings…"))
        self.b_params.clicked.connect(self.edit_params)
        row.addWidget(self.b_active)
        row.addWidget(self.b_params)
        lay.addLayout(row)
        self.info = QLabel("")
        self.info.setWordWrap(True)
        lay.addWidget(self.info)
        self.refresh()

    # ------------------------------------------------------------ building

    @contextmanager
    def _quiet(self):
        """No reaction to our own changes in the table (they come from the model)."""
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
                it.setToolTip(tr("Double-click to rename"))
                self.table.setItem(row, 3, it)
                self.table.setCellWidget(row, 0, self._button(
                    row, "eye" if lay.visible else "eye_off",
                    tr("Hide layer (is not exported)") if lay.visible
                    else tr("Show layer"), self.toggle_visible))
                self.table.setCellWidget(row, 1, self._button(
                    row, "lock" if lay.locked else "unlock",
                    tr("Lock layer (can no longer be selected)") if not lay.locked
                    else tr("Unlock layer"), self.toggle_locked))
                self.table.setCellWidget(row, 2, self._button(
                    row, "paint", tr("Layer color ({color})").format(color=lay.color), self.pick_color))
                self.table.setCellWidget(row, 4, self._op_box(row, lay))
                self.table.setCellWidget(row, 5, self._button(
                    row, "check", tr("Settings of this layer"), self.edit_params))
            self.table.selectRow(max(0, self.names.index(active)) if active in self.names else 0)
        self.info.setText(self._summary(sheet, active))

    def _summary(self, sheet, active) -> str:
        lay = sheet.layers.get(active)
        if lay is None:
            return ""
        n = sum(1 for p in sheet.paths if p.layer == active)
        if lay.op in MILL_OPS:
            details = tr("cutter Ø{tool:g} mm, depth {depth:g} mm").format(
                tool=layer_param(lay, 'tool_d'), depth=layer_param(lay, 'depth'))
        elif lay.op in LASER_OPS:
            details = tr("{power:g} %, {speed:g} mm/s, {passes}×, kerf {kerf:g} mm").format(
                power=layer_param(lay, 'power'), speed=layer_param(lay, 'speed'),
                passes=layer_param(lay, 'passes'), kerf=layer_param(lay, 'kerf'))
        else:
            details = tr("not exported")
        return tr("Active layer: {name} — {op}; {n} outlines; {details}.").format(
            name=active, op=OP_LABELS.get(lay.op, lay.op), n=n, details=details)

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
        box.setToolTip(tr("What the machine does with this layer"))
        box.currentIndexChanged.connect(lambda _=0, r=row, b=box: self.set_op(r, b))
        return box

    def _layer(self, row):
        try:
            return self.win.doc.sheet.layers[self.names[row]]
        except (IndexError, KeyError):
            return None

    def _edit(self, change, message):
        """One undo step on the worksheet, then update everything."""
        self.win.doc.checkpoint()
        change()
        self.win.doc.notify()
        self.refresh()
        self.win.hint.setText(message)

    # ------------------------------------------------------------ buttons

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
            self.win.hint.setText(tr("Layer {name} is locked; it cannot be the active layer now.")
                                  .format(name=lay.name))
            return
        self.win.sheet_view.set_current_layer(lay.name)
        self.refresh()
        self.win.hint.setText(tr("New outlines go to layer {name}.").format(name=lay.name))

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
            self.win.hint.setText(tr("There is already a layer named {name}.").format(name=new))
            self.refresh()
            return
        with self._quiet():
            self.win.doc.checkpoint()
            sheet.rename_layer(old, new)
            if self.win.sheet_view.current_layer == old:
                self.win.sheet_view.current_layer = new
            self.win.doc.notify()
        self.refresh()
        self.win.hint.setText(tr("Layer {old} is now called {new}.").format(old=old, new=new))

    def add_layer(self):
        sheet = self.win.doc.sheet
        name = sheet.free_layer_name()
        color = ["#00a0a0", "#a000a0", "#0000a0", "#a0a000", "#00a000"][len(sheet.layers) % 5]

        def change():
            sheet.ensure_layer(name, color, "cut")
        self._edit(change, tr("Layer {name} added; draw or move outlines onto it.").format(name=name))
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
            self.win.hint.setText(tr("Layer 0 is the base layer and cannot be removed."))
            return
        sheet = self.win.doc.sheet
        n = sum(1 for p in sheet.paths if p.layer == lay.name)
        if n:
            box = QMessageBox(self)
            box.setIcon(QMessageBox.Question)
            box.setWindowTitle(tr("Delete Layer {name}").format(name=lay.name))
            box.setText(tr("Layer {name} has {n} outlines.").format(name=lay.name, n=n))
            b_move = box.addButton(tr("Paths to layer 0"), QMessageBox.AcceptRole)
            b_del = box.addButton(tr("Delete paths too"), QMessageBox.DestructiveRole)
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
        if move and n:
            message = tr("Layer {name} deleted ({n} outlines to layer 0).")
        elif n:
            message = tr("Layer {name} deleted with {n} outlines.")
        else:
            message = tr("Layer {name} deleted.")
        self._edit(change, message.format(name=lay.name, n=n))

    def toggle_visible(self, row):
        lay = self._layer(row)
        if lay is None:
            return
        new = not lay.visible

        def change():
            lay.visible = new
        self._edit(change, (tr("Layer {name} visible again.") if new
                            else tr("Layer {name} hidden; it does not go to the machine."))
                   .format(name=lay.name))

    def toggle_locked(self, row):
        lay = self._layer(row)
        if lay is None:
            return
        new = not lay.locked

        def change():
            lay.locked = new
        self._edit(change, (tr("Layer {name} locked: you can no longer click its outlines.") if new
                            else tr("Layer {name} unlocked.")).format(name=lay.name))

    def pick_color(self, row):
        lay = self._layer(row)
        if lay is None:
            return
        color = QColorDialog.getColor(QColor(lay.color or "#000000"), self,
                                      tr("Color of Layer {name}").format(name=lay.name))
        if not color.isValid():
            return

        new = color.name().lower()

        def change():
            lay.color = new
        self._edit(change, tr("Layer {name} is now {color}.").format(name=lay.name, color=new))

    def set_op(self, row, box):
        lay = self._layer(row)
        if lay is None or self._busy:
            return
        op = box.currentData()
        if op == lay.op:
            return

        def change():
            lay.op = op
        self._edit(change, tr("Layer {name}: {op}.").format(name=lay.name, op=OP_LABELS.get(op, op)))

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
        self._edit(change, tr("Settings of layer {name} changed.").format(name=lay.name))


# ---------------------------------------------------------------- dialogs

SHORTCUTS_KEY = "shortcuts/"
SHOW_WELCOME_KEY = "show_welcome"
# the group of the commands that are only in the right-click menu; English, as it is part
# of the settings keys (shown translated)
OTHER_COMMANDS = tr_noop("Other")
SOURCE_TEXT = "source_text"   # QAction property: the English text, when tr() can't tell


def qsettings():
    """The application settings, the same store as Settings.q."""
    return QSettings("Volura", "Volura")


def clean_text(text):
    """Menu text without the & accelerator and without a trailing ellipsis."""
    return text.replace("&", "").replace("…", "").replace("...", "").strip()


def source_text_of(action):
    """The English text of an action or menu (with & and …), for language-independent keys."""
    text = action.property(SOURCE_TEXT)
    if text:
        return str(text)
    text = action.text()
    return text if i18n.language() == "en" else i18n.source_text(text)


_TABLES = None


def translation_tables():
    """The translation tables of the other languages, to recognise their old settings keys."""
    global _TABLES
    if _TABLES is None:
        _TABLES = [t for t in (i18n.table(code) for code, _name in i18n.LANGUAGES if code != "en")
                   if t]
    return _TABLES


def welcome_enabled():
    """Help > Welcome to Volura at startup, unless the user unchecked the box."""
    return qsettings().value(SHOW_WELCOME_KEY, True) in (True, "true", "1", 1)


UPDATE_DELAY_MS = 4000             # not right at startup: the window comes first
POOL_PREWARM_MS = 2000             # boot the worker processes shortly after the window shows
UPDATE_INTERVAL = 24 * 3600        # automatically at most once a day
UPDATE_LAST_KEY = "update/last_check"
UPDATE_SKIP_KEY = "update/skip"


class _Bridge(QObject):
    """Brings the outcome of a background thread back to the main thread."""

    done = Signal(object, object)          # result, error
    progress = Signal(int, int)


def run_in_background(work, on_done, on_progress=None):
    """`work(progress)` in a thread; `on_done(result, error)` in the main thread."""
    bridge = _Bridge()
    bridge.done.connect(on_done)
    if on_progress is not None:
        bridge.progress.connect(on_progress)

    def run():
        try:
            result = work(bridge.progress.emit)
        except Exception as e:               # network errors belong in the message, not in a crash
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
AUTOSAVE_INTERVAL_MS = 5 * 60 * 1000      # autosave, like SketchUp


def autosave_path():
    """<AppDataLocation>/autosave.volura: the model of the last session."""
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
            p.drawText(box, Qt.AlignCenter, tr("no outlines"))
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
    """The 'Export 2D Vector' dialog: what to export and a small preview.

    The choices: the selected faces (only when there are faces), the top view or a
    section at height Z -- of the selection or of everything. The height is a number
    with a unit as in the Measurements box (12 or 0.5 in) and starts halfway up the
    model. The preview shows what will be written.
    """

    def __init__(self, win):
        super().__init__(win)
        self.win = win
        self.setWindowTitle(tr("Export 2D Vector"))
        form = QFormLayout(self)
        faces = [i for i in win.doc.selection if i[0] == "f"]

        self.b_faces = QRadioButton(tr("Selected faces ({n})").format(n=len(faces)))
        self.b_faces.setEnabled(bool(faces))
        self.b_faces.setToolTip(tr("Every face at true size, in its own plane"))
        self.b_top = QRadioButton(tr("Top view (the outline from above)"))
        self.b_section = QRadioButton(tr("Section at height Z"))
        (self.b_faces if faces else self.b_top).setChecked(True)
        box = QGroupBox(tr("What to export"))
        col = QVBoxLayout(box)
        for b in (self.b_faces, self.b_top, self.b_section):
            col.addWidget(b)

        self.z_edit = QLineEdit(format_length(vector_export.section_z_default(win.doc)))
        self.z_edit.setToolTip(tr("Height of the section, for example 12 or 0.5 in"))
        self.z_edit.setFixedWidth(90)

        self.only = QCheckBox(tr("Selection only"))
        self.only.setToolTip(tr("Only the selected groups; otherwise the whole model"))

        self.preview = VectorPreview()

        form.addRow(box)
        form.addRow(tr("Height Z"), self.z_edit)
        form.addRow("", self.only)
        form.addRow(tr("Preview"), self.preview)
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
        """The choices as a dict: source, selection only and the height of the section."""
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
            self.z_edit.setStyleSheet(f"border: 1px solid {theme.color('invalid')};")
        self.preview.set_drawing(self.win.vector_export_drawing(self.options()))


class SheetExportDialog(QDialog):
    """The "Export for Laser/Mill" dialog: which layers may go and which compensation.

    By default the visible layers that mean something to the machine go along (except
    "None", which is for information only). The kerf/cutter compensation is on as soon
    as a mill layer goes along — a cutter has its own diameter — and off for laser,
    because laser software often does that compensation itself. With "a separate file
    per layer" every layer gets its own file, as some cutting software likes. The
    preview shows what will be written.
    """

    def __init__(self, win, kind="svg"):
        super().__init__(win)
        self.win = win
        self.kind = "dxf" if str(kind).lower() == "dxf" else "svg"
        self._busy = False
        self.setWindowTitle(tr("Export for Laser/Mill"))
        self.resize(430, 520)
        lay = QVBoxLayout(self)
        lay.addWidget(QLabel(tr("Layers that go to the machine:")))
        self.list = QListWidget()
        self.list.setIconSize(QSize(16, 16))
        lay.addWidget(self.list, 1)
        self.compensate = QCheckBox(tr("Apply kerf/cutter compensation"))
        self.compensate.setToolTip(tr("Move outlines sideways by half the kerf or the cutter "
                                      "radius, so the workpiece gets the drawn size"))
        self.per_layer = QCheckBox(tr("A separate file per layer"))
        self.per_layer.setToolTip(tr("Every layer gets its own file: name_layername.{ext}")
                                  .format(ext=self.kind))
        self.per_sheet = QCheckBox(tr("A separate file per sheet"))
        self.per_sheet.setToolTip(tr("Every sheet becomes its own file with that sheet as the page "
                                     "and its origin as the zero point: name_sheet1.svg ..."))
        lay.addWidget(self.compensate)
        lay.addWidget(self.per_layer)
        lay.addWidget(self.per_sheet)
        self.warning = QLabel("")
        self.warning.setWordWrap(True)
        self.warning.setStyleSheet(f"color: {theme.color('invalid')};")
        lay.addWidget(self.warning)
        self.preview = VectorPreview()
        lay.addWidget(QLabel(tr("Preview of what is exported:")))
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
                it.setToolTip(tr("This layer is for information only and normally does not go along"))
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
        except Exception:                       # the preview must never break a dialog
            self.preview.set_drawing(None)
        try:
            n = len(nesting.outside_paths(self.win.doc.sheet))
        except Exception:
            n = 0
        self.warning.setText("" if not n else
                             tr("Note: {n} paths lie outside the sheet or in the margin and go "
                                "to the machine like that.").format(n=n))


class SheetSettingsDialog(QDialog):
    """The "Sheet…" dialog: preset sizes or custom, plus thickness, margin, gap and count."""

    PRESETS = [("A4 — 297 × 210", 297.0, 210.0),
               ("A3 — 420 × 297", 420.0, 297.0),
               ("300 × 200", 300.0, 200.0),
               ("400 × 300", 400.0, 300.0),
               ("600 × 400", 600.0, 400.0),
               (tr("1220 × 610 (half sheet)"), 1220.0, 610.0),
               (tr("2440 × 1220 (full sheet)"), 2440.0, 1220.0),
               (tr("Custom size"), None, None)]

    def __init__(self, win):
        super().__init__(win)
        self.win = win
        self.setWindowTitle(tr("Sheet"))
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
        self.thickness.setToolTip(tr("Thickness of the material (used by the milling paths)"))
        self.margin = QDoubleSpinBox()
        self.margin.setRange(0.0, 1000.0)
        self.margin.setDecimals(2)
        self.margin.setSuffix(" mm")
        self.margin.setToolTip(tr("Distance that stays free along the edge of the sheet"))
        self.gap = QDoubleSpinBox()
        self.gap.setRange(0.0, 1000.0)
        self.gap.setDecimals(2)
        self.gap.setSuffix(" mm")
        self.gap.setToolTip(tr("Minimum distance between two parts"))
        self.count = QSpinBox()
        self.count.setRange(1, 99)
        self.count.setToolTip(tr("Number of sheets side by side on the worksheet"))
        thick_row = QHBoxLayout()
        self.thick_off = QCheckBox(tr("no thickness"))
        self.thick_off.toggled.connect(self.thickness.setDisabled)
        thick_row.addWidget(self.thickness)
        thick_row.addWidget(self.thick_off)
        lay.addRow(tr("Preset"), self.preset)
        lay.addRow(tr("Width"), self.w)
        lay.addRow(tr("Height"), self.h)
        lay.addRow(tr("Material thickness"), thick_row)
        lay.addRow(tr("Margin along the edge"), self.margin)
        lay.addRow(tr("Gap"), self.gap)
        lay.addRow(tr("Number of sheets"), self.count)
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
    """The "Auto Arrange" dialog: rotation, gap, margin and new sheets."""

    def __init__(self, win):
        super().__init__(win)
        self.win = win
        sheet = win.doc.sheet
        self.setWindowTitle(tr("Auto Arrange"))
        lay = QFormLayout(self)
        self.rot_all = QRadioButton(tr("0°, 90°, 180° and 270°"))
        self.rot_none = QRadioButton(tr("0° only (do not rotate)"))
        self.rot_all.setChecked(True)
        rot_row = QHBoxLayout()
        rot_row.addWidget(self.rot_all)
        rot_row.addWidget(self.rot_none)
        lay.addRow(tr("Allow rotation"), rot_row)
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
        self.add_sheets = QCheckBox(tr("add new sheets if it does not fit"))
        self.add_sheets.setChecked(True)
        lay.addRow(tr("Gap"), self.gap)
        lay.addRow(tr("Margin"), self.margin)
        lay.addRow(tr("Sheets"), self.add_sheets)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addRow(bb)

    def options(self) -> dict:
        return {"rotations": (0, 90, 180, 270) if self.rot_all.isChecked() else (0,),
                "gap": round(self.gap.value(), 6), "margin": round(self.margin.value(), 6),
                "add_sheets": self.add_sheets.isChecked()}


class BoxJointDialog(QDialog):
    """The "Box with Finger Joints…" dialog: sizes, material, fingers, kerf and lid.

    The two buttons at the bottom both do something with the sizes: "On Worksheet" puts
    the cutting outlines on the 2D worksheet and arranges them on the sheet, "Preview in
    3D" puts the panels extruded into the model. The thickness comes from the sheet
    settings and the kerf from the "Cut" layer; the finger width follows the thickness
    (2.5 ×) as long as it was not chosen by the user.
    """

    SHEET = "sheet"
    MODEL = "model"

    def __init__(self, win):
        super().__init__(win)
        self.win = win
        self.mode = self.SHEET
        self._sync = False
        self._finger_fixed = False
        self.setWindowTitle(tr("Box with Finger Joints"))
        sheet = win.doc.sheet
        t = float(sheet.sheet_thickness or 0.0) or boxgen.DEFAULT_THICKNESS
        kerf = float(layer_param(sheet.layers.get(sheet.default_layer("cut")), "kerf") or 0.0)
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
        self.kind.addItem(tr("inside dimensions (the space inside the box)"), True)
        self.kind.addItem(tr("outside dimensions (the size of the box)"), False)
        self.t = spin(0.1, 100.0, t, 0.1)
        self.t.setToolTip(tr("Thickness of the panel; the fingers are as deep as the thickness"))
        self.finger = spin(0.5, 200.0, boxgen.FINGER_FACTOR * t, 0.1)
        self.finger.setToolTip(tr("Desired finger width; the number of fingers becomes the "
                                  "largest odd number that fits"))
        self.kerf = spin(0.0, 5.0, kerf, 0.05)
        self.kerf.setToolTip(tr("Width of the cut gap: it makes the fingers wider and the "
                                "slots narrower, so the joint fits tightly"))
        self.lid = QComboBox()
        for label, key in ((tr("no lid (open box)"), "none"), (tr("fixed lid"), "closed")):
            self.lid.addItem(label, key)
        self.labels = QCheckBox(tr("engrave the panel name small in the middle"))
        self.labels.setChecked(True)
        self.info = QLabel()
        self.info.setWordWrap(True)
        lay.addRow(tr("Width"), self.w)
        lay.addRow(tr("Depth"), self.d)
        lay.addRow(tr("Height"), self.h)
        lay.addRow(tr("Sizes"), self.kind)
        lay.addRow(tr("Material thickness"), self.t)
        lay.addRow(tr("Finger width"), self.finger)
        lay.addRow(tr("Kerf"), self.kerf)
        lay.addRow(tr("Lid"), self.lid)
        lay.addRow("", self.labels)
        lay.addRow(self.info)
        bb = QDialogButtonBox(QDialogButtonBox.Cancel)
        to_sheet = QPushButton(tr("On Worksheet"))
        to_sheet.setToolTip(tr("Cut the panels on the worksheet and arrange them on the sheet "
                               "automatically"))
        to_model = QPushButton(tr("Preview in 3D"))
        to_model.setToolTip(tr("Put the panels extruded into the model, the way the glued box "
                               "looks"))
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
        """The finger width follows 2.5 × the thickness, unless the user chose it."""
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
            self.info.setText(tr("Box of {w:g} × {d:g} × {h:g} mm, fingers of {step:.2f} mm "
                                 "({n} along the longest edge).").format(w=W, d=D, h=H, step=step,
                                                                          n=n))
        except (ValueError, ZeroDivisionError) as e:
            self.info.setText(str(e))

    def options(self) -> dict:
        return {"w": round(self.w.value(), 6), "d": round(self.d.value(), 6),
                "h": round(self.h.value(), 6), "inner": bool(self.kind.currentData()),
                "t": round(self.t.value(), 6), "finger_w": round(self.finger.value(), 6),
                "kerf": round(self.kerf.value(), 6), "lid": str(self.lid.currentData()),
                "labels": self.labels.isChecked(), "mode": self.mode}


class SimplifyDialog(QDialog):
    """Tools > Simplify Model: a quality slider and a live estimate of the result.

    The slider has the fixed steps of ops.SIMPLIFY_STEPS; under it the maximum deviation
    in mm and what that step does. The "N triangles -> M triangles" estimate runs in the
    background (no progress dialog, so the slider never stutters on a big model) and only
    the last calculation counts: an older one that finishes later is ignored. OK returns
    the chosen tolerance; the window does the actual simplification."""

    def __init__(self, win, groups):
        super().__init__(win)
        self.win = win
        self.groups = list(groups)
        self.tolerance = ops.SIMPLIFY_DEFAULT_TOLERANCE
        self.preview_before = self.preview_after = None
        self.manifolds = {}             # group id -> its closed manifold (None: not closed)
        self._manifold_lock = threading.Lock()
        self._token = 0
        self._job = None
        self.setWindowTitle(tr("Simplify Model"))
        lay = QVBoxLayout(self)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, len(ops.SIMPLIFY_STEPS) - 1)
        self.slider.setPageStep(1)
        self.slider.setValue(self._default_index())
        lay.addWidget(self.slider)

        self.tol_label = QLabel()
        lay.addWidget(self.tol_label)
        self.desc_label = QLabel()
        self.desc_label.setWordWrap(True)
        lay.addWidget(self.desc_label)
        self.preview_label = QLabel()
        self.preview_label.setWordWrap(True)
        lay.addWidget(self.preview_label)
        if any(ops.has_painted_faces(g) for g in self.groups):
            warn = QLabel(tr("Painted face colors are lost; the parts keep the group color."))
            warn.setWordWrap(True)
            lay.addWidget(warn)

        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)

        self.slider.valueChanged.connect(self._step_changed)
        self._step_changed(self.slider.value())

    @staticmethod
    def _default_index():
        for i, (tol, _d) in enumerate(ops.SIMPLIFY_STEPS):
            if tol == ops.SIMPLIFY_DEFAULT_TOLERANCE:
                return i
        return len(ops.SIMPLIFY_STEPS) // 2

    def _step_changed(self, index):
        tol, desc = ops.SIMPLIFY_STEPS[index]
        self.tolerance = tol
        self.tol_label.setText(tr("Maximum deviation: {tol} mm").format(tol=tol))
        self.desc_label.setText(tr(desc))
        self._estimate()

    def _estimate(self):
        """Estimate the triangle count after simplifying, off the GUI thread."""
        if self._job is not None and not self._job.done:
            self._job.cancel()
        self._token += 1
        token = self._token
        self.preview_before = self.preview_after = None
        self.preview_label.setText(tr("Calculating…"))
        groups, tol = list(self.groups), self.tolerance
        manifolds, lock = self.manifolds, self._manifold_lock

        def task(report):
            report(0.0)
            before = after = 0
            for g in groups:
                # the manifold of a group is made once (seconds on a big model); every
                # step of the slider after that is only manifold's own simplify
                with lock:              # a quick second step waits instead of doing it again
                    if g.id not in manifolds:
                        try:
                            manifolds[g.id] = ops.closed_manifold(g)
                        except ValueError:
                            manifolds[g.id] = None
                m = manifolds[g.id]
                if m is None:
                    continue
                before += m.num_tri()
                after += m.simplify(tol).num_tri()
            return before, after

        def done(result):
            if token == self._token:
                self._show_estimate(*result)

        self._job = background.run(None, task, text=tr("Calculating…"), can_cancel=False,
                                   show_dialog=False, on_done=done)

    def _show_estimate(self, before, after):
        self.preview_before, self.preview_after = before, after
        if before <= 0:
            self.preview_label.setText(tr("Nothing to simplify."))
            return
        pct = round(100 * (before - after) / before)
        self.preview_label.setText(
            tr("{before} triangles -> {after} triangles ({pct} % fewer)")
            .format(before=before, after=after, pct=pct))


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
            if page not in self.page_titles:     # the English title, or one in another language
                english = [i18n.source_text(t) for t in self.page_titles]
                page = (tr(page) if tr(page) in self.page_titles
                        else self.page_titles[english.index(i18n.source_text(page))])
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
    """Window > Model Info: Units, Statistics and Print Bed (SketchUp 2017)."""

    PAGES = (tr("Units"), tr("Statistics"), tr("Print Bed"))
    SIZE = (660, 430)

    def __init__(self, win, page=0):
        super().__init__(win, tr("Model Info"), self.PAGES, page)
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
        box = QGroupBox(tr("Length"))
        inner = QFormLayout(box)
        self.snap_on = QCheckBox(tr("Enable length snapping"))
        self.snap_on.setChecked(snap > 0)
        self.snap = QDoubleSpinBox()
        self.snap.setRange(0.01, 100)
        self.snap.setDecimals(2)
        self.snap.setSingleStep(0.5)
        self.snap.setSuffix(" mm")
        self.snap.setValue(snap or 1.0)
        self.snap.setEnabled(snap > 0)
        self.snap_on.toggled.connect(self.snap.setEnabled)
        self.snap.setToolTip(tr("Lengths that do not snap to a point or edge are rounded to this, "
                                "like 'Length Snapping' in SketchUp."))
        inner.addRow(self.snap_on)
        inner.addRow(tr("Round to"), self.snap)
        self.precision = QComboBox()
        self.precision.addItems(["0", "0.0", "0.00"])
        self.precision.setCurrentIndex(min(self.win.settings.length_precision, 2))
        self.sample = QLabel()
        self.precision.currentIndexChanged.connect(self._show_sample)
        self._show_sample()
        inner.addRow(tr("Display precision"), self.precision)
        inner.addRow(tr("Sample"), self.sample)
        form.addRow(box)
        form.addRow(note(tr("The display precision is the number of decimals of the Measurements "
                            "input and of the dimensions in the drawing window, like 'Precision' "
                            "in SketchUp.")))
        form.addRow(note(tr("The print bed and the printer are on the Print Bed page.")))
        self.add_page(page)

    def _show_sample(self):
        self.sample.setText(format_length(123.456, self.precision.currentIndex()))

    def _build_statistics(self):
        page = QWidget()
        form = QFormLayout(page)
        self.stats = {}
        # keyed by the English label, shown translated
        for label in (tr_noop("Groups"), tr_noop("Loose faces"), tr_noop("Loose edges"),
                      tr_noop("Total volume")):
            value = QLabel()
            form.addRow(tr(label), value)
            self.stats[label] = value
        form.addRow(note(tr("The volume is the sum of all closed parts: every group and every "
                            "loose shape that forms a closed volume.")))
        self.add_page(page)

    def refresh_statistics(self):
        doc = self.win.doc
        self.stats["Groups"].setText(str(len(doc.groups)))
        self.stats["Loose faces"].setText(str(len(doc.root.F)))
        self.stats["Loose edges"].setText(str(len(doc.root.E)))
        volume = sum(p.manifold.volume() for p in ops.parts(doc) if p.manifold is not None)
        self.stats["Total volume"].setText(f"{volume / 1000:.3f} cm³")

    def _build_bed(self):
        page = QWidget()
        form = QFormLayout(page)
        self.printer = QComboBox()
        self.printer.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.printer.addItem(tr("Custom"), None)
        for name, size in PRINTERS:
            self.printer.addItem(f"{name}  ({size[0]}×{size[1]}×{size[2]})", size)
        self.bed_spins = []
        for label, v in zip((tr("Bed X"), tr("Bed Y"), tr("Height Z")), self.win.settings.bed_size):
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
        form.insertRow(0, tr("Printer"), self.printer)
        form.addRow(note(tr("The bed sets the grid of the drawing area and the limits of the "
                            "print check.")))
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
    """Window > Preferences: General, Display, Drawing, 3D Mouse and Shortcuts (SketchUp 2017)."""

    PAGES = (tr("General"), tr("Display"), tr("Drawing"), tr("3D Mouse"), tr("Shortcuts"))
    SIZE = (680, 470)
    SPEED_STEPS = 20                       # slider 1..40 = speed 0.05..2 times

    def __init__(self, win, page=0):
        super().__init__(win, tr("Preferences"), self.PAGES, page)
        self._build_general()
        self._build_looks()
        self._build_drawing()
        self._build_spacemouse()
        self.shortcuts = ShortcutsPanel(win)
        self.add_page(self.shortcuts)

    def _build_general(self):
        page = QWidget()
        form = QFormLayout(page)
        self.segments = QSpinBox()
        self.segments.setRange(8, 512)
        self.segments.setValue(self.win.settings.circle_segments)
        self.segments.setToolTip(tr("Number of segments of a circle, arc or polygon. While "
                                    "drawing, type for example 32s to change it for that shape."))
        form.addRow(tr("Circle segments"), self.segments)
        form.addRow(note(tr("The more segments, the rounder a circle looks and the longer "
                            "drawing and exporting take.")))
        self.updates = QCheckBox(tr("Check for updates at startup"))
        self.updates.setChecked(self.win.settings.check_updates)
        self.updates.setToolTip(tr("Once a day Volura checks on GitHub whether there is a newer "
                                   "version."))
        form.addRow(self.updates)
        self.ask_simplify = QCheckBox(tr("Ask to simplify a large import"))
        self.ask_simplify.setChecked(self.win.settings.ask_simplify_import)
        self.ask_simplify.setToolTip(tr("After importing a model with many triangles, ask "
                                        "whether to simplify it to keep Volura fast."))
        form.addRow(self.ask_simplify)
        self.language = QComboBox()
        self.language.addItem(tr("System"), i18n.SYSTEM)
        for code, name in i18n.LANGUAGES:       # language names stay in their own language
            self.language.addItem(name, code)
        self.language.setCurrentIndex(max(0, self.language.findData(i18n.saved_choice())))
        self.language.setToolTip(tr("System uses the language of Windows (English if Volura "
                                    "does not have it)."))
        form.addRow(tr("Language"), self.language)
        form.addRow(note(tr("Takes effect after restarting Volura.")))
        self.add_page(page)

    def _build_looks(self):
        page = QWidget()
        form = QFormLayout(page)
        self.theme = QComboBox()
        for key, label in theme.CHOICES:
            self.theme.addItem(label, key)
        self.theme.setCurrentIndex(max(0, self.theme.findData(self.win.settings.theme)))
        self.theme.setToolTip(tr("System follows the Windows setting (light or dark)."))
        form.addRow(tr("Theme"), self.theme)
        form.addRow(note(tr("In the dark theme the sheet on the 2D worksheet stays light, so the "
                            "layer colors stay readable, and the 3D window gets a muted gray "
                            "background.")))
        self.add_page(page)

    def _build_spacemouse(self):
        page = QWidget()
        form = QFormLayout(page)
        s = self.win.settings
        self.sm_on = QCheckBox(tr("Use a 3Dconnexion 3D mouse"))
        self.sm_on.setChecked(s.spacemouse)
        form.addRow(self.sm_on)
        self.sm_speed = QSlider(Qt.Horizontal)
        self.sm_speed.setRange(1, 2 * self.SPEED_STEPS)
        self.sm_speed.setValue(round(s.spacemouse_speed * self.SPEED_STEPS))
        self.sm_speed.setToolTip(tr("How fast the model moves with the cap."))
        form.addRow(tr("Speed"), self.sm_speed)
        form.addRow(tr("Connection"), QLabel(self.win.spacemouse_status()))
        self.sm_live = QLabel(tr("move the cap to test"))
        form.addRow(tr("Motion"), self.sm_live)
        self.win.spacemouse.motion.connect(self._show_motion)
        self.win.spacemouse.command.connect(self._show_command)
        self.finished.connect(self._stop_live)
        form.addRow(note(tr("Works with the SpaceMouse (Wireless, Pro, Compact, Enterprise) and "
                            "older devices such as SpaceNavigator, SpaceExplorer and SpacePilot. "
                            "With 3DxWare installed you set up buttons and axes in the Volura "
                            "profile in 3DxWare; without a driver Volura reads the device itself.\n"
                            "Sliding the cap pans the model, pulling or pushing zooms, twisting and "
                            "tilting rotates the model. In 2D the cap pans and zooms the "
                            "worksheet.")))
        self.add_page(page)

    def _show_motion(self, m):
        self.sm_live.setText(tr("pan {tx:+.2f} {ty:+.2f} {tz:+.2f}   rotate {rx:+.2f} {ry:+.2f} "
                                "{rz:+.2f}").format(tx=m.tx, ty=m.ty, tz=m.tz,
                                                    rx=m.rx, ry=m.ry, rz=m.rz))

    def _show_command(self, name):
        self.sm_live.setText(tr("button: {name}").format(name=name))

    def _stop_live(self, *args):
        for signal, slot in ((self.win.spacemouse.motion, self._show_motion),
                             (self.win.spacemouse.command, self._show_command)):
            try:
                signal.disconnect(slot)
            except (RuntimeError, TypeError):
                pass

    def _build_drawing(self):
        page = QWidget()
        form = QFormLayout(page)
        self.split_lines = QCheckBox(tr("Split lines automatically"))
        self.split_lines.setChecked(True)
        self.split_lines.setEnabled(False)
        self.split_lines.setToolTip(tr("As in SketchUp: edges that cross always split each other, "
                                       "and a line across a face splits that face."))
        form.addRow(self.split_lines)
        form.addRow(note(tr("Volura always does this, like SketchUp; it is only shown here.")))
        self.add_page(page)

    def apply(self):
        s = self.win.settings
        s.circle_segments = self.segments.value()
        s.check_updates = self.updates.isChecked()
        s.ask_simplify_import = self.ask_simplify.isChecked()
        s.spacemouse_speed = self.sm_speed.value() / self.SPEED_STEPS
        sm_changed = s.spacemouse != self.sm_on.isChecked()
        s.spacemouse = self.sm_on.isChecked()
        theme_changed = s.theme != self.theme.currentData()
        s.theme = self.theme.currentData()
        # the language is read once at startup (i18n.saved_choice), so it applies after a restart
        s.q.setValue(i18n.SETTINGS_KEY, str(self.language.currentData() or i18n.SYSTEM))
        s.save()
        # also when the choice did not change: the windows can still show another theme
        if theme_changed or theme.wants_dark(s.theme) != theme.is_dark():
            self.win.apply_theme()
        if sm_changed:
            self.win.start_spacemouse()
        self.win.view.tools["circle"].segments = s.circle_segments


class ShortcutsPanel(QWidget):
    """Preferences > Shortcuts: filter, the commands per menu, and add/remove a shortcut."""

    def __init__(self, win):
        super().__init__()
        self.win = win
        self.items = {}
        lay = QVBoxLayout(self)
        top = QHBoxLayout()
        top.addWidget(QLabel(tr("Filter")))
        self.filter = QLineEdit()
        self.filter.setPlaceholderText(tr("Type to search"))
        self.filter.textChanged.connect(self._filter)
        top.addWidget(self.filter, 1)
        lay.addLayout(top)
        self.tree = QTreeWidget()
        self.tree.setColumnCount(2)
        self.tree.setHeaderLabels([tr("Command"), tr("Shortcut")])
        self.tree.header().setSectionResizeMode(0, QHeaderView.Stretch)
        self.tree.setColumnWidth(1, 150)
        lay.addWidget(self.tree, 1)
        row = QHBoxLayout()
        row.addWidget(QLabel(tr("Add shortcut")))
        self.key_edit = QKeySequenceEdit()
        row.addWidget(self.key_edit, 1)
        self.add_button = QPushButton("+")
        self.add_button.setToolTip(tr("Puts the typed shortcut on the chosen command"))
        self.add_button.clicked.connect(self.add_shortcut)
        self.remove_button = QPushButton("−")
        self.remove_button.setToolTip(tr("Removes the shortcut from the chosen command"))
        self.remove_button.clicked.connect(self.remove_shortcut)
        self.restore_button = QPushButton(tr("Reset All"))
        self.restore_button.setToolTip(tr("Puts all shortcuts back to their defaults"))
        self.restore_button.clicked.connect(self.restore_all)
        for b in (self.add_button, self.remove_button, self.restore_button):
            b.setMinimumWidth(40)
            row.addWidget(b)
        lay.addLayout(row)
        self.reload()

    def reload(self, keep=None):
        """Every command of the menus, grouped per menu, with the shortcut it has now.

        The paths are the English ones (the settings keys); the tree shows the menu texts."""
        keep = keep or self.selected_path()
        self.tree.blockSignals(True)
        self.tree.clear()
        self.items = {}
        self.labels = {}
        groups = {}
        for path, action in self.win.command_paths().items():
            label = self.win.command_label(path)
            self.labels[path] = label
            menu, _, name = label.partition("/")
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
        """Reset All: back to the shortcuts Volura ships with."""
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
            label = self.labels.get(path, path)
            it.setHidden(bool(text) and text not in label.lower() and text not in path.lower())
        for i in range(self.tree.topLevelItemCount()):
            head = self.tree.topLevelItem(i)
            found = any(not head.child(j).isHidden() for j in range(head.childCount()))
            head.setHidden(bool(text) and not found)


class WelcomeDialog(QDialog):
    """Help > Welcome to Volura, the welcome screen of SketchUp 2017."""

    STEPS = ((tr("Draw a shape"), tr("Line (L), Rectangle (R), Circle (C) or Polygon: Volura makes "
                                     "a face of every closed shape.")),
             (tr("Push/Pull"), tr("Choose Push/Pull (P), click the face and type a distance: a "
                                  "volume appears. Going down makes a recess or hole.")),
             (tr("Export to your slicer"), tr("File > Export as 3MF (multicolor) or STL, and open "
                                              "the file in PrusaSlicer, OrcaSlicer or Bambu "
                                              "Studio.")))

    def __init__(self, win):
        super().__init__(win)
        self.setWindowTitle(tr("Welcome to Volura"))
        self.resize(700, 430)
        lay = QVBoxLayout(self)
        title = QLabel("Volura")
        title.setStyleSheet(f"font-size: 34px; font-weight: bold; color: {theme.color('title')};")
        lay.addWidget(title)
        lay.addWidget(QLabel(tr("<b>3D modeling for 3D printing, like in SketchUp 2017.</b>")))
        lay.addSpacing(10)
        muted = theme.color("muted")
        for i, (head, text) in enumerate(self.STEPS, 1):
            step = QLabel(f"<b>{i}. {head}</b><br><span style='color:{muted}'>{text}</span>")
            step.setWordWrap(True)
            step.setTextFormat(Qt.RichText)
            lay.addWidget(step)
        lay.addStretch(1)
        self.show_at_start = QCheckBox(tr("Always show at startup"))
        self.show_at_start.setChecked(welcome_enabled())
        self.show_at_start.toggled.connect(self.save)
        lay.addWidget(self.show_at_start)
        row = QHBoxLayout()
        row.addStretch(1)
        start = QPushButton(tr("Start Using Volura"))
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

#: Help > Keyboard Shortcuts; {rows} are the rows of the tools with a shortcut
HELP_HTML = tr_noop("""
<h2>Volura - how it works</h2>
<p>Volura works like SketchUp: draw a flat face and make a volume of it with <b>Push/Pull</b>.</p>
<ol>
<li>Draw edges with Line (L), Rectangle (R), Circle (C) or Polygon. A closed shape becomes a face.</li>
<li>Edges split each other, a line across a face splits the face; with the Eraser (E) you erase edges
(or parts of them) and faces are merged again.</li>
<li>Push/Pull (P) works on every face or part of a face: up = add, down = recess or hole.
Type a distance and Enter; the direction follows your mouse.</li>
<li>Select a shape (triple-click) and make a <b>group</b> of it (G): every group is a separate
printable part with its own color. Double-click a group to edit it.</li>
<li><b>Multicolor</b>: hold <b>Alt</b> while pushing/pulling: inward makes an inlay (recess +
filling as a separate group), outward a separate part. Give it a color with the Paint Bucket (B) and
export as 3MF.</li>
</ol>
<h3>Keyboard shortcuts</h3>
<table cellpadding="3">{rows}
<tr><td><b>A</b></td><td>2 Point Arc (in Draw &gt; Arcs also Arc, 3 Point Arc and Pie)</td></tr>
<tr><td><b>Shift+Z</b></td><td>Zoom Extents</td></tr>
<tr><td><b>Middle mouse button</b></td><td>Orbit, with Shift: pan. Scroll wheel: zoom</td></tr>
<tr><td><b>Arrow keys</b></td><td>Lock an axis (right = red/X, left = green/Y, up = blue/Z)</td></tr>
<tr><td><b>Shift</b> (hold)</td><td>Lock the current direction</td></tr>
<tr><td><b>Ctrl</b></td><td>Move/rotate: copy. Push/Pull: new start face</td></tr>
<tr><td><b>Alt</b></td><td>Push/Pull: separate part / inlay</td></tr>
<tr><td><b>G</b></td><td>Make a component or group of the selection</td></tr>
<tr><td><b>Esc</b></td><td>Cancel the current operation</td></tr>
<tr><td><b>Ctrl+Z / Ctrl+Y</b></td><td>Undo / redo</td></tr>
</table>
<h3>Measurements (bottom right)</h3>
<p>Just start typing during an operation. Examples: <code>25</code>, <code>2.5cm</code>,
<code>40;20</code> (rectangle), <code>64s</code> (segments), <code>10;0;5</code> (move),
<code>5x</code> or <code>/5</code> after a copy (array), <code>45</code> (degrees when rotating).</p>
""")


class MainWindow(QMainWindow):
    #: the active environment: "3d" or "2d" (the 2D worksheet); not part of the file
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
        self._busy = False          # a background job is reading the model: keep it still
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
        QTimer.singleShot(POOL_PREWARM_MS, workers.prewarm)
        self.spacemouse = spacemouse.SpaceMouse(self)
        self.spacemouse.motion.connect(self.on_spacemouse_motion)
        self.spacemouse.command.connect(self.on_spacemouse_command)
        QTimer.singleShot(0, self.start_spacemouse)
        QApplication.instance().styleHints().colorSchemeChanged.connect(self._system_scheme_changed)
        self.autosave_timer = QTimer(self)
        self.autosave_timer.setInterval(AUTOSAVE_INTERVAL_MS)
        self.autosave_timer.timeout.connect(self.autosave)
        self.autosave_timer.start()

    # ------------------------------------------------------------ autosave
    def autosave(self):
        """Autosave: every 5 minutes, only when something has changed."""
        if not self.doc.dirty:
            return False
        try:
            fileio.save_document(self.doc, autosave_path())
        except Exception as e:
            errors.log_exception(note=f"autosave failed: {e}")
            return False
        return True

    def offer_autosave_restore(self):
        """At startup, ask whether the autosaved model should be restored."""
        if is_offscreen() or not os.path.exists(autosave_path()):
            return False
        r = QMessageBox.question(self, tr("Autosaved Model"),
                                 tr("There is an autosaved model from a previous session.\n\n"
                                    "Do you want to restore that model?"),
                                 QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
        if r != QMessageBox.Yes:
            remove_autosave()
            return False
        return self.restore_autosave()

    def restore_autosave(self):
        """Load the autosaved model; the document keeps its own name."""
        path = self.doc.path
        try:
            self.view.tool.reset()
            fileio.load_document(self.doc, autosave_path())
        except Exception as e:
            QMessageBox.critical(self, tr("Restore Failed"), str(e))
            return False
        self.doc.path = path
        self.doc.dirty = True
        self.doc.notify()
        self.update_title()
        self.view.zoom_extents()
        self.hint.setText(tr("Autosaved model restored."))
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
        self.coords.setToolTip(tr("Position of the cursor on the 2D worksheet"))
        self.vcb_label = QLabel(tr("Measurements"))
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
        """The '3D | 2D' slide switch at the bottom right: the half of the current mode is blue."""
        two = self.mode == "2d"
        self.mode_button.setText("3D | 2D")
        self.mode_button.setChecked(two)
        self.mode_button.setToolTip(
            tr("Now: 2D worksheet — click for the 3D model (Ctrl+Shift+2)") if two
            else tr("Now: 3D model — click for the 2D worksheet (Ctrl+Shift+2)"))

    # ------------------------------------------------------------ 3D or 2D
    def set_mode(self, mode) -> bool:
        """To the 3D model or the 2D worksheet; the mode is not part of the file."""
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

    #: The panels of the Default Tray that belong to each mode (English keys, see PANELS).
    MODE_PANELS = {"3d": (PANEL_INFO, PANEL_MATERIALS, PANEL_OUTLINER, PANEL_PRINT_CHECK),
                   "2d": (PANEL_INFO, PANEL_LAYERS)}

    def _show_layers_panel(self, two_d: bool):
        """Only the panels that belong to the mode: filaments, outliner and print check with
        the model, the layers with the worksheet. Entity Info shows what belongs to the mode."""
        if not hasattr(self, "layers_panel"):
            return
        mode = "2d" if two_d else "3d"
        mode_titles = {tr(p) for p in self.MODE_PANELS[mode]}
        for title in list(self.tray.panel_titles) + list(self.tray.extra_titles):
            shown = title in mode_titles
            self.tray.set_panel_shown(title, shown)
            a = self.tray_actions.get(title)
            if a is not None:
                a.setVisible(shown)
        if two_d and self.tray.is_open(tr(PANEL_PRINT_CHECK)):
            self.print_panel.clear_marks()
        self.layers_panel.refresh()
        self.tray.set_panel_open(tr(PANEL_LAYERS), bool(two_d))
        self.info.refresh()

    def toggle_sheet_mode(self, checked=None):
        """View > 2D Worksheet and the button at the bottom right of the status bar."""
        want = "2d" if (self.mode == "3d" if checked is None else bool(checked)) else "3d"
        return self.set_mode(want)

    def zoom_extents(self):
        """Zoom Extents: the model in the 3D view, the drawing on the worksheet."""
        (self.sheet_view if self.mode == "2d" else self.view).zoom_extents()

    def active_view(self):
        """The view that belongs to the current mode: the 3D view or the worksheet."""
        return self.sheet_view if self.mode == "2d" else self.view

    def _show_mode_toolbars(self, changed=True):
        """In 2D the 3D toolbars make way for those of the worksheet; going back restores all.

        What was visible is only remembered when entering 2D: a second set_mode("2d") (for
        example from Sheet… or the box generator) would otherwise remember the hidden state
        and leave the toolbars away on return.
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
        """The 3D tools and what goes with them are out of reach in 2D."""
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
        a.setToolTip(tr("Not available yet"))
        a.setStatusTip(tr("Not available yet"))
        a.setEnabled(False)
        return a

    def tool_act(self, text, name, shortcut=None, icon_name=None):
        """Action that selects a drawing tool; also reachable as self.tool_actions[name]."""
        a = self.act(text, lambda checked=False, n=name: self.set_tool(n), shortcut, icon_name,
                     checkable=True)
        self.tool_actions[name] = a
        return a

    def face_style_act(self, text, value):
        """One choice of the exclusive View > Face Style group."""
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
        self.a_new = A(tr("&New"), self.file_new, "Ctrl+N", "new")
        self.a_open = A(tr("&Open"), self.file_open, "Ctrl+O", "open")
        self.a_save = A(tr("&Save"), self.file_save, "Ctrl+S", "save")
        self.a_save_as = A(tr("Save &As"), self.file_save_as)
        self.a_save_copy = A(tr("Save a Cop&y As"), self.file_save_copy_as)
        self.a_import = A(tr("&Import…"), self.file_import)
        self.a_exp_stl = A(tr("&3D Model (STL)…"), self.export_stl, None, "stl")
        self.a_exp_3mf = A(tr("&Multicolor 3MF…"), self.export_3mf, None, "3mf")
        self.a_exp_colors = A(tr("STL per &Color…"), self.export_stl_colors)
        self.a_exp_svg2d = A(tr("2D Vector (&SVG)…"), lambda checked=False: self.export_vector("svg"))
        self.a_exp_dxf2d = A(tr("2D Vector (&DXF)…"), lambda checked=False: self.export_vector("dxf"))
        self.a_print_setup = A(tr("Printer Set&up…"), lambda checked=False: self.print_setup())
        self.a_check = A(tr("Print &Check"), self.print_check, "F5", "check")
        self.a_print = A(tr("&Print"), self.export_3mf, "Ctrl+P", "3mf")
        self.a_quit = A(tr("E&xit"), self.close, "Ctrl+Q")

        self.a_undo = A(tr("&Undo"), self.undo, ["Alt+Backspace", "Ctrl+Z"], "undo")
        self.a_redo = A(tr("&Redo"), self.redo, "Ctrl+Y", "redo")
        self.a_cut = A(tr("Cu&t"), self.cut, ["Shift+Del", "Ctrl+X"])
        self.a_copy = A(tr("&Copy"), self.copy, "Ctrl+C")
        self.a_paste = A(tr("&Paste"), self.paste, "Ctrl+V")
        self.a_paste_loc = A(tr("Paste &In Place"), self.paste_at_location)
        self.a_delete = A(tr("&Delete"), self.delete, "Del")
        self.a_erase_guides = A(tr("Delete Guid&es"), self.erase_guides)
        self.a_sel_all = A(tr("Select &All"), self.select_all, "Ctrl+A")
        self.a_sel_none = A(tr("Select None"), self.select_none, "Ctrl+T")
        self.a_hide = A(tr("&Hide"), self.hide_selected)
        self.a_unhide_sel = A(tr("&Selected"), self.unhide_selected)
        self.a_unhide_last = A(tr("&Last"), self.unhide_last)
        self.a_unhide = A(tr("&All"), self.unhide_all)
        self.a_lock = A(tr("Loc&k"), self.lock_selected)
        self.a_unlock = A(tr("Unl&ock"), self.unlock_selected)
        self.a_unlock_sel = A(tr("&Selected"), self.unlock_selected)
        self.a_unlock_all = A(tr("&All"), self.unlock_all)
        self.a_cut_face_model = A(tr("With &Model"), lambda checked=False: self.intersect_faces(True),
                                  None, "cut_faces", False,
                                  tr("Intersection lines of the selection with the rest of the model"))
        self.a_cut_face_sel = A(tr("With &Selection"), lambda checked=False: self.intersect_faces(False),
                                None, "cut_faces", False,
                                tr("Intersection lines between the selected items"))
        self.a_component = A(tr("Make Co&mponent"), self.make_group, "G")
        self.a_group = A(tr("Make &Group"), self.make_group)
        self.a_close_group = A(tr("Clo&se Group/Component"), self.close_group)
        self.a_explode = A(tr("E&xplode Group"), self.explode_group)
        self.a_edit_group = A(tr("&Edit Group"), self.edit_group)

        for name, label, ic, sc in TOOL_INFO:
            a = A(tr(label), lambda checked=False, n=name: self.set_tool(n), sc, ic, checkable=True)
            a.setProperty(SOURCE_TEXT, label)     # "Orbit" and "Turn" are both "Draaien" in Dutch
            self.tool_actions[name] = a

        self.a_freehand = self.tool_act(tr("&Freehand"), "freehand", None, "freehand")
        self.a_arc = self.tool_act(tr("&Arc"), "arc", None, "arc")
        self.a_arc2 = self.tool_act(tr("&2 Point Arc"), "arc2", "A", "arc2")
        self.a_arc3 = self.tool_act(tr("&3 Point Arc"), "arc3", None, "arc3")
        self.a_pie = self.tool_act(tr("&Pie"), "pie", None, "pie")
        self.a_rotrect = self.tool_act(tr("R&otated Rectangle"), "rotrect", None, "rotrect")
        self.a_box = A(tr("&Box…"), lambda checked=False: self.insert_primitive("box"))
        self.a_cylinder = A(tr("&Cylinder…"), lambda checked=False: self.insert_primitive("cylinder"))
        self.a_cone = A(tr("C&one…"), lambda checked=False: self.insert_primitive("cone"))
        self.a_sphere = A(tr("&Sphere…"), lambda checked=False: self.insert_primitive("sphere"))
        self.a_torus = A(tr("&Torus…"), lambda checked=False: self.insert_primitive("torus"))

        self.a_extents = A(tr("Zoom Extents"), self.zoom_extents, ["Ctrl+Shift+E", "Shift+Z"],
                           "zoom_extents")
        self.views = {}
        for key, label, ic in (("top", tr("Top"), "view_top"), ("bottom", tr("Bottom"), "view_top"),
                               ("front", tr("Front"), "view_front"), ("back", tr("Back"), "view_back"),
                               ("left", tr("Left"), "view_left"), ("right", tr("Right"), "view_right"),
                               ("iso", tr("Iso"), "view_iso")):
            self.views[key] = A(label, lambda checked=False, k=key: self.view.set_view(k), None, ic)
        self.a_prev_view = A(tr("&Previous"), self.camera_prev)
        self.a_next_view = A(tr("&Next"), self.camera_next)
        self.a_prev_view.setEnabled(self.view.can_prev_camera())
        self.a_next_view.setEnabled(self.view.can_next_camera())
        self.a_ortho = A(tr("Parallel Projection"), lambda checked=False: self.apply_projection(),
                         None, "ortho", checkable=True)
        self.a_persp = A(tr("P&erspective"), lambda checked=False: self.apply_projection(), None, None,
                         checkable=True)
        proj = QActionGroup(self)
        proj.setExclusive(True)
        proj.addAction(self.a_ortho)
        proj.addAction(self.a_persp)
        self.a_persp.setChecked(True)
        self.a_fov = A(tr("&Field of View"), self.camera_fov)
        self.a_zoom_window = A(tr("Zoom &Window"), self.camera_zoom_window, "Ctrl+Shift+W")

        self.a_toolbars = A(tr("&Toolbars"), self.show_toolbar_dialog)
        self.a_hidden_geometry = A(tr("Hidden &Geometry"), self.toggle_show_hidden, None, None,
                                   checkable=True)
        self.a_hidden_geometry.setChecked(self.settings.show_hidden)
        self.a_guides = A(tr("G&uides"), self.toggle_guides, None, None, checkable=True)
        self.a_guides.setChecked(self.settings.show_guides)
        self.a_grid = A(tr("Print Bed"), self.toggle_grid, None, None, checkable=True)
        self.a_grid.setChecked(self.settings.show_grid)
        self.a_axes = A(tr("&Axes"), self.toggle_axes, None, None, checkable=True)
        self.a_axes.setChecked(self.settings.show_axes)
        self.a_edges = A(tr("&Edges"), self.toggle_edges, None, None, checkable=True)
        self.a_edges.setChecked(self.settings.show_edges)
        self.a_back_edges = A(tr("&Back Edges"), self.toggle_back_edges, "K", None,
                              checkable=True)
        self.a_back_edges.setChecked(self.settings.back_edges)
        self.a_xray = A(tr("&X-ray"), self.toggle_xray, "Alt+X", "xray", checkable=True)
        self.a_xray.setChecked(self.settings.xray)
        self.face_style_group = QActionGroup(self)
        self.face_style_group.setExclusive(True)
        self.face_style_actions = {}
        self.a_wireframe = self.face_style_act(tr("&Wireframe"), WIREFRAME)
        self.a_hidden_lines = self.face_style_act(tr("&Hidden Line"), HIDDENLINE)
        self.a_shaded = self.face_style_act(tr("&Shaded"), SHADED)
        self.a_monochrome = self.face_style_act(tr("&Monochrome"), MONOCHROME)
        self.a_hide_rest = A(tr("Hide &Rest of Model"), self.toggle_hide_rest, None, None,
                             checkable=True)
        self.a_hide_rest.setChecked(self.settings.hide_rest)

        self.a_union = A(tr("Union"), lambda checked=False: self.solid_op("union"), None, "union")
        self.a_subtract = A(tr("Subtract"), lambda checked=False: self.solid_op("subtract"), None,
                            "subtract")
        self.a_intersect = A(tr("Intersect"), lambda checked=False: self.solid_op("intersect"), None,
                             "intersect")
        self.a_trim = A(tr("&Trim"), lambda checked=False: self.trim_solids(), None, "trim",
                        False, tr("Trim the second group with the first (B = B - A)"))
        self.a_split = A(tr("Split"), self.split_parts)
        self.a_follow_me = self.tool_act(tr("&Follow Me"), "followme", None, "followme")
        self.a_shell = A(tr("&Outer Shell"), self.outer_shell, None, "shell",
                         False, tr("Keep only the outside of the selected groups"))
        self.a_simplify = A(tr("Simplify &Model…"), self.simplify_model, None, None,
                            False, tr("Fewer triangles on the selected groups, within a chosen deviation"))
        self.a_protractor = self.tool_act(tr("&Protractor"), "protractor")
        self.a_drop = A(tr("Drop to Bed"), self.drop_to_bed, "Ctrl+B", "bed")
        self.a_center = A(tr("Center on Bed"), self.center_on_bed)

        self.a_thread = A(tr("Thread..."), lambda checked=False: self.thread_dialog(), None, "thread",
                          False, tr("Add a screw thread or make a bolt or nut (ISO metric)"))
        self.a_thread_ext = A(tr("External Thread on Cylinder..."),
                              lambda checked=False: self.thread_dialog("external"), None, "thread_ext")
        self.a_thread_int = A(tr("Internal Thread in Hole..."),
                              lambda checked=False: self.thread_dialog("internal"), None, "thread_int")
        self.a_thread_bolt = A(tr("New Bolt..."),
                               lambda checked=False: self.thread_dialog("bolt"), None, "bolt")
        self.a_thread_nut = A(tr("New Nut or Sleeve..."),
                              lambda checked=False: self.thread_dialog("nut"), None, "nut")

        self.a_cutplane = self.tool_act(tr("Cut"), "cutplane", None, "cutplane")
        self.a_cut_join = A(tr("Cut and Join..."), lambda checked=False: self.cut_join_dialog(),
                            None, "cut_join", False,
                            tr("Cut the model at a plane and make pins, holes or a dovetail"))
        self.a_fillet = self.tool_act(tr("Fillet"), "fillet", None, "fillet")
        self.a_fillet.setToolTip(tr("Click edges and type a radius: a convex or concave rounding"))
        self.a_chamfer = self.tool_act(tr("Chamfer"), "chamfer", None, "chamfer")
        self.a_chamfer.setToolTip(tr("Click edges and type a distance: a bevel along the edge"))

        self.a_prefs = A(tr("&Preferences"), self.preferences)
        self.a_model_info = A(tr("&Model Info"), lambda checked=False: self.model_info())
        self.a_welcome = A(tr("&Welcome to Volura"), self.show_welcome)
        self.a_updates = A(tr("Check for &Updates…"), lambda checked=False: self.check_for_updates())
        self.a_help = A(tr("&Keyboard Shortcuts"), self.show_help, "F1")
        self.a_about = A(tr("&About Volura"), self.about)

        # ---- the 2D worksheet
        self.a_sheet = A(tr("2D &Worksheet"), self.toggle_sheet_mode, "Ctrl+Shift+2", None,
                         checkable=True, tip=tr("Drawing in 2D for laser cutter and mill"))
        self.a_to_sheet = A(tr("Copy to 2D &Worksheet…"), self.copy_to_sheet, None, None, False,
                            tr("Selected faces, top view or section to the worksheet"))
        self.a_to_laser = A(tr("To &Laser or Mill…"), self.send_selection_to_sheet, None, "to_laser",
                            False, tr("Put the selected faces or plates on a layer of the worksheet "
                                      "to cut, engrave or mill them"))
        self.a_sheet_to_model = A(tr("Parts to &3D Model"), self.sheet_to_model, None, "to_model",
                                  False, tr("Put what is cut into the 3D model as plates with the "
                                            "sheet thickness"))
        for name, label, tip in SHEET_TOOL_INFO:
            self.sheet_tool_act(tr(label), name, tip).setProperty(SOURCE_TEXT, label)
        self.a_mirror_h = A(tr("Flip &Horizontally"),
                            lambda checked=False: self.sheet_view.mirror(True), None, "mirror_h",
                            False, tr("Swap left and right around the center of the selection"))
        self.a_mirror_v = A(tr("Flip &Vertically"),
                            lambda checked=False: self.sheet_view.mirror(False), None, "mirror_v",
                            False, tr("Swap top and bottom around the center of the selection"))
        # ---- operations on the selection of the 2D worksheet
        self.a_sheet_union = A(tr("Union Outlines"),
                               lambda checked=False: self.sheet_op("union"), None, "union", False,
                               tr("The selected closed outlines as one shape"))
        self.a_sheet_subtract = A(tr("Subtract Outlines"),
                                  lambda checked=False: self.sheet_op("subtract"), None, "subtract",
                                  False, tr("The first selected path minus the rest"))
        self.a_sheet_intersect = A(tr("Intersect Outlines"),
                                   lambda checked=False: self.sheet_op("intersect"), None,
                                   "intersect", False, tr("Only what all outlines have in common"))
        self.a_sheet_join = A(tr("Join Lines"),
                              lambda checked=False: self.sheet_view.join_selected(), None, "join",
                              False, tr("Make loose segments into polylines and shapes"))
        self.a_sheet_split = A(tr("Split into Segments"),
                               lambda checked=False: self.sheet_view.split_selected(), None, "split",
                               False, tr("Make every segment of the selection separate"))
        self.a_sheet_align = A(tr("Align…"), lambda checked=False: self.sheet_view.align_selected("left"),
                               None, "align", False,
                               tr("Line up the selection left, center, right, top or bottom"))
        self.sheet_align_actions = {}
        for mode, label, tip in (
                ("left", tr("Align Left"), tr("Align the selection to the left")),
                ("center", tr("Align Center"), tr("Center the selection")),
                ("right", tr("Align Right"), tr("Align the selection to the right")),
                ("top", tr("Align Top"), tr("Align the selection to the top")),
                ("bottom", tr("Align Bottom"), tr("Align the selection to the bottom"))):
            self.sheet_align_actions[mode] = A(
                label, lambda checked=False, m=mode: self.sheet_view.align_selected(m), None,
                None, False, tip)
        self.a_sheet_distribute = A(tr("Distribute…"),
                                    lambda checked=False: self.sheet_view.distribute_selected("x"),
                                    None, "distribute", False,
                                    tr("Put the same spacing throughout the selection"))
        self.sheet_distribute_actions = {
            "x": A(tr("Distribute Horizontally"),
                   lambda checked=False: self.sheet_view.distribute_selected("x"), None, None,
                   False, tr("Space the centers evenly horizontally")),
            "y": A(tr("Distribute Vertically"),
                   lambda checked=False: self.sheet_view.distribute_selected("y"), None, None,
                   False, tr("Space the centers evenly vertically")),
        }
        self.a_sheet_duplicate = A(tr("Duplicate in Row or Grid…"),
                                   lambda checked=False: self.sheet_view.duplicate_dialog(), None,
                                   "duplicate", False,
                                   tr("Place copies of the selection at a fixed spacing"))
        self.a_sheet_grid = A(tr("Snap to Grid"), self.set_sheet_grid_snap, None, "grid",
                              checkable=True, tip=tr("Pull the cursor to grid intersections"))
        self.a_sheet_grid.setChecked(True)
        self.a_sheet_round = A(tr("Rounded Corners on Offset"), self.set_sheet_round_join, None,
                               "round_join", checkable=True,
                               tip=tr("Offsets with round corners (off) or sharp corners (on)"))
        self.a_sheet_round.setChecked(True)
        # ---- layers of the worksheet
        self.a_sheet_to_layer = A(tr("To &Layer"), lambda checked=False: self.move_selection_to_layer(),
                                  None, "layers", False,
                                  tr("Move the selection to the active layer; the arrow picks "
                                     "another one"))
        self.a_sheet_layers = A(tr("&Worksheet Layers"), self.open_layers_panel, None, None, False,
                                tr("Open the Layers panel: color, operation and settings per layer"))
        # ---- sheet size and automatic nesting
        self.a_sheet_plaat = A(tr("&Sheet…"), lambda checked=False: self.open_sheet_settings(), None,
                               "sheet", False,
                               tr("Sheet size, material thickness, margin, gap and number of sheets"))
        self.a_sheet_nest = A(tr("Auto Arra&nge…"), lambda checked=False: self.auto_arrange(),
                              None, "nest", False,
                              tr("Arrange the parts over the sheets as economically as possible"))
        # ---- box with finger joints
        self.a_box_joint = A(tr("Box with &Finger Joints…"),
                             lambda checked=False: self.box_joint_dialog(), None, "boxjoint",
                             False,
                             tr("Box with finger joints for the laser cutter, on the worksheet "
                                "or as a preview in 3D"))

    def sheet_tool_act(self, label, name, tip=None):
        """Button for a 2D tool; the key works on the worksheet (M, Q, S, Space)."""
        a = self.act(label, lambda checked=False, n=name: self.sheet_view.set_tool(n), None,
                     name, checkable=True, tip=tip)
        self.sheet_tool_actions[name] = a
        return a

    def _build_menus(self):
        mb = self.menuBar()

        m = mb.addMenu(tr("&File"))
        export = QMenu(tr("&Export"), m)
        for a in (self.a_exp_stl, self.a_exp_3mf, self.a_exp_colors, None,
                  self.a_exp_svg2d, self.a_exp_dxf2d):
            if a is None:
                export.addSeparator()
            else:
                export.addAction(a)
        self.recent_menu = QMenu(tr("&Recent Files"), m)
        self.update_recent_menu()
        self.fill(m, (self.a_new, self.a_open, self.recent_menu, None,
                      self.a_save, self.a_save_as, self.a_save_copy, None,
                      self.a_import, export, None,
                      self.a_print_setup, self.a_check, self.a_print, None,
                      self.a_quit))

        m = mb.addMenu(tr("&Edit"))
        unhide = QMenu(tr("U&nhide"), m)
        self.fill(unhide, (self.a_unhide_sel, self.a_unhide_last, self.a_unhide))
        unlock = QMenu(tr("Unl&ock"), m)
        self.fill(unlock, (self.a_unlock_sel, self.a_unlock_all))
        cut_faces = QMenu(tr("Intersect &Faces"), m)
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

        m = mb.addMenu(tr("&View"))
        edge_style = QMenu(tr("&Edge Style"), m)
        self.fill(edge_style, (self.a_edges, self.a_back_edges))
        face_style = QMenu(tr("Face St&yle"), m)
        self.fill(face_style, (self.a_xray, None, self.a_wireframe, self.a_hidden_lines,
                               self.a_shaded, self.a_monochrome))
        comp_edit = QMenu(tr("Co&mponent Edit"), m)
        self.fill(comp_edit, (self.a_hide_rest,))
        self.fill(m, (self.a_toolbars, self.a_sheet, None,
                      self.a_sheet_plaat, self.a_sheet_nest, self.a_box_joint,
                      self.a_sheet_to_model, None,
                      self.a_hidden_geometry, self.a_axes, self.a_guides, self.a_grid, None,
                      edge_style, face_style, comp_edit))

        m = mb.addMenu(tr("&Camera"))
        std_views = QMenu(tr("&Standard Views"), m)
        self.fill(std_views, tuple(self.views[k] for k in
                                   ("top", "bottom", "front", "back", "left", "right", "iso")))
        self.fill(m, (self.a_prev_view, self.a_next_view, std_views, None,
                      self.a_ortho, self.a_persp, None,
                      self.tool_actions["orbit"], self.tool_actions["pan"], self.tool_actions["zoom"],
                      self.a_fov, self.a_zoom_window, self.a_extents))

        m = mb.addMenu(tr("&Draw"))
        lines = QMenu(tr("&Lines"), m)
        self.fill(lines, (self.tool_actions["line"], self.a_freehand))
        arcs = QMenu(tr("&Arcs"), m)
        self.fill(arcs, (self.a_arc, self.a_arc2, self.a_arc3, self.a_pie))
        shapes = QMenu(tr("&Shapes"), m)
        self.fill(shapes, (self.tool_actions["rectangle"], self.a_rotrect, self.tool_actions["circle"],
                           self.tool_actions["polygon"]))
        primitives = QMenu(tr("&Basic Shapes"), m)
        self.fill(primitives, (self.a_box, self.a_cylinder, self.a_cone, self.a_sphere, self.a_torus))
        self.fill(m, (lines, arcs, shapes, None, primitives))

        m = mb.addMenu(tr("&Tools"))
        solids = QMenu(tr("&Solid Tools"), m)
        self.fill(solids, (self.a_intersect, self.a_union, self.a_subtract, self.a_trim, self.a_split))
        self.print_tools_menu = QMenu(tr("P&rint Tools"), m)
        self.fill(self.print_tools_menu, (self.a_thread, self.a_thread_ext, self.a_thread_int,
                                          self.a_thread_bolt, self.a_thread_nut, None,
                                          self.a_cutplane, self.a_cut_join, None,
                                          self.a_fillet, self.a_chamfer))
        self.fill(m, (self.tool_actions["select"], self.tool_actions["eraser"], self.tool_actions["paint"],
                      None,
                      self.tool_actions["move"], self.tool_actions["rotate"], self.tool_actions["scale"],
                      None,
                      self.tool_actions["pushpull"], self.a_follow_me, self.tool_actions["offset"], None,
                      self.a_shell, self.a_simplify, solids, None,
                      self.tool_actions["tape"], self.a_protractor, None,
                      self.tool_actions["text"], None,
                      self.print_tools_menu))

        m = mb.addMenu(tr("&Window"))
        self.standard_menu = QMenu(tr("Default &Tray"), m)
        self.fill(m, (self.standard_menu, None, self.a_model_info, self.a_prefs))

        m = mb.addMenu(tr("&Help"))
        self.fill(m, (self.a_welcome, self.a_help, None, self.a_updates, self.a_about))

    def _build_toolbars(self):
        self._dropdown_buttons = {}
        self._toolbar_tool_icons = {}
        T = self.tool_actions

        tb = QToolBar(tr("Getting Started"))
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

        tools = QToolBar(tr("Large Tool Set"))
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
        """The '2D Worksheet' toolbar on the left; only visible in 2D mode."""
        bar = self.findChild(QToolBar, "SheetToolbar")
        if bar is None:
            bar = QToolBar(tr("2D Worksheet"))
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
        bar.hide()          # SketchUp shows no toolbar on the left by default

    def _add_layer_dropdown(self, tb, action):
        """Button with a drop-down menu of the layers; the menu is filled each time it opens."""
        btn = tb.widgetForAction(action)
        if not isinstance(btn, QToolButton):
            return
        menu = QMenu(btn)
        menu.aboutToShow.connect(lambda m=menu: self._fill_layer_menu(m))
        btn.setMenu(menu)
        btn.setPopupMode(QToolButton.MenuButtonPopup)
        self._dropdown_buttons[action] = btn

    def _fill_layer_menu(self, menu):
        """The "To Layer" menu: one line per layer of the worksheet."""
        menu.clear()
        for name, lay in self.doc.sheet.layers.items():
            a = menu.addAction(f"{name}   —   {OP_LABELS.get(lay.op, lay.op)}")
            a.setIcon(swatch(lay.color or "#000000", 14))
            if lay.locked:
                a.setEnabled(False)
                a.setToolTip(tr("This layer is locked"))
            a.triggered.connect(lambda checked=False, n=name: self.move_selection_to_layer(n))
        if not self.doc.sheet.layers:
            menu.addAction(tr("(no layers)")).setEnabled(False)

    def sheet_op(self, op):
        """An operation on the selection: Union, Subtract or Intersect."""
        self.set_mode("2d")
        return self.sheet_view.sheet_boolean(op)

    def set_sheet_grid_snap(self, checked):
        self.sheet_view.set_grid_snap(checked)

    def set_sheet_round_join(self, checked):
        self.sheet_view.set_offset_round(checked)

    def move_selection_to_layer(self, name=None) -> bool:
        """"To Layer": the selection to the active layer, or to the chosen layer."""
        self.set_mode("2d")
        sel = self.sheet_view.selected_paths()
        if not sel:
            self.hint.setText(tr("First select the outlines that should go to a layer."))
            return False
        if name is None:
            name = self.sheet_view.draw_layer()
            if all(p.layer == name for p in sel):
                self.hint.setText(tr("Those outlines are already on layer {name}; choose another "
                                     "layer with the arrow.").format(name=name))
                return False
        return self.sheet_view.move_to_layer(name)

    def open_layers_panel(self):
        """Window > Default Tray > Layers: color, operation and settings per layer."""
        self.tray.show()
        self.tray.raise_()
        self.tray.open_panel(tr(PANEL_LAYERS))
        self.layers_panel.refresh()

    # ------------------------------------------------------------ sheet and nesting
    def open_sheet_settings(self):
        """View > Sheet…: size, thickness, margin, gap and number of sheets."""
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
        message = (tr("Sheet {w:g} × {h:g} mm, {count} sheet, margin {margin:g} mm, gap {gap:g} mm")
                   if values["count"] == 1 else
                   tr("Sheet {w:g} × {h:g} mm, {count} sheets, margin {margin:g} mm, gap {gap:g} mm"))
        self.hint.setText(message.format(w=values["w"], h=values["h"], count=values["count"],
                                         margin=values["margin"], gap=values["gap"]))
        self.setFocus()

    def auto_arrange(self, options=None):
        """View > Auto Arrange…: arrange parts economically over the sheets."""
        self.set_mode("2d")
        if not self.doc.sheet.paths:
            self.hint.setText(tr("There is nothing on the worksheet to arrange."))
            return
        if options is None:
            dlg = AutoArrangeDialog(self)
            if dlg.exec() != QDialog.Accepted:
                self.setFocus()
                return
            options = dlg.options()
        self.sheet_view.auto_arrange(options)
        self.setFocus()

    # ------------------------------------------------------------ box with finger joints
    def box_joint_dialog(self):
        """View > Box with Finger Joints…: the panels on the worksheet or as a 3D preview."""
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
        """The panels of the box from an options dict; `ValueError` for impossible sizes."""
        o = dict(options or {})
        sheet = self.doc.sheet
        t = float(o.get("t") or sheet.sheet_thickness or boxgen.DEFAULT_THICKNESS)
        return boxgen.box_panels(float(o.get("w", 100.0)), float(o.get("d", 60.0)),
                                 float(o.get("h", 40.0)), t, o.get("finger_w"),
                                 bool(o.get("inner", True)), str(o.get("lid", "none")),
                                 float(o.get("kerf", 0.0) or 0.0))

    def _box_failure(self, error, note, message):
        """A failed box: a clear text in the status bar, never a crash.

        `note` is the English text for the log, `message` the translated template with
        {error} for the status bar."""
        if isinstance(error, ValueError):
            self.hint.setText(str(error))
        else:
            errors.log_exception(note=f"{note}: {error}")
            self.hint.setText(message.format(error=error))
        return None

    def make_box_sheet(self, options):
        """"On Worksheet": the cutting outlines on the worksheet, auto arranged on the sheet.

        One undo step: the panels go on in the same step and their place comes straight
        from the auto arrange. Without a sheet size they stay side by side.
        """
        self.set_mode("2d")
        sheet = self.doc.sheet
        try:
            panels = self.box_panels_from(options)
            drawing = boxgen.panels_to_drawing(panels, layer=sheet.default_layer("cut") or None,
                                               label_layer=sheet.default_layer("engrave") or None,
                                               labels=bool(options.get("labels", True)))
        except Exception as e:                      # the generator must never crash
            return self._box_failure(e, "the box", tr("The box failed: {error}"))
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
                return self._box_failure(e, "arranging", tr("The arranging failed: {error}"))
            else:
                drawing = self._nest_box_drawing(drawing, placements, sheet)
                used = max((si for (_p, si, *_r) in placements), default=0) + 1
        else:
            note = tr(" (no sheet size: choose View > Sheet… to arrange)")
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
        if not used:
            message = tr("{n} panels of the box on the worksheet ({w:g} × {d:g} × {h:g} mm){note}")
        elif used == 1:
            message = tr("{n} panels of the box on the worksheet ({w:g} × {d:g} × {h:g} mm), "
                         "arranged on {used} sheet{note}")
        else:
            message = tr("{n} panels of the box on the worksheet ({w:g} × {d:g} × {h:g} mm), "
                         "arranged on {used} sheets{note}")
        self.hint.setText(message.format(n=len(panels), w=W, d=D, h=H, used=used, note=note))
        return added

    def _nest_box_drawing(self, drawing, placements, sheet):
        """Put the arranged places on the drawing: every panel with its engraving."""
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
        """"Preview in 3D": every panel as its own group, in its place in the box."""
        self.set_mode("3d")
        try:
            panels = self.box_panels_from(options)
            solids = boxgen.panels_to_manifolds(panels)
        except Exception as e:                      # the generator must never crash
            return self._box_failure(e, "the box", tr("The box failed: {error}"))
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
                                        self.settings.current_color,
                                        tr("Box - {name}").format(name=p.name))
            groups.append(g)
        self.doc.set_selection([("g", g.id, g.id) for g in groups])
        self.doc.notify()
        self.view.zoom_extents()
        self.hint.setText(tr("{n} panels of the box placed in the model ({w:g} × {d:g} × {h:g} mm)")
                          .format(n=len(groups), w=W, d=D, h=H))
        return groups

    def ask_sheet_text(self):
        """Dialog of the 2D text tool: text, font, height and bold."""
        from .fonts import default_family, PREFERRED
        tool = self.sheet_view.tools["text"]
        dlg = QDialog(self)
        dlg.setWindowTitle(tr("Text on the Worksheet"))
        form = QFormLayout(dlg)
        ed = QLineEdit(tool.text or getattr(self, "_last_sheet_text", "Volura"))
        family = QComboBox()
        family.addItems(PREFERRED)
        if tool.family:
            family.setCurrentText(tool.family)
        elif default_family() in PREFERRED:
            family.setCurrentText(default_family())
        bold = QCheckBox(tr("Bold"))
        bold.setChecked(bool(tool.bold))
        h = QDoubleSpinBox()
        h.setRange(0.5, 2000)
        h.setDecimals(2)
        h.setSuffix(" mm")
        h.setValue(float(tool.height) if tool.height else 10.0)
        form.addRow(tr("Text"), ed)
        form.addRow(tr("Font"), family)
        form.addRow("", bold)
        form.addRow(tr("Cap height"), h)
        form.addRow(QLabel(tr("The letters are placed as closed outlines, with the holes\n"
                              "of o, a and e in them. Type a height in the Measurements box to "
                              "change it.")))
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(dlg.accept)
        bb.rejected.connect(dlg.reject)
        form.addRow(bb)
        if dlg.exec() != QDialog.Accepted or not ed.text().strip():
            return False
        self._last_sheet_text = ed.text()
        if not tool.set_text(ed.text(), family.currentText(), h.value(), bold.isChecked()):
            self.hint.setText(tr("This text gave no outlines; choose another font."))
            return False
        self.sheet_view.set_tool("text")
        self.sheet_view.setFocus()
        return True

    def _build_print_tools_toolbar(self):
        """The 'Print Tools' toolbar on the right; other features can add buttons to it too."""
        bar = self.findChild(QToolBar, "PrintToolsToolbar")
        if bar is None:
            bar = QToolBar(tr("Print Tools"))
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
        dlg.setWindowTitle(tr("Toolbars"))
        lay = QVBoxLayout(dlg)
        lay.addWidget(QLabel(tr("Show or hide the toolbars:")))
        for bar, label in ((self.tb_main, tr("Getting Started")), (self.tb_tools, tr("Large Tool Set")),
                           (getattr(self, "tb_print", None), tr("Print Tools")),
                           (getattr(self, "tb_sheet", None), tr("2D Worksheet"))):
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
        self.tray = Tray(((tr(PANEL_INFO), self.info), (tr(PANEL_MATERIALS), self.materials),
                          (tr(PANEL_OUTLINER), self.outliner)),
                         open_panels=(tr(PANEL_INFO), tr(PANEL_MATERIALS)), parent=self,
                         extra_panels=((tr(PANEL_PRINT_CHECK), self.print_panel),
                                       (tr(PANEL_LAYERS), self.layers_panel)))
        self.addDockWidget(Qt.RightDockWidgetArea, self.tray)
        self.resizeDocks([self.tray], [TRAY_WIDTH], Qt.Horizontal)
        self._build_tray_menu()
        self._show_layers_panel(False)
        self.info.refresh()
        self.outliner.refresh()

    def _build_tray_menu(self):
        """Window > Default Tray: an item per panel that opens/closes that panel."""
        self.tray_actions = {}
        english = {tr(p): p for p in PANELS}
        for title in list(self.tray.panel_titles) + list(self.tray.extra_titles):
            a = self.act(f"&{title}", lambda checked=False, t=title: self.tray.toggle_panel(t),
                         checkable=True)
            # the shortcut path uses the English title, whatever the language
            a.setProperty(SOURCE_TEXT, f"&{english.get(title, title)}")
            a.setChecked(self.tray.is_open(title))
            self.tray_actions[title] = a
            self.standard_menu.addAction(a)
        self.tray.panel_toggled.connect(self.on_tray_panel_toggled)
        show = self.tray.toggleViewAction()
        show.setText(tr("&Show Default Tray"))
        self.standard_menu.addAction(show)

    def on_tray_panel_toggled(self, title, is_open):
        a = self.tray_actions.get(title)
        if a is not None and a.isChecked() != is_open:
            a.setChecked(is_open)
        if title == tr(PANEL_PRINT_CHECK):
            if is_open:
                self.print_panel.refresh()
            else:
                self.print_panel.clear_marks()
        if title == tr(PANEL_LAYERS) and is_open:
            self.layers_panel.refresh()

    def open_print_check(self):
        """Window > Default Tray > Print Check: open the panel and check the model."""
        self.tray.show()
        self.tray.raise_()
        if self.tray.is_open(tr(PANEL_PRINT_CHECK)):
            self.print_panel.refresh()
        else:
            self.tray.open_panel(tr(PANEL_PRINT_CHECK))   # opening runs the check through the toggle
        self.view.update()

    # ------------------------------------------------------------ shortcuts
    def command_paths(self):
        """{'File/New': QAction} for every command in the menus; the first menu wins.

        The paths are made of the English menu texts, so the shortcuts stored under them
        (QSettings "shortcuts/<path>") work in every language. `command_label` gives the
        path as the menus show it; `_legacy_paths` maps the path the same command had in
        another language (the Dutch Volura 0.1.0 stored those) to the English one."""
        out, labels, legacy = {}, {}, {}
        tables = translation_tables()

        def segments(a):
            """The path segment of a menu or command: English, shown, per other language."""
            src = source_text_of(a)
            return (clean_text(src), clean_text(a.text()),
                    *(clean_text(t.get(src) or t.get(clean_text(src)) or src) for t in tables))

        def add(paths, a, overwrite=False):
            if paths[0] in out and not overwrite:
                return
            out[paths[0]] = a
            labels[paths[0]] = paths[1]
            for old in paths[2:]:
                legacy.setdefault(old, paths[0])

        def walk(menu, prefixes):
            for a in menu.actions():
                if a.isSeparator():
                    continue
                sub = a.menu()
                if sub is not None:
                    if sub is not getattr(self, "recent_menu", None):
                        walk(sub, tuple(f"{p}/{s}" for p, s in zip(prefixes, segments(a))))
                    continue
                segs = segments(a)
                if segs[0]:
                    add(tuple(f"{p}/{s}" for p, s in zip(prefixes, segs)), a)

        for a in self.menuBar().actions():
            menu = a.menu()
            if menu is not None:
                walk(menu, segments(a))
        seen = {id(a) for a in out.values()}
        other = (OTHER_COMMANDS, tr(OTHER_COMMANDS),
                 *(t.get(OTHER_COMMANDS, OTHER_COMMANDS) for t in tables))
        for a in self.actions():          # the commands that only live in the right-click menu
            segs = segments(a)
            if segs[0] and id(a) not in seen:
                add(tuple(f"{p}/{s}" for p, s in zip(other, segs)), a, overwrite=True)
        self._command_labels = labels
        self._legacy_paths = legacy
        return out

    def command_label(self, path):
        """The path of a command as the menus show it (in the current language)."""
        return getattr(self, "_command_labels", {}).get(path, path)

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
        keys = [k for k in q.allKeys() if k.startswith(SHORTCUTS_KEY)]
        for key in keys:
            path = key[len(SHORTCUTS_KEY):]
            if path in paths:
                stored[path] = self.read_shortcut_value(q.value(key))
        legacy = getattr(self, "_legacy_paths", {})
        for key in keys:                      # paths in another language, e.g. 'Bestand/Nieuw'
            path = key[len(SHORTCUTS_KEY):]
            new = legacy.get(path)
            if path in paths or new is None:
                continue
            value = self.read_shortcut_value(q.value(key))
            q.remove(key)                     # move it to the English path for good
            if new not in stored:
                stored[new] = value
                q.setValue(SHORTCUTS_KEY + new, value)
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
        """Preferences > Shortcuts > '+': the shortcut goes to this command."""
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
                    self, tr("Shortcut in Use"),
                    tr("{key} is now the shortcut of '{command}'.\n\nDo you want to move that "
                       "shortcut here?").format(key=seq.toString(QKeySequence.NativeText),
                                                command=clean_text(other.text()))
            ) != QMessageBox.Yes:
                return False
            self.change_shortcuts(other, [t for t in self.sequence_texts(other) if t != text])
        self.change_shortcuts(action, current + [text])
        return True

    def remove_shortcut(self, path, sequence=None):
        """Preferences > Shortcuts > '−': this shortcut, or all of them, off the command."""
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
        """Preferences > Shortcuts > Reset All: back to the shortcuts Volura ships with."""
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
        """Window > Model Info; File > Printer Setup lands on the Print Bed page."""
        dlg = ModelInfoDialog(self, page)
        dlg.exec()
        return dlg

    def print_setup(self):
        """File > Printer Setup: Model Info, page Print Bed (SketchUp)."""
        return self.model_info("Print Bed")

    def preferences(self, checked=False):
        """Window > Preferences: General, Display, Drawing, 3D Mouse and Shortcuts."""
        dlg = PreferencesDialog(self)
        dlg.exec()
        return dlg

    # ------------------------------------------------------------ theme
    def apply_theme(self):
        """Preferences > Display > Theme: the new theme everywhere at once."""
        dark = theme.apply(self.settings.theme)
        self.tray.apply_theme()
        self.info.refresh()
        for w in (self.view, self.sheet_view, self.mode_button):
            w.update()
        return dark

    def _system_scheme_changed(self, scheme=None):
        """Windows switches between light and dark: follow it when the theme is System."""
        if self.settings.theme == theme.SYSTEM and theme.wants_dark(theme.SYSTEM) != theme.is_dark():
            self.apply_theme()

    # ------------------------------------------------------------ 3D mouse
    def start_spacemouse(self):
        """Connect the 3D mouse (3DxWare, Raw Input or spacenavd); never in a test."""
        self.spacemouse.stop()
        if is_offscreen() or not self.settings.spacemouse:
            return None
        return self.spacemouse.start(int(self.winId()))

    def spacemouse_status(self):
        if not self.settings.spacemouse:
            return tr("off")
        if self.spacemouse.name:
            return tr("connected through {name}").format(name=self.spacemouse.name)
        return tr("no 3D mouse found")

    def on_spacemouse_motion(self, m):
        if self._busy or not self.isActiveWindow():
            return
        self.active_view().spacemouse_move(m, self.settings.spacemouse_speed)

    def on_spacemouse_command(self, name):
        """A button of the 3D mouse: fit to view, or a standard view in 3D."""
        if self._busy:
            return
        if name == "fit":
            self.zoom_extents()
        elif self.mode == "3d":
            self.view.set_view(name)

    def show_welcome(self, checked=False):
        """Help > Welcome to Volura: the welcome screen."""
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
        name = os.path.basename(self.doc.path) if self.doc.path else tr("Untitled")
        star = "*" if self.doc.dirty else ""
        self.setWindowTitle(f"{name}{star} - Volura")

    def show_entity_info(self):
        self.tray.show()
        self.tray.raise_()
        self.tray.open_panel(tr(PANEL_INFO))

    def select_color(self, color):
        self.settings.current_color = color
        if not any(c.lower() == color.lower() for _, c in self.doc.materials):
            self.doc.checkpoint()
            self.doc.materials.append([tr("Color {n}").format(n=len(self.doc.materials) + 1), color])
            self.doc.notify()
        self.materials.refresh()
        self.view.set_status(tr("Current color: {color}").format(color=color))

    # ------------------------------------------------------------ files
    def confirm_discard(self):
        if not self.doc.dirty or (not self.doc.groups and not self.doc.root.E
                                  and not self.doc.sheet.paths):
            return True
        r = QMessageBox.question(self, "Volura", tr("Save changes to the current model?"),
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
            path, _ = QFileDialog.getOpenFileName(
                self, tr("Open Model"), self.last_dir(),
                ";;".join([tr("Volura model (*.volura)"), mesh_filter()] + MESH_FILTERS))
        if not path:
            return
        if is_mesh(path):
            self.doc.reset()
            self.import_mesh(path)
            return
        try:
            self.view.tool.reset()
            fileio.load_document(self.doc, path)
        except Exception as e:
            QMessageBox.critical(self, tr("Open Failed"), str(e))
            return
        self.view.engine.cache.warm([g.ents for g in self.doc.groups])
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
            QMessageBox.critical(self, tr("Save Failed"), str(e))
            return False
        self.doc.dirty = False
        self.update_title()
        self.hint.setText(tr("Saved: {path}").format(path=self.doc.path))
        return True

    def file_save_as(self):
        path, _ = QFileDialog.getSaveFileName(self, tr("Save As"), self.last_dir("model.volura"),
                                              tr("Volura model (*.volura)"))
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
        path, _ = QFileDialog.getSaveFileName(self, tr("Save a Copy As"), self.last_dir(name),
                                              tr("Volura model (*.volura)"))
        if not path:
            return False
        if not path.lower().endswith(".volura"):
            path += ".volura"
        try:
            fileio.save_document(self.doc, path)
        except Exception as e:
            QMessageBox.critical(self, tr("Saving the Copy Failed"), str(e))
            return False
        self.add_recent(path)
        self.hint.setText(tr("Copy saved: {path}").format(path=path))
        return True

    def file_import(self):
        vector = tr("Vector drawings") + " (*.svg *.dxf)"
        everything = " ".join(f"*{e}" for e in MESH_EXTENSIONS + vector_import.VECTOR_EXTENSIONS)
        path, _ = QFileDialog.getOpenFileName(
            self, tr("Import"), self.last_dir(),
            ";;".join([tr("All supported files") + f" ({everything})", mesh_filter(), vector]
                      + MESH_FILTERS + ["SVG (*.svg)", "DXF (*.dxf)"]))
        if not path:
            return
        self.import_file(path)

    def import_file(self, path):
        """Import any supported file: a vector drawing or a 3D model."""
        if vector_import.is_vector(path):
            if self.mode == "2d":
                return self.import_vector_to_sheet(path)     # on the worksheet it stays flat
            return self.import_vector_interactive(path)
        return self.import_mesh(path)

    def import_mesh(self, path):
        """Import a 3D model (STL, OBJ, 3MF, ...): every closed volume in it becomes a group.

        The file is read and the groups are built in the background, with a progress bar;
        only adding them to the document happens on the GUI thread. Returns the job."""
        base = os.path.basename(path)
        file_name = os.path.splitext(base)[0]
        default_color = self.settings.current_color

        def read(report):
            # everything heavy happens here, on geometry that is not in the document yet:
            # reading, building the faces and the render and snap cache of a model with
            # tens of thousands of triangles takes seconds
            report(0.0, tr("Reading {name}…").format(name=base))
            volumes, notes = fileio.import_mesh_report(path, lambda f: report(0.3 * f))
            n = sum(v.manifold.num_tri() for v in volumes)
            done, parts = 0.0, []
            for v in volumes:
                share = v.manifold.num_tri() / max(n, 1)
                name = file_name if len(volumes) == 1 else v.name
                color = v.color or default_color

                def step(f, at=done, share=share):
                    report(0.3 + 0.7 * (at + share * f))
                report(0.3 + 0.7 * done, tr("Building {name}…").format(name=name))
                ents = Entities.from_manifold(v.manifold, progress=lambda f: step(0.7 * f))
                step(0.7)
                cache = GeoCache.build(ents)
                step(0.8)
                summary = ops.part_summary(name, color, ents, pool=True)
                parts.append((name, color, ents, cache, summary))
                done += share
            report(1.0, tr("Preparing the view…"))
            removed = sum(v.duplicates for v in volumes)
            loose = sum(v.loose for v in volumes)
            for v in volumes:
                if v.repair is not None and v.repair.changed:
                    name = file_name if len(volumes) == 1 else v.name
                    notes.insert(0, tr("{name} was repaired: {what}").format(
                        name=name, what=", ".join(repair.describe(v.repair))))
            return n, removed, loose, notes, parts

        def imported(result):
            n, removed, loose, notes, parts = result
            self.doc.checkpoint()
            groups = []
            for name, color, ents, cache, summary in parts:
                g = self.doc.add_group(ents, color, name)
                self.view.engine.cache.put(cache)
                self.info.remember_part(g, summary)
                groups.append(g)
            self.doc.active = ROOT
            self.doc.set_selection([("g", g.id, g.id) for g in groups])
            self.doc.notify()
            self.view.zoom_extents()
            said = []
            if len(groups) > 1:
                said.append(tr("{n} parts imported from {name}").format(n=len(groups), name=base))
            if removed:
                said.append(tr("{n} duplicate triangles removed from {name}").format(
                    n=removed, name=base))
            if loose:
                said.append(tr("{n} loose triangles removed from {name}").format(
                    n=loose, name=base))
            said += notes
            if said:
                self.hint.setText("; ".join(said))
            self._maybe_suggest_simplify(groups, n)

        def failed(e):
            QMessageBox.critical(self, tr("Import Failed"), str(e))

        return background.run(self, read, text=tr("Importing…"),
                              on_done=imported, on_error=failed)

    import_stl = import_mesh       # the name the STL import had

    # ------------------------------------------------------------ vector import
    def ask_vector_options(self, drawing, name="Vector"):
        """The 'Import Vector File' dialog; None on cancel, else an options dict."""
        b = drawing.bounds()
        w = (b[2] - b[0]) if b else 0.0
        h = (b[3] - b[1]) if b else 0.0
        dlg = QDialog(self)
        dlg.setWindowTitle(tr("Import Vector File"))
        form = QFormLayout(dlg)
        form.addRow(tr("Dimensions"), QLabel(f"{w:.1f} x {h:.1f} mm"))
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
        anchor.addItem(tr("Middle"), "center")
        anchor.addItem(tr("Bottom left"), "lower_left")
        open_edges = QCheckBox(tr("Import open lines as edges"))
        open_edges.setChecked(True)
        extrude = QCheckBox(tr("Extrude to a volume"))
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

        form.addRow(tr("Scale factor"), scale)
        form.addRow(tr("Target width"), width)
        form.addRow(tr("Anchor point"), anchor)
        form.addRow("", open_edges)
        form.addRow("", extrude)
        form.addRow(tr("Height"), height)
        form.addRow(tr("Color"), cbtn)
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
        """Place a drawing directly (without dialog or tool) as one undo step."""
        plane = plane or Plane((0, 0, 0), (0, 0, 1))
        point = np.zeros(3) if point is None else np.asarray(point, dtype=float)
        regios, open_paden = vector_import.prepare(
            drawing, scale=options.get("scale", 1.0),
            target_width=options.get("target_width"),
            anchor=options.get("anchor", "center"))
        if not regios and not open_paden:
            self.view.set_status(tr("There was nothing to import from the vector file."))
            return None
        if not options.get("open_edges", True):
            open_paden = []
        self.doc.checkpoint()
        offset = plane.to_local(point)
        group = None
        try:
            if options.get("extrude"):
                group = vector_import.place_extruded(
                    self.doc, plane, regios, offset, options.get("height", 3.0),
                    options.get("color", self.settings.current_color),
                    options.get("name", "Vector"))
                if options.get("open_edges", True) and open_paden:
                    self.place_faces(self.doc.active_ents(), plane, [], open_paden, offset)
            else:
                self.place_faces(self.doc.active_ents(), plane, regios, open_paden, offset)
        except background.BackgroundCancelled:
            self.doc.undo()
            self.doc.redo_stack.clear()
            return None
        self.doc.active = ROOT
        if group is not None:
            self.doc.set_selection([("g", group.id, group.id)])
        self.doc.notify()
        self.view.zoom_extents()
        return group

    def load_vector(self, path):
        """Read an SVG or DXF in the background, with a progress dialog when it takes a while.

        Returns the drawing, or None after showing the error (or when cancelled)."""
        name = os.path.basename(path)

        def read(report):
            report(None, tr("Reading {name}…").format(name=name))
            return vector_import.load_vector(path)
        job = background.run(self, read, text=tr("Reading {name}…").format(name=name), wait=True)
        if job.error is not None:
            QMessageBox.critical(self, tr("Import Failed"), str(job.error))
        return job.result

    #: from this many points on, a drawing is placed in the background
    BACKGROUND_PLACE_POINTS = 2000

    def place_faces(self, ents, plane, regios, open_paden, offset):
        """`vector_import.place_faces`, for a big drawing off the GUI thread with a progress
        bar: on a copy of `ents`, whose state then replaces that of `ents`. Raises
        BackgroundCancelled when the user cancels, and the error of the placing."""
        points = sum(len(loop) for shape in regios for loop in shape) + sum(
            len(p) for p in open_paden or ())
        if points < self.BACKGROUND_PLACE_POINTS:
            return vector_import.place_faces(ents, plane, regios, open_paden, offset)
        work = ents.copy()

        def place(report):
            report(0.0, tr("Placing the drawing…"))
            return vector_import.place_faces(work, plane, regios, open_paden, offset,
                                             progress=report)
        job = background.run(self, place, text=tr("Placing the drawing…"), wait=True)
        if job.was_cancelled:
            raise background.BackgroundCancelled()
        if job.error is not None:
            raise job.error
        ents.load_state(work.state())
        ents.touch()
        return job.result

    def import_vector(self, path, options=None):
        """Import an SVG/DXF onto the ground; `options` skips the dialog."""
        drawing = self.load_vector(path)
        if drawing is None:
            return None
        if options is None:
            options = self.ask_vector_options(drawing,
                                              os.path.splitext(os.path.basename(path))[0])
            if options is None:
                return None
        return self.place_vector(drawing, options)

    def import_vector_interactive(self, path, drawing=None, name=None):
        """Choose options and place the drawing with the placement tool behind the cursor."""
        if drawing is None:
            drawing = self.load_vector(path)
            if drawing is None:
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
            self.view.set_status(tr("There was nothing to import from the vector file."))
            return
        options["name"] = name
        self.view.tools["vectorplace"].set_drawing(regios, open_paden, options)
        self.set_tool("vectorplace")

    def add_drawing_to_sheet(self, drawing, message=None, **fields):
        """Put a drawing on the worksheet as one undo step and select the new paths.

        `message` is a translated template with {n} (the number of paths) and `fields`."""
        if drawing is None or not drawing.paths:
            self.active_view().set_status(tr("There was nothing to add to the worksheet."))
            self.hint.setText(tr("Nothing was added."))
            return None
        self.doc.checkpoint()
        added = self.doc.sheet.merge(drawing)
        self.doc.notify()
        self.set_mode("2d")
        self.sheet_view.set_selection(added)
        self.sheet_view.zoom_extents()
        message = message or tr("{n} outlines added to the worksheet")
        self.hint.setText(message.format(n=len(added), **fields))
        return added

    def import_vector_to_sheet(self, path, drawing=None):
        """File > Import in 2D: the SVG or DXF on the worksheet, in its own place.

        The positions from the file stay as they are and layers with the same name are
        merged; the new paths are selected right away.
        """
        name = os.path.splitext(os.path.basename(path))[0] if path else "Vector"
        if drawing is None:
            drawing = self.load_vector(path)
            if drawing is None:
                return None
        added = self.add_drawing_to_sheet(drawing, tr("{n} outlines added from {name}"), name=name)
        if added is not None and drawing.warnings:
            self.hint.setText(tr("{n} outlines added from {name} ({warnings})").format(
                n=len(added), name=name, warnings="; ".join(drawing.warnings)))
        return added

    def copy_to_sheet(self, options=None):
        """Edit > Copy to 2D Worksheet: the same choices as the 2D export."""
        opts = self.vector_export_options() if options is None else dict(options)
        if opts is None:
            return None
        try:
            drawing = self.vector_export_drawing(opts)
        except Exception as e:
            QMessageBox.critical(self, tr("Copy Failed"), str(e))
            return None
        if not drawing.paths:
            msg = self.vector_export_empty_message(opts, drawing)
            if drawing.warnings:
                msg += "\n\n" + "\n".join(drawing.warnings)
            QMessageBox.information(self, tr("Copy to 2D Worksheet"), msg)
            self.hint.setText(tr("Nothing was copied."))
            return None
        return self.add_drawing_to_sheet(drawing, tr("{n} outlines copied to the worksheet"))

    # ------------------------------------------------------------ 3D and worksheet together
    def _laser_layers(self):
        """The layers of the worksheet that have an operation (not the guides)."""
        return [(n, lay) for n, lay in self.doc.sheet.layers.items()
                if lay.op != "none" and not lay.locked]

    def _add_laser_menu(self, menu):
        """Right click in 3D: "Laser / Mill" with a line per layer of the worksheet."""
        sm = menu.addMenu(icon("to_laser"), tr("Laser / Mill"))
        face = getattr(self, "_context_face", None)
        for name, lay in self._laser_layers():
            a = sm.addAction(swatch(lay.color or "#000000", 14),
                             f"{name}   —   {OP_LABELS.get(lay.op, lay.op)}")
            a.triggered.connect(lambda checked=False, n=name, f=face:
                                self.send_selection_to_sheet(n, f))
        return sm

    def send_selection_to_sheet(self, layer=None, face=None):
        """The 3D selection as an operation on the worksheet: faces at true size,
        selected groups (plates) as a top view.

        `face` is the face (ctx, fid) that was right-clicked, also when it lies in a
        group: then that face goes, not the top view of the whole group (unless it is
        part of a selection of more faces). Without `layer` the layer is asked. The
        outlines go to the right of what is already on the worksheet. If there is no
        sheet thickness yet, the worksheet takes the thickness of the selected plates
        when they are all equally thick.
        """
        if layer is None or isinstance(layer, bool):
            names = [f"{n}   —   {OP_LABELS.get(l.op, l.op)}" for n, l in self._laser_layers()]
            if not names:
                self.hint.setText(tr("The worksheet has no layer with an operation."))
                return None
            choice, ok = QInputDialog.getItem(self, tr("To Laser or Mill"),
                                              tr("On which layer (operation)?"), names, 0, False)
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
            QMessageBox.critical(self, tr("To Laser or Mill"), str(e))
            return None
        if drawing is None or not drawing.paths:
            self.hint.setText(tr("First select faces or plates (groups) in the model."))
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
        added = self.add_drawing_to_sheet(drawing, tr("{n} outlines put on layer {layer}"),
                                          layer=layer)
        if added and thickness and not sheet.sheet_thickness:
            sheet.sheet_thickness = thickness
            self.hint.setText(tr("{n} outlines put on layer {layer}; sheet thickness {t:g} mm "
                                 "taken over").format(n=len(added), layer=layer, t=thickness))
        return added

    def _face_plate_thickness(self, faces):
        """The sheet thickness for horizontal faces in a group: the height of that group."""
        groups = []
        for c, i in faces:
            g = next((g for g in self.doc.groups if g.id == c), None)
            if g is None or abs(g.ents.F[i].normal[2]) < 0.999:
                return None
            groups.append(g)
        return sheet3d.solid_thickness(groups)

    def _place_beside_sheet(self, drawing):
        """Move a new drawing to the right of the worksheet's content (or into the corner
        of the sheet when it is still empty), so nothing overlaps."""
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
        """Worksheet > Parts to 3D Model: everything that is cut, as a plate.

        Every part becomes its own group with the sheet thickness, at the same X/Y as
        on the worksheet. Parts put into the model an earlier time are replaced; with a
        selection on the worksheet only those parts go.
        """
        sheet = self.doc.sheet
        t = thickness or sheet.sheet_thickness
        if not t:
            t, ok = QInputDialog.getDouble(self, tr("Sheet Thickness"), tr("Thickness of the sheet (mm):"),
                                           sheet3d.DEFAULT_THICKNESS, 0.1, 500.0, 2)
            if not ok:
                return None
        selected = self.sheet_view.selected_paths() if self.mode == "2d" else []
        try:
            solids = sheet3d.part_solids(sheet, t, selected or None)
        except Exception as e:
            QMessageBox.critical(self, tr("Parts to 3D Model"), str(e))
            return None
        if not solids:
            self.hint.setText(tr("There are no closed cutting outlines on the worksheet "
                                 "(layer Cut, Mill or Drill)."))
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
        message = (tr("{n} part of {t:g} mm placed in the model") if len(groups) == 1
                   else tr("{n} parts of {t:g} mm placed in the model"))
        self.hint.setText(message.format(n=len(groups), t=float(t)))
        return groups

    # ------------------------------------------------------------ export for laser and mill
    def sheet_layers_for_export(self) -> list:
        """The layers of the worksheet that mean something to the machine."""
        return [n for n, lay in self.doc.sheet.layers.items()
                if lay.visible and lay.op != "none"]

    def default_sheet_export_options(self) -> dict:
        """The default choices: all visible work layers, compensation only for milling."""
        names = self.sheet_layers_for_export()
        mill = any(self.doc.sheet.layers[n].op in MILL_OPS
                   and any(p.layer == n for p in self.doc.sheet.paths) for n in names)
        return {"layers": names, "compensate": mill, "per_layer": False, "per_sheet": False}

    def sheet_export_options(self, kind="svg"):
        """The "Export for Laser/Mill" dialog; None on cancel, else an options dict."""
        dlg = SheetExportDialog(self, kind)
        if dlg.exec() != QDialog.Accepted:
            return None
        return dlg.options()

    def sheet_export_drawing(self, options) -> "Drawing":
        """The drawing of the chosen layers, with the compensation when it is on."""
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
        """`filename_layername.svg`: the layer name made fit for a file name."""
        base = os.path.splitext(path)[0]
        safe = ("".join(ch for ch in str(name) if ch not in '<>:"/\\|?*\n').strip(" .")
                or tr("layer"))
        return f"{base}_{safe}.{kind}"

    @staticmethod
    def sheet_file_name(path, index, kind) -> str:
        """`filename_sheet1.svg`: the file of one sheet (the word in the user's language)."""
        base = os.path.splitext(path)[0]
        return tr("{base}_sheet{index}.{ext}").format(base=base, index=int(index), ext=kind)

    def export_sheet(self, kind="svg", path=None, options=None):
        """Export > 2D vector on the worksheet: "Export for Laser/Mill".

        Without `path` the dialog asks which layers may go, whether the kerf and cutter
        compensation should be applied and whether every layer gets its own file; with a
        `path` the default choices are used. Raster layers go out filled, the rest as
        hairlines, and the operation and settings of every layer go into the SVG so
        Volura finds them again when reading it.
        """
        kind = "dxf" if str(kind).lower() == "dxf" else "svg"
        if options is None:
            options = (self.sheet_export_options(kind) if path is None
                       else self.default_sheet_export_options())
            if options is None:
                return None
        names = [n for n in (options.get("layers") or ()) if n in self.doc.sheet.layers]
        if not names:
            QMessageBox.information(self, tr("Export for Laser/Mill"),
                                    tr("Choose at least one layer with an operation. A layer set "
                                       "to \"None\" or a hidden layer does not go along."))
            self.hint.setText(tr("Nothing was exported."))
            return None
        try:
            drawing = self.sheet_export_drawing(options)
        except Exception as e:
            QMessageBox.critical(self, tr("Export Failed"), str(e))
            return None
        if not drawing.paths:
            QMessageBox.information(self, tr("Export for Laser/Mill"),
                                    tr("There are no outlines on the chosen layers. Draw "
                                       "something, or import an SVG or DXF file."))
            self.hint.setText(tr("Nothing was exported."))
            return None
        if path is None:
            path, _ = QFileDialog.getSaveFileName(
                self, tr("Export for Laser/Mill"), self.export_name("." + kind),
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
            QMessageBox.critical(self, tr("Export Failed"), str(e))
            return None
        where = (files[0] if len(files) == 1
                 else tr("{n} files ({first})").format(n=len(files), first=files[0]))
        message = (tr("{n} outlines exported to {where} with kerf/cutter compensation")
                   if options.get("compensate") else tr("{n} outlines exported to {where}"))
        self.hint.setText(message.format(n=len(drawing.paths), where=where))
        return files[0]

    @staticmethod
    def _write_vector(drawing, path, kind, layers, fill_layers=()):
        """Write one drawing to an SVG or DXF file, with the chosen layers."""
        if kind == "dxf":
            dxf.write_dxf(drawing, path, layers=layers)
        else:
            svg.write_svg(drawing, path, layers=layers, fill_layers=fill_layers)

    def import_dropped_file(self, path):
        """A dropped file: a 3D model or a vector drawing."""
        if vector_import.is_vector(path) or is_mesh(path):
            self.import_file(path)

    def dragEnterEvent(self, e):
        md = e.mimeData()
        if md.hasUrls() and any(vector_import.is_vector(u.toLocalFile()) or
                                is_mesh(u.toLocalFile()) for u in md.urls()):
            e.acceptProposedAction()

    def dragMoveEvent(self, e):
        self.dragEnterEvent(e)

    def dropEvent(self, e):
        md = e.mimeData()
        if not md.hasUrls():
            return
        paths = [u.toLocalFile() for u in md.urls() if u.isLocalFile()]
        supported = [p for p in paths if vector_import.is_vector(p) or is_mesh(p)]
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
        return self._validated_parts(ops.parts(self.doc, only_selected=only))

    def _validated_parts(self, part_list):
        """The export warnings for an already computed part list; the list, or None to stop."""
        if not part_list:
            QMessageBox.information(self, tr("Export"), tr("There are no volumes to export. "
                                                           "Draw a face and use Push/Pull (P)."))
            return None
        bad = [p for p in part_list if p.problem]
        if bad:
            text = "\n".join(f"• {p.name}: {p.problem}" for p in bad)
            if not [p for p in part_list if p.manifold is not None]:
                self.ask_export(tr("Not a single part is a closed volume:\n\n{parts}").format(
                    parts=text))
                return None
            if not self.ask_export(tr("These parts are not closed and are skipped:\n\n{parts}"
                                      "\n\nExport anyway?").format(parts=text)):
                return None
        return part_list

    def ask_export(self, text):
        """The export warning, with a button that opens the Print Check panel."""
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Warning)
        box.setWindowTitle(tr("Export"))
        box.setText(text)
        b_open = box.addButton(tr("Open Print Check"), QMessageBox.ActionRole)
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
        self._export_parts_async("stl")

    def export_3mf(self):
        self._export_parts_async("3mf")

    def export_stl_colors(self):
        self._export_parts_async("colors")

    def _export_parts_async(self, kind):
        """Export the printable parts in the background: prepare, ask, write.

        ``kind`` is "stl", "3mf" or "colors". Turning the model into solids and
        writing the file run in the background so the window keeps responding; the
        warnings and the file-name dialog stay on the GUI thread, in the same order
        as a synchronous export. Only the preparation can be cancelled, and then no
        file is written; the write itself is atomic and runs to its end.
        """
        only = bool(self.doc.selected_groups())

        def prepare(report):
            report(0.0, tr("Preparing…"))
            return ops.parts(self.doc, only_selected=only)

        def prepared(part_list):
            part_list = self._validated_parts(part_list)
            if part_list is None:
                return
            if kind == "stl":
                path, _ = QFileDialog.getSaveFileName(self, tr("Export as STL"),
                                                      self.export_name(".stl"), "STL (*.stl)")
            elif kind == "3mf":
                path, _ = QFileDialog.getSaveFileName(self, tr("Export as 3MF"),
                                                      self.export_name(".3mf"), "3MF (*.3mf)")
            else:
                path, _ = QFileDialog.getSaveFileName(self, tr("STL per Color (base name)"),
                                                      self.export_name(".stl"), "STL (*.stl)")
            if path:
                self._write_export_async(kind, path, part_list)

        background.run(self, prepare, text=tr("Preparing…"),
                       on_done=prepared, on_error=self._export_failed)

    def _write_export_async(self, kind, path, part_list):
        """Write an already validated part list to `path` in the background."""
        materials = [list(m) for m in self.doc.materials]
        name = os.path.splitext(os.path.basename(path))[0]

        def write(report):
            report(0.0, tr("Exporting…"))
            if kind == "stl":
                return ("stl", fileio.export_stl(path, part_list))
            if kind == "3mf":
                return ("3mf", fileio.export_3mf(path, part_list, materials, name))
            return ("colors", fileio.export_stl_per_color(path, part_list, materials))

        def written(outcome):
            what, value = outcome
            if what == "stl":
                self.hint.setText(tr("STL exported ({n} triangles, watertight): {path}")
                                  .format(n=value, path=path))
            elif what == "3mf":
                lines = "\n".join(tr("  extruder {e}: {name}").format(e=e, name=n)
                                  for n, e in value)
                QMessageBox.information(self, tr("3MF Exported"),
                                        tr("{path}\n\nColors per filament:\n{lines}\n\n"
                                           "Open the file in PrusaSlicer, OrcaSlicer or Bambu Studio; "
                                           "every color is already on the right filament. The slicer "
                                           "shows its own filament colors: give its filaments these "
                                           "colors to see the model as in Volura.").format(
                                            path=path, lines=lines))
            else:
                QMessageBox.information(self, tr("STL per Color"),
                                        tr("Written:\n{files}\n\nLoad them together in your slicer as "
                                           "one object with several parts.").format(
                                            files="\n".join(value)))

        background.run(self, write, text=tr("Exporting…"), can_cancel=False,
                       on_done=written, on_error=self._export_failed)

    def _export_failed(self, exc):
        QMessageBox.critical(self, tr("Exporting Failed"), str(exc))

    # ------------------------------------------------------------ 2D vector export
    def vector_export_options(self):
        """The 'Export 2D Vector' dialog; None on cancel, else an options dict."""
        dlg = VectorExportDialog(self)
        if dlg.exec() != QDialog.Accepted:
            return None
        return dlg.options()

    def vector_export_drawing(self, options):
        """The drawing for these choices: selected faces, top view or section."""
        doc = self.doc
        source = options.get("source", "top")
        if source == "faces":
            return vector_export.drawing_from_faces(doc)
        only = bool(options.get("only_selected"))
        if source == "section":
            return vector_export.drawing_section(doc, float(options.get("z", 0.0)), only)
        return vector_export.drawing_top(doc, only)

    def vector_export_empty_message(self, options, drawing):
        """Why there is nothing to write, in words that help the user."""
        source = options.get("source", "top")
        if source == "faces":
            return tr("No face is selected. Click a face (or draw a rectangle) and try again.")
        only = bool(options.get("only_selected"))
        if source == "section" and ops.parts(self.doc, only_selected=only):
            return tr("At height {z} the model cuts nothing: there is no volume there.").format(
                z=format_length(float(options.get('z', 0.0))))
        return tr("There are no closed volumes to export. Draw a face and use Push/Pull (P), "
                  "or close the group with Print Check.")

    def export_vector(self, kind="svg", path=None, options=None):
        """File > Export > 2D Vector: write the drawing to SVG or DXF.

        Without `options` the dialog asks what should go; without `path` Qt asks for a
        file name. On the 2D worksheet the dialog is about the worksheet itself and only
        the visible layers are written. Returns the written path, or None.

        With a `path` the whole export runs synchronously (and returns the path); the
        interactive export without one runs in the background, so a big model does not
        freeze the window, and returns None right away.
        """
        kind = "dxf" if str(kind).lower() == "dxf" else "svg"
        if self.mode == "2d":
            return self.export_sheet(kind, path)
        opts = self.vector_export_options() if options is None else dict(options)
        if opts is None:
            return None
        if path is not None:
            return self._export_vector_sync(kind, path, opts)
        return self._export_vector_async(kind, opts)

    def _export_vector_sync(self, kind, path, opts):
        """Compute, check and write the drawing now; returns the path, or None."""
        try:
            drawing = self.vector_export_drawing(opts)
        except Exception as e:
            QMessageBox.critical(self, tr("Export Failed"), str(e))
            return None
        if not drawing.paths:
            msg = self.vector_export_empty_message(opts, drawing)
            if drawing.warnings:
                msg += "\n\n" + "\n".join(drawing.warnings)
            QMessageBox.information(self, tr("Export 2D Vector"), msg)
            self.hint.setText(tr("Nothing was exported."))
            return None
        try:
            (dxf.write_dxf if kind == "dxf" else svg.write_svg)(drawing, path)
        except Exception as e:
            QMessageBox.critical(self, tr("Export Failed"), str(e))
            return None
        self.hint.setText(tr("{n} outlines exported to {where}").format(n=len(drawing.paths),
                                                                       where=path))
        return path

    def _export_vector_async(self, kind, opts):
        """The interactive export: build the drawing in the background, then ask and write."""
        def compute(report):
            report(0.0, tr("Preparing…"))
            return self.vector_export_drawing(opts)

        def computed(drawing):
            if not drawing.paths:
                msg = self.vector_export_empty_message(opts, drawing)
                if drawing.warnings:
                    msg += "\n\n" + "\n".join(drawing.warnings)
                QMessageBox.information(self, tr("Export 2D Vector"), msg)
                self.hint.setText(tr("Nothing was exported."))
                return
            path, _ = QFileDialog.getSaveFileName(
                self, tr("Export 2D Vector"), self.export_name("." + kind),
                "SVG (*.svg)" if kind == "svg" else "DXF (*.dxf)")
            if not path:
                return
            if not os.path.splitext(path)[1]:
                path += "." + kind
            self._write_vector_async(drawing, path, kind)

        background.run(self, compute, text=tr("Preparing…"), on_done=computed,
                       on_error=self._vector_export_failed)
        return None

    def _write_vector_async(self, drawing, path, kind):
        """Write an already built drawing to `path` in the background."""
        def write(report):
            report(0.0, tr("Exporting…"))
            (dxf.write_dxf if kind == "dxf" else svg.write_svg)(drawing, path)
            return len(drawing.paths)

        def written(n):
            self.hint.setText(tr("{n} outlines exported to {where}").format(n=n, where=path))

        background.run(self, write, text=tr("Exporting…"), can_cancel=False, on_done=written,
                       on_error=self._vector_export_failed)

    def _vector_export_failed(self, exc):
        QMessageBox.critical(self, tr("Export Failed"), str(exc))

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
        """Edit > Lock: say in the status bar why nothing happened."""
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
        """Edit > Delete Guides: every guide of the document, in one undo step."""
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
        """Edit > Hide: hide the selection and remember what it was."""
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
        """Edit > Unhide > Selected (useful with Hidden Geometry on)."""
        items = self._hidden_items(self.doc.selection)
        if not items:
            return
        self.doc.checkpoint()
        self._show_items(items)
        self.doc.notify()

    def unhide_last(self):
        """Edit > Unhide > Last: what Edit > Hide hid last."""
        items = self._hidden_items(self.doc.last_hidden)
        if not items:
            return
        self.doc.checkpoint()
        self._show_items(items)
        self.doc.last_hidden = []
        self.doc.notify()

    def unhide_all(self):
        """Edit > Unhide > All: every hidden face, edge and group."""
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
        """Edit > Lock: the selected groups can no longer be changed."""
        groups = [g for g in self.doc.selected_groups() if not g.locked]
        if not groups:
            return
        self.doc.checkpoint()
        for g in groups:
            g.locked = True
        self.doc.notify()

    def unlock_selected(self):
        """Edit > Unlock > Selected."""
        groups = [g for g in self.doc.selected_groups() if g.locked]
        if not groups:
            return
        self.doc.checkpoint()
        for g in groups:
            g.locked = False
        self.doc.notify()

    def unlock_all(self):
        """Edit > Unlock > All."""
        groups = [g for g in self.doc.groups if g.locked]
        if not groups:
            return
        self.doc.checkpoint()
        for g in groups:
            g.locked = False
        self.doc.notify()

    def copy(self):
        self.clipboard = list(self.doc.selection)
        self.hint.setText(tr("{n} item(s) copied.").format(n=len(self.clipboard)))

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
        """Edit > Paste: import SVG from the clipboard (e.g. from Inkscape)."""
        md = QApplication.clipboard().mimeData()
        if md is None or not md.hasFormat("image/svg+xml"):
            return False
        raw = bytes(md.data("image/svg+xml"))
        try:
            drawing = vector_import.load_svg_text(raw.decode("utf-8", "replace"))
        except Exception as e:
            QMessageBox.critical(self, tr("Paste Failed"), str(e))
            return True
        self.import_vector_interactive(None, drawing=drawing, name=tr("Pasted SVG"))
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
        """Camera > Field of View: the Zoom tool with the field of view in the Measurements box."""
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
        """View > Guides: show or hide the construction guides."""
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
            QMessageBox.information(self, "Volura", tr("Select at least {n} group(s). First make a "
                                                       "group of loose geometry (G).").format(n=minimum))
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
        """Tools > Solid Tools > Trim: the second group minus the first."""
        groups = self._sel_groups(2)
        if not groups:
            return
        if len(groups) != 2:
            QMessageBox.information(self, "Volura", tr("Select exactly two groups: the second is "
                                                       "trimmed with the first."))
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
        """Tools > Outer Shell: only the outside of the selected groups."""
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

    # ------------------------------------------------------------ simplify model

    def simplify_model(self):
        """Tools > Simplify Model: fewer triangles on the selected groups."""
        groups = self.doc.selected_groups()
        if not groups:
            self.hint.setText(tr("Select the group(s) to simplify first."))
            return
        groups = self.unlocked(groups)
        if not groups:
            return
        self.simplify_groups(groups)

    def simplify_groups(self, groups):
        """Show the Simplify Model dialog for `groups` and apply the chosen quality."""
        dlg = SimplifyDialog(self, groups)
        if dlg.exec() != QDialog.Accepted:
            return None
        return self._run_simplify(groups, dlg.tolerance, dlg.manifolds)

    def _run_simplify(self, groups, tolerance, manifolds=None):
        """Simplify every group in the background, then swap the new geometry in with one
        checkpoint, so a single undo restores them all. `manifolds` ({group id: Manifold})
        are those the dialog already made for its estimate."""
        groups = [g for g in groups if not g.locked]
        manifolds = dict(manifolds or {})

        def work(report):
            # building the new faces and their render cache is the slow part: seconds
            # for a big model, so it happens here and the window keeps responding
            out = []
            for i, g in enumerate(groups):
                report(i / len(groups), tr("Simplifying {name}…").format(name=g.name))
                before, after, ents = ops.simplified(g, tolerance, manifolds.get(g.id))
                out.append((g, before, after, ents, GeoCache.build(ents),
                            ops.part_summary(g.name, g.color, ents, pool=True)))
            return out

        def done(out):
            self.doc.checkpoint()
            for g, _b, _a, ents, cache, summary in out:
                g.ents = ents
                self.view.engine.cache.put(cache)
                self.info.remember_part(g, summary)
            self.doc.notify()
            before = sum(o[1] for o in out)
            after = sum(o[2] for o in out)
            self.hint.setText(tr("Simplified: {n} fewer triangles ({before} -> {after}).")
                              .format(n=before - after, before=before, after=after))

        if not groups:
            return None
        return background.run(self, work, text=tr("Simplifying…"),
                              on_done=done, on_error=self._failed)

    def _maybe_suggest_simplify(self, g, n=None):
        """After a large import: ask whether to simplify the new group (or list of groups)
        right away. `n` is the triangle count when the caller already knows it (the import
        does)."""
        groups = [x for x in (g if isinstance(g, (list, tuple)) else [g]) if x is not None]
        if not groups or not self.settings.ask_simplify_import:
            return
        n = sum(ops.tri_count(x.ents) for x in groups) if n is None else n
        if n <= ops.LARGE_IMPORT_TRIANGLES:
            return
        if self._ask_simplify_import(n):
            self.simplify_groups(groups)

    def _ask_simplify_import(self, n):
        """The large-import question; True when the user wants to simplify now. The
        'Don't ask again' box turns the advice off in the settings (Preferences)."""
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Question)
        box.setWindowTitle(tr("Large Import"))
        box.setText(tr("This model has {n} triangles, which can make Volura slow. Simplify "
                       "it now? Recommended keeps the print quality (deviation under 0.02 "
                       "mm).").format(n=n))
        simplify_btn = box.addButton(tr("Simplify…"), QMessageBox.AcceptRole)
        box.addButton(tr("Not now"), QMessageBox.RejectRole)
        again = QCheckBox(tr("Don't ask again"))
        box.setCheckBox(again)
        box.exec()
        if again.isChecked():
            self.settings.ask_simplify_import = False
            self.settings.save()
        return box.clickedButton() is simplify_btn

    def intersect_faces(self, with_model):
        """Edit > Intersect Faces: the cut lines of the selection go into the model.

        "With Model" cuts with everything else in the model, "With Selection" only with
        the other selected items. The edges are added to the active context."""
        g = self.doc.group(self.doc.active) if self.doc.active != ROOT else None
        if g is not None and g.locked:
            self.warn_locked()
            return
        sel = [it for it in self.doc.selection if it[0] in ("f", "g")]
        if not sel:
            QMessageBox.information(self, "Volura", tr("First select the faces or groups whose "
                                                       "intersection lines you want."))
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
            self.view.set_status(tr("No intersection lines found."))
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
            "box": [(tr("Width (X)"), 20), (tr("Depth (Y)"), 20), (tr("Height (Z)"), 20)],
            "cylinder": [(tr("Radius"), 10), (tr("Height"), 20)],
            "cone": [(tr("Bottom radius"), 10), (tr("Top radius"), 0), (tr("Height"), 20)],
            "sphere": [(tr("Radius"), 10)],
            "torus": [(tr("Major radius"), 15), (tr("Tube radius"), 4)],
        }[kind]
        dlg = QDialog(self)
        dlg.setWindowTitle(tr("Insert"))
        form = QFormLayout(dlg)
        spins = []
        for label, v in fields:
            sp = QDoubleSpinBox()
            sp.setRange(0 if label == tr("Top radius") else 0.01, 10000)
            sp.setDecimals(2)
            sp.setSuffix(" mm")
            sp.setValue(v)
            form.addRow(label, sp)
            spins.append(sp)
        seg = QSpinBox()
        seg.setRange(3, 512)
        seg.setValue(self.settings.circle_segments)
        if kind != "box":
            form.addRow(tr("Segments"), seg)
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
            QMessageBox.critical(self, tr("Insert"), str(e))
            return
        bx, by, _ = self.settings.bed_size
        lo, hi = np.asarray(m.bounding_box()[:3]), np.asarray(m.bounding_box()[3:])
        m = m.translate((bx / 2 - (lo[0] + hi[0]) / 2, by / 2 - (lo[1] + hi[1]) / 2, -lo[2]))
        names = {"box": tr("Box"), "cylinder": tr("Cylinder"), "cone": tr("Cone"),
                 "sphere": tr("Sphere"), "torus": tr("Torus")}
        self.doc.checkpoint()
        g = ops.group_from_manifold(self.doc, m, self.settings.current_color,
                                    f"{names[kind]} {self.doc.group_counter}")
        self.doc.group_counter += 1
        self.doc.active = ROOT
        self.doc.set_selection([("g", g.id, g.id)])
        self.doc.notify()
        self.set_tool("move")

    # ------------------------------------------------------------ screw thread
    THREAD_KINDS = [(tr("External thread on cylinder"), "external"),
                    (tr("Internal thread in hole"), "internal"),
                    (tr("New bolt"), "bolt"), (tr("New nut or sleeve"), "nut")]

    def _thread_target(self, kind):
        """(ents, group, center, axis, radius, length) from the selection, or None.

        Valid: a selected circular end face (external thread) or the opening of a hole
        (internal thread), or a group that is a cylinder/sleeve."""
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
        dlg.setWindowTitle(tr("Screw Thread"))
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
        left_cb = QCheckBox(tr("Left-hand thread"))
        outer_sp = QDoubleSpinBox()
        outer_sp.setRange(2.0, 500.0)
        outer_sp.setDecimals(1)
        outer_sp.setSuffix(" mm")
        outer_sp.setValue(13.6)
        hex_cb = QCheckBox(tr("Hexagon (uncheck for a round sleeve)"))
        hex_cb.setChecked(True)
        mate_cb = QCheckBox(tr("Also make a matching nut/bolt"))
        form.addRow(tr("Kind"), kind_cb)
        form.addRow(tr("Diameter"), dia)
        form.addRow(tr("Pitch (ISO coarse, free to enter)"), pitch_cb)
        form.addRow(tr("Length / height"), len_sp)
        form.addRow(tr("Clearance"), clr_sp)
        form.addRow(tr("Number of starts"), starts_sp)
        form.addRow("", left_cb)
        form.addRow(tr("Outer size of nut/sleeve"), outer_sp)
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
            self.view.set_status(tr("Screw thread failed: {error}").format(error=e))

    def _mate_group(self, man, label, near_bb, dx):
        """Put a matching part next to the edited part and make a group of it."""
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
                    tr("First select a group (or its circular face) with a cylinder.")
                    if kind == "external" else
                    tr("First select a group (or the face with the opening) with a hole."))
                return
            ents, g, center, axis, radius, depth = t
            if g is not None and g.locked:
                self.warn_locked()
                return
            m = ents.to_manifold()
            if m.status() != mf.Error.NoError or m.is_empty():
                self.view.set_status(tr("This part is not a closed volume."))
                return
            near = g.name if g else tr("Loose geometry")
            if kind == "external":
                res = threads.external_on(m, center, axis, radius, length, pitch=pitch,
                                          starts=starts, left=left)
                message = (tr("External thread M{d:g}x{p:g} put on {near} including a matching "
                              "part.") if mate else
                           tr("External thread M{d:g}x{p:g} put on {near}."))
            else:
                res = threads.internal_in(m, center, axis, radius, length, clearance,
                                          pitch=pitch, starts=starts, left=left)
                message = (tr("Internal thread M{d:g}x{p:g} put on {near} including a matching "
                              "part.") if mate else
                           tr("Internal thread M{d:g}x{p:g} put on {near}."))
            if res.status() != mf.Error.NoError or res.is_empty():
                self.view.set_status(tr("Screw thread failed: the result is not a valid volume."))
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
                    mg = self._mate_group(nm, tr("Nut M{d:g}x{p:g}").format(d=d, p=pitch), bb,
                                          outer_d / 2 + 5)
                else:
                    nm = threads.thread_solid(d, pitch, max(length, 2 * d), starts=starts,
                                              left=left)
                    mg = self._mate_group(nm, tr("Bolt M{d:g}x{p:g}").format(d=d, p=pitch), bb,
                                          d + 5)
                if mg is not None:
                    sel.append(("g", mg.id, mg.id))
            doc.set_selection(sel)
            doc.notify()
            self.view.set_status(message.format(d=d, p=pitch, near=near))
            return

        if kind == "bolt":
            man = threads.thread_solid(d, pitch, length, starts=starts, left=left)
            label = tr("Bolt M{d:g}x{p:g}").format(d=d, p=pitch)
        else:
            man = threads.nut(d, pitch, length, outer_d, clearance, hex=hexagon,
                              starts=starts, left=left)
            label = (tr("Nut M{d:g}x{p:g}") if hexagon else tr("Sleeve M{d:g}x{p:g}")).format(
                d=d, p=pitch)
        if man.status() != mf.Error.NoError or man.is_empty():
            self.view.set_status(tr("Screw thread failed: invalid dimensions."))
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
                mate_label = (tr("Nut M{d:g}x{p:g}") if hexagon
                              else tr("Sleeve M{d:g}x{p:g}")).format(d=d, p=pitch)
                mg = self._mate_group(nm, mate_label, man.bounding_box(), outer_d / 2 + 5)
            else:
                nm = threads.thread_solid(d, pitch, max(2 * d, length), starts=starts, left=left)
                mg = self._mate_group(nm, tr("Bolt M{d:g}x{p:g}").format(d=d, p=pitch),
                                      man.bounding_box(), d + 5)
            if mg is not None:
                sel.append(("g", mg.id, mg.id))
        doc.active = ROOT
        doc.set_selection(sel)
        doc.notify()
        self.view.set_status((tr("{label} placed with a matching part.") if mate
                              else tr("{label} placed.")).format(label=label))

    # ------------------------------------------------------------ cut & join
    # (label, joints.KINDS identifier; the identifiers stay Dutch)
    CUT_JOINTS = [(tr("No joint (cut only)"), "geen"),
                  (tr("Pins (on the upper part)"), "pennen"),
                  (tr("Holes for loose pins"), "losse pennen"),
                  (tr("Dovetail"), "zwaluwstaart")]

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
            self.view.set_status(tr("Cut: the selection is not a closed volume."))
            return
        if origin is None:
            bb = np.asarray(model.bounding_box())
            origin = tuple((bb[:3] + bb[3:]) / 2)
            normal = (0.0, 0.0, 1.0)

        d = joints.DEFAULTS
        dlg = QDialog(self)
        dlg.setWindowTitle(tr("Cut and Join"))
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
        sep = QCheckBox(tr("Move the parts apart (10 mm apart)"))
        flat = QCheckBox(tr("Lay the parts flat on the bed"))
        form.addRow(tr("Joint"), kind_cb)
        form.addRow(tr("Pin diameter"), dia)
        form.addRow(tr("Pin length"), length)
        form.addRow(tr("Number of pins"), count_cb)
        form.addRow(tr("Wall thickness around hole"), wall)
        form.addRow(tr("Tail width"), tw)
        form.addRow(tr("Tail depth"), td)
        form.addRow(tr("Tail angle"), ta)
        form.addRow(tr("Clearance"), clr)
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
            self.view.set_status(tr("Cut failed: {error}").format(error=e))
            return
        if (a.is_empty() or b.is_empty() or a.status() != mf.Error.NoError
                or b.status() != mf.Error.NoError):
            self.view.set_status(tr("The cut did not give two valid parts."))
            return
        doc = self.doc
        color, name = groups[0].color, groups[0].name
        doc.checkpoint()
        first = groups[0]
        first.ents = Entities.from_manifold(a)
        for g in groups[1:]:
            doc.remove_group(g.id)
        part_b = doc.add_group(Entities.from_manifold(b), color,
                               tr("{name} (part B)").format(name=name))
        sel = [("g", first.id, first.id), ("g", part_b.id, part_b.id)]
        for i, pin in enumerate(extras):
            pg = ops.group_from_manifold(doc, pin, self.settings.current_color,
                                         tr("Pin {n}").format(n=i + 1))
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
        self.view.set_status((tr("Model cut with {joint} and {n} loose pin(s).") if extras
                              else tr("Model cut with {joint}.")).format(joint=label, n=len(extras)))

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
        dlg.setWindowTitle(tr("3D Text"))
        form = QFormLayout(dlg)
        ed = QLineEdit(getattr(self, "_last_text", "Volura"))
        font = QFontComboBox()
        font.setCurrentFont(self.font())
        bold = QCheckBox(tr("Bold"))
        bold.setChecked(True)
        h = QDoubleSpinBox()
        h.setRange(0.5, 1000)
        h.setValue(10)
        h.setSuffix(" mm")
        form.addRow(tr("Text"), ed)
        form.addRow(tr("Font"), font)
        form.addRow("", bold)
        form.addRow(tr("Cap height"), h)
        form.addRow(QLabel(tr("After placing: Push/Pull. Alt held = separate part\n"
                              "(e.g. inlaid text in another color).")))
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
            msgs = [("info", tr("The model has no volumes yet."))]
        loose = len(self.doc.root.loose_edges())
        if loose:
            msgs.append(("info", tr("{n} loose edge(s) without a face are not printed.").format(
                n=loose)))
        # the levels are the identifiers of ops.check_parts (Dutch, not shown)
        icons = {"ok": "✔", "info": "ℹ", "waarschuwing": "⚠", "fout": "✖"}
        colors = {"ok": theme.color("ok"), "info": theme.color("info"),
                  "waarschuwing": theme.color("warn"), "fout": theme.color("error")}
        html = tr("<h3>Print Check</h3><p>Every part (group or loose shape) must be a closed "
                  "volume: no holes, no double or loose faces. Only closed volumes are "
                  "exported.</p>") + "<ul>"
        for level, text in msgs:
            html += f'<li style="color:{colors[level]}">{icons[level]} {text}</li>'
        html += "</ul>"
        dlg = QDialog(self)
        dlg.setWindowTitle(tr(PANEL_PRINT_CHECK))
        dlg.resize(620, 420)
        lay = QVBoxLayout(dlg)
        tb = QTextBrowser()
        tb.setHtml(html)
        lay.addWidget(tb)
        row = QHBoxLayout()
        b1 = QPushButton(tr("Drop to Bed"))
        b1.clicked.connect(lambda: (self.drop_to_bed(), dlg.accept()))
        b2 = QPushButton(tr("Close"))
        b2.clicked.connect(dlg.accept)
        row.addWidget(b1)
        row.addStretch(1)
        row.addWidget(b2)
        lay.addLayout(row)
        dlg.exec()

    def show_help(self):
        rows = "".join(f"<tr><td><b>{sc}</b></td><td>{tr(label)}</td></tr>"
                       for _, label, _, sc in TOOL_INFO if sc)
        html = tr(HELP_HTML).format(rows=rows)
        dlg = QDialog(self)
        dlg.setWindowTitle(tr("Volura Help"))
        dlg.resize(700, 640)
        lay = QVBoxLayout(dlg)
        tb = QTextBrowser()
        tb.setHtml(html)
        lay.addWidget(tb)
        dlg.exec()

    # ------------------------------------------------------------ updates
    def check_for_updates_on_startup(self):
        """Check quietly once a day; only in the standalone version, never in a test."""
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
        """Help > Check for Updates…: fetch the newest release from GitHub."""
        if not automatic:
            self.hint.setText(tr("Looking for a newer version…"))
        self._update_bridge = run_in_background(
            lambda progress: updater.latest_release(),
            lambda rel, err: self.on_update_result(rel, err, automatic))
        return self._update_bridge

    def on_update_result(self, release, error, automatic=False):
        if error is not None or release is None:
            if not automatic:
                self.hint.setText("")
                detail = f"\n\n{error}" if error is not None else ""
                QMessageBox.information(self, tr("Updates"),
                                        tr("Volura could not reach GitHub or found no release.")
                                        + detail)
            return None
        if not updater.is_newer(release.version):
            if not automatic:
                self.hint.setText("")
                QMessageBox.information(self, tr("Updates"),
                                        tr("You have the newest version ({version}).").format(
                                            version=__version__))
            return None
        if automatic and qsettings().value(UPDATE_SKIP_KEY, "") == release.version:
            return None
        self.hint.setText(tr("Volura {version} is available.").format(version=release.version))
        return self.offer_update(release)

    def offer_update(self, release):
        """Ask whether the new version should go on now; the installer does the rest."""
        box = QMessageBox(self)
        box.setWindowTitle(tr("New Version of Volura"))
        box.setIcon(QMessageBox.Information)
        box.setText(tr("<b>Volura {version}</b> is available. You now have {current}.").format(
            version=release.version, current=__version__))
        if release.notes.strip():
            box.setDetailedText(release.notes.strip())
        self_install = updater.can_self_install() and bool(release.setup_url)
        if self_install:
            box.setInformativeText(tr("Volura downloads the new version, closes, installs and "
                                      "then starts again. Your model is saved first if you "
                                      "want."))
            go = box.addButton(tr("Update Now"), QMessageBox.AcceptRole)
        else:
            box.setInformativeText(tr("Download the new version from the release page on GitHub."))
            go = box.addButton(tr("Open Download Page"), QMessageBox.AcceptRole)
        box.addButton(tr("Later"), QMessageBox.RejectRole)
        skip = box.addButton(tr("Skip This Version"), QMessageBox.DestructiveRole)
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
        """Save the model, download the installer and let it replace Volura."""
        if not self.confirm_discard():
            return None
        dlg = QProgressDialog(tr("Downloading Volura {version}…").format(version=release.version),
                              tr("Cancel"), 0, 100, self)
        dlg.setWindowTitle(tr("Updating"))
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
                    QMessageBox.warning(self, tr("Updating"),
                                        tr("The download failed:\n{error}").format(error=error))
                return
            try:
                updater.run_installer(path)
            except OSError as e:
                QMessageBox.warning(self, tr("Updating"),
                                    tr("The installer could not start:\n{error}").format(error=e))
                return
            self.doc.dirty = False             # already saved or deliberately discarded
            self.close()

        self._update_bridge = run_in_background(
            lambda report: updater.download(release, report, stop.is_set), finished, progress)
        return dlg

    def about(self):
        QMessageBox.about(self, tr("About Volura"),
                          f"<h3>Volura {__version__}</h3>"
                          + tr("<p>Open source 3D modeling program for 3D printing, with the "
                               "workflow of SketchUp 2017.</p><p>On export every part is checked "
                               "for a closed volume and written flawlessly as STL or multicolor "
                               "3MF.</p>"
                               "<p>© 2026 L. van der Poel · License: MIT</p>"
                               "<p>Volura uses third-party open source software:</p><ul>"
                               "<li>Qt 6 and PySide6 (The Qt Company), LGPL v3. The Qt libraries "
                               "are separate files in the program folder and may be replaced; "
                               "source code at <a href='https://code.qt.io'>code.qt.io</a>.</li>"
                               "<li>Manifold (manifold3d), Apache 2.0</li>"
                               "<li>NumPy, BSD 3-clause</li>"
                               "<li>PyOpenGL, BSD-style</li>"
                               "<li>Python, PSF license</li></ul>"
                               "<p>The full license texts are in THIRD_PARTY_LICENSES.txt next to "
                               "the program.</p>"))

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
        QMessageBox.information(self, tr("Area"),
                                tr("Total area of the selection: {area:.2f} mm²").format(area=total))

    def show_area_material(self):
        ref = self._clicked_face()
        if ref is None:
            return
        c, fid = ref
        ents = self.doc.ents(c)
        color = ents.F[fid].color
        total = sum(self.face_area(ents, f) for f in ents.F if ents.F[f].color == color)
        QMessageBox.information(self, tr("Area"),
                                tr("Total area of this material: {area:.2f} mm²").format(area=total))

    def align_view_to_face(self):
        """Align View: look straight at the face (view direction = -normal)."""
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
            m.addAction(tr(PANEL_INFO), self.show_entity_info)
            m.addAction(self.a_delete)
            m.addAction(self.a_hide)
            sel_groups = self.doc.selected_groups()
            if sel_groups:
                # SketchUp: right click > Lock / Unlock
                m.addAction(self.a_unlock if all(g.locked for g in sel_groups) else self.a_lock)
            if self._selected_faces():
                m.addSeparator()
                sm = m.addMenu(tr("Select"))
                sm.addAction(tr("Bounding Edges"), self.select_face_edges)
                sm.addAction(tr("Connected Faces"), self.select_connected_faces)
                sm.addAction(tr("All Connected"), self.select_all_connected)
                sm.addAction(tr("All with Same Material"), self.select_same_material)
                om = m.addMenu(tr("Area"))
                om.addAction(tr("Selection"), self.show_area_selection)
                om.addAction(tr("Material"), self.show_area_material)
                fm = m.addMenu(tr("Intersect Faces"))
                fm.addAction(self.a_cut_face_model)
                fm.addAction(self.a_cut_face_sel)
                m.addAction(tr("Align View"), self.align_view_to_face)
                m.addAction(tr("Reverse Faces"), self.reverse_selected_faces)
                m.addAction(tr("Orient Faces"), self.orient_selected_faces)
                m.addAction(tr("Zoom Selection"), self.zoom_selection)
            if self._selected_faces() or self.doc.selected_groups():
                m.addSeparator()
                self._add_laser_menu(m)
            m.addSeparator()
            if any(k in ("f", "g") for k, _, _ in sel):
                cm = m.addMenu(tr("Filament Color"))
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
            mm = m.addMenu(tr("Flip Along"))
            for i, naam in enumerate((tr("Red Axis"), tr("Green Axis"), tr("Blue Axis"))):
                mm.addAction(naam, lambda checked=False, ax=i: self.mirror(ax))
            m.addAction(self.a_drop)
            m.addAction(self.a_center)
        else:
            m.addAction(self.a_sel_all)
            m.addAction(tr("Unhide All"), self.unhide_all)
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
        # the face under the mouse, also in a group: "Laser / Mill" takes that very face
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
        self.spacemouse.stop()
        workers.shutdown()          # stop the worker processes on the way out
        view = getattr(self, "view", None)
        if view is not None and view.isVisible():
            # the vertex buffers can only be given back while the context lives
            try:
                view.makeCurrent()
                view.release_gl()
            except Exception:
                pass        # a view that cannot be made current has nothing left to give back
        self.settings.save()
        remove_autosave()       # closed properly: there is nothing to restore
        e.accept()


def excepthook(exc_type, exc, tb):
    """An error outside the event handlers: the same log as there, plus a message."""
    text = errors.log_exception(exc_type, exc, tb, note="unhandled error")
    if is_offscreen():
        return
    app = QApplication.instance()
    if app is not None:
        QMessageBox.critical(None, tr("Volura - Unexpected Error"), text[-3000:])


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
    theme.apply(str(QSettings("Volura", "Volura").value("theme", theme.SYSTEM)), app)
    sys.excepthook = excepthook
    win = MainWindow()
    win.show()
    if len(argv) > 1 and os.path.exists(argv[1]):
        win.file_open(argv[1])
    return app.exec()

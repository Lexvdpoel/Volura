"""The dialogs of the 2D worksheet: the exports, the sheet and the cut generators.

The vector export of the 3D model and the export for laser and mill, the size of the
sheet, the auto arrange, and the dialogs of the Laser and Mill menu: the box with finger
joints, the living hinge, the vent pattern, the fit test, holding tabs, round and dogbone
corners and the job estimate. `volura/app.py` builds the actions that open them.
"""
from __future__ import annotations

from PySide6.QtCore import QPointF, QSize, Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox, QDialog, QDialogButtonBox,
                               QDoubleSpinBox, QFormLayout, QGroupBox, QHBoxLayout, QHeaderView,
                               QLabel, QLineEdit, QListWidget, QListWidgetItem, QPushButton,
                               QRadioButton, QSizePolicy, QSpinBox, QTableWidget, QTableWidgetItem,
                               QVBoxLayout, QWidget)

from . import boxgen, nesting, theme, vector_export
from .i18n import tr
from .panels import note, swatch
from .settings import qsettings
from .tools import format_length, parse_length
from .vector2d import OP_LABELS, layer_param


def fmt_time(seconds) -> str:
    """A duration as m:ss, and as h:mm:ss from an hour onwards; a dash when there is no estimate."""
    if seconds is None:
        return "—"
    h, rest = divmod(int(round(float(seconds))), 3600)
    m, s = divmod(rest, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


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


#: what the worksheet can be written as, and the file filter of each of those kinds
SHEET_EXPORT_KINDS = ("svg", "dxf", "gcode")
EXPORT_FILTERS = {"svg": "SVG (*.svg)", "dxf": "DXF (*.dxf)",
                  "gcode": "G-code (*.gcode *.nc *.gc)"}
#: the S word maximum and M4 of the G-code export: keys of the machine, not of the drawing
GCODE_S_MAX_KEY = "gcode_s_max"
GCODE_DYNAMIC_KEY = "gcode_dynamic"
GCODE_S_MAX_DEFAULT = 1000         # $30 of a modern 32-bit board, 255 on an old 8-bit one


def sheet_export_kind(kind) -> str:
    """The kind of a worksheet export: "svg", "dxf" or "gcode"; anything else is SVG."""
    kind = str(kind).lower()
    return kind if kind in SHEET_EXPORT_KINDS else "svg"


def gcode_export_defaults() -> dict:
    """The S maximum and M4 choice of the last G-code export, as the dialog starts them."""
    q = qsettings()
    try:
        s_max = int(q.value(GCODE_S_MAX_KEY, GCODE_S_MAX_DEFAULT))
    except (TypeError, ValueError):
        s_max = GCODE_S_MAX_DEFAULT
    return {"s_max": min(100000, max(1, s_max)),
            "dynamic": q.value(GCODE_DYNAMIC_KEY, True) in (True, "true", "1", 1)}


class SheetExportDialog(QDialog):
    """The "Export for Laser/Mill" dialog: which layers may go and which compensation.

    By default the visible layers that mean something to the machine go along (except
    "None", which is for information only). The kerf/cutter compensation is on as soon
    as a mill layer goes along — a cutter has its own diameter — and off for laser,
    because laser software often does that compensation itself. With "a separate file
    per layer" every layer gets its own file, as some cutting software likes. The
    preview shows what will be written.

    For G-code the dialog also asks what the machine understands of the power: the
    highest S word (`$30`) and whether the beam follows the feed (M4) or burns at a fixed
    power (M3). Both come back next time, since they belong to the machine and not to the
    drawing.
    """

    def __init__(self, win, kind="svg"):
        super().__init__(win)
        self.win = win
        self.kind = sheet_export_kind(kind)
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
        self.s_max = self.dynamic = None
        if self.kind == "gcode":
            power = QHBoxLayout()
            power.addWidget(QLabel(tr("Max power (S)")))
            self.s_max = QSpinBox()
            self.s_max.setRange(1, 100000)
            self.s_max.setToolTip(tr("The highest S word the controller understands ($30)"))
            self.dynamic = QCheckBox(tr("Dynamic power (M4)"))
            self.dynamic.setToolTip(tr("The beam gives way where the head slows down in a "
                                       "corner, so it burns less there"))
            defaults = gcode_export_defaults()
            self.s_max.setValue(int(defaults["s_max"]))
            self.dynamic.setChecked(bool(defaults["dynamic"]))
            power.addWidget(self.s_max)
            power.addStretch(1)
            lay.addLayout(power)
            lay.addWidget(self.dynamic)
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
        out = {"layers": names, "compensate": self.compensate.isChecked(),
               "per_layer": self.per_layer.isChecked(), "per_sheet": self.per_sheet.isChecked()}
        if self.kind == "gcode":
            out["s_max"] = int(self.s_max.value())
            out["dynamic"] = self.dynamic.isChecked()
        return out

    def accept(self):
        if self.kind == "gcode":
            q = qsettings()
            q.setValue(GCODE_S_MAX_KEY, int(self.s_max.value()))
            q.setValue(GCODE_DYNAMIC_KEY, bool(self.dynamic.isChecked()))
            q.sync()
        super().accept()

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


class LivingHingeDialog(QDialog):
    """The "Living Hinge…" dialog: the slits of a bendable cut in plywood.

    A living hinge is a field of straight slits in a panel: the laser cuts them full
    through and the panel bends there. The values are the same for the same plywood, so
    the dialog starts where the user left them (Settings) and remembers the new ones.
    The direction says around which axis the panel has to bend: with slits along Y the
    hinge is a vertical line in the drawing and the panel bends around a vertical axis.
    """

    def __init__(self, win):
        super().__init__(win)
        s = win.settings
        self.setWindowTitle(tr("Living Hinge"))
        lay = QFormLayout(self)

        def spin(lo, hi, value, step=0.5):
            sp = QDoubleSpinBox()
            sp.setRange(lo, hi)
            sp.setDecimals(2)
            sp.setSingleStep(step)
            sp.setSuffix(" mm")
            sp.setValue(float(value))
            return sp

        self.slit = spin(5.0, 100.0, s.hinge_slit, 1.0)
        self.slit.setToolTip(tr("Length of one slit; the longer the slits, the easier the "
                                "panel bends"))
        self.gap = spin(1.0, 20.0, s.hinge_gap, 0.5)
        self.gap.setToolTip(tr("Width of the material between two slits; too thin and it "
                               "tears out"))
        self.pitch = spin(0.5, 10.0, s.hinge_pitch, 0.5)
        self.pitch.setToolTip(tr("Distance from one column of slits to the next"))
        self.margin = spin(0.0, 20.0, s.hinge_margin, 0.5)
        self.margin.setToolTip(tr("Material that stays free of slits along the edge, so the "
                                  "pattern cannot cut the panel loose"))
        self.vertical = QComboBox()
        self.vertical.addItem(tr("Bend around a vertical axis"), True)
        self.vertical.addItem(tr("Bend around a horizontal axis"), False)
        self.vertical.setCurrentIndex(0 if s.hinge_vertical else 1)
        lay.addRow(tr("Slit length"), self.slit)
        lay.addRow(tr("Bridge"), self.gap)
        lay.addRow(tr("Column spacing"), self.pitch)
        lay.addRow(tr("Edge margin"), self.margin)
        lay.addRow(tr("Direction"), self.vertical)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addRow(bb)

    def options(self) -> dict:
        return {"slit": round(self.slit.value(), 6), "gap": round(self.gap.value(), 6),
                "pitch": round(self.pitch.value(), 6), "margin": round(self.margin.value(), 6),
                "vertical": bool(self.vertical.currentData())}


class VentPatternDialog(QDialog):
    """The "Vent Pattern…" dialog: the holes of a grille or a speaker cover.

    The holes stand on a triangular grid, `web` millimetres of material between two
    neighbours, and only holes that lie whole inside the contour survive the edge margin.
    Hexagons pack tightest, circles are the classic speaker cover; the values depend on the
    project and the machine, so the dialog starts where the user left them (Settings) and
    remembers the new ones.
    """

    def __init__(self, win):
        super().__init__(win)
        s = win.settings
        self.setWindowTitle(tr("Vent Pattern"))
        lay = QFormLayout(self)
        intro = QLabel(tr("Fills the selected closed contour with a grid of holes, for a "
                          "ventilation grille or a speaker cover. The holes go on the cut "
                          "layer."))
        intro.setWordWrap(True)
        lay.addRow(intro)

        def spin(lo, hi, value, step=0.5):
            sp = QDoubleSpinBox()
            sp.setRange(lo, hi)
            sp.setDecimals(2)
            sp.setSingleStep(step)
            sp.setSuffix(" mm")
            sp.setValue(float(value))
            return sp

        self.shape = QComboBox()
        self.shape.addItem(tr("Hexagons"), "hex")
        self.shape.addItem(tr("Circles"), "round")
        self.shape.setCurrentIndex(0 if s.vent_shape == "hex" else 1)
        self.shape.setToolTip(tr("Hexagons pack tightest in the material, circles are the "
                                 "classic speaker cover"))
        self.size = spin(1.0, 50.0, s.vent_size, 1.0)
        self.size.setToolTip(tr("Width of one hole: across the flats for a hexagon, the "
                                "diameter of a circle"))
        self.web = spin(0.5, 20.0, s.vent_web, 0.5)
        self.web.setToolTip(tr("Width of the material between two holes; too thin and it "
                               "warps or tears out"))
        self.margin = spin(0.0, 50.0, s.vent_margin, 0.5)
        self.margin.setToolTip(tr("Material that stays free of holes along the edge, so the "
                                  "grille keeps its place in the panel"))
        lay.addRow(tr("Shape"), self.shape)
        lay.addRow(tr("Hole size"), self.size)
        lay.addRow(tr("Web"), self.web)
        lay.addRow(tr("Edge margin"), self.margin)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addRow(bb)

    def options(self) -> dict:
        return {"shape": str(self.shape.currentData()), "size": round(self.size.value(), 6),
                "web": round(self.web.value(), 6), "margin": round(self.margin.value(), 6)}


class FitTestDialog(QDialog):
    """The "Fit Test…" dialog: the slots of a press-fit test for the sheet in use.

    A fit test is a comb of slots, one per offset on the thickness of the sheet: the user
    cuts it from the same plywood, pushes an offcut in every slot and keeps the offset of
    the slot that holds best. That offset comes back in the kerf and the finger width of a
    box, so the dialog starts at the last values (Settings) and remembers the new ones.
    The thickness is not remembered: it is a property of the sheet on the table, and the
    worksheet already knows that.
    """

    def __init__(self, win):
        super().__init__(win)
        s = win.settings
        self.setWindowTitle(tr("Fit Test"))
        lay = QFormLayout(self)

        def spin(lo, hi, value, step=0.5):
            sp = QDoubleSpinBox()
            sp.setRange(lo, hi)
            sp.setDecimals(2)
            sp.setSingleStep(step)
            sp.setSuffix(" mm")
            sp.setValue(float(value))
            return sp

        self.t = spin(0.5, 20.0, win.doc.sheet.sheet_thickness or boxgen.DEFAULT_THICKNESS, 0.5)
        self.t.setToolTip(tr("Thickness of the sheet you are cutting"))
        self.first = spin(-1.0, 0.0, s.fit_test_first, 0.05)
        self.first.setToolTip(tr("Offset of the first slot: a negative offset makes a slot "
                                 "tighter than the sheet"))
        self.step = spin(0.02, 0.5, s.fit_test_step, 0.05)
        self.step.setToolTip(tr("How much wider every next slot is than the one before it"))
        self.count = QSpinBox()
        self.count.setRange(2, 10)
        self.count.setValue(s.fit_test_count)
        self.count.setToolTip(tr("Number of slots, narrow on the left and wide on the right"))
        lay.addRow(tr("Sheet thickness"), self.t)
        lay.addRow(tr("First offset"), self.first)
        lay.addRow(tr("Step"), self.step)
        lay.addRow(tr("Number of slots"), self.count)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addRow(bb)

    def options(self) -> dict:
        return {"t": round(self.t.value(), 6), "first": round(self.first.value(), 6),
                "step": round(self.step.value(), 6), "count": int(self.count.value())}


class HoldingTabsDialog(QDialog):
    """The "Holding Tabs…" dialog: the small bridges that keep a part in the sheet.

    A laser or mill that cuts a contour clean through lets the part fall out of the sheet
    or come loose in the clamp before the job is done: it tips into the beam or rattles
    under the cutter. Holding tabs are a few gaps left uncut in the contour, evenly
    spread, that you snap off afterwards. How many and how wide depends on the machine and
    the material, so the dialog starts where the user left them (Settings) and remembers
    the new ones; one millimetre is about the width of a laser beam, a few for a mill.

    The scrap of a hole has nowhere to hang, so it falls out on its own: by default only
    the outer contours get tabs and the holes are left alone ("Outer contours only").
    """

    def __init__(self, win):
        super().__init__(win)
        s = win.settings
        self.setWindowTitle(tr("Holding Tabs"))
        lay = QFormLayout(self)
        self.count = QSpinBox()
        self.count.setRange(1, 20)
        self.count.setValue(s.tab_count)
        self.count.setToolTip(tr("Number of tabs spread evenly around each contour"))
        self.width = QDoubleSpinBox()
        self.width.setRange(0.2, 20.0)
        self.width.setDecimals(2)
        self.width.setSingleStep(0.1)
        self.width.setSuffix(" mm")
        self.width.setValue(float(s.tab_width))
        self.width.setToolTip(tr("Width of one tab: about the laser beam for a laser, a few "
                                 "millimetres for a mill"))
        self.outer_only = QCheckBox(tr("Outer contours only"))
        self.outer_only.setToolTip(tr("Leave the holes alone: their scrap falls out of the sheet"))
        self.outer_only.setChecked(bool(s.tab_outer_only))
        lay.addRow(tr("Tabs per contour"), self.count)
        lay.addRow(tr("Tab width"), self.width)
        lay.addRow("", self.outer_only)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addRow(bb)

    def options(self) -> dict:
        return {"count": int(self.count.value()), "width": round(self.width.value(), 6),
                "outer_only": bool(self.outer_only.isChecked())}


class RoundCornersDialog(QDialog):
    """The "Round Corners…" dialog: the radius that takes the sharp corners out of a panel.

    A panel that comes off the laser or the mill with its corners still sharp cuts into
    your hand and chips whatever it rubs against. One radius is not one radius everywhere:
    a corner of which the two segments are shorter than the radius needs is rounded as far
    as they let it, and a corner without room stays sharp. Where the sheet is milled the
    radius is that of the cutter, so the dialog starts where the user left it (Settings)
    and remembers the new radius.
    """

    def __init__(self, win):
        super().__init__(win)
        s = win.settings
        self.setWindowTitle(tr("Round Corners"))
        lay = QFormLayout(self)
        self.radius = QDoubleSpinBox()
        self.radius.setRange(0.1, 100.0)
        self.radius.setDecimals(2)
        self.radius.setSingleStep(0.1)
        self.radius.setSuffix(" mm")
        self.radius.setValue(float(s.corner_radius))
        self.radius.setToolTip(tr("Radius of the arc in every corner, as far as the "
                                  "segments next to it allow"))
        lay.addRow(tr("Radius"), self.radius)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addRow(bb)

    def options(self) -> dict:
        return {"radius": round(self.radius.value(), 6)}


class DogboneCornersDialog(QDialog):
    """The "Dogbone Corners…" dialog: the diameter of the mill whose corners must be cleared.

    A round cutter cannot mill a sharp inside corner: it leaves a fillet of its own radius
    standing there, so a square tab does not fit into a milled square slot. The relief that
    clears the corner is a circle of the cutter itself, so the only number here is the
    diameter of the cutter that will run the job. Where the sheet is milled that is the
    diameter in the tool table, so the dialog starts where the user left it (Settings) and
    remembers the new one.
    """

    def __init__(self, win):
        super().__init__(win)
        s = win.settings
        self.setWindowTitle(tr("Dogbone Corners"))
        lay = QFormLayout(self)
        self.tool = QDoubleSpinBox()
        self.tool.setRange(0.5, 25.0)
        self.tool.setDecimals(3)
        self.tool.setSingleStep(0.5)
        self.tool.setSuffix(" mm")
        self.tool.setValue(float(s.dogbone_tool_d))
        self.tool.setToolTip(tr("Diameter of the round cutter: it leaves a fillet of half "
                                "this diameter in every inside corner"))
        lay.addRow(tr("Cutter diameter"), self.tool)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addRow(bb)

    def options(self) -> dict:
        return {"tool_d": round(self.tool.value(), 6)}


class JobEstimateDialog(QDialog):
    """The "Job Estimate…" dialog: what the worksheet takes on the laser.

    Read-only. One row per layer that the beam follows, with the length of its paths, the
    number of pierces, the passes and the speed, and below them the rapid moves between the
    paths and the total. Times stand as m:ss, and as h:mm:ss from an hour onwards. A layer
    without a usable speed gets a dash instead of a time, and a raster or mill layer is only
    named under the table, since its time does not follow from the layer.
    """

    WIDTHS = (120, 70, 60, 55, 80, 60)

    def __init__(self, estimate, parent=None):
        super().__init__(parent)
        self.estimate = estimate
        self.setWindowTitle(tr("Job Estimate"))
        lay = QVBoxLayout(self)
        rows = list(estimate["layers"])
        self.table = QTableWidget(len(rows) + 2, 6)
        self.table.setHorizontalHeaderLabels(
            [tr("Layer"), tr("Length (m)"), tr("Pierces"), tr("Passes"),
             tr("Speed (mm/s)"), tr("Time")])
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setSelectionMode(QAbstractItemView.NoSelection)
        self.table.setFocusPolicy(Qt.NoFocus)
        for col, width in enumerate(self.WIDTHS):
            self.table.setColumnWidth(col, width)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        for r, row in enumerate(rows):
            self._cell(r, 0, row["layer"])
            self._cell(r, 1, f'{row["length"] / 1000.0:.3f}')
            self._cell(r, 2, row["pierces"])
            self._cell(r, 3, f'{row["passes"]}×')
            self._cell(r, 4, f'{row["speed"]:g}')
            self._cell(r, 5, fmt_time(row["time"]))
        self._cell(len(rows), 0, tr("Travel"))
        self._cell(len(rows), 1, f'{estimate["travel"] / 1000.0:.3f}')
        self._cell(len(rows), 5, fmt_time(estimate["travel_time"]))
        self._cell(len(rows) + 1, 0, tr("Total"))
        self._cell(len(rows) + 1, 5, fmt_time(estimate["total"]))
        bold = self.table.item(len(rows) + 1, 0)
        font = QFont()
        font.setBold(True)
        bold.setFont(font)
        lay.addWidget(self.table)
        if estimate["skipped"]:
            lay.addWidget(note(tr("Not included: {names}.").format(
                names=", ".join(estimate["skipped"]))))
        row = QHBoxLayout()
        row.addStretch(1)
        b = QPushButton(tr("Close"))
        b.clicked.connect(self.accept)
        row.addWidget(b)
        lay.addLayout(row)

    def _cell(self, r, c, text):
        """One read-only cell of the table; the numbers line up on the right."""
        it = QTableWidgetItem(str(text))
        if c:
            it.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.table.setItem(r, c, it)

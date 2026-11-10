"""Main window of Volura."""
from __future__ import annotations

import math
import os
import sys
import threading
import time
from dataclasses import replace

import numpy as np
import manifold3d as mf
from PySide6.QtCore import (QObject, QRectF, QSize, Qt, QTimer, QUrl,
                            QVariantAnimation, Signal)
from PySide6.QtGui import (QAction, QActionGroup, QColor, QKeySequence, QPainter,
                           QPen, QDesktopServices)
from PySide6.QtWidgets import (QApplication, QCheckBox, QColorDialog, QComboBox,
                               QDialog, QDialogButtonBox, QDoubleSpinBox, QFileDialog,
                               QFontComboBox, QFormLayout, QHBoxLayout,
                               QInputDialog,
                                QLabel, QLineEdit,
                                QMainWindow, QMenu, QMessageBox, QPlainTextEdit,
                                QProgressDialog, QPushButton,
                               QSpinBox, QStackedWidget,
                               QTextBrowser,
                               QToolBar, QToolButton, QVBoxLayout)

from . import (__version__, background, battery_holder, boxgen, cable_clip, cam, dxf, enclosure,
                   errors, fileio, gcode, gears, hinge, i18n, joints, knob, label_plate, linking,
                   nesting, ops, sheet3d, sheettools, svg, spacemouse, standoff, storage_bin,
                   text3d, theme, threads, vector_export, vector_import, workers)
from .brep import Entities
from .inference import GeoCache
from .mesh_formats import MESH_EXTENSIONS, is_mesh
from . import repair
from .geom import Plane
from .i18n import tr, tr_noop, trn
from .icons import icon
from .model import ROOT, LOCKED_MSG, Document
from .app_dialogs import (FILAMENT_MATERIALS, ModelInfoDialog, PageDialog,
                          PreferencesDialog, PRINTERS, ShortcutsPanel, SimplifyDialog)
from .panels import (EntityInfoPanel, LayerParamsDialog, LayersPanel, MaterialsPanel,
                     OutlinerPanel, PrintCheckPanel, note, swatch)
from .part_dialogs import (CableClipDialog, HingeDialog, KnobDialog, LabelPlateDialog,
                           ProjectBoxDialog, SpurGearDialog, StandoffDialog, Text3DDialog)
from .settings import qsettings
from .sheet_dialogs import (AutoArrangeDialog, BoxJointDialog, DogboneCornersDialog, EXPORT_FILTERS,
                            FitTestDialog, GCODE_DYNAMIC_KEY, GCODE_S_MAX_DEFAULT, GCODE_S_MAX_KEY,
                            HoldingTabsDialog, JobEstimateDialog, LivingHingeDialog,
                            RoundCornersDialog, SHEET_EXPORT_KINDS, SheetExportDialog,
                            SheetSettingsDialog, VectorExportDialog, VectorPreview,
                            VentPatternDialog, fmt_time, gcode_export_defaults, sheet_export_kind)
from .sheettools import joined_paths
from .sheetview import SheetView
from .tools import TOOL_CLASSES, fmt_deg, format_length, set_length_precision
from .tray import TRAY_WIDTH, Tray
from .vector2d import (SHEET_SPACING, Drawing, MILL_OPS, OP_LABELS, VPath, _transform_meta,
                       default_layer_names, job_estimate, polygon_area)
from .viewport import (FACE_STYLES, HIDDENLINE, MONOCHROME, SHADED, WIREFRAME, Viewport)
from .window_print import PANEL_PRINT_CHECK, PrintMixin, TOOL_INFO
from .window_sheet import SheetMixin
from .window_ui import (MOVED_COMMANDS, OTHER_COMMANDS, PANEL_INFO,
                        SHORTCUTS_KEY, SOURCE_TEXT, UiMixin, clean_text,
                        source_text_of, translation_tables)

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
PANEL_MATERIALS = tr_noop("Materials")
PANEL_OUTLINER = tr_noop("Outliner")
PANEL_LAYERS = tr_noop("Layers")
PANELS = (PANEL_INFO, PANEL_MATERIALS, PANEL_OUTLINER, PANEL_PRINT_CHECK, PANEL_LAYERS)


class Settings:
    def __init__(self):
        self.q = qsettings()
        self.bed_size = tuple(float(v) for v in self.q.value("bed_size", [256, 256, 256]))
        self.show_grid = self._bool("show_grid", True)
        self.show_axes = self._bool("show_axes", True)
        self.show_edges = self._bool("show_edges", True)
        self.xray = self._bool("xray", False)
        self.show_overhangs = self._bool("show_overhangs", False)
        try:
            self.overhang_limit = int(self.q.value("overhang_limit", 45))
        except (TypeError, ValueError):
            self.overhang_limit = 45
        self.overhang_limit = max(0, min(89, self.overhang_limit))
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
        self.filament_material = str(self.q.value("filament_material", "PLA"))
        try:
            self.filament_density = float(self.q.value("filament_density", 1.24))
        except (TypeError, ValueError):
            self.filament_density = 1.24
        self.filament_density = max(0.1, min(20.0, self.filament_density))
        try:
            self.filament_diameter = float(self.q.value("filament_diameter", 1.75))
        except (TypeError, ValueError):
            self.filament_diameter = 1.75
        self.filament_diameter = max(0.5, min(5.0, self.filament_diameter))
        try:
            self.filament_price = float(self.q.value("filament_price", 20.0))
        except (TypeError, ValueError):
            self.filament_price = 20.0
        self.filament_price = max(0.0, min(100000.0, self.filament_price))
        try:
            self.print_shell = float(self.q.value("print_shell", 1.2))
        except (TypeError, ValueError):
            self.print_shell = 1.2
        self.print_shell = max(0.0, min(5.0, self.print_shell))
        try:
            self.print_infill = int(self.q.value("print_infill", 15))
        except (TypeError, ValueError):
            self.print_infill = 15
        self.print_infill = max(0, min(100, self.print_infill))
        try:
            self.elephant_foot_size = float(self.q.value("elephant_foot_size",
                                                         ops.ELEPHANT_FOOT_SIZE))
        except (TypeError, ValueError):
            self.elephant_foot_size = ops.ELEPHANT_FOOT_SIZE
        self.elephant_foot_size = max(0.1, min(2.0, self.elephant_foot_size))
        try:
            self.mouse_ear_diameter = float(self.q.value("mouse_ear_diameter",
                                                         ops.MOUSE_EAR_DIAMETER))
        except (TypeError, ValueError):
            self.mouse_ear_diameter = ops.MOUSE_EAR_DIAMETER
        self.mouse_ear_diameter = max(5.0, min(30.0, self.mouse_ear_diameter))
        try:
            self.mouse_ear_height = float(self.q.value("mouse_ear_height", ops.MOUSE_EAR_HEIGHT))
        except (TypeError, ValueError):
            self.mouse_ear_height = ops.MOUSE_EAR_HEIGHT
        self.mouse_ear_height = max(0.1, min(2.0, self.mouse_ear_height))
        self.screw_hole_size = str(self.q.value("screw_hole_size", "M3"))
        if self.screw_hole_size not in threads.SCREW_HOLES:
            self.screw_hole_size = "M3"
        self.screw_hole_kind = str(self.q.value("screw_hole_kind", "through"))
        if self.screw_hole_kind not in threads.SCREW_HOLE_KINDS:
            self.screw_hole_kind = "through"
        try:
            self.screw_hole_clearance = float(self.q.value("screw_hole_clearance",
                                                           threads.DEFAULT_HOLE_CLEARANCE))
        except (TypeError, ValueError):
            self.screw_hole_clearance = threads.DEFAULT_HOLE_CLEARANCE
        self.screw_hole_clearance = max(0.0, min(1.0, self.screw_hole_clearance))
        self.screw_hole_teardrop = self._bool("screw_hole_teardrop", False)
        try:
            self.clearance_test_pin = float(self.q.value("clearance_test_pin", 5.0))
        except (TypeError, ValueError):
            self.clearance_test_pin = 5.0
        self.clearance_test_pin = max(2.0, min(20.0, self.clearance_test_pin))
        try:
            self.clearance_test_first = float(self.q.value("clearance_test_first", 0.0))
        except (TypeError, ValueError):
            self.clearance_test_first = 0.0
        self.clearance_test_first = max(0.0, min(1.0, self.clearance_test_first))
        try:
            self.clearance_test_step = float(self.q.value("clearance_test_step", 0.1))
        except (TypeError, ValueError):
            self.clearance_test_step = 0.1
        self.clearance_test_step = max(0.05, min(0.5, self.clearance_test_step))
        try:
            self.clearance_test_count = int(float(self.q.value("clearance_test_count", 6)))
        except (TypeError, ValueError):
            self.clearance_test_count = 6
        self.clearance_test_count = max(2, min(10, self.clearance_test_count))
        try:
            self.overhang_test_first = float(self.q.value("overhang_test_first", 30.0))
        except (TypeError, ValueError):
            self.overhang_test_first = 30.0
        self.overhang_test_first = max(0.0, min(85.0, self.overhang_test_first))
        try:
            self.overhang_test_step = float(self.q.value("overhang_test_step", 10.0))
        except (TypeError, ValueError):
            self.overhang_test_step = 10.0
        self.overhang_test_step = max(1.0, min(45.0, self.overhang_test_step))
        try:
            self.overhang_test_count = int(float(self.q.value("overhang_test_count", 6)))
        except (TypeError, ValueError):
            self.overhang_test_count = 6
        self.overhang_test_count = max(1, min(10, self.overhang_test_count))
        try:
            self.hinge_slit = float(self.q.value("hinge_slit", 15.0))
        except (TypeError, ValueError):
            self.hinge_slit = 15.0
        self.hinge_slit = max(5.0, min(100.0, self.hinge_slit))
        try:
            self.hinge_gap = float(self.q.value("hinge_gap", 3.0))
        except (TypeError, ValueError):
            self.hinge_gap = 3.0
        self.hinge_gap = max(1.0, min(20.0, self.hinge_gap))
        try:
            self.hinge_pitch = float(self.q.value("hinge_pitch", 2.0))
        except (TypeError, ValueError):
            self.hinge_pitch = 2.0
        self.hinge_pitch = max(0.5, min(10.0, self.hinge_pitch))
        try:
            self.hinge_margin = float(self.q.value("hinge_margin", 3.0))
        except (TypeError, ValueError):
            self.hinge_margin = 3.0
        self.hinge_margin = max(0.0, min(20.0, self.hinge_margin))
        self.hinge_vertical = self._bool("hinge_vertical", True)
        self.vent_shape = str(self.q.value("vent_shape", "hex") or "hex").strip().lower()
        if self.vent_shape not in boxgen.VENT_SHAPES:
            self.vent_shape = "hex"
        try:
            self.vent_size = float(self.q.value("vent_size", 6.0))
        except (TypeError, ValueError):
            self.vent_size = 6.0
        self.vent_size = max(1.0, min(50.0, self.vent_size))
        try:
            self.vent_web = float(self.q.value("vent_web", 2.0))
        except (TypeError, ValueError):
            self.vent_web = 2.0
        self.vent_web = max(0.5, min(20.0, self.vent_web))
        try:
            self.vent_margin = float(self.q.value("vent_margin", 3.0))
        except (TypeError, ValueError):
            self.vent_margin = 3.0
        self.vent_margin = max(0.0, min(50.0, self.vent_margin))
        try:
            self.fit_test_first = float(self.q.value("fit_test_first", -0.2))
        except (TypeError, ValueError):
            self.fit_test_first = -0.2
        self.fit_test_first = max(-1.0, min(0.0, self.fit_test_first))
        try:
            self.fit_test_step = float(self.q.value("fit_test_step", 0.1))
        except (TypeError, ValueError):
            self.fit_test_step = 0.1
        self.fit_test_step = max(0.02, min(0.5, self.fit_test_step))
        try:
            self.fit_test_count = int(float(self.q.value("fit_test_count", 5)))
        except (TypeError, ValueError):
            self.fit_test_count = 5
        self.fit_test_count = max(2, min(10, self.fit_test_count))
        try:
            self.gear_teeth = int(float(self.q.value("gear_teeth", 20)))
        except (TypeError, ValueError):
            self.gear_teeth = 20
        self.gear_teeth = max(gears.TEETH_MIN, min(gears.TEETH_MAX, self.gear_teeth))
        try:
            self.gear_module = float(self.q.value("gear_module", 2.0))
        except (TypeError, ValueError):
            self.gear_module = 2.0
        self.gear_module = max(gears.MODULE_MIN, min(gears.MODULE_MAX, self.gear_module))
        try:
            self.gear_angle = float(self.q.value("gear_angle", 20.0))
        except (TypeError, ValueError):
            self.gear_angle = 20.0
        self.gear_angle = max(gears.ANGLE_MIN, min(gears.ANGLE_MAX, self.gear_angle))
        try:
            self.gear_bore = float(self.q.value("gear_bore", 5.0))
        except (TypeError, ValueError):
            self.gear_bore = 5.0
        self.gear_bore = max(0.0, self.gear_bore)
        try:
            self.gear_backlash = float(self.q.value("gear_backlash", 0.1))
        except (TypeError, ValueError):
            self.gear_backlash = 0.1
        self.gear_backlash = max(0.0, min(1.0, self.gear_backlash))
        try:
            self.gear_thickness = float(self.q.value("gear_thickness", 8.0))
        except (TypeError, ValueError):
            self.gear_thickness = 8.0
        self.gear_thickness = max(1.0, min(100.0, self.gear_thickness))
        try:
            self.box3d_length = float(self.q.value("box3d_length", 80.0))
        except (TypeError, ValueError):
            self.box3d_length = 80.0
        self.box3d_length = max(1.0, min(500.0, self.box3d_length))
        try:
            self.box3d_width = float(self.q.value("box3d_width", 50.0))
        except (TypeError, ValueError):
            self.box3d_width = 50.0
        self.box3d_width = max(1.0, min(500.0, self.box3d_width))
        try:
            self.box3d_height = float(self.q.value("box3d_height", 30.0))
        except (TypeError, ValueError):
            self.box3d_height = 30.0
        self.box3d_height = max(1.0, min(500.0, self.box3d_height))
        try:
            self.box3d_wall = float(self.q.value("box3d_wall", 2.0))
        except (TypeError, ValueError):
            self.box3d_wall = 2.0
        self.box3d_wall = max(enclosure.WALL_MIN, min(20.0, self.box3d_wall))
        try:
            self.box3d_floor = float(self.q.value("box3d_floor", 2.0))
        except (TypeError, ValueError):
            self.box3d_floor = 2.0
        self.box3d_floor = max(enclosure.FLOOR_MIN, min(20.0, self.box3d_floor))
        try:
            self.box3d_radius = float(self.q.value("box3d_radius", 3.0))
        except (TypeError, ValueError):
            self.box3d_radius = 3.0
        self.box3d_radius = max(0.0, min(100.0, self.box3d_radius))
        try:
            self.box3d_lid = float(self.q.value("box3d_lid", 2.0))
        except (TypeError, ValueError):
            self.box3d_lid = 2.0
        self.box3d_lid = max(enclosure.LID_MIN, min(20.0, self.box3d_lid))
        try:
            self.box3d_lip = float(self.q.value("box3d_lip", 3.0))
        except (TypeError, ValueError):
            self.box3d_lip = 3.0
        self.box3d_lip = max(0.0, min(50.0, self.box3d_lip))
        try:
            # the clearance of the screw holes is the clearance this printer already proved
            self.box3d_clearance = float(self.q.value("box3d_clearance",
                                                      self.screw_hole_clearance))
        except (TypeError, ValueError):
            self.box3d_clearance = self.screw_hole_clearance
        self.box3d_clearance = max(0.0, min(enclosure.CLEARANCE_MAX, self.box3d_clearance))
        self.box3d_screws = str(self.q.value("box3d_screws", "") or "").strip()
        if self.box3d_screws not in enclosure.SCREW_SIZES:
            self.box3d_screws = ""
        try:
            self.box3d_vents = int(float(self.q.value("box3d_vents", 0)))
        except (TypeError, ValueError):
            self.box3d_vents = 0
        self.box3d_vents = max(0, min(enclosure.VENTS_MAX, self.box3d_vents))
        try:
            self.knob_diameter = float(self.q.value("knob_diameter", 30.0))
        except (TypeError, ValueError):
            self.knob_diameter = 30.0
        self.knob_diameter = max(knob.DIAMETER_MIN, min(knob.DIAMETER_MAX, self.knob_diameter))
        try:
            self.knob_height = float(self.q.value("knob_height", 12.0))
        except (TypeError, ValueError):
            self.knob_height = 12.0
        self.knob_height = max(knob.HEIGHT_MIN, min(knob.HEIGHT_MAX, self.knob_height))
        self.knob_grip = str(self.q.value("knob_grip", "knurl"))
        if self.knob_grip not in knob.KNOB_GRIPS:
            self.knob_grip = "knurl"
        try:
            self.knob_lobes = int(float(self.q.value("knob_lobes", 6)))
        except (TypeError, ValueError):
            self.knob_lobes = 6
        self.knob_lobes = max(knob.LOBES_MIN, min(knob.LOBES_MAX, self.knob_lobes))
        self.knob_kind = str(self.q.value("knob_kind", "nut"))
        if self.knob_kind not in knob.KNOB_KINDS:
            self.knob_kind = "nut"
        self.knob_bore = str(self.q.value("knob_bore", "M4"))
        if self.knob_bore not in threads.SCREW_HOLES:
            self.knob_bore = "M4"
        try:
            # the clearance of the screw holes is the clearance this printer already proved
            self.knob_clearance = float(self.q.value("knob_clearance",
                                                     self.screw_hole_clearance))
        except (TypeError, ValueError):
            self.knob_clearance = self.screw_hole_clearance
        self.knob_clearance = max(0.0, min(knob.CLEARANCE_MAX, self.knob_clearance))
        try:
            self.knob_shaft_d = float(self.q.value("knob_shaft_d", 6.0))
        except (TypeError, ValueError):
            self.knob_shaft_d = 6.0
        self.knob_shaft_d = max(0.5, min(50.0, self.knob_shaft_d))
        try:
            self.knob_flat = float(self.q.value("knob_flat", 0.5))
        except (TypeError, ValueError):
            self.knob_flat = 0.5
        self.knob_flat = max(0.0, min(10.0, self.knob_flat))
        try:
            self.standoff_height = float(self.q.value("standoff_height", 10.0))
        except (TypeError, ValueError):
            self.standoff_height = 10.0
        self.standoff_height = max(standoff.HEIGHT_MIN,
                                   min(standoff.HEIGHT_MAX, self.standoff_height))
        try:
            self.standoff_outer = float(self.q.value("standoff_outer", 6.0))
        except (TypeError, ValueError):
            self.standoff_outer = 6.0
        self.standoff_outer = max(standoff.OUTER_MIN, min(standoff.OUTER_MAX, self.standoff_outer))
        self.standoff_shape = str(self.q.value("standoff_shape", "round"))
        if self.standoff_shape not in standoff.STANDOFF_SHAPES:
            self.standoff_shape = "round"
        self.standoff_kind = str(self.q.value("standoff_kind", "through"))
        if self.standoff_kind not in standoff.STANDOFF_KINDS:
            self.standoff_kind = "through"
        self.standoff_bore = str(self.q.value("standoff_bore", "M3"))
        if self.standoff_bore not in threads.SCREW_HOLES:
            self.standoff_bore = "M3"
        try:
            # the clearance of the screw holes is the clearance this printer already proved
            self.standoff_clearance = float(self.q.value("standoff_clearance",
                                                         self.screw_hole_clearance))
        except (TypeError, ValueError):
            self.standoff_clearance = self.screw_hole_clearance
        self.standoff_clearance = max(0.0,
                                      min(standoff.CLEARANCE_MAX, self.standoff_clearance))
        try:
            self.standoff_count = int(float(self.q.value("standoff_count", 1)))
        except (TypeError, ValueError):
            self.standoff_count = 1
        self.standoff_count = max(standoff.COUNT_MIN, min(standoff.COUNT_MAX, self.standoff_count))
        self.label_text = str(self.q.value("label_text", "Volura")).strip() or "Volura"
        try:
            self.label_height = float(self.q.value("label_height", 8.0))
        except (TypeError, ValueError):
            self.label_height = 8.0
        self.label_height = max(label_plate.HEIGHT_MIN,
                                min(label_plate.HEIGHT_MAX, self.label_height))
        try:
            self.label_thickness = float(self.q.value("label_thickness", 2.0))
        except (TypeError, ValueError):
            self.label_thickness = 2.0
        self.label_thickness = max(label_plate.THICKNESS_MIN,
                                   min(label_plate.THICKNESS_MAX, self.label_thickness))
        try:
            self.label_relief = float(self.q.value("label_relief", 0.8))
        except (TypeError, ValueError):
            self.label_relief = 0.8
        self.label_relief = max(label_plate.RELIEF_MIN,
                                min(label_plate.RELIEF_MAX, self.label_relief))
        self.label_mode = str(self.q.value("label_mode", "raised"))
        if self.label_mode not in label_plate.LABEL_MODES:
            self.label_mode = "raised"
        try:
            self.label_margin = float(self.q.value("label_margin", 3.0))
        except (TypeError, ValueError):
            self.label_margin = 3.0
        self.label_margin = max(label_plate.MARGIN_MIN,
                                min(label_plate.MARGIN_MAX, self.label_margin))
        try:
            self.label_radius = float(self.q.value("label_radius", 2.0))
        except (TypeError, ValueError):
            self.label_radius = 2.0
        self.label_radius = max(0.0, min(20.0, self.label_radius))
        try:
            self.label_hole = float(self.q.value("label_hole", 0.0))
        except (TypeError, ValueError):
            self.label_hole = 0.0
        self.label_hole = max(0.0, min(label_plate.HOLE_MAX, self.label_hole))
        self.text3d_text = str(self.q.value("text3d_text", "Volura")).strip() or "Volura"
        self.text3d_family = str(self.q.value("text3d_family", "")).strip()
        try:
            self.text3d_height = float(self.q.value("text3d_height", 10.0))
        except (TypeError, ValueError):
            self.text3d_height = 10.0
        self.text3d_height = max(text3d.HEIGHT_MIN,
                                 min(text3d.HEIGHT_MAX, self.text3d_height))
        try:
            self.text3d_depth = float(self.q.value("text3d_depth", 3.0))
        except (TypeError, ValueError):
            self.text3d_depth = 3.0
        self.text3d_depth = max(text3d.DEPTH_MIN, min(text3d.DEPTH_MAX, self.text3d_depth))
        self.text3d_bold = self._bool("text3d_bold", False)
        self.text3d_align = str(self.q.value("text3d_align", "left"))
        if self.text3d_align not in text3d.ALIGNMENTS:
            self.text3d_align = "left"
        try:
            self.clip_cable_d = float(self.q.value("clip_cable_d", 6.0))
        except (TypeError, ValueError):
            self.clip_cable_d = 6.0
        self.clip_cable_d = max(cable_clip.CABLE_MIN,
                                min(cable_clip.CABLE_MAX, self.clip_cable_d))
        try:
            self.clip_width = float(self.q.value("clip_width", 8.0))
        except (TypeError, ValueError):
            self.clip_width = 8.0
        self.clip_width = max(cable_clip.WIDTH_MIN,
                              min(cable_clip.WIDTH_MAX, self.clip_width))
        try:
            self.clip_wall = float(self.q.value("clip_wall", 1.6))
        except (TypeError, ValueError):
            self.clip_wall = 1.6
        self.clip_wall = max(cable_clip.WALL_MIN, min(cable_clip.WALL_MAX, self.clip_wall))
        try:
            self.clip_opening = float(self.q.value("clip_opening", 0.8))
        except (TypeError, ValueError):
            self.clip_opening = 0.8
        self.clip_opening = max(cable_clip.OPENING_MIN,
                                min(cable_clip.OPENING_MAX, self.clip_opening))
        try:
            self.clip_base = float(self.q.value("clip_base", 3.0))
        except (TypeError, ValueError):
            self.clip_base = 3.0
        self.clip_base = max(cable_clip.BASE_MIN, min(cable_clip.BASE_MAX, self.clip_base))
        self.clip_bore = str(self.q.value("clip_bore", "M3"))
        if self.clip_bore not in threads.SCREW_HOLES:
            self.clip_bore = "M3"
        try:
            self.clip_count = int(float(self.q.value("clip_count", 1)))
        except (TypeError, ValueError):
            self.clip_count = 1
        self.clip_count = max(cable_clip.COUNT_MIN, min(cable_clip.COUNT_MAX, self.clip_count))
        try:
            self.hinge_length = float(self.q.value("hinge_length", 40.0))
        except (TypeError, ValueError):
            self.hinge_length = 40.0
        self.hinge_length = max(hinge.LENGTH_MIN, min(hinge.LENGTH_MAX, self.hinge_length))
        try:
            self.hinge_width = float(self.q.value("hinge_width", 15.0))
        except (TypeError, ValueError):
            self.hinge_width = 15.0
        self.hinge_width = max(hinge.WIDTH_MIN, min(hinge.WIDTH_MAX, self.hinge_width))
        try:
            self.hinge_thickness = float(self.q.value("hinge_thickness", 2.0))
        except (TypeError, ValueError):
            self.hinge_thickness = 2.0
        self.hinge_thickness = max(hinge.THICKNESS_MIN,
                                   min(hinge.THICKNESS_MAX, self.hinge_thickness))
        try:
            self.hinge_knuckles = int(float(self.q.value("hinge_knuckles", 5)))
        except (TypeError, ValueError):
            self.hinge_knuckles = 5
        self.hinge_knuckles = max(hinge.KNUCKLES_MIN,
                                  min(hinge.KNUCKLES_MAX, self.hinge_knuckles))
        try:
            self.hinge_clearance = float(self.q.value("hinge_clearance", 0.3))
        except (TypeError, ValueError):
            self.hinge_clearance = 0.3
        self.hinge_clearance = max(hinge.CLEARANCE_MIN,
                                   min(hinge.CLEARANCE_MAX, self.hinge_clearance))
        try:
            self.hinge_pin_d = float(self.q.value("hinge_pin_d", 3.0))
        except (TypeError, ValueError):
            self.hinge_pin_d = 3.0
        self.hinge_pin_d = max(hinge.PIN_MIN, min(hinge.PIN_MAX, self.hinge_pin_d))
        self.battery_cell = str(self.q.value("battery_cell", "AA"))
        if self.battery_cell not in battery_holder.CELLS:
            self.battery_cell = "AA"
        try:
            self.battery_count = int(float(self.q.value("battery_count", 2)))
        except (TypeError, ValueError):
            self.battery_count = 2
        self.battery_count = max(battery_holder.COUNT_MIN,
                                 min(battery_holder.COUNT_MAX, self.battery_count))
        try:
            self.battery_wall = float(self.q.value("battery_wall", 1.6))
        except (TypeError, ValueError):
            self.battery_wall = 1.6
        self.battery_wall = max(battery_holder.WALL_MIN,
                                min(battery_holder.WALL_MAX, self.battery_wall))
        try:
            self.battery_clearance = float(self.q.value("battery_clearance", 0.4))
        except (TypeError, ValueError):
            self.battery_clearance = 0.4
        self.battery_clearance = max(battery_holder.CLEARANCE_MIN,
                                     min(battery_holder.CLEARANCE_MAX, self.battery_clearance))
        try:
            self.battery_grip = float(self.q.value("battery_grip", 0.05))
        except (TypeError, ValueError):
            self.battery_grip = 0.05
        self.battery_grip = max(battery_holder.GRIP_MIN,
                                min(battery_holder.GRIP_MAX, self.battery_grip))
        try:
            self.bin_length = float(self.q.value("bin_length", 80.0))
        except (TypeError, ValueError):
            self.bin_length = 80.0
        self.bin_length = max(storage_bin.LENGTH_MIN,
                              min(storage_bin.LENGTH_MAX, self.bin_length))
        try:
            self.bin_width = float(self.q.value("bin_width", 60.0))
        except (TypeError, ValueError):
            self.bin_width = 60.0
        self.bin_width = max(storage_bin.WIDTH_MIN, min(storage_bin.WIDTH_MAX, self.bin_width))
        try:
            self.bin_height = float(self.q.value("bin_height", 40.0))
        except (TypeError, ValueError):
            self.bin_height = 40.0
        self.bin_height = max(storage_bin.HEIGHT_MIN,
                              min(storage_bin.HEIGHT_MAX, self.bin_height))
        try:
            self.bin_wall = float(self.q.value("bin_wall", 1.6))
        except (TypeError, ValueError):
            self.bin_wall = 1.6
        self.bin_wall = max(storage_bin.WALL_MIN, min(storage_bin.WALL_MAX, self.bin_wall))
        try:
            self.bin_floor = float(self.q.value("bin_floor", 1.2))
        except (TypeError, ValueError):
            self.bin_floor = 1.2
        self.bin_floor = max(storage_bin.FLOOR_MIN, min(storage_bin.FLOOR_MAX, self.bin_floor))
        try:
            self.bin_radius = float(self.q.value("bin_radius", 4.0))
        except (TypeError, ValueError):
            self.bin_radius = 4.0
        self.bin_radius = max(storage_bin.RADIUS_MIN,
                              min(storage_bin.RADIUS_MAX, self.bin_radius))
        try:
            self.bin_cols = int(float(self.q.value("bin_cols", 1)))
        except (TypeError, ValueError):
            self.bin_cols = 1
        self.bin_cols = max(storage_bin.COLS_MIN, min(storage_bin.COLS_MAX, self.bin_cols))
        try:
            self.bin_rows = int(float(self.q.value("bin_rows", 1)))
        except (TypeError, ValueError):
            self.bin_rows = 1
        self.bin_rows = max(storage_bin.COLS_MIN, min(storage_bin.COLS_MAX, self.bin_rows))
        self.bin_label = self._bool("bin_label", True)
        try:
            self.tab_count = int(float(self.q.value("tab_count", 4)))
        except (TypeError, ValueError):
            self.tab_count = 4
        self.tab_count = max(1, min(20, self.tab_count))
        try:
            self.tab_width = float(self.q.value("tab_width", 1.0))
        except (TypeError, ValueError):
            self.tab_width = 1.0
        self.tab_width = max(0.2, min(20.0, self.tab_width))
        self.tab_outer_only = self._bool("tab_outer_only", True)
        try:
            self.corner_radius = float(self.q.value("corner_radius", 2.0))
        except (TypeError, ValueError):
            self.corner_radius = 2.0
        self.corner_radius = max(0.1, min(100.0, self.corner_radius))
        try:
            # the default is the 1/8 inch end mill that is on most desktop mills
            self.dogbone_tool_d = float(self.q.value("dogbone_tool_d", 3.175))
        except (TypeError, ValueError):
            self.dogbone_tool_d = 3.175
        self.dogbone_tool_d = max(0.5, min(25.0, self.dogbone_tool_d))
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
        self.q.setValue("show_overhangs", self.show_overhangs)
        self.q.setValue("overhang_limit", self.overhang_limit)
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
        self.q.setValue("filament_material", self.filament_material)
        self.q.setValue("filament_density", self.filament_density)
        self.q.setValue("filament_diameter", self.filament_diameter)
        self.q.setValue("filament_price", self.filament_price)
        self.q.setValue("print_shell", self.print_shell)
        self.q.setValue("print_infill", self.print_infill)
        self.q.setValue("elephant_foot_size", self.elephant_foot_size)
        self.q.setValue("mouse_ear_diameter", self.mouse_ear_diameter)
        self.q.setValue("mouse_ear_height", self.mouse_ear_height)
        self.q.setValue("screw_hole_size", self.screw_hole_size)
        self.q.setValue("screw_hole_kind", self.screw_hole_kind)
        self.q.setValue("screw_hole_clearance", self.screw_hole_clearance)
        self.q.setValue("screw_hole_teardrop", self.screw_hole_teardrop)
        self.q.setValue("clearance_test_pin", self.clearance_test_pin)
        self.q.setValue("clearance_test_first", self.clearance_test_first)
        self.q.setValue("clearance_test_step", self.clearance_test_step)
        self.q.setValue("clearance_test_count", self.clearance_test_count)
        self.q.setValue("overhang_test_first", self.overhang_test_first)
        self.q.setValue("overhang_test_step", self.overhang_test_step)
        self.q.setValue("overhang_test_count", self.overhang_test_count)
        self.q.setValue("hinge_slit", self.hinge_slit)
        self.q.setValue("hinge_gap", self.hinge_gap)
        self.q.setValue("hinge_pitch", self.hinge_pitch)
        self.q.setValue("hinge_margin", self.hinge_margin)
        self.q.setValue("hinge_vertical", self.hinge_vertical)
        self.q.setValue("vent_shape", self.vent_shape)
        self.q.setValue("vent_size", self.vent_size)
        self.q.setValue("vent_web", self.vent_web)
        self.q.setValue("vent_margin", self.vent_margin)
        self.q.setValue("fit_test_first", self.fit_test_first)
        self.q.setValue("fit_test_step", self.fit_test_step)
        self.q.setValue("fit_test_count", self.fit_test_count)
        self.q.setValue("gear_teeth", self.gear_teeth)
        self.q.setValue("gear_module", self.gear_module)
        self.q.setValue("gear_angle", self.gear_angle)
        self.q.setValue("gear_bore", self.gear_bore)
        self.q.setValue("gear_backlash", self.gear_backlash)
        self.q.setValue("gear_thickness", self.gear_thickness)
        self.q.setValue("box3d_length", self.box3d_length)
        self.q.setValue("box3d_width", self.box3d_width)
        self.q.setValue("box3d_height", self.box3d_height)
        self.q.setValue("box3d_wall", self.box3d_wall)
        self.q.setValue("box3d_floor", self.box3d_floor)
        self.q.setValue("box3d_radius", self.box3d_radius)
        self.q.setValue("box3d_lid", self.box3d_lid)
        self.q.setValue("box3d_lip", self.box3d_lip)
        self.q.setValue("box3d_clearance", self.box3d_clearance)
        self.q.setValue("box3d_screws", self.box3d_screws)
        self.q.setValue("box3d_vents", self.box3d_vents)
        self.q.setValue("knob_diameter", self.knob_diameter)
        self.q.setValue("knob_height", self.knob_height)
        self.q.setValue("knob_grip", self.knob_grip)
        self.q.setValue("knob_lobes", self.knob_lobes)
        self.q.setValue("knob_kind", self.knob_kind)
        self.q.setValue("knob_bore", self.knob_bore)
        self.q.setValue("knob_clearance", self.knob_clearance)
        self.q.setValue("knob_shaft_d", self.knob_shaft_d)
        self.q.setValue("knob_flat", self.knob_flat)
        self.q.setValue("standoff_height", self.standoff_height)
        self.q.setValue("standoff_outer", self.standoff_outer)
        self.q.setValue("standoff_shape", self.standoff_shape)
        self.q.setValue("standoff_kind", self.standoff_kind)
        self.q.setValue("standoff_bore", self.standoff_bore)
        self.q.setValue("standoff_clearance", self.standoff_clearance)
        self.q.setValue("standoff_count", self.standoff_count)
        self.q.setValue("label_text", self.label_text)
        self.q.setValue("label_height", self.label_height)
        self.q.setValue("label_thickness", self.label_thickness)
        self.q.setValue("label_relief", self.label_relief)
        self.q.setValue("label_mode", self.label_mode)
        self.q.setValue("label_margin", self.label_margin)
        self.q.setValue("label_radius", self.label_radius)
        self.q.setValue("label_hole", self.label_hole)
        self.q.setValue("text3d_text", self.text3d_text)
        self.q.setValue("text3d_family", self.text3d_family)
        self.q.setValue("text3d_height", self.text3d_height)
        self.q.setValue("text3d_depth", self.text3d_depth)
        self.q.setValue("text3d_bold", self.text3d_bold)
        self.q.setValue("text3d_align", self.text3d_align)
        self.q.setValue("clip_cable_d", self.clip_cable_d)
        self.q.setValue("clip_width", self.clip_width)
        self.q.setValue("clip_wall", self.clip_wall)
        self.q.setValue("clip_opening", self.clip_opening)
        self.q.setValue("clip_base", self.clip_base)
        self.q.setValue("clip_bore", self.clip_bore)
        self.q.setValue("clip_count", self.clip_count)
        self.q.setValue("hinge_length", self.hinge_length)
        self.q.setValue("hinge_width", self.hinge_width)
        self.q.setValue("hinge_thickness", self.hinge_thickness)
        self.q.setValue("hinge_knuckles", self.hinge_knuckles)
        self.q.setValue("hinge_clearance", self.hinge_clearance)
        self.q.setValue("hinge_pin_d", self.hinge_pin_d)
        self.q.setValue("battery_cell", self.battery_cell)
        self.q.setValue("battery_count", self.battery_count)
        self.q.setValue("battery_wall", self.battery_wall)
        self.q.setValue("battery_clearance", self.battery_clearance)
        self.q.setValue("battery_grip", self.battery_grip)
        self.q.setValue("bin_length", self.bin_length)
        self.q.setValue("bin_width", self.bin_width)
        self.q.setValue("bin_height", self.bin_height)
        self.q.setValue("bin_wall", self.bin_wall)
        self.q.setValue("bin_floor", self.bin_floor)
        self.q.setValue("bin_radius", self.bin_radius)
        self.q.setValue("bin_cols", self.bin_cols)
        self.q.setValue("bin_rows", self.bin_rows)
        self.q.setValue("bin_label", self.bin_label)
        self.q.setValue("tab_count", self.tab_count)
        self.q.setValue("tab_width", self.tab_width)
        self.q.setValue("tab_outer_only", self.tab_outer_only)
        self.q.setValue("corner_radius", self.corner_radius)
        self.q.setValue("dogbone_tool_d", self.dogbone_tool_d)
        self.q.setValue("recent", self.recent[:8])
        self.q.sync()   # now on disk, so the next Settings (or the next start) reads these


#: one file-dialog filter per 3D format Volura imports
MESH_FILTERS = ["STL (*.stl)", "OBJ (*.obj)", "3MF (*.3mf)", "PLY (*.ply)", "OFF (*.off)",
                "AMF (*.amf)", "glTF (*.glb *.gltf)", "COLLADA (*.dae)"]


def mesh_filter():
    """The file-dialog filter for every 3D format Volura imports."""
    return tr("3D models") + " (" + " ".join(f"*{e}" for e in MESH_EXTENSIONS) + ")"


# ---------------------------------------------------------------- dialogs

SHOW_WELCOME_KEY = "show_welcome"


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


class MainWindow(QMainWindow, SheetMixin, PrintMixin, UiMixin):
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
        self._show_mode_menus(want)
        self._enable_3d_tools(want == "3d")
        if want == "2d":
            self.sheet_view.set_tool("select")
            self.sheet_view.setFocus()
            self.sheet_view.announce()
        else:
            self.view.setFocus()
            self.view.announce()
        self._show_layers_panel(want == "2d")
        if changed:
            self.refit_vents()        # a grille fits its outline again, also without a link
            self.sync_links("to_sheet" if want == "2d" else "to_model")
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
            for bar in self._sheet_toolbars():
                bar.show()
        else:
            for bar in self._sheet_toolbars():
                bar.hide()
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
        self.a_exp_obj = A(tr("3D Model with &Colors (OBJ)…"), self.export_obj)
        self.a_exp_svg2d = A(tr("2D Vector (&SVG)…"), lambda checked=False: self.export_vector("svg"))
        self.a_exp_dxf2d = A(tr("2D Vector (&DXF)…"), lambda checked=False: self.export_vector("dxf"))
        self.a_exp_gcode = A(tr("G-code for a GRBL Laser (.gcode)…"),
                             lambda checked=False: self.export_sheet("gcode"))
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
            a.setProperty(SOURCE_TEXT, label)     # the English label: the key of its shortcut
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
        # ---- 3D text
        self.a_text3d = A(tr("3D Text…"), lambda checked=False: self.text3d_dialog(), None,
                          "text3d", False,
                          tr("Add a part made of letters to the model: your text in the font, "
                             "the height and the depth you choose, standing on the bed"))

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
        self.a_overhangs = A(tr("Overhangs"), self.toggle_overhangs, None, None, checkable=True,
                             tip=tr("Mark the faces that hang over too far to print without support"))
        self.a_overhangs.setChecked(self.settings.show_overhangs)
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
        self.a_arrange = A(tr("Arrange on Bed"), self.arrange_on_bed, None, None, False,
                           tr("Lay the groups side by side on the print bed, without overlapping"))

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
        self.a_screw_hole = A(tr("Screw Hole…"),
                              lambda checked=False: self.screw_hole_dialog(), None, None, False,
                              tr("A printable hole for a metric screw: straight, with a "
                                 "counterbore, a countersunk head or a nut pocket"))
        self.a_clearance_test = A(tr("Clearance Test…"),
                                  lambda checked=False: self.clearance_test_dialog(), None, None,
                                  False,
                                  tr("A small test print that shows which clearance fits your "
                                     "printer: a row of holes with a loose pin"))
        self.a_overhang_test = A(tr("Overhang Test…"),
                                 lambda checked=False: self.overhang_test_dialog(), None, None,
                                 False,
                                 tr("A small test print that shows up to which angle your "
                                    "printer prints an overhang without support"))

        self.a_cutplane = self.tool_act(tr("Cut"), "cutplane", None, "cutplane")
        self.a_cut_join = A(tr("Cut and Join..."), lambda checked=False: self.cut_join_dialog(),
                            None, "cut_join", False,
                            tr("Cut the model at a plane and make pins, holes or a dovetail"))
        self.a_fillet = self.tool_act(tr("Fillet"), "fillet", None, "fillet")
        self.a_fillet.setToolTip(tr("Click edges and type a radius: a convex or concave rounding"))
        self.a_chamfer = self.tool_act(tr("Chamfer"), "chamfer", None, "chamfer")
        self.a_chamfer.setToolTip(tr("Click edges and type a distance: a bevel along the edge"))
        self.a_elephant = A(tr("Elephant Foot Chamfer…"),
                            lambda checked=False: self.elephant_foot_chamfer(), None, None, False,
                            tr("A small chamfer along the bottom edges, so the first printed "
                               "layer does not stick out"))
        self.a_mouse_ears = A(tr("Mouse Ears…"),
                              lambda checked=False: self.mouse_ears_dialog(), None, None, False,
                              tr("Thin discs on the bed under the sharp bottom corners of a "
                                 "part, so those corners do not lift while printing"))
        self.a_orient_print = A(tr("Orient for Printing"),
                                lambda checked=False: self.orient_for_printing(), None, None,
                                False,
                                tr("Turn every part so that it stands on the face that needs "
                                   "the least support, like the auto-orient of a slicer"))

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
        self.a_job_estimate = A(tr("Job Estimate…"),
                                lambda checked=False: self.job_estimate_dialog(), None,
                                "job_estimate", False,
                                tr("How long the laser takes for the worksheet and how many "
                                   "metres it cuts"))
        self.a_sheet_to_model = A(tr("Parts to &3D Model"), self.sheet_to_model, None, "to_model",
                                  False, tr("Put what is cut into the 3D model as plates with the "
                                            "sheet thickness"))
        self.a_update_links = A(tr("&Update Links"), self.update_links, None, None, False,
                                tr("Carry a shape that changed on one side to the other, in both "
                                   "directions"))
        self.a_unlink = A(tr("&Unlink"), self.unlink, None, None, False,
                          tr("Remove the links of the selected parts and outlines"))
        for name, label, tip in SHEET_TOOL_INFO:
            self.sheet_tool_act(tr(label), name, tip).setProperty(SOURCE_TEXT, label)
        self.a_mirror_h = A(tr("Flip &Horizontally"),
                            lambda checked=False: self.sheet_view.mirror(True), None, "mirror_h",
                            False, tr("Swap left and right around the center of the selection"))
        self.a_mirror_v = A(tr("Flip &Vertically"),
                            lambda checked=False: self.sheet_view.mirror(False), None, "mirror_v",
                            False, tr("Swap top and bottom around the center of the selection"))
        # ---- the Edit menu of the worksheet (the keys work on the worksheet itself)
        self.a_sheet_delete = A(tr("&Delete"),
                                lambda checked=False: self.sheet_view.delete_selected(), None,
                                None, False, tr("Delete the selected outlines (Del)"))
        self.a_sheet_sel_all = A(tr("Select &All"),
                                 lambda checked=False: self.sheet_view.select_all(), None, None,
                                 False, tr("Select everything on the worksheet (Ctrl+A)"))
        self.a_sheet_sel_none = A(tr("Select None"),
                                  lambda checked=False: self.sheet_view.clear_selection(), None,
                                  None, False, tr("Clear the selection on the worksheet"))
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
        self.a_sheet_duplicate_circle = A(
            tr("Duplicate in a Circle…"),
            lambda checked=False: self.sheet_view.duplicate_circle_dialog(), None, "duplicate",
            False, tr("Place copies of the selection around a centre"))
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
        # ---- living hinge
        self.a_hinge = A(tr("&Living Hinge…"),
                         lambda checked=False: self.living_hinge_dialog(), None, "hinge", False,
                         tr("Fill a rectangle on the worksheet with the slits of a living "
                            "hinge, so the panel bends there"))
        # ---- vent pattern
        self.a_vent_pattern = A(tr("Vent Pattern…"),
                                lambda checked=False: self.vent_pattern_dialog(), None, "vent",
                                False,
                                tr("Fill the selected closed contour on the worksheet with a "
                                   "pattern of holes, for a ventilation grille or a speaker "
                                   "cover"))
        # ---- fit test
        self.a_fit_test = A(tr("Fit Test…"),
                            lambda checked=False: self.fit_test_dialog(), None, "fit_test", False,
                             tr("A comb of slots that shows which offset gives a good press "
                                "fit for the sheet you are cutting"))
        # ---- spur gear
        self.a_spur_gear = A(tr("Spur Gear…"),
                             lambda checked=False: self.spur_gear_dialog(), None, "gear", False,
                             tr("Add an involute spur gear to the worksheet, with the teeth, "
                                "the module and the bore you choose"))
        self.a_spur_gear_3d = A(tr("Spur Gear…"),
                                lambda checked=False: self.spur_gear_dialog(solid=True), None,
                                None, False,
                                tr("Add a printable involute spur gear to the model, with "
                                   "the teeth, the module, the bore and the thickness you "
                                   "choose"))
        # ---- project box
        self.a_project_box = A(tr("Project Box…"),
                               lambda checked=False: self.project_box_dialog(), None, None,
                               False,
                               tr("Add a printed enclosure to the model: a box with the "
                                  "inside size you choose and a lid that slides into it"))
        # ---- knob
        self.a_knob = A(tr("Knob…"),
                        lambda checked=False: self.knob_dialog(), None, None, False,
                        tr("Add a printable thumb knob to the model, with the grip you choose "
                           "and a mount for a nut, a bolt head or a shaft"))
        # ---- standoff
        self.a_standoff = A(tr("Standoff…"),
                            lambda checked=False: self.standoff_dialog(), None, None, False,
                            tr("Add printable standoffs to the model: spacers for a PCB or a "
                               "panel, round or hexagonal, with a through hole, a self-tapping "
                               "pilot or a hole for a heat-set insert"))
        # ---- label plate
        self.a_label_plate = A(tr("Label Plate…"),
                               lambda checked=False: self.label_plate_dialog(), None, None,
                               False,
                               tr("Add a printable name tag or drawer label to the model: your "
                                  "text raised or engraved on a plate, with a hole to hang it up"))
        # ---- cable clip
        self.a_cable_clip = A(tr("Cable Clip…"),
                              lambda checked=False: self.cable_clip_dialog(), None, None,
                              False,
                              tr("Add printable cable clips to the model: a ring that snaps "
                                 "round a cable and a foot to screw against a wall or under a "
                                 "desk"))
        # ---- print-in-place hinge
        self.a_print_hinge = A(tr("Print-in-Place Hinge…"),
                               lambda checked=False: self.hinge_dialog(), None, "print_hinge",
                               False,
                                tr("Add a printable hinge to the model: two leaves that lie flat "
                                   "on the bed, joined by a row of knuckles around a pin, so it "
                                   "prints as one piece and turns as soon as it comes off the "
                                   "printer"))
        # ---- battery holder
        self.a_battery_holder = A(tr("Battery Holder…"),
                                  lambda checked=False: self.battery_holder_dialog(), None,
                                  "battery_holder", False,
                                  tr("Add a printable battery holder to the model: a block with "
                                     "an open cradle per cell, so the cells press in from above "
                                     "and the wires come through the slots in the end walls and "
                                     "the holes in the floor"))
        # ---- storage bin
        self.a_storage_bin = A(tr("Storage Bin…"),
                               lambda checked=False: self.storage_bin_dialog(), None,
                               "storage_bin", False,
                               tr("Add a printable storage bin to the model: an open-top box "
                                  "whose compartments are made by straight dividers, with a "
                                  "sloped ledge along the inside of the front wall on which a "
                                  "label lies readable"))
        # ---- holding tabs
        self.a_holding_tabs = A(tr("Holding &Tabs…"),
                                lambda checked=False: self.holding_tabs_dialog(), None, "tabs",
                                False,
                                tr("Leave small uncut bridges in the selected closed contours, "
                                   "so a part stays in the sheet until the job is done"))
        # ---- round corners
        self.a_round_corners = A(tr("Round &Corners…"),
                                 lambda checked=False: self.round_corners_dialog(), None,
                                 "round_corners",
                                 False,
                                 tr("Round the sharp corners of the selected paths with a "
                                    "radius, so a cut panel has no sharp edges left"))
        # ---- dogbone corners
        self.a_dogbone_corners = A(tr("&Dogbone Corners…"),
                                   lambda checked=False: self.dogbone_corners_dialog(),
                                   None, "dogbone", False,
                                   tr("Cut a round relief in every inside corner of the "
                                      "selected closed contours, so a round mill leaves no "
                                      "fillet in the corner"))

    def sheet_tool_act(self, label, name, tip=None):
        """Button for a 2D tool; the key works on the worksheet (M, Q, S, Space)."""
        a = self.act(label, lambda checked=False, n=name: self.sheet_view.set_tool(n), None,
                     name, checkable=True, tip=tip)
        self.sheet_tool_actions[name] = a
        return a

    def _build_menus(self):
        """The menu bar. Every command that only works on the 3D model or only on the 2D
        worksheet is marked for its mode (`_menu_modes`), and `_show_mode_menus` shows just
        the commands of the mode you are in; what is not marked works in both."""
        mb = self.menuBar()
        self._menu_modes = {}
        T, S = self.tool_actions, self.sheet_tool_actions

        def only(mode, *items):
            for a in items:
                if a is not None:
                    act = a.menuAction() if isinstance(a, QMenu) else a
                    self._menu_modes[id(act)] = (act, mode)
            return items

        def sub(title, parent, items, mode=None):
            menu = QMenu(title, parent)
            self.fill(menu, items)
            if mode:
                only(mode, menu)
            return menu

        # ---- File
        m = mb.addMenu(tr("&File"))
        export = QMenu(tr("&Export"), m)
        self.fill(export, only("3d", self.a_exp_stl, self.a_exp_3mf, self.a_exp_colors,
                               self.a_exp_obj) + (None,) +
                  only("2d", self.a_exp_svg2d, self.a_exp_dxf2d, self.a_exp_gcode))
        self.recent_menu = QMenu(tr("&Recent Files"), m)
        self.update_recent_menu()
        self.fill(m, (self.a_new, self.a_open, self.recent_menu, None,
                      self.a_save, self.a_save_as, self.a_save_copy, None,
                      self.a_import, export, None)
                  + only("3d", self.a_print) + (None, self.a_quit))

        # ---- Edit
        m = mb.addMenu(tr("&Edit"))
        unhide = sub(tr("U&nhide"), m, (self.a_unhide_sel, self.a_unhide_last, self.a_unhide))
        unlock = sub(tr("Unl&ock"), m, (self.a_unlock_sel, self.a_unlock_all))
        self.fill(m, (self.a_undo, self.a_redo, None)
                  + only("3d", self.a_cut, self.a_copy, self.a_paste, self.a_paste_loc,
                         self.a_delete, self.a_erase_guides, None,
                         self.a_sel_all, self.a_sel_none, None,
                         self.a_hide, unhide, None,
                         self.a_lock, unlock, None,
                         self.a_component, self.a_group, self.a_close_group)
                  + only("2d", self.a_sheet_delete, None,
                         self.a_sheet_sel_all, self.a_sheet_sel_none))

        # ---- View
        m = mb.addMenu(tr("&View"))
        edge_style = sub(tr("&Edge Style"), m, (self.a_edges, self.a_back_edges))
        face_style = sub(tr("Face St&yle"), m,
                         (self.a_xray, self.a_overhangs, None, self.a_wireframe,
                          self.a_hidden_lines, self.a_shaded, self.a_monochrome))
        comp_edit = sub(tr("Co&mponent Edit"), m, (self.a_hide_rest,))
        self.fill(m, (self.a_toolbars, self.a_sheet, None)
                  + only("3d", self.a_hidden_geometry, self.a_axes, self.a_guides, self.a_grid,
                         None, edge_style, face_style, comp_edit)
                  + only("2d", self.a_sheet_grid, self.a_sheet_layers))

        # ---- Camera
        m = mb.addMenu(tr("&Camera"))
        std_views = sub(tr("&Standard Views"), m,
                        tuple(self.views[k] for k in
                              ("top", "bottom", "front", "back", "left", "right", "iso")))
        self.fill(m, only("3d", self.a_prev_view, self.a_next_view, std_views, None,
                          self.a_ortho, self.a_persp, None,
                          T["orbit"], T["pan"], T["zoom"], self.a_fov, self.a_zoom_window)
                  + (self.a_extents,))

        # ---- Draw: lines and shapes, and the parts Volura builds for you
        m = mb.addMenu(tr("&Draw"))
        lines = sub(tr("&Lines"), m, (T["line"], self.a_freehand))
        arcs = sub(tr("&Arcs"), m, (self.a_arc, self.a_arc2, self.a_arc3, self.a_pie))
        shapes = sub(tr("&Shapes"), m, (T["rectangle"], self.a_rotrect, T["circle"],
                                        T["polygon"]))
        primitives = sub(tr("&Basic Shapes"), m, (self.a_box, self.a_cylinder, self.a_cone,
                                                   self.a_sphere, self.a_torus))
        parts_menu = sub(tr("&Parts"), m,
                         (self.a_spur_gear_3d, self.a_project_box, self.a_knob, self.a_standoff,
                          self.a_label_plate, self.a_cable_clip, self.a_print_hinge,
                          self.a_battery_holder, self.a_storage_bin))
        self.fill(m, only("3d", lines, arcs, shapes, None, primitives, parts_menu,
                          self.a_text3d)
                  + only("2d", S["line"], S["rectangle"], S["roundrect"], S["circle"],
                         S["ellipse"], S["arc"], S["centerarc"], S["polygon"], None, S["text"],
                         None, self.a_box_joint, self.a_spur_gear))

        # ---- Tools
        m = mb.addMenu(tr("&Tools"))
        solids = sub(tr("&Solid Tools"), m, (self.a_intersect, self.a_union, self.a_subtract,
                                             self.a_trim, self.a_split))
        cut_faces = sub(tr("Intersect &Faces"), m, (self.a_cut_face_model, self.a_cut_face_sel))
        threads_menu = sub(tr("&Screw Threads"), m,
                           (self.a_thread, self.a_thread_ext, self.a_thread_int,
                            self.a_thread_bolt, self.a_thread_nut))
        align = sub(tr("&Align"), m, tuple(self.sheet_align_actions.values()))
        distribute = sub(tr("&Distribute"), m, tuple(self.sheet_distribute_actions.values()))
        self.fill(m, only("3d", T["select"], T["eraser"], T["paint"], None,
                          T["move"], T["rotate"], T["scale"], None,
                          T["pushpull"], self.a_follow_me, T["offset"], None,
                          self.a_fillet, self.a_chamfer, None,
                          solids, cut_faces, self.a_shell, self.a_simplify, None,
                          threads_menu, self.a_screw_hole, None,
                          T["tape"], self.a_protractor, None,
                          T["text"], None,
                          self.a_to_sheet, self.a_to_laser)
                  + only("2d", S["select"], S["move"], S["rotate"], S["scale"], None,
                         S["offset"], self.a_sheet_round, None,
                         self.a_sheet_union, self.a_sheet_subtract, self.a_sheet_intersect,
                         None, self.a_sheet_join, self.a_sheet_split, None,
                         self.a_mirror_h, self.a_mirror_v, align, distribute, None,
                         self.a_sheet_duplicate, self.a_sheet_duplicate_circle, None,
                          self.a_round_corners, self.a_hinge, self.a_vent_pattern, None,
                          self.a_sheet_to_layer, self.a_sheet_to_model)
                   + (None, self.a_update_links, self.a_unlink))

        # ---- 3D Printing: what only matters for the printer
        m = mb.addMenu(tr("3D &Printing"))
        only("3d", m)
        self.print_tools_menu = m
        self.fill(m, (self.a_print_setup, self.a_check, None,
                      self.a_orient_print, self.a_drop, self.a_center, self.a_arrange, None,
                      self.a_cutplane, self.a_cut_join, None,
                      self.a_elephant, self.a_mouse_ears, None,
                      self.a_clearance_test, self.a_overhang_test))

        # ---- Laser and Mill: what only matters for the machine
        m = mb.addMenu(tr("&Laser and Mill"))
        only("2d", m)
        self.laser_menu = m
        self.fill(m, (self.a_sheet_plaat, self.a_sheet_nest, None,
                      self.a_fit_test, None,
                      self.a_holding_tabs, self.a_dogbone_corners, None,
                      self.a_job_estimate))

        # ---- Window and Help
        m = mb.addMenu(tr("&Window"))
        self.standard_menu = QMenu(tr("Default &Tray"), m)
        self.fill(m, (self.standard_menu, None, self.a_model_info, self.a_prefs))

        m = mb.addMenu(tr("&Help"))
        self.fill(m, (self.a_welcome, self.a_help, None, self.a_updates, self.a_about))
        self._show_mode_menus(getattr(self, "mode", "3d"))

    def _show_mode_menus(self, mode):
        """Only the commands of the mode: the model's in 3D, the worksheet's in 2D.

        A hidden command also gives up its shortcut, so a key of the model never acts on
        the worksheet or the other way round. Qt folds the separators that are left over."""
        for act, want in getattr(self, "_menu_modes", {}).values():
            act.setVisible(want == mode)

    @staticmethod
    def flat_actions(menu):
        """The commands of a menu and of its submenus, in the order the menu shows them."""
        out = []
        for a in menu.actions():
            if a.isSeparator():
                continue
            if a.menu() is not None:
                out.extend(MainWindow.flat_actions(a.menu()))
            else:
                out.append(a)
        return out

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

    def _sheet_bar(self, name, title, new_column=False):
        """One of the toolbars of the worksheet on the left; made once, hidden in 3D.
        With `new_column` it starts a column of its own next to the previous one."""
        bar = self.findChild(QToolBar, name)
        if bar is None:
            bar = QToolBar(title)
            bar.setObjectName(name)
            bar.setIconSize(QSize(24, 24))
            if new_column:
                self.addToolBarBreak(Qt.LeftToolBarArea)
            self.addToolBar(Qt.LeftToolBarArea, bar)
        return bar

    def _build_sheet_toolbar(self):
        """The toolbars of the worksheet: three columns on the left, only in 2D mode.

        One column with every button would be taller than the window, and Qt then folds
        the rest behind an arrow that closes again as soon as the mouse leaves it. So:
        drawing, editing, and what is for the laser or the mill, side by side."""
        T = self.sheet_tool_actions
        bars = (
            ("SheetToolbar", tr("2D Worksheet"),
             (T["select"], T["move"], T["rotate"], T["scale"], None,
              T["line"], T["rectangle"], T["roundrect"], T["circle"], T["ellipse"],
              T["arc"], T["centerarc"], T["polygon"], T["text"], None,
              T["offset"], self.a_sheet_round, self.a_sheet_grid, None,
              self.a_extents)),
            ("SheetEditToolbar", tr("2D Editing"),
             (self.a_sheet_union, self.a_sheet_subtract, self.a_sheet_intersect, None,
              self.a_sheet_join, self.a_sheet_split, None,
              self.a_mirror_h, self.a_mirror_v, self.a_sheet_align,
              self.a_sheet_distribute, self.a_sheet_duplicate,
              self.a_sheet_duplicate_circle, None,
              self.a_round_corners, self.a_hinge, self.a_vent_pattern, None,
              self.a_sheet_to_layer)),
            ("LaserToolbar", tr("Laser and Mill"),
             (self.a_sheet_plaat, self.a_sheet_nest, None,
              self.a_box_joint, self.a_spur_gear, self.a_fit_test, None,
              self.a_holding_tabs, self.a_dogbone_corners, None,
              self.a_sheet_to_model, self.a_job_estimate)),
        )
        made = []
        for name, title, items in bars:
            bar = self._sheet_bar(name, title, new_column=bool(made))
            if not bar.actions():
                for a in items:
                    bar.addSeparator() if a is None else bar.addAction(a)
            made.append(bar)
        self.tb_sheet, self.tb_sheet_edit, self.tb_laser = made
        edit = self.tb_sheet_edit
        if self.a_sheet_align not in self._dropdown_buttons:
            self._add_dropdown(edit, self.a_sheet_align,
                               tuple(self.sheet_align_actions.values()))
            self._add_dropdown(edit, self.a_sheet_distribute,
                               tuple(self.sheet_distribute_actions.values()))
            self._add_layer_dropdown(edit, self.a_sheet_to_layer)
        for bar in made:
            bar.hide()      # SketchUp shows no toolbar on the left by default

    def _sheet_toolbars(self):
        return [bar for bar in (getattr(self, "tb_sheet", None),
                                getattr(self, "tb_sheet_edit", None),
                                getattr(self, "tb_laser", None)) if bar is not None]

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

    def job_estimate_dialog(self):
        """Edit > Job Estimate…: what the worksheet takes on the laser, per layer."""
        sheet = self.doc.sheet
        if not sheet.paths:
            self.hint.setText(tr("The worksheet is empty."))
            self.setFocus()
            return None
        JobEstimateDialog(job_estimate(sheet), self).exec()
        self.setFocus()

    def open_layers_panel(self):
        """Window > Default Tray > Layers: color, operation and settings per layer."""
        self.tray.show()
        self.tray.raise_()
        self.tray.open_panel(tr(PANEL_LAYERS))
        self.layers_panel.refresh()

    # ------------------------------------------------------------ round corners
    def apply_round_corners(self, options):
        """The rounded copy of every selected path that has a corner: one undo step.

        Open and closed paths both count, and the two end points of an open path stay where
        they are. Each path is replaced by its copy on the same layer with the same source
        number, so a part still moves and nests as one. A path whose corners are all blunter
        than the arc needs is left alone, and when nothing changes there is no undo step.

        A contour from an import easily has tens of thousands of points, so from
        `BACKGROUND_ROUND_POINTS` on the copies are made in the background, like the placing
        of a big drawing on the sheet.
        """
        sel = self.sheet_view.selected_paths()
        self.set_mode("2d")
        sheet = self.doc.sheet
        radius = float(options["radius"])

        def work(report):
            out = {}
            for i, p in enumerate(sel):
                if i:
                    report(i / len(sel))
                out[id(p)] = sheettools.round_corners(p.pts, radius, closed=p.closed)
            return out

        if sum(len(p.pts) for p in sel) >= self.BACKGROUND_ROUND_POINTS:
            job = background.run(self, work, text=tr("Rounding the corners…"), wait=True)
            if job.was_cancelled:
                self.hint.setText(tr("Rounding the corners was cancelled."))
                return None
            if job.error is not None:
                QMessageBox.critical(self, tr("Round Corners Failed"), str(job.error))
                return None
            copies = job.result
        else:
            copies = work(lambda *args: None)
        replacements, rounded, blunt = {}, 0, 0
        for p in sel:
            pts = copies[id(p)]
            if np.array_equal(pts, np.asarray(p.pts, dtype=float)):
                blunt += 1
                continue
            replacements[id(p)] = pts
            rounded += 1
        if not rounded:
            self.hint.setText(trn("{n} path has no corners to round.",
                                  "{n} paths have no corners to round.", blunt).format(n=blunt)
                              if blunt else tr("Nothing to round."))
            return None
        self.doc.checkpoint()
        live = set(replacements)
        keep, fresh = [], []
        for p in sheet.paths:
            if id(p) not in live:
                keep.append(p)
                continue
            q = VPath(np.asarray(replacements[id(p)], dtype=float), p.closed, p.layer, p.filled,
                      p.rule, p.src, {})
            keep.append(q)
            fresh.append(q)
        sheet.paths = keep
        self.doc.notify()
        self.sheet_view.set_selection(fresh)
        self.sheet_view._refresh_outside(announce=False)
        self.hint.setText(trn("{n} path rounded.", "{n} paths rounded.", rounded)
                          .format(n=rounded))
        return fresh

    # ------------------------------------------------------------ dogbone corners
    def apply_dogbone_corners(self, options):
        """The relieved copy of every selected closed contour: one undo step.

        A round cutter leaves a fillet of its own radius in every inside corner of the
        material, so a square tab does not fit into a milled square slot; the relief that
        clears such a corner is a circle of the cutter. Only closed contours count and each
        is replaced by its copy on the same layer with the same source number, so a part
        still moves and nests as one.

        Whether the material lies inside a contour comes from its nesting depth over all the
        closed paths of its own layer on the sheet, exactly as the holding tabs find their
        holes: an even depth is an outer boundary of a part (a reflex vertex is an inside
        corner), an odd depth a hole or slot (there a convex vertex is one). A contour
        without inside corners is left alone, and when nothing changes there is no undo step.

        A contour from an import easily has tens of thousands of points, so from
        `BACKGROUND_DOGBONE_POINTS` on the copies are made in the background, like rounding
        the corners of the same selection.
        """
        sel = [p for p in self.sheet_view.selected_paths() if p.closed and len(p.pts) >= 3]
        skipped = len(self.sheet_view.selected_paths()) - len(sel)
        self.set_mode("2d")
        sheet = self.doc.sheet
        tool_d = float(options["tool_d"])
        depths = {}
        by_layer = {}
        for p in sheet.paths:
            if p.closed and len(p.pts) >= 3:
                by_layer.setdefault(p.layer, []).append(p)
        for group in by_layer.values():
            for p, d in zip(group, cam.nesting_depth(group)):
                depths[id(p)] = d

        def work(report):
            out = {}
            for i, p in enumerate(sel):
                if i:
                    report(i / len(sel))
                inside = depths.get(id(p), 0) % 2 == 0
                pts = sheettools.dogbone_corners(p.pts, tool_d, material_inside=inside)
                out[id(p)] = (pts, len(sheettools.dogbone_spots(p.pts, tool_d,
                                                                material_inside=inside)))
            return out

        if sum(len(p.pts) for p in sel) >= self.BACKGROUND_DOGBONE_POINTS:
            job = background.run(self, work, text=tr("Cutting the dogbones…"), wait=True)
            if job.was_cancelled:
                self.hint.setText(tr("Cutting the dogbones was cancelled."))
                return None
            if job.error is not None:
                QMessageBox.critical(self, tr("Dogbone Corners Failed"), str(job.error))
                return None
            copies = job.result
        else:
            copies = work(lambda *args: None)
        replacements, corners, plain = {}, 0, 0
        for p in sel:
            pts, made = copies[id(p)]
            if not made:
                plain += 1
                continue
            replacements[id(p)] = pts
            corners += made
        notes = []
        if corners:
            notes.append(trn("{n} inside corner got a dogbone.",
                             "{n} inside corners got a dogbone.", corners).format(n=corners))
        elif plain:
            notes.append(trn("{n} contour has no inside corners.",
                             "{n} contours have no inside corners.", plain).format(n=plain))
        if skipped:
            notes.append(trn("{n} path was skipped (not a closed contour).",
                             "{n} paths were skipped (not a closed contour).", skipped)
                         .format(n=skipped))
        if not corners:
            self.hint.setText(" ".join(notes) if notes else tr("Nothing to dogbone."))
            return None
        self.doc.checkpoint()
        live = set(replacements)
        keep, fresh = [], []
        for p in sheet.paths:
            if id(p) not in live:
                keep.append(p)
                continue
            q = VPath(np.asarray(replacements[id(p)], dtype=float), p.closed, p.layer, p.filled,
                      p.rule, p.src, {})
            keep.append(q)
            fresh.append(q)
        sheet.paths = keep
        self.doc.notify()
        self.sheet_view.set_selection(fresh)
        self.sheet_view._refresh_outside(announce=False)
        self.hint.setText(" ".join(notes))
        return fresh

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
                           (getattr(self, "tb_sheet", None), tr("2D Worksheet")),
                           (getattr(self, "tb_sheet_edit", None), tr("2D Editing")),
                           (getattr(self, "tb_laser", None), tr("Laser and Mill"))):
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

    # ------------------------------------------------------------ 3D mouse
    def start_spacemouse(self):
        """Connect the 3D mouse (3DxWare, Raw Input or spacenavd); never in a test."""
        self.spacemouse.stop()
        if is_offscreen() or not self.settings.spacemouse:
            return None
        return self.spacemouse.start(int(self.winId()))

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
                said.append(trn("{n} duplicate triangle removed from {name}",
                                "{n} duplicate triangles removed from {name}",
                                removed).format(n=removed, name=base))
            if loose:
                said.append(trn("{n} loose triangle removed from {name}",
                                "{n} loose triangles removed from {name}",
                                loose).format(n=loose, name=base))
            said += notes
            if said:
                self.hint.setText("; ".join(said))
            self._maybe_suggest_scale_up(groups)
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

        `message` is a translated template with {n} (the number of paths) and `fields`, or a
        (singular, plural) pair of texts to translate; the pair follows that number."""
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
        if message is None:
            message = (tr_noop("{n} outline added to the worksheet"),
                       tr_noop("{n} outlines added to the worksheet"))
        if isinstance(message, tuple):
            message = trn(message[0], message[1], len(added))
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
        added = self.add_drawing_to_sheet(
            drawing, (tr_noop("{n} outline added from {name}"),
                      tr_noop("{n} outlines added from {name}")), name=name)
        if added is not None and drawing.warnings:
            self.hint.setText(trn("{n} outline added from {name} ({warnings})",
                                  "{n} outlines added from {name} ({warnings})", len(added))
                              .format(n=len(added), name=name,
                                      warnings="; ".join(drawing.warnings)))
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
        return self.add_drawing_to_sheet(
            drawing, (tr_noop("{n} outline copied to the worksheet"),
                      tr_noop("{n} outlines copied to the worksheet")))

    # ------------------------------------------------------------ 3D and worksheet together
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
        added = self.add_drawing_to_sheet(
            drawing, (tr_noop("{n} outline put on layer {layer}"),
                      tr_noop("{n} outlines put on layer {layer}")), layer=layer)
        if added and thickness and not sheet.sheet_thickness:
            sheet.sheet_thickness = thickness
            self.hint.setText(trn("{n} outline put on layer {layer}; sheet thickness {t:g} mm "
                                  "taken over",
                                  "{n} outlines put on layer {layer}; sheet thickness {t:g} mm "
                                  "taken over", len(added))
                              .format(n=len(added), layer=layer, t=thickness))
        if added:
            self._link_sent_to_sheet(added, groups, faces)
        return added

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
            plates = sheet3d.part_plates(sheet, t, selected or None)
        except Exception as e:
            QMessageBox.critical(self, tr("Parts to 3D Model"), str(e))
            return None
        if not plates:
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
            self.doc.links = [l for l in self.doc.links if int(l.group) in live]
        groups = [ops.group_from_manifold(self.doc, man, self.settings.current_color, name)
                  for name, man, _src in plates]
        self.doc.sheet_part_groups = ([] if not selected else
                                      [g for g in getattr(self.doc, "sheet_part_groups", ())
                                       if g in live]) + [g.id for g in groups]
        self.doc.set_selection([("g", g.id, g.id) for g in groups])
        self.doc.notify()
        self.set_mode("3d")
        self.view.zoom_extents()
        self._link_parts_to_sheet(groups, plates)
        message = trn("{n} part of {t:g} mm placed in the model",
                      "{n} parts of {t:g} mm placed in the model", len(groups))
        self.hint.setText(message.format(n=len(groups), t=float(t)))
        return groups

    # ------------------------------------------------------------ export for laser and mill
    def export_sheet(self, kind="svg", path=None, options=None):
        """Export > 2D vector on the worksheet: "Export for Laser/Mill".

        Without `path` the dialog asks which layers may go, whether the kerf and cutter
        compensation should be applied and whether every layer gets its own file; with a
        `path` the default choices are used. Raster layers go out filled, the rest as
        hairlines, and the operation and settings of every layer go into the SVG so
        Volura finds them again when reading it.

        `kind` "gcode" writes the same drawing as G-code for a GRBL laser
        (`volura/gcode.py`), with the S maximum and M4 of the dialog, and says in the
        status bar how long that job takes according to the job estimate.
        """
        kind = sheet_export_kind(kind)
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
                EXPORT_FILTERS[kind])
            if not path:
                return None
            if not os.path.splitext(path)[1]:
                path += "." + kind
        filled = [n for n in names
                  if drawing.layers.get(n) is not None and drawing.layers[n].op == "raster"]
        s_max = int(options.get("s_max") or GCODE_S_MAX_DEFAULT)
        dynamic = bool(options.get("dynamic", True))
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
                                               [nm] if nm in filled else [],
                                               s_max=s_max, dynamic=dynamic)
                            files.append(one)
                    else:
                        self._write_vector(sub, target, kind, names, filled,
                                           s_max=s_max, dynamic=dynamic)
                        files.append(target)
            elif options.get("per_layer"):
                for n in names:
                    one = self.layer_file_name(path, n, kind)
                    self._write_vector(drawing, one, kind, [n], [n] if n in filled else [],
                                       s_max=s_max, dynamic=dynamic)
                    files.append(one)
            else:
                self._write_vector(drawing, path, kind, names, filled,
                                   s_max=s_max, dynamic=dynamic)
                files.append(path)
        except Exception as e:
            QMessageBox.critical(self, tr("Export Failed"), str(e))
            return None
        where = (files[0] if len(files) == 1
                 else tr("{n} files ({first})").format(n=len(files), first=files[0]))
        n_paths = len(drawing.paths)
        if kind == "gcode":
            message = (trn("{n} outline exported to {where} with kerf/cutter compensation, "
                           "estimated time {time}",
                           "{n} outlines exported to {where} with kerf/cutter compensation, "
                           "estimated time {time}", n_paths) if options.get("compensate")
                       else trn("{n} outline exported to {where}, estimated time {time}",
                                "{n} outlines exported to {where}, estimated time {time}",
                                n_paths))
        else:
            message = (trn("{n} outline exported to {where} with kerf/cutter compensation",
                           "{n} outlines exported to {where} with kerf/cutter compensation",
                           n_paths) if options.get("compensate")
                       else trn("{n} outline exported to {where}",
                                "{n} outlines exported to {where}", n_paths))
        self.hint.setText(message.format(n=n_paths, where=where,
                                         time=(fmt_time(job_estimate(drawing)["total"])
                                               if kind == "gcode" else None)))
        return files[0]

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

    def export_obj(self):
        self._export_parts_async("obj")

    def _export_parts_async(self, kind):
        """Export the printable parts in the background: prepare, ask, write.

        ``kind`` is "stl", "3mf", "obj" or "colors". Turning the model into solids and
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
            elif kind == "obj":
                path, _ = QFileDialog.getSaveFileName(self, tr("Export as OBJ"),
                                                      self.export_name(".obj"), "OBJ (*.obj)")
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
            if kind == "obj":
                # the .mtl holds one material per color of the parts that are written
                colors = len({fileio.obj_material(p.color)[0]
                              for p in part_list if p.manifold is not None})
                return ("obj", (fileio.export_obj(path, part_list), colors))
            return ("colors", fileio.export_stl_per_color(path, part_list, materials))

        def written(outcome):
            what, value = outcome
            if what == "stl":
                self.hint.setText(tr("STL exported ({n} triangles, watertight): {path}")
                                  .format(n=value, path=path))
            elif what == "obj":
                n, colors = value
                self.hint.setText(tr("OBJ exported ({n} triangles, {colors} colors): {path}")
                                  .format(n=n, colors=colors, path=path))
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
        self.hint.setText(trn("{n} outline exported to {where}",
                              "{n} outlines exported to {where}", len(drawing.paths))
                          .format(n=len(drawing.paths), where=path))
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

    def _vector_export_failed(self, exc):
        QMessageBox.critical(self, tr("Export Failed"), str(exc))

    # ------------------------------------------------------------ hide / lock
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

    # ------------------------------------------------------------ solids
    def _sel_groups(self, minimum=1):
        groups = self.doc.selected_groups()
        if len(groups) < minimum:
            QMessageBox.information(self, "Volura",
                                    trn("Select at least {n} group. First make a group of loose "
                                        "geometry (G).",
                                        "Select at least {n} groups. First make a group of loose "
                                        "geometry (G).", minimum).format(n=minimum))
            return None
        return groups

    def _failed(self, e):
        """Undo the checkpoint of an operation that turned out to be impossible."""
        self.doc.undo()
        self.doc.redo_stack.clear()
        QMessageBox.warning(self, "Volura", str(e))

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

    # ------------------------------------------------------------ elephant foot chamfer

    def elephant_foot_chamfer(self):
        """Tools > Print Tools > Elephant Foot Chamfer: a small chamfer along the bottom
        edges of the selected groups, so the squashed first layer does not stick out."""
        gids = self._target_groups()
        if not gids:
            return
        size, ok = QInputDialog.getDouble(self, tr("Elephant Foot Chamfer"),
                                          tr("Chamfer of the bottom edges (mm):"),
                                          self.settings.elephant_foot_size, 0.1, 2.0, 2)
        if not ok:
            return
        self.settings.elephant_foot_size = float(size)
        self.settings.save()
        segments = int(self.settings.circle_segments)

        def work(report):
            # cutting the chamfer of every part is the slow part: seconds for a big model,
            # so it happens here and the window keeps responding
            out, skipped = [], 0
            for n, gid in enumerate(gids):
                report(n / len(gids), tr("Chamfering the bottom edges…"))
                g = self.doc.group(gid)
                new = ops.elephant_foot_chamfer(g, size, segments)
                if new is None:
                    skipped += 1
                else:
                    out.append((g, new))
            return out, skipped

        def done(result):
            out, skipped = result
            if not out:
                # nothing changed: no undo step either
                self.hint.setText(tr("None of these parts has bottom edges to chamfer."))
                return
            self.doc.checkpoint()         # one undo brings all the old shapes back
            for g, new in out:
                g.ents = new
            self.doc.notify()
            msg = trn("Chamfered the bottom edges of {n} part.",
                      "Chamfered the bottom edges of {n} parts.", len(out)).format(n=len(out))
            if skipped:
                msg += " " + trn("{n} part was skipped: no bottom edges, or the size is "
                                 "too large.",
                                 "{n} parts were skipped: no bottom edges, or the size is "
                                 "too large.", skipped).format(n=skipped)
            self.hint.setText(msg)

        def failed(exc):
            QMessageBox.warning(self, "Volura", str(exc))    # nothing changed, so no undo

        background.run(self, work, text=tr("Chamfering the bottom edges…"),
                       on_done=done, on_error=failed)

    # ------------------------------------------------------------ mouse ears

    def _run_mouse_ears(self, gids, diameter, height):
        """The background run of Mouse Ears, one undo step for all the parts together."""
        segments = int(self.settings.circle_segments)

        def work(report):
            # the boolean of every part is the slow part: seconds for a big model, so it
            # happens here and the window keeps responding
            out, skipped = [], 0
            for n, gid in enumerate(gids):
                report(n / len(gids), tr("Adding mouse ears…"))
                g = self.doc.group(gid)
                new = ops.mouse_ears_part(g, diameter, height, segments=segments)
                if new is None:
                    skipped += 1
                else:
                    out.append((g, new))
            return out, skipped

        def done(result):
            out, skipped = result
            if not out:
                # nothing changed: no undo step either
                self.hint.setText(tr("None of these parts has a sharp bottom corner."))
                return
            self.doc.checkpoint()         # one undo brings all the old shapes back
            for g, new in out:
                g.ents = new
            self.doc.notify()
            msg = trn("Added mouse ears to {n} part.",
                      "Added mouse ears to {n} parts.", len(out)).format(n=len(out))
            if skipped:
                msg += " " + trn("{n} part was skipped: no sharp bottom corners, or not a "
                                 "closed volume.",
                                 "{n} parts were skipped: no sharp bottom corners, or not a "
                                 "closed volume.", skipped).format(n=skipped)
            self.hint.setText(msg)

        def failed(exc):
            QMessageBox.warning(self, "Volura", str(exc))    # nothing changed, so no undo

        background.run(self, work, text=tr("Adding mouse ears…"),
                       on_done=done, on_error=failed)

    # ------------------------------------------------------------ orient for printing

    def orient_for_printing(self):
        """Tools > Print Tools > Orient for Printing: turn the selected groups so that each
        stands on the face that needs the least support, like the auto-orient of a slicer."""
        gids = self._target_groups()
        if not gids:
            return
        limit = float(self.settings.overhang_limit)

        def work(report):
            # looking for the best face of every part is the slow part: seconds for a big
            # model, so it happens here and the window keeps responding
            out, skipped = [], 0
            for n, gid in enumerate(gids):
                report(n / len(gids), tr("Looking for the best way to print…"))
                d = ops.orient_direction(self.doc.group(gid), limit)
                if d is None:
                    skipped += 1
                else:
                    out.append((gid, d))
            return out, skipped

        def done(result):
            out, skipped = result
            if not out:
                # nothing to turn and nothing to drop: no undo step either
                self.hint.setText(tr("None of these parts can be turned: locked or not a "
                                     "closed volume."))
                return
            # a part that already stands the right way round on the bed does not move, so
            # when no part moves there is nothing to undo and no checkpoint is taken
            moving = {gid for gid, d in out
                      if not ops.is_down(d) or abs(float(self.doc.group(gid).bbox()[0][2])) > 1e-6}
            if moving:
                self.doc.checkpoint()     # one undo brings all the old places back
            turned = unchanged = 0
            for gid, d in out:
                if gid in moving and ops.orient_group(self.doc, gid, d):
                    turned += 1
                else:
                    unchanged += 1
            self.doc.notify()
            msg = trn("Turned {n} part to need the least support.",
                      "Turned {n} parts to need the least support.", turned).format(n=turned)
            if unchanged:
                msg += " " + trn("{n} part was already the right way round.",
                                 "{n} parts were already the right way round.",
                                 unchanged).format(n=unchanged)
            if skipped:
                msg += " " + trn("{n} part was skipped: locked or not a closed volume.",
                                 "{n} parts were skipped: locked or not a closed volume.",
                                 skipped).format(n=skipped)
            self.hint.setText(msg)

        def failed(exc):
            QMessageBox.warning(self, "Volura", str(exc))    # nothing moved, so no undo

        background.run(self, work, text=tr("Orienting for printing…"),
                       on_done=done, on_error=failed)

    def _maybe_suggest_scale_up(self, groups):
        """After an import that came in tiny: ask whether the file was in other units and
        scale the new groups up around the origin, as one undo step of its own."""
        bb = ops.selection_bbox(self.doc, [("g", g.id, g.id) for g in groups])
        if bb is None:
            return
        size = float((bb[1] - bb[0]).max())
        if size >= ops.SMALL_IMPORT_MM:
            return
        factor = self._ask_scale_up(size)
        if not factor:
            return
        self.doc.checkpoint()
        ops.transform_selection(self.doc, np.eye(3) * factor, np.zeros(3),
                                items=[("g", g.id, g.id) for g in groups])
        self.doc.notify()
        self.view.zoom_extents()
        self.hint.setText(tr("Scaled up {factor}×: now {size} mm across").format(
            factor=f"{factor:g}", size=f"{size * factor:.1f}"))

    def _ask_scale_up(self, size):
        """The small-import question; the scale factor the user picked, or None."""
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Question)
        box.setWindowTitle(tr("Very Small Import"))
        box.setText(tr("This model is only {size} mm across, so its scale is probably wrong: "
                       "the file was likely made in other units than millimetres. Scale it "
                       "up?").format(size=f"{size:.2g}"))
        buttons = {}
        for unit, factor in ops.IMPORT_UNITS:
            label = tr("From {unit} (×{factor})").format(unit=tr(unit), factor=f"{factor:g}")
            buttons[box.addButton(label, QMessageBox.AcceptRole)] = factor
        box.addButton(tr("Keep size"), QMessageBox.RejectRole)
        box.exec()
        return buttons.get(box.clickedButton())

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

    def arrange_on_bed(self):
        """Tools > Arrange on Bed: lay the groups side by side on the bed, like a slicer."""
        items = [("g", gid, gid) for gid in self._target_groups()]
        if not items:
            return

        def work(report):
            # looking for a place for every part is the slow part: seconds for a pile of
            # parts, so it happens here and the window keeps responding
            report(0.1, tr("Arranging parts…"))
            return ops.arrange_placements(self.doc, items, self.settings.bed_size)

        def done(moves):
            # a part that already stands exactly where the nest puts it (no turn, no move,
            # already on the bed) does not change: it is no reason for an undo step
            real = [(gid, M3, t) for gid, M3, t in moves
                    if not np.allclose(M3, np.eye(3)) or not np.allclose(t, 0.0)
                    or abs(float(self.doc.group(gid).bbox()[0][2])) > 1e-6]
            if not real:
                return                        # nothing to move: no undo step either
            self.doc.checkpoint()             # one undo brings all the old places back
            for gid, M3, t in real:
                sel = [("g", gid, gid)]
                ops.transform_selection(self.doc, M3, t, items=sel)
                ops.drop_to_bed(self.doc, sel)
            self.doc.notify()
            self.hint.setText(trn("Arranged {n} part on the bed.",
                                  "Arranged {n} parts on the bed.", len(real))
                              .format(n=len(real)))

        def failed(exc):
            QMessageBox.warning(self, "Volura", str(exc))    # nothing moved, so no undo either

        background.run(self, work, text=tr("Arranging…"), on_done=done, on_error=failed)

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
        from . import updater       # network code: only here, not at startup
        if not automatic:
            self.hint.setText(tr("Looking for a newer version…"))
        self._update_bridge = run_in_background(
            lambda progress: updater.latest_release(),
            lambda rel, err: self.on_update_result(rel, err, automatic))
        return self._update_bridge

    def on_update_result(self, release, error, automatic=False):
        from . import updater       # network code: only here, not at startup
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
        from . import updater       # network code: only here, not at startup
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
        from . import updater       # network code: only here, not at startup
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
    theme.apply(str(qsettings().value("theme", theme.SYSTEM)), app)
    sys.excepthook = excepthook
    win = MainWindow()
    win.show()
    if len(argv) > 1 and os.path.exists(argv[1]):
        win.file_open(argv[1])
    return app.exec()

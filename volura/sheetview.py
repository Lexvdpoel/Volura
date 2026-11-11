"""The 2D worksheet: a flat drawing next to the 3D model.

`SheetView` is a plain QWidget with QPainter (no OpenGL): the world is in
millimetres with Y up, the grid adapts to the zoom level, and the paths of
`doc.sheet` are drawn in their layer colour. The widget is the game master of the
tools from `volura.sheettools`: it passes on mouse, keys and the Measurements box,
keeps track of the selection and does every operation as one undo step on the document.

The 3D view (`volura.viewport.Viewport`) and this worksheet sit together in the
central widget of the window; `MainWindow.set_mode` switches between them.
"""
from __future__ import annotations

import math

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox,
                               QFormLayout, QHBoxLayout, QLabel, QMenu, QSpinBox, QWidget)

from . import errors, nesting, theme
from .sheettools import (ALIGN_LABELS, MEASURE_LABEL, PICK_PX, SHEET_TOOLS, align_deltas,
                         boolean_loops, copy_of, distribute_deltas, grid_offsets,
                         joined_paths, mirror_matrix, next_src, path_distance, path_in_rect,
                         path_touches_rect, paths_bbox, polar_angles, rotation,
                         split_into_segments, translation, transformed_points)
from .i18n import tr, trn
from .sheetsnap import SnapEngine, grid_step
from .vector2d import (DEFAULT_LAYERS, SHEET_SPACING, OP_LABELS, Layer, VPath, _transform_meta,
                       default_layer_names)

_NO_LAYER = Layer("0", "#000000")   # for a path whose layer is gone

GRID_MIN_PX = 8.0            # grid lines must be at least this far apart
SELECT_COLOR = "#2f6fd6"     # selected paths: blue and thicker
HOVER_COLOR = "#f08a24"      # the path under the cursor
AXIS_X = "#d23b3b"
AXIS_Y = "#3fa34d"
OUTSIDE_COLOR = "#e02020"     # paths outside the sheet or in the margin: red outline
LINK_BADGE = 14.0            # px: the chain that marks an outline linked to a part
LINK_BADGE_GAP = 4.0         # px: between that badge and the corner of the outline
# The sheet and the grid: colours per theme in theme.py. The sheet stays light in the
# dark theme too, so black and blue layer colours remain readable on it.
VCB_CHARS = set("0123456789.,;-+=\"' ")

#: the default Cut layer of a new sheet (in the current language)
CUT_LAYER = next(name for name, _color, op in DEFAULT_LAYERS if op == "cut")

#: shortcuts on the worksheet, the same letters as in 3D (L, R, C, A, F, ...)
SHEET_KEYS = {
    Qt.Key_L: "line", Qt.Key_R: "rectangle", Qt.Key_N: "roundrect", Qt.Key_C: "circle",
    Qt.Key_E: "ellipse", Qt.Key_A: "arc", Qt.Key_B: "centerarc", Qt.Key_P: "polygon",
    Qt.Key_T: "text", Qt.Key_F: "offset", Qt.Key_O: "offset",
}


class SheetView(QWidget):
    """The 2D environment: grid, sheet, paths, selection and the tools that go with it."""

    status_changed = Signal(str)
    vcb_changed = Signal(str, str)        # label, value
    tool_changed = Signal(str)
    coords_changed = Signal(str)          # the cursor position in the status bar
    selection_changed = Signal()
    sheet_settings_requested = Signal()   # "Sheet…" asks for the sheet settings
    auto_arrange_requested = Signal()     # "Auto Arrange" asks for the options
    to_model_requested = Signal()         # "Parts to 3D Model": plates in the model
    go_to_part_requested = Signal(int)    # "Go to Linked Part": the part of a linked outline

    MIN_SCALE = 0.005                      # pixels per mm
    MAX_SCALE = 500.0

    def __init__(self, doc, settings=None, parent=None):
        super().__init__(parent)
        self.doc = doc
        self.settings = settings
        self.tools = {cls.name: cls(self) for cls in SHEET_TOOLS}
        self.tool = None
        self.scale = 4.0                   # pixels per mm
        self.origin = QPointF(0.0, 0.0)    # the screen position of world (0, 0)
        self.selected: list = []           # the selected VPath objects
        self.hover = None
        self.preview_transform = None      # for errors.recover
        self.vcb_text = ""
        self.snap_to_grid = True           # snap to grid, on or off on the toolbar
        self.offset_round = True           # rounded or sharp corners for Offset
        self.current_layer = CUT_LAYER     # the layer of newly drawn paths
        self.snap_engine = SnapEngine(self)
        self._status_note = None
        self._status_shown = ""
        self._vcb_shown = None
        self._pan_from = None
        self._space = False
        self._placed_origin = False
        self._outside_ids: set = set()     # ids of paths outside the sheet or in the margin
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setMinimumSize(320, 240)
        doc.changed.connect(self._on_doc_changed)
        self.set_tool("select")

    # ------------------------------------------------------------ worksheet settings
    def join_type_name(self) -> str:
        return tr("Rounded") if self.offset_round else tr("Sharp")

    def set_offset_round(self, round_corners: bool):
        self.offset_round = bool(round_corners)
        self.update()

    def set_grid_snap(self, on: bool):
        """Snap to grid: pull the cursor to grid crossings or not."""
        self.snap_to_grid = bool(on)
        self.update()

    def _follow_cut_layer(self):
        """The active layer is the default Cut layer under another name in this sheet
        (a file from Volura 0.1.0 has Dutch layer names): use the name of this sheet."""
        names = default_layer_names("cut")
        layers = self.doc.sheet.layers
        if self.current_layer not in layers and self.current_layer in names:
            self.current_layer = next((n for n in names if n in layers), self.current_layer)

    def draw_layer(self) -> str:
        """The layer of newly drawn paths; a locked layer is not one."""
        self._follow_cut_layer()
        lay = self.doc.sheet.layers.get(self.current_layer)
        if lay is None or lay.locked:
            return "0"
        return self.current_layer

    def set_current_layer(self, name) -> bool:
        """The active layer: new paths end up here (the "Layers" panel)."""
        name = str(name)
        if name not in self.doc.sheet.layers:
            return False
        self.current_layer = name
        self.update()
        return True

    def move_to_layer(self, name) -> bool:
        """Move the selection to another layer, as one undo step."""
        sel = self.selected_paths()
        if not sel:
            self.set_status(tr("First select the contours that should go to a layer."))
            return False
        name = str(name)
        lay = self.doc.sheet.layers.get(name)
        if lay is None:
            self.set_status(tr("Layer {name} does not exist.").format(name=name))
            return False
        if lay.locked:
            self.set_status(tr("Layer {name} is locked; unlock it first.").format(name=name))
            return False
        moved = [p for p in sel if p.layer != name]
        if not moved:
            self.set_status(tr("Those contours are already on layer {name}.").format(name=name))
            return False

        def change():
            for p in moved:
                p.layer = name
        self.edit(change)
        self.set_status(trn("{n} contour to layer {name}", "{n} contours to layer {name}",
                            len(moved)).format(n=len(moved), name=name))
        return True

    # ------------------------------------------------------------ world and screen
    def to_screen(self, pt) -> QPointF:
        """World point (mm) to screen coordinates (pixels, Y down)."""
        return QPointF(self.origin.x() + float(pt[0]) * self.scale,
                       self.origin.y() - float(pt[1]) * self.scale)

    def to_world(self, pos) -> np.ndarray:
        """Screen point (QPoint or QPointF) to world coordinates (mm, Y up)."""
        x = float(pos.x()) if hasattr(pos, "x") else float(pos[0])
        y = float(pos.y()) if hasattr(pos, "y") else float(pos[1])
        return self.to_world_xy(x, y)

    def to_world_xy(self, x, y) -> np.ndarray:
        x, y = float(x), float(y)
        return np.array([(x - self.origin.x()) / self.scale,
                         (self.origin.y() - y) / self.scale])

    def world_rect(self):
        """(xmin, ymin, xmax, ymax) of what is visible now."""
        a = self.to_world_xy(0.0, float(self.height()))
        b = self.to_world_xy(float(self.width()), 0.0)
        return float(a[0]), float(a[1]), float(b[0]), float(b[1])

    def _clamp_scale(self, value):
        return max(self.MIN_SCALE, min(self.MAX_SCALE, float(value)))

    def zoom_extents(self):
        """Everything on the worksheet (or the sheet) as large as possible in view."""
        s = self.doc.sheet
        box = s.bounds()
        if s.sheet_w and s.sheet_h:          # the sheets always belong in it entirely
            for ox, oy in nesting.sheet_origins(s) or [(0.0, 0.0)]:
                r = (ox, oy, ox + float(s.sheet_w), oy + float(s.sheet_h))
                box = r if box is None else (min(box[0], r[0]), min(box[1], r[1]),
                                             max(box[2], r[2]), max(box[3], r[3]))
        w, h = max(self.width(), 1), max(self.height(), 1)
        if box is None:
            self.scale = self._clamp_scale(min(w, h) / 200.0)
            self.origin = QPointF(0.35 * w, 0.65 * h)
        else:
            pad, label = 24.0, 20.0          # room around it, and on top for "Sheet · size"
            dw = max(box[2] - box[0], 1e-6)
            dh = max(box[3] - box[1], 1e-6)
            self.scale = self._clamp_scale(min((w - 2 * pad) / dw, (h - 2 * pad - label) / dh))
            cx, cy = 0.5 * (box[0] + box[2]), 0.5 * (box[1] + box[3])
            self.origin = QPointF(0.5 * w - cx * self.scale,
                                  0.5 * (h + label) + cy * self.scale)
        self._placed_origin = True
        self.update()

    def zoom_rect(self, rect):
        """A rectangle (xmin, ymin, xmax, ymax) of the world as large as possible, centred."""
        if rect is None:
            return
        w, h = max(self.width(), 1), max(self.height(), 1)
        pad = 24.0                           # room around it, as in zoom_extents
        dw = max(rect[2] - rect[0], 1e-6)
        dh = max(rect[3] - rect[1], 1e-6)
        self.scale = self._clamp_scale(min((w - 2 * pad) / dw, (h - 2 * pad) / dh))
        self.origin = QPointF(0.5 * w - 0.5 * (rect[0] + rect[2]) * self.scale,
                              0.5 * h + 0.5 * (rect[1] + rect[3]) * self.scale)
        self._placed_origin = True
        self.update()

    def zoom_by(self, factor, about: QPointF | None = None):
        """Zoom around a screen point: that point stays in its place."""
        about = QPointF(self.width() / 2.0, self.height() / 2.0) if about is None else about
        before = self.to_world(about)
        self.scale = self._clamp_scale(self.scale * float(factor))
        after = self.to_world(about)
        self.origin = QPointF(self.origin.x() + (after[0] - before[0]) * self.scale,
                              self.origin.y() - (after[1] - before[1]) * self.scale)
        self.update()

    def spacemouse_move(self, m, speed=1.0):
        """The 3D mouse on the worksheet: slide the cap to pan, pull it towards you to zoom in."""
        k = float(speed) * m.dt
        step = 1.2 * self.height() * k
        self.origin = QPointF(self.origin.x() + m.tx * step, self.origin.y() - m.ty * step)
        if m.tz:
            self.zoom_by(float(math.exp(m.tz * 1.6 * k)))
        self.update()

    # ------------------------------------------------------------ tool and status
    def set_tool(self, name):
        if name not in self.tools:
            return
        if self.tool is not None:
            self.tool.deactivate()
        self.tool = self.tools[name]
        self.vcb_text = ""
        self.hover = None
        self._vcb_shown = None
        self._status_note = None
        self.tool.activate()
        self.setCursor(self.tool.cursor_shape)
        self.tool_changed.emit(name)
        self.refresh_status()
        self.update()

    def set_status(self, text):
        self._status_note = text
        self.show_status(text)

    def show_status(self, text):
        if text and text != self._status_shown:
            self._status_shown = text
            self.status_changed.emit(text)

    def refresh_status(self):
        if self.tool is None:
            return
        note, self._status_note = self._status_note, None
        self.show_status(note or self.tool.status_text())
        self.update_vcb()

    def announce(self):
        """Make the status bar say again what the tool wants now (after a mode switch)."""
        self._status_shown = ""
        self._vcb_shown = None
        self.refresh_status()

    def update_vcb(self):
        if self.tool is None:
            return
        label = self.tool.vcb_label() or MEASURE_LABEL
        value = self.vcb_text if self.vcb_text else self.tool.vcb_value
        if (label, value) == self._vcb_shown:
            return
        self._vcb_shown = (label, value)
        self.vcb_changed.emit(label, value)

    def submit_vcb(self, text):
        self.vcb_text = ""
        if self.tool is not None and text.strip():
            try:
                self.tool.vcb_enter(text.strip())
            except ValueError as e:
                self.set_status(tr("Invalid input: {error}").format(error=e))
            except Exception as e:                       # a tool must never crash
                errors.log_exception(note=f"2D tool: {e}")
                self.set_status(tr("Input failed: {error}").format(error=e))
        self._vcb_shown = None
        self.refresh_status()
        self.update()

    def set_vcb(self, value):
        self.tool.vcb_value = value
        self.update_vcb()

    # ------------------------------------------------------------ selection
    def _on_doc_changed(self):
        self._follow_cut_layer()
        self._prune_selection()
        self._refresh_outside()
        self.update()

    def _refresh_outside(self, announce=True) -> int:
        """Mark paths outside the sheet or in the margin and report how many."""
        try:
            outside = nesting.outside_paths(self.doc.sheet)
        except Exception as e:                       # a warning must never crash
            errors.log_exception(note=f"sheet check: {e}")
            outside = []
        self._outside_ids = {id(p) for p in outside}
        n = len(outside)
        if n and announce:
            self.show_status(trn("{n} path lies outside the sheet or in the margin",
                                 "{n} paths lie outside the sheet or in the margin", n).format(n=n))
        return n

    def _prune_selection(self):
        live = {id(p) for p in self.doc.sheet.paths}
        keep = [p for p in self.selected if id(p) in live]
        if len(keep) != len(self.selected):
            self.selected = keep

    def selected_paths(self) -> list:
        self._prune_selection()
        return list(self.selected)

    def set_selection(self, paths):
        out, seen = [], set()
        for p in list(paths or ()):
            if id(p) not in seen:
                seen.add(id(p))
                out.append(p)
        self.selected = out
        self.selection_changed.emit()
        self.update()

    def add_to_selection(self, paths):
        self.set_selection(list(self.selected) + list(paths or ()))

    def toggle_selection(self, path):
        if any(p is path for p in self.selected):
            self.selected = [p for p in self.selected if p is not path]
        else:
            self.selected = list(self.selected) + [path]
        self.selection_changed.emit()
        self.update()

    def clear_selection(self):
        if self.selected:
            self.selected = []
            self.selection_changed.emit()
        self.update()

    def select_all(self):
        self.set_selection(self._selectable_paths())

    def _layer(self, path) -> Layer:
        """The layer of a path; a path without a layer behaves like layer "0"."""
        return self.doc.sheet.layers.get(path.layer) or _NO_LAYER

    def _visible(self, path) -> bool:
        return self._layer(path).visible

    def _selectable_paths(self):
        return [p for p in self.doc.sheet.paths if self._visible(p)
                and not self._layer(p).locked and len(p.pts)]

    def _link_of_selection(self):
        """The link of the selection: one when every selected path is of one linked src.

        A selection of two outlines, or of an outline and something drawn by hand, is not
        one part, and has no link to go to.
        """
        sel = self.selected_paths()
        if not sel:
            return None
        srcs = {int(p.src) for p in sel}
        if len(srcs) != 1:
            return None
        src = srcs.pop()
        if not src:
            return None
        return next((l for l in self.doc.links if int(l.src) == src), None)

    def selection_center(self):
        box = paths_bbox(self.selected_paths())
        if box is None:
            box = self.doc.sheet.bounds()
        if box is None:
            return None
        return np.array([0.5 * (box[0] + box[2]), 0.5 * (box[1] + box[3])])

    def path_at(self, x, y):
        """The path closest to the cursor, within a few pixels."""
        tol = PICK_PX / self.scale
        p = self.to_world_xy(x, y)
        best, dist = None, tol
        for path in self._selectable_paths():
            d = path_distance(path, p)
            if d <= dist:
                best, dist = path, d
        return best

    def paths_in_band(self, a, b):
        """Window selection: left to right only what lies inside, right to left also what it touches."""
        w0 = self.to_world_xy(min(a[0], b[0]), max(a[1], b[1]))
        w1 = self.to_world_xy(max(a[0], b[0]), min(a[1], b[1]))
        rect = (float(w0[0]), float(w0[1]), float(w1[0]), float(w1[1]))
        window = b[0] >= a[0]
        out = []
        for path in self._selectable_paths():
            if path_in_rect(path, rect) if window else path_touches_rect(path, rect):
                out.append(path)
        return out

    # ------------------------------------------------------------ editing
    def edit(self, change):
        """One undo step: checkpoint, change the paths, notify the document."""
        self.doc.checkpoint()
        change()
        self.doc.notify()
        self.update()

    def transform_paths(self, paths, M, copy=False):
        """Apply matrix `M` to the paths; with `copy` the original paths stay."""
        M = np.asarray(M, dtype=float)
        lin, off = M[:2, :2], M[:2, 2]
        if copy:
            fresh = []
            for p in paths:
                q = copy_of(p)
                q.pts = transformed_points(q.pts, M)
                q.meta = _transform_meta(p.meta, lin, off)
                fresh.append(q)
            self.doc.sheet.paths = list(self.doc.sheet.paths) + fresh
            self.set_selection(fresh)
            return fresh
        for p in paths:
            p.pts = transformed_points(p.pts, M)
            p.meta = _transform_meta(p.meta, lin, off)
        return list(paths)

    def delete_selected(self):
        sel = self.selected_paths()
        if not sel:
            self.set_status(tr("Nothing is selected to delete."))
            return 0
        live = {id(p) for p in sel}
        sheet = self.doc.sheet

        def drop():
            sheet.paths = [p for p in sheet.paths if id(p) not in live]

        self.edit(drop)
        self.clear_selection()
        self.set_status(trn("{n} contour deleted", "{n} contours deleted",
                            len(sel)).format(n=len(sel)))
        return len(sel)

    def mirror(self, horizontal=True):
        """Mirror around the centre of the selection: horizontal = swap left and right."""
        sel = self.selected_paths()
        if not sel:
            self.set_status(tr("First select what you want to mirror."))
            return False
        center = self.selection_center()
        if center is None:
            return False
        self.edit(lambda: self.transform_paths(sel, mirror_matrix(bool(horizontal), center)))
        self.set_status(tr("Contours mirrored"))
        return True

    # ------------------------------------------------------------ operations on the selection
    def _replace_selected(self, shapes, message):
        """Replace the selection with new paths, as one undo step.

        `shapes` is a list `(points, closed, meta, filled)`; all paths that belong to it
        get one source number, so holes are read as one shape.
        """
        sel = self.selected_paths()
        if not sel:
            self.set_status(tr("First select the contours for this operation."))
            return []
        sheet = self.doc.sheet
        src = next_src(sheet.paths)
        live = {id(p) for p in sel}
        fresh: list = []

        def change():
            keep = [p for p in sheet.paths if id(p) not in live]
            for pts, closed, meta, filled in shapes:
                q = VPath(np.asarray(pts, dtype=float), bool(closed), sel[0].layer,
                          bool(filled), "nonzero", src, dict(meta or {}))
                keep.append(q)
                fresh.append(q)
            sheet.paths = keep
        self.edit(change)
        self.set_selection(fresh)
        self.set_status(message)
        return fresh

    def sheet_boolean(self, op) -> bool:
        """Union, subtract or intersect the selected closed paths."""
        sel = self.selected_paths()
        done = {"union": trn("{n} contour united", "{n} contours united", len(sel)),
                "subtract": trn("{n} contour subtracted", "{n} contours subtracted", len(sel)),
                "intersect": trn("{n} contour intersected", "{n} contours intersected",
                                 len(sel))}
        if len([p for p in sel if p.closed and len(p.pts) >= 3]) < (1 if op == "union" else 2):
            self.set_status(tr("First select two closed contours for this operation."))
            return False
        try:
            loops = boolean_loops(sel, op)
        except ValueError as e:
            self.set_status(tr("The operation failed: {error}").format(error=e))
            return False
        except Exception as e:                       # manifold3d must not crash anything here
            errors.log_exception(note=f"2D boolean operation: {e}")
            self.set_status(tr("The operation failed: {error}").format(error=e))
            return False
        if not loops:
            self.set_status(tr("Nothing is left of these contours."))
            return False
        shapes = [(loop, True, {}, True) for loop in loops]
        self._replace_selected(shapes, done[op].format(n=len(sel)))
        return True

    def join_selected(self) -> int:
        """Join open segments into polylines and closed shapes."""
        sel = self.selected_paths()
        if not sel:
            self.set_status(tr("First select the lines you want to join."))
            return 0
        joined = joined_paths(sel)
        src = next_src(self.doc.sheet.paths)
        out = []
        for p in joined:
            p = VPath(p.pts, p.closed, p.layer, p.filled, p.rule, src, dict(p.meta))
            out.append((p.pts, p.closed, p.meta, p.filled))
        live = {id(p) for p in sel}
        sheet = self.doc.sheet

        def change():
            keep = [p for p in sheet.paths if id(p) not in live]
            for pts, closed, meta, filled in out:
                keep.append(VPath(np.asarray(pts, dtype=float), closed, sel[0].layer,
                                  filled, "nonzero", src, meta))
            sheet.paths = keep
        self.edit(change)
        self.set_selection([p for p in sheet.paths if int(p.src) == src])
        made = len(out)
        self.set_status(tr("{made} paths from {n} lines").format(made=made, n=len(sel)))
        return made

    def split_selected(self) -> int:
        """Split every selected path into loose segments."""
        sel = self.selected_paths()
        if not sel:
            self.set_status(tr("First select the contours you want to split."))
            return 0
        segs = split_into_segments(sel)
        src = next_src(self.doc.sheet.paths)
        sheet = self.doc.sheet
        live = {id(p) for p in sel}

        def change():
            keep = [p for p in sheet.paths if id(p) not in live]
            for pts, closed in segs:
                keep.append(VPath(np.asarray(pts, dtype=float), closed, sel[0].layer,
                                  False, "nonzero", src))
            sheet.paths = keep
        self.edit(change)
        self.set_selection([p for p in sheet.paths if int(p.src) == src])
        self.set_status(trn("{n} segment", "{n} segments", len(segs)).format(n=len(segs)))
        return len(segs)

    def align_selected(self, mode) -> bool:
        """Align: left, middle, right, top or bottom, across the whole selection."""
        sel = self.selected_paths()
        if len(sel) < 2:
            self.set_status(tr("Select at least two contours to align."))
            return False
        if mode not in ALIGN_LABELS:
            return False
        deltas = align_deltas(sel, mode)
        self.edit(lambda: [self.transform_paths([p], translation(d))
                           for p, d in zip(sel, deltas)])
        self.set_status(tr("contours aligned ({side})").format(side=ALIGN_LABELS[mode]))
        return True

    def distribute_selected(self, axis="x") -> bool:
        """Distribute: the same distance between the centres everywhere."""
        sel = self.selected_paths()
        if len(sel) < 3:
            self.set_status(tr("Select at least three contours to distribute."))
            return False
        deltas = distribute_deltas(sel, axis)
        self.edit(lambda: [self.transform_paths([p], translation(d))
                           for p, d in zip(sel, deltas)])
        self.set_status(tr("contours distributed"))
        return True

    def duplicate_grid(self, cols=2, rows=1, dx=10.0, dy=10.0) -> list:
        """Duplicate in a row or grid: copies next to and above the original."""
        sel = self.selected_paths()
        if not sel:
            self.set_status(tr("First select what you want to duplicate."))
            return []
        offsets = grid_offsets(cols, rows, dx, dy)
        if not offsets:
            self.set_status(tr("A row of one column and one row is not a duplication."))
            return []
        fresh: list = []

        def change():
            src = next_src(self.doc.sheet.paths)
            for off in offsets:
                for p in sel:
                    q = copy_of(p)
                    q.pts = transformed_points(q.pts, translation(off))
                    q.meta = _transform_meta(p.meta, np.eye(3)[:2, :2], off)
                    q.src = src
                    self.doc.sheet.paths.append(q)
                    fresh.append(q)
        self.edit(change)
        self.set_selection(fresh)
        self.set_status(trn("{n} copy placed", "{n} copies placed",
                            len(fresh)).format(n=len(fresh)))
        return fresh

    def duplicate_circle(self, count=6, total=360.0, center=None, rotate=True) -> list:
        """Duplicate in a circle: copies around a centre, turned along the circle or not."""
        sel = self.selected_paths()
        if not sel:
            self.set_status(tr("First select what you want to duplicate."))
            return []
        try:
            angles = polar_angles(count, total)
        except ValueError:
            self.set_status(tr("A circle needs at least two copies."))
            return []
        center = self.selection_center() if center is None else np.asarray(center, dtype=float)
        fresh: list = []

        def change():
            src = next_src(self.doc.sheet.paths)
            for a in angles[1:]:
                turn = rotation(center, math.radians(a))
                for p in sel:
                    M = turn
                    if not rotate:      # the copies only travel along the circle
                        box = paths_bbox([p])
                        c = np.array([0.5 * (box[0] + box[2]), 0.5 * (box[1] + box[3])])
                        M = translation(transformed_points(c[None, :], turn)[0] - c)
                    lin, off = M[:2, :2], M[:2, 2]
                    q = copy_of(p)
                    q.pts = transformed_points(q.pts, M)
                    q.meta = _transform_meta(p.meta, lin, off)
                    q.src = src
                    self.doc.sheet.paths.append(q)
                    fresh.append(q)
        self.edit(change)
        self.set_selection(list(sel) + fresh)
        self.set_status(trn("{n} copy placed", "{n} copies placed",
                            len(fresh)).format(n=len(fresh)))
        return fresh

    # ------------------------------------------------------------ auto arrange
    def arrange_source(self) -> list:
        """The paths that take part: the selection, or everything visible when nothing is chosen."""
        sel = self.selected_paths()
        if sel:
            return sel
        return [p for p in self.doc.sheet.paths
                if self._visible(p) and not self._layer(p).locked and len(p.pts)]

    def auto_arrange(self, options=None) -> bool:
        """Arrange the parts as economically as possible over the sheets, as one undo step."""
        options = dict(options or {})
        sheet = self.doc.sheet
        if not sheet.sheet_w or not sheet.sheet_h:
            self.set_status(tr("First choose a sheet size (Sheet…) before you arrange."))
            return False
        source = self.arrange_source()
        if not source:
            self.set_status(tr("There is nothing to arrange."))
            return False
        gap = float(options.get("gap", sheet.sheet_gap))
        margin = float(options.get("margin", sheet.sheet_margin))
        rotations = options.get("rotations", (0, 90, 180, 270))
        add_sheets = bool(options.get("add_sheets", True))
        max_sheets = int(sheet.sheet_count or 1) if not add_sheets else int(options.get("max_sheets", 20))
        parts = nesting.parts_of(sheet, source)
        try:
            placements = nesting.nest(parts, sheet.sheet_w, sheet.sheet_h, margin, gap,
                                      rotations=rotations, max_sheets=max_sheets)
        except ValueError as e:
            self.set_status(str(e))
            return False
        except Exception as e:                       # nesting must never crash
            errors.log_exception(note=f"auto arrange: {e}")
            self.set_status(tr("Arranging failed: {error}").format(error=e))
            return False
        spacing = float(sheet.sheet_w) + SHEET_SPACING
        used = max((si for (_p, si, *_r) in placements), default=0) + 1

        def change():
            for (part, si, angle, dx, dy) in placements:
                M = nesting.place_transform(part, angle, dx, dy, origin=(si * spacing, 0.0))
                lin, off = M[:2, :2], M[:2, 2]
                for p in part.paths:
                    p.pts = transformed_points(p.pts, M)
                    p.meta = _transform_meta(p.meta, lin, off)
            if add_sheets:
                sheet.sheet_count = max(int(sheet.sheet_count or 1), used)
        self.edit(change)
        util = nesting.utilization(placements, sheet.sheet_w, sheet.sheet_h)
        self._refresh_outside(announce=False)
        self.set_status(tr("{n} parts on {sheets} sheets, utilization {pct}%").format(
            n=len(parts), sheets=used, pct=round(util * 100)))
        return True

    # ------------------------------------------------------------ context menu
    @errors.safe_handler
    def contextMenuEvent(self, event):
        pos = event.pos()
        path = self.path_at(pos.x(), pos.y())
        if path is not None and path not in self.selected:
            self.set_selection([path])
        menu = self.build_context_menu()
        menu.exec(event.globalPos())

    def build_context_menu(self) -> QMenu:
        """The context menu of the worksheet: tools and the operations."""
        menu = QMenu(self)
        for name, label in (("select", tr("Select")), ("line", tr("Line")),
                            ("rectangle", tr("Rectangle")), ("circle", tr("Circle")),
                            ("text", tr("Text…")), ("offset", tr("Offset"))):
            act = menu.addAction(label)
            act.triggered.connect(
                lambda checked=False, n=name: self._menu_tool(n))
        link = self._link_of_selection()
        if link is not None:
            menu.addSeparator()
            menu.addAction(tr("Go to Linked Part"),
                           lambda checked=False, g=int(link.group):
                           self.go_to_part_requested.emit(g))
        menu.addSeparator()
        for label, slot in ((tr("Union"), lambda: self.sheet_boolean("union")),
                            (tr("Subtract"), lambda: self.sheet_boolean("subtract")),
                            (tr("Intersect"), lambda: self.sheet_boolean("intersect")),
                            (tr("Join"), self.join_selected),
                            (tr("Split into Segments"), self.split_selected)):
            menu.addAction(label, slot)
        align = menu.addMenu(tr("Align"))
        for mode in ALIGN_LABELS:
            align.addAction(ALIGN_LABELS[mode].capitalize(),
                            lambda checked=False, m=mode: self.align_selected(m))
        dist = menu.addMenu(tr("Distribute"))
        dist.addAction(tr("Horizontally"), lambda checked=False: self.distribute_selected("x"))
        dist.addAction(tr("Vertically"), lambda checked=False: self.distribute_selected("y"))
        layers = menu.addMenu(tr("Move to Layer"))
        for name, lay in self.doc.sheet.layers.items():
            act = layers.addAction(f"{name}   —   {OP_LABELS.get(lay.op, lay.op)}")
            act.setEnabled(not lay.locked)
            if lay.locked:
                act.setToolTip(tr("This layer is locked"))
            act.triggered.connect(lambda checked=False, n=name: self.move_to_layer(n))
        menu.addSeparator()
        menu.addAction(tr("Duplicate in Row or Grid…"), self.duplicate_dialog)
        menu.addAction(tr("Duplicate in a Circle…"), self.duplicate_circle_dialog)
        menu.addAction(tr("Delete"), self.delete_selected)
        menu.addAction(tr("Select All"), self.select_all)
        menu.addSeparator()
        menu.addAction(tr("Auto Arrange…"), lambda: self.auto_arrange_requested.emit())
        menu.addAction(tr("Sheet…"), lambda: self.sheet_settings_requested.emit())
        menu.addAction(tr("Parts to 3D Model"), lambda: self.to_model_requested.emit())
        return menu

    def _menu_tool(self, name):
        if name == "text":
            self.set_tool("text")
            self.tools["text"].ask_text()
            self.setFocus()
            return
        self.set_tool(name)

    def duplicate_dialog(self):
        """Dialog 'Duplicate in a row or grid' (count and distance)."""
        dlg = DuplicateSheetDialog(self)
        if dlg.exec() == QDialog.Accepted:
            c, r, dx, dy = dlg.values()
            self.duplicate_grid(c, r, dx, dy)
        self.setFocus()

    def duplicate_circle_dialog(self):
        """Dialog 'Duplicate in a circle' (count, angle and centre)."""
        dlg = DuplicateCircleSheetDialog(self)
        if dlg.exec() == QDialog.Accepted:
            n, a, center, rot = dlg.values()
            self.duplicate_circle(n, a, center, rot)
        self.setFocus()

    # ------------------------------------------------------------ painting
    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        self._draw_background(p)
        self._draw_grid(p)
        self._draw_sheet_rect(p)
        self._draw_margin(p)
        self._draw_axes(p)
        self._draw_paths(p)
        if self.tool is not None:
            try:
                self.tool.paint(p)
            except Exception as e:                       # a preview must never crash
                errors.log_exception(note=f"2D preview: {e}")
        p.end()

    def _draw_grid(self, p):
        step = grid_step(self.scale)
        xmin, ymin, xmax, ymax = self.world_rect()
        if (xmax - xmin) / step > 4000 or (ymax - ymin) / step > 4000:
            return
        rects = self._sheet_screen_rects()
        if not rects or not theme.is_dark():
            self._grid_lines(p, step, "grid", "grid_major")
            return
        # dark: a light grid on the light sheet, a dark grid next to it
        plates = QPainterPath()
        for r in rects:
            plates.addRect(r)
        desk = QPainterPath()
        desk.addRect(QRectF(self.rect()))
        p.save()
        p.setClipPath(desk.subtracted(plates))
        self._grid_lines(p, step, "desk_grid", "desk_grid_major")
        p.setClipPath(plates)
        self._grid_lines(p, step, "grid", "grid_major")
        p.restore()

    def _grid_lines(self, p, step, minor_color, major_color):
        xmin, ymin, xmax, ymax = self.world_rect()
        minor = QPen(theme.qcolor(minor_color))
        minor.setWidthF(1.0)
        major = QPen(theme.qcolor(major_color))
        major.setWidthF(1.0)
        start = math.floor(xmin / step) * step
        x = start
        while x <= xmax:
            i = round(x / step)
            p.setPen(major if i % 10 == 0 else minor)
            a = self.to_screen((x, ymin))
            b = self.to_screen((x, ymax))
            p.drawLine(a, b)
            x += step
        start = math.floor(ymin / step) * step
        y = start
        while y <= ymax:
            i = round(y / step)
            p.setPen(major if i % 10 == 0 else minor)
            a = self.to_screen((xmin, y))
            b = self.to_screen((xmax, y))
            p.drawLine(a, b)
            y += step

    def _sheet_screen_rects(self):
        """The sheets as screen rectangles; empty without a sheet size."""
        s = self.doc.sheet
        if not s.sheet_w or not s.sheet_h:
            return []
        out = []
        for (ox, oy) in nesting.sheet_origins(s) or [(0.0, 0.0)]:
            a = self.to_screen((ox, oy))
            b = self.to_screen((ox + float(s.sheet_w), oy + float(s.sheet_h)))
            out.append(QRectF(a, b).normalized())
        return out

    def _draw_background(self, p):
        """White without a sheet; with a sheet everything around it is grey and the sheet
        light with a shadow, so it is clear where the material is."""
        rects = self._sheet_screen_rects()
        if not rects:
            p.fillRect(self.rect(), theme.qcolor("sheet_empty"))
            return
        p.fillRect(self.rect(), theme.qcolor("sheet_desk"))
        for r in rects:
            p.fillRect(r.translated(3.0, 3.0), theme.qcolor("sheet_shadow"))
        for r in rects:
            p.fillRect(r, theme.qcolor("sheet_plate"))

    def _draw_sheet_rect(self, p):
        s = self.doc.sheet
        rects = self._sheet_screen_rects()
        if not rects:
            return
        pen = QPen(theme.qcolor("sheet_border"))
        pen.setWidthF(1.6)
        pen.setJoinStyle(Qt.MiterJoin)
        p.setBrush(Qt.NoBrush)
        font = p.font()
        font.setPointSizeF(10.0)
        font.setBold(True)
        size = (f"{float(s.sheet_w):g} × {float(s.sheet_h):g}"
                + (f" × {float(s.sheet_thickness):g}" if s.sheet_thickness else "") + " mm")
        for i, r in enumerate(rects):
            p.setPen(pen)
            p.drawRect(r)
            p.setPen(QPen(theme.qcolor("sheet_label")))
            p.setFont(font)
            label = (tr("Sheet {n}  ·  {size}").format(n=i + 1, size=size) if len(rects) > 1
                     else tr("Sheet  ·  {size}").format(size=size))
            p.drawText(QPointF(r.left() + 2.0, r.top() - 6.0), label)

    def _draw_margin(self, p):
        """The margin along the edge as a dashed line, so it is visible what may go on it."""
        s = self.doc.sheet
        m = float(s.sheet_margin or 0.0)
        if not s.sheet_w or not s.sheet_h or m <= 0:
            return
        pen = QPen(theme.qcolor("sheet_margin"))
        pen.setStyle(Qt.DashLine)
        pen.setWidthF(1.0)
        p.setPen(pen)
        p.setBrush(Qt.NoBrush)
        for (x0, y0, x1, y1) in nesting.usable_rects(s):
            a = self.to_screen((x0, y0))
            b = self.to_screen((x1, y1))
            p.drawRect(QRectF(a, b).normalized())

    def _draw_axes(self, p):
        xmin, ymin, xmax, ymax = self.world_rect()
        pen = QPen(QColor(AXIS_X))
        pen.setWidthF(1.2)
        p.setPen(pen)
        o = self.to_screen((0.0, 0.0))
        p.drawLine(self.to_screen((xmin, 0.0)), self.to_screen((xmax, 0.0)))
        pen = QPen(QColor(AXIS_Y))
        pen.setWidthF(1.2)
        p.setPen(pen)
        p.drawLine(self.to_screen((0.0, ymin)), self.to_screen((0.0, ymax)))
        p.setPen(QPen(theme.qcolor("origin")))
        p.drawLine(QPointF(o.x() - 6, o.y()), QPointF(o.x() + 6, o.y()))
        p.drawLine(QPointF(o.x(), o.y() - 6), QPointF(o.x(), o.y() + 6))

    def _draw_paths(self, p):
        sheet = self.doc.sheet
        chosen = {id(path) for path in self.selected}
        for path in sheet.paths:
            if not self._visible(path) or len(path.pts) < 1:
                continue
            color = self._layer(path).color or "#000000"
            if id(path) in self._outside_ids:      # outside the sheet or in the margin
                warn = QPen(QColor(OUTSIDE_COLOR))
                warn.setWidthF(4.2)
                p.setPen(warn)
                p.drawPath(self._path_of(path))
            if path is self.hover:
                pen = QPen(QColor(HOVER_COLOR))
                pen.setWidthF(3.0)
                p.setPen(pen)
                p.drawPath(self._path_of(path))
            pen = QPen(QColor(SELECT_COLOR if id(path) in chosen else color))
            pen.setWidthF(2.6 if id(path) in chosen else 1.4)
            p.setPen(pen)
            p.drawPath(self._path_of(path))
        self._draw_link_badges(p)

    def _link_badges(self) -> list:
        """(QPointF, link) for every link whose outline lies on the worksheet.

        The point is the top-left corner of the bounding box of the visible paths of that
        link, in screen coordinates: the badge hangs just outside it, so it never covers
        the contour it belongs to. It is only a mark on the screen - exporting and printing
        read the paths of the drawing and know nothing about it.
        """
        out = []
        for link in self.doc.links:
            pts = [np.asarray(path.pts, dtype=float) for path in self.doc.sheet.paths
                   if int(path.src) == int(link.src) and self._visible(path) and len(path.pts)]
            if not pts:
                continue
            q = np.concatenate(pts)
            out.append((self.to_screen((float(q[:, 0].min()), float(q[:, 1].max()))), link))
        return out

    def _draw_link_badges(self, p):
        """The chain that marks a linked outline: two interlocked rounded rectangles."""
        badges = self._link_badges()
        if not badges:
            return
        pen = QPen(theme.qcolor("link_badge"))
        pen.setWidthF(1.6)
        p.setPen(pen)
        p.setBrush(Qt.NoBrush)
        s, r = LINK_BADGE, 0.22 * LINK_BADGE
        for pos, _link in badges:
            x = pos.x() - LINK_BADGE_GAP - s
            y = pos.y() - LINK_BADGE_GAP - s
            p.drawRoundedRect(QRectF(x, y + 0.42 * s, 0.66 * s, 0.58 * s), r, r)
            p.drawRoundedRect(QRectF(x + 0.34 * s, y, 0.66 * s, 0.58 * s), r, r)

    def _path_of(self, path) -> QPainterPath:
        pts = path.pts
        out = QPainterPath()
        if not len(pts):
            return out
        out.moveTo(self.to_screen(pts[0]))
        for q in pts[1:]:
            out.lineTo(self.to_screen(q))
        if path.closed:
            out.closeSubpath()
        return out

    # ------------------------------------------------------------ input
    def event(self, e):
        """Digits for the Measurements box must never become a shortcut, as in the 3D view."""
        from PySide6.QtCore import QEvent
        if e.type() == QEvent.ShortcutOverride:
            t = e.text()
            if self.vcb_text and t and (t in VCB_CHARS or t.isalnum() or t == " "):
                e.accept()
                return True
            if t and t in "0123456789.-,;=":
                e.accept()
                return True
            if e.key() in (Qt.Key_Escape, Qt.Key_Backspace, Qt.Key_Return, Qt.Key_Enter):
                e.accept()
                return True
        return super().event(e)

    @errors.safe_handler
    def keyPressEvent(self, e):
        k = e.key()
        t = e.text()
        mods = e.modifiers()
        if k in (Qt.Key_Return, Qt.Key_Enter):
            if self.vcb_text:
                self.submit_vcb(self.vcb_text)
            elif self.tool is not None:
                self.tool.finish_input()          # Enter without text: done drawing
                self.refresh_status()
                self.update()
            return
        if k == Qt.Key_Backspace:
            if self.vcb_text:
                self.vcb_text = self.vcb_text[:-1]
                self.update_vcb()
            return
        if k == Qt.Key_Escape:
            if self.vcb_text:
                self.vcb_text = ""
                self.update_vcb()
                return
            if self.tool is not None and self.tool.name != "select":
                self.set_tool("select")
                return
            self.clear_selection()
            return
        if k == Qt.Key_Space:
            self._space = True
            if self.tool is not None and self.tool.name != "select":
                self.set_tool("select")
            return
        if mods & Qt.ControlModifier and k == Qt.Key_A:
            self.select_all()
            return
        if k in (Qt.Key_Delete, Qt.Key_Backspace) and not self.vcb_text:
            self.delete_selected()
            return
        if not mods and k in (Qt.Key_M, Qt.Key_Q, Qt.Key_S):
            self.set_tool({Qt.Key_M: "move", Qt.Key_Q: "rotate", Qt.Key_S: "scale"}[k])
            return
        if not mods and k in SHEET_KEYS:
            self.set_tool(SHEET_KEYS[k])          # L, R, C, A, F … as in 3D
            return
        if t and (t in "0123456789.-,;="
                  or (self.vcb_text and (t in VCB_CHARS or t.isalnum() or t == "<"))):
            self.vcb_text += t
            self.update_vcb()
            return
        if self.tool is not None and self.tool.key_press(e):
            self.refresh_status()
            self.update()
            return
        super().keyPressEvent(e)

    @errors.safe_handler
    def keyReleaseEvent(self, e):
        if e.key() == Qt.Key_Space:
            self._space = False
            return
        super().keyReleaseEvent(e)

    @errors.safe_handler
    def mousePressEvent(self, e):
        self.setFocus()
        pos = e.position()
        self._pan_from = None
        if e.button() == Qt.MiddleButton or (e.button() == Qt.LeftButton and self._space):
            self._pan_from = pos
            self.setCursor(Qt.ClosedHandCursor)
            return
        if e.button() == Qt.RightButton:
            if self.tool is not None:
                self.tool.cancel()
            self.refresh_status()
            self.update()
            return
        if e.button() == Qt.LeftButton and self.tool is not None:
            self.tool.mouse_press(e, pos.x(), pos.y())
            self.refresh_status()
            self.update()

    @errors.safe_handler
    def mouseDoubleClickEvent(self, e):
        """A double click ends an open path, as in 3D."""
        self.setFocus()
        if e.button() != Qt.LeftButton or self.tool is None:
            return
        self._pan_from = None
        pos = e.position()
        self.tool.double_click(e, pos.x(), pos.y())
        self.refresh_status()
        self.update()

    @errors.safe_handler
    def mouseMoveEvent(self, e):
        pos = e.position()
        self.show_coords(pos)
        if self._pan_from is not None:
            d = pos - self._pan_from
            self.origin = QPointF(self.origin.x() + d.x(), self.origin.y() + d.y())
            self._pan_from = pos
            self.update()
            return
        if self.tool is not None:
            self.tool.mouse_move(e, pos.x(), pos.y())
            self.refresh_status()
            self.update()

    @errors.safe_handler
    def mouseReleaseEvent(self, e):
        if e.button() in (Qt.MiddleButton, Qt.LeftButton) and self._pan_from is not None:
            self._pan_from = None
            self.setCursor(self.tool.cursor_shape if self.tool else Qt.ArrowCursor)
            return
        if e.button() == Qt.LeftButton and self.tool is not None:
            pos = e.position()
            self.tool.mouse_release(e, pos.x(), pos.y())
            self.refresh_status()
            self.update()

    @errors.safe_handler
    def wheelEvent(self, e):
        steps = e.angleDelta().y() / 120.0
        if steps == 0:
            return
        self.zoom_by(1.25 ** steps, e.position())

    def show_coords(self, pos):
        w = self.to_world(pos)
        self.coords_changed.emit(f"X {w[0]:.2f} mm   Y {w[1]:.2f} mm")

    def resizeEvent(self, event):
        if not self._placed_origin and self.width() > 0 and self.height() > 0:
            self._placed_origin = True
            self.origin = QPointF(0.35 * self.width(), 0.65 * self.height())
        super().resizeEvent(event)

    def showEvent(self, event):
        """The first time the worksheet comes into view: show everything right away."""
        super().showEvent(event)
        if not self._placed_origin:
            self.zoom_extents()


class DuplicateSheetDialog(QDialog):
    """Duplicate in a row or grid: number of columns and rows, and the distance between them."""

    def __init__(self, view: SheetView, parent=None):
        super().__init__(parent or view)
        self.setWindowTitle(tr("Duplicate in a Row or Grid"))
        box = self.selected_box(view)
        lay = QFormLayout(self)
        self.cols = QSpinBox()
        self.cols.setRange(1, 100)
        self.cols.setValue(3)
        self.cols.setToolTip(tr("Number of copies side by side"))
        self.rows = QSpinBox()
        self.rows.setRange(1, 100)
        self.rows.setValue(1)
        self.rows.setToolTip(tr("Number of copies above each other"))
        self.dx = QDoubleSpinBox()
        self.dy = QDoubleSpinBox()
        for spin in (self.dx, self.dy):
            spin.setRange(-10000.0, 10000.0)
            spin.setDecimals(2)
            spin.setSuffix(" mm")
        self.dx.setValue(max(10.0, box[2] - box[0]) if box else 10.0)
        self.dy.setValue(max(10.0, box[3] - box[1]) if box else 10.0)
        lay.addRow(tr("Columns"), self.cols)
        lay.addRow(tr("Rows"), self.rows)
        lay.addRow(tr("Horizontal distance"), self.dx)
        lay.addRow(tr("Vertical distance"), self.dy)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        lay.addRow(buttons)

    @staticmethod
    def selected_box(view: SheetView):
        return paths_bbox(view.selected_paths())

    def values(self):
        return (int(self.cols.value()), int(self.rows.value()),
                float(self.dx.value()), float(self.dy.value()))


class DuplicateCircleSheetDialog(QDialog):
    """Duplicate in a circle: how many copies, over what angle, around which centre."""

    def __init__(self, view: SheetView, parent=None):
        super().__init__(parent or view)
        self.setWindowTitle(tr("Duplicate in a Circle"))
        center = view.selection_center()
        lay = QFormLayout(self)
        self.copies = QSpinBox()
        self.copies.setRange(2, 360)
        self.copies.setValue(6)
        self.copies.setToolTip(tr("Number of copies in the circle, the original included"))
        self.angle = QDoubleSpinBox()
        self.angle.setRange(1.0, 360.0)
        self.angle.setDecimals(1)
        self.angle.setSuffix("°")
        self.angle.setValue(360.0)
        self.angle.setToolTip(tr("The angle of the arc the copies stand on"))
        self.centre = QComboBox()
        self.centre.addItems([tr("Centre of the selection"), tr("Origin"), tr("Point")])
        self.centre.currentIndexChanged.connect(self._centre_changed)
        self.px = QDoubleSpinBox()
        self.py = QDoubleSpinBox()
        for spin in (self.px, self.py):
            spin.setRange(-10000.0, 10000.0)
            spin.setDecimals(2)
            spin.setSuffix(" mm")
            spin.setEnabled(False)
        if center is not None:
            self.px.setValue(float(center[0]))
            self.py.setValue(float(center[1]))
        spot = QWidget()
        row = QHBoxLayout(spot)
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(QLabel(tr("X")))
        row.addWidget(self.px)
        row.addWidget(QLabel(tr("Y")))
        row.addWidget(self.py)
        self.rotate = QCheckBox(tr("Rotate the copies"))
        self.rotate.setChecked(True)
        self.rotate.setToolTip(tr("Turn the copies as they go around the circle"))
        lay.addRow(tr("Copies"), self.copies)
        lay.addRow(tr("Angle"), self.angle)
        lay.addRow(tr("Centre"), self.centre)
        lay.addRow(spot)
        lay.addRow("", self.rotate)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        lay.addRow(buttons)

    def _centre_changed(self, index):
        """The X and Y of the centre only matter for 'Point'."""
        on = index == self.centre.count() - 1
        self.px.setEnabled(on)
        self.py.setEnabled(on)

    def values(self):
        """Count, angle, the centre or None for the centre of the selection, and rotate."""
        mode = self.centre.currentIndex()
        if mode == 0:
            center = None
        elif mode == 1:
            center = (0.0, 0.0)
        else:
            center = (float(self.px.value()), float(self.py.value()))
        return (int(self.copies.value()), float(self.angle.value()), center,
                bool(self.rotate.isChecked()))

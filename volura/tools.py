"""Interactive tools, modelled after the SketchUp 2017 tool set."""
from __future__ import annotations

import math
import re
import time

import numpy as np
import manifold3d as mf
from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QColor, QFont, QFontMetricsF, QPainterPath, QPen

from . import fillet, icons, ops, vector_import
from .brep import Entities
from .geom import (AXES, MeshData, Plane, arc_from_bulge, arc_from_center, arc_through_3,
                   circle_points, closest_on_line_to_ray, extrude_cs, normalize, point_in_polys,
                   rotated_rect, rotation_matrix, simplify_polyline)
from .model import ROOT, LOCKED_MSG

ARROWS = {Qt.Key_Right: 0, Qt.Key_Left: 1, Qt.Key_Up: 2}
AXIS_COLORS = ((0.9, 0.1, 0.1, 1), (0.1, 0.65, 0.1, 1), (0.15, 0.3, 0.95, 1))
UNITS = {"mm": 1.0, "cm": 10.0, "m": 1000.0, "in": 25.4, '"': 25.4, "ft": 304.8, "'": 304.8}


# ------------------------------------------------------------------ parsing

def split_values(text, counts=(1, 2, 3)):
    """Split typed values. ';' always separates (then ',' is a decimal comma);
    otherwise ',' separates unless that gives an unexpected count ("0,5")."""
    text = text.strip().strip("[]<>")
    if ";" in text:
        return [p.strip().replace(",", ".") for p in text.split(";")]
    parts = [p.strip() for p in text.split(",")]
    if len(parts) in counts:
        return parts
    if len(parts) == 2 and 1 in counts:
        return [text.replace(",", ".")]
    return parts


def parse_length(text):
    t = text.strip().lower().replace(" ", "").replace(",", ".")
    m = re.fullmatch(r"([-+]?(?:\d+\.?\d*|\.\d+))(mm|cm|m|in|\"|ft|')?", t)
    if not m:
        raise ValueError(text)
    return float(m.group(1)) * UNITS.get(m.group(2) or "mm", 1.0)


def parse_number(text):
    try:
        return float(text.strip().replace(",", "."))
    except ValueError:
        raise ValueError(text)


MEASURE_LABEL = "Afmetingen"     # the Measurements box label of a tool without one of its own

LENGTH_PRECISION = 2      # decimals shown in length labels (Modelinfo > Eenheden)


def set_length_precision(decimals):
    """Digits of the length display, like SketchUp's Model info > Units > Length precision."""
    global LENGTH_PRECISION
    LENGTH_PRECISION = max(0, min(3, int(decimals)))


def format_length(v, decimals=None):
    """A length in millimetres as shown in the Measurements box and the labels."""
    decimals = LENGTH_PRECISION if decimals is None else int(decimals)
    s = f"{float(v):.{decimals}f}"
    if decimals:
        s = s.rstrip("0").rstrip(".")
    if s in ("-0", ""):
        s = "0"
    return f"{s} mm"


def fmt_len(v):
    return format_length(v, LENGTH_PRECISION)


def fmt_angle(a):
    return f"{a:.1f}°"


def fmt_deg(a):
    """Field of view in the Measurements box, like SketchUp's 35.0 deg."""
    return f"{a:.1f} deg"


def pick_item(pick):
    """Selection item for a pick_entity() result."""
    kind = pick[0]
    if kind == "edge":
        return ("e", pick[1], pick[2])
    if kind == "face":
        return ("f", pick[1], pick[2])
    if kind == "group":
        return ("g", pick[1], pick[1])
    if kind == "guide":
        return ("guide", ROOT, pick[2])
    return None


# ------------------------------------------------------------------ base

class Tool:
    name = ""
    status = ""                     # the instruction of the first step, see status_text()
    measure_label = MEASURE_LABEL   # the label of the Measurements box, see vcb_label()
    cursor_icon = "select"          # SketchUp toont het icoon van het gereedschap als aanwijzer
    _cursor = None

    def __init__(self, view):
        self.view = view
        self.inf = None
        self.vcb_value = ""
        self.lock_dir = None
        self.lock_axis = None
        self.shift_lock = False
        self.reset()

    @property
    def cursor(self):
        """The QCursor of this tool, made on demand: it needs a running QApplication."""
        if self._cursor is None:
            self._cursor = icons.cursor(self.cursor_icon)
        return self._cursor

    @cursor.setter
    def cursor(self, c):
        self._cursor = c

    @property
    def doc(self):
        return self.view.doc

    @property
    def engine(self):
        return self.view.engine

    @property
    def settings(self):
        return self.view.settings

    @property
    def ents(self):
        return self.doc.active_ents()

    def activate(self):
        self.reset()

    def deactivate(self):
        self.reset()

    def reset(self):
        self.inf = None
        self.vcb_value = ""
        self.unlock()

    def cancel(self):
        self.reset()
        self.view.set_status(self.status_text())
        self.view.update_vcb()

    def status_text(self):
        """SketchUp: the status bar shows one short instruction per step."""
        return self.status

    def vcb_label(self):
        """The label of the Measurements box, which belongs to the tool and its step."""
        return self.measure_label

    def busy(self):
        return False

    def mouse_press(self, e, x, y):
        pass

    def mouse_move(self, e, x, y):
        pass

    def mouse_release(self, e, x, y):
        pass

    def double_click(self, e, x, y):
        pass

    def key_press(self, e):
        return False

    def key_release(self, e):
        if e.key() == Qt.Key_Shift and self.shift_lock and not e.isAutoRepeat():
            self.unlock()
            return True
        return False

    def vcb_enter(self, text):
        raise ValueError("geen invoer verwacht")

    def draw(self, view):
        pass

    def paint_overlay(self, painter):
        pass

    def set_vcb(self, value):
        self.vcb_value = value
        self.view.update_vcb()

    # -- helpers ---------------------------------------------------------
    def commit(self, fn):
        """Run a modification as one undoable step."""
        self.doc.checkpoint()
        try:
            result = fn()
        except Exception as ex:  # keep the document consistent on errors
            import traceback
            traceback.print_exc()
            self.doc.undo()
            self.doc.redo_stack.clear()
            self.view.set_status(f"Bewerking mislukt: {ex}")
            return None
        self.doc.notify()
        return result

    @property
    def eye(self):
        return self.view.cam.eye

    def unlock(self):
        self.lock_dir = None
        self.lock_axis = None
        self.shift_lock = False

    def lock_line(self, anchor):
        if self.lock_dir is None or anchor is None:
            return None
        return (np.asarray(anchor, dtype=float), self.lock_dir)

    def handle_lock_keys(self, e, anchor):
        """Arrow keys lock to an axis, Shift locks the current inference."""
        k = e.key()
        if k in ARROWS:
            axis = ARROWS[k]
            if self.lock_axis == axis:
                self.unlock()
            else:
                self.lock_axis, self.lock_dir = axis, AXES[axis].copy()
                self.shift_lock = False
            return True
        if k == Qt.Key_Shift and not e.isAutoRepeat() and anchor is not None and self.inf is not None:
            inf = self.inf
            if inf.kind == "axis":
                self.lock_axis, self.lock_dir = inf.axis, AXES[inf.axis].copy()
            elif inf.kind == "edge" and inf.axis is None:
                self.lock_dir = normalize(inf.edge[1] - inf.edge[0])
            else:
                d = inf.point - anchor
                if np.linalg.norm(d) < 1e-9:
                    return False
                self.lock_dir = normalize(d)
            self.shift_lock = True
            return True
        return False

    def guide_color(self):
        if self.lock_axis is not None:
            return AXIS_COLORS[self.lock_axis]
        if self.lock_dir is not None:
            return (0.7, 0.1, 0.7, 1)
        if self.inf is not None and self.inf.axis is not None and self.inf.kind in ("axis", "edge"):
            return AXIS_COLORS[self.inf.axis]
        return (0.05, 0.05, 0.05, 1)

    def face_under(self, x, y, point=None):
        """(ctx, fid, plane) of the face under the cursor (point must lie on it)."""
        o, d = self.view.cam.ray(x, y)
        fh = self.engine.pick_face(o, d)
        if not fh:
            return None
        t, ctx, fid, p = fh
        pl = self.doc.ents(ctx).face_plane(fid)
        if point is not None and abs(pl.distance(point)) > 1e-3:
            return None
        return ctx, fid, pl

    def drawing_plane(self, inf, x, y, forced_axis=None):
        """The plane a shape is drawn on: forced axis, face under the cursor, or ground."""
        p = inf.point
        if forced_axis is not None:
            return Plane(p, AXES[forced_axis])
        if inf.kind == "face" and inf.plane is not None:
            return Plane(p, inf.plane.n, inf.plane.u)
        fu = self.face_under(x, y, p)
        if fu is not None:
            return Plane(p, fu[2].n, fu[2].u)
        return Plane(p, (0, 0, 1))

    def current_color(self):
        return self.settings.current_color

    def snap_len(self, v):
        """SketchUp length snapping (rounds free lengths, default 1 mm)."""
        step = getattr(self.settings, "length_snap", 1.0)
        if not step:
            return v
        return round(v / step) * step

    def free(self, inf=None):
        """True when the point is not snapped to geometry (length snapping applies)."""
        inf = self.inf if inf is None else inf
        return inf is None or inf.kind not in ("endpoint", "midpoint", "edge", "origin",
                                               "guide", "guidepoint")

    def snap_point(self, anchor, p, inf=None):
        """Round the distance from anchor to p when p is not snapped to geometry."""
        if anchor is None or not self.free(inf):
            return p
        d = p - anchor
        L = float(np.linalg.norm(d))
        if L < 1e-9:
            return p
        return anchor + d / L * self.snap_len(L)

    def hover_item(self, x, y):
        pick = self.engine.pick_entity(x, y)
        if pick[0] == "edge":
            self.view.hover = ("edge", pick[1], pick[2])
        elif pick[0] == "face":
            self.view.hover = ("face", pick[1], pick[2])
        elif pick[0] == "group":
            self.view.hover = ("group", pick[1])
        elif pick[0] == "guide":
            self.view.hover = ("guide", pick[2])
        else:
            self.view.hover = None
        return pick

    def warn_locked(self):
        """Bewerken > Vergrendelen: say in the status bar why nothing happened."""
        self.view.set_status(LOCKED_MSG)

    def show_hidden(self):
        """Beeld > Verborgen geometrie: hidden faces and edges can be selected."""
        return bool(getattr(self.settings, "show_hidden", False))


# ------------------------------------------------------------------ navigation

class OrbitTool(Tool):
    name = "orbit"
    status = ("Sleep met de linkermuisknop om het model te draaien. "
              "Shift + slepen pant, de middelste muisknop doet dat altijd.")
    measure_label = ""
    cursor_icon = "orbit"

    def mouse_press(self, e, x, y):
        self._last = (x, y)

    def mouse_move(self, e, x, y):
        if e.buttons() & Qt.LeftButton:
            dx, dy = x - self._last[0], y - self._last[1]
            if e.modifiers() & Qt.ShiftModifier:
                self.view.cam.pan(dx, dy)
            else:
                self.view.cam.orbit(dx, dy)
            self._last = (x, y)

    def mouse_release(self, e, x, y):
        self.view.commit_camera()


class PanTool(OrbitTool):
    name = "pan"
    status = ("Sleep met de linkermuisknop om het model te verschuiven. "
              "Middelste muisknop pant altijd, Shift + slepen ook.")
    measure_label = ""
    cursor_icon = "pan"

    def mouse_move(self, e, x, y):
        if e.buttons() & Qt.LeftButton:
            self.view.cam.pan(x - self._last[0], y - self._last[1])
            self._last = (x, y)


class ZoomTool(OrbitTool):
    name = "zoom"
    status = ("Sleep met de linkermuisknop omhoog/omlaag om te zoomen. "
              "Typ een gezichtsveld in graden en druk op Enter.")
    measure_label = "Gezichtsveld"
    cursor_icon = "zoom"

    def activate(self):
        super().activate()
        self.show_fov()

    def mouse_move(self, e, x, y):
        if e.buttons() & Qt.LeftButton:
            self.view.cam.zoom(1.01 ** (y - self._last[1]))
            self._last = (x, y)

    def vcb_enter(self, text):
        self.view.cam.fov = max(5.0, min(120.0, parse_number(text.replace("deg", ""))))
        self.show_fov()
        self.view.commit_camera()

    def show_fov(self):
        """Camera > Gezichtsveld: the current field of view in the Measurements box."""
        self.vcb_value = fmt_deg(self.view.cam.fov)
        self.view.update_vcb()


class ZoomWindowTool(Tool):
    """Camera > Zoomvenster: drag a rectangle and zoom the camera into it."""
    name = "zoomwindow"
    status = "Sleep een kader over het gebied waarop je wilt inzoomen."
    measure_label = ""
    cursor_icon = "zoom"

    def reset(self):
        super().reset()
        self.start = None
        self.rect = None

    def busy(self):
        return self.rect is not None

    def mouse_press(self, e, x, y):
        self.start = (x, y)
        self.rect = None

    def mouse_move(self, e, x, y):
        if self.start is not None and e.buttons() & Qt.LeftButton:
            self.rect = (self.start, (x, y))
            self.view.hover = None

    def mouse_release(self, e, x, y):
        if self.rect is not None:
            self.view.zoom_window(self.rect)
        self.back_to_previous_tool()

    def cancel(self):
        super().cancel()
        self.back_to_previous_tool()

    def back_to_previous_tool(self):
        self.start = self.rect = None
        self.view.set_tool(self.view.prev_tool or "select")

    def paint_overlay(self, p):
        if self.rect is None:
            return
        (x0, y0), (x1, y1) = self.rect
        p.setPen(QPen(QColor(30, 30, 30)))
        p.setBrush(QColor(80, 120, 255, 25))
        p.drawRect(QRectF(min(x0, x1), min(y0, y1), abs(x1 - x0), abs(y1 - y0)))


# ------------------------------------------------------------------ select

class SelectTool(Tool):
    name = "select"
    status = ("Selecteer objecten. Shift = toevoegen/verwijderen. Ctrl = toevoegen. "
              "Ctrl+Shift = verwijderen.")
    measure_label = ""
    cursor_icon = "select"

    def reset(self):
        super().reset()
        self.start = None
        self.rect = None
        self._dbl_time = 0.0

    def busy(self):
        return self.rect is not None

    def mouse_press(self, e, x, y):
        self.start = (x, y)
        self.rect = None
        from PySide6.QtWidgets import QApplication
        if time.time() - self._dbl_time < QApplication.doubleClickInterval() / 1000.0:
            self._triple(x, y)
            self.start = None

    def mouse_move(self, e, x, y):
        if self.start and e.buttons() & Qt.LeftButton:
            if abs(x - self.start[0]) + abs(y - self.start[1]) > 4:
                self.rect = (self.start, (x, y))
            self.view.hover = None
            return
        self.hover_item(x, y)

    def mouse_release(self, e, x, y):
        mods = e.modifiers()
        if self.rect is not None:
            (x0, y0), (x1, y1) = self.rect
            lo = np.array([min(x0, x1), min(y0, y1)])
            hi = np.array([max(x0, x1), max(y0, y1)])
            crossing = x1 < x0
            items = []
            ctx = self.doc.active
            ents = self.doc.active_ents()
            show = self.show_hidden()
            cand = [("f", ctx, f) for f in (ents.F if show else ents.visible_faces())]
            cand += [("e", ctx, e) for e in (ents.E if show else ents.visible_edges())
                     if e not in ents.soft]
            if ctx == ROOT:
                cand += [("g", g.id, g.id) for g in self.doc.visible_groups()]
            for it in cand:
                bb = self.view.item_screen_bbox(it)
                if bb is None:
                    continue
                if crossing:
                    hit = np.all(bb[0] <= hi) and np.all(bb[1] >= lo)
                else:
                    hit = np.all(bb[0] >= lo) and np.all(bb[1] <= hi)
                if hit:
                    items.append(it)
            # soft edges belong to the faces they smooth
            for k, c, f in list(items):
                if k == "f":
                    items += [("e", c, e) for e in ents.face_edges(ents.F[f])
                              if e in ents.soft and (show or e not in ents.hidden_e)]
            self._apply(items, mods)
        elif self.start is not None:
            pick = self.engine.pick_entity(x, y)
            if pick[0] in (None, "outside") and self.doc.active != ROOT:
                self.doc.exit_group()
            else:
                it = pick_item(pick)
                self._apply([it] if it else [], mods)
        self.start = None
        self.rect = None

    def double_click(self, e, x, y):
        pick = self.engine.pick_entity(x, y)
        self._dbl_time = time.time()
        if pick[0] == "group":
            if not self.doc.enter_group(pick[1]):
                self.warn_locked()      # a locked group does not open
            return
        ents = self.doc.active_ents()
        ctx = self.doc.active
        items = []
        if pick[0] == "face":
            f = pick[2]
            items = [("f", ctx, f)] + [("e", ctx, e) for e in ents.face_edges(ents.F[f])]
        elif pick[0] == "edge":
            items = [("e", ctx, pick[2])] + [("f", ctx, f) for f in ents.e2f.get(pick[2], ())]
        if items:
            self._apply(items, e.modifiers())

    def _triple(self, x, y):
        pick = self.engine.pick_entity(x, y)
        ents = self.doc.active_ents()
        ctx = self.doc.active
        if pick[0] == "face":
            fs, es = ents.connected(faces=[pick[2]])
        elif pick[0] == "edge":
            fs, es = ents.connected(edges=[pick[2]])
        else:
            return
        self.doc.set_selection([("f", ctx, f) for f in fs] + [("e", ctx, e) for e in es])

    def _apply(self, items, mods):
        sel = list(self.doc.selection)
        ctrl = bool(mods & Qt.ControlModifier)
        shift = bool(mods & Qt.ShiftModifier)
        if ctrl and shift:
            sel = [i for i in sel if i not in items]
        elif ctrl:
            sel += items
        elif shift:
            for i in items:
                if i in sel:
                    sel.remove(i)
                else:
                    sel.append(i)
        else:
            sel = items
        self.doc.set_selection(sel)

    def paint_overlay(self, p):
        if self.rect is None:
            return
        (x0, y0), (x1, y1) = self.rect
        pen = QPen(QColor(30, 30, 30))
        if x1 < x0:
            pen.setStyle(Qt.DashLine)
        p.setPen(pen)
        p.setBrush(QColor(80, 120, 255, 25))
        p.drawRect(QRectF(min(x0, x1), min(y0, y1), abs(x1 - x0), abs(y1 - y0)))


# ------------------------------------------------------------------ eraser / paint

class EraserTool(Tool):
    name = "eraser"
    status = "Selecteer objecten om te wissen. Ctrl = verzachten, Shift = verbergen."
    measure_label = ""
    cursor_icon = "eraser"

    def reset(self):
        super().reset()
        self.marked = []
        self.groups = []
        self.guides = []

    def busy(self):
        return bool(self.marked or self.groups or self.guides)

    def _pick(self, x, y):
        eh = self.engine.pick_edge(x, y, only={self.doc.active}, px=7.0)
        gh = self.engine.pick_guide(x, y, px=5.0)
        if gh is not None and (eh is None or gh[0] <= eh[0]):
            return ("guide", gh[1])
        if eh is not None:
            return ("edge", eh[1], eh[2])
        pick = self.engine.pick_entity(x, y)
        if pick[0] == "group":
            return ("group", pick[1])
        return None

    def _mark(self, x, y):
        it = self._pick(x, y)
        if it is None:
            return
        if it[0] == "edge" and (it[1], it[2]) not in self.marked:
            self.marked.append((it[1], it[2]))
        elif it[0] == "guide" and it[1] not in self.guides:
            self.guides.append(it[1])
        elif it[0] == "group" and it[1] not in self.groups:
            g = self.doc.group(it[1])
            if g is not None and g.locked:
                self.warn_locked()      # a locked group cannot be erased
                return
            self.groups.append(it[1])

    def mouse_press(self, e, x, y):
        self.marked, self.groups, self.guides = [], [], []
        self._mark(x, y)
        self._show()

    def mouse_move(self, e, x, y):
        if e.buttons() & Qt.LeftButton:
            self._mark(x, y)
            self._show()
            return
        it = self._pick(x, y)
        if it is None:
            self.view.hover = None
        elif it[0] == "edge":
            self.view.hover = ("edge", it[1], it[2])
        elif it[0] == "guide":
            self.view.hover = ("guide", it[1])
        else:
            self.view.hover = ("group", it[1])

    def _show(self):
        if self.marked:
            self.view.hover = ("edges", list(self.marked))

    def mouse_release(self, e, x, y):
        soften = bool(e.modifiers() & Qt.ControlModifier)
        marked, groups, guides = list(self.marked), list(self.groups), list(self.guides)
        if marked or groups or guides:
            def run():
                by_ctx = {}
                for ctx, eid in marked:
                    by_ctx.setdefault(ctx, []).append(eid)
                for ctx, eids in by_ctx.items():
                    ents = self.doc.ents(ctx)
                    if soften:
                        ents.soft.update(e for e in eids if e in ents.E)
                        ents.touch()
                    else:
                        ents.erase_edges(eids)
                for gid in groups:
                    self.doc.remove_group(gid)
                self.doc.remove_guides(guides)
            self.commit(run)
        self.marked, self.groups, self.guides = [], [], []
        self.view.hover = None


class PaintTool(Tool):
    name = "paint"
    status = "Selecteer een object om te verven."
    measure_label = ""
    cursor_icon = "paint"

    def mouse_move(self, e, x, y):
        pick = self.engine.pick_entity(x, y)
        if pick[0] == "face":
            self.view.hover = ("face", pick[1], pick[2])
        elif pick[0] == "group":
            self.view.hover = ("group", pick[1])
        else:
            self.view.hover = None

    def is_locked(self, gid):
        g = self.doc.group(gid)
        return g is not None and g.locked

    def mouse_press(self, e, x, y):
        pick = self.engine.pick_entity(x, y)
        item = pick_item(pick)
        if item is None or item[0] == "e":
            return
        if item[0] == "g" and self.is_locked(item[1]):
            self.warn_locked()      # a locked group cannot be painted
            return
        if e.modifiers() & Qt.AltModifier:
            if item[0] == "g":
                color = self.doc.group(item[1]).color
            else:
                color = self.doc.ents(item[1]).F[item[2]].color
            if color:
                self.view.window().select_color(color)
            return
        color = self.current_color()
        targets = [item]
        if item in self.doc.selection:
            targets = [it for it in self.doc.selection if it[0] in ("f", "g")
                       and not (it[0] == "g" and self.is_locked(it[1]))]
        if not targets:
            self.warn_locked()
            return

        def apply():
            for k, ctx, i in targets:
                if k == "g":
                    self.doc.group(i).color = color
                else:
                    self.doc.ents(ctx).set_face_color(i, color)
        self.commit(apply)


# ------------------------------------------------------------------ line

class LineTool(Tool):
    name = "line"
    status = "Selecteer beginpunt."
    measure_label = "Lengte"
    cursor_icon = "line"

    def status_text(self):
        if not self.pts:
            return "Selecteer beginpunt."
        return ("Selecteer eindpunt of typ een lengte. "
                "Shift = vergrendel richting; pijltoetsen = vergrendel as.")

    def reset(self):
        super().reset()
        self.pts = []

    def busy(self):
        return bool(self.pts)

    def mouse_move(self, e, x, y):
        anchor = self.pts[-1] if self.pts else None
        extra = np.array(self.pts) if self.pts else None
        self.inf = self.engine.infer(x, y, anchor=anchor, lock=self.lock_line(anchor), extra=extra)
        self.inf.point = self.snap_point(anchor, self.inf.point)
        if anchor is not None:
            self.set_vcb(fmt_len(np.linalg.norm(self.inf.point - anchor)))

    def mouse_press(self, e, x, y):
        self.mouse_move(e, x, y)
        self.add_point(self.inf.point)

    def add_point(self, p):
        p = np.asarray(p, dtype=float)
        if not self.pts:
            self.pts = [p]
            return
        a = self.pts[-1]
        if np.linalg.norm(p - a) < 1e-6:
            return
        ents = self.ents
        nf = len(ents.F)
        before = set(ents.F)
        self.commit(lambda: ents.add_line(a, p, hint=self.eye))
        self.unlock()
        made_face = any(f not in before for f in ents.F) and len(ents.F) >= nf
        if made_face or (len(self.pts) >= 2 and np.linalg.norm(p - self.pts[0]) < 1e-6):
            self.reset()  # SketchUp ends the line when it creates a face
            self.view.update_vcb()
            return
        self.pts.append(p)

    def key_press(self, e):
        return self.handle_lock_keys(e, self.pts[-1] if self.pts else None)

    def vcb_enter(self, text):
        if not self.pts or self.inf is None:
            raise ValueError("klik eerst een beginpunt")
        L = parse_length(text)
        anchor = self.pts[-1]
        d = self.lock_dir if self.lock_dir is not None else self.inf.point - anchor
        if np.linalg.norm(d) < 1e-9:
            raise ValueError("geen richting")
        self.add_point(anchor + normalize(d) * L)

    def draw(self, view):
        if self.pts and self.inf is not None:
            a = self.pts[-1]
            view.draw_lines([a, self.inf.point], self.guide_color(), 2.2, on_top=True)
            view.add_label((a + self.inf.point) / 2, fmt_len(np.linalg.norm(self.inf.point - a)))
        view.draw_inference(self.inf)


class FreehandTool(Tool):
    """Tekenen > Lijnen > Uit de vrije hand: sleep een lijn in het vlak onder de
    cursor; bij het loslaten wordt ze vereenvoudigd en als randen toegevoegd."""
    name = "freehand"
    status = "Sleep met de linkermuisknop om uit de vrije hand te tekenen."
    measure_label = "Afmetingen"
    cursor_icon = "freehand"

    TOL_PX = 3.0    # points that deviate less than this many pixels disappear
    MIN_PX = 4.0    # the shortest segment that survives, in pixels
    CLOSE_PX = 8.0  # begin and end within this many pixels: the shape closes

    def status_text(self):
        if self.drawing:
            return ("Sleep de lijn uit de vrije hand; eindig bij het beginpunt en de vorm "
                    "sluit zich tot een vlak.")
        return self.status

    def reset(self):
        super().reset()
        self.samples = []
        self.plane = None
        self.drawing = False

    def busy(self):
        return self.drawing

    def mouse_press(self, e, x, y):
        inf = self.engine.infer(x, y)
        self.plane = self.drawing_plane(inf, x, y)
        self.samples = [np.asarray(inf.point, dtype=float)]
        self.drawing = True
        self.inf = None
        self.view.hover = None
        self.set_vcb("")

    def mouse_move(self, e, x, y):
        if not self.drawing:
            self.inf = self.engine.infer(x, y)
            return
        o, d = self.view.cam.ray(x, y)
        p = self.plane.intersect_ray(o, d)
        if p is None:
            return
        if float(np.linalg.norm(p - self.samples[-1])) > 1e-9:
            self.samples.append(p)
        self.set_vcb(fmt_len(self.stroke_length()))

    def stroke_length(self):
        return float(sum(np.linalg.norm(b - a) for a, b in zip(self.samples[:-1], self.samples[1:])))

    def stroke_pixel_size(self):
        return self.view.cam.pixel_size(self.samples[len(self.samples) // 2])

    def points(self):
        """The stroke as it will be added: simplified, closed if it ends where it started."""
        if len(self.samples) < 2:
            return None
        ps = self.stroke_pixel_size()
        pts = simplify_polyline(self.samples, self.TOL_PX * ps, self.MIN_PX * ps)
        if len(pts) < 2:
            return None
        gap = float(np.linalg.norm(pts[-1] - pts[0]))
        if gap <= self.CLOSE_PX * ps:
            if gap > 1e-9:
                pts = np.concatenate([pts, pts[:1]])   # laatste punt = eerste punt
            if len(pts) < 4:
                return None                            # te weinig hoekpunten voor een vlak
        return pts

    def mouse_release(self, e, x, y):
        if not self.drawing:
            return
        pts = self.points()
        self.reset()
        self.view.update_vcb()
        if pts is None:
            return
        ents, hint = self.ents, self.eye

        def run():
            for a, b in zip(pts[:-1], pts[1:]):
                ents.add_line(a, b, hint=hint)
        self.commit(run)

    def draw(self, view):
        if self.drawing and len(self.samples) > 1:
            view.draw_strip(self.samples, (0.0, 0.0, 0.0, 1), 2, on_top=True)
            view.add_label(self.samples[-1], fmt_len(self.stroke_length()))
        view.draw_inference(self.inf)


# ------------------------------------------------------------------ 2D shapes

class ShapeTool(Tool):
    """Base for rectangle/circle/polygon: the first click fixes the plane."""
    cursor_icon = "rectangle"
    step1 = "Selecteer eerste hoek."
    step2 = "Selecteer tegenoverliggende hoek of typ waarden."

    def status_text(self):
        return self.step1 if self.p0 is None else self.step2

    def reset(self):
        super().reset()
        self.p0 = None
        self.plane = None
        self.dynamic_plane = False
        if not hasattr(self, "forced_axis"):
            self.forced_axis = None
        self.cur = None

    def busy(self):
        return self.p0 is not None

    def key_press(self, e):
        if e.key() in ARROWS and self.p0 is None:
            ax = ARROWS[e.key()]
            self.forced_axis = None if self.forced_axis == ax else ax
            names = ("rode", "groene", "blauwe")
            self.view.set_status(
                f"Tekenvlak loodrecht op de {names[ax]} as." if self.forced_axis is not None else self.status_text())
            return True
        return False

    def mouse_move(self, e, x, y):
        if self.p0 is None:
            self.inf = self.engine.infer(x, y)
            return
        if self.dynamic_plane:
            # started on a corner or edge: like SketchUp, use the face under the cursor
            # that contains the first point, otherwise a horizontal plane
            fu = self.face_under(x, y, self.p0)
            self.plane = Plane(self.p0, fu[2].n, fu[2].u) if fu else Plane(self.p0, (0, 0, 1))
        self.inf = self.engine.infer(x, y, anchor=self.p0, plane=self.plane)
        self.cur = self.plane.to_local(self.inf.point) - self.plane.to_local(self.p0)
        if self.free():
            self.cur = self.snap_cur(self.cur)
        self.update_value()

    def snap_cur(self, cur):
        return np.array([self.snap_len(cur[0]), self.snap_len(cur[1])])

    def mouse_press(self, e, x, y):
        if self.p0 is None:
            inf = self.engine.infer(x, y)
            self.plane = self.drawing_plane(inf, x, y, self.forced_axis)
            self.dynamic_plane = self.forced_axis is None and inf.kind in ("endpoint", "midpoint", "edge", "origin")
            self.p0 = inf.point
            self.cur = np.zeros(2)
            return
        self.mouse_move(e, x, y)
        self.finish()

    def mouse_release(self, e, x, y):
        if self.p0 is not None and self.cur is not None and np.linalg.norm(self.cur) > 0:
            s, _ = self.view.cam.project(self.p0)
            if np.hypot(s[0][0] - x, s[0][1] - y) > 8:
                self.finish()

    def loop2d(self):
        raise NotImplementedError

    def update_value(self):
        pass

    def finish(self):
        loop = self.loop2d()
        if loop is None:
            return
        plane, ents = self.plane, self.ents
        self.commit(lambda: ents.add_shape(plane, [loop], hint=self.eye))
        self.reset()
        self.view.update_vcb()

    def draw(self, view):
        if self.p0 is not None:
            loop = self.loop2d()
            if loop is not None:
                view.draw_loop(self.plane.to_world(loop), (0, 0, 0, 1), 2, on_top=True)
        view.draw_inference(self.inf)


class RectangleTool(ShapeTool):
    name = "rectangle"
    status = "Selecteer eerste hoek."
    measure_label = "Afmetingen"

    def loop2d(self):
        if self.cur is None:
            return None
        o = self.plane.to_local(self.p0)
        w, h = self.cur
        if abs(w) < 1e-6 or abs(h) < 1e-6:
            return None
        x0, x1 = sorted((o[0], o[0] + w))
        y0, y1 = sorted((o[1], o[1] + h))
        return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])

    def update_value(self):
        self.set_vcb(f"{abs(self.cur[0]):.2f}; {abs(self.cur[1]):.2f}")

    def vcb_enter(self, text):
        if self.p0 is None:
            raise ValueError("klik eerst de eerste hoek")
        vals = split_values(text, (2,))
        if len(vals) != 2:
            raise ValueError("geef breedte;hoogte")
        w, h = parse_length(vals[0]), parse_length(vals[1])
        sx = -1 if self.cur is not None and self.cur[0] < 0 else 1
        sy = -1 if self.cur is not None and self.cur[1] < 0 else 1
        self.cur = np.array([w * sx, h * sy])
        self.finish()

    def draw(self, view):
        super().draw(view)
        if self.p0 is not None and self.cur is not None:
            c = self.plane.to_world(self.plane.to_local(self.p0) + self.cur / 2)
            view.add_label(c, f"{abs(self.cur[0]):.2f} × {abs(self.cur[1]):.2f} mm")


class RotatedRectangleTool(Tool):
    """Tekenen > Vormen > Gedraaide rechthoek: de eerste twee hoeken bepalen de eerste
    zijde (ook schuin), de muis bepaalt daarna de breedte loodrecht op die zijde."""
    name = "rotrect"
    status = "Selecteer eerste hoek."
    measure_label = "Lengte"
    cursor_icon = "rotrect"

    def status_text(self):
        if self.p0 is None:
            return "Selecteer eerste hoek."
        if self.p1 is None:
            return "Selecteer de tweede hoek van de eerste zijde of typ een lengte."
        return "Selecteer de breedte of typ een waarde."

    def vcb_label(self):
        return "Lengte" if self.p1 is None else "Breedte"

    def reset(self):
        super().reset()
        self.p0 = self.p1 = None
        self.plane = None
        self.hover_plane = None
        self.cur = None
        self.width = 0.0
        if not hasattr(self, "forced_axis"):
            self.forced_axis = None

    def busy(self):
        return self.p0 is not None

    def key_press(self, e):
        if e.key() in ARROWS and self.p0 is None:
            ax = ARROWS[e.key()]
            self.forced_axis = None if self.forced_axis == ax else ax
            names = ("rode", "groene", "blauwe")
            self.view.set_status(
                f"Tekenvlak loodrecht op de {names[ax]} as." if self.forced_axis is not None else self.status_text())
            return True
        return False

    # -- the three steps -------------------------------------------------
    def length(self):
        return 0.0 if self.cur is None else float(np.linalg.norm(self.cur - self.p0))

    def width_at(self, p):
        """Signed distance from the first side to the cursor, in the drawing plane."""
        d = self.p1 - self.p0
        L = float(np.linalg.norm(d))
        if L < 1e-9:
            return 0.0
        w = np.cross(self.plane.n, d / L)
        s = float(np.dot(self.plane.project(p) - self.p0, w))
        return self.snap_len(abs(s)) * (1.0 if s >= 0 else -1.0) if self.free() else s

    def mouse_move(self, e, x, y):
        if self.p0 is None:
            self.inf = self.engine.infer(x, y)
            self.hover_plane = self.drawing_plane(self.inf, x, y, self.forced_axis)
            self.cur = None
            return
        if self.p1 is None:
            self.inf = self.engine.infer(x, y, anchor=self.p0, plane=self.plane)
            self.cur = self.snap_point(self.p0, self.inf.point)
            self.set_vcb(fmt_len(self.length()))
            return
        self.inf = self.engine.infer(x, y, anchor=self.p1, plane=self.plane)
        self.width = self.width_at(self.inf.point)
        self.set_vcb(fmt_len(abs(self.width)))

    def set_side(self):
        """Second click: the first side of the rectangle is now known."""
        if self.length() < 1e-6:
            return False
        self.p1 = self.cur
        self.width = 0.0
        self.view.set_status("Beweeg de breedte en klik, of typ een breedte.")
        return True

    def mouse_press(self, e, x, y):
        if self.p0 is None:
            inf = self.engine.infer(x, y)
            self.plane = self.drawing_plane(inf, x, y, self.forced_axis)
            self.p0 = np.asarray(inf.point, dtype=float)
            self.cur = None
            self.view.hover = None
            return
        self.mouse_move(e, x, y)
        if self.p1 is None:
            self.set_side()
            return
        self.finish()

    def mouse_release(self, e, x, y):
        """Sleep van de ene hoek naar de andere bepaalt die zijde ook."""
        if self.p0 is None:
            return
        anchor = self.p0 if self.p1 is None else self.p1
        s, _ = self.view.cam.project(anchor)
        if np.hypot(s[0][0] - x, s[0][1] - y) <= 8:
            return
        self.mouse_move(e, x, y)
        if self.p1 is None:
            self.set_side()
        else:
            self.finish()

    def loop(self):
        if self.p0 is None or self.p1 is None or abs(self.width) < 1e-6:
            return None
        return rotated_rect(self.p0, self.p1, self.width, self.plane)

    def finish(self):
        loop = self.loop()
        if loop is None or len(loop) < 4:
            return
        plane, ents = self.plane, self.ents
        loop2d = plane.to_local(loop)
        self.commit(lambda: ents.add_shape(plane, [loop2d], hint=self.eye))
        self.reset()
        self.view.update_vcb()

    def vcb_enter(self, text):
        if self.p0 is None:
            raise ValueError("klik eerst de eerste hoek")
        vals = split_values(text, (1, 2))
        L = parse_length(vals[0])
        if self.p1 is None:
            if len(vals) > 1:                      # lengte;hoek vanaf de u-as van het tekenvlak
                a = math.radians(parse_number(vals[1]))
                d = math.cos(a) * self.plane.u + math.sin(a) * self.plane.v
            else:
                d = None if self.cur is None else self.cur - self.p0
                if d is None or np.linalg.norm(d) < 1e-9:
                    d = self.plane.u
                d = normalize(d)
            self.p1 = self.p0 + d * L
            self.cur = self.p1
            self.width = 0.0
            self.view.set_status("Beweeg de breedte en klik, of typ een breedte.")
            return
        if len(vals) > 1:
            raise ValueError("typ nu alleen de breedte")
        self.width = abs(L) * (1.0 if self.width >= 0 else -1.0)
        self.finish()

    def draw(self, view):
        if self.p0 is not None and self.p1 is None and self.cur is not None:
            view.draw_lines([self.p0, self.cur], self.guide_color(), 2.2, on_top=True)
            view.add_label(0.5 * (self.p0 + self.cur), fmt_len(self.length()))
        elif self.p1 is not None:
            loop = self.loop()
            if loop is not None:
                view.draw_loop(loop, (0.0, 0.0, 0.0, 1), 2, on_top=True)
                view.add_label(0.5 * (loop[2] + loop[3]), fmt_len(abs(self.width)))
            else:
                view.draw_lines([self.p0, self.p1], (0.35, 0.35, 0.35, 1), 1.5, on_top=True)
        elif self.hover_plane is not None and self.inf is not None:
            r = view.cam.pixel_size(self.inf.point) * 22
            c = self.inf.point
            u, v = self.hover_plane.u, self.hover_plane.v
            view.draw_lines([c - u * r, c + u * r, c - v * r, c + v * r], (0.5, 0.5, 0.5, 1), 1,
                            stipple=True, on_top=True)
        view.draw_inference(self.inf)


class CircleTool(ShapeTool):
    name = "circle"
    cursor_icon = "circle"

    def snap_cur(self, cur):
        L = float(np.linalg.norm(cur))
        return cur if L < 1e-9 else cur / L * self.snap_len(L)
    status = "Selecteer middelpunt."
    step1 = "Selecteer middelpunt."
    step2 = "Selecteer straal of typ een waarde."
    measure_label = "Straal"
    default_segments = None

    def vcb_label(self):
        """SketchUp: before the centre is picked the box holds the number of sides."""
        return "Zijden" if self.p0 is None else "Straal"

    def reset(self):
        super().reset()
        if not hasattr(self, "segments"):
            self.segments = self.default_segments or self.settings.circle_segments
        self.radius = None
        self.angle = 0.0

    def mouse_move(self, e, x, y):
        super().mouse_move(e, x, y)
        if self.p0 is not None and self.cur is not None:
            self.radius = float(np.linalg.norm(self.cur))
            self.angle = math.atan2(self.cur[1], self.cur[0])

    def loop2d(self):
        if not self.radius or self.radius < 1e-6:
            return None
        c = self.plane.to_local(self.p0)
        start = self.angle if self.name == "polygon" else 0.0
        return circle_points(self.radius, self.segments, start) + c

    def update_value(self):
        if self.cur is not None:
            self.set_vcb(f"{np.linalg.norm(self.cur):.2f}")

    def vcb_enter(self, text):
        t = text.strip().lower()
        if t.endswith("s"):
            n = int(parse_number(t[:-1]))
            if n < 3:
                raise ValueError("minimaal 3 segmenten")
            self.segments = n
            self.view.set_status(f"{n} segmenten.")
            return
        if self.p0 is None:
            raise ValueError("klik eerst het middelpunt")
        self.radius = abs(parse_length(t))
        self.finish()

    def draw(self, view):
        super().draw(view)
        if self.p0 is not None and self.radius and self.inf is not None:
            view.draw_lines([self.p0, self.inf.point], (0.3, 0.3, 0.3, 1), 1, stipple=True, on_top=True)
            view.add_label(self.inf.point, f"r = {fmt_len(self.radius)}")


class PolygonTool(CircleTool):
    name = "polygon"
    cursor_icon = "polygon"
    default_segments = 6


# ------------------------------------------------------------------ arcs

class ArcBase(Tool):
    """Shared behaviour of the four arc tools: the first click fixes the drawing
    plane and the arc is one undo step of edges, like the Line tool."""
    cursor_icon = "arc"
    default_segments = 12
    measure_label = "Hoek"

    def reset(self):
        super().reset()
        if not hasattr(self, "segments"):
            self.segments = self.default_segments
        self.plane = None
        self.forced_axis = None

    def key_press(self, e):
        if e.key() in ARROWS and not self.busy():
            ax = ARROWS[e.key()]
            self.forced_axis = None if self.forced_axis == ax else ax
            names = ("rode", "groene", "blauwe")
            self.view.set_status(
                f"Tekenvlak loodrecht op de {names[ax]} as." if self.forced_axis is not None else self.status_text())
            return True
        return False

    def segments_input(self, text):
        """Handle "12s" in the measurements box; True when that was the input."""
        t = text.strip().lower()
        if not t.endswith("s"):
            return False
        n = int(parse_number(t[:-1]))
        if n < 2:
            raise ValueError("minimaal 2 segmenten")
        self.segments = n
        self.view.set_status(f"{n} segmenten.")
        return True

    def commit_edges(self, pairs):
        ents, hint = self.ents, self.eye
        pairs = list(pairs)

        def run():
            for a, b in pairs:
                ents.add_line(a, b, hint=hint)
        self.commit(run)

    def draw_arc(self, view, pts):
        if pts is not None and len(pts) > 1:
            view.draw_strip(pts, (0.0, 0.0, 0.0, 1), 2, on_top=True)
        view.draw_inference(self.inf)


class Arc2Tool(ArcBase):
    name = "arc2"
    cursor_icon = "arc2"
    status = "Selecteer beginpunt."
    measure_label = "Lengte"

    def status_text(self):
        if self.p0 is None:
            return "Selecteer beginpunt."
        if self.p1 is None:
            return "Selecteer eindpunt of typ een lengte."
        return "Selecteer een uitstulping of typ een hoogte."

    def vcb_label(self):
        return "Lengte" if self.p1 is None else "Uitstulping"

    def reset(self):
        super().reset()
        self.p0 = self.p1 = self.end = self.apex = None
        self.height = 0.0

    def busy(self):
        return self.p0 is not None

    def mouse_move(self, e, x, y):
        if self.p0 is None:
            self.inf = self.engine.infer(x, y)
            self.end = None
            return
        if self.p1 is None:
            self.inf = self.engine.infer(x, y, anchor=self.p0, plane=self.plane)
            self.inf.point = self.snap_point(self.p0, self.inf.point)
            self.end = self.inf.point
            self.set_vcb(fmt_len(np.linalg.norm(self.end - self.p0)))
            return
        self.inf = self.engine.infer(x, y, anchor=0.5 * (self.p0 + self.p1), plane=self.plane)
        self.apex = self.apex_at(self.inf.point)
        self.set_vcb(fmt_len(abs(self.height)))

    def apex_at(self, p):
        """Top of the bulge for the cursor point (length snapped, half circle at 50%)."""
        d = self.p1 - self.p0
        L = float(np.linalg.norm(d))
        mid = 0.5 * (self.p0 + self.p1)
        if L < 1e-9:
            self.height = 0.0
            return mid
        w = np.cross(self.plane.n, d / L)
        h = float(np.dot(p - mid, w))
        if self.free():
            h = self.snap_len(abs(h)) * (1.0 if h >= 0 else -1.0)
        if abs(abs(h) - 0.5 * L) <= 0.05 * L:
            h = math.copysign(0.5 * L, h)  # a full half circle, like SketchUp
        self.height = h
        return mid + w * h

    def points(self):
        if self.p0 is None or self.p1 is None or self.apex is None:
            return None
        pts = arc_from_bulge(self.p0, self.p1, self.apex, self.plane, self.segments)
        return pts if len(pts) > 2 else None

    def mouse_press(self, e, x, y):
        if self.p0 is None:
            inf = self.engine.infer(x, y)
            self.plane = self.drawing_plane(inf, x, y, self.forced_axis)
            self.p0 = inf.point
            return
        if self.p1 is None:
            self.mouse_move(e, x, y)
            if self.end is None or np.linalg.norm(self.end - self.p0) < 1e-6:
                return
            self.p1 = self.end
            self.view.set_status(self.status_text())
            return
        self.mouse_move(e, x, y)
        self.finish()

    def mouse_release(self, e, x, y):
        """Sleep van het eerste naar het tweede punt bepaalt ook de koorde."""
        if self.p0 is not None and self.p1 is None and self.end is not None:
            s, _ = self.view.cam.project(self.p0)
            if np.hypot(s[0][0] - x, s[0][1] - y) > 8:
                self.p1 = self.end

    def finish(self):
        pts = self.points()
        if pts is None:
            return
        self.commit_edges(zip(pts[:-1], pts[1:]))
        self.reset()
        self.view.update_vcb()

    def vcb_enter(self, text):
        if self.segments_input(text):
            return
        if self.p0 is None:
            raise ValueError("klik eerst het beginpunt")
        if self.p1 is None:
            d = None if self.end is None else self.end - self.p0
            if d is None or np.linalg.norm(d) < 1e-9:
                raise ValueError("beweeg eerst naar het eindpunt")
            self.p1 = self.p0 + normalize(d) * abs(parse_length(text))
            self.view.set_status(self.status_text())
            return
        w = np.cross(self.plane.n, normalize(self.p1 - self.p0))
        h = abs(parse_length(text)) * (1.0 if self.height >= 0 else -1.0)
        self.apex = self.apex_at(0.5 * (self.p0 + self.p1) + w * h)
        self.finish()

    def draw(self, view):
        if self.p0 is not None and self.p1 is None and self.end is not None:
            view.draw_lines([self.p0, self.end], self.guide_color(), 2.2, on_top=True)
            view.add_label(0.5 * (self.p0 + self.end), fmt_len(np.linalg.norm(self.end - self.p0)))
        elif self.p1 is not None:
            view.draw_lines([self.p0, self.p1], (0.35, 0.35, 0.35, 1), 1, stipple=True, on_top=True)
            self.draw_arc(view, self.points())
            if self.apex is not None:
                view.add_label(self.apex, fmt_len(abs(self.height)))
            return
        view.draw_inference(self.inf)


class ArcTool(ArcBase):
    name = "arc"
    status = "Selecteer middelpunt."
    measure_label = "Straal"
    pie = False

    def status_text(self):
        if self.center is None:
            return "Selecteer middelpunt."
        if self.start is None:
            return "Selecteer beginpunt van de boog of typ een straal."
        return "Draai de boog of typ een hoek in graden."

    def vcb_label(self):
        return "Straal" if self.start is None else "Hoek"

    def reset(self):
        super().reset()
        self.center = self.start = self.cur = None
        self.angle = 0.0

    def busy(self):
        return self.center is not None

    @property
    def radius(self):
        return None if self.start is None else float(np.linalg.norm(self.start - self.center))

    def mouse_move(self, e, x, y):
        if self.center is None:
            self.inf = self.engine.infer(x, y)
            self.cur = None
            return
        self.inf = self.engine.infer(x, y, anchor=self.center, plane=self.plane)
        if self.start is None:
            self.inf.point = self.snap_point(self.center, self.inf.point)
            self.cur = self.inf.point
            self.set_vcb(fmt_len(np.linalg.norm(self.cur - self.center)))
            return
        self.cur = self.inf.point
        self.angle = self.angle_at(self.cur)
        self.set_vcb(fmt_angle(self.angle))

    def angle_at(self, p):
        v = p - self.center
        if np.linalg.norm(v) < 1e-9:
            return self.angle
        ref = normalize(self.start - self.center)
        deg = math.degrees(math.atan2(float(np.dot(np.cross(ref, v), self.plane.n)), float(ref @ v)))
        snap = round(deg / 15.0) * 15.0
        return snap if abs(deg - snap) < 2.5 else deg  # 15° steps, like Roteren

    def mouse_press(self, e, x, y):
        if self.center is None:
            inf = self.engine.infer(x, y)
            self.plane = self.drawing_plane(inf, x, y, self.forced_axis)
            self.center = inf.point
            return
        if self.start is None:
            self.mouse_move(e, x, y)
            if self.cur is None or np.linalg.norm(self.cur - self.center) < 1e-6:
                return
            self.start = self.cur
            self.view.set_status(self.status_text())
            return
        self.mouse_move(e, x, y)
        self.finish()

    def mouse_release(self, e, x, y):
        """Sleep vanaf het middelpunt om direct de straal te kiezen."""
        if self.center is not None and self.start is None and self.cur is not None:
            s, _ = self.view.cam.project(self.center)
            if np.hypot(s[0][0] - x, s[0][1] - y) > 8:
                self.start = self.cur

    def points(self):
        if self.center is None or self.start is None or abs(self.angle) < 1e-9:
            return None
        return arc_from_center(self.center, self.start, math.radians(self.angle),
                               self.plane, self.segments)

    def finish(self):
        pts = self.points()
        if pts is None:
            return
        pairs = list(zip(pts[:-1], pts[1:]))
        if self.pie:  # the two radii close the shape, so it becomes a face
            pairs += [(pts[0], self.center), (self.center, pts[-1])]
        self.commit_edges(pairs)
        self.reset()
        self.view.update_vcb()

    def vcb_enter(self, text):
        if self.segments_input(text):
            return
        if self.center is None:
            raise ValueError("klik eerst het middelpunt")
        if self.start is None:
            d = None if self.cur is None else self.cur - self.center
            if d is None or np.linalg.norm(d) < 1e-9:
                d = self.plane.u
            self.start = self.center + normalize(d) * abs(parse_length(text))
            self.view.set_status(self.status_text())
            return
        self.angle = parse_number(text.replace("°", ""))
        self.finish()

    def draw(self, view):
        if self.center is not None:
            if self.start is None:
                if self.cur is not None:
                    view.draw_lines([self.center, self.cur], (0.3, 0.3, 0.3, 1), 1, stipple=True,
                                    on_top=True)
                    view.add_label(self.cur, f"r = {fmt_len(np.linalg.norm(self.cur - self.center))}")
            else:
                pts = self.points()
                view.draw_lines([self.center, self.start], (0.3, 0.3, 0.3, 1), 1, stipple=True,
                                on_top=True)
                if pts is not None and self.pie:
                    view.draw_lines([self.center, pts[0], self.center, pts[-1]], (0.0, 0.0, 0.0, 1), 2,
                                    on_top=True)
                self.draw_arc(view, pts)
                if self.cur is not None:
                    view.add_label(self.cur, fmt_angle(self.angle))
                return
        view.draw_inference(self.inf)


class PieTool(ArcTool):
    name = "pie"
    cursor_icon = "pie"
    status = "Selecteer middelpunt."
    pie = True

    def status_text(self):
        if self.center is not None and self.start is not None:
            return "Draai de taartpunt of typ een hoek in graden."
        return super().status_text()


class Arc3Tool(ArcBase):
    name = "arc3"
    cursor_icon = "arc3"
    status = "Selecteer beginpunt."
    measure_label = "Segmenten"

    def status_text(self):
        if not self.pts:
            return "Selecteer beginpunt."
        if len(self.pts) == 1:
            return "Selecteer een punt op de boog."
        return "Selecteer het eindpunt van de boog."

    def reset(self):
        super().reset()
        self.pts = []
        self.cur = None

    def busy(self):
        return bool(self.pts)

    def mouse_move(self, e, x, y):
        if not self.pts:
            self.inf = self.engine.infer(x, y)
            self.cur = None
            return
        self.inf = self.engine.infer(x, y, anchor=self.pts[-1], plane=self.plane)
        self.cur = self.inf.point
        self.set_vcb(f"{self.segments}s")

    def mouse_press(self, e, x, y):
        if not self.pts:
            inf = self.engine.infer(x, y)
            self.plane = self.drawing_plane(inf, x, y, self.forced_axis)
            self.pts = [inf.point]
            return
        self.mouse_move(e, x, y)
        if np.linalg.norm(self.cur - self.pts[-1]) < 1e-6:
            return
        self.pts.append(self.cur)
        if len(self.pts) == 3:
            self.finish()

    def points(self):
        if len(self.pts) < 3:
            return None
        pts = arc_through_3(self.pts[0], self.pts[1], self.pts[2], self.plane, self.segments)
        return pts if len(pts) > 2 else None

    def finish(self):
        pts = self.points()
        if pts is None:
            return
        self.commit_edges(zip(pts[:-1], pts[1:]))
        self.reset()
        self.view.update_vcb()

    def vcb_enter(self, text):
        if self.segments_input(text):
            return
        raise ValueError("typ het aantal segmenten, bv. 12s")

    def draw(self, view):
        if self.pts:
            segs = []
            for a, b in zip(self.pts[:-1], self.pts[1:]):
                segs += [a, b]
            if self.cur is not None:
                segs += [self.pts[-1], self.cur]
            if segs:
                view.draw_lines(segs, (0.35, 0.35, 0.35, 1), 1, stipple=True, on_top=True)
        view.draw_inference(self.inf)


class TextTool(Tool):
    """3D text: places the outlines of a text as faces that you push/pull."""
    name = "text"
    status = "Klik op een vlak of de grond om de tekst te plaatsen. Daarna duwen/trekken (Alt = inleg)."
    measure_label = ""
    cursor_icon = "text"

    def reset(self):
        super().reset()
        if not hasattr(self, "loops"):
            self.loops = None
        self.preview = None

    def set_text(self, text, family, height, bold=False):
        font = QFont(family)
        font.setPixelSize(200)
        font.setBold(bold)
        path = QPainterPath()
        path.addText(0, 0, font, text)
        cap = QFontMetricsF(font).capHeight() or 140.0
        s = height / cap
        loops = []
        for poly in path.toSubpathPolygons():
            pts = np.array([[p.x() * s, -p.y() * s] for p in poly])
            if len(pts) > 1 and np.allclose(pts[0], pts[-1]):
                pts = pts[:-1]
            if len(pts) >= 3:
                loops.append(pts)
        if not loops:
            self.loops = None
            return
        cs = mf.CrossSection(loops, mf.FillRule.NonZero)
        loops = [np.asarray(p) for p in cs.to_polygons()]
        allp = np.concatenate(loops)
        c = np.array([(allp[:, 0].min() + allp[:, 0].max()) / 2, allp[:, 1].min()])
        self.loops = [L - c for L in loops]

    def mouse_move(self, e, x, y):
        self.inf = self.engine.infer(x, y)
        self.preview = None
        if self.loops:
            plane = self.drawing_plane(self.inf, x, y)
            self.preview = [plane.to_world(L + plane.to_local(self.inf.point)) for L in self.loops]

    def mouse_press(self, e, x, y):
        if not self.loops:
            self.view.window().ask_text()
            return
        inf = self.engine.infer(x, y)
        plane = self.drawing_plane(inf, x, y)
        off = plane.to_local(inf.point)
        loops = [L + off for L in self.loops]
        ents = self.ents
        self.commit(lambda: ents.add_shape(plane, loops, hint=self.eye))
        self.view.window().set_tool("pushpull")

    def draw(self, view):
        if self.preview:
            for L in self.preview:
                view.draw_loop(L, (0.1, 0.1, 0.1, 1), 1.5, on_top=True)
        view.draw_inference(self.inf)


class VectorPlaceTool(Tool):
    """Import: place the contours of an SVG/DXF so they follow the cursor.

    Zelfde opbouw als `TextTool`: de vormen volgen de cursor op het vlak of de
    grond eronder, klik plaatst als één undo-stap, Esc annuleert. In het meetvak
    (label "Hoek") kan een draaihoek getypt worden. Zonder extrusie gaat het daarna
    naar Duwen/trekken (zoals tekst); met extrusie wordt de nieuwe groep geselecteerd.
    """
    name = "vectorplace"
    status = ("Klik op een vlak of de grond om de tekening te plaatsen. "
              "Typ een hoek in graden om te draaien; Esc annuleert.")
    measure_label = "Hoek"
    cursor_icon = "text"

    def reset(self):
        super().reset()
        self.preview = None
        self.angle = 0.0
        self._last_xy = None
        if not hasattr(self, "regios"):
            self.regios = []
        if not hasattr(self, "open_paden"):
            self.open_paden = []
        if not hasattr(self, "options"):
            self.options = {}

    def set_drawing(self, regios, open_paden, options=None):
        self.regios = list(regios or [])
        self.open_paden = [np.asarray(p, dtype=float) for p in (open_paden or [])]
        self.options = dict(options or {})
        self.angle = 0.0

    def has_content(self):
        return bool(self.regios or self.open_paden)

    def _rotated(self):
        a = math.radians(self.angle)
        R = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
        regios = [[loop @ R.T for loop in shape] for shape in self.regios]
        open_paden = [p @ R.T for p in self.open_paden]
        return regios, open_paden

    def mouse_move(self, e, x, y):
        self._last_xy = (x, y)
        self.inf = self.engine.infer(x, y)
        self.preview = None
        if self.has_content():
            plane = self.drawing_plane(self.inf, x, y)
            off = plane.to_local(self.inf.point)
            regios, open_paden = self._rotated()
            lussen = [plane.to_world(loop + off) for shape in regios for loop in shape]
            openen = [plane.to_world(p + off) for p in open_paden if len(p) >= 2]
            self.preview = (lussen, openen)

    def mouse_press(self, e, x, y):
        if not self.has_content():
            self.view.window().set_tool("select")
            return
        inf = self.engine.infer(x, y)
        self.place(self.drawing_plane(inf, x, y), inf.point)

    def place(self, plane, point):
        regios, open_paden = self._rotated()
        opts = self.options
        extrude = bool(opts.get("extrude"))
        open_edges = bool(opts.get("open_edges", True))
        offset = plane.to_local(point)
        color = opts.get("color") or self.current_color()
        name = opts.get("name") or "Vector"
        height = float(opts.get("height", 3.0))
        ents = self.ents
        made = {}
        ok = {"done": False}

        def fn():
            if extrude:
                made["g"] = vector_import.place_extruded(self.doc, plane, regios, offset,
                                                          height, color, name)
                if open_edges and open_paden:
                    vector_import.place_faces(ents, plane, [], open_paden, offset)
            else:
                vector_import.place_faces(ents, plane, regios,
                                          open_paden if open_edges else [], offset)
            ok["done"] = True
        self.commit(fn)
        if not ok["done"]:
            return
        if extrude:
            g = made.get("g")
            self.doc.active = ROOT
            if g is not None:
                self.doc.set_selection([("g", g.id, g.id)])
            self.doc.notify()
            self.reset()
            self.view.window().set_tool("select")
        else:
            self.view.window().set_tool("pushpull")

    def vcb_enter(self, text):
        self.angle = parse_number(text.replace("°", ""))
        self.set_vcb(fmt_angle(self.angle))
        if self._last_xy is not None:
            self.mouse_move(None, self._last_xy[0], self._last_xy[1])
            self.view.update()

    def draw(self, view):
        if self.preview:
            lussen, openen = self.preview
            for L in lussen:
                view.draw_loop(L, (0.1, 0.1, 0.1, 1), 1.5, on_top=True)
            for p in openen:
                view.draw_strip(p, (0.1, 0.1, 0.1, 1), 1.5, on_top=True)
        view.draw_inference(self.inf)


# ------------------------------------------------------------------ push / pull

class PushPullTool(Tool):
    name = "pushpull"
    status = "Selecteer een vlak om te duwen of te trekken."
    measure_label = "Afstand"
    cursor_icon = "pushpull"
    last_distance = None

    def status_text(self):
        if self.target is None:
            return "Selecteer een vlak om te duwen of te trekken."
        return "Beweeg om te duwen/trekken of typ een afstand. Ctrl = nieuw vlak."

    def reset(self):
        super().reset()
        self.target = None
        self.dist = 0.0
        self.preview = None
        self._pkey = None
        self.press_pos = None
        self.last_commit = None
        self.mods = Qt.NoModifier
        self.sign = 1.0

    def busy(self):
        return self.target is not None

    def _set_target(self, ctx, fid, p):
        """Pick the face; `sign` turns the distance along the normal into the shown distance.

        A loose face drawn on the ground points down (as in SketchUp), so along its normal
        pulling up would be negative; like SketchUp, up is shown and typed as positive."""
        ents = self.doc.ents(ctx)
        n = ents.F[fid].normal.copy()
        loose = all(len(ents.e2f.get(e, ())) <= 1 for e in ents.face_edges(ents.F[fid]))
        self.sign = -1.0 if loose and n[2] < -1e-6 else 1.0
        self.target = (ctx, fid, p, n)

    def _face_at(self, x, y):
        pick = self.engine.pick_entity(x, y)
        if pick[0] == "face":
            return pick[1], pick[2], pick[3]
        if pick[0] == "edge":
            o, d = self.view.cam.ray(x, y)
            fh = self.engine.pick_face(o, d, only={self.doc.active})
            if fh:
                return fh[1], fh[2], fh[3]
        return None

    def mouse_move(self, e, x, y):
        self.mods = e.modifiers()
        if self.target is None:
            f = self._face_at(x, y)
            self.view.hover = ("face", f[0], f[1]) if f else None
            return
        ctx, fid, p0, n = self.target
        self.inf = self.engine.infer(x, y)
        if self.inf.kind in ("endpoint", "midpoint", "edge", "origin"):
            self.dist = float(np.dot(self.inf.point - p0, n))
        else:
            self.inf = None
            o, d = self.view.cam.ray(x, y)
            s = closest_on_line_to_ray(p0, n, o, d)
            self.dist = 0.0 if s is None else self.snap_len(s)
        self.set_vcb(f"{self.sign * self.dist:.2f}")

    def mouse_press(self, e, x, y):
        self.mods = e.modifiers()
        if self.target is None:
            f = self._face_at(x, y)
            if f is None:
                return
            self._set_target(*f)
            self.dist = 0.0
            self.press_pos = (x, y)
            self.view.hover = ("face", f[0], f[1])
            return
        self.apply(self.dist)

    def mouse_release(self, e, x, y):
        if self.target is not None and self.press_pos is not None:
            if np.hypot(x - self.press_pos[0], y - self.press_pos[1]) > 6 and abs(self.dist) > 1e-6:
                self.apply(self.dist)
        self.press_pos = None

    def double_click(self, e, x, y):
        if PushPullTool.last_distance is None:
            return
        if self.target is None:
            f = self._face_at(x, y)
            if f is None:
                return
            self._set_target(*f)
        self.apply(self.sign * PushPullTool.last_distance)

    def vcb_enter(self, text):
        v = parse_length(text)
        if self.target is not None:
            # like SketchUp the typed distance follows the current mouse direction
            self.apply(self.sign * (-v if self.sign * self.dist < 0 else v))
        elif self.last_commit is not None:
            tgt, d0, sign = self.last_commit
            self.doc.undo()
            self.doc.redo_stack.clear()
            self.target, self.sign = tgt, sign
            self.apply(sign * (-v if sign * d0 < 0 else v))
        else:
            raise ValueError("klik eerst op een vlak")

    def apply(self, dist):
        if self.target is None or abs(dist) < 1e-6:
            return
        ctx, fid, p0, n = self.target
        ents = self.doc.ents(ctx)
        if ents is None or fid not in ents.F:
            self.reset()
            return
        mods = self.mods
        alt = bool(mods & Qt.AltModifier)
        ctrl = bool(mods & Qt.ControlModifier)
        color = self.current_color()

        def run():
            if alt:
                ops.inlay(self.doc, ents, fid, dist, color)
            else:
                ents.pushpull(fid, dist, keep=ctrl)
        tgt, sign = self.target, self.sign
        self.commit(run)
        PushPullTool.last_distance = sign * dist
        self.reset()
        self.last_commit = (tgt, dist, sign)
        self.set_vcb(f"{sign * dist:.2f}")
        self.view.hover = None

    def draw(self, view):
        if self.target is None:
            return
        ctx, fid, p0, n = self.target
        ents = self.doc.ents(ctx)
        if ents is None or fid not in ents.F:
            return
        key = (round(self.dist, 6), fid, ents.version)
        if key != self._pkey:
            self._pkey = key
            plane = ents.face_plane(fid)
            polys = [plane.to_local(L) for L in ents.face_points(fid)]
            m = extrude_cs(mf.CrossSection(polys, mf.FillRule.EvenOdd), plane, 0.0, self.dist)
            self.preview = MeshData(m) if not m.is_empty() else None
        if self.preview is not None:
            col = (0.95, 0.3, 0.25) if self.sign * self.dist < 0 else (0.35, 0.6, 1.0)
            if self.mods & Qt.AltModifier:
                col = (0.95, 0.7, 0.15)
            view.draw_solid_preview(self.preview, col)
        p1 = p0 + n * self.dist
        view.draw_lines([p0, p1], (0.3, 0.3, 0.3, 1), 1, stipple=True, on_top=True)
        label = fmt_len(self.sign * self.dist)
        if self.mods & Qt.AltModifier:
            label += "  (apart onderdeel)"
        view.add_label(p1, label)
        view.draw_inference(self.inf)


# ------------------------------------------------------------------ offset

class OffsetTool(Tool):
    name = "offset"
    status = "Selecteer een vlak om een verschoven kopie van de rand te tekenen."
    measure_label = "Afstand"
    cursor_icon = "offset"

    def status_text(self):
        if self.target is None:
            return "Selecteer een vlak om te verschuiven."
        return "Beweeg om te verschuiven of typ een afstand."

    def reset(self):
        super().reset()
        self.target = None
        self.off = 0.0

    def busy(self):
        return self.target is not None

    def _signed(self, local):
        best = np.inf
        for P in self.target["polys"]:
            Q = np.roll(P, -1, axis=0)
            ab = Q - P
            t = np.clip(np.einsum("ij,ij->i", local - P, ab) / np.maximum(np.einsum("ij,ij->i", ab, ab), 1e-18), 0, 1)
            best = min(best, float(np.min(np.linalg.norm(P + ab * t[:, None] - local, axis=1))))
        return -best if point_in_polys(self.target["polys"], local) else best

    def mouse_move(self, e, x, y):
        if self.target is None:
            f = PushPullTool._face_at(self, x, y)
            self.view.hover = ("face", f[0], f[1]) if f else None
            return
        o, d = self.view.cam.ray(x, y)
        p = self.target["plane"].intersect_ray(o, d, allow_behind=True)
        if p is None:
            return
        self.off = self._signed(self.target["plane"].to_local(p))
        self.set_vcb(f"{abs(self.off):.2f}")

    def mouse_press(self, e, x, y):
        if self.target is None:
            f = PushPullTool._face_at(self, x, y)
            if f is None:
                return
            ents = self.doc.ents(f[0])
            plane = ents.face_plane(f[1])
            polys = [plane.to_local(L) for L in ents.face_points(f[1])]
            self.target = {"ctx": f[0], "plane": plane, "polys": polys,
                           "cs": mf.CrossSection(polys, mf.FillRule.EvenOdd)}
            self.off = 0.0
            return
        self.apply(self.off)

    def vcb_enter(self, text):
        if self.target is None:
            raise ValueError("klik eerst op een vlak")
        v = abs(parse_length(text))
        self.apply(-v if self.off <= 0 else v)

    def shape(self, off):
        return [np.asarray(p) for p in self.target["cs"].offset(off, mf.JoinType.Miter, 2.0).to_polygons()]

    def apply(self, off):
        if abs(off) < 1e-6:
            return
        tgt = self.target
        loops = self.shape(off)
        ents = self.doc.ents(tgt["ctx"])
        self.commit(lambda: ents.add_shape(tgt["plane"], loops, hint=self.eye))
        self.reset()
        self.view.hover = None

    def draw(self, view):
        if self.target is None or abs(self.off) < 1e-6:
            return
        for P in self.shape(self.off):
            view.draw_loop(self.target["plane"].to_world(P), (0, 0, 0, 1), 2, on_top=True)


# ------------------------------------------------------------------ follow me

def edge_chain(ents, eids):
    """The edges ordered as one chain: (points, closed, eids), or None when they are
    not a single connected, unbranched run."""
    eids = [e for e in eids if e in ents.E and ents.E[e][0] != ents.E[e][1]]
    if not eids:
        return None
    adj = {}
    for e in eids:
        a, b = ents.E[e]
        adj.setdefault(a, []).append((e, b))
        adj.setdefault(b, []).append((e, a))
    if any(len(v) > 2 for v in adj.values()):
        return None
    ends = [v for v, nb in adj.items() if len(nb) == 1]
    if len(ends) > 2:
        return None
    start = ends[0] if ends else next(iter(adj))
    pts, used, v = [ents.V[start].copy()], set(), start
    while True:
        step = [nb for nb in adj[v] if nb[0] not in used]
        if not step:
            break
        e, w = step[0]
        used.add(e)
        v = w
        pts.append(ents.V[v].copy())
        if v == start:
            break
    if len(used) != len(eids):
        return None
    return np.array(pts, dtype=float), not ends, used


def follow_path(ents, fids, eids):
    """The path to sweep along: the selected edges, or the outer edge of a selected face."""
    fids = [f for f in fids if f in ents.F]
    eids = [e for e in eids if e in ents.E]
    if fids:
        face = ents.F[fids[0]]
        loop = list(face.loops[0])
        if len(loop) < 3:
            return None
        ids = set()
        for i in range(len(loop)):
            e = ents.edge_between(loop[i], loop[(i + 1) % len(loop)])
            if e is not None:
                ids.add(e)
        return np.array([ents.V[v] for v in loop], dtype=float), True, ids
    return edge_chain(ents, eids)


def sweep_solid(profile, pts, closed):
    """follow_me along the path, starting where the profile is.

    A closed path is swept in the direction that makes the largest solid, which is
    the outside of the loop when the profile stands beside it."""
    pts = np.asarray(pts, dtype=float)
    loops = [np.asarray(L, dtype=float).reshape(-1, 3) for L in profile]
    if len(pts) > 1:
        c = np.concatenate(loops).mean(axis=0)
        if np.linalg.norm(pts[-1] - c) < np.linalg.norm(pts[0] - c):
            pts = pts[::-1]
    best = ops.follow_me(profile, pts, closed)
    if not closed or len(pts) < 3:
        return best, pts
    back = np.roll(pts[::-1], 1, axis=0)      # the same loop, other way round
    other = ops.follow_me(profile, back, closed)
    if other.status() == mf.Error.NoError and (best.status() != mf.Error.NoError or
                                               other.volume() > best.volume()):
        return other, back
    return best, pts


class FollowMeTool(Tool):
    name = "followme"
    status = ("Klik op het profielvlak dat langs het pad geveegd wordt. Selecteer daarvoor de "
              "randen van het pad (of een vlak; dan is zijn buitenrand het pad).")
    measure_label = ""
    cursor_icon = "followme"

    def reset(self):
        super().reset()
        self.preview = None
        self.path = None
        self._pkey = None

    def _profile_at(self, x, y):
        return PushPullTool._face_at(self, x, y)

    def _sweep(self, fid):
        """(manifold, MeshData) of the sweep of fid along the selected path, or (None, None)."""
        path = follow_path(self.ents, *self.doc.selected_geometry())
        self.path = path
        if path is None or fid not in self.ents.F:
            return None, None
        pts, closed, _ = path
        try:
            man, _ = sweep_solid(self.ents.face_points(fid), pts, closed)
        except Exception:
            return None, None
        if man.is_empty() or man.status() != mf.Error.NoError:
            return None, None
        return man, MeshData(man)

    def mouse_move(self, e, x, y):
        f = self._profile_at(x, y)
        self.view.hover = ("face", f[0], f[1]) if f else None
        if f is None or f[0] != self.doc.active:
            self.preview, self._pkey = None, None
            return
        key = (f[1], self.ents.version, tuple(sorted(self.doc.selected_vertices())))
        if key != self._pkey:
            self._pkey = key
            _, self.preview = self._sweep(f[1])
        self.inf = self.engine.infer(x, y)

    def mouse_press(self, e, x, y):
        f = self._profile_at(x, y)
        if f is None:
            return
        if f[0] != self.doc.active:
            self.view.set_status("Volg mij: het profiel moet in de huidige context liggen.")
            return
        self.apply(f[1])

    def apply(self, fid):
        path = follow_path(self.ents, *self.doc.selected_geometry())
        self.path = path
        if path is None:
            self.view.set_status("Volg mij: de selectie is geen aaneengesloten pad van randen.")
            return
        _, _, eids = path
        man, _ = self._sweep(fid)
        if man is None:
            self.view.set_status("Volg mij gaf geen gesloten volume")
            return
        result = Entities.from_manifold(man, color=self.ents.F[fid].color)
        ents = self.ents

        def run():
            ents.remove([fid], list(eids))       # SketchUp uses up the profile and the path
            ents.merge_from(result, hint=self.eye)
            self.doc.set_selection([])

        self.commit(run)
        self.reset()
        self.view.hover = None

    def draw(self, view):
        if self.path is not None:
            pts, closed, _ = self.path
            col = (0.95, 0.35, 0.15, 1)
            if closed:
                loop = pts[:-1] if np.linalg.norm(pts[0] - pts[-1]) < 1e-9 else pts
                view.draw_loop(loop, col, 3, on_top=True)
            else:
                view.draw_strip(pts, col, 3, on_top=True)
        if self.preview is not None:
            view.draw_solid_preview(self.preview, (0.35, 0.6, 1.0))
        view.draw_inference(self.inf)


# ------------------------------------------------------------------ transforms

class TransformTool(Tool):
    """Shared selection handling for move / rotate / scale."""

    def reset(self):
        super().reset()
        self.items = []
        self.view.preview_transform = None

    def grab_selection(self, x, y):
        if self.doc.selection:
            if any(g.locked for g in self.doc.selected_groups()):
                self.warn_locked()      # locked groups stay where they are
            return list(self.doc.selection)
        pick = self.engine.pick_entity(x, y)
        item = pick_item(pick)
        if item is None:
            return []
        if item[0] == "g":
            g = self.doc.group(item[1])
            if g is not None and g.locked:
                self.warn_locked()
                return []
        items = [item]
        if item[0] == "f":
            ents = self.doc.ents(item[1])
            items += [("e", item[1], e) for e in ents.face_edges(ents.F[item[2]])]
        self.doc.set_selection(items)
        return items

    def hover_pick(self, x, y):
        if self.doc.selection:
            self.view.hover = None
            return
        self.hover_item(x, y)

    def moving_groups(self):
        return {i for k, _, i in self.items if k == "g"}


class MoveTool(TransformTool):
    name = "move"
    status = "Selecteer object om te verplaatsen. Ctrl = kopieren."
    measure_label = "Afstand"
    cursor_icon = "move"

    def status_text(self):
        if self.base is None:
            return "Selecteer object om te verplaatsen. Ctrl = kopieren."
        return "Selecteer doelpunt of typ een afstand."

    def reset(self):
        super().reset()
        self.base = None
        self.offset = np.zeros(3)
        self.copy = False
        self.press_pos = None
        if not hasattr(self, "last_copy"):
            self.last_copy = None

    def busy(self):
        return self.base is not None

    def mouse_move(self, e, x, y):
        if self.base is None:
            self.inf = self.engine.infer(x, y)
            self.hover_pick(x, y)
            return
        self.inf = self.engine.infer(x, y, anchor=self.base, exclude=self.moving_groups(),
                                     lock=self.lock_line(self.base))
        self.inf.point = self.snap_point(self.base, self.inf.point)
        self.offset = self.inf.point - self.base
        M = np.eye(4)
        M[:3, 3] = self.offset
        self.view.preview_transform = (list(self.items), M, self.copy)
        self.set_vcb(fmt_len(np.linalg.norm(self.offset)))

    def mouse_press(self, e, x, y):
        if self.base is None:
            items = self.grab_selection(x, y)
            if not items:
                return
            self.items = items
            self.inf = self.engine.infer(x, y)
            self.base = self.inf.point
            self.copy = bool(e.modifiers() & Qt.ControlModifier)
            self.press_pos = (x, y)
            self.view.hover = None
            self.last_copy = None
            return
        self.mouse_move(e, x, y)
        self.apply(self.offset)

    def mouse_release(self, e, x, y):
        if self.base is not None and self.press_pos is not None:
            if np.hypot(x - self.press_pos[0], y - self.press_pos[1]) > 6:
                self.apply(self.offset)
        self.press_pos = None

    def key_press(self, e):
        if e.key() == Qt.Key_Control:
            self.copy = not self.copy
            if self.view.preview_transform:
                items, M, _ = self.view.preview_transform
                self.view.preview_transform = (items, M, self.copy)
            return True
        return self.handle_lock_keys(e, self.base)

    def vcb_enter(self, text):
        t = text.strip().lower()
        if self.base is None and self.last_copy is not None:
            items, off = self.last_copy
            m = re.fullmatch(r"(\d+)\s*[x*]", t) or re.fullmatch(r"[x*]\s*(\d+)", t)
            dv = re.fullmatch(r"/\s*(\d+)", t)
            if m or dv:
                n = int((m or dv).group(1))
                if n < 1:
                    raise ValueError(t)
                step = off if m else off / n
                self.doc.undo()
                self.doc.redo_stack.clear()

                def run():
                    made = []
                    for k in range(1, n + 1):
                        made += ops.transform_selection(self.doc, np.eye(3), step * k, copy=True, items=items)
                    self.doc.set_selection(made)
                self.commit(run)
                return
        if self.base is None:
            raise ValueError("klik eerst een basispunt")
        vals = split_values(t, (1, 3))
        if len(vals) == 3:
            off = np.array([parse_length(v) for v in vals])
        else:
            L = parse_length(vals[0])
            d = self.lock_dir if self.lock_dir is not None else self.offset
            if np.linalg.norm(d) < 1e-9:
                raise ValueError("beweeg eerst in een richting")
            off = normalize(d) * L
        self.apply(off)

    def apply(self, off):
        if np.linalg.norm(off) < 1e-9:
            self.reset()
            return
        items, copy = list(self.items), self.copy

        def run():
            new = ops.transform_selection(self.doc, np.eye(3), off, copy=copy, items=items)
            self.doc.set_selection(new)
        self.commit(run)
        self.reset()
        if copy:
            self.last_copy = (items, np.array(off))
        self.view.set_status(self.status_text())

    def draw(self, view):
        if self.base is not None and self.inf is not None:
            view.draw_lines([self.base, self.inf.point], self.guide_color(), 1.5, stipple=True, on_top=True)
        view.draw_inference(self.inf)


class RotateTool(TransformTool):
    name = "rotate"
    status = "Selecteer middelpunt."
    measure_label = "Hoek"
    cursor_icon = "rotate"

    def status_text(self):
        if self.center is None:
            return "Selecteer middelpunt."
        if self.ref is None:
            return "Selecteer beginpunt van de hoek."
        return "Selecteer eindpunt van de hoek of typ een hoek."

    def reset(self):
        super().reset()
        self.center = None
        self.plane = None
        self.ref = None
        self.angle = 0.0
        if not hasattr(self, "forced_axis"):
            self.forced_axis = None
        self.copy = False
        self.hover_plane = None

    def busy(self):
        return self.center is not None

    def key_press(self, e):
        if e.key() in ARROWS:
            ax = ARROWS[e.key()]
            self.forced_axis = None if self.forced_axis == ax else ax
            if self.center is not None and self.forced_axis is not None:
                self.plane = Plane(self.center, AXES[ax])
                self.ref = None
            return True
        if e.key() == Qt.Key_Control:
            self.copy = not self.copy
            return True
        return False

    def mouse_move(self, e, x, y):
        if self.center is None:
            self.inf = self.engine.infer(x, y)
            self.hover_pick(x, y)
            self.hover_plane = self.drawing_plane(self.inf, x, y, self.forced_axis)
            return
        self.inf = self.engine.infer(x, y, anchor=self.center, plane=self.plane, exclude=self.moving_groups())
        if self.ref is None:
            return
        v = self.inf.point - self.center
        if np.linalg.norm(v) < 1e-9:
            return
        a = math.atan2(np.dot(np.cross(self.ref, v), self.plane.n), np.dot(self.ref, v))
        deg = math.degrees(a)
        snap = round(deg / 15.0) * 15.0
        if abs(deg - snap) < 2.5:
            deg = snap
        self.angle = deg
        self.view.preview_transform = (list(self.items), self.matrix(deg), self.copy)
        self.set_vcb(fmt_angle(deg))

    def matrix(self, deg):
        R = rotation_matrix(self.plane.n, math.radians(deg))
        M = np.eye(4)
        M[:3, :3] = R
        M[:3, 3] = self.center - R @ self.center
        return M

    def mouse_press(self, e, x, y):
        if self.center is None:
            items = self.grab_selection(x, y)
            if not items:
                return
            self.items = items
            self.inf = self.engine.infer(x, y)
            plane = self.drawing_plane(self.inf, x, y, self.forced_axis)
            self.center = self.inf.point
            self.plane = Plane(self.center, plane.n)
            self.copy = bool(e.modifiers() & Qt.ControlModifier)
            self.view.hover = None
            return
        if self.ref is None:
            v = self.inf.point - self.center if self.inf is not None else None
            if v is None or np.linalg.norm(v) < 1e-9:
                return
            self.ref = normalize(v)
            return
        self.apply(self.angle)

    def vcb_enter(self, text):
        if self.center is None:
            raise ValueError("klik eerst het draaipunt")
        if self.ref is None:
            self.ref = self.plane.u
        self.apply(parse_number(text.replace("°", "")))

    def apply(self, deg):
        M = self.matrix(deg)
        items, copy = list(self.items), self.copy

        def run():
            new = ops.transform_selection(self.doc, M[:3, :3], M[:3, 3], copy=copy, items=items)
            self.doc.set_selection(new)
        self.commit(run)
        self.reset()

    def draw(self, view):
        c = self.center if self.center is not None else (self.inf.point if self.inf else None)
        pl = self.plane if self.center is not None else self.hover_plane
        if c is not None and pl is not None:
            r = view.cam.pixel_size(c) * 45
            circ = [c + (pl.u * math.cos(a) + pl.v * math.sin(a)) * r for a in np.linspace(0, 2 * math.pi, 49)]
            col = (0.15, 0.3, 0.95, 1)
            if abs(pl.n[0]) > 0.99:
                col = AXIS_COLORS[0]
            elif abs(pl.n[1]) > 0.99:
                col = AXIS_COLORS[1]
            view.draw_strip(circ, col, 2, on_top=True)
            ticks = []
            for k in range(24):
                a = k * math.pi / 12
                d = pl.u * math.cos(a) + pl.v * math.sin(a)
                ticks += [c + d * r * 0.88, c + d * r]
            view.draw_lines(ticks, col, 1, on_top=True)
        if self.center is not None and self.ref is not None:
            r = view.cam.pixel_size(self.center) * 60
            view.draw_lines([self.center, self.center + self.ref * r], (0.2, 0.2, 0.2, 1), 1.5,
                            stipple=True, on_top=True)
            if self.inf is not None:
                view.draw_lines([self.center, self.inf.point], (0.2, 0.2, 0.2, 1), 1.5, on_top=True)
                view.add_label(self.inf.point, fmt_angle(self.angle))
        view.draw_inference(self.inf)


class ScaleTool(TransformTool):
    name = "scale"
    status = "Selecteer een greep om te schalen. Ctrl = rond middelpunt, Shift = uniform."
    measure_label = "Schaal"
    cursor_icon = "scale"

    def status_text(self):
        if self.p0 is None:
            return "Selecteer een greep om te schalen. Ctrl = rond middelpunt, Shift = uniform."
        return "Beweeg om te schalen of typ een factor."

    def reset(self):
        super().reset()
        self.p0 = None
        self.anchor = None
        self.factor = np.ones(3)

    def busy(self):
        return self.p0 is not None

    def mouse_move(self, e, x, y):
        if self.p0 is None:
            self.inf = self.engine.infer(x, y)
            self.hover_pick(x, y)
            return
        self.inf = self.engine.infer(x, y, anchor=self.p0, exclude=self.moving_groups())
        ref = self.p0 - self.anchor
        cur = self.inf.point - self.anchor
        L = float(np.dot(ref, ref))
        if L < 1e-12:
            return
        f = float(np.dot(cur, ref)) / L
        if abs(f) < 1e-3:
            return
        self.factor = np.array([f, f, f])
        self.view.preview_transform = (list(self.items), self.matrix(self.factor), False)
        self.set_vcb(f"{f:.3f}")

    def matrix(self, f):
        S = np.diag(f)
        M = np.eye(4)
        M[:3, :3] = S
        M[:3, 3] = self.anchor - S @ self.anchor
        return M

    def mouse_press(self, e, x, y):
        if self.p0 is None:
            items = self.grab_selection(x, y)
            if not items:
                return
            self.items = items
            bb = ops.selection_bbox(self.doc, items)
            self.inf = self.engine.infer(x, y)
            p = self.inf.point
            c = bb.mean(axis=0)
            if e.modifiers() & Qt.ControlModifier:
                self.anchor = c
            else:
                self.anchor = np.where(p >= c, bb[0], bb[1])
                self.anchor[2] = bb[0][2]  # keep the part on the print bed
            if np.linalg.norm(p - self.anchor) < 1e-6:
                self.anchor = c
            self.p0 = p
            self.view.hover = None
            return
        self.apply(self.factor)

    def vcb_enter(self, text):
        vals = split_values(text, (1, 3))
        if len(vals) == 3:
            f = np.array([parse_number(v) for v in vals])
        else:
            f = np.full(3, parse_number(vals[0]))
        if np.any(np.abs(f) < 1e-6):
            raise ValueError("factor mag niet 0 zijn")
        if self.p0 is None:
            items = list(self.doc.selection)
            if not items:
                raise ValueError("selecteer eerst iets")
            self.items = items
            bb = ops.selection_bbox(self.doc, items)
            self.anchor = np.array([(bb[0][0] + bb[1][0]) / 2, (bb[0][1] + bb[1][1]) / 2, bb[0][2]])
        self.apply(f)

    def apply(self, f):
        M = self.matrix(f)
        items = list(self.items)
        self.commit(lambda: self.doc.set_selection(
            ops.transform_selection(self.doc, M[:3, :3], M[:3, 3], items=items)))
        self.reset()

    def draw(self, view):
        if self.p0 is not None:
            view.draw_points([self.anchor], (0.9, 0.1, 0.1, 1), 9)
            if self.inf is not None:
                view.draw_lines([self.anchor, self.inf.point], (0.3, 0.3, 0.3, 1), 1, stipple=True, on_top=True)
                view.add_label(self.inf.point, f"× {self.factor[0]:.3f}")
        elif not self.doc.selection:
            view.draw_inference(self.inf)
        bb = ops.selection_bbox(self.doc) if self.doc.selection else None
        if bb is not None and self.p0 is None:
            lo, hi = bb
            grips = [[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])]
            view.draw_points(grips, (0.1, 0.6, 0.1, 1), 7)


# ------------------------------------------------------------------ tape measure

GUIDE_PREVIEW = (0.45, 0.45, 0.52, 1)
POINT_KINDS = ("endpoint", "midpoint", "origin", "guidepoint")
_PLUS_CURSOR = None


def plus_cursor():
    """The cursor of the Meetlint when it makes guides: a crosshair with a green +."""
    global _PLUS_CURSOR
    if _PLUS_CURSOR is None:
        from PySide6.QtGui import QCursor, QPainter, QPixmap
        pm = QPixmap(32, 32)
        pm.fill(Qt.transparent)
        p = QPainter(pm)
        p.setRenderHint(QPainter.Antialiasing)
        p.setPen(QPen(QColor(30, 30, 30), 1.4))
        p.drawLine(12, 2, 12, 22)
        p.drawLine(2, 12, 22, 12)
        p.setPen(QPen(QColor(255, 255, 255), 3.2))
        p.drawLine(25, 17, 25, 31)
        p.drawLine(17, 25, 31, 25)
        p.setPen(QPen(QColor(20, 130, 40), 2.0))
        p.drawLine(25, 17, 25, 31)
        p.drawLine(17, 25, 31, 25)
        p.end()
        _PLUS_CURSOR = QCursor(pm, 12, 12)
    return _PLUS_CURSOR


class TapeTool(Tool):
    name = "tape"
    status = "Selecteer beginpunt om te meten."
    measure_label = "Lengte"
    cursor_icon = "tape"

    def status_text(self):
        if self.edge is not None:
            return "Beweeg om de hulplijn te plaatsen of typ een afstand."
        if self.a is None:
            return "Selecteer beginpunt om te meten."
        return "Selecteer eindpunt of typ een afstand."

    def reset(self):
        super().reset()
        self.a = None
        self.b = None
        self.edge = None               # (a, b) of the edge the guide is parallel to
        self.dir = None
        self.base = None               # the point on that edge closest to the cursor
        self.offset = None             # unit offset direction, perpendicular to the edge
        self.distance = 0.0
        if not hasattr(self, "make_guides"):
            self.make_guides = True    # Ctrl wisselt naar "alleen meten"

    def activate(self):
        self.reset()
        self.set_cursor()

    def set_cursor(self):
        self.cursor = plus_cursor() if self.make_guides else None
        self.view.setCursor(self.cursor)

    def busy(self):
        return self.edge is not None or self.a is not None

    # -- offset mode ------------------------------------------------------
    def edge_under(self, x, y):
        """The segment under the cursor, as the line a parallel guide follows."""
        eh = self.engine.pick_edge(x, y, px=7.0)
        return None if eh is None else (np.asarray(eh[4][0], float), np.asarray(eh[4][1], float))

    def start_offset(self, seg, p):
        d = normalize(seg[1] - seg[0])
        if np.linalg.norm(d) < 1e-9:
            return False
        self.edge, self.dir, self.offset, self.distance = seg, d, None, 0.0
        self.update_offset(p)
        return True

    def update_offset(self, p):
        a, d = self.edge[0], self.dir
        foot = a + d * float(np.dot(p - a, d))
        w = p - foot
        perp = w - d * float(np.dot(w, d))
        L = float(np.linalg.norm(perp))
        if L > 1e-9:
            self.offset = perp / L
        self.base = foot
        self.distance = L

    def shown_distance(self):
        """The distance in the Measurements box: length snapping applies to free points."""
        return self.snap_len(self.distance) if self.free() else self.distance

    def guide_point(self):
        return self.base + self.offset * self.shown_distance()

    def place_offset(self, dist=None):
        if self.edge is None or self.offset is None:
            return
        d = self.shown_distance() if dist is None else self.snap_len(abs(float(dist)))
        p, dvec = self.base + self.offset * d, self.dir
        self.commit(lambda: self.doc.add_guide(p, dvec))
        self.reset()

    # -- measuring --------------------------------------------------------
    def finish_measure(self, p=None, dist=None):
        if p is None:
            if self.inf is None:
                return
            d = self.inf.point - self.a
            if dist is None:
                p = self.inf.point
            else:
                if np.linalg.norm(d) < 1e-9:
                    return
                p = self.a + d / np.linalg.norm(d) * abs(float(dist))
        self.b = p
        self.unlock()
        if self.make_guides:
            self.commit(lambda: self.doc.add_guide(p))

    def mouse_move(self, e, x, y):
        if self.edge is not None:
            self.inf = self.engine.infer(x, y)
            self.update_offset(self.inf.point)
            self.set_vcb(fmt_len(self.shown_distance()))
            self.view.set_status(f"Hulplijn evenwijdig aan de rand op {fmt_len(self.shown_distance())}")
            return
        anchor = self.a if self.b is None else None
        self.inf = self.engine.infer(x, y, anchor=anchor, lock=self.lock_line(anchor))
        self.inf.point = self.snap_point(anchor, self.inf.point)
        if self.a is not None and self.b is None:
            d = self.inf.point - self.a
            self.set_vcb(fmt_len(np.linalg.norm(d)))
            self.view.set_status(f"Δx {d[0]:.2f}   Δy {d[1]:.2f}   Δz {d[2]:.2f} mm")

    def mouse_press(self, e, x, y):
        self.mouse_move(e, x, y)
        if self.edge is not None:
            self.place_offset()
            return
        if self.a is not None:
            if self.b is None:
                self.finish_measure()
            else:
                self.a, self.b = self.inf.point, None
            return
        if self.make_guides and self.inf.kind not in POINT_KINDS:
            seg = self.edge_under(x, y)
            if seg is not None:
                self.start_offset(seg, self.inf.point)
                return
        if self.inf.kind in POINT_KINDS:
            self.a = self.inf.point
            self.view.hover = None
        else:
            # SketchUp: on a face or in empty space the tape measure does nothing
            self.view.set_status("Klik op een rand voor een hulplijn, of op een punt om te meten.")

    def key_press(self, e):
        if e.key() == Qt.Key_Control and not e.isAutoRepeat():
            self.make_guides = not self.make_guides
            self.set_cursor()
            self.view.set_status("Meetlint maakt hulplijnen." if self.make_guides else
                                 "Meetlint meet alleen, zonder hulplijnen.")
            return True
        return self.handle_lock_keys(e, None if self.edge is not None else
                                     (self.a if self.b is None else None))

    def vcb_enter(self, text):
        dist = parse_length(text)
        if self.edge is not None:
            self.place_offset(dist)
            return
        if self.a is None:
            raise ValueError("klik eerst een punt")
        self.finish_measure(dist=dist)

    def draw(self, view):
        if self.edge is not None and self.offset is not None:
            p = self.guide_point()
            L = view.cam.pixel_size(p) * 400
            view.draw_lines([p - self.dir * L, p + self.dir * L], GUIDE_PREVIEW, 1.4,
                            stipple=True, on_top=True)
            view.draw_lines([self.base, p], (0.1, 0.1, 0.1, 1), 1.2, stipple=True, on_top=True)
            view.draw_points([self.base, p], (0.1, 0.1, 0.1, 1), 6)
            view.add_label((self.base + p) / 2, fmt_len(self.shown_distance()))
        elif self.a is not None:
            end = self.b if self.b is not None else (self.inf.point if self.inf else None)
            if end is not None:
                view.draw_lines([self.a, end], (0.1, 0.1, 0.1, 1), 1.5, stipple=True, on_top=True)
                view.draw_points([self.a, end], (0.1, 0.1, 0.1, 1), 6)
                view.add_label((self.a + end) / 2, fmt_len(np.linalg.norm(end - self.a)))
        view.draw_inference(self.inf)


# ------------------------------------------------------------------ protractor

class ProtractorTool(Tool):
    """Gereedschap > Gradenboog: a guide line through a centre at a typed angle."""

    name = "protractor"
    status = "Selecteer middelpunt van de gradenboog."
    measure_label = "Hoek"
    cursor_icon = "rotate"

    def status_text(self):
        if self.center is None:
            return "Selecteer middelpunt van de gradenboog."
        if self.ref is None:
            return "Selecteer beginrichting van de hoek."
        return "Selecteer de hoek of typ een waarde in graden."

    def reset(self):
        super().reset()
        self.center = None
        self.plane = None
        self.ref = None
        self.angle = 0.0
        self.hover_plane = None
        if not hasattr(self, "forced_axis"):
            self.forced_axis = None

    def busy(self):
        return self.center is not None

    def key_press(self, e):
        if e.key() in ARROWS:
            ax = ARROWS[e.key()]
            self.forced_axis = None if self.forced_axis == ax else ax
            if self.center is not None and self.forced_axis is not None:
                self.plane = Plane(self.center, AXES[ax])
                self.ref = None
            return True
        return False

    def angle_at(self, p):
        """The angle from the reference direction to p, snapped to 15 degrees."""
        v = p - self.center
        if np.linalg.norm(v) < 1e-9:
            return self.angle
        a = math.degrees(math.atan2(float(np.dot(np.cross(self.ref, v), self.plane.n)),
                                    float(np.dot(self.ref, v))))
        snap = round(a / 15.0) * 15.0
        return snap if abs(a - snap) < 2.5 else a

    def mouse_move(self, e, x, y):
        if self.center is None:
            self.inf = self.engine.infer(x, y)
            self.hover_plane = self.drawing_plane(self.inf, x, y, self.forced_axis)
            return
        self.inf = self.engine.infer(x, y, anchor=self.center, plane=self.plane)
        if self.ref is None:
            return
        self.angle = self.angle_at(self.inf.point)
        self.set_vcb(fmt_angle(self.angle))

    def mouse_press(self, e, x, y):
        self.mouse_move(e, x, y)
        if self.center is None:
            self.center = self.inf.point
            n = self.hover_plane.n if self.hover_plane is not None else (0, 0, 1)
            self.plane = Plane(self.center, n)
            self.view.hover = None
            return
        if self.ref is None:
            v = self.inf.point - self.center
            if np.linalg.norm(v) < 1e-9:
                return
            self.ref = normalize(v)
            return
        self.place(self.angle)

    def direction(self, deg):
        return normalize(rotation_matrix(self.plane.n, math.radians(deg)) @ self.ref)

    def place(self, deg):
        c, d = self.center, self.direction(deg)
        self.commit(lambda: self.doc.add_guide(c, d))
        self.reset()

    def vcb_enter(self, text):
        if self.center is None:
            raise ValueError("klik eerst het middelpunt")
        if self.ref is None:
            self.ref = self.plane.u
        self.place(parse_number(text.replace("°", "")))

    def draw(self, view):
        c = self.center if self.center is not None else (self.inf.point if self.inf else None)
        pl = self.plane if self.center is not None else self.hover_plane
        if c is not None and pl is not None:
            r = view.cam.pixel_size(c) * 45
            circ = [c + (pl.u * math.cos(a) + pl.v * math.sin(a)) * r for a in np.linspace(0, 2 * math.pi, 49)]
            col = (0.15, 0.3, 0.95, 1)
            if abs(pl.n[0]) > 0.99:
                col = AXIS_COLORS[0]
            elif abs(pl.n[1]) > 0.99:
                col = AXIS_COLORS[1]
            view.draw_strip(circ, col, 2, on_top=True)
            ticks = []
            for k in range(24):
                a = k * math.pi / 12
                d = pl.u * math.cos(a) + pl.v * math.sin(a)
                ticks += [c + d * r * 0.88, c + d * r]
            view.draw_lines(ticks, col, 1, on_top=True)
        if self.center is not None and self.ref is not None:
            r = view.cam.pixel_size(self.center) * 60
            view.draw_lines([self.center, self.center + self.ref * r], (0.2, 0.2, 0.2, 1), 1.5,
                            stipple=True, on_top=True)
            d = self.direction(self.angle)
            view.draw_lines([self.center - d * r, self.center + d * r], GUIDE_PREVIEW, 1.4,
                            stipple=True, on_top=True)
            if self.inf is not None:
                view.draw_lines([self.center, self.inf.point], (0.2, 0.2, 0.2, 1), 1.5, on_top=True)
                view.add_label(self.inf.point, fmt_angle(self.angle))
        view.draw_inference(self.inf)


# ------------------------------------------------------------------ cut plane

class CutPlaneTool(Tool):
    """Printhulpmiddelen > Snijden: leg een snijvlak en knip het model in tweeën."""
    name = "cutplane"
    status = ("Klik op een vlak (of in het model) om het snijvlak te leggen; daarna kies je de "
              "verbinding: pennen, losse pennen of een zwaluwstaart.")
    measure_label = ""
    cursor_icon = "cutplane"

    def reset(self):
        super().reset()
        self.origin = None
        self.normal = None

    def mouse_move(self, e, x, y):
        self.inf = self.engine.infer(x, y)
        self.origin = np.asarray(self.inf.point, dtype=float)
        self.normal = self.drawing_plane(self.inf, x, y).n

    def mouse_press(self, e, x, y):
        self.mouse_move(e, x, y)
        if self.origin is None or self.normal is None:
            return
        origin, normal = tuple(self.origin), tuple(self.normal)
        self.reset()
        self.view.set_tool("select")
        self.view.window().cut_join_dialog(origin, normal)

    def draw(self, view):
        if self.origin is not None and self.normal is not None:
            r = view.cam.pixel_size(self.origin) * 45
            pl = Plane(self.origin, self.normal)
            loop = [self.origin + (pl.u * a + pl.v * b) * r for a, b in
                    ((-1, -1), (1, -1), (1, 1), (-1, 1))]
            view.draw_loop(loop, (0.85, 0.25, 0.2, 1), 2, on_top=True)
            view.draw_lines([self.origin, self.origin + self.normal * r], (0.85, 0.25, 0.2, 1),
                            1.5, stipple=True, on_top=True)
        view.draw_inference(self.inf)


class EdgeSizeTool(Tool):
    """Afronden en afschuinen: randen kiezen, maat in het meetvak, met Enter toe passen.

    Een klik op een gladde (soft) rand kiest de hele aaneengesloten keten, een klik
    op een vlak kiest al zijn randen. Shift doet er een bij of haalt er een af.
    De bewerking werkt op de actieve context: de groep die bewerkt wordt, of de
    losse geometrie (die dan een gesloten volume moet zijn)."""
    mode = "fillet"
    measure_label = "Straal"
    cursor_icon = "fillet"
    default_size = 1.0
    last_size = None

    def reset(self):
        super().reset()
        self.picked = []              # (ctx, eid) van de gekozen randen
        self.size = None
        self._toggled = None          # wat de laatste klik deed (om een dubbelklik terug te draaien)
        self.vcb_value = format_length(self.last_size or self.default_size)

    def busy(self):
        return bool(self.picked)

    def status_text(self):
        if not self.picked:
            return self.status
        what = "straal" if self.mode == "fillet" else "afstand"
        return (f"{self.vcb_label()}: typ een {what} en druk op Enter, "
                "of dubbelklik om toe te passen.")

    # -- kiezen ------------------------------------------------------------
    def _pick(self, x, y):
        eh = self.engine.pick_edge(x, y, only={self.doc.active}, px=7.0)
        gh = self.engine.pick_guide(x, y, px=5.0)
        if eh is not None and (gh is None or eh[0] <= gh[0]):
            return ("e", eh[1], eh[2])
        pick = self.engine.pick_entity(x, y)
        if pick[0] == "face":
            return ("f", pick[1], pick[2])
        return None

    def _edges_of(self, item):
        """De randen die bij een klik horen: een gladde keten of alle randen van een vlak."""
        kind, ctx, i = item
        ents = self.doc.ents(ctx)
        if ents is None:
            return ctx, []
        if kind == "f":
            if i not in ents.F:
                return ctx, []
            return ctx, list(ents.face_edges(ents.F[i]))
        return ctx, fillet.soft_chain(ents, i)

    def _show(self, extra=None):
        """De gekozen randen oranje markeren (met de rand onder de cursor erbij)."""
        if not self.picked and extra is None:
            self.view.hover = None
            return
        items = list(self.picked)
        if extra is not None and extra not in items:
            items.append(extra)
        self.view.hover = ("edges", items)

    def mouse_move(self, e, x, y):
        it = self._pick(x, y)
        self._show(it if it and it[0] == "e" else None)

    def mouse_press(self, e, x, y):
        self._toggled = None
        it = self._pick(x, y)
        if it is None:
            if not e.modifiers() & Qt.ShiftModifier:
                self.picked = []
            self._show()
            return
        ctx, eids = self._edges_of(it)
        if not eids:
            return
        have = all((ctx, i) in self.picked for i in eids)
        if e.modifiers() & Qt.ShiftModifier:
            if have:
                self.picked = [p for p in self.picked if p not in {(ctx, i) for i in eids}]
                self._toggled = ("removed", ctx, eids)
            else:
                self.picked += [(ctx, i) for i in eids]
                self._toggled = ("added", ctx, eids)
        elif have and len(eids) == len(self.picked):
            self.picked = []
            self._toggled = ("removed", ctx, eids)
        else:
            self.picked = [(ctx, i) for i in eids]
            self._toggled = ("added", ctx, eids)
        self._show()
        self.view.update_vcb()

    def double_click(self, e, x, y):
        """Dubbelklik past toe met de laatste maat; de klik erbij koos geen rand bij."""
        if self._toggled and self._toggled[0] == "added":
            self.picked = [p for p in self.picked if p not in {(self._toggled[1], i)
                                                               for i in self._toggled[2]}]
        self.apply(self.last_size or self.default_size)

    def vcb_enter(self, text):
        self.apply(parse_length(text))

    # -- toepassen ----------------------------------------------------------
    def apply(self, size):
        size = float(size)
        if size <= 0:
            raise ValueError("de maat moet groter dan 0 zijn")
        if not self.picked:
            self.view.set_status("Klik eerst op een rand die afgerond of afgeschuind moet worden.")
            return
        ctx = self.picked[0][0]
        if any(c != ctx for c, _i in self.picked):
            self.view.set_status("Afronden werkt binnen één groep of één gesloten volume.")
            return
        ents = self.doc.ents(ctx)
        if ents is None:
            self.view.set_status("Selecteer eerst een groep.")
            return
        eids = [i for _c, i in self.picked]
        try:
            man = ents.to_manifold()
        except Exception:
            man = mf.Manifold()
        if man.is_empty() or man.status() != mf.Error.NoError:
            self.view.set_status("Selecteer eerst een groep met een gesloten volume.")
            return
        ok, bad = fillet.edge_data(ents, eids)
        if not ok:
            self.view.set_status("Die randen horen niet bij twee vlakken; kies randen van een "
                                 "gesloten volume.")
            return
        edges = [(p0, p1, n1, n2) for (_eid, p0, p1, n1, n2) in ok]
        colors = ops.ents_face_colors(ents)
        color = colors[0] if colors else None
        segments = int(getattr(self.settings, "circle_segments", 64))

        def run():
            res = fillet.apply(man, edges, size, self.mode, segments)
            if res.is_empty() or res.status() != mf.Error.NoError:
                raise ValueError("de maat is te groot voor dit volume")
            new = Entities.from_manifold(res)
            g = self.doc.group(ctx) if ctx != ROOT else None
            if g is not None:
                g.ents = new                  # naam en kleur van de groep blijven
            else:
                for f in new.F:
                    new.set_face_color(f, color or self.current_color())
                self.doc.root = new

        self.commit(run)
        EdgeSizeTool.last_size = size
        self.size = size
        self.reset()
        self.set_vcb(format_length(size))
        if bad:
            self.view.set_status(f"{len(bad)} randen overgeslagen: die horen niet bij twee vlakken.")

    def draw(self, view):
        self._show()


class FilletTool(EdgeSizeTool):
    name = "fillet"
    mode = "fillet"
    status = ("Klik op de randen die afgerond moeten worden. Shift = toevoegen/verwijderen, "
              "klik op een gladde rand = de hele keten, klik op een vlak = al zijn randen.")
    measure_label = "Straal"
    cursor_icon = "fillet"


class ChamferTool(EdgeSizeTool):
    name = "chamfer"
    mode = "chamfer"
    status = ("Klik op de randen die afgeschuind moeten worden. Shift = toevoegen/verwijderen, "
              "klik op een gladde rand = de hele keten, klik op een vlak = al zijn randen.")
    measure_label = "Afstand"
    cursor_icon = "chamfer"


TOOL_CLASSES = [SelectTool, EraserTool, PaintTool, LineTool, FreehandTool, RectangleTool,
                RotatedRectangleTool, CircleTool,
                PolygonTool, Arc2Tool, ArcTool, Arc3Tool, PieTool, TextTool, VectorPlaceTool,
                PushPullTool,
                FollowMeTool, OffsetTool, MoveTool, RotateTool, ScaleTool, TapeTool, ProtractorTool,
                CutPlaneTool, FilletTool, ChamferTool, OrbitTool, PanTool, ZoomTool, ZoomWindowTool]

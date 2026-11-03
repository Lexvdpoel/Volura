"""The tools of the 2D worksheet: select, move, rotate, scale and the drawing tools
(line, rectangle, rounded rectangle, circle, ellipse, arc, polygon, text and offset).

Every tool is a small class with its own base class `SheetTool`; `SHEET_TOOLS` is the
registry that `volura.sheetview` builds in. The drawing tools inherit from
`DrawSheetTool`, which takes care of snapping (`volura.sheetsnap`), drawing the
preview and committing new paths. At the bottom are the operations on the selection:
union, subtract, intersect, join, split, align, distribute and duplicate in a row or
grid.

A tool works on `doc.sheet` (the shared `vector2d.Drawing`) and does every operation
as one undo step: first `doc.checkpoint()`, then change the paths, then
`doc.notify()`. Sizes are millimetres with Y up; the mouse positions the widget passes
on are widget pixels, which the tool converts with `view.to_world_xy`.
"""
from __future__ import annotations

import math

import numpy as np
import manifold3d as mf
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QPen

from .fonts import default_family, text_loops
from .i18n import tr
from .sheetsnap import paint_snap
from .tools import MEASURE_LABEL, format_length, parse_length, parse_number, split_values
from .vector2d import (TOL, VPath, arc_points, ellipse_arc_points, join_open_paths,
                       next_src, path_segments, polygon_area)

PICK_PX = 4.0            # pixels around a line that still count as a click on that line
DRAG_PX = 3.0            # only from this distance on is a drag a selection window
SELECT_COLOR = "#2f6fd6"


# --------------------------------------------------------------------------
#  measuring and calculating
# --------------------------------------------------------------------------

def seg_distance(p, a, b) -> float:
    """Distance from point `p` to segment `a`-`b`."""
    d = b - a
    L2 = float(d @ d)
    t = 0.0 if L2 <= 1e-30 else max(0.0, min(1.0, float((p - a) @ d) / L2))
    return float(np.hypot(*(p - (a + t * d))))


def path_distance(path, p) -> float:
    """Distance from point `p` to polyline `path` (to the nearest segment)."""
    pts = np.asarray(path.pts, dtype=float)
    if not len(pts):
        return np.inf
    if len(pts) == 1:
        return float(np.hypot(*(p - pts[0])))
    segs = list(zip(pts[:-1], pts[1:]))
    if path.closed:
        segs.append((pts[-1], pts[0]))
    return min((seg_distance(p, a, b) for a, b in segs), default=np.inf)


def paths_bbox(paths):
    """(xmin, ymin, xmax, ymax) of a list of paths, or None when it is empty."""
    pts = [np.asarray(p.pts, dtype=float) for p in paths if len(p.pts)]
    if not pts:
        return None
    P = np.concatenate(pts)
    return (float(P[:, 0].min()), float(P[:, 1].min()),
            float(P[:, 0].max()), float(P[:, 1].max()))


def _seg_hits_seg(p1, p2, p3, p4) -> bool:
    """Do two segments intersect (also when they touch at an endpoint)?"""
    r, s = p2 - p1, p4 - p3
    den = float(r[0] * s[1] - r[1] * s[0])
    q = p3 - p1
    if abs(den) <= 1e-12:
        return False
    t = float(q[0] * s[1] - q[1] * s[0]) / den
    u = float(q[0] * r[1] - q[1] * r[0]) / den
    return -1e-12 <= t <= 1.0 + 1e-12 and -1e-12 <= u <= 1.0 + 1e-12


def rect_edges(rect):
    """The four corners of (xmin, ymin, xmax, ymax), counter-clockwise."""
    xmin, ymin, xmax, ymax = rect
    return [np.array([xmin, ymin]), np.array([xmax, ymin]),
            np.array([xmax, ymax]), np.array([xmin, ymax])]


def path_touches_rect(path, rect):
    """Does `path` touch the rectangle `rect`: inside it, through it or against it."""
    pts = np.asarray(path.pts, dtype=float)
    if not len(pts):
        return False
    xmin, ymin, xmax, ymax = rect

    def inside(q):
        return xmin - 1e-9 <= q[0] <= xmax + 1e-9 and ymin - 1e-9 <= q[1] <= ymax + 1e-9

    if any(inside(q) for q in pts):
        return True
    segs = list(zip(pts[:-1], pts[1:]))
    if path.closed:
        segs.append((pts[-1], pts[0]))
    corners = rect_edges(rect)
    for a, b in segs:
        for i, c in enumerate(corners):
            if _seg_hits_seg(a, b, c, corners[(i + 1) % 4]):
                return True
    return False


def path_in_rect(path, rect):
    """Does `path` lie entirely inside rectangle `rect`?"""
    pts = np.asarray(path.pts, dtype=float)
    if not len(pts):
        return False
    xmin, ymin, xmax, ymax = rect
    return (float(pts[:, 0].min()) >= xmin - 1e-9 and float(pts[:, 0].max()) <= xmax + 1e-9
            and float(pts[:, 1].min()) >= ymin - 1e-9 and float(pts[:, 1].max()) <= ymax + 1e-9)


def translation(delta) -> np.ndarray:
    dx, dy = float(delta[0]), float(delta[1])
    return np.array([[1.0, 0.0, dx], [0.0, 1.0, dy], [0.0, 0.0, 1.0]])


def rotation(center, angle) -> np.ndarray:
    c, s = math.cos(float(angle)), math.sin(float(angle))
    cx, cy = float(center[0]), float(center[1])
    return np.array([[c, -s, cx - c * cx + s * cy],
                     [s, c, cy - s * cx - c * cy],
                     [0.0, 0.0, 1.0]])


def scaling(center, factor) -> np.ndarray:
    k = float(factor)
    cx, cy = float(center[0]), float(center[1])
    return np.array([[k, 0.0, cx - k * cx], [0.0, k, cy - k * cy], [0.0, 0.0, 1.0]])


def mirror_matrix(horizontal: bool, center) -> np.ndarray:
    """Mirror around the vertical axis through `center` (horizontal) or the horizontal one."""
    cx, cy = float(center[0]), float(center[1])
    if horizontal:
        return np.array([[-1.0, 0.0, 2.0 * cx], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    return np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 2.0 * cy], [0.0, 0.0, 1.0]])


def transformed_points(pts, M) -> np.ndarray:
    p = np.asarray(pts, dtype=float)
    if not len(p):
        return np.ascontiguousarray(p, dtype=float)
    return np.ascontiguousarray(p @ M[:2, :2].T + M[:2, 2], dtype=float)


def copy_of(path: VPath) -> VPath:
    return VPath(np.asarray(path.pts, dtype=float).copy(), path.closed, path.layer,
                 path.filled, path.rule, path.src, dict(path.meta))


def parse_offset(text, direction=None) -> np.ndarray:
    """The typed offset: one distance along `direction`, or `x;y`."""
    parts = split_values(text, counts=(1, 2))
    if len(parts) == 1:
        d = parse_length(parts[0])
        v = np.asarray(direction, dtype=float) if direction is not None else np.array([1.0, 0.0])
        n = float(np.hypot(*v))
        if n <= 1e-12:
            v, n = np.array([1.0, 0.0]), 1.0
        return v / n * d
    if len(parts) == 2:
        return np.array([parse_length(parts[0]), parse_length(parts[1])])
    raise ValueError(text)


# --------------------------------------------------------------------------
#  base class
# --------------------------------------------------------------------------

class SheetTool:
    """A tool of the 2D worksheet: mouse, keys, Measurements box and drawing."""

    name = ""
    status = ""                     # the text in the status bar
    measure_label = MEASURE_LABEL   # the label of the Measurements box
    cursor_shape = Qt.ArrowCursor

    def __init__(self, view):
        self.view = view
        self.vcb_value = ""           # what the tool puts in the Measurements box
        self.reset()                  # every step of the tool has its own state

    # -- the document ---------------------------------------------------

    @property
    def doc(self):
        return self.view.doc

    @property
    def sheet(self):
        return self.view.doc.sheet

    def selected(self) -> list[VPath]:
        return self.view.selected_paths()

    def edit(self, change):
        """One undo step: checkpoint, change the paths, notify everything."""
        self.view.edit(change)

    # -- life cycle ------------------------------------------------------

    def activate(self):
        self.reset()

    def deactivate(self):
        pass

    def reset(self):
        pass

    def cancel(self):
        self.reset()

    def status_text(self) -> str:
        return self.status

    def vcb_label(self) -> str:
        return self.measure_label

    # -- input -----------------------------------------------------------

    def mouse_press(self, e, x, y):
        pass

    def mouse_move(self, e, x, y):
        pass

    def mouse_release(self, e, x, y):
        pass

    def double_click(self, e, x, y):
        pass

    def finish_input(self):
        """Enter without text: the drawing tools are done then."""
        pass

    def key_press(self, e) -> bool:
        return False

    def vcb_enter(self, text):
        raise ValueError(text)

    # -- painting --------------------------------------------------------

    def paint(self, p):
        pass

    def set_vcb(self, value):
        self.view.set_vcb(value)

    def warn(self, text):
        self.view.set_status(text)
        self.reset()
        self.view.update()


# --------------------------------------------------------------------------
#  select
# --------------------------------------------------------------------------

class SelectSheetTool(SheetTool):
    """Click a line or drag a window: right to left also catches what touches it."""

    name = "select"
    status = tr("Click a line to select it, or drag a selection window")

    def reset(self):
        self.start = None
        self.cur = None
        self.dragging = False

    def activate(self):
        self.reset()
        self.view.setCursor(Qt.ArrowCursor)

    def mouse_press(self, e, x, y):
        if e.button() != Qt.LeftButton:
            return
        self.start = (float(x), float(y))
        self.cur = (float(x), float(y))
        self.dragging = False
        self.view.update()

    def mouse_move(self, e, x, y):
        if self.start is not None:
            self.cur = (float(x), float(y))
            if (abs(self.cur[0] - self.start[0]) > DRAG_PX
                    or abs(self.cur[1] - self.start[1]) > DRAG_PX):
                self.dragging = True
                self.view.hover = None
            self.view.update()
            return
        self.view.hover = self.view.path_at(x, y)
        self.view.update()

    def mouse_release(self, e, x, y):
        if self.start is None:
            return
        shift = bool(e.modifiers() & Qt.ShiftModifier)
        if self.dragging:
            self.cur = (float(x), float(y))
            paths = self.view.paths_in_band(self.start, self.cur)
            if shift:
                self.view.add_to_selection(paths)
            else:
                self.view.set_selection(paths)
        else:
            path = self.view.path_at(x, y)
            if path is None:
                if not shift:
                    self.view.clear_selection()
            elif shift:
                self.view.toggle_selection(path)
            else:
                self.view.set_selection([path])
        self.reset()
        self.view.update()

    def paint(self, p):
        if not self.dragging or self.start is None or self.cur is None:
            return
        a, b = QPointF(*self.start), QPointF(*self.cur)
        rect = QRectF(a, b).normalized()
        left_to_right = self.cur[0] >= self.start[0]
        pen = QPen(QColor("#2ea44f" if left_to_right else SELECT_COLOR))
        pen.setStyle(Qt.SolidLine if left_to_right else Qt.DashLine)
        p.setPen(pen)
        # SketchUp: green and solid = everything inside, blue dashed = also what it touches
        p.setBrush(QColor(46, 164, 79, 28) if left_to_right else QColor(47, 111, 214, 28))
        p.drawRect(rect)
        p.setBrush(Qt.NoBrush)


# --------------------------------------------------------------------------
#  move
# --------------------------------------------------------------------------

class MoveSheetTool(SheetTool):
    """Click a base point, click the target or type a distance ("30" or "10;5")."""

    name = "move"
    status = tr("Click a base point and then the target; type a distance or x;y")
    measure_label = tr("Distance")
    cursor_shape = Qt.SizeAllCursor

    def reset(self):
        self.base = None
        self.cursor = None
        self.copy = False

    def activate(self):
        self.reset()
        self.view.setCursor(self.cursor_shape)

    def mouse_press(self, e, x, y):
        if e.button() != Qt.LeftButton:
            return
        p = self.view.to_world_xy(x, y)
        if self.base is None:
            if not self.selected():
                self.warn(tr("First select what you want to move."))
                return
            self.base = p
            self.copy = bool(e.modifiers() & Qt.ControlModifier)
            self.cursor = p
            self.set_vcb("0 mm")
            return
        self.finish(p - self.base)

    def mouse_move(self, e, x, y):
        self.cursor = self.view.to_world_xy(x, y)
        if self.base is not None:
            self.set_vcb(format_length(float(np.hypot(*(self.cursor - self.base)))))
        self.view.update()

    def finish(self, delta):
        paths = self.selected()
        delta = np.asarray(delta, dtype=float)
        if not paths:
            self.warn(tr("First select what you want to move."))
            return
        if float(np.hypot(*delta)) <= 1e-9:
            self.reset()
            self.view.update()
            return
        self.edit(lambda: self.view.transform_paths(paths, translation(delta), copy=self.copy))
        self.reset()
        self.view.update()

    def vcb_enter(self, text):
        if self.base is None:
            self.warn(tr("First click the base point of the move."))
            return
        direction = None
        if self.cursor is not None:
            direction = self.cursor - self.base
        self.finish(parse_offset(text, direction))

    def paint(self, p):
        if self.base is None or self.cursor is None:
            return
        a, b = self.view.to_screen(self.base), self.view.to_screen(self.cursor)
        pen = QPen(QColor("#d23b3b"))
        p.setPen(pen)
        p.drawLine(a, b)
        p.setBrush(QColor("#d23b3b"))
        p.drawEllipse(a, 2.5, 2.5)
        p.setBrush(Qt.NoBrush)


# --------------------------------------------------------------------------
#  rotate
# --------------------------------------------------------------------------

class RotateSheetTool(SheetTool):
    """Click the centre, then click the angle or type it in degrees (positive = counter-clockwise)."""

    name = "rotate"
    status = tr("Click the centre of the rotation and then the angle; type for example 90")
    measure_label = tr("Angle")
    cursor_shape = Qt.CrossCursor

    def reset(self):
        self.center = None
        self.ref_angle = 0.0
        self.cursor = None

    def activate(self):
        self.reset()
        self.view.setCursor(self.cursor_shape)

    def _angle_at(self, p) -> float:
        d = np.asarray(p, dtype=float) - self.center
        if float(np.hypot(*d)) <= 1e-9:
            return self.ref_angle
        return math.atan2(float(d[1]), float(d[0]))

    def mouse_press(self, e, x, y):
        if e.button() != Qt.LeftButton:
            return
        p = self.view.to_world_xy(x, y)
        if self.center is None:
            if not self.selected():
                self.warn(tr("First select what you want to rotate."))
                return
            self.center = np.asarray(p, dtype=float)
            self.ref_angle = self._angle_at(p)
            self.cursor = p
            self.set_vcb("0°")
            return
        self.finish(self._angle_at(p) - self.ref_angle)

    def mouse_move(self, e, x, y):
        self.cursor = self.view.to_world_xy(x, y)
        if self.center is not None:
            deg = math.degrees(self._angle_at(self.cursor) - self.ref_angle)
            self.set_vcb(f"{deg:.1f}°")
        self.view.update()

    def finish(self, angle):
        paths = self.selected()
        if not paths:
            self.warn(tr("First select what you want to rotate."))
            return
        center = self.center if self.center is not None else self.view.selection_center()
        if center is None:
            self.warn(tr("There is no selection to rotate."))
            return
        if abs(float(angle)) <= 1e-9:
            self.reset()
            self.view.update()
            return
        self.edit(lambda: self.view.transform_paths(paths, rotation(center, angle)))
        self.reset()
        self.view.update()

    def vcb_enter(self, text):
        deg = parse_number(text)
        self.finish(math.radians(deg))

    def paint(self, p):
        if self.center is None or self.cursor is None:
            return
        c = self.view.to_screen(self.center)
        r = float(np.hypot(*(self.cursor - self.center))) * self.view.scale
        pen = QPen(QColor("#d23b3b"))
        p.setPen(pen)
        if r > 2.0:
            p.setBrush(Qt.NoBrush)
            p.drawEllipse(c, r, r)
        p.drawLine(c, self.view.to_screen(self.cursor))
        p.setBrush(QColor("#d23b3b"))
        p.drawEllipse(c, 2.5, 2.5)
        p.setBrush(Qt.NoBrush)


# --------------------------------------------------------------------------
#  scale
# --------------------------------------------------------------------------

class ScaleSheetTool(SheetTool):
    """Uniform scaling: drag from the centre of the selection, or type a factor."""

    name = "scale"
    status = tr("Drag from the centre of the selection, or type a factor such as 2")
    measure_label = tr("Factor")
    cursor_shape = Qt.SizeFDiagCursor

    def reset(self):
        self.base = None
        self.cursor = None

    def activate(self):
        self.reset()
        self.view.setCursor(self.cursor_shape)

    def center(self):
        return self.view.selection_center()

    def mouse_press(self, e, x, y):
        if e.button() != Qt.LeftButton:
            return
        if not self.selected():
            self.warn(tr("First select what you want to scale."))
            return
        p = self.view.to_world_xy(x, y)
        if self.center() is None:
            self.warn(tr("There is no selection to scale."))
            return
        self.base = p
        self.cursor = p
        self.set_vcb("1.00")
        self.view.update()

    def mouse_release(self, e, x, y):
        if e.button() != Qt.LeftButton or self.base is None:
            return
        self.finish_at(self.view.to_world_xy(x, y))

    def mouse_move(self, e, x, y):
        self.cursor = self.view.to_world_xy(x, y)
        if self.base is not None:
            self.set_vcb(f"{self.factor_at(self.cursor):.2f}")
        self.view.update()

    def factor_at(self, p) -> float:
        c = self.center()
        r0 = float(np.hypot(*(np.asarray(self.base, dtype=float) - c)))
        r1 = float(np.hypot(*(np.asarray(p, dtype=float) - c)))
        return r1 / r0 if r0 > 1e-9 else 1.0

    def finish_at(self, p):
        factor = self.factor_at(p)
        self.apply(factor)

    def apply(self, factor):
        paths = self.selected()
        c = self.center()
        if not paths or c is None:
            self.warn(tr("First select what you want to scale."))
            return
        if abs(float(factor) - 1.0) <= 1e-12 or float(factor) <= 0.0:
            self.reset()
            self.view.update()
            return
        self.edit(lambda: self.view.transform_paths(paths, scaling(c, factor)))
        self.reset()
        self.view.update()

    def vcb_enter(self, text):
        self.apply(parse_number(text))

    def paint(self, p):
        if self.base is None or self.cursor is None:
            return
        c = self.center()
        if c is None:
            return
        box = paths_bbox(self.selected())
        if box is None:
            return
        k = self.factor_at(self.cursor)
        a = self.view.to_screen((c[0] + (box[0] - c[0]) * k, c[1] + (box[1] - c[1]) * k))
        b = self.view.to_screen((c[0] + (box[2] - c[0]) * k, c[1] + (box[3] - c[1]) * k))
        pen = QPen(QColor("#d23b3b"))
        pen.setStyle(Qt.DashLine)
        p.setPen(pen)
        p.drawRect(QRectF(a, b).normalized())


# --------------------------------------------------------------------------
#  building curves and shapes
# --------------------------------------------------------------------------

#: curves on the worksheet are flattened more finely than the default 0.01 mm: for a
#: circle the area of the polygon is otherwise well over 0.1 % smaller than pi r^2
CURVE_TOL = TOL / 4.0


def circle_loop(center, r, segments=None, tol=CURVE_TOL) -> np.ndarray:
    """A circle as a closed polyline: flattened with `tol`, or with `segments` sides."""
    c = np.asarray(center, dtype=float)
    r = abs(float(r))
    if r <= 0.0:
        return np.zeros((0, 2))
    if segments:
        n = max(3, int(segments))
        t = np.linspace(0.0, 2.0 * math.pi, n + 1)[:-1]
        return np.ascontiguousarray(c + np.column_stack([r * np.cos(t), r * np.sin(t)]))
    return np.ascontiguousarray(arc_points(c, r, 0.0, 2.0 * math.pi, tol)[:-1], dtype=float)


def ellipse_loop(center, rx, ry, rot=0.0, segments=None, tol=CURVE_TOL) -> np.ndarray:
    """An ellipse as a closed polyline (with `segments` sides when given)."""
    rx, ry = abs(float(rx)), abs(float(ry))
    if rx <= 0.0 or ry <= 0.0:
        return np.zeros((0, 2))
    if segments:
        n = max(3, int(segments))
        t = np.linspace(0.0, 2.0 * math.pi, n + 1)[:-1]
        pts = np.column_stack([rx * np.cos(t), ry * np.sin(t)])
    else:
        pts = ellipse_arc_points(np.zeros(2), rx, ry, 0.0, 2.0 * math.pi, tol, rot)[:-1]
        rot = 0.0
    pts = np.asarray(pts, dtype=float)
    if rot:
        co, si = math.cos(rot), math.sin(rot)
        pts = pts @ np.array([[co, -si], [si, co]]).T
    return np.ascontiguousarray(pts + np.asarray(center, dtype=float), dtype=float)


def rounded_rect_loop(a, b, r, tol=CURVE_TOL):
    """Rectangle with rounded corners: `r` is the radius, counter-clockwise."""
    p0, p1 = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    x0, x1 = sorted((float(p0[0]), float(p1[0])))
    y0, y1 = sorted((float(p0[1]), float(p1[1])))
    w, h = x1 - x0, y1 - y0
    if w <= 1e-9 or h <= 1e-9:
        return None
    r = max(0.0, min(float(r), 0.5 * min(w, h)))
    if r <= 1e-9:
        return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=float)
    corners = (((x0 + r), (y0 + r)), math.pi, 1.5 * math.pi), \
              (((x1 - r), (y0 + r)), 1.5 * math.pi, 2.0 * math.pi), \
              (((x1 - r), (y1 - r)), 0.0, 0.5 * math.pi), \
              (((x0 + r), (y1 - r)), 0.5 * math.pi, math.pi)
    pts = []
    for c, a0, a1 in corners:
        for q in arc_points(np.array(c, dtype=float), r, a0, a1, tol):
            q = np.asarray(q, dtype=float)
            if pts and float(np.hypot(*(q - pts[-1]))) <= 1e-9:
                continue
            pts.append(q)
    while len(pts) > 1 and float(np.hypot(*(pts[0] - pts[-1]))) <= 1e-9:
        pts.pop()
    return np.ascontiguousarray(np.asarray(pts, dtype=float).reshape(-1, 2), dtype=float)


def rect_loop(a, b) -> np.ndarray:
    """The four corners of a rectangle through two opposite corner points."""
    p0, p1 = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    x0, x1 = sorted((float(p0[0]), float(p1[0])))
    y0, y1 = sorted((float(p0[1]), float(p1[1])))
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=float)


def arc_through_3(a, b, c) -> tuple | None:
    """Centre, radius and the angles of the arc through three points."""
    a, b, c = (np.asarray(q, dtype=float) for q in (a, b, c))
    d = 2.0 * (a[0] * (b[1] - c[1]) + b[0] * (c[1] - a[1]) + c[0] * (a[1] - b[1]))
    if abs(d) < 1e-12:
        return None
    a2, b2, c2 = float(a @ a), float(b @ b), float(c @ c)
    ux = (a2 * (b[1] - c[1]) + b2 * (c[1] - a[1]) + c2 * (a[1] - b[1])) / d
    uy = (a2 * (c[0] - b[0]) + b2 * (a[0] - c[0]) + c2 * (b[0] - a[0])) / d
    center = np.array([ux, uy])
    r = float(np.hypot(*(a - center)))
    if r <= 1e-9:
        return None
    ang = lambda q: math.atan2(float(q[1] - center[1]), float(q[0] - center[0]))
    a0, am, a1 = ang(a), ang(b), ang(c)
    return center, r, a0, am, a1


def arc_span(a0, am, a1) -> float:
    """The angle from `a0` to `a1` that passes `am`, in radians (signed)."""
    two = 2.0 * math.pi

    def delta(from_a, to_a):
        d = to_a - from_a
        while d <= -math.pi:
            d += two
        while d > math.pi:
            d -= two
        return d
    d0, d1 = delta(a0, am), delta(am, a1)
    if d0 * d1 <= 0.0:                       # no arc through the middle: the long way round
        return d0 + d1 + (two if d0 + d1 < 0.0 else -two)
    return d0 + d1


def parse_angle_value(text):
    """`"30"`, `"30<45"` or `"45<30"`: length and angle in degrees, in either order."""
    if "<" not in text:
        return parse_length(text), None
    parts = text.split("<", 1)
    first, second = parse_length(parts[0]), parse_number(parts[1])
    return first, second


# --------------------------------------------------------------------------
#  base class of the drawing tools
# --------------------------------------------------------------------------

class DrawSheetTool(SheetTool):
    """A tool that draws something new: snapping, preview and one undo step."""

    cursor_shape = Qt.CrossCursor

    def __init__(self, view):
        super().__init__(view)
        self.snap = None
        self.cursor = None

    def reset(self):
        self.snap = None
        self.cursor = None

    # -- snapping ----------------------------------------------------------

    def snap_at(self, x, y, anchor=None, allow_grid=True):
        self.snap = self.view.snap_engine.snap(x, y, anchor=anchor, allow_grid=allow_grid)
        self.cursor = np.asarray(self.snap.point, dtype=float)
        return self.snap

    def raw_at(self, x, y):
        return self.view.to_world_xy(x, y)

    # -- committing --------------------------------------------------------

    def add_shapes(self, shapes):
        """New paths as one undo step; the new paths become selected."""
        shapes = list(shapes)
        if not shapes:
            return []
        sheet = self.sheet
        src = next_src(sheet.paths)
        layer = self.view.draw_layer()
        made: list[VPath] = []

        def change():
            for pts, closed, meta, filled in shapes:
                made.append(sheet.add_path(np.asarray(pts, dtype=float), closed, layer,
                                           bool(filled), "nonzero", src, dict(meta or {})))
        self.edit(change)
        self.view.set_selection(made)
        return made

    def finish_input(self):
        """Enter without text: for the drawing tools that means done."""
        self.reset()
        self.view.update()

    def double_click(self, e, x, y):
        self.finish_input()

    # -- painting ----------------------------------------------------------

    def preview(self, p):
        pass

    def paint(self, p):
        self.preview(p)
        paint_snap(p, self.view, self.snap)

    def draw_line(self, p, pts, closed=False, color="#1f3b63"):
        if len(pts) < 1:
            return
        pen = QPen(QColor(color))
        pen.setWidthF(1.6)
        p.setPen(pen)
        screen = [self.view.to_screen(q) for q in pts]
        for a, b in zip(screen[:-1], screen[1:]):
            p.drawLine(a, b)
        if closed and len(screen) > 1:
            p.drawLine(screen[-1], screen[0])

    def draw_rubber(self, p, a, b, color="#8a8a8a"):
        pen = QPen(QColor(color))
        pen.setStyle(Qt.DashLine)
        pen.setWidthF(1.0)
        p.setPen(pen)
        p.drawLine(self.view.to_screen(a), self.view.to_screen(b))


# --------------------------------------------------------------------------
#  line and polyline
# --------------------------------------------------------------------------

class LineSheetTool(DrawSheetTool):
    """Click points; a click on the start point closes the line, Enter or double click ends it."""

    name = "line"
    status = tr("Click the start point of the line")
    measure_label = tr("Length")

    def reset(self):
        super().reset()
        self.pts: list[np.ndarray] = []

    def status_text(self):
        if not self.pts:
            return tr("Click the start point of the line, or type a length")
        return tr("Click the next point; click the start point to close. "
                  "Type a length or length<angle; Enter or double click ends the line.")

    def mouse_move(self, e, x, y):
        anchor = self.pts[-1] if self.pts else None
        self.snap_at(x, y, anchor=anchor)
        if anchor is not None:
            self.set_vcb(format_length(float(np.hypot(*(self.cursor - anchor)))))
        self.view.update()

    def mouse_press(self, e, x, y):
        if e.button() != Qt.LeftButton:
            return
        self.mouse_move(e, x, y)
        self.add_point(self.cursor)
        self.view.update()

    def add_point(self, p):
        p = np.asarray(p, dtype=float)
        if not self.pts:
            self.pts = [p]
            return
        if float(np.hypot(*(p - self.pts[-1]))) <= 1e-6:
            return
        if len(self.pts) >= 2 and float(np.hypot(*(p - self.pts[0]))) <= 1e-6:
            self.finish(closed=True)
            return
        self.pts.append(p)

    def finish(self, closed=False):
        pts = list(self.pts)
        self.reset()
        if len(pts) >= 2:
            if closed:
                self.add_shapes([(pts, True, {}, False)])
            else:
                self.add_shapes([(pts, False, {}, False)])

    def finish_input(self):
        if len(self.pts) >= 2:
            self.finish(closed=False)
        else:
            self.reset()
            self.view.update()

    def double_click(self, e, x, y):
        """The second click of a double click does not count as an extra point."""
        if len(self.pts) >= 2:
            tol = 3.0 / max(self.view.scale, 1e-9)
            if float(np.hypot(*(self.pts[-1] - self.pts[-2]))) <= tol:
                self.pts = self.pts[:-1]
        self.finish_input()

    def vcb_enter(self, text):
        if not self.pts:
            raise ValueError(tr("first click the start point of the line"))
        L, angle = parse_angle_value(text)
        anchor = self.pts[-1]
        if angle is not None:
            a = math.radians(float(angle))
            d = np.array([math.cos(a), math.sin(a)])
        else:
            raw = self.snap.raw if self.snap is not None else self.cursor
            d = np.asarray(raw, dtype=float) - anchor
            if float(np.hypot(*d)) <= 1e-9:
                d = np.array([1.0, 0.0])
            d = d / float(np.hypot(*d))
        self.add_point(anchor + d * float(L))
        self.view.update()

    def preview(self, p):
        if not self.pts:
            return
        pts = list(self.pts) + ([self.cursor] if self.cursor is not None else [])
        self.draw_line(p, pts, color="#1f3b63")
        if len(self.pts) >= 2 and self.cursor is not None:
            self.draw_rubber(p, self.pts[0], self.cursor, "#2e7d32")


# --------------------------------------------------------------------------
#  rectangle and rounded rectangle
# --------------------------------------------------------------------------

class RectangleSheetTool(DrawSheetTool):
    """Click two opposite corners, or type the dimensions ("30;20")."""

    name = "rectangle"
    status = tr("Click the first corner of the rectangle")
    measure_label = tr("Dimensions")

    def reset(self):
        super().reset()
        self.p0 = None

    def status_text(self):
        if self.p0 is None:
            return tr("Click the first corner of the rectangle")
        return tr("Click the opposite corner, or type width;height such as 30;20")

    def mouse_move(self, e, x, y):
        self.snap_at(x, y, anchor=self.p0)
        if self.p0 is not None:
            w, h = self.size()
            self.set_vcb(f"{w:.2f}; {h:.2f}")
        self.view.update()

    def size(self):
        d = np.asarray(self.cursor, dtype=float) - np.asarray(self.p0, dtype=float)
        return abs(float(d[0])), abs(float(d[1]))

    def mouse_press(self, e, x, y):
        if e.button() != Qt.LeftButton:
            return
        self.mouse_move(e, x, y)
        if self.p0 is None:
            self.p0 = np.asarray(self.cursor, dtype=float)
        else:
            self.finish()
        self.view.update()

    def loop(self):
        if self.p0 is None or self.cursor is None:
            return None
        return rect_loop(self.p0, self.cursor)

    def finish(self, size=None):
        if self.p0 is None:
            return
        p0 = np.asarray(self.p0, dtype=float)
        if size is not None:
            w, h = float(size[0]), float(size[1])
            cur = np.asarray(self.cursor, dtype=float) - p0 if self.cursor is not None \
                else np.array([1.0, 1.0])
            sx = -1.0 if cur[0] < 0 else 1.0
            sy = -1.0 if cur[1] < 0 else 1.0
            p1 = p0 + np.array([w * sx, h * sy])
        else:
            p1 = np.asarray(self.cursor, dtype=float)
        loop = rect_loop(p0, p1)
        self.reset()
        if abs(polygon_area(loop)) > 1e-9:
            self.add_shapes([(loop, True, {}, False)])

    def vcb_enter(self, text):
        if self.p0 is None:
            raise ValueError(tr("first click the first corner of the rectangle"))
        vals = split_values(text, (2,))
        if len(vals) != 2:
            raise ValueError(tr("give width;height, for example 30;20"))
        self.finish((parse_length(vals[0]), parse_length(vals[1])))

    def preview(self, p):
        loop = self.loop()
        if loop is not None:
            self.draw_line(p, loop, closed=True)


class RoundedRectSheetTool(RectangleSheetTool):
    """Rectangle with rounded corners: drag the radius, or type "r5"."""

    name = "roundrect"
    status = tr("Click the first corner of the rounded rectangle")
    measure_label = tr("Radius")

    def reset(self):
        super().reset()
        self.box = None
        self.radius = 0.0

    def status_text(self):
        if self.p0 is None:
            return tr("Click the first corner of the rounded rectangle")
        if self.box is None:
            return tr("Click the opposite corner, or type width;height such as 30;20")
        return tr("Drag the radius of the corners, click it, or type a radius such as r5")

    def vcb_label(self):
        return tr("Dimensions") if self.box is None else tr("Radius")

    def mouse_move(self, e, x, y):
        self.snap_at(x, y, anchor=self.p0)
        if self.p0 is not None and self.box is None:
            w, h = self.size()
            self.set_vcb(f"{w:.2f}; {h:.2f}")
        elif self.box is not None:
            self.radius = self.radius_at(x, y)
            self.set_vcb(format_length(self.radius))
        self.view.update()

    def radius_at(self, x, y) -> float:
        """The radius is the distance from the mouse to the nearest corner."""
        raw = self.view.to_world_xy(x, y)
        x0, y0, x1, y1 = self.box
        best = min(math.hypot(raw[0] - cx, raw[1] - cy)
                   for cx, cy in ((x0, y0), (x1, y0), (x1, y1), (x0, y1)))
        return max(0.0, min(best, 0.5 * min(x1 - x0, y1 - y0)))

    def mouse_press(self, e, x, y):
        if e.button() != Qt.LeftButton:
            return
        if self.box is not None:
            self.finish(self.radius)
            self.view.update()
            return
        self.mouse_move(e, x, y)
        if self.p0 is None:
            self.p0 = np.asarray(self.cursor, dtype=float)
        else:
            self.set_box(self.p0, self.cursor)
        self.view.update()

    def set_box(self, a, b):
        self.box = (min(float(a[0]), float(b[0])), min(float(a[1]), float(b[1])),
                    max(float(a[0]), float(b[0])), max(float(a[1]), float(b[1])))
        self.radius = 0.0

    def finish(self, radius=None):
        if self.box is None:
            RectangleSheetTool.finish(self, radius)
            return
        x0, y0, x1, y1 = self.box
        r = float(self.radius if radius is None else radius)
        loop = rounded_rect_loop((x0, y0), (x1, y1), r)
        self.reset()
        if loop is not None and abs(polygon_area(loop)) > 1e-9:
            self.add_shapes([(loop, True, {}, False)])

    def vcb_enter(self, text):
        t = text.strip().lower()
        if t.startswith("r"):
            r = parse_length(t[1:])
            if self.box is None:
                raise ValueError(tr("first click the rectangle, then the radius"))
            self.finish(r)
            return
        if self.box is None:
            RectangleSheetTool.vcb_enter(self, text)
            return
        self.finish(parse_length(t))

    def loop(self):
        if self.box is None:
            return RectangleSheetTool.loop(self)
        x0, y0, x1, y1 = self.box
        return rounded_rect_loop((x0, y0), (x1, y1), self.radius)


# --------------------------------------------------------------------------
#  circle, ellipse, polygon
# --------------------------------------------------------------------------

class CircleSheetTool(DrawSheetTool):
    """Click the centre and click or type the radius; "12s" sets the number of sides."""

    name = "circle"
    status = tr("Click the centre of the circle")
    measure_label = tr("Radius")
    default_segments = None

    def reset(self):
        super().reset()
        self.center = None
        self.radius = None
        if not hasattr(self, "segments"):
            self.segments = self.default_segments

    def status_text(self):
        if self.center is None:
            return tr("Click the centre of the circle, or type the number of sides such as 12s")
        return tr("Click the radius of the circle, or type a radius")

    def vcb_label(self):
        return tr("Sides") if self.center is None else tr("Radius")

    def mouse_move(self, e, x, y):
        self.snap_at(x, y, anchor=self.center)
        if self.center is not None:
            self.radius = float(np.hypot(*(self.cursor - self.center)))
            self.set_vcb(format_length(self.radius))
        self.view.update()

    def mouse_press(self, e, x, y):
        if e.button() != Qt.LeftButton:
            return
        self.mouse_move(e, x, y)
        if self.center is None:
            self.center = np.asarray(self.cursor, dtype=float)
        else:
            self.finish()
        self.view.update()

    def loop(self):
        if self.center is None or not self.radius:
            return None
        return circle_loop(self.center, self.radius, self.segments)

    def finish(self, radius=None):
        if self.center is None:
            return
        r = float(self.radius if radius is None else radius)
        center = np.asarray(self.center, dtype=float)
        segs = self.segments
        self.reset()
        if r > 1e-9:
            self.add_shapes([(circle_loop(center, r, segs), True,
                              {"kind": "circle", "center": center.tolist(),
                               "radius": r}, True)])

    def vcb_enter(self, text):
        t = text.strip().lower()
        if t.endswith("s"):
            n = int(parse_number(t[:-1]))
            if n < 3:
                raise ValueError(tr("at least 3 sides"))
            self.segments = n
            self.view.set_status(tr("{n} sides; the circle is flattened with them.").format(n=n))
            return
        if self.center is None:
            raise ValueError(tr("first click the centre of the circle"))
        self.finish(abs(parse_length(t)))

    def preview(self, p):
        loop = self.loop()
        if loop is not None:
            self.draw_line(p, loop, closed=True)
        if self.center is not None and self.cursor is not None:
            self.draw_rubber(p, self.center, self.cursor)
            p.setBrush(QColor("#1f3b63"))
            p.drawEllipse(self.view.to_screen(self.center), 2.0, 2.0)
            p.setBrush(Qt.NoBrush)


class EllipseSheetTool(CircleSheetTool):
    """Click the centre and drag the semi-axes; "30;20" are the two semi-axes."""

    name = "ellipse"
    status = tr("Click the centre of the ellipse")
    measure_label = tr("Semi-axes")

    def reset(self):
        super().reset()
        self.rx = self.ry = None

    def status_text(self):
        if self.center is None:
            return tr("Click the centre of the ellipse")
        return tr("Drag the ellipse or type the two semi-axes, for example 30;20")

    def vcb_label(self):
        return tr("Semi-axes")

    def mouse_move(self, e, x, y):
        self.snap_at(x, y, anchor=self.center)
        if self.center is not None:
            self.rx, self.ry = self.radii()
            self.set_vcb(f"{self.rx:.2f}; {self.ry:.2f}")
        self.view.update()

    def radii(self):
        d = np.asarray(self.cursor, dtype=float) - np.asarray(self.center, dtype=float)
        return abs(float(d[0])), abs(float(d[1]))

    def mouse_press(self, e, x, y):
        if e.button() != Qt.LeftButton:
            return
        self.mouse_move(e, x, y)
        if self.center is None:
            self.center = np.asarray(self.cursor, dtype=float)
        else:
            self.finish()
        self.view.update()

    def loop(self):
        if self.center is None or not self.rx or not self.ry:
            return None
        return ellipse_loop(self.center, self.rx, self.ry, 0.0, self.segments)

    def finish(self, radii=None):
        if self.center is None:
            return
        rx, ry = (self.rx, self.ry) if radii is None else (abs(float(radii[0])),
                                                           abs(float(radii[1])))
        center = np.asarray(self.center, dtype=float)
        segs = self.segments
        self.reset()
        if rx > 1e-9 and ry > 1e-9:
            self.add_shapes([(ellipse_loop(center, rx, ry, 0.0, segs), True,
                              {"kind": "ellipse", "center": center.tolist(),
                               "rx": rx, "ry": ry, "rot": 0.0}, True)])

    def vcb_enter(self, text):
        if self.center is None:
            raise ValueError(tr("first click the centre of the ellipse"))
        vals = split_values(text, (2,))
        if len(vals) != 2:
            raise ValueError(tr("give the two semi-axes, for example 30;20"))
        self.finish((abs(parse_length(vals[0])), abs(parse_length(vals[1]))))


class PolygonSheetTool(CircleSheetTool):
    """Regular polygon: number of sides with "6s", click or type the radius."""

    name = "polygon"
    status = tr("Click the centre of the polygon")
    measure_label = tr("Radius")
    default_segments = 6

    def status_text(self):
        if self.center is None:
            return tr("Click the centre of the polygon, or type the number of sides such as 6s")
        return tr("Click a corner of the polygon, or type a radius")

    def vcb_label(self):
        return tr("Sides") if self.center is None else tr("Radius")


# --------------------------------------------------------------------------
#  arcs
# --------------------------------------------------------------------------

class ArcSheetTool(DrawSheetTool):
    """Arc through three points: start, a point on the arc and the end point."""

    name = "arc"
    status = tr("Click the start point of the arc")
    measure_label = tr("Radius")

    def reset(self):
        super().reset()
        self.pts: list[np.ndarray] = []

    def status_text(self):
        if not self.pts:
            return tr("Click the start point of the arc")
        if len(self.pts) == 1:
            return tr("Click a point on the arc")
        return tr("Click the end point of the arc")

    def mouse_move(self, e, x, y):
        self.snap_at(x, y, anchor=self.pts[-1] if self.pts else None)
        got = self.points()
        if got is not None:
            self.set_vcb(format_length(got[1]))
        self.view.update()

    def mouse_press(self, e, x, y):
        if e.button() != Qt.LeftButton:
            return
        self.mouse_move(e, x, y)
        if self.pts and float(np.hypot(*(self.cursor - self.pts[-1]))) <= 1e-6:
            return
        self.pts.append(np.asarray(self.cursor, dtype=float))
        if len(self.pts) == 3:
            self.finish()
        self.view.update()

    def points(self):
        if len(self.pts) < 2:
            return None
        third = self.cursor if len(self.pts) == 2 else None
        trio = list(self.pts) + ([third] if third is not None else [])
        if len(trio) < 3:
            return None
        return arc_through_3(trio[0], trio[1], trio[2])

    def loop(self):
        got = self.points()
        if got is None:
            return None
        center, r, a0, am, a1 = got
        span = arc_span(a0, am, a1)
        if abs(span) < 1e-9:
            return None
        pts = arc_points(center, r, a0, a0 + span, CURVE_TOL)
        return np.ascontiguousarray(pts, dtype=float), center, r, a0, a0 + span

    def finish(self):
        got = self.loop()
        self.reset()
        if got is None:
            return
        pts, center, r, a0, a1 = got
        self.add_shapes([(pts, False, {"kind": "arc", "center": center.tolist(),
                                       "radius": r, "a0": float(a0), "a1": float(a1)},
                          False)])

    def preview(self, p):
        got = self.loop()
        if got is not None:
            self.draw_line(p, got[0])
        elif self.pts:
            pts = list(self.pts) + ([self.cursor] if self.cursor is not None else [])
            self.draw_line(p, pts)


class CenterArcSheetTool(DrawSheetTool):
    """Centre arc: centre, radius and then the angle (click or type degrees)."""

    name = "centerarc"
    status = tr("Click the centre of the arc")
    measure_label = tr("Radius")
    default_segments = None

    def reset(self):
        super().reset()
        self.center = None
        self.start = None
        self.angle = 0.0
        if not hasattr(self, "segments"):
            self.segments = self.default_segments

    def status_text(self):
        if self.center is None:
            return tr("Click the centre of the arc")
        if self.start is None:
            return tr("Click the start point of the arc, or type a radius")
        return tr("Click the end point of the arc, or type an angle in degrees "
                  "(counter-clockwise positive)")

    def vcb_label(self):
        return tr("Radius") if self.start is None else tr("Angle")

    def radius(self):
        return None if self.start is None else float(np.hypot(*(self.start - self.center)))

    def angle_at(self, p) -> float:
        ref = np.asarray(self.start, dtype=float) - self.center
        v = np.asarray(p, dtype=float) - self.center
        if float(np.hypot(*ref)) <= 1e-9 or float(np.hypot(*v)) <= 1e-9:
            return self.angle
        a = math.atan2(float(ref[0] * v[1] - ref[1] * v[0]), float(ref @ v))
        return math.degrees(a)

    def mouse_move(self, e, x, y):
        self.snap_at(x, y, anchor=self.center)
        if self.center is not None and self.start is None:
            self.set_vcb(format_length(float(np.hypot(*(self.cursor - self.center)))))
        elif self.center is not None:
            self.angle = self.angle_at(self.cursor)
            self.set_vcb(f"{self.angle:.1f}°")
        self.view.update()

    def mouse_press(self, e, x, y):
        if e.button() != Qt.LeftButton:
            return
        self.mouse_move(e, x, y)
        if self.center is None:
            self.center = np.asarray(self.cursor, dtype=float)
        elif self.start is None:
            if float(np.hypot(*(self.cursor - self.center))) <= 1e-9:
                return
            self.start = np.asarray(self.cursor, dtype=float)
        else:
            self.finish()
        self.view.update()

    def loop(self):
        r = self.radius()
        if self.center is None or r is None or abs(self.angle) < 1e-9:
            return None
        a0 = math.atan2(float(self.start[1] - self.center[1]),
                        float(self.start[0] - self.center[0]))
        span = math.radians(self.angle)
        if self.segments:
            n = max(2, int(self.segments))
            t = a0 + np.linspace(0.0, span, n + 1)
            pts = self.center + np.column_stack([r * np.cos(t), r * np.sin(t)])
        else:
            pts = arc_points(self.center, r, a0, a0 + span, CURVE_TOL)
        return np.ascontiguousarray(pts, dtype=float), a0, a0 + span

    def finish(self, angle=None):
        if angle is not None:
            self.angle = float(angle)
        got = self.loop()
        center = None if self.center is None else np.asarray(self.center, dtype=float)
        r = self.radius()
        self.reset()
        if got is None or center is None:
            return
        pts, a0, a1 = got
        self.add_shapes([(pts, False, {"kind": "arc", "center": center.tolist(),
                                       "radius": r, "a0": float(a0), "a1": float(a1)},
                          False)])

    def vcb_enter(self, text):
        t = text.strip().lower()
        if t.endswith("s"):
            n = int(parse_number(t[:-1]))
            if n < 2:
                raise ValueError(tr("at least 2 segments"))
            self.segments = n
            self.view.set_status(tr("{n} segments.").format(n=n))
            return
        if self.center is None:
            raise ValueError(tr("first click the centre of the arc"))
        if self.start is None:
            raw = self.snap.raw if self.snap is not None else self.cursor
            d = np.asarray(raw, dtype=float) - self.center
            if float(np.hypot(*d)) <= 1e-9:
                d = np.array([1.0, 0.0])
            self.start = self.center + d / float(np.hypot(*d)) * abs(parse_length(t))
            return
        self.finish(parse_number(t.replace("°", "")))

    def preview(self, p):
        got = self.loop()
        if got is not None:
            self.draw_line(p, got[0])
        if self.center is not None and self.cursor is not None:
            self.draw_rubber(p, self.center, self.cursor)


# --------------------------------------------------------------------------
#  text
# --------------------------------------------------------------------------

class TextSheetTool(DrawSheetTool):
    """Text as contours on the worksheet: one undo step, with the holes of o and a."""

    name = "text"
    status = tr("Click where the text should start")
    measure_label = tr("Height")

    def reset(self):
        super().reset()
        self.preview_loops = None

    def __init__(self, view):
        super().__init__(view)
        self.loops = None
        self.text = ""
        self.family = ""
        self.height = 10.0
        self.bold = False

    def status_text(self):
        if not self.loops:
            return tr("First choose the text (Text button or the dialog), then click the place")
        return tr("Click the place for “{text}” (capitals {height:g} mm)").format(
            text=self.text, height=self.height)

    def set_text(self, text, family=None, height=10.0, bold=False):
        """The contours of `text`; capital height in millimetres."""
        self.text = str(text or "")
        self.family = family or default_family()
        self.height = max(0.1, float(height))
        self.bold = bool(bold)
        self.loops = text_loops(self.text, self.family, self.height, self.bold)
        self.preview_loops = None
        return bool(self.loops)

    def ask_text(self):
        win = self.view.window()
        if hasattr(win, "ask_sheet_text"):
            win.ask_sheet_text()
        return bool(self.loops)

    def mouse_move(self, e, x, y):
        self.snap_at(x, y)
        self.preview_loops = None
        if self.loops:
            off = np.asarray(self.cursor, dtype=float)
            self.preview_loops = [L + off for L in self.loops]
        self.view.update()

    def mouse_press(self, e, x, y):
        if e.button() != Qt.LeftButton:
            return
        if not self.loops:
            self.ask_text()
            self.view.update()
            return
        self.mouse_move(e, x, y)
        off = np.asarray(self.cursor, dtype=float)
        shapes = [(L + off, True, {}, True) for L in self.loops]
        self.add_shapes(shapes)
        self.view.set_status(tr("text “{text}” placed").format(text=self.text))
        self.view.update()

    def vcb_enter(self, text):
        h = abs(parse_length(text))
        if not self.text:
            raise ValueError(tr("first choose the text you want to place"))
        self.set_text(self.text, self.family, h, self.bold)
        self.view.set_status(tr("text height {h:g} mm").format(h=h))

    def preview(self, p):
        for L in self.preview_loops or ():
            self.draw_line(p, L, closed=True, color="#1f3b63")


# --------------------------------------------------------------------------
#  offset
# --------------------------------------------------------------------------

class OffsetSheetTool(DrawSheetTool):
    """Offset a closed path inwards or outwards with `mf.CrossSection.offset`."""

    name = "offset"
    status = tr("Click a closed contour you want to offset")
    measure_label = tr("Distance")

    def reset(self):
        super().reset()
        self.target = None
        self.distance = 0.0

    def status_text(self):
        if self.target is None:
            return tr("Click a closed contour you want to offset")
        if self.view.offset_round:
            return tr("Drag the offset or type a distance (rounded corners)")
        return tr("Drag the offset or type a distance (sharp corners)")

    def mouse_move(self, e, x, y):
        if self.target is None:
            self.view.hover = self.view.path_at(x, y)
            self.view.update()
            return
        p = self.view.to_world_xy(x, y)
        self.cursor = p
        self.distance = self.signed_distance(p)
        self.set_vcb(format_length(abs(self.distance)))
        self.view.update()

    def signed_distance(self, p):
        """Positive outside the contour, negative inside it (as in 3D)."""
        from .vector2d import _point_in_poly
        pts = np.asarray(p, dtype=float)
        best = np.inf
        for a, b in path_segments(self.target.pts, True):
            d = b - a
            L2 = float(d @ d)
            t = 0.0 if L2 <= 1e-30 else max(0.0, min(1.0, float((pts - a) @ d) / L2))
            best = min(best, float(np.hypot(*(a + t * d - pts))))
        inside = _point_in_poly(np.asarray(self.target.pts, dtype=float), pts)
        return -best if inside else best

    def mouse_press(self, e, x, y):
        if e.button() != Qt.LeftButton:
            return
        if self.target is None:
            path = self.view.path_at(x, y)
            if path is None:
                self.warn(tr("Click a closed contour to offset it."))
                return
            if not path.closed or len(path.pts) < 3:
                self.warn(tr("Only a closed contour can be offset."))
                return
            self.target = path
            self.distance = 0.0
            self.view.set_status(self.status_text())
            self.view.update()
            return
        self.apply(self.distance)

    def cross_section(self):
        return mf.CrossSection([np.ascontiguousarray(np.asarray(self.target.pts, float))],
                               mf.FillRule.NonZero)

    def shape(self, off):
        join = mf.JoinType.Round if self.view.offset_round else mf.JoinType.Miter
        cs = self.cross_section().offset(float(off), join, 2.0)
        return [np.asarray(q, dtype=float) for q in cs.to_polygons()]

    def apply(self, off):
        target = self.target
        off = float(off)
        if target is None or abs(off) < 1e-6:
            self.warn(tr("Give a distance greater than zero to offset."))
            return
        try:
            loops = self.shape(off)
        except Exception as exc:                    # manifold3d may refuse a distance that is too large
            self.warn(tr("Offset failed: {error}").format(error=exc))
            return
        layer, src = target.layer, next_src(self.sheet.paths)
        made: list[VPath] = []

        def change():
            for loop in loops:
                made.append(self.sheet.add_path(loop, True, layer, True, "nonzero", src))
        self.edit(change)
        self.reset()
        self.view.set_selection(made)
        self.view.set_status((tr("contour offset {d:.2f} mm inwards") if off < 0
                              else tr("contour offset {d:.2f} mm outwards")).format(d=abs(off)))
        self.view.update()

    def vcb_enter(self, text):
        if self.target is None:
            raise ValueError(tr("first click a closed contour"))
        v = parse_length(text)
        self.apply(-abs(v) if self.distance < 0 else abs(v))

    def preview(self, p):
        if self.target is None or abs(self.distance) < 1e-6:
            return
        try:
            loops = self.shape(self.distance)
        except Exception:
            return
        for loop in loops:
            self.draw_line(p, loop, closed=True, color="#1f3b63")


SHEET_TOOLS = [SelectSheetTool, MoveSheetTool, RotateSheetTool, ScaleSheetTool,
               LineSheetTool, RectangleSheetTool, RoundedRectSheetTool, CircleSheetTool,
               EllipseSheetTool, ArcSheetTool, CenterArcSheetTool, PolygonSheetTool,
               TextSheetTool, OffsetSheetTool]


# --------------------------------------------------------------------------
#  operations on the selection
# --------------------------------------------------------------------------

BOOLEAN_OPS = {"union": mf.OpType.Add, "subtract": mf.OpType.Subtract,
               "intersect": mf.OpType.Intersect}

ALIGN_MODES = ("left", "center", "right", "top", "bottom")
ALIGN_LABELS = {"left": tr("left"), "center": tr("middle"), "right": tr("right"),
                "top": tr("top"), "bottom": tr("bottom")}


def closed_paths(paths):
    """The paths that can be a shape: closed and with at least three points."""
    return [p for p in paths if p.closed and len(p.pts) >= 3]


def boolean_loops(paths, op) -> list[np.ndarray]:
    """Union, subtract or intersect closed paths with `mf.CrossSection`.

    With `subtract` the first path is the whole and the rest is the hole. The loops
    come out with the outer boundary counter-clockwise and the holes clockwise, exactly
    as `vector2d.regions` and the tools expect.
    """
    if op not in BOOLEAN_OPS:
        raise ValueError(op)
    ps = closed_paths(paths)
    if len(ps) < (2 if op != "union" else 1):
        raise ValueError(tr("select two closed contours first"))
    sections = []
    for p in ps:
        cs = mf.CrossSection([np.ascontiguousarray(np.asarray(p.pts, float))],
                             mf.FillRule.NonZero)
        if not cs.is_empty():
            sections.append(cs)
    if len(sections) < (2 if op != "union" else 1):
        raise ValueError(tr("there is nothing to cut from"))
    if op == "subtract":
        whole = sections[0]
        for other in sections[1:]:
            whole = mf.CrossSection.batch_boolean([whole, other], BOOLEAN_OPS[op])
    elif len(sections) == 1:
        whole = sections[0]
    else:
        whole = mf.CrossSection.batch_boolean(sections, BOOLEAN_OPS[op])
    if whole.is_empty():
        return []
    loops = [np.asarray(q, dtype=float) for q in whole.to_polygons()]
    return [L for L in loops if len(L) >= 3 and abs(polygon_area(L)) > 1e-9]


def split_into_segments(paths) -> list[tuple[np.ndarray, bool]]:
    """Every segment of the paths separately: (points, closed)."""
    out = []
    for p in paths:
        for a, b in path_segments(np.asarray(p.pts, dtype=float), p.closed):
            out.append((np.array([a, b], dtype=float), False))
    return out


def joined_paths(paths) -> list[VPath]:
    """Join open segments into polylines and shapes."""
    return join_open_paths(paths)


def align_deltas(paths, mode) -> list[np.ndarray]:
    """The move per path so they line up (left, middle, right, top, bottom)."""
    if mode not in ALIGN_MODES:
        raise ValueError(mode)
    box = paths_bbox(paths)
    if box is None or len(paths) < 2:
        return [np.zeros(2) for _ in paths]
    xmin, ymin, xmax, ymax = box
    target = {"left": xmin, "right": xmax, "bottom": ymin, "top": ymax,
              "center": 0.5 * (xmin + xmax)}[mode]
    out = []
    for p in paths:
        b = paths_bbox([p])
        if b is None:
            out.append(np.zeros(2))
            continue
        if mode == "left":
            out.append(np.array([target - b[0], 0.0]))
        elif mode == "right":
            out.append(np.array([target - b[2], 0.0]))
        elif mode == "bottom":
            out.append(np.array([0.0, target - b[1]]))
        elif mode == "top":
            out.append(np.array([0.0, target - b[3]]))
        else:
            out.append(np.array([target - 0.5 * (b[0] + b[2]),
                                 0.5 * (ymin + ymax) - 0.5 * (b[1] + b[3])]))
    return out


def distribute_deltas(paths, axis="x") -> list[np.ndarray]:
    """Distribute: the centres get the same spacing everywhere."""
    axis = "x" if str(axis).lower().startswith("x") else "y"
    i = 0 if axis == "x" else 1
    boxes = [(p, paths_bbox([p])) for p in paths]
    if len(paths) < 3 or any(b is None for _p, b in boxes):
        return [np.zeros(2) for _ in paths]
    order = sorted(range(len(paths)), key=lambda k: 0.5 * (boxes[k][1][i] + boxes[k][1][i + 2]))
    first, last = order[0], order[-1]
    a = 0.5 * (boxes[first][1][i] + boxes[first][1][i + 2])
    b = 0.5 * (boxes[last][1][i] + boxes[last][1][i + 2])
    step = (b - a) / (len(paths) - 1)
    out = [np.zeros(2) for _ in paths]
    for rank, k in enumerate(order):
        mid = 0.5 * (boxes[k][1][i] + boxes[k][1][i + 2])
        delta = (a + step * rank) - mid
        out[k][i] = delta
    return out


def grid_offsets(cols, rows, dx, dy) -> list[np.ndarray]:
    """The offsets of a row or grid, without the place of the original."""
    cols, rows = max(1, int(cols)), max(1, int(rows))
    out = []
    for j in range(rows):
        for i in range(cols):
            if i == 0 and j == 0:
                continue
            out.append(np.array([i * float(dx), j * float(dy)]))
    return out

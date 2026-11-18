"""The view cube: the 26 regions of a cube and the camera angle each one looks from.

The functions at the top do the geometry in plain numpy, so it can be tested without
an application; the widget at the bottom only draws what they return."""
from __future__ import annotations

import math
import weakref
from itertools import combinations

import numpy as np
from PySide6.QtCore import (QEasingCurve, QEvent, QLineF, QPointF, QRectF, Qt,
                            QVariantAnimation)
from PySide6.QtGui import QFont, QFontMetrics, QPainter, QPainterPath, QPen, QPolygonF, QTransform
from PySide6.QtWidgets import QMenu, QWidget

from . import errors, theme
from .camera import VIEWS
from .i18n import tr, tr_noop
from .settings import flag, qsettings

# Order matters: it is the order of the faces inside a region, so a cell of the
# front face that touches the top is ("front", "top") and not ("top", "front").
FACES = ("front", "back", "left", "right", "top", "bottom")

NORMALS = {
    "front": (0.0, -1.0, 0.0),
    "back": (0.0, 1.0, 0.0),
    "left": (-1.0, 0.0, 0.0),
    "right": (1.0, 0.0, 0.0),
    "top": (0.0, 0.0, 1.0),
    "bottom": (0.0, 0.0, -1.0),
}

# face -> (outward normal, right, up) in world directions, as the face appears in
# its own VIEWS entry; the two in-plane axes say which neighbour a cell touches.
FACE_AXES = {
    "front": ((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
    "back": ((0.0, 1.0, 0.0), (-1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
    "right": ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
    "left": ((-1.0, 0.0, 0.0), (0.0, -1.0, 0.0), (0.0, 0.0, 1.0)),
    "top": ((0.0, 0.0, 1.0), (-1.0, 0.0, 0.0), (0.0, -1.0, 0.0)),
    "bottom": ((0.0, 0.0, -1.0), (1.0, 0.0, 0.0), (0.0, -1.0, 0.0)),
}

_FACE_OF = {tuple(np.round(np.asarray(v), 6)): f for f, v in NORMALS.items()}

GRID = 3   # cells per face: the middle one is the face, the sides edges, the rest corners


def _perpendicular(group):
    """True if the faces of a group touch each other, so they form an edge or corner."""
    for a, b in combinations(group, 2):
        if abs(float(np.dot(NORMALS[a], NORMALS[b]))) > 1e-9:
            return False
    return True


EDGES = tuple(g for g in combinations(FACES, 2) if _perpendicular(g))
CORNERS = tuple(g for g in combinations(FACES, 3) if _perpendicular(g))
REGIONS = tuple(FACES) + EDGES + CORNERS


def canonical(region):
    """A region as a tuple of faces in the order of FACES, whatever order it came in."""
    return tuple(sorted(region, key=FACES.index))


def _unit(v):
    v = np.asarray(v, dtype=float)
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else v


def region_direction(region):
    """Unit vector from the centre of the cube to a region: the sum of its normals."""
    d = np.zeros(3)
    for face in region:
        d += np.asarray(NORMALS[face])
    return _unit(d)


def view_for(region, keep_yaw=None):
    """(yaw, pitch) of the camera that looks at the cube from a region.

    The six faces give exactly camera.VIEWS. Only straight top and bottom have no
    yaw of their own; like the big CAD programs they then keep the yaw they are
    given, so looking down does not turn the model."""
    region = canonical(region)
    if len(region) == 1 and region[0] in ("top", "bottom"):
        yaw = VIEWS["top"][0] if keep_yaw is None else float(keep_yaw)
        return yaw, VIEWS[region[0]][1]
    d = region_direction(region)
    yaw = math.degrees(math.atan2(d[1], d[0]))
    pitch = math.degrees(math.asin(max(-1.0, min(1.0, float(d[2])))))
    return yaw, pitch


def basis(yaw, pitch):
    """(eye direction, right, up) of a camera with these angles, as camera.Camera has them."""
    y, p = math.radians(yaw), math.radians(pitch)
    dirv = np.array([math.cos(p) * math.cos(y), math.cos(p) * math.sin(y), math.sin(p)])
    f = -dirv
    r = np.cross(f, [0.0, 0.0, 1.0])
    if float(np.linalg.norm(r)) < 1e-6:
        r = np.array([-math.sin(y), math.cos(y), 0.0])
        if p > 0:
            r = -r
    r = _unit(r)
    return dirv, r, np.cross(r, f)


def _face_of(direction):
    return _FACE_OF[tuple(np.round(np.asarray(direction, dtype=float), 6))]


def _screen(v, r, u, h):
    """A point of the cube as widget pixels around the centre (0, 0)."""
    return (float(np.dot(v * h, r)), -float(np.dot(v * h, u)))


def cube_polygons(yaw, pitch, size):
    """The visible cells of the cube as (region, polygon), drawn back to front.

    A polygon is four (x, y) points in widget pixels around the centre (0, 0); the
    cube is drawn in parallel projection whatever the view does, like the big CAD
    programs. Every face is split in a 3x3 grid so a hit test can tell a face, an
    edge and a corner apart."""
    dirv, r, u = basis(yaw, pitch)
    h = size / 2.0
    step = 2.0 / GRID
    cells = []
    for face in FACES:
        n, ax, ay = (np.asarray(v, dtype=float) for v in FACE_AXES[face])
        if float(np.dot(n, dirv)) <= 1e-9:
            continue
        for col in range(GRID):
            for row in range(GRID):
                extra = []
                if col == 0:
                    extra.append(_face_of(-ax))
                elif col == GRID - 1:
                    extra.append(_face_of(ax))
                if row == 0:
                    extra.append(_face_of(-ay))
                elif row == GRID - 1:
                    extra.append(_face_of(ay))
                u0, u1 = -1.0 + col * step, -1.0 + (col + 1) * step
                v0, v1 = -1.0 + row * step, -1.0 + (row + 1) * step
                poly = []
                for cu, cv in ((u0, v0), (u1, v0), (u1, v1), (u0, v1)):
                    poly.append(_screen(n + ax * cu + ay * cv, r, u, h))
                c = (n + ax * (u0 + u1) / 2 + ay * (v0 + v1) / 2) * h
                cells.append((float(np.dot(c, dirv)),
                              canonical((face,) + tuple(extra)), poly))
    cells.sort(key=lambda c: c[0])
    return [(region, poly) for _, region, poly in cells]


def face_frames(yaw, pitch, size):
    """The faces that look towards the eye, as (face, centre, right, up, facing).

    Centre and the two in-plane axes are vectors in widget pixels around (0, 0);
    `facing` is how much the face looks at the eye, 1 straight on and 0 edge on."""
    dirv, r, u = basis(yaw, pitch)
    h = size / 2.0
    out = []
    for face in FACES:
        n, ax, ay = (np.asarray(v, dtype=float) for v in FACE_AXES[face])
        facing = float(np.dot(n, dirv))
        if facing <= 1e-9:
            continue
        out.append((face, _screen(n, r, u, h), _screen(ax, r, u, h), _screen(ay, r, u, h),
                    facing))
    return out


def _inside(poly, x, y):
    """True if the point lies inside the polygon (the even-odd rule)."""
    inside = False
    n = len(poly)
    for i in range(n):
        x0, y0 = poly[i]
        x1, y1 = poly[(i + 1) % n]
        if (y0 > y) != (y1 > y) and x < (x1 - x0) * (y - y0) / (y1 - y0) + x0:
            inside = not inside
    return inside


def hit(cells, x, y):
    """The region of the front-most cell that contains the point (x, y), or None."""
    for region, poly in reversed(cells):
        if _inside(poly, x, y):
            return region
    return None


def shortest_turn(a, b):
    """b - a wrapped to -180..180: the way round a camera animation should turn."""
    return (b - a + 180.0) % 360.0 - 180.0


# ------------------------------------------------------------------------ widget

FACE_LABEL = {"top": tr_noop("TOP"), "bottom": tr_noop("BOTTOM"), "front": tr_noop("FRONT"),
              "back": tr_noop("BACK"), "left": tr_noop("LEFT"), "right": tr_noop("RIGHT")}

# how light every face is drawn, so the cube reads as a block and not as a hexagon
FACE_SHADE = {"top": 1.18, "front": 1.0, "back": 0.94, "right": 0.88, "left": 0.82,
              "bottom": 0.70}

ARROW_DIRS = ((1.0, 0.0), (0.0, -1.0), (-1.0, 0.0), (0.0, 1.0))    # right, up, left, down

HOME_KEY = "view_cube/home"     # + the key of the view: main, print, sla

_cubes = []                     # the live cubes, so View > View Cube can show and hide them


def set_shown(on):
    """Show or hide the cube in every 3D view: View > View Cube."""
    alive = []
    for ref in _cubes:
        cube = ref()
        try:
            if cube is not None:
                cube.setVisible(bool(on))
                alive.append(ref)
        except RuntimeError:                # the window it lived in is gone already
            pass
    _cubes[:] = alive


class ViewCube(QWidget):
    """The cube in the top right corner of a 3D view, like the big CAD programs have.

    It shows the camera of the view it belongs to: click a face, edge or corner to look
    at the model from there, drag the cube to orbit it, click the house to go home. The
    camera of Volura has no roll and never gets one, so the cube only turns yaw and pitch.
    """

    SIZE = 110                # the widget: the cube, the arrows beside it and the house
    CUBE = 52.0               # the edge of the cube in pixels
    MARGIN = 12               # the distance to the corner of the view
    ARROW = 10.0              # the arrows that turn to the neighbouring face
    HOME = QRectF(2, 2, 18, 18)           # the house in the top left
    CLICK_LIMIT = 4.0         # less movement than this and a press is a click
    TURN_MS = 300

    def __init__(self, view, key="main"):
        super().__init__(view)
        self.view = view
        self.key = key                      # which home view in the settings is ours
        self.setFixedSize(int(self.SIZE), int(self.SIZE))
        self.setAttribute(Qt.WA_TranslucentBackground)   # only the cube, no box around it
        self.setCursor(Qt.PointingHandCursor)
        self.setMouseTracking(True)
        self.setAccessibleName(tr("View cube"))
        self.setToolTip(tr("Click a face, edge or corner to look from there; drag to orbit"))
        self._hover = None                  # the cell of the cube under the mouse
        self._arrow = None                  # the arrow under the mouse
        self._house = False
        self._press = self._last = None
        self._dragging = False
        self._from = self._to = None
        self._angles = None                 # the camera the cube was last drawn with
        self._anim = QVariantAnimation(self)
        self._anim.setDuration(self.TURN_MS)
        self._anim.setEasingCurve(QEasingCurve.InOutCubic)
        self._anim.valueChanged.connect(self._turning)
        self._anim.finished.connect(self.finish_turn)
        view.installEventFilter(self)
        _cubes.append(weakref.ref(self))
        self.setVisible(flag("show_view_cube"))
        self._place()

    # ------------------------------------------------------------- what it shows
    def sync(self):
        """Repaint when the camera of the view turned since the last frame it was drawn in."""
        angles = (self.view.cam.yaw, self.view.cam.pitch)
        if angles != self._angles:
            self._angles = angles
            self.update()

    def centre(self):
        """The middle of the cube, in the coordinates of the widget."""
        return QPointF(self.width() / 2.0, self.height() / 2.0)

    def cells(self):
        """The cells of the cube as the camera of the view sees it now."""
        return cube_polygons(self.view.cam.yaw, self.view.cam.pitch, self.CUBE)

    def region_at(self, pos):
        """The region of the cube under a point of the widget, or None."""
        c = self.centre()
        return hit(self.cells(), pos.x() - c.x(), pos.y() - c.y())

    def face_view(self):
        """The face the camera looks at straight on, or None; only then the arrows come."""
        cam = self.view.cam
        for face in FACES:
            yaw, pitch = view_for((face,), keep_yaw=cam.yaw)
            if abs(shortest_turn(cam.yaw, yaw)) <= 0.5 and abs(cam.pitch - pitch) <= 0.5:
                return face
        return None

    def arrows(self):
        """(region, triangle) of the four arrows outside the cube, empty unless a face
        looks straight at the camera; every arrow turns to the neighbour on that side."""
        if self.face_view() is None:
            return []
        _dirv, r, u = basis(self.view.cam.yaw, self.view.cam.pitch)
        out = []
        for sx, sy in ARROW_DIRS:
            w = r * sx - u * sy
            face = max(FACES, key=lambda f: float(np.dot(NORMALS[f], w)))
            out.append(((face,), self._triangle(sx, sy)))
        return out

    def _triangle(self, sx, sy):
        """A small triangle outside the cube, pointing away from its centre."""
        c, h = self.centre(), self.CUBE / 2.0
        px, py = -sy, sx
        tip = c + QPointF(sx * (h + self.ARROW), sy * (h + self.ARROW))
        base = c + QPointF(sx * h, sy * h)
        half = self.ARROW * 0.6
        return [(base.x() + px * half, base.y() + py * half),
                (base.x() - px * half, base.y() - py * half), (tip.x(), tip.y())]

    # ------------------------------------------------------------------- paint
    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setOpacity(1.0 if self.underMouse() else 0.6)     # quiet until the mouse is here
        c = self.centre()
        edge = QPen(theme.qcolor("switch_border"), 1.0)
        for region, poly in self.cells():
            p.setPen(edge)
            p.setBrush(self._fill(region))
            p.drawPolygon(QPolygonF([c + QPointF(x, y) for x, y in poly]))
        self._paint_labels(p, c)
        for i, (_region, tri) in enumerate(self.arrows()):
            p.setPen(edge)
            p.setBrush(theme.qcolor("switch_on", 210) if self._arrow == i
                       else theme.qcolor("tray_bg", 235))
            p.drawPolygon(QPolygonF([QPointF(x, y) for x, y in tri]))
        self._paint_house(p)
        p.end()

    def _fill(self, region):
        """The color of a cell: its face, lighter on top; blue when the mouse is on it."""
        if region == self._hover:
            return theme.qcolor("switch_on", 210)
        col, k = theme.qcolor("tray_bg"), FACE_SHADE[region[0]]
        return col.lighter(int(100 * k)) if k >= 1.0 else col.darker(int(100 / k))

    def _paint_labels(self, p, c):
        """The name of every face that looks at the eye, drawn on the face so it turns with it."""
        f = QFont(self.font())
        f.setPixelSize(10)
        f.setBold(True)
        fm = QFontMetrics(f)
        h = self.CUBE / 2.0
        p.setPen(Qt.NoPen)
        p.setBrush(theme.qcolor("view_text"))
        frames = face_frames(self.view.cam.yaw, self.view.cam.pitch, self.CUBE)
        for face, centre, ax, ay, facing in frames:
            if facing < 0.25:
                continue                    # too much on its edge to read
            label = tr(FACE_LABEL[face])
            path = QPainterPath()
            path.addText(-fm.horizontalAdvance(label) / 2.0, fm.ascent() / 2.0, f, label)
            m = QTransform(ax[0] / h, ax[1] / h, -ay[0] / h, -ay[1] / h,
                           c.x() + centre[0], c.y() + centre[1])
            p.drawPath(m.map(path))

    def _paint_house(self, p):
        """The house in the top left: the way back to the home view."""
        r = self.HOME
        p.setPen(QPen(theme.qcolor("switch_border"), 1.0))
        p.setBrush(theme.qcolor("switch_on", 210) if self._house else theme.qcolor("tray_bg", 235))
        p.drawRoundedRect(r, 4.0, 4.0)
        p.setPen(QPen(theme.qcolor("view_text"), 1.4))
        p.setBrush(Qt.NoBrush)
        p.drawPolygon(QPolygonF([QPointF(r.left() + 3.5, r.top() + 9.0),
                                 QPointF(r.center().x(), r.top() + 4.0),
                                 QPointF(r.right() - 3.5, r.top() + 9.0)]))
        p.drawRect(QRectF(r.left() + 5.5, r.top() + 9.0, r.width() - 11.0, r.height() - 12.0))

    # ------------------------------------------------------------- interaction
    def _place(self):
        """Keep the cube in the top right corner of the view."""
        self.move(self.view.width() - self.width() - self.MARGIN, self.MARGIN)
        self.raise_()

    @errors.safe_handler(ret=False, boolean=True)
    def eventFilter(self, obj, e):
        if obj is self.view and e.type() in (QEvent.Resize, QEvent.Show):
            self._place()
        return super().eventFilter(obj, e)

    def enterEvent(self, event):
        self.update()
        super().enterEvent(event)

    def leaveEvent(self, event):
        self._hover = self._arrow = None
        self._house = False
        self.update()
        super().leaveEvent(event)

    @errors.safe_handler
    def mousePressEvent(self, e):
        if e.button() == Qt.RightButton:
            self.menu()
            return
        if e.button() != Qt.LeftButton:
            return
        self._press = self._last = e.position()
        self._dragging = False

    @errors.safe_handler
    def mouseMoveEvent(self, e):
        pos = e.position()
        if self._press is None:
            self._set_hover(pos)
        else:
            if not self._dragging and QLineF(self._press, pos).length() > self.CLICK_LIMIT:
                self._dragging = True       # from here it orbits, live and without animation
                self._anim.stop()
                self._from = self._to = None
            if self._dragging and self._last is not None:
                self.view.cam.orbit(pos.x() - self._last.x(), pos.y() - self._last.y())
                self.view.update()
                self.update()
        self._last = pos

    @errors.safe_handler
    def mouseReleaseEvent(self, e):
        if e.button() != Qt.LeftButton or self._press is None:
            return
        dragging, self._press, self._dragging = self._dragging, None, False
        if dragging:
            self._commit()
        else:
            self.click(e.position())

    def click(self, pos):
        """A press without dragging: the house, an arrow or the region under the mouse."""
        if self.HOME.contains(pos):
            self.go_home()
            return
        for region, tri in self.arrows():
            if _inside(tri, pos.x(), pos.y()):
                self.turn_to(region)
                return
        region = self.region_at(pos)
        if region is not None:
            self.turn_to(region)

    def _set_hover(self, pos):
        """What is under the mouse: the house, an arrow or a cell of the cube."""
        self._house = self.HOME.contains(pos)
        self._arrow = None
        for i, (_region, tri) in enumerate(self.arrows()):
            if _inside(tri, pos.x(), pos.y()):
                self._arrow = i
                break
        self._hover = None if (self._house or self._arrow is not None) else self.region_at(pos)
        self.update()

    # ------------------------------------------------------------------ camera
    def turn_to(self, region):
        """Look at the model from a region of the cube, the shortest way round."""
        cam = self.view.cam
        yaw, pitch = view_for(region, keep_yaw=cam.yaw)
        if abs(shortest_turn(cam.yaw, yaw)) < 1e-9 and abs(cam.pitch - pitch) < 1e-9:
            return                          # this is the view you are already looking at
        self._from = (cam.yaw, cam.pitch)
        self._to = (yaw, pitch)
        self._anim.stop()
        self._anim.setStartValue(0.0)
        self._anim.setEndValue(1.0)
        self._anim.start()

    def _turning(self, value):
        """One frame of the turn."""
        if self._from is None or self._to is None:
            return
        t = float(value)
        (yaw0, pitch0), (yaw1, pitch1) = self._from, self._to
        self.view.cam.yaw = yaw0 + shortest_turn(yaw0, yaw1) * t
        self.view.cam.pitch = pitch0 + (pitch1 - pitch0) * t
        self.view.update()
        self.update()

    def finish_turn(self):
        """Stop the turn on its target and let the view remember it, as View > Front does."""
        to, self._to, self._from = self._to, None, None
        self._anim.stop()
        if to is not None:
            self.view.cam.yaw, self.view.cam.pitch = to
        self._commit()
        self.view.update()
        self.update()

    def _commit(self):
        """The view remembers the camera as the newest step of its history."""
        commit = getattr(self.view, "commit_camera", None)
        if commit is not None:
            commit()

    def go_home(self):
        """The house: the home view, by default iso with everything in view."""
        yaw, pitch, fit = self.home()
        self._anim.stop()
        self._from = self._to = None
        self.view.cam.yaw, self.view.cam.pitch = yaw, pitch
        if fit:
            if hasattr(self.view, "zoom_extents"):
                self.view.zoom_extents()
            else:
                self.view.fit(None)
        self._commit()
        self.view.update()
        self.update()

    def set_projection(self, ortho):
        """Camera > Parallel Projection and Perspective, for the view this cube is in."""
        win = self.view.window()
        if hasattr(win, "apply_projection") and hasattr(win, "a_ortho"):
            win.a_ortho.setChecked(bool(ortho))
            win.a_persp.setChecked(not ortho)
            win.apply_projection()          # the window keeps its own actions in step
            self.update()
            return
        set_projection = getattr(self.view, "set_projection", None)
        if set_projection is not None:
            set_projection(ortho)
        else:
            self.view.cam.ortho = bool(ortho)
            self.view.update()
        self.update()

    # ------------------------------------------------------------- the home view
    def home_key(self):
        return f"{HOME_KEY}/{self.key}"

    def custom_home(self):
        """The stored home view as (yaw, pitch, zoom extents), or None when there is none."""
        parts = str(qsettings().value(self.home_key(), "")).split(",")
        try:
            return float(parts[0]), float(parts[1]), parts[2] == "1"
        except ValueError:
            return None

    def home(self):
        """(yaw, pitch, zoom extents) of the home view: the stored one, else iso."""
        return self.custom_home() or (VIEWS["iso"][0], VIEWS["iso"][1], True)

    def store_home(self):
        """Remember the camera of the view as its home view."""
        cam = self.view.cam
        qsettings().setValue(self.home_key(), f"{cam.yaw},{cam.pitch},1")

    def reset_home(self):
        """Back to the home view Volura starts with: iso with everything in view."""
        qsettings().remove(self.home_key())

    def menu(self):
        """The right-click menu of the cube."""
        ortho = bool(self.view.cam.ortho)
        menu = QMenu(self)
        menu.addAction(tr("Home View"), self.go_home)
        menu.addAction(tr("Set Current View as Home"), self.store_home)
        reset = menu.addAction(tr("Reset Home View"), self.reset_home)
        reset.setEnabled(self.custom_home() is not None)
        menu.addSeparator()
        persp = menu.addAction(tr("Perspective"))
        persp.setCheckable(True)
        persp.setChecked(not ortho)
        par = menu.addAction(tr("Orthographic"))
        par.setCheckable(True)
        par.setChecked(ortho)
        chosen = menu.exec(self.mapToGlobal(self.rect().bottomLeft()))
        if chosen is persp:
            self.set_projection(False)
        elif chosen is par:
            self.set_projection(True)

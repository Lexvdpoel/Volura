"""The 3D viewport: OpenGL rendering, navigation and event routing to tools."""
from __future__ import annotations

import numpy as np
from OpenGL import GL as gl
from OpenGL.GL import GLubyte
from PySide6.QtCore import QEvent, QPointF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPainterPath, QPen, QSurfaceFormat
from PySide6.QtOpenGLWidgets import QOpenGLWidget

from . import errors, icons
from .camera import Camera, horizon_line
from .inference import InferenceEngine
from .tools import MEASURE_LABEL

VCB_CHARS = set("0123456789.,;-+/*xXsS\"'mcin<>[]=")
SELECT_COLOR = (0.13, 0.35, 1.0)
LOCK_COLOR = (0.80, 0.13, 0.10)        # a selected locked group gets a red frame, not a blue one
HIDDEN_EDGE_COLOR = (0.45, 0.45, 0.55, 0.85)
PROBLEM_COLOR = (0.86, 0.06, 0.06, 1.0)   # de problemen van Printcontrole
FRONT = (1.0, 1.0, 1.0)                # the front of a face is white, like SketchUp
BACK = (0.64, 0.70, 0.78)              # the back of a face is blue grey

# The SketchUp background: a light blue grey sky that fades to almost white at the
# horizon, and a grey ground below it.
SKY_TOP = (0.62, 0.72, 0.82)
SKY_HORIZON = (0.97, 0.97, 0.96)
GROUND_COLOR = (204 / 255, 203 / 255, 198 / 255)

# The inference markers of SketchUp: per soort een vorm, getekend in 2D op de cursor.
INFERENCE_COLORS = {
    "endpoint": (0.10, 0.72, 0.10, 1), "midpoint": (0.10, 0.72, 0.85, 1),
    "origin": (0.98, 0.80, 0.10, 1),
    "center": (0.10, 0.72, 0.85, 1), "edge": (0.92, 0.10, 0.10, 1),
    "guide": (0.10, 0.70, 0.85, 1), "guidepoint": (0.10, 0.70, 0.85, 1),
    "face": (0.20, 0.30, 0.95, 1), "region": (0.20, 0.30, 0.95, 1),
    "lock": (0.60, 0.20, 0.70, 1),
}
INFERENCE_SHAPES = {
    "endpoint": "ring", "origin": "ring", "midpoint": "dot", "center": "dot",
    "guidepoint": "dot", "edge": "square", "guide": "square",
    "face": "diamond", "region": "diamond", "lock": "diamond",
}
TOOLTIP_FILL = (255, 255, 153)         # the yellow SketchUp tooltip
TOOLTIP_BORDER = (140, 130, 40)

# Beeld > Vlakstijl (SketchUp's face styles); SHADED is the default.
WIREFRAME = "wireframe"
HIDDENLINE = "hiddenline"
SHADED = "shaded"
MONOCHROME = "monochrome"
FACE_STYLES = (WIREFRAME, HIDDENLINE, SHADED, MONOCHROME)
HIDDEN_FACE = (1.0, 1.0, 1.0)          # hidden line: faces in the white background colour
BACK_EDGE_COLOR = (0.60, 0.60, 0.60, 0.85)
SOFT_EDGE_COLOR = (0.35, 0.50, 0.85, 0.9)
GUIDE_COLOR = (0.52, 0.52, 0.58, 1)    # hulplijnen: dashed light grey, like SketchUp


def hex_rgb(h):
    c = QColor(h)
    return c.redF(), c.greenF(), c.blueF()


def _tint(col, on):
    """Blend a solid face colour with the selection colour, as SketchUp tints selected faces."""
    if not on:
        return col
    return tuple(c * 0.8 + s * 0.2 for c, s in zip(col, SELECT_COLOR))


_STIPPLE = None


def stipple_pattern():
    """32x32 polygon stipple of loose dots, every fourth pixel (SketchUp's selection)."""
    global _STIPPLE
    if _STIPPLE is None:
        on = [1 if (x % 4 == 0 and y % 4 == 0) else 0 for y in range(32) for x in range(32)]
        _STIPPLE = (GLubyte * len(on))(*on)
    return _STIPPLE


def _clip_below(poly, a, b):
    """The part of the convex polygon `poly` on the lower side of the line a -> b (y up)."""
    dx, dy = b[0] - a[0], b[1] - a[1]

    def side(p):
        return dx * (p[1] - a[1]) - dy * (p[0] - a[0])

    out = []
    for i, p in enumerate(poly):
        q = poly[(i + 1) % len(poly)]
        sp, sq = side(p), side(q)
        if sp <= 0:
            out.append(p)
            if sq > 0:
                out.append(_crossing(p, q, a, b))
        elif sq <= 0:
            out.append(_crossing(p, q, a, b))
    return out


def _crossing(p, q, a, b):
    """Where the segment p -> q crosses the line a -> b."""
    ex, ey = q[0] - p[0], q[1] - p[1]
    dx, dy = b[0] - a[0], b[1] - a[1]
    den = ex * dy - ey * dx
    if abs(den) < 1e-12:
        return p
    t = ((a[0] - p[0]) * dy - (a[1] - p[1]) * dx) / den
    return (p[0] + t * ex, p[1] + t * ey)


def surface_format():
    fmt = QSurfaceFormat()
    fmt.setDepthBufferSize(24)
    fmt.setSamples(4)
    fmt.setProfile(QSurfaceFormat.CompatibilityProfile)
    return fmt


def states_equal(a, b):
    """Compare two Camera.state() tuples."""
    return a[5] == b[5] and all(abs(x - y) < 1e-6 for x, y in zip(a[0] + a[1:5], b[0] + b[1:5]))


class Viewport(QOpenGLWidget):
    status_changed = Signal(str)
    vcb_changed = Signal(str, str)      # label, value
    tool_changed = Signal(str)
    context_menu = Signal(object)       # global position
    cam_history_changed = Signal(bool, bool)   # can go back, can go forward

    CAM_HISTORY_MAX = 50
    CAM_SCROLL_SETTLE_MS = 500

    def __init__(self, doc, settings, parent=None):
        super().__init__(parent)
        self.setFormat(surface_format())
        self.doc = doc
        self.settings = settings
        self.cam = Camera()
        self.engine = InferenceEngine(doc, self.cam, settings)
        self.tool = None
        self.tools = {}
        self.prev_tool = ""
        self.cam_hist = [self.cam.state()]
        self.cam_idx = 0
        self._cam_settle = QTimer(self)
        self._cam_settle.setSingleShot(True)
        self._cam_settle.setInterval(self.CAM_SCROLL_SETTLE_MS)
        self._cam_settle.timeout.connect(self.commit_camera)
        self.vcb_text = ""
        self._status_note = None          # a message that is not the instruction of a step
        self._status_shown = ""
        self._vcb_shown = None            # last (label, value) sent to the Measurements box
        self.hover = None               # ("face", obj, rid) | ("object", obj) | ("region", r) | ...
        self.preview_transform = None   # (ids, 4x4, keep_original)
        self.problem_items = []         # items the Printcontrole panel marks in red
        self.labels = []                # (screen QPointF, text)
        self.markers = []               # (screen QPointF, inference kind)
        self._rcache = {}
        self._drag = None
        self._last = None
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setMinimumSize(320, 240)
        doc.changed.connect(self.update)
        doc.selection_changed.connect(self.update)

    # ------------------------------------------------------------- tools
    def set_tool(self, name):
        if self.tool is not None:
            self.tool.deactivate()
            if name != self.tool.name:
                self.prev_tool = self.tool.name
        self.tool = self.tools[name]
        self.vcb_text = ""
        self.hover = None
        self.preview_transform = None
        self.tool.activate()
        self.setCursor(self.tool.cursor)
        self.tool_changed.emit(name)
        self._status_note = None
        self._vcb_shown = None            # the box always starts again with the new tool
        self.refresh_status()
        self.update()

    def update_vcb(self):
        """Label and value of the Measurements box, as the tool and its step ask for them."""
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
                self.set_status(f"Ongeldige invoer: {e}")
        self._vcb_shown = None            # whatever was typed always leaves the Measurements box
        self.refresh_status()
        self.update()

    def set_status(self, text):
        """A message of a tool or the window: it stays above the step text until the next event."""
        self._status_note = text
        self.show_status(text)

    def show_status(self, text):
        if text and text != self._status_shown:
            self._status_shown = text
            self.status_changed.emit(text)

    def refresh_status(self):
        """SketchUp: after every mouse or key event the status bar shows the instruction of
        the current step of the tool, unless something had something else to say, and the
        Measurements box follows the same step."""
        if self.tool is None:
            return
        note, self._status_note = self._status_note, None
        self.show_status(note or self.tool.status_text())
        self.update_vcb()

    def announce(self):
        """Say it again, even when the status bar already showed this: after a mode switch."""
        self._status_shown = ""
        self._vcb_shown = None
        self.refresh_status()

    # ------------------------------------------------------------- GL
    def initializeGL(self):
        # openstaande fouten weg: anders geeft PyOpenGL GLError 1282 terug bij glClearColor
        errors.clear_gl_errors()
        gl.glClearColor(0.95, 0.95, 0.95, 1)

    def resizeGL(self, w, h):
        self.cam.width, self.cam.height = max(1, w), max(1, h)

    @errors.safe_handler
    def paintGL(self):
        self.cam.width, self.cam.height = max(1, self.width()), max(1, self.height())
        dpr = self.devicePixelRatioF()
        gl.glViewport(0, 0, int(self.width() * dpr), int(self.height() * dpr))
        self._reset_state()
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        self._draw_background()

        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glLoadMatrixd(self.cam.proj_matrix().T.ravel())
        gl.glMatrixMode(gl.GL_MODELVIEW)
        gl.glLoadIdentity()
        gl.glLightfv(gl.GL_LIGHT0, gl.GL_POSITION, (0.35, 0.55, 1.0, 0.0))
        gl.glLightfv(gl.GL_LIGHT1, gl.GL_POSITION, (-0.5, -0.3, 0.4, 0.0))
        gl.glLoadMatrixd(self.cam.view_matrix().T.ravel())
        gl.glEnable(gl.GL_DEPTH_TEST)

        if self.settings.show_grid:
            self._draw_bed()
        self._draw_model()
        if getattr(self.settings, "show_guides", True):
            self._draw_guides()
        if self.settings.show_axes:
            self._draw_axes()
        self._draw_hover()
        self._draw_problems()
        self.labels = []
        self.markers = []
        if self.tool is not None:
            self.tool.draw(self)
        self._draw_overlay()

    def _reset_state(self):
        try:
            gl.glUseProgram(0)
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)
            gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, 0)
            for i in range(4):
                gl.glDisableVertexAttribArray(i)
        except Exception:
            pass
        gl.glDisable(gl.GL_TEXTURE_2D)
        gl.glDisable(gl.GL_CULL_FACE)
        gl.glDisable(gl.GL_STENCIL_TEST)
        gl.glDisable(gl.GL_SCISSOR_TEST)
        gl.glDisable(gl.GL_LIGHTING)
        gl.glDepthMask(gl.GL_TRUE)
        gl.glDepthFunc(gl.GL_LEQUAL)
        gl.glEnable(gl.GL_BLEND)
        # keep destination alpha at 1, otherwise the widget composites translucent pixels
        gl.glBlendFuncSeparate(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA, gl.GL_ONE, gl.GL_ONE_MINUS_SRC_ALPHA)
        gl.glEnable(gl.GL_LINE_SMOOTH)
        gl.glHint(gl.GL_LINE_SMOOTH_HINT, gl.GL_NICEST)
        gl.glEnable(gl.GL_NORMALIZE)
        gl.glShadeModel(gl.GL_SMOOTH)
        gl.glLightModelfv(gl.GL_LIGHT_MODEL_AMBIENT, (0.55, 0.55, 0.56, 1.0))
        gl.glLightModeli(gl.GL_LIGHT_MODEL_TWO_SIDE, gl.GL_TRUE)
        gl.glLightfv(gl.GL_LIGHT0, gl.GL_DIFFUSE, (0.48, 0.48, 0.46, 1.0))
        gl.glLightfv(gl.GL_LIGHT0, gl.GL_SPECULAR, (0, 0, 0, 1))
        gl.glLightfv(gl.GL_LIGHT1, gl.GL_DIFFUSE, (0.10, 0.10, 0.13, 1.0))
        gl.glLightfv(gl.GL_LIGHT1, gl.GL_SPECULAR, (0, 0, 0, 1))
        gl.glEnable(gl.GL_LIGHT0)
        gl.glEnable(gl.GL_LIGHT1)
        gl.glColorMaterial(gl.GL_FRONT_AND_BACK, gl.GL_AMBIENT_AND_DIFFUSE)
        gl.glDisableClientState(gl.GL_COLOR_ARRAY)
        gl.glDisableClientState(gl.GL_TEXTURE_COORD_ARRAY)
        gl.glDisableClientState(gl.GL_NORMAL_ARRAY)
        gl.glDisableClientState(gl.GL_VERTEX_ARRAY)

    def _draw_background(self):
        """The SketchUp background: sky above the horizon, grey ground below it."""
        gl.glDisable(gl.GL_DEPTH_TEST)
        gl.glDepthMask(gl.GL_FALSE)     # the sky must not fill the depth buffer (back edges need it)
        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glLoadIdentity()
        gl.glMatrixMode(gl.GL_MODELVIEW)
        gl.glLoadIdentity()
        h = float(max(1, self.height()))
        y0, y1 = horizon_line(self.cam)

        def sky(y):
            """Sky colour at widget y: almost white at the horizon, blue grey one screen up."""
            t = min(1.0, max(0.0, (0.5 * (y0 + y1) - y) / h))
            return tuple(a * t + b * (1 - t) for a, b in zip(SKY_TOP, SKY_HORIZON))

        gl.glBegin(gl.GL_QUADS)
        gl.glColor3f(*sky(0.0))
        gl.glVertex2f(-1, 1)
        gl.glColor3f(*sky(0.0))
        gl.glVertex2f(1, 1)
        gl.glColor3f(*sky(h))
        gl.glVertex2f(1, -1)
        gl.glColor3f(*sky(h))
        gl.glVertex2f(-1, -1)
        gl.glEnd()

        def ndc_y(y):
            return max(-1e4, min(1e4, 1 - 2 * y / h))

        ground = _clip_below([(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)],
                             (-1.0, ndc_y(y0)), (1.0, ndc_y(y1)))
        if len(ground) > 2:
            gl.glColor3f(*GROUND_COLOR)
            gl.glBegin(gl.GL_POLYGON)
            for x, y in ground:
                gl.glVertex2f(x, y)
            gl.glEnd()
        gl.glDepthMask(gl.GL_TRUE)

    def _draw_bed(self):
        bx, by, _ = self.settings.bed_size
        gl.glDepthMask(gl.GL_FALSE)
        gl.glColor4f(0.55, 0.60, 0.55, 0.13)
        gl.glBegin(gl.GL_QUADS)
        for x, y in ((0, 0), (bx, 0), (bx, by), (0, by)):
            gl.glVertex3f(x, y, 0)
        gl.glEnd()
        lines = []
        step = 10.0
        for x in np.arange(0, bx + 1e-6, step):
            lines += [(x, 0, 0), (x, by, 0)]
        for y in np.arange(0, by + 1e-6, step):
            lines += [(0, y, 0), (bx, y, 0)]
        self.draw_lines(lines, (0.45, 0.48, 0.45, 0.25), 1.0)
        major = []
        for x in np.arange(0, bx + 1e-6, 50.0):
            major += [(x, 0, 0), (x, by, 0)]
        for y in np.arange(0, by + 1e-6, 50.0):
            major += [(0, y, 0), (bx, y, 0)]
        self.draw_lines(major, (0.35, 0.38, 0.35, 0.40), 1.0)
        self.draw_loop([(0, 0, 0), (bx, 0, 0), (bx, by, 0), (0, by, 0)], (0.2, 0.25, 0.2, 0.8), 1.6)
        gl.glDepthMask(gl.GL_TRUE)

    def _draw_axes(self):
        L = 100000.0
        cols = ((0.85, 0.1, 0.1, 1), (0.1, 0.6, 0.1, 1), (0.15, 0.25, 0.9, 1))
        for i, c in enumerate(cols):
            a = np.zeros(3)
            b = np.zeros(3)
            b[i] = L
            self.draw_lines([a, b], c, 1.5)
            b[i] = -L
            self.draw_lines([a, b], (c[0], c[1], c[2], 0.6), 1.2, stipple=True)

    # ------------------------------------------------------------- model
    def _render_data(self, ents, gcolor):
        key = (id(ents), ents.version, gcolor)
        rc = self._rcache.get(id(ents))
        if rc is not None and rc["key"] == key:
            return rc
        c = self.engine.cache.get(ents)
        base = hex_rgb(gcolor) if gcolor else FRONT
        colors = {f: face.color for f, face in ents.F.items()}
        T, fids = c["T"], c["tri_fid"]

        def faces(keep):
            """Vertex, normal and colour arrays of the faces selected by `keep`."""
            n = int(keep.sum())
            front = np.ones((n * 3, 4), dtype=np.float32)
            back = np.ones((n * 3, 4), dtype=np.float32)
            if n:
                ids = fids[keep]
                fc = np.array([hex_rgb(colors[f]) if colors.get(f) else base for f in ids], dtype=np.float32)
                bc = np.array([hex_rgb(colors[f]) if colors.get(f) else (base if gcolor else BACK) for f in ids],
                              dtype=np.float32)
                front[:, :3] = np.repeat(fc, 3, axis=0)
                back[:, :3] = np.repeat(bc, 3, axis=0)
            return {"pos": T[keep].reshape(-1, 3).astype(np.float32),
                    "nrm": np.repeat(c["tri_nrm"][keep], 3, axis=0).astype(np.float32),
                    "front": front, "back": back}

        hid = c["tri_hidden"]
        vis, hidd = faces(~hid), faces(hid)
        if len(c["A"]):
            segs = np.stack([c["A"], c["B"]], axis=1).reshape(-1, 3)
            edges = segs[np.repeat(~c["ehidden"], 2)].astype(np.float32)
        else:
            edges = np.zeros((0, 3), np.float32)
        sshown = np.array([e not in ents.hidden_e for e in c["sids"]], dtype=bool)
        if len(c["SA"]):
            soft = np.stack([c["SA"], c["SB"]], axis=1).reshape(-1, 3)
            soft = soft[np.repeat(sshown, 2)].astype(np.float32)
        else:
            soft = np.zeros((0, 3), np.float32)
        hidden_edges = [e for e in sorted(ents.hidden_e) if e in ents.E]
        if hidden_edges:
            hedges = np.array([p for e in hidden_edges for p in ents.edge_points(e)],
                              dtype=np.float32).reshape(-1, 3)
        else:
            hedges = np.zeros((0, 3), np.float32)
        rc = {"key": key, "pos": vis["pos"], "nrm": vis["nrm"], "front": vis["front"], "back": vis["back"],
              "edges": edges, "soft": soft, "hidden": hidd, "hedges": hedges}
        self._rcache[id(ents)] = rc
        return rc

    def _draw_contexts(self, hiding_rest):
        """(ctx, ents, group, render data, matrices) of everything to draw, with the preview applied."""
        doc = self.doc
        pt = self.preview_transform
        moving = {i for k, _, i in pt[0] if k == "g"} if pt else set()
        for ctx, ents, g in doc.contexts():
            if hiding_rest and ctx != doc.active:
                continue
            mats = [None]
            if ctx in moving:
                mats = [pt[1]] + ([None] if pt[2] else [])
            yield ctx, ents, g, self._render_data(ents, g.color if g else None), mats

    def _draw_model(self):
        doc = self.doc
        sel_groups = {i for k, _, i in doc.selection if k == "g"}
        xray = self.settings.xray
        style = getattr(self.settings, "face_style", SHADED)
        editing = doc.active != 0
        hiding_rest = editing and getattr(self.settings, "hide_rest", False)
        show_hidden = getattr(self.settings, "show_hidden", False)
        show_back = getattr(self.settings, "back_edges", False) and not xray and style != WIREFRAME
        for ctx, ents, g, rc, mats in self._draw_contexts(hiding_rest):
            faded = editing and ctx != doc.active
            selected = g is not None and g.id in sel_groups
            sel_col = LOCK_COLOR if selected and g.locked else SELECT_COLOR
            for M in mats:
                self._push(M)
                if show_hidden:
                    # Bewerken > Verbergen geometry: translucent faces and dashed edges
                    self._draw_faces(rc["hidden"], alpha=0.35, tint=selected, style=style)
                    if len(rc["hedges"]):
                        self.draw_lines(rc["hedges"], HIDDEN_EDGE_COLOR, 1.2, stipple=True)
                self._draw_faces(rc, alpha=0.3 if faded else (0.5 if xray else 1.0), tint=selected,
                                 style=style)
                if self.settings.show_edges and len(rc["edges"]):
                    if faded:
                        col = (0.6, 0.6, 0.6, 1)
                    elif selected:
                        col = (*sel_col, 1)
                    else:
                        col = (0.05, 0.05, 0.05, 1)
                    self.draw_lines(rc["edges"], col, 1.4)
                    if show_hidden and len(rc["soft"]):
                        self.draw_lines(rc["soft"], (0.7, 0.7, 0.7, 1) if faded else SOFT_EDGE_COLOR,
                                        1.0, stipple=True)
                self._pop(M)
            if selected:
                self._draw_bbox(g.bbox(), (*sel_col, 1))
            if g is not None and doc.active == g.id:
                self._draw_bbox(g.bbox(), (0.3, 0.3, 0.3, 1), stipple=True)
        if show_back:
            # second pass: the edges behind the faces, dashed and light grey (SketchUp's back edges)
            for ctx, ents, g, rc, mats in self._draw_contexts(hiding_rest):
                if not len(rc["edges"]):
                    continue
                for M in mats:
                    self._push(M)
                    self._draw_back_edges(rc["edges"])
                    self._pop(M)
        self._draw_selection()

    def _draw_guides(self):
        """Beeld > Hulplijnen: oneindige gestippelde lijnen en hulppunten als kruisjes."""
        doc = self.doc
        if not doc.guides:
            return
        sel = {i for k, _, i in doc.selection if k == "guide"}
        hov = self.hover
        hover_id = hov[1] if hov is not None and hov[0] == "guide" else None
        for g in doc.guides:
            if g.id in sel:
                col = (*SELECT_COLOR, 1)
            elif g.id == hover_id:
                col = (0.95, 0.55, 0.05, 1)
            else:
                col = GUIDE_COLOR
            if g.is_line:
                self.draw_lines(list(g.endpoints()), col, 1.3, stipple=True)
            else:
                _, r, u = self.cam.basis()
                s = self.cam.pixel_size(g.point) * 4.0
                p = g.point
                self.draw_lines([p - r * s, p + r * s, p - u * s, p + u * s], col, 1.5)

    def _draw_back_edges(self, pts):
        """The edges behind the faces: only the fragments that fail the depth test are drawn."""
        gl.glDepthMask(gl.GL_FALSE)
        gl.glDepthFunc(gl.GL_GREATER)
        self.draw_lines(pts, BACK_EDGE_COLOR, 1.0, stipple=True)
        gl.glDepthFunc(gl.GL_LEQUAL)
        gl.glDepthMask(gl.GL_TRUE)

    def _draw_faces(self, rc, alpha=1.0, tint=False, style=None):
        style = style or getattr(self.settings, "face_style", SHADED)
        if style == WIREFRAME or not len(rc["pos"]):
            return
        lit = style != HIDDENLINE
        solid = {HIDDENLINE: (HIDDEN_FACE, HIDDEN_FACE), MONOCHROME: (FRONT, BACK)}.get(style)
        gl.glEnable(gl.GL_POLYGON_OFFSET_FILL)
        gl.glPolygonOffset(1.5, 1.5)
        gl.glEnable(gl.GL_CULL_FACE)
        gl.glFrontFace(gl.GL_CCW)
        if alpha < 1:
            gl.glDepthMask(gl.GL_FALSE)
        if lit:
            gl.glEnable(gl.GL_LIGHTING)
            gl.glEnable(gl.GL_COLOR_MATERIAL)
            gl.glEnableClientState(gl.GL_NORMAL_ARRAY)
            gl.glNormalPointer(gl.GL_FLOAT, 0, rc["nrm"])
        else:
            gl.glDisable(gl.GL_LIGHTING)
        gl.glEnableClientState(gl.GL_VERTEX_ARRAY)
        gl.glVertexPointer(3, gl.GL_FLOAT, 0, rc["pos"])
        passes = ((gl.GL_BACK, rc["front"]), (gl.GL_FRONT, rc["back"])) if solid is None else \
            ((gl.GL_BACK, solid[0]), (gl.GL_FRONT, solid[1]))
        for cull, cols in passes:
            gl.glCullFace(cull)
            if isinstance(cols, np.ndarray):
                if alpha < 1 or tint:
                    cols = cols.copy()
                    if tint:
                        cols[:, :3] = cols[:, :3] * 0.8 + np.array(SELECT_COLOR, np.float32) * 0.2
                    cols[:, 3] = alpha
                gl.glEnableClientState(gl.GL_COLOR_ARRAY)
                gl.glColorPointer(4, gl.GL_FLOAT, 0, cols)
            else:
                gl.glDisableClientState(gl.GL_COLOR_ARRAY)
                gl.glColor4f(*_tint(cols, tint), alpha)
            gl.glDrawArrays(gl.GL_TRIANGLES, 0, len(rc["pos"]))
        gl.glDisableClientState(gl.GL_COLOR_ARRAY)
        gl.glDisableClientState(gl.GL_NORMAL_ARRAY)
        gl.glDisableClientState(gl.GL_VERTEX_ARRAY)
        gl.glDisable(gl.GL_CULL_FACE)
        gl.glDisable(gl.GL_LIGHTING)
        gl.glDisable(gl.GL_COLOR_MATERIAL)
        gl.glDisable(gl.GL_POLYGON_OFFSET_FILL)
        gl.glDepthMask(gl.GL_TRUE)

    def _draw_bbox(self, bb, color, stipple=False):
        if bb is None:
            return
        lo, hi = bb
        c = [[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])]
        idx = [(0, 1), (2, 3), (4, 5), (6, 7), (0, 2), (1, 3), (4, 6), (5, 7), (0, 4), (1, 5), (2, 6), (3, 7)]
        self.draw_lines([c[i] for pair in idx for i in pair], color, 1.5, stipple=stipple)

    def _push(self, M):
        if M is not None:
            gl.glPushMatrix()
            gl.glMultMatrixd(np.asarray(M, dtype=float).T.ravel())

    def _pop(self, M):
        if M is not None:
            gl.glPopMatrix()

    def face_tris(self, ctx, fid):
        ents = self.doc.ents(ctx)
        if ents is None or fid not in ents.F:
            return np.zeros((0, 3))
        c = self.engine.cache.get(ents)
        return c["face_tris"].get(fid, np.zeros((0, 3, 3))).reshape(-1, 3)

    def edge_seg(self, ctx, eid):
        ents = self.doc.ents(ctx)
        if ents is None or eid not in ents.E:
            return None
        return ents.edge_points(eid)

    def _draw_selection(self):
        pt = self.preview_transform
        Ms = [None]
        if pt and any(k != "g" for k, _, _ in pt[0]):
            Ms.append(pt[1])
        gl.glEnable(gl.GL_POLYGON_OFFSET_FILL)
        gl.glPolygonOffset(-1.0, -1.0)
        for M in Ms:
            self._push(M)
            for k, ctx, i in self.doc.selection:
                if k == "f":
                    self._draw_stippled(self.face_tris(ctx, i), SELECT_COLOR)
                elif k == "e":
                    seg = self.edge_seg(ctx, i)
                    if seg is not None:
                        self.draw_lines(seg, (*SELECT_COLOR, 1), 2.6)
            self._pop(M)
        gl.glDisable(gl.GL_POLYGON_OFFSET_FILL)

    def _draw_stippled(self, tris, color):
        """Geselecteerde vlakken: het blauwe stippelpatroon van SketchUp, geen dekkende laag."""
        tris = np.asarray(tris, dtype=np.float32).reshape(-1, 3)
        if not len(tris):
            return
        gl.glDepthMask(gl.GL_FALSE)
        gl.glPolygonStipple(stipple_pattern())
        gl.glEnable(gl.GL_POLYGON_STIPPLE)
        self.draw_tris(tris, (*color, 1.0))
        gl.glDisable(gl.GL_POLYGON_STIPPLE)

    def _draw_hover(self):
        h = self.hover
        if not h:
            return
        gl.glEnable(gl.GL_POLYGON_OFFSET_FILL)
        gl.glPolygonOffset(-2.0, -2.0)
        if h[0] == "face":
            self.draw_tris(self.face_tris(h[1], h[2]), (0.25, 0.45, 1.0, 0.35))
            ents = self.doc.ents(h[1])
            if ents is not None and h[2] in ents.F:
                for L in ents.face_points(h[2]):
                    self.draw_loop(L, (*SELECT_COLOR, 1), 2.2)
        elif h[0] == "edge":
            seg = self.edge_seg(h[1], h[2])
            if seg is not None:
                self.draw_lines(seg, (0.95, 0.55, 0.05, 1), 3.5, on_top=True)
        elif h[0] == "edges":
            for ctx, e in h[1]:
                seg = self.edge_seg(ctx, e)
                if seg is not None:
                    self.draw_lines(seg, (0.95, 0.3, 0.1, 1), 3.5, on_top=True)
        elif h[0] == "group":
            g = self.doc.group(h[1])
            if g is not None:
                self._draw_bbox(g.bbox(), (0.95, 0.55, 0.05, 1))
        gl.glDisable(gl.GL_POLYGON_OFFSET_FILL)

    def _draw_problems(self):
        """Printcontrole: de randen en vlakken van het gekozen probleem, dik in het rood."""
        if not self.problem_items:
            return
        gl.glEnable(gl.GL_POLYGON_OFFSET_FILL)
        gl.glPolygonOffset(-3.0, -3.0)
        for k, ctx, i in self.problem_items:
            if k == "f":
                self.draw_tris(self.face_tris(ctx, i), (*PROBLEM_COLOR[:3], 0.3))
                ents = self.doc.ents(ctx)
                if ents is not None and i in ents.F:
                    for L in ents.face_points(i):
                        self.draw_loop(L, PROBLEM_COLOR, 3.4, on_top=True)
            elif k == "e":
                seg = self.edge_seg(ctx, i)
                if seg is not None:
                    self.draw_lines(seg, PROBLEM_COLOR, 3.6, on_top=True)
        gl.glDisable(gl.GL_POLYGON_OFFSET_FILL)

    # ------------------------------------------------------------- helpers
    def draw_lines(self, pts, color, width=1.0, stipple=False, on_top=False):
        pts = np.asarray(pts, dtype=np.float32).reshape(-1, 3)
        if not len(pts):
            return
        self._prim(gl.GL_LINES, pts, color, width, stipple, on_top)

    def draw_strip(self, pts, color, width=1.0, stipple=False, on_top=False):
        pts = np.asarray(pts, dtype=np.float32).reshape(-1, 3)
        if len(pts) > 1:
            self._prim(gl.GL_LINE_STRIP, pts, color, width, stipple, on_top)

    def draw_loop(self, pts, color, width=1.0, stipple=False, on_top=False):
        pts = np.asarray(pts, dtype=np.float32).reshape(-1, 3)
        if len(pts) > 1:
            self._prim(gl.GL_LINE_LOOP, pts, color, width, stipple, on_top)

    def draw_points(self, pts, color, size=8.0, on_top=True):
        pts = np.asarray(pts, dtype=np.float32).reshape(-1, 3)
        if len(pts):
            gl.glPointSize(size)
            gl.glEnable(gl.GL_POINT_SMOOTH)
            self._prim(gl.GL_POINTS, pts, color, 1, False, on_top)

    def draw_tris(self, pts, color, on_top=False, depth_write=False):
        pts = np.asarray(pts, dtype=np.float32).reshape(-1, 3)
        if not len(pts):
            return
        if not depth_write:
            gl.glDepthMask(gl.GL_FALSE)
        self._prim(gl.GL_TRIANGLES, pts, color, 1, False, on_top)
        gl.glDepthMask(gl.GL_TRUE)

    def _prim(self, mode, pts, color, width, stipple, on_top):
        if on_top:
            gl.glDisable(gl.GL_DEPTH_TEST)
        if stipple:
            gl.glEnable(gl.GL_LINE_STIPPLE)
            gl.glLineStipple(2, 0x3333)
        gl.glLineWidth(width)
        gl.glColor4f(*color) if len(color) == 4 else gl.glColor3f(*color)
        gl.glEnableClientState(gl.GL_VERTEX_ARRAY)
        gl.glVertexPointer(3, gl.GL_FLOAT, 0, np.ascontiguousarray(pts))
        gl.glDrawArrays(mode, 0, len(pts))
        gl.glDisableClientState(gl.GL_VERTEX_ARRAY)
        if stipple:
            gl.glDisable(gl.GL_LINE_STIPPLE)
        if on_top:
            gl.glEnable(gl.GL_DEPTH_TEST)

    def draw_solid_preview(self, md, color, alpha=0.45):
        gl.glDepthMask(gl.GL_FALSE)
        gl.glEnable(gl.GL_LIGHTING)
        gl.glEnable(gl.GL_COLOR_MATERIAL)
        gl.glColor4f(*color, alpha)
        gl.glEnableClientState(gl.GL_VERTEX_ARRAY)
        gl.glEnableClientState(gl.GL_NORMAL_ARRAY)
        gl.glVertexPointer(3, gl.GL_FLOAT, 0, md.tri_positions)
        gl.glNormalPointer(gl.GL_FLOAT, 0, md.tri_normals)
        gl.glDrawArrays(gl.GL_TRIANGLES, 0, len(md.tri_positions))
        gl.glDisableClientState(gl.GL_NORMAL_ARRAY)
        gl.glDisableClientState(gl.GL_VERTEX_ARRAY)
        gl.glDisable(gl.GL_LIGHTING)
        gl.glDisable(gl.GL_COLOR_MATERIAL)
        gl.glDepthMask(gl.GL_TRUE)
        if len(md.edge_positions):
            self.draw_lines(md.edge_positions, (0.1, 0.1, 0.1, 0.9), 1.2)

    def draw_inference(self, inf):
        """Draw the SketchUp style marker + tooltip for an inference result."""
        if inf is None:
            return
        if inf.kind in INFERENCE_COLORS:
            s, ok = self.cam.project(inf.point)
            if ok[0]:
                self.markers.append((QPointF(float(s[0][0]), float(s[0][1])), inf.kind))
        if inf.label:
            self.add_label(inf.point, inf.label, offset=(14, -26), tooltip=True)

    def _paint_marker(self, p, pos, kind):
        """The inference point of SketchUp: per soort een andere vorm in zijn eigen kleur."""
        col = QColor(*[int(round(c * 255)) for c in INFERENCE_COLORS[kind][:3]])
        x, y = float(pos.x()), float(pos.y())
        p.setPen(QPen(QColor(35, 35, 35), 1.2))
        p.setBrush(col)
        shape = INFERENCE_SHAPES.get(kind, "dot")
        if shape == "square":
            p.drawRect(x - 3, y - 3, 6, 6)
        elif shape == "diamond":
            path = QPainterPath(QPointF(x, y - 4.5))
            for px, py in ((x + 4.5, y), (x, y + 4.5), (x - 4.5, y)):
                path.lineTo(px, py)
            path.closeSubpath()
            p.drawPath(path)
        elif shape == "ring":
            p.setBrush(Qt.NoBrush)
            p.drawEllipse(QPointF(x, y), 4.5, 4.5)
        else:
            p.drawEllipse(QPointF(x, y), 3.5, 3.5)

    def draw_axis_guide(self, anchor, point, axis):
        cols = ((0.9, 0.1, 0.1, 1), (0.1, 0.65, 0.1, 1), (0.15, 0.3, 0.95, 1))
        self.draw_lines([anchor, point], cols[axis], 1.5, stipple=True, on_top=True)

    def add_label(self, world, text, offset=(10, -10), tooltip=False):
        s, ok = self.cam.project(world)
        if ok[0]:
            self.labels.append((QPointF(s[0][0] + offset[0], s[0][1] + offset[1]), text, tooltip))

    def _draw_overlay(self):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        f = QFont(self.font())
        f.setPointSizeF(9)
        p.setFont(f)
        for pos, kind in self.markers:
            self._paint_marker(p, pos, kind)
        for pos, text, tooltip in self.labels:
            fm = p.fontMetrics()
            w = fm.horizontalAdvance(text) + 10
            h = fm.height() + 4
            if tooltip:
                p.setPen(QPen(QColor(*TOOLTIP_BORDER), 1.0))
                p.setBrush(QColor(*TOOLTIP_FILL, 245))
            else:
                p.setPen(QPen(QColor(60, 60, 60)))
                p.setBrush(QColor(255, 255, 255, 220))
            p.drawRect(int(pos.x()), int(pos.y()), w, h)
            p.setPen(QColor(0, 0, 0))
            p.drawText(int(pos.x()) + 5, int(pos.y()) + fm.ascent() + 2, text)
        if self.tool is not None:
            self.tool.paint_overlay(p)
        # view name corner
        p.setPen(QColor(90, 90, 90))
        p.drawText(10, self.height() - 10,
                   "Parallel" if self.cam.ortho else "Perspectief")
        p.end()

    # ------------------------------------------------------------- events
    @errors.safe_handler(ret=False, boolean=True)
    def event(self, e):
        if e.type() == QEvent.ShortcutOverride:
            t = e.text()
            if self.vcb_text and t and (t in VCB_CHARS or t.isalnum() or t == " "):
                e.accept()
                return True
            if t and t in "0123456789.-,;=":
                e.accept()
                return True
            if e.key() in (Qt.Key_Escape, Qt.Key_Backspace, Qt.Key_Return, Qt.Key_Enter,
                           Qt.Key_Left, Qt.Key_Right, Qt.Key_Up, Qt.Key_Down) and (
                    self.vcb_text or e.key() != Qt.Key_Backspace):
                e.accept()
                return True
        return super().event(e)

    @errors.safe_handler
    def keyPressEvent(self, e):
        k = e.key()
        t = e.text()
        if k in (Qt.Key_Return, Qt.Key_Enter):
            if self.vcb_text:
                self.submit_vcb(self.vcb_text)
            elif self.tool:
                self.tool.key_press(e)
                self.refresh_status()
            return
        if k == Qt.Key_Backspace and self.vcb_text:
            self.vcb_text = self.vcb_text[:-1]
            self.update_vcb()
            return
        if k == Qt.Key_Escape:
            if self.vcb_text:
                self.vcb_text = ""
                self.update_vcb()
                return
            if self.tool:
                self.tool.cancel()
                self.hover = None
                self.refresh_status()
                self.update()
            return
        if t and (t in "0123456789.-,;=" or (self.vcb_text and (t in VCB_CHARS or t.isalnum() or t == " "))):
            self.vcb_text += t
            self.update_vcb()
            return
        if self.tool and self.tool.key_press(e):
            self.refresh_status()
            self.update()
            return
        super().keyPressEvent(e)

    @errors.safe_handler
    def keyReleaseEvent(self, e):
        if self.tool and self.tool.key_release(e):
            self.refresh_status()
            self.update()
            return
        super().keyReleaseEvent(e)

    @errors.safe_handler
    def mousePressEvent(self, e):
        self.setFocus()
        pos = e.position()
        self._last = pos
        if e.button() == Qt.MiddleButton:
            self._drag = "pan" if e.modifiers() & Qt.ShiftModifier else "orbit"
            self.setCursor(icons.cursor("pan" if self._drag == "pan" else "orbit"))
            return
        if e.button() == Qt.RightButton:
            if self.tool and self.tool.busy():
                self.tool.cancel()
                self.refresh_status()
                self.update()
                return
            self.context_menu.emit(e.globalPosition().toPoint())
            return
        if e.button() == Qt.LeftButton and self.tool:
            self.tool.mouse_press(e, pos.x(), pos.y())
            self.refresh_status()
            self.update()

    @errors.safe_handler
    def mouseMoveEvent(self, e):
        pos = e.position()
        if self._drag:
            dx = pos.x() - self._last.x()
            dy = pos.y() - self._last.y()
            if self._drag == "orbit":
                self.cam.orbit(dx, dy)
            else:
                self.cam.pan(dx, dy)
            self._last = pos
            self.update()
            return
        self._last = pos
        if self.tool:
            self.tool.mouse_move(e, pos.x(), pos.y())
            self.refresh_status()
            self.update()

    @errors.safe_handler
    def mouseReleaseEvent(self, e):
        if e.button() == Qt.MiddleButton and self._drag:
            self._drag = None
            self.commit_camera()
            if self.tool:
                self.setCursor(self.tool.cursor)
            return
        if e.button() == Qt.LeftButton and self.tool:
            pos = e.position()
            self.tool.mouse_release(e, pos.x(), pos.y())
            self.refresh_status()
            self.update()

    @errors.safe_handler
    def mouseDoubleClickEvent(self, e):
        if e.button() == Qt.MiddleButton:
            self.zoom_extents()
            return
        if e.button() == Qt.LeftButton and self.tool:
            pos = e.position()
            self.tool.double_click(e, pos.x(), pos.y())
            self.refresh_status()
            self.update()

    @errors.safe_handler
    def wheelEvent(self, e):
        steps = e.angleDelta().y() / 120.0
        if steps == 0:
            return
        pos = e.position()
        o, d = self.cam.ray(pos.x(), pos.y())
        about = None
        hit = self.engine.pick_face(o, d)
        if hit:
            about = hit[3]
        else:
            from .geom import GROUND
            p = GROUND.intersect_ray(o, d)
            if p is not None and np.linalg.norm(p - o) < self.cam.distance * 5:
                about = p
        factor = 0.85 ** steps
        self.cam.zoom(factor, about)
        self._cam_settle.start()       # bundle scrolling: store one state after 0,5 s of quiet
        self.update()

    # ------------------------------------------------------------- camera
    def commit_camera(self):
        """Remember the current camera as the newest step of the view history (Camera > Vorige)."""
        self._cam_settle.stop()
        s = self.cam.state()
        if self.cam_hist and states_equal(s, self.cam_hist[self.cam_idx]):
            return
        del self.cam_hist[self.cam_idx + 1:]
        self.cam_hist.append(s)
        if len(self.cam_hist) > self.CAM_HISTORY_MAX:
            del self.cam_hist[:len(self.cam_hist) - self.CAM_HISTORY_MAX]
        self.cam_idx = len(self.cam_hist) - 1
        self._notify_cam_history()

    def can_prev_camera(self):
        return self.cam_idx > 0

    def can_next_camera(self):
        return self.cam_idx < len(self.cam_hist) - 1

    def _notify_cam_history(self):
        self.cam_history_changed.emit(self.can_prev_camera(), self.can_next_camera())

    def _goto_camera(self, index):
        self._cam_settle.stop()
        self.cam_idx = index
        self.cam.set_state(self.cam_hist[index])
        self._notify_cam_history()
        self.update()

    def prev_camera(self):
        """Camera > Vorige: back to the previous camera position."""
        if self.can_prev_camera():
            self._goto_camera(self.cam_idx - 1)

    def next_camera(self):
        """Camera > Volgende: forward again after Vorige."""
        if self.can_next_camera():
            self._goto_camera(self.cam_idx + 1)

    def point_at_pixel(self, x, y):
        """World point under widget pixel (x, y), on the view plane through the target."""
        w, h = max(1, self.width()), max(1, self.height())
        _, r, u = self.cam.basis()
        hh = self.cam.half_height()
        nx, ny = (2 * x / w - 1), (1 - 2 * y / h)
        return self.cam.target + r * (nx * hh * w / h) + u * (ny * hh)

    def zoom_window(self, rect):
        """Camera > Zoomvenster: make the screen rectangle fill the view."""
        (x0, y0), (x1, y1) = rect
        w, h = abs(x1 - x0), abs(y1 - y0)
        if w < 2 or h < 2:
            return
        centre = ((x0 + x1) / 2, (y0 + y1) / 2)
        factor = min(1.0, max(w / max(1, self.width()), h / max(1, self.height())))
        target = self.point_at_pixel(*centre)
        self.cam.target = target
        self.cam.distance = max(0.5, self.cam.distance * factor)
        self.commit_camera()
        self.update()

    def set_projection(self, ortho):
        self.cam.ortho = bool(ortho)
        self.commit_camera()
        self.update()

    def zoom_extents(self):
        bb = self.doc.model_bbox()
        if bb is None:
            bx, by, bz = self.settings.bed_size
            bb = np.array([[0, 0, 0], [bx, by, 0]], dtype=float)
        self.cam.fit((bb[0], bb[1]))
        self.commit_camera()
        self.update()

    def set_view(self, name):
        self.cam.set_view(name)
        self.commit_camera()
        self.update()

    def zoom_items(self, items):
        """Zoom in on the given items (the Printcontrole panel uses this on double click)."""
        pts = [p for p in (self.item_points(it) for it in items) if p is not None and len(p)]
        if not pts:
            return
        P = np.concatenate(pts)
        self.cam.fit((P.min(axis=0), P.max(axis=0)))
        self.commit_camera()
        self.update()

    def item_points(self, item):
        k, ctx, i = item
        if k == "guide":
            g = self.doc.guide(i)
            if g is None:
                return None
            return np.array([g.point, g.point] if not g.is_line else list(g.endpoints()))
        if k == "g":
            g = self.doc.group(i)
            bb = g.bbox() if g else None
            if bb is None:
                return None
            lo, hi = bb
            return np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
        ents = self.doc.ents(ctx)
        if ents is None:
            return None
        if k == "f":
            return np.concatenate(ents.face_points(i)) if i in ents.F else None
        return np.array(ents.edge_points(i)) if i in ents.E else None

    def item_screen_bbox(self, item):
        pts = self.item_points(item)
        if pts is None:
            return None
        s, ok = self.cam.project(pts)
        if not ok.all():
            return None
        return s.min(axis=0), s.max(axis=0)

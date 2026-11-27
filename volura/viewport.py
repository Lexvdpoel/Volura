"""The 3D viewport: OpenGL rendering, navigation and event routing to tools."""
from __future__ import annotations

import ctypes

import numpy as np
from PySide6.QtCore import QEvent, QPointF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPainterPath, QPen, QSurfaceFormat
from PySide6.QtOpenGLWidgets import QOpenGLWidget

from . import background, clearance, collisions, errors, icons, ops, plates, section, theme, thin_walls
from .camera import Camera, horizon_line
from .i18n import tr, trn
from .inference import InferenceEngine
from .tools import MEASURE_LABEL
from .viewcube import ViewCube

errors.configure_pyopengl()          # before OpenGL loads, so it stops checking every call

from OpenGL import GL as gl          # noqa: E402
from OpenGL.GL import GLubyte        # noqa: E402

VCB_CHARS = set("0123456789.,;-+/*xXsS\"'mcin<>[]=")
SELECT_COLOR = (0.13, 0.35, 1.0)
LOCK_COLOR = (0.80, 0.13, 0.10)        # a selected locked group gets a red frame, not a blue one
HIDDEN_EDGE_COLOR = (0.45, 0.45, 0.55, 0.85)
PROBLEM_COLOR = (0.86, 0.06, 0.06, 1.0)   # the problems of Print Check
FRONT = (1.0, 1.0, 1.0)                # the front of a face is white, like SketchUp
BACK = (0.64, 0.70, 0.78)              # the back of a face is blue grey
# the two support volumes of 3D Printing: always in the colour of their role and see-through, so a
# volume in the model is never mistaken for a part of it (the colours live in theme.py)
ROLE_COLORS = {ops.ROLE_BLOCKER: "blocker", ops.ROLE_ENFORCER: "enforcer"}
ROLE_ALPHA = 0.35                      # as translucent as the hidden faces of the model

# The SketchUp background: a light blue grey sky that fades to almost white at the
# horizon, and a grey ground below it. In the dark theme a muted grey, in the high
# contrast theme black (theme.py).

# The inference markers of SketchUp: one shape per kind, drawn in 2D at the cursor.
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

# View > Face Style (SketchUp's face styles); SHADED is the default.
WIREFRAME = "wireframe"
HIDDENLINE = "hiddenline"
SHADED = "shaded"
MONOCHROME = "monochrome"
FACE_STYLES = (WIREFRAME, HIDDENLINE, SHADED, MONOCHROME)
HIDDEN_FACE = (1.0, 1.0, 1.0)          # hidden line: faces in the white background colour
BACK_EDGE_COLOR = (0.60, 0.60, 0.60, 0.85)
SOFT_EDGE_COLOR = (0.35, 0.50, 0.85, 0.9)
GUIDE_COLOR = (0.52, 0.52, 0.58, 1)
# the print bed: fill, fine and coarse grid lines; light lines on the dark background
BED_LIGHT = ((0.55, 0.60, 0.55, 0.13), (0.45, 0.48, 0.45, 0.25), (0.35, 0.38, 0.35, 0.40))
BED_DARK = ((0.85, 0.88, 0.85, 0.08), (0.86, 0.89, 0.86, 0.16), (0.90, 0.92, 0.90, 0.32))    # guides: dashed light grey, like SketchUp
# The bed of the plates after the first: an outline and a big number, so a project that is
# spread over several beds shows which part belongs to which print. The number is drawn from
# seven segments instead of a font, because it stands flat on the bed and grows with the view.
PLATE_BED_COLOR = (0.30, 0.36, 0.30, 0.85)
PLATE_NUMBER_COLOR = (0.34, 0.40, 0.34, 0.75)
PLATE_NUMBER_HIGH = 0.34       # of the depth of the bed
PLATE_NUMBER_WIDE = 0.20       # of the depth of the bed, per digit
PLATE_NUMBER_GAP = 0.07        # between two digits, of the depth of the bed
#: which of the seven segments a digit lights up
DIGIT_SEGMENTS = {"0": "abcdef", "1": "bc", "2": "abged", "3": "abgcd", "4": "fgbc",
                  "5": "afgcd", "6": "afgedc", "7": "abc", "8": "abcdefg", "9": "abcfgd"}
#: the two ends of a segment in a cell of one by one: x to the right, y up
SEGMENT_ENDS = {"a": ((0.0, 1.0), (1.0, 1.0)), "b": ((1.0, 0.5), (1.0, 1.0)),
                "c": ((1.0, 0.0), (1.0, 0.5)), "d": ((0.0, 0.0), (1.0, 0.0)),
                "e": ((0.0, 0.0), (0.0, 0.5)), "f": ((0.0, 0.5), (0.0, 1.0)),
                "g": ((0.0, 0.5), (1.0, 0.5))}


def number_segments(plate) -> list:
    """The plate number as line segments: `(x0, y0, x1, y1)` per segment.

    A digit is one by one and the row of digits stands on y = 0, centred on x = 0, so the
    caller has only to scale it and put it on the bed."""
    digits = str(plates.plate_number(plate))
    step = 1.0 + PLATE_NUMBER_GAP
    total = len(digits) * step - PLATE_NUMBER_GAP
    out = []
    for k, char in enumerate(digits):
        x = k * step - 0.5 * total
        for seg in DIGIT_SEGMENTS.get(char, ""):
            (ax, ay), (bx, by) = SEGMENT_ENDS[seg]
            out.append((x + ax, ay, x + bx, by))
    return out


def hex_rgb(h):
    c = QColor(h)
    return c.redF(), c.greenF(), c.blueF()


def edge_color() -> tuple:
    """The color of the edges between the faces: white in the high contrast theme."""
    return (*theme.rgb("edge"), 1)


def _role_of(g):
    """The role a group takes in the print, or None when it is a printed part.

    A support blocker or enforcer is drawn in the see-through colour of its role and is no part of
    the print: its faces are never counted among its overhangs, thin walls, tight fits or
    collisions."""
    role = ops.group_role(g) if g is not None else None
    return role if role in ROLE_COLORS else None


def _boxes_touch(a, b):
    """Whether the boxes `a` and `b` of two parts touch at all.

    A box is what ``Entities.bbox()`` gives back: ``[[min], [max]]``. Boxes that only touch at
    a face or at an edge do touch: the parts they stand for may still stand in one another."""
    return bool(np.all(a[0] <= b[1]) and np.all(b[0] <= a[1]))


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


# Vertex buffer objects: a render cache goes to the GPU once instead of every frame.
GEOM_STRIDE = 6 * 4          # bytes per vertex of the buffer with position + normal
COLOUR_STRIDE = 8 * 4        # bytes per vertex of the buffer with front + back colour
_TINT_MIX = np.array(SELECT_COLOR, np.float32) * 0.2

# the state the model pass switches through _FaceState: caps, and the arrays of the client state
_CAPS = (gl.GL_LIGHTING, gl.GL_COLOR_MATERIAL, gl.GL_CULL_FACE, gl.GL_POLYGON_OFFSET_FILL)
_ARRAYS = (gl.GL_VERTEX_ARRAY, gl.GL_NORMAL_ARRAY, gl.GL_COLOR_ARRAY)


def _at(offset):
    """A byte offset in a buffer object: PyOpenGL wants a pointer, a plain int draws nothing."""
    return ctypes.c_void_p(offset)


def _new_buffer(data):
    """A buffer object with `data` (float32, contiguous) in it; 0 when this driver refuses."""
    name = int(gl.glGenBuffers(1))
    if name <= 0:
        return 0            # no name, so nothing to fill: buffer 0 is "no buffer", not ours
    gl.glBindBuffer(gl.GL_ARRAY_BUFFER, name)
    gl.glBufferData(gl.GL_ARRAY_BUFFER, data.nbytes, data, gl.GL_STATIC_DRAW)
    gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)
    if gl.glGetError() != gl.GL_NO_ERROR:
        _delete_buffers([name])
        return 0
    return name


def _delete_buffers(names):
    """Give the buffer objects back to the driver; a context that is gone simply raises."""
    names = [int(n) for n in names if n]
    if not names:
        return
    try:
        gl.glDeleteBuffers(len(names), np.array(names, dtype=np.uint32))
    except Exception:
        pass


def _colour_variant(cols, tint, alpha):
    """A colour array with the selection tint and the face alpha mixed in; `cols` itself if plain.

    The same mix the old per frame copy made, but now made once per render cache."""
    if not tint and alpha >= 1:
        return cols
    out = cols.copy()
    if tint:
        out[:, :3] = out[:, :3] * 0.8 + _TINT_MIX
    out[:, 3] = alpha
    return out


def _colour_data(front, back):
    """The two colour arrays of one set of faces in one array: 8 floats per vertex."""
    cols = np.empty((len(front), 8), dtype=np.float32)
    cols[:, :4] = front
    cols[:, 4:] = back
    return cols


class _FaceBuffers:
    """The GPU copy of one set of faces of a render cache (see `Viewport._face_buffers`)."""

    __slots__ = ("geom", "colours", "count")

    def __init__(self):
        self.geom = 0                 # position + normal, GEOM_STRIDE per vertex
        self.colours = {}             # (tint, alpha) -> buffer with COLOUR_STRIDE per vertex
        self.count = 0

    def free(self):
        """Back to the driver: this render cache is replaced or its group is gone."""
        _delete_buffers([self.geom] + list(self.colours.values()))
        self.geom = 0
        self.colours = {}
        self.count = 0


def _free_render_data(rc):
    """The buffers of one render cache: its faces and the edges between them."""
    for fs in (rc, rc.get("hidden")):
        if fs is None:
            continue
        b = fs.get("gpu")
        if b is not None:
            b.free()
        _delete_buffers(list(fs.get("lbuf", {}).values()))
        fs.pop("lbuf", None)


class _FaceState:
    """What the 3D view already switched on: one GL call per change instead of per group.

    A frame goes back and forth between faces (lighting, culling, polygon offset, colour
    array) and edges (none of that), for every group of the model, and every step of that
    used to be an explicit glEnable/glDisable pair. This remembers the state, so only the
    calls that really change something are made.

    `set` is for the caps of glEnable/glDisable, `array` for the arrays of
    glEnableClientState/glDisableClientState; the two are different kinds of state and
    only the right call changes the right one. `_reset_state` starts a frame from what the
    driver really has switched on (`capture`), so a tool or a Qt paint engine that
    changed one of these behind its back is followed correctly by the next frame."""

    __slots__ = ("on",)

    def __init__(self, on=None):
        self.on = dict(on or {})

    @classmethod
    def capture(cls):
        """A tracker that starts from the state the driver is really in right now."""
        on = {}
        for cap in _CAPS + _ARRAYS:
            try:
                on[cap] = bool(gl.glIsEnabled(cap))
            except Exception:
                on[cap] = False
        return cls(on)

    def _change(self, cap, want, on, off):
        if self.on.get(cap, False) == want:
            return
        self.on[cap] = want
        (on if want else off)(cap)

    def set(self, cap, want):
        """glEnable or glDisable `cap`, but only when it is not already in that state."""
        self._change(cap, want, gl.glEnable, gl.glDisable)

    def array(self, cap, want):
        """glEnableClientState or glDisableClientState `cap`, only on a real change."""
        self._change(cap, want, gl.glEnableClientState, gl.glDisableClientState)


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
    #: how long the model may keep changing before its collisions are looked at again
    COLLISION_SETTLE_MS = 300
    #: a model of more parts than this is only measured near the selection (see _collision_near)
    COLLISION_PARTS = 60

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
        self.overlay_lines = []         # (pts, color, width) a dialog draws over the model
        self.overlay_solid = None       # (MeshData, color, alpha) a dialog draws over the model
        self.problem_items = []         # items the Print Check panel marks in red
        self.plate = plates.ALL         # the plate the view shows; plates.ALL is every plate
        self.labels = []                # (screen QPointF, text)
        self.markers = []               # (screen QPointF, inference kind)
        self._rcache = {}
        self._vbo = False             # set in initializeGL: can this driver take buffer objects?
        self._gl_ctx = None           # the GL context the buffers in _rcache belong to
        self._fstate = _FaceState()   # which GL states the model pass already set
        self._overhang_cache = None   # (key, (bed_z, faces)) of the overhang view
        self._overhang_ctx = None     # (bed_z, limit) the current frame tints faces with
        self._thin_cache = {}         # id(ents) -> (ents, version, wall, bool per triangle)
        self._thin_busy = {}          # id(ents) -> that same key while its cut is running
        self._thin_answers = {}       # id(ents) -> answers landed; see _thin_context
        self._tight_answer = None     # (key, {id(ents): (ents, bool per triangle)}) of one check
        self._tight_busy = None       # that same key while the check of the model is running
        self._tight_landed = {}       # id(ents) -> answers landed; see _tight_context
        self._collision_answer = None  # (key, overlaps, parts measured) of one check of the model
        self._collision_busy = None    # that same key while the check of the model is running
        self._collision_timer = QTimer(self)
        self._collision_timer.setSingleShot(True)
        self._collision_timer.setInterval(self.COLLISION_SETTLE_MS)
        self._collision_timer.timeout.connect(self._check_collisions)
        self._cap_cache = {}          # (id(ents), version, axis, pos) -> triangles of the cut
        self._cap_solids = {}         # id(ents) -> (ents, version, manifold) to cut in
        self._cap_busy = {}           # id(ents) -> that same key while its solid is being made
        self._view_hint = ""          # what the colours in the model mean, "" when nothing does
        self._drag = None
        self._last = None
        self.middle_pans = False        # the Fusion scheme pans with the middle button
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setAccessibleName(tr("3D view"))
        self.setMinimumSize(320, 240)
        self.view_cube = ViewCube(self, "main")   # the cube in the top right corner
        doc.changed.connect(self.update)
        doc.selection_changed.connect(self.update)
        doc.changed.connect(self.update_view_status)
        doc.changed.connect(self.collisions_changed)
        doc.selection_changed.connect(self.collisions_changed)

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
                self.set_status(tr("Invalid input: {error}").format(error=e))
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
        # clear pending errors: otherwise PyOpenGL returns GLError 1282 on glClearColor
        errors.clear_gl_errors()
        ctx = self.context()
        if ctx is self._gl_ctx:
            self.release_gl()        # this context again: the old names are still buffers, give them back
        else:
            self._rcache = {}        # another context: its buffers are gone with the old one
        self._gl_ctx = ctx
        self._vbo = self._probe_buffers()
        gl.glClearColor(*theme.rgb("sky_horizon"), 1)

    def _probe_buffers(self):
        """Whether this driver accepts vertex buffer objects (an old driver or offscreen: not).

        Everything is tried once here, so `_face_buffers` can simply fall back to the
        arrays in memory when the answer is no."""
        try:
            name = _new_buffer(np.zeros(4, dtype=np.float32))
        except Exception:
            return False
        if not name:
            return False
        _delete_buffers([name])
        return True

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
        self._draw_collisions()
        self._draw_overlay_solid()
        self._draw_overlay_lines()
        self.labels = []
        self.markers = []
        if self.tool is not None:
            self.tool.draw(self)
        self._draw_overlay()
        self.view_cube.sync()              # the cube in the corner follows the camera

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
        gl.glDisable(gl.GL_COLOR_MATERIAL)
        gl.glDisable(gl.GL_POLYGON_OFFSET_FILL)
        gl.glDisable(gl.GL_CLIP_PLANE0)     # View > Section; the model pass switches it on
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
        gl.glFrontFace(gl.GL_CCW)        # counter clockwise faces, the whole frame through
        # what the driver really has switched on now, so the tracker of the model pass
        # never assumes a state that some earlier drawing left behind
        self._fstate = _FaceState.capture()

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
        sky_top, sky_horizon = theme.rgb("sky_top"), theme.rgb("sky_horizon")

        def sky(y):
            """Sky colour at widget y: almost white at the horizon, blue grey one screen up."""
            t = min(1.0, max(0.0, (0.5 * (y0 + y1) - y) / h))
            return tuple(a * t + b * (1 - t) for a, b in zip(sky_top, sky_horizon))

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
            gl.glColor3f(*theme.rgb("ground"))
            gl.glBegin(gl.GL_POLYGON)
            for x, y in ground:
                gl.glVertex2f(x, y)
            gl.glEnd()
        gl.glDepthMask(gl.GL_TRUE)

    def _draw_bed(self):
        bx, by, _ = self.settings.bed_size
        fill, minor_col, major_col = BED_DARK if theme.is_dark() else BED_LIGHT
        gl.glDepthMask(gl.GL_FALSE)
        gl.glColor4f(*fill)
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
        self.draw_lines(lines, minor_col, 1.0)
        major = []
        for x in np.arange(0, bx + 1e-6, 50.0):
            major += [(x, 0, 0), (x, by, 0)]
        for y in np.arange(0, by + 1e-6, 50.0):
            major += [(0, y, 0), (bx, y, 0)]
        self.draw_lines(major, major_col, 1.0)
        self.draw_loop([(0, 0, 0), (bx, 0, 0), (bx, by, 0), (0, by, 0)], (0.2, 0.25, 0.2, 0.8), 1.6)
        self._draw_plate_beds(bx, by)
        gl.glDepthMask(gl.GL_TRUE)

    def _plate_beds(self) -> list:
        """The plates after the first whose bed is on the screen.

        As many as the model needs, and at least the one the plate chooser is looking at, so a
        plate that is still empty is a bed you can already put parts on."""
        if getattr(self, "doc", None) is None:
            return []
        need = plates.plate_count(self.doc, self.settings.bed_size)
        if self.plate != plates.ALL:
            # the chosen plate, and the empty bed after it: where the parts go when this one fills up
            need = max(need, self.plate + 1)
        return list(range(plates.FIRST + 1, need + 1))

    def _draw_plate_beds(self, bx, by):
        """The beds of the plates after the first: the same bed, next to the real one along +X,
        drawn as an outline with a big number on it. Only plate 1 is the bed of the printer;
        the others are places on the same machine for the parts that did not fit on it."""
        for plate in self._plate_beds():
            ox = float(plates.origin(plate, self.settings.bed_size)[0])
            self.draw_loop([(ox, 0.0, 0.0), (ox + bx, 0.0, 0.0), (ox + bx, by, 0.0),
                            (ox, by, 0.0)], PLATE_BED_COLOR, 1.6)
            wide = PLATE_NUMBER_WIDE * by
            high = PLATE_NUMBER_HIGH * by
            cx, y0 = ox + 0.5 * bx, 0.5 * (by - high)
            lines = []
            for x0, y0s, x1, y1 in number_segments(plate):
                lines += [(cx + x0 * wide, y0 + y0s * high, 0.05),
                          (cx + x1 * wide, y0 + y1 * high, 0.05)]
            self.draw_lines(lines, PLATE_NUMBER_COLOR, 2.5 * self.settings.line_scale)

    def set_plate(self, plate) -> None:
        """Look at one plate of the model, or at all of them (`plates.ALL`)."""
        self.plate = plates.plate_choice(plate)
        self.update()

    def _plate_shown(self, g) -> bool:
        """Whether this group is on the plate the view is looking at.

        Loose geometry of the model itself has no group to stand on a bed, so it belongs to the
        first plate; it is only hidden when the view picked another plate."""
        if self.plate == plates.ALL:
            return True
        if g is None:
            return self.plate == plates.FIRST
        return plates.on_plate(g, self.plate, self.settings.bed_size)

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
    def _render_data(self, ents, gcolor, role=None):
        # a support volume is drawn as what it is: the overhang, thin wall and tight fit colours
        # belong to the print, and this volume is no part of it
        octx = None if role else self._overhang_ctx   # (bed_z, limit) while the overhang view is on
        wctx = None if role else self._thin_context(ents)   # (wall, answers) thin wall view on
        fctx = None if role else self._tight_context(ents)  # (room, answers) tight fit view on
        key = (id(ents), ents.version, gcolor, role, octx, wctx, fctx)
        rc = self._rcache.get(id(ents))
        if rc is not None and rc["key"] == key and rc["ents"] is ents:
            return rc
        if rc is not None:
            _free_render_data(rc)         # another version: the old buffers are no longer drawn
        c = self.engine.cache.get(ents)
        base = hex_rgb(gcolor) if gcolor else FRONT
        role_rgb = theme.rgb(ROLE_COLORS[role]) if role else None
        colors = {f: face.color for f, face in ents.F.items()}
        T, fids = c["T"], c["tri_fid"]
        warn = theme.rgb("overhang") if octx else None
        thin = theme.rgb("thin") if wctx else None
        tight = theme.rgb("tight") if fctx else None
        mask = self._thin_mask_of(ents) if wctx else None
        fmask = self._tight_mask_of(ents) if fctx else None

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
                if role_rgb is not None:
                    # both sides of every face in the colour of the role: the volume hangs in the
                    # model as a red or green shape, whatever way it is turned
                    front[:, :3] = role_rgb
                    back[:, :3] = role_rgb
                if tight is not None and fmask is not None:
                    # the tight places go on under the thin walls and the overhangs: of two
                    # troubles on one face the machine is the worse one to hear about
                    vmask = np.repeat(fmask[keep], 3)
                    front[vmask, :3] = tight
                    back[vmask, :3] = tight
                if thin is not None and mask is not None:
                    # the thin walls go on first: a face that both hangs over and is thin
                    # stays red, because support is the worse of the two problems
                    vmask = np.repeat(mask[keep], 3)
                    front[vmask, :3] = thin
                    back[vmask, :3] = thin
                if warn is not None:
                    bed_z, limit = octx
                    over = ops.overhang_mask(c["tri_nrm"][keep], T[keep].mean(axis=1)[:, 2],
                                             bed_z, limit)
                    vmask = np.repeat(over, 3)
                    front[vmask, :3] = warn
                    back[vmask, :3] = warn
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
        rc = {"key": key, "ents": ents, "pos": vis["pos"], "nrm": vis["nrm"],
              "front": vis["front"], "back": vis["back"],
              "edges": edges, "soft": soft, "hidden": hidd, "hedges": hedges}
        self._rcache[id(ents)] = rc
        return rc

    def _plate_contexts(self, hiding_rest):
        """(ctx, ents, group) of everything on the plate the view is looking at.

        The heart of the plate chooser: the parts of the other plates are simply not drawn, so
        one plate at a time is what you see and what you pick."""
        for ctx, ents, g in self.doc.contexts():
            if hiding_rest and ctx != self.doc.active:
                continue
            if not self._plate_shown(g):
                continue
            yield ctx, ents, g

    def _draw_contexts(self, hiding_rest):
        """(ctx, ents, group, render data, matrices) of everything to draw, with the preview applied."""
        pt = self.preview_transform
        moving = {i for k, _, i in pt[0] if k == "g"} if pt else set()
        for ctx, ents, g in self._plate_contexts(hiding_rest):
            mats = [None]
            if ctx in moving:
                mats = [pt[1]] + ([None] if pt[2] else [])
            yield ctx, ents, g, self._render_data(ents, g.color if g else None,
                                                  _role_of(g)), mats

    def _draw_model(self):
        doc = self.doc
        sel_groups = {i for k, _, i in doc.selection if k == "g"}
        xray = self.settings.xray
        style = getattr(self.settings, "face_style", SHADED)
        editing = doc.active != 0
        hiding_rest = editing and getattr(self.settings, "hide_rest", False)
        show_hidden = getattr(self.settings, "show_hidden", False)
        show_back = getattr(self.settings, "back_edges", False) and not xray and style != WIREFRAME
        show_edges = self.settings.show_edges
        self._overhang_ctx = self._overhang_context()
        cut = self.engine.section_cut()
        if cut is None:
            self._forget_caps()
        else:
            self._start_section(cut[3])
        for ctx, ents, g, rc, mats in self._draw_contexts(hiding_rest):
            faded = editing and ctx != doc.active
            selected = g is not None and g.id in sel_groups
            sel_col = LOCK_COLOR if selected and g.locked else SELECT_COLOR
            # the edges of this group go to the GPU once, whatever how often the group is
            # drawn below (a move preview draws the same group twice)
            edges = self._line_buffer(rc, "edges") if show_edges and len(rc["edges"]) else 0
            hedges = self._line_buffer(rc, "hedges") if show_hidden and len(rc["hedges"]) else 0
            soft = self._line_buffer(rc, "soft") if show_edges and show_hidden and len(rc["soft"]) else 0
            for M in mats:
                self._push(M)
                if show_hidden:
                    # Edit > Hide geometry: translucent faces and dashed edges
                    self._draw_faces(rc["hidden"], alpha=0.35, tint=selected, style=style)
                    if len(rc["hedges"]):
                        self.draw_lines(rc["hedges"], HIDDEN_EDGE_COLOR, 1.2, stipple=True, buf=hedges)
                self._draw_faces(rc, alpha=(ROLE_ALPHA if _role_of(g)
                                            else (0.3 if faded else (0.5 if xray else 1.0))),
                                 tint=selected, style=style)
                if show_edges and len(rc["edges"]):
                    if faded:
                        col = (0.6, 0.6, 0.6, 1)
                    elif selected:
                        col = (*sel_col, 1)
                    else:
                        col = edge_color()
                    self.draw_lines(rc["edges"], col, 1.4, buf=edges)
                    if show_hidden and len(rc["soft"]):
                        self.draw_lines(rc["soft"], (0.7, 0.7, 0.7, 1) if faded else SOFT_EDGE_COLOR,
                                        1.0, stipple=True, buf=soft)
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
                buf = self._line_buffer(rc, "edges")
                for M in mats:
                    self._push(M)
                    self._draw_back_edges(rc["edges"], buf)
                    self._pop(M)
        if cut is not None:
            self._draw_caps(cut, hiding_rest)
        self._end_model_pass()
        self._draw_selection()
        self._end_section()
        self._prune_rcache()

    def _prune_rcache(self):
        """The buffers of geometry that left the document: a deleted group, another file.

        Called from the model pass, so the context is current and the names still mean
        something; a render cache of a group that is drawn again simply stays."""
        doc = getattr(self, "doc", None)
        if doc is None:
            return
        live = {id(e) for e in doc.all_ents()}
        for key in [k for k in self._rcache if k not in live]:
            _free_render_data(self._rcache.pop(key))
        for table in (self._thin_cache, self._thin_busy, self._thin_answers,
                      self._tight_landed):
            for key in [k for k in table if k not in live]:
                table.pop(key)
        for key in (self._tight_answer[0] if self._tight_answer is not None else None,
                    self._tight_busy):
            # the tight fit check belongs to the whole model: a part that left the document
            # makes its answer nothing, and a check on its way to it too
            if key is not None and any(k not in live for k, _v in key[:-1]):
                self._tight_answer = None
                self._tight_busy = None
                break
        for table in (self._cap_solids, self._cap_busy):
            for key in [k for k in table if k not in live]:
                table.pop(key)
        for key in [k for k in self._cap_cache if k[0] not in live]:
            self._cap_cache.pop(key)

    # ------------------------------------------------------------- overhangs
    def _overhang_faces(self):
        """``(bed_z, faces that need support)`` of the model, cached until it or the limit changes.

        The bed is the lowest z of all visible geometry that prints; a face is counted when its
        outward normal hangs over further than the limit (see :func:`ops.overhang_mask`). The count
        is per face, read off the triangle normals the render cache already holds, so nothing is
        triangulated twice and no frame pays for it twice. A support volume is in neither: it is no
        part, and it hangs where the support is wanted instead of where the print stands."""
        if not getattr(self.settings, "show_overhangs", False):
            return None
        limit = getattr(self.settings, "overhang_limit", 45)
        ents_list = [e for _, e, g in self.doc.contexts() if _role_of(g) is None]
        key = tuple((id(e), e.version) for e in ents_list) + (limit,)
        if self._overhang_cache is not None and self._overhang_cache[0] == key:
            return self._overhang_cache[1]
        tris = [(e, T) for e, T in ((e, self.engine.cache.get(e)["T"]) for e in ents_list)
                if len(T)]
        if not tris:
            self._overhang_cache = (key, (0.0, 0))
            return self._overhang_cache[1]
        bed_z = min(float(T[:, 2].min()) for _e, T in tris)
        total = 0
        for e, T in tris:
            c = self.engine.cache.get(e)
            over = ops.overhang_mask(c["tri_nrm"], T.mean(axis=1)[:, 2], bed_z, limit)
            total += int(len(np.unique(c["tri_fid"][over])))
        self._overhang_cache = (key, (bed_z, total))
        return self._overhang_cache[1]

    def _overhang_context(self):
        """``(bed_z, limit)`` the faces of this frame are tinted with, or None when off."""
        if not getattr(self.settings, "show_overhangs", False):
            return None
        info = self._overhang_faces()
        return None if info is None else (info[0], getattr(self.settings, "overhang_limit", 45))

    # ------------------------------------------------------------- thin walls
    def _thin_context(self, ents):
        """``(wall, answers)`` the faces of `ents` are tinted with, or None when off.

        The answers that landed for this geometry belong in the key of its render cache: while
        a cut is on its way there is nothing to tint yet, and the frame right after one
        arrives may not be skipped as though nothing in the model had changed."""
        if not getattr(self.settings, "show_thin_walls", False):
            return None
        return getattr(self.settings, "min_wall", 0.8), self._thin_answers.get(id(ents), 0)

    def _thin_mask_of(self, ents):
        """The bool per triangle of `ents` whether it lies in a wall thinner than the setting,
        or None while that is not known yet.

        The cut takes seconds on a big model, so it runs in the background (see
        :meth:`_run_thin_job`) and the faces are drawn as they always were until its answer
        has landed. An answer belongs to one version of one geometry and to one wall width:
        a cut of a model that has moved on since is never shown."""
        if not getattr(self.settings, "show_thin_walls", False):
            return None
        wall = getattr(self.settings, "min_wall", 0.8)
        got = self._thin_cache.get(id(ents))
        if got is not None and got[0] is ents and got[1] == ents.version and got[2] == wall:
            mask = got[3]
            # the triangles of this version are counted the same way the answer was made;
            # when the render cache says otherwise, the answer is not the one for this frame
            return mask if len(mask) == len(self.engine.cache.get(ents)["tri_fid"]) else None
        job = (ents, ents.version, wall)
        if self._thin_busy.get(id(ents)) != job:
            self._thin_busy[id(ents)] = job
            self._run_thin_job(job)
        return None

    def _run_thin_job(self, job):
        """The cut of one geometry off the GUI thread; the answer is checked before it lands.

        The window takes no input while it runs (like every job that reads the model), so the
        faces cannot change under the cut; a big model takes seconds, and the status bar says
        what the wait is for. Until the answer has landed the model is drawn without tint,
        which is why a cut that is too late for this frame costs nothing."""
        ents, wall = job[0], job[2]
        # the centres of the triangles the view draws; the answer comes back for these
        points = np.array(self.engine.cache.get(ents)["T"].mean(axis=1), dtype=float)
        checking = tr("Checking wall thickness…")

        def work(report):
            return thin_walls.thin_mask(ents.to_manifold(), points, wall)

        def told():
            """The bar showed what it was waiting for; now it shows the count, or - when
            nothing is thin - the instruction of the tool again."""
            if self._status_note == checking:
                self._status_note = None
            self._view_hint = ""            # so the count comes back even when it is the same
            self.update_view_status()
            self.refresh_status()
            if self._status_shown == checking:
                # nothing thin and a tool without an instruction: the wait is over all the same
                self._status_shown = ""
                self.status_changed.emit("")

        def stored(mask):
            if not background.alive(self) or self._thin_busy.get(id(ents)) != job:
                return                     # another file, another version, another wall width
            self._thin_busy.pop(id(ents), None)
            self._thin_cache[id(ents)] = job + (np.asarray(mask, dtype=bool),)
            self._thin_answers[id(ents)] = self._thin_answers.get(id(ents), 0) + 1
            told()
            self.update()

        def failed(_exc):
            # nothing to show: the faces stay as they are, and the next frame asks again
            if self._thin_busy.get(id(ents)) == job:
                self._thin_busy.pop(id(ents), None)
                told()

        background.run(self.window(), work, text=checking, can_cancel=False,
                       show_dialog=False, on_done=stored, on_error=failed)
        self.set_status(checking)

    def _thin_faces(self):
        """How many faces of the model lie in a wall thinner than the setting; 0 when unknown.

        A face counts as soon as one of its triangles does, read off the triangles the render
        cache holds for the answer of :meth:`_thin_mask_of`. While a cut still runs its
        geometry counts as nothing, so the number can still rise before the view is done."""
        if not getattr(self.settings, "show_thin_walls", False):
            return 0
        total = 0
        for e in [e for _, e, g in self.doc.contexts() if _role_of(g) is None]:
            mask = self._thin_mask_of(e)
            if mask is None or not mask.any():
                continue
            total += int(len(np.unique(self.engine.cache.get(e)["tri_fid"][mask])))
        return total

    # ------------------------------------------------------------- tight fits
    def _tight_clearance(self):
        """The room two parts have to keep between them, or None when the view has nothing
        to ask about.

        The room is the clearance of 3D Printing > Clearance Test, the same one the threads
        and the hole tools use; a clearance of nothing says the printer lays nothing between
        two parts, and under that distance there is nothing left to look for."""
        if not getattr(self.settings, "show_tight_fits", False):
            return None
        room = getattr(self.settings, "screw_hole_clearance", 0.0)
        return room if room > 0 else None

    def _tight_context(self, ents):
        """``(room, answers)`` the faces of `ents` are tinted with, or None when off; the
        answers that landed belong in the key of its render cache, for the same reason as
        with the thin walls (see :meth:`_thin_context`)."""
        room = self._tight_clearance()
        if room is None:
            return None
        return room, self._tight_landed.get(id(ents), 0)

    def _tight_key(self, room):
        """What one check of the whole model belongs to: every part of it at the version it
        was measured at, and the room it was measured with.

        A part is only tight in respect to the others, so one part that is edited asks for
        the whole model again - the part it was close to may have moved away by now. A support
        volume is not one of the parts: it is made to touch them."""
        return tuple((id(e), e.version)
                     for _c, e, g in self.doc.contexts() if _role_of(g) is None) + (room,)

    def _tight_mask_of(self, ents):
        """The bool per triangle of `ents` whether it stands closer to another part than the
        clearance, or None while that is not known yet.

        One part tells nothing without the others, so the check is of the whole model at
        once (see :meth:`_run_tight_job`) and every face is drawn as it always was until the
        answer has landed. An answer belongs to one version of every part of the model and
        to one clearance; a check of a model that has moved on since is never shown."""
        room = self._tight_clearance()
        if room is None:
            return None
        key = self._tight_key(room)
        got = self._tight_answer
        if got is not None and got[0] == key:
            held = got[1].get(id(ents))
            # the triangles of this version are counted the same way the answer was made;
            # when the render cache says otherwise, the answer is not the one for this frame
            if held is None or held[0] is not ents:
                return None
            mask = held[1]
            return mask if len(mask) == len(self.engine.cache.get(ents)["tri_fid"]) else None
        if self._tight_busy != key:
            self._tight_busy = key
            self._run_tight_job(key)
        return None

    def _run_tight_job(self, key):
        """The check of every part against every other one off the GUI thread.

        The parts are gathered here, on the GUI thread, because the window takes no input
        while a job runs and the model cannot change under it; their solids are made in the
        worker, which is what costs seconds on a big model. The answer is one mask per part,
        in the order the parts stood; a check that is too late for the model it was made for
        is thrown away, which is why a check too late for this frame costs nothing."""
        room = key[-1]
        # the centres of the triangles the view draws, in the coordinates the model is drawn
        # in: a part can only be measured against the others where they really stand
        parts = [(e, np.array(self.engine.cache.get(e)["T"].mean(axis=1), dtype=float))
                 for _c, e, g in self.doc.contexts() if _role_of(g) is None]
        checking = tr("Checking clearances…")

        def work(report):
            return clearance.tight_masks([(e.to_manifold(), points) for e, points in parts],
                                         room)

        def told():
            """The bar showed what it was waiting for; now it shows the count, or - when
            nothing is tight - the instruction of the tool again."""
            if self._status_note == checking:
                self._status_note = None
            self._view_hint = ""            # so the count comes back even when it is the same
            self.update_view_status()
            self.refresh_status()
            if self._status_shown == checking:
                # nothing tight and a tool without an instruction: the wait is over all the same
                self._status_shown = ""
                self.status_changed.emit("")

        def stored(masks):
            if not background.alive(self) or self._tight_busy != key:
                return                     # another file, another model, another clearance
            by = {}
            for (e, _), mask in zip(parts, masks):
                by[id(e)] = (e, np.asarray(mask, dtype=bool))
            if len(by) != len(parts):
                # an answer without every part is no answer, and asking for the same broken
                # check again next frame would only wait for it again
                told()
                return
            self._tight_busy = None
            for landed in by:
                self._tight_landed[landed] = self._tight_landed.get(landed, 0) + 1
            self._tight_answer = (key, by)
            told()
            self.update()

        def failed(_exc):
            # nothing to show: the faces stay as they are, and the next frame asks again
            if self._tight_busy == key:
                self._tight_busy = None
                told()

        background.run(self.window(), work, text=checking, can_cancel=False,
                       show_dialog=False, on_done=stored, on_error=failed)
        self.set_status(checking)

    def _tight_faces(self):
        """How many faces of the model stand closer to another part than the clearance; 0
        when unknown. A face counts as soon as one of its triangles does, read off the
        triangles the render cache holds (see :meth:`_thin_faces`); while a check still runs
        its parts count as nothing, so the number can still rise before the view is done."""
        if self._tight_clearance() is None:
            return 0
        total = 0
        for e in [e for _, e, g in self.doc.contexts() if _role_of(g) is None]:
            mask = self._tight_mask_of(e)
            if mask is None or not mask.any():
                continue
            total += int(len(np.unique(self.engine.cache.get(e)["tri_fid"][mask])))
        return total

    # ------------------------------------------------------------- collisions
    def _collision_on(self):
        """Whether View > Face Style > Collisions asks about the model at all."""
        return bool(getattr(self.settings, "show_collisions", False))

    def _collision_parts(self):
        """(entities, group) of every part the view measures: everything that is drawn, the
        support volumes left out, because those are made to touch the parts (see
        :meth:`_role_of`). The loose geometry of the model itself is a part like any other."""
        return [(e, g) for _c, e, g in self.doc.contexts() if _role_of(g) is None]

    def _collision_key(self):
        """What one check of the model belongs to: every part of it at the version it was
        measured at, and - for a model that is too big to measure whole - the groups that were
        selected when it was measured.

        Two parts only collide in respect to one another, so one part that is moved asks for
        the whole model again: the part it stood in may have moved away by now."""
        parts = self._collision_parts()
        key = tuple((id(e), e.version) for e, _g in parts)
        if len(parts) > self.COLLISION_PARTS:
            key += tuple(sorted(i for k, _c, i in self.doc.selection if k == "g"))
        return key

    def _collision_near(self):
        """The parts to measure and, when the model was too big to measure whole, how many
        parts there were to measure - ``None`` when all of them were.

        More than :data:`COLLISION_PARTS` parts is a boolean for every pair of them, and the
        one who selected a part wants to know about that part: then only the selected parts and
        the parts whose box touches a selected one are measured, and the bar says how much of
        the model that was. Nothing selected says nothing about where to look, so then the
        whole model is measured as well."""
        parts = self._collision_parts()
        if len(parts) <= self.COLLISION_PARTS:
            return parts, None
        sel = {i for k, _c, i in self.doc.selection if k == "g"}
        boxes = [e.bbox() for e, g in parts if g is not None and g.id in sel]
        boxes = [b for b in boxes if b is not None]
        if not boxes:
            return parts, None
        chosen = [(e, g) for e, g in parts
                  if e.bbox() is not None and any(_boxes_touch(e.bbox(), b) for b in boxes)]
        return chosen, len(parts)

    def _collision_names(self):
        """id(entities) -> what that part is called in the bar; the loose geometry of the model
        itself has no name of its own."""
        return {id(e): (g.name if g else tr("Loose geometry"))
                for e, g in self._collision_parts()}

    def collisions_changed(self):
        """The model changed, the selection changed, or the view was switched on: the model is
        measured again, but only once the work that changes it has come to rest.

        Moving a part with the Move tool changes the model at every snap of a step and at every
        press of an arrow key; asking for a boolean of the whole model at each of those is what
        a big model cannot pay. The timer starts again at every change, so the model is
        measured after it has been left alone for :data:`COLLISION_SETTLE_MS` - which is why the
        red shapes of an overlap follow a part a moment after it is put down."""
        if self._collision_on():
            self._collision_timer.start()

    def forget_collisions(self):
        """View > Face Style > Collisions off: the red shapes go at once, and with them the
        answer of a check that is still on its way (see :meth:`_forget_caps`)."""
        self._collision_timer.stop()
        self._collision_busy = None
        self._collision_answer = None
        self.update()

    def _check_collisions(self):
        """What the timer of :meth:`collisions_changed` waits for: the check, but only when
        there is something new to check.

        A click in the model changes the selection without changing a part, and the view is
        switched on for a model that was measured a moment ago; those calls end here, because
        the key of the model has not moved and the answer of the last check still stands."""
        if not self._collision_on():
            return
        key = self._collision_key()
        if self._collision_busy == key:
            return
        if self._collision_answer is not None and self._collision_answer[0] == key:
            return
        chosen, of_model = self._collision_near()
        self._collision_busy = key
        self._run_collision_job(key, chosen, len(chosen) if of_model is not None else None)

    def _run_collision_job(self, key, parts, measured):
        """Every part against every other one, off the GUI thread (see :meth:`_run_tight_job`).

        The parts are gathered before the job starts, on the GUI thread, because the window
        takes no input while a job runs and the model cannot change under it; their solids and
        their booleans are made in the worker, which is what costs seconds on a big model.
        `measured` is how many parts of the model were measured, ``None`` when all of them
        were; a check too late for the model it was made for is thrown away, which is why a
        check too late for this frame costs nothing."""
        ents = [e for e, _g in parts]
        checking = tr("Checking for collisions…")

        def work(report):
            found = collisions.collisions([(id(e), e.to_manifold()) for e in ents])
            for hit in found:             # the triangles of the overlap, as one draw wants them
                verts, tris = hit.pop("mesh")
                hit["pts"] = np.asarray(verts, dtype=np.float32)[
                    np.asarray(tris, dtype=np.int32)].reshape(-1, 3)
            return found

        def told():
            """The bar showed what it was waiting for; now it says what was found, or - when
            there is nothing to say - the instruction of the tool again."""
            if self._status_note == checking:
                self._status_note = None
            self._view_hint = ""            # so the count comes back even when it is the same
            self.update_view_status()
            self.refresh_status()
            if self._status_shown == checking:
                # nothing to say and a tool without an instruction: the wait is over all the same
                self._status_shown = ""
                self.status_changed.emit("")

        def stored(found):
            if not background.alive(self) or self._collision_busy != key:
                return                     # another file, another model, another selection
            self._collision_busy = None
            self._collision_answer = (key, list(found), measured)
            told()
            self.update()

        def failed(_exc):
            # nothing to show: the model stays as it is, and the next change asks again
            if self._collision_busy == key:
                self._collision_busy = None
                told()

        background.run(self.window(), work, text=checking, can_cancel=False,
                       show_dialog=False, on_done=stored, on_error=failed)
        self.set_status(checking)

    def _collision_hits(self):
        """The pairs of parts that stand in one another, the biggest overlap first, or [] when
        that is not known - or when the view is off.

        An answer belongs to one version of every part of the model, so a check that came in
        too late for the model it was made for is never shown (see :meth:`_tight_mask_of`)."""
        if not self._collision_on():
            return []
        got = self._collision_answer
        if got is None or got[0] != self._collision_key():
            return []
        return got[1]

    def _collision_hint(self):
        """What the collision view has to say in the bar: the pairs that stand in one another,
        the biggest first and three at most, or that none does. "" when the view has nothing
        to say - off, or a model that is being looked at.

        The volume of an overlap is said in cm³, as the print tools do: 420 mm³ of plastic that
        should not be there is a lot, and reads as nothing in the millimetres a model is
        measured in."""
        if not self._collision_on():
            return ""
        got = self._collision_answer
        if got is None or got[0] != self._collision_key():
            return ""                    # the wait: the bar already says what it waits for
        hits = got[1]
        if not hits:
            text = tr("No collisions")
        else:
            names = self._collision_names()
            listed = ", ".join(
                tr("{a} and {b} ({vol} cm³)").format(
                    a=names.get(h["a"], tr("Loose geometry")),
                    b=names.get(h["b"], tr("Loose geometry")),
                    vol=f"{h['volume'] / 1000.0:.2f}")
                for h in hits[:3])
            if len(hits) > 3:
                listed += ", …"          # the three biggest are enough to find the trouble
            text = trn("{n} collision: {list}", "{n} collisions: {list}",
                       len(hits)).format(n=len(hits), list=listed)
        if got[2] is not None:
            # a big model is only measured near the selection; the bar says how much of the
            # model the answer is about, so it is never taken for the whole of it
            text += " " + trn("(only the {n} part near the selection was checked)",
                              "(only the {n} parts near the selection were checked)",
                              got[2]).format(n=got[2])
        return text

    def _draw_collisions(self):
        """View > Face Style > Collisions: the solid that two parts have in common, in red.

        The overlap stands inside both parts, so the depth test of the model would hide every
        bit of it: the red is painted over the model the way the shape of Text or Logo on a Face
        is, and gives the depth buffer back as it was (see :meth:`_draw_overlay_solid`)."""
        for hit in self._collision_hits():
            self.draw_tris(hit["pts"], (*theme.rgb("collision"), 1.0), on_top=True)

    # ------------------------------------------------------------- section
    def _start_section(self, eq):
        """View > Section: the model pass cuts with the plane, so what stands behind it is gone.

        The driver reads a clip plane once, when it is handed over, and keeps it for every
        vertex after that; that is why this happens before the first group is drawn, while
        only the matrix of the camera stands (see :meth:`_draw_model`)."""
        gl.glClipPlane(gl.GL_CLIP_PLANE0, (float(eq[0]), float(eq[1]), float(eq[2]), float(eq[3])))
        self._fstate.set(gl.GL_CLIP_PLANE0, True)

    def _end_section(self):
        """The plane leaves with the model: the overlays and the next frame draw whole again."""
        self._fstate.set(gl.GL_CLIP_PLANE0, False)

    def _forget_caps(self):
        """View > Section off: the cut surfaces go, and with them the answers on their way.

        Called from a frame in which the view is off, so the context is current; the clip
        plane is switched off in `_reset_state`, but a frame that turns the view off between
        two draws has to give the driver back what a frame that turns it on takes."""
        self._end_section()
        self._cap_solids.clear()
        self._cap_cache.clear()
        self._cap_busy.clear()

    def _cap_solid_of(self, ents):
        """The solid to cut `ents` open in, or None while that solid is not there yet.

        Making a manifold of a part takes seconds on a big model and may not stop the view,
        so it runs in the background (see :meth:`_run_cap_job`). An answer of a geometry that
        moved on since is never used; the plane itself is cheap to cut, so that happens here
        on the GUI thread, as often as the panel moves it."""
        got = self._cap_solids.get(id(ents))
        if got is not None and got[0] is ents and got[1] == ents.version:
            return got[2]
        job = (ents, ents.version)
        if self._cap_busy.get(id(ents)) != job:
            self._cap_busy[id(ents)] = job
            self._run_cap_job(job)
        return None

    def _run_cap_job(self, job):
        """The solid of one geometry, made off the GUI thread.

        The window takes no input while it runs (like every job that reads the model), so the
        geometry cannot change under it; until the answer has landed the model is simply drawn
        without a cut surface, which is why a job that is too late for this frame costs
        nothing more than the wait."""
        ents = job[0]
        making = tr("Preparing the section…")

        def work(report):
            return ents.to_manifold()

        def told():
            """The bar showed what it waited for; now it shows the plane again, or - when the
            view went off in the meantime - the instruction of the tool."""
            if self._status_note == making:
                self._status_note = None
            self._view_hint = ""            # so the position of the plane comes back
            self.update_view_status()
            self.refresh_status()
            if self._status_shown == making:
                # the view is off and a tool without an instruction: the wait is over too
                self._status_shown = ""
                self.status_changed.emit("")

        def stored(md):
            if not background.alive(self) or self._cap_busy.get(id(ents)) != job:
                return                     # another file, another version, another view
            self._cap_busy.pop(id(ents), None)
            self._cap_solids[id(ents)] = job + (md,)
            told()
            self.update()

        def failed(_exc):
            # nothing to draw: the model stays cut open and the next frame asks again
            if self._cap_busy.get(id(ents)) == job:
                self._cap_busy.pop(id(ents), None)
                told()

        background.run(self.window(), work, text=making, can_cancel=False, show_dialog=False,
                       on_done=stored, on_error=failed)
        self.set_status(making)

    def _cap_triangles_of(self, ents, axis, pos):
        """The cut surface of `ents` at `pos`, or None while its solid is still being made.

        One surface per geometry is kept: a plane only stands in one place at a time, so the
        cut of the position before it is worth nothing."""
        solid = self._cap_solid_of(ents)
        if solid is None:
            return None
        key = (id(ents), ents.version, axis, pos)
        at = self._cap_cache.get(key)
        if at is None:
            at = section.cap_triangles(solid, axis, pos)
            for gone in [k for k in self._cap_cache if k[0] == id(ents)]:
                self._cap_cache.pop(gone)      # the plane moved on; the old cut goes with it
            self._cap_cache[key] = at
        return at

    def _draw_caps(self, cut, hiding_rest):
        """The surface the plane cuts open, so that a part hollow inside also looks hollow.

        The cap comes from the solid of the part (see :func:`section.cap_triangles`), so it
        covers nothing but the material the plane cut: the hollow of a tray stays empty. It is
        drawn with the clip plane switched off - its own vertices stand in that plane, where a
        hair of round-off is enough to clip them away - and lifted towards the eye with the
        polygon offset, so a flat face of the part that happens to lie in the same plane does
        not win the depth test."""
        if getattr(self.settings, "face_style", SHADED) == WIREFRAME:
            return                       # no faces in a wireframe, so nothing to close off
        axis, pos, flip, _eq = cut
        st = self._fstate
        st.set(gl.GL_CLIP_PLANE0, False)
        st.set(gl.GL_LIGHTING, True)
        st.set(gl.GL_COLOR_MATERIAL, True)
        st.set(gl.GL_POLYGON_OFFSET_FILL, True)
        gl.glPolygonOffset(-1.5, -1.5)
        gl.glColor4f(*theme.rgb("section"), 1.0)
        # the cut surface looks into the half that was cut away, so its normal stands that way
        normal = np.zeros(3, dtype=np.float32)
        normal[section.AXES.index(axis)] = -1.0 if flip else 1.0
        for ctx, ents, g, rc, mats in self._draw_contexts(hiding_rest):
            tris = self._cap_triangles_of(ents, axis, pos)
            if tris is None or not len(tris):
                continue
            pts = np.ascontiguousarray(tris.reshape(-1, 3), dtype=np.float32)
            nrm = np.repeat(normal[None, :], len(pts), axis=0)
            st.array(gl.GL_VERTEX_ARRAY, True)
            st.array(gl.GL_NORMAL_ARRAY, True)
            for M in mats:
                self._push(M)
                gl.glVertexPointer(3, gl.GL_FLOAT, 0, pts)
                gl.glNormalPointer(gl.GL_FLOAT, 0, nrm)
                gl.glDrawArrays(gl.GL_TRIANGLES, 0, len(pts))
                self._pop(M)
        st.array(gl.GL_NORMAL_ARRAY, False)
        st.array(gl.GL_VERTEX_ARRAY, False)
        st.set(gl.GL_LIGHTING, False)
        st.set(gl.GL_COLOR_MATERIAL, False)
        st.set(gl.GL_POLYGON_OFFSET_FILL, False)
        st.set(gl.GL_CLIP_PLANE0, True)      # the selection is clipped along with the model

    def update_view_status(self):
        """The status bar hint of the face style views: what the colours in the model mean.

        All of them in one line, the overhang count first, then the thin walls and the tight
        places, then what the collision view found, and the place of the section plane last. Off,
        or nothing to see: the hint goes away and the instruction of the current tool comes back,
        so the bar never keeps a count that is no longer true."""
        parts = []
        info = self._overhang_faces()
        if info and info[1]:
            parts.append(trn("Overhangs: {n} face needs support (over {limit}°)",
                             "Overhangs: {n} faces need support (over {limit}°)", info[1]).format(
                n=info[1], limit=int(self.settings.overhang_limit)))
        thin = self._thin_faces()
        if thin:
            parts.append(trn("Thin walls: {n} face under {mm} mm",
                             "Thin walls: {n} faces under {mm} mm", thin).format(
                n=thin, mm=f"{self.settings.min_wall:g}"))
        tight = self._tight_faces()
        if tight:
            parts.append(trn("Tight fits: {n} face closer than {mm} mm to another part",
                             "Tight fits: {n} faces closer than {mm} mm to another part",
                             tight).format(n=tight,
                                           mm=f"{self.settings.screw_hole_clearance:g}"))
        collide = self._collision_hint()
        if collide:
            parts.append(collide)
        cut = self.engine.section_cut()
        if cut is not None:
            # where the plane stands is otherwise invisible: the panel is often closed and the
            # slider does not say it either; the section is the one view with a place to show
            parts.append(tr("Section: {axis} at {mm} mm").format(axis=cut[0], mm=f"{cut[1]:g}"))
        text = " · ".join(parts)
        if text == self._view_hint:
            return
        was, self._view_hint = self._view_hint, text
        if text:
            self.set_status(text)
        else:
            if self._status_note == was:
                self._status_note = None
            self.refresh_status()

    def update_overhang_status(self):
        """The overhang hint on its own (see :meth:`update_view_status`), as the overhang
        view and its tests ask for it."""
        self.update_view_status()

    def release_gl(self):
        """Everything this viewport gave the driver, back before the window closes."""
        for rc in list(self._rcache.values()):
            _free_render_data(rc)
        self._rcache.clear()
        self._vbo = False

    def _draw_guides(self):
        """View > Guides: infinite dashed lines, and guide points as small crosses."""
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

    def _draw_back_edges(self, pts, buf=0):
        """The edges behind the faces: only the fragments that fail the depth test are drawn."""
        gl.glDepthMask(gl.GL_FALSE)
        gl.glDepthFunc(gl.GL_GREATER)
        self.draw_lines(pts, BACK_EDGE_COLOR, 1.0, stipple=True, buf=buf)
        gl.glDepthFunc(gl.GL_LEQUAL)
        gl.glDepthMask(gl.GL_TRUE)

    def _face_colours(self, rc, tint, alpha):
        """The front and back colours of one variant of `rc`: mixed once, never per frame."""
        v = rc.setdefault("cols", {})
        key = (bool(tint), alpha)
        out = v.get(key)
        if out is None:
            out = v[key] = (_colour_variant(rc["front"], tint, alpha),
                            _colour_variant(rc["back"], tint, alpha))
        return out

    def _face_buffers(self, rc, tint, alpha, colours=True):
        """(geometry, colour) buffer names of one set of faces; 0 means draw from memory.

        The geometry of a render cache - position and normal of every triangle - goes to
        the GPU the first time it is drawn and stays there until another version of the
        same geometry takes its place. The colours are a second buffer, so the tint of a
        selection or the alpha of x-ray is one extra colour buffer and never a new copy
        of the positions. Without buffer objects everything stays as it was: the arrays
        of `_render_data` are drawn straight from memory."""
        if not self._vbo:
            return 0, 0
        b = rc.get("gpu")
        if b is None:
            b = rc["gpu"] = _FaceBuffers()
        if not b.count:
            try:
                geom = np.empty((len(rc["pos"]), 6), dtype=np.float32)
                geom[:, :3] = rc["pos"]
                geom[:, 3:] = rc["nrm"]
                b.geom = _new_buffer(geom)
            except Exception:
                b.geom = 0
            if not b.geom:
                self._vbo = False         # this driver will not have buffers: draw from memory
                return 0, 0
            b.count = len(rc["pos"])
        if not colours:
            return b.geom, 0
        key = (bool(tint), alpha)
        cbuf = b.colours.get(key)
        if cbuf is None:
            try:
                front, back = self._face_colours(rc, tint, alpha)
                cbuf = _new_buffer(_colour_data(front, back))
            except Exception:
                cbuf = 0
            b.colours[key] = cbuf
        return b.geom, cbuf

    def _line_buffer(self, rc, key):
        """The buffer with the edge positions of `rc[key]`: uploaded once, like the faces."""
        if not self._vbo or not len(rc[key]):
            return 0
        b = rc.setdefault("lbuf", {})
        name = b.get(key)
        if name is None:
            try:
                name = _new_buffer(np.ascontiguousarray(rc[key], dtype=np.float32))
            except Exception:
                name = 0
            b[key] = name
        return name

    def _draw_faces(self, rc, alpha=1.0, tint=False, style=None):
        style = style or getattr(self.settings, "face_style", SHADED)
        if style == WIREFRAME or not len(rc["pos"]):
            return
        lit = style != HIDDENLINE
        solid = {HIDDENLINE: (HIDDEN_FACE, HIDDEN_FACE), MONOCHROME: (FRONT, BACK)}.get(style)
        gbuf, cbuf = self._face_buffers(rc, tint, alpha, solid is None)
        st = self._fstate
        st.set(gl.GL_POLYGON_OFFSET_FILL, True)
        gl.glPolygonOffset(1.5, 1.5)
        st.set(gl.GL_CULL_FACE, True)
        if alpha < 1:
            gl.glDepthMask(gl.GL_FALSE)
        st.set(gl.GL_LIGHTING, lit)
        st.set(gl.GL_COLOR_MATERIAL, lit)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, gbuf)      # 0: draw from the arrays in memory
        st.array(gl.GL_VERTEX_ARRAY, True)
        gl.glVertexPointer(3, gl.GL_FLOAT, GEOM_STRIDE if gbuf else 0,
                           _at(0) if gbuf else rc["pos"])
        st.array(gl.GL_NORMAL_ARRAY, lit)
        if lit:
            gl.glNormalPointer(gl.GL_FLOAT, GEOM_STRIDE if gbuf else 0,
                               _at(3 * 4) if gbuf else rc["nrm"])
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)
        passes = ((gl.GL_BACK, rc["front"]), (gl.GL_FRONT, rc["back"])) if solid is None else \
            ((gl.GL_BACK, solid[0]), (gl.GL_FRONT, solid[1]))
        cols = self._face_colours(rc, tint, alpha) if solid is None else None
        for cull, want in passes:
            gl.glCullFace(cull)
            if isinstance(want, np.ndarray):
                st.array(gl.GL_COLOR_ARRAY, True)
                gl.glBindBuffer(gl.GL_ARRAY_BUFFER, cbuf)   # 0: the colours in memory
                if cbuf:
                    gl.glColorPointer(4, gl.GL_FLOAT, COLOUR_STRIDE,
                                      _at(0 if want is rc["front"] else 4 * 4))
                else:
                    gl.glColorPointer(4, gl.GL_FLOAT, 0, cols[0] if want is rc["front"] else cols[1])
                gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)
            else:
                st.array(gl.GL_COLOR_ARRAY, False)
                gl.glColor4f(*_tint(want, tint), alpha)
            gl.glDrawArrays(gl.GL_TRIANGLES, 0, len(rc["pos"]))
        # lighting and the colour and normal arrays off, because the edges between the
        # groups are drawn with a plain colour: an array left on would be read for every
        # line vertex too, past its end when a group has more line than face vertices.
        # Culling, the polygon offset and the position array may stay on: lines ignore
        # the first two and set their own position pointer. `_end_model_pass` returns to
        # the state of `_reset_state`.
        st.array(gl.GL_COLOR_ARRAY, False)
        st.array(gl.GL_NORMAL_ARRAY, False)
        st.set(gl.GL_LIGHTING, False)
        st.set(gl.GL_COLOR_MATERIAL, False)
        gl.glDepthMask(gl.GL_TRUE)

    def _end_model_pass(self):
        """The state a frame starts with, so the overlays draw exactly as they always did."""
        st = self._fstate
        for cap in _ARRAYS:
            st.array(cap, False)
        st.set(gl.GL_CULL_FACE, False)
        st.set(gl.GL_POLYGON_OFFSET_FILL, False)
        st.set(gl.GL_LIGHTING, False)
        st.set(gl.GL_COLOR_MATERIAL, False)

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
        """Selected faces: the blue dot pattern of SketchUp, not an opaque layer."""
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
        """Print Check: the edges and faces of the chosen problem, thick and red."""
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
    def set_overlay_lines(self, lines):
        """Lines a dialog wants to see over the model, until it asks for nothing again.

        `lines` holds (points, colour, width) per polyline; they are drawn above the
        model, so the 3D toolpaths of the mill stay visible also behind a face.
        """
        self.overlay_lines = list(lines or ())
        self.update()

    def clear_overlay_lines(self):
        """What `set_overlay_lines` put over the model is gone again."""
        self.set_overlay_lines(())

    def set_overlay_solid(self, md, color=(0.35, 0.6, 1.0), alpha=0.45):
        """One see-through solid a dialog wants to see over the model, until it asks for nothing
        again.

        The shape "Text or Logo on Face…" puts on a face is shown this way: whether a logo is big
        enough and stands where it should is judged in the model, not in a picture beside the
        fields, and the user keeps turning the model round while they choose. It is drawn over the
        model, so the engraving that is cut *in* the part is visible too.
        """
        self.overlay_solid = None if md is None else (md, tuple(color), float(alpha))
        self.update()

    def clear_overlay_solid(self):
        """What `set_overlay_solid` put over the model is gone again."""
        self.set_overlay_solid(None)

    def _draw_overlay_solid(self):
        """The see-through shape of a dialog: the ghost of a shape on a face of the model."""
        if self.overlay_solid is None:
            return
        md, color, alpha = self.overlay_solid
        gl.glDisable(gl.GL_DEPTH_TEST)        # also the part in front of it must let it through
        try:
            self.draw_solid_preview(md, color, alpha)
        finally:
            gl.glEnable(gl.GL_DEPTH_TEST)

    def _draw_overlay_lines(self):
        """The lines of a dialog: the toolpaths of the 3D mill and the like."""
        for pts, color, width in self.overlay_lines:
            self.draw_strip(pts, color, width, on_top=True)

    def draw_lines(self, pts, color, width=1.0, stipple=False, on_top=False, buf=0):
        pts = np.asarray(pts, dtype=np.float32).reshape(-1, 3)
        if not len(pts):
            return
        self._prim(gl.GL_LINES, pts, color, width, stipple, on_top, buf)

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
            gl.glPointSize(self._width(size))
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

    def _width(self, width):
        """A line width times the line width of the preferences (Window > Preferences > Display)."""
        return width * getattr(self.settings, "line_scale", 1.0)

    def _prim(self, mode, pts, color, width, stipple, on_top, buf=0):
        st = self._fstate
        if on_top:
            gl.glDisable(gl.GL_DEPTH_TEST)
        if stipple:
            gl.glEnable(gl.GL_LINE_STIPPLE)
            gl.glLineStipple(2, 0x3333)
        gl.glLineWidth(self._width(width))
        gl.glColor4f(*color) if len(color) == 4 else gl.glColor3f(*color)
        st.array(gl.GL_VERTEX_ARRAY, True)
        if buf:
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, buf)
            gl.glVertexPointer(3, gl.GL_FLOAT, 0, _at(0))
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)
        else:
            gl.glVertexPointer(3, gl.GL_FLOAT, 0, np.ascontiguousarray(pts))
        gl.glDrawArrays(mode, 0, len(pts))
        st.array(gl.GL_VERTEX_ARRAY, False)
        if stipple:
            gl.glDisable(gl.GL_LINE_STIPPLE)
        if on_top:
            gl.glEnable(gl.GL_DEPTH_TEST)

    def draw_solid_preview(self, md, color, alpha=0.45):
        st = self._fstate
        gl.glDepthMask(gl.GL_FALSE)
        st.set(gl.GL_LIGHTING, True)
        st.set(gl.GL_COLOR_MATERIAL, True)
        gl.glColor4f(*color, alpha)
        st.array(gl.GL_VERTEX_ARRAY, True)
        st.array(gl.GL_NORMAL_ARRAY, True)
        gl.glVertexPointer(3, gl.GL_FLOAT, 0, md.tri_positions)
        gl.glNormalPointer(gl.GL_FLOAT, 0, md.tri_normals)
        gl.glDrawArrays(gl.GL_TRIANGLES, 0, len(md.tri_positions))
        st.array(gl.GL_NORMAL_ARRAY, False)
        st.array(gl.GL_VERTEX_ARRAY, False)
        st.set(gl.GL_LIGHTING, False)
        st.set(gl.GL_COLOR_MATERIAL, False)
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
        """The inference point of SketchUp: a different shape per kind, in its own colour."""
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
        p.setPen(theme.qcolor("view_text"))
        p.drawText(10, self.height() - 10,
                   tr("Parallel") if self.cam.ortho else tr("Perspective"))
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
            shift = bool(e.modifiers() & Qt.ShiftModifier)
            self._drag = "orbit" if shift == self.middle_pans else "pan"
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
        self._cam_settle.start()       # bundle scrolling: store one state after 0.5 s of quiet
        self.update()

    # A 3D mouse at full deflection: degrees per second, screen heights per second, zoom factor.
    SPACEMOUSE_ROTATE = 120.0
    SPACEMOUSE_PAN = 1.2
    SPACEMOUSE_ZOOM = 1.6

    def spacemouse_move(self, m, speed=1.0):
        """A motion of the 3D mouse: the model moves along with the cap.

        Pushing the cap right or up moves the model along, pulling it towards you zooms in,
        twisting the cap turns the model about the vertical axis and tilting tilts it.
        """
        k = float(speed) * m.dt
        cam = self.cam
        cam.yaw -= m.ry * self.SPACEMOUSE_ROTATE * k
        cam.pitch = max(-89.99, min(89.99, cam.pitch + m.rx * self.SPACEMOUSE_ROTATE * k))
        _, r, u = cam.basis()
        step = 2 * cam.half_height() * self.SPACEMOUSE_PAN * k
        cam.target = cam.target - r * (m.tx * step) - u * (m.ty * step)
        cam.distance = max(0.5, cam.distance * float(np.exp(-m.tz * self.SPACEMOUSE_ZOOM * k)))
        self._cam_settle.start()       # after the motion, one step in Camera > Previous
        self.update()

    # ------------------------------------------------------------- camera
    def commit_camera(self):
        """Remember the current camera as the newest step of the view history (Camera > Previous)."""
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
        """Camera > Previous: back to the previous camera position."""
        if self.can_prev_camera():
            self._goto_camera(self.cam_idx - 1)

    def next_camera(self):
        """Camera > Next: forward again after Previous."""
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
        """Camera > Zoom Window: make the screen rectangle fill the view."""
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

    def visible_bbox(self):
        """The box of everything this view draws: the visible groups of the plate it shows.

        `Document.model_bbox` knows every group, but the plate chooser takes the other plates off
        the screen, and Zoom Selection frames what you see - otherwise the plate you picked would
        stay in a corner of the view."""
        pts = []
        for _ctx, ents, _g in self._plate_contexts(False):
            bb = ents.bbox()
            if bb is not None:
                pts.extend(bb)
        if not pts:
            return None
        P = np.asarray(pts, dtype=float)
        return np.array([P.min(axis=0), P.max(axis=0)])

    def zoom_extents(self):
        bb = self.visible_bbox()
        if bb is None:
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
        """Zoom in on the given items (the Print Check panel uses this on double click)."""
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

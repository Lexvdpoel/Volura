"""Fast drawing of the 3D view: PyOpenGL without a check after every call, and vertex buffers.

Runs without a display (Qt offscreen platform), like tests/test_ui_spec.py. Most of it
looks at what the view asks of the driver through a stand-in that remembers the GL state,
so it also works on a machine without OpenGL; only the picture comparison needs a real
context and is skipped when this machine has none.
"""
import os
import subprocess
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from PySide6.QtGui import QImage  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

_app = QApplication.instance() or QApplication(sys.argv[:1])

from volura import _glconfig, brep, viewport  # noqa: E402
from volura.errors import configure_pyopengl  # noqa: E402
from volura.geom import Plane  # noqa: E402

GROUND = Plane((0, 0, 0), (0, 0, 1))
RECT = np.array([[0, 0], [40, 0], [40, 30], [0, 30]], dtype=float)


@pytest.fixture(scope="module")
def win():
    from volura.app import MainWindow
    w = MainWindow()
    w.doc.root.add_shape(GROUND, [RECT])
    yield w
    w.doc.dirty = False
    w.close()


# ----------------------------------------------------- PyOpenGL error checking

def _run_snippet(code, extra_env):
    """Run a piece of Python in a fresh process and return what it printed."""
    env = dict(os.environ)
    env["PYTHONPATH"] = ROOT + os.pathsep + env.get("PYTHONPATH", "")
    for key in ("VOLURA_GL_DEBUG", "PYOPENGL_ERROR_CHECKING", "PYOPENGL_ERROR_LOGGING"):
        env.pop(key, None)
    env.update(extra_env)
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         env=env, cwd=ROOT)
    assert out.returncode == 0, out.stderr
    return out.stdout.strip()


IMPORT_FIRST = "import volura; from OpenGL import GL; import volura._glconfig as g; print(g.error_checking())"
IMPORT_LATE = "from OpenGL import GL; import volura._glconfig as g; print(g.error_checking())"


def test_pyopengl_stops_checking_every_call():
    """A process that imports volura first gets a PyOpenGL that does not call glGetError."""
    assert _run_snippet(IMPORT_FIRST, {}) == "False"


def test_volura_gl_debug_keeps_the_checks_on():
    """VOLURA_GL_DEBUG=1 is the way back to the slow, checking PyOpenGL."""
    assert _run_snippet(IMPORT_FIRST, {"VOLURA_GL_DEBUG": "1"}) == "True"


def test_the_choice_has_to_be_made_before_opengl():
    """Without volura the check is on: it is volura that switches it off, and only in time."""
    assert _run_snippet(IMPORT_LATE, {}) == "True"


def test_configure_pyopengl_follows_the_environment(monkeypatch):
    saved = {flag: os.environ.get(flag) for flag in _glconfig.FLAGS}
    try:
        for value, want in ((None, False), ("", False), ("0", False), ("1", True), ("true", True)):
            if value is None:
                monkeypatch.delenv("VOLURA_GL_DEBUG", raising=False)
            else:
                monkeypatch.setenv("VOLURA_GL_DEBUG", value)
            assert configure_pyopengl() is want
            assert os.environ["PYOPENGL_ERROR_CHECKING"] == ("1" if want else "0")
            assert os.environ["PYOPENGL_ERROR_LOGGING"] == ("1" if want else "0")
    finally:
        monkeypatch.delenv("VOLURA_GL_DEBUG", raising=False)
        for flag, value in saved.items():
            if value is None:
                os.environ.pop(flag, None)
            else:
                os.environ[flag] = value
        configure_pyopengl()      # the real OpenGL module goes back to what it was


def test_configure_pyopengl_survives_a_stub_module(monkeypatch):
    """clear_gl_errors works with a stand-in for PyOpenGL, so configure has to accept it too."""
    import types
    stub = types.ModuleType("OpenGL")
    monkeypatch.setitem(sys.modules, "OpenGL", stub)
    assert configure_pyopengl() is False
    assert stub.ERROR_CHECKING is False


# ------------------------------------------------------------- the GL stand-in

class FakeGL:
    """A stand-in for PyOpenGL that keeps the state, so a test can look over its shoulder.

    glEnable and glEnableClientState are kept apart on purpose: they are different kinds
    of state and switching a client array with the wrong one is a drawing bug."""

    def __init__(self, names=True):
        self.calls = []
        self.caps = {}
        self.arrays = {}
        self.pointer = {}
        self.draws = []
        self.names = names          # can this driver hand out buffer names?
        self.bound = 0
        self.bindings = []
        self.uploads = []           # (buffer, bytes) of every glBufferData
        self.handed_out = []        # every name glGenBuffers gave
        self.freed = []             # every name glDeleteBuffers took back
        self._next = 0

    # -- the state
    def glEnable(self, cap):
        self.calls.append(("enable", cap))
        self.caps[cap] = True

    def glDisable(self, cap):
        self.calls.append(("disable", cap))
        self.caps[cap] = False

    def glEnableClientState(self, cap):
        self.calls.append(("enable_client", cap))
        self.arrays[cap] = True

    def glDisableClientState(self, cap):
        self.calls.append(("disable_client", cap))
        self.arrays[cap] = False

    def glIsEnabled(self, cap):
        return 1 if (self.caps.get(cap) or self.arrays.get(cap)) else 0

    def is_on(self, cap):
        return bool(self.caps.get(cap) or self.arrays.get(cap))

    # -- the arrays
    def glVertexPointer(self, size, type_, stride, ptr):
        self.pointer["v"] = (self.bound, stride, ptr)

    def glNormalPointer(self, type_, stride, ptr):
        self.pointer["n"] = (self.bound, stride, ptr)

    def glColorPointer(self, size, type_, stride, ptr):
        self.pointer["c"] = (self.bound, stride, ptr)

    def glDrawArrays(self, mode, first, count):
        self.draws.append({"mode": mode, "count": count, "caps": dict(self.caps),
                           "arrays": dict(self.arrays), "pointer": dict(self.pointer)})

    # -- buffer objects
    def glGenBuffers(self, n):
        if not self.names:
            return 0
        self._next += 1
        name = self._next
        self.handed_out.append(name)
        return name

    def glBindBuffer(self, target, name):
        self.bound = int(name)
        self.bindings.append(self.bound)

    def glBufferData(self, target, size, data, usage):
        self.uploads.append((self.bound, int(size)))

    def glDeleteBuffers(self, n, names):
        gone = [int(x) for x in np.asarray(names).reshape(-1)]
        self.freed += gone

    def glGetError(self):
        return 0

    def __getattr__(self, name):
        if name.startswith("GL_"):
            from OpenGL import GL as real
            return getattr(real, name)

        def call(*args, **kw):
            self.calls.append((name,) + args)
            return 0
        return call


def start(monkeypatch, win, vbo=True, names=True):
    """The view with the stand-in installed, an empty render cache and no leftover state."""
    fake = FakeGL(names=names)
    monkeypatch.setattr(viewport, "gl", fake)
    view = win.view
    view._rcache.clear()
    view._gl_ctx = view.context()
    view._vbo = vbo
    view._fstate = viewport._FaceState()
    return fake


def draw(view):
    view._draw_model()


def face_draws(fake):
    return [d for d in fake.draws if d["mode"] == viewport.gl.GL_TRIANGLES]


def geometry_buffers(view):
    """{id(ents): buffer name of the positions and normals of that geometry}."""
    return {k: rc["gpu"].geom for k, rc in view._rcache.items() if rc.get("gpu")}


# ------------------------------------------------------------------- the state

def test_client_arrays_are_switched_with_client_state_calls(win, monkeypatch):
    """GL_VERTEX_ARRAY and friends are client state: glEnable would simply be wrong GL."""
    fake = start(monkeypatch, win)
    draw(win.view)
    wrong = [c for c in fake.calls if c[0] in ("enable", "disable") and c[1] in viewport._ARRAYS]
    assert not wrong, f"client arrays switched with glEnable/glDisable: {wrong}"
    assert fake.calls.count(("enable_client", viewport.gl.GL_VERTEX_ARRAY)) >= 1


def test_every_group_is_drawn_with_the_state_it_needs(win, monkeypatch):
    """The state after a group is not the end of the story: the next group has to be right too."""
    fake = start(monkeypatch, win)
    win.doc.root.pushpull(next(iter(win.doc.root.F)), 10.0)
    win.doc.add_group(_tri_group(), "#cc3333")
    draw(win.view)
    tri = face_draws(fake)
    assert len(tri) >= 4, f"two groups of faces should give at least four draws, got {len(tri)}"
    for d in tri:
        assert d["arrays"].get(viewport.gl.GL_VERTEX_ARRAY), "positions are not being read"
        assert d["arrays"].get(viewport.gl.GL_NORMAL_ARRAY), "normals are not being read"
        assert d["arrays"].get(viewport.gl.GL_COLOR_ARRAY), "face colours are not being read"
        assert d["caps"].get(viewport.gl.GL_LIGHTING), "the faces are drawn unlit"
        assert d["caps"].get(viewport.gl.GL_COLOR_MATERIAL), "glColor does not reach the faces"
        assert d["caps"].get(viewport.gl.GL_CULL_FACE), "nothing is culled"
        assert d["caps"].get(viewport.gl.GL_POLYGON_OFFSET_FILL), "edges are not lifted off the faces"


def test_the_model_pass_leaves_the_frame_state_behind(win, monkeypatch):
    """What a frame starts with is what the overlays and the next frame have to find again."""
    fake = start(monkeypatch, win)
    draw(win.view)
    left = [cap for cap in viewport._CAPS + viewport._ARRAYS if fake.is_on(cap)]
    assert not left, f"the model pass left these switched on: {left}"


def test_state_is_taken_from_the_driver_at_the_start_of_a_frame(win, monkeypatch):
    """A frame that starts with lighting on (something else left it) is followed correctly."""
    fake = start(monkeypatch, win)
    fake.caps[viewport.gl.GL_LIGHTING] = True
    fake.caps[viewport.gl.GL_COLOR_MATERIAL] = True
    fake.arrays[viewport.gl.GL_VERTEX_ARRAY] = True
    win.view._fstate = viewport._FaceState.capture()
    draw(win.view)
    for d in face_draws(fake):
        assert d["caps"].get(viewport.gl.GL_LIGHTING)
    assert not [cap for cap in viewport._CAPS + viewport._ARRAYS if fake.is_on(cap)]


def test_a_state_change_is_made_once_instead_of_per_group(win, monkeypatch):
    """Two groups that both want culling on cost one glEnable together instead of two."""
    fake = start(monkeypatch, win)
    win.doc.add_group(_tri_group(), "#cc3333")
    win.doc.add_group(_tri_group(), "#33aa33")
    draw(win.view)
    groups = len(list(win.doc.contexts()))
    assert groups >= 3, "this test is about more than one group"
    culls = fake.calls.count(("enable", viewport.gl.GL_CULL_FACE))
    assert culls < groups, f"one glEnable for {groups} groups is the point, not {culls}"


# ------------------------------------------------------------------- buffers

def _tri_group():
    """A small group of geometry that is not the model itself."""
    from volura.brep import Entities
    e = Entities()
    e.add_shape(GROUND, [np.array([[0, 0], [20, 0], [0, 20]], dtype=float)])
    return e


def test_one_set_of_buffers_per_version_of_the_geometry(win, monkeypatch):
    """The first frame uploads, the frames after that only draw: nothing is sent again."""
    fake = start(monkeypatch, win)
    draw(win.view)
    first = geometry_buffers(win.view)
    uploads = len(fake.uploads)
    assert first and all(first.values()), f"every group wants a buffer: {first}"
    draw(win.view)
    draw(win.view)
    assert len(fake.uploads) == uploads, "a frame sent data to the GPU again"
    assert geometry_buffers(win.view) == first, "the buffers were replaced without a reason"


def test_a_moved_face_gets_new_buffers(win, monkeypatch):
    """After a change the GPU has to hold the new positions, and the old ones are given back."""
    fake = start(monkeypatch, win)
    draw(win.view)
    ents = win.doc.root
    old = geometry_buffers(win.view)[id(ents)]
    win.doc.checkpoint()
    ents.pushpull(next(iter(ents.F)), 12.0)
    draw(win.view)
    new = geometry_buffers(win.view)[id(ents)]
    assert new != old, "the buffer of the old version is still being drawn"
    assert old in fake.freed, f"the old buffer {old} was not given back"
    assert new in fake.handed_out and new not in fake.freed


def test_a_drawn_line_gets_new_buffers(win, monkeypatch):
    """The same, for the everyday change: a line in the model."""
    fake = start(monkeypatch, win)
    draw(win.view)
    ents = win.doc.root
    old = geometry_buffers(win.view)[id(ents)]
    old_edges = win.view._rcache[id(ents)]["lbuf"]["edges"]
    win.doc.checkpoint()
    ents.add_line((0, 0, 0), (40, 30, 0))
    draw(win.view)
    now = win.view._rcache[id(ents)]
    assert now["gpu"].geom != old, "a new line has to reach the GPU"
    assert now["lbuf"]["edges"] != old_edges, "the edges changed, the buffer may not stay"
    assert old in fake.freed and old_edges in fake.freed


def test_undo_and_redo_refresh_the_buffers(win, monkeypatch):
    """Undo draws the old geometry again: the buffer of the undone version may not be reused."""
    fake = start(monkeypatch, win)
    ents = win.doc.root
    draw(win.view)
    before = geometry_buffers(win.view)[id(ents)]
    win.doc.checkpoint()
    ents.pushpull(next(iter(ents.F)), 12.0)
    draw(win.view)
    moved = geometry_buffers(win.view)[id(ents)]
    assert win.doc.undo()
    draw(win.view)
    assert geometry_buffers(win.view)[id(ents)] not in (before, moved)
    assert win.doc.redo()
    draw(win.view)
    after = geometry_buffers(win.view)[id(ents)]
    assert after not in (before, moved)
    assert fake.freed.count(before) >= 1 and fake.freed.count(moved) >= 1


def test_a_removed_group_gives_its_buffers_back(win, monkeypatch):
    """A deleted group is never drawn again, so its buffers go back to the driver."""
    fake = start(monkeypatch, win)
    group = win.doc.add_group(_tri_group(), "#cc3333")
    draw(win.view)
    gone = geometry_buffers(win.view)[id(group.ents)]
    win.doc.remove_group(group.id)
    draw(win.view)
    assert gone in fake.freed, "the buffers of the deleted group are still in use"
    assert gone not in geometry_buffers(win.view).values()


def test_release_gl_gives_everything_back(win, monkeypatch):
    """Closing the window: no buffer of this view may stay behind."""
    fake = start(monkeypatch, win)
    draw(win.view)
    live = set(fake.handed_out) - set(fake.freed)
    assert live, "the model pass should have left buffers behind to give back"
    win.view.release_gl()
    assert win.view._rcache == {}
    assert win.view._vbo is False
    assert set(fake.freed) >= live, f"these stayed: {live - set(fake.freed)}"


def test_initialize_gl_of_the_same_context_does_not_leak(win, monkeypatch):
    """Qt may initialize again in the same context: then the old names are still buffers."""
    fake = start(monkeypatch, win)
    draw(win.view)
    live = set(fake.handed_out) - set(fake.freed)
    ctx = win.view._gl_ctx
    win.view.initializeGL()
    assert win.view._gl_ctx is ctx
    assert set(fake.freed) >= live, "the buffers of the old render cache were not given back"
    assert win.view._rcache == {}


def test_initialize_gl_of_another_context_only_forgets(win, monkeypatch):
    """Another context: those names mean nothing there, so deleting them would be wrong."""
    fake = start(monkeypatch, win)
    draw(win.view)
    live = set(fake.handed_out) - set(fake.freed)
    freed_before = set(fake.freed)
    monkeypatch.setattr(win.view, "context", lambda: object(), raising=False)
    win.view.initializeGL()
    new_frees = set(fake.freed) - freed_before
    assert not (new_frees & live), f"names of a context that is gone were deleted: {new_frees & live}"
    assert win.view._rcache == {}


def test_prune_rcache_without_a_document(win, monkeypatch):
    """A paint before the document exists must not break the frame."""
    start(monkeypatch, win)
    win.view._rcache[1234] = {"pos": np.zeros((0, 3), np.float32)}
    monkeypatch.setattr(win.view, "doc", None)
    win.view._prune_rcache()
    assert 1234 in win.view._rcache


# ------------------------------------------------------------------- fallback

def test_no_buffers_means_drawing_from_memory(win, monkeypatch):
    """An old driver or an offscreen test: the arrays of the render cache are drawn as they are."""
    fake = start(monkeypatch, win, vbo=False)
    draw(win.view)
    assert fake.uploads == [], "nothing can be uploaded without buffer objects"
    assert fake.bindings == [] or set(fake.bindings) == {0}
    tri = face_draws(fake)
    assert tri, "the fallback still has to draw the faces"
    for d in tri:
        assert d["pointer"]["v"][0] == 0, "the positions come from memory, not from a buffer"
        assert isinstance(d["pointer"]["v"][2], np.ndarray)
        assert d["arrays"].get(viewport.gl.GL_VERTEX_ARRAY)


def test_a_driver_without_buffer_names_is_not_tried_again(win, monkeypatch):
    """glGenBuffers that hands out 0 is answered without filling buffer 0, and tried once."""
    fake = start(monkeypatch, win, vbo=True, names=False)
    assert viewport._new_buffer(np.zeros(4, dtype=np.float32)) == 0
    assert "glBindBuffer" not in [c[0] for c in fake.calls]
    assert "glBufferData" not in [c[0] for c in fake.calls]
    draw(win.view)
    assert fake.uploads == []
    assert win.view._vbo is False, "the view falls back to client-side arrays"
    assert face_draws(fake), "and still draws the faces"


# ------------------------------------------------------------------- the picture

def _pixels(img):
    img = img.convertToFormat(QImage.Format.Format_RGBA8888)
    data = img.constBits()
    data.setsize(img.sizeInBytes())
    return np.frombuffer(data, np.uint8).reshape(img.height(), img.width(), 4)


def _has_opengl(win):
    win.resize(480, 360)
    win.show()
    for _ in range(4):
        QApplication.processEvents()
    view = win.view
    if not view.makeCurrent():
        return False
    try:
        from OpenGL import GL as gl
        return gl.glGetString(gl.GL_VERSION) is not None
    except Exception:
        return False
    finally:
        view.doneCurrent()


def test_the_picture_is_the_same_as_without_buffers(win):
    """The point of the whole exercise: the same picture, only faster."""
    if not _has_opengl(win):
        pytest.skip("this machine gives the offscreen window no OpenGL context")
    from OpenGL import GL as gl
    view = win.view
    shots = {}
    for buffers in (True, False):
        view.release_gl()
        view._vbo = buffers and view._probe_buffers()
        view.update()
        for _ in range(4):
            QApplication.processEvents()
        view.makeCurrent()
        gl.glFinish()
        view.doneCurrent()
        shots[buffers] = _pixels(view.grabFramebuffer())
    a, b = shots[True], shots[False]
    assert a.shape == b.shape
    assert len(np.unique(a.reshape(-1, 4), axis=0)) > 8, "the picture is empty, the test says nothing"
    different = np.any(np.abs(a.astype(int) - b.astype(int)) > 2, axis=-1)
    assert different.mean() < 0.01, f"{different.mean():.1%} of the pixels is different"


# --------------------------------------------------------- triangular faces

def _general(ents, fid):
    """What face_triangles returns without the fast way: always mf.triangulate."""
    real, brep._triangle = brep._triangle, lambda V, loop, normal: None
    try:
        return ents.face_triangles(fid)
    finally:
        brep._triangle = real


def _face(points, normal):
    e = brep.Entities()
    ids = []
    for p in points:
        v = e._new_id()
        e.V[v] = np.asarray(p, dtype=float)
        ids.append(v)
    fid = e._add_face_raw([ids], np.asarray(normal, dtype=float))
    return e, fid


@pytest.mark.parametrize("points", [
    [(0, 0, 0), (10, 0, 0), (0, 10, 0)],                  # counter clockwise seen from +z
    [(0, 0, 0), (0, 10, 0), (10, 0, 0)],                  # the same face, the other way round
    [(1, 2, 3), (4, 1, 5), (2, 6, 0)],                    # in the air, tilted
    [(0, 0, 0), (10, 0, 0), (5, 0.0001, 0)],              # a sliver
    [(0, 0, 0), (10, 0, 0), (10, 0, 0)],                  # a vertex twice
    [(0, 0, 0), (5, 0, 0), (10, 0, 0)],                   # on one line
])
def test_the_fast_triangle_is_the_general_triangle(points):
    """The shortcut for a triangular face answers exactly what mf.triangulate answers."""
    e, fid = _face(points, (0, 0, 1))
    fast_pts, fast_tris = e.face_triangles(fid)
    gen_pts, gen_tris = _general(e, fid)
    assert np.array_equal(fast_tris, gen_tris), f"{fast_tris} instead of {gen_tris}"
    assert np.array_equal(fast_pts, gen_pts), f"{fast_pts} instead of {gen_pts}"


def test_the_fast_triangle_of_random_faces():
    """Hundreds of triangles in the air: the shortcut answers what mf.triangulate answers."""
    rng = np.random.default_rng(11)
    checked = 0
    for _ in range(150):
        pts = rng.uniform(-100.0, 100.0, (3, 3))
        n = np.cross(pts[1] - pts[0], pts[2] - pts[0])
        if float(np.linalg.norm(n)) < 1e-6:
            continue
        e, fid = _face(pts, n)
        fast_pts, fast_tris = e.face_triangles(fid)
        gen_pts, gen_tris = _general(e, fid)
        assert np.array_equal(fast_tris, gen_tris)
        assert np.array_equal(fast_pts, gen_pts)
        assert fast_tris is brep.TRI_TRIS, "a normal triangle has to take the fast way"
        checked += 1
    assert checked > 100


def test_the_fast_triangle_follows_the_normal():
    """The winding of the answer is the winding around the face normal, not around +z."""
    for normal, want_fast in (((0, 0, 1), True), ((0, 0, -1), False)):
        e, fid = _face([(0, 0, 0), (10, 0, 0), (0, 10, 0)], normal)
        fast_pts, fast_tris = e.face_triangles(fid)
        gen_pts, gen_tris = _general(e, fid)
        assert np.array_equal(fast_pts, gen_pts)
        assert np.array_equal(fast_tris, gen_tris)
        took_fast = fast_tris is brep.TRI_TRIS
        assert took_fast is want_fast, f"normal {normal}: fast way {took_fast}"


def test_tri_tris_cannot_be_changed():
    """The indices of a triangular face are one shared array: reading only."""
    assert not brep.TRI_TRIS.flags.writeable
    e, fid = _face([(0, 0, 0), (10, 0, 0), (0, 10, 0)], (0, 0, 1))
    _, tris = e.face_triangles(fid)
    assert tris is brep.TRI_TRIS
    with pytest.raises(ValueError):
        tris[0, 0] = 7
    assert np.array_equal(brep.TRI_TRIS, [[0, 1, 2]])


def test_triangles_of_a_real_model_are_unchanged(win):
    """Every face of the model: the shortcut gives the same triangles as before."""
    ents = win.doc.root
    ents.pushpull(next(iter(ents.F)), 12.0)
    ents.add_line((0, 0, 0), (40, 30, 12.0))
    assert len(ents.F) >= 3, "a box and a line, so more than one face"
    for fid in list(ents.F):
        fast_pts, fast_tris = ents.face_triangles(fid)
        gen_pts, gen_tris = _general(ents, fid)
        assert np.array_equal(fast_tris, gen_tris), f"face {fid}: {fast_tris} instead of {gen_tris}"
        assert np.array_equal(fast_pts, gen_pts), f"face {fid}: other points"

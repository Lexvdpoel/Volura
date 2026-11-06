"""Tools > Simplify Model: a quality slider, a live estimate and a large-import advice.

volura/ops.simplify_group() replaces a group's geometry with a manifold-simplified copy
within a chosen deviation. These tests (Qt offscreen) check:

* the geometry after simplifying: fewer triangles, the volume within half a percent, the
  group still closed, name/color/id unchanged, every vertex on an original vertex;
* a cube (already minimal) is untouched at every step;
* undo/redo restore the exact old and new triangle counts and volumes;
* a locked group is left alone and an open group raises with its name;
* painted face colors do not survive the manifold (the group keeps its own color, and the
  dialog warns about it beforehand);
* the step table is increasing, defaults to 0.02 mm and is fully translated;
* the dialog shows the tolerance and explanation of every step and an estimate that
  matches simplify_group on a copy;
* an import above LARGE_IMPORT_TRIANGLES asks to simplify, one below does not, and the
  advice stays off once "Don't ask again" was chosen.
"""
import os
import sys
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import manifold3d as mf  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from PySide6.QtWidgets import QApplication, QLabel  # noqa: E402

from volura import app as app_module, fileio, ops  # noqa: E402
from volura.brep import Entities  # noqa: E402
from volura.locale import nl  # noqa: E402

_app = QApplication.instance() or QApplication(sys.argv[:1])

SPHERE_SEGMENTS = 256
SPHERE_TRIS = 32768            # manifold sphere(20, 256)
SPHERE_TRIS_AT_002 = 20576     # simplify_group(0.02) on that sphere (via the entities)
CUBE_TRIS = 12
CUBE_VOLUME = 8000.0


@pytest.fixture()
def win(monkeypatch, tmp_path):
    """A window that logs and autosaves into the folder of the test."""
    monkeypatch.setattr(app_module.errors, "log_dir", lambda: str(tmp_path))
    monkeypatch.setattr(app_module, "autosave_path", lambda: str(tmp_path / "autosave.volura"))
    w = app_module.MainWindow()
    w.doc.dirty = False
    yield w
    w.doc.dirty = False
    w.close()


def add_sphere(win, segments=SPHERE_SEGMENTS, name="sphere", color="#888888"):
    return ops.group_from_manifold(win.doc, mf.Manifold.sphere(20, segments), color, name)


def add_cube(win, size=(20.0, 20.0, 20.0), name="cube", color="#cccccc"):
    return ops.group_from_manifold(win.doc, mf.Manifold.cube(size), color, name)


def tris(ents):
    return ops.tri_count(ents)


def volume(ents):
    return ents.to_manifold().volume()


def vertices(ents):
    return np.round(np.array([ents.V[k] for k in sorted(ents.V)], dtype=float), 6)


def pump_until(predicate, timeout_ms=15000):
    """Spin the event loop until `predicate()` holds or the time is up; returns its value."""
    deadline = time.monotonic() + timeout_ms / 1000.0
    while not predicate() and time.monotonic() < deadline:
        _app.processEvents()
        time.sleep(0.005)
    return predicate()


# ---------------------------------------------------------------- simplify_group

def test_simplify_sphere_keeps_shape_and_group(win):
    g = add_sphere(win)
    before_n, before_v = tris(g.ents), volume(g.ents)
    name, color, gid = g.name, g.color, g.id
    b, a = ops.simplify_group(win.doc, g, 0.02)

    assert b == SPHERE_TRIS
    assert a == SPHERE_TRIS_AT_002 < before_n
    assert tris(g.ents) == a
    assert abs(volume(g.ents) - before_v) / before_v < 0.005
    # still a closed, printable part
    part = next(p for p in ops.parts(win.doc) if p.name == name)
    assert part.problem is None
    assert part.manifold is not None
    # name, color and id of the group are unchanged
    assert (g.name, g.color, g.id) == (name, color, gid)


def test_simplified_vertices_lie_on_original_vertices(win):
    g = add_sphere(win, segments=128)
    original = vertices(g.ents)
    ops.simplify_group(win.doc, g, 0.1)
    result = vertices(g.ents)
    assert len(result) < len(original)           # the coarse sphere really lost vertices
    nearest = np.full(len(result), np.inf)
    for i in range(0, len(original), 4000):
        block = original[i:i + 4000]
        nearest = np.minimum(nearest,
                             np.linalg.norm(result[:, None, :] - block[None, :, :], axis=2).min(axis=1))
    assert nearest.max() == 0.0            # manifold keeps a subset of the vertices


def test_cube_is_untouched_at_every_step(win):
    g = add_cube(win)
    assert len(g.ents.V) == 8 and tris(g.ents) == CUBE_TRIS
    for tol, _desc in ops.SIMPLIFY_STEPS:
        b, a = ops.simplify_group(win.doc, g, tol)
        assert (b, a) == (CUBE_TRIS, CUBE_TRIS)
        assert len(g.ents.V) == 8 and tris(g.ents) == CUBE_TRIS
        assert abs(volume(g.ents) - CUBE_VOLUME) < 1e-6


def test_undo_redo_restore_counts_and_volumes(win):
    g = add_sphere(win, segments=128)
    before_n, before_v = tris(g.ents), volume(g.ents)
    win.doc.checkpoint()
    ops.simplify_group(win.doc, g, 0.1)
    after_n, after_v = tris(g.ents), volume(g.ents)
    assert after_n < before_n

    assert win.doc.undo()
    assert tris(g.ents) == before_n
    assert abs(volume(g.ents) - before_v) < 1e-6
    assert win.doc.redo()
    assert tris(g.ents) == after_n
    assert abs(volume(g.ents) - after_v) < 1e-6


def test_locked_group_is_not_changed(win):
    g = add_sphere(win, segments=128)
    g.locked = True
    before = tris(g.ents)
    b, a = ops.simplify_group(win.doc, g, 0.2)
    assert (b, a) == (before, before)
    assert tris(g.ents) == before


def test_open_group_raises_with_its_name(win):
    g = add_cube(win, name="lid")
    g.ents.erase_faces([sorted(g.ents.F)[-1]])
    with pytest.raises(ValueError) as e:
        ops.simplify_group(win.doc, g, 0.02)
    assert "lid" in str(e)


# ------------------------------------------------------------------- face colors

def test_painted_face_colors_are_lost_group_keeps_its_color(win):
    g = add_cube(win, color="#3fa34d")
    for fid in list(g.ents.F):
        g.ents.set_face_color(fid, "#ff0000")
    assert ops.has_painted_faces(g) is True
    ops.simplify_group(win.doc, g, 0.02)
    assert {f.color for f in g.ents.F.values()} == {None}   # no face keeps its own color
    assert g.color == "#3fa34d"                             # the group keeps its own color


def test_dialog_warns_about_painted_faces(win):
    g = add_cube(win)
    for fid in list(g.ents.F):
        g.ents.set_face_color(fid, "#ff0000")
    dlg = app_module.SimplifyDialog(win, [g])
    warning = "Painted face colors are lost; the parts keep the group color."
    assert any(warning in lbl.text() for lbl in dlg.findChildren(QLabel))
    dlg.deleteLater()


# ------------------------------------------------------------------- step table

def test_step_table_is_increasing_with_a_default_and_translations():
    tolerances = [tol for tol, _desc in ops.SIMPLIFY_STEPS]
    assert tolerances == sorted(tolerances)
    assert len(set(tolerances)) == len(tolerances)
    assert ops.SIMPLIFY_DEFAULT_TOLERANCE == 0.02
    assert ops.SIMPLIFY_DEFAULT_TOLERANCE in tolerances
    for _tol, desc in ops.SIMPLIFY_STEPS:
        assert desc in nl.TRANSLATIONS
        assert nl.TRANSLATIONS[desc] != desc               # a real Dutch translation


# ---------------------------------------------------------------------- dialog

def test_dialog_shows_each_step_and_matches_simplify_group(win):
    g = add_sphere(win, segments=96)
    dlg = app_module.SimplifyDialog(win, [g])
    assert dlg.tolerance == ops.SIMPLIFY_DEFAULT_TOLERANCE
    for index, (tol, desc) in enumerate(ops.SIMPLIFY_STEPS):
        dlg.slider.setValue(index)
        assert dlg.tolerance == tol
        assert dlg.desc_label.text() == desc
        assert f"{tol:g}" in dlg.tol_label.text()
        assert pump_until(lambda: dlg.preview_after is not None), "no estimate arrived"

        copy = ops.group_from_manifold(win.doc, mf.Manifold.sphere(20, 96), "#888888", "copy")
        b, a = ops.simplify_group(win.doc, copy, tol)
        win.doc.remove_group(copy.id)
        assert dlg.preview_before == b
        assert dlg.preview_after == a
    dlg.deleteLater()


# ------------------------------------------------------------------ the advice

def _write_sphere_stl(tmp_path, segments):
    path = tmp_path / f"sphere{segments}.stl"
    fileio.write_stl(str(path), mf.Manifold.sphere(20, segments))
    return str(path)


def _import(win, path):
    before = len(win.doc.groups)
    job = win.import_stl(path)
    assert pump_until(lambda: len(win.doc.groups) == before + 1), "the group never appeared"
    job.wait()
    return win.doc.groups[-1]


BIG_SEGMENTS = 208             # ~21600 triangles: just above LARGE_IMPORT_TRIANGLES
SMALL_SEGMENTS = 128           # 8192 triangles: below it


def test_large_import_asks_to_simplify(win, monkeypatch, tmp_path):
    asked = []
    monkeypatch.setattr(app_module.MainWindow, "_ask_simplify_import",
                        lambda self, n: asked.append(n) or False)
    win.settings.ask_simplify_import = True

    big = _import(win, _write_sphere_stl(tmp_path, BIG_SEGMENTS))
    assert len(asked) == 1
    assert asked[0] == tris(big.ents)
    assert asked[0] > ops.LARGE_IMPORT_TRIANGLES

    asked.clear()
    _import(win, _write_sphere_stl(tmp_path, SMALL_SEGMENTS))
    assert asked == []


def test_advice_stays_off_after_dont_ask_again(win, monkeypatch, tmp_path):
    asked = []
    monkeypatch.setattr(app_module.MainWindow, "_ask_simplify_import",
                        lambda self, n: asked.append(n) or False)
    win.settings.ask_simplify_import = False               # what "Don't ask again" stores

    _import(win, _write_sphere_stl(tmp_path, BIG_SEGMENTS))
    assert asked == []


def test_dont_ask_again_box_turns_the_advice_off(win, monkeypatch):
    """Checking 'Don't ask again' in the question stores the setting off."""
    class FakeBox:
        Question = 0
        AcceptRole = 1
        RejectRole = 2

        def __init__(self, parent=None):
            self._clicked = None
            self._checkbox = None
        def setIcon(self, *a): pass
        def setWindowTitle(self, *a): pass
        def setText(self, *a): pass
        def addButton(self, button, role):
            if self._clicked is None:
                self._clicked = button
            return button
        def setCheckBox(self, box):
            self._checkbox = box
        def exec(self):
            self._checkbox.setChecked(True)
            return 0
        def clickedButton(self):
            return self._clicked

    monkeypatch.setattr(app_module, "QMessageBox", FakeBox)
    win.settings.ask_simplify_import = True
    win._ask_simplify_import(50000)
    assert win.settings.ask_simplify_import is False


# ---------------------------------------------------------------- off the GUI thread
# A model of tens of thousands of triangles takes seconds to build into faces: that work
# must not run on the GUI thread, or the window freezes while it happens.

def record_threads(monkeypatch):
    """The threads Entities.from_manifold runs on, from now on."""
    import threading
    seen = []
    real = Entities.from_manifold

    def spy(*a, **k):
        seen.append(threading.current_thread() is threading.main_thread())
        return real(*a, **k)

    monkeypatch.setattr(Entities, "from_manifold", staticmethod(spy))
    return seen


def test_import_builds_the_model_off_the_gui_thread(win, monkeypatch, tmp_path):
    monkeypatch.setattr(app_module.MainWindow, "_ask_simplify_import", lambda self, n: False)
    seen = record_threads(monkeypatch)
    g = _import(win, _write_sphere_stl(tmp_path, SMALL_SEGMENTS))
    assert seen == [False]                         # built once, in the background
    assert tris(g.ents) == 8192
    # and the render and snap cache of the new group is ready before the first frame
    c = win.view.engine.cache._c.get(id(g.ents))
    assert c is not None and c["ents"] is g.ents and c["version"] == g.ents.version
    assert len(c["T"]) == 8192


def test_simplify_runs_off_the_gui_thread_and_undoes_in_one_step(win, monkeypatch):
    a = add_sphere(win, segments=128, name="a")
    b = add_sphere(win, segments=96, name="b")
    old = (tris(a.ents), tris(b.ents))
    seen = record_threads(monkeypatch)
    job = win._run_simplify([a, b], 0.1)
    job.wait()
    assert pump_until(lambda: tris(a.ents) < old[0]), "the result never arrived"
    assert seen == [False, False]
    assert tris(b.ents) < old[1]
    assert "fewer triangles" in win.hint.text()
    assert win.doc.undo() is True                  # one checkpoint for both groups
    assert (tris(a.ents), tris(b.ents)) == old


def test_cancelled_simplify_changes_nothing(win):
    g = add_sphere(win, segments=128)
    old, ents = tris(g.ents), g.ents
    undo = len(win.doc.undo_stack)
    job = win._run_simplify([g], 0.1)
    job.cancel()
    job.wait()
    for _ in range(20):
        _app.processEvents()
    assert g.ents is ents and tris(g.ents) == old
    assert len(win.doc.undo_stack) == undo


def test_dialog_makes_the_manifold_of_a_group_only_once(win, monkeypatch):
    g = add_sphere(win, segments=128)
    calls = []
    real = ops.closed_manifold
    monkeypatch.setattr(ops, "closed_manifold", lambda grp: calls.append(grp.id) or real(grp))
    dlg = app_module.SimplifyDialog(win, [g])
    for index in range(len(ops.SIMPLIFY_STEPS)):
        dlg.slider.setValue(index)
        assert pump_until(lambda: dlg.preview_after is not None), "no estimate arrived"
    assert calls == [g.id]
    assert dlg.preview_before == 8192


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))

"""Mouse Ears: thin round discs on the bed under the sharp bottom corners of a part.

A sharp corner of a flat bottom lifts off the bed while printing (warping), most of all with
ABS and ASA and on large flat parts. `ops.mouse_ears()` puts a thin disc under every sharp
corner of the convex hull of the vertices at the lowest z of the given groups; the discs are
cut off after printing. A corner is sharp when its interior angle is at most 100 degrees, so
a box (four corners of 90°) gets an ear on each corner and a cylinder (nearly flat hull
corners) gets none. A disc of Ø 10 mm under the corner of a box stands for 3/4 outside the
box (the quarter under the corner already lies inside it), so a box of 20 × 20 × 10 mm with
ears of Ø 10 × 0.4 mm grows by 4 * 0.75 * pi * 5² * 0.4 mm³, apart from the polygon of the
segments differing a little from pi. These tests (Qt offscreen) check:

* the volume and the bounding box of a box with ears, and that its height is untouched;
* a cylinder, a part that stands on one vertex and a locked part, which are skipped;
* an L-shaped part, whose convex hull has three sharp corners and gets exactly three ears;
* the menu action: right after the Elephant Foot Chamfer, the values in the dialog, one undo
  for the whole operation and the hint in the status bar.

Run with:  python -m pytest tests/test_mouse_ears.py
"""
import math
import os
import sys
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import manifold3d as mf  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from PySide6.QtWidgets import QApplication, QDialog, QDoubleSpinBox  # noqa: E402

from volura import app as app_module, ops  # noqa: E402
from volura.geom import rotation_matrix  # noqa: E402
from volura.model import Document  # noqa: E402

_app = QApplication.instance() or QApplication(sys.argv[:1])

BOX = (20.0, 20.0, 10.0)
BOX_VOLUME = BOX[0] * BOX[1] * BOX[2]
DIAMETER = 10.0
HEIGHT = 0.4
R = DIAMETER / 2.0


class FakeBox:
    """Stands in for QMessageBox: remembers what the window asked."""

    def __init__(self):
        self.asked = []

    def warning(self, *args, **kwargs):
        self.asked.append(args[1:])


@pytest.fixture()
def win(monkeypatch, tmp_path):
    """A window that logs into the folder of the test and never shows a message box."""
    monkeypatch.setattr(app_module.errors, "log_dir", lambda: str(tmp_path))
    monkeypatch.setattr(app_module, "autosave_path", lambda: str(tmp_path / "autosave.volura"))
    monkeypatch.setattr(app_module, "QMessageBox", FakeBox())
    w = app_module.MainWindow()
    w.doc.dirty = False
    yield w
    w.doc.dirty = False
    w.close()


def clean(text):
    return text.replace("&", "").replace("…", "").replace("...", "").strip()


def add_box(doc, name="box", color="#cccccc"):
    """A closed box of 20 × 20 × 10 mm standing on the bed, as its own group."""
    return ops.group_from_manifold(doc, mf.Manifold.cube(BOX), color, name)


def add_cylinder(doc, name="cylinder"):
    """A cylinder of r = 10 mm and h = 20 mm on the bed, out of 32 segments."""
    return ops.group_from_manifold(doc, mf.Manifold.cylinder(20.0, 10.0, 10.0, 32), "#cccccc", name)


def tilt_about_a_vertex(doc, name="tilted"):
    """A box turned 45° about X and then 45° about Y: it stands on one vertex."""
    g = add_box(doc, name=name)
    for axis in (np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])):
        g.ents.transform(rotation_matrix(axis, math.radians(45)), np.zeros(3))
    return g


def add_l_part(doc, name="L"):
    """An L-shaped part: [0, 20] × [0, 10] unioned with [0, 10] × [0, 20], 10 mm high."""
    man = ops.union_all([mf.Manifold.cube((20.0, 10.0, 10.0)), mf.Manifold.cube((10.0, 20.0, 10.0))])
    return ops.group_from_manifold(doc, man, "#cccccc", name)


def ear_volume(n):
    """The volume `n` discs add under the corners of a box: 3/4 of each disc outside it."""
    return n * 0.75 * math.pi * R ** 2 * HEIGHT


def pump_until(predicate, timeout_ms=15000):
    """Spin the event loop until `predicate()` holds or the time is up; returns its value."""
    deadline = time.monotonic() + timeout_ms / 1000.0
    while not predicate() and time.monotonic() < deadline:
        _app.processEvents()
        time.sleep(0.005)
    return predicate()


# ---------------------------------------------------------------- the operation

def test_a_box_gets_an_ear_under_every_corner():
    doc = Document()
    g = add_box(doc)
    assert ops.mouse_ears(doc, [("g", g.id, g.id)], DIAMETER, HEIGHT) == (1, 0)
    after = g.ents.to_manifold().volume()
    assert after - BOX_VOLUME == pytest.approx(ear_volume(4), rel=0.01)


def test_the_ears_widen_the_footprint_but_not_the_height():
    doc = Document()
    g = add_box(doc)
    ops.mouse_ears(doc, [("g", g.id, g.id)], DIAMETER, HEIGHT)
    lo, hi = g.ents.bbox()
    assert hi[0] - lo[0] == pytest.approx(30.0, abs=0.1), "the discs did not reach 5 mm out"
    assert hi[1] - lo[1] == pytest.approx(30.0, abs=0.1), "the discs did not reach 5 mm out"
    assert lo[2] == pytest.approx(0.0) and hi[2] == pytest.approx(10.0), "the height changed"


def test_a_cylinder_gets_no_ears_and_is_skipped():
    doc = Document()
    g = add_cylinder(doc)
    before = g.ents.to_manifold().volume()
    assert ops.mouse_ears(doc, [("g", g.id, g.id)], DIAMETER, HEIGHT) == (0, 1)
    assert g.ents.to_manifold().volume() == pytest.approx(before)


def test_a_part_that_stands_on_one_vertex_is_skipped():
    doc = Document()
    g = tilt_about_a_vertex(doc)
    before = g.ents.to_manifold().volume()
    assert ops.mouse_ears(doc, [("g", g.id, g.id)], DIAMETER, HEIGHT) == (0, 1)
    assert g.ents.to_manifold().volume() == pytest.approx(before)


def test_a_locked_group_is_skipped():
    doc = Document()
    g = add_box(doc)
    g.locked = True
    assert ops.mouse_ears(doc, [("g", g.id, g.id)], DIAMETER, HEIGHT) == (0, 1)
    assert g.ents.to_manifold().volume() == pytest.approx(BOX_VOLUME)


def test_an_l_shaped_part_gets_ears_on_its_three_sharp_corners():
    doc = Document()
    g = add_l_part(doc)
    before = g.ents.to_manifold().volume()
    assert ops.mouse_ears(doc, [("g", g.id, g.id)], DIAMETER, HEIGHT) == (1, 0)
    after = g.ents.to_manifold().volume()
    assert after - before == pytest.approx(ear_volume(3), rel=0.01), "not exactly three discs"


def test_a_group_keeps_its_name_and_color():
    doc = Document()
    g = add_box(doc, name="my part", color="#d23b3b")
    ops.mouse_ears(doc, [("g", g.id, g.id)], DIAMETER, HEIGHT)
    assert g.name == "my part" and g.color == "#d23b3b"


def test_only_the_groups_of_the_items_are_counted_once():
    doc = Document()
    g = add_box(doc)
    items = [("g", g.id, g.id), ("g", g.id, g.id), ("f", 0, 1)]
    assert ops.mouse_ears(doc, items, DIAMETER, HEIGHT) == (1, 0)


# ---------------------------------------------------------------- the window

def test_the_action_is_in_the_print_tools_menu_after_the_chamfer(win):
    labels = [clean(a.text()) for a in win.print_tools_menu.actions() if not a.isSeparator()]
    assert labels[-2:] == ["Elephant Foot Chamfer", "Mouse Ears"]


def test_the_action_adds_ears_to_the_selected_group(win, monkeypatch):
    monkeypatch.setattr(app_module.QDialog, "exec", lambda dlg: QDialog.Accepted)
    g = add_box(win.doc)
    win.doc.set_selection([("g", g.id, g.id)])
    win.a_mouse_ears.trigger()
    assert pump_until(lambda: g.ents.to_manifold().volume() > BOX_VOLUME + 1), "no ears added"
    assert pump_until(lambda: "Added mouse ears to 1 parts." in win.hint.text()), win.hint.text()


def test_the_action_takes_every_visible_group_when_nothing_is_selected(win, monkeypatch):
    monkeypatch.setattr(app_module.QDialog, "exec", lambda dlg: QDialog.Accepted)
    a, b = add_box(win.doc, name="a"), add_box(win.doc, name="b")
    hidden = add_box(win.doc, name="hidden")
    hidden.visible = False
    before = hidden.ents.to_manifold().volume()
    win.a_mouse_ears.trigger()
    ready = lambda: a.ents.to_manifold().volume() > BOX_VOLUME + 1 and \
        b.ents.to_manifold().volume() > BOX_VOLUME + 1
    assert pump_until(ready), "not every group got ears"
    assert hidden.ents.to_manifold().volume() == pytest.approx(before)


def test_a_part_without_sharp_corners_leaves_no_undo_step(win, monkeypatch):
    monkeypatch.setattr(app_module.QDialog, "exec", lambda dlg: QDialog.Accepted)
    tilted = tilt_about_a_vertex(win.doc)
    steps = len(win.doc.undo_stack)
    win.doc.set_selection([("g", tilted.id, tilted.id)])
    win.a_mouse_ears.trigger()
    assert pump_until(lambda: "None of these parts" in win.hint.text()), win.hint.text()
    assert len(win.doc.undo_stack) == steps, "nothing changed: there is nothing to undo"


def test_the_action_is_one_undo_step(win, monkeypatch):
    monkeypatch.setattr(app_module.QDialog, "exec", lambda dlg: QDialog.Accepted)
    g = add_box(win.doc)
    win.doc.set_selection([("g", g.id, g.id)])
    win.a_mouse_ears.trigger()
    assert pump_until(lambda: g.ents.to_manifold().volume() > BOX_VOLUME + 1)
    win.doc.undo()
    assert g.ents.to_manifold().volume() == pytest.approx(BOX_VOLUME), "one undo did not restore"


def test_the_dialog_starts_at_the_last_values_and_remembers_the_new_ones(win, monkeypatch):
    seen = {}

    def accept(dlg):
        spins = dlg.findChildren(QDoubleSpinBox)
        seen["start"] = (spins[0].value(), spins[1].value())
        spins[0].setValue(12.0)
        spins[1].setValue(0.6)
        return QDialog.Accepted

    monkeypatch.setattr(app_module.QDialog, "exec", accept)
    win.settings.mouse_ear_diameter = 8.0
    win.settings.mouse_ear_height = 0.3
    g = add_box(win.doc)
    win.doc.set_selection([("g", g.id, g.id)])
    win.mouse_ears_dialog()
    assert seen["start"] == (pytest.approx(8.0), pytest.approx(0.3))
    assert pump_until(lambda: g.ents.to_manifold().volume() > BOX_VOLUME + 1)
    assert win.settings.mouse_ear_diameter == pytest.approx(12.0)
    assert win.settings.mouse_ear_height == pytest.approx(0.6)

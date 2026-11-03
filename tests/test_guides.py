"""Hulplijnen en hulppunten zoals in SketchUp 2017 (Meetlint en Gradenboog).

Runs without a display (Qt offscreen platform), like tests/test_ui_spec.py.
"""
import json
import math
import os
import sys
import zipfile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

_app = QApplication.instance() or QApplication(sys.argv[:1])

from volura import fileio, ops  # noqa: E402
from volura.geom import Plane  # noqa: E402
from volura.model import ROOT, Document  # noqa: E402

GROUND = Plane((0, 0, 0), (0, 0, 1))


@pytest.fixture()
def win():
    from volura.app import MainWindow
    w = MainWindow()
    w.settings.show_guides = True
    w.view.cam.width, w.view.cam.height = 800, 600
    w.view.cam.set_view("iso")
    yield w
    w.doc.dirty = False
    w.close()


class Ev:
    """A stand-in for a Qt mouse/keyboard event."""

    def __init__(self, key=None, mods=Qt.KeyboardModifiers(0), buttons=Qt.MouseButton(0)):
        self._key, self._mods, self._buttons = key, mods, buttons

    def key(self):
        return self._key

    def isAutoRepeat(self):
        return False

    def modifiers(self):
        return self._mods

    def buttons(self):
        return self._buttons


def draw_rect(win, size=(40.0, 30.0)):
    """A loose rectangle in the ground plane; returns its face id."""
    loop = np.array([[0, 0], [size[0], 0], [size[0], size[1]], [0, size[1]]], dtype=float)
    created = win.doc.root.add_shape(GROUND, [loop])
    assert created, "rechthoek gaf geen vlak"
    win.view.zoom_extents()
    return created[0]


def box_group(win):
    """A closed 20x20x10 box as a group."""
    win.doc.root.add_shape(GROUND, [np.array([[0, 0], [20, 0], [20, 20], [0, 20]], float)])
    f = next(iter(win.doc.root.F))
    win.doc.root.pushpull(f, -10)
    return ops.make_group(win.doc, [("f", ROOT, f) for f in win.doc.root.F])


def px(win, p):
    """The widget pixel of a world point."""
    s, ok = win.view.cam.project(np.asarray(p, dtype=float)[None])
    assert ok[0], "punt valt buiten beeld"
    return float(s[0][0]), float(s[0][1])


def move(tool, win, p):
    x, y = px(win, p)
    tool.mouse_move(Ev(), x, y)


def click(tool, win, p):
    x, y = px(win, p)
    tool.mouse_move(Ev(), x, y)
    tool.mouse_press(None, x, y)


def click_release(tool, win, p):
    x, y = px(win, p)
    return tool.mouse_release(Ev(), x, y)


def close(a, b, tol=1e-6):
    return np.allclose(np.asarray(a, float), np.asarray(b, float), atol=tol)


# ------------------------------------------------------------------ document API

def test_add_and_remove_guides():
    doc = Document()
    line = doc.add_guide([10, 20, 30], [2, 0, 0])
    point = doc.add_guide([1, 2, 3])
    assert line.is_line and not point.is_line
    assert close(line.direction, [1, 0, 0]), "de richting wordt genormaliseerd"
    assert doc.guide(line.id) is line and doc.guide(point.id) is point
    a, b = line.endpoints()
    assert close(a - b, -2 * 100000 * np.array([1.0, 0, 0])), "een hulplijn is oneindig lang"
    assert doc.guides == [line, point]
    doc.remove_guides([line.id])
    assert doc.guides == [point] and doc.guide(line.id) is None
    doc.clear_guides()
    assert doc.guides == []


def test_guides_live_in_the_undo_state():
    doc = Document()
    doc.checkpoint()
    g = doc.add_guide([0, 0, 0], [0, 0, 1])
    doc.notify()
    assert len(doc.guides) == 1
    doc.undo()
    assert doc.guides == [], "hulplijnen gaan mee in ongedaan maken"
    doc.redo()
    assert len(doc.guides) == 1 and close(doc.guides[0].direction, [0, 0, 1])
    assert doc.guides[0].id == g.id


def test_erase_guides_menu_is_one_undo_step(win):
    assert win.a_erase_guides.isEnabled()
    doc = win.doc
    doc.add_guide([0, 0, 0], [1, 0, 0])
    doc.add_guide([10, 10, 0])
    doc.notify()
    win.a_erase_guides.trigger()
    assert doc.guides == []
    doc.undo()
    assert len(doc.guides) == 2, "Hulplijnen wissen is één stap in ongedaan maken"


def test_save_and_load_keeps_guides(tmp_path, win):
    draw_rect(win)
    line = win.doc.add_guide([20, 15, 0], [1, 1, 0])
    point = win.doc.add_guide([5, 5, 5])
    path = str(tmp_path / "model.volura")
    fileio.save_document(win.doc, path)

    other = Document()
    fileio.load_document(other, path)
    assert len(other.guides) == 2
    l2, p2 = other.guides[0], other.guides[1]
    assert close(l2.point, line.point) and close(l2.direction, line.direction)
    assert close(p2.point, point.point) and not p2.is_line
    assert other.next_guide_id == win.doc.next_guide_id


def test_old_files_without_guides_still_load(tmp_path, win):
    draw_rect(win)
    win.doc.add_guide([0, 0, 0], [1, 0, 0])
    path = str(tmp_path / "oud.volura")
    fileio.save_document(win.doc, path)
    with zipfile.ZipFile(path) as z:
        data = json.loads(z.read("document.json"))
    data.pop("guides")
    data.pop("next_guide_id", None)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("document.json", json.dumps(data))

    other = Document()
    fileio.load_document(other, path)
    assert other.guides == []
    assert len(other.root.F) == len(win.doc.root.F)


# ------------------------------------------------------------------ inference

def test_infer_gives_a_point_on_a_guide_line(win):
    """A point near a hulplijn snaps to a point ON that line."""
    win.view.zoom_extents()
    g = win.doc.add_guide([20, 15, 0], [1, 0, 0])
    x, y = px(win, [26.0, 15.0, 0.0])
    inf = win.view.engine.infer(x, y)
    assert inf.kind == "guide" and inf.label == "Op hulplijn"
    assert abs(inf.point[1] - 15.0) < 1e-9 and abs(inf.point[2]) < 1e-9, "punt ligt op de hulplijn"
    assert inf.guide == g.id

    x, y = px(win, [26.0, 15.35, 0.0])       # a few pixels beside the line
    inf = win.view.engine.infer(x, y)
    assert inf.kind == "guide"
    assert abs(inf.point[1] - 15.0) < 1e-9, "ook dicht bij de lijn snapt de cursor erop"


def test_infer_uses_guide_points(win):
    win.view.zoom_extents()
    win.doc.add_guide([10, 25, 0])
    x, y = px(win, [10.0, 25.0, 0.0])
    inf = win.view.engine.infer(x, y)
    assert inf.kind == "guidepoint" and inf.label == "Hulppunt"
    assert close(inf.point, [10, 25, 0])


def test_infer_uses_the_crossing_of_a_guide_with_an_edge(win):
    draw_rect(win)
    win.doc.add_guide([20, 0, 5], [0, 0, 1])      # a vertical guide through the front edge
    x, y = px(win, [20.0, 0.0, 0.0])
    inf = win.view.engine.infer(x, y)
    assert inf.kind in ("guide", "endpoint", "midpoint")
    assert close(inf.point[:2], [20, 0], tol=1e-6)


def test_hidden_guides_are_not_used(win):
    draw_rect(win)
    win.doc.add_guide([20, 15, 0], [1, 0, 0])
    x, y = px(win, [26.0, 15.0, 0.0])
    assert win.view.engine.infer(x, y).kind == "guide"
    win.settings.show_guides = False
    assert win.view.engine.infer(x, y).kind != "guide"
    assert win.view.engine.pick_guide(x, y) is None
    win.settings.show_guides = True


def test_guides_are_never_printed_or_exported(win):
    box_group(win)
    before = ops.parts(win.doc)
    assert len(before) == 1
    win.doc.add_guide([0, 0, 0], [1, 0, 0])
    win.doc.add_guide([5, 5, 5])
    assert len(ops.parts(win.doc)) == len(before), "hulplijnen zijn geen printonderdelen"
    assert not ops.check_parts(ops.parts(win.doc), win.settings.bed_size)[0][1].startswith("Hulplijn")
    assert ops.selection_bbox(win.doc, [("guide", ROOT, win.doc.guides[0].id)]) is None
    assert win.doc.model_bbox() is not None


# ------------------------------------------------------------------ the tape measure

def test_tape_makes_a_guide_parallel_to_an_edge(win):
    draw_rect(win)
    win.set_tool("tape")
    tape = win.view.tools["tape"]
    assert tape.make_guides, "hulplijnen maken staat standaard aan"
    click(tape, win, [10.0, 0.0, 0.0])           # on the front edge, away from its midpoint
    assert tape.edge is not None, "klik op een rand start de hulplijn"
    click(tape, win, [20.0, 10.0, 0.0])
    assert len(win.doc.guides) == 1
    g = win.doc.guides[0]
    assert g.is_line and close(g.direction, [1, 0, 0]), "evenwijdig aan de rand"
    assert close(g.point, [20, 10, 0], tol=1e-6), "op de afstand van de muis"


def test_tape_places_a_guide_point_between_two_points(win):
    draw_rect(win)
    win.set_tool("tape")
    tape = win.view.tools["tape"]
    click(tape, win, [40.0, 0.0, 0.0])           # an endpoint
    assert tape.a is not None and tape.edge is None, "klik op een punt meet"
    click(tape, win, [40.0, 30.0, 0.0])
    assert tape.b is not None
    assert len(win.doc.guides) == 1
    g = win.doc.guides[0]
    assert not g.is_line and close(g.point, [40, 30, 0], tol=1e-6)
    assert abs(np.linalg.norm(tape.b - tape.a) - 30.0) < 1e-6


def test_tape_types_a_distance(win):
    draw_rect(win)
    win.set_tool("tape")
    tape = win.view.tools["tape"]
    click(tape, win, [10.0, 0.0, 0.0])
    move(tape, win, [20.0, 10.0, 0.0])
    tape.vcb_enter("25")
    assert len(win.doc.guides) == 1
    assert close(win.doc.guides[0].point, [20, 25, 0], tol=1e-6)

    tape.reset()
    click(tape, win, [40.0, 0.0, 0.0])
    move(tape, win, [40.0, 12.0, 0.0])
    tape.vcb_enter("12")
    assert len(win.doc.guides) == 2
    assert close(win.doc.guides[1].point, [40, 12, 0], tol=1e-6)


def test_tape_on_a_face_makes_nothing(win):
    """Dubbelklikken op een vlak maakt geen hulplijn."""
    draw_rect(win)
    win.set_tool("tape")
    tape = win.view.tools["tape"]
    click(tape, win, [20.0, 15.0, 0.0])
    click(tape, win, [20.0, 15.0, 0.0])
    assert win.doc.guides == []
    assert tape.a is None and tape.edge is None


def test_tape_ctrl_toggles_making_guides(win):
    draw_rect(win)
    win.set_tool("tape")
    tape = win.view.tools["tape"]
    assert tape.key_press(Ev(Qt.Key_Control))
    assert not tape.make_guides
    click(tape, win, [10.0, 0.0, 0.0])
    assert tape.edge is None, "in de stand alleen meten ontstaat er geen hulplijn"
    click(tape, win, [40.0, 0.0, 0.0])
    click(tape, win, [40.0, 30.0, 0.0])
    assert win.doc.guides == [], "alleen meten plaatst ook geen hulppunt"
    assert tape.b is not None, "meten doet het nog wel"
    assert tape.key_press(Ev(Qt.Key_Control))
    assert tape.make_guides


def test_tape_snaps_lengths(win):
    draw_rect(win)
    win.set_tool("tape")
    tape = win.view.tools["tape"]
    click(tape, win, [10.0, 0.0, 0.0])
    move(tape, win, [20.0, 10.37, 0.0])
    assert tape.shown_distance() == 10.0, "de lengte-snapping blijft gelden"


# ------------------------------------------------------------------ the protractor

def test_protractor_makes_a_guide_at_the_angle(win):
    draw_rect(win)
    win.set_tool("protractor")
    pro = win.view.tools["protractor"]
    assert pro.key_press(Ev(Qt.Key_Up)), "pijltjes kiezen de as"
    click(pro, win, [0.0, 0.0, 0.0])
    assert pro.center is not None and close(pro.plane.n, [0, 0, 1])
    click(pro, win, [30.0, 0.0, 0.0])
    assert close(pro.ref, [1, 0, 0])
    pro.vcb_enter("30")
    assert len(win.doc.guides) == 1
    g = win.doc.guides[0]
    assert close(g.point, [0, 0, 0])
    assert close(g.direction, [math.cos(math.radians(30)), math.sin(math.radians(30)), 0])


def test_protractor_snaps_to_15_degrees(win):
    draw_rect(win)
    win.set_tool("protractor")
    pro = win.view.tools["protractor"]
    pro.key_press(Ev(Qt.Key_Up))
    click(pro, win, [0.0, 0.0, 0.0])
    click(pro, win, [30.0, 0.0, 0.0])
    a = math.radians(28.0)
    move(pro, win, [30 * math.cos(a), 30 * math.sin(a), 0.0])
    assert pro.angle == 30.0, "hoeken snappen op 15°"
    assert pro.vcb_value.startswith("30.0")
    click(pro, win, [30 * math.cos(a), 30 * math.sin(a), 0.0])
    assert len(win.doc.guides) == 1
    g = win.doc.guides[0]
    assert close(g.direction, [math.cos(math.radians(30)), math.sin(math.radians(30)), 0])


def test_protractor_needs_a_centre_first(win):
    win.set_tool("protractor")
    with pytest.raises(ValueError):
        win.view.tools["protractor"].vcb_enter("45")


# ------------------------------------------------------------------ selecting and erasing

def test_select_a_guide_and_delete_it(win):
    draw_rect(win)
    g = win.doc.add_guide([20, 15, 0], [1, 0, 0])
    x, y = px(win, [20.0, 15.0, 0.0])
    assert win.view.engine.pick_entity(x, y)[0] == "guide"
    win.set_tool("select")
    sel = win.view.tools["select"]
    sel.mouse_press(None, x, y)
    click_release(sel, win, [20.0, 15.0, 0.0])
    assert list(win.doc.selection) == [("guide", ROOT, g.id)]
    win.delete()
    assert win.doc.guides == []


def test_eraser_erases_a_guide(win):
    draw_rect(win)
    g = win.doc.add_guide([20, 15, 0], [1, 0, 0])
    win.set_tool("eraser")
    er = win.view.tools["eraser"]
    click(er, win, [20.0, 15.0, 0.0])
    assert er.guides == [g.id]
    click_release(er, win, [20.0, 15.0, 0.0])
    assert win.doc.guides == []
    win.doc.undo()
    assert len(win.doc.guides) == 1, "wissen met de Gum is één undo-stap"


# ------------------------------------------------------------------ Beeld > Hulplijnen

def test_beeld_hulplijnen_shows_and_hides(win):
    assert win.a_guides.isCheckable() and win.a_guides.isEnabled()
    assert win.settings.show_guides is True
    win.a_guides.trigger()
    assert win.settings.show_guides is False
    win.a_guides.trigger()
    assert win.settings.show_guides is True
    win.settings.save()
    assert type(win.settings)().show_guides is True

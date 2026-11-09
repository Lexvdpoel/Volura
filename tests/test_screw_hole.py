"""Screw Hole: a circle on a part becomes a printable hole for a metric screw.

The sizes are in `threads.SCREW_HOLES`: (clearance hole Ø, socket head Ø, head height,
countersunk head Ø) per size, from ISO 273 (medium clearance holes), ISO 4762 (socket
head) and ISO 10642 (countersunk head, 90°). Every diameter grows with the clearance,
because a printed hole comes out smaller than drawn. So an M3 through hole with the
default clearance of 0.2 mm has Ø 3.4 + 0.2 = 3.6 mm, and a block of 20 × 20 × 10 mm
loses pi * 1.8² * 10 = 101.8 mm³. A counterbore adds a cylinder of the head Ø + clearance
to the depth of the head + 0.2 mm; a countersink adds a 90° cone that ends in the hole; a
nut pocket adds a hexagonal prism of the nut (`threads.HEX_NUTS`: width across flats and
height, ISO 4032), so a hex nut lies in the pocket while the screw goes through the hole.
A teardrop puts a 45° roof on a hole that runs sideways, so it prints without support.

These tests (Qt offscreen) check the volumes of the four kinds on a block with a circle
drawn on top, that the hole goes through, the openings in the top and bottom face, the
roof of a sideways hole, the error of a face that is not a circle, and one undo for the
whole operation.

Run with:  python -m pytest tests/test_screw_hole.py
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
from PySide6.QtWidgets import QApplication, QCheckBox, QComboBox, QDialog, QDoubleSpinBox  # noqa: E402

from volura import app as app_module, ops, threads  # noqa: E402
from volura.brep import Entities  # noqa: E402
from volura.geom import Plane, circle_points  # noqa: E402
from volura.model import Document  # noqa: E402

_app = QApplication.instance() or QApplication(sys.argv[:1])

BLOCK = (20.0, 20.0, 10.0)
BLOCK_VOLUME = BLOCK[0] * BLOCK[1] * BLOCK[2]


class FakeBox:
    """Stands in for QMessageBox: remembers what the window asked."""

    def __init__(self):
        self.asked = []

    def warning(self, *args, **kwargs):
        self.asked.append(args[1:])

    def information(self, *args, **kwargs):
        self.asked.append(args[1:])


@pytest.fixture()
def win(monkeypatch, tmp_path):
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


def block(doc, name="block", color="#cccccc"):
    """A closed block of 20 × 20 × 10 mm standing on the bed, as its own group."""
    return ops.group_from_manifold(doc, mf.Manifold.cube(BLOCK), color, name)


def draw_circle(g, radius=3.0, center=None, normal=None):
    """A circle drawn on a face, as the Circle tool makes it; by default in the middle of
    the top face, but `center` and `normal` put it on any face.

    Returns the id of the round face; the circle's own size never matters for the hole,
    only its centre and the direction of the face."""
    if center is None:
        center = (BLOCK[0] / 2, BLOCK[1] / 2, BLOCK[2])
    if normal is None:
        normal = (0, 0, 1)
    g.ents.add_shape(Plane(center, normal), [circle_points(radius, 48)])
    for fid in list(g.ents.F):
        if threads.face_thread_context(g.ents, fid, "outer") is not None:
            return fid
    raise AssertionError("the circle did not become a circular face")


def cut(g, fid, size, kind, clearance=0.2, teardrop=False):
    """The block with the hole, as `screw_hole` puts it in from the drawn circle."""
    center, axis, _radius, length = threads.face_thread_context(g.ents, fid, "outer")
    return threads.screw_hole(g.ents.to_manifold(), center, axis, length, size, kind,
                              clearance=clearance, teardrop=teardrop)


def hole_top(ents, x):
    """The highest point of the hole in the side face at x = `x`, or None."""
    for fid in ents.F:
        if abs(float(ents.F[fid].normal[0])) < 0.999:
            continue
        pts = np.vstack(ents.face_points(fid))
        if np.any(np.abs(pts[:, 0] - x) > 1e-6):
            continue
        for loop in ents.face_points(fid):
            P = np.asarray(loop, dtype=float)
            if len(P) < 12:
                continue
            return float(P[:, 2].max())
    return None


def opening(ents, z):
    """The radius of the round hole in the horizontal face at height `z`, or None."""
    for fid in ents.F:
        if abs(float(ents.F[fid].normal[2])) < 0.999:
            continue
        pts = np.vstack(ents.face_points(fid))
        if np.any(np.abs(pts[:, 2] - z) > 1e-6):
            continue
        for loop in ents.face_points(fid):
            c, r = threads.circle_loop(np.asarray(loop, dtype=float))
            if c is not None:
                return r
    return None


def hexagon_opening(ents, z):
    """The width across flats of the hexagonal hole in the horizontal face at height `z`.

    The loop can carry extra points where the drawn circle crossed a side, so the centre is
    the middle of the loop and the width twice the closest distance from it to a side."""
    for fid in ents.F:
        if abs(float(ents.F[fid].normal[2])) < 0.999:
            continue
        pts = np.vstack(ents.face_points(fid))
        if np.any(np.abs(pts[:, 2] - z) > 1e-6):
            continue
        for loop in ents.face_points(fid):
            P = np.asarray(loop, dtype=float)
            if len(P) < 6 or threads.circle_loop(P)[0] is not None:
                continue
            c = (P.max(axis=0) + P.min(axis=0)) / 2.0
            d = np.roll(P, -1, axis=0) - P
            n = np.stack([-d[:, 1], d[:, 0], np.zeros(len(P))], axis=1)
            dist = np.abs(np.sum((P - c) * n, axis=1)) / np.linalg.norm(n, axis=1)
            return 2.0 * float(np.min(dist))
    return None


def pump_until(predicate, timeout_ms=15000):
    """Spin the event loop until `predicate()` holds or the time is up; returns its value."""
    deadline = time.monotonic() + timeout_ms / 1000.0
    while not predicate() and time.monotonic() < deadline:
        _app.processEvents()
        time.sleep(0.005)
    return predicate()


# ------------------------------------------------------------------ the geometry

def test_a_through_hole_takes_the_clearance_hole_out_of_the_block():
    doc = Document()
    g = block(doc)
    fid = draw_circle(g)
    before = g.ents.to_manifold().volume()
    assert before == pytest.approx(BLOCK_VOLUME, rel=1e-6)
    res = cut(g, fid, "M3", "through")
    assert res.status() == mf.Error.NoError and not res.is_empty()
    lost = before - res.volume()
    assert lost == pytest.approx(math.pi * 1.8 ** 2 * 10.0, rel=0.01)


def test_the_hole_goes_all_the_way_through():
    doc = Document()
    g = block(doc)
    fid = draw_circle(g)
    g.ents = Entities.from_manifold(cut(g, fid, "M3", "through"))
    assert opening(g.ents, BLOCK[2]) == pytest.approx(1.8, abs=0.02)
    assert opening(g.ents, 0.0) == pytest.approx(1.8, abs=0.02), "the bottom has no hole"


def test_a_counterbore_leaves_room_for_a_socket_head():
    doc = Document()
    g = block(doc)
    fid = draw_circle(g)
    through = BLOCK_VOLUME - cut(g, fid, "M3", "through").volume()
    res = cut(g, fid, "M3", "counterbore")
    lost = BLOCK_VOLUME - res.volume()
    # the hole plus the ring around it up to the head height + 0.2 mm
    extra = math.pi * (2.85 ** 2 - 1.8 ** 2) * 3.2
    assert lost == pytest.approx(math.pi * 1.8 ** 2 * 10.0 + extra, rel=0.01)
    assert lost > through
    g.ents = Entities.from_manifold(res)
    assert opening(g.ents, BLOCK[2]) == pytest.approx(2.85, abs=0.02), "the head does not fit"


def test_a_countersink_takes_less_than_a_counterbore_and_more_than_a_through_hole():
    doc = Document()
    g = block(doc)
    fid = draw_circle(g)
    through = BLOCK_VOLUME - cut(g, fid, "M3", "through").volume()
    counterbore = BLOCK_VOLUME - cut(g, fid, "M3", "counterbore").volume()
    res = cut(g, fid, "M3", "countersink")
    lost = BLOCK_VOLUME - res.volume()
    assert through < lost < counterbore, (through, lost, counterbore)
    # a 90° cone from Ø 6.9 at the face down to the hole of Ø 3.6, next to the hole itself
    r_head, r_hole = 3.45, 1.8
    depth = r_head - r_hole
    cone = math.pi * depth / 3.0 * (r_head ** 2 + r_head * r_hole + r_hole ** 2)
    assert lost == pytest.approx(math.pi * r_hole ** 2 * 10.0 + cone
                                 - math.pi * r_hole ** 2 * depth, rel=0.01)
    g.ents = Entities.from_manifold(res)
    assert opening(g.ents, BLOCK[2]) == pytest.approx(3.45, abs=0.02)
    assert opening(g.ents, 0.0) == pytest.approx(1.8, abs=0.02)


def test_a_nut_pocket_holds_a_hex_nut_and_the_screw_goes_through():
    doc = Document()
    g = block(doc)
    fid = draw_circle(g)
    through = BLOCK_VOLUME - cut(g, fid, "M3", "through").volume()
    counterbore = BLOCK_VOLUME - cut(g, fid, "M3", "counterbore").volume()
    res = cut(g, fid, "M3", "nutpocket")
    lost = BLOCK_VOLUME - res.volume()
    # the hole plus the pocket around it up to the nut height + 0.2 mm, a hexagon here
    hexagon = math.sqrt(3.0) / 2.0 * 5.7 ** 2
    assert lost == pytest.approx(math.pi * 1.8 ** 2 * 10.0
                                 + (hexagon - math.pi * 1.8 ** 2) * 2.6, rel=0.01)
    assert through < lost < counterbore, (through, lost, counterbore)
    g.ents = Entities.from_manifold(res)
    flats = hexagon_opening(g.ents, BLOCK[2])
    assert flats == pytest.approx(5.7, abs=0.02), "the nut does not fit in the pocket"
    assert opening(g.ents, 0.0) == pytest.approx(1.8, abs=0.02), "the screw cannot go through"


def test_a_sideways_teardrop_hole_gets_a_roof_of_45_degrees():
    doc = Document()
    g = block(doc)
    fid = draw_circle(g, center=(BLOCK[0], BLOCK[1] / 2, BLOCK[2] / 2), normal=(1, 0, 0))
    r = 1.8
    res = cut(g, fid, "M3", "through", teardrop=True)
    # the circle plus the roof: the square of r on its corner, minus the quarter of the
    # circle that lies inside it, over the whole width of the block
    area = math.pi * r ** 2 + r ** 2 - math.pi * r ** 2 / 4.0
    assert BLOCK_VOLUME - res.volume() == pytest.approx(area * BLOCK[0], rel=0.01)
    g.ents = Entities.from_manifold(res)
    top = hole_top(g.ents, BLOCK[0])
    assert top == pytest.approx(BLOCK[2] / 2 + r * math.sqrt(2.0), abs=0.02), "no roof"


def test_a_sideways_hole_without_teardrop_stays_round():
    doc = Document()
    g = block(doc)
    fid = draw_circle(g, center=(BLOCK[0], BLOCK[1] / 2, BLOCK[2] / 2), normal=(1, 0, 0))
    res = cut(g, fid, "M3", "through")
    assert BLOCK_VOLUME - res.volume() == pytest.approx(math.pi * 1.8 ** 2 * BLOCK[0],
                                                        rel=0.01)
    g.ents = Entities.from_manifold(res)
    assert hole_top(g.ents, BLOCK[0]) == pytest.approx(BLOCK[2] / 2 + 1.8, abs=0.02)


def test_an_upright_hole_needs_no_roof():
    doc = Document()
    g = block(doc)
    fid = draw_circle(g)
    straight = BLOCK_VOLUME - cut(g, fid, "M3", "through").volume()
    roofed = BLOCK_VOLUME - cut(g, fid, "M3", "through", teardrop=True).volume()
    assert roofed == pytest.approx(straight, rel=1e-9)


def test_the_clearance_is_added_to_every_diameter():
    doc = Document()
    g = block(doc)
    fid = draw_circle(g)
    tight = BLOCK_VOLUME - cut(g, fid, "M4", "through", clearance=0.0).volume()
    loose = BLOCK_VOLUME - cut(g, fid, "M4", "through", clearance=0.6).volume()
    assert tight == pytest.approx(math.pi * 2.25 ** 2 * 10.0, rel=0.01)
    assert loose == pytest.approx(math.pi * 2.55 ** 2 * 10.0, rel=0.01)


def test_an_unknown_size_or_kind_is_refused():
    block_man = mf.Manifold.cube(BLOCK)
    with pytest.raises(ValueError, match="unknown screw size"):
        threads.screw_hole(block_man, (0, 0, 10), (0, 0, -1), 10.0, "M99", "through")
    with pytest.raises(ValueError, match="unknown screw hole kind"):
        threads.screw_hole(block_man, (0, 0, 10), (0, 0, -1), 10.0, "M3", "oval")


# -------------------------------------------------------------------- the window

def select_circle(win, g, fid):
    win.doc.set_selection([("f", g.id, fid)])


def test_the_action_cuts_the_hole_and_keeps_the_name_and_color(win):
    g = block(win.doc, name="my part", color="#d23b3b")
    fid = draw_circle(g)
    select_circle(win, g, fid)
    win._apply_screw_hole("M3", "counterbore", "counterbore", 0.2)
    assert pump_until(lambda: g.ents.to_manifold().volume() < BLOCK_VOLUME - 120), "no hole"
    assert g.name == "my part" and g.color == "#d23b3b"
    assert opening(g.ents, BLOCK[2]) == pytest.approx(2.85, abs=0.02)
    assert pump_until(lambda: "M3" in win.hint.text()), win.hint.text()


def test_the_action_gives_a_sideways_hole_its_roof(win):
    g = block(win.doc)
    fid = draw_circle(g, center=(BLOCK[0], BLOCK[1] / 2, BLOCK[2] / 2), normal=(1, 0, 0))
    select_circle(win, g, fid)
    win._apply_screw_hole("M3", "through", "through", 0.2, teardrop=True)
    # a round M3 hole takes 203 mm³ out of the block, the teardrop 217 mm³
    assert pump_until(lambda: g.ents.to_manifold().volume() < BLOCK_VOLUME - 210), "no roof"
    assert hole_top(g.ents, BLOCK[0]) == pytest.approx(BLOCK[2] / 2 + 1.8 * math.sqrt(2.0),
                                                       abs=0.02)


def test_the_action_is_one_undo_step(win):
    g = block(win.doc)
    fid = draw_circle(g)
    select_circle(win, g, fid)
    win._apply_screw_hole("M3", "through", "through", 0.2)
    assert pump_until(lambda: g.ents.to_manifold().volume() < BLOCK_VOLUME - 50)
    win.doc.undo()
    assert win.doc.group(g.id).ents.to_manifold().volume() == pytest.approx(BLOCK_VOLUME,
                                                                            rel=1e-3)


def test_a_face_that_is_not_a_circle_gives_a_clear_message(win):
    msgs = []
    win.view.status_changed.connect(msgs.append)
    g = block(win.doc)
    draw_circle(g)
    square = next(fid for fid in g.ents.F
                  if threads.face_thread_context(g.ents, fid, "outer") is None
                  and abs(float(g.ents.F[fid].normal[2])) > 0.999)
    select_circle(win, g, square)
    win._apply_screw_hole("M3", "through", "through", 0.2)
    assert any("circle" in m for m in msgs), msgs
    assert g.ents.to_manifold().volume() == pytest.approx(BLOCK_VOLUME, rel=1e-6)


def test_a_group_without_a_selected_circle_is_left_alone(win):
    msgs = []
    win.view.status_changed.connect(msgs.append)
    g = block(win.doc)
    win.doc.set_selection([("g", g.id, g.id)])
    steps = len(win.doc.undo_stack)
    win._apply_screw_hole("M3", "through", "through", 0.2)
    assert any("circle" in m for m in msgs), msgs
    assert len(win.doc.undo_stack) == steps, "nothing changed: there is nothing to undo"


def test_the_dialog_starts_at_the_last_values_and_remembers_the_new_ones(win, monkeypatch):
    seen = {}

    def accept(dlg):
        combos = dlg.findChildren(QComboBox)
        spin = dlg.findChild(QDoubleSpinBox)
        seen["start"] = (combos[0].currentText(), combos[1].currentText(), spin.value())
        combos[0].setCurrentText("M6")
        combos[1].setCurrentIndex(1)
        spin.setValue(0.5)
        return QDialog.Accepted

    monkeypatch.setattr(app_module.QDialog, "exec", accept)
    win.settings.screw_hole_size = "M5"
    win.settings.screw_hole_kind = "countersink"
    win.settings.screw_hole_clearance = 0.45
    win.screw_hole_dialog()
    assert seen["start"] == ("M5", "Countersunk", pytest.approx(0.45))
    assert (win.settings.screw_hole_size, win.settings.screw_hole_kind) == ("M6", "counterbore")
    assert win.settings.screw_hole_clearance == pytest.approx(0.5)


def test_the_dialog_offers_the_nut_pocket_and_remembers_it(win, monkeypatch):
    seen = {}

    def accept(dlg):
        kinds = dlg.findChildren(QComboBox)[1]
        seen["items"] = [kinds.itemText(i) for i in range(kinds.count())]
        kinds.setCurrentIndex(kinds.count() - 1)
        return QDialog.Accepted

    monkeypatch.setattr(app_module.QDialog, "exec", accept)
    win.screw_hole_dialog()
    assert seen["items"] == ["Through", "Counterbore", "Countersunk", "Nut pocket"]
    assert win.settings.screw_hole_kind == "nutpocket"


def test_the_dialog_remembers_the_teardrop(win, monkeypatch):
    seen = []

    def accept(dlg):
        box = dlg.findChild(QCheckBox)
        seen.append((box.isChecked(), box.toolTip()))
        box.setChecked(not box.isChecked())
        return QDialog.Accepted

    monkeypatch.setattr(app_module.QDialog, "exec", accept)
    win.settings.screw_hole_teardrop = False
    win.screw_hole_dialog()
    assert seen[0][0] is False and seen[0][1], seen[0]
    assert win.settings.screw_hole_teardrop is True
    win.screw_hole_dialog()
    assert seen[1][0] is True, "the box does not come back checked"
    assert win.settings.screw_hole_teardrop is False


def test_the_action_is_in_the_print_tools_menu_after_the_threads(win):
    labels = [clean(a.text()) for a in win.flat_actions(win.menuBar())]
    at = labels.index("Thread")
    assert labels[at:at + 6] == ["Thread", "External Thread on Cylinder", "Internal Thread in Hole",
                                 "New Bolt", "New Nut or Sleeve", "Screw Hole"]

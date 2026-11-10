"""Project box: an enclosure to print, with a lid that slides into it.

`enclosure.project_box` gives the base and the lid of a box as two closed volumes. The
inside is what the user says, because that is what has to fit: the electronics. The wall is
added around it, so the box stands `length + 2 × wall` wide, `width + 2 × wall` deep and
`floor + height` high, with the vertical corners rounded by `radius` outside and
`radius − wall` inside, which keeps the wall the same thickness all the way round.

The lid is a plate of the same outline with a lip on its underside, a ring as thick as the
wall, that goes into the opening with `clearance` millimetres of room on every side. The lid
comes back flipped for printing: its flat top on the bed and its lip in the air, so neither
part needs support. Flipped back over the base it must fit without touching it, and with no
clearance at all the lip must fill the opening exactly.

These tests (Qt offscreen) measure the box against the analytic volume of a rounded
rectangle (`L × W − (4 − π) × r²`), let the lid slide into the base, check the refused input
and the place in the interface (Print Tools next to the spur gear, both parts in one undo
step, 10 mm apart on the bed, and the dialog that remembers its values).

With a screw size the base also gets a post in every inside corner with a blind pilot hole
and the lid a clearance hole above it; the lip makes room for the posts, so the lid still
slides down without touching them.

With vent slots the two long walls get the same vertical slots through them, evenly spaced
and centred along the length, that stop below the lip of the lid; the lid is not changed.

Run with:  python -m pytest tests/test_enclosure.py
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
from PySide6.QtCore import QCoreApplication  # noqa: E402
from PySide6.QtWidgets import (QApplication, QComboBox, QDialog,  # noqa: E402
                               QDoubleSpinBox, QSpinBox)

import volura.app as app_module  # noqa: E402
from volura import enclosure  # noqa: E402
from volura.settings import qsettings  # noqa: E402

_app = QApplication.instance() or QApplication(sys.argv[:1])

BOX = {"length": 80.0, "width": 50.0, "height": 30.0, "wall": 2.0, "floor": 2.0,
       "radius": 3.0, "lid": 2.0, "lip": 3.0, "clearance": 0.2}
OPTIONS = dict(BOX)
#: the arc of an offset corner is a polygon, so a rounded box is at most this far off
REL = 0.01


def clean(text):
    return text.replace("&", "").replace("…", "").replace("...", "").strip()


def pump_until(predicate, timeout_ms=15000):
    """Spin the event loop until `predicate()` holds or the time is up; returns its value."""
    deadline = time.monotonic() + timeout_ms / 1000.0
    while not predicate() and time.monotonic() < deadline:
        _app.processEvents()
        time.sleep(0.005)
    return predicate()


def rounded_area(length, width, radius):
    """The area of a rectangle with rounded corners: the rectangle minus the corners."""
    r = max(0.0, min(float(radius), 0.5 * min(float(length), float(width))))
    return float(length) * float(width) - (4.0 - math.pi) * r * r


def box(**kw):
    return dict(BOX, **kw)


def base_volume(**kw):
    o = box(**kw)
    return (rounded_area(o["length"] + 2 * o["wall"], o["width"] + 2 * o["wall"], o["radius"])
            * (o["floor"] + o["height"])
            - rounded_area(o["length"], o["width"], max(0.0, o["radius"] - o["wall"]))
            * o["height"])


def lid_volume(**kw):
    o = box(**kw)
    t, c = o["wall"], o["clearance"]
    return (rounded_area(o["length"] + 2 * t, o["width"] + 2 * t, o["radius"]) * o["lid"]
            + (rounded_area(o["length"] - 2 * c, o["width"] - 2 * c,
                            max(0.0, o["radius"] - t - c))
               - rounded_area(o["length"] - 2 * (c + t), o["width"] - 2 * (c + t),
                              max(0.0, o["radius"] - 2 * t - c))) * o["lip"])


def outline_at(m, z):
    """The outline of a part where it is cut at `z`, as a bounding box."""
    return np.asarray(m.slice(z).bounds(), dtype=float)


def bbox(m):
    """The bounding box of a manifold as (low, high)."""
    b = np.asarray(m.bounding_box(), dtype=float)
    return b[:3], b[3:]


def fitted(lid, o):
    """The lid flipped back the way it sits on the box: the plate on the walls."""
    return lid.mirror((0.0, 0.0, 1.0)).translate((0.0, 0.0, o["floor"] + o["height"]
                                                  + o["lid"]))


@pytest.fixture()
def win(monkeypatch, tmp_path):
    monkeypatch.setattr(app_module.errors, "log_dir", lambda: str(tmp_path))
    from volura.app import MainWindow
    w = MainWindow()
    w.doc.dirty = False
    QCoreApplication.processEvents()            # handle the deferred initial view now
    w.resize(900, 600)
    yield w
    w.doc.dirty = False
    w.close()


# ------------------------------------------------------------------ the box

def test_the_base_is_the_inside_with_wall_around_it():
    base, _lid = enclosure.project_box(**BOX)
    lo, hi = bbox(base)
    assert lo[:2] == pytest.approx([-42.0, -27.0]), "the outside is the inside plus two walls"
    assert hi[:2] == pytest.approx([42.0, 27.0])
    assert lo[2] == pytest.approx(0.0), "the box does not stand on the bed"
    assert hi[2] == pytest.approx(BOX["floor"] + BOX["height"])


def test_the_base_holds_the_volume_it_promises():
    base, _lid = enclosure.project_box(**BOX)
    assert base.volume() == pytest.approx(base_volume(**BOX), rel=REL)
    small = enclosure.project_box(**box(length=23.5, width=17.0, height=9.0, radius=0.0))
    assert small[0].volume() == pytest.approx(base_volume(length=23.5, width=17.0, height=9.0,
                                                          radius=0.0), rel=1e-6)


def test_the_lid_is_a_plate_with_a_lip_up():
    _base, lid = enclosure.project_box(**BOX)
    lo, hi = bbox(lid)
    assert lo[:2] == pytest.approx([-42.0, -27.0]), "the lid covers the box"
    assert lo[2] == pytest.approx(0.0), "the flat top of the lid lies on the bed"
    assert hi[2] == pytest.approx(BOX["lid"] + BOX["lip"]), "the lip stands up"
    assert lid.volume() == pytest.approx(lid_volume(**BOX), rel=REL)


def test_both_parts_are_valid_solids_without_holes():
    base, lid = enclosure.project_box(**BOX)
    for m in (base, lid):
        assert m.status() == mf.Error.NoError and not m.is_empty()
        assert m.genus() == 0, "a tray and a lid with a blind groove are both solid"


def test_the_lid_slides_into_the_base_without_touching_it():
    o = box(clearance=0.4)
    base, lid = enclosure.project_box(**o)
    inter = base ^ fitted(lid, o)
    assert inter.volume() < 1e-6, "the lip runs into the wall"
    assert fitted(lid, o).status() == mf.Error.NoError


def test_without_clearance_the_lip_fills_the_opening():
    o = box(clearance=0.0)
    base, lid = enclosure.project_box(**o)
    inside = np.asarray(enclosure.rounded_rect(o["length"], o["width"],
                                               o["radius"] - o["wall"]).bounds(), dtype=float)
    assert np.max(np.abs(outline_at(lid, o["lid"] + 0.5 * o["lip"]) - inside)) < 1e-3, \
        "the lip must touch the inner wall"
    assert (base ^ fitted(lid, o)).volume() < 1e-6
    loose = box(clearance=0.5)
    loose_lid = enclosure.project_box(**loose)[1]
    lip = outline_at(loose_lid, loose["lid"] + 0.5 * loose["lip"])
    assert lip[0] - inside[0] == pytest.approx(0.5), "the clearance is room on every side"


def test_a_box_without_rounded_corners_works():
    o = box(radius=0.0)
    base, lid = enclosure.project_box(**o)
    lo, hi = bbox(base)
    assert [hi[0] - lo[0], hi[1] - lo[1]] == pytest.approx([84.0, 54.0])
    assert base.genus() == 0 and lid.genus() == 0
    assert base.volume() == pytest.approx(base_volume(**o), rel=1e-6)
    assert (base ^ fitted(lid, o)).volume() < 1e-6


def test_a_rounding_over_half_the_box_makes_a_cylinder():
    o = box(length=40.0, width=40.0, height=20.0, radius=25.0)
    base, lid = enclosure.project_box(**o)
    lo, hi = bbox(base)
    assert hi[0] - lo[0] == pytest.approx(44.0)
    assert base.status() == mf.Error.NoError and base.genus() == 0
    assert lid.status() == mf.Error.NoError and lid.genus() == 0


@pytest.mark.parametrize("kw, wrong", (
    ({"length": 0.0}, "greater than zero"), ({"width": -10.0}, "greater than zero"),
    ({"height": 0.0}, "greater than zero"),
    ({"wall": 0.5}, "wall"), ({"wall": -2.0}, "wall"),
    ({"floor": 0.2}, "floor"), ({"lid": 0.2}, "lid"),
    ({"clearance": -0.1}, "clearance"), ({"clearance": 1.5}, "clearance"),
    ({"lip": 30.0}, "lip"), ({"lip": -1.0}, "lip"),
    ({"radius": -1.0}, "corner radius")))
def test_impossible_boxes_raise(kw, wrong):
    with pytest.raises(ValueError, match=wrong):
        enclosure.project_box(**box(**kw))


# ------------------------------------------------------------------ screw posts

def screw_volume(size, **kw):
    """The plastic the four posts add to the base, with the pilot holes taken out again."""
    o = box(**kw)
    nominal = float(size[1:])
    post = enclosure.POST_OVER_SCREW * nominal
    pilot = enclosure.PILOT_OVER_SCREW * nominal
    h = o["height"] - o["lip"]
    return 4.0 * (math.pi * (0.5 * post) ** 2 * h
                  - math.pi * (0.5 * pilot) ** 2 * (h - enclosure.POST_FLOOR))


def test_without_a_screw_size_the_box_is_the_plain_one():
    plain = enclosure.project_box(**BOX)
    for off in (None, ""):
        base, lid = enclosure.project_box(**BOX, screws=off)
        assert base.volume() == plain[0].volume()
        assert lid.volume() == plain[1].volume()


def test_the_posts_stand_in_the_inside_corners():
    plain = enclosure.project_box(**BOX)[0]
    base = enclosure.project_box(**BOX, screws="M3")[0]
    assert base.volume() - plain.volume() == pytest.approx(screw_volume("M3"), rel=0.15)
    assert base.status() == mf.Error.NoError and base.genus() == 0


def test_the_lid_has_a_hole_above_every_post():
    base, lid = enclosure.project_box(**BOX, screws="M3")
    assert lid.status() == mf.Error.NoError
    assert lid.genus() == 4, "a clearance hole through the plate for every post"
    assert (base ^ fitted(lid, BOX)).volume() < 1e-6, "the lip of the lid runs into a post"


def test_the_pilot_hole_stops_short_of_the_floor():
    o = box()
    base = enclosure.project_box(**o, screws="M3")[0]
    floor = base.slice(0.5 * o["floor"])
    assert floor.area() == pytest.approx(rounded_area(o["length"] + 2 * o["wall"],
                                                      o["width"] + 2 * o["wall"],
                                                      o["radius"]), rel=REL), \
        "the pilot hole goes through the bottom"
    wall = (rounded_area(o["length"] + 2 * o["wall"], o["width"] + 2 * o["wall"], o["radius"])
            - rounded_area(o["length"], o["width"], max(0.0, o["radius"] - o["wall"])))
    post = 0.5 * enclosure.POST_OVER_SCREW * 3.0
    pilot = 0.5 * enclosure.PILOT_OVER_SCREW * 3.0
    above = base.slice(o["floor"] + enclosure.POST_FLOOR + 0.5)
    assert above.area() == pytest.approx(wall + 4.0 * math.pi * (post * post - pilot * pilot),
                                         rel=REL), "the posts have no pilot hole"


@pytest.mark.parametrize("kw, screws, wrong", (
    ({"width": 15.0}, "M4", "posts"), ({"length": 12.0, "width": 12.0}, "M3", "posts"),
    ({}, "M9", "screw size")))
def test_posts_that_cannot_be_built_are_refused(kw, screws, wrong):
    with pytest.raises(ValueError, match=wrong):
        enclosure.project_box(**box(**kw), screws=screws)


# ------------------------------------------------------------------ vent slots

def slot_volume(**kw):
    """The plastic one vent slot takes out of one wall: a slab straight through the wall."""
    o = box(**kw)
    h = o["height"] - o["lip"] - enclosure.VENT_BOTTOM - enclosure.VENT_TOP
    return enclosure.VENT_WIDTH * o["wall"] * h


def test_without_vent_slots_the_box_is_the_plain_one():
    plain = enclosure.project_box(**BOX)
    for off in (0, None):
        base, lid = enclosure.project_box(**BOX, vents=off)
        assert base.volume() == plain[0].volume()
        assert lid.volume() == plain[1].volume()


def test_the_slots_go_through_both_long_walls():
    plain = enclosure.project_box(**BOX)[0]
    base = enclosure.project_box(**BOX, vents=5)[0]
    assert plain.volume() - base.volume() == pytest.approx(2 * 5 * slot_volume(), rel=0.01)
    assert base.status() == mf.Error.NoError
    assert base.genus() == plain.genus() + 10, "every slot through a wall is one hole"
    lid = enclosure.project_box(**BOX, vents=5)[1]
    assert lid.volume() == pytest.approx(enclosure.project_box(**BOX)[1].volume()), \
        "the vent slots may not change the lid"


def test_the_slots_run_between_the_floor_and_the_lip_of_the_lid():
    o = box()
    base = enclosure.project_box(**o, vents=3)[0]
    wall = (rounded_area(o["length"] + 2 * o["wall"], o["width"] + 2 * o["wall"], o["radius"])
            - rounded_area(o["length"], o["width"], max(0.0, o["radius"] - o["wall"])))
    slots = 2 * 3 * enclosure.VENT_WIDTH * o["wall"]
    assert base.slice(o["floor"] + 1.0).area() == pytest.approx(wall, rel=REL), \
        "a slot reaches into the floor"
    assert base.slice(o["floor"] + o["height"] - o["lip"] - 1.0).area() == pytest.approx(
        wall, rel=REL), "a slot reaches the top edge of the wall"
    assert base.slice(o["floor"] + enclosure.VENT_BOTTOM + 1.0).area() == pytest.approx(
        wall - slots, rel=REL), "the slots are not through both long walls"


def test_the_slots_stay_clear_of_the_screw_posts():
    posts = enclosure.project_box(**BOX, screws="M3")[0]
    both = enclosure.project_box(**BOX, screws="M3", vents=5)[0]
    assert posts.volume() - both.volume() == pytest.approx(2 * 5 * slot_volume(), rel=1e-6), \
        "a slot cuts into a post"
    assert both.status() == mf.Error.NoError and both.genus() == 10


@pytest.mark.parametrize("kw", ({"vents": -1}, {"vents": 21}, {"vents": 20},
                                {"length": 25.0, "vents": 4}, {"height": 9.0, "vents": 2}))
def test_vent_slots_that_cannot_be_cut_are_refused(kw):
    with pytest.raises(ValueError, match="vent slots"):
        enclosure.project_box(**box(**kw))


# ------------------------------------------------------------------ in the window

def test_the_box_and_the_lid_are_two_parts_in_one_undo_step(win):
    steps = len(win.doc.undo_stack)
    win.make_project_box_model(OPTIONS)
    assert pump_until(lambda: len(win.doc.groups) == 2), "no box in the model"
    assert len(win.doc.undo_stack) == steps + 1, "the box is not one undo step"
    assert [g.name for g in win.doc.groups] == ["Box", "Lid"]
    base, lid = (g.ents.to_manifold() for g in win.doc.groups)
    assert base.volume() == pytest.approx(base_volume(**BOX), rel=REL)
    assert lid.volume() == pytest.approx(lid_volume(**BOX), rel=REL)
    assert [g.id for g in win.doc.selected_groups()] == [g.id for g in win.doc.groups]
    win.doc.undo()
    assert win.doc.groups == []


def test_the_two_parts_stand_next_to_each_other_on_the_bed(win):
    win.make_project_box_model(OPTIONS)
    assert pump_until(lambda: len(win.doc.groups) == 2), "no box in the model"
    assert pump_until(lambda: all(g.bbox() is not None for g in win.doc.groups))
    (blo, bhi), (llo, lhi) = (np.asarray(g.bbox(), dtype=float) for g in win.doc.groups)
    assert blo[2] == pytest.approx(0.0) and llo[2] == pytest.approx(0.0), "not on the bed"
    assert bhi[2] == pytest.approx(BOX["floor"] + BOX["height"])
    assert lhi[2] == pytest.approx(BOX["lid"] + BOX["lip"])
    assert llo[0] - bhi[0] == pytest.approx(10.0), "the parts do not stand 10 mm apart"
    bx, by, _bz = win.settings.bed_size
    assert 0.5 * (blo[0] + lhi[0]) == pytest.approx(0.5 * bx), "the pair is not on the bed"
    assert 0.5 * (blo[1] + bhi[1]) == pytest.approx(0.5 * by)
    assert pump_until(lambda: "inside" in win.hint.text()), win.hint.text()
    assert win.hint.text() == "Project box 80 × 50 × 30 mm added (inside)."


def test_a_box_that_cannot_be_built_is_a_clear_text(win):
    win.make_project_box_model(dict(OPTIONS, wall=0.2))
    assert pump_until(lambda: "Project box failed" in win.hint.text()), win.hint.text()
    assert win.doc.groups == []


def test_the_project_box_follows_the_spur_gear_in_print_tools(win):
    labels = [clean(a.text()) for a in win.flat_actions(win.menuBar())]
    assert labels.index("Project Box") == labels.index("Spur Gear") + 1


def test_the_dialog_starts_at_the_last_values_and_remembers_the_new_ones(win, monkeypatch):
    seen = {}

    def fill(dlg):
        spins = dlg.findChildren(QDoubleSpinBox)
        seen["start"] = [round(sp.value(), 2) for sp in spins]
        seen["title"] = dlg.windowTitle()
        spins[0].setValue(100.0)
        spins[3].setValue(3.0)
        return QDialog.Accepted

    monkeypatch.setattr(app_module.QDialog, "exec", fill)
    win.settings.box3d_length, win.settings.box3d_width = 70.0, 45.0
    win.settings.box3d_height, win.settings.box3d_wall = 25.0, 1.6
    win.settings.box3d_floor, win.settings.box3d_radius = 1.5, 4.0
    win.settings.box3d_lid, win.settings.box3d_lip = 2.5, 4.0
    win.settings.box3d_clearance, win.settings.box3d_board_height = 0.3, 5.0
    win.project_box_dialog()
    assert seen["title"] == "Project Box"
    assert seen["start"] == [70.0, 45.0, 25.0, 1.6, 1.5, 4.0, 2.5, 4.0, 0.3, 5.0]
    assert pump_until(lambda: len(win.doc.groups) == 2), "no box in the model"
    lo, hi = np.asarray(win.doc.groups[0].bbox(), dtype=float)
    assert hi[0] - lo[0] == pytest.approx(100.0 + 2 * 3.0), "the length is not as asked"
    assert win.settings.box3d_length == pytest.approx(100.0)
    assert win.settings.box3d_wall == pytest.approx(3.0)
    s = app_module.Settings()
    assert (s.box3d_length, s.box3d_width, s.box3d_wall) == (pytest.approx(100.0),
                                                             pytest.approx(45.0),
                                                             pytest.approx(3.0))


def test_the_dialog_starts_at_the_last_screws_and_remembers_the_new_ones(win, monkeypatch):
    seen = {}

    def fill(dlg):
        cb = dlg.findChild(QComboBox)
        seen["start"] = cb.currentData()
        cb.setCurrentIndex(cb.findData("M3"))
        return QDialog.Accepted

    monkeypatch.setattr(app_module.QDialog, "exec", fill)
    win.settings.box3d_screws = "M2"
    win.project_box_dialog()
    assert seen["start"] == "M2", "the dialog did not start at the last screw size"
    assert pump_until(lambda: len(win.doc.groups) == 2), "no box in the model"
    lid = win.doc.groups[1].ents.to_manifold()
    assert lid.genus() == 4, "the screw size did not reach the box"
    assert win.settings.box3d_screws == "M3"
    assert app_module.Settings().box3d_screws == "M3"


def test_the_dialog_starts_at_the_last_vents_and_remembers_the_new_ones(win, monkeypatch):
    seen = {}

    def fill(dlg):
        sp = dlg.findChild(QSpinBox)
        seen["start"] = sp.value()
        sp.setValue(6)
        return QDialog.Accepted

    monkeypatch.setattr(app_module.QDialog, "exec", fill)
    win.settings.box3d_vents = 3
    win.project_box_dialog()
    assert seen["start"] == 3, "the dialog did not start at the last number of vent slots"
    assert pump_until(lambda: len(win.doc.groups) == 2), "no box in the model"
    base = win.doc.groups[0].ents.to_manifold()
    assert base.genus() == 12, "the number of vent slots did not reach the box"
    assert win.settings.box3d_vents == 6
    assert app_module.Settings().box3d_vents == 6


def test_the_clearance_starts_at_the_one_of_the_screw_holes():
    """Without a clearance of its own the box takes the one this printer already proved."""
    q = qsettings()
    saved = {k: q.value(k) for k in ("box3d_clearance", "screw_hole_clearance")}
    try:
        q.remove("box3d_clearance")
        q.setValue("screw_hole_clearance", 0.45)
        q.sync()
        assert app_module.Settings().box3d_clearance == pytest.approx(0.45)
    finally:
        for key, value in saved.items():
            q.setValue(key, value)
        q.sync()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))

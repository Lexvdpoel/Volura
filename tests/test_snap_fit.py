"""Snap-fit clips: a printable hook and catch that click together after they come off the bed.

`snap_fit.snap_fit_clip` gives the clip as one closed volume that stands on `z = 0` and is
centred on the origin in plan: a hook with a base plate of `w` by `BASE_T` by `BASE_L`
millimetres and a beam of `L` by `t` by `w` millimetres standing on the bed behind it, ending
in a barb that points `undercut` millimetres out on one side, and a catch: a wall of
`CATCH_WALL` millimetres with a slot of `t + clearance` by `CATCH_FLOOR + w + 2 clearance`
clean through it — only as wide as the beam, so the barb has to push the beam aside to get
through and the retaining face hooks behind the wall — on two foot strips that keep the wall
clear of the beam and of where the barb swings. The two pieces stand `PIECE_ROOM` millimetres
of bed apart. `snap_fit.strain` and `snap_fit.shortest_beam` say what the click does to the
beam; `snap_fit.min_length` says when the beam is too short to reach behind the wall at all.

These tests weigh the clip against what it promises: one volume of exactly two pieces standing
on the bed; the beam with material at z = 0.05 all along its length; the barb sticking out
exactly `undercut`; the slot `t + clearance` wide and open right through the wall, so a box
along the slot touches nothing while a box as wide as the beam with its barb touches the wall;
at least `undercut` plus 1.5 mm of wall beside the slot on the barb side for the retaining face
to hook on, two millimetres of wall over the slot, and a foot that neither blocks the slot nor
stands where the barb swings; a clip that grows with the beam and with the width; a releasable
clip whose retaining face leans over, so it has plastic where the square face has air; the
strain of a click against the hand-computed number and against the limit of the material; a
beam too short to reach behind the wall refused; and the input that cannot be printed refused.
The place in the interface is checked too (Parts right after the print-in-place hinge, one
group in one undo step, and the dialog that remembers its values, warns in the warning colour
when the click strains the beam past what the material takes, and says so when the beam is too
short).

Run with:  python -m pytest tests/test_snap_fit.py
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
from PySide6.QtWidgets import QApplication, QDialog  # noqa: E402

import volura.app as app_module  # noqa: E402
from volura import snap_fit  # noqa: E402

_app = QApplication.instance() or QApplication(sys.argv[:1])

MAT, L, T, W, U = "PLA", 15.0, 1.6, 8.0, 1.0
C = snap_fit.DEFAULT_CLEARANCE
BASE = dict(material=MAT, L=L, t=T, w=W, undercut=U)
OPTIONS = dict(BASE, releasable=False)
#: half the slot, the top of the slot, and how far the lead-in slope eats behind the beam tip
SLOT_C = 0.5 * (T + C)
SLOT_TOP = snap_fit.CATCH_FLOOR + W + 2.0 * C
LEAD = U / math.tan(math.radians(snap_fit.LEAD_IN))


def clean(text):
    return text.replace("&", "").replace("…", "").replace("...", "").strip()


def pump_until(predicate, timeout_ms=15000):
    """Spin the event loop until `predicate()` holds or the time is up; returns its value."""
    deadline = time.monotonic() + timeout_ms / 1000.0
    while not predicate() and time.monotonic() < deadline:
        _app.processEvents()
        time.sleep(0.005)
    return predicate()


def bbox(m):
    """The bounding box of a manifold as (low, high)."""
    b = np.asarray(m.bounding_box(), dtype=float)
    return b[:3], b[3:]


def clip(**kw):
    return snap_fit.snap_fit_clip(**dict(BASE, **kw))


def pieces(m):
    """The clip's two pieces: the hook, whose far end is the beam tip, and the catch."""
    got = m.decompose()
    assert len(got) == 2, "the clip is not a hook and a catch"
    hook, catch = sorted(got, key=lambda p: float(np.asarray(p.bounding_box())[0]))
    return hook, catch


def box(x0, x1, y0, y1, z0, z1) -> mf.Manifold:
    """The rectangular block from (`x0`, `y0`, `z0`) to (`x1`, `y1`, `z1`)."""
    lo = [float(v) for v in (x0, y0, z0)]
    hi = [float(v) for v in (x1, y1, z1)]
    return mf.Manifold.cube([b - a for a, b in zip(lo, hi)]).translate(lo)


def touches(m, probe) -> bool:
    """Whether a probe runs into the part anywhere; a sliver of a touch is not a touch."""
    return (m ^ probe).volume() > 1e-6


def fills(m, probe) -> bool:
    """Whether the part is solid everywhere inside the probe."""
    return abs((m ^ probe).volume() - probe.volume()) < 1e-6


def tip_of(m) -> float:
    """The x of the beam tip: the far end of the hook piece."""
    return float(bbox(m)[1][0])


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


# ------------------------------------------------------------------ the clip

def test_the_default_clip_is_one_valid_volume_of_two_pieces_on_the_bed():
    m = clip()
    assert m.status() == mf.Error.NoError and not m.is_empty()
    hook, catch = pieces(m)
    assert hook.status() == mf.Error.NoError and catch.status() == mf.Error.NoError
    lo, hi = bbox(m)
    assert lo[2] == pytest.approx(0.0), "the clip does not stand on the bed"
    assert lo[0] == pytest.approx(-hi[0]), "the clip is not centred in x on the bed"
    assert lo[1] == pytest.approx(-hi[1]), "the clip is not centred in y on the bed"
    hl, hh = bbox(hook)
    assert hh[1] - hl[1] == pytest.approx(W), "the beam is not as wide as asked"
    assert hh[2] - hl[2] == pytest.approx(W), "the beam does not stand as high as it is wide"
    assert hh[0] - hl[0] == pytest.approx(snap_fit.BASE_L + L), \
        "the hook is not the plate plus the beam long"
    cl, ch = bbox(catch)
    assert cl[2] == pytest.approx(0.0), "the catch does not stand on the bed"
    assert cl[0] - hh[0] == pytest.approx(snap_fit.PIECE_ROOM), \
        "the pieces are not five millimetres of bed apart"
    assert ch[0] - cl[0] == pytest.approx(snap_fit.CATCH_WALL + snap_fit.BASE_L), \
        "the catch is not the wall plus the foot deep"
    assert ch[2] == pytest.approx(SLOT_TOP + snap_fit.WALL_ABOVE), \
        "the wall does not stand WALL_ABOVE over the slot"


def test_the_beam_stands_on_the_bed_all_along_its_length():
    m = clip()
    hook, _catch = pieces(m)
    tip = float(bbox(hook)[1][0])
    for x0 in (tip - L + 0.2, tip - 0.55 * L, tip - 0.25 * L):
        bed = box(x0, tip - 0.05, -0.5 * T + 0.05, 0.5 * T - 0.05, 0.03, 0.07)
        assert fills(m, bed), f"the beam has no plastic on the bed near x = {x0:.1f}"
    assert float(bbox(hook)[0][2]) == pytest.approx(0.0), "the hook does not stand on the bed"


def test_the_barb_sticks_out_exactly_undercut():
    m = clip()
    hook, _catch = pieces(m)
    tip = tip_of(hook)
    z_band = (0.05, W - 0.05)
    # the corner of the barb, where the slope crests at full undercut, runs up the whole beam
    full = box(tip - LEAD + 0.01, tip - LEAD + 0.06, 0.5 * T + 0.05, 0.5 * T + U - 0.05,
               *z_band)
    assert fills(m, full), "the barb does not reach the full undercut"
    over = box(tip - LEAD - 1.0, tip, 0.5 * T + U + 0.05, 0.5 * T + U + 0.5, *z_band)
    assert not touches(m, over), "the barb sticks out further than the undercut"


def test_the_slot_is_as_wide_as_the_beam_and_open_right_through():
    m = clip()
    _hook, catch = pieces(m)
    cl, ch = bbox(catch)
    x0, x1 = cl[0] + 0.2, cl[0] + snap_fit.CATCH_WALL - 0.2   # inside the wall along x
    free = box(cl[0] - 0.9, ch[0] - snap_fit.BASE_L + 0.9,
               -SLOT_C + 0.05, SLOT_C - 0.05, 0.05, SLOT_TOP - 0.05)
    assert not touches(m, free), "the slot is smaller than the beam with its clearance"
    for y0 in (SLOT_C + 0.05, -SLOT_C - 0.45):
        edge = box(x0, x1, y0, y0 + 0.4, 0.05, SLOT_TOP - 0.05)
        assert fills(m, edge), "the wall does not close on the slot at the beam's width"


def test_the_barb_does_not_fit_through_the_slot():
    m = clip()
    _hook, catch = pieces(m)
    cl, _ch = bbox(catch)
    through = box(cl[0] - 0.2, cl[0] + snap_fit.CATCH_WALL + 0.2,
                  -0.5 * (T + U), 0.5 * (T + U), 2.5, SLOT_TOP - 1.0)
    assert touches(m, through), "a beam with its barb slides through without bending"


def test_there_is_room_for_the_retaining_face_to_hook_behind_the_wall():
    m = clip()
    _hook, catch = pieces(m)
    cl, ch = bbox(catch)
    assert ch[1] == pytest.approx(SLOT_C + max(U + 1.5, snap_fit.WALL_ABOVE)), \
        "the wall is not tall and wide enough beside the slot on the barb side"
    assert cl[1] == pytest.approx(-SLOT_C - snap_fit.WALL_ABOVE), \
        "the wall is not two millimetres wide beside the slot on the other side"
    lip = box(cl[0] + 0.1, cl[0] + snap_fit.CATCH_WALL - 0.1,
              SLOT_C + 0.05, SLOT_C + U + 1.5 - 0.05, 0.05, SLOT_TOP + 1.95)
    assert fills(m, lip), "the barb would hook on nothing when it springs back"
    over = box(cl[0] + 0.1, cl[0] + snap_fit.CATCH_WALL - 0.1, -0.5, 0.5,
               SLOT_TOP + 0.05, SLOT_TOP + 1.95)
    assert fills(m, over), "there is not WALL_ABOVE of wall over the slot"


def test_the_foot_holds_the_catch_without_blocking_the_slot():
    m = clip()
    _hook, catch = pieces(m)
    _cl, ch = bbox(catch)
    foot = box(ch[0] - snap_fit.BASE_L + 0.1, ch[0] - 0.1, ch[1] - 0.45, ch[1] - 0.05,
               0.05, snap_fit.BASE_T - 0.05)
    assert fills(m, foot), "there is no foot strip behind the wall on the barb side"
    foot = box(ch[0] - snap_fit.BASE_L + 0.1, ch[0] - 0.1,
               bbox(catch)[0][1] + 0.05, bbox(catch)[0][1] + 0.45,
               0.05, snap_fit.BASE_T - 0.05)
    assert fills(m, foot), "there is no foot strip behind the wall on the other side"
    path = box(ch[0] - snap_fit.BASE_L + 0.05, ch[0] - 0.05,
               -0.5 * T - U + 0.1, 0.5 * T + U - 0.1, 0.05, snap_fit.BASE_T - 0.05)
    assert not touches(m, path), "the foot blocks the beam or the place it swings into"
    swing = box(ch[0] - snap_fit.BASE_L, ch[0] - snap_fit.BASE_L + 0.6,
                -0.5 * T - U + 0.1, 0.5 * T + U - 0.1, 0.05, 2.95)
    assert not touches(m, swing), "the wall back leaves no room for the barb to spring out"


def test_a_longer_beam_makes_a_longer_clip_with_more_plastic():
    last_volume, last_end = 0.0, 0.0
    need = snap_fit.min_length(U, C)
    for length in (max(10.0, need), 15.0, 20.0):
        m = clip(L=length)
        lo, hi = bbox(m)
        assert hi[0] - lo[0] == pytest.approx(snap_fit.BASE_L + length + snap_fit.PIECE_ROOM
                                              + snap_fit.CATCH_WALL + snap_fit.BASE_L)
        assert m.volume() > last_volume and hi[0] > last_end, "the clip does not grow with the beam"
        last_volume, last_end = m.volume(), hi[0]


def test_a_wider_clip_stands_taller_and_holds_more_plastic():
    last_volume, last_top = 0.0, 0.0
    for width in (7.0, 8.0, 10.0):
        m = clip(w=width)
        lo, hi = bbox(m)
        hook, _catch = pieces(m)
        assert bbox(hook)[1][1] - bbox(hook)[0][1] == pytest.approx(width), \
            "the beam is not as wide as asked"
        assert hi[2] - lo[2] == pytest.approx(snap_fit.CATCH_FLOOR + width + 2.0 * C
                                              + snap_fit.WALL_ABOVE), \
            "the slot does not grow with the beam width"
        assert m.volume() > last_volume and hi[2] > last_top, \
            "the clip does not grow with the width"
        last_volume, last_top = m.volume(), hi[2]


def test_the_releasable_leans_the_retaining_face_over():
    holding = clip()
    pull_out = clip(releasable=True)
    tip = tip_of(pieces(pull_out)[0])
    z_band = (0.05, W - 0.05)
    # on the root side of the crest, the leaning 45-degree face has plastic where the square
    # face of the holding clip has air: the edge of the wall can ride back up that face
    face = box(tip - LEAD - 0.7 * U, tip - LEAD - 0.3 * U, 0.5 * T + 0.15, 0.5 * T + 0.35,
               *z_band)
    assert touches(pull_out, face), "the releasable retaining face does not lean over"
    assert not touches(holding, face), "the holding clip has plastic where its face is square"
    assert pull_out.volume() > holding.volume(), "the releasable clip is not a different barb"


def test_the_beam_strains_by_the_book():
    assert snap_fit.strain(1.6, 1.0, 15.0) == pytest.approx(0.010666666666666668, rel=1e-9)
    assert snap_fit.strain(1.6, 1.0, 30.0) == pytest.approx(
        0.25 * snap_fit.strain(1.6, 1.0, 15.0), rel=1e-12)
    assert snap_fit.strain(3.2, 1.0, 15.0) == pytest.approx(
        2.0 * snap_fit.strain(1.6, 1.0, 15.0), rel=1e-12)
    assert snap_fit.strain(1.6, 0.0, 15.0) == pytest.approx(0.0)


def test_the_material_limits_are_the_usual_printables():
    assert snap_fit.MATERIALS == {"PLA": 0.025, "PETG": 0.04, "ABS/ASA": 0.035,
                                  "Nylon": 0.06, "TPU": None}


def test_the_shortest_beam_sits_exactly_on_the_limit():
    for material in ("PLA", "PETG", "ABS/ASA", "Nylon"):
        limit = snap_fit.MATERIALS[material]
        need = snap_fit.shortest_beam(1.6, 1.0, material)
        assert snap_fit.strain(1.6, 1.0, need) == pytest.approx(limit, rel=1e-9)
        assert snap_fit.strain(1.6, 1.0, need + 1.0) < limit, "a longer beam strains more"
        assert snap_fit.strain(1.6, 1.0, need - 1.0) > limit, "a shorter beam strains less"
    assert snap_fit.shortest_beam(1.6, 1.0, "TPU") is None, "TPU may bend freely"


def test_the_default_clip_clicks_within_what_pla_takes():
    eps = snap_fit.strain(T, U, L)
    assert eps < snap_fit.MATERIALS[MAT], "the default parameters would break the beam"
    assert snap_fit.shortest_beam(T, U, MAT) <= L


def test_the_shortest_reaching_beam_is_the_one_the_lead_in_leaves_behind():
    tan30 = math.tan(math.radians(snap_fit.LEAD_IN))
    for u in (snap_fit.UNDERCUT_MIN, 1.0, 2.0, snap_fit.UNDERCUT_MAX):
        assert snap_fit.min_length(u, C) == pytest.approx(
            u / tan30 + snap_fit.CATCH_WALL + C + snap_fit.HOOK_ROOM, rel=1e-12)
    assert snap_fit.min_length(1.0, 1.0) - snap_fit.min_length(1.0, 0.2) == pytest.approx(0.8)
    assert snap_fit.min_length(2.0) > snap_fit.min_length(1.0), "a deeper bite eats more beam"
    # the beam tip is the last of the hook: at a 2 mm undercut the reach limit sits just above
    # the minimum length, and one micrometre longer the clip is a clip
    edge = snap_fit.min_length(2.0, C)
    assert edge > snap_fit.LENGTH_MIN
    assert clip(L=edge + 1e-6, undercut=2.0).status() == mf.Error.NoError
    assert clip(L=snap_fit.min_length(2.0) + 1e-6, undercut=2.0).status() == mf.Error.NoError


@pytest.mark.parametrize("kw, wrong", (
    ({"L": 4.0}, "beam of a snap-fit clip"), ({"L": 61.0}, "beam of a snap-fit clip"),
    ({"t": 0.7}, "beam of a snap-fit clip"), ({"t": 5.1}, "beam of a snap-fit clip"),
    ({"w": 2.0}, "width of a snap-fit clip"), ({"w": 31.0}, "width of a snap-fit clip"),
    ({"undercut": 0.2}, "undercut of a snap-fit clip"),
    ({"undercut": 3.1}, "undercut of a snap-fit clip"),
    ({"t": 0.8, "undercut": 1.3}, "undercut of a snap-fit clip"),
    ({"clearance": 1.2}, "clearance of a snap-fit clip"),
    ({"clearance": -0.1}, "clearance of a snap-fit clip"),
    ({"material": "BPA"}, "material of a snap-fit clip")))
def test_a_clip_that_cannot_be_printed_or_would_not_work_raises(kw, wrong):
    with pytest.raises(ValueError, match=wrong):
        clip(**kw)


@pytest.mark.parametrize("kw", (
    {"L": 5.0, "undercut": 2.0},                       # L 5 < min_length(2) = 5.9 mm
    {"L": 7.8, "undercut": 3.0, "t": 2.0},             # L 7.8 < min_length(3) = 7.9 mm
    {"L": 8.5, "undercut": 3.0, "t": 2.0, "clearance": 1.0},   # deep room, longer bite
    {"L": 5.0, "undercut": 2.0, "releasable": True}))  # the lean eats even more of the reach
def test_a_beam_that_never_reaches_behind_the_wall_raises(kw):
    with pytest.raises(ValueError, match="reach.*behind the wall|behind the wall"):
        clip(**kw)


def test_a_beam_of_five_still_reaches_behind_the_default_barb():
    assert clip(L=5.0).status() == mf.Error.NoError


# ------------------------------------------------------------------ in the window

def test_the_clip_is_one_group_that_one_undo_takes_away(win):
    steps = len(win.doc.undo_stack)
    win.make_snap_fit_model(OPTIONS)
    assert pump_until(lambda: len(win.doc.groups) == 1), "no snap-fit clip in the model"
    assert len(win.doc.undo_stack) == steps + 1, "the clip is not one undo step"
    g = win.doc.groups[0]
    assert g.name == "Snap-fit clip"
    m = g.ents.to_manifold()
    assert m.status() == mf.Error.NoError
    # the window cuts the slot with the clearance this printer proved, which another test
    # may have changed in the stored settings: compare with a clip of that same clearance
    want = clip(clearance=win.settings.screw_hole_clearance)
    assert m.volume() == pytest.approx(want.volume(), rel=1e-6)
    assert [g.id for g in win.doc.selected_groups()] == [g.id for g in win.doc.groups]
    win.doc.undo()
    assert win.doc.groups == []


def test_the_clip_stands_on_the_bed_to_the_right_of_the_model(win):
    win.make_snap_fit_model(OPTIONS)
    assert pump_until(lambda: len(win.doc.groups) == 1), "no snap-fit clip in the model"
    assert pump_until(lambda: win.doc.groups[0].bbox() is not None)
    lo, hi = np.asarray(win.doc.groups[0].bbox(), dtype=float)
    assert lo[2] == pytest.approx(0.0), "the clip does not stand on the bed"
    bx, by, _bz = win.settings.bed_size
    assert 0.5 * (lo[0] + hi[0]) == pytest.approx(0.5 * bx), "the clip is not on the bed"
    assert 0.5 * (lo[1] + hi[1]) == pytest.approx(0.5 * by)
    win.make_snap_fit_model(dict(OPTIONS, L=20.0))
    assert pump_until(lambda: len(win.doc.groups) == 2), "the second clip is missing"
    first, second = (np.asarray(g.bbox(), dtype=float) for g in win.doc.groups)
    assert second[0][0] - first[1][0] == pytest.approx(10.0), "the clips are not 10 mm apart"
    assert pump_until(lambda: "Snap-fit clip added" in win.hint.text()), win.hint.text()
    assert win.hint.text() == "Snap-fit clip added: hook of 20 mm for PLA."


def test_a_clip_that_cannot_be_built_is_a_clear_text(win):
    win.make_snap_fit_model(dict(OPTIONS, L=5.0, t=2.0, undercut=3.0))
    assert pump_until(lambda: "Snap-fit clip failed" in win.hint.text()), win.hint.text()
    assert "behind the wall" in win.hint.text(), win.hint.text()
    assert win.doc.groups == []


def test_the_snap_fit_follows_the_hinge_in_parts(win):
    labels = [clean(a.text()) for a in win.flat_actions(win.parts_category("Mechanical"))]
    assert labels.index("Snap-Fit Clip") == labels.index("Print-in-Place Hinge") + 1


def test_the_dialog_starts_at_the_last_values_and_remembers_the_new_ones(win, monkeypatch):
    seen = {}

    def fill(dlg):
        seen["title"] = dlg.windowTitle()
        seen["start"] = [round(dlg.length.value(), 2), round(dlg.thickness.value(), 2),
                         round(dlg.width.value(), 2), round(dlg.undercut.value(), 2)]
        seen["material"] = dlg.material.currentData()
        seen["info"] = [dlg.info.text(), dlg.info.styleSheet()]
        dlg.material.setCurrentIndex(dlg.material.findData("PLA"))
        dlg.length.setValue(5.0)
        dlg.thickness.setValue(0.8)              # this clamps the undercut to 1.2 mm
        seen["clamped"] = [round(dlg.undercut.maximum(), 2), round(dlg.undercut.value(), 2)]
        dlg.undercut.setValue(0.6)
        seen["warn"] = [dlg.info.text(), dlg.info.styleSheet()]
        dlg.thickness.setValue(5.0)
        dlg.undercut.setValue(3.0)               # this barb needs a longer beam to reach
        seen["reach"] = [dlg.info.text(), dlg.info.styleSheet()]
        dlg.thickness.setValue(0.8)
        dlg.undercut.setValue(0.6)
        dlg.releasable.setChecked(True)
        return QDialog.Accepted

    monkeypatch.setattr(app_module.QDialog, "exec", fill)
    win.settings.snap_material, win.settings.snap_length = "PETG", 12.0
    win.settings.snap_thickness, win.settings.snap_width = 2.0, 10.0
    win.settings.snap_undercut, win.settings.snap_releasable = 0.8, False
    win.snap_fit_dialog()
    assert seen["title"] == "Snap-Fit Clip"
    assert seen["start"] == [12.0, 2.0, 10.0, 0.8]
    assert seen["material"] == "PETG"
    assert "1.7 %" in seen["info"][0] and "PETG" in seen["info"][0]
    assert "behind the wall" not in seen["info"][0], "a reaching beam is called too short"
    assert seen["info"][1] == "", "a clip within the limit is shown in the warning colour"
    assert seen["clamped"] == [1.2, 0.8], "the undercut was not clamped to 1.5 times the beam"
    assert "2.9 %" in seen["warn"][0] and "breaks when it clicks" in seen["warn"][0]
    assert "at least 5.4 mm long" in seen["warn"][0], seen["warn"][0]
    assert "color" in seen["warn"][1] and "#" in seen["warn"][1], seen["warn"][1]
    assert "behind the wall" in seen["reach"][0] and "7.9 mm" in seen["reach"][0]
    assert "90.0 %" in seen["reach"][0], "the reach message lost the strain message"
    assert pump_until(lambda: len(win.doc.groups) == 1), "no snap-fit clip in the model"
    assert win.doc.groups[0].name == "Snap-fit clip"
    assert win.settings.snap_material == "PLA"
    assert win.settings.snap_length == pytest.approx(5.0)
    assert win.settings.snap_thickness == pytest.approx(0.8)
    assert win.settings.snap_undercut == pytest.approx(0.6)
    assert win.settings.snap_releasable is True
    s = app_module.Settings()
    assert (s.snap_material, s.snap_length, s.snap_thickness, s.snap_width, s.snap_undercut,
            s.snap_releasable) == ("PLA", pytest.approx(5.0), pytest.approx(0.8),
                                   pytest.approx(10.0), pytest.approx(0.6), True)


def test_the_remembered_values_are_clamped_when_they_are_read():
    s = app_module.Settings()
    s.q.setValue("snap_material", "Wood")
    s.q.setValue("snap_length", 500.0)
    s.q.setValue("snap_thickness", 0.01)
    s.q.setValue("snap_width", 900.0)
    s.q.setValue("snap_undercut", 9.0)
    s.q.sync()
    t = app_module.Settings()
    assert t.snap_material == "PLA"
    assert t.snap_length == pytest.approx(snap_fit.LENGTH_MAX)
    assert t.snap_thickness == pytest.approx(snap_fit.THICKNESS_MIN)
    assert t.snap_width == pytest.approx(snap_fit.WIDTH_MAX)
    assert t.snap_undercut == pytest.approx(
        min(snap_fit.UNDERCUT_MAX, snap_fit.UNDERCUT_MAX_OF_T * t.snap_thickness))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))

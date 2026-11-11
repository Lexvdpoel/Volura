"""Desk grommets: a printed cable pass-through for the hole in a desk top.

`grommet.grommet` gives the grommet as one closed volume that prints flange down on `z = 0`: a
flange of `hole + 2 * flange` across that lies on the desk with a chamfer on the edge you see, a
sleeve of `hole - 0.4` across that hangs below it into the hole with four crush ribs on it, and a
recess in the flange that a lid drops into. The lid is a disc of `lid_d` with a slot of `slot`
millimetres deep to the rim for the cables; it lies next to the grommet on the bed and in use it
rests on the cone that steps the recess down to the bore.

These tests weigh the part against that: it is one piece that lies flat on the bed and is as wide
as its flange, the sleeve is the hole minus a hair and the ribs are a hair more, the bore is open
from end to end, the lid fits the recess it closes off, the slot is open at the rim, and nothing
overhangs steeper than 45 degrees, because the grommet prints without support.

Run with:  python -m pytest tests/test_grommet.py
"""
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import manifold3d as mf  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from volura import geom, grommet  # noqa: E402

BASE = dict(hole=60.0)
#: the sleeve, the ribs and the cone are polygons, so a part is at most this far off
ABS = 0.02
#: the angle of a point on the sleeve halfway between two of the four crush ribs
HALFWAY = 45.0


def part(**kw) -> mf.Manifold:
    return grommet.grommet(**dict(BASE, **kw))


def numbers(**kw) -> dict:
    return grommet.sizes(**dict(BASE, **kw))


def bbox(m):
    """The bounding box of a manifold as (low, high)."""
    b = np.asarray(m.bounding_box(), dtype=float)
    return b[:3], b[3:]


def pieces(m) -> tuple:
    """The grommet and the lid, from left to right; the lid is None when there is none."""
    out = sorted(m.decompose(), key=lambda p: 0.5 * (bbox(p)[0][0] + bbox(p)[1][0]))
    return out[0], out[1] if len(out) > 1 else None


def lid_x(s) -> float:
    """Where the lid lies on the bed, next to the grommet."""
    return s["flange_d"] + grommet.LID_GAP


def outside(m, z) -> np.ndarray:
    """The points of the widest contour of the part in the section at `z`."""
    return max((np.asarray(p, dtype=float) for p in m.slice(z).to_polygons()),
               key=lambda p: np.hypot(p[:, 0], p[:, 1]).max())


def inside(m, z) -> np.ndarray:
    """The points of the narrowest contour of the part in the section at `z`: the passage."""
    return min((np.asarray(p, dtype=float) for p in m.slice(z).to_polygons()),
               key=lambda p: np.hypot(p[:, 0], p[:, 1]).max())


def box(dx, dy, dz, x, y, z) -> mf.Manifold:
    """A brick `dx` by `dy` by `dz`, centred on (x, y, z)."""
    return mf.Manifold.cube((dx, dy, dz), True).translate((x, y, z))


def touches(m, probe) -> bool:
    """Whether a probe runs into the part anywhere."""
    return (m ^ probe).volume() > 1e-9


def triangles(m):
    """The unit normals and the centres of all triangles, pointing out of the part."""
    v, t = geom.mesh_arrays(m)
    a, b, c = v[t[:, 0]], v[t[:, 1]], v[t[:, 2]]
    n = np.cross(b - a, c - a)
    return n / np.linalg.norm(n, axis=1)[:, None], (a + b + c) / 3.0


# ------------------------------------------------------------------ the part

@pytest.mark.parametrize("hole", (20.0, 35.0, 60.0, 80.0, 120.0))
@pytest.mark.parametrize("with_lid", (True, False))
def test_every_grommet_is_one_piece_that_lies_on_the_bed(hole, with_lid):
    s, m = numbers(hole=hole, lid=with_lid), part(hole=hole, lid=with_lid)
    assert m.status() == mf.Error.NoError and not m.is_empty()
    body, cap = pieces(m)
    assert (cap is not None) == with_lid, "the lid is not made when it is not asked for"
    assert len(m.decompose()) == (2 if with_lid else 1), "the grommet fell apart"
    lo, hi = bbox(body)
    assert lo[2] == pytest.approx(0.0), "the grommet does not lie on the bed"
    assert hi[2] - lo[2] == pytest.approx(grommet.THICK + s["sleeve_len"]), \
        "the grommet is not the flange with the sleeve under it"
    assert hi[0] - lo[0] == pytest.approx(s["flange_d"], abs=ABS), "the flange is not as wide"
    assert hi[1] - lo[1] == pytest.approx(s["flange_d"], abs=ABS), "the flange is not round"
    assert body.genus() == 1, "the grommet is not a tube"


def test_the_lid_lies_flat_next_to_the_grommet():
    s, m = numbers(), part()
    _, cap = pieces(m)
    lo, hi = bbox(cap)
    assert lo[2] == pytest.approx(0.0), "the lid does not lie on the bed"
    assert hi[2] - lo[2] == pytest.approx(grommet.THICK), "the lid is not as thick as the flange"
    assert hi[1] - lo[1] == pytest.approx(s["lid_d"], abs=ABS), "the lid is not as wide as asked"
    assert lo[0] == pytest.approx(lid_x(s) - 0.5 * s["lid_d"], abs=ABS), \
        "the lid does not lie next to the grommet"


def test_the_sleeve_is_as_long_as_the_desk_and_never_longer_than_28():
    assert numbers(desk=10.0)["sleeve_len"] == pytest.approx(10.0 - grommet.SLEEVE_KEEP)
    assert numbers(desk=25.0)["sleeve_len"] == pytest.approx(25.0 - grommet.SLEEVE_KEEP)
    assert numbers(desk=60.0)["sleeve_len"] == pytest.approx(30.0 - grommet.SLEEVE_KEEP)
    for desk in (10.0, 25.0, 60.0):
        lo, hi = bbox(part(desk=desk))
        assert hi[2] - lo[2] == pytest.approx(grommet.THICK + numbers(desk=desk)["sleeve_len"])


# ------------------------------------------------------------------ the sleeve

@pytest.mark.parametrize("hole", (20.0, 35.0, 60.0, 80.0, 120.0))
def test_the_sleeve_fits_the_hole_and_the_ribs_keep_it_there(hole):
    s, m = numbers(hole=hole), part(hole=hole)
    body, _ = pieces(m)
    assert s["sleeve_d"] == pytest.approx(hole - grommet.FIT)
    z = grommet.THICK + 0.5 * s["sleeve_len"]
    poly = outside(body, z)
    radius = np.hypot(poly[:, 0], poly[:, 1])
    angle = np.degrees(np.arctan2(poly[:, 1], poly[:, 0])) % 90.0
    halfway = radius[np.abs(angle - HALFWAY) < 1.0]
    assert halfway.min() == pytest.approx(0.5 * s["sleeve_d"], abs=ABS), \
        "the sleeve is not the hole minus a hair"
    assert halfway.max() == pytest.approx(0.5 * s["sleeve_d"], abs=ABS), "the sleeve is not round"
    b = np.asarray(body.slice(z).bounds(), dtype=float)
    assert max(abs(b[0]), abs(b[2])) == pytest.approx(0.5 * s["sleeve_d"] + grommet.RIB, abs=ABS), \
        "the ribs do not reach 0.4 mm past the sleeve"


@pytest.mark.parametrize("hole", (20.0, 60.0, 120.0))
def test_the_bore_is_open_from_end_to_end(hole):
    s, m = numbers(hole=hole), part(hole=hole)
    r = 0.5 * (s["bore_d"] - 0.2)
    probe = (mf.Manifold.cylinder(s["sleeve_len"] + 4.0 * grommet.THICK, r, r, grommet.SEGMENTS)
             .translate((0.0, 0.0, -grommet.THICK)))
    assert not touches(m, probe), "the bore is blocked"


def test_the_flange_lies_on_the_desk_with_a_chamfer_on_its_edge():
    s, m = numbers(), part()
    body, _ = pieces(m)
    c = grommet.CHAMFER
    assert np.hypot(*outside(body, 0.01).T).max() == pytest.approx(
        0.5 * s["flange_d"] - c + 0.01, abs=ABS), "the edge of the flange is not chamfered"
    for z in (c + 0.1, grommet.THICK - 0.1):
        assert np.hypot(*outside(body, z).T).max() == pytest.approx(0.5 * s["flange_d"], abs=ABS)


def test_the_recess_is_as_deep_as_the_flange_and_steps_down_on_a_cone():
    s, m = numbers(), part()
    body, _ = pieces(m)
    assert s["recess_d"] == pytest.approx(min(
        s["hole"] - 2.0 * s["wall"] + grommet.RECESS_ROOM, s["sleeve_d"] - 2.0 * grommet.NECK))
    assert s["bore_d"] == pytest.approx(s["sleeve_d"] - 2.0 * s["wall"])
    for z in (0.1, grommet.THICK - 0.1):
        assert np.hypot(*inside(body, z).T).max() == pytest.approx(0.5 * s["recess_d"], abs=ABS), \
            "the recess is not as deep as the flange is thick"
    for step in (0.2, 0.6):
        assert np.hypot(*inside(body, grommet.THICK + step).T).max() == pytest.approx(
            0.5 * s["recess_d"] - step, abs=ABS), "the step is not a cone of 45 degrees"
    cone = 0.5 * (s["recess_d"] - s["bore_d"])
    assert np.hypot(*inside(body, grommet.THICK + cone + 0.1).T).max() == pytest.approx(
        0.5 * s["bore_d"], abs=ABS), "the cone does not reach the bore"


# ------------------------------------------------------------------ the lid

@pytest.mark.parametrize("hole", (20.0, 60.0, 120.0))
def test_the_lid_fits_the_recess_with_play_all_round(hole):
    s, m = numbers(hole=hole), part(hole=hole)
    body, cap = pieces(m)
    assert s["lid_d"] == pytest.approx(s["recess_d"] - 2.0 * grommet.PLAY)
    # the lid dropped into the recess, resting on the cone at the bottom of it
    probe = cap.translate((-lid_x(s), 0.0, grommet.PLAY))
    assert (body ^ probe).volume() == pytest.approx(0.0, abs=1e-6), "the lid does not fit"


@pytest.mark.parametrize("slot", (5.0, 12.0, 15.0))
def test_the_slot_is_open_at_the_rim_and_as_wide_as_asked(slot):
    s, m = numbers(slot=slot), part(slot=slot)
    _, cap = pieces(m)
    assert 0.5 * s["lid_d"] - grommet.LID_EDGE > slot, "this slot does not reach its full depth"
    rim, free = 0.5 * s["lid_d"], slot - 2.0 * grommet.SLOT_R
    inner = lid_x(s) + rim - slot
    assert not touches(cap, box(free, 11.8, 6.0, inner + grommet.SLOT_R + 0.5 * free, 0.0, 1.0)), \
        "there is no room for a cable in the slot"
    assert touches(cap, box(free, 12.2, 6.0, inner + grommet.SLOT_R + 0.5 * free, 0.0, 1.0)), \
        "the slot is not 12 mm across"
    assert not touches(cap, box(1.0, 11.8, 6.0, lid_x(s) + rim - 0.5, 0.0, 1.0)), \
        "the slot is closed at the rim"
    assert touches(cap, box(0.4, 0.4, 0.4, inner - 0.5, 0.0, 1.0)), "the slot is deeper than asked"
    assert not touches(cap, box(0.4, 0.4, 0.4, inner + 0.5, 0.0, 1.0)), "the slot is too shallow"


def test_a_slot_never_eats_more_than_half_the_lid():
    for hole in (35.0, 60.0):
        s, m = numbers(hole=hole, slot=99.0), part(hole=hole, slot=99.0)
        assert s["slot"] == pytest.approx(0.5 * hole), "the slot is not clamped to the hole"
        _, cap = pieces(m)
        inner = lid_x(s) + grommet.LID_EDGE
        assert touches(cap, box(0.4, 0.4, 0.4, inner - 0.5, 0.0, 1.0)), "the slot is too deep"
        assert not touches(cap, box(0.4, 0.4, 0.4, inner + 0.5, 0.0, 1.0)), \
            "the slot does not reach as far as it may"
        whole = mf.Manifold.cylinder(grommet.THICK, grommet.LID_EDGE - 0.2,
                                     grommet.LID_EDGE - 0.2, grommet.SEGMENTS)
        probe = whole.translate((lid_x(s), 0.0, 0.0))
        assert (probe - cap).volume() == pytest.approx(0.0, abs=1e-6), \
            "there is no whole lid left around the slot"


# ------------------------------------------------------------------ printing flange down

@pytest.mark.parametrize("wall", (grommet.WALL_MIN, 3.0, grommet.WALL_MAX))
@pytest.mark.parametrize("hole", (grommet.HOLE_MIN, 60.0, grommet.HOLE_MAX))
def test_the_sleeve_holds_on_to_the_flange_and_the_lid_has_a_ledge(hole, wall):
    """The ring between the recess and the outside joins sleeve and flange; the lid rests inside it."""
    s = numbers(hole=hole, wall=wall)
    assert 0.5 * (s["sleeve_d"] - s["recess_d"]) >= grommet.NECK - 1e-9, \
        "the sleeve hangs on a ring too thin to print"
    assert 0.5 * (s["recess_d"] - s["bore_d"]) >= 0.8 - 1e-9, "the lid has no ledge to rest on"
    assert s["lid_d"] > s["bore_d"] + 0.4, "the lid falls through the bore"
    body, _ = pieces(part(hole=hole, wall=wall))
    assert body.slice(grommet.THICK + 0.01).area() >= 0.9 * np.pi * (
        0.25 * s["sleeve_d"] ** 2 - 0.25 * (s["sleeve_d"] - 2.0 * grommet.NECK) ** 2)


@pytest.mark.parametrize("hole, wall", ((20.0, 2.0), (60.0, 3.0), (120.0, 4.0)))
def test_nothing_overhangs_steeper_than_45_degrees(hole, wall):
    """The flange lies on the bed and the sleeve points up: the cone and the chamfer are 45."""
    m = part(hole=hole, wall=wall)
    n, c = triangles(m)
    assert (n[c[:, 2] < 1e-9][:, 2] < -0.99).all(), "the triangles do not face out of the part"
    down = (n[:, 2] < -np.sin(np.pi / 4.0) - 0.01) & (c[:, 2] > 1e-6)
    assert not down.any(), "a face overhangs steeper than 45 degrees, so it needs support"


# ------------------------------------------------------------------ the numbers

def test_sizes_reports_the_grommet_and_the_lid():
    s = numbers(hole=80.0, desk=40.0, flange=10.0, wall=3.0, slot=25.0, lid=False)
    assert s["hole"] == 80.0 and s["desk"] == 40.0 and s["flange"] == 10.0
    assert s["wall"] == 3.0 and s["slot"] == 25.0 and s["lid"] is False
    assert s["sleeve_d"] == pytest.approx(79.6) and s["sleeve_len"] == pytest.approx(28.0)
    assert s["flange_d"] == pytest.approx(100.0) and s["bore_d"] == pytest.approx(73.6)
    assert s["recess_d"] == pytest.approx(76.0) and s["lid_d"] == pytest.approx(75.4)
    assert set(s) == {"hole", "desk", "flange", "wall", "slot", "lid", "sleeve_d", "sleeve_len",
                      "flange_d", "recess_d", "lid_d", "bore_d"}


def test_the_numbers_are_clamped_to_what_prints():
    thin = numbers(hole=1.0, desk=1.0, flange=1.0, wall=0.1, slot=-4.0)
    assert (thin["hole"], thin["desk"], thin["flange"], thin["wall"], thin["slot"]) == (
        grommet.HOLE_MIN, grommet.DESK_MIN, grommet.FLANGE_MIN, grommet.WALL_MIN, 0.0)
    fat = numbers(hole=999.0, desk=999.0, flange=999.0, wall=999.0, slot=999.0)
    assert (fat["hole"], fat["desk"], fat["flange"], fat["wall"], fat["slot"]) == (
        grommet.HOLE_MAX, grommet.DESK_MAX, grommet.FLANGE_MAX, grommet.WALL_MAX,
        0.5 * grommet.HOLE_MAX)
    body, _ = pieces(part(hole=1.0))
    lo, hi = bbox(body)
    assert hi[0] - lo[0] == pytest.approx(numbers(hole=1.0)["flange_d"], abs=ABS), \
        "the smallest grommet this makes is not the smallest hole"


@pytest.mark.parametrize("bad", (float("nan"), float("inf"), None, "wide"))
def test_a_number_that_is_no_number_is_refused(bad):
    for key in ("hole", "desk", "flange", "wall", "slot"):
        with pytest.raises(ValueError):
            grommet.grommet(**dict(BASE, **{key: bad}))
        with pytest.raises(ValueError):
            grommet.sizes(**dict(BASE, **{key: bad}))


# ------------------------------------------------------------------ the interface

def test_the_dialog_builds_a_grommet_and_remembers_it(monkeypatch, tmp_path):
    import time
    from PySide6.QtCore import QCoreApplication
    from PySide6.QtWidgets import QApplication, QDialog
    app = QApplication.instance() or QApplication(sys.argv[:1])
    import volura.app as app_module
    monkeypatch.setattr(app_module.errors, "log_dir", lambda: str(tmp_path))
    w = app_module.MainWindow()
    w.doc.dirty = False
    QCoreApplication.processEvents()

    def fill(dlg):
        dlg.hole.setValue(80.0)
        dlg.desk.setValue(18.0)
        dlg.slot.setValue(25.0)
        assert dlg.lid.isChecked(), "the lid is made along with the grommet"
        assert "79.6" in dlg.info.text(), "the dialog does not show the sleeve it builds"
        return QDialog.Accepted

    monkeypatch.setattr(app_module.QDialog, "exec", fill)
    try:
        w.grommet_dialog()
        deadline = time.monotonic() + 15.0
        while not w.doc.groups and time.monotonic() < deadline:
            app.processEvents()
            time.sleep(0.005)
        assert len(w.doc.groups) == 1, "no grommet in the model"
        lo, hi = np.asarray(w.doc.groups[0].bbox(), dtype=float)
        s = grommet.sizes(hole=80.0, desk=18.0, slot=25.0)
        assert lo[2] == pytest.approx(0.0, abs=1e-6), "the grommet does not lie on the bed"
        assert hi[2] - lo[2] == pytest.approx(grommet.THICK + s["sleeve_len"], abs=1e-4)
        settings = app_module.Settings()
        assert (settings.grommet_hole, settings.grommet_desk, settings.grommet_slot) == (
            pytest.approx(80.0), pytest.approx(18.0), pytest.approx(25.0))
        labels = [a.text().replace("&", "").replace("…", "") for a in w.flat_actions(w.menuBar())]
        assert labels.index("Desk Grommet") == labels.index("Fan Guard") + 1
    finally:
        w.settings.grommet_hole, w.settings.grommet_desk, w.settings.grommet_slot = 60.0, 25.0, 20.0
        w.settings.save()
        w.doc.dirty = False
        w.close()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))

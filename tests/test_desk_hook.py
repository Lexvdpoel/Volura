"""Desk hooks: a hook that slides over the edge of a desk top and hangs below it.

`desk_hook.desk_hook` gives the hook as one closed volume on `z = 0`: the side profile of a top arm
that lies on the desk, an upright that runs down along the desk's edge, an under arm that clamps
against its bottom with a bump pressing into the wood, a drop and a U whose opening is `hook`
millimetres wide, all extruded `width` millimetres straight up.

These tests weigh the hook against that: one volume on the bed and centred, `grip + 0.3 + 2 * wall
+ hook` by `desk + 3 * wall + drop` outside and `width` high; the desk it slides over, with play at
its edge and under it and the bump the only thing that presses into it; an opening as wide as asked
and plastic on both sides of it; no face that is not vertical or horizontal, so nothing overhangs;
every range reached by clamping and every number that is no number refused; and the dialog that
builds the hook and remembers it.

Run with:  python -m pytest tests/test_desk_hook.py
"""
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import manifold3d as mf  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from PySide6.QtWidgets import QApplication  # noqa: E402

from volura import desk_hook, geom  # noqa: E402

_app = QApplication.instance() or QApplication(sys.argv[:1])

BASE = dict(desk=25.0, grip=40.0, drop=50.0, hook=30.0, wall=6.0, width=25.0)
#: the rounded corners are polygons, so a part is at most this far off
ABS = 0.05
#: mm across a probe that feels for plastic
PROBE = 0.4
#: mm the test desk reaches in from its edge
DEEP = 30.0
#: mm³ of sliver a boolean leaves behind where two faces of the parts coincide exactly
SLIVER = 1e-4


def part(**kw) -> mf.Manifold:
    return desk_hook.desk_hook(**dict(BASE, **kw))


def numbers(**kw) -> dict:
    return desk_hook.sizes(**dict(BASE, **kw))


def bbox(m):
    """The bounding box of a manifold as (low, high)."""
    b = np.asarray(m.bounding_box(), dtype=float)
    return b[:3], b[3:]


def overlap(m, probe) -> float:
    """How much of two solids is the same: nothing when they only touch.

    Two solids that only touch leave a sliver of a few ten-millionths of a mm³ behind, so that is
    not a touch; plastic that is really there is off by whole tenths of a millimetre.
    """
    return (m ^ probe).volume()


def post(x, y, z0, z1, r=PROBE) -> mf.Manifold:
    """A round post of radius `r` at (`x`, `y`), between the heights `z0` and `z1`."""
    return mf.Manifold.cylinder(z1 - z0, r, r, 32).translate((x, y, z0))


def block(x0, x1, y0, y1, z0, z1) -> mf.Manifold:
    """The block between the corners (`x0`, `y0`, `z0`) and (`x1`, `y1`, `z1`)."""
    return mf.Manifold.cube([x1 - x0, y1 - y0, z1 - z0]).translate((x0, y0, z0))


def frame(s) -> dict:
    """The edge of the desk and its top face, in the millimetres the hook stands on the bed.

    The profile is drawn with the edge of the desk at `x = 0` and its top face at `y = 0`, the desk
    itself at `x < 0`, and the finished hook is centred, so these two say where the desk goes.
    """
    return {"edge": s["grip"] - 0.5 * s["size_x"], "top": 0.5 * s["size_y"] - s["wall"]}


def normals(m):
    """The unit normals and the areas of all triangles, pointing out of the part."""
    v, t = geom.mesh_arrays(m)
    a, b, c = v[t[:, 0]], v[t[:, 1]], v[t[:, 2]]
    n = np.cross(b - a, c - a)
    area = 0.5 * np.linalg.norm(n, axis=1)
    return n / np.maximum(2.0 * area[:, None], 1e-30), area


# ------------------------------------------------------------------ the piece

@pytest.mark.parametrize("size", ((10.0, 20.0, 20.0, 15.0, 4.0, 15.0),
                                  (25.0, 40.0, 50.0, 30.0, 6.0, 25.0),
                                  (50.0, 80.0, 120.0, 60.0, 10.0, 50.0)))
def test_the_hook_is_one_volume_that_stands_on_the_bed_and_is_centred(size):
    desk, grip, drop, hook, wall, width = size
    kw = dict(desk=desk, grip=grip, drop=drop, hook=hook, wall=wall, width=width)
    s, m = desk_hook.sizes(**kw), part(**kw)
    assert m.status() == mf.Error.NoError and not m.is_empty()
    assert len(m.decompose()) == 1, "the hook fell apart"
    assert m.genus() == 0, "the hook has a hole through it"
    lo, hi = bbox(m)
    assert lo[2] == pytest.approx(0.0, abs=1e-6), "the hook does not stand on the bed"
    assert hi[2] - lo[2] == pytest.approx(width, abs=ABS), "not as wide as asked"
    assert np.allclose(lo[:2], -0.5 * np.array([s["size_x"], s["size_y"]]), atol=ABS), \
        "not centred in x and y"
    assert np.allclose(hi[:2], -lo[:2], atol=ABS), "not centred in x and y"


def test_the_outside_is_the_profile_by_its_width():
    s, m = numbers(), part()
    lo, hi = bbox(m)
    assert hi[0] - lo[0] == pytest.approx(s["size_x"], abs=ABS), \
        "not grip plus play plus two walls plus the hook across"
    assert hi[1] - lo[1] == pytest.approx(s["size_y"], abs=ABS), \
        "not desk plus three walls plus the drop high"
    assert hi[2] - lo[2] == pytest.approx(s["width"], abs=ABS), "not as thick as asked"


def test_it_reaches_grip_over_the_desk_and_hangs_drop_below_it():
    s, m = numbers(), part()
    f = frame(s)
    lo, _hi = bbox(m)
    assert lo[0] == pytest.approx(f["edge"] - s["grip"], abs=ABS), \
        "the top arm does not reach over the desk as far as asked"
    assert lo[1] == pytest.approx(f["top"] - s["desk"] - 2 * s["wall"] - s["drop"], abs=ABS), \
        "the hook does not hang as far below the desk as asked"


# ------------------------------------------------------------------ the desk

@pytest.mark.parametrize("kw", (dict(), dict(desk=10.0, grip=20.0, drop=20.0, hook=15.0,
                                             wall=4.0, width=15.0),
                                dict(desk=50.0, grip=80.0, drop=120.0, hook=60.0,
                                     wall=10.0, width=50.0)))
def test_the_desk_slides_over_the_edge_and_only_the_bump_presses(kw):
    """A desk `desk` millimetres thick stands where the desk goes: the upright leaves it `PLAY`
    millimetres at the edge and the under arm `PLAY` millimetres under it, and the only plastic
    that reaches into the wood further in than the rounded corners is the bump."""
    s, m = numbers(**kw), part(**kw)
    f, w = frame(s), s["width"]
    desk = block(f["edge"] - DEEP, f["edge"], f["top"] - s["desk"], f["top"], -1.0, w + 1.0)
    assert overlap(m, desk) < 0.02 * m.volume(), "the hook does not fit over the desk"
    at_edge = block(f["edge"], f["edge"] + desk_hook.PLAY,
                    f["top"] - s["desk"] + desk_hook.INSIDE + 1.0,
                    f["top"] - desk_hook.INSIDE - 1.0, -1.0, w + 1.0)
    assert overlap(m, at_edge) == pytest.approx(0.0, abs=SLIVER), \
        "the upright does not leave the desk its play at the edge"
    under = block(f["edge"] + desk_hook.BUMP_NEAR + desk_hook.INSIDE + 0.5,
                  f["edge"] - desk_hook.INSIDE - 0.5,
                  f["top"] - s["desk"] - desk_hook.PLAY, f["top"] - s["desk"], -1.0, w + 1.0)
    assert overlap(m, under) == pytest.approx(0.0, abs=SLIVER), \
        "the under arm does not leave the desk its play underneath"
    inwood = block(f["edge"] - DEEP, f["edge"] - desk_hook.INSIDE,
                   f["top"] - s["desk"], f["top"], -1.0, w + 1.0)
    assert overlap(m, inwood) > 0.0, "the bump does not clamp the desk"
    assert overlap(m, block(f["edge"] - DEEP, f["edge"] - desk_hook.INSIDE,
                            f["top"] - s["desk"], f["top"] - s["desk"] + 0.5 * desk_hook.BUMP,
                            -1.0, w + 1.0)) > SLIVER, "the bump does not press into the desk"
    assert overlap(m, block(f["edge"] - DEEP, f["edge"] - desk_hook.INSIDE,
                            f["top"] - s["desk"] + desk_hook.BUMP, f["top"],
                            -1.0, w + 1.0)) == pytest.approx(0.0, abs=SLIVER), \
        "the bump presses further into the desk than it is deep"


def test_the_top_arm_lies_on_the_desk():
    """The arm that holds the hook up lies flat on top of the desk, from its edge back over it."""
    s, m = numbers(), part()
    f, w = frame(s), s["width"]
    x0, x1 = f["edge"] - s["grip"] + 1.0, f["edge"] - desk_hook.INSIDE - 1.0
    assert overlap(m, block(x0, x1, f["top"], f["top"] + s["wall"], -1.0, w + 1.0)) \
        > 0.5 * (x1 - x0) * s["wall"] * w, "the top arm does not lie on the desk"
    assert overlap(m, block(x0, x1, f["top"] - 1.0, f["top"], -1.0, w + 1.0)) \
        == pytest.approx(0.0, abs=SLIVER), "the top arm hangs through the desk"


# ------------------------------------------------------------------ the opening

@pytest.mark.parametrize("kw", (dict(), dict(hook=15.0, wall=4.0, desk=10.0, drop=20.0),
                                dict(hook=60.0, wall=10.0, desk=50.0, drop=120.0)))
def test_the_opening_of_the_hook_is_as_wide_as_asked(kw):
    """The U at the bottom of the drop opens upwards with a `hook` millimetre wide gap between the
    drop and its lip, so a bag band or a pair of headphones fits."""
    s, m = numbers(**kw), part(**kw)
    f, w = frame(s), s["width"]
    near = f["edge"] + desk_hook.PLAY + s["wall"]              # the far face of the drop
    under = f["top"] - s["desk"] - s["wall"] - s["drop"]       # the top of the U's opening
    lip = under + desk_hook.LIP_RATIO * s["hook"]              # how high the lip reaches
    y0, y1 = under + desk_hook.INSIDE + 1.0, under + desk_hook.INSIDE + 3.0
    assert overlap(m, block(near, near + s["hook"], y0, y1, -1.0, w + 1.0)) \
        == pytest.approx(0.0, abs=SLIVER), "the opening is narrower than asked"
    assert overlap(m, block(near - 1.0, near, y0, y1, -1.0, w + 1.0)) > 0.0, \
        "there is no drop on this side of the opening"
    assert overlap(m, block(near + s["hook"], near + s["hook"] + 1.0, y0, y1, -1.0, w + 1.0)) \
        > 0.0, "there is no lip on that side of the opening"
    y = 0.5 * (y0 + lip)
    assert overlap(m, post(near + 0.5 * s["hook"], y, 1.0, w - 1.0)) \
        == pytest.approx(0.0, abs=SLIVER), "the opening is not empty"
    assert overlap(m, post(near + s["hook"] + 0.5 * s["wall"], y, 1.0, w - 1.0)) > 0.0, \
        "the lip is not where the opening ends"
    assert overlap(m, post(near + 0.5 * s["hook"], under - 0.5 * s["wall"], 1.0, w - 1.0)) > 0.0, \
        "the bottom of the hook is missing under the opening"


def test_the_lip_is_shorter_than_the_opening_is_wide():
    """The lip reaches 0.6 of the opening up, so what hangs in the hook can still be lifted out."""
    s, m = numbers(), part()
    f, w = frame(s), s["width"]
    near = f["edge"] + desk_hook.PLAY + s["wall"]
    under = f["top"] - s["desk"] - s["wall"] - s["drop"]
    lip = under + desk_hook.LIP_RATIO * s["hook"]
    assert overlap(m, block(near + s["hook"], near + 2 * s["hook"], lip + 1.0, lip + 3.0,
                            -1.0, w + 1.0)) == pytest.approx(0.0, abs=SLIVER), \
        "the lip reaches higher than it should"
    assert overlap(m, post(near + s["hook"] + 0.5 * s["wall"], lip - 1.0, 1.0, w - 1.0)) > 0.0, \
        "the lip stops before it reaches 0.6 of the opening"


# ------------------------------------------------------------------ printing

@pytest.mark.parametrize("kw", (dict(), dict(desk=10.0, grip=20.0, drop=20.0, hook=15.0,
                                             wall=4.0, width=15.0),
                                dict(desk=50.0, grip=80.0, drop=120.0, hook=60.0,
                                     wall=10.0, width=50.0)))
def test_every_face_is_vertical_or_horizontal(kw):
    """The hook is one profile pulled straight up, so no face leans and nothing needs support."""
    n, area = normals(part(**kw))
    keep = area > 1e-9
    vertical = np.abs(n[:, 2]) < 1e-6
    horizontal = np.abs(n[:, 2]) > 1.0 - 1e-6
    assert (vertical | horizontal)[keep].all(), "a face leans, so the hook would overhang"


# ------------------------------------------------------------------ the numbers

def test_sizes_reports_the_hook_it_makes():
    s = desk_hook.sizes(desk=30.0, grip=55.0, drop=70.0, hook=45.0, wall=8.0, width=35.0)
    assert s["desk"] == 30.0 and s["grip"] == 55.0 and s["drop"] == 70.0
    assert s["hook"] == 45.0 and s["wall"] == 8.0 and s["width"] == 35.0
    assert s["size_x"] == pytest.approx(55.0 + desk_hook.PLAY + 16.0 + 45.0)
    assert s["size_y"] == pytest.approx(30.0 + 24.0 + 70.0)
    assert s["size_z"] == pytest.approx(35.0)
    assert set(s) == {"desk", "grip", "drop", "hook", "wall", "width",
                      "size_x", "size_y", "size_z"}


def test_the_default_hook_is_for_a_desk_of_25_mm():
    s = desk_hook.sizes()
    assert s["desk"] == 25.0 and s["grip"] == 40.0 and s["drop"] == 50.0
    assert s["hook"] == 30.0 and s["wall"] == 6.0 and s["width"] == 25.0


def test_the_numbers_are_clamped_to_what_prints():
    small = numbers(desk=0.0, grip=0.0, drop=0.0, hook=0.0, wall=0.0, width=0.0)
    assert small["desk"] == desk_hook.DESK_MIN and small["grip"] == desk_hook.GRIP_MIN
    assert small["drop"] == desk_hook.DROP_MIN and small["hook"] == desk_hook.HOOK_MIN
    assert small["wall"] == desk_hook.WALL_MIN and small["width"] == desk_hook.WIDTH_MIN
    big = numbers(desk=999.0, grip=999.0, drop=999.0, hook=999.0, wall=999.0, width=999.0)
    assert big["desk"] == desk_hook.DESK_MAX and big["grip"] == desk_hook.GRIP_MAX
    assert big["drop"] == desk_hook.DROP_MAX and big["hook"] == desk_hook.HOOK_MAX
    assert big["wall"] == desk_hook.WALL_MAX and big["width"] == desk_hook.WIDTH_MAX
    for kw in (dict(desk=0.0, grip=0.0, drop=0.0, hook=0.0, wall=0.0, width=0.0),
               dict(desk=999.0, grip=999.0, drop=999.0, hook=999.0, wall=999.0, width=999.0)):
        s, m = numbers(**kw), part(**kw)
        lo, hi = bbox(m)
        assert hi[0] - lo[0] == pytest.approx(s["size_x"], abs=ABS), "a clamped size was lost"
        assert hi[1] - lo[1] == pytest.approx(s["size_y"], abs=ABS), "a clamped size was lost"
        assert hi[2] - lo[2] == pytest.approx(s["width"], abs=ABS), "a clamped width was lost"


@pytest.mark.parametrize("bad", (float("nan"), float("inf"), None, "wide"))
def test_a_number_that_is_no_number_is_refused(bad):
    for key in ("desk", "grip", "drop", "hook", "wall", "width"):
        with pytest.raises(ValueError):
            desk_hook.desk_hook(**dict(BASE, **{key: bad}))
        with pytest.raises(ValueError):
            desk_hook.sizes(**dict(BASE, **{key: bad}))


# ------------------------------------------------------------------ the interface

def test_the_dialog_builds_a_hook_and_remembers_it(monkeypatch, tmp_path):
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
        dlg.desk.setValue(30.0)
        dlg.grip.setValue(55.0)
        dlg.drop.setValue(70.0)
        dlg.hook.setValue(45.0)
        dlg.wall.setValue(8.0)
        dlg.width.setValue(35.0)
        assert "116.3" in dlg.info.text(), "the dialog does not show how wide the hook is"
        assert "124.0" in dlg.info.text(), "the dialog does not show how high the hook is"
        return QDialog.Accepted

    monkeypatch.setattr(app_module.QDialog, "exec", fill)
    try:
        w.desk_hook_dialog()
        deadline = time.monotonic() + 20.0
        while len(w.doc.groups) < 1 and time.monotonic() < deadline:
            app.processEvents()
            time.sleep(0.005)
        assert len(w.doc.groups) == 1, "no hook in the model"
        assert [g.name for g in w.doc.groups] == ["Desk hook"], "the group has another name"
        s = desk_hook.sizes(desk=30.0, grip=55.0, drop=70.0, hook=45.0, wall=8.0, width=35.0)
        lo, hi = np.asarray(w.doc.groups[0].bbox(), dtype=float)
        assert lo[2] == pytest.approx(0.0, abs=1e-6), "the hook does not stand on the bed"
        assert hi[0] - lo[0] == pytest.approx(s["size_x"], abs=ABS), "not as wide as computed"
        assert hi[1] - lo[1] == pytest.approx(s["size_y"], abs=ABS), "not as high as computed"
        assert hi[2] - lo[2] == pytest.approx(s["width"], abs=ABS), "not as thick as computed"
        settings = app_module.Settings()
        assert (settings.desk_hook_desk, settings.desk_hook_grip, settings.desk_hook_drop,
                settings.desk_hook_hook, settings.desk_hook_wall, settings.desk_hook_width) == (
            30.0, 55.0, 70.0, 45.0, 8.0, 35.0)
        labels = [a.text().replace("&", "").replace("…", "")
                  for a in w.flat_actions(w.parts_category("Home and Desk"))]
        assert labels.index("Desk Hook") == labels.index("Coaster") + 1
    finally:
        for key, value in (("desk_hook_desk", 25.0), ("desk_hook_grip", 40.0),
                           ("desk_hook_drop", 50.0), ("desk_hook_hook", 30.0),
                           ("desk_hook_wall", 6.0), ("desk_hook_width", 25.0)):
            setattr(w.settings, key, value)
        w.settings.save()
        w.doc.dirty = False
        w.close()


@pytest.mark.parametrize("desk, wall", ((25.0, 6.0), (10.0, 4.0), (50.0, 10.0)))
def test_the_rounded_corners_leave_the_edge_of_the_desk_free(desk, wall):
    """Rounding the inside corners must not fill the corners the edge of the desk goes into: the
    profile meets the desk only with the bump that clamps it."""
    s = desk_hook._sizes(desk, 40.0, 50.0, 30.0, wall, 25.0)
    top = mf.CrossSection.square([30.0, desk]).translate([-30.0, -desk])
    bump = (desk_hook.BUMP_NEAR - desk_hook.BUMP_FAR) * desk_hook.BUMP
    assert (desk_hook._profile(s) ^ top).area() == pytest.approx(bump, abs=1e-6)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))

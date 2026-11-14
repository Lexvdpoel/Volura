"""Bag clips: a flat spring clip that closes a bag and prints lying flat.

`bag_clip.bag_clip` gives the clip as one closed volume on `z = 0`: two arms `length` long and `arm`
wide with `gap` between them, joined at one end by a half ring - the spring loop - and closed at the
other end by a hook that snaps over the upper arm. With `teeth` the inner faces are serrated.

These tests weigh the clip against that: one volume on the bed and centred, `thickness` high; the gap
between the arms open along their length; the loop joining the arms and its wall standing where it
should; the hook rising past the upper arm without the arm touching it; the teeth jagging the inner
faces when they are on and leaving them flat when they are off; no face that is not vertical or
horizontal, so nothing overhangs; every range reached by clamping and every number that is no number
refused; and the dialog that builds the clip and remembers it.

Run with:  python -m pytest tests/test_bag_clip.py
"""
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import manifold3d as mf  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from PySide6.QtWidgets import QApplication  # noqa: E402

from volura import bag_clip, geom  # noqa: E402

_app = QApplication.instance() or QApplication(sys.argv[:1])

BASE = dict(length=100.0, arm=5.0, thickness=6.0, gap=1.0, loop=4.0, teeth=True)
#: the round corners are polygons, so a part is at most this far off
ABS = 0.05


def part(**kw) -> mf.Manifold:
    return bag_clip.bag_clip(**dict(BASE, **kw))


def numbers(**kw) -> dict:
    return bag_clip.sizes(**dict(BASE, **kw))


def frame(s) -> tuple:
    """The shift from the design point (`x`, `y`) to the centred piece: the loop is the leftmost and
    the lowest thing on the clip and the hook upright the rightmost."""
    r_o = s["outside_length"] - s["length"] - s["arm"]
    return (-0.5 * (s["length"] + s["arm"] - r_o), -0.5 * (s["outside_width"] - 2.0 * r_o))


def bbox(m):
    """The bounding box of a manifold as (low, high)."""
    b = np.asarray(m.bounding_box(), dtype=float)
    return b[:3], b[3:]


def touches(m, probe) -> bool:
    """Whether a probe runs into the part anywhere.

    Two solids that only touch leave a sliver of a few ten-millionths of a mm³ behind, so that is
    not a touch; plastic that is really there is off by whole tenths of a millimetre.
    """
    return (m ^ probe).volume() > 1e-6


def post(s, x, y, r, z0=None, z1=None) -> mf.Manifold:
    """A round post of radius `r` on the design point (`x`, `y`), over the whole clip by default."""
    dx, dy = frame(s)
    z0, z1 = -1.0 if z0 is None else z0, (s["thickness"] + 1.0) if z1 is None else z1
    return mf.Manifold.cylinder(z1 - z0, r, r, 32).translate((x + dx, y + dy, z0))


def triangles(m):
    """The unit normals, the centres and the areas of all triangles."""
    v, t = geom.mesh_arrays(m)
    a, b, c = v[t[:, 0]], v[t[:, 1]], v[t[:, 2]]
    n = np.cross(b - a, c - a)
    area = 0.5 * np.linalg.norm(n, axis=1)
    return n / np.maximum(2.0 * area[:, None], 1e-30), (a + b + c) / 3.0, area


# ------------------------------------------------------------------ the piece

@pytest.mark.parametrize("kw", (dict(), dict(teeth=False), dict(length=50.0, arm=3.0),
                                dict(length=200.0, arm=10.0, thickness=12.0, gap=3.0, loop=10.0),
                                dict(gap=0.4, loop=2.0, teeth=False)))
def test_the_clip_is_one_volume_that_lies_on_the_bed_and_is_centred(kw):
    s, m = numbers(**kw), part(**kw)
    assert m.status() == mf.Error.NoError and not m.is_empty()
    assert len(m.decompose()) == 1, "the clip fell apart"
    lo, hi = bbox(m)
    assert lo[2] == pytest.approx(0.0, abs=1e-6), "the clip does not lie on the bed"
    assert hi[2] - lo[2] == pytest.approx(s["thickness"], abs=ABS), "not as thick as asked"
    assert np.allclose(lo[:2], -0.5 * np.array([s["outside_length"], s["outside_width"]]),
                       atol=ABS), "not centred in x and y"
    assert np.allclose(hi[:2], 0.5 * np.array([s["outside_length"], s["outside_width"]]),
                       atol=ABS), "not the size the clip needs"


def test_the_arms_lie_side_by_side_and_the_loop_joins_them():
    """The piece is one volume because the loop runs from one arm to the other: plastic stands in the
    wall of the loop and none inside it."""
    s, m = numbers(), part()
    r_i = 0.5 * s["gap"] + s["loop"]
    r_o = s["outside_length"] - s["length"] - s["arm"]
    assert r_o == pytest.approx(r_i + 0.6 * s["arm"], abs=ABS), "the loop wall is not an arm's 0.6"
    assert touches(m, post(s, -0.5 * (r_i + r_o), 0.0, 0.2)), "there is no loop between the arms"
    assert not touches(m, post(s, -0.5 * r_i, 0.0, 0.2)), "the loop is not a loop but a disc"
    assert not touches(m, post(s, 0.0, 0.0, 0.2)), "the loop closes the gap between the arms"


def test_the_gap_between_the_arms_is_open_along_their_length():
    """The bag goes between the arms, so nothing stands there - except the teeth when they are on."""
    for teeth in (False, True):
        s, m = numbers(teeth=teeth), part(teeth=teeth)
        r = 0.05 if teeth else 0.5 * s["gap"] - 0.1
        for x in (2.0 * s["loop"] + 2.0, 0.5 * s["length"], s["length"] - 8.0):
            assert not touches(m, post(s, x, 0.0, r)), \
                "something stands between the arms at %g mm" % x


# ------------------------------------------------------------------ the latch

def test_the_hook_rises_past_the_upper_arm_without_touching_it():
    """The upright stands at the end of the lower arm, over the upper one, and the upper arm stops
    short of it and under its lip: the clip snaps shut and is still one piece."""
    s, m = numbers(teeth=False), part(teeth=False)
    a, g, L = s["arm"], s["gap"], s["length"]
    assert touches(m, post(s, L + 0.5 * a, 0.5 * g + 1.5 * a, 0.2)), \
        "the hook does not rise past the upper arm"
    assert touches(m, post(s, L - 1.5 * a, 0.5 * g + 1.5 * a, 0.2)), \
        "the lip does not bend back over the upper arm"
    assert not touches(m, post(s, L - 0.5 * bag_clip.HOOK_GAP, 0.5 * g + 0.5 * a, 0.1)), \
        "the upper arm touches the upright"
    assert not touches(m, post(s, L - 0.5 * bag_clip.HOOK_GAP,
                               0.5 * g + a + 0.5 * bag_clip.HOOK_LIFT, 0.05)), \
        "the upper arm touches the lip"
    assert touches(m, post(s, L - 1.5 * a, 0.5 * g + 0.5 * a, 0.1)), "the upper arm ends too early"


# ------------------------------------------------------------------ the teeth

def test_the_teeth_jag_the_inner_faces_and_interleave():
    s, m = numbers(), part(teeth=True)
    plain = part(teeth=False, gap=s["gap"])
    g = s["gap"]
    assert g == pytest.approx(bag_clip.TEETH_GAP), "the teeth do not ask for room to bite"
    x = 2.0 * s["loop"] + bag_clip.TOOTH_START
    y = 0.5 * g - 0.5 * bag_clip.TOOTH
    assert touches(m, post(s, x, y, 0.1)), "there is no tooth on the upper arm"
    assert touches(m, post(s, x + 0.5 * bag_clip.TOOTH_PITCH, -y, 0.1)), \
        "there is no tooth on the lower arm where the upper one has none"
    assert not touches(plain, post(s, x, y, 0.1)), "the inner face is jagged without the teeth"
    assert not touches(plain, post(s, x + 0.5 * bag_clip.TOOTH_PITCH, -y, 0.1)), \
        "the inner face is jagged without the teeth"


def test_the_teeth_stop_short_of_the_end_of_the_arms():
    s, m = numbers(), part()
    g = s["gap"]
    assert not touches(m, post(s, s["length"] - 1.0, 0.5 * g - 0.5 * bag_clip.TOOTH, 0.1)), \
        "a tooth hangs over the end of the arm"


# ------------------------------------------------------------------ printing

@pytest.mark.parametrize("kw", (dict(), dict(teeth=False), dict(length=50.0, arm=3.0, gap=0.4),
                                dict(length=200.0, arm=10.0, thickness=4.0, loop=2.0)))
def test_every_face_is_vertical_or_horizontal(kw):
    """The clip is one outline pulled straight up, so nothing overhangs and nothing needs support."""
    n, _c, area = triangles(part(**kw))
    keep = area > 1e-9
    up = np.abs(n[keep][:, 2])
    assert np.all((up > 0.999) | (up < 1e-6)), "a face is tilted, so the clip overhangs"


# ------------------------------------------------------------------ the numbers

def test_sizes_reports_the_clip_it_makes():
    s = bag_clip.sizes(length=120.0, arm=6.0, thickness=8.0, gap=2.0, loop=7.0, teeth=False)
    assert (s["length"], s["arm"], s["thickness"], s["gap"], s["loop"], s["teeth"]) == (
        120.0, 6.0, 8.0, 2.0, 7.0, False)
    assert s["outside_length"] == pytest.approx(120.0 + 6.0 + 1.0 + 7.0 + 0.6 * 6.0)
    assert s["outside_width"] == pytest.approx(1.0 + 2.0 * 6.0 + bag_clip.HOOK_LIFT
                                               + 1.0 + 7.0 + 0.6 * 6.0)
    assert set(s) == {"length", "arm", "thickness", "gap", "loop", "teeth", "outside_length",
                      "outside_width"}


def test_the_default_clip_closes_an_ordinary_bag():
    s = bag_clip.sizes()
    assert s["length"] == 100.0 and s["arm"] == 5.0 and s["thickness"] == 6.0
    assert s["teeth"] is True


def test_the_numbers_are_clamped_to_what_prints():
    small = numbers(length=0.0, arm=0.0, thickness=0.0, gap=0.0, loop=0.0)
    assert small["length"] == bag_clip.LENGTH_MIN and small["arm"] == bag_clip.ARM_MIN
    assert small["thickness"] == bag_clip.THICKNESS_MIN
    assert small["gap"] == bag_clip.TEETH_GAP and small["loop"] == bag_clip.ARM_MIN
    big = numbers(length=999.0, arm=999.0, thickness=999.0, gap=999.0, loop=999.0)
    assert big["length"] == bag_clip.LENGTH_MAX and big["arm"] == bag_clip.ARM_MAX
    assert big["thickness"] == bag_clip.THICKNESS_MAX
    assert big["gap"] == bag_clip.GAP_MAX and big["loop"] == bag_clip.LOOP_MAX
    for kw in (dict(length=0.0, arm=0.0, thickness=0.0, gap=0.0, loop=0.0),
               dict(length=999.0, arm=999.0, thickness=999.0, gap=999.0, loop=999.0),
               dict(gap=0.4, teeth=True)):
        s, m = numbers(**kw), part(**kw)
        lo, hi = bbox(m)
        assert hi[2] - lo[2] == pytest.approx(s["thickness"], abs=ABS), "a clamped thickness was lost"
        assert hi[0] - lo[0] == pytest.approx(s["outside_length"], abs=ABS), "a clamped length was lost"
        assert hi[1] - lo[1] == pytest.approx(s["outside_width"], abs=ABS), "a clamped width was lost"


def test_the_loop_is_never_smaller_than_an_arm():
    """The neck from the loop to an arm runs from the inside of the loop to the arm: a loop smaller
    than the arm would make it a point."""
    s = numbers(arm=10.0, loop=2.0)
    assert s["loop"] == 10.0, "a loop smaller than an arm was kept"
    assert len(part(arm=10.0, loop=2.0).decompose()) == 1, "the thin neck cut the clip in two"


@pytest.mark.parametrize("bad", (float("nan"), float("inf"), None, "wide"))
def test_a_number_that_is_no_number_is_refused(bad):
    for key in ("length", "arm", "thickness", "gap", "loop"):
        with pytest.raises(ValueError):
            bag_clip.bag_clip(**dict(BASE, **{key: bad}))
        with pytest.raises(ValueError):
            bag_clip.sizes(**dict(BASE, **{key: bad}))


# ------------------------------------------------------------------ the interface

def test_the_dialog_builds_a_clip_and_remembers_it(monkeypatch, tmp_path):
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
        dlg.length.setValue(80.0)
        dlg.arm.setValue(4.0)
        dlg.thickness.setValue(5.0)
        dlg.gap.setValue(1.2)
        dlg.loop.setValue(6.0)
        assert dlg.teeth.isChecked(), "the teeth are off by default"
        dlg.teeth.setChecked(False)
        assert "93.0" in dlg.info.text(), "the dialog does not show the clip it makes"
        assert "18.0" in dlg.info.text(), "the dialog does not show how wide the clip is"
        return QDialog.Accepted

    monkeypatch.setattr(app_module.QDialog, "exec", fill)
    try:
        w.bag_clip_dialog()
        deadline = time.monotonic() + 20.0
        while len(w.doc.groups) < 1 and time.monotonic() < deadline:
            app.processEvents()
            time.sleep(0.005)
        assert len(w.doc.groups) == 1, "no clip in the model"
        assert [g.name for g in w.doc.groups] == ["Bag clip"]
        s = bag_clip.sizes(length=80.0, arm=4.0, thickness=5.0, gap=1.2, loop=6.0, teeth=False)
        lo, hi = np.asarray(w.doc.groups[0].bbox(), dtype=float)
        assert lo[2] == pytest.approx(0.0, abs=1e-6), "the clip does not lie on the bed"
        assert hi[0] - lo[0] == pytest.approx(s["outside_length"], abs=ABS), "not as long as computed"
        assert hi[1] - lo[1] == pytest.approx(s["outside_width"], abs=ABS), "not as wide as computed"
        assert hi[2] - lo[2] == pytest.approx(s["thickness"], abs=1e-3), "not as thick as computed"
        settings = app_module.Settings()
        assert (settings.bag_clip_length, settings.bag_clip_arm, settings.bag_clip_thickness,
                settings.bag_clip_gap, settings.bag_clip_loop,
                settings.bag_clip_teeth) == (80.0, 4.0, 5.0, 1.2, 6.0, False)
        labels = [a.text().replace("&", "").replace("…", "") for a in w.flat_actions(w.menuBar())]
        assert labels.index("Bag Clip") == labels.index("Screw-Top Jar") + 1
    finally:
        for key, value in (("bag_clip_length", 100.0), ("bag_clip_arm", 5.0),
                           ("bag_clip_thickness", 6.0), ("bag_clip_gap", 1.0),
                           ("bag_clip_loop", 4.0), ("bag_clip_teeth", True)):
            setattr(w.settings, key, value)
        w.settings.save()
        w.doc.dirty = False
        w.close()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))

"""Filament clips: a clip that pushes onto the rim of a spool and holds the end of the filament.

`filament_clip.filament_clip` gives the clip as one closed volume on `z = 0`: two jaws of `WALL`
millimetres that straddle the flange of a spool in a gap as wide as `flange` and are held by a grip
bump in each jaw, joined by a bridge under which the tab hangs to the right. The tab carries
`count` holes of `filament` plus `CLEAR` millimetres through the whole clip, and every hole opens
to the edge of the tab through a slit of `SLIT` times the filament diameter, so the thread is
pushed in sideways and the narrower slit holds it.

These tests weigh the clip against that: one volume on the bed and centred, `thickness` high; the
gap between the jaws as wide as the flange above the bumps and narrower between them; a hole for
every filament that a probe of the thread itself passes through, each with its slit open to the
edge of the tab; no face that is not vertical or horizontal, so nothing overhangs; every range
reached by clamping, every filament that is unknown fallen back and every number that is no number
refused; and the dialog that builds the clip and remembers it.

Run with:  python -m pytest tests/test_filament_clip.py
"""
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import manifold3d as mf  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from PySide6.QtWidgets import QApplication  # noqa: E402

from volura import filament_clip, geom  # noqa: E402

_app = QApplication.instance() or QApplication(sys.argv[:1])

BASE = dict(flange=2.5, filament="1.75", count=1, thickness=5.0)
#: the holes and the rounded corners are polygons, so a part is at most this far off
ABS = 0.05


def part(**kw) -> mf.Manifold:
    return filament_clip.filament_clip(**dict(BASE, **kw))


def numbers(**kw) -> dict:
    return filament_clip.sizes(**dict(BASE, **kw))


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


def post(x, y, r, z0=-1.0, z1=40.0) -> mf.Manifold:
    """A round post of radius `r` on (`x`, `y`) from `z0` to `z1`, to feel the piece with."""
    return mf.Manifold.cylinder(z1 - z0, r, r, 32).translate((x, y, z0))


def frame(s) -> dict:
    """The faces of the clip, in the millimetres the outline is drawn in, and how it lies.

    The outline is drawn with its left face on `x = 0` and its bottom on `y = 0` and the finished
    clip is centred, so `off` is what moves such a point onto the bed.
    """
    w, f, n = filament_clip.WALL, s["flange"], s["count"]
    outer = 2.0 * w + f
    return {"off": (-0.5 * s["length"], -0.5 * s["width"]),
            "left": w, "right": w + f, "outer": outer,
            "tab": outer + filament_clip.HOLE_STEP * (n + 1),
            "top": filament_clip.JAW + w,
            "bottom": filament_clip.JAW + w - filament_clip.TAB,
            "hole_y": filament_clip.JAW + w - filament_clip.HOLE_EDGE,
            "holes": [outer + filament_clip.HOLE_STEP * (k + 1) for k in range(n)]}


def plastic(m, p, x, y, r=0.2) -> bool:
    """Whether there is plastic above (`x`, `y`) of the outline, wherever the clip lies."""
    return touches(m, post(x + p["off"][0], y + p["off"][1], r))


def room(p, x0, x1, y0, y1) -> mf.Manifold:
    """The box between (`x0`, `y0`) and (`x1`, `y1`) of the outline, through the whole clip."""
    return mf.Manifold.cube([x1 - x0, y1 - y0, 41.0]).translate(
        (x0 + p["off"][0], y0 + p["off"][1], -1.0))


def triangles(m):
    """The unit normals and the areas of all triangles."""
    v, t = geom.mesh_arrays(m)
    a, b, c = v[t[:, 0]], v[t[:, 1]], v[t[:, 2]]
    n = np.cross(b - a, c - a)
    area = 0.5 * np.linalg.norm(n, axis=1)
    return n / np.maximum(2.0 * area[:, None], 1e-30), area


# ------------------------------------------------------------------ the piece

@pytest.mark.parametrize("kw", (dict(), dict(filament="2.85"),
                                dict(flange=1.0, count=2, thickness=4.0),
                                dict(flange=6.0, count=3, thickness=10.0),
                                dict(flange=4.0, filament="2.85", count=3)))
def test_the_clip_is_one_volume_that_lies_on_the_bed_and_is_centred(kw):
    s, m = numbers(**kw), part(**kw)
    assert m.status() == mf.Error.NoError and not m.is_empty()
    assert len(m.decompose()) == 1, "the clip fell apart"
    lo, hi = bbox(m)
    assert lo[2] == pytest.approx(0.0, abs=1e-6), "the clip does not lie on the bed"
    assert hi[2] - lo[2] == pytest.approx(s["thickness"], abs=ABS), "not as high as asked"
    assert hi[0] - lo[0] == pytest.approx(s["length"], abs=ABS), "not as long as computed"
    assert hi[1] - lo[1] == pytest.approx(s["width"], abs=ABS), "not as wide as computed"
    assert lo[0] == pytest.approx(-hi[0], abs=ABS) and lo[1] == pytest.approx(-hi[1], abs=ABS), \
        "not centred in x and y"


# ------------------------------------------------------------------ the grip

def test_the_gap_between_the_jaws_is_as_wide_as_the_flange():
    """Above the bumps nothing stands between the jaws, and the jaws stop where the flange ends."""
    s, m = numbers(), part()
    p = frame(s)
    edge = 0.05
    assert (m ^ room(p, p["left"], p["right"], filament_clip.BUMP_HIGH + 1.0,
                     filament_clip.JAW)).volume() == pytest.approx(0.0, abs=1e-6), \
        "the gap is narrower than the flange"
    wall = 2.0 * edge * (filament_clip.JAW - filament_clip.BUMP_HIGH - 1.0) * s["thickness"]
    got = (m ^ room(p, p["left"] - edge, p["right"] + edge, filament_clip.BUMP_HIGH + 1.0,
                    filament_clip.JAW)).volume()
    assert got == pytest.approx(wall, abs=0.02), "the jaws do not grip a flange of this thickness"


def test_the_grip_bumps_pinch_the_gap():
    """Both bumps together are two triangles of `BUMP` millimetres, and they reach no further."""
    s, m = numbers(), part()
    p = frame(s)
    tip = 0.5 * (filament_clip.BUMP_LOW + filament_clip.BUMP_HIGH)
    bumps = (filament_clip.BUMP_HIGH - filament_clip.BUMP_LOW) * filament_clip.BUMP \
        * s["thickness"]
    got = (m ^ room(p, p["left"], p["right"], filament_clip.BUMP_LOW,
                    filament_clip.BUMP_HIGH)).volume()
    assert got == pytest.approx(bumps, abs=1e-3), "the bumps do not pinch the flange"
    assert plastic(m, p, p["left"] + 0.5 * filament_clip.BUMP, tip), \
        "no bump at the middle of its base"
    assert plastic(m, p, p["right"] - 0.5 * filament_clip.BUMP, tip), \
        "no bump at the middle of its base"
    assert not plastic(m, p, p["left"] + filament_clip.BUMP + 0.2, tip), \
        "a bump reaches past its tip into the gap"
    assert not plastic(m, p, p["right"] - filament_clip.BUMP - 0.2, tip), \
        "a bump reaches past its tip into the gap"
    for y in (filament_clip.BUMP_LOW - 1.0, filament_clip.BUMP_HIGH):
        assert (m ^ room(p, p["left"], p["right"], y, y + 1.0)).volume() \
            == pytest.approx(0.0, abs=1e-6), "a bump stands outside the base it is given"


def test_a_jaw_slides_onto_the_flange_over_its_chamfer():
    """The outer bottom corner of a jaw is gone and the far corners of the tab are rounded, so
    nothing digs into the spool the clip slides onto."""
    m = part()
    p = frame(numbers())
    assert not plastic(m, p, 0.25, 0.25), "the left jaw is not chamfered on the outside"
    assert not plastic(m, p, p["outer"] - 0.25, 0.25), "the right jaw is not chamfered outside"
    assert plastic(m, p, 0.25, 1.5) and plastic(m, p, p["outer"] - 0.25, 1.5), \
        "a jaw is chamfered higher than its millimetre"
    assert not plastic(m, p, p["tab"] - 0.25, p["top"] - 0.25), \
        "the far top corner of the tab is not rounded"
    assert not plastic(m, p, p["tab"] - 0.25, p["bottom"] + 0.25), \
        "the far bottom corner of the tab is not rounded"
    assert plastic(m, p, p["tab"] - 1.5, p["top"] - 1.5), "the tab is rounded further in than 2 mm"


# ------------------------------------------------------------------ the holes

@pytest.mark.parametrize("kw", (dict(), dict(filament="2.85"), dict(count=2),
                                dict(count=3, filament="2.85", flange=6.0)))
def test_the_tab_has_a_hole_for_every_filament(kw):
    """A hole of the thread plus `CLEAR` for every hole asked, and plastic between two of them."""
    s, m = numbers(**kw), part(**kw)
    p = frame(s)
    d = float(s["filament"])
    assert len(p["holes"]) == s["count"], "the tab has another number of holes"
    for x in p["holes"]:
        assert not touches(m, post(x + p["off"][0], p["hole_y"] + p["off"][1], 0.5 * d)), \
            "the thread does not pass through the hole"
        assert not plastic(m, p, x + 0.5 * s["hole"] - 0.15, p["hole_y"], 0.1), \
            "the hole is smaller than the thread plus its play"
        assert plastic(m, p, x + 0.5 * s["hole"] + 0.15, p["hole_y"], 0.1), \
            "the hole is bigger than the thread plus its play"
    for k in range(s["count"] - 1):
        assert plastic(m, p, 0.5 * (p["holes"][k] + p["holes"][k + 1]), p["hole_y"]), \
            "two holes ran into one another"
    assert plastic(m, p, p["tab"] - 3.0, p["hole_y"]), "there is a hole where none was asked"


@pytest.mark.parametrize("kw", (dict(), dict(filament="2.85"), dict(count=3, filament="2.85")))
def test_every_hole_opens_to_the_edge_of_the_tab_through_a_slit(kw):
    """The slit is `SLIT` times the thread wide and runs from the hole to the edge of the tab."""
    s, m = numbers(**kw), part(**kw)
    p = frame(s)
    half = 0.5 * s["slit"]
    y = p["bottom"] + 2.0
    assert y < p["hole_y"] - 0.5 * s["hole"], "the slit was measured beside a hole"
    for x in p["holes"]:
        assert not plastic(m, p, x, p["bottom"] + 0.3), "the slit does not reach the edge"
        for side in (-1.0, 1.0):
            assert not plastic(m, p, x + side * (half - 0.25), y), "the slit is narrower than it is"
            assert plastic(m, p, x + side * (half + 0.25), y), "the slit is wider than it is"


# ------------------------------------------------------------------ printing

@pytest.mark.parametrize("kw", (dict(), dict(filament="2.85"), dict(count=2),
                                dict(flange=6.0, count=3, thickness=10.0),
                                dict(flange=1.0, thickness=4.0)))
def test_every_face_is_vertical_or_horizontal(kw):
    """The clip is an outline pulled straight up with vertical holes, so nothing overhangs and
    nothing needs support."""
    n, area = triangles(part(**kw))
    keep = area > 1e-9
    up = np.abs(n[keep][:, 2])
    assert np.all((up > 0.999) | (up < 1e-6)), "a face is tilted, so the clip overhangs"


# ------------------------------------------------------------------ the numbers

def test_sizes_reports_the_clip_it_makes():
    s = filament_clip.sizes(flange=4.0, filament="2.85", count=3, thickness=8.0)
    assert (s["flange"], s["filament"], s["count"], s["thickness"]) == (4.0, "2.85", 3, 8.0)
    assert s["hole"] == pytest.approx(3.25) and s["slit"] == pytest.approx(1.71)
    assert s["length"] == pytest.approx(2.0 * filament_clip.WALL + 4.0 + 24.0)
    assert s["width"] == pytest.approx(filament_clip.JAW + filament_clip.WALL)
    assert set(s) == {"flange", "filament", "count", "thickness", "hole", "slit", "length",
                      "width"}


def test_the_default_clip_holds_a_thin_thread():
    s = filament_clip.sizes()
    assert s["flange"] == 2.5 and s["filament"] == "1.75" and s["count"] == 1
    assert s["thickness"] == 5.0
    assert s["hole"] == pytest.approx(2.15) and s["slit"] == pytest.approx(1.05)


def test_the_numbers_are_clamped_to_what_prints():
    small = numbers(flange=0.0, count=0, thickness=0.0)
    assert small["flange"] == filament_clip.FLANGE_MIN and small["count"] == filament_clip.COUNT_MIN
    assert small["thickness"] == filament_clip.THICKNESS_MIN
    big = numbers(flange=999.0, count=999.0, thickness=999.0)
    assert big["flange"] == filament_clip.FLANGE_MAX and big["count"] == filament_clip.COUNT_MAX
    assert big["thickness"] == filament_clip.THICKNESS_MAX
    for kw in (dict(flange=0.0, count=0, thickness=0.0),
               dict(flange=999.0, count=999.0, thickness=999.0)):
        s, m = numbers(**kw), part(**kw)
        lo, hi = bbox(m)
        assert hi[2] - lo[2] == pytest.approx(s["thickness"], abs=ABS), "a clamped height was lost"
        assert hi[0] - lo[0] == pytest.approx(s["length"], abs=ABS), "a clamped length was lost"
        assert hi[1] - lo[1] == pytest.approx(s["width"], abs=ABS), "a clamped width was lost"


@pytest.mark.parametrize("filament", (None, "", "3.00", "1.75mm", 1.75))
def test_a_filament_that_is_no_filament_is_the_thin_one(filament):
    assert filament_clip.sizes(filament=filament)["filament"] == "1.75"
    assert filament_clip.filament_clip(filament=filament).volume() \
        == pytest.approx(part().volume(), rel=1e-9)


@pytest.mark.parametrize("bad", (float("nan"), float("inf"), None, "wide"))
def test_a_number_that_is_no_number_is_refused(bad):
    for key in ("flange", "count", "thickness"):
        with pytest.raises(ValueError):
            filament_clip.filament_clip(**dict(BASE, **{key: bad}))
        with pytest.raises(ValueError):
            filament_clip.sizes(**dict(BASE, **{key: bad}))


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
        dlg.flange.setValue(3.0)
        dlg.filament.setCurrentIndex(dlg.filament.findData("2.85"))
        dlg.count.setValue(2)
        dlg.thickness.setValue(6.0)
        assert "25.4" in dlg.info.text(), "the dialog does not show how long the clip is"
        assert "16.2" in dlg.info.text(), "the dialog does not show how wide the clip is"
        return QDialog.Accepted

    monkeypatch.setattr(app_module.QDialog, "exec", fill)
    try:
        w.filament_clip_dialog()
        deadline = time.monotonic() + 20.0
        while len(w.doc.groups) < 1 and time.monotonic() < deadline:
            app.processEvents()
            time.sleep(0.005)
        assert len(w.doc.groups) == 1, "no clip in the model"
        assert [g.name for g in w.doc.groups] == ["Filament clip"]
        s = filament_clip.sizes(flange=3.0, filament="2.85", count=2, thickness=6.0)
        lo, hi = np.asarray(w.doc.groups[0].bbox(), dtype=float)
        assert lo[2] == pytest.approx(0.0, abs=1e-6), "the clip does not lie on the bed"
        assert hi[0] - lo[0] == pytest.approx(s["length"], abs=ABS), "not as long as computed"
        assert hi[1] - lo[1] == pytest.approx(s["width"], abs=ABS), "not as wide as computed"
        assert hi[2] - lo[2] == pytest.approx(s["thickness"], abs=1e-3), "not as high as computed"
        settings = app_module.Settings()
        assert (settings.filament_clip_flange, settings.filament_clip_filament,
                settings.filament_clip_count, settings.filament_clip_thickness) == (
            3.0, "2.85", 2, 6.0)
        labels = [a.text().replace("&", "").replace("…", "") for a in w.flat_actions(w.menuBar())]
        assert labels.index("Filament Clip") == labels.index("Coaster") + 1
    finally:
        for key, value in (("filament_clip_flange", 2.5), ("filament_clip_filament", "1.75"),
                           ("filament_clip_count", 1), ("filament_clip_thickness", 5.0)):
            setattr(w.settings, key, value)
        w.settings.save()
        w.doc.dirty = False
        w.close()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))

"""Hex key holders: a block that stands a set of Allen keys upright and prints on its bottom.

`hex_key_holder.hex_key_holder` gives the holder as one closed volume on `z = 0`: a block with one
hexagonal hole per key in a row, smallest to the left, every hole the size of its key plus
`clearance` across the flats and `depth` deep, cut from the top face above it. The top steps down
towards the small keys when `steps` and every size engraves itself in front of its hole when
`labels`; the outside vertical edges round with `CORNER` millimetres.

These tests weigh the holder against that: one volume on the bed and centred; the block as long as
the row needs with its gap and its margin and its corners rounded; a hole for every key, smallest
left, as wide and as deep as the key plus its clearance and with a flat to the front, and plastic
between two holes; the sizes read out of the text - commas, spaces, junk, a size twice, sizes that
are no keys and a list that is too long; the steps lower towards the small keys while every hole
keeps its depth and the top flat without them; the sizes engraved in front of the holes and the
block still one volume; no face that looks down above the bed, so nothing needs support; every range
reached by clamping and every number that is no number refused; and the dialog that builds the
holder and remembers it.

Run with:  python -m pytest tests/test_hex_key_holder.py
"""
import math
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import manifold3d as mf  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from PySide6.QtWidgets import QApplication  # noqa: E402

from volura import geom, hex_key_holder  # noqa: E402

_app = QApplication.instance() or QApplication(sys.argv[:1])   # the engraved sizes need it

BASE = dict(sizes_mm="2 3 4 5 6", clearance=0.3, depth=12.0, gap=3.0, margin=4.0, steps=True,
            labels=True)
#: the round edges are polygons, so a part is at most this far off
ABS = 0.05
#: mm across a probe that feels for plastic
PROBE = 0.3


def part(**kw) -> mf.Manifold:
    return hex_key_holder.hex_key_holder(**dict(BASE, **kw))


def numbers(**kw) -> dict:
    return hex_key_holder.sizes(**dict(BASE, **kw))


def holes(s) -> list:
    """The holes of a holder: every key with the play of `clearance` all round it."""
    return [key + s["clearance"] for key in s["keys"]]


def centres(s) -> list:
    """The axes of the holes: smallest left, `gap` of plastic between two of them and the row
    centred on x = 0. A hexagon is 2 / sqrt(3) wider across its corners than across its flats, and
    its corners point along the row."""
    at, got = 0.0, []
    for across in holes(s):
        r = across / math.sqrt(3.0)
        got.append(at + r)
        at += 2.0 * r + s["gap"]
    return [x - 0.5 * (at - s["gap"]) for x in got]


def tops(s) -> list:
    """The height of the top face above every hole: the block steps down `STEP_DROP` of the depth
    in all, spread over the holes in proportion to their sizes, and the lowest step stands
    `FLOOR` above the bed."""
    fall = hex_key_holder.STEP_DROP * s["depth"] if s["steps"] and len(s["keys"]) > 1 else 0.0
    height = s["depth"] + hex_key_holder.FLOOR + fall
    big, small = s["keys"][-1], s["keys"][0]
    return [height - fall * (big - key) / (big - small) if fall > 0.0 else height
            for key in s["keys"]]


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


def post(x, y, z0, z1, r=PROBE) -> mf.Manifold:
    """A round post of radius `r` at (`x`, `y`), between the heights `z0` and `z1`."""
    return mf.Manifold.cylinder(z1 - z0, r, r, 32).translate((x, y, z0))


def hex_post(across, x, z0, z1) -> mf.Manifold:
    """A hexagonal post `across` millimetres across the flats with a flat to the front, on the axis
    through (`x`, 0), between the heights `z0` and `z1`: a key that is looking for its hole."""
    r = across / math.sqrt(3.0)
    cs = mf.CrossSection([[(r * math.cos(k * math.pi / 3.0), r * math.sin(k * math.pi / 3.0))
                           for k in range(6)]])
    return mf.Manifold.extrude(cs, z1 - z0).translate((x, 0.0, z0))


def slab(m, z0, z1) -> float:
    """The volume of the part between the heights `z0` and `z1`."""
    return (m ^ mf.Manifold.cube([1e4, 1e4, z1 - z0]).translate((-5e3, -5e3, z0))).volume()


def top_at(m, x, y) -> float:
    """The height of the part under a post at (`x`, `y`); 0.0 when there is nothing standing there."""
    hit = m ^ mf.Manifold.cylinder(1e3, PROBE, PROBE, 32).translate((x, y, -5e2))
    return 0.0 if hit.is_empty() else float(np.asarray(hit.bounding_box(), dtype=float)[5])


def back(s, i) -> float:
    """The y of the strip behind hole `i`, where nothing is engraved."""
    return 0.25 * (holes(s)[i] + s["width"])


def triangles(m):
    """The unit normals, the centres and the areas of all triangles, and the mesh itself."""
    v, t = geom.mesh_arrays(m)
    a, b, c = v[t[:, 0]], v[t[:, 1]], v[t[:, 2]]
    n = np.cross(b - a, c - a)
    area = 0.5 * np.linalg.norm(n, axis=1)
    return n / np.maximum(2.0 * area[:, None], 1e-30), (a + b + c) / 3.0, area, v, t


# ------------------------------------------------------------------ the piece

@pytest.mark.parametrize("kw", (dict(), dict(steps=False), dict(labels=False),
                                dict(sizes_mm="5"), dict(sizes_mm="0.7 19"),
                                dict(clearance=0.6, depth=30.0, gap=10.0, margin=10.0)))
def test_the_holder_is_one_volume_that_stands_on_the_bed_and_is_centred(kw):
    s, m = numbers(**kw), part(**kw)
    assert m.status() == mf.Error.NoError and not m.is_empty()
    assert len(m.decompose()) == 1, "the holder fell apart"
    lo, hi = bbox(m)
    assert lo[2] == pytest.approx(0.0, abs=1e-6), "the holder does not stand on the bed"
    assert np.allclose(lo[:2], -0.5 * np.array([s["length"], s["width"]]), atol=ABS), \
        "not centred in x and y"
    assert np.allclose(hi[:2], 0.5 * np.array([s["length"], s["width"]]), atol=ABS), \
        "not the size the row needs"
    assert hi[2] - lo[2] == pytest.approx(s["height"], abs=ABS), "not as high as computed"


def test_the_block_is_the_row_with_its_gap_and_its_margin():
    """As long as the holes with their plastic between them, as wide as the biggest hole with its
    margin twice, and its outside vertical edges rounded away."""
    s, m = numbers(), part()
    lo, hi = bbox(m)
    span = 2.0 * sum(h / math.sqrt(3.0) for h in holes(s)) + s["gap"] * (len(s["keys"]) - 1)
    assert s["length"] == pytest.approx(span + 2.0 * s["margin"]), "the row does not fit its block"
    assert s["width"] == pytest.approx(holes(s)[-1] + 2.0 * s["margin"]), "the wrong width"
    assert hi[0] - lo[0] == pytest.approx(s["length"], abs=ABS), "not as long as the row needs"
    assert hi[1] - lo[1] == pytest.approx(s["width"], abs=ABS), "not as wide as the biggest hole"
    half_l, half_w = 0.5 * s["length"], 0.5 * s["width"]
    for sx in (1.0, -1.0):
        for sy in (1.0, -1.0):
            assert not touches(m, post(sx * (half_l - 0.2), sy * (half_w - 0.2), 1.0,
                                       s["height"] - 1.0)), "the corner is not rounded away"
    assert touches(m, post(half_l - hex_key_holder.CORNER - 0.5, half_w - 0.2, 1.0,
                           s["height"] - 1.0)), "the flat beside the corner is gone"


# ------------------------------------------------------------------ the holes

def test_there_is_a_hole_for_every_key_and_the_smallest_stands_left():
    s, m = numbers(), part()
    cs, ts = centres(s), tops(s)
    assert len(cs) == len(s["keys"]), "a key has no hole"
    assert s["keys"] == sorted(s["keys"]), "the keys are not smallest first"
    assert cs == sorted(cs), "the holes are not in a row along x"
    for i, (x, key, top) in enumerate(zip(cs, s["keys"], ts)):
        assert not touches(m, hex_post(key, x, top - s["depth"] + 0.1, top - 0.1)), \
            "the %g mm key does not fit its hole" % key
        assert touches(m, hex_post(key + s["clearance"] + 0.5, x, top - s["depth"] + 0.1,
                                   top - 0.1)), "the hole is wider than the key plus its clearance"
        if i:
            mid = 0.5 * (cs[i - 1] + x)
            assert touches(m, post(mid, 0.0, ts[i - 1] - s["depth"] + 1.0, ts[i - 1] - 1.0)), \
                "there is no plastic between two holes"


def test_a_hole_is_as_deep_as_asked_and_stands_on_its_floor():
    """The bottom of a hole is `depth` under the top above it and the deepest one leaves `FLOOR`
    millimetres of plastic under it, so the short leg of a key goes in up to there."""
    for steps in (True, False):
        s, m = numbers(steps=steps), part(steps=steps)
        for x, key, top in zip(centres(s), s["keys"], tops(s)):
            assert not touches(m, hex_post(key, x, top - s["depth"] + 0.1,
                                           top - s["depth"] + 0.9)), "the hole is shallower than asked"
            assert touches(m, hex_post(key, x, top - s["depth"] - 0.4,
                                       top - s["depth"] + 0.05)), "the hole is deeper than asked"
        assert min(tops(s)) - s["depth"] == pytest.approx(hex_key_holder.FLOOR), \
            "the deepest hole does not stand on its floor"


def test_the_holes_are_hexagons_with_a_flat_to_the_front():
    """The wall a key lies against at the front is a flat of the hole across, not a corner: the key
    turns the same way every time and the top face in front of it lies flat for its size."""
    s, m = numbers(labels=False), part(labels=False)
    n, c, area, v, t = triangles(m)
    for x, across, top in zip(centres(s), holes(s), tops(s)):
        edge = across / math.sqrt(3.0)                # the flat of a hexagon is that wide
        wall = (area > 1e-9) & (n[:, 1] > 0.999) & (np.abs(c[:, 1] + 0.5 * across) < 0.02) \
               & (np.abs(c[:, 0] - x) < 0.5 * across)
        assert wall.any(), "there is no flat wall at the front of a hole"
        xs = v[t[wall]][:, :, 0]
        zs = v[t[wall]][:, :, 2]
        assert xs.min() == pytest.approx(x - 0.5 * edge, abs=ABS), "the front of the hole is a corner"
        assert xs.max() == pytest.approx(x + 0.5 * edge, abs=ABS), "the front of the hole is a corner"
        assert zs.min() == pytest.approx(top - s["depth"], abs=ABS), \
            "the wall of the hole does not run its full depth"
        assert zs.max() == pytest.approx(top, abs=ABS), \
            "the wall of the hole does not run its full depth"


# ------------------------------------------------------------------ the stepped top

def test_the_top_steps_down_towards_the_small_keys():
    """A small key is a short one, so its step is lower: every key sticks out as far as the others
    and every hole keeps its full depth below its own step."""
    s, m = numbers(), part()
    cs, ts = centres(s), tops(s)
    assert all(a < b for a, b in zip(ts, ts[1:])), "the top does not step down to the small keys"
    assert max(b - a for a, b in zip(ts, ts[1:])) <= 0.5 * s["depth"] + 1e-9, \
        "a step stands more than half a depth below the next"
    assert ts[0] == pytest.approx(s["height"] - hex_key_holder.STEP_DROP * s["depth"], abs=ABS)
    for i, (x, top) in enumerate(zip(cs, ts)):
        assert top_at(m, x, back(s, i)) == pytest.approx(top, abs=ABS), \
            "the step above a hole is not where it should be"


def test_without_steps_the_top_is_flat():
    s, m = numbers(steps=False), part(steps=False)
    assert s["height"] == pytest.approx(s["depth"] + hex_key_holder.FLOOR)
    for i, x in enumerate(centres(s)):
        assert top_at(m, x, back(s, i)) == pytest.approx(s["height"], abs=ABS), \
            "the top is not flat without the steps"


# ------------------------------------------------------------------ the sizes

def test_the_sizes_engrave_themselves_in_front_of_the_holes():
    from volura import fonts
    if not fonts.has_fonts():
        pytest.skip("this system has no fonts, so there are no letters to engrave")
    s = numbers()
    plain, made = part(labels=False), part()
    assert len(made.decompose()) == 1, "the engraving cut the holder in two"
    for x, across, top in zip(centres(s), holes(s), tops(s)):
        band = mf.Manifold.cube([across, 0.5 * (s["width"] - across), 0.5]) \
            .translate((x - 0.5 * across, -0.5 * s["width"], top - 0.55))
        assert (made ^ band).volume() < (plain ^ band).volume(), \
            "the size in front of a hole is not engraved"
    low = min(tops(s)) - hex_key_holder.ENGRAVE - 0.05
    assert slab(made, 0.1, low) == pytest.approx(slab(plain, 0.1, low), abs=1e-6), \
        "the engraving cuts deeper than it should"


# ------------------------------------------------------------------ printing

@pytest.mark.parametrize("kw", (dict(), dict(steps=False), dict(labels=False),
                                dict(sizes_mm="0.7", clearance=0.1, gap=2.0, margin=3.0),
                                dict(sizes_mm="1 2 3 4 5 6 7 8 9 10 11 12 13 14 15", depth=30.0)))
def test_nothing_looks_down_above_the_bed(kw):
    """The holder prints on its bottom: the only faces that look down are the ones that lie on the
    bed, so nothing hangs over and nothing needs support."""
    n, c, area, _v, _t = triangles(part(**kw))
    keep = area > 1e-9
    assert (n[keep][c[keep][:, 2] < 1e-9][:, 2] < -0.99).all(), "the triangles do not face out"
    down = keep & (n[:, 2] < -1e-6) & (c[:, 2] > 0.1)
    assert not down.any(), "a face looks down above the bed, so the holder needs support"


# ------------------------------------------------------------------ the numbers

def test_the_key_sizes_are_read_out_of_the_text():
    for text in ("1.5 2 2.5", "1.5, 2,2.5", "  2.5 ,  2  ,1.5  ", "1.5 mm, 2 mm and 2.5 mm"):
        assert hex_key_holder.sizes(sizes_mm=text)["keys"] == [1.5, 2.0, 2.5], text
    assert hex_key_holder.sizes(sizes_mm="5 3 1")["keys"] == [1.0, 3.0, 5.0], "not sorted"
    assert hex_key_holder.sizes(sizes_mm="2 2 2")["keys"] == [2.0], "a size twice is two holes"
    assert hex_key_holder.sizes(sizes_mm="0.1 0.7 19 19.5")["keys"] == [0.7, 19.0], \
        "a size that is no hex key was kept"


def test_a_long_list_keeps_the_fifteen_smallest_keys():
    keys = hex_key_holder.sizes(sizes_mm=" ".join(f"{0.5 * i:g}" for i in range(1, 40)))["keys"]
    assert len(keys) == hex_key_holder.COUNT_MAX, "more keys than a holder holds"
    assert keys == sorted(keys) and keys[0] == 1.0 and keys[-1] == 8.0, "not the smallest fifteen"


@pytest.mark.parametrize("bad", ("", "   ", "no keys here", None, "0.5 20", float("nan")))
def test_a_text_without_a_key_size_is_refused(bad):
    with pytest.raises(ValueError):
        hex_key_holder.sizes(sizes_mm=bad)
    with pytest.raises(ValueError):
        hex_key_holder.hex_key_holder(sizes_mm=bad)


def test_sizes_reports_the_keys_and_the_block():
    s = hex_key_holder.sizes(sizes_mm="3, 6, 10", clearance=0.4, depth=20.0, gap=5.0, margin=6.0,
                             steps=True, labels=True)
    assert s["keys"] == [3.0, 6.0, 10.0]
    assert s["clearance"] == 0.4 and s["depth"] == 20.0 and s["gap"] == 5.0
    assert s["margin"] == 6.0 and s["steps"] is True and s["labels"] is True
    assert s["width"] == pytest.approx(10.4 + 2.0 * 6.0)
    assert s["height"] == pytest.approx(20.0 + hex_key_holder.FLOOR + 10.0)
    assert set(s) == {"keys", "clearance", "depth", "gap", "margin", "steps", "labels", "length",
                      "width", "height"}


def test_the_default_holder_holds_the_common_set():
    assert hex_key_holder.sizes()["keys"] == [1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0]


def test_the_numbers_are_clamped_to_what_prints():
    small = numbers(clearance=0.0, depth=0.0, gap=0.0, margin=0.0)
    assert small["clearance"] == hex_key_holder.CLEARANCE_MIN
    assert small["depth"] == hex_key_holder.DEPTH_MIN
    assert small["gap"] == hex_key_holder.GAP_MIN
    assert small["margin"] == hex_key_holder.MARGIN_MIN
    big = numbers(clearance=999.0, depth=999.0, gap=999.0, margin=999.0)
    assert big["clearance"] == hex_key_holder.CLEARANCE_MAX
    assert big["depth"] == hex_key_holder.DEPTH_MAX
    assert big["gap"] == hex_key_holder.GAP_MAX
    assert big["margin"] == hex_key_holder.MARGIN_MAX
    for kw in (dict(clearance=0.0, depth=0.0, gap=0.0, margin=0.0),
               dict(clearance=999.0, depth=999.0, gap=999.0, margin=999.0)):
        s, m = numbers(**kw), part(**kw)
        lo, hi = bbox(m)
        assert hi[2] - lo[2] == pytest.approx(s["height"], abs=ABS), "the clamped numbers were ignored"
        assert not touches(m, post(0.5 * s["length"] - 0.2, 0.5 * s["width"] - 0.2, 1.0,
                                   s["height"] - 1.0)), "a clamped block lost its rounded corner"


@pytest.mark.parametrize("bad", (float("nan"), float("inf"), None, "wide"))
def test_a_number_that_is_no_number_is_refused(bad):
    for key in ("clearance", "depth", "gap", "margin"):
        with pytest.raises(ValueError):
            hex_key_holder.hex_key_holder(**dict(BASE, **{key: bad}))
        with pytest.raises(ValueError):
            hex_key_holder.sizes(**dict(BASE, **{key: bad}))


# ------------------------------------------------------------------ the interface

def test_the_dialog_builds_a_holder_and_remembers_it(monkeypatch, tmp_path):
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
        dlg.sizes.setText("2 3 4 5")
        dlg.clearance.setValue(0.4)
        dlg.depth.setValue(15.0)
        dlg.gap.setValue(4.0)
        dlg.margin.setValue(5.0)
        assert dlg.steps.isChecked(), "the stepped top is off by default"
        assert dlg.labels.isChecked(), "the sizes are off by default"
        dlg.labels.setChecked(False)
        assert "40.0" in dlg.info.text(), "the dialog does not show the block it makes"
        assert "25.5" in dlg.info.text(), "the dialog does not show how high the block stands"
        return QDialog.Accepted

    monkeypatch.setattr(app_module.QDialog, "exec", fill)
    try:
        w.hex_key_holder_dialog()
        deadline = time.monotonic() + 20.0
        while len(w.doc.groups) < 1 and time.monotonic() < deadline:
            app.processEvents()
            time.sleep(0.005)
        assert len(w.doc.groups) == 1, "no holder in the model"
        assert [g.name for g in w.doc.groups] == ["Hex key holder"]
        s = hex_key_holder.sizes(sizes_mm="2 3 4 5", clearance=0.4, depth=15.0, gap=4.0,
                                 margin=5.0, labels=False)
        lo, hi = np.asarray(w.doc.groups[0].bbox(), dtype=float)
        assert lo[2] == pytest.approx(0.0, abs=1e-6), "the holder does not stand on the bed"
        assert hi[0] - lo[0] == pytest.approx(s["length"], abs=1e-3), "not as long as computed"
        assert hi[1] - lo[1] == pytest.approx(s["width"], abs=1e-3), "not as wide as computed"
        assert hi[2] - lo[2] == pytest.approx(s["height"], abs=1e-3), "not as high as computed"
        settings = app_module.Settings()
        assert (settings.hex_key_holder_sizes, settings.hex_key_holder_clearance,
                settings.hex_key_holder_depth, settings.hex_key_holder_gap,
                settings.hex_key_holder_margin, settings.hex_key_holder_steps,
                settings.hex_key_holder_labels) == ("2 3 4 5", 0.4, 15.0, 4.0, 5.0, True, False)
        labels = [a.text().replace("&", "").replace("…", "") for a in w.flat_actions(w.menuBar())]
        assert labels.index("Hex Key Holder") == labels.index("Lithophane") + 1
    finally:
        for key, value in (("hex_key_holder_sizes", hex_key_holder.SIZES_TEXT),
                           ("hex_key_holder_clearance", 0.3), ("hex_key_holder_depth", 12.0),
                           ("hex_key_holder_gap", 3.0), ("hex_key_holder_margin", 4.0),
                           ("hex_key_holder_steps", True), ("hex_key_holder_labels", True)):
            setattr(w.settings, key, value)
        w.settings.save()
        w.doc.dirty = False
        w.close()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))

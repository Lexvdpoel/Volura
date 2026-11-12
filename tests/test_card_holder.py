"""Memory card holders: a block that stands SD cards, microSD cards and USB sticks upright.

`card_holder.card_holder` gives the holder as one closed volume that lies on `z = 0` and is centred
on the origin in plan: a block with a column of slots for every kind of card that was asked for,
every slot the size of its card with play all round, open at the top and chamfered at its mouth, and
with a half-round dip in the top in front of every column for a fingertip.

These tests weigh the part against that: it is one piece that lies on the bed and is as big as its
block, a card of every kind stands in every slot of its kind without touching anything and a card
that is thicker does, every slot keeps a floor of a millimetre and a half above the bed, a microSD
slot stays shallow so the card sticks out, a kind without cards leaves no slot behind, the dip in
front of a column is there and no wider or deeper than it is asked to be, and nothing hangs over
steeper than 45 degrees, because the holder prints without support.

Run with:  python -m pytest tests/test_card_holder.py
"""
import math
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import manifold3d as mf  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from volura import card_holder, geom  # noqa: E402

BASE = dict(sd=4, micro=6, usb=0, rows=1, fit=0.3, depth=12.0)
#: the rounded corners, the chamfers and the notch are polygons, so a part is at most this far off
ABS = 0.05


def part(**kw) -> mf.Manifold:
    return card_holder.card_holder(**dict(BASE, **kw))


def numbers(**kw) -> dict:
    return card_holder.sizes(**dict(BASE, **kw))


def bbox(m):
    """The bounding box of a manifold as (low, high)."""
    b = np.asarray(m.bounding_box(), dtype=float)
    return b[:3], b[3:]


def box(w, d, h, x, y, z) -> mf.Manifold:
    """A box `w` by `d` and `h` high, centred on (`x`, `y`) in plan and standing on `z`."""
    return mf.Manifold.cube([w, d, h]).translate((x - 0.5 * w, y - 0.5 * d, z))


def kinds(s) -> list:
    """The columns the numbers describe: where each of them stands and how big its slot is.

    The kinds follow each other along X with a wall of `GAP` between two of them, the whole of it
    centred on the origin, and a microSD slot is never deeper than `MICRO_DEPTH`.
    """
    out, x = [], 0.0
    for name, w, t, pitch in card_holder.KINDS:
        n = s[name]
        if not n:
            continue
        sx, sy = w + 2.0 * s["fit"], t + 2.0 * s["fit"]
        out.append({"name": name, "count": n, "sx": sx, "sy": sy, "pitch": pitch,
                    "depth": min(s["depth"], card_holder.MICRO_DEPTH) if name == "micro"
                    else s["depth"], "x": x + 0.5 * sx})
        x += sx + card_holder.GAP
    for k in out:
        k["x"] -= 0.5 * (x - card_holder.GAP)
    return out


def slots(s) -> list:
    """Every slot of the holder: its place, the size of its slot and how deep it is.

    The first slot of a column stands with its front edge a wall behind the front of the block, the
    others of that column follow at its pitch and a second row begins where the longest column of
    the first ends, plus a wall.
    """
    reach = max(k["pitch"] * (k["count"] - 1) + k["sy"] for k in kinds(s))
    step = reach + card_holder.GAP
    out = []
    for k in kinds(s):
        for r in range(s["rows"]):
            for i in range(k["count"]):
                out.append({"name": k["name"], "x": k["x"], "sx": k["sx"], "sy": k["sy"],
                            "depth": k["depth"],
                            "y": y_edge(s) + 0.5 * k["sy"] + r * step + i * k["pitch"]})
    return out


def y_edge(s) -> float:
    """The front edge of the first slot of every column: a wall behind the front of the block."""
    return -0.5 * s["length"] + card_holder.WALL


def card(s, k) -> mf.Manifold:
    """The card the slot `k` is made for, standing on the floor of that slot and two millimetres
    out of its mouth."""
    return box(k["sx"] - 2.0 * s["fit"], k["sy"] - 2.0 * s["fit"], k["depth"] + 2.0,
               k["x"], k["y"], s["height"] - k["depth"])


def too_thick(s, k) -> mf.Manifold:
    """A card 0.3 mm thicker than the slot `k`, so it cannot go down into it."""
    return box(k["sx"] + 0.3, k["sy"] + 0.3, k["depth"] + 2.0, k["x"], k["y"],
               s["height"] - k["depth"])


def touches(m, probe) -> bool:
    """Whether a probe runs into the part anywhere.

    Two solids that only touch leave a sliver of a few ten-millionths of a mm³ behind, so that is
    not a touch; a card that does not fit is off by whole millimetres.
    """
    return (m ^ probe).volume() > 1e-6


def triangles(m):
    """The unit normals, the centres and the areas of all triangles, pointing out of the part."""
    v, t = geom.mesh_arrays(m)
    a, b, c = v[t[:, 0]], v[t[:, 1]], v[t[:, 2]]
    n = np.cross(b - a, c - a)
    area = 0.5 * np.linalg.norm(n, axis=1)
    return n / np.maximum(2.0 * area[:, None], 1e-30), (a + b + c) / 3.0, area


# ------------------------------------------------------------------ the part

@pytest.mark.parametrize("sd, micro, usb", ((0, 1, 0), (1, 0, 0), (0, 0, 1), (4, 6, 0),
                                            (12, 12, 12)))
@pytest.mark.parametrize("rows", (1, 2))
def test_the_holder_is_one_piece_that_lies_on_the_bed(sd, micro, usb, rows):
    kw = dict(sd=sd, micro=micro, usb=usb, rows=rows)
    s, m = numbers(**kw), part(**kw)
    assert m.status() == mf.Error.NoError and not m.is_empty()
    assert len(m.decompose()) == 1, "the holder fell apart"
    lo, hi = bbox(m)
    assert np.allclose(lo, (-0.5 * s["width"], -0.5 * s["length"], 0.0), atol=ABS), \
        "the block is not centred on the origin, or does not lie on the bed"
    assert np.allclose(hi, (0.5 * s["width"], 0.5 * s["length"], s["height"]), atol=ABS), \
        "the block is not as big as sizes says"


@pytest.mark.parametrize("fit, depth", ((0.1, 6.0), (0.3, 12.0), (0.6, 20.0)))
def test_the_holder_is_valid_whatever_it_is_asked_for(fit, depth):
    m = part(sd=8, micro=3, usb=2, rows=2, fit=fit, depth=depth)
    assert m.status() == mf.Error.NoError and not m.is_empty()
    assert len(m.decompose()) == 1, "the holder fell apart"


@pytest.mark.parametrize("rows", (1, 2))
def test_the_block_is_a_box_with_rounded_corners(rows):
    """Below the slots the block is a rectangle with corners of CORNER, over the whole
    footprint."""
    s, m = numbers(rows=rows), part(rows=rows)
    want = s["width"] * s["length"] - (4.0 - math.pi) * card_holder.CORNER ** 2
    assert m.slice(0.5).area() == pytest.approx(want, rel=0.01), "the corners are not rounded"


# ------------------------------------------------------------------ the slots

@pytest.mark.parametrize("fit", (0.1, 0.3, 0.6))
@pytest.mark.parametrize("depth", (6.0, 12.0, 20.0))
def test_a_card_stands_in_every_slot(fit, depth):
    kw = dict(fit=fit, depth=depth, rows=2)
    s, m = numbers(**kw), part(**kw)
    for k in slots(s):
        assert not touches(m, card(s, k)), f"a {k['name']} card does not stand in its slot"
        assert touches(m, too_thick(s, k)), \
            f"the {k['name']} slot is wider than the card with its play"


@pytest.mark.parametrize("depth", (6.0, 12.0, 20.0))
def test_a_microsd_slot_stays_shallow(depth):
    """A microSD card is only 15 mm tall: its slot is never deeper than MICRO_DEPTH, so the card
    keeps standing out of the block far enough to be grabbed."""
    s = numbers(depth=depth)
    k = next(k for k in slots(s) if k["name"] == "micro")
    assert k["depth"] == pytest.approx(min(depth, card_holder.MICRO_DEPTH))
    assert 15.0 - k["depth"] >= 7.0, "the card does not stand out of the block"


@pytest.mark.parametrize("fit, depth", ((0.1, 6.0), (0.3, 12.0), (0.6, 20.0)))
def test_the_floor_under_every_slot_is_at_least_the_floor(fit, depth):
    kw = dict(fit=fit, depth=depth, rows=2)
    s, m = numbers(**kw), part(**kw)
    for k in slots(s):
        low = s["height"] - k["depth"]
        assert low >= card_holder.FLOOR - 1e-9, "a slot has too little floor under it"
        floor = box(0.6, 0.6, low - 0.04, k["x"], k["y"], 0.02)
        assert (m ^ floor).volume() == pytest.approx(floor.volume(), rel=1e-3), \
            "there is nothing solid under the floor of the slot"


def test_the_mouth_of_a_slot_is_chamfered():
    """The lead-in is a chamfer of MOUTH at 45 degrees: it is gone MOUTH down the slot."""
    s, m = numbers(), part()
    k = slots(s)[0]
    shallow = box(k["sx"] + 0.4, k["sy"] + 0.4, 0.5, k["x"], k["y"], s["height"] - 0.05)
    deep = box(k["sx"] + 0.4, k["sy"] + 0.4, 0.5, k["x"], k["y"], s["height"] - 0.5)
    assert not touches(m, shallow), "the mouth has no chamfer for a card to find"
    assert touches(m, deep), "the chamfer runs deeper down the slot than it is wide"


def test_a_kind_that_is_left_out_makes_the_block_narrower():
    """A kind without cards has no column, so the block is only as wide as the columns that are
    there: where the microSD column would have stood is one wall."""
    kw = dict(sd=3, micro=0, usb=0)
    s, m = numbers(**kw), part(**kw)
    assert len(slots(s)) == 3
    assert s["width"] == pytest.approx(card_holder.SD_W + 2.0 * s["fit"] + 2.0 * card_holder.WALL,
                                       abs=ABS)
    assert s["width"] < numbers(sd=3, micro=3)["width"], "the block did not get narrower"
    wall = box(1.0, 1.0, 1.0, 0.5 * s["width"] - 1.5, 0.0, s["height"] - 1.3)
    assert touches(m, wall), "there is a slot where no card was asked for"


# ------------------------------------------------------------------ the finger notch

@pytest.mark.parametrize("rows", (1, 2))
def test_the_front_of_a_column_has_a_finger_notch(rows):
    """In front of every column the top dips NOTCH_DEPTH millimetres down over NOTCH millimetres
    of its width: a fingertip comes down there and pinches the front card out."""
    kw = dict(rows=rows)
    s, m = numbers(**kw), part(**kw)
    for k in kinds(s):
        y = y_edge(s) - 1.5                        # in the wall in front of the first card
        dip = box(2.0, 1.0, 1.0, k["x"], y, s["height"] - 0.9)
        assert not touches(m, dip), f"the top does not dip in front of the {k['name']} column"
        under = box(2.0, 1.0, 1.0, k["x"], y, s["height"] - card_holder.NOTCH_DEPTH - 0.9)
        assert touches(m, under), "the dip is deeper than it is asked to be"
        side = box(2.0, 1.0, 1.0, k["x"] + 0.5 * card_holder.NOTCH + 1.0, y, s["height"] - 0.9)
        assert touches(m, side), "the dip is wider than it is asked to be"


def test_the_notch_stops_at_the_card_it_belongs_to():
    """The dip runs from the front of the block to the first slot and no further: the wall between
    the first two slots is as tall as the top of the block."""
    s, m = numbers(sd=3), part(sd=3)
    k = next(k for k in slots(s) if k["name"] == "sd")
    y = k["y"] + 0.5 * k["sy"] + 0.5 * card_holder.GAP
    wall = box(2.0, 1.0, 1.0, k["x"], y, s["height"] - 0.9)
    assert touches(m, wall), "the dip runs into the column behind the first card"


# ------------------------------------------------------------------ printing

@pytest.mark.parametrize("rows", (1, 2))
@pytest.mark.parametrize("fit, depth", ((0.1, 6.0), (0.6, 20.0)))
def test_nothing_hangs_over_steeper_than_45_degrees(rows, fit, depth):
    """The holder lies on its own block: the walls of a slot are vertical, their floors look up and
    the dip in front of a column is a hollow in the top, so no face hangs over steeper than
    45 degrees."""
    m = part(sd=6, micro=4, usb=2, rows=rows, fit=fit, depth=depth)
    n, c, area = triangles(m)
    keep = area > 1e-9
    assert (n[keep][c[keep][:, 2] < 1e-9][:, 2] < -0.99).all(), \
        "the triangles do not face out of the part"
    down = keep & (n[:, 2] < -np.sin(np.pi / 4.0) - 0.01) & (c[:, 2] > 1e-6)
    assert not down.any(), "a face overhangs steeper than 45 degrees, so it needs support"


# ------------------------------------------------------------------ the numbers

def test_sizes_reports_the_block_and_its_cards():
    s = card_holder.sizes(sd=2, micro=3, usb=1, rows=2, fit=0.2, depth=14.0)
    assert s["sd"] == 2 and s["micro"] == 3 and s["usb"] == 1 and s["rows"] == 2
    assert s["fit"] == 0.2 and s["depth"] == 14.0
    assert s["width"] == pytest.approx(24.4 + 11.4 + 12.4 + 2.0 * card_holder.GAP
                                       + 2.0 * card_holder.WALL)
    reach = max(4.5 * 1 + 2.5, 4.5 * 2 + 1.1, 8.0 * 0 + 4.9)
    assert s["length"] == pytest.approx(2.0 * reach + card_holder.GAP + 2.0 * card_holder.WALL)
    assert s["height"] == pytest.approx(card_holder.FLOOR + 14.0)
    assert set(s) == {"sd", "micro", "usb", "rows", "fit", "depth", "width", "length", "height"}


def test_the_block_grows_with_every_column_and_row():
    """A column adds the width of its slots and a wall, a row the reach of the longest column and
    a wall."""
    one = numbers(sd=1, micro=0, usb=0, rows=1)
    assert numbers(sd=1, micro=1, usb=0, rows=1)["width"] - one["width"] == pytest.approx(
        card_holder.MICRO_W + 2.0 * one["fit"] + card_holder.GAP)
    assert numbers(sd=1, micro=0, usb=1, rows=1)["width"] - one["width"] == pytest.approx(
        card_holder.USB_W + 2.0 * one["fit"] + card_holder.GAP)
    assert numbers(sd=1, micro=0, usb=0, rows=2)["length"] - one["length"] == pytest.approx(
        card_holder.SD_T + 2.0 * one["fit"] + card_holder.GAP)
    assert numbers(sd=2, micro=0, usb=0)["length"] - one["length"] == pytest.approx(
        card_holder.SD_PITCH)


def test_the_numbers_are_clamped_to_what_prints():
    small = numbers(sd=-3, micro=1, usb=-1, rows=0, fit=0.0, depth=1.0)
    assert small["sd"] == card_holder.SD_MIN and small["usb"] == card_holder.USB_MIN
    assert small["micro"] == 1 and small["rows"] == card_holder.ROWS_MIN
    assert small["fit"] == card_holder.FIT_MIN and small["depth"] == card_holder.DEPTH_MIN
    big = numbers(sd=99, micro=99, usb=99, rows=9, fit=9.0, depth=99.0)
    assert big["sd"] == card_holder.SD_MAX and big["micro"] == card_holder.MICRO_MAX
    assert big["usb"] == card_holder.USB_MAX and big["rows"] == card_holder.ROWS_MAX
    assert big["fit"] == card_holder.FIT_MAX and big["depth"] == card_holder.DEPTH_MAX
    lo, hi = bbox(part(sd=-3, micro=1, usb=-1, rows=0, fit=0.0, depth=1.0))
    assert hi[0] - lo[0] == pytest.approx(card_holder.MICRO_W + 2.0 * card_holder.FIT_MIN
                                          + 2.0 * card_holder.WALL, abs=ABS), \
        "the smallest holder is not one microSD slot with a wall around it"


@pytest.mark.parametrize("bad", (float("nan"), float("inf"), None, "wide"))
def test_a_number_that_is_no_number_is_refused(bad):
    for key in ("sd", "micro", "usb", "rows", "fit", "depth"):
        with pytest.raises(ValueError):
            card_holder.card_holder(**dict(BASE, **{key: bad}))
        with pytest.raises(ValueError):
            card_holder.sizes(**dict(BASE, **{key: bad}))


@pytest.mark.parametrize("kw", (dict(sd=0, micro=0, usb=0), dict(sd=-2, micro=0, usb=0)))
def test_a_holder_without_a_single_slot_is_refused(kw):
    for name in (card_holder.card_holder, card_holder.sizes):
        with pytest.raises(ValueError):
            name(**dict(BASE, **kw))


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
        dlg.sd.setValue(3)
        dlg.micro.setValue(2)
        dlg.usb.setValue(1)
        dlg.rows.setValue(2)
        dlg.depth.setValue(14.0)
        assert "3 SD" in dlg.info.text(), "the dialog does not show the slots it makes"
        return QDialog.Accepted

    monkeypatch.setattr(app_module.QDialog, "exec", fill)
    try:
        w.card_holder_dialog()
        deadline = time.monotonic() + 15.0
        while not w.doc.groups and time.monotonic() < deadline:
            app.processEvents()
            time.sleep(0.005)
        assert len(w.doc.groups) == 1, "no holder in the model"
        lo, hi = np.asarray(w.doc.groups[0].bbox(), dtype=float)
        s = card_holder.sizes(sd=3, micro=2, usb=1, rows=2, depth=14.0)
        assert lo[2] == pytest.approx(0.0, abs=1e-6), "the holder does not lie on the bed"
        assert hi[2] - lo[2] == pytest.approx(s["height"], abs=1e-4)
        assert hi[0] - lo[0] == pytest.approx(s["width"], abs=0.2), "the block is not as wide"
        assert hi[1] - lo[1] == pytest.approx(s["length"], abs=0.2), "the block is not as long"
        settings = app_module.Settings()
        assert (settings.card_holder_sd, settings.card_holder_micro, settings.card_holder_usb,
                settings.card_holder_rows, settings.card_holder_depth) == (
            3, 2, 1, 2, pytest.approx(14.0))
        labels = [a.text().replace("&", "").replace("…", "") for a in w.flat_actions(w.menuBar())]
        assert labels.index("Memory Card Holder") == labels.index("Bit Holder") + 1
    finally:
        w.settings.card_holder_sd, w.settings.card_holder_micro = 4, 6
        w.settings.card_holder_usb, w.settings.card_holder_rows = 0, 1
        w.settings.card_holder_depth = 12.0
        w.settings.save()
        w.doc.dirty = False
        w.close()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))

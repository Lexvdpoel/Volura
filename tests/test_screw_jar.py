"""Screw-top jars: a round jar with an outside thread and the lid that screws onto it.

`screw_jar.screw_jar` gives the jar and its lid as closed volumes that stand side by side on
`z = 0`: the jar on its floor, `diameter` wide and `height` deep inside, with the thread of
`threads.thread_solid` round its top, and the lid on its top, upside down, with the matching
inside thread `clearance` larger and `grip` ribs round it.

These tests weigh the pair against that: one volume each, standing on the bed, `GAP` apart and
centred together; a jar that is hollow and `diameter` wide up to its rim and as wide as the thread
over its crests; a lid that screws onto the jar without running into it, and that would with no
play; the ribs on the lid; nothing that faces straight down above the bed; every range reached by
clamping and a number that is no number refused; and the dialog, which builds the two as two
groups and remembers what it was given.

Run with:  python -m pytest tests/test_screw_jar.py
"""
import math
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import manifold3d as mf  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from volura import screw_jar, threads  # noqa: E402

BASE = dict(diameter=40.0, height=30.0, wall=2.0, pitch=3.0, thread_len=10.0, clearance=0.3,
            grip=24)
#: mm across a probe that feels for plastic or for open space
PROBE = 0.4


def parts(**kw) -> list:
    return screw_jar.screw_jar(**dict(BASE, **kw))


def numbers(**kw) -> dict:
    return screw_jar.sizes(**dict(BASE, **kw))


def bbox(m):
    b = np.asarray(m.bounding_box(), dtype=float)
    return b[:3], b[3:]


def touches(m, probe) -> bool:
    """Whether a probe runs into the part; a sliver of a few ten-millionths of a mm³ is no touch."""
    return (m ^ probe).volume() > 1e-6


def centred(m) -> mf.Manifold:
    """A piece moved so its axis is the z axis, the way it was built."""
    lo, hi = bbox(m)
    return m.translate((-0.5 * (lo[0] + hi[0]), -0.5 * (lo[1] + hi[1]), 0.0))


def tube(r0, r1, z0, z1) -> mf.Manifold:
    """A ring between the radii `r0` and `r1` and the heights `z0` and `z1`."""
    outer = mf.Manifold.cylinder(z1 - z0, r1, r1, 128)
    inner = mf.Manifold.cylinder(z1 - z0 + 2.0, r0, r0, 128).translate((0.0, 0.0, -1.0))
    return (outer - inner).translate((0.0, 0.0, z0))


# ------------------------------------------------------------------ the pieces

@pytest.mark.parametrize("kw", (dict(), dict(grip=0), dict(diameter=20.0, height=15.0, pitch=5.0),
                                dict(diameter=80.0, height=60.0, pitch=2.0, thread_len=20.0)))
def test_two_pieces_each_one_volume_on_the_bed_and_centred(kw):
    made = parts(**kw)
    assert len(made) == 2, "not a jar and a lid"
    for m in made:
        assert m.status() == mf.Error.NoError and not m.is_empty()
        assert len(m.decompose()) == 1, "a piece fell apart"
        lo, _hi = bbox(m)
        assert lo[2] == pytest.approx(0.0, abs=1e-6), "a piece does not stand on the bed"
    (lo0, hi0), (lo1, hi1) = bbox(made[0]), bbox(made[1])
    assert lo1[0] - hi0[0] == pytest.approx(screw_jar.GAP, abs=1e-6), "not GAP apart"
    assert lo0[0] + hi1[0] == pytest.approx(0.0, abs=1e-6), "the pair is not centred in x"
    for lo, hi in ((lo0, hi0), (lo1, hi1)):
        assert lo[1] + hi[1] == pytest.approx(0.0, abs=1e-6), "a piece is not centred in y"


def test_the_jar_is_as_high_as_its_floor_and_inside_and_the_lid_as_its_top_and_thread():
    s = numbers()
    jar, lid = parts()
    assert bbox(jar)[1][2] == pytest.approx(screw_jar.FLOOR + s["height"], abs=1e-6)
    assert bbox(lid)[1][2] == pytest.approx(screw_jar.LID_TOP + s["thread_len"], abs=1e-6)
    assert s["closed_height"] == pytest.approx(screw_jar.FLOOR + s["height"] + screw_jar.LID_TOP)


# ------------------------------------------------------------------ the jar

def test_the_jar_is_hollow_and_as_wide_as_asked_up_to_its_rim():
    s = numbers()
    jar = centred(parts()[0])
    r = 0.5 * s["diameter"]
    rim = screw_jar.FLOOR + s["height"]
    inside = mf.Manifold.cylinder(rim + 1.0 - screw_jar.FLOOR - 0.05, r - 0.1, r - 0.1, 128) \
        .translate((0.0, 0.0, screw_jar.FLOOR + 0.05))
    assert not touches(jar, inside), "the inside is narrower than asked somewhere"
    assert touches(jar, tube(r + 0.1, r + 0.3, screw_jar.FLOOR + 1.0, rim - 1.0)), \
        "there is no wall round the inside"
    floor = mf.Manifold.cylinder(screw_jar.FLOOR - 0.2, r - 1.0, r - 1.0, 64) \
        .translate((0.0, 0.0, 0.1))
    assert (jar ^ floor).volume() == pytest.approx(floor.volume(), rel=1e-3), "no floor"


def test_the_thread_stands_out_of_the_wall_by_its_depth():
    s = numbers()
    jar = centred(parts()[0])
    lo, hi = bbox(jar)
    assert hi[0] - lo[0] == pytest.approx(s["jar_outside"], abs=0.05), "not the crest diameter"
    outer = 0.5 * s["diameter"] + s["wall"]
    assert s["jar_outside"] == pytest.approx(2.0 * (outer + screw_jar.THREAD_DEPTH * s["pitch"]))
    below = screw_jar.FLOOR + s["height"] - s["thread_len"] - 1.0
    assert not touches(jar, tube(outer + 0.05, outer + 3.0, 0.0, below)), \
        "the cup is wider than its wall below the thread"
    mid = screw_jar.FLOOR + s["height"] - 0.5 * s["thread_len"]
    assert touches(jar, tube(outer + 0.2, outer + 0.6, mid - 0.5, mid + 0.5)), \
        "no thread round the top"


# ------------------------------------------------------------------ the fit

def lid_on_jar(jar, lid, s, turn):
    """The lid turned back over, as it is used, screwed down on the jar and turned by `turn`."""
    rim = screw_jar.FLOOR + s["height"]
    used = lid.rotate((180.0, 0.0, 0.0))                  # a turn over keeps the thread's hand
    lo, hi = bbox(used)
    used = used.translate((0.0, 0.0, rim + screw_jar.LID_TOP - hi[2]))
    return used.rotate((0.0, 0.0, turn))


def overlap(**kw) -> float:
    """The least the lid runs into the jar over a turn of the lid, when it sits on the rim."""
    s = numbers(**kw)
    jar, lid = (centred(m) for m in parts(**kw))
    return min((jar ^ lid_on_jar(jar, lid, s, t)).volume() for t in range(0, 360, 10))


def test_the_lid_screws_down_onto_the_jar_without_running_into_it():
    assert overlap() < 0.5, "the lid's thread runs into the jar's however it is turned"


def test_the_threads_of_jar_and_lid_leave_the_clearance_between_them():
    """The two thread solids themselves: the cut of the lid holds the jar's thread with room,
    and without the play, shifted by a quarter turn, the two run into each other."""
    d, p, L = 50.0, 3.0, 10.0
    outer = threads.thread_solid(d, p, L, segments=96)
    for c in (0.3, 0.1):
        skirt = mf.Manifold.cylinder(L, 0.5 * d + c + 2.0, 0.5 * d + c + 2.0, 96) \
            - threads.thread_solid(d, p, L, internal=True, clearance=c, segments=96)
        assert (outer ^ skirt).volume() < 0.05, f"no room with a clearance of {c} mm"
    tight = mf.Manifold.cylinder(L, 0.5 * d + 2.0, 0.5 * d + 2.0, 96) \
        - threads.thread_solid(d, p, L, internal=True, clearance=0.0, segments=96) \
        .translate((0.0, 0.0, 0.25 * p))
    assert (outer ^ tight).volume() > 1.0, "the threads do not hold each other"


# ------------------------------------------------------------------ the lid

def bumps(m, radius, z):
    """How many separate pieces of plastic a thin ring at `radius` and height `z` meets."""
    ring = tube(radius - 0.1, radius + 0.1, z - 0.1, z + 0.1)
    return len((m ^ ring).decompose())


@pytest.mark.parametrize("grip", (12, 24, 48))
def test_the_lid_has_its_ribs(grip):
    s = numbers(grip=grip)
    lid = centred(parts(grip=grip)[1])
    outer = 0.5 * s["lid_outside"] - screw_jar.RIB_DEPTH
    assert bumps(lid, outer + 0.5 * screw_jar.RIB_DEPTH, 5.0) == grip


def test_a_lid_without_ribs_is_smooth():
    s = numbers(grip=0)
    lid = centred(parts(grip=0)[1])
    lo, hi = bbox(lid)
    assert hi[0] - lo[0] == pytest.approx(s["lid_outside"], abs=0.05)
    assert bumps(lid, 0.5 * s["lid_outside"] + 0.3, 5.0) == 0


# ------------------------------------------------------------------ printing

@pytest.mark.parametrize("piece", (0, 1))
def test_nothing_faces_straight_down_above_the_bed(piece):
    m = parts()[piece].to_mesh()
    v = np.asarray(m.vert_properties)[:, :3]
    t = np.asarray(m.tri_verts)
    a, b, c = v[t[:, 0]], v[t[:, 1]], v[t[:, 2]]
    n = np.cross(b - a, c - a)
    area = np.linalg.norm(n, axis=1)
    keep = area > 1e-9
    down = n[keep, 2] / area[keep] < -math.cos(math.radians(5.0))
    low = np.minimum(np.minimum(a[keep, 2], b[keep, 2]), c[keep, 2])
    assert not (down & (low > 0.1)).any(), "a face hangs straight down over nothing"


# ------------------------------------------------------------------ the numbers

def test_sizes_reports_the_jar():
    s = screw_jar.sizes()
    assert s["volume"] == pytest.approx(math.pi * 30.0 ** 2 * 60.0 / 1000.0)
    assert s["lid_outside"] == pytest.approx(
        s["jar_outside"] + 2.0 * (s["clearance"] + s["wall"] + screw_jar.RIB_DEPTH))
    assert set(s) == {"diameter", "height", "wall", "pitch", "thread_len", "clearance", "grip",
                      "jar_outside", "lid_outside", "closed_height", "volume"}


@pytest.mark.parametrize("key, low, high", (
        ("diameter", screw_jar.D_MIN, screw_jar.D_MAX),
        ("height", screw_jar.HEIGHT_MIN, screw_jar.HEIGHT_MAX),
        ("wall", screw_jar.WALL_MIN, screw_jar.WALL_MAX),
        ("pitch", screw_jar.PITCH_MIN, screw_jar.PITCH_MAX),
        ("clearance", screw_jar.CLEARANCE_MIN, screw_jar.CLEARANCE_MAX)))
def test_every_number_is_clamped(key, low, high):
    kw = dict(height=100.0)
    assert screw_jar.sizes(**{**kw, key: -1e9})[key] == low
    assert screw_jar.sizes(**{**kw, key: 1e9})[key] == high


def test_the_thread_is_clamped_to_the_jar():
    assert screw_jar.sizes(thread_len=1.0)["thread_len"] == screw_jar.THREAD_MIN
    assert screw_jar.sizes(height=100.0, thread_len=99.0)["thread_len"] == screw_jar.THREAD_MAX
    s = screw_jar.sizes(height=15.0, thread_len=20.0)
    assert s["thread_len"] == pytest.approx(15.0 - screw_jar.BELOW_THREAD)


@pytest.mark.parametrize("grip, want", ((-5, 0), (0, 0), (3, screw_jar.GRIP_MIN), (30, 30),
                                        (99, screw_jar.GRIP_MAX)))
def test_the_ribs_are_none_or_enough(grip, want):
    assert screw_jar.sizes(grip=grip)["grip"] == want


@pytest.mark.parametrize("bad", (math.nan, math.inf, None, "wide"))
def test_a_number_that_is_no_number_is_refused(bad):
    with pytest.raises(ValueError):
        screw_jar.sizes(diameter=bad)


# ------------------------------------------------------------------ the dialog

def test_the_dialog_builds_a_jar_and_a_lid_as_two_groups_and_remembers_it(monkeypatch, tmp_path):
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
        dlg.diameter.setValue(50.0)
        dlg.height.setValue(40.0)
        dlg.wall.setValue(2.4)
        dlg.pitch.setValue(2.5)
        dlg.thread_len.setValue(8.0)
        dlg.clearance.setValue(0.4)
        dlg.grip.setValue(0)
        assert "79 ml" in dlg.info.text(), "the dialog does not say what the jar holds"
        return QDialog.Accepted

    monkeypatch.setattr(app_module.QDialog, "exec", fill)
    try:
        w.screw_jar_dialog()
        deadline = time.monotonic() + 20.0
        while len(w.doc.groups) < 2 and time.monotonic() < deadline:
            app.processEvents()
            time.sleep(0.005)
        assert [g.name for g in w.doc.groups] == ["Jar", "Lid"]
        lo, hi = np.asarray(w.doc.groups[0].bbox(), dtype=float)
        assert lo[2] == pytest.approx(0.0, abs=1e-6), "the jar does not stand on the bed"
        assert hi[2] - lo[2] == pytest.approx(screw_jar.FLOOR + 40.0, abs=1e-4)
        settings = app_module.Settings()
        assert (settings.screw_jar_diameter, settings.screw_jar_height, settings.screw_jar_wall,
                settings.screw_jar_pitch, settings.screw_jar_thread_len,
                settings.screw_jar_clearance, settings.screw_jar_grip) == \
            (50.0, 40.0, 2.4, 2.5, 8.0, 0.4, 0)
    finally:
        for key, value in (("screw_jar_diameter", 60.0), ("screw_jar_height", 60.0),
                           ("screw_jar_wall", 2.0), ("screw_jar_pitch", 3.0),
                           ("screw_jar_thread_len", 10.0), ("screw_jar_clearance", 0.3),
                           ("screw_jar_grip", 24)):
            setattr(w.settings, key, value)
        w.settings.save()
        w.doc.dirty = False
        w.close()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))

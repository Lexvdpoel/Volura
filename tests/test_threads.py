"""Screw thread on a cylinder and in a hole/sleeve that fit each other.

Tests the computing core of volura/threads.py (thread_solid, external_on, internal_in,
nut) and its place in the interface (submenu + Print Tools toolbar).
"""
import math
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import manifold3d as mf  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from PySide6.QtWidgets import QApplication, QToolBar  # noqa: E402

from volura import ops, threads  # noqa: E402

_app = QApplication.instance() or QApplication(sys.argv[:1])


def clean(text):
    return text.replace("&", "").replace("…", "").replace("...", "").strip()


# ---------------------------------------------------------------- thread_solid

def test_thread_solid_is_valid_and_between_core_and_major_cylinder():
    m = threads.thread_solid(10, 1.5, 20)
    assert m.status() == mf.Error.NoError
    assert not m.is_empty()
    assert m.volume() > 0
    L = 20.0
    r_core = (10 - 1.0825 * 1.5) / 2
    v_core = math.pi * r_core ** 2 * L
    v_major = math.pi * (10 / 2) ** 2 * L
    assert v_core < m.volume() < v_major


def test_thread_solid_runs_along_z_from_zero_to_length():
    m = threads.thread_solid(10, 1.5, 20)
    bb = np.array(m.bounding_box())
    lo, hi = bb[:3], bb[3:]
    assert abs(lo[2]) < 1e-9 and abs(hi[2] - 20) < 1e-9
    assert abs(lo[0] + 5) < 1e-6 and abs(hi[0] - 5) < 1e-6


# ------------------------------------------------------------ bolt and nut

def test_bolt_and_nut_fit_with_clearance_and_phase():
    bolt = threads.thread_solid(8, 1.25, 12)
    nut_fit = threads.nut(8, 1.25, 6.5, 13, 0.15)
    nut_tight = threads.nut(8, 1.25, 6.5, 13, 0.0)
    for m in (bolt, nut_fit, nut_tight):
        assert m.status() == mf.Error.NoError
    assert (bolt ^ nut_fit).volume() < 0.005 * bolt.volume()
    assert (bolt ^ nut_tight).volume() < 0.005 * bolt.volume()
    assert nut_fit.volume() < nut_tight.volume()


def test_nut_round_bus_variant_is_valid():
    bus = threads.nut(10, 1.5, 12, 18, 0.15, hex=False)
    assert bus.status() == mf.Error.NoError
    assert 0 < bus.volume() < math.pi * 9 ** 2 * 12


# ------------------------------------------------------- external_on / internal_in

def test_external_on_cylinder_gives_a_valid_smaller_solid():
    cyl = mf.Manifold.cylinder(20, 5, 5, 64)
    res = threads.external_on(cyl, (0, 0, 0), (0, 0, 1), 5, 20, pitch=1.5)
    assert res.status() == mf.Error.NoError
    assert 0 < res.volume() < cyl.volume()


def test_internal_in_block_with_hole_gives_a_valid_solid():
    block = (mf.Manifold.cube((20, 20, 20)).translate((-10, -10, 0))
             - mf.Manifold.cylinder(20, 5, 5, 64))
    res = threads.internal_in(block, (0, 0, 0), (0, 0, 1), 5, 20, 0.15, pitch=1.5)
    assert res.status() == mf.Error.NoError
    assert 0 < res.volume() < block.volume()


# ------------------------------------------------------------ left-hand thread

def test_left_hand_thread_is_the_mirror_image():
    right = threads.thread_solid(10, 1.5, 20)
    left = threads.thread_solid(10, 1.5, 20, left=True)
    assert left.status() == mf.Error.NoError
    assert abs(right.volume() - left.volume()) < 1e-3 * right.volume()
    mirrored = right.transform(np.array([[1.0, 0, 0, 0], [0, -1.0, 0, 0], [0, 0, 1.0, 0]]))
    assert mirrored.status() == mf.Error.NoError
    assert abs((mirrored ^ left).volume() - left.volume()) < 1e-3 * left.volume()
    assert (right ^ left).volume() < 0.95 * right.volume()


# ------------------------------------------------------------------- interface

@pytest.fixture()
def win():
    from volura.app import MainWindow
    w = MainWindow()
    yield w
    w.doc.dirty = False
    w.close()


def test_print_tools_toolbar_is_visible_on_the_right(win):
    bar = win.findChild(QToolBar, "PrintToolsToolbar")
    assert bar is not None, "toolbar PrintToolsToolbar is missing"
    assert not bar.isHidden(), "Print Tools must be visible by default"
    assert win.toolBarArea(bar).name == "RightToolBarArea"
    labels = [clean(a.text()) for a in bar.actions() if not a.isSeparator()]
    assert "Thread" in labels


def test_print_tools_is_the_last_submenu_of_tools(win):
    tools = next(a.menu() for a in win.menuBar().actions()
                 if a.menu() is not None and clean(a.text()) == "Tools")
    acts = [a for a in tools.actions() if not a.isSeparator()]
    assert acts[-1].menu() is not None
    assert clean(acts[-1].text()) == "Print Tools"
    assert tools.actions()[-2].isSeparator()
    labels = [clean(a.text()) for a in acts[-1].menu().actions() if not a.isSeparator()]
    thread_labels = ["Thread", "External Thread on Cylinder", "Internal Thread in Hole",
                     "New Bolt", "New Nut or Sleeve"]
    assert labels[:len(thread_labels)] == thread_labels


def test_external_thread_on_a_selected_cylinder_group_is_one_undo_step(win):
    g = ops.group_from_manifold(win.doc, mf.Manifold.cylinder(20, 5, 5, 64), "#f2f2f2", "cylinder")
    cyl_vol = g.ents.to_manifold().volume()
    win.doc.set_selection([("g", g.id, g.id)])
    win._apply_thread("external", 10, 1.5, 20, 0.15, 1, False, 17.0, True, False)
    m = g.ents.to_manifold()
    assert m.status() == mf.Error.NoError
    assert 0 < m.volume() < cyl_vol
    win.undo()
    assert abs(win.doc.group(g.id).ents.to_manifold().volume() - cyl_vol) < 1e-3


def test_thread_without_selection_reports_in_the_status_bar(win):
    msgs = []
    win.view.status_changed.connect(msgs.append)
    win.doc.set_selection([])
    win._apply_thread("external", 10, 1.5, 20, 0.15, 1, False, 17.0, True, False)
    assert any("select" in m for m in msgs), msgs


def test_new_bolt_with_matching_nut_makes_two_groups(win):
    win._apply_thread("bolt", 8, 1.25, 12, 0.15, 1, False, 13.6, True, True)
    assert len(win.doc.groups) == 2
    vols = sorted(g.ents.to_manifold().volume() for g in win.doc.groups)
    assert all(v > 0 for v in vols)
    names = sorted(g.name for g in win.doc.groups)
    assert names == ["Bolt M8x1.25", "Nut M8x1.25"]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))

"""Finger-joint box: the kerf compensation leaves no slivers at the corners of a panel.

`boxgen.finger_edge` makes the recesses of a female edge `kerf / 2` narrower on each side where a
finger of the other panel comes in. A recess that runs to the end of the edge has no finger
beyond that end, only the corner of the panel; narrowing it there left a sliver of `kerf / 2` by
the plate thickness at every such corner, which the worksheet showed as a short loose line
sticking out of the left and the right side of the default box.

These tests look at every edge of every panel of the box, with and without a lid, inside or
outside measures, at the kerfs a laser needs: no segment of an outline is shorter than the
kerf, and the assembled box without kerf is still a closed shell of exactly the right volume.

Run with:  python -m pytest tests/test_box_kerf_corners.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import manifold3d as mf  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from volura import boxgen  # noqa: E402

W, D, H, T = 100.0, 60.0, 40.0, 3.0


def segments(pts):
    p = np.asarray(pts, dtype=float)
    return np.hypot(*(np.roll(p, -1, axis=0) - p).T)


@pytest.mark.parametrize("kerf", (0.0, 0.1, 0.15, 0.3))
@pytest.mark.parametrize("lid", ("none", "closed"))
@pytest.mark.parametrize("inner", (True, False))
def test_no_panel_has_a_sliver_left_by_the_kerf(kerf, lid, inner):
    for panel in boxgen.box_panels(W, D, H, T, finger_w=7.5, inner=inner, lid=lid, kerf=kerf):
        shortest = float(segments(panel.points).min())
        assert shortest > max(kerf, 1e-6) + 1e-9, \
            f"{panel.name}: a segment of {shortest:.4f} mm with a kerf of {kerf}"


def test_the_default_side_panels_have_clean_corners():
    for panel in boxgen.box_panels(W, D, H, T, finger_w=7.5, kerf=0.15):
        pts = np.asarray(panel.points, dtype=float)
        lo, hi = pts.min(axis=0), pts.max(axis=0)
        for corner in ((lo[0], lo[1]), (hi[0], lo[1]), (lo[0], hi[1]), (hi[0], hi[1])):
            near = np.hypot(pts[:, 0] - corner[0], pts[:, 1] - corner[1])
            close = near[(near > 1e-9) & (near < 0.5)]
            assert not len(close), f"{panel.name}: points crowd the corner {corner}"


@pytest.mark.parametrize("lid", ("none", "closed"))
def test_the_assembled_box_is_still_a_closed_shell(lid):
    panels = boxgen.box_panels(W, D, H, T, finger_w=7.5, lid=lid, kerf=0.0)
    mans = boxgen.panels_to_manifolds(panels)
    box = mf.Manifold.batch_boolean(mans, mf.OpType.Add)
    total = sum(m.volume() for m in mans)
    assert box.volume() == pytest.approx(total, rel=1e-9), "two panels overlap"
    outer_h = H + (2.0 if lid == "closed" else 1.0) * T
    room = W * D * H
    assert box.volume() == pytest.approx((W + 2 * T) * (D + 2 * T) * outer_h - room, rel=1e-9)

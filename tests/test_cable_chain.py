"""Cable chain: a drag chain printed in one piece, links already in each other.

`cable_chain.cable_chain` lays `links` links in a row: the narrow front of every link, with a pin
on each side, lies in the wide rear of the link before it, with `CLEARANCE` of room around the
pin and between the plates. These tests weigh the chain against that: every link is a part of
its own (nothing is printed fast to its neighbour), the links turn on their pins far both ways
without running into each other, and a link that is pulled along the chain is held by its pin.

Run with:  python -m pytest tests/test_cable_chain.py
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
from volura import cable_chain  # noqa: E402

_app = QApplication.instance() or QApplication(sys.argv[:1])


def clean(text):
    return text.replace("&", "").replace("…", "").replace("...", "").strip()


def common(a, b):
    return mf.Manifold.batch_boolean([a, b], mf.OpType.Intersect).volume()


def turned(link, deg, s):
    """`link` placed as the next one in the chain and turned `deg` degrees on its pin."""
    P, r = s["pitch"], 0.5 * s["outer_h"]
    a = math.radians(deg)
    c, sn = math.cos(a), math.sin(a)
    M = np.array([[c, 0.0, sn, P - c * P - sn * r], [0.0, 1.0, 0.0, 0.0],
                  [-sn, 0.0, c, r + sn * P - c * r]])
    return link.translate((P, 0.0, 0.0)).transform(M)


@pytest.fixture()
def win(monkeypatch, tmp_path):
    monkeypatch.setattr(app_module.errors, "log_dir", lambda: str(tmp_path))
    from volura.app import MainWindow
    w = MainWindow()
    w.doc.dirty = False
    QCoreApplication.processEvents()
    yield w
    w.doc.dirty = False
    w.close()


def test_every_link_is_a_part_of_its_own():
    for links in (2, 5):
        chain = cable_chain.cable_chain(links=links)
        assert chain.status() == mf.Error.NoError
        assert len(chain.decompose()) == links, "links are printed fast to each other"
        lo = np.asarray(chain.bounding_box()[:3])
        assert lo[2] == pytest.approx(0.0)


def test_the_inside_is_the_room_asked_for():
    s = cable_chain.sizes(width=24.0, height=12.0, wall=2.0)
    link = cable_chain.chain_link(width=24.0, height=12.0, wall=2.0)
    room = mf.Manifold.cube([1.0, 24.0 - 0.2, 12.0 - 0.2]).translate(
        (1.0, -12.0 + 0.1, 2.0 + 0.1))
    assert common(link, room) == pytest.approx(0.0, abs=1e-6), "something stands in the cables"
    lo, hi = np.asarray(link.bounding_box()[:3]), np.asarray(link.bounding_box()[3:])
    assert hi[2] - lo[2] == pytest.approx(s["outer_h"])


@pytest.mark.parametrize("deg", [-90, -45, -20, 20, 45, 90])
def test_a_link_turns_on_its_pin_without_running_into_the_one_before(deg):
    s = cable_chain.sizes()
    link = cable_chain.chain_link()
    assert common(link, turned(link, deg, s)) == pytest.approx(0.0, abs=1e-6)


def test_the_pin_holds_the_next_link():
    s = cable_chain.sizes()
    link = cable_chain.chain_link()
    nxt = link.translate((s["pitch"], 0.0, 0.0))
    assert common(link, nxt) == pytest.approx(0.0, abs=1e-6)
    for d in ((1.0, 0.0, 0.0), (-1.0, 0.0, 0.0), (0.0, 0.0, 1.0)):
        assert common(link, nxt.translate(d)) > 1.0, f"the next link comes loose along {d}"


def test_a_size_that_cannot_be_printed_becomes_the_nearest_one_that_can():
    s = cable_chain.sizes(width=0.0, height=999.0, wall=99.0)
    assert (s["width"], s["height"], s["wall"]) == (
        cable_chain.WIDTH_MIN, cable_chain.HEIGHT_MAX, cable_chain.WALL_MAX)
    assert len(cable_chain.cable_chain(links=999).decompose()) == cable_chain.LINKS_MAX


def test_the_chain_follows_the_rack_in_parts(win):
    labels = [clean(a.text()) for a in win.flat_actions(win.menuBar())]
    assert labels.index("Cable Chain") == labels.index("Bin Wall Rack") + 1


def test_the_dialog_remembers_and_builds_one_part(win, monkeypatch):
    def fill(dlg):
        dlg.width.setValue(18.0)
        dlg.links.setValue(4)
        return QDialog.Accepted

    monkeypatch.setattr(app_module.QDialog, "exec", fill)
    win.cable_chain_dialog()
    deadline = time.monotonic() + 15.0
    while not win.doc.groups and time.monotonic() < deadline:
        _app.processEvents()
        time.sleep(0.005)
    assert len(win.doc.groups) == 1, "no chain in the model"
    assert len(win.doc.groups[0].ents.to_manifold().decompose()) == 4
    s = app_module.Settings()
    assert (s.chain_width, s.chain_links) == (pytest.approx(18.0), 4)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))

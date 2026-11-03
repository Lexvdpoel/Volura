"""Duwen/trekken: omhoog is positief, ook bij een los vlak op de grond (zoals SketchUp).

Een vlak dat op de grond getekend wordt wijst naar beneden; de getoonde en getypte
afstand moet toch positief zijn als het vlak omhoog getrokken wordt.
"""
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from PySide6.QtCore import QCoreApplication  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

_app = QApplication.instance() or QApplication(sys.argv[:1])


@pytest.fixture()
def win():
    from volura.app import MainWindow
    w = MainWindow()
    QCoreApplication.processEvents()
    yield w
    w.doc.dirty = False
    w.close()


def ground_rect(ents, z=0.0):
    pts = [(0, 0, z), (10, 0, z), (10, 5, z), (0, 5, z)]
    for a, b in zip(pts, pts[1:] + pts[:1]):
        ents.add_line(np.array(a, float), np.array(b, float))


def face_at_z(ents, z, nz):
    for fid, f in ents.F.items():
        if abs(f.normal[2] - nz) < 1e-6 and all(abs(ents.V[v][2] - z) < 1e-6 for v in f.loops[0]):
            return fid
    raise AssertionError("geen vlak")


def zmax(ents):
    return max(float(p[2]) for p in ents.V.values())


def pick(t, ents, fid):
    t.reset()
    t._set_target(t.doc.active, fid, ents.V[ents.F[fid].loops[0][0]].copy())


def test_getypte_afstand_op_grondvlak_gaat_omhoog(win):
    ents = win.doc.root
    ground_rect(ents)
    fid = face_at_z(ents, 0, -1)
    win.set_tool("pushpull")
    t = win.view.tools["pushpull"]
    pick(t, ents, fid)
    t.vcb_enter("10")
    assert zmax(ents) == pytest.approx(10.0)
    assert win.vcb.text() == "10.00"


def test_omhoog_slepen_toont_positieve_afstand(win):
    ents = win.doc.root
    ground_rect(ents)
    fid = face_at_z(ents, 0, -1)
    win.set_tool("pushpull")
    t = win.view.tools["pushpull"]
    pick(t, ents, fid)
    t.dist = -7.0                       # langs de normaal (omlaag gericht) = 7 mm omhoog
    assert t.sign * t.dist == pytest.approx(7.0)
    t.apply(t.dist)
    assert zmax(ents) == pytest.approx(7.0)
    assert win.vcb.text() == "7.00"


def test_bovenvlak_van_blok_omhoog_positief(win):
    ents = win.doc.root
    ground_rect(ents)
    win.set_tool("pushpull")
    t = win.view.tools["pushpull"]
    pick(t, ents, face_at_z(ents, 0, -1))
    t.vcb_enter("10")
    pick(t, ents, face_at_z(ents, 10, 1))
    assert t.sign == 1.0
    t.vcb_enter("5")
    assert zmax(ents) == pytest.approx(15.0)
    # een negatieve getypte waarde duwt het bovenvlak omlaag
    pick(t, ents, face_at_z(ents, 15, 1))
    t.vcb_enter("-3")
    assert zmax(ents) == pytest.approx(12.0)


class _Ev:
    def modifiers(self):
        from PySide6.QtCore import Qt
        return Qt.KeyboardModifier.NoModifier


def test_klikken_kiest_vlak(win, monkeypatch):
    ents = win.doc.root
    ground_rect(ents)
    fid = face_at_z(ents, 0, -1)
    win.set_tool("pushpull")
    t = win.view.tools["pushpull"]
    p = ents.V[ents.F[fid].loops[0][0]].copy()
    monkeypatch.setattr(t, "_face_at", lambda x, y: (win.doc.active, fid, p))
    t.mouse_press(_Ev(), 10, 10)
    assert t.target is not None and t.target[1] == fid
    assert t.sign == -1.0
    assert win.view.hover == ("face", win.doc.active, fid)

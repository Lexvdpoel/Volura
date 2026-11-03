"""The right-click menu on a face mimics SketchUp 2017 (ui_spec.CONTEXT_FACE).

Runs without a display (Qt offscreen platform).
"""
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

import ui_spec  # noqa: E402
from volura.geom import Plane  # noqa: E402
from volura.model import ROOT  # noqa: E402

_app = QApplication.instance() or QApplication(sys.argv[:1])


def clean(text):
    return text.replace("&", "").replace("…", "").replace("...", "").strip()


def menu_items(menu):
    out = []
    for a in menu.actions():
        if a.isSeparator():
            out.append("---")
        elif a.menu() is not None:
            out.append(clean(a.text()) + " >")
        else:
            out.append(clean(a.text()))
    return out


@pytest.fixture()
def win():
    from volura.app import MainWindow
    w = MainWindow()
    yield w
    w.doc.dirty = False
    w.close()


def draw_rect(win):
    loop = np.array([[0, 0], [40, 0], [40, 30], [0, 30]], dtype=float)
    created = win.doc.root.add_shape(Plane((0, 0, 0), (0, 0, 1)), [loop])
    assert created, "rectangle gave no face"
    return created[0]


def test_face_context_menu_matches_sketchup(win):
    fid = draw_rect(win)
    win.doc.set_selection([("f", ROOT, fid)])
    items = menu_items(win.build_context_menu())
    assert items[:len(ui_spec.CONTEXT_FACE)] == ui_spec.CONTEXT_FACE


def test_face_context_menu_submenus(win):
    fid = draw_rect(win)
    win.doc.set_selection([("f", ROOT, fid)])
    menu = win.build_context_menu()
    subs = {clean(a.text()): [clean(x.text()) for x in a.menu().actions() if not x.isSeparator()]
            for a in menu.actions() if a.menu() is not None}
    assert subs["Select"] == ["Bounding Edges", "Connected Faces",
                              "All Connected", "All with Same Material"]
    assert subs["Area"] == ["Selection", "Material"]
    assert subs["Intersect Faces"] == ["With Model", "With Selection"]
    assert subs["Flip Along"] == ["Red Axis", "Green Axis", "Blue Axis"]
    cut = next(a.menu() for a in menu.actions() if clean(a.text()) == "Intersect Faces")
    assert all(a.isEnabled() for a in cut.actions()), "Intersect Faces must work"


def test_edge_context_menu_starts_with_info_delete_hide(win):
    draw_rect(win)
    eid = sorted(win.doc.root.E)[0]
    win.doc.set_selection([("e", ROOT, eid)])
    items = menu_items(win.build_context_menu())
    assert items[:3] == ["Entity Info", "Delete", "Hide"]
    assert "Reverse Faces" not in items

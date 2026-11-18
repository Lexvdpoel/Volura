"""The view cube: the 26 regions of a cube, the view each one gives, and the widget
that floats in the top right corner of every 3D view.

The geometry needs no application; the widget runs offscreen, like the other GUI tests.

Run with:  python -m pytest tests/test_viewcube.py
"""
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from PySide6.QtCore import QPoint, Qt  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from volura import gcode_preview as gp  # noqa: E402
from volura.camera import VIEWS  # noqa: E402
from volura.i18n import tr  # noqa: E402
from volura.settings import qsettings  # noqa: E402
from volura.viewcube import (CORNERS, EDGES, FACES, REGIONS, ViewCube,  # noqa: E402
                             cube_polygons, hit, region_direction, shortest_turn, view_for)

_app = QApplication.instance() or QApplication(sys.argv[:1])

SIZE = ViewCube.CUBE


# ------------------------------------------------------------------- the geometry

def test_the_cube_has_26_regions():
    assert (len(FACES), len(EDGES), len(CORNERS), len(REGIONS)) == (6, 12, 8, 26)


@pytest.mark.parametrize("face", FACES)
def test_a_face_gives_exactly_the_standard_view_of_that_name(face):
    assert view_for((face,)) == VIEWS[face]


def test_straight_up_and_down_keep_the_yaw_you_came_from():
    assert view_for(("top",), keep_yaw=20.0) == (20.0, 89.99)
    assert view_for(("bottom",), keep_yaw=-135.0) == (-135.0, -89.99)


def test_a_corner_looks_at_the_iso_angle():
    yaw, pitch = view_for(("front", "right", "top"))
    assert pitch == pytest.approx(35.264389682754654)
    assert yaw == pytest.approx(-45.0), "45 degrees between front and right"


@pytest.mark.parametrize("region, yaw", ((("front", "top"), -90.0), (("right", "top"), 0.0),
                                         (("back", "bottom"), 90.0)))
def test_an_edge_looks_at_45_degrees(region, yaw):
    angle, pitch = view_for(region)
    assert pitch == pytest.approx(45.0 if "top" in region else -45.0)
    assert angle == pytest.approx(yaw)


def test_a_corner_points_at_a_diagonal_of_the_cube():
    assert np.allclose(region_direction(("front", "right", "top")),
                       np.array([1.0, -1.0, 1.0]) / np.sqrt(3.0))
    assert np.allclose(region_direction(("back",)), [0.0, 1.0, 0.0])


def test_at_the_front_view_the_middle_hits_the_front_face():
    assert hit(cube_polygons(*VIEWS["front"], SIZE), 0, 0) == ("front",)


def test_near_the_top_of_the_front_face_hits_the_top_edge():
    cells = cube_polygons(*VIEWS["front"], SIZE)
    assert hit(cells, 0, -SIZE / 4) == ("front", "top")
    assert hit(cells, -SIZE / 4, 0) == ("front", "left")
    assert hit(cells, SIZE / 4, -SIZE / 4) == ("front", "right", "top")


def test_the_face_at_the_back_is_never_hit():
    cells = cube_polygons(*VIEWS["front"], SIZE)
    assert all("back" not in region for region, _poly in cells)
    for x in range(-40, 41, 4):
        for y in range(-40, 41, 4):
            assert "back" not in (hit(cells, x, y) or ())


def test_shortest_turn_goes_the_short_way_round():
    assert shortest_turn(170, -170) == pytest.approx(20.0)
    assert shortest_turn(-170, 170) == pytest.approx(-20.0)
    assert shortest_turn(10, 30) == pytest.approx(20.0)


# ---------------------------------------------------------------------- the widget

@pytest.fixture(scope="module")
def win():
    q = qsettings()
    saved = q.value("show_view_cube")
    from volura.app import MainWindow
    w = MainWindow()
    w.resize(1200, 800)
    w.show()
    QApplication.processEvents()
    yield w
    w.doc.dirty = False
    w.close()
    if saved is None:
        q.remove("show_view_cube")
    else:
        q.setValue("show_view_cube", saved)
    q.sync()


def spot(cube, dx=0, dy=0):
    """A point of the cube widget, dx and dy pixels from its centre."""
    return QPoint(int(cube.width() / 2) + dx, int(cube.height() / 2) + dy)


def click(cube, dx=0, dy=0):
    QTest.mouseClick(cube, Qt.LeftButton, Qt.NoModifier, spot(cube, dx, dy))


def test_the_viewport_has_a_cube_in_its_top_right_corner(win):
    cube = win.view.view_cube
    assert isinstance(cube, ViewCube) and cube.isVisible()
    assert cube.x() == win.view.width() - cube.width() - cube.MARGIN
    assert cube.y() == cube.MARGIN


def test_the_cube_follows_the_view_when_it_is_resized(win):
    cube = win.view.view_cube
    win.resize(900, 700)
    QApplication.processEvents()
    assert cube.x() == win.view.width() - cube.width() - cube.MARGIN
    win.resize(1200, 800)
    QApplication.processEvents()


def test_the_cube_only_redraws_when_the_camera_turned(win, monkeypatch):
    cube = win.view.view_cube
    seen = []
    monkeypatch.setattr(cube, "update", lambda *a: seen.append(1))
    cube._angles = None                      # a camera the cube has not been drawn with yet
    cube.sync()
    win.view.cam.set_view("top")
    cube.sync()
    assert len(seen) == 2
    cube.sync()                              # the same camera again: nothing to draw
    assert len(seen) == 2


def test_a_click_on_the_face_you_already_look_at_keeps_the_view(win):
    cube = win.view.view_cube
    win.view.cam.set_view("front")
    click(cube)
    cube.finish_turn()
    assert (win.view.cam.yaw, win.view.cam.pitch) == VIEWS["front"]


def test_a_click_on_the_top_face_looks_straight_down(win):
    cube = win.view.view_cube
    win.view.cam.set_view("iso")
    click(cube, 0, -22)                      # the middle cell of the top face, seen from iso
    cube.finish_turn()
    assert win.view.cam.pitch == pytest.approx(89.99, abs=1e-6)
    assert win.view.cam.yaw == pytest.approx(VIEWS["iso"][0]), "looking down keeps the yaw"


def test_a_click_on_an_edge_looks_at_45_degrees(win):
    cube = win.view.view_cube
    win.view.cam.set_view("front")
    click(cube, 0, -int(cube.CUBE / 4))       # the top edge of the front face
    cube.finish_turn()
    assert win.view.cam.pitch == pytest.approx(45.0)
    assert win.view.cam.yaw == pytest.approx(VIEWS["front"][0])


def test_a_click_on_a_corner_looks_at_the_iso_angle(win):
    cube = win.view.view_cube
    win.view.cam.set_view("front")
    click(cube, int(cube.CUBE / 4), -int(cube.CUBE / 4))
    cube.finish_turn()
    assert win.view.cam.pitch == pytest.approx(35.264389682754654, abs=0.1)
    assert win.view.cam.yaw == pytest.approx(-45.0, abs=0.1)


def test_the_turn_takes_the_short_way_round(win):
    cube = win.view.view_cube
    win.view.cam.yaw, win.view.cam.pitch = -170.0, 0.0
    cube.turn_to(("left",))                  # yaw 180: ten degrees on, not 350 degrees back
    assert cube._to == (180.0, 0.0)
    cube._turning(0.5)
    assert win.view.cam.yaw == pytest.approx(-175.0)
    cube.finish_turn()
    assert win.view.cam.yaw == pytest.approx(180.0)


def test_dragging_the_cube_orbits_the_view(win):
    cube = win.view.view_cube
    win.view.cam.set_view("iso")
    yaw = win.view.cam.yaw
    QTest.mousePress(cube, Qt.LeftButton, Qt.NoModifier, spot(cube))
    QTest.mouseMove(cube, spot(cube, 30, 10))
    QTest.mouseRelease(cube, Qt.LeftButton, Qt.NoModifier, spot(cube, 30, 10))
    assert win.view.cam.yaw == pytest.approx(yaw - 12.0), "the same speed as orbiting in the view"


def test_the_house_goes_back_to_iso_with_everything_in_view(win):
    cube = win.view.view_cube
    win.view.cam.set_view("top")
    distance = win.view.cam.distance
    win.view.cam.distance = distance * 3
    click(cube, -int(cube.width() / 2) + 6, -int(cube.height() / 2) + 6)
    assert (win.view.cam.yaw, win.view.cam.pitch) == VIEWS["iso"]
    assert win.view.cam.distance < distance * 3, "home also shows the whole model again"


def test_the_home_view_can_be_stored_and_reset(win):
    cube = win.view.view_cube
    win.view.cam.yaw, win.view.cam.pitch = -20.0, 15.0
    cube.store_home()
    assert cube.home() == (-20.0, 15.0, True)
    cube.go_home()
    assert (win.view.cam.yaw, win.view.cam.pitch) == (-20.0, 15.0)
    cube.reset_home()
    assert cube.custom_home() is None and qsettings().value(cube.home_key()) is None
    cube.go_home()
    assert (win.view.cam.yaw, win.view.cam.pitch) == VIEWS["iso"]


def test_the_view_menu_shows_and_hides_the_cube(win):
    cube = win.view.view_cube
    win.a_view_cube.trigger()
    assert cube.isHidden() and win.settings.show_view_cube is False
    win.a_view_cube.trigger()
    assert cube.isVisible() and win.settings.show_view_cube is True


def test_the_cube_says_what_it_is(win):
    cube = win.view.view_cube
    assert cube.accessibleName() == tr("View cube")
    assert "drag to orbit" in cube.toolTip()


def test_the_cube_keeps_clicks_of_its_own(win):
    """A click on the cube never reaches the tool of the view; one next to it always does."""
    win.set_tool("select")
    tool = win.view.tool
    seen, real = [], tool.mouse_press
    tool.mouse_press = lambda e, x, y: seen.append((x, y))
    try:
        click(win.view.view_cube)
        assert seen == []
        QTest.mouseClick(win.view, Qt.LeftButton, Qt.NoModifier, QPoint(60, 60))
        assert seen, "the view still works next to the cube"
    finally:
        tool.mouse_press = real


def test_the_preview_views_have_a_cube_too():
    from volura import print_preview as pp, sla_preview as sp
    layer = pp.LayerView(gp.Preview(), (0.0, 0.0, 100.0, 100.0))
    assert isinstance(layer.view_cube, ViewCube) and layer.view_cube.key == "print"
    mesh = sp.MeshView((100.0, 100.0))
    assert isinstance(mesh.view_cube, ViewCube) and mesh.view_cube.key == "sla"

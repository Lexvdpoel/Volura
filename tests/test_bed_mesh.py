"""The bed mesh of a Klipper printer: `volura/bed_mesh.py`, its asking in `printer_link.py`
and Window > Bed Mesh… (`volura/bed_mesh_dialog.py`).

A bed mesh is what the probe of a printer touched on its bed, so it is asked from the printer:
Moonraker answers it and nothing else has one, which is what the first block tests against the
same fake printers tests/test_printer_link.py stands up. The numbers next to it - how flat a bed
is, what the window says of a mesh and what colour a point gets - need no network at all. The
last block opens the window over fake printers and looks at what it paints, what it says when the
mouse stands over a cell, and how often it asks.

Run with:  python -m pytest tests/test_bed_mesh.py
"""
import os
import sys
import threading

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from PySide6.QtCore import QCoreApplication, QEvent, QPointF, Qt  # noqa: E402
from PySide6.QtGui import QColor, QMouseEvent  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from test_printer_link import FakePrinter, free_port  # noqa: E402
from volura import bed_mesh as bm  # noqa: E402
from volura import bed_mesh_dialog as bd  # noqa: E402
from volura import printer_link as pl  # noqa: E402

_app = QApplication.instance() or QApplication(sys.argv[:1])

#: the path a Klipper printer is asked about its bed, as volura/printer_link.py asks it
MOON_MESH = "/printer/objects/query?bed_mesh"


def grid(rows=5, cols=5) -> list:
    """A probed grid of z heights that climbs with every row and column: no two points are
    alike, and the one in the middle is 0 mm, so a painted cell says where it came from."""
    return [[(row * cols + col - (rows * cols - 1) / 2) / 100.0 for col in range(cols)]
            for row in range(rows)]


PROBED = grid()                                    # -0,12 mm ... 0 ... +0,12 mm over 5 by 5
BED = dict(mesh_min=[0.0, 0.0], mesh_max=[200.0, 200.0])


def moon_mesh(**fields) -> dict:
    """What Klipper (Moonraker) answers on the bed mesh it was queried for, with these fields
    of that mesh; None leaves a field out of the answer altogether."""
    bed = {"profile_name": "thuis", "mesh_min": [0.0, 0.0], "mesh_max": [200.0, 200.0],
           "probed_matrix": PROBED, "mesh_matrix": PROBED}
    bed.update({one: value for one, value in fields.items() if value is not None})
    for one, value in fields.items():
        if value is None:
            bed.pop(one, None)
    return {"result": {"status": {"bed_mesh": bed}}}


def asked(fake, path=MOON_MESH) -> list:
    """The requests this fake printer got for that path, so far."""
    return [got for got in fake.requests if got.path == path]


@pytest.fixture()
def printer():
    """Fake printers on ports of this computer; they are closed after the test."""
    made = []

    def make(kind, api_key="", code=0):
        made.append(FakePrinter(kind, api_key, code))
        return made[-1]

    yield make
    for fake in made:
        fake.stop()


@pytest.fixture(scope="module")
def win():
    """The window these bed mesh windows sit in.

    Every setting these tests read is set before and after, so none of them depends on what
    another test left in the settings of this computer."""
    from volura.app import MainWindow
    w = MainWindow()
    empty(w)
    yield w
    empty(w)
    w.doc.dirty = False
    w.close()


def empty(w):
    """The settings of the window as they stand before and after these tests."""
    w.settings.printers = []
    w.settings.save()


# ------------------------------------------------------------------ asking a printer

def test_klipper_says_the_mesh_of_its_bed(printer):
    klipper = printer("moonraker")
    klipper.answer(MOON_MESH, moon_mesh())
    got = pl.bed_mesh(klipper.connection("Klipper"))
    assert got.profile == "thuis"
    assert (got.x_min, got.y_min, got.x_max, got.y_max) == (0.0, 0.0, 200.0, 200.0)
    assert got.probed.shape == (5, 5)
    assert got.probed[0][0] == pytest.approx(-0.12), "the first row is the lowest y of the bed"
    assert got.probed[4][4] == pytest.approx(0.12)
    assert got.mesh.shape == (5, 5)
    assert klipper.last.path == MOON_MESH


def test_the_mesh_the_printer_interpolated_is_kept(printer):
    klipper = printer("moonraker")
    conn = klipper.connection("Klipper")
    klipper.answer(MOON_MESH, moon_mesh(mesh_matrix=grid(9, 9)))
    got = pl.bed_mesh(conn)
    assert (got.probed.shape, got.mesh.shape) == ((5, 5), (9, 9)), "a finer grid is kept finer"
    klipper.answer(MOON_MESH, moon_mesh(mesh_matrix=None))
    gone = pl.bed_mesh(conn)
    assert gone.probed.shape == (5, 5) and gone.mesh.size == 0, "a printer may send no grid"
    klipper.answer(MOON_MESH, moon_mesh(mesh_matrix=[[0.1], [0.1, 0.2]]))
    half = pl.bed_mesh(conn)
    assert half.probed.shape == (5, 5) and half.mesh.size == 0, \
        "a broken interpolated grid leaves the picture to what was probed"


def test_a_bed_that_was_never_probed_has_no_mesh(printer):
    klipper = printer("moonraker")
    conn = klipper.connection("Klipper")
    for junk in ({}, {"result": {}}, {"result": "nonsens"},
                 {"result": {"status": {}}}, {"result": {"status": {"bed_mesh": "nonsens"}}},
                 moon_mesh(probed_matrix=[]), moon_mesh(probed_matrix=[[]]),
                 moon_mesh(probed_matrix=[[], []]), moon_mesh(probed_matrix="nonsens"),
                 moon_mesh(probed_matrix=[[0.1], [0.1, 0.2]]),
                 moon_mesh(probed_matrix=[[0.1, "beneden"]]),
                 moon_mesh(probed_matrix=[[0.1, float("nan")]]),
                 moon_mesh(probed_matrix=[[0.1, float("inf")]]),
                 moon_mesh(mesh_min=None), moon_mesh(mesh_min=[0.0]),
                 moon_mesh(mesh_min=["x", "y"]), moon_mesh(mesh_max=None),
                 moon_mesh(mesh_min=[100.0, 100.0], mesh_max=[100.0, 200.0])):
        klipper.answer(MOON_MESH, junk)
        assert pl.bed_mesh(conn) is None, junk


def test_an_answer_that_is_no_answer_at_all(printer):
    klipper = printer("moonraker")
    klipper.body(MOON_MESH, b"<!DOCTYPE html><html><body>Mainsail</body></html>", "text/html")
    assert pl.bed_mesh(klipper.connection("Klipper")) is None, "a web page is no mesh"


def test_a_mesh_may_stand_straight_under_result(printer):
    klipper = printer("moonraker")
    answer = moon_mesh()["result"]["status"]["bed_mesh"]
    klipper.answer(MOON_MESH, {"result": {"bed_mesh": answer}})
    got = pl.bed_mesh(klipper.connection("Klipper"))
    assert got is not None and got.probed.shape == (5, 5)


def test_the_profile_may_be_absent_or_wrong(printer):
    klipper = printer("moonraker")
    conn = klipper.connection("Klipper")
    klipper.answer(MOON_MESH, moon_mesh(profile_name=None))
    assert pl.bed_mesh(conn).profile == ""
    klipper.answer(MOON_MESH, moon_mesh(profile_name=7))
    assert pl.bed_mesh(conn).profile == "", "a number is no name for a profile"


def test_other_printers_have_no_bed_mesh_to_ask(printer):
    octo = printer("octoprint")
    prusa = printer("prusalink")
    assert pl.bed_mesh(octo.connection()) is None
    assert pl.bed_mesh(prusa.connection("Prusa")) is None
    assert pl.bed_mesh(pl.Connection("Octo", "toslink", octo.url)) is None
    assert octo.requests == [] and prusa.requests == [], \
        "OctoPrint has a bed mesh only through a plugin and PrusaLink not at all"


def test_a_printer_that_answers_nothing_is_said_the_way_check_says_it(printer):
    with pytest.raises(pl.LinkError) as bad:
        pl.bed_mesh(pl.Connection("Thuis", "moonraker", f"127.0.0.1:{free_port()}"))
    assert "Nothing answers at" in str(bad.value)
    klipper = printer("moonraker", api_key="the right one")
    klipper.answer(MOON_MESH, moon_mesh())
    with pytest.raises(pl.LinkError) as bad:
        pl.bed_mesh(klipper.connection("Klipper", api_key="a wrong one"))
    assert "API key" in str(bad.value)
    assert "right one" not in str(bad.value) and "wrong one" not in str(bad.value)
    with pytest.raises(pl.LinkError) as bad:
        pl.bed_mesh(pl.Connection("Klipper", "moonraker", "het printernetwerk"))
    assert "is not the address of a printer" in str(bad.value)
    klipper.refuse(MOON_MESH, 500)
    with pytest.raises(pl.LinkError) as bad:
        pl.bed_mesh(klipper.connection("Klipper", api_key="the right one"))
    assert "error 500" in str(bad.value)


# ---------------------------------------------------------------- the numbers of a mesh

def test_flatness_is_the_highest_point_minus_the_lowest():
    assert bm.flatness(PROBED) == pytest.approx(0.24)
    assert bm.flatness([[1.0, 2.0], [3.0, 0.5]]) == pytest.approx(2.5)
    assert bm.flatness(np.asarray([[0.1, float("nan")], [-0.2, 0.0]])) == pytest.approx(0.3)
    assert bm.flatness([[0.3]]) == 0.0, "one point is as flat as itself"


def test_flatness_of_nothing_is_flat_as_a_mirror():
    for nothing in ([], [[]], [[], []], [[float("nan")]], None, "nonsens",
                    np.zeros((0, 0))):
        assert bm.flatness(nothing) == 0.0, nothing


def test_the_window_says_what_a_mesh_means():
    mesh = bm.BedMesh("keuken", 0.0, 0.0, 200.0, 200.0, np.asarray(PROBED))
    assert bm.summary(mesh) == ("Range 0.240 mm (from -0.120 to 0.120 mm), 5×5 points, "
                                "profile keuken")
    wide = bm.BedMesh("p", probed=np.zeros((3, 7)))
    assert "3×7 points" in bm.summary(wide), "the rows over y come first"


def test_a_mesh_without_points_or_name_says_what_it_has():
    assert bm.summary(bm.BedMesh()) == ("Range 0.000 mm (from 0.000 to 0.000 mm), 0×0 points, "
                                        "profile —")
    assert bm.summary(bm.BedMesh("", probed=np.asarray([[0.01]]))) == \
        "Range 0.000 mm (from 0.010 to 0.010 mm), 1×1 points, profile —"


def test_a_bed_wholly_above_or_below_zero_is_said_as_it_stands():
    above = bm.BedMesh("a", probed=np.asarray([[0.20], [0.40]]))
    below = bm.BedMesh("b", probed=np.asarray([[-0.40], [-0.10]]))
    assert bm.summary(above) == "Range 0.200 mm (from 0.200 to 0.400 mm), 2×1 points, profile a"
    assert bm.summary(below) == "Range 0.300 mm (from -0.400 to -0.100 mm), 2×1 points, profile b"
    assert bm.limits(above.probed) == (0.0, 0.40)
    assert bm.limits(below.probed) == (-0.40, 0.0)


def test_the_colours_always_hold_zero():
    assert bm.limits(np.asarray(PROBED)) == pytest.approx((-0.12, 0.12))
    assert bm.limits([[0.28, 0.31], [0.29, 0.33]]) == (0.0, 0.33), \
        "a bed that stands above 0 keeps its white on 0"
    assert bm.limits([[-0.2, -0.1]]) == (-0.2, 0.0)
    assert bm.limits([]) == (0.0, 0.0) and bm.limits(None) == (0.0, 0.0)
    assert bm.limits([[float("nan")]]) == (0.0, 0.0)


def test_zero_is_white_and_a_bed_is_blue_below_and_red_above():
    assert bm.colour(0.0, -0.09, 0.05) == bm.MID, "0 mm is always the white in the middle"
    assert bm.colour(-0.09, -0.09, 0.09) == bm.COLD
    assert bm.colour(0.09, -0.09, 0.09) == bm.WARM
    assert bm.colour(-9.0, -0.09, 0.09) == bm.COLD, "what lies deeper is painted as deep as it can"
    assert bm.colour(9.0, -0.09, 0.09) == bm.WARM, "and what lies higher as high"


def test_the_scale_is_as_wide_on_both_sides_of_zero():
    assert bm.colour(0.045, -0.09, 0.045) == bm.colour(0.045, -0.09, 0.09), \
        "one side of 0 does not make the colours twice as strong"
    under, over = bm.colour(-0.06, -0.1, 0.1), bm.colour(0.06, -0.1, 0.1)
    assert under[0] < under[2] and over[2] < over[0], "blue below 0, red above it"
    for got, end in ((under, bm.COLD), (over, bm.WARM)):
        near = [(255 - one) / (255 - two) for one, two in zip(got, end)]
        assert all(one == pytest.approx(0.6, abs=0.01) for one in near), \
            "the same distance from 0 is equally far from white"
    assert 0.0 < under[2] - under[0] < 255.0


def test_a_colour_between_zero_and_the_end_lies_between_white_and_that_end():
    for z in (-0.07, -0.02, 0.02, 0.07):
        got = bm.colour(z, -0.1, 0.1)
        end = bm.COLD if z < 0 else bm.WARM
        for one, middle, away in zip(bm.MID, got, end):
            assert min(one, away) <= middle <= max(one, away), z


def test_a_colour_that_cannot_be_known_is_white():
    for junk in ((float("nan"), -0.1, 0.1), (0.05, 0.0, 0.0), (0.05, None, 0.1),
                 (None, -0.1, 0.1), (0.05, "x", "y"), (0.05, float("inf"), 0.1)):
        assert bm.colour(*junk) == bm.MID, junk


def test_the_picture_is_the_smooth_mesh_when_there_is_one():
    one = bm.BedMesh(probed=np.asarray([[0.05]]))
    assert one.picture().shape == (1, 1), "a bed of one point is painted with that point"
    smooth = bm.BedMesh(probed=np.asarray(PROBED), mesh=np.zeros((9, 9)))
    assert smooth.picture().shape == (9, 9)
    assert bm.BedMesh().picture().size == 0, "nothing at all to paint"


def test_a_cell_says_where_on_the_bed_it_stands():
    mesh = bm.BedMesh("p", 10.0, 20.0, 210.0, 120.0, np.asarray(PROBED))
    x, y, z = mesh.point(0, 0)
    assert (x, y) == pytest.approx((30.0, 30.0)), "the middle of the first cell of the bed"
    assert z == pytest.approx(-0.12)
    assert mesh.point(4, 4)[:2] == pytest.approx((190.0, 110.0))
    for wrong in ((5, 0), (-1, 0), (0, 5), (0, -1)):
        assert mesh.point(*wrong) is None, wrong
    assert bm.BedMesh().point(0, 0) is None


# ----------------------------------------------------------------------- the window

class Asks:
    """Stands in for `printer_link.bed_mesh`: counts how often it is asked and says what the test
    said it should. `hold` keeps an asking busy, so a test can look at a window that is still
    waiting for an answer; `inside` says that an asking really started."""

    def __init__(self, result=None, hold=None):
        self.calls = 0
        self.result = result
        self.hold = hold
        self.inside = threading.Event()

    def __call__(self, conn, timeout=5):
        self.calls += 1
        self.inside.set()
        if self.hold is not None:
            assert self.hold.wait(20), "the test let no asking come to an end"
        return self.result


def quiet(panel):
    """What this window has on its way, come to an end; nothing to wait for when it asks nothing.
    """
    job = panel._job
    if job is None:
        return
    assert job.wait(20000), f"asking the printer never came to an end: {job.error}"
    QCoreApplication.processEvents()              # the answer, so it is on the screen


def closed(panel):
    """Close this window with nothing of its own still on its way."""
    quiet(panel)
    panel.close()


def fresh(win, conns=()):
    """The window on the screen over these printers, with nothing of the test before it on its
    way and nothing waited for.

    The window Volura keeps lives from one test to the next, so every test says again which
    printers there are; and every fake printer stands on a port of its own, which is another
    printer to that window and takes the mesh of the previous test off the screen."""
    panel = getattr(win, "_bed_mesh", None)
    if panel is not None:
        quiet(panel)
        panel.close()
    win.settings.printers = list(conns)
    win.settings.save()
    return win.bed_mesh()


def opened(win, *conns):
    """The window on the screen over these printers, with what showing it asked come to an end."""
    panel = fresh(win, conns)
    quiet(panel)
    return panel


def ink_of(image, view, row, col) -> tuple:
    """The colour the picture shows in the middle of this cell of the map."""
    cell = view.rect_of(row, col)
    scale = image.width() / max(1, view.width())
    got = image.pixelColor(int(cell.center().x() * scale), int(cell.center().y() * scale))
    return (got.red(), got.green(), got.blue())


def move(view, x, y):
    """A mouse move over this pixel, through the event routing of the widget."""
    where = QPointF(float(x), float(y))
    view.mouseMoveEvent(QMouseEvent(QEvent.MouseMove, where, where, Qt.NoButton, Qt.NoButton,
                                    Qt.NoModifier))


def test_the_box_lists_only_the_printers_with_a_bed_mesh(printer, win):
    klipper = printer("moonraker")
    klipper.answer(MOON_MESH, moon_mesh())
    octo, prusa = printer("octoprint"), printer("prusalink")
    panel = opened(win, octo.connection("Octo"), klipper.connection("Klipper"),
                   prusa.connection("Prusa"))
    try:
        assert [panel.printer.itemText(i) for i in range(panel.printer.count())] == ["Klipper"]
        assert panel.printer.currentData().kind == "moonraker"
        assert "127.0.0.1" in panel.printer.itemData(0, Qt.ToolTipRole)
        assert panel.mesh is not None and panel.mesh.profile == "thuis"
        assert len(asked(klipper)) == 1, "one asking when the window opens"
        assert octo.requests == [] and prusa.requests == []
    finally:
        closed(panel)


def test_without_a_klipper_printer_the_window_says_what_it_needs(printer, win):
    octo = printer("octoprint")
    panel = opened(win, octo.connection("Octo"))
    try:
        assert panel.printer.count() == 0
        assert panel.needs.text() == bd.NEEDS_KLIPPER
        assert "Klipper (Moonraker)" in panel.needs.text()
        assert "Preferences > Printers" in panel.needs.text()
        assert panel.needs.isVisible() and not panel.printer.isVisible()
        assert not panel.refresh.isVisible(), "nothing to refresh without a printer"
        assert not panel.view.isVisible()
        assert panel.mesh is None and octo.requests == []
    finally:
        closed(panel)
    panel = opened(win)                       # not a printer on the list at all
    try:
        assert panel.needs.isVisible() and panel.mesh is None
    finally:
        closed(panel)


def test_the_mesh_is_painted_blue_below_zero_and_red_above_it(printer, win):
    klipper = printer("moonraker")
    klipper.answer(MOON_MESH, moon_mesh())
    panel = opened(win, klipper.connection("Klipper"))
    try:
        view = panel.view
        assert view.isVisible() and not panel.grab().isNull()
        image = view.grab().toImage()
        lo, hi = bm.limits(np.asarray(PROBED))
        assert (lo, hi) == pytest.approx((-0.12, 0.12))
        for row in range(5):
            for col in range(5):
                want = QColor(*bm.colour(PROBED[row][col], lo, hi))
                got = ink_of(image, view, row, col)
                assert got == (want.red(), want.green(), want.blue()), f"row {row}, column {col}"
        front_ink = ink_of(image, view, 0, 0)
        back_ink = ink_of(image, view, 4, 4)
        assert front_ink[2] > front_ink[0], "the front is the low side"
        assert back_ink[0] > back_ink[2], "the back is the high side"
        assert panel.legend.isVisible() and not panel.error.isVisible()
        assert panel.summary.text() == ("Range 0.240 mm (from -0.120 to 0.120 mm), 5×5 points, "
                                        "profile thuis")
    finally:
        closed(panel)


def test_row_zero_of_the_mesh_is_drawn_at_the_bottom(printer, win):
    klipper = printer("moonraker")
    klipper.answer(MOON_MESH, moon_mesh())
    panel = opened(win, klipper.connection("Klipper"))
    try:
        view = panel.view
        front, back = view.rect_of(0, 0), view.rect_of(4, 0)
        assert not front.isNull() and not back.isNull()
        assert front.y() > back.y(), "row 0 is the lowest y of the bed, which is its front"
        assert front.bottom() == view.place().bottom()
        left, right = view.rect_of(0, 0), view.rect_of(0, 4)
        assert left.x() < right.x(), "x goes to the right"
        where = view.cell_at(front.center().x(), front.center().y())
        assert where == (0, 0), "the front left cell of the bed is probed cell 0, 0"
        assert view.cell_at(1, 1) is None, "next to the bed is nothing to read"
        assert view.rect_of(5, 0).isNull() and view.rect_of(-1, 0).isNull()
    finally:
        closed(panel)


def test_a_rectangular_bed_is_painted_in_its_own_proportions(printer, win):
    klipper = printer("moonraker")
    klipper.answer(MOON_MESH, moon_mesh(mesh_min=[0.0, 0.0], mesh_max=[300.0, 100.0]))
    panel = opened(win, klipper.connection("Klipper"))
    try:
        view = panel.view
        where = view.place()
        assert not where.isNull()
        assert where.width() / where.height() == pytest.approx(3.0, abs=0.02)
        front, back = view.rect_of(0, 0), view.rect_of(4, 0)
        assert front.bottom() == where.bottom() and front.y() > back.y()
        left, right = view.rect_of(0, 0), view.rect_of(0, 4)
        assert left.x() == where.x() and right.right() == where.right()
        x, y, _ = panel.mesh.point(0, 0)
        assert (x, y) == pytest.approx((30.0, 10.0)), "the middle of the first cell of this bed"
    finally:
        closed(panel)


def test_the_smooth_mesh_makes_the_picture_and_the_numbers_stay_probed(printer, win):
    klipper = printer("moonraker")
    klipper.answer(MOON_MESH, moon_mesh(probed_matrix=[[0.10]], mesh_matrix=[[-0.20, 0.0],
                                                                             [0.0, 0.20]]))
    panel = opened(win, klipper.connection("Klipper"))
    try:
        view = panel.view
        assert view.cell_at(*_middle_of(view.rect_of(0, 0))) == (0, 0), \
            "the numbers come from the one point that was probed"
        image = view.grab().toImage()
        cold, warm = QColor(*bm.COLD), QColor(*bm.WARM)
        assert ink_of(image, view, 0, 0) == (cold.red(), cold.green(), cold.blue())
        assert ink_of(image, view, 1, 1) == (warm.red(), warm.green(), warm.blue())
        assert "1×1 points" in panel.summary.text()
    finally:
        closed(panel)


def test_an_entirely_flat_bed_needs_no_colour_bar(printer, win):
    klipper = printer("moonraker")
    flat = [[0.0, 0.0], [0.0, 0.0]]
    klipper.answer(MOON_MESH, moon_mesh(probed_matrix=flat, mesh_matrix=flat))
    panel = opened(win, klipper.connection("Klipper"))
    try:
        assert panel.mesh is not None and not panel.error.text()
        assert panel.summary.text() == ("Range 0.000 mm (from 0.000 to 0.000 mm), 2×2 points, "
                                        "profile thuis")
        assert not panel.legend.isVisible(), "one colour has no bar to say it"
        assert not panel.grab().isNull(), "the flat bed is still a bed to paint"
    finally:
        closed(panel)


def _middle_of(cell) -> tuple:
    """The pixel in the middle of a cell, for a mouse to stand on."""
    return int(cell.center().x()), int(cell.center().y())


def test_the_line_under_the_picture_says_where_the_mouse_stands(printer, win):
    klipper = printer("moonraker")
    klipper.answer(MOON_MESH, moon_mesh())
    panel = opened(win, klipper.connection("Klipper"))
    try:
        view = panel.view
        move(view, *_middle_of(view.rect_of(1, 3)))
        assert panel.status.text() == "x 140.0 mm, y 60.0 mm, z -0.040 mm"
        move(view, *_middle_of(view.rect_of(4, 4)))
        assert panel.status.text() == "x 180.0 mm, y 180.0 mm, z 0.120 mm"
        move(view, 1, 1)
        assert panel.status.text() == "", "outside the bed there is nothing to say"
        view.leaveEvent(QEvent(QEvent.Leave))
        assert panel.status.text() == ""
    finally:
        closed(panel)


def test_the_mesh_is_asked_on_opening_and_on_refresh(win, monkeypatch):
    asks = Asks(bm.BedMesh("thuis", 0.0, 0.0, 100.0, 100.0, np.asarray(PROBED)))
    monkeypatch.setattr(pl, "bed_mesh", asks)
    panel = opened(win, pl.Connection("Bed mesh test", "moonraker", "klipper.thuis"))
    try:
        assert asks.calls == 1, "opening the window asks once"
        assert panel.mesh is not None and panel.summary.text().endswith("profile thuis")
        panel.refresh.click()
        quiet(panel)
        assert asks.calls == 2, "Refresh asks again"
    finally:
        closed(panel)


def test_a_printer_that_is_still_answering_is_not_asked_again(win, monkeypatch):
    hold = threading.Event()
    asks = Asks(bm.BedMesh("thuis", probed=np.asarray(PROBED)), hold=hold)
    monkeypatch.setattr(pl, "bed_mesh", asks)
    panel = fresh(win, [pl.Connection("Langzaam", "moonraker", "trage.klipper.thuis")])
    try:
        assert asks.inside.wait(20000), "the window asked nothing at all"
        panel.refresh.click()
        panel.ask()
        panel.ask()
        assert asks.calls == 1, "a printer that is still answering is asked a second time"
        hold.set()
        quiet(panel)
        assert asks.calls == 1, "the asking that was on its way was itself enough"
    finally:
        closed(panel)


def test_nothing_is_asked_by_itself_and_a_window_that_comes_back_keeps_its_mesh(printer, win):
    klipper = printer("moonraker")
    klipper.answer(MOON_MESH, moon_mesh())
    conn = klipper.connection("Klipper")
    panel = opened(win, conn)
    try:
        assert len(asked(klipper)) == 1
        panel.close()
        panel.show()                                  # back on the screen, and still asked nothing
        assert len(asked(klipper)) == 1, "a mesh that is there is asked over again"
        assert panel.mesh is not None
        win.settings.printers = [conn]                # nothing changed: reload asks nothing
        panel.reload()
        assert len(asked(klipper)) == 1
    finally:
        closed(panel)


def test_the_window_follows_the_printers_under_preferences(printer, win):
    klipper = printer("moonraker")
    klipper.answer(MOON_MESH, moon_mesh())
    panel = opened(win, klipper.connection("Klipper"))
    try:
        assert panel.printer.count() == 1 and panel.mesh is not None
        win.settings.printers = []                    # the printer is gone from Preferences
        win.settings.save()
        panel.reload()
        assert panel.printer.count() == 0 and panel.needs.isVisible()
        assert panel.mesh is None and not panel.legend.isVisible()
        win.settings.printers = [klipper.connection("Klipper")]
        win.settings.save()
        panel.reload()
        assert panel.printer.count() == 1 and not panel.needs.isVisible()
        assert panel.mesh is None
        assert len(asked(klipper)) == 1, "reload asks nothing of a printer by itself"
        panel.refresh.click()
        quiet(panel)
        assert len(asked(klipper)) == 2 and panel.mesh is not None
    finally:
        closed(panel)


def test_another_printer_in_the_box_takes_the_mesh_of_the_other_one_away(printer, win):
    one, other = printer("moonraker"), printer("moonraker")
    one.answer(MOON_MESH, moon_mesh(profile_name="de ene"))
    other.answer(MOON_MESH, moon_mesh(profile_name="de andere"))
    panel = opened(win, one.connection("Ene"), other.connection("Andere"))
    try:
        assert panel.mesh.profile == "de ene"
        panel.printer.setCurrentIndex(1)
        quiet(panel)
        assert panel.mesh.profile == "de andere"
        assert panel.summary.text().endswith("profile de andere")
        assert len(asked(other)) == 1 and len(asked(one)) == 1
        panel.printer.setCurrentIndex(0)
        quiet(panel)
        assert panel.mesh.profile == "de ene"
        assert len(asked(one)) == 2, "a mesh that was taken away is asked again"
    finally:
        closed(panel)


def test_a_printer_that_answers_nothing_says_so_under_the_picture(printer, win):
    klipper = printer("moonraker")
    klipper.answer(MOON_MESH, moon_mesh())
    klipper.refuse(MOON_MESH, 500)
    conn = klipper.connection("Klipper")
    panel = fresh(win, [conn])
    quiet(panel)
    try:
        assert "error 500" in panel.error.text() and klipper.url in panel.error.text()
        assert panel.error.isVisible() and panel.error.styleSheet().startswith("color:")
        assert panel.mesh is None and not panel.legend.isVisible()
        assert panel.printer.currentData().name == "Klipper"
        assert not panel.grab().isNull(), "the window paints even with nothing to show"
        klipper.accept(MOON_MESH)                     # it answers again: Refresh brings the mesh
        panel.refresh.click()
        quiet(panel)
        assert panel.mesh is not None and not panel.error.text()
        assert not panel.error.isVisible() and panel.legend.isVisible()
    finally:
        closed(panel)


def test_a_bed_that_was_never_probed_says_so_and_is_no_error(printer, win):
    klipper = printer("moonraker")
    klipper.answer(MOON_MESH, moon_mesh(probed_matrix=[[]]))
    panel = opened(win, klipper.connection("Klipper"))
    try:
        assert panel.error.text() == bd.NO_MESH
        assert "Refresh" in panel.error.text()
        assert panel.error.styleSheet().startswith("color:"), "muted, like everything here"
        assert panel.mesh is None and not panel.legend.isVisible()
        assert not panel.summary.text() and not panel.status.text()
        assert not panel.view.grab().isNull(), "the empty bed says what to press"
        panel.refresh.click()
        quiet(panel)
        assert len(asked(klipper)) == 2, "Refresh tries a printer that had nothing again"
    finally:
        closed(panel)


def test_the_window_has_a_name_for_everything_it_paints(win):
    panel = fresh(win, [])
    try:
        assert panel.windowTitle() == "Bed Mesh"
        assert panel.view.accessibleName() and panel.legend.accessibleName()
        assert panel.refresh.toolTip() and panel.printer.toolTip()
    finally:
        closed(panel)

"""G-code for a GRBL laser: the order in which the paths are cut and the text itself.

`volura/gcode.py` is the core without Qt. `cut_order` puts the paths of one layer in the
order the laser should follow them (deepest nesting first, so a part never drops out
before its holes are cut, and within one depth the nearest path next), and `grbl_laser`
writes the G-code of the worksheet on top of that order, with the power and the speed of
every layer. `vector2d.job_estimate` counts its rapid moves over the same order, so the
estimate and the G-code stand for one and the same job.

Run with:  python -m pytest tests/test_gcode.py
"""
import math
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from volura import __version__  # noqa: E402
from volura.gcode import cut_order, grbl_laser  # noqa: E402
from volura.vector2d import DEFAULT_TRAVEL_SPEED, Drawing, job_estimate  # noqa: E402


# ------------------------------------------------------------------ helpers

def rect(w, h, x=0.0, y=0.0):
    return np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=float)


def circle(cx, cy, r, n=72):
    t = np.linspace(0.0, 2.0 * math.pi, n, endpoint=False)
    return np.column_stack([cx + r * np.cos(t), cy + r * np.sin(t)])


def sheet(layer="Cut", *shapes, **params) -> Drawing:
    """An empty worksheet whose default layers carry `params`; `shapes` go on `layer`."""
    d = Drawing.new_sheet()
    for lay in d.layers.values():
        lay.params.update(params)
    for pts in shapes:
        d.add_path(pts, closed=True, layer=layer, src=1)
    return d


def width(path) -> float:
    """The width of a path, to tell the squares of a test apart after reordering."""
    p = np.asarray(path.pts, dtype=float)
    return float(p[:, 0].max() - p[:, 0].min())


def words(line) -> dict:
    """The words of a G-code line, as {letter: value}."""
    return {w[0]: float(w[1:]) for w in line.split() if w[0].isalpha()}


def moves(text) -> list[tuple[str, tuple[float, float], float | None]]:
    """The moves of a G-code text as ("G0"/"G1", (x, y), feed).

    A line without a G word continues the motion mode that is already on, so the modal
    moves inside a path are read back as G1 too; `feed` is None where the line has no F.
    """
    out, mode = [], "G1"
    for raw in text.splitlines():
        line = raw.split(";")[0].strip()
        if not line:
            continue
        w = words(line)
        if w.get("G") in (0.0, 1.0):
            mode = f"G{int(w['G'])}"
        if "X" in w and "Y" in w:
            out.append((mode, (w["X"], w["Y"]), w.get("F")))
    return out


def cut_points(text) -> set[tuple[float, float]]:
    """Every point the beam cuts, so the coordinates of the G1 moves."""
    return {xy for mode, xy, _f in moves(text) if mode == "G1"}


def rapid_targets(text) -> list[tuple[float, float]]:
    """The targets of the rapid moves with a feed, so without the run home at the end."""
    return [xy for mode, xy, feed in moves(text) if mode == "G0" and feed is not None]


# ------------------------------------------------------------------ the order of the paths

def test_the_island_and_its_hole_are_cut_before_the_outer_contour():
    d = sheet("Cut", rect(100, 100), circle(50, 50, 20), circle(50, 50, 5))
    assert [width(p) for p in cut_order(d.paths)] == pytest.approx([10.0, 40.0, 100.0])


def test_the_nearest_path_of_a_depth_is_cut_next():
    d = Drawing.new_sheet()
    d.add_path(rect(10, 10, 300), closed=True, layer="Cut")     # far from the origin
    d.add_path(rect(10, 10, 20), closed=True, layer="Cut")      # near
    assert [p.pts[:, 0].min() for p in cut_order(d.paths)] == [20.0, 300.0]


def test_an_open_line_is_reversed_when_its_end_lies_nearer():
    d = Drawing.new_sheet()
    d.add_path([[0, 100], [0, 10]], layer="Engrave")
    (path,) = cut_order(d.paths)
    assert path.pts[0].tolist() == [0.0, 10.0]
    assert path.pts[-1].tolist() == [0.0, 100.0]


def test_a_closed_loop_starts_at_the_vertex_nearest_the_head():
    d = Drawing.new_sheet()
    # corners (-20, -10), (-10, -10), (-10, 0), (-20, 0): (-10, 0) is nearest, third in line
    d.add_path(rect(10, 10, -20, -10), closed=True, layer="Cut")
    (path,) = cut_order(d.paths)
    assert path.pts[0].tolist() == [-10.0, 0.0]
    assert path.pts[1].tolist() == [-20.0, 0.0]                # turned, not reversed
    assert path.closed


def test_cut_order_keeps_the_layer_and_leaves_the_input_as_it_was():
    d = sheet("Cut", rect(100, 100), circle(50, 50, 20))
    before = [p.pts.copy() for p in d.paths]
    order = cut_order(d.paths)
    assert all(np.array_equal(p.pts, q) for p, q in zip(d.paths, before))
    assert all(new is not old for new, old in zip(order, d.paths))
    assert all(p.layer == "Cut" and p.src == 1 for p in order)


# ------------------------------------------------------------------ the text for the machine

def test_the_text_opens_with_the_header_and_closes_with_the_footer():
    lines = grbl_laser(sheet("Cut", rect(10, 10))).splitlines()
    assert lines[0] == f"; Volura {__version__}"
    assert lines[1:3] == ["G21", "G90"]
    assert lines[-2:] == ["M5", "G0 X0 Y0"]


@pytest.mark.parametrize("dynamic, mode", ((True, "M4"), (False, "M3")))
def test_dynamic_power_is_M4_and_constant_power_is_M3(dynamic, mode):
    assert mode in grbl_laser(sheet("Cut", rect(10, 10)), dynamic=dynamic).splitlines()


def test_the_feed_of_a_path_is_the_speed_of_its_layer_in_mm_per_minute():
    text = grbl_laser(sheet("Cut", rect(10, 10), speed=12.5))
    g1 = [line for line in text.splitlines() if line.startswith("G1 ")]
    assert words(g1[0])["F"] == pytest.approx(12.5 * 60.0)
    assert words(g1[0])["G"] == 1.0


def test_the_power_is_the_percentage_of_the_layer_scaled_to_s_max():
    d = sheet("Cut", rect(10, 10), power=55.0)
    g1 = [line for line in grbl_laser(d).splitlines() if line.startswith("G1 ")]
    assert words(g1[0])["S"] == pytest.approx(550.0)
    g1 = [line for line in grbl_laser(d, s_max=100).splitlines() if line.startswith("G1 ")]
    assert words(g1[0])["S"] == pytest.approx(55.0)


def test_the_power_and_the_feed_stand_on_the_first_G1_of_a_path_only():
    text = grbl_laser(sheet("Cut", rect(10, 10), speed=20.0, power=100.0))
    assert len([line for line in text.splitlines() if line.startswith("G1 ")]) == 1
    assert sum(1 for mode, _xy, f in moves(text) if mode == "G1" and f is not None) == 1
    assert sum(1 for mode, _xy, _f in moves(text) if mode == "G1") == 4


def test_every_pass_runs_the_whole_layer_again():
    text = grbl_laser(sheet("Cut", rect(10, 10), passes=3))
    assert [line for line in text.splitlines() if line.startswith("; layer")] == [
        "; layer Cut, pass 1/3", "; layer Cut, pass 2/3", "; layer Cut, pass 3/3"]
    assert len(cut_points(text)) == 4
    assert sum(1 for mode, _xy, _f in moves(text) if mode == "G1") == 3 * 4


def test_a_closed_square_ends_on_the_point_where_it_began():
    text = grbl_laser(sheet("Cut", rect(10, 10)))
    cut = [xy for mode, xy, _f in moves(text) if mode == "G1"]
    assert len(cut) == 4
    assert cut[-1] == rapid_targets(text)[0]


def test_the_rapid_moves_travel_at_the_rapid_speed():
    text = grbl_laser(sheet("Cut", rect(10, 10), passes=2), travel_speed=80.0)
    assert {f for mode, _xy, f in moves(text) if mode == "G0" and f is not None} == {4800.0}


def test_only_the_layers_the_beam_follows_are_cut():
    d = Drawing.new_sheet()
    d.add_path(rect(10, 10), closed=True, layer="Cut")
    d.add_path(rect(10, 10, 20), closed=True, layer="Raster")
    d.add_path(rect(10, 10, 40), closed=True, layer="Mill outside")
    d.add_path(rect(10, 10, 60), closed=True, layer="Pocket")
    d.add_path(rect(10, 10, 80), closed=True, layer="Guides")
    assert cut_points(grbl_laser(d)) == {(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)}


def test_a_hidden_layer_goes_to_the_laser_neither():
    d = sheet("Cut", rect(10, 10))
    d.layers["Cut"].visible = False
    with pytest.raises(ValueError, match="nothing on the worksheet"):
        grbl_laser(d)


@pytest.mark.parametrize("layer", ("Raster", "Mill outside", "Pocket", "Guides"))
def test_a_layer_that_is_not_for_the_laser_gives_no_G_code_of_its_own(layer):
    d = Drawing.new_sheet()
    d.add_path(rect(10, 10), closed=True, layer=layer)
    with pytest.raises(ValueError, match="nothing on the worksheet"):
        grbl_laser(d)


def test_every_vertex_of_the_worksheet_comes_back_in_the_G_code():
    d = Drawing.new_sheet()
    d.add_path(rect(40, 30), closed=True, layer="Cut", src=1)
    d.add_path(circle(20, 20, 6), closed=True, layer="Cut", src=1)
    d.add_path([[70, 5], [90, 5]], layer="Engrave")
    text = grbl_laser(d)
    drawn = {xy for mode, xy, feed in moves(text) if mode == "G1" or feed is not None}
    wanted = {(round(float(x), 3), round(float(y), 3))
              for p in d.paths for x, y in np.asarray(p.pts, dtype=float)}
    assert wanted <= drawn, "the start of an open path is reached in rapid"
    assert drawn <= wanted


def test_the_layers_come_in_the_order_of_the_worksheet():
    d = Drawing.new_sheet()
    d.add_path(rect(10, 10), closed=True, layer="Engrave")
    d.add_path(rect(10, 10, 20), closed=True, layer="Cut")
    text = grbl_laser(d)
    comments = [line for line in text.splitlines() if line.startswith("; layer")]
    assert comments == [f"; layer {name}, pass 1/1" for name in d.layers
                        if name in ("Engrave", "Cut")]


def test_the_job_estimate_counts_the_same_rapid_moves_as_the_G_code():
    d = sheet("Cut", rect(100, 100), circle(50, 50, 20), rect(20, 20, 200))
    targets = [(0.0, 0.0)] + rapid_targets(grbl_laser(d))
    travel = sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(targets, targets[1:]))
    # the G-code stands in thousandths of a millimetre, so the sum may differ a little
    assert job_estimate(d, travel_speed=DEFAULT_TRAVEL_SPEED)["travel"] == pytest.approx(
        travel, abs=0.01)


# ------------------------------------------------------------------ nothing to cut

def test_an_empty_worksheet_is_an_error():
    with pytest.raises(ValueError, match="nothing on the worksheet"):
        grbl_laser(Drawing.new_sheet())


def test_a_drawing_without_paths_is_an_error():
    with pytest.raises(ValueError, match="nothing on the worksheet"):
        grbl_laser([])


# ------------------------------------------------------------------ a layer without a feed

def test_a_layer_without_speed_is_an_error_that_names_the_layer():
    d = sheet("Cut", rect(10, 10))
    d.layers["Cut"].params["speed"] = 0.0
    with pytest.raises(ValueError, match=r"layer Cut has no speed"):
        grbl_laser(d)


def test_a_layer_with_a_negative_speed_is_an_error_too():
    d = sheet("Engrave", rect(10, 10))
    d.layers["Engrave"].params["speed"] = -5.0
    with pytest.raises(ValueError, match=r"layer Engrave has no speed"):
        grbl_laser(d)


def test_a_layer_without_passes_is_an_error_that_names_the_layer():
    d = sheet("Cut", rect(10, 10))
    d.layers["Cut"].params["passes"] = 0
    with pytest.raises(ValueError, match=r"layer Cut has no passes"):
        grbl_laser(d)


def test_a_layer_the_beam_never_follows_may_stay_without_speed():
    """Only a layer that is cut needs a feed, so a silent layer is no error."""
    d = Drawing.new_sheet()
    d.add_path(rect(10, 10), closed=True, layer="Cut")
    for name in ("Mill outside", "Pocket", "Guides", "Raster"):
        d.layers[name].params["speed"] = 0.0
        d.layers[name].params["passes"] = 0
    grbl_laser(d)

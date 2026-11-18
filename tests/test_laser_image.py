"""An image to engrave: the modes, the dots and the rows of the raster.

`volura/laser_image.py` is the picture half of the laser controller: grey values in,
blocks of G-code out. What is tested here is the *meaning* of the modes, not their
appearance: a threshold burns the dark pixels and nothing else, a dither keeps the
average grey of the picture, a halftone burns a grid, and a row of pixels becomes a head
that travels with the beam off and burns with it on. A picture that engraves as a black
square because the S word was scaled wrong is the mistake to catch here.

Run with:  python -m pytest tests/test_laser_image.py
"""
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from volura import laser_image as li  # noqa: E402


def photo(rows=8, cols=8):
    """A picture that goes from nothing to everything, left to right."""
    return np.tile(np.linspace(0.0, 1.0, cols), (rows, 1))


# ------------------------------------------------------------------ grey
def test_black_burns_and_white_does_not():
    """What is dark on the screen is dark on the sheet: black is full power."""
    pixels = np.array([[[255, 255, 255], [0, 0, 0]]], dtype=np.uint8)
    assert np.allclose(li.to_gray(pixels), [[0.0, 1.0]])


def test_the_eye_sees_green_brighter_than_blue_so_green_burns_less():
    green = li.to_gray(np.array([[[0, 255, 0]]], dtype=np.uint8))[0, 0]
    blue = li.to_gray(np.array([[[0, 0, 255]]], dtype=np.uint8))[0, 0]
    assert green < blue


def test_a_flat_array_becomes_one_row_when_nothing_else_is_known():
    assert li.to_gray(np.array([0, 255], dtype=np.uint8)).shape == (1, 2)
    assert li.to_gray(np.array([0, 255, 255, 0], dtype=np.uint8), width=2).shape == (2, 2)


def test_brighter_is_brighter_and_invert_is_the_other_way_round():
    gray = np.array([[0.2, 0.5, 0.8]])
    assert np.all(li.adjust(gray, brightness=0.2) >= gray)
    assert np.allclose(li.adjust(gray, invert=True), 1.0 - gray)
    assert np.allclose(li.adjust(li.adjust(gray, invert=True), invert=True), gray)


def test_gamma_moves_the_middle_and_leaves_the_ends():
    gray = np.array([[0.0, 0.25, 1.0]])
    out = li.adjust(gray, gamma=2.0)
    assert out[0, 0] == 0.0 and out[0, 2] == 1.0
    assert out[0, 1] > 0.25


def test_nothing_ever_leaves_the_scale():
    gray = np.array([[0.0, 0.5, 1.0]])
    assert li.adjust(gray, brightness=5.0).max() == 1.0
    assert li.adjust(gray, brightness=-5.0).min() == 0.0
    assert li.adjust(gray, contrast=9.0).max() == 1.0


# ------------------------------------------------------------------ the modes
def test_threshold_burns_the_pixels_that_ask_for_power():
    out = li.bitmap(np.array([[0.1, 0.4, 0.6, 0.9]]), "threshold")
    assert np.array_equal(out, [[0, 0, 1, 1]])


def test_a_gray_mode_keeps_the_values_because_the_beam_follows_them():
    out = li.bitmap(np.array([[0.1, 0.5, 1.0]]), "gray")
    assert np.allclose(out, [[0.1, 0.5, 1.0]])


def test_a_dither_makes_dots_and_keeps_the_average():
    gray = np.full((16, 16), 0.4)
    out = li.bitmap(gray, "dither", "floyd")
    assert set(np.unique(out)) <= {0.0, 1.0}
    assert abs(out.mean() - 0.4) < 0.06


def test_every_dither_keeps_the_average():
    gray = photo(16, 16)
    for kind in li.DITHERS:
        out = li.bitmap(gray, "dither", kind)
        assert abs(out.mean() - gray.mean()) < 0.08, kind


def test_every_matrix_spreads_all_of_the_error_and_only_forwards():
    """The weights add up to the divisor, and no tap points at a pixel already done."""
    for kind, (label, taps, total) in li.DITHERS.items():
        assert label, kind
        # Atkinson passes on only 6/8 of the error on purpose: that keeps light areas clean
        want = 6 if kind == "atkinson" else total
        assert sum(w for _dy, _dx, w in taps) == want, kind
        for dy, dx, _w in taps:
            assert dy > 0 or (dy == 0 and dx > 0), (kind, dy, dx)


def test_floyd_steinberg_is_the_classic_matrix():
    _label, taps, total = li.DITHERS["floyd"]
    assert sorted(taps) == sorted(((0, 1, 7), (1, -1, 3), (1, 0, 5), (1, 1, 1)))
    assert total == 16


def test_the_old_name_of_burkes_still_works():
    gray = photo(8, 8)
    assert np.array_equal(li.dither(gray, "burrows"), li.dither(gray, "burkes"))


def test_a_dither_spreads_the_error_the_way_the_matrix_says():
    """One grey pixel next to nothing: Floyd-Steinberg gives 7/16 of it to the right."""
    gray = np.array([[0.4, 0.0], [0.0, 0.0]])
    # 0.4 is not burned: its error 0.4 goes 7/16 right, 5/16 down, 1/16 down-right
    out = li.dither(gray, "floyd")
    assert np.array_equal(out, np.zeros((2, 2)))
    gray = np.array([[0.4, 0.4]])
    assert np.array_equal(li.dither(gray, "floyd"), [[0.0, 1.0]])   # 0.4 + 0.175 >= 0.5


def test_a_big_picture_dithers_in_reasonable_time():
    import time
    gray = np.random.default_rng(1).random((300, 300))
    start = time.perf_counter()
    out = li.dither(gray, "stucki")
    assert time.perf_counter() - start < 5.0
    assert abs(out.mean() - gray.mean()) < 0.02


def test_a_halftone_burns_a_grid_and_not_a_stain():
    out = li.bitmap(np.full((8, 8), 0.5), "halftone", "4x4")
    assert out.sum() == pytest.approx(32, abs=2)
    assert out[0, 0] != out[0, 1]


def test_a_halftone_of_every_size_holds_its_place():
    for size in li.HALFTONES:
        out = li.bitmap(np.full((8, 8), 0.25), "halftone", size)
        assert 0.1 < out.mean() < 0.45, size


def test_full_power_is_always_burned_and_none_is_never_burned():
    for mode in ("threshold", "dither", "halftone"):
        assert li.bitmap(np.ones((4, 4)), mode).sum() == 16, mode
        assert li.bitmap(np.zeros((4, 4)), mode).sum() == 0, mode


def test_the_modes_are_the_ones_the_choice_box_shows():
    assert [key for key, _ in li.MODES] == ["gray", "threshold", "halftone", "dither"]
    assert all(label for _key, label in li.MODES)


# ------------------------------------------------------------------ the rows
def test_the_S_word_is_the_power_of_the_maximum_of_the_board():
    assert li.s_value(1.0, 100, 1000) == 1000
    assert li.s_value(1.0, 50, 1000) == 500
    assert li.s_value(0.5, 100, 255) == 128
    assert li.s_value(0.0, 100, 1000) == 0


def test_a_faint_pixel_still_gets_a_beam():
    assert li.s_value(0.001, 1, 255) == 1


def test_a_row_is_stretches_of_one_power():
    row = np.array([0.0, 1.0, 1.0, 0.0, 0.5])
    assert li.runs(row, 100, 1000) == [(1, 3, 1000), (4, 5, 500)]


def test_an_empty_row_is_nothing_to_burn():
    assert li.runs(np.zeros(6)) == []
    assert li.row_lines(np.zeros(6), 0.0) == []


def test_a_row_travels_to_the_first_dot_with_the_beam_off():
    lines = li.row_lines(np.array([0.0, 1.0, 1.0]), 2.0, x0=0.0, dot=0.1, feed=2000,
                         s_max=255)
    assert lines == ["G0 X0.100 Y2.000 S0", "G1 X0.300 Y2.000 F2000 S255"]


def test_a_stretch_of_one_power_is_one_block_and_not_one_per_pixel():
    lines = li.row_lines(np.ones(500), 0.0, dot=0.1)
    assert len(lines) == 2
    assert lines[1].startswith("G1 X50.000")


def test_stretches_side_by_side_need_no_travel_in_between():
    lines = li.row_lines(np.array([1.0, 0.5, 0.0, 1.0]), 0.0, dot=1.0)
    assert lines == ["G0 X0.000 Y0.000 S0", "G1 X1.000 Y0.000 F2000 S1000",
                     "G1 X2.000 Y0.000 F2000 S500",
                     "G0 X3.000 Y0.000 S0", "G1 X4.000 Y0.000 F2000 S1000"]


def test_a_row_never_switches_the_beam_off():
    """An M5 in a row leaves every row after it dark: only S0 may say 'no burn'."""
    assert "M5" not in li.row_lines(np.array([1.0, 0.0, 1.0]), 0.0)


def test_a_row_backwards_burns_from_the_right():
    forward = li.row_lines(np.array([1.0, 0.0, 1.0]), 0.0, dot=1.0)
    back = li.row_lines(np.array([1.0, 0.0, 1.0]), 0.0, dot=1.0, forward=False)
    assert back[0] == "G0 X3.000 Y0.000 S0"
    assert back[1] == "G1 X2.000 Y0.000 F2000 S1000"
    assert forward[0] == "G0 X0.000 Y0.000 S0"


def test_the_picture_starts_at_the_top_and_ends_at_the_origin():
    lines = li.engrave(np.ones((2, 2)), dot=1.0, feed=1000).splitlines()
    assert lines[0].startswith("; Volura")
    assert lines[1:4] == ["G21", "G90", "M3"]
    burns = [line for line in lines if line.startswith("G1")]
    assert "Y1.000" in burns[0] and "Y0.000" in burns[-1]
    assert lines[-3:] == ["S0", "M5", "G0 X0.000 Y0.000"]


def test_every_row_of_the_picture_burns():
    """The beam is switched on once and off once: every row has its G1 with power."""
    text = li.engrave(np.ones((5, 3)), dot=1.0, feed=1000)
    lines = text.splitlines()
    assert lines.count("M5") == 1 and lines.index("M5") > lines.index("M3")
    burns = [line for line in lines if line.startswith("G1")]
    assert len(burns) == 5 and all("S1000" in b for b in burns)


def test_the_rows_alternate_unless_one_way_is_asked():
    both = li.engrave(np.ones((2, 2)), dot=1.0, feed=2000).splitlines()
    one = li.engrave(np.ones((2, 2)), dot=1.0, feed=2000, one_way=True).splitlines()
    assert "G0 X2.000 Y0.000 S0" in both and "G0 X0.000 Y0.000 S0" in one


def test_the_board_that_follows_the_feed_gets_M4():
    assert "M4" in li.engrave(np.ones((1, 2)), dynamic=True)
    assert "M4" not in li.engrave(np.ones((1, 2)), dynamic=False)


def test_a_picture_is_burned_as_often_as_asked():
    once = li.engrave(np.ones((1, 2)), dot=1.0)
    twice = li.engrave(np.ones((1, 2)), dot=1.0, passes=2)
    assert twice.count("G1") == 2 * once.count("G1")


def test_the_power_of_the_beam_is_the_power_of_the_picture():
    text = li.engrave(np.ones((1, 2)), power=40, s_max=1000)
    assert "S400" in text


def test_nothing_to_engrave_says_so():
    with pytest.raises(ValueError):
        li.engrave(np.zeros((0, 0)))


def test_the_estimate_knows_the_size_and_the_time():
    info = li.estimate(np.ones((10, 20)), dot=0.1, feed=1200)
    assert info["width"] == pytest.approx(2.0)
    assert info["height"] == pytest.approx(1.0)
    assert info["rows"] == 10 and info["columns"] == 20
    assert info["cut_mm"] == pytest.approx(20.0)      # ten rows of two millimetres
    assert info["seconds"] == pytest.approx((info["cut_mm"] + info["travel_mm"]) / 20.0)


def test_a_picture_that_is_all_black_burns_nothing_at_all():
    black = li.estimate(np.zeros((10, 20)), dot=0.1, feed=1200)
    white = li.estimate(np.ones((10, 20)), dot=0.1, feed=1200)
    assert black["cut_mm"] == 0.0
    assert white["cut_mm"] > black["cut_mm"]


# ------------------------------------------------------------------ the size on the sheet
def test_resample_gives_one_value_per_dot():
    out = li.resample(photo(100, 200), 20.0, 0.1)
    assert out.shape == (100, 200)
    out = li.resample(photo(100, 200), 10.0, 0.1)
    assert out.shape == (50, 100)                    # the shape of the picture is kept


def test_resample_to_a_given_height():
    assert li.resample(photo(10, 10), 5.0, 0.5, height_mm=2.0).shape == (4, 10)


def test_shrinking_keeps_the_grey_of_the_picture():
    gray = np.random.default_rng(3).random((90, 120))
    out = li.resample(gray, 4.0, 0.1)                # 40 columns out of 120
    assert out.shape == (30, 40)
    assert out.mean() == pytest.approx(gray.mean(), abs=1e-9)
    assert out[0, 0] == pytest.approx(gray[:3, :3].mean())


def test_enlarging_stays_between_the_pixels_it_comes_from():
    gray = np.array([[0.0, 1.0]])
    out = li.resample(gray, 8.0, 1.0)
    assert out.shape == (4, 8)
    assert out.min() >= 0.0 and out.max() <= 1.0
    assert np.all(np.diff(out[0]) >= 0.0)            # from dark to light, never back


def test_resample_refuses_a_size_of_nothing():
    with pytest.raises(ValueError):
        li.resample(photo(), 0.0, 0.1)
    with pytest.raises(ValueError):
        li.resample(photo(), 10.0, 0.0)

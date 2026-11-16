"""An image to engrave: the modes of LightBurn, and the raster the machine reads.

No Qt in this file. An image is a grid of grey values and a laser is a pen with two things
to say: where the head is, and how hard the beam burns there. Everything between those two
is what this module decides, and the choices are the ones LightBurn has, because that is
the program everyone with a hobby laser has used at least once.

**The mode** (`MODES`) says what one pixel becomes:

* ``gray`` - the grey goes to the beam as it is, as the S word of the line it is in. Only
  good on a board that can follow an S word; on a board that switches, use a mode below.
* ``threshold`` - full power or nothing: the pixels at or over the level are burned,
  the rest is not. A logo cut from vinyl wants this, a photograph never does.
* ``dither`` - an error-diffusion matrix (`DITHERS`) turns grey into dots and spreads what
  each dot got wrong over the pixels that come after it. Floyd-Steinberg is the well-known
  one; Atkinson gives the error only to neighbours further away and so keeps white clean,
  which is why photo programs like it for light images; the others are its cousins with a
  different reach.
* ``halftone`` - an ordered matrix (`HALFTONES`, 4x4 or 8x8) burns a pixel when its grey is
  over the place it holds in the matrix. The dots lie in a grid and no error travels, so
  the picture stays calm - which is what a laser at one power likes best.

**The rows** are the scan lines of the raster: one row of pixels becomes one line of
travel and burn. The head travels to the first dot of a stretch with the beam off (a G0
carries no power in GRBL laser mode, and a beam that is on while travelling draws a line
that is not in the picture), burns the stretch, and travels on. `one_way` sends the head
back over the row instead of turning it around at the end: slower, but the return is dark
on a machine that cannot switch the beam off fast enough, so some people want it.

**The power** is the S word: a grey of 1.0 is the full power the user chose, and `s_max`
is the maximum of the board (`$30`), so the same picture is right on a board that counts
to 255 and on one that counts to 1000.
"""
from __future__ import annotations

import numpy as np

from . import __version__
from .i18n import tr

#: the modes of an image, in the order of the choice box: key, name
MODES = (
    ("gray", tr("Grayscale")),
    ("threshold", tr("Threshold")),
    ("halftone", tr("Halftone")),
    ("dither", tr("Dithering")),
)

#: the error-diffusion matrices, as LightBurn names them: every tap says how far the
#: error of a pixel travels (row, column, weight) and the sum divides it
DITHERS = {
    "floyd": (tr("Floyd-Steinberg"),
              ((0, 1, 7), (1, -1, 3), (1, 0, 5), (1, 1, 1)), 16),
    "atkinson": (tr("Atkinson"),
                 ((0, 1, 1), (0, 2, 1), (1, -1, 1), (1, 0, 1), (1, 1, 1), (2, 0, 1)), 8),
    "jarvis": (tr("Jarvis-Judice-Ninke"),
               ((1, -1, 7), (1, 0, 12), (1, 1, 7), (2, -2, 5), (2, -1, 7), (2, 0, 7),
                (2, 1, 5), (2, 2, 3), (3, -1, 2), (3, 0, 3), (3, 1, 2)), 48),
    "stucki": (tr("Stucki"),
               ((1, -1, 2), (1, 0, 4), (1, 1, 2), (2, -2, 1), (2, -1, 2), (2, 0, 4),
                (2, 1, 2), (2, 2, 1), (3, -2, 1), (3, -1, 2), (3, 0, 1)), 16),
    "burrows": (tr("Burrows"),
                ((0, -1, 2), (0, 0, 1), (0, 1, 2), (0, 2, 1), (1, -2, 1), (1, -1, 1),
                 (1, 0, 1), (1, 1, 1), (1, 2, 1)), 16),
    "sierra": (tr("Sierra"),
               ((0, 1, 5), (1, -1, 3), (1, 0, 2), (1, 1, 1), (2, 0, 2)), 16),
    "sierra_lite": (tr("Sierra Lite"),
                    ((0, 1, 2), (1, -1, 1), (1, 0, 1)), 4),
}

#: the ordered matrices: the grey a pixel must be over to be burned
HALFTONES = {
    "4x4": (np.array([[0, 8, 2, 10],
                      [12, 4, 14, 6],
                      [3, 11, 1, 9],
                      [15, 7, 13, 5]], dtype=float) + 0.5) / 16.0,
    "8x8": (np.array([
        [0, 32, 8, 40, 2, 34, 10, 42],
        [48, 16, 56, 24, 50, 18, 58, 26],
        [12, 44, 4, 36, 14, 46, 6, 38],
        [60, 28, 52, 20, 62, 30, 54, 22],
        [3, 35, 11, 43, 1, 33, 9, 41],
        [51, 19, 59, 27, 49, 17, 57, 25],
        [15, 47, 7, 39, 13, 45, 5, 37],
        [63, 31, 55, 23, 61, 29, 53, 21]], dtype=float) + 0.5) / 64.0,
}


def _num(value) -> str:
    """A coordinate the way GRBL reads it: three decimals of a millimetre."""
    return f"{float(value):.3f}"


# --------------------------------------------------------------------------
#  from pixels to grey, from grey to dots
# --------------------------------------------------------------------------

def to_gray(pixels, width=None) -> np.ndarray:
    """The grey values of an image, 0.0 (the beam stays off) to 1.0 (full power).

    A white pixel is therefore burned and a black one is not, which is what a CO2 laser
    engraving a dark sheet wants; the "Invert" of the window is the other way round.


    `pixels` is what a file or a screen holds: bytes or a flat array, with three or four
    values per pixel (RGB, RGBA) or one. The eye does not see the colours the same, so the
    grey is the Rec. 601 mix every scanner and every video chip uses.
    """
    data = np.asarray(pixels)
    if data.ndim == 1:
        if not width:
            data = data.reshape(1, -1)
        else:
            data = data.reshape(-1, int(width))
    if data.ndim == 3:
        channels = data.shape[2]
        if channels in (3, 4):
            rgb = data[:, :, :3].astype(float)
            data = rgb[:, :, 0] * 0.299 + rgb[:, :, 1] * 0.587 + rgb[:, :, 2] * 0.114
        else:
            data = data[:, :, 0]
    return np.clip(np.asarray(data, dtype=float) / 255.0, 0.0, 1.0)


def adjust(gray, invert=False, brightness=0.0, contrast=1.0, gamma=1.0) -> np.ndarray:
    """The grey values as the user wants them: brighter, harder, or the other way round.

    Gamma moves the middle of the scale without touching black and white, contrast
    stretches around the middle, brightness adds; the order is what a photo program uses,
    so the same numbers do the same thing there. Inverting turns a photo into a stamp and
    is what a negative needs.
    """
    out = np.asarray(gray, dtype=float).copy()
    if gamma and float(gamma) != 1.0:
        out = np.power(np.clip(out, 0.0, 1.0), 1.0 / float(gamma))
    if float(contrast) != 1.0:
        out = (out - 0.5) * float(contrast) + 0.5
    if float(brightness):
        out = out + float(brightness)
    if invert:
        out = 1.0 - out
    return np.clip(out, 0.0, 1.0)


def dither(gray, kind="floyd") -> np.ndarray:
    """The grey values as dots, 0.0 or 1.0, with the error of every pixel spread around.

    A pixel is burned or not, and what that got wrong goes to the pixels that come after
    it in the shape of the matrix of `kind`. This is the one loop that cannot be a numpy
    expression: a pixel sees what happened to its neighbours.
    """
    _label, taps, total = DITHERS.get(str(kind), DITHERS["floyd"])
    out = np.asarray(gray, dtype=float).copy()
    rows, cols = out.shape
    for y in range(rows):
        for x in range(cols):
            old = out[y, x]
            on = 1.0 if old >= 0.5 else 0.0
            out[y, x] = on
            error = old - on
            if not error:
                continue
            for dy, dx, weight in taps:
                ny, nx = y + dy, x + dx
                if 0 <= ny < rows and 0 <= nx < cols:
                    out[ny, nx] += error * weight / float(total)
    return out


def halftone(gray, size="8x8") -> np.ndarray:
    """The grey values as a grid of dots: burned when the grey is over its place."""
    matrix = HALFTONES.get(str(size), HALFTONES["8x8"])
    gray = np.asarray(gray, dtype=float)
    rows, cols = gray.shape
    tile = np.tile(matrix, (rows // matrix.shape[0] + 1, cols // matrix.shape[1] + 1))
    return (gray > tile[:rows, :cols]).astype(float)


def threshold(gray, level=0.5) -> np.ndarray:
    """Full power or nothing: everything at or over `level` is burned."""
    return (np.asarray(gray, dtype=float) >= float(level)).astype(float)


def bitmap(gray, mode="dither", dither_kind="floyd", halftone_size="8x8",
           threshold_level=0.5) -> np.ndarray:
    """The grey values as the machine should burn them.

    Everything gives 0.0 and 1.0 per pixel except ``gray``, which keeps the values: that
    mode asks the beam to follow them, so it has no dots to make.
    """
    mode = str(mode)
    if mode == "threshold":
        return threshold(gray, threshold_level)
    if mode == "halftone":
        return halftone(gray, halftone_size)
    if mode == "gray":
        return np.clip(np.asarray(gray, dtype=float), 0.0, 1.0)
    return dither(gray, dither_kind)


# --------------------------------------------------------------------------
#  from dots to the rows the machine reads
# --------------------------------------------------------------------------

def s_value(value, power=100.0, s_max=1000) -> int:
    """The S word of a grey value: the chosen power of the layer, of the maximum of the board."""
    value = float(value)
    if value <= 0.0:
        return 0
    return max(1, int(round(value * float(power) * float(s_max) / 100.0)))


def runs(row, power=100.0, s_max=1000) -> list:
    """The stretches of one row that burn with the same power: ``(first, last, S)``.

    `last` is the first pixel after the stretch, so a run of one pixel at 0 is (0, 1, S).
    Pixels that burn with nothing are no run at all: the head travels over them.
    """
    out = []
    start = None
    current = 0
    for x, value in enumerate(np.asarray(row, dtype=float)):
        s = s_value(value, power, s_max)
        if start is None:
            if s:
                start, current = x, s
            continue
        if s != current:
            out.append((start, x, current))
            start, current = (x, s) if s else (None, 0)
    if start is not None:
        out.append((start, len(row), current))
    return out


def row_lines(row, y, x0=0.0, dot=0.1, feed=2000.0, power=100.0, s_max=1000,
              forward=True) -> list:
    """One row of the picture as blocks: travel to a stretch, burn it, travel on.

    The head comes to a stretch with the beam off, so the blank between two stretches
    draws nothing, and it leaves it with `M5` so a row that ends in the middle of the
    sheet does not burn while the head moves to the next row.
    """
    found = runs(row, power, s_max)
    if not found:
        return []
    if not forward:
        found = list(reversed(found))
    lines = []
    for first, last, s in found:
        begin = float(x0) + (first if forward else last) * float(dot)
        lines.append(f"G0 X{_num(begin)} Y{_num(y)} F{float(feed):g}")
        steps = range(first + 1, last + 1) if forward else range(last - 1, first - 1, -1)
        for i, x in enumerate(steps):
            px = float(x0) + x * float(dot)
            if i == 0:
                lines.append(f"G1 X{_num(px)} Y{_num(y)} F{float(feed):g} S{s}")
            else:
                lines.append(f"X{_num(px)}")
    lines.append("M5")
    return lines


def engrave(dots, dot=0.1, origin=(0.0, 0.0), feed=2000.0, power=100.0, s_max=1000,
            one_way=False, dynamic=False, passes=1, title="") -> str:
    """The dots as G-code for a GRBL laser: the whole text, ready to send or save.

    `dots` is what :func:`bitmap` gives: rows of values, the first row the top of the
    picture. The picture is engraved **from its top downwards in Y**, so what is on the
    screen is what is on the sheet when the origin of the work is the bottom left of the
    picture; `origin` is where that bottom left is.

    One row of pixels is one line of the raster, `dot` millimetres wide. `passes` burns
    the whole picture again that many times, which is how a deeper engraving is made
    without slowing the beam down to where it stalls.
    """
    dots = np.atleast_2d(np.asarray(dots, dtype=float))
    rows, cols = dots.shape
    if not rows or not cols:
        raise ValueError(tr("There is nothing to engrave."))
    dot = float(dot)
    x0, y0 = float(origin[0]), float(origin[1])
    height = rows * dot
    lines = [f"; Volura {__version__}"]
    if title:
        lines.append(f"; {title}")
    lines += ["G21", "G90", "M4" if dynamic else "M3", "S0"]
    for _pass in range(max(1, int(passes))):
        for row in range(rows):
            y = y0 + (height - (row + 1) * dot)
            forward = one_way or row % 2 == 0
            lines += row_lines(dots[row], y, x0, dot, feed, power, s_max, forward)
    lines.append(f"G0 X{_num(x0)} Y{_num(y0)}")
    return "\n".join(lines) + "\n"


def estimate(dots, dot=0.1, feed=2000.0, one_way=False, passes=1) -> dict:
    """What engraving this picture costs: how big it is and how long it takes.

    The head burns the stretches and travels over the blanks between them; the time is
    both divided by the feed, the same kind of guess as the estimate of a contour job.
    What it does not know is the board: a machine that slows down in a corner or that
    waits for its buffer takes longer than the numbers here say.
    """
    dots = np.atleast_2d(np.asarray(dots, dtype=float))
    rows, cols = dots.shape
    dot = float(dot)
    passes = max(1, int(passes))
    cut = travel = 0.0
    head = 0.0
    for _pass in range(passes):
        for i, row in enumerate(dots):
            found = runs(row)
            forward = one_way or i % 2 == 0
            if not forward:
                found = list(reversed(found))
            if not found:
                travel += dot
                continue
            travel += abs((found[0][0] if forward else found[0][1]) * dot - head)
            for first, last, _s in found:
                cut += (last - first) * dot
            for here, there in zip(found, found[1:]):
                travel += (there[0] - here[1]) * dot
            head = (found[-1][1] if forward else found[-1][0]) * dot
            travel += dot
    return {"width": cols * dot, "height": rows * dot, "rows": rows, "columns": cols,
            "cut_mm": cut, "travel_mm": travel,
            "seconds": (cut + travel) / (float(feed) / 60.0)}

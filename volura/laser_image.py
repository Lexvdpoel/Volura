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
               ((0, 1, 7), (0, 2, 5),
                (1, -2, 3), (1, -1, 5), (1, 0, 7), (1, 1, 5), (1, 2, 3),
                (2, -2, 1), (2, -1, 3), (2, 0, 5), (2, 1, 3), (2, 2, 1)), 48),
    "stucki": (tr("Stucki"),
               ((0, 1, 8), (0, 2, 4),
                (1, -2, 2), (1, -1, 4), (1, 0, 8), (1, 1, 4), (1, 2, 2),
                (2, -2, 1), (2, -1, 2), (2, 0, 4), (2, 1, 2), (2, 2, 1)), 42),
    "burkes": (tr("Burkes"),
               ((0, 1, 8), (0, 2, 4),
                (1, -2, 2), (1, -1, 4), (1, 0, 8), (1, 1, 4), (1, 2, 2)), 32),
    "sierra": (tr("Sierra"),
               ((0, 1, 5), (0, 2, 3),
                (1, -2, 2), (1, -1, 4), (1, 0, 5), (1, 1, 4), (1, 2, 2),
                (2, -1, 2), (2, 0, 3), (2, 1, 2)), 32),
    "sierra_lite": (tr("Sierra Lite"),
                    ((0, 1, 2), (1, -1, 1), (1, 0, 1)), 4),
}

#: names a matrix had before, so a choice that was saved still finds its matrix
DITHER_ALIASES = {"burrows": "burkes"}

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
    """The grey values of an image as power: 0.0 (the beam stays off) to 1.0 (full power).

    A black pixel is burned and a white one is not, the way a picture is engraved on wood
    or card: what is dark on the screen is dark on the sheet. The "Invert" of the window
    is the other way round, for a light mark on a dark sheet (slate, anodised aluminium).

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
    return np.clip(1.0 - np.asarray(data, dtype=float) / 255.0, 0.0, 1.0)


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
    kind = DITHER_ALIASES.get(str(kind), str(kind))
    _label, taps, total = DITHERS.get(kind, DITHERS["floyd"])
    gray = np.atleast_2d(np.asarray(gray, dtype=float))
    rows, cols = gray.shape
    # plain lists of floats: a numpy scalar per pixel is many times slower than this
    depth = max(dy for dy, _dx, _w in taps)
    weights = [(dy, dx, w / float(total)) for dy, dx, w in taps]
    lines = [gray[y].tolist() for y in range(min(rows, depth + 1))]
    out = np.empty((rows, cols), dtype=float)
    for y in range(rows):
        row = lines[0]
        for x in range(cols):
            old = row[x]
            on = 1.0 if old >= 0.5 else 0.0
            row[x] = on
            error = old - on
            if error:
                for dy, dx, w in weights:
                    nx = x + dx
                    if 0 <= nx < cols and dy < len(lines):
                        lines[dy][nx] += error * w
        out[y] = row
        lines.pop(0)
        nxt = y + depth + 1
        if nxt < rows:
            lines.append(gray[nxt].tolist())
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


def resample(gray, width_mm, dot, height_mm=None) -> np.ndarray:
    """The picture at the size it gets on the sheet: one value per dot of the laser.

    `width_mm` divided by the `dot` is the number of columns; the rows follow from
    `height_mm` the same way, or keep the shape of the picture when it is not given.
    Shrinking averages every pixel that falls into a dot (a photo keeps its grey and does
    not alias); enlarging interpolates linearly between the pixels.
    """
    gray = np.atleast_2d(np.asarray(gray, dtype=float))
    rows, cols = gray.shape
    dot = float(dot)
    if dot <= 0 or float(width_mm) <= 0 or not rows or not cols:
        raise ValueError(tr("The picture needs a size and a dot bigger than nothing."))
    new_cols = max(1, int(round(float(width_mm) / dot)))
    if height_mm:
        new_rows = max(1, int(round(float(height_mm) / dot)))
    else:
        new_rows = max(1, int(round(new_cols * rows / float(cols))))
    return _resize_axis(_resize_axis(gray, new_rows, 0), new_cols, 1)


def _resize_axis(a, n, axis) -> np.ndarray:
    """`a` with `n` samples along `axis`: averaged areas when fewer, linear when more."""
    a = np.moveaxis(np.asarray(a, dtype=float), axis, 0)
    m = a.shape[0]
    shape = (-1,) + (1,) * (a.ndim - 1)
    if n == m:
        out = a.copy()
    elif n < m:
        # every new sample is the mean of the stretch of old samples it covers
        edges = np.linspace(0.0, float(m), n + 1)
        cum = np.concatenate([np.zeros((1,) + a.shape[1:]), np.cumsum(a, axis=0)])

        def integral(t):
            i = np.minimum(np.floor(t).astype(int), m - 1)
            return cum[i] + (t - i).reshape(shape) * a[i]
        out = (integral(edges[1:]) - integral(edges[:-1])) * (n / float(m))
    else:
        centres = np.clip((np.arange(n) + 0.5) * m / float(n) - 0.5, 0.0, m - 1.0)
        lo = np.floor(centres).astype(int)
        hi = np.minimum(lo + 1, m - 1)
        frac = (centres - lo).reshape(shape)
        out = a[lo] * (1.0 - frac) + a[hi] * frac
    return np.moveaxis(out, 0, axis)


def image_frame(pts):
    """Where a picture on the worksheet lies: its bottom left corner and its two sides.

    `pts` are the four points of the rectangle of the path, in the order of a picture:
    bottom left, bottom right, top right, top left. The sides are the vectors from the
    bottom left to its neighbours, so a rotated or a mirrored picture carries its
    rotation and its mirror in them.
    """
    pts = np.asarray(pts, dtype=float).reshape(-1, 2)
    origin = pts[0].astype(float)
    return origin, pts[1] - pts[0], pts[3] - pts[0]


def sample_on_grid(gray, pts, dot):
    """The picture as it lies on the sheet, on an axis-aligned grid of `dot` millimetres.

    The laser only knows rows that run along X, so a picture that was rotated on the
    worksheet is first sampled onto a grid that does: the bounding box of the four
    points is cut into squares of `dot`, and the centre of every square is asked what
    pixel of the picture it shows. A square outside the picture gets 0.0 - nothing
    burns there - which is what makes a rotated picture keep its corners clean.

    Returns the grid, and the bottom left of the box it stands on. Row 0 of the grid is
    the **top** of the box, the way :func:`engrave` wants it.
    """
    gray = np.atleast_2d(np.asarray(gray, dtype=float))
    rows, cols = gray.shape
    dot = float(dot)
    origin, ux, uy = image_frame(pts)
    pts = np.asarray(pts, dtype=float).reshape(-1, 2)
    x0, x1 = float(pts[:, 0].min()), float(pts[:, 0].max())
    y0, y1 = float(pts[:, 1].min()), float(pts[:, 1].max())
    ncols = max(1, int(np.ceil((x1 - x0) / dot))) if dot > 0 else 1
    nrows = max(1, int(np.ceil((y1 - y0) / dot))) if dot > 0 else 1
    grid_x, grid_y = np.meshgrid(x0 + (np.arange(ncols) + 0.5) * dot,
                                 y1 - (np.arange(nrows) + 0.5) * dot)
    out = np.zeros((nrows, ncols), dtype=float)
    det = float(ux[0] * uy[1] - uy[0] * ux[1])
    if not rows or not cols or abs(det) < 1e-12:
        return out, x0, y0
    dx, dy = grid_x - origin[0], grid_y - origin[1]
    u = (dx * uy[1] - dy * uy[0]) / det
    v = (dy * ux[0] - dx * ux[1]) / det
    inside = (u >= 0.0) & (u < 1.0) & (v >= 0.0) & (v < 1.0)
    col = np.clip((u * cols).astype(int), 0, cols - 1)
    row = np.clip(((1.0 - v) * rows).astype(int), 0, rows - 1)
    out[inside] = gray[row[inside], col[inside]]
    return out, x0, y0


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

    The head comes to a stretch with `G0 ... S0`, so the blank between two stretches
    draws nothing even on a board that is not in laser mode, and burns every stretch of
    one power with a single `G1`: one block per pixel would flood the small buffer of the
    board and make the head stutter. The beam stays switched on (M3/M4) for the whole
    picture - the S word alone says how hard it burns - so no row ends with M5, which
    would leave every row after it dark.
    """
    found = runs(row, power, s_max)
    if not found:
        return []
    if not forward:
        found = list(reversed(found))
    lines = []
    head = None
    for first, last, s in found:
        begin = float(x0) + (first if forward else last) * float(dot)
        end = float(x0) + (last if forward else first) * float(dot)
        if head is None or abs(head - begin) > 1e-9:
            lines.append(f"G0 X{_num(begin)} Y{_num(y)} S0")
        lines.append(f"G1 X{_num(end)} Y{_num(y)} F{float(feed):g} S{s}")
        head = end
    return lines


def engrave(dots, dot=0.1, origin=(0.0, 0.0), feed=2000.0, power=100.0, s_max=1000,
            one_way=False, dynamic=False, passes=1, title="", header=True) -> str:
    """The dots as G-code for a GRBL laser: the whole text, ready to send or save.

    `dots` is what :func:`bitmap` gives: rows of values, the first row the top of the
    picture. The picture is engraved **from its top downwards in Y**, so what is on the
    screen is what is on the sheet when the origin of the work is the bottom left of the
    picture; `origin` is where that bottom left is.

    One row of pixels is one line of the raster, `dot` millimetres wide. `passes` burns
    the whole picture again that many times, which is how a deeper engraving is made
    without slowing the beam down to where it stalls.

    `header` says whether the text stands on its own: without it the first lines (the
    name, the millimetres, the absolute coordinates and the switching of the beam) are
    left out, so the picture can go inside a job that already said all of that.
    """
    dots = np.atleast_2d(np.asarray(dots, dtype=float))
    rows, cols = dots.shape
    if not rows or not cols:
        raise ValueError(tr("There is nothing to engrave."))
    dot = float(dot)
    x0, y0 = float(origin[0]), float(origin[1])
    height = rows * dot
    lines = []
    if header:
        lines = [f"; Volura {__version__}"]
        if title:
            lines.append(f"; {title}")
        lines += ["G21", "G90", "M4" if dynamic else "M3", "S0"]
    for _pass in range(max(1, int(passes))):
        for row in range(rows):
            y = y0 + (height - (row + 1) * dot)
            forward = one_way or row % 2 == 0
            lines += row_lines(dots[row], y, x0, dot, feed, power, s_max, forward)
    lines += ["S0", "M5", f"G0 X{_num(x0)} Y{_num(y0)}"]
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

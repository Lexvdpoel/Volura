"""The mesh of a print bed: the numbers of it, and how they look.

No Qt in this file. `volura/printer_link.py` asks a Klipper printer for the mesh of its bed and
hands the answer over as a `BedMesh`, and Window > Bed Mesh… (volura/bed_mesh_dialog.py) paints
it. A mesh is what the probe of the printer touched: a grid of z heights over the x/y of the bed,
with the row of the lowest y at the front of the bed. Klipper interpolates that grid to a finer
one, which is what a print follows, so both are kept: the probed points to read numbers from and
the interpolated mesh to paint.

What a user came to look at is how flat the bed is: how much the highest and the lowest point
lie apart (`flatness`). The colours say where a point stands with respect to 0 - blue under it,
red above it - so one high side of a bed is told apart from one low side at a glance, and 0 is
always the white in between.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .i18n import tr

#: the colours of the heat map: a point under 0 slides to COLD, one above 0 to WARM
COLD = (33, 102, 172)
WARM = (178, 24, 43)
#: what 0 mm is: the middle of the map and the middle of the scale
MID = (255, 255, 255)
#: what stands where a printer named no profile
NOTHING = "—"


def no_grid(rows: int = 0, cols: int = 0) -> np.ndarray:
    """An empty grid: the shape of "there is nothing here"."""
    return np.zeros((rows, cols), dtype=float)


@dataclass
class BedMesh:
    """The mesh one printer has of its bed: what the probe touched and what Klipper made of it.

    `probed` is a 2D array of z heights in mm, one row per y and one column per x, with row 0 at
    the lowest y - the front of the bed; `mesh` is the interpolated grid a print follows and is
    empty when the printer did not say one. `x_min`, `y_min`, `x_max` and `y_max` bound the part
    of the bed the mesh covers, in mm, and `profile` is what the printer calls this mesh.
    `volura/printer_link.py` hands over either one of these or nothing at all.
    """

    profile: str = ""
    x_min: float = 0.0
    y_min: float = 0.0
    x_max: float = 0.0
    y_max: float = 0.0
    probed: np.ndarray = field(default_factory=no_grid)
    mesh: np.ndarray = field(default_factory=no_grid)

    def picture(self) -> np.ndarray:
        """What the heat map is painted with: the smooth mesh when there is one, and the probed
        points when the printer named none; a bed of one point is painted with that point."""
        return self.mesh if _whole(self.mesh) else self.probed

    def point(self, row, col):
        """The middle of probed cell (row, col) on the bed and the z that stands there:
        (x, y, z) in mm; None when there is no such cell."""
        grid = self.probed if _whole(self.probed) else no_grid()
        if not (0 <= row < grid.shape[0] and 0 <= col < grid.shape[1]):
            return None
        rows, cols = grid.shape
        x = self.x_min + (col + 0.5) * (self.x_max - self.x_min) / cols
        y = self.y_min + (row + 0.5) * (self.y_max - self.y_min) / rows
        return float(x), float(y), float(grid[row][col])


def _whole(grid) -> bool:
    """Is this a grid of cells to draw? An empty one, a row of none and a thing that is not a
    grid at all are not."""
    return (isinstance(grid, np.ndarray) and grid.ndim == 2 and grid.size > 0)


def finite(grid) -> np.ndarray:
    """Every number of a grid that is really there, in one line.

    An empty grid, a grid of nothing but NaN and a thing that is no grid at all come back empty,
    so a range or a flatness can always be asked of the result."""
    try:
        data = np.asarray(grid, dtype=float).ravel()
    except (TypeError, ValueError):
        return no_grid(0)
    return data[np.isfinite(data)]


def flatness(probed) -> float:
    """How much the highest and the lowest point of a bed lie apart (mm).

    Nothing to compare is no flatness at all: 0.0 for an empty grid, and for one that holds
    nothing but numbers that are not numbers."""
    got = finite(probed)
    return 0.0 if not got.size else float(got.max() - got.min())


def limits(grid) -> tuple:
    """The smallest and the biggest z of a grid, with 0 between them: what the heat map and the
    bar under it span. A grid with nothing in it spans nothing, which is 0 to 0."""
    got = finite(grid)
    if not got.size:
        return 0.0, 0.0
    return min(0.0, float(got.min())), max(0.0, float(got.max()))


def summary(mesh) -> str:
    """What the window says of a mesh under the picture: how flat the bed is, how big the probed
    grid is (`n` rows over y by `m` columns over x) and which profile of the printer it is.

    The range is said as the distance between the highest and the lowest point, with both points
    beside it; a mesh without points or without a name says 0 and a dash rather than nothing."""
    got = finite(mesh.probed)
    lo = hi = 0.0
    if got.size:
        lo, hi = float(got.min()), float(got.max())
    rows, cols = (mesh.probed.shape if _whole(mesh.probed) else (0, 0))
    return tr("Range {r} mm (from {lo} to {hi} mm), {n}×{m} points, profile {p}").format(
        r=f"{flatness(mesh.probed):.3f}", lo=f"{lo:.3f}", hi=f"{hi:.3f}",
        n=rows, m=cols, p=mesh.profile or NOTHING)


def colour(z, lo, hi) -> tuple:
    """The colour of one z of a mesh as (r, g, b) from 0 to 255: blue under 0, white at 0, red
    above it.

    The scale is as wide on both sides of 0 as the widest of `lo` and `hi`, so 0 is always the
    white in the middle and a point of 0 mm on a bed that stands high on one side is not painted
    as though it were low. What stands outside that span gets the colour of its end; a z that is
    no number, and a span of nothing at all, is white."""
    try:
        value, span = float(z), max(abs(float(lo)), abs(float(hi)))
    except (TypeError, ValueError):
        return MID
    if not math.isfinite(value) or not math.isfinite(span) or span <= 0:
        return MID
    part = max(-1.0, min(1.0, value / span))
    away = COLD if part < 0 else WARM
    near = abs(part)
    return tuple(int(round(one + (two - one) * near)) for one, two in zip(MID, away))

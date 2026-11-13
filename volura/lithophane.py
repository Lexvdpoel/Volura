"""Lithophanes: the computational core of a plate whose thickness follows the brightness of a photo.

No Qt in this file: it works on a numpy array of brightness values, so it can be tested on its own
and reused in the model and the export.

A **lithophane** is a thin plate that hides a photo in its thickness: where the picture is dark the
plastic is `thick` millimetres thick and where it is light it is `thin`, so with a light behind the
plate the photo appears. `gray` is the picture, a brightness of 0 for black and 1 for white for
every pixel, row 0 at the top of the image.

The picture is read on a grid of square cells of `pixel` millimetres, `width / pixel` of them
across the picture, and the height of the picture follows the aspect of the image. Every cell takes
the average brightness of the pixels that fall in it, so a photo of any size gives the same plate,
and the grid never grows past `GRID_MAX` cells on a side: a picture that would ask for more is read
with coarser cells.

The plastic between `thin` and `thick` follows the brightness on a straight scale and `invert`
swaps light and dark, for a negative of the photo. Every corner of the grid takes the mean of the
cells that meet in it, so the relief is a smooth surface and not a staircase. A flat frame of
`thick` millimetres plastic, `border` millimetres wide, stands round the picture, so the plate has
an edge that holds and the thin edge of the photo is not where it breaks.

The plate prints **standing up**, which is how a lithophane gets its quality: it lies in the x-z
plane with the top of the picture at the top, its flat back on `y = 0` and its relief growing
towards `-y`, the front the viewer looks at. It stands on `z = 0` with the bottom of that frame and
is centred in x. The mesh is built from the grid itself - a front surface of quads, one per cell,
split into triangles, a flat back and walls on the four sides - as one closed manifold.
"""
from __future__ import annotations

import math

import manifold3d as mf
import numpy as np

from .i18n import tr

#: mm the narrowest and the widest picture: small enough to print, big enough to read
WIDTH_MIN, WIDTH_MAX = 40.0, 250.0
#: mm the thinnest and the thickest plastic: white goes to `thin`, black to `thick`
THIN_MIN, THIN_MAX = 0.4, 2.0
THICK_MIN, THICK_MAX = 1.0, 6.0
#: mm the black has to stand out above the white, or the photo does not appear
THICK_ROOM = 0.4
#: mm the narrowest and the widest flat frame of thick plastic round the picture
BORDER_MIN, BORDER_MAX = 0.0, 10.0
#: mm the smallest and the largest cell of the grid: fine keeps the detail, coarse prints fast
PIXEL_MIN, PIXEL_MAX = 0.1, 1.0
#: the most cells the grid has on a side; a bigger picture is read with coarser cells
GRID_MAX = 1000


def _number(value, lo, hi):
    """A number of the plate, clamped to what prints; anything but a number is refused."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        v = math.nan
    if not math.isfinite(v):
        raise ValueError(tr("the sizes of a lithophane must be numbers"))
    return max(lo, min(hi, v))


def _brightness(gray) -> np.ndarray:
    """The picture as a 2D array of brightnesses between 0 and 1, row 0 at the top of the image."""
    try:
        b = np.asarray(gray, dtype=float)
    except (TypeError, ValueError):
        raise ValueError(tr("a lithophane needs a picture of brightness values"))
    if b.ndim != 2 or b.size == 0 or not np.isfinite(b).all():
        raise ValueError(tr("a lithophane needs a picture of brightness values"))
    return np.clip(b, 0.0, 1.0)


def _shape(shape) -> tuple:
    """The (rows, columns) of a picture, checked: the grid is made from them."""
    try:
        rows, cols = int(shape[0]), int(shape[1])
    except (TypeError, ValueError, IndexError, KeyError):
        raise ValueError(tr("a lithophane needs a picture of brightness values"))
    if rows < 1 or cols < 1:
        raise ValueError(tr("a lithophane needs a picture of brightness values"))
    return rows, cols


def grid_size(shape, width=100.0, pixel=0.2) -> tuple:
    """The (columns, rows) a picture of `shape` is read on.

    The cells are square and `pixel` millimetres wide, `width / pixel` of them across the picture,
    and the number of rows follows the aspect of the picture. When that would give more than
    `GRID_MAX` cells on a side the cells are made coarser instead, so the grid stays printable and
    the picture keeps its shape.
    """
    rows, cols = _shape(shape)
    W = _number(width, WIDTH_MIN, WIDTH_MAX)
    p = _number(pixel, PIXEL_MIN, PIXEL_MAX)
    p = max(p, W / GRID_MAX, W * rows / cols / GRID_MAX)
    return max(1, int(round(W / p))), max(1, int(round(W * rows / cols / p)))


def _sizes(shape, width, thin, thick, border, invert, pixel) -> dict:
    """The clamped numbers of the plate, the grid it is read on and the size it comes out.

    The black is never thinner than the white plus `THICK_ROOM` millimetres, or there is no photo
    to see, and the cell is made coarser when the grid would pass `GRID_MAX` cells on a side.
    `height` is how high the picture is, which follows its aspect, and `outside_width` and
    `outside_height` add the frame of `border` millimetres that stands round it.
    """
    rows, cols = _shape(shape)
    W = _number(width, WIDTH_MIN, WIDTH_MAX)
    t = _number(thin, THIN_MIN, THIN_MAX)
    T = _number(thick, max(THICK_MIN, t + THICK_ROOM), THICK_MAX)
    b = _number(border, BORDER_MIN, BORDER_MAX)
    p = _number(pixel, PIXEL_MIN, PIXEL_MAX)
    high = W * rows / cols
    p = max(p, W / GRID_MAX, high / GRID_MAX)
    return {"width": W, "thin": t, "thick": T, "border": b, "invert": bool(invert), "pixel": p,
            "height": high, "cols": max(1, int(round(W / p))), "rows": max(1, int(round(high / p))),
            "outside_width": W + 2.0 * b, "outside_height": high + 2.0 * b}


def sizes(shape, *, width=100.0, thin=0.8, thick=3.0, border=3.0, invert=False,
          pixel=0.2) -> dict:
    """The numbers a lithophane is made of, so a dialog can show them.

    `shape` is the shape of the picture, `(rows, columns)`; `width` is how wide the picture is
    printed and `height` - which the picture's aspect says - how high. `thin` is the plastic where
    the picture is white and `thick` where it is black, `border` how wide the flat frame round the
    picture is and `pixel` how big one cell of the grid is; `invert` swaps light and dark.
    `outside_width` and `outside_height` are the whole plate, frame included. Every number is
    clamped to what prints; anything but a number is a ValueError.
    """
    return _sizes(shape, width, thin, thick, border, invert, pixel)


def _integral(b) -> np.ndarray:
    """The running total of the picture: `I[i, j]` is the brightness of all pixels above-left."""
    I = np.zeros((b.shape[0] + 1, b.shape[1] + 1))
    I[1:, 1:] = np.cumsum(np.cumsum(b, axis=0), axis=1)
    return I


def _sample(I, xs, ys) -> np.ndarray:
    """The running total at every edge of the grid, interpolated between whole pixels.

    Between two pixels the running total is a bilinear surface, so interpolating its four corners
    gives the exact total of a picture made of flat pixels, however the grid is laid over it.
    """
    x0 = np.clip(np.floor(xs).astype(np.intp), 0, I.shape[1] - 2)
    y0 = np.clip(np.floor(ys).astype(np.intp), 0, I.shape[0] - 2)
    fx = (xs - x0)[None, :]
    fy = (ys - y0)[:, None]
    return (I[np.ix_(y0, x0)] * (1.0 - fx) * (1.0 - fy)
            + I[np.ix_(y0, x0 + 1)] * fx * (1.0 - fy)
            + I[np.ix_(y0 + 1, x0)] * (1.0 - fx) * fy
            + I[np.ix_(y0 + 1, x0 + 1)] * fx * fy)


def _resample(b, rows, cols) -> np.ndarray:
    """The picture read on a grid of `rows` by `cols`: every cell is the average of its pixels.

    A cell takes the brightness of every pixel that falls in it, weighted by how much of the pixel
    is inside, so a photo of any size comes out the same and nothing is aliased into stripes.
    """
    if b.shape == (rows, cols):
        return b.copy()
    I = _integral(b)
    xs = np.linspace(0.0, b.shape[1], cols + 1)
    ys = np.linspace(0.0, b.shape[0], rows + 1)
    s = _sample(I, xs, ys)
    area = np.diff(ys)[:, None] * np.diff(xs)[None, :]
    return (s[1:, 1:] - s[:-1, 1:] - s[1:, :-1] + s[:-1, :-1]) / area


def _thickness(b, g) -> np.ndarray:
    """The thickness of every cell of the picture, from the bottom of the plate up.

    Row 0 of the picture is its top, which is the top of the plate, so the rows come back the
    other way round from the resampling.
    """
    pic = _resample(b, g["rows"], g["cols"])
    if g["invert"]:
        pic = 1.0 - pic
    return (g["thick"] - pic * (g["thick"] - g["thin"]))[::-1]


def _nodes(H, g) -> np.ndarray:
    """The thickness at every corner of the grid: the mean of the cells that meet in it.

    A corner outside the picture takes the cell next to it as its own, so the plate ends on the
    thickness of its edge cell instead of leaning away from it. With a frame the edge of the
    picture stands as high as the frame, so the frame is flat `thick` plastic all the way over.
    """
    e = np.pad(H, 1, mode="edge")
    N = 0.25 * (e[:-1, :-1] + e[1:, :-1] + e[:-1, 1:] + e[1:, 1:])
    if g["border"] > 0.0:
        for edge in (N[0], N[-1], N[:, 0], N[:, -1]):
            edge[:] = g["thick"]
        N = np.pad(N, 1, constant_values=g["thick"])
    return N


def _edges(lo, hi, n, border) -> np.ndarray:
    """The places of the `n + 1` grid corners between `lo` and `hi`, with the frame's two added."""
    e = np.linspace(lo, hi, n + 1)
    if border > 0.0:
        e = np.concatenate(([lo - border], e, [hi + border]))
    return e


def _mesh(N, xs, zs) -> mf.Manifold:
    """The closed plate whose grid corners have the thicknesses of `N`.

    `N` gives the thickness at every corner of the grid, its rows from the bottom of the plate up
    and its columns from the left, and `xs` and `zs` are the places of those corners. The back is
    flat on `y = 0` and the front is the surface that runs over the thicknesses towards `-y`: every
    cell is one quad there, split into two triangles, and the four sides are walled from the edge
    of the front down to the edge of the back, so the whole is one closed surface.
    """
    nr, nc = N.shape
    front = np.empty((nr, nc, 3))
    front[:, :, 0] = xs
    front[:, :, 1] = -N
    front[:, :, 2] = zs[:, None]
    back = np.empty((nr, nc, 3))
    back[:, :, 0] = xs
    back[:, :, 1] = 0.0
    back[:, :, 2] = zs[:, None]
    verts = np.concatenate([front.reshape(-1, 3), back.reshape(-1, 3)])
    f = np.arange(nr * nc).reshape(nr, nc)
    k = nr * nc + f
    tris = [np.column_stack([f[:-1, :-1].ravel(), f[:-1, 1:].ravel(), f[1:, 1:].ravel()]),
            np.column_stack([f[:-1, :-1].ravel(), f[1:, 1:].ravel(), f[1:, :-1].ravel()]),
            np.column_stack([k[:-1, :-1].ravel(), k[1:, 1:].ravel(), k[:-1, 1:].ravel()]),
            np.column_stack([k[:-1, :-1].ravel(), k[1:, :-1].ravel(), k[1:, 1:].ravel()]),
            np.column_stack([f[0, :-1], k[0, 1:], f[0, 1:]]),
            np.column_stack([f[0, :-1], k[0, :-1], k[0, 1:]]),
            np.column_stack([f[-1, :-1], f[-1, 1:], k[-1, 1:]]),
            np.column_stack([f[-1, :-1], k[-1, 1:], k[-1, :-1]]),
            np.column_stack([f[:-1, 0], f[1:, 0], k[1:, 0]]),
            np.column_stack([f[:-1, 0], k[1:, 0], k[:-1, 0]]),
            np.column_stack([f[:-1, -1], k[1:, -1], f[1:, -1]]),
            np.column_stack([f[:-1, -1], k[:-1, -1], k[1:, -1]])]
    v = np.ascontiguousarray(verts, dtype=np.float32)
    t = np.ascontiguousarray(np.concatenate(tris), dtype=np.uint32)
    m = mf.Manifold(mf.Mesh(vert_properties=v, tri_verts=t))
    if m.status() != mf.Error.NoError or m.volume() <= 0.0:
        m = mf.Manifold(mf.Mesh(vert_properties=v, tri_verts=np.ascontiguousarray(t[:, ::-1])))
    return m


def lithophane(gray, *, width=100.0, thin=0.8, thick=3.0, border=3.0, invert=False,
               pixel=0.2) -> mf.Manifold:
    """One printable lithophane: a plate whose thickness is the brightness of a photo.

    `gray` is the picture, a brightness between 0 for black and 1 for white for every pixel and
    row 0 at the top of the image. The picture is printed `width` millimetres wide and as high as
    its aspect says, on a grid of square cells of `pixel` millimetres; every cell is as thick as
    its brightness says, `thin` for white and `thick` for black, and `invert` swaps the two. A flat
    frame of `thick` millimetres plastic, `border` wide, stands round the picture.

    The plate lies in the x-z plane with its flat back on `y = 0` and its relief towards `-y`, the
    top of the picture at the top, standing on `z = 0` with the bottom of that frame and centred in
    x, so it prints standing up. Every number is clamped to what prints and anything but a number
    is a ValueError.
    """
    b = _brightness(gray)
    g = _sizes(b.shape, width, thin, thick, border, invert, pixel)
    N = _nodes(_thickness(b, g), g)
    xs = _edges(-0.5 * g["width"], 0.5 * g["width"], g["cols"], g["border"])
    zs = _edges(g["border"], g["border"] + g["height"], g["rows"], g["border"])
    m = _mesh(N, xs, zs)
    if m.status() != mf.Error.NoError or m.is_empty() or len(m.decompose()) != 1:
        raise ValueError(tr("the lithophane is not a valid volume"))
    return m

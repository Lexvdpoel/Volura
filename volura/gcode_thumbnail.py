"""A picture of the print for the screen of the printer, in the G-code.

A printer with a screen - a Prusa MK4, XL or Mini, a Klipper printer behind Mainsail, Fluidd or
KlipperScreen, an OctoPrint with a thumbnail plugin - shows what is about to be printed when the
G-code carries a picture of it. PrusaSlicer's format is a comment block at the top of the file:

    ;
    ; thumbnail begin 300x300 12345
    ; iVBORw0KGgo...   (the base64 of a PNG, at most 78 characters behind "; ")
    ; ...
    ; thumbnail end
    ;

`render` makes that picture from the meshes of the print and their colours, `block` puts a PNG in
the comment block and `insert` writes the blocks behind the header of a G-code file.

Nothing here touches OpenGL: the picture is painted with QPainter into a QImage, which works off
the GUI thread, without a window and without a GL context, so a slice can make it in its
background job and the tests can check it.

The picture is the model as the window shows it: the same corner (see `volura.viewcube`), flat
shaded with one lamp and a little light from the room, painted back to front (the painter's
algorithm). Everything about it is done with numpy first and handed to Qt in one call per
triangle, so a model of two hundred thousand triangles is painted in well under two seconds.
"""
from __future__ import annotations

import base64
import os
import tempfile

import numpy as np
from PySide6.QtCore import QBuffer, QIODevice, QPointF, Qt
from PySide6.QtGui import QColor, QImage, QPainter, QPen

from .camera import VIEWS
from .i18n import tr, tr_noop
from .viewcube import basis

#: the view of the picture: the corner of `VIEWS["iso"]`, the front, a side and the top
YAW, PITCH = VIEWS["iso"]
#: the lamp that lights the model: in front of it, above it and a little to one side
LIGHT = np.array([0.15, -0.55, 0.82])
#: how much of a colour is always there (the light of the room) and at most how much the lamp
#: adds on top of that; the two together can go over 1, which is what makes a face that points
#: straight at the lamp come out in its full colour
AMBIENT, LAMP = 0.34, 0.86
#: how many brightness steps a colour is cut in: enough that a round surface shows no bands,
#: few enough that the colours of the print are prepared per mesh and not per triangle
STEPS = 64
#: the colour of a part that has none of its own: the grey the viewport paints it in
GREY = "#c8c8c8"
#: the part of the picture the model may not touch, so it never grows into the edge
MARGIN = 0.06
#: how thick the edge of a triangle is painted in its own colour: the antialiased edge of one
#: triangle covers about half a pixel, so an edge this thick makes sure that a pixel between two
#: triangles is painted by one of them completely - otherwise the picture shows a net of hairline
#: cracks of a few percent transparency, also in the middle of a ball of yarn
EDGE = 1.4
#: how many characters of base64 fit on one comment line, PrusaSlicer's own bound
LINE = 78
#: how much of the G-code is copied at a time by `insert`, so a big file is streamed
CHUNK = 1 << 20
#: the thumbnail choices of File > Slice (see the Extras group in `volura.slice_dialog`):
#: (label, the value in Settings, the sizes to draw). Off draws nothing, the rest is what the
#: screens of those printers want; PNG is understood by all of them, also the newest Prusa
#: firmware that would take a QOI of 440 x 240.
THUMBNAILS = [(tr_noop("Off"), "off", ()),
              (tr_noop("Small and large (48 × 48 + 300 × 300)"), "48x48/300x300",
               ((48, 48), (300, 300))),
              (tr_noop("Prusa (16 × 16 + 313 × 173 + 440 × 240)"), "16x16/313x173/440x240",
               ((16, 16), (313, 173), (440, 240))),
              (tr_noop("Klipper (32 × 32 + 300 × 300)"), "32x32/300x300", ((32, 32), (300, 300)))]
#: what a slice draws when nothing was chosen yet: the pair most screens read
DEFAULT = THUMBNAILS[1][1]


def sizes(value) -> list:
    """The sizes to draw for the value of the Thumbnail choice: `[(width, height)]`, and nothing
    for Off or for a value that names none of the choices."""
    for _label, kind, chosen in THUMBNAILS:
        if kind == str(value or ""):
            return list(chosen)
    return []


def _shades(colour) -> list:
    """The colour of a part in every brightness step, darkest first: what to paint a triangle
    with when its face catches that much of the lamp."""
    picked = QColor(str(colour or ""))
    # a part without a colour of its own is the grey of the viewport, not black
    rgb = (picked if picked.isValid() else QColor(GREY)).getRgb()[:3]
    out = []
    for k in range(STEPS):
        tone = QColor(*[max(0, min(255, int(round(c * (AMBIENT + LAMP * k / (STEPS - 1))))))
                        for c in rgb])
        out.append((tone, QPen(tone, EDGE)))
    return out


def _paint(painter, corners, tris, lamp, shades, size, flip):
    """One mesh of the model, flat shaded and painted back to front.

    `corners` are the corners of the model in the picture's own coordinates: x to the right, y
    up and z towards the eye, so a face that looks away from the eye has a normal with a
    negative z and is left out - what is behind it is hidden by the front of the model anyway.
    `size` and `flip` turn its x and y into pixels of the image, which counts y downwards.
    """
    face = corners[tris]                      # the corners of every triangle
    n = np.cross(face[:, 1] - face[:, 0], face[:, 2] - face[:, 0])
    keep = n[:, 2] > 0                        # the faces that look at the eye
    front, n = tris[keep], n[keep]
    if not len(front):
        return
    del face                                  # the corners are gathered, that was the big one
    ln = np.maximum(np.linalg.norm(n, axis=1), 1e-12)
    light = np.clip((n / ln[:, None]) @ lamp, 0.0, 1.0)  # how much of the lamp a face catches
    step = np.clip(np.rint(light * (STEPS - 1)), 0, STEPS - 1).astype(np.int32).tolist()
    # the moment each face reaches towards the eye: the far ones are painted first
    order = np.argsort(corners[:, 2][front].sum(axis=1), kind="stable").tolist()
    pixel = corners[:, :2] * size + flip      # the corners of the model in pixels of the image
    points = list(map(QPointF, pixel[:, 0].tolist(), pixel[:, 1].tolist()))
    corners_of = front.tolist()
    level = -1
    for k in order:
        if step[k] != level:                  # only touch the brush when the colour changes
            level = step[k]
            painter.setBrush(shades[level][0])
            painter.setPen(shades[level][1])
        a, b, c = corners_of[k]
        painter.drawConvexPolygon((points[a], points[b], points[c]))


def render(meshes_with_colours, width, height) -> bytes:
    """A PNG of the model: a list of ``(vertices, triangles, colour)`` painted in an isometric
    view from the same corner as the window looks from.

    The colours are the ``#rrggbb`` of the parts. The background stays transparent, so a printer
    screen puts the picture over its own background. Nothing is decimated: every triangle of
    every mesh is in the picture.
    """
    parts = [(np.asarray(v, dtype=float), np.asarray(t, dtype=np.int64), colour)
             for v, t, colour in meshes_with_colours if len(np.asarray(t))]
    if not parts:
        raise ValueError(tr("there is nothing to draw"))
    eye, right, up = basis(YAW, PITCH)
    turn = np.stack([right, up, eye], axis=1)       # world directions to picture directions
    mid = 0.5 * (np.min([v.min(axis=0) for v, _t, _c in parts], axis=0)
                 + np.max([v.max(axis=0) for v, _t, _c in parts], axis=0))
    placed = [(t, (v - mid) @ turn, c) for v, t, c in parts]
    low = np.min([q.min(axis=0) for _t, q, _c in placed], axis=0)
    high = np.max([q.max(axis=0) for _t, q, _c in placed], axis=0)
    room = np.array([width, height], dtype=float) * (1.0 - 2 * MARGIN)
    scale = float(np.min(room / np.maximum(high[:2] - low[:2], 1e-9)))
    middle = 0.5 * (low[:2] + high[:2]) * scale     # the model's middle, in picture pixels
    # the picture's x and y as the pixels of the image, which counts y downwards from the top
    size = np.array([scale, -scale])
    flip = np.array([0.5 * width - middle[0], 0.5 * height + middle[1]])
    img = QImage(width, height, QImage.Format_ARGB32_Premultiplied)
    img.fill(Qt.transparent)
    painter = QPainter(img)
    painter.setRenderHint(QPainter.Antialiasing)
    lamp = LIGHT @ turn                       # the lamp in the picture's own coordinates
    # one mesh after another, the one further from the eye first: parts that overlap in the
    # picture then lie the right way round, also when the model lists them the other way
    placed.sort(key=lambda item: item[1].mean(axis=0)[2])
    try:
        for tris, q, colour in placed:
            _paint(painter, q, tris, lamp, _shades(colour), size, flip)
    finally:
        painter.end()
    buf = QBuffer()
    buf.open(QIODevice.WriteOnly)
    img.save(buf, "PNG")
    return bytes(buf.data())


def block(png, width, height) -> str:
    """A PNG as the comment block a printer screen reads: the size and the length of the base64
    between `; thumbnail begin` and `; thumbnail end`, the picture itself in pieces of at most
    `LINE` characters, every line of the block a comment."""
    text = base64.b64encode(bytes(png)).decode("ascii")
    lines = [";", f"; thumbnail begin {width}x{height} {len(text)}"]
    lines += [f"; {text[i:i + LINE]}" for i in range(0, len(text), LINE)]
    lines += ["; thumbnail end", ";"]
    return "\n".join(lines) + "\n"


def _copy(src, dst) -> int:
    """The rest of `src` into `dst`, in pieces of `CHUNK`; how many bytes were copied."""
    copied = 0
    while True:
        piece = src.read(CHUNK)
        if not piece:
            return copied
        dst.write(piece)
        copied += len(piece)


def insert(path, blocks) -> None:
    """Put the blocks of `block()` behind the header comments of the G-code file at `path`.

    A printer that knows nothing about thumbnails skips the lines as the comments they are, so
    everything else in the file keeps its bytes: it is streamed into a temporary file next to it
    and replaced, never read into memory whole. Nothing happens with no blocks to put in.
    """
    blocks = [text for text in blocks if text]
    if not blocks:
        return
    head, rest, ends = [], b"", b"\n"
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(os.path.abspath(path)),
                               prefix=".volura-thumbnail-", suffix=".gcode")
    try:
        with open(path, "rb") as src, os.fdopen(fd, "wb") as dst:
            while True:
                line = src.readline()
                if not line or (line.strip() and not line.startswith(b";")):
                    rest = line            # the first line that is not a comment stays after them
                    break
                if line.endswith(b"\r\n"):
                    ends = b"\r\n"
                head.append(line)
            dst.write(b"".join(head))
            for text in blocks:
                dst.write(text.encode("ascii").replace(b"\n", ends))
            dst.write(rest)
            _copy(src, dst)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)

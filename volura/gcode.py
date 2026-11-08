"""G-code for a GRBL laser: the order in which to cut and the text for the machine.

No Qt in this file. The layers and paths of the worksheet (`volura/vector2d.py`) become
the plain text a hobby laser understands, so it can be tested on its own and reused by
the export in `volura/app.py`.

A **GRBL laser** is a diode or CO2 laser on a board with the GRBL firmware: it reads
G-code over the serial port and moves the head in millimetres (G21) with absolute
coordinates (G90). The beam is switched with the words of the spindle: **M3** puts it on
at a fixed power as soon as the head moves, **M4** follows the feed rate, so the beam
gives way where the head slows down in a corner and burns less there, and **M5** switches
it off. The power is the **S word**, a whole number up to the maximum of the controller
(`$30`: 1000 on a modern 32-bit board, 255 on an old 8-bit one), which is why `s_max` is
an input and not a constant.

The order in which the paths are cut is not free either. A part drops out of the sheet as
soon as its outer contour is cut loose, and then it floats under the head and is torn
about by the rest of the job; the holes are therefore cut **before** the outer contour,
and an island that stands in such a hole before that hole. That is the nesting depth of
`cam.nesting_depth`, from the deepest path to the outermost. Within one depth the head
takes the nearest path next, because every move between two paths is a rapid move that
costs time and cuts nothing.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from . import __version__
from .cam import nesting_depth
from .i18n import tr
from .vector2d import DEFAULT_TRAVEL_SPEED, Drawing, LASER_OPS, VPath, layer_param

#: decimals of a coordinate; GRBL works in thousandths of a millimetre
DECIMALS = 3


# --------------------------------------------------------------------------
#  helpers
# --------------------------------------------------------------------------

def _num(v) -> str:
    """A coordinate as the machine reads it: always the same number of decimals."""
    return f"{float(v):.{DECIMALS}f}"


def _laser_layers(drawing) -> list:
    """The layers the beam follows, in the order of the drawing.

    The visible layers with a laser operation other than "raster": the time of a raster
    fill follows from the line spacing and not from the layer, the mill layers have their
    own export and "none" is for information only. Those layers, and only those, are what
    `vector2d.job_estimate` times.
    """
    out = []
    for name, lay in drawing.layers.items():
        if not getattr(lay, "visible", True) or not any(p.layer == name for p in drawing.paths):
            continue
        if lay.op == "raster" or lay.op not in LASER_OPS:
            continue
        out.append(lay)
    return out


def _entry(path, pos) -> tuple[np.ndarray, float]:
    """The points of `path` as the head runs them when it comes from `pos`, and that distance.

    A closed loop may start at any vertex, so it is turned to the vertex nearest the head;
    an open path is run backwards when its end lies nearer than its start.
    """
    pts = np.asarray(path.pts, dtype=float)
    if len(pts) < 2:
        return pts, float(np.hypot(*(pts[0] - pos))) if len(pts) else 0.0
    if path.closed:
        d = np.hypot(pts[:, 0] - pos[0], pts[:, 1] - pos[1])
        i = int(np.argmin(d))
        return np.roll(pts, -i, axis=0), float(d[i])
    a, b = float(np.hypot(*(pts[0] - pos))), float(np.hypot(*(pts[-1] - pos)))
    return (np.ascontiguousarray(pts[::-1]), b) if b < a else (pts, a)


# --------------------------------------------------------------------------
#  the order of the paths
# --------------------------------------------------------------------------

def cut_order(paths) -> list[VPath]:
    """The order in which one layer is cut, as new paths; the input is left as it is.

    The deepest path is cut first, so an island goes before the hole around it and that
    hole before the outer contour: a part never drops out of the sheet before its holes
    are cut (`cam.nesting_depth`). Within one depth the path nearest to the head is cut
    next, from the current position, which starts at the origin. A closed loop starts at
    its vertex nearest the head and an open path is reversed when that shortens the run,
    so the head does not travel further than it must.
    """
    items = list(paths.paths if isinstance(paths, Drawing) else paths)
    depths = nesting_depth(items)
    out: list[VPath] = []
    pos = np.zeros(2)
    for depth in sorted(set(depths), reverse=True):
        rest = [p for p, d in zip(items, depths) if d == depth]
        while rest:
            runs = [_entry(p, pos) for p in rest]
            i = int(np.argmin([d for _pts, d in runs]))
            pts, _d = runs[i]
            path = rest.pop(i)
            out.append(replace(path, pts=np.ascontiguousarray(pts)))
            if len(pts):
                pos = pts[0] if path.closed else pts[-1]  # a contour ends where it began
    return out


# --------------------------------------------------------------------------
#  the text for the machine
# --------------------------------------------------------------------------

def grbl_laser(drawing, s_max=1000, dynamic=True, travel_speed=None) -> str:
    """The drawing as G-code for a GRBL laser: the whole text, ready to send or save.

    Per laser layer, in the order of the drawing, and per pass, the paths run in
    `cut_order`: the head travels to the start of a path in rapid with the beam off (a G0
    move has no power in GRBL laser mode), after which every G1 carries the feed in
    mm/min -- the speed of the layer in mm/s times 60 -- and the power as `S` word, the
    percentage of the layer scaled to `s_max`. Only the first G1 of a path names them;
    the moves that follow are modal and give nothing but the coordinates, and a closed
    path ends where it began.

    `dynamic` chooses M4 (power follows the feed, so corners burn less) or M3 (constant
    power). Between the paths the head travels at `travel_speed` mm/s, the same rapid as
    `vector2d.job_estimate` counts. A worksheet without a laser layer with paths gives a
    ValueError, since G-code that does nothing is not a result.
    """
    drawing = drawing if isinstance(drawing, Drawing) else Drawing(paths=list(drawing))
    travel_speed = DEFAULT_TRAVEL_SPEED if travel_speed is None else float(travel_speed)
    layers = [(lay, [p for p in drawing.paths if p.layer == lay.name and len(p.pts) >= 2])
              for lay in _laser_layers(drawing)]
    if not any(paths for _lay, paths in layers):
        raise ValueError(tr("There is nothing on the worksheet for the laser."))
    lines = [f"; Volura {__version__}", "G21", "G90", "M4" if dynamic else "M3", "S0"]
    for lay, paths in layers:
        if not paths:
            continue
        passes = int(layer_param(lay, "passes"))
        feed = float(layer_param(lay, "speed") or 0.0) * 60.0
        power = round(float(layer_param(lay, "power") or 0.0) * float(s_max) / 100.0)
        ordered = cut_order(paths)
        for k in range(1, passes + 1):
            lines.append(f"; layer {lay.name}, pass {k}/{passes}")
            for path in ordered:
                pts = np.asarray(path.pts, dtype=float)
                run = np.vstack([pts, pts[:1]]) if path.closed else pts
                lines.append(f"G0 X{_num(pts[0][0])} Y{_num(pts[0][1])} F{travel_speed * 60.0:g}")
                for i, q in enumerate(run[1:]):
                    if i == 0:
                        lines.append(f"G1 X{_num(q[0])} Y{_num(q[1])} F{feed:g} S{power}")
                    else:
                        lines.append(f"X{_num(q[0])} Y{_num(q[1])}")
    lines += ["M5", "G0 X0 Y0"]
    return "\n".join(lines) + "\n"

"""G-code for a GRBL mill: the 2.5D toolpaths of the worksheet and the text for the machine.

No Qt in this file. The layers and paths of the worksheet (`volura/vector2d.py`) become
the plain text a hobby mill understands, so it can be tested on its own and reused by
the export in `volura/app.py` and by the CNC Controller (`volura/laser_window.py`).

A **2.5D mill** goes down in levels: the cutter follows the same contour again at every
level, until the depth of the layer is reached. **Z = 0 is the top of the material**, so
everything that cuts stands negative in the G-code; `safe_z` is the height between the
contours, above the material, where the cutter may travel. The depth and the step come
from the layer, the same settings the compensation already uses.

Which way the cutter runs is told by `cam.compensate`: it moves the drawn contour
sideways by the radius of the cutter, fills a pocket with concentric rings and gives the
inside corners a relief, so what is written here is the path of the **centre** of the
cutter. The order of the layers is not free either: holes are drilled first, a pocket is
cleared before the contour around it is milled, and a part is cut loose **last**, because
a part that floats in the plate is torn about by the rest of the job. Within one layer
the same rule holds as for the laser, and `gcode.cut_order` gives that order.

The spindle is the word of the laser turned motor: **M3** starts it at the revolutions of
the `S` word, **G4 P2** gives it the time to come up to speed before the cutter touches
anything, and **M5** stops it at the end of the job.
"""
from __future__ import annotations

import numpy as np

from . import __version__, cam
from .gcode import _num, cut_order
from .i18n import tr
from .vector2d import MILL_PARAMS, Drawing, layer_param

#: the order in which the mill visits the operations: a part is cut loose last
MILL_ORDER = ("drill", "pocket", "mill_inside", "mill_outside")

#: the settings of a mill layer, also for a layer that is not a mill layer ("drill")
_MILL_DEFAULTS = {key: value for key, value in MILL_PARAMS}


# --------------------------------------------------------------------------
#  helpers
# --------------------------------------------------------------------------

def _param(lay, key):
    """A mill setting of a layer.

    A "drill" layer stands in `vector2d.LASER_OPS` and so has no mill settings of its
    own; `layer_param` then finds nothing at all, so the default of the mill is given
    here and the layer still says what it wants to say.
    """
    return layer_param(lay, key, _MILL_DEFAULTS[key])


def _mill_layers(drawing) -> list:
    """The layers the mill follows, with their paths, in the order it visits them.

    The visible layers with a mill operation or "drill", and only those that carry a path
    of at least two points. A hidden layer goes to no machine at all, and "none" is for
    information only.
    """
    out = []
    for op in MILL_ORDER:
        for name, lay in drawing.layers.items():
            if lay.op != op or not getattr(lay, "visible", True):
                continue
            paths = [p for p in drawing.paths if p.layer == name and len(p.pts) >= 2]
            if paths:
                out.append((lay, paths))
    return out


def _levels(lay) -> list[float]:
    """The Z of every pass of a layer: from the first step down to the depth.

    The depth is a positive number that says how deep the cutter goes, while the mill
    works negative below the top of the material. A step of zero, less than nothing or
    deeper than the layer cuts in one pass; the last level always lands exactly on the
    depth, also when the step does not divide it.
    """
    depth = abs(float(_param(lay, "depth") or 0.0))
    step = float(_param(lay, "step_depth") or 0.0)
    if depth <= 0.0:
        return [-0.0]
    if step <= 0.0:
        return [-depth]
    out: list[float] = []
    z = 0.0
    while z > -depth + 1e-9:
        z = max(z - step, -depth)
        out.append(z)
    return out


def _center(path) -> np.ndarray:
    """The middle of the bounding box of a path: where a drill goes in."""
    pts = np.asarray(path.pts, dtype=float)
    return (pts.min(axis=0) + pts.max(axis=0)) / 2.0


# --------------------------------------------------------------------------
#  the moves
# --------------------------------------------------------------------------

def _contour_lines(paths, safe_z, feed, plunge, levels) -> list:
    """The plunges and the cuts of one contour layer, level by level.

    Every level begins with a rapid above the material and a rapid to the start of the
    contour, after which the cutter goes down at the plunge feed and follows the path at
    the feed of the layer. A closed contour ends where it began, so the next level is
    taken from there without coming up; an open path cannot do that and travels back to
    its own start for every level.
    """
    lines = []
    for path in cut_order(paths):
        pts = np.asarray(path.pts, dtype=float)
        run = np.vstack([pts, pts[:1]]) if path.closed else pts
        for i, z in enumerate(levels):
            if i == 0 or not path.closed:
                lines.append(f"G0 Z{_num(safe_z)}")
                lines.append(f"G0 X{_num(run[0][0])} Y{_num(run[0][1])}")
            lines.append(f"G1 Z{_num(z)} F{plunge:g}")
            for q in run[1:]:
                lines.append(f"G1 X{_num(q[0])} Y{_num(q[1])} F{feed:g}")
    return lines


def _drill_lines(paths, safe_z, plunge, depth) -> list:
    """One hole per path, at the middle of its bounding box: in and straight out again."""
    lines = []
    for path in cut_order(paths):
        c = _center(path)
        lines.append(f"G0 Z{_num(safe_z)}")
        lines.append(f"G0 X{_num(c[0])} Y{_num(c[1])}")
        lines.append(f"G1 Z{_num(-abs(float(depth)))} F{plunge:g}")
        lines.append(f"G0 Z{_num(safe_z)}")
    return lines


# --------------------------------------------------------------------------
#  the text for the machine
# --------------------------------------------------------------------------

def grbl_mill(drawing, safe_z=5.0, compensate=True) -> str:
    """The drawing as G-code for a GRBL mill: the whole text, ready to send or save.

    Per layer in `MILL_ORDER` - drilled holes first, then the pockets, then the inside
    contours, and the outside contours that free the part last - the paths run in
    `cut_order`, at every level down to the depth of the layer. `safe_z` is the height
    the cutter travels at, in millimetres above the top of the material, which is Z = 0.

    `compensate` runs `cam.compensate` over the drawing first, so the paths are those of
    the centre of the cutter; leave it on when the drawing still shows what was drawn,
    and turn it off when the caller compensated it already.

    The spindle starts with `M3 S<rpm>` as soon as a layer wants other revolutions than
    the layer before, with `G4 P2` so it can come up to speed. A layer that cuts must say
    how fast: a feed or a plunge feed of zero or less would write `F0`, which GRBL
    refuses, so that gives a ValueError that names the layer. A worksheet without a mill
    layer with paths gives a ValueError too, since G-code that does nothing is no result.
    """
    drawing = drawing if isinstance(drawing, Drawing) else Drawing(paths=list(drawing))
    if compensate:
        drawing = cam.compensate(drawing)
    layers = _mill_layers(drawing)
    if not layers:
        raise ValueError(tr("There is nothing on the worksheet for the mill."))
    safe_z = float(safe_z)
    lines = [f"; Volura {__version__}", "G21", "G90", "G17", f"G0 Z{_num(safe_z)}"]
    rpm = None
    for lay, paths in layers:
        feed = float(_param(lay, "feed") or 0.0)
        plunge = float(_param(lay, "plunge") or 0.0)
        if feed <= 0.0:
            raise ValueError(tr("The layer {layer} has no feed; set one in the layer "
                                "settings.").format(layer=lay.name))
        if plunge <= 0.0:
            raise ValueError(tr("The layer {layer} has no plunge feed; set one in the layer "
                                "settings.").format(layer=lay.name))
        speed = int(round(float(_param(lay, "rpm") or 0.0)))
        if speed != rpm:
            lines += [f"M3 S{speed}", "G4 P2"]      # let the spindle come up to speed
            rpm = speed
        lines.append(f"; layer {lay.name}")
        if lay.op == "drill":
            lines += _drill_lines(paths, safe_z, plunge, _param(lay, "depth"))
        else:
            lines += _contour_lines(paths, safe_z, feed, plunge, _levels(lay))
    lines += [f"G0 Z{_num(safe_z)}", "M5", "G0 X0 Y0"]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------
#  what the job costs
# --------------------------------------------------------------------------

def _words(line) -> dict:
    """The words of a G-code line, as {letter: value}; anything without a number is left out."""
    out = {}
    for word in line.split():
        try:
            out[word[0]] = float(word[1:])
        except (IndexError, ValueError):
            continue
    return out


def mill_estimate(drawing, safe_z=5.0, rapid=3000.0, compensate=True) -> dict:
    """What the mill takes for a drawing: the millimetres it cuts and the seconds.

    The same G-code as `grbl_mill`, read back the way a controller follows it: a G1 goes
    at the feed of its own line - the plunge feed of a plunge and the feed of the layer of
    a cut - and a G0 at `rapid` millimetres per minute. The Z of the moves counts along,
    so a deep layer costs what it costs. `compensate` goes on to `grbl_mill`: off for a
    drawing that was compensated already.

    {"cut_mm", "travel_mm", "seconds"}
    """
    cut = travel = seconds = 0.0
    pos = np.zeros(3)
    feed, mode = float(rapid), 1
    for raw in grbl_mill(drawing, safe_z=safe_z, compensate=compensate).splitlines():
        w = _words(raw.split(";")[0])
        if not w:
            continue
        if "G" in w:
            mode = int(w["G"])
        if "F" in w:
            feed = max(float(w["F"]), 1e-9)
        q = np.array([w.get("X", pos[0]), w.get("Y", pos[1]), w.get("Z", pos[2])])
        d = float(np.linalg.norm(q - pos))
        pos = q
        if d <= 0.0:
            continue
        if mode == 1:
            cut += d
            seconds += d / feed
        else:
            travel += d
            seconds += d / float(rapid)
    return {"cut_mm": cut, "travel_mm": travel, "seconds": seconds}

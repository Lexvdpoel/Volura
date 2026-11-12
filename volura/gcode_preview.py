"""Reading back sliced G-code for the print preview: the moves of every layer, in print order.

No Qt in this file. `read_gcode` follows the nozzle through a G-code file the way a printer does -
absolute or relative positions (G90/G91), absolute or relative extrusion (M82/M83), G92 resets -
and cuts the print into layers at CuraEngine's `;LAYER:` comments. Every layer keeps its moves as
segments with the kind of line they print (`;TYPE:` - outer wall, infill, support, ...), whether
they extrude or only travel, and the moment the printer reaches the end of each, so the preview
can play the layer back as the printer would print it.

The times come from length and feed rate, scaled per layer so they add up to the times CuraEngine
writes after every layer (`;TIME_ELAPSED:`), which include acceleration.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

#: the kinds of line CuraEngine marks with ;TYPE:, in the order of KIND_NAMES
KIND_NAMES = ("travel", "WALL-OUTER", "WALL-INNER", "SKIN", "FILL", "SUPPORT",
              "SUPPORT-INTERFACE", "SKIRT", "PRIME-TOWER", "OTHER")
KIND = {name: i for i, name in enumerate(KIND_NAMES)}
TRAVEL = KIND["travel"]

_WORD = re.compile(r"([A-Z])\s*(-?\d*\.?\d+)")


@dataclass
class Layer:
    """One layer: its height and its moves in print order.

    `seg` holds one row per move: x0, y0, x1, y1; `kind` the kind of each (TRAVEL for a move that
    does not extrude) and `t` the moment, in seconds from the start of the layer, the move ends.
    """

    number: int
    z: float
    seg: np.ndarray
    kind: np.ndarray
    t: np.ndarray

    @property
    def seconds(self) -> float:
        return float(self.t[-1]) if len(self.t) else 0.0

    def position(self, at: float):
        """Where the nozzle is `at` seconds into the layer, and how many moves are done by then."""
        if not len(self.t):
            return None, 0
        i = int(np.searchsorted(self.t, at, side="right"))
        if i >= len(self.t):
            return (float(self.seg[-1, 2]), float(self.seg[-1, 3])), len(self.t)
        start = float(self.t[i - 1]) if i > 0 else 0.0
        span = float(self.t[i]) - start
        f = 0.0 if span <= 0 else (at - start) / span
        x0, y0, x1, y1 = (float(v) for v in self.seg[i])
        return (x0 + f * (x1 - x0), y0 + f * (y1 - y0)), i


@dataclass
class Preview:
    """A sliced print: its layers and the box its extruded lines fill."""

    layers: list = field(default_factory=list)
    lo: tuple = (0.0, 0.0)
    hi: tuple = (0.0, 0.0)

    @property
    def seconds(self) -> float:
        return sum(layer.seconds for layer in self.layers)


def read_gcode(path: str) -> Preview:
    """The layers of the G-code at `path`; moves before the first ;LAYER: are left out."""
    with open(path, encoding="utf-8", errors="replace") as fh:
        return parse(fh)


def parse(lines) -> Preview:
    x = y = z = e = 0.0
    feed = 1500.0                               # mm/min until the G-code says otherwise
    abs_pos = abs_e = True
    kind = KIND["OTHER"]
    layers = []
    cur = None                                  # [number, z, rows, kinds, seconds, elapsed]

    def close(elapsed=None):
        if cur is None:
            return
        rows = np.asarray(cur[2], dtype=np.float32).reshape(-1, 4)
        kinds = np.asarray(cur[3], dtype=np.uint8)
        secs = np.asarray(cur[4], dtype=np.float64)
        total = float(secs.sum())
        if elapsed is not None and elapsed > 0 and total > 0:
            secs *= elapsed / total             # CuraEngine's own time includes acceleration
        layers.append(Layer(cur[0], cur[1], rows, kinds, np.cumsum(secs)))

    last_elapsed = 0.0
    pending_elapsed = None
    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        if line[0] == ";":
            if line.startswith(";LAYER:"):
                close(pending_elapsed)
                pending_elapsed = None
                try:
                    number = int(line[7:])
                except ValueError:
                    number = len(layers)
                cur = [number, z, [], [], []]
            elif line.startswith(";TYPE:"):
                kind = KIND.get(line[6:].strip(), KIND["OTHER"])
            elif line.startswith(";TIME_ELAPSED:"):
                try:
                    now = float(line[14:])
                except ValueError:
                    continue
                pending_elapsed = now - last_elapsed
                last_elapsed = now
            continue
        code = line.split(";", 1)[0].strip()
        if not code:
            continue
        head = code.split(None, 1)[0].upper()
        if head in ("G0", "G1", "G00", "G01"):
            words = dict((k, float(v)) for k, v in _WORD.findall(code[len(head):].upper()))
            nx, ny, nz, ne = x, y, z, e
            if "X" in words:
                nx = words["X"] if abs_pos else x + words["X"]
            if "Y" in words:
                ny = words["Y"] if abs_pos else y + words["Y"]
            if "Z" in words:
                nz = words["Z"] if abs_pos else z + words["Z"]
            if "E" in words:
                ne = words["E"] if abs_e else e + words["E"]
            if "F" in words and words["F"] > 0:
                feed = words["F"]
            if cur is not None:
                if nz != z and not cur[2]:
                    cur[1] = nz                 # the height the layer prints at
                length = float(np.hypot(nx - x, ny - y))
                if length > 0:
                    extrudes = ne > e + 1e-9
                    cur[2].extend((x, y, nx, ny))
                    cur[3].append(kind if extrudes else TRAVEL)
                    cur[4].append(length / feed * 60.0)
            x, y, z, e = nx, ny, nz, ne
        elif head == "G90":
            abs_pos = True
        elif head == "G91":
            abs_pos = False
        elif head == "M82":
            abs_e = True
        elif head == "M83":
            abs_e = False
        elif head == "G92":
            words = dict((k, float(v)) for k, v in _WORD.findall(code[3:].upper()))
            x, y, z, e = (words.get("X", x), words.get("Y", y), words.get("Z", z),
                          words.get("E", e))
    close(pending_elapsed)
    preview = Preview(layers)
    printed = [layer.seg[layer.kind != TRAVEL] for layer in layers]
    printed = [p for p in printed if len(p)]
    if printed:
        pts = np.concatenate([np.concatenate([p[:, :2], p[:, 2:]]) for p in printed])
        preview.lo = (float(pts[:, 0].min()), float(pts[:, 1].min()))
        preview.hi = (float(pts[:, 0].max()), float(pts[:, 1].max()))
    return preview

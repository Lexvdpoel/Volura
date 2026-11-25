"""Reading back sliced G-code for the print preview: the moves of every layer, in print order.

No Qt in this file. `read_gcode` follows the nozzle through a G-code file the way a printer does -
absolute or relative positions (G90/G91), absolute or relative extrusion (M82/M83), G92 resets -
and cuts the print into layers at CuraEngine's `;LAYER:` comments. Every layer keeps its moves as
segments with the kind of line they print (`;TYPE:` - outer wall, infill, support, ...), whether
they extrude or only travel, and the moment the printer reaches the end of each, so the preview
can play the layer back as the printer would print it.

The times come from length and feed rate, scaled per layer so they add up to the times CuraEngine
writes after every layer (`;TIME_ELAPSED:`), which include acceleration.

Every printed move also gets the width of the line it lays down: the filament it pushes out,
spread over its length and the height of the layer, so the preview can draw the lines as wide as
they print and neighbouring lines touch the way they do on the bed.

Every move also keeps the filament that lays it: the extruder of the last `T<n>`, plus how often
the filament of that nozzle was changed (`M600`, inserted by a colour change by height). That is
what lets the preview draw a multi-material print in the colors it prints with.

And every move keeps the mesh it belongs to. CuraEngine writes `;MESH:<name>` whenever it moves on
to another mesh of the print, and `;MESH:NONMESH` for everything that lays no plastic of a part of
the model: the skirt and brim, the prime tower, the support it builds under an overhang and the
travel from one part to the next. That is what `per_part` weighs every part of the print with.
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
#: the narrowest and widest line the preview believes (a short move that primes or wipes can
#: push out much more or less than a real line)
WIDTH_MIN, WIDTH_MAX = 0.1, 3.0
#: the code that asks for a filament change of a printer whose filaments share a nozzle; what
#: comes after it prints with the next filament behind that nozzle
FILAMENT_CHANGE = "M600"
#: the mesh CuraEngine names for everything that lays no plastic of a part of the model: the skirt
#: and brim, the prime tower, the support under an overhang and the travel between the parts
NONMESH = "NONMESH"
#: the code of the moves before the first `;MESH:` of the file, which no part owns either
NO_MESH = -1

_WORD = re.compile(r"([A-Z])\s*(-?\d*\.?\d+)")


@dataclass
class Layer:
    """One layer: its height and its moves in print order.

    `seg` holds one row per move: x0, y0, x1, y1; `kind` the kind of each (TRAVEL for a move that
    does not extrude) and `t` the moment, in seconds from the start of the layer, the move ends.
    `height` is how thick the layer is and `width` the width of the line of every move (0 for a
    travel move). `pushed` is the length of filament every move pushes out (mm) and `material`
    which filament that is: the extruder of the move or, after a filament change of a shared
    nozzle (`M600`), the filament loaded then. Both are None when nothing was extruded. `mesh`
    says which mesh of the print lays every move: a code into `Preview.mesh_names`, `NO_MESH` for
    the moves before the first `;MESH:` of the file.
    """

    number: int
    z: float
    seg: np.ndarray
    kind: np.ndarray
    t: np.ndarray
    height: float = 0.2
    width: np.ndarray | None = None
    pushed: np.ndarray | None = None
    material: np.ndarray | None = None
    mesh: np.ndarray | None = None

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
    """A sliced print: its layers, the box its extruded lines fill, its filaments and its meshes.

    `mesh_names` names every mesh the G-code says it is printing, in the order the file first
    mentions one: `part0`, `part1`, ... and `NONMESH` for everything that lays no plastic of a
    part. Every move of every layer carries a code into this list.
    """

    layers: list = field(default_factory=list)
    lo: tuple = (0.0, 0.0)
    hi: tuple = (0.0, 0.0)
    materials: int = 1
    mesh_names: list = field(default_factory=list)

    @property
    def seconds(self) -> float:
        return sum(layer.seconds for layer in self.layers)

    def filament_mm(self) -> np.ndarray:
        """How much filament every material pushes out (mm), summed over every layer."""
        out = np.zeros(max(1, self.materials), dtype=np.float64)
        for layer in self.layers:
            if layer.pushed is not None and layer.material is not None:
                np.add.at(out, layer.material, layer.pushed)
        return out


@dataclass
class PartRow:
    """What one part of a sliced print costs, a row of the table the preview shows after a slice.

    `name` is the name the part has in the model and `group_id` the group it stands in; the row of
    the moves that print no part of the model carries neither and is what the table calls "Other".
    `seconds` is how long the moves of the part take, `grams` and `metres` what they push out of
    the filament and `share` how much of the time of the whole print that is. `codes` are the
    meshes of the G-code the row adds up: codes into `Preview.mesh_names` (plus `NO_MESH` for the
    row of "Other"), which is what the view of the preview lights up when the row is chosen.
    """

    name: str = ""
    group_id: object = None
    seconds: float = 0.0
    grams: float = 0.0
    metres: float = 0.0
    share: float = 0.0
    codes: tuple = ()

    @property
    def other(self) -> bool:
        """Whether this is the row of everything that prints no part of the model."""
        return self.group_id is None and not self.name


def read_gcode(path: str, filament: float = 1.75) -> Preview:
    """The layers of the G-code at `path`; moves before the first ;LAYER: are left out.

    `filament` is the diameter of the filament in mm, for the width of the lines.
    """
    with open(path, encoding="utf-8", errors="replace") as fh:
        return parse(fh, filament)


def parse(lines, filament: float = 1.75) -> Preview:
    x = y = z = e = 0.0
    feed = 1500.0                               # mm/min until the G-code says otherwise
    abs_pos = abs_e = True
    kind = KIND["OTHER"]
    tool = loaded = 0                           # the extruder at work and how often its
                                                # filament was changed since
    layers = []
    seen = set()                                # every filament this print laid a line with
    names = []                                  # every mesh the G-code names, in the order it
    codes = {}                                  # first mentions one; what a `;MESH:` names
    mesh = NO_MESH                              # the mesh the moves being read belong to
    area = np.pi * (filament / 2.0) ** 2 if filament > 0 else np.pi * 0.875 ** 2
    cur = None                                  # [number, z, rows, kinds, seconds, filament,
                                                # material, mesh]

    def mesh_code(name):
        """The code of the mesh `name`; a name the file mentions for the first time gets one."""
        if name not in codes:
            codes[name] = len(names)
            names.append(name)
        return codes[name]

    def close(elapsed=None):
        if cur is None:
            return
        rows = np.asarray(cur[2], dtype=np.float32).reshape(-1, 4)
        kinds = np.asarray(cur[3], dtype=np.uint8)
        secs = np.asarray(cur[4], dtype=np.float64)
        total = float(secs.sum())
        if elapsed is not None and elapsed > 0 and total > 0:
            secs *= elapsed / total             # CuraEngine's own time includes acceleration
        below = layers[-1].z if layers else 0.0
        height = cur[1] - below
        if not 0.01 < height < 2.0:
            height = layers[-1].height if layers else 0.2
        length = np.hypot(rows[:, 2] - rows[:, 0], rows[:, 3] - rows[:, 1])
        pushed = np.asarray(cur[5], dtype=np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            width = pushed * area / (length * height)
        width = np.where(kinds == TRAVEL, 0.0,
                         np.clip(np.nan_to_num(width), WIDTH_MIN, WIDTH_MAX)).astype(np.float32)
        material = np.asarray(cur[6], dtype=np.uint8)
        seen.update(int(m) for m in np.unique(material))
        layers.append(Layer(cur[0], cur[1], rows, kinds, np.cumsum(secs), height, width,
                            pushed, material, np.asarray(cur[7], dtype=np.int16)))

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
                cur = [number, z, [], [], [], [], [], []]
            elif line.startswith(";TYPE:"):
                kind = KIND.get(line[6:].strip(), KIND["OTHER"])
            elif line.startswith(";MESH:"):
                mesh = mesh_code(line[6:].strip())     # the engine moved on to another mesh
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
                    cur[5].append(ne - e if extrudes else 0.0)
                    cur[6].append(tool + loaded)
                    cur[7].append(mesh)
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
        elif head == FILAMENT_CHANGE:
            loaded += 1                         # the next filament behind this nozzle
        elif head.startswith("T") and head[1:].isdigit():
            tool = int(head[1:])                # the extruder this part prints with
    close(pending_elapsed)
    preview = Preview(layers, materials=1 + max(seen, default=0), mesh_names=names)
    printed = [layer.seg[layer.kind != TRAVEL] for layer in layers]
    printed = [p for p in printed if len(p)]
    if printed:
        pts = np.concatenate([np.concatenate([p[:, :2], p[:, 2:]]) for p in printed])
        preview.lo = (float(pts[:, 0].min()), float(pts[:, 1].min()))
        preview.hi = (float(pts[:, 0].max()), float(pts[:, 1].max()))
    return preview


def per_part(preview: Preview, parts=None, density: float = 1.24,
             filament: float = 1.75) -> list:
    """What every part of the print costs: a `PartRow` per part, the longest first.

    CuraEngine writes a `;MESH:` comment whenever it moves on to another mesh: `part0`, `part1`,
    ... for the meshes the slicer handed it and `NONMESH` for everything that lays no plastic of a
    part of the model - the skirt and brim, the prime tower of a multicolour print, the travel from
    one part to the next and, what CuraEngine does for every part of the print together, the
    support it builds under an overhang. Every move of the preview knows its mesh from that, so a
    row is what the moves of one mesh cost: the time the preview worked out for them (its per-move
    times, scaled to the `;TIME_ELAPSED:` CuraEngine writes after every layer, so the acceleration
    is in there too) and the filament they push out, weighed with `density` in g/cm³ and the
    diameter `filament` in mm - the same measure with which the slice weighs the whole print.

    `parts` is what the slice knows of its meshes, `{mesh name: (group id, part name)}`. A part
    that prints with more than one mesh - faces painted in a colour that comes off another
    extruder - has those meshes add up to one row, and a part whose mesh printed nothing keeps a
    row of 0 seconds, so it is to see that it printed nothing. Every mesh `parts` does not name is
    no part of the print and comes, together with the moves before the first `;MESH:` of the file,
    in the row without name and group that the table shows as "Other".

    Every move of the print is in exactly one row and no move in two, so the rows add up to what
    the print took: `preview.seconds` and all of the filament the G-code pushed out. A print of one
    part answers that part and the row of "Other"; a G-code that names no mesh at all answers only
    that one row. Every part `parts` names is always to see, even when the print has no moves at
    all; a print whose every mesh belongs to a part and that never travels - no skirt, no support -
    has no row of "Other".
    """
    seconds, pushed = {}, {}
    for layer in preview.layers:
        codes = (layer.mesh if layer.mesh is not None
                 else np.full(len(layer.kind), NO_MESH, dtype=np.int16))
        if not len(codes):
            continue
        mm = layer.pushed if layer.pushed is not None else np.zeros(len(codes))
        secs = np.diff(np.concatenate(([0.0], layer.t)))
        for code in np.unique(codes):
            hit = codes == code
            key = int(code)
            seconds[key] = seconds.get(key, 0.0) + float(secs[hit].sum())
            pushed[key] = pushed.get(key, 0.0) + float(np.asarray(mm)[hit].sum())

    area = np.pi * (filament / 2.0) ** 2 if filament and filament > 0 else 0.0

    def row(codes, name, group_id):
        """The row of `PartRow` that adds the meshes of `codes` up."""
        secs = sum(seconds.get(int(code), 0.0) for code in codes)
        mm = sum(pushed.get(int(code), 0.0) for code in codes)
        return PartRow(name=name, group_id=group_id, seconds=secs,
                       grams=mm * area * density / 1000.0,
                       metres=mm / 1000.0, codes=tuple(int(code) for code in codes))

    codes_of = {name: code for code, name in enumerate(preview.mesh_names)}
    rows, mine = {}, set()
    for name, (group_id, part_name) in (parts or {}).items():
        row_of = rows.setdefault((group_id, part_name), PartRow(name=part_name, group_id=group_id))
        code = codes_of.get(name)               # a mesh that printed nothing is not named
        if code is not None:
            row_of.codes += (code,)
            mine.add(code)
    out = [row(one.codes, one.name, one.group_id) for one in rows.values()]
    spare = [code for code in range(len(preview.mesh_names)) if code not in mine]
    if NO_MESH in seconds:
        spare.append(NO_MESH)
    if spare:                               # a print with no skirt, brim, travel or support at all
        out.append(row(spare, "", None))    # has nothing to fill a row of "Other" with
    total = sum(one.seconds for one in out)
    for one in out:
        one.share = one.seconds / total if total > 0 else 0.0
    out.sort(key=lambda one: (-one.seconds, one.name))
    return out

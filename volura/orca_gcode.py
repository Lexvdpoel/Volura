"""The start and end G-code of the printers that come from OrcaSlicer's profiles.

Cura writes `{setting}` in its start G-code and lets CuraEngine fill it in; OrcaSlicer has a small
language of its own for it: `[name]` and `{expression}`, `{if ...}{elsif ...}{else}{endif}`,
`{local x = ...}`, arithmetic, comparisons, `min`, `max`, `int`, ... CuraEngine does not know that
language, so the printers that tools/fetch_orca_profiles.py makes out of OrcaSlicer's profiles keep
their G-code as it is, and `render` works it out here just before slicing, with the values of the
chosen printer, nozzle, material and quality (`variables`). CuraEngine then gets plain G-code.

No Qt in this file.
"""
from __future__ import annotations

import math
import re
import time

__all__ = ["OrcaGcodeError", "render", "variables", "names_used", "points"]


class OrcaGcodeError(ValueError):
    """A G-code template that cannot be worked out: wrong syntax or a name nobody knows."""


# ---- values
_INT = re.compile(r"^[+-]?\d+$")
_FLOAT = re.compile(r"^[+-]?(\d+\.\d*|\.\d+|\d+)([eE][+-]?\d+)?$")


def typed(value):
    """A value of an OrcaSlicer profile, which keeps everything as text, as a number when it is."""
    if isinstance(value, list):
        return [typed(v) for v in value]
    if not isinstance(value, str):
        return value
    if _INT.match(value):
        return int(value)
    if _FLOAT.match(value):
        return float(value)
    if value in ("true", "false"):
        return value == "true"
    return value


def _text(value) -> str:
    """A value the way it lands in the G-code."""
    if isinstance(value, list):
        value = value[0] if value else ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if value != value or value in (math.inf, -math.inf):
            raise OrcaGcodeError("a number that is not a number")
        s = f"{value:.4f}".rstrip("0").rstrip(".")
        return "0" if s in ("-0", "") else s
    return str(value)


def _scalar(value):
    if isinstance(value, list):
        return value[0] if value else 0
    return value


def _number(value):
    value = _scalar(value)
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return value
    t = typed(str(value).strip())
    if isinstance(t, (int, float)) and not isinstance(t, bool):
        return t
    raise OrcaGcodeError(f"not a number: {value!r}")


def _truth(value) -> bool:
    value = _scalar(value)
    if isinstance(value, str):
        return value not in ("", "0", "false")
    return bool(value)


def _equal(a, b) -> bool:
    a, b = _scalar(a), _scalar(b)
    if isinstance(a, str) != isinstance(b, str):
        try:
            return float(_number(a)) == float(_number(b))
        except OrcaGcodeError:
            return _text(a) == _text(b)
    return a == b


# ---- expressions
_TOKEN = re.compile(r"""\s*(?:
    (?P<num>\d+\.\d*(?:[eE][+-]?\d+)?|\.\d+(?:[eE][+-]?\d+)?|\d+(?:[eE][+-]?\d+)?)
  | (?P<str>"(?:[^"\\]|\\.)*")
  | (?P<name>[A-Za-z_][A-Za-z0-9_]*)
  | (?P<op>\|\||&&|==|!=|<>|<=|>=|=~|!~|[-+*/%<>!()\[\],?:=])
)""", re.X)


def _tokens(text: str) -> list:
    out, pos = [], 0
    text = text.strip()
    while pos < len(text):
        m = _TOKEN.match(text, pos)
        if not m or m.end() == pos:
            raise OrcaGcodeError(f"cannot read {text[pos:pos + 20]!r}")
        pos = m.end()
        kind = m.lastgroup
        value = m.group(kind)
        if kind == "op" and value in ("=~", "!~"):
            out.append((kind, value))
            rest = text[pos:].lstrip()
            if rest.startswith("/"):
                pos = len(text) - len(rest) + 1
                end = pos
                while end < len(text) and text[end] != "/":
                    end += 2 if text[end] == "\\" else 1
                if end >= len(text):
                    raise OrcaGcodeError("a regular expression without its closing /")
                out.append(("str", text[pos:end]))
                pos = end + 1
            continue
        if kind == "num":
            value = int(value) if _INT.match(value) else float(value)
        elif kind == "str":
            value = re.sub(r"\\(.)", r"\1", value[1:-1])
        elif kind == "name" and value in ("and", "or", "not"):
            kind, value = "op", {"and": "&&", "or": "||", "not": "!"}[value]
        out.append((kind, value))
    return out


class _Parser:
    """One expression, worked out while it is read (recursive descent, C-like precedence)."""

    def __init__(self, text, lookup, functions):
        self.toks = _tokens(text)
        self.i = 0
        self.lookup, self.functions = lookup, functions

    def peek(self, value=None):
        if self.i >= len(self.toks):
            return None
        tok = self.toks[self.i]
        if value is not None:
            return tok if tok[0] == "op" and tok[1] == value else None
        return tok

    def take(self, value=None):
        tok = self.peek(value)
        if tok is None:
            raise OrcaGcodeError(f"expected {value!r}" if value else "the expression stops early")
        self.i += 1
        return tok

    def done(self):
        if self.i != len(self.toks):
            raise OrcaGcodeError(f"unexpected {self.toks[self.i][1]!r}")

    def expression(self):
        cond = self.or_()
        if self.peek("?"):
            self.take("?")
            a = self.expression()
            self.take(":")
            b = self.expression()
            return a if _truth(cond) else b
        return cond

    def or_(self):
        v = self.and_()
        while self.peek("||"):
            self.take()
            w = self.and_()
            v = _truth(v) or _truth(w)
        return v

    def and_(self):
        v = self.compare()
        while self.peek("&&"):
            self.take()
            w = self.compare()
            v = _truth(v) and _truth(w)
        return v

    def compare(self):
        v = self.additive()
        while True:
            tok = self.peek()
            if not tok or tok[0] != "op" or tok[1] not in ("==", "!=", "<>", "<", ">", "<=", ">=",
                                                          "=~", "!~"):
                return v
            op = self.take()[1]
            w = self.additive()
            if op in ("=~", "!~"):
                try:
                    found = re.search(str(_scalar(w)), _text(v)) is not None
                except re.error as e:
                    raise OrcaGcodeError(str(e)) from None
                v = found if op == "=~" else not found
            elif op == "==":
                v = _equal(v, w)
            elif op in ("!=", "<>"):
                v = not _equal(v, w)
            else:
                a, b = _scalar(v), _scalar(w)
                if not (isinstance(a, str) and isinstance(b, str)):
                    a, b = _number(a), _number(b)
                v = {"<": a < b, ">": a > b, "<=": a <= b, ">=": a >= b}[op]

    def additive(self):
        v = self.term()
        while self.peek("+") or self.peek("-"):
            op = self.take()[1]
            w = self.term()
            if op == "+" and (isinstance(_scalar(v), str) or isinstance(_scalar(w), str)):
                v = _text(v) + _text(w)
            else:
                a, b = _number(v), _number(w)
                v = a + b if op == "+" else a - b
        return v

    def term(self):
        v = self.unary()
        while self.peek("*") or self.peek("/") or self.peek("%"):
            op = self.take()[1]
            a, b = _number(v), _number(self.unary())
            if op == "*":
                v = a * b
            elif b == 0:
                raise OrcaGcodeError("division by zero")
            elif op == "/":
                # as in OrcaSlicer (C++): whole numbers divide into a whole number
                v = int(a / b) if isinstance(a, int) and isinstance(b, int) else a / b
            else:
                v = math.fmod(a, b) if isinstance(a, float) or isinstance(b, float) \
                    else int(math.fmod(a, b))
        return v

    def unary(self):
        if self.peek("-"):
            self.take()
            return -_number(self.unary())
        if self.peek("+"):
            self.take()
            return _number(self.unary())
        if self.peek("!"):
            self.take()
            return not _truth(self.unary())
        return self.postfix()

    def postfix(self):
        v = self.primary()
        while self.peek("["):
            self.take()
            index = _number(self.expression())
            self.take("]")
            v = _index(v, index)
        return v

    def primary(self):
        kind, value = self.take()
        if kind in ("num", "str"):
            return value
        if kind == "op" and value == "(":
            v = self.expression()
            self.take(")")
            return v
        if kind == "name":
            if value in ("true", "false"):
                return value == "true"
            if self.peek("("):
                self.take()
                args = []
                if not self.peek(")"):
                    args.append(self.expression())
                    while self.peek(","):
                        self.take()
                        args.append(self.expression())
                self.take(")")
                f = self.functions.get(value)
                if f is None:
                    raise OrcaGcodeError(f"unknown function {value}()")
                try:
                    return f(*args)
                except TypeError:
                    raise OrcaGcodeError(f"wrong arguments for {value}()") from None
            return self.lookup(value)
        raise OrcaGcodeError(f"unexpected {value!r}")


def _index(value, index):
    i = int(index)
    if isinstance(value, list):
        if not value:
            raise OrcaGcodeError("an empty list")
        return value[min(max(i, 0), len(value) - 1)]
    return value


def _digits(value, places, decimals=0, zeros=False):
    v, places, decimals = float(_number(value)), int(_number(places)), int(_number(decimals))
    s = f"{v:.{decimals}f}"
    return s.rjust(places, "0" if zeros else " ") if places > 0 else s


_FUNCTIONS = {
    "min": lambda a, b: min(_number(a), _number(b)),
    "max": lambda a, b: max(_number(a), _number(b)),
    "int": lambda a: int(_number(a)),
    "round": lambda a: int(math.floor(_number(a) + 0.5)),
    "digits": lambda v, p, d=0: _digits(v, p, d),
    "zdigits": lambda v, p, d=0: _digits(v, p, d, zeros=True),
    "is_nil": lambda a: False,
    "one_of": lambda v, *opts: any(_equal(v, o) for o in opts),
    "random": lambda a, b: _number(a),
    "size": lambda v: len(v) if isinstance(v, list) else 1,
    "empty": lambda v: not v if isinstance(v, (list, str)) else False,
}


# ---- templates
_LEGACY = re.compile(r"\[([A-Za-z_][A-Za-z0-9_]*)(?:\[(\d+)\])?\]")
_KEYWORD = re.compile(r"^(if|elsif|elseif|else|endif)(?![A-Za-z0-9_])(.*)$", re.S)
_INDEXED = re.compile(r"^(.+)_(\d+)$")
_ASSIGN = re.compile(r"^(?:(local|global)\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=(?!=)\s*(.*)$", re.S)


def _blocks(template: str):
    """The template as ("text", str) and ("code", str) pieces; `{...}` is code."""
    out, pos, n = [], 0, len(template)
    while pos < n:
        start = template.find("{", pos)
        if start < 0:
            out.append(("text", template[pos:]))
            break
        if start > pos:
            out.append(("text", template[pos:start]))
        i, depth, quote = start + 1, 1, False
        while i < n and depth:
            c = template[i]
            if quote:
                if c == "\\":
                    i += 1
                elif c == '"':
                    quote = False
            elif c == '"':
                quote = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
            i += 1
        if depth:
            raise OrcaGcodeError("a { without its }")
        out.append(("code", template[start + 1:i - 1]))
        pos = i
    return out


def _split_statements(code: str) -> list[str]:
    parts, cur, quote = [], [], False
    for c in code:
        if c == '"':
            quote = not quote
        if c == ";" and not quote:
            parts.append("".join(cur))
            cur = []
        else:
            cur.append(c)
    parts.append("".join(cur))
    return [p.strip() for p in parts if p.strip()]


def render(template: str, values: dict) -> str:
    """`template` with every placeholder, condition and expression worked out from `values`.

    `values` holds the variables by name, as numbers, text or lists (one entry per extruder).
    Raises OrcaGcodeError for a template that cannot be worked out.
    """
    own: dict = {}

    def lookup(name):
        if name in own:
            return own[name]
        if name in values:
            return values[name]
        raise OrcaGcodeError(f"unknown name {name!r}")

    def evaluate(expr):
        p = _Parser(expr, lookup, _FUNCTIONS)
        v = p.expression()
        p.done()
        return v

    def legacy(text):
        def sub(m):
            name, index = m.group(1), m.group(2)
            split = _INDEXED.match(name)
            if index is None and name not in own and name not in values and split                     and (split.group(1) in own or split.group(1) in values):
                name, index = split.group(1), split.group(2)    # [name_0]: [name[0]]
            if name in own or name in values:
                v = lookup(name)
                return _text(_index(v, int(index)) if index is not None else v)
            return m.group(0)               # not ours: [ and ] are plain G-code here
        return _LEGACY.sub(sub, text)

    out = []
    # the conditions around the current place: [shown?, taken already?] per {if}
    stack: list[list[bool]] = []

    def live():
        return all(s[0] for s in stack)

    for kind, piece in _blocks(template):
        if kind == "text":
            if live():
                out.append(legacy(piece))
            continue
        code = piece.strip()
        m = _KEYWORD.match(code)
        head, rest = (m.group(1), m.group(2).strip()) if m else ("", code)
        if head == "if":
            on = live() and _truth(evaluate(rest))
            stack.append([on, on])
        elif head in ("elsif", "elseif"):
            if not stack:
                raise OrcaGcodeError("{elsif} without {if}")
            top = stack[-1]
            stack.pop()
            on = live() and not top[1] and _truth(evaluate(rest))
            stack.append([on, top[1] or on])
        elif head == "else" and not rest:
            if not stack:
                raise OrcaGcodeError("{else} without {if}")
            top = stack.pop()
            stack.append([live() and not top[1], True])
        elif head == "endif" and not rest:
            if not stack:
                raise OrcaGcodeError("{endif} without {if}")
            stack.pop()
        elif live():
            for statement in _split_statements(code):
                m = _ASSIGN.match(statement)
                if m:
                    own[m.group(2)] = evaluate(m.group(3))
                else:
                    out.append(_text(evaluate(statement)))
    if stack:
        raise OrcaGcodeError("{if} without {endif}")
    return "".join(out)


def names_used(template: str) -> set[str]:
    """Every name a template might ask for: what `variables` has to give for it."""
    names = set()
    for kind, piece in _blocks(template):
        if kind == "text":
            for m in _LEGACY.finditer(piece):
                split = _INDEXED.match(m.group(1))
                names.add(split.group(1) if split and m.group(2) is None else m.group(1))
        else:
            for k, v in _tokens(piece.replace(";", " ")) if piece.strip() else []:
                if k == "name":
                    names.add(v)
    return names - set(_FUNCTIONS) - {"if", "elsif", "elseif", "else", "endif", "local", "global",
                                      "true", "false"}


# ---- the values a template gets
_POINT = re.compile(r"^\s*([-+]?[\d.]+(?:[eE][-+]?\d+)?)\s*x\s*([-+]?[\d.]+(?:[eE][-+]?\d+)?)\s*$")


def points(area) -> list[tuple[float, float]]:
    """The corners of an OrcaSlicer area ("0x0", "220x0", ...); what is no point is left out."""
    out = []
    for p in area if isinstance(area, list) else str(area).split(","):
        m = _POINT.match(str(p))
        if m:
            try:
                out.append((float(m.group(1)), float(m.group(2))))
            except ValueError:
                continue
    return out


def variables(config: dict, *, nozzle_temp0: float, nozzle_temp: float, bed_temp0: float,
              bed_temp: float, material: str, layer_height: float, first_layer_height: float,
              nozzle: float, retraction: float, travel_speed: float, print_min=(0.0, 0.0),
              print_max=(0.0, 0.0), top=0.0, layers=0, printer="", quality="",
              filament="", now=None) -> dict:
    """The values for `render`: the printer's own OrcaSlicer settings in `config`, with what
    OrcaSlicer would work out for this print put over them.

    `print_min` and `print_max` are the corners of the first layer on the bed, in the printer's
    coordinates; `top` is the height of the print and `layers` its number of layers.
    """
    v = {k: typed(x) for k, x in config.items()}
    count = max(1, len(v["nozzle_diameter"]) if isinstance(v.get("nozzle_diameter"), list) else 1)

    def each(x):
        return [x] * count

    t0, t, b0, b = (int(round(x)) for x in (nozzle_temp0, nozzle_temp, bed_temp0, bed_temp))
    for k in ("nozzle_temperature_initial_layer", "first_layer_temperature"):
        v[k] = each(t0)
    for k in ("nozzle_temperature", "temperature"):
        v[k] = each(t)
    v["bed_temperature_initial_layer_single"] = b0
    for plate in ("hot_plate_temp", "cool_plate_temp", "eng_plate_temp", "textured_plate_temp",
                  "supertack_plate_temp", "textured_cool_plate_temp"):
        v[plate + "_initial_layer"] = each(b0)
        v[plate] = each(b)
    for k in ("first_layer_bed_temperature", "bed_temperature_initial_layer"):
        v[k] = each(b0)
    v["bed_temperature"] = each(b)
    v["filament_type"] = each(material)
    v["nozzle_diameter"] = each(nozzle)
    v["retraction_length"] = each(retraction)
    v["layer_height"] = layer_height
    v["initial_layer_print_height"] = v["first_layer_height"] = first_layer_height
    v["travel_speed"] = travel_speed
    for k in ("initial_extruder", "initial_tool", "initial_no_support_extruder",
              "current_extruder", "previous_extruder", "next_extruder", "layer_num",
              "total_toolchanges", "plate_number"):
        v[k] = 0
    v["is_extruder_used"] = [True] + [False] * (count - 1)
    v["has_wipe_tower"] = False
    v["e_retracted"] = each(0.0)
    v["e_restart_extra"] = each(0.0)
    v["layer_z"] = first_layer_height
    v["max_layer_z"] = float(top)
    v["total_layer_count"] = int(layers)
    lo, hi = [float(x) for x in print_min], [float(x) for x in print_max]
    v["first_layer_print_min"], v["first_layer_print_max"] = lo, hi
    v["first_layer_print_size"] = [hi[0] - lo[0], hi[1] - lo[1]]
    area = points(config.get("printable_area", [])) or [(0.0, 0.0)]
    bmin = [min(p[0] for p in area), min(p[1] for p in area)]
    bmax = [max(p[0] for p in area), max(p[1] for p in area)]
    v["print_bed_min"], v["print_bed_max"] = bmin, bmax
    v["print_bed_size"] = [bmax[0] - bmin[0], bmax[1] - bmin[1]]
    v["printer_preset"] = v["printer_model"] = printer or v.get("printer_model", "")
    v["print_preset"] = quality
    v["filament_preset"] = each(filament)
    v["plate_name"] = v["input_filename_base"] = v["input_filename"] = "Volura"
    v["scan_first_layer"] = False
    v["max_print_height"] = v.get("printable_height", top)
    v.setdefault("curr_bed_type", "Textured PEI Plate")
    v.setdefault("z_offset", 0.0)
    for k in ("chamber_temperature", "overall_chamber_temperature"):
        v.setdefault(k, each(0))
    v["adaptive_bed_mesh_min"], v["adaptive_bed_mesh_max"] = lo, hi
    v.setdefault("bed_mesh_probe_count", [5, 5])
    v.setdefault("bed_mesh_algo", "bicubic")
    v.setdefault("enable_pressure_advance", each(False))
    v.setdefault("pressure_advance", each(0.02))
    v.setdefault("outer_wall_volumetric_speed", 12.0)
    v.setdefault("machine_max_acceleration_travel", each(1500))
    now = time.localtime() if now is None else now
    v["year"], v["month"], v["day"] = now.tm_year, now.tm_mon, now.tm_mday
    v["hour"], v["minute"], v["second"] = now.tm_hour, now.tm_min, now.tm_sec
    v["timestamp"] = time.strftime("%Y%m%d%H%M%S", now)
    return v

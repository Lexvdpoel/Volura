"""Send a sliced print straight to the printer over the network.

No Qt in this file: the print preview asks `check` and `upload` in a background thread
and shows what comes back. Three programs answer on the network for a printer:
OctoPrint, Moonraker (the API of Klipper) and PrusaLink on a Prusa printer. A
`Connection` says which of them it is, where it stands and with which key it lets you
in; the settings keep them (volura.app.Settings) and Preferences > Printers edits them.
`status` asks what one of them is doing right now, which Window > Printer Status shows.
`bed_mesh` asks a Klipper printer how flat its bed is, which Window > Bed Mesh… paints.
`snapshot_url` says where the printer's camera keeps a still picture and `snapshot` asks
that camera for one, which the same window can put beside the status of that printer.

An API key is a password: it never stands in a message or in the log, only in the
`X-Api-Key` header of the request. It is kept in the settings like every other
preference (QSettings: the registry of the user on Windows, an ini file elsewhere), so
whoever can read your account can read it too; on your own computer that is fine and on
one you share it is not.

`urllib` and `ssl` are imported where they are used, as with the updater: starting
Volura may not wait on them (tests/test_startup_imports.py).
"""
from __future__ import annotations

import json
import math
import os
import re
import uuid
from dataclasses import dataclass, replace

import numpy as np

from . import __version__
from .bed_mesh import BedMesh, no_grid
from .i18n import tr, tr_noop

#: every kind of printer, with the program that answers at it and what Preferences calls it
KINDS = {"octoprint": ("OctoPrint", tr_noop("OctoPrint")),
         "moonraker": ("Moonraker", tr_noop("Klipper (Moonraker)")),
         "prusalink": ("PrusaLink", tr_noop("PrusaLink"))}
#: how much of the G-code is read and sent at a time: a big print is tens of megabytes
CHUNK = 256 * 1024
#: what a G-code is called on the printer when nothing says what it should be
DEFAULT_NAME = "volura"


class LinkError(Exception):
    """What went wrong speaking to a printer, in words for the user.

    `code` is the answer the printer gave with it, or None when the request never got an
    answer; `status` asks for it, because OctoPrint answers 409 when it has no printer and
    that is a state and not an error."""

    def __init__(self, text, code=None):
        super().__init__(text)
        self.code = code


@dataclass
class Connection:
    """One printer: which program it runs, where it stands and which key it wants."""

    name: str
    kind: str
    url: str
    api_key: str = ""


def server_name(kind) -> str:
    """What the program at a kind of printer is really called, as in "Moonraker"."""
    return KINDS[kind][0]


def kind_label(kind) -> str:
    """How Preferences names a kind of printer, as in "Klipper (Moonraker)"."""
    return tr(KINDS[kind][1])


# ------------------------------------------------------------------ the address


def normalise_url(text) -> str:
    """The address of a printer as it is asked: with a scheme and without a last slash.

    "192.168.1.20" becomes "http://192.168.1.20"; https, a port and a path are kept, so
    "octoprint.local:5000/" is "http://octoprint.local:5000" - and so is
    "octoprint.local:5000", because a name before a colon is not yet a scheme."""
    text = str(text or "").strip()
    if text and not re.match(r"[A-Za-z][A-Za-z0-9+.-]*://", text):
        text = "http://" + text
    return text.rstrip("/")


def _address(conn, url) -> str:
    """The address to ask, or a LinkError when what was filled in is not an address."""
    import urllib.parse
    where = urllib.parse.urlsplit(url)
    try:
        host = where.hostname or ""
    except ValueError:                            # a port that is not a number
        host = ""
    if where.scheme not in ("http", "https") or not re.fullmatch(r"[\w.\-:]+", host):
        raise LinkError(tr("{text} is not the address of a printer. Use for example "
                           "192.168.1.20:5000 or octoprint.local.").format(text=conn.url))
    return url


# ------------------------------------------------------------------ file names


def safe_name(text) -> str:
    """A file name the printer will take: no path, no spaces, nothing it cannot store."""
    name = str(text).replace("\\", "/").rsplit("/", 1)[-1]
    name = re.sub(r"[^A-Za-z0-9._+-]", "_", name)
    name = re.sub(r"_{2,}", "_", name).strip("._")
    return name or DEFAULT_NAME


def file_name(*names) -> str:
    """What a G-code is called on the printer: the parts that print, or "volura"."""
    name = "-".join(str(part).strip() for part in names if part) or DEFAULT_NAME
    if not name.lower().endswith(".gcode"):
        name += ".gcode"
    return safe_name(name)


# ------------------------------------------------------------------ in the settings


def to_json(conn) -> str:
    """One connection as a line of JSON, for the list of printers in the settings."""
    return json.dumps({"name": conn.name, "kind": conn.kind, "url": conn.url,
                       "api_key": conn.api_key})


def from_json(text) -> Connection | None:
    """The connection that stands in that line of JSON; None when it is not one."""
    try:
        data = json.loads(text)
    except (TypeError, ValueError):
        return None
    if not isinstance(data, dict) or data.get("kind") not in KINDS:
        return None
    return Connection(str(data.get("name", "")), str(data["kind"]), str(data.get("url", "")),
                      str(data.get("api_key", "")))


# ------------------------------------------------------------------ what went wrong


def _wrong_kind(url, what, code=None) -> LinkError:
    """There is another kind of printer at that address than the one that was asked for."""
    return LinkError(tr("{url} does not answer as {what}. Is this printer really a "
                        "{what}?").format(url=url, what=what), code)


def _error(exc, url, what) -> LinkError:
    """What a failed request means for the user; an API key is never one of the words.

    The answer the printer gave with it stays on the LinkError, so a caller can tell one
    of them apart without reading words back out of a message."""
    import ssl
    import urllib.error
    if isinstance(exc, urllib.error.HTTPError):
        if exc.code in (401, 403):
            return LinkError(tr("{url} will not take this API key. Copy it again from the "
                                "settings of the printer.").format(url=url), exc.code)
        if exc.code == 404:
            return _wrong_kind(url, what, exc.code)
        return LinkError(tr("{url} answered error {code}.").format(url=url, code=exc.code),
                         exc.code)
    if isinstance(exc, ssl.SSLError) or isinstance(getattr(exc, "reason", None), ssl.SSLError):
        return LinkError(tr("{url} has a certificate this computer does not trust. Is the "
                            "address right?").format(url=url))
    if isinstance(exc, OSError):            # refused, timed out, no such address
        return LinkError(tr("Nothing answers at {url}. Is the printer switched on, and is the "
                            "address right?").format(url=url))
    return _wrong_kind(url, what)           # an answer that is not an answer


def _open(request, url, what, timeout):
    """urlopen with every error turned into what it means for the user."""
    import urllib.request
    try:
        return urllib.request.urlopen(request, timeout=timeout)
    except Exception as exc:
        raise _error(exc, url, what) from exc


def _headers(conn, extra=None) -> dict:
    """The headers of a request to a printer: the key is only ever a header, never a word
    of a message."""
    head = {"User-Agent": f"Volura/{__version__}"}
    if conn.api_key:
        head["X-Api-Key"] = conn.api_key
    head.update(extra or {})
    return head


# ------------------------------------------------------------------ what answers


def _get(conn, url, path, what, timeout) -> str:
    """What a printer answers to a small GET, as text."""
    import urllib.request
    request = urllib.request.Request(url + path, method="GET",
                                     headers=_headers(conn, {"Accept": "application/json"}))
    with _open(request, url, what, timeout) as resp:
        return resp.read().decode("utf-8", "replace")


def _field(text, name) -> str:
    """One field of an answer: JSON, or the XML tag that PrusaLink answers with by default."""
    try:
        data = json.loads(text)
    except ValueError:
        data = None
    if isinstance(data, dict) and data.get(name) is not None:
        return str(data[name])
    found = re.search(rf"<{name}>(.*?)</{name}>", text, re.S)
    return found.group(1).strip() if found else ""


def _moonraker_version(text) -> str:
    """The version of Moonraker out of what /server/info answered, without its tail."""
    try:
        answer = json.loads(text)
    except ValueError:
        return ""
    result = answer.get("result") if isinstance(answer, dict) else None
    version = str(result.get("moonraker_version", "")) if isinstance(result, dict) else ""
    # v0.9.1-23-g72b0e4a is the same program as v0.9.1: what stands behind it is git talk
    return version.split("-")[0]


def check(conn, timeout=5) -> str:
    """What answers at `conn`, as "OctoPrint 1.10.2", "Moonraker v0.9.1" or "PrusaLink MK4".

    A `LinkError` says what went wrong instead: nothing answers, the API key is not
    taken, or the printer at that address is another kind than the one filled in.
    """
    if conn.kind not in KINDS:
        raise LinkError(tr("Say what kind of printer {name} is.").format(name=conn.name))
    what = server_name(conn.kind)
    url = _address(conn, normalise_url(conn.url))
    if conn.kind == "octoprint":
        version = _field(_get(conn, url, "/api/version", what, timeout), "server")
    elif conn.kind == "moonraker":
        version = _moonraker_version(_get(conn, url, "/server/info", what, timeout))
    else:
        version = _field(_get(conn, url, "/api/version", what, timeout), "product")
    if not version:
        raise _wrong_kind(url, what)
    return f"{what} {version}"


# ------------------------------------------------------------------ what it is doing

#: what a printer can be doing, and what the status window says of it
STATES = ("idle", "printing", "paused", "error", "offline")
STATE_LABELS = {"idle": tr_noop("Idle"), "printing": tr_noop("Printing"),
                "paused": tr_noop("Paused"), "error": tr_noop("Error"),
                "offline": tr_noop("Offline")}
#: how the three programs word those states: the first group an answer agrees with wins
STATE_WORDS = (("error", ("error", "attention")),
               ("paused", ("paus",)),
               ("offline", ("offline", "closed", "without printer")),
               ("printing", ("print", "busy", "preparing", "operating", "starting",
                             "cancelling", "queued", "suspend")),
               ("idle", ("idle", "ready", "standby", "operational", "operations",
                         "complete", "finished", "cancel")))


@dataclass
class Status:
    """What a printer is doing: the state it is in and, while it prints, how far it got.

    `state` is one of `STATES` and `file` is what the printer calls the G-code; `progress`
    is a part of the whole print (0..1), `elapsed` and `left` are seconds and the four
    temperatures are degrees Celsius. What a printer does not say stays None, and an API
    key is never part of any of this.
    """

    state: str
    file: str = ""
    progress: float | None = None
    elapsed: float | None = None
    left: float | None = None
    nozzle: float | None = None
    nozzle_target: float | None = None
    bed: float | None = None
    bed_target: float | None = None


def state_label(state) -> str:
    """What the status window calls a state, as "Printing"; "Offline" for anything unknown."""
    return tr(STATE_LABELS.get(state, STATE_LABELS["offline"]))


def format_duration(seconds) -> str:
    """How long a stretch of seconds is, as the window says it: "1 h 05 min", "12 min",
    "< 1 min"; nothing at all when there is no stretch to say."""
    if seconds is None:
        return ""
    minutes = int(float(seconds) // 60)
    if minutes < 1:
        return tr("< 1 min")
    hours, mins = divmod(minutes, 60)
    return f"{hours} h {mins:02d} min" if hours else f"{mins} min"


def _state(word, default="idle") -> str:
    """One of `STATES` out of the word a program uses for it; `default` for a word there is not."""
    word = str(word or "").strip().lower()
    for name, words in STATE_WORDS:
        if any(part in word for part in words):
            return name
    return default


def _json(text) -> dict:
    """What a printer answered to a small GET, as the dict it is; an empty one if it is not JSON."""
    try:
        answer = json.loads(text)
    except ValueError:
        return {}
    return answer if isinstance(answer, dict) else {}


def _object(value) -> dict:
    """A piece of an answer that holds fields of its own: nothing when the printer left it out."""
    return value if isinstance(value, dict) else {}


def _number(value):
    """A time or a temperature as the printer gave it, or None when it gave nothing."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _fraction(percent) -> float | None:
    """A percentage, as OctoPrint and PrusaLink answer it, as a part of the whole."""
    value = _number(percent)
    return None if value is None else max(0.0, min(1.0, value / 100.0))


def _octoprint_status(conn, url, what, timeout) -> Status:
    """What OctoPrint says of the print it is running, and how hot its nozzle and bed are."""
    job = _json(_get(conn, url, "/api/job", what, timeout))
    done = _object(job.get("progress"))
    status = Status(_state(job.get("state")),
                    str(_object(_object(job.get("job")).get("file")).get("name", "")),
                    _fraction(done.get("completion")), _number(done.get("printTime")),
                    _number(done.get("printTimeLeft")))
    try:
        heat = _object(_json(_get(conn, url, "/api/printer", what, timeout)
                             ).get("temperature"))
    except LinkError as exc:
        if exc.code != 409:                 # 409: OctoPrint answers, the printer does not
            raise
        return replace(status, state="offline")
    hot, bed = _object(heat.get("tool0")), _object(heat.get("bed"))
    return replace(status, nozzle=_number(hot.get("actual")),
                   nozzle_target=_number(hot.get("target")), bed=_number(bed.get("actual")),
                   bed_target=_number(bed.get("target")))


def _moonraker_status(conn, url, what, timeout) -> Status:
    """What Klipper (Moonraker) says of the print it is running, and how hot it is."""
    got = _json(_get(conn, url, "/printer/objects/query?print_stats&display_status"
                               "&extruder&heater_bed", what, timeout))
    result = _object(got.get("result"))
    stats = _object(result.get("print_stats"))
    shown = _object(result.get("display_status"))
    hot, bed = _object(result.get("extruder")), _object(result.get("heater_bed"))
    went = _number(stats.get("print_duration"))
    part = _number(shown.get("progress"))
    part = None if part is None else max(0.0, min(1.0, part))
    # how long it still needs is not said anywhere: what is left of the print is what is
    # left of the time it has taken so far to get this far
    left = None if not part or not went else went / part - went
    return Status(_state(stats.get("state")), str(stats.get("filename", "")), part, went,
                  left, _number(hot.get("temperature")), _number(hot.get("target")),
                  _number(bed.get("temperature")), _number(bed.get("target")))


def _file_name_of(job) -> str:
    """The name PrusaLink gives the file of a job: what stands on the printer, or its name."""
    stored = _object(job.get("file"))
    return str(stored.get("display_name") or stored.get("name") or "")


def _prusalink_status(conn, url, what, timeout) -> Status:
    """What PrusaLink says of the print on its USB stick, and how hot the printer is."""
    got = _json(_get(conn, url, "/api/v1/status", what, timeout))
    printer, job = _object(got.get("printer")), _object(got.get("job"))
    state = _state(printer.get("state"))
    name = ""                               # the status names a job by its number, not its file
    if state == "printing":
        name = _file_name_of(_json(_get(conn, url, "/api/v1/job", what, timeout)))
    return Status(state, name, _fraction(job.get("progress")),
                  _number(job.get("time_printing")), _number(job.get("time_remaining")),
                  _number(printer.get("temp_nozzle")), _number(printer.get("target_nozzle")),
                  _number(printer.get("temp_bed")), _number(printer.get("target_bed")))


def status(conn, timeout=5) -> Status:
    """What the printer at `conn` is doing right now.

    What went wrong is the very `LinkError` `check` gives: nothing answers, the API key is
    not taken, or the printer at that address is another kind than the one filled in. A
    printer that is switched off while its OctoPrint answers is not said as an error but
    as state "offline". An API key stands in neither the `Status` and what comes with it,
    nor in anything asked of the printer.
    """
    if conn.kind not in KINDS:
        raise LinkError(tr("Say what kind of printer {name} is.").format(name=conn.name))
    what = server_name(conn.kind)
    url = _address(conn, normalise_url(conn.url))
    if conn.kind == "octoprint":
        return _octoprint_status(conn, url, what, timeout)
    if conn.kind == "moonraker":
        return _moonraker_status(conn, url, what, timeout)
    return _prusalink_status(conn, url, what, timeout)


# ------------------------------------------------------------------ a picture from the camera

#: the biggest picture a camera is asked for (bytes): more is nobody's status window
MAX_PICTURE = 5 * 1024 * 1024
#: how a JPEG and a PNG say what they are, at the very front of their bytes
PICTURE_MAGIC = ((b"\xff\xd8", "jpeg"), (b"\x89PNG\r\n\x1a\n", "png"))
#: how much of an answer has to be read to know which of those it is
PICTURE_HEAD = max(len(magic) for magic, _ in PICTURE_MAGIC)


def _picture_kind(data) -> str:
    """What kind of picture the front of those bytes is: "jpeg", "png"; "" for anything else."""
    for magic, name in PICTURE_MAGIC:
        if data.startswith(magic):
            return name
    return ""


def _text(value) -> str:
    """A field of an answer as the text it is; "" when the printer put something else there."""
    return str(value).strip() if isinstance(value, str) else ""


def _first(value) -> dict:
    """A piece of an answer that can also come as a list of them: the first of those."""
    if isinstance(value, list):
        value = value[0] if value else {}
    return _object(value)


def _absolute(base, url) -> str:
    """What a printer says a picture stands at, as a whole address to ask.

    A printer often names only "webcam/?action=snapshot": that picture comes from the
    printer itself, also one that stands under a path of its own. `base` is the address of
    that printer, made ready with `normalise_url`."""
    import urllib.parse
    text = _text(url)
    if not text or re.match(r"[A-Za-z][A-Za-z0-9+.-]*://", text):
        return text
    return urllib.parse.urljoin(base.rstrip("/") + "/", text)


def _camera_on(cam) -> bool:
    """Whether a webcam of a Moonraker answer is switched on; one that does not say is on."""
    return str(cam.get("enabled", True)).strip().lower() not in ("", "false", "0", "off", "no")


def _webcams(answer) -> list:
    """The webcams a Moonraker answer names, whether they stand under `result` or behind it."""
    found = answer.get("webcams")
    if found is None:
        found = _object(answer.get("result")).get("webcams")
    if isinstance(found, dict):                 # one webcam, not wrapped in a list
        found = [found]
    return [cam for cam in found if isinstance(cam, dict)] if isinstance(found, list) else []


def snapshot_url(conn, timeout=5) -> str | None:
    """Where the camera of `conn` has a still picture, or None when it has none to offer.

    OctoPrint keeps the address of its webcam in its settings, under the `webcam` section;
    a plugin that brings a camera of its own puts the same fields under `plugins` there, and
    the newest OctoPrints hand out a list of webcams instead of one, so all of those shapes
    are taken as they come. Moonraker lists every webcam it manages with the address of its
    picture, and which of them are switched on; the first one that is on is the one to look
    at. PrusaLink has no camera API of its own, so a Prusa printer never has a picture here.

    Half an address is a picture from the printer itself, so `conn.url` goes before it.
    Nothing answers, the printer will not say, or what it says is junk: all of that is no
    camera and never an error for whoever asks - a window that lays out statuses then has
    nothing to lay out.
    """
    where = ""
    if conn.kind not in KINDS:
        return None
    what = server_name(conn.kind)
    try:
        url = _address(conn, normalise_url(conn.url))
        if conn.kind == "octoprint":
            saved = _json(_get(conn, url, "/api/settings", what, timeout))
            for part in (saved.get("webcam"), _object(saved.get("plugins")).get("webcam")):
                where = _text(_first(part).get("snapshotUrl"))
                if where:
                    break
        elif conn.kind == "moonraker":
            got = _json(_get(conn, url, "/server/webcams/list", what, timeout))
            for cam in _webcams(got):
                where = _text(cam.get("snapshot_url")) if _camera_on(cam) else ""
                if where:
                    break
    except (LinkError, OSError, ValueError):        # nothing answers, or nothing to understand
        return None
    return _absolute(url, where) if where else None


def _head(resp) -> bytes:
    """The front of an answer, until there is enough of it to say which picture it is."""
    got = b""
    while len(got) < PICTURE_HEAD:
        block = resp.read(PICTURE_HEAD - len(got))
        if not block:
            break
        got += block
    return got


def _picture(resp, url) -> bytes:
    """What a camera answered, as bytes: a JPEG or a PNG, and at most MAX_PICTURE of them.

    The front is read first and looked at at once, so an answer that is a web page instead
    of a picture is said before any of it is put together, and a camera that will not stop
    sending is stopped before it fills the memory."""
    got = _head(resp)
    if not _picture_kind(got):
        raise LinkError(tr("{url} answers with nothing that is a picture to look "
                           "at.").format(url=url))
    while True:
        block = resp.read(CHUNK)
        if not block:
            return got
        got += block
        if len(got) > MAX_PICTURE:
            raise LinkError(tr("{url} sends a picture of more than {size} MB, which is too "
                               "much for this window.").format(
                                   url=url, size=MAX_PICTURE // (1024 * 1024)))


def snapshot(conn, url, timeout=5) -> bytes:
    """A picture from the camera of `conn`, as the JPEG or PNG bytes the camera sent.

    `url` is what `snapshot_url` said; half an address is asked from the printer itself.
    The same headers go with the request as with every other call, because an OctoPrint that
    wants a key wants that key of its camera too.

    What comes back has to say it is a JPEG or a PNG at its front, so a proxy or a login
    page that answers HTML instead of a picture is said as such; at most `MAX_PICTURE` bytes
    are read, in pieces of `CHUNK`. What went wrong is the very `LinkError` `check` gives.
    """
    if conn.kind not in KINDS:
        raise LinkError(tr("Say what kind of printer {name} is.").format(name=conn.name))
    import urllib.request
    what = server_name(conn.kind)
    base = _address(conn, normalise_url(conn.url))        # first the printer, then its camera
    where = _address(conn, _absolute(base, url))
    head = _headers(conn, {"Accept": "image/jpeg, image/png"})
    if not _same_host(base, where):
        # a camera on another machine never gets the key of the printer
        head.pop("X-Api-Key", None)
    request = urllib.request.Request(where, method="GET", headers=head)
    with _open(request, where, what, timeout) as resp:
        return _picture(resp, where)


def _same_host(a, b) -> bool:
    """Whether two addresses are the same machine: scheme, host and port alike."""
    import urllib.parse
    one, two = urllib.parse.urlsplit(a), urllib.parse.urlsplit(b)
    return (one.scheme.lower(), (one.hostname or "").lower(), one.port) ==         (two.scheme.lower(), (two.hostname or "").lower(), two.port)


# ------------------------------------------------------------------ the mesh of the bed

#: what a Klipper printer is asked for to know how flat its bed is
MOON_MESH = "/printer/objects/query?bed_mesh"


def _pair(value):
    """One corner of the box a mesh covers, as (x, y); None when that is not a pair of numbers."""
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    got = [_number(item) for item in value]
    if any(one is None or not math.isfinite(one) for one in got):
        return None
    return float(got[0]), float(got[1])


def _matrix(value):
    """A matrix an answer holds, as a 2D array of numbers; None when it is not a whole one.

    A bed that was never probed comes back as [] or as [[]], and a half answer as rows that are
    not all as long as each other, or that hold something that is no number, or a NaN; none of
    that is a mesh a window can paint numbers on."""
    if not isinstance(value, list) or not value:
        return None
    rows = []
    for row in value:
        if not isinstance(row, list) or not row:
            return None
        numbers = [_number(item) for item in row]
        if any(one is None or not math.isfinite(one) for one in numbers):
            return None
        rows.append(numbers)
    if len({len(row) for row in rows}) != 1:
        return None
    return np.asarray(rows, dtype=float)


def bed_mesh(conn, timeout=5) -> BedMesh | None:
    """The mesh of the bed of `conn`, or None when there is none to look at.

    Only Klipper has a bed mesh, and Moonraker answers it on `MOON_MESH` under
    `result.status.bed_mesh`: the profile it is saved under, the two corners of the bed it
    covers, the grid the probe touched (`probed_matrix`, whose first row is the lowest y of the
    bed) and the finer grid a print follows (`mesh_matrix`). The probed grid is what the numbers
    of a window come from; the interpolated one is gone when the printer did not send one, and
    the window then paints what was probed.

    None says there is nothing to look at, and is no error: the printer has no mesh loaded (an
    empty grid, or [[]]), or what it answers is of no use (rows of different length, a value
    that is no number, corners that bound nothing). OctoPrint is answered None without being
    asked, because it has a bed mesh only through a plugin, and PrusaLink not at all. What does
    not answer at all, or will not take the key, raises the very `LinkError` `check` gives: a
    printer that is silent is something else than a bed that was never probed, and a window says
    that difference.
    """
    if conn.kind != "moonraker":
        return None                     # nothing to ask of a printer without a bed mesh
    what = server_name(conn.kind)
    url = _address(conn, normalise_url(conn.url))
    got = _json(_get(conn, url, MOON_MESH, what, timeout))
    result = _object(got.get("result"))
    # the mesh stands under `status`; a Moonraker that answers it straight under `result` is
    # met the same way, as with the webcams of `snapshot_url`
    mesh = _object(_object(result.get("status")).get("bed_mesh"))
    mesh = mesh or _object(result.get("bed_mesh"))
    corner, other = _pair(mesh.get("mesh_min")), _pair(mesh.get("mesh_max"))
    probed = _matrix(mesh.get("probed_matrix"))
    if corner is None or other is None or probed is None:
        return None
    if other[0] <= corner[0] or other[1] <= corner[1]:
        return None                     # a bed without width or depth is no bed to paint
    # the interpolated grid is only for the picture: one printer does not send it and another
    # sends it wrong, and neither is a reason to hide a bed that was probed
    smooth = _matrix(mesh.get("mesh_matrix"))
    return BedMesh(_text(mesh.get("profile_name")), corner[0], corner[1], other[0], other[1],
                   probed, no_grid() if smooth is None else smooth)


# ------------------------------------------------------------------ sending a file


def _chunks(path, before: bytes, after: bytes, total: int, on_progress):
    """The pieces of a request body: `before`, the file, `after`, `CHUNK` bytes at a time.

    Handing urllib the body in pieces is what keeps a G-code of 50 MB out of the memory:
    urllib sends a piece and only then asks for the next one, so nothing is read or put
    together twice. `on_progress` follows every piece and may raise to stop the send;
    volura.background does that when the user presses Cancel.
    """
    done = 0

    def went(number: int):
        nonlocal done
        done += number
        if on_progress is not None:
            on_progress(1.0 if total <= 0 else min(1.0, done / total))

    if before:
        yield before
        went(len(before))
    with open(path, "rb") as handle:
        while True:
            block = handle.read(CHUNK)
            if not block:
                break
            yield block
            went(len(block))
    if after:
        yield after
        went(len(after))


def _multipart(boundary, name, fields):
    """The bytes around the file of a multipart/form-data body: the form fields, the head
    of the file part and the closing line of the body."""
    part = f"--{boundary}\r\n"
    before = ""
    for key, value in fields:
        before += part + f'Content-Disposition: form-data; name="{key}"\r\n\r\n{value}\r\n'
    before += (part + f'Content-Disposition: form-data; name="file"; filename="{name}"\r\n'
               'Content-Type: application/octet-stream\r\n\r\n')
    return before.encode("utf-8"), f"\r\n--{boundary}--\r\n".encode("utf-8")


def _send(conn, method, url, body, total, headers, what, timeout):
    """Send one whole G-code to the printer and wait for the answer."""
    import urllib.request
    head = _headers(conn, headers)
    # urllib counts a body itself and puts it in one bytes object unless it is told how
    # long it is; with the length in the header it sends the pieces as they come
    head["Content-Length"] = str(total)
    request = urllib.request.Request(url, data=body, method=method, headers=head)
    with _open(request, url, what, timeout) as resp:
        resp.read()


def upload(conn, path, name=None, start=False, on_progress=None, timeout=60):
    """Send the G-code at `path` to `conn`, under `name`, and start it at once when `start`.

    `name` defaults to `file_name()`. The file goes out in pieces of `CHUNK` bytes and
    `on_progress(fraction)` follows how far it got, so a send of 50 MB costs neither 50
    MB of memory nor a silent minute; what goes wrong is a `LinkError`, the same as
    `check` gives.
    """
    if conn.kind not in KINDS:
        raise LinkError(tr("Say what kind of printer {name} is.").format(name=conn.name))
    import urllib.parse
    what = server_name(conn.kind)
    url = _address(conn, normalise_url(conn.url))
    name = file_name(name)                      # safe for the printer, with its .gcode
    size = os.path.getsize(path)
    if conn.kind == "prusalink":            # PrusaLink takes the file itself, flags in headers
        head = {"Content-Type": "application/octet-stream", "Overwrite": "?1"}
        if start:
            head["Print-After-Upload"] = "?1"
        _send(conn, "PUT", url + "/api/v1/files/usb/" + urllib.parse.quote(name),
              _chunks(path, b"", b"", size, on_progress), size, head, what, timeout)
        return
    if conn.kind == "moonraker":
        target = "/server/files/upload"
        fields = [("print", "true")] if start else []
    else:                                   # OctoPrint also wants the file selected
        target = "/api/files/local"
        fields = [("select", "true"), ("print", "true")] if start else []
    boundary = uuid.uuid4().hex
    before, after = _multipart(boundary, name, fields)
    total = len(before) + size + len(after)
    _send(conn, "POST", url + target, _chunks(path, before, after, total, on_progress), total,
          {"Content-Type": f"multipart/form-data; boundary={boundary}"}, what, timeout)

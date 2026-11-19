"""Send a sliced print straight to the printer over the network.

No Qt in this file: the print preview asks `check` and `upload` in a background thread
and shows what comes back. Three programs answer on the network for a printer:
OctoPrint, Moonraker (the API of Klipper) and PrusaLink on a Prusa printer. A
`Connection` says which of them it is, where it stands and with which key it lets you
in; the settings keep them (volura.app.Settings) and Preferences > Printers edits them.

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
import os
import re
import uuid
from dataclasses import dataclass

from . import __version__
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
    """What went wrong speaking to a printer, in words for the user."""


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


def _wrong_kind(url, what) -> LinkError:
    """There is another kind of printer at that address than the one that was asked for."""
    return LinkError(tr("{url} does not answer as {what}. Is this printer really a "
                        "{what}?").format(url=url, what=what))


def _error(exc, url, what) -> LinkError:
    """What a failed request means for the user; an API key is never one of the words."""
    import ssl
    import urllib.error
    if isinstance(exc, urllib.error.HTTPError):
        if exc.code in (401, 403):
            return LinkError(tr("{url} will not take this API key. Copy it again from the "
                                "settings of the printer.").format(url=url))
        if exc.code == 404:
            return _wrong_kind(url, what)
        return LinkError(tr("{url} answered error {code}.").format(url=url, code=exc.code))
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

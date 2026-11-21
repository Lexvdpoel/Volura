"""Speaking to a printer over the network (`volura/printer_link.py`).

Every kind of printer is played here by a small HTTP server on 127.0.0.1 that answers
what OctoPrint, Moonraker or PrusaLink answer and remembers everything they sent, so the
tests look at the requests without anything leaving this computer. The printers page of
Window > Preferences, Send to printer… of the print preview and the Printer Status window
use the same fakes.

Run with:  python -m pytest tests/test_printer_link.py
"""
import http.server
import io
import json
import os
import socket
import ssl
import sys
import threading
import urllib.error
import urllib.request
from collections import namedtuple

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest  # noqa: E402

from volura import printer_link as pl  # noqa: E402

GCODE = ";FLAVOR:Marlin\nG28\nG1 X10 Y10 E1.0\nM600\n;LAYER_CHANGE\nG1 X11 Y11 E2.0\n"
#: one request, as a fake printer took it in
Request = namedtuple("Request", "method path headers body")


class Fake:
    """What a fake printer is, and everything it was asked to do."""

    def __init__(self, kind, api_key="", code=0):
        self.kind = kind                # "octoprint", "moonraker" or "prusalink"
        self.api_key = api_key          # the key it wants, "" when it wants none
        self.code = code                # a status to answer everything else with
        self.answers = {}               # a path to answer with this JSON from now on
        self.codes = {}                 # a path to answer with this status instead
        self.bodies = {}                # a path to answer with these bytes and this content type
        self.requests = []


class Handler(http.server.BaseHTTPRequestHandler):
    """Answers as the kind of printer of its server and keeps every request."""

    def do_GET(self):
        self._taken()

    def do_POST(self):
        self._taken()

    def do_PUT(self):
        self._taken()

    def _taken(self):
        fake = self.server.fake
        body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
        fake.requests.append(Request(self.command, self.path, dict(self.headers), body))
        if fake.code:
            return self._reply(fake.code, b"")
        if fake.api_key and (self.headers.get("X-Api-Key") or "") != fake.api_key:
            return self._reply(401, b'{"error": "Unauthorized"}')
        return self._as_kind()

    def _as_kind(self):
        """Only the routes the program of this fake really has; the rest is a 404."""
        fake, path = self.server.fake, self.path
        if path in fake.codes:                   # what a test asked this printer to answer
            return self._reply(fake.codes[path], b'{"error": "Not Doing That"}')
        if path in fake.answers:
            return self._reply(200, json.dumps(fake.answers[path]).encode())
        if path in fake.bodies:                  # what a camera or a page behind it answers
            status, kind, data = fake.bodies[path]
            return self._reply(status, data, kind)
        if fake.kind == "octoprint" and self.command == "GET" and path == "/api/version":
            return self._reply(200, json.dumps({"server": "1.10.2", "api": "0.1"}).encode())
        if fake.kind == "octoprint" and self.command == "POST" and path == "/api/files/local":
            return self._reply(200, b'{"files": {"local": {"name": "x.gcode"}}}')
        if fake.kind == "moonraker" and self.command == "GET" and path == "/server/info":
            return self._reply(200, json.dumps(
                {"result": {"klipper_status": "ready",
                            "moonraker_version": "v0.9.1-23-g72b0e4a"}}).encode())
        if (fake.kind == "moonraker" and self.command == "POST"
                and path == "/server/files/upload"):
            return self._reply(200, b'{"result": {"action": "start"}}')
        if fake.kind == "prusalink" and self.command == "GET" and path == "/api/version":
            return self._reply(200, b'<?xml version="1.0" encoding="UTF-8"?>\n'
                                    b"<api><version>1.0.4</version><product>MK4</product></api>")
        if (fake.kind == "prusalink" and self.command == "PUT"
                and path.startswith("/api/v1/files/usb/")):
            return self._reply(200, b"")
        return self._reply(404, b'{"error": "Not Found"}')

    def _reply(self, status, body, kind="application/json"):
        """Answer with exactly these bytes, in pieces.

        A reader who stops reading halfway - which is what a picture bigger than Volura
        asks for is about - is not something a printer is wrong about, so it is said here
        and dropped."""
        self.send_response(status)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        for start in range(0, len(body), pl.CHUNK):
            try:
                self.wfile.write(body[start:start + pl.CHUNK])
            except OSError:
                self.close_connection = True
                return

    def log_message(self, *args):
        pass


class FakePrinter:
    """One fake printer of one kind, on a free port of this computer."""

    def __init__(self, kind, api_key="", code=0):
        self.fake = Fake(kind, api_key, code)
        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.fake = self.fake
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server.server_address[1]}"

    @property
    def requests(self) -> list:
        return self.fake.requests

    @property
    def last(self) -> Request:
        return self.fake.requests[-1]

    def answer(self, path, payload):
        """What this printer answers to that path from now on: sent back as JSON."""
        self.fake.answers[path] = payload

    def refuse(self, path, code):
        """The status this printer answers that path with, instead of an answer."""
        self.fake.codes[path] = code

    def accept(self, path):
        """This printer answers that path again: what a test asked it to refuse is forgotten."""
        self.fake.codes.pop(path, None)

    def body(self, path, data, content_type="image/jpeg", status=200):
        """What this printer answers that path with from now on: exactly these bytes.

        What a camera or the web server behind one really sends: a JPEG, a PNG, or the
        web page that stands where the camera should stand."""
        self.fake.bodies[path] = (status, content_type, data)

    def connection(self, name="Octo", api_key="") -> pl.Connection:
        """A connection to this printer, of the kind it answers as."""
        return pl.Connection(name, self.fake.kind, self.url, api_key)

    def stop(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)


def header(request, name) -> str:
    """One header of a request a fake got, whatever case the sender used."""
    return next((v for k, v in request.headers.items() if k.lower() == name.lower()), "")


class Answer(io.BytesIO):
    """A response for a test that only looks at what was sent."""

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


@pytest.fixture()
def printer():
    """Fake printers on ports of this computer; they are closed after the test."""
    made = []

    def make(kind, api_key="", code=0):
        made.append(FakePrinter(kind, api_key, code))
        return made[-1]

    yield make
    for fake in made:
        fake.stop()


@pytest.fixture()
def gcode(tmp_path):
    """The G-code of a slice, as the print preview has it: standing in a temporary file."""
    path = tmp_path / "part.gcode"
    path.write_bytes(GCODE.encode())          # exactly these bytes, also on Windows
    return str(path)


def free_port() -> int:
    """A port that stands open, so asking it goes against nothing."""
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


# ------------------------------------------------------------------- the address


def test_addresses_are_made_ready():
    assert pl.normalise_url("192.168.1.20") == "http://192.168.1.20"
    assert pl.normalise_url("http://octoprint.local:5000/") == "http://octoprint.local:5000"
    assert pl.normalise_url("octoprint.local:5000/") == "http://octoprint.local:5000"
    assert (pl.normalise_url(" https://print.example.nl/octoprint/ ")
            == "https://print.example.nl/octoprint")
    assert pl.normalise_url("") == ""
    assert pl.normalise_url(None) == ""


def test_names_are_made_safe():
    assert pl.safe_name("C:\\prints\\my box.gcode") == "my_box.gcode"
    assert pl.safe_name("a/b/../bad*name?") == "bad_name"
    assert pl.safe_name("   ") == "volura"
    assert pl.file_name() == "volura.gcode"
    assert pl.file_name("my box", "lid") == "my_box-lid.gcode"
    assert pl.file_name("part.gcode") == "part.gcode"
    assert pl.file_name("  ", "") == "volura.gcode"
    assert pl.file_name(None) == "volura.gcode"


def test_a_connection_survives_the_settings():
    conn = pl.Connection("Octo, de printer", "octoprint", "http://a:5000", "a key")
    assert pl.from_json(pl.to_json(conn)) == conn
    assert pl.from_json("not json") is None
    assert pl.from_json('{"name": "x", "kind": "toslink"}') is None
    assert pl.from_json("[]") is None
    back = pl.from_json('{"name": "P", "kind": "prusalink", "url": "b"}')
    assert (back.name, back.kind, back.url, back.api_key) == ("P", "prusalink", "b", "")


# ---------------------------------------------------------------- what answered


def test_check_says_what_answered(printer):
    octo = printer("octoprint", api_key="secret")
    assert pl.check(octo.connection(api_key="secret")) == "OctoPrint 1.10.2"
    klipper = printer("moonraker")
    assert pl.check(klipper.connection("Klipper")) == "Moonraker v0.9.1"
    prusa = printer("prusalink", api_key="key")
    assert pl.check(prusa.connection("Prusa", api_key="key")) == "PrusaLink MK4"


def test_the_key_is_a_header_and_nothing_else(printer):
    octo = printer("octoprint", api_key="secret")
    pl.check(octo.connection(api_key="secret"))
    assert header(octo.last, "X-Api-Key") == "secret"
    assert "secret" not in octo.last.path and not octo.last.body
    klipper = printer("moonraker")
    pl.check(klipper.connection("Klipper"))            # on a lan a key is often not wanted
    assert header(klipper.last, "X-Api-Key") == ""
    pl.check(klipper.connection("Klipper", api_key="also secret"))
    assert header(klipper.last, "X-Api-Key") == "also secret"


def test_a_wrong_key_is_said_without_saying_it(printer):
    octo = printer("octoprint", api_key="the right one")
    with pytest.raises(pl.LinkError) as bad:
        pl.check(octo.connection(api_key="a wrong one"))
    said = str(bad.value)
    # the address of the fake printer has a random port, which can hold "401" itself
    told = said.replace(octo.url, "")
    assert "API key" in said and "401" not in told
    assert "right one" not in said and "wrong one" not in said


def test_nothing_answers_there():
    with pytest.raises(pl.LinkError) as bad:
        pl.check(pl.Connection("Thuis", "octoprint", f"127.0.0.1:{free_port()}"))
    assert "Nothing answers at" in str(bad.value)


def test_another_kind_of_printer_at_that_address(printer):
    klipper = printer("moonraker")                            # no /api/version here: 404
    with pytest.raises(pl.LinkError) as bad:
        pl.check(pl.Connection("Octo", "octoprint", klipper.url))
    assert "does not answer as OctoPrint" in str(bad.value)
    octo = printer("octoprint")                               # it answers, only not as PrusaLink
    with pytest.raises(pl.LinkError) as bad:
        pl.check(pl.Connection("Prusa", "prusalink", octo.url))
    assert "does not answer as PrusaLink" in str(bad.value)


def test_an_answer_that_is_no_answer(printer):
    octo = printer("octoprint", code=500)
    with pytest.raises(pl.LinkError) as bad:
        pl.check(octo.connection())
    assert "error 500" in str(bad.value)


def test_an_address_that_is_not_one(printer):
    with pytest.raises(pl.LinkError) as bad:
        pl.check(pl.Connection("Octo", "octoprint", "het printernetwerk"))
    assert "is not the address of a printer" in str(bad.value)
    with pytest.raises(pl.LinkError) as bad:                  # a kind of printer there is not
        pl.check(pl.Connection("Octo", "toslink", printer("octoprint").url))
    assert "kind of printer" in str(bad.value)


def test_a_certificate_that_is_not_trusted():
    problem = urllib.error.URLError(ssl.SSLCertVerificationError(1, "certificate verify failed"))
    said = str(pl._error(problem, "https://print.thuis", "OctoPrint"))
    assert "certificate" in said and "address" in said


# ------------------------------------------------------------ what a printer is doing

#: the paths a printer is asked about what it is doing, as volura/printer_link.py asks them
OCTO_JOB = "/api/job"
OCTO_PRINTER = "/api/printer"
MOON_QUERY = "/printer/objects/query?print_stats&display_status&extruder&heater_bed"
PRUSA_STATUS = "/api/v1/status"
PRUSA_JOB = "/api/v1/job"


def octo_job(state="Printing", name="box.gcode", done=41.0, went=600, left=900) -> dict:
    """What OctoPrint answers on /api/job: the G-code it has, and how far it got in it."""
    return {"job": {"file": {"name": name}}, "state": state,
            "progress": {"completion": done, "printTime": went, "printTimeLeft": left}}


def moon_print(state="printing", name="box.gcode", went=300.0, part=0.25) -> dict:
    """What Klipper (Moonraker) answers on the objects Volura asks for."""
    return {"result": {"print_stats": {"state": state, "filename": name,
                                       "print_duration": went},
                       "display_status": {"progress": part},
                       "extruder": {"temperature": 210.2, "target": 210.0},
                       "heater_bed": {"temperature": 59.4, "target": 60.0}}}


def prusa_status(state="PRINTING", done=60, went=1200, left=800, **temps) -> dict:
    """What PrusaLink answers on /api/v1/status: the printer, and its job if it has one."""
    printer = {"state": state}
    printer.update(temps)
    return {"printer": printer,
            "job": {"id": "1", "progress": done, "time_printing": went,
                    "time_remaining": left}}


def test_octoprint_says_what_it_is_doing(printer):
    octo = printer("octoprint", api_key="secret")
    octo.answer(OCTO_JOB, octo_job(name="my_box.gcode"))
    octo.answer(OCTO_PRINTER, {"temperature": {"tool0": {"actual": 205.4, "target": 210.0},
                                               "bed": {"actual": 60.2, "target": 60.0}}})
    got = pl.status(octo.connection(api_key="secret"))
    assert (got.state, got.file) == ("printing", "my_box.gcode")
    assert got.progress == pytest.approx(0.41), "OctoPrint says a percentage of the whole"
    assert (got.elapsed, got.left) == (600, 900)
    assert got.nozzle == pytest.approx(205.4) and got.nozzle_target == 210.0
    assert got.bed == pytest.approx(60.2) and got.bed_target == 60.0
    assert [r.path for r in octo.requests] == [OCTO_JOB, OCTO_PRINTER]
    assert header(octo.requests[1], "X-Api-Key") == "secret"
    assert "secret" not in repr(got), "the key came back with the answer"


def test_octoprint_that_is_not_printing_says_so(printer):
    octo = printer("octoprint")
    octo.answer(OCTO_JOB, octo_job(state="Operational", name="", done=None, went=None,
                                   left=None))
    octo.answer(OCTO_PRINTER, {"temperature": {"tool0": {"actual": 24.0, "target": 0.0}}})
    got = pl.status(octo.connection())
    assert (got.state, got.file, got.progress) == ("idle", "", None)
    assert got.elapsed is None and got.left is None
    assert got.nozzle == 24.0 and got.nozzle_target == 0.0
    assert got.bed is None, "this OctoPrint has nothing to say about a bed"


def test_octoprint_without_a_printer_is_offline_and_not_wrong(printer):
    octo = printer("octoprint")
    octo.answer(OCTO_JOB, octo_job(state="Offline", name="", done=None, went=None, left=None))
    octo.refuse(OCTO_PRINTER, 409)         # OctoPrint answers; the printer behind it does not
    got = pl.status(octo.connection())
    assert got.state == "offline"
    assert (got.file, got.progress, got.nozzle) == ("", None, None)


def test_klipper_says_what_it_is_doing(printer):
    klipper = printer("moonraker")
    klipper.answer(MOON_QUERY, moon_print())
    got = pl.status(klipper.connection("Klipper"))
    assert (got.state, got.file) == ("printing", "box.gcode")
    assert got.progress == pytest.approx(0.25), "Moonraker already speaks in parts of one"
    assert got.elapsed == 300
    assert got.left == pytest.approx(900), "a quarter of the print took five minutes"
    assert got.nozzle == pytest.approx(210.2) and got.nozzle_target == 210.0
    assert got.bed == pytest.approx(59.4) and got.bed_target == 60.0
    assert klipper.last.path == MOON_QUERY


def test_klipper_that_stands_by_needs_nothing(printer):
    klipper = printer("moonraker")
    klipper.answer(MOON_QUERY, {"result": {"print_stats": {"state": "standby", "filename": "",
                                                           "print_duration": 0},
                                            "display_status": {"progress": 0.0}}})
    got = pl.status(klipper.connection("Klipper"))
    assert (got.state, got.file, got.progress) == ("idle", "", 0.0)
    assert got.left is None, "nothing printing has no time left"
    assert got.nozzle is None and got.bed is None, "this Klipper did not answer those objects"


def test_klipper_that_is_done_is_not_stuck(printer):
    klipper = printer("moonraker")
    klipper.answer(MOON_QUERY, moon_print(state="complete", part=1.0, went=1200.0))
    got = pl.status(klipper.connection("Klipper"))
    assert got.state == "idle" and got.progress == 1.0 and got.left == 0


def test_prusalink_says_what_it_is_doing(printer):
    prusa = printer("prusalink", api_key="key")
    prusa.answer(PRUSA_STATUS, prusa_status(temp_nozzle=210.5, target_nozzle=210.0,
                                            temp_bed=60.0, target_bed=60.0))
    prusa.answer(PRUSA_JOB, {"file": {"display_name": "Box with lid",
                                      "path": "/usb/box_with_lid.gcode"}})
    got = pl.status(prusa.connection("Prusa", api_key="key"))
    assert (got.state, got.file) == ("printing", "Box with lid")
    assert got.progress == pytest.approx(0.6), "PrusaLink says a percentage of the whole"
    assert (got.elapsed, got.left) == (1200, 800)
    assert (got.nozzle, got.nozzle_target) == (210.5, 210.0)
    assert (got.bed, got.bed_target) == (60.0, 60.0)
    assert [r.path for r in prusa.requests] == [PRUSA_STATUS, PRUSA_JOB]
    assert header(prusa.requests[0], "X-Api-Key") == "key"


def test_prusalink_names_a_print_from_its_own_page(printer):
    prusa = printer("prusalink")
    prusa.answer(PRUSA_STATUS, prusa_status(state="PAUSED"))
    got = pl.status(prusa.connection("Prusa"))
    assert (got.state, got.file) == ("paused", "")
    assert [r.path for r in prusa.requests] == [PRUSA_STATUS], "only a print is asked for its file"
    prusa.answer(PRUSA_STATUS, prusa_status())
    prusa.answer(PRUSA_JOB, {"file": {"name": "box.gcode"}})
    assert pl.status(prusa.connection("Prusa")).file == "box.gcode", "a plain name is enough"


def test_prusalink_states_there_are_are_the_closest_of_the_five(printer):
    prusa = printer("prusalink")
    prusa.answer(PRUSA_JOB, {"file": {"name": "box.gcode"}})
    for word, state in (("IDLE", "idle"), ("PRINTING", "printing"), ("PAUSED", "paused"),
                        ("BUSY", "printing"), ("PREPARING", "printing"),
                        ("FINISHED", "idle"), ("ERROR", "error"), ("ATTENTION", "error")):
        prusa.answer(PRUSA_STATUS, prusa_status(state=word))
        assert pl.status(prusa.connection("Prusa")).state == state, word


def test_every_program_words_its_states_differently(printer):
    """Every answer of every program comes back as one of the five states."""
    octo = printer("octoprint")
    octo.answer(OCTO_PRINTER, {"temperature": {}})
    for word, state in (("Printing", "printing"), ("Paused", "paused"),
                        ("Cancelling", "printing"), ("Starting", "printing"),
                        ("Operational", "idle"), ("Offline", "offline"), ("Error", "error")):
        octo.answer(OCTO_JOB, octo_job(state=word, name="", done=None, went=None, left=None))
        assert pl.status(octo.connection()).state == state, f"OctoPrint says {word}"
    klipper = printer("moonraker")
    for word, state in (("printing", "printing"), ("paused", "paused"),
                        ("complete", "idle"), ("cancelled", "idle"), ("standby", "idle"),
                        ("error", "error")):
        klipper.answer(MOON_QUERY, moon_print(state=word))
        assert pl.status(klipper.connection("Klipper")).state == state, f"Klipper says {word}"


def test_a_state_there_is_not_is_said_as_the_closest_one():
    assert pl._state("PRINTING FROM SD CARD") == "printing"
    assert pl._state("Closed / without printer") == "offline"
    assert pl._state("Suspended") == "printing"
    assert pl._state(None) == "idle" and pl._state("") == "idle"


def test_the_states_a_window_shows_are_the_states_a_printer_has():
    assert set(pl.STATE_LABELS) == set(pl.STATES)
    assert pl.state_label("idle") == "Idle"
    assert pl.state_label("printing") == "Printing"
    assert pl.state_label("paused") == "Paused"
    assert pl.state_label("error") == "Error"
    assert pl.state_label("offline") == "Offline"
    assert pl.state_label("a state there is not") == "Offline"


def test_times_are_said_the_way_a_window_does():
    assert pl.format_duration(None) == ""
    assert pl.format_duration(9) == "< 1 min"
    assert pl.format_duration(59) == "< 1 min"
    assert pl.format_duration(720) == "12 min"
    assert pl.format_duration(3599) == "59 min"
    assert pl.format_duration(3600) == "1 h 00 min"
    assert pl.format_duration(3930) == "1 h 05 min"
    assert pl.format_duration(7320) == "2 h 02 min"


def test_what_went_wrong_is_said_the_way_check_says_it(printer):
    octo = printer("octoprint", api_key="the right one")
    octo.answer(OCTO_JOB, octo_job())
    with pytest.raises(pl.LinkError) as bad:
        pl.status(octo.connection(api_key="a wrong one"))
    assert "API key" in str(bad.value)
    assert "right one" not in str(bad.value) and "wrong one" not in str(bad.value)
    with pytest.raises(pl.LinkError) as bad:
        pl.status(pl.Connection("Thuis", "octoprint", f"127.0.0.1:{free_port()}"))
    assert "Nothing answers at" in str(bad.value)
    klipper = printer("moonraker")                       # it has no /api/job: another kind
    with pytest.raises(pl.LinkError) as bad:
        pl.status(pl.Connection("Octo", "octoprint", klipper.url))
    assert "does not answer as OctoPrint" in str(bad.value)
    with pytest.raises(pl.LinkError) as bad:
        pl.status(pl.Connection("Octo", "toslink", octo.url))
    assert "kind of printer" in str(bad.value)
    with pytest.raises(pl.LinkError) as bad:
        pl.status(pl.Connection("Octo", "octoprint", "het printernetwerk"))
    assert "is not the address of a printer" in str(bad.value)


def test_an_answer_that_is_no_answer_at_all(printer):
    klipper = printer("moonraker")
    klipper.answer(MOON_QUERY, {"result": {"print_stats": "not a dict at all"}})
    got = pl.status(klipper.connection("Klipper"))
    assert (got.state, got.file, got.progress) == ("idle", "", None)


# ------------------------------------------------------------- a picture from the camera

#: the paths a printer is asked about its camera, as volura/printer_link.py asks them
OCTO_SETTINGS = "/api/settings"
MOON_WEBCAMS = "/server/webcams/list"
#: where the picture of a camera stands, on the printer itself
SNAP = "/snapshot"
#: what a camera sends: the front of a JPEG and of a PNG, with a little of the picture after it
JPEG = b"\xff\xd8\xff\xe0" + b"j" * 40
PNG = b"\x89PNG\r\n\x1a\n" + b"p" * 40
HTML = b"<!DOCTYPE html><html><head></head><body>Sign in</body></html>"


def octo_webcam(url) -> dict:
    """What OctoPrint answers on /api/settings when its webcam has a picture at that address."""
    return {"webcam": {"streamUrl": "webcam/?action=stream", "snapshotUrl": url}}


def webcam(name, url, **fields) -> dict:
    """One webcam as Moonraker lists it: its picture address and what is switched on."""
    cam = {"uid": f"webcam_{name}", "name": name, "enabled": True, "type": "mjpeg_streamer",
           "stream_url": f"/{name}/stream", "snapshot_url": url}
    cam.update(fields)
    return cam


def moon_webcams(*cams) -> dict:
    """What Moonraker answers on /server/webcams/list about the webcams it manages."""
    return {"result": {"webcams": list(cams)}}


def test_octoprint_says_where_its_camera_has_a_picture(printer):
    octo = printer("octoprint", api_key="secret")
    octo.answer(OCTO_SETTINGS, octo_webcam("http://192.168.1.20:8080/?action=snapshot"))
    conn = octo.connection(api_key="secret")
    assert pl.snapshot_url(conn) == "http://192.168.1.20:8080/?action=snapshot"
    assert octo.last.path == OCTO_SETTINGS
    assert header(octo.last, "X-Api-Key") == "secret"
    assert "secret" not in octo.last.path, "the key stands in the address of a request"


def test_octoprint_that_names_its_camera_half_or_behind_plugins(printer):
    octo = printer("octoprint")
    conn = octo.connection()
    octo.answer(OCTO_SETTINGS, {"plugins":
                                {"webcam": {"snapshotUrl": "http://cam.thuis/snapshot.jpg"}}})
    assert pl.snapshot_url(conn) == "http://cam.thuis/snapshot.jpg", "a plugin camera is a camera"
    octo.answer(OCTO_SETTINGS, octo_webcam("webcam/?action=snapshot"))
    assert pl.snapshot_url(conn) == octo.url + "/webcam/?action=snapshot"
    octo.answer(OCTO_SETTINGS, octo_webcam("/webcam/?action=snapshot"))
    assert pl.snapshot_url(conn) == octo.url + "/webcam/?action=snapshot", "/ in front is the same"
    octo.answer(OCTO_SETTINGS, {"webcam": [{"snapshotUrl": "http://eerste.thuis/s.jpg"},
                                           {"snapshotUrl": "http://tweede.thuis/s.jpg"}]})
    assert pl.snapshot_url(conn) == "http://eerste.thuis/s.jpg", "the first camera is the one"
    both = octo_webcam("http://octoprint.thuis/a.jpg")
    both["plugins"] = {"webcam": {"snapshotUrl": "http://plugin.thuis/b.jpg"}}
    octo.answer(OCTO_SETTINGS, both)
    assert pl.snapshot_url(conn) == "http://octoprint.thuis/a.jpg", "the camera of OctoPrint first"


def test_a_camera_behind_an_octoprint_under_a_path_stands_under_that_path(printer):
    octo = printer("octoprint")
    octo.answer("/octoprint" + OCTO_SETTINGS, octo_webcam("webcam/?action=snapshot"))
    conn = pl.Connection("Octo", "octoprint", octo.url + "/octoprint")
    assert pl.snapshot_url(conn) == octo.url + "/octoprint/webcam/?action=snapshot"
    assert octo.last.path == "/octoprint" + OCTO_SETTINGS, "its settings are under that path too"


def test_an_octoprint_without_a_camera_or_without_an_answer_has_no_camera(printer):
    octo = printer("octoprint")
    conn = octo.connection()
    for answer in ({}, {"webcam": {}}, {"webcam": None}, {"webcam": []}, {"webcam": "camera"},
                   {"webcam": {"snapshotUrl": None}}, {"webcam": {"snapshotUrl": "   "}},
                   {"webcam": {"snapshotUrl": 7}}, {"plugins": None},
                   {"plugins": {"webcam": {"streamUrl": "webcam/?action=stream"}}}):
        octo.answer(OCTO_SETTINGS, answer)
        assert pl.snapshot_url(conn) is None, answer
    octo.refuse(OCTO_SETTINGS, 500)
    assert pl.snapshot_url(conn) is None, "an answer that is an error is no camera"
    assert pl.snapshot_url(pl.Connection("Thuis", "octoprint",
                                         f"127.0.0.1:{free_port()}")) is None, "nothing answers"
    klipper = printer("moonraker")        # it answers, only not as OctoPrint
    assert pl.snapshot_url(pl.Connection("Octo", "octoprint", klipper.url)) is None
    assert pl.snapshot_url(pl.Connection("Octo", "toslink", octo.url)) is None
    assert pl.snapshot_url(pl.Connection("Octo", "octoprint", "het printernetwerk")) is None


def test_klipper_lists_the_webcams_it_manages(printer):
    klipper = printer("moonraker")
    conn = klipper.connection("Klipper")
    klipper.answer(MOON_WEBCAMS, moon_webcams(
        webcam("uit", "http://uit.thuis/snapshot.jpg", enabled=False),
        webcam("aan", "/webcam/?action=snapshot")))
    assert pl.snapshot_url(conn) == klipper.url + "/webcam/?action=snapshot", \
        "the first webcam that is on is the one to look at"
    assert klipper.last.path == MOON_WEBCAMS
    klipper.answer(MOON_WEBCAMS, {"result": {"webcams": [
        {"name": "geen woord over aan", "snapshot_url": "http://stille.thuis/s.jpg"}]}})
    assert pl.snapshot_url(conn) == "http://stille.thuis/s.jpg", \
        "a webcam that does not say it is off, is on"
    klipper.answer(MOON_WEBCAMS, {"webcams": [webcam("los", "http://los.thuis/s.jpg")]})
    assert pl.snapshot_url(conn) == "http://los.thuis/s.jpg", "that list need not stand wrapped"


def test_klipper_without_a_webcam_to_look_at_says_so(printer):
    klipper = printer("moonraker")
    conn = klipper.connection("Klipper")
    for answer in ({}, {"result": {}}, {"result": "nonsens"}, {"result": {"webcams": []}},
                   {"result": {"webcams": "nonsens"}}, {"result": {"webcams": [7]}},
                   {"result": {"webcams": [webcam("uit", "http://uit.thuis/s.jpg",
                                                  enabled=False)]}},
                   {"result": {"webcams": [webcam("dicht", "http://dicht.thuis/s.jpg",
                                                  enabled="false")]}},
                   {"result": {"webcams": [{"name": "geen plaatje", "enabled": True}]}}):
        klipper.answer(MOON_WEBCAMS, answer)
        assert pl.snapshot_url(conn) is None, answer
    klipper.refuse(MOON_WEBCAMS, 404)
    assert pl.snapshot_url(conn) is None
    assert pl.snapshot_url(pl.Connection("Thuis", "moonraker",
                                         f"127.0.0.1:{free_port()}")) is None, "nothing answers"


def test_prusalink_has_no_camera_to_ask(printer):
    prusa = printer("prusalink", api_key="key")
    assert pl.snapshot_url(prusa.connection("Prusa", api_key="key")) is None
    assert prusa.requests == [], "PrusaLink has no camera API, so there is nothing to ask"


def test_a_camera_picture_is_the_bytes_the_camera_sent(printer):
    octo = printer("octoprint", api_key="secret")
    octo.answer(OCTO_SETTINGS, octo_webcam(SNAP))
    conn = octo.connection(api_key="secret")
    url = pl.snapshot_url(conn)
    assert url == octo.url + SNAP
    octo.body(SNAP, JPEG)
    assert pl.snapshot(conn, url) == JPEG
    got = octo.last
    assert (got.method, got.path) == ("GET", SNAP)
    assert header(got, "X-Api-Key") == "secret", "the camera hangs behind OctoPrint"
    assert header(got, "Accept").startswith("image/"), "a picture is asked, not an answer"
    octo.body("/shot.png", PNG, "image/png")
    assert pl.snapshot(conn, octo.url + "/shot.png") == PNG, "a PNG is as good as a JPEG"
    assert pl.snapshot(conn, SNAP) == JPEG, "half an address is asked from the printer itself"


def test_a_camera_on_another_machine_never_gets_the_key_of_the_printer(printer):
    """OctoPrint may name a camera on another machine; that one gets the picture asked, but
    the key of the printer stays with the printer."""
    octo = printer("octoprint", api_key="secret")
    cam = printer("octoprint")                       # another machine: another port
    cam.body(SNAP, JPEG)
    assert pl.snapshot(octo.connection(api_key="secret"), cam.url + SNAP) == JPEG
    assert header(cam.last, "X-Api-Key") == "", "the key went to another machine"


def test_a_camera_that_answers_a_web_page_says_so(printer):
    octo = printer("octoprint")
    for body in (HTML, b"nope", b""):
        octo.body(SNAP, body)
        with pytest.raises(pl.LinkError) as bad:
            pl.snapshot(octo.connection(), octo.url + SNAP)
        assert "picture" in str(bad.value) and octo.url in str(bad.value), body
    octo.body(SNAP, JPEG[:1])
    with pytest.raises(pl.LinkError):     # too little of a picture to know what it is
        pl.snapshot(octo.connection(), octo.url + SNAP)


def test_a_picture_bigger_than_five_megabytes_is_stopped(printer):
    octo = printer("octoprint")
    octo.body("/big", JPEG + b"0" * pl.MAX_PICTURE)
    with pytest.raises(pl.LinkError) as bad:
        pl.snapshot(octo.connection(), octo.url + "/big")
    assert "5 MB" in str(bad.value)
    whole = JPEG + b"0" * (pl.MAX_PICTURE - len(JPEG))
    octo.body("/groot", whole)
    assert pl.snapshot(octo.connection(), octo.url + "/groot") == whole, "exactly five is five"


def test_a_camera_that_will_not_answer_is_said_the_way_check_says_it(printer):
    octo = printer("octoprint", api_key="the right one")
    octo.body(SNAP, JPEG)
    with pytest.raises(pl.LinkError) as bad:
        pl.snapshot(octo.connection(api_key="a wrong one"), octo.url + SNAP)
    assert "API key" in str(bad.value)
    assert "right one" not in str(bad.value) and "wrong one" not in str(bad.value)
    with pytest.raises(pl.LinkError) as bad:
        pl.snapshot(pl.Connection("Thuis", "octoprint", f"127.0.0.1:{free_port()}"), SNAP)
    assert "Nothing answers at" in str(bad.value)
    with pytest.raises(pl.LinkError) as bad:
        pl.snapshot(pl.Connection("Octo", "toslink", octo.url), octo.url + SNAP)
    assert "kind of printer" in str(bad.value)
    with pytest.raises(pl.LinkError) as bad:
        pl.snapshot(pl.Connection("Octo", "octoprint", "het printernetwerk"), SNAP)
    assert "is not the address of a printer" in str(bad.value)


# ------------------------------------------------------------------- what is sent


def multipart_file(body, boundary) -> bytes:
    """The bytes of the `file` part of a multipart body, as a printer reads them."""
    _, _, rest = body.partition(b'name="file"')
    assert rest, "the body has no file part"
    _, _, content = rest.partition(b"\r\n\r\n")
    return content.partition(b"\r\n--" + boundary.encode())[0]


def test_octoprint_takes_the_file_and_wants_it_printed(printer, gcode):
    octo = printer("octoprint", api_key="secret")
    seen = []
    pl.upload(octo.connection(api_key="secret"), gcode, "my box.gcode", start=True,
              on_progress=seen.append)
    got = octo.last
    assert (got.method, got.path) == ("POST", "/api/files/local")
    assert header(got, "X-Api-Key") == "secret"
    assert header(got, "Content-Type").startswith("multipart/form-data; boundary=")
    boundary = header(got, "Content-Type").split("boundary=")[1]
    assert f'filename="my_box.gcode"' in got.body.decode("utf-8", "replace")
    assert multipart_file(got.body, boundary) == GCODE.encode()
    assert b'name="select"' in got.body and b'name="print"' in got.body
    assert got.body.endswith(b"\r\n--" + boundary.encode() + b"--\r\n")
    assert int(header(got, "Content-Length")) == len(got.body)
    assert seen[-1] == 1.0 and seen == sorted(seen)


def test_octoprint_without_start_leaves_the_flags_away(printer, gcode):
    octo = printer("octoprint")
    pl.upload(octo.connection(), gcode)
    assert b'name="print"' not in octo.last.body and b'name="select"' not in octo.last.body
    assert f'name="file"; filename="{pl.DEFAULT_NAME}.gcode"'.encode() in octo.last.body


def test_moonraker_takes_the_file_and_prints_it_when_asked(printer, gcode):
    klipper = printer("moonraker")
    pl.upload(klipper.connection("Klipper"), gcode, "box.gcode", start=True)
    got = klipper.last
    assert (got.method, got.path) == ("POST", "/server/files/upload")
    assert GCODE.encode() in got.body and got.body.count(GCODE.encode()) == 1
    assert b'name="print"' in got.body and b'name="select"' not in got.body
    assert header(klipper.last, "X-Api-Key") == ""


def test_prusalink_takes_the_file_alone_on_its_usb(printer, gcode):
    prusa = printer("prusalink", api_key="nonsense")
    steps = []
    pl.upload(prusa.connection("Prusa", api_key="nonsense"), gcode, "box with lid.gcode",
              start=True, on_progress=steps.append)
    got = prusa.last
    assert (got.method, got.path) == ("PUT", "/api/v1/files/usb/box_with_lid.gcode")
    assert got.body == GCODE.encode()                 # the raw file, nothing around it
    assert header(got, "X-Api-Key") == "nonsense"
    assert header(got, "Content-Type") == "application/octet-stream"
    assert header(got, "Overwrite") == "?1"
    assert header(got, "Print-After-Upload") == "?1"
    assert steps == [1.0]                             # one piece, all of it


def test_prusalink_leaves_the_printing_flag_away(printer, gcode):
    prusa = printer("prusalink")
    pl.upload(prusa.connection("Prusa"), gcode, "box.gcode")
    assert header(prusa.last, "Print-After-Upload") == ""


def test_a_printer_that_refuses_the_key_says_so(printer, gcode):
    prusa = printer("prusalink", api_key="the right one")
    with pytest.raises(pl.LinkError) as bad:
        pl.upload(prusa.connection("Prusa", api_key="wrong"), gcode)
    assert "API key" in str(bad.value) and "right one" not in str(bad.value)
    octo = printer("octoprint", code=404)
    with pytest.raises(pl.LinkError) as bad:
        pl.upload(octo.connection(), gcode)
    assert "does not answer as OctoPrint" in str(bad.value)


def test_a_big_file_goes_in_pieces_and_only_once(tmp_path, monkeypatch):
    """5 MB of G-code is handed over in pieces: never twice, and never all at once."""
    path = tmp_path / "big.gcode"
    path.write_bytes(b";LAYER\n" * (5 * 1024 * 1024 // 7))
    asked = []

    def urlopen(request, timeout=None, context=None):
        asked.append(request)
        return Answer(b"{}")

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    steps = []
    pl.upload(pl.Connection("Octo", "octoprint", "http://127.0.0.1:5000"), str(path),
              on_progress=steps.append)
    body = asked[0].data
    assert not isinstance(body, (bytes, bytearray, str)), "the whole body was made in memory"
    pieces = list(body)                               # what urllib was given to send
    size = path.stat().st_size
    assert max(len(piece) for piece in pieces) <= pl.CHUNK
    assert sum(len(piece) for piece in pieces) > size, "the file itself was not sent in pieces"
    assert sum(len(piece) for piece in pieces) == int(header(asked[0], "Content-Length"))
    assert steps and steps[-1] == 1.0 and steps == sorted(steps)


# --------------------------------------------------------------- in the windows

#: sliced G-code with two layers, so the print preview has something to show
SLICED = """\
;FLAVOR:Marlin
G28
;LAYER:0
G0 F6000 X10 Y10 Z0.2
;TYPE:WALL-OUTER
G1 F1200 X20 Y10 E1.0
G1 X20 Y20 E2.0
;TIME_ELAPSED:6.0
;LAYER:1
G0 F6000 X10 Y10 Z0.4
G1 F1200 X20 Y20 E3.0
;TIME_ELAPSED:12.0
M104 S0
"""


@pytest.fixture(scope="module")
def win():
    """The window the windows below sit in; its printers go once they all have been."""
    from PySide6.QtWidgets import QApplication
    from volura.app import MainWindow
    QApplication.instance() or QApplication(sys.argv[:1])
    w = MainWindow()
    yield w
    w.settings.printers = []
    w.settings.printer_cameras_on = []
    w.settings.save()
    w.doc.dirty = False
    w.close()


@pytest.fixture()
def sliced(tmp_path):
    """A slice as the print preview gets it: G-code in a temporary file, and read in."""
    from volura import gcode_preview as gp
    path = tmp_path / "box.gcode"
    path.write_bytes(SLICED.encode())          # exactly these bytes, also on Windows
    return str(path), gp.parse(SLICED.splitlines(), filament=1.75)


def press(dialog):
    """Stands in for a user who presses Send: the dialog accepts, and remembers what is on it."""
    from PySide6.QtWidgets import QDialog
    dialog.accept()
    return QDialog.Accepted


def test_a_printer_of_the_preferences_comes_back_after_a_restart(win):
    from volura.app import PreferencesDialog
    win.settings.printers = []
    win.settings.save()
    dlg = PreferencesDialog(win, "Printers")
    assert not dlg.printers and not dlg.p_name.isEnabled(), \
        "with no printer on the list there is nothing to fill in"
    dlg._add_printer()
    dlg.p_name.setText("Keuken")
    dlg.p_kind.setCurrentIndex(dlg.p_kind.findData("prusalink"))
    dlg.p_url.setText("prusa.thuis")
    dlg.p_key.setText("nonsense")
    dlg.accept()
    again = PreferencesDialog(win, "Printers")
    assert [again.printer_list.item(i).text()
            for i in range(again.printer_list.count())] == ["Keuken"]
    assert again.p_kind.currentData() == "prusalink"
    assert again.p_url.text() == "prusa.thuis" and again.p_key.text() == "nonsense"
    back = type(win.settings)()               # as after a restart of the program
    assert [(c.name, c.kind, c.url, c.api_key) for c in back.printers] == \
        [("Keuken", "prusalink", "prusa.thuis", "nonsense")]


def test_a_printer_is_only_kept_once_it_has_a_name_and_an_address(win):
    from volura.app import PreferencesDialog
    win.settings.printers = []
    win.settings.save()
    dlg = PreferencesDialog(win, "Printers")
    dlg._add_printer()                         # the name is there, the address is not
    dlg.apply()
    assert win.settings.printers == []
    dlg.p_url.setText("192.168.1.20:5000")
    dlg.apply()
    assert [c.url for c in win.settings.printers] == ["192.168.1.20:5000"]
    dlg._remove_printer()
    dlg.apply()
    assert win.settings.printers == []


def test_the_preferences_ask_the_printer_what_it_is(printer, win):
    from volura.app import PreferencesDialog
    octo = printer("octoprint", api_key="secret")
    dlg = PreferencesDialog(win, "Printers")
    dlg._add_printer()
    dlg.p_url.setText(octo.url)
    dlg.p_key.setText("a wrong one")
    dlg._test_printer()
    job = dlg._printer_job
    assert job.wait(20000), "the test connection did not come to an end"
    assert "API key" in dlg.p_answer.text() and "secret" not in dlg.p_answer.text()
    assert dlg.p_test.isEnabled(), "the button must work again after a wrong key"
    dlg.p_key.setText("secret")
    dlg._test_printer()
    job = dlg._printer_job
    assert job.wait(20000), job.error
    assert dlg.p_answer.text() == "OctoPrint 1.10.2"
    assert octo.requests, "the printer was never asked"


def test_send_waits_for_a_printer_under_the_preferences(win, sliced):
    from volura import print_preview as pp
    win.settings.printers = []
    win.settings.save()
    path, preview = sliced
    dlg = pp.PrintPreviewDialog(win, path, preview, (0.0, 0.0, 100.0, 100.0), "")
    try:
        assert not dlg.send_button.isEnabled(), "Send works without a printer to send to"
        assert "Preferences > Printers" in dlg.send_button.toolTip()
        assert dlg.save_button.isEnabled(), "saving the G-code needs no printer"
    finally:
        dlg.close()


def test_the_preview_sends_exactly_the_gcode_it_shows(printer, win, sliced, monkeypatch):
    from PySide6.QtCore import Qt
    from volura import print_preview as pp
    from volura.settings import qsettings
    octo = printer("octoprint", api_key="secret")
    win.settings.printers = [octo.connection(api_key="secret")]
    win.settings.save()
    path, preview = sliced

    def fill(dlg):
        """Send to printer…: a name of your own and start printing right away."""
        assert dlg.name.text() == f"{pl.DEFAULT_NAME}.gcode"
        assert dlg.printer.itemText(0) == "Octo"
        assert dlg.printer.itemData(0, Qt.ToolTipRole) == octo.url
        dlg.name.setText("my box")
        dlg.start.setChecked(True)
        return press(dlg)

    monkeypatch.setattr(pp.SendDialog, "exec", fill)
    dlg = pp.PrintPreviewDialog(win, path, preview, (0.0, 0.0, 100.0, 100.0), "")
    try:
        assert dlg.send_button.isEnabled() and not dlg.send_button.toolTip()
        dlg._send()
        assert dlg._job.wait(20000), dlg._job.error
        got = octo.last
        assert (got.method, got.path) == ("POST", "/api/files/local")
        boundary = header(got, "Content-Type").split("boundary=")[1]
        assert multipart_file(got.body, boundary) == SLICED.encode(), \
            "the printer got other bytes than the preview showed"
        assert 'filename="my_box.gcode"' in got.body.decode("utf-8", "replace")
        assert b'name="print"' in got.body, "starting printing was asked but not said"
        assert dlg.sent == "Sent my_box.gcode to Octo and started printing"
        assert win.hint.text() == dlg.sent
        assert str(qsettings().value("send_printer")) == "Octo"
        assert bool(qsettings().value("send_start"))
        assert dlg.send_button.isEnabled()
    finally:
        dlg.close()


def test_the_send_dialog_remembers_the_printer_it_last_went_to(printer, win):
    from PySide6.QtCore import Qt
    from volura import print_preview as pp
    from volura.settings import qsettings
    octo, prusa = printer("octoprint"), printer("prusalink", api_key="key")
    conns = [octo.connection("Octo"), prusa.connection("Prusa", "key")]
    was = qsettings().value("send_printer")
    try:
        qsettings().setValue("send_printer", "the printer of before there were any")
        dlg = pp.SendDialog(win, conns, "box.gcode")
        assert dlg.connection().name == "Octo", "the first printer on the list is not shown"
        qsettings().setValue("send_printer", "Prusa")
        again = pp.SendDialog(win, conns, "box")
        assert again.connection().name == "Prusa", "the printer of last time is not shown"
        assert again.printer.itemData(1, Qt.ToolTipRole) == prusa.url
        assert again.file_name() == "box.gcode"
        again.name.setText("  two boxes ")
        assert again.file_name() == "two_boxes.gcode"
        press(again)
        assert str(qsettings().value("send_printer")) == "Prusa"
    finally:
        qsettings().setValue("send_printer", was if was is not None else "")


class Box:
    """Stands in for QMessageBox, so a test can look at what the preview says."""

    def __init__(self):
        self.said = []

    def critical(self, parent, title, text):
        self.said.append((title, text))


def test_a_printer_that_refuses_the_key_is_said_in_the_preview(printer, win, sliced,
                                                               monkeypatch):
    from volura import print_preview as pp
    octo = printer("octoprint", api_key="the right one")
    win.settings.printers = [octo.connection(api_key="a wrong one")]
    win.settings.save()
    box = Box()
    monkeypatch.setattr(pp, "QMessageBox", box)
    monkeypatch.setattr(pp.SendDialog, "exec", press)
    path, preview = sliced
    dlg = pp.PrintPreviewDialog(win, path, preview, (0.0, 0.0, 100.0, 100.0), "")
    try:
        dlg._send()
        assert dlg._job.wait(20000), "a printer that refuses must still come to an end"
        assert box.said and "API key" in box.said[0][1]
        assert "right one" not in box.said[0][1] and "wrong one" not in box.said[0][1]
        assert dlg.sent is None, "nothing reached the printer, so nothing may be said"
        assert octo.requests and octo.last.method == "POST"
        assert dlg.send_button.isEnabled(), "Send may not be stuck after a refusal"
    finally:
        dlg.close()


def test_the_file_on_the_printer_is_called_after_the_model(win, sliced):
    from volura import print_preview as pp
    path, preview = sliced
    dlg = pp.PrintPreviewDialog(win, path, preview, (0.0, 0.0, 100.0, 100.0), "")
    old = win.doc.path
    try:
        win.doc.path = None
        assert dlg._send_name() == f"{pl.DEFAULT_NAME}.gcode"
        win.doc.path = os.path.join("somewhere", "My Box.skp")
        assert dlg._send_name() == "My_Box.gcode"
    finally:
        win.doc.path = old
        dlg.close()


# --------------------------------------------------------------- the status window


def opened(win):
    """The status window, asking only when the test says so.

    An open window asks every printer again each `POLL_SECONDS` by itself. A test that waits
    a while - a printer on a closed port takes seconds to fail on Windows, and longer when the
    machine is busy - would otherwise get an asking of the timer in between its own, and count
    one asking too many."""
    panel = win.printer_status()
    panel.timer.stop()
    return panel


def asked(panel):
    """Every asking the window has on its way, come to an end."""
    for job in [row.job for row in panel.rows]:
        assert job is not None, "the window asked nothing"
        assert job.wait(20000), f"asking the printer never came to an end: {job.error}"


def test_the_window_shows_what_two_printers_are_doing(printer, win):
    octo = printer("octoprint")
    octo.answer(OCTO_JOB, octo_job(name="my_box.gcode"))
    octo.answer(OCTO_PRINTER, {"temperature": {"tool0": {"actual": 205.0, "target": 210.0},
                                               "bed": {"actual": 60.0, "target": 60.0}}})
    klipper = printer("moonraker")
    klipper.answer(MOON_QUERY, moon_print(state="paused"))
    win.settings.printers = [octo.connection("Octo"), klipper.connection("Klipper")]
    win.settings.save()
    panel = opened(win)
    try:
        assert panel is win.printer_status(), "the second call makes a second window"
        panel.timer.stop()                # shown again, and its timer with it
        assert [row.conn.name for row in panel.rows] == ["Octo", "Klipper"]
        assert "127.0.0.1" in panel.rows[0].name.toolTip()
        panel.poll()
        asked(panel)
        first, second = panel.rows
        assert first.state.text() == "Printing" and first.file.text() == "my_box.gcode"
        assert first.bar.value() == 41
        assert first.times.text() == "10 min / 15 min"
        assert first.temps.text() == "nozzle 205 / 210 °C   ·   bed 60 / 60 °C"
        assert not first.error.text()
        assert second.state.text() == "Paused" and second.file.text() == "box.gcode"
        assert second.bar.value() == 25 and second.times.text() == "5 min / 15 min"
    finally:
        panel.close()


def test_a_printer_that_answers_nothing_says_so_under_its_own_row(printer, win):
    octo = printer("octoprint")
    octo.answer(OCTO_JOB, octo_job(state="Paused"))
    octo.answer(OCTO_PRINTER, {"temperature": {}})
    gone = pl.Connection("Weg", "octoprint", f"127.0.0.1:{free_port()}")
    win.settings.printers = [gone, octo.connection("Octo")]
    win.settings.save()
    panel = opened(win)
    try:
        panel.poll()
        asked(panel)
        bad, good = panel.rows
        assert bad.state.text() == "Offline" and bad.times.text() == ""
        assert "Nothing answers at" in bad.error.text()
        assert "127.0.0.1" in bad.error.text()
        assert good.state.text() == "Paused" and good.file.text() == "box.gcode"
        asked_jobs = len([r for r in octo.requests if r.path == OCTO_JOB])
        panel.poll()
        asked(panel)
        assert len([r for r in octo.requests if r.path == OCTO_JOB]) == asked_jobs + 1, \
            "a printer that was wrong once is asked again"
        assert panel.rows[1].state.text() == "Paused"
    finally:
        panel.close()


def test_a_window_without_printers_says_where_to_add_one(win):
    win.settings.printers = []
    win.settings.save()
    panel = win.printer_status()
    try:
        assert panel.rows == [] and panel.conns == []
        assert "Preferences > Printers" in panel.hint.text()
    finally:
        panel.close()


def test_the_window_asks_while_it_is_on_the_screen(win):
    from volura import printer_status as ps
    win.settings.printers = []
    win.settings.save()
    panel = win.printer_status()
    assert panel.timer.isActive(), "an open window follows its printers"
    assert panel.timer.interval() == ps.POLL_SECONDS * 1000
    panel.close()
    assert not panel.timer.isActive(), "a window that is gone asks nothing"
    assert win.printer_status() is panel, "the window there is is brought back, not made again"
    panel.close()


def test_the_window_follows_the_printers_under_preferences(printer, win):
    from volura.app import PreferencesDialog
    octo = printer("octoprint")
    prusa = printer("prusalink")
    win.settings.printers = [octo.connection("Octo")]
    win.settings.save()
    panel = win.printer_status()
    panel.close()                        # out of sight, and still following along
    dlg = PreferencesDialog(win, "Printers")
    dlg._add_printer()
    dlg.p_name.setText("Prusa")
    dlg.p_kind.setCurrentIndex(dlg.p_kind.findData("prusalink"))
    dlg.p_url.setText(prusa.url)
    dlg.apply()
    assert [row.conn.name for row in panel.rows] == ["Octo", "Prusa"]
    win.settings.printers = []
    win.settings.save()
    win.printer_status()                  # shown again: the rows follow what is there now
    assert panel.rows == [] and "Preferences > Printers" in panel.hint.text()
    panel.close()


# ------------------------------------------------------------ the cameras in that window


def shot(width=640, height=120) -> bytes:
    """A picture such a camera sends: a PNG of these measurements, as the bytes it is."""
    from PySide6.QtCore import QByteArray, QBuffer, QIODevice
    from PySide6.QtGui import QColor, QPixmap
    from PySide6.QtWidgets import QApplication
    QApplication.instance() or QApplication(sys.argv[:1])
    picture = QPixmap(width, height)
    picture.fill(QColor("forestgreen"))
    data = QByteArray()
    store = QBuffer(data)
    store.open(QIODevice.WriteOnly)
    assert picture.save(store, "PNG")
    store.close()
    return bytes(data)


def asks(fake, path) -> list:
    """The requests this fake printer got for that path, so far."""
    return [got for got in fake.requests if got.path == path]


def picture_of(row):
    """The picture this row shows, or None when it shows no picture."""
    picture = row.picture.pixmap()
    return picture if picture is not None and not picture.isNull() else None


def camera_panel(win, conns, on=()):
    """The status window over these printers, watching the cameras of these names.

    The window Volura keeps lives on from one test to the next, so a test says what it needs
    every time again: which printers there are and which of their cameras are watched. What a
    printer said about its camera is kept as long as that window lives, which is also why a
    test names printers of its own: a new port is another printer to that window. `on=None`
    leaves the remembered cameras alone, which is what a test that looks at remembering needs.
    """
    win.settings.printers = list(conns)
    if on is not None:
        win.settings.printer_cameras_on = list(on)
    win.settings.save()
    before = getattr(win, "_printer_status", None)
    if before is not None and before.isVisible():
        quiet(before)             # nothing of that window may answer the printers of this one
        before.close()            # a window takes its printers anew when it comes back on screen
    return opened(win)


def looked(panel):
    """Every asking where a printer's camera has a picture, come to an end."""
    ended(list(panel._cam_jobs.values()), "the printer about its camera")


def ended(jobs, what):
    """Every asking in `jobs`, come to an end."""
    for job in jobs:
        assert job is not None, f"{what} asked for nothing"
        assert job.wait(20000), f"{what} never came to an end: {job.error}"


def ticked(panel):
    """One asking of the whole window, come to an end: the status of every printer and the
    picture of every watched camera.

    A test calls this right after `panel.poll()`, while every row has its askings on its way.
    The jobs are taken before anything is waited, because waiting hands over the outcome of
    every job there is - also of the other rows - and then a job is gone before its turn."""
    jobs = []
    for row in panel.rows:
        jobs.append(row.job)
        if row.camera.isChecked():
            jobs.append(row.cam_job)
    ended(jobs, "the window")


def quiet(panel):
    """Every asking this window has on its way, come to an end: what a printer says about its
    camera, the state of every printer and the picture of every watched camera."""
    jobs = list(panel._cam_jobs.values())
    jobs += [job for job in (row.job for row in panel.rows) if job is not None]
    jobs += [job for job in (row.cam_job for row in panel.rows) if job is not None]
    for job in jobs:
        assert job.wait(20000), f"an asking of the window never came to an end: {job.error}"


def closed(panel):
    """Close this window with nothing of its own still on its way.

    A job that is still running when its window goes, answers a printer that may be gone with
    the test and says so to a window that is not there anymore; the noise of that lands in the
    test that follows, which has nothing to do with it."""
    quiet(panel)
    panel.close()


def cam_printer(printer, api_key="") -> FakePrinter:
    """A fake printer with a camera: its settings say where the picture stands, and it is there.

    What that camera really answers - a web page, nothing, a JPEG too big - is up to the test,
    which is why the fake comes back. What the printer is doing is a print that runs, so the
    row has something to say beside its picture."""
    octo = printer("octoprint", api_key=api_key)
    octo.answer(OCTO_SETTINGS, octo_webcam(SNAP))
    octo.answer(OCTO_JOB, octo_job())
    octo.answer(OCTO_PRINTER, {"temperature": {}})
    octo.body(SNAP, shot())
    return octo


def test_the_camera_button_is_only_there_for_a_printer_that_has_one(printer, win):
    octo = cam_printer(printer)
    klipper = printer("moonraker")
    klipper.answer(MOON_QUERY, moon_print())
    klipper.answer(MOON_WEBCAMS, moon_webcams(webcam("uit", "http://uit.thuis/s.jpg",
                                                     enabled=False)))
    panel = camera_panel(win, [octo.connection("Octo"), klipper.connection("Klipper")])
    try:
        looked(panel)
        watched, off = panel.rows
        assert watched.camera.isVisible() and not watched.camera.isChecked()
        assert "camera" in watched.camera.toolTip().lower()
        assert not off.camera.isVisible(), "a webcam that is switched off is nothing to watch"
        assert len(asks(octo, OCTO_SETTINGS)) == 1, "a printer is asked about its camera once"
        assert asks(klipper, MOON_WEBCAMS), "the Klipper was asked, and said nothing to watch"
    finally:
        closed(panel)


def test_a_watched_camera_puts_its_picture_under_the_row(printer, win):
    from volura import printer_status as ps
    octo = cam_printer(printer, api_key="secret")
    panel = camera_panel(win, [octo.connection("Octo", api_key="secret")], on=("Octo",))
    try:
        looked(panel)
        row = panel.rows[0]
        assert row.camera.isChecked(), "a camera the window just learned of is watched at once"
        first = row.cam_job
        # waiting for the lookups hands over every outcome there is: on a busy machine the first
        # picture can already be in by now, and then there is no asking left to look at
        assert first is not None or picture_of(row) is not None,             "watching a camera asks for its picture at once"
        if first is not None:
            panel.poll()                 # the asking of the printer comes along with it
            assert row.cam_job is first, "a camera that is still answering is not asked again"
            ticked(panel)
        assert len(asks(octo, SNAP)) == 1, "one asking of a printer is one picture"
        picture = picture_of(row)
        assert picture is not None, "the row shows no picture"
        assert (picture.width(), picture.height()) == (ps.CAMERA_WIDTH, 60), "the shape is kept"
        assert row.picture.isVisible() and not row.cam_error.isVisible()
        assert header(asks(octo, SNAP)[-1], "X-Api-Key") == "secret", "a camera hangs behind it"
        panel.poll()
        ticked(panel)
        assert len(asks(octo, SNAP)) == 2, "a watched picture follows the asking of the printer"
    finally:
        closed(panel)


def test_a_camera_that_answers_nothing_says_so_under_its_own_row(printer, win):
    octo = cam_printer(printer)
    octo.refuse(SNAP, 500)
    panel = camera_panel(win, [octo.connection("Octo")], on=("Octo",))
    try:
        looked(panel)
        row = panel.rows[0]
        panel.poll()
        ticked(panel)
        assert "error 500" in row.cam_error.text() and octo.url in row.cam_error.text()
        assert row.cam_error.isVisible() and row.cam_error.styleSheet().startswith("color:")
        assert picture_of(row) is None and not row.picture.isVisible()
        assert not row.error.text(), "the printer itself answers, only its camera does not"
        assert row.state.text() == "Printing", "a wrong camera says no wrong printer"
        octo.accept(SNAP)                          # the camera answers again
        octo.body(SNAP, shot(200, 100))            # and the next asking brings the picture back
        panel.poll()
        ticked(panel)
        assert not row.cam_error.text() and not row.cam_error.isVisible()
        assert picture_of(row).width() == 200, "a picture that fits is not made bigger"
    finally:
        closed(panel)


def test_bytes_the_camera_calls_a_picture_but_are_one_say_so(printer, win):
    octo = cam_printer(printer)
    octo.body(SNAP, JPEG)                    # the right beginning and nothing useable after it
    panel = camera_panel(win, [octo.connection("Octo")], on=("Octo",))
    try:
        looked(panel)
        row = panel.rows[0]
        panel.poll()
        ticked(panel)
        assert row.cam_error.text() == "The camera sent nothing that is a picture."
        assert row.cam_error.isVisible() and picture_of(row) is None
        octo.body(SNAP, shot(320, 200))      # the camera comes right with the next asking
        panel.poll()
        ticked(panel)
        assert picture_of(row) is not None, "a picture that does come is shown again"
        assert not row.cam_error.isVisible()
    finally:
        closed(panel)


def test_a_printer_that_said_nothing_about_its_camera_is_asked_again(printer, win):
    octo = cam_printer(printer)
    octo.refuse(OCTO_SETTINGS, 500)
    panel = camera_panel(win, [octo.connection("Octo")], on=("Octo",))
    try:
        looked(panel)
        row = panel.rows[0]
        assert not row.camera.isVisible(), "a printer that will not say has nothing to watch yet"
        assert row.cam_job is None, "a row does not watch a camera the printer never named"
        octo.accept(OCTO_SETTINGS)                 # it answers again, so the next time asks again
        quiet(panel)
        panel.close()
        panel.show()
        panel.timer.stop()
        looked(panel)
        assert row.camera.isVisible(), "what a printer would not say once, is asked a second time"
        assert len(asks(octo, OCTO_SETTINGS)) == 2
    finally:
        closed(panel)


def test_toggling_the_camera_off_stops_asking_the_camera(printer, win):
    octo = cam_printer(printer)
    panel = camera_panel(win, [octo.connection("Octo")], on=("Octo",))
    try:
        looked(panel)
        row = panel.rows[0]
        panel.poll()
        ticked(panel)
        assert row.picture.isVisible() and win.settings.printer_cameras_on == ["Octo"]
        row.camera.setChecked(False)
        assert row.cam_job is None and picture_of(row) is None
        assert not row.picture.isVisible() and not row.cam_error.text()
        assert win.settings.printer_cameras_on == [], "off is remembered too"
        went = len(asks(octo, SNAP))
        panel.poll()
        panel.poll()
        ticked(panel)
        assert len(asks(octo, SNAP)) == went, "a camera that is off is asked nothing at all"
    finally:
        closed(panel)


def test_which_cameras_are_on_comes_back_next_time_the_window_opens(printer, win):
    octo = cam_printer(printer)
    conns = [octo.connection("Octo")]
    panel = camera_panel(win, conns)
    try:
        looked(panel)
        assert not panel.rows[0].camera.isChecked()
        panel.rows[0].camera.setChecked(True)
        assert win.settings.printer_cameras_on == ["Octo"]
        assert type(win.settings)().printer_cameras_on == ["Octo"], "kept like any preference"
        quiet(panel)
        panel.close()
        panel = camera_panel(win, [], on=None)   # printers gone; what was on is not said here
        assert panel.rows == []
        panel = camera_panel(win, conns, on=None)        # nothing said: the settings say it
        row = panel.rows[0]
        assert row.camera.isChecked(), "the window forgot which camera was on"
        assert row.camera.isVisible(), "what it knows of that camera stayed known"
        assert len(asks(octo, OCTO_SETTINGS)) == 1, "once per printer, not once per opening"
    finally:
        closed(panel)


def test_a_window_that_is_out_of_sight_asks_nothing_of_a_camera(printer, win):
    octo = cam_printer(printer)
    panel = camera_panel(win, [octo.connection("Octo")], on=("Octo",))
    try:
        looked(panel)
        panel.poll()
        ticked(panel)
        row = panel.rows[0]
        assert row.picture.isVisible()
        panel.close()                          # out of sight: nothing asks a printer anything
        assert not panel.timer.isActive()
        went = len(asks(octo, SNAP))
        row.ask_picture()                      # the asking of a tick that never comes
        assert row.cam_job is None, "off the screen a watched camera asks for nothing"
        assert len(asks(octo, SNAP)) == went
        panel.show()
        panel.timer.stop()                     # back on the screen: the asking starts again
        ticked(panel)
        assert len(asks(octo, SNAP)) == went + 1, "a window that is back asks its cameras again"
        assert len(asks(octo, OCTO_SETTINGS)) == 1, "what it knew of that camera stayed known"
    finally:
        closed(panel)

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

    def _reply(self, status, body):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if body:
            self.wfile.write(body)

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
    assert "API key" in said and "401" not in said
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

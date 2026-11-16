"""The conversation with a GRBL board: what goes out, what comes back, on a thread.

`volura/laser.py` is the cable without Qt: `Session` holds what the board is and does and
what is waiting to go out, and `Worker` reads and writes a port on a thread of its own.
The tests here run against a board made of a list of bytes, so the whole conversation -
character counting, feed hold, a status report cut in two pieces, a cable that breaks -
is checked without a laser and without a serial port.

Run with:  python -m pytest tests/test_laser.py
"""
import os
import sys
import time
import types

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest  # noqa: E402

from volura import laser  # noqa: E402

REPORT = "<Idle|MPos:12.000,3.000,0.000|FS:0,0|Ov:100,100,100>"


class FakePort:
    """A GRBL board made of two lists: what it was told and what it answers."""

    def __init__(self, boot="Grbl 1.1h ['$' for help]", report=REPORT):
        self.written = []
        self.report = report
        self.closed = False
        self.raises = None
        self._in = []
        if boot:
            self.say(boot)

    # -- the serial port the thread sees
    @property
    def in_waiting(self):
        self._check()
        return len(self._in)

    def read(self, n):
        self._check()
        data = bytes(list(self._in)[:n])
        del self._in[:n]
        return data
    def write(self, data):
        self._check()
        data = bytes(data)
        self.written.append(data)
        self._answer(data)
        return len(data)

    def close(self):
        self.closed = True

    def _check(self):
        if self.raises is not None:
            raise self.raises

    # -- what a board would do with it
    def say(self, text):
        self._in.extend((str(text) + "\n").encode("ascii", "replace"))

    def _answer(self, data):
        if data == b"?":
            self.say(self.report)
            return
        for block in data.decode("ascii", "replace").split("\n"):
            block = block.strip()
            if block:
                self.say("error:1" if block.startswith("G99") else "ok")

    def text(self):
        """Everything the thread wrote, as one string."""
        return b"".join(self.written).decode("ascii", "replace")


def wait_for(what, timeout=3.0):
    """Wait until `what()` is true; the last value of `what()` when it is not."""
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        value = what()
        if value:
            return value
        time.sleep(0.005)
    return what()


# --------------------------------------------------------------------------  the cable

def test_a_computer_without_pyserial_says_what_to_install(monkeypatch):
    monkeypatch.setitem(sys.modules, "serial", None)
    with pytest.raises(laser.CableError) as e:
        laser.open_port("COM3")
    assert "pyserial" in str(e.value)


def test_a_computer_without_pyserial_sees_no_ports(monkeypatch):
    monkeypatch.setitem(sys.modules, "serial", None)
    assert laser.ports() == []
    assert laser.has_serial() is False


def test_a_port_that_cannot_be_opened_says_which_one_and_why(monkeypatch):
    def no_serial(**_kwargs):
        raise OSError("Access is denied")
    monkeypatch.setitem(sys.modules, "serial", types.SimpleNamespace(Serial=no_serial))
    with pytest.raises(laser.CableError) as e:
        laser.open_port("COM3", 115200)
    assert "COM3" in str(e.value) and "Access is denied" in str(e.value)


def test_the_port_that_opens_is_the_port_of_the_board(monkeypatch):
    opened = {}

    def fake_serial(**kwargs):
        opened.update(kwargs)
        return FakePort()
    monkeypatch.setitem(sys.modules, "serial", types.SimpleNamespace(Serial=fake_serial))
    port = laser.open_port("COM3", 57600)
    assert port is not None and opened["port"] == "COM3" and opened["baudrate"] == 57600
    assert opened["timeout"] == 0            # the thread never waits for the board


# -------------------------------------------------------------------------- the session

def test_a_block_goes_out_with_a_newline_behind_it():
    session = laser.Session()
    session.send("G21")
    assert session.pending() == b"G21\n"
    assert session.pending() == b""


def test_an_empty_block_is_nothing_to_write():
    session = laser.Session()
    session.send("   ")
    assert session.pending() == b""


def test_what_may_not_waits_goes_out_before_the_job():
    """A feed hold is read between the blocks that are waiting, so it may not queue up."""
    session = laser.Session()
    session.start(["G21", "G90"])
    session.realtime(laser.grbl.FEED_HOLD)
    assert session.pending() == b"!"
    assert session.pending() == b"G21\nG90\n"


def test_an_ok_frees_the_board_for_the_next_block():
    session = laser.Session(buffer=8)
    session.start(["G21", "G90", "M4"])
    assert session.pending() == b"G21\nG90\n"
    session.feed("ok")
    session.feed("ok")
    assert session.pending() == b"M4\n"


def test_a_refused_block_is_said_in_words():
    session = laser.Session()
    session.start(["G99 nonsense"])
    session.pending()
    session.feed("error:1")
    assert session.pending() == b""
    assert session.errors == [laser.grbl.error_text("1")]
    assert not session.running                 # the job ended with the block it refused


def test_a_status_report_is_taken_into_the_session():
    session = laser.Session()
    session.feed("<Run:34|MPos:20.000,5.000,0.000|WPos:10.000,2.000,0.000|FS:800,600"
                 "|Ov:90,50,80>")
    assert session.state == "Run" and session.line == 34
    assert session.mpos == (20.0, 5.0, 0.0) and session.wpos == (10.0, 2.0, 0.0)
    assert session.feed_rate == 800.0 and session.speed == 600.0
    assert (session.feed_override, session.rapid_override, session.power_override) \
        == (90.0, 50.0, 80.0)


def test_a_report_that_says_less_leaves_the_rest_as_it_was():
    session = laser.Session()
    session.feed(REPORT)
    session.feed("<Run|MPos:1.000,1.000,0.000>")
    assert session.state == "Run"
    assert session.mpos == (1.0, 1.0, 0.0)
    assert session.feed_override == 100.0


def test_the_first_thing_a_board_says_is_who_it_is():
    session = laser.Session()
    session.feed("Grbl 1.1h ['$' for help]")
    session.feed("$0=10 (step pulse, usec, X axis)")
    assert session.boot == "Grbl 1.1h ['$' for help]"
    assert session.messages == ["$0=10 (step pulse, usec, X axis)"]


def test_a_message_of_the_board_is_kept_for_the_log():
    session = laser.Session()
    session.feed("[MSG:GCODE source pinned]")
    assert session.messages == ["GCODE source pinned"]


def test_an_alarm_lands_in_the_state_and_in_the_errors():
    session = laser.Session()
    session.feed("Alarm:8")
    assert session.state == "Alarm"
    assert session.state_label == laser.grbl.state_label("Alarm")
    assert session.errors == ["Alarm:8"]


def test_a_job_cannot_be_started_while_another_one_runs():
    session = laser.Session()
    assert session.start(["G21"])
    assert session.start(["M5"]) is False
    assert session.sender.total == 1


def test_a_new_job_starts_after_the_first_one_is_over():
    session = laser.Session()
    session.start(["G21"])
    session.pending()
    session.feed("ok")
    assert session.pending() == b"" and not session.running
    assert session.start(["M5"])
    assert session.pending() == b"M5\n"


def test_pause_and_resume_go_through_the_board():
    session = laser.Session()
    session.start(["G21", "G90"])
    session.pause()
    assert session.paused and session.pending() == b"!"
    assert session.pending() == b""                  # nothing new goes out while paused
    session.resume()
    assert not session.paused and session.pending() == b"~"
    assert session.pending() == b"G21\nG90\n"


def test_stop_ends_the_job_and_resets_the_board():
    session = laser.Session()
    session.start(["G21", "G90"])
    session.stop()
    assert not session.running
    assert session.pending() == laser.grbl.RESET.encode()


def test_the_job_is_over_when_the_board_answered_the_last_block():
    session = laser.Session()
    session.start(["G21"])
    session.pending()
    session.feed("ok")
    assert session.pending() == b""
    assert not session.running and session.fraction == 1.0


def test_the_position_shows_the_work_and_else_the_machine():
    session = laser.Session()
    session.feed("<Idle|MPos:9.000,9.000,0.000>")
    assert session.position == (9.0, 9.0, 0.0)
    session.feed("<Idle|MPos:9.000,9.000,0.000|WPos:1.000,1.000,0.000>")
    assert session.position == (1.0, 1.0, 0.0)


def test_an_override_that_does_not_exist_writes_nothing():
    session = laser.Session()
    session.override("feed", "up1000")
    assert session.pending() == b""


# -------------------------------------------------------------------------- the thread

def test_the_thread_answers_the_board_and_asks_where_it_is():
    port, session = FakePort(), laser.Session()
    worker = laser.Worker(port, session, poll=0.01)
    worker.start()
    try:
        assert wait_for(lambda: session.boot)
        session.send("G21")
        assert wait_for(lambda: b"G21\n" in port.written)
        assert wait_for(lambda: session.state == "Idle"), session.state
        assert b"?" in port.written                    # the report was asked for
    finally:
        worker.shutdown()


def test_a_job_goes_out_block_by_block_as_the_board_answers():
    port = FakePort()
    session = laser.Session(buffer=16)
    worker = laser.Worker(port, session, poll=0.01)
    worker.start()
    try:
        session.start([f"G1 X{i}.000 F600" for i in range(6)])
        assert wait_for(lambda: not session.running)
        assert session.fraction == 1.0
        assert session.errors == []
    finally:
        worker.shutdown()


def test_a_line_that_arrives_in_two_pieces_still_arrives():
    port, session = FakePort(boot=""), laser.Session()
    port._in.clear()
    worker = laser.Worker(port, session, poll=10.0)
    worker.start()
    try:
        port._in.extend(b"<Idle|MPos:1.000,2.000")
        time.sleep(0.05)
        port._in.extend(b",3.000>\n")
        assert wait_for(lambda: session.mpos == (1.0, 2.0, 3.0))
        assert session.state == "Idle"
    finally:
        worker.shutdown()


def test_a_broken_cable_stops_the_thread_and_says_why():
    port, session = FakePort(), laser.Session()
    worker = laser.Worker(port, session, poll=0.01)
    worker.start()
    assert wait_for(lambda: session.boot)
    port.raises = OSError("the USB cable is gone")
    assert wait_for(lambda: not worker.is_alive())
    assert "USB cable" in worker.problem
    assert port.closed


def test_shutdown_ends_the_thread_and_lets_go_of_the_port():
    port, session = FakePort(), laser.Session()
    worker = laser.Worker(port, session, poll=0.01)
    worker.start()
    worker.shutdown()
    assert not worker.is_alive() and port.closed


def test_the_board_that_refuses_a_block_says_so_in_the_session():
    port = FakePort()
    session = laser.Session()
    worker = laser.Worker(port, session, poll=0.01)
    worker.start()
    try:
        session.start(["G21", "G99 nonsense", "M5"])
        assert wait_for(lambda: session.errors)
        assert session.errors[0] == laser.grbl.error_text("1")
        assert wait_for(lambda: not session.running)   # the rest ran to its end anyway
    finally:
        worker.shutdown()

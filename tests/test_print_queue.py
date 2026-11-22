"""The print queue: volura/print_queue.py and Window > Print Queue…

A queued print waits for the first printer that is free, and Volura starts nothing on a printer
before the user said that printer's bed is clear - a printer that just finished still has its
last print on the bed, and the next one would crash the nozzle into it. That word comes from one
button and is taken back by the printer saying it prints or stands paused, by Volura starting
something there, and by closing the window. `next_job` is that rule in one function; the window
is tested with fake printers that say what they are doing and say which G-code they got.

Run with:  python -m pytest tests/test_print_queue.py
"""
import os
import sys
import threading
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest  # noqa: E402
from PySide6.QtCore import QCoreApplication, QEvent, QEventLoop  # noqa: E402
from PySide6.QtGui import QPalette  # noqa: E402
from PySide6.QtWidgets import QApplication, QDialog  # noqa: E402

from volura import gcode_preview as gp, print_history, print_queue, printer_link  # noqa: E402
from volura.settings import qsettings  # noqa: E402

_app = QApplication.instance() or QApplication(sys.argv[:1])

#: a slice that changes filament and needs 42.5 minutes of 60 grams of red PLA
GCODE = """\
;FLAVOR:Marlin
;LAYER:0
G0 F6000 X10 Y10 Z0.2
;TYPE:WALL-OUTER
G1 F1200 X20 Y10 E1.0
G1 X20 Y20 E2.0
;LAYER:1
G0 F6000 X10 Y10 Z0.4
T1
;TYPE:FILL
G1 F1200 X20 Y10 E3.0
;TIME_ELAPSED:2550.0
"""

RED = [("Red PLA", "#ff0000", 1000.0, 60.0)]         # what the slice says it printed with
OCTO = printer_link.Connection("Octo", "octoprint", "192.168.1.20", "k")
PRUSA = printer_link.Connection("Prusa", "prusalink", "prusa.thuis")
#: a queue window that asks by itself asks every POLL_SECONDS; a test does not wait that long
FAST_MS = 20


@pytest.fixture(scope="module")
def win():
    """The window these previews, dialogs and queue windows sit in.

    Every setting these tests read is set before and after, so none of them depends on what
    another test left in the settings of this computer."""
    from volura.app import MainWindow
    w = MainWindow()
    empty(w)
    yield w
    empty(w)
    w.doc.dirty = False
    w.close()


def empty(w):
    """The settings of the window as they stand before and after these tests."""
    w.settings.print_queue = []
    w.settings.print_history = []
    w.settings.printers = []
    w.settings.spools = []
    w.settings.spool_for_slot = []
    w.settings.save()


@pytest.fixture()
def copies(tmp_path, monkeypatch):
    """The folder of the G-codes Volura keeps itself, in the folder of this test."""
    folder = tmp_path / "history" / "gcode"
    folder.mkdir(parents=True)
    monkeypatch.setattr(print_history, "gcode_dir", lambda create=True: str(folder))
    return folder


def reset(win, *items, printers=(), history=()):
    """The settings of the window as the next test starts them."""
    win.settings.print_queue = print_queue.from_values(items)
    win.settings.print_history = print_history.from_values(history)
    win.settings.printers = list(printers)
    win.settings.save()
    return win.settings.print_queue


def queued(win):
    """What the settings remember of the queue, as prints, in the order they wait."""
    return print_queue.records(win.settings.print_queue)


def slice_of(tmp_path, name="box"):
    """A G-code that is really there, for a print that can go to a printer."""
    path = tmp_path / f"{name}.gcode"
    path.write_text(GCODE, encoding="utf-8")
    return path


def waiting(name, source, printer=print_queue.ANY, grams=0.0, minutes=0.0):
    """A print on the queue with a G-code of its own, as the preview hands it over."""
    item = print_queue.QueueItem(name=name, printer=printer, grams=grams, minutes=minutes)
    item.gcode = print_history.keep_copy(str(source), item.id)
    return item


def kept(folder):
    """The names of the G-codes that stand in the folder Volura keeps itself."""
    return sorted(os.listdir(str(folder))) if os.path.isdir(str(folder)) else []


def in_folder(folder, path):
    """Is `path` one of the G-codes of the folder Volura keeps itself?"""
    return os.path.basename(str(path)) in kept(folder)


class States:
    """Stands in for printer_link.status: what every printer says of itself when it is asked.

    A printer that is not named says `idle`, so a test only says what it cares about; `say`
    changes the answer between two asking, and "wrong" is the one printer that answers nothing
    at all."""

    def __init__(self, **states):
        self.states = states            # the name of a printer to what it says it is doing
        self.asks = []                  # every printer that was asked, in the order they were

    def say(self, name, mode):
        """What this printer answers from now on."""
        self.states[name] = mode

    def __call__(self, conn, timeout=5):
        self.asks.append(conn.name)
        state = self.states.get(conn.name, "idle")
        if state == "wrong":
            raise printer_link.LinkError("Nothing answers at that address.")
        return printer_link.Status(state)


class Uploads:
    """Stands in for printer_link.upload: which G-code went to which printer, or nothing.

    `fails` says which prints do not reach the printer: True every time, or a number that often
    - so a test can let a printer be wrong once and answer well again after."""

    def __init__(self, fails=False):
        self.calls = []
        self.fails = fails

    def __call__(self, conn, path, name, start=False, on_progress=None):
        if self.fails is True or (isinstance(self.fails, int) and self.fails > 0):
            if isinstance(self.fails, int):
                self.fails -= 1
            raise printer_link.LinkError("nothing answers at the printer")
        self.calls.append((conn.name, os.path.basename(path), name, bool(start)))


class Waiting(Uploads):
    """Stands in for a G-code that stands waiting with the printer until the test lets it go.

    `on_the_way` says the file has reached the printer and is being handed over, `let_go` says
    it may; so a test can change the printers while a print is on its way."""

    def __init__(self):
        super().__init__()
        self.on_the_way = threading.Event()
        self.let_go = threading.Event()

    def __call__(self, conn, path, name, start=False, on_progress=None):
        self.on_the_way.set()
        assert self.let_go.wait(20), "the test did not let this G-code go on"
        return super().__call__(conn, path, name, start=start, on_progress=on_progress)


class Answer:
    """Stands in for QMessageBox: answers the question about a print without a dialog."""

    Yes, No = 1, 0

    def __init__(self, answer=1):
        self.answer = answer
        self.asked = []

    def question(self, *args, **kwargs):
        self.asked.append(args)
        return self.answer


def state(conn, mode="idle", bed_clear=False) -> dict:
    """How the queue looks at a printer: what it says it is doing, and what the user said about
    its bed."""
    return print_queue.printer_state(conn, mode, bed_clear)


# ------------------------------------------------------------------ one print on the queue


def test_a_queued_print_comes_back_out_of_the_settings_as_it_went_in():
    waits = print_queue.QueueItem(name="Bracket", gcode="/history/gcode/ab.gcode", grams=62.5,
                                 minutes=42.5, printer="Octo", added="2026-03-04T15:30:00")
    assert print_queue.from_dict(print_queue.to_dict(waits)) == waits
    assert print_queue.to_dict(waits)["grams"] == 62.5
    assert waits.added == "2026-03-04T15:30:00"
    # a print of nothing at all still waits: it says when it was put on the queue and that any
    # printer may take it
    blank = print_queue.QueueItem()
    assert blank.id and blank.added and blank.name == "Untitled"
    assert blank.gcode == "" and blank.printer == ""
    assert (blank.grams, blank.minutes) == (0.0, 0.0)
    assert print_queue.QueueItem().id != print_queue.QueueItem().id, \
        "two prints of nothing at all are still two prints"


def test_a_queued_print_with_junk_in_it_falls_back_to_what_is_known():
    junk = print_queue.from_dict({"id": None, "name": 5, "gcode": 7, "grams": "a lot",
                                 "minutes": -5, "printer": None, "added": "yesterday",
                                 "where_it_waits": "the kitchen"})
    assert junk.id, "a print without an id is still this print and not another one"
    assert junk.name == "5" and junk.gcode == "7" and junk.printer == ""
    assert junk.grams == 0.0 and junk.minutes == 0.0
    assert junk.added != "yesterday", "a moment that is no moment is the moment of writing it"
    assert "where_it_waits" not in print_queue.to_dict(junk)
    assert print_queue.from_dict("not a print").name == "Untitled"
    odd = print_queue.from_dict({"grams": "nan", "minutes": float("inf")})
    assert odd.grams == 0.0 and odd.minutes == 0.0
    assert print_queue.records(["junk", 7, None, {"id": "a"}])[0].id == "a"
    assert print_queue.records("a stray text") == [] and print_queue.records(7) == []
    assert print_queue.records(None) == []
    assert print_queue.from_values(["junk", {"name": " Box "}])[0]["name"] == "Box"
    loud = print_queue.QueueItem(name="base   plate\nwith\tlid", printer="  Octo  Print  ")
    assert loud.name == "base plate with lid" and loud.printer == "Octo Print"
    assert len(print_queue.QueueItem(name="k" * 900).name) == print_queue.MAX_TEXT
    assert print_queue.QueueItem(name="0").name == "0", "a name that is a zero is still a name"


def test_the_queue_keeps_its_line_and_lets_a_print_step_aside():
    one, two, three = (print_queue.QueueItem(name=name) for name in ("one", "two", "three"))
    line = print_queue.add(print_queue.add([], one), two)
    line = print_queue.add(line, three)
    assert [item["name"] for item in line] == ["one", "two", "three"]
    assert all(isinstance(item, dict) for item in line), "the settings keep plain dicts"
    back = print_queue.move(line, line[2]["id"], True)
    assert [item["name"] for item in back] == ["one", "three", "two"]
    assert [item["name"] for item in print_queue.move(back, back[0]["id"], True)] == \
        ["one", "three", "two"], "the first print in line does not go further forward"
    assert [item["name"] for item in print_queue.move(back, back[2]["id"], False)] == \
        ["one", "three", "two"], "the last print in line does not go further back"
    assert print_queue.move(line, "nobody", True) == line
    assert print_queue.move(line, 7, True) == line, "an id that is no id is nobody's print"
    assert [item["name"] for item in print_queue.add(["junk", 7, None], one)] == ["one"], \
        "what is no print is no print"
    assert print_queue.add([], None) == print_queue.from_values([])


def test_a_print_that_leaves_the_queue_loses_the_gcode_volura_kept(copies, tmp_path):
    mine = tmp_path / "not mine.gcode"
    mine.write_text(GCODE, encoding="utf-8")
    one, two = copies / "one.gcode", copies / "two.gcode"
    for path in (one, two):
        path.write_text(GCODE, encoding="utf-8")
    line = print_queue.records([print_queue.QueueItem(name="one", gcode=str(one)),
                                print_queue.QueueItem(name="two", gcode=str(two)),
                                print_queue.QueueItem(name="mine", gcode=str(mine))])
    out = print_queue.remove(line, line[1].id)
    assert [item["name"] for item in out] == ["one", "mine"]
    assert kept(copies) == ["one.gcode"], "the copy of the print that left is gone"
    assert print_queue.remove(out, "nobody") == out
    assert print_queue.cleared(out) == [] and kept(copies) == []
    assert mine.is_file(), "what the user saved itself is nobody else's to take away"
    assert print_queue.cleared([]) == [] and print_queue.remove([], "x") == []


def test_the_gcode_of_a_queued_print_stands_where_the_history_keeps_its_copies(copies, tmp_path):
    source = slice_of(tmp_path, "slice")
    item = print_queue.QueueItem(name="Box")
    assert item.gcode == "", "a print that says nothing has nothing to send yet"
    assert print_queue.store(None, item, source=source) is item
    assert in_folder(copies, item.gcode) and os.path.isfile(item.gcode)
    assert open(item.gcode, encoding="utf-8").read() == GCODE
    assert print_queue.gcode_of(item) == item.gcode
    assert source.is_file(), "the file it was copied from stays where it is"
    assert print_queue.gcode_of(print_queue.QueueItem(name="gone",
                                                     gcode=tmp_path / "gone.gcode")) is None
    assert print_queue.gcode_of(print_queue.QueueItem()) is None, "nothing to send is nothing"
    elsewhere = tmp_path / "somewhere"
    elsewhere.mkdir()
    assert print_queue.gcode_of(print_queue.QueueItem(name="a folder",
                                                     gcode=elsewhere)) is None
    assert print_queue.gcode_of("junk") is None
    assert print_queue.store(None, print_queue.QueueItem(name="nothing"), source="").gcode == ""


def test_a_queued_print_says_what_it_waits_for():
    waits = print_queue.QueueItem(name="Bracket", printer="Octo", grams=62.5, minutes=105.0)
    assert print_queue.printer_text(waits) == "Octo"
    assert print_queue.item_text(waits) == "Bracket on Octo - 62.5 g in 1 h 45 min"
    free = print_queue.QueueItem(name="Box")
    assert print_queue.printer_text(free) == "Any printer"
    assert print_queue.item_text(free) == "Box on Any printer", "a print that needs nothing "\
        "says no numbers"
    assert print_queue.added_text(1) == "Added to the queue (1 waiting)"
    assert print_queue.added_text(3) == "Added to the queue (3 waiting)"
    assert print_queue.added_text("junk") == "Added to the queue (0 waiting)"
    assert print_queue.waiting_text(1) == "1 print waiting"
    assert print_queue.waiting_text(2) == "2 prints waiting"
    assert print_queue.waiting_text("junk") == "0 prints waiting"


# ------------------------------------------------------------------ what may start where


def test_a_printer_that_is_idle_with_a_clear_bed_takes_the_first_print():
    item = print_queue.QueueItem(name="Box")
    got = print_queue.next_job([item], [state(OCTO, "idle", True)])
    assert got is not None and got[0].name == "Box" and got[1].name == "Octo"


def test_a_printer_whose_bed_nobody_said_clear_starts_nothing():
    item = print_queue.QueueItem(name="Box")
    assert print_queue.next_job([item], [state(OCTO, "idle", False)]) is None, \
        "a bed nobody looked at is a bed nothing is started on"
    assert print_queue.next_job([item], [state(OCTO)]) is None, "nothing was asked or said yet"


def test_a_printer_that_prints_or_stands_paused_takes_nothing_and_loses_the_word():
    item = print_queue.QueueItem(name="Box")
    for mode in print_queue.BUSY:
        assert print_queue.next_job([item], [state(OCTO, mode, True)]) is None, \
            f"a printer that says {mode} has something on its bed"
        assert not print_queue.keep_clear(mode, True), \
            "the word that the bed is clear is worth nothing from that moment on"
    assert print_queue.keep_clear("idle", True) and print_queue.keep_clear("error", True), \
        "a printer that is wrong says nothing about what lies on its bed"
    assert print_queue.keep_clear("printing", False) is False
    assert print_queue.keep_clear(None, True) is True, "a printer that says nothing says nothing"


def test_a_printer_in_error_or_offline_is_never_used():
    item = print_queue.QueueItem(name="Box")
    for mode in print_queue.BROKEN:
        assert print_queue.next_job([item], [state(OCTO, mode, True)]) is None
    assert print_queue.next_job([item], [state(OCTO, "what is that", True)]) is None, \
        "a state there is not is said as offline, and that is never used"
    assert state(OCTO, "what is that")["state"] == "offline"
    assert not print_queue.free(None) and not print_queue.free({})
    assert not print_queue.free({"conn": None, "state": "idle", "bed_clear": True})
    assert not print_queue.fits(item, {"conn": None, "state": "idle", "bed_clear": True})


def test_a_print_that_wants_a_printer_of_its_own_waits_for_that_one():
    wants = print_queue.QueueItem(name="Box", printer="Prusa")
    free = [state(OCTO, "idle", True), state(PRUSA, "idle", True)]
    assert print_queue.next_job([wants], [state(OCTO, "idle", True)]) is None, \
        "the wrong printer is not taken, however free and clear it is"
    got = print_queue.next_job([wants], free)
    assert got is not None and got[1].name == "Prusa"
    gone = print_queue.QueueItem(name="Box", printer="Keuken")
    assert print_queue.next_job([gone], free) is None, "a printer that is gone is not invented"


def test_two_free_printers_take_two_prints_and_one_printer_takes_one():
    one, two = (print_queue.QueueItem(name=name) for name in ("one", "two"))
    free = [state(OCTO, "idle", True), state(PRUSA, "idle", True)]
    first = print_queue.next_job([one, two], free)
    second = print_queue.next_job([two], free, taken=[first[1].name])
    assert first[0].name == "one", "the print that waits longest goes first"
    assert {first[1].name, second[1].name} == {"Octo", "Prusa"}, "both go in one round"
    only = [state(OCTO, "idle", True)]
    one_more = print_queue.next_job([one, two], only)
    assert print_queue.next_job([two], only, taken=[one_more[1].name]) is None, \
        "one printer never gets two prints at once"


def test_the_first_print_in_line_takes_the_first_printer_that_suits_it():
    wants, free_item = (print_queue.QueueItem(name=n, printer=p)
                        for n, p in (("wants", "Prusa"), ("free", "")))
    free = [state(OCTO, "idle", True), state(PRUSA, "idle", True)]
    got = print_queue.next_job([wants, free_item], free)
    assert (got[0].name, got[1].name) == ("wants", "Prusa"), \
        "the print that cares about the printer goes first, and takes its printer"
    assert print_queue.next_job([], free) is None and print_queue.next_job([wants], []) is None
    assert print_queue.next_job([wants], None) is None
    assert print_queue.next_job(["junk"], free) is None


# ------------------------------------------------------------------ the window


def made(win):
    """The queue window as it is before it comes on the screen: nothing was asked yet.

    Only a window that is shown asks its printers, so a test that looks at the list, the
    buttons or the word about a bed takes the window like this and never speaks to a network."""
    from volura import print_queue_dialog as qd
    panel = qd.PrintQueueWindow(win)
    panel.timer.stop()
    return panel


def opened(win):
    """The queue window on the screen, asking only when the test says so.

    A test that uses this has already said what every printer answers (see `States`): a window
    that is shown asks its printers at once, and that would be a real asking here. An open
    window also asks again every `POLL_SECONDS` by itself, and a test that waits a while - a job
    on its way delivers itself with the events of `wait` - would otherwise get an asking of the
    timer in between its own, and start something it never asked for."""
    panel = win.print_queue()
    panel.timer.stop()
    asked(panel)                            # what showing the window already asked
    panel.timer.stop()                      # `wait` may have brought the window back on screen
    return panel


def asked(panel):
    """Every asking of the printers the window has on its way, come to an end.

    The asking are taken together first and waited after that: waiting the answer of one row
    hands the answer of the next one over as well, and that row has none on its way any more by
    the time it is its turn (see tests/test_printer_link.py). With the fake of `States` in place
    this also says whether every printer was asked at all - which is the only thing to look at
    when the answers came in already, while the window was being shown."""
    said = printer_link.status
    for job in [row.job for row in panel.rows]:
        if job is not None:
            assert job.wait(20000), f"asking the printer never came to an end: {job.error}"
    QCoreApplication.processEvents()        # the last answer, so every row has it on screen
    if isinstance(said, States):            # the asking is over: now it is there to see
        missing = [row.conn.name for row in panel.rows if row.conn.name not in said.asks]
        assert not missing, f"the window asked nothing of {' and '.join(missing)}"


def sent_off(panel) -> bool:
    """The G-code the window is sending, come to an end; True when it reached the printer."""
    job = panel._job
    assert job is not None, "the queue started nothing"
    assert job.wait(20000), f"sending never came to an end: {job.error}"
    return job.done and not job.was_cancelled and job.error is None


def ticking(what, ms=1500) -> bool:
    """Let the event loop run until `what` is true, at most `ms` milliseconds; was it true?

    No nested event loop: on Windows such a loop sometimes sleeps while there is nothing to
    sleep for, and then nothing comes due until some unrelated window message arrives (see
    volura/background.py). So the events are handed over by hand, in short rounds."""
    deadline = time.monotonic() + ms / 1000.0
    while True:
        QCoreApplication.sendPostedEvents(None, QEvent.MetaCall)
        QCoreApplication.processEvents(QEventLoop.AllEvents)
        if what():
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.002)


def look(panel):
    """Ask every printer again and let the answers come in: the window now knows what it can do."""
    panel.poll()
    asked(panel)


def clear_all(panel, *names):
    """The user presses *Bed is clear* for every printer of these names, or for all of them."""
    for name in (names or [row.conn.name for row in panel.rows]):
        row = panel._row_of(name)
        assert row is not None, f"there is no printer {name}"
        row.clear.click()
        assert row.bed_clear, f"the word about the bed of {name} did not land"


def test_the_window_lists_the_prints_in_the_order_they_wait(win, tmp_path, copies):
    from volura import print_queue_dialog as qd
    source = slice_of(tmp_path, "bracket")
    reset(win, waiting("Bracket", source, grams=100.0, minutes=90.0),
          waiting("Box", source, printer="Prusa"), printers=[OCTO, PRUSA])
    panel = made(win)
    try:
        assert panel.windowTitle() == "Print Queue" and panel.table.columnCount() == 4
        assert [panel.table.item(r, 0).text() for r in range(2)] == ["Bracket", "Box"]
        assert [panel.table.item(r, 1).text() for r in range(2)] == ["Any printer", "Prusa"]
        assert panel.table.item(0, 2).text() == "100 g"
        assert panel.table.item(0, 3).text() == "1 h 30 min"
        assert panel.table.item(1, 2).text() == "" and panel.table.item(1, 3).text() == ""
        assert "Bracket on Any printer" in panel.table.item(0, 0).toolTip()
        assert [row.conn.name for row in panel.rows] == ["Octo", "Prusa"]
        assert "192.168.1.20" in panel.rows[0].name.toolTip()
        assert panel.rows[0].state.text() == "Offline", "nothing was asked yet"
        assert panel.rows[0].bed.text() == "Bed is not clear", "nobody said anything about a bed"
        assert panel.rows[0].clear.text() == "Bed is clear"
        assert panel.pause.text() == "Resume" and panel.paused, \
            "the queue starts paused, so nothing runs behind the user's back"
        assert panel.line.text() == "The queue stands paused, so nothing is sent"
        assert "Bed is clear" in panel.safety.text(), "the window says its own rule out loud"
        assert panel.timer.interval() == qd.POLL_SECONDS * 1000
        assert not panel.b_up.isEnabled() and not panel.b_down.isEnabled() and \
            not panel.b_remove.isEnabled(), "nothing is chosen"
        panel.table.selectRow(0)
        assert not panel.b_up.isEnabled() and panel.b_down.isEnabled() and \
            panel.b_remove.isEnabled()
        panel.table.selectRow(1)
        assert panel.b_up.isEnabled() and not panel.b_down.isEnabled()
    finally:
        panel.close()


def test_a_window_without_printers_says_where_to_add_one(win):
    reset(win)
    panel = made(win)
    try:
        assert panel.rows == [] and panel.conns == [] and panel.table.rowCount() == 0
        assert "Preferences > Printers" in panel.hint.text()
    finally:
        panel.close()


def test_a_clear_bed_starts_the_first_print_and_takes_the_word_back(win, tmp_path, copies,
                                                                   monkeypatch):
    source = slice_of(tmp_path, "bracket")
    one = waiting("One", source, grams=60.0, minutes=42.5)
    reset(win, one, waiting("Two", source), printers=[OCTO])
    monkeypatch.setattr(printer_link, "status", States())
    sent = Uploads()
    monkeypatch.setattr(printer_link, "upload", sent)
    panel = opened(win)
    try:
        assert panel.rows[0].state.text() == "Idle"
        panel.poll()
        assert panel._job is None and sent.calls == [], \
            "a paused queue sends nothing, even to a printer whose bed is clear"
        clear_all(panel, "Octo")
        panel.pause.click()                 # Resume: the queue runs, and this print can go
        assert sent_off(panel)
        assert sent.calls == [("Octo", os.path.basename(one.gcode), "One.gcode", True)], \
            "the queued G-code goes out and starts right away"
        assert [item.name for item in queued(win)] == ["Two"], "the print that went is off the "\
            "queue"
        history = print_history.records(win.settings.print_history)
        assert [entry.how for entry in history] == ["started"], "the print stands in the history"
        assert (history[0].name, history[0].printer) == ("One", "Octo")
        assert (history[0].grams, history[0].minutes) == (60.0, 42.5)
        assert in_folder(copies, history[0].gcode) and history[0].gcode != one.gcode
        assert not os.path.isfile(one.gcode), "the copy of the queue went with the print"
        assert not panel.rows[0].bed_clear, "there is a print on that bed now"
        assert panel.rows[0].bed.text() == "Bed is not clear"
        assert panel.table.rowCount() == 1 and panel.table.item(0, 0).text() == "Two"
        assert panel.line.text() == "1 print waiting"
        # the second print waits for a new word about that bed: the first one is still on it
        panel.poll()
        assert panel._job is None and len(sent.calls) == 1
        clear_all(panel, "Octo")
        panel.poll()
        assert sent_off(panel)
        assert [call[2] for call in sent.calls] == ["One.gcode", "Two.gcode"]
        assert queued(win) == [] and panel.table.rowCount() == 0
    finally:
        empty(win)
        panel.close()


def test_a_paused_queue_sends_nothing_at_all(win, tmp_path, copies, monkeypatch):
    source = slice_of(tmp_path, "bracket")
    reset(win, waiting("One", source), printers=[OCTO])
    monkeypatch.setattr(printer_link, "status", States())
    sent = Uploads()
    monkeypatch.setattr(printer_link, "upload", sent)
    panel = opened(win)
    try:
        clear_all(panel)
        for _ in range(3):
            look(panel)
        assert panel._job is None and sent.calls == [], "a paused queue keeps its hands off"
        assert panel.paused and panel.rows[0].bed_clear, "and it does not touch the word about "\
            "the bed either"
        panel.pause.click()
        assert sent_off(panel), "Resume is the only way the queue starts"
        assert len(sent.calls) == 1
        assert panel.line.text() == "0 prints waiting"
        panel.pause.click()                 # Pause again: the next print waits for the user
        assert panel.paused and panel.pause.text() == "Resume"
        assert panel.line.text() == "The queue stands paused, so nothing is sent"
    finally:
        empty(win)
        panel.close()


def test_a_print_waits_for_the_printer_it_was_pointed_at(win, tmp_path, copies, monkeypatch):
    source = slice_of(tmp_path, "bracket")
    reset(win, waiting("One", source, printer="Prusa"), printers=[OCTO, PRUSA])
    monkeypatch.setattr(printer_link, "status", States())
    sent = Uploads()
    monkeypatch.setattr(printer_link, "upload", sent)
    panel = opened(win)
    try:
        clear_all(panel, "Octo")            # the wrong one is said clear
        panel.pause.click()
        assert panel._job is None and sent.calls == []
        clear_all(panel, "Prusa")
        panel.poll()
        assert sent_off(panel)
        assert [call[0] for call in sent.calls] == ["Prusa"]
    finally:
        empty(win)
        panel.close()


def test_two_free_printers_take_the_next_two_prints_one_at_a_time(win, tmp_path, copies,
                                                                 monkeypatch):
    source = slice_of(tmp_path, "bracket")
    reset(win, waiting("One", source), waiting("Two", source, printer="Prusa"),
          waiting("Three", source), printers=[OCTO, PRUSA])
    monkeypatch.setattr(printer_link, "status", States())
    sent = Uploads()
    monkeypatch.setattr(printer_link, "upload", sent)
    panel = opened(win)
    try:
        clear_all(panel)
        panel.pause.click()
        assert sent_off(panel)
        assert len(sent.calls) == 1, "never two G-codes on their way at once"
        panel.poll()
        assert sent_off(panel)
        assert [call[0] for call in sent.calls] == ["Octo", "Prusa"]
        assert [call[2] for call in sent.calls] == ["One.gcode", "Two.gcode"], \
            "the print that cares about a printer went to that one"
        assert [item.name for item in queued(win)] == ["Three"]
    finally:
        empty(win)
        panel.close()


def test_a_printer_that_says_it_prints_loses_the_word_about_its_bed(win, tmp_path, copies,
                                                                   monkeypatch):
    """The rule of the queue in the window: only the button says a bed is clear, and a printer
    that starts something takes that word back."""
    source = slice_of(tmp_path, "bracket")
    reset(win, waiting("One", source), printers=[OCTO])
    said = States()
    monkeypatch.setattr(printer_link, "status", said)
    sent = Uploads()
    monkeypatch.setattr(printer_link, "upload", sent)
    panel = opened(win)
    try:
        clear_all(panel)
        assert panel.rows[0].bed.text() == "Bed clear"
        said.say("Octo", "printing")
        look(panel)
        assert panel.rows[0].state.text() == "Printing"
        assert not panel.rows[0].bed_clear, "somebody started something: there is something on "\
            "that bed"
        panel.rows[0].clear.click()
        said.say("Octo", "paused")
        look(panel)
        assert not panel.rows[0].bed_clear, "a paused print still lies on the bed"
        said.say("Octo", "idle")
        look(panel)
        assert panel.rows[0].state.text() == "Idle"
        assert not panel.rows[0].bed_clear, "being idle says nothing about what lies on the bed"
        panel.pause.click()                 # resumed, but there is no word about the bed left
        asked(panel)                        # what resuming asked, in before the printer says more
        assert panel._job is None and sent.calls == []
        said.say("Octo", "wrong")
        look(panel)
        assert panel.rows[0].state.text() == "Offline" and not panel.rows[0].bed_clear
        said.say("Octo", "idle")
        look(panel)
        panel.poll()
        assert panel._job is None and sent.calls == [], \
            "a printer that answers nothing, or has no word about its bed, is not used"
    finally:
        empty(win)
        panel.close()


def test_a_failing_upload_keeps_the_print_and_pauses_the_queue(win, tmp_path, copies,
                                                              monkeypatch):
    source = slice_of(tmp_path, "bracket")
    one = waiting("One", source)
    reset(win, one, waiting("Two", source), printers=[OCTO])
    monkeypatch.setattr(printer_link, "status", States())
    sent = Uploads(fails=1)                 # the printer is wrong this once
    monkeypatch.setattr(printer_link, "upload", sent)
    panel = opened(win)
    try:
        clear_all(panel)
        panel.pause.click()
        job = panel._job
        assert job is not None and job.wait(20000), \
            "a printer that does not answer must still come to an end"
        assert job.error is not None
        assert [item.name for item in queued(win)] == ["One", "Two"], \
            "the print stays on the queue"
        assert panel.table.rowCount() == 2
        assert "nothing answers" in panel.rows[0].error.text()
        # an answer about the state of that printer that comes in after the failed upload - an
        # asking that was already on its way - leaves the reason under the row standing
        row = panel.rows[0]
        row.job = object()
        row.answered(printer_link.Status(state="idle"))
        assert "nothing answers" in row.error.text(), "a late answer wiped what went wrong"
        assert panel.paused and panel.pause.text() == "Resume", \
            "a printer that is wrong may not eat the line behind it"
        assert panel.line.text() == "The queue stands paused, so nothing is sent"
        assert win.settings.print_history == [], "nothing reached the printer"
        assert os.path.isfile(one.gcode), "the print keeps its G-code to try again with"
        assert not panel.rows[0].bed_clear,             "an upload that failed halfway may have started the print: the bed is asked again"
        panel.pause.click()                 # resumed, but the bed was not said clear again
        assert panel._job is None, "a bed nobody said clear again got a print"
        panel.pause.click()                 # paused once more
        clear_all(panel)                    # the user looked at the bed and says it again
        panel.pause.click()                 # and the same print goes
        assert sent_off(panel)
        assert panel.rows[0].error.text() == "", "what was wrong is said away"
        assert [item.name for item in queued(win)] == ["Two"]
        assert [entry.how for entry in print_history.records(win.settings.print_history)] == \
            ["started"]
    finally:
        empty(win)
        panel.close()


def test_a_print_whose_gcode_is_gone_waits_and_does_not_block_the_line(win, tmp_path, copies,
                                                                      monkeypatch):
    source = slice_of(tmp_path, "bracket")
    gone = print_queue.QueueItem(name="Gone", gcode=tmp_path / "gone.gcode")
    reset(win, gone, waiting("Here", source), printers=[OCTO])
    monkeypatch.setattr(printer_link, "status", States())
    sent = Uploads()
    monkeypatch.setattr(printer_link, "upload", sent)
    dim = win.palette().color(QPalette.Disabled, QPalette.Text)
    panel = opened(win)
    try:
        assert panel.table.item(0, 0).foreground().color() == dim, \
            "a print with nothing to send stands grey"
        assert "no longer there" in panel.table.item(0, 0).toolTip()
        assert panel.table.item(1, 0).foreground().color() != dim
        clear_all(panel)
        panel.pause.click()
        assert sent_off(panel), "a print with no G-code does not block the ones behind it"
        assert [call[2] for call in sent.calls] == ["Here.gcode"]
        assert [item.name for item in queued(win)] == ["Gone"], "and it waits where it waited"
        assert panel.table.rowCount() == 1 and panel.table.item(0, 0).text() == "Gone"
    finally:
        empty(win)
        panel.close()


def test_the_window_moves_prints_up_and_down_and_removes_one(win, tmp_path, copies, monkeypatch):
    from volura import print_queue_dialog as qd
    source = slice_of(tmp_path, "bracket")
    reset(win, waiting("One", source), waiting("Two", source), waiting("Three", source),
          printers=[OCTO])
    answer = Answer(Answer.No)
    monkeypatch.setattr(qd, "QMessageBox", answer)
    panel = made(win)
    try:
        panel.table.selectRow(2)
        panel.b_up.click()
        assert [item.name for item in queued(win)] == ["One", "Three", "Two"]
        assert panel.table.currentRow() == 1
        assert [panel.table.item(r, 0).text() for r in range(3)] == ["One", "Three", "Two"]
        panel.b_down.click()
        assert [item.name for item in queued(win)] == ["One", "Two", "Three"]
        assert panel.table.currentRow() == 2
        panel.table.selectRow(0)
        panel._move(True)
        assert [item.name for item in queued(win)] == ["One", "Two", "Three"], \
            "the first print in line does not go further forward"
        panel.table.selectRow(0)
        panel._move(False)
        assert [item.name for item in queued(win)] == ["Two", "One", "Three"]
        assert panel.table.currentRow() == 1
        panel.table.selectRow(0)
        panel._remove_item()
        assert answer.asked and "Two" in answer.asked[0][2], "the print is named in the question"
        assert [item.name for item in queued(win)] == ["Two", "One", "Three"], \
            "no said no, so the print stays"
        answer.answer = Answer.Yes
        for _ in range(2):
            panel.table.selectRow(0)
            panel._remove_item()
        assert [panel.table.item(r, 0).text() for r in range(1)] == ["Three"]
        assert [item.name for item in queued(win)] == ["Three"]
        panel.table.selectRow(0)
        panel._remove_item()
        assert panel.table.rowCount() == 0 and queued(win) == []
        assert not panel.b_remove.isEnabled() and not panel.b_up.isEnabled()
        answer.asked.clear()
        panel._remove_item()                # nothing is chosen, and nothing is there
        panel.table.selectRow(0)
        panel._remove_item()
        assert answer.asked == [], "an empty queue is not asked about again"
    finally:
        empty(win)
        panel.close()


def test_removing_a_print_takes_the_gcode_volura_kept_of_it(win, tmp_path, copies, monkeypatch):
    from volura import print_queue_dialog as qd
    mine = tmp_path / "not mine.gcode"
    mine.write_text(GCODE, encoding="utf-8")
    one = waiting("One", slice_of(tmp_path, "one"))
    reset(win, one, print_queue.QueueItem(name="Mine", gcode=str(mine)), printers=[OCTO])
    monkeypatch.setattr(qd, "QMessageBox", Answer(Answer.Yes))
    panel = made(win)
    try:
        panel.table.selectRow(0)
        panel._remove_item()
        assert kept(copies) == [], "the copy Volura kept itself went with the print"
        assert [item.name for item in queued(win)] == ["Mine"]
        panel.table.selectRow(0)
        panel._remove_item()
        assert mine.is_file(), "the file the user saved stays exactly where it is"
        assert queued(win) == []
    finally:
        empty(win)
        panel.close()


def test_the_queue_takes_a_print_that_a_preview_put_on_it_afterwards(win, tmp_path, copies,
                                                                    monkeypatch):
    from volura import print_preview as pp
    source = slice_of(tmp_path, "bracket")
    reset(win, waiting("One", source), printers=[OCTO])
    monkeypatch.setattr(printer_link, "status", States())
    sent = Uploads()
    monkeypatch.setattr(printer_link, "upload", sent)
    monkeypatch.setattr(pp.QueueDialog, "exec", choose_any)
    panel = made(win)
    try:
        look(panel)
        clear_all(panel)
        panel.pause.click()
        assert sent_off(panel)
        assert [call[2] for call in sent.calls] == ["One.gcode"]
        assert not panel.rows[0].bed_clear, "there is a print on that bed now"
        pp.add_to_queue(panel, win.settings, [OCTO], str(source), "Two")
        assert [item.name for item in queued(win)] == ["Two"]
        panel.poll()
        assert panel.table.rowCount() == 1 and panel.table.item(0, 0).text() == "Two", \
            "a print a preview put on the queue comes onto the list by itself"
        assert panel._job is None and len(sent.calls) == 1, \
            "and nothing starts on that bed before the user said it clear again"
        clear_all(panel)
        panel.poll()
        assert sent_off(panel)
        assert [call[2] for call in sent.calls] == ["One.gcode", "Two.gcode"]
    finally:
        empty(win)
        panel.close()


def test_the_queue_keeps_working_behind_another_window_and_stops_when_it_closes(win, tmp_path,
                                                                               copies,
                                                                               monkeypatch):
    from volura import print_queue_dialog as qd
    source = slice_of(tmp_path, "bracket")
    reset(win, waiting("One", source), printers=[OCTO])
    said = States()
    monkeypatch.setattr(printer_link, "status", said)
    sent = Uploads()
    monkeypatch.setattr(printer_link, "upload", sent)
    panel = win.print_queue()
    try:
        assert panel.timer.isActive(), "an open window follows its queue and its printers"
        assert panel.timer.interval() == qd.POLL_SECONDS * 1000
        assert panel.paused, "and it stands paused until the user says Resume"
        asked(panel)
        panel.timer.setInterval(FAST_MS)    # the window asks by itself, also behind a window
        clear_all(panel)
        panel.pause.click()
        assert sent_off(panel)
        assert [call[2] for call in sent.calls] == ["One.gcode"]
        panel.hide()                        # behind another window: the queue goes on
        assert panel.timer.isActive(), "the queue works while this window exists"
        said.asks.clear()
        assert ticking(lambda: "Octo" in said.asks), \
            "a window that stands behind another window still asks its printers"
        said.say("Octo", "printing")
        assert ticking(lambda: panel.rows[0].state.text() == "Printing"), \
            "and it takes the answer of an asking it did off the screen"
        reset(win, waiting("Two", source), printers=[OCTO])
        assert ticking(lambda: panel.table.rowCount() == 1), \
            "a print that came along in the meantime is taken along off the screen too"
        assert panel.table.item(0, 0).text() == "Two"
        assert panel._job is None and len(sent.calls) == 1, \
            "and still nothing goes to a bed that nobody said clear"
    finally:
        panel.show()
        panel.close()
        empty(win)
    assert not panel.timer.isActive(), "closing this window stops the queue"
    assert panel.paused and panel.pause.text() == "Resume", "and puts it back paused"
    assert win._print_queue is None, "the next Window > Print Queue… is a window of itself"
    again = win.print_queue()
    try:
        assert again is not panel and again.paused, "a queue that comes back starts paused"
    finally:
        again.close()


def test_the_window_follows_the_printers_under_preferences(win, tmp_path, copies, monkeypatch):
    from volura.app import PreferencesDialog
    reset(win, printers=[OCTO])
    monkeypatch.setattr(printer_link, "status", States())
    panel = opened(win)
    try:
        dlg = PreferencesDialog(win, "Printers")
        dlg._add_printer()
        dlg.p_name.setText("Prusa")
        dlg.p_kind.setCurrentIndex(dlg.p_kind.findData("prusalink"))
        dlg.p_url.setText("prusa.thuis")
        dlg.apply()
        assert [row.conn.name for row in panel.rows] == ["Octo", "Prusa"]
        assert panel.rows[1].bed.text() == "Bed is not clear", "a new printer has no word yet"
        clear_all(panel)                    # the word about a bed is said anew, per printer
        assert [row.bed_clear for row in panel.rows] == [True, True]
    finally:
        empty(win)
        panel.close()


def test_a_printer_taken_away_while_its_print_is_on_its_way_changes_nothing(win, tmp_path,
                                                                           copies, monkeypatch):
    source = slice_of(tmp_path, "bracket")
    one = waiting("One", source)
    reset(win, one, waiting("Two", source), printers=[OCTO, PRUSA])
    monkeypatch.setattr(printer_link, "status", States())
    sent = Waiting()
    monkeypatch.setattr(printer_link, "upload", sent)
    panel = opened(win)
    try:
        clear_all(panel, "Octo")
        panel.pause.click()
        assert panel._job is not None, "the queue started nothing"
        assert sent.on_the_way.wait(20), "the G-code never got on its way"
        # while that file is being handed over, that printer is taken away under Preferences
        win.settings.printers = [PRUSA]
        win.settings.save()
        win.refresh_print_queue()
        QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
        assert [row.conn.name for row in panel.rows] == ["Prusa"]
        sent.let_go.set()
        assert sent_off(panel), "a print that was on its way is finished off"
        assert [call[0] for call in sent.calls] == ["Octo"]
        assert [item.name for item in queued(win)] == ["Two"]
        history = print_history.records(win.settings.print_history)
        assert [(entry.name, entry.printer, entry.how) for entry in history] == \
            [("One", "Octo", "started")], "the history says what printer took the print"
        assert in_folder(copies, history[0].gcode) and not os.path.isfile(one.gcode)
        assert panel._job is None and panel._going is None
        assert not panel.rows[0].bed_clear, "the printer that is left has no word about its bed"
        look(panel)
        assert panel._job is None, "and so the next print stands waiting"
    finally:
        sent.let_go.set()
        empty(win)
        panel.close()


def test_a_print_that_was_on_its_way_is_finished_off_after_the_window_closes(win, tmp_path,
                                                                            copies, monkeypatch):
    source = slice_of(tmp_path, "bracket")
    one = waiting("One", source)
    reset(win, one, waiting("Two", source), printers=[OCTO])
    monkeypatch.setattr(printer_link, "status", States())
    sent = Waiting()
    monkeypatch.setattr(printer_link, "upload", sent)
    panel = opened(win)
    try:
        clear_all(panel)
        panel.pause.click()
        assert sent.on_the_way.wait(20), "the G-code never got on its way"
        panel.close()                     # stopped while that file was still being handed over
        assert panel.paused and [item.name for item in queued(win)] == ["One", "Two"], \
            "closing the window stops the queue but takes nothing off it"
        sent.let_go.set()
        assert sent_off(panel), "what was on its way is finished off, not stopped halfway"
        assert [call[0] for call in sent.calls] == ["Octo"]
        assert [item.name for item in queued(win)] == ["Two"], \
            "and the print that reached the printer is off the queue"
        history = print_history.records(win.settings.print_history)
        assert [(entry.name, entry.printer, entry.how) for entry in history] == \
            [("One", "Octo", "started")], "the history says it even when the window is gone"
        assert in_folder(copies, history[0].gcode) and not os.path.isfile(one.gcode)
    finally:
        sent.let_go.set()
        empty(win)
        panel.close()


def test_the_window_offers_the_print_queue(win):
    reset(win)
    assert "Print Queue" in win.a_print_queue.text().replace("&", "")
    win.a_print_queue.trigger()
    panel = win._print_queue
    try:
        assert panel is not None and panel.windowTitle() == "Print Queue"
        assert win.print_queue() is panel, "the window there is is brought back, not made again"
    finally:
        panel.close()
    assert win._print_queue is None
    again = win.print_queue()
    try:
        assert again is not panel and again.paused, "a queue that comes back starts paused"
    finally:
        again.close()


def test_the_settings_remember_the_queue_and_survive_junk(win):
    from volura.app import Settings
    waits = print_queue.QueueItem(name="Bracket", gcode="/prints/bracket.gcode", grams=12.0,
                                 minutes=3.0, printer="Octo", added="2026-03-04T15:30:00")
    reset(win, waits)
    assert Settings().print_queue == [print_queue.to_dict(waits)]
    assert queued(win)[0].name == "Bracket" and queued(win)[0].added == "2026-03-04T15:30:00"
    odd = print_queue.QueueItem(name='Keutel «ç» – 中文 😀 "quoted" plate', printer="Ünïcode ✓")
    reset(win, odd)
    assert Settings().print_queue == [print_queue.to_dict(odd)]
    assert queued(win)[0].name == odd.name and queued(win)[0].printer == odd.printer
    qsettings().setValue("print_queue", "nonsense")
    qsettings().sync()
    assert Settings().print_queue == []
    qsettings().setValue("print_queue", ["junk", "also junk"])
    qsettings().sync()
    assert Settings().print_queue == [], "what is no print is no print"
    qsettings().remove("print_queue")
    qsettings().sync()
    assert Settings().print_queue == []


# ------------------------------------------------------------------ adding to the queue


def preview(win, tmp_path, grams=0.0, minutes=0.0):
    """The print preview of a small slice, with what the slice worked out for it."""
    from volura import print_preview as pp
    path = tmp_path / "box.gcode"
    path.write_text(GCODE, encoding="utf-8")
    return pp.PrintPreviewDialog(win, str(path), gp.parse(GCODE.splitlines()),
                                 (0.0, 0.0, 100.0, 100.0), "", None, RED,
                                 grams=grams, minutes=minutes)


def choose_any(dlg):
    """Stands in for a user who presses Add for a print any printer may take."""
    assert dlg.printer.itemText(0) == "Any printer"
    dlg.printer.setCurrentIndex(0)
    dlg.accept()
    return QDialog.Accepted


def choose_prusa(dlg):
    """Stands in for a user who points the print at one printer of their own."""
    assert dlg.printer.itemText(0) == "Any printer", "a print may wait for any printer"
    assert dlg.printer.itemText(1) == "Octo" and dlg.printer.itemText(2) == "Prusa"
    dlg.printer.setCurrentIndex(2)
    dlg.accept()
    return QDialog.Accepted


def neither(dlg):
    """Stands in for a user who cancels the question which printer this is for."""
    return QDialog.Rejected


def test_a_preview_offers_the_queue_only_with_a_printer_under_the_preferences(win, tmp_path,
                                                                             copies):
    reset(win)
    dlg = preview(win, tmp_path)
    try:
        assert not dlg.queue_button.isEnabled(), "a queue needs a printer to wait for"
        assert "Preferences > Printers" in dlg.queue_button.toolTip()
        assert dlg.save_button.isEnabled(), "saving the G-code needs no printer at all"
        assert not dlg.send_button.isEnabled(), "sending needs one just as much"
        reset(win, printers=[OCTO])
        dlg.printers = [OCTO]
        dlg._show_send()
        assert dlg.queue_button.isEnabled() and dlg.send_button.isEnabled()
        assert not dlg.queue_button.toolTip(), "a button that works has nothing to wait for"
    finally:
        empty(win)
        dlg.close()


def test_a_preview_puts_its_gcode_on_the_queue_and_keeps_a_copy(win, tmp_path, copies,
                                                               monkeypatch):
    from volura import print_preview as pp
    reset(win, printers=[OCTO])
    monkeypatch.setattr(pp.QueueDialog, "exec", choose_any)
    was, win.doc.path = win.doc.path, str(tmp_path / "bracket.volura")
    try:
        dlg = preview(win, tmp_path, grams=60.0, minutes=42.5)
        try:
            win.hint.setText("")
            dlg._add_queue()
            line = queued(win)
            assert [item.name for item in line] == ["bracket"]
            assert line[0].printer == "" and (line[0].grams, line[0].minutes) == (60.0, 42.5)
            assert line[0].added, "the queue says when the print started waiting"
            assert in_folder(copies, line[0].gcode)
            assert line[0].gcode != str(tmp_path / "box.gcode"), "the queue works on a copy of "\
                "its own: the file of a preview does not outlive it"
            assert open(line[0].gcode, encoding="utf-8").read() == GCODE
            assert win.hint.text() == "Added to the queue (1 waiting)"
            assert os.path.isfile(tmp_path / "box.gcode"), "the file of the preview itself stays"
            dlg._add_queue()                # the same print a second time is a second print
            assert [item.name for item in queued(win)] == ["bracket", "bracket"]
            assert len({item.id for item in queued(win)}) == 2
            assert win.hint.text() == "Added to the queue (2 waiting)"
            assert len(kept(copies)) == 2, "every print on the queue has a copy of its own"
            dlg.done(QDialog.Rejected)
            assert print_history.records(win.settings.print_history) == [], \
                "a print that only waits has printed nothing"
        finally:
            dlg.close()
    finally:
        win.doc.path = was
        empty(win)


def test_a_preview_can_point_its_print_at_one_printer(win, tmp_path, copies, monkeypatch):
    from volura import print_preview as pp
    reset(win, printers=[OCTO, PRUSA])
    monkeypatch.setattr(pp.QueueDialog, "exec", choose_prusa)
    dlg = preview(win, tmp_path)
    try:
        dlg._add_queue()
        assert [item.printer for item in queued(win)] == ["Prusa"]
        assert print_queue.printer_text(queued(win)[0]) == "Prusa"
    finally:
        empty(win)
        dlg.close()


def test_a_cancelled_question_queues_nothing(win, tmp_path, copies, monkeypatch):
    from volura import print_preview as pp
    reset(win, printers=[OCTO])
    monkeypatch.setattr(pp.QueueDialog, "exec", neither)
    dlg = preview(win, tmp_path)
    try:
        win.hint.setText("")
        dlg._add_queue()
        assert queued(win) == [] and kept(copies) == [], "nothing was copied and nothing waits"
        assert win.hint.text() == "", "nothing is said about a print that was not queued"
    finally:
        empty(win)
        dlg.close()


def test_a_preview_without_a_printer_queues_nothing(win, tmp_path, copies, monkeypatch):
    """A queue needs a printer to wait for: with none there is nothing to choose from, so the
    question is not even asked."""
    from volura import print_preview as pp
    seen = []

    def ask(dlg):
        seen.append(dlg.printer.count())        # "Any printer" and every printer of its own
        return choose_any(dlg)

    monkeypatch.setattr(pp.QueueDialog, "exec", ask)
    reset(win)
    dlg = preview(win, tmp_path)
    try:
        assert not dlg.printers
        assert pp.add_to_queue(dlg, win.settings, dlg.printers, str(tmp_path / "box.gcode"),
                               "Box") is None
        assert queued(win) == [] and kept(copies) == []
        assert seen == [], "there is nothing to choose from, so nothing is asked"
        assert pp.add_to_queue(dlg, win.settings, [OCTO], str(tmp_path / "box.gcode"),
                               "Box") is not None
        assert seen == [2] and [item.name for item in queued(win)] == ["Box"]
    finally:
        empty(win)
        dlg.close()


def test_the_history_can_send_a_print_to_the_queue(win, tmp_path, copies, monkeypatch):
    from volura import print_history_dialog as hd, print_preview as pp
    source = slice_of(tmp_path, "bracket")
    reset(win, printers=[OCTO],
          history=[print_history.Entry(name="Bracket", how=print_history.STARTED,
                                       gcode=str(source), grams=60.0, minutes=42.5)])
    monkeypatch.setattr(pp.QueueDialog, "exec", choose_any)
    dlg = hd.PrintHistoryDialog(win)
    try:
        dlg.table.selectRow(0)
        assert dlg.b_queue.isEnabled(), "the G-code is there and a printer is set up"
        dlg._add_queue()
        line = queued(win)
        assert [item.name for item in line] == ["Bracket"]
        assert (line[0].grams, line[0].minutes) == (60.0, 42.5)
        assert in_folder(copies, line[0].gcode) and line[0].gcode != str(source)
        assert source.is_file(), "the print works on a copy of its own"
        assert win.hint.text() == "Added to the queue (1 waiting)"
        assert [entry.how for entry in print_history.records(win.settings.print_history)] == \
            ["started"], "a print that waits changes nothing of the history"
    finally:
        empty(win)
        dlg.close()


def test_add_to_queue_needs_a_gcode_and_a_printer(win, tmp_path, copies, monkeypatch):
    from volura import print_history_dialog as hd, print_preview as pp
    monkeypatch.setattr(pp.QueueDialog, "exec", choose_any)
    there = slice_of(tmp_path, "again")
    reset(win, printers=[OCTO],
          history=[print_history.Entry(name="Gone", how=print_history.STARTED,
                                       gcode=tmp_path / "gone.gcode"),
                   print_history.Entry(name="Here", how=print_history.STARTED, gcode=str(there),
                                       grams=5.0, minutes=2.5)])
    dlg = hd.PrintHistoryDialog(win)
    try:
        dlg.table.selectRow(0)
        assert not dlg.b_queue.isEnabled(), "there is no G-code left to send"
        assert "no longer there" in dlg.b_queue.toolTip()
        dlg._add_queue()
        assert queued(win) == [] and kept(copies) == [], "nothing is copied for a print of "\
            "which nothing is left"
        dlg.table.selectRow(1)
        assert dlg.b_queue.isEnabled()
        dlg.printers = []                    # the printer went away under Preferences
        dlg._show_buttons()
        assert not dlg.b_queue.isEnabled() and "Preferences > Printers" in dlg.b_queue.toolTip()
        dlg._add_queue()
        assert queued(win) == [], "without a printer there is nothing to wait for"
    finally:
        empty(win)
        dlg.close()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))

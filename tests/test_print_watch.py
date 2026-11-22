"""A word on the desktop when a print is done, stops or waits (volura/print_watch.py).

What a transition says is Qt-free (`transition` and `message`), so those rules are checked
without a window. The watching around them is checked with a fake in front of every printer and
a fake tray, with its timer stopped: a test says when the printers are asked, so nothing comes
due while it is busy (see the `opened` helper of tests/test_printer_link.py).
"""
import os
import sys
import threading
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import pytest  # noqa: E402
from PySide6.QtCore import QCoreApplication  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from volura import printer_link as pl  # noqa: E402
from volura import print_watch as pw  # noqa: E402

OCTO = pl.Connection("Octo", "octoprint", "192.168.1.20", "k")
PRUSA = pl.Connection("Prusa", "prusalink", "prusa.thuis")


# ---------------------------------------------------------------- the rules, without a window

def state(word, file="box.gcode", progress=None, elapsed=None) -> pl.Status:
    """What a printer says of itself: the state it is in and, while it prints, the rest."""
    return pl.Status(word, file, progress, elapsed)


def test_the_first_answer_of_a_printer_only_sets_what_is_known():
    for word in pl.STATES:
        for part in (None, 0.4, 1.0):
            assert pw.transition(None, state(word, progress=part)) == "", \
                f"{word} is said as news, while it was already {word} before Volura started"


def test_a_state_that_only_stays_says_nothing():
    for word in pl.STATES:
        for part in (None, 0.4, 1.0):
            assert pw.transition(state(word, progress=part), state(word, progress=part)) == "", \
                f"a printer that stays {word} has something to say about it"
    # a printer that prints on, however much further it gets, says nothing
    assert pw.transition(state("printing", progress=0.1), state("printing", progress=0.9)) == ""


def test_a_print_that_came_to_its_end_is_said_as_done():
    for part in (1.0, pw.DONE_FRACTION, None):
        got = pw.transition(state("printing", progress=0.6), state("idle", progress=part))
        assert got == pw.DONE, f"idle at {part} is not said as done"


def test_a_print_that_stopped_before_its_end_is_said_as_stopped():
    for part in (0.0, 0.4, pw.DONE_FRACTION - 0.01):
        got = pw.transition(state("printing", progress=0.6), state("idle", progress=part))
        assert got == pw.STOPPED, f"idle at {part} is not said as stopped"
    assert pw.transition(state("printing", progress=0.4), state("error")) == pw.STOPPED


def test_a_print_that_waits_for_somebody_is_said_as_waiting():
    got = pw.transition(state("printing", progress=0.4), state("paused", progress=0.4))
    assert got == pw.WAITING, "a filament change that stops the print is not said"


def test_only_a_printer_that_was_printing_has_something_to_say():
    for word in pl.STATES:
        if word == "printing":
            continue
        for new in ("idle", "error", "paused"):
            assert pw.transition(state(word), state(new)) == "", \
                f"{word} to {new} is said, and nothing was printing"


def test_a_printer_that_answers_nothing_is_gone_after_three_asking():
    printing = state("printing", progress=0.4)
    for count in range(1, pw.OFFLINE_ASKS):
        assert pw.transition(printing, state("offline"), count) == "", \
            f"a printer is said gone after {count} of the {pw.OFFLINE_ASKS} asking"
    assert pw.transition(printing, state("offline"), pw.OFFLINE_ASKS) == pw.GONE
    assert pw.transition(printing, state("offline"), pw.OFFLINE_ASKS + 1) == "", \
        "a printer that stays gone is said again with every asking after it"


def test_a_printer_that_was_not_printing_can_be_gone_quietly():
    for word in ("idle", "paused", "error"):
        for count in range(1, pw.OFFLINE_ASKS + 2):
            assert pw.transition(state(word), state("offline"), count) == ""


def test_a_silence_or_two_and_answering_again_says_nothing():
    printing = state("printing", progress=0.4)
    for count in range(1, pw.OFFLINE_ASKS):
        assert pw.transition(printing, state("offline"), count) == ""
        assert pw.transition(printing, printing, 0) == "", \
            "a printer that answers again after a silence is said as something"


def test_a_print_that_ended_while_nothing_answered_is_still_said_as_done():
    """A printer that was printing, answered nothing for a while and now says idle with a full
    bar: the print is over, and that is exactly what the user was waiting to hear."""
    printing = state("printing", progress=0.9, elapsed=3000)
    assert pw.transition(printing, state("offline"), 2) == ""
    assert pw.transition(printing, state("idle", progress=1.0, elapsed=3900), 0) == pw.DONE


def test_an_answer_there_is_says_nothing():
    assert pw.transition(state("printing", progress=0.4), None) == ""
    assert pw.transition(None, None) == ""


# ----------------------------------------------------------------------- the words themselves

def test_the_words_name_the_printer_its_file_and_the_time_it_took():
    assert pw.message(pw.DONE, "Octo", state("idle", progress=1.0)) == "Octo: box.gcode is done"
    got = pw.message(pw.DONE, "Octo", state("idle", progress=1.0, elapsed=3900))
    assert got == "Octo: box.gcode is done in 1 h 05 min"


def test_the_words_work_for_a_printer_that_stopped_naming_its_file():
    """PrusaLink names a file only while it prints, so a print that is over has no file left to
    say - the words stand without it, and without the time it took."""
    assert pw.message(pw.DONE, "Prusa", pl.Status("idle")) == "Prusa is done"
    got = pw.message(pw.DONE, "Prusa", pl.Status("idle", elapsed=45))
    assert got == "Prusa is done in < 1 min"
    assert pw.message(pw.STOPPED, "Prusa", pl.Status("error")) == "Prusa stopped"
    assert pw.message(pw.WAITING, "Prusa", pl.Status("paused")) == "Prusa is waiting"


def test_the_words_of_a_stopped_print_say_what_went_wrong_when_that_is_known():
    got = pw.message(pw.STOPPED, "Octo", state("idle", progress=0.2))
    assert got == "Octo stopped: box.gcode"
    got = pw.message(pw.STOPPED, "Octo", state("idle", progress=0.2), "nothing answers at it")
    assert got == "Octo stopped: box.gcode: nothing answers at it"
    got = pw.message(pw.STOPPED, "Octo", pl.Status("error"), "nothing answers")
    assert got == "Octo stopped: nothing answers"


def test_the_words_of_a_waiting_and_of_a_gone_printer():
    assert pw.message(pw.WAITING, "Octo", state("paused")) == "Octo is waiting: box.gcode paused"
    assert pw.message(pw.GONE, "Octo") == "Octo no longer answers"
    assert pw.message(pw.GONE, "Octo", state("offline")) == "Octo no longer answers"


def test_everything_a_transition_can_say_has_words():
    assert all(pw.message(key, "Octo") for key in (pw.DONE, pw.STOPPED, pw.WAITING, pw.GONE))


def test_a_printer_that_forgot_its_file_while_idle_still_hears_which_print_it_was():
    printing = state("printing", file="bracket.gcode", progress=0.9, elapsed=3000)
    done = pw.said(pl.Status("idle"), printing)
    assert done.file == "bracket.gcode" and done.elapsed == 3000
    assert pw.message(pw.DONE, "Prusa", done) == "Prusa: bracket.gcode is done in 50 min"
    assert pw.said(state("paused"), None).file == "box.gcode"     # nothing to fill in from


# --------------------------------------------------------------- the watching around a window

@pytest.fixture(scope="module")
def win():
    """The window whose printers are watched.

    Every setting these tests read is set before and after, so none of them depends on what
    another test left in the settings of this run."""
    from volura.app import MainWindow
    QApplication.instance() or QApplication(sys.argv[:1])
    w = MainWindow()
    quiet(w)
    yield w
    quiet(w)
    w.doc.dirty = False
    w.close()


def quiet(w):
    """The settings as they stand before and after these tests: nothing to watch, and nothing
    switched on to say about it."""
    w.settings.printers = []
    w.settings.print_notifications = False
    w.settings.save()
    w.print_watch.stop()


class Says:
    """Stands in for printer_link.status: what every printer answers when it is asked.

    A printer that is not named says it is idle with nothing to say about a print; `say` says
    what one of them answers from now on, and "wrong" is a printer that answers nothing at all.
    Every asking is counted, so a test sees whether one printer was asked twice."""

    def __init__(self):
        self.answers = {}
        self.asks = []

    def say(self, name, word, file="box.gcode", progress=None, elapsed=None):
        """What this printer answers from now on; "wrong" answers nothing at all."""
        self.answers[name] = pl.Status(word, file, progress, elapsed)

    def asks_of(self, name) -> int:
        """How often this printer was asked."""
        return self.asks.count(name)

    def __call__(self, conn, timeout=5):
        self.asks.append(conn.name)
        answer = self.answers.get(conn.name) or pl.Status("idle")
        if answer.state == "wrong":
            raise pl.LinkError("Nothing answers at that address.")
        return answer


class Held(Says):
    """Stands in for asking a printer that takes its time: the asking stays on its way until the
    test lets it go, which is how a test sees a second asking next to the first one."""

    def __init__(self):
        super().__init__()
        self.on_the_way = threading.Event()
        self.let_go = threading.Event()

    def __call__(self, conn, timeout=5):
        self.asks.append(conn.name)
        self.on_the_way.set()
        assert self.let_go.wait(20), "the test never let this asking go"
        return self.answers.get(conn.name) or pl.Status("idle")


class Click:
    """The little signal of the fake tray: `connect` to hear it, `press` to press it."""

    def __init__(self):
        self.slots = []

    def connect(self, slot):
        self.slots.append(slot)

    def press(self, *args):
        for slot in self.slots:
            slot(*args)


class Tray:
    """Stands in for QSystemTrayIcon: it remembers what was put on the desktop and can be
    pressed, which is all the watching ever asks of a tray.

    `messages_ok` says whether this tray can show a message at all - a desktop with an icon but
    without anything to show messages has that False, and then the watching has other words to
    use."""

    def __init__(self, image=None, parent=None):
        self.image = image
        self.tip = ""
        self.shown = False
        self.messages = []
        self.messages_ok = True
        self.activated = Click()
        self.messageClicked = Click()

    def supportsMessages(self):
        return self.messages_ok

    def setToolTip(self, text):
        self.tip = text

    def show(self):
        self.shown = True

    def hide(self):
        self.shown = False

    def isVisible(self):
        return self.shown

    def showMessage(self, title, text, image=None, msecs=0):
        self.messages.append((title, text))


@pytest.fixture()
def watch(win, monkeypatch):
    """The watching of the window: nothing watched and nothing said yet, a fake in front of every
    printer, a tray of its own and its timer stopped, so the test says when printers are asked.

    `watch.says` is the fake of the printers and `watch.tray` the fake tray - the latter only from
    the moment the watching started, because the icon comes with it. A test that needs another
    fake puts it in front of the printers with monkeypatch itself.

    A Window > Printer Status left open is taken in and closed afterwards: an asking that is still
    on its way outlives the fake that answers it, and would then speak to the real network. The
    cameras of that window are answered too - it asks about them next to the printers - and a
    printer that has no camera there is nothing to watch."""
    says = Says()
    monkeypatch.setattr(pl, "status", says)
    monkeypatch.setattr(pl, "snapshot_url", lambda conn, **kw: None)     # nothing to watch
    monkeypatch.setattr(pl, "snapshot", lambda conn, url, **kw: b"")
    monkeypatch.setattr(pw, "QSystemTrayIcon", Tray)
    monkeypatch.setattr(pw, "has_tray", lambda: True)
    panel = getattr(win, "_printer_status", None)          # a status window of a test before
    if panel is not None:
        close(panel)
    w = win.print_watch
    w.timer.stop()
    w.stop()
    w.tray = None                 # an icon of this test alone, made when the watching starts
    win.statusBar().clearMessage()          # a word of a test before is nothing this test said
    w.says = says
    yield w
    panel = getattr(win, "_printer_status", None)
    if panel is not None:
        close(panel)
    asked(w)                                    # nothing of this test may outlive its fake


def switch_on(win, watch, *conns, notify=True):
    """The printers under Preferences and the messages switched on: the watching follows, and its
    timer stands still, so only the test asks the printers."""
    win.settings.printers = list(conns) or [OCTO]
    win.settings.print_notifications = notify
    win.settings.save()
    watch.follow()
    watch.timer.stop()            # `follow` started it: a test drives the asking by hand
    return watch


def asked(watch, ms=20000):
    """Every asking the watching has on its way, come to an end, and the answers handed over -
    also the asking that were given up on, which still have to be waited for, because their
    answer comes anyway and may not arrive after the test has put the fake away."""
    jobs = list(watch.asking.values()) + list(watch.given_up)
    for job in [job for job in jobs if not job.done]:
        assert job.wait(ms), f"asking the printer never came to an end: {job.error}"
    # the answer of a job is handed over by the event loop, and on a busy machine that can take
    # more than one round of it: wait until every job says it is done and its answer is taken in
    settle(lambda: all(job.done for job in jobs), ms)


def settle(ready, ms=20000):
    """Spin the event loop until `ready()` holds, then once more for what it set off."""
    deadline = time.monotonic() + ms / 1000.0
    while not ready() and time.monotonic() < deadline:
        QCoreApplication.processEvents()
        time.sleep(0.005)
    assert ready(), "the event loop never handed the answer over"
    QCoreApplication.processEvents()


def answered(panel, ms=20000):
    """Every asking Window > Printer Status has on its way, come to an end."""
    for row in panel.rows:
        if row.job is not None:
            assert row.job.wait(ms), f"the window never got an answer: {row.job.error}"
    # a row lets go of its job when it took the answer in, which the event loop hands over
    settle(lambda: all(row.job is None for row in panel.rows), ms)


def close(panel):
    """Window > Printer Status, closed after every asking it has on its way was taken in, the
    asking about its cameras included."""
    answered(panel)
    for job in list(panel._cam_jobs.values()):
        assert job.wait(20000), f"asking the camera never came to an end: {job.error}"
    for row in panel.rows:
        if row.cam_job is not None:
            assert row.cam_job.wait(20000), "asking the camera never came to an end"
    QCoreApplication.processEvents()
    panel.close()
    return panel


def look(watch, ms=20000):
    """Ask every printer once, and take in every answer that came back."""
    watch.poll()
    asked(watch, ms)


def said(watch):
    """What the watching said to the user since this test began, oldest first: the messages of
    its tray, or, where there is no tray or none that shows messages, what stands in the bar
    under the window."""
    if watch.tray is not None and watch.tray.supportsMessages():
        return [text for _title, text in watch.tray.messages]
    word = watch.win.statusBar().currentMessage()
    return [word] if word else []


# -------------------------------------------------------------------------------- switched off

def test_switched_off_nothing_is_asked_nothing_is_said_and_no_icon(win, watch):
    switch_on(win, watch, notify=False)
    assert not watch.timer.isActive(), "the messages are off, yet something asks the printers"
    assert watch.tray is None, "the messages are off, yet there is an icon on the desktop"
    watch.says.say("Octo", "printing", progress=0.4)
    look(watch)
    assert watch.says.asks == [] and said(watch) == []


def test_the_switch_toggles_the_icon_and_the_asking(win, watch):
    win.settings.printers = [OCTO]
    win.settings.print_notifications = True
    win.settings.save()
    watch.follow()
    assert watch.timer.isActive(), "the messages are on, yet nothing asks the printers"
    assert watch.timer.interval() == pw.POLL_SECONDS * 1000
    assert watch.tray is not None and watch.tray.shown, "the messages are on and no icon came"
    assert watch.tray.tip == pw.TIP
    watch.timer.stop()
    win.settings.print_notifications = False
    watch.follow()
    assert not watch.timer.isActive(), "switched off, and still asking"
    assert not watch.tray.shown, "switched off, and the icon stays on the desktop"
    look(watch)
    assert watch.says.asks == [], "switched off, yet a printer was asked"


def test_an_icon_comes_with_the_first_printer_under_preferences(win, watch):
    """Without a printer there is nothing to watch and no word to say, so there is no icon
    either; a printer that is added afterwards brings it along."""
    win.settings.printers = []
    win.settings.print_notifications = True
    win.settings.save()
    watch.follow()
    watch.timer.stop()
    assert watch.tray is None, "an icon that watches nothing has nothing to say"
    win.settings.printers = [OCTO]
    watch.poll()
    assert watch.tray is not None and watch.tray.shown, "a printer arrived and no icon came"


def test_a_printer_off_the_list_is_nothing_to_know(win, watch):
    switch_on(win, watch, OCTO, PRUSA)
    watch.says.say("Octo", "printing", progress=0.4)
    watch.says.say("Prusa", "printing", progress=0.4)
    look(watch)
    assert set(watch.known) == {"Octo", "Prusa"}
    win.settings.printers = [OCTO]
    watch.follow()
    assert set(watch.known) == {"Octo"}, "a printer that left Preferences is still watched"


# ---------------------------------------------------------------------------------- what is said

def test_the_first_answer_only_says_what_is_known_and_a_change_says_a_word(win, watch):
    switch_on(win, watch)
    watch.says.say("Octo", "printing", progress=0.4)
    look(watch)
    assert said(watch) == [], "a print that was already printing is said as news"
    watch.says.say("Octo", "idle", progress=1.0, elapsed=3900)
    look(watch)
    assert said(watch) == ["Octo: box.gcode is done in 1 h 05 min"]
    assert watch.tray.messages == [("Printer Status", "Octo: box.gcode is done in 1 h 05 min")]


def test_a_state_that_stays_is_not_said_again(win, watch):
    switch_on(win, watch)
    watch.says.say("Octo", "printing", progress=0.3)
    look(watch)                                 # the first answer only sets what is known
    watch.says.say("Octo", "printing", progress=0.6)
    look(watch)
    watch.says.say("Octo", "paused", progress=0.6)
    look(watch)
    assert said(watch) == ["Octo is waiting: box.gcode paused"]
    look(watch)                                 # the same pause, asked again
    look(watch)
    assert len(said(watch)) == 1, "a state that only stays is said again"


def test_a_print_that_stopped_halfway_is_said_as_stopped(win, watch):
    switch_on(win, watch)
    watch.says.say("Octo", "printing", progress=0.3)
    look(watch)
    watch.says.say("Octo", "idle", progress=0.3)
    look(watch)
    assert said(watch) == ["Octo stopped: box.gcode"]
    assert watch.tray.messages[-1][1] == "Octo stopped: box.gcode"


def test_a_print_that_stopped_with_an_error_is_said_as_stopped(win, watch):
    switch_on(win, watch)
    watch.says.say("Octo", "printing", progress=0.3)
    look(watch)
    watch.says.say("Octo", "error")
    look(watch)
    assert said(watch) == ["Octo stopped: box.gcode"]


def test_the_words_of_an_asking_that_went_wrong_are_said_once(win, watch):
    """A printer that answers nothing for a moment and then says the print stopped: what went
    wrong belongs to that word, and does not come back with the next one."""
    switch_on(win, watch)
    watch.says.say("Octo", "printing", progress=0.3)
    look(watch)
    watch.says.say("Octo", "wrong")
    look(watch)                                 # one silence: nothing to say about it yet
    watch.says.say("Octo", "idle", progress=0.3)
    look(watch)
    assert said(watch) == ["Octo stopped: box.gcode: Nothing answers at that address."]
    watch.says.say("Octo", "printing", progress=0.4)
    look(watch)
    watch.says.say("Octo", "error")
    look(watch)
    assert said(watch)[-1] == "Octo stopped: box.gcode", "the words of a silence came back"


def test_a_filament_change_is_said_as_waiting(win, watch):
    switch_on(win, watch)
    watch.says.say("Octo", "printing", progress=0.5)
    look(watch)
    watch.says.say("Octo", "paused", progress=0.5)
    look(watch)
    assert said(watch) == ["Octo is waiting: box.gcode paused"]


def test_three_asking_without_an_answer_say_that_the_printer_is_gone(win, watch):
    switch_on(win, watch)
    watch.says.say("Octo", "printing", progress=0.4)
    look(watch)
    watch.says.say("Octo", "wrong")
    for _ in range(pw.OFFLINE_ASKS - 1):
        look(watch)
        assert said(watch) == [], "a printer is said gone too soon"
    look(watch)
    assert said(watch) == ["Octo no longer answers"]
    look(watch)
    look(watch)
    assert len(said(watch)) == 1, "a printer that stays gone is said again"


def test_a_silence_or_two_and_answering_again_says_nothing_at_all(win, watch):
    switch_on(win, watch)
    watch.says.say("Octo", "printing", progress=0.4)
    look(watch)
    for _ in range(pw.OFFLINE_ASKS - 1):
        watch.says.say("Octo", "wrong")
        look(watch)
        watch.says.say("Octo", "printing", progress=0.5)
        look(watch)
        assert said(watch) == [], \
            "a network that stands still for a moment is no printer that is gone"


def test_a_printer_that_is_idle_and_stops_answering_says_nothing(win, watch):
    switch_on(win, watch)
    watch.says.say("Octo", "idle")
    look(watch)
    watch.says.say("Octo", "wrong")
    for _ in range(pw.OFFLINE_ASKS + 2):
        look(watch)
    assert said(watch) == []


def test_switching_the_messages_on_starts_from_nothing_known(win, watch):
    switch_on(win, watch)
    watch.says.say("Octo", "printing", progress=0.9)
    look(watch)
    win.settings.print_notifications = False
    watch.follow()
    win.settings.print_notifications = True
    watch.follow()
    watch.timer.stop()
    watch.says.say("Octo", "idle", progress=1.0)
    look(watch)
    assert said(watch) == [], "a print that was already over is said after switching on"


def test_an_answer_that_arrives_after_the_messages_went_off_says_nothing(win, watch, monkeypatch):
    """Switching these messages off does not stop an asking that is already on its way - a
    printer is asked in a way that does not end half way - so its answer still comes in."""
    held = Held()
    monkeypatch.setattr(pl, "status", held)
    switch_on(win, watch, OCTO)
    held.say("Octo", "printing", progress=0.9)
    watch.poll()
    assert held.on_the_way.wait(10), "the watching never asked the printer"
    win.settings.print_notifications = False
    watch.follow()                              # switched off while the answer was on its way
    held.let_go.set()
    asked(watch)                                # the answer comes anyway, and is held for
    assert said(watch) == [], "an answer of after the switch went off was still said"
    win.settings.print_notifications = True
    watch.follow()                              # back on: the asking that was held is let go
    assert watch.given_up == [], "a held asking was held for longer than it answered"


def test_a_printer_that_left_preferences_while_answering_says_nothing(win, watch, monkeypatch):
    held = Held()
    monkeypatch.setattr(pl, "status", held)
    switch_on(win, watch, OCTO)
    held.say("Octo", "printing", progress=1.0)
    watch.poll()
    assert held.on_the_way.wait(10), "the watching never asked the printer"
    win.settings.printers = []                  # the printer is gone from Preferences
    watch.follow()
    held.let_go.set()
    asked(watch)
    assert said(watch) == [], "a word was said about a printer that is no longer there"


def test_every_printer_under_preferences_gets_its_own_word(win, watch):
    switch_on(win, watch, OCTO, PRUSA)
    watch.says.say("Octo", "printing", progress=0.4)
    watch.says.say("Prusa", "printing", progress=0.9, file="lid.gcode")
    look(watch)
    assert watch.says.asks_of("Octo") == watch.says.asks_of("Prusa") == 1
    watch.says.say("Prusa", "idle", progress=1.0, file="")      # it forgot which print it was
    look(watch)
    assert said(watch) == ["Prusa: lid.gcode is done"], "one printer ate the others"
    watch.says.say("Octo", "error")
    look(watch)
    assert said(watch)[-1] == "Octo stopped: box.gcode"


# ------------------------------------------------------------------------ the asking itself

def test_one_printer_is_not_asked_twice_at_once(win, watch, monkeypatch):
    held = Held()
    monkeypatch.setattr(pl, "status", held)
    switch_on(win, watch, OCTO)
    watch.poll()
    assert held.on_the_way.wait(10), "the watching never asked the printer"
    watch.poll()                                # the first asking is still on its way
    watch.poll()
    assert held.asks_of("Octo") == 1, "asking went out next to the asking still running"
    held.let_go.set()
    asked(watch)
    watch.poll()                                # answered: the next asking may go
    asked(watch)                                # the fake counts an asking when its thread runs
    assert held.asks_of("Octo") == 2


def test_the_watching_asks_every_printer_under_preferences(win, watch):
    switch_on(win, watch, OCTO, PRUSA)
    look(watch)
    # the askings run side by side on threads, so the order they reach the fake in is free
    assert sorted(watch.says.asks) == ["Octo", "Prusa"]
    look(watch)
    twice = "a printer was asked twice"
    assert sorted(watch.says.asks) == ["Octo", "Octo", "Prusa", "Prusa"], twice


def test_the_status_window_on_the_screen_shares_its_answers(win, watch):
    """Window > Printer Status asks every printer every five seconds anyway; while it is on the
    screen the watching takes what its rows show instead of asking a second time."""
    switch_on(win, watch)
    panel = win.printer_status()
    panel.timer.stop()
    try:
        watch.says.say("Octo", "printing", progress=0.4)
        answered(panel)                         # what showing that window already asked
        before = watch.says.asks_of("Octo")
        watch.poll()
        asked(watch)
        assert watch.says.asks_of("Octo") == before, "the watching asked what the window asks"
        assert said(watch) == [], "the first answer of the window is said as news"
        watch.says.say("Octo", "idle", progress=1.0, elapsed=3900)
        panel.poll()
        answered(panel)                         # the window sees the print come to its end
        watch.poll()
        asked(watch)
        assert said(watch) == ["Octo: box.gcode is done in 1 h 05 min"], \
            "the watching heard nothing of what the window showed"
        assert watch.says.asks_of("Octo") == before + 1, "the watching asked next to the window"
    finally:
        close(panel)


def test_a_status_window_off_the_screen_is_not_listened_to(win, watch):
    switch_on(win, watch)
    panel = win.printer_status()
    panel.timer.stop()
    close(panel)                          # off the screen: it asks nothing and knows nothing
    before = watch.says.asks_of("Octo")
    watch.poll()
    asked(watch)
    assert watch.says.asks_of("Octo") == before + 1, \
        "a window that is gone asks nothing, so the watching may ask itself"


# -------------------------------------------------------------------------- without a tray

def test_without_a_tray_the_words_go_to_the_status_bar(win, watch, monkeypatch):
    monkeypatch.setattr(pw, "has_tray", lambda: False)
    switch_on(win, watch)
    assert watch.tray is None, "this desktop has no tray, yet an icon was made"
    watch.says.say("Octo", "printing", progress=0.9)
    look(watch)
    watch.says.say("Octo", "idle", progress=1.0)
    look(watch)
    assert said(watch) == ["Octo: box.gcode is done"]
    assert win.statusBar().currentMessage() == "Octo: box.gcode is done"


def test_a_tray_that_shows_no_messages_puts_the_words_in_the_bar(win, watch):
    """A desktop can have an icon place and nothing that shows a message on it; the word then
    goes to the bar, while the icon stays: clicking it still leads to Window > Printer Status."""
    switch_on(win, watch)
    watch.tray.messages_ok = False          # this tray swallows every message
    watch.says.say("Octo", "printing", progress=0.9)
    look(watch)
    watch.says.say("Octo", "idle", progress=1.0)
    look(watch)
    assert watch.tray.messages == [], "a word was given to a tray that shows no messages"
    assert watch.tray.shown, "the icon went away with the messages it cannot show"
    assert said(watch) == ["Octo: box.gcode is done"]


# --------------------------------------------------------------------------------- clicking

def test_clicking_the_message_or_the_icon_opens_printer_status(win, watch):
    switch_on(win, watch)
    watch.tray.messageClicked.press()
    panel = win._printer_status
    assert panel is not None and panel.isVisible(), "clicking a message opened nothing"
    panel.timer.stop()
    close(panel)
    assert not panel.isVisible()
    watch.tray.activated.press(1)         # a click on the icon; what kind of click does not matter
    assert win._printer_status is panel and panel.isVisible(), "clicking the icon opened nothing"


# ------------------------------------------------------------------------ the preferences page

def test_the_printers_page_holds_the_switch(win, watch):
    from volura.app import PreferencesDialog
    switch_on(win, watch)
    dlg = PreferencesDialog(win, "Printers")
    assert dlg.p_notify.text() == "Tell me when a print finishes, stops or waits"
    assert dlg.p_notify.isChecked(), "the page does not show the switch as it stands"
    assert "desktop" in dlg.p_notify.toolTip()
    dlg.p_notify.setChecked(False)
    dlg.apply()
    assert win.settings.print_notifications is False
    assert not watch.timer.isActive() and not watch.tray.shown, "off on the page, on in Volura"
    again = PreferencesDialog(win, "Printers")
    assert not again.p_notify.isChecked(), "the word of the page does not come back"
    again.p_notify.setChecked(True)
    again.apply()
    assert type(win.settings)().print_notifications is True, "the switch is not kept"
    assert watch.timer.isActive() and watch.tray.shown, "on on the page, and nothing asks"
    watch.timer.stop()


def test_the_switch_is_on_once_a_printer_is_set_up(win):
    """Without a word of its own, a printer that is set up also gets these messages; without a
    printer there is nothing to tell, so the switch stands off."""
    try:
        for printers, want in (([OCTO], True), ([], False)):
            win.settings.printers = list(printers)
            win.settings.save()                 # the printers have to stand in the file for it
            win.settings.q.remove("print_notifications")
            win.settings.q.sync()
            assert type(win.settings)().print_notifications is want
    finally:
        quiet(win)

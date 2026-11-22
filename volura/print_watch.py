"""A word on the desktop when a print is done, stops or waits for you.

**Window > Preferences > Printers** has *Tell me when a print finishes, stops or waits*. With
that on - and on by itself as soon as there is a printer - every printer under Preferences is
asked every `POLL_SECONDS` while Volura runs, also when Window > Printer Status is closed. The
asking is the one of `volura/printer_status.py`, so one printer is never asked twice at once,
and while that window is on the screen its answers are taken instead of asking a second time.

Only what *changes* is said: a printer that was printing and is done, one that stopped, one that
stands waiting for a filament change, and one that answers nothing three asking in a row while
it printed. A state that merely stays says nothing, and the first answer after Volura starts or
after the messages are switched on only fills in what is known - a print that was finished
before they were switched on is no news. `transition` holds those rules, with no Qt and no
printer anywhere in it, and `message` says what they mean in words.

Where a desktop has a tray the words go there, and an icon with the Volura picture stands there
as long as there is a printer to watch; clicking that icon or a message brings Window > Printer
Status. Where it has none, or none that will show a message, the same words go to the status bar
under the model.
"""
from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtCore import QObject, QTimer
from PySide6.QtWidgets import QSystemTrayIcon

from . import printer_link, printer_status
from .i18n import tr
from .icons import icon

#: how often every printer is asked while Volura watches in the background (seconds)
POLL_SECONDS = 30
#: how often a printer has to answer nothing in a row before it is said as gone
OFFLINE_ASKS = 3
#: how far a print has to be before a printer that turned idle is said to have finished
DONE_FRACTION = 0.98
#: how long a message stands in the status bar where there is no tray (milliseconds)
MESSAGE_MS = 10000
#: the four things a transition can have to say, as `message` knows them
DONE, STOPPED, WAITING, GONE = "done", "stopped", "waiting", "gone"
#: what the tray icon and every message are said to lead to
TIP = tr("What every printer on the network is doing right now")


def has_tray() -> bool:
    """Does this desktop have a tray to put a message in?"""
    return QSystemTrayIcon.isSystemTrayAvailable()


def _finished(new) -> bool:
    """Is this idle answer the end of a print that really came that far?

    A printer that stopped halfway still has that part of the print on its completion, so the
    number says it; one that says nothing about a number is taken as finished, because most of
    them forget everything about a print once it is off the bed."""
    return new.progress is None or new.progress >= DONE_FRACTION


def transition(old, new, offline_count=0) -> str:
    """What this answer has to say, or "" when it has nothing to say.

    `old` is what that printer last said it was doing (None as long as nothing is known of it),
    `new` the answer that came in and `offline_count` how often it answered nothing in a row,
    this one included. An answer that says nothing leaves what is known alone, so a printer that
    went away is still compared with the state it had before it went away.

    A state that only stays says nothing, and a printer that answers nothing says nothing before
    `OFFLINE_ASKS` asking in a row - a network that stands still for a moment is no printer that
    is gone - with that word said at exactly that third asking, so it comes once and not again
    with every asking after it. Only a printer that *was printing* has something to say: idle,
    error and paused of themselves are no news.
    """
    if new is None:
        return ""                                  # nothing came in, so nothing is known
    if new.state == "offline":
        if old is not None and old.state == "printing" and offline_count == OFFLINE_ASKS:
            return GONE
        return ""
    if old is None or old.state != "printing":
        return ""                                  # nothing known yet, or nothing that printed
    if new.state == "idle":
        return DONE if _finished(new) else STOPPED   # finished, or stopped before its end
    if new.state == "error":
        return STOPPED
    if new.state == "paused":
        return WAITING                               # a filament change looks like this
    return ""                                        # still printing: a state that stays


def said(new, old):
    """What to tell of this printer: what it answers now, filled in with what it said while it
    printed.

    A printer that is done has often stopped naming its file and counting its time, and that
    file and that time are just what the user wants to hear."""
    if old is None:
        return new
    return printer_link.Status(
        new.state, new.file or old.file,
        new.progress if new.progress is not None else old.progress,
        new.elapsed if new.elapsed is not None else old.elapsed,
        new.left if new.left is not None else old.left)


def message(key, name, status=None, error="") -> str:
    """The words that go with a transition: the printer, which file it had, how long it took and
    what it says went wrong - every one of those only when it is known."""
    file = str(getattr(status, "file", "") or "").strip()
    went = printer_link.format_duration(getattr(status, "elapsed", None))
    if key == DONE:
        if file:
            text = (tr("{printer}: {file} is done in {time}") if went
                    else tr("{printer}: {file} is done"))
        else:
            text = (tr("{printer} is done in {time}") if went else tr("{printer} is done"))
        return text.format(printer=name, file=file, time=went)
    if key == STOPPED:
        if error:
            text = (tr("{printer} stopped: {file}: {error}") if file
                    else tr("{printer} stopped: {error}"))
            return text.format(printer=name, file=file, error=error)
        return (tr("{printer} stopped: {file}") if file else tr("{printer} stopped")).format(
            printer=name, file=file)
    if key == WAITING:
        return (tr("{printer} is waiting: {file} paused") if file
                else tr("{printer} is waiting")).format(printer=name, file=file)
    return tr("{printer} no longer answers").format(printer=name)


@dataclass
class Watch:
    """What Volura knows of one printer: the last thing it said it was doing, how often it has
    answered nothing in a row since, and what the last asking said went wrong.

    An answer that says nothing leaves `status` alone, so a printer that is back after being
    gone is still compared with the state it had before it went away."""

    status: printer_link.Status | None = None
    offline: int = 0
    wrong: str = ""


class PrintWatcher(QObject):
    """Watch the printers of the preferences while Volura runs, and say on the desktop what
    changes: a print that is done, one that stopped, one that waits and a printer that no longer
    answers.

    Unseen and without a window, like the autosave timer: with these messages on, every printer
    is asked again every `POLL_SECONDS` in the background - the same asking Window > Printer
    Status uses, so a printer that is still answering, or that that window is asking already, is
    not asked a second time. What `transition` finds new goes into the tray, or into the status
    bar of the window where the desktop has no tray.
    """

    def __init__(self, win):
        super().__init__(win)
        self.win = win
        self.icon = icon("app")
        self.known = {}                 # the name of a printer to what is known of it
        self.asking = {}                # the printers that are answering, so none is asked twice
        self.given_up = []              # asking that were given up on, until they are done
        self.started = False            # watching since these messages went on, from nothing known
        self.tray = None                # the icon, on the desktop as long as the messages are on
        self.timer = QTimer(self)
        self.timer.setInterval(POLL_SECONDS * 1000)
        self.timer.timeout.connect(self.poll)
        self.follow()                   # the messages can be on from the first moment

    # ------------------------------------------------------------------ on and off
    def on(self) -> bool:
        """Does the user want a word on the desktop when a print ends?"""
        return bool(self.win.settings.print_notifications)

    def watching(self) -> bool:
        """Is there something to watch? Without a printer under Preferences there is no word to
        say, and no icon to say it with."""
        return self.on() and bool(self.win.settings.printers)

    def follow(self):
        """Preferences > Printers: the switch of these messages, the icon that goes with them and
        the printers that may have changed along with them."""
        if not self.on():
            self.stop()
            return
        if not self.started:
            # the first moment since these messages went on: nothing is known of any printer, so
            # that first answer only says what is and never a print that was already over
            self.started = True
            self.known.clear()
            self.asking.clear()
        self.timer.start()
        self._look_icon()
        self._let_go()
        names = {conn.name for conn in self.win.settings.printers}
        for name in [n for n in self.known if n not in names]:
            self.known.pop(name, None)          # a printer off the list is nothing to know

    def _look_icon(self):
        """The icon comes while there is something to watch and goes when there is not; a desktop
        without a tray never gets one at all."""
        want = self.watching() and has_tray()
        if want and self.tray is None:
            self.tray = QSystemTrayIcon(self.icon, self)
            self.tray.setToolTip(TIP)
            self.tray.activated.connect(self._clicked)
            self.tray.messageClicked.connect(self._clicked)
        if self.tray is None:
            return
        if want:
            self.tray.show()
        else:
            self.tray.hide()

    def stop(self):
        """Nothing is asked and nothing is said any more: the icon goes away, and what was known
        with it, so switching the messages back on never brings an old print back. An asking that
        is still on its way is not stopped - a printer is asked in a way that does not end half
        way - but it is held for a while and said nothing about."""
        self.timer.stop()
        self.started = False
        self.given_up.extend(self.asking.values())
        self.asking.clear()
        self.known.clear()
        if self.tray is not None:
            self.tray.hide()

    def _let_go(self):
        """The asking that were given up on: out of sight as soon as they are done, for as long
        as they answer, a printer has to be held for."""
        self.given_up = [job for job in self.given_up if not job.done]

    # ------------------------------------------------------------------------ the asking
    def poll(self):
        """Ask every printer what it is doing and take in what that says; a printer that is
        still answering waits its turn, and one that the status window on the screen is asking
        needs no second asking. Switched off nothing is asked, and the icon that goes with these
        messages comes along with the first printer that showed up since the last time."""
        if not self.on():
            self._let_go()
            return
        self._look_icon()
        shown = self._shown()
        for conn in list(self.win.settings.printers):
            row = shown.get(conn.name)
            if row is not None:
                self.observe(conn.name, row.status)     # this one is asked by the window
            else:
                self.ask(conn)

    def _shown(self) -> dict:
        """The printers Window > Printer Status is asking, by their name; a window off the
        screen asks nothing, so nothing of it is to be had either."""
        panel = getattr(self.win, "_printer_status", None)
        if panel is None or not panel.isVisible():
            return {}
        return {row.conn.name: row for row in panel.rows}

    def ask(self, conn):
        """Ask one printer what it is doing, in the background; one that is still answering is
        not asked again."""
        job = self.asking.get(conn.name)
        if job is not None and not job.done:
            return
        name = conn.name

        def answered(status):
            self.observe(name, status)

        def failed(exc):
            one = self.known.setdefault(name, Watch())
            one.wrong = str(exc)                # what went wrong belongs to what is said next
            self.observe(name, printer_link.Status("offline"))

        self.asking[name] = printer_status.ask_status(conn, answered, failed)

    # ------------------------------------------------------------------------ what is said
    def observe(self, name, status):
        """What one printer now says, and the word on the desktop when that says something new.
        An answer that comes when nothing is watched, or about a printer that is no longer under
        Preferences, is nothing to know and so nothing to say."""
        if status is None or not self.started:
            return                              # nothing answered, or nothing watched any more
        if name not in {c.name for c in self.win.settings.printers}:
            return                              # off the list, and so nothing to know or to say
        one = self.known.setdefault(name, Watch())
        old, error = one.status, one.wrong      # what an asking said wrong, said once
        if status.state == "offline":
            one.offline += 1
        else:
            one.offline, one.status, one.wrong = 0, status, ""
        key = transition(old, status, one.offline)
        if key:
            self.say(key, name, said(status, old), error)

    def say(self, key, name, status, error=""):
        """The words of this transition to the desktop: into the tray where there is one that can
        show a message, and into the status bar of the window everywhere else. An icon that
        swallows messages is no place for a word either, and the icon itself stays on the desktop
        in that case: clicking it still leads to Window > Printer Status."""
        text = message(key, name, status, error)
        if self.tray is not None and self.tray.supportsMessages():
            self.tray.showMessage(tr("Printer Status"), text, self.icon, MESSAGE_MS)
        else:
            self.win.statusBar().showMessage(text, MESSAGE_MS)

    def _clicked(self, _reason=None):
        """The icon or a message was clicked: Window > Printer Status, with everything the
        printers have to say about themselves."""
        self.win.printer_status()

"""The print history: volura/print_history.py, what the preview writes and Window > Print History…

An entry remembers one print: when it was made, what it was called, where its G-code stands and
what became of it. The preview records the strongest thing that happened to it - started beats
marked, marked beats sent, sent beats saved - and nothing at all when it was only looked at. Of
a print that went to a printer a copy of its G-code is kept, so Print Again… needs no new slice,
and that copy goes when the print leaves the list; a file the user saved is never copied and
never taken away.

Run with:  python -m pytest tests/test_print_history.py
"""
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest  # noqa: E402
from PySide6.QtCore import QUrl  # noqa: E402
from PySide6.QtGui import QPalette  # noqa: E402
from PySide6.QtWidgets import QApplication, QDialog  # noqa: E402

from volura import gcode_preview as gp, printer_link, print_history  # noqa: E402
from volura.settings import qsettings  # noqa: E402

_app = QApplication.instance() or QApplication(sys.argv[:1])

#: a print that changes filament and needs 42.5 minutes of 60 grams of red PLA
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
;TIME_ELAPSED:6.0
"""

RED = [("Red PLA", "#ff0000", 1000.0, 60.0)]      # what the slice says it printed with
PRINTER = printer_link.Connection("Octo", "octoprint", "192.168.1.20", "k")
NO_PRINTS = "0 prints, 0.0 kg filament, 0.0 h printing"


@pytest.fixture(scope="module")
def win():
    """The window the previews and the dialog sit in; its history is empty again when done.

    Every setting these tests read is set here, so none of them depends on what another test
    left in the settings of this computer."""
    from volura.app import MainWindow
    w = MainWindow()
    empty(w)
    yield w
    empty(w)
    w.doc.dirty = False
    w.close()


def empty(w):
    """The settings of the window as they stand before and after these tests."""
    w.settings.print_history = []
    w.settings.printers = []
    w.settings.spools = []
    w.settings.spool_for_slot = []
    w.settings.save()


@pytest.fixture()
def copies(tmp_path, monkeypatch):
    """The folder of the G-codes the history keeps itself, in the folder of this test."""
    folder = tmp_path / "history" / "gcode"
    folder.mkdir(parents=True)
    monkeypatch.setattr(print_history, "gcode_dir", lambda create=True: str(folder))
    return folder


def reset(win, *records, printers=()):
    """The history of the settings as the next test starts it."""
    win.settings.print_history = [print_history.to_dict(record) for record in records]
    win.settings.printers = list(printers)
    win.settings.save()
    return win.settings.print_history


def recorded(win):
    """What the settings remember of the prints, as entries, newest first."""
    return print_history.records(win.settings.print_history)


def entry(gcode, **kwargs):
    """A print whose G-code is the file at `gcode`."""
    return print_history.Entry(gcode=str(gcode), **kwargs)


def slice_of(tmp_path, name="box"):
    """A G-code that is really there, for a print that can go to a printer again."""
    path = tmp_path / f"{name}.gcode"
    path.write_text(GCODE, encoding="utf-8")
    return path


def in_folder(folder, path):
    """Is `path` one of the G-codes of the folder the history keeps itself?"""
    return os.path.basename(str(path)) in kept(folder)


def kept(folder):
    """The names of the G-codes that stand in the folder of the history."""
    return sorted(os.listdir(str(folder))) if os.path.isdir(str(folder)) else []


def preview(win, tmp_path, filaments=(), grams=0.0, minutes=0.0):
    """The print preview of a small slice, with what the slice worked out for it."""
    from volura import print_preview as pp
    path = tmp_path / "box.gcode"
    path.write_text(GCODE, encoding="utf-8")
    return pp.PrintPreviewDialog(win, str(path), gp.parse(GCODE.splitlines()),
                                 (0.0, 0.0, 100.0, 100.0), "", None, filaments,
                                 grams=grams, minutes=minutes)


def press(dlg):
    """Stands in for a user who presses Send and starts printing right away."""
    dlg.start.setChecked(True)
    dlg.accept()
    return QDialog.Accepted


def hand_over(dlg):
    """Stands in for a user who presses Send but only puts the file on the printer."""
    dlg.start.setChecked(False)
    dlg.accept()
    return QDialog.Accepted


def neither(dlg):
    """Stands in for a user who cancels the question of which printer takes the file."""
    return QDialog.Rejected


class Answer:
    """Stands in for QMessageBox: answers the question about a print without a dialog."""

    Yes, No = 1, 0

    def __init__(self, answer=1):
        self.answer = answer
        self.asked, self.said = [], []

    def question(self, *args, **kwargs):
        self.asked.append(args)
        return self.answer

    def critical(self, *args, **kwargs):
        self.said.append(args)


class Uploads:
    """Stands in for printer_link.upload: which G-code went to which printer, or nothing."""

    def __init__(self, fails=False):
        self.calls = []
        self.fails = fails

    def __call__(self, conn, path, name, start=False, on_progress=None):
        if self.fails:
            raise printer_link.LinkError("nothing answers at the printer")
        self.calls.append((conn.name, os.path.basename(path), name, bool(start)))


class Opened:
    """Stands in for QDesktopServices: which folder was opened in the file manager."""

    def __init__(self):
        self.urls = []

    def openUrl(self, url):
        self.urls.append(url.toString())
        return True


# ------------------------------------------------------------------ one print on its own

def test_a_print_comes_back_out_of_the_settings_as_it_went_in():
    made = print_history.Entry(name="Bracket", gcode="/prints/bracket.gcode", printer="Octo",
                               grams=62.5, minutes=42.5, filaments=[("PLA", "#ff0000")],
                               how="started", when="2026-03-04T15:30:00")
    assert print_history.from_dict(print_history.to_dict(made)) == made
    assert print_history.to_dict(made)["filaments"] == [["PLA", "#ff0000"]], \
        "the settings keep lists, also for the colours of a filament"
    assert print_history.to_dict(made)["grams"] == 62.5
    assert made.when == "2026-03-04T15:30:00"
    # a print of nothing at all is still a print: it says when it was made and that nothing
    # of it was printed
    blank = print_history.Entry()
    assert blank.id and blank.when and blank.name == "Untitled"
    assert blank.gcode == "" and blank.printer == "" and blank.filaments == []
    assert (blank.grams, blank.minutes) == (0.0, 0.0) and blank.how == "saved"
    # every kind of print the history knows keeps its own name
    for how in print_history.HOWS:
        assert print_history.from_dict(print_history.to_dict(entry("", how=how))).how == how


def test_a_print_with_junk_in_it_falls_back_to_what_is_known():
    junk = print_history.from_dict({"id": None, "when": "yesterday", "name": 5, "gcode": 7,
                                   "printer": None, "grams": "a lot", "minutes": -5,
                                   "filaments": ["PLA", 7, None, ["", "#ff0000"],
                                                 ("PETG", "#00ff00", "too much")],
                                   "how": "printed", "where_it_was_bought": "the fair"})
    assert junk.id, "a print without an id is still this print and not another one"
    assert junk.when != "yesterday", "a moment that is no moment is the moment of writing it"
    assert junk.name == "5" and junk.gcode == "7" and junk.printer == ""
    assert junk.grams == 0.0 and junk.minutes == 0.0 and junk.how == "saved"
    assert junk.filaments == [["PLA", ""], ["PETG", "#00ff00"]]
    assert "where_it_was_bought" not in print_history.to_dict(junk)
    assert print_history.from_dict("not a print").name == "Untitled"
    assert print_history.from_dict({"grams": "nan", "minutes": float("inf")}).grams == 0.0
    assert print_history.from_dict({"when": 12345}).when
    assert print_history.records(["junk", 7, None, {"id": "a"}])[0].id == "a"
    assert print_history.records("a stray text") == []
    assert print_history.records(None) == [] and print_history.records(7) == []


def test_the_strongest_thing_that_happened_is_what_is_kept():
    assert print_history.strongest(["saved", "sent", "marked", "started"]) == "started"
    assert print_history.strongest(["saved", "marked", "sent"]) == "marked"
    assert print_history.strongest(["saved", "sent"]) == "sent"
    assert print_history.strongest(["saved"]) == "saved"
    assert print_history.strongest([]) == "" and print_history.strongest(None) == ""
    assert print_history.strongest(["nonsense", None]) == ""


def test_only_a_print_that_ran_counts_in_the_totals():
    """A G-code on an SD card has printed nothing, so it weighs nothing (see volura.spools)."""
    ran = [entry("", how="started", grams=100.0, minutes=60.0),
           entry("", how="marked", grams=50.0, minutes=30.0),
           entry("", how="sent", grams=900.0, minutes=900.0),
           entry("", how="saved", grams=900.0, minutes=900.0)]
    assert print_history.totals(ran) == (2, 150.0, 90.0)
    assert print_history.totals([]) == (0, 0.0, 0.0)
    assert print_history.totals(["junk", None, 7]) == (0, 0.0, 0.0)
    assert print_history.printed(entry("", how="marked"))
    assert not print_history.printed(entry("", how="sent"))
    assert not print_history.printed("a print of junk")


def test_the_history_says_what_it_printed_in_total():
    ran = [entry("", how="started", grams=1000.0, minutes=120.0),
           entry("", how="marked", grams=250.0, minutes=15.0),
           entry("", how="saved", grams=5000.0, minutes=600.0)]
    assert print_history.totals_text(ran) == "2 prints, 1.25 kg filament, 2.2 h printing"
    assert print_history.totals_text([]) == NO_PRINTS
    assert print_history.totals_text(["junk"]) == NO_PRINTS


def test_a_print_says_what_it_was_and_what_became_of_it():
    made = print_history.Entry(name="Bracket", printer="Octo", grams=62.5, minutes=105.0,
                               how="started", when="2026-03-04T15:30:00")
    assert print_history.entry_text(made) == \
        "Started 2026-03-04 15:30: Bracket on Octo - 62.5 g in 1 h 45 min"
    assert print_history.entry_text(print_history.Entry(name="Box",
                                                       when="2026-03-04T15:30:00")) == \
        "Saved 2026-03-04 15:30: Box", "a print that says nothing more says what it was"
    assert print_history.when_text("yesterday") == "yesterday", "junk stands as it was"
    assert print_history.when_text(None) == ""
    assert print_history.how_text("nonsense") == "Saved"
    assert print_history.time_text(0.0) == "< 1 min"
    assert print_history.time_text("junk") == "< 1 min"
    assert print_history.time_text(None) == "< 1 min"
    assert print_history.grams_text("junk") == "0 g"


# ------------------------------------------------------------------ the list in the settings

def test_the_list_keeps_the_newest_first_and_lets_the_oldest_go():
    older = [entry("", name="old one"), entry("", name="old two")]
    out = print_history.add(older, entry("", name="new"))
    assert [item["name"] for item in out] == ["new", "old one", "old two"]
    assert all(isinstance(item, dict) for item in out), "the settings keep plain dicts"
    assert print_history.add(None, entry("", name="only"))[0]["name"] == "only"
    assert [item["name"] for item in print_history.add(["junk", 7, None], entry("", name="a"))] \
        == ["a"], "what is no print is no print"
    many = print_history.records([entry("", name=f"p{i}") for i in range(5)])
    assert [item["name"] for item in print_history.add(many, entry("", name="new"), limit=3)] \
        == ["new", "p0", "p1"]
    assert len(print_history.add(many, entry("", name="x"), limit="junk")) == 6, \
        "a limit that is no limit is the usual limit of 200 prints"
    assert len(print_history.add(many, entry("", name="x"), limit=0)) == 1, \
        "the print that was just made is never the one that drops off"
    assert len(print_history.add(many, entry("", name="x"), limit=None)) == 6
    huge = print_history.records([entry("", name=f"h{i}") for i in range(1000)])
    cut = print_history.add(huge, entry("", name="newest"), limit=print_history.LIMIT)
    assert len(cut) == print_history.LIMIT == 200, "a long history is cut to the usual length"
    assert cut[0]["name"] == "newest" and cut[-1]["name"] == "h198", "the oldest prints fall off"


def test_a_print_that_drops_off_the_list_loses_the_copy_of_its_gcode(copies, tmp_path):
    """The folder the history keeps its G-codes in may not grow without end."""
    last = copies / "last.gcode"
    last.write_text(GCODE, encoding="utf-8")
    mine = tmp_path / "not mine.gcode"
    mine.write_text(GCODE, encoding="utf-8")
    out = print_history.add([entry(last, name="kept"), entry(mine, name="saved")],
                            entry("", name="new"), limit=2)
    assert [item["name"] for item in out] == ["new", "kept"], "the oldest went off the list"
    assert kept(copies) == ["last.gcode"], "the file the user saved stands where it was"
    out = print_history.add(out, entry("", name="newer"), limit=2)
    assert kept(copies) == [], "the copy of the print that dropped off is gone"
    assert mine.is_file(), "what the user saved itself is nobody else's to take away"


def test_removing_a_print_takes_its_copy_and_clearing_takes_every_one(copies):
    one, two = copies / "one.gcode", copies / "two.gcode"
    for path in (one, two):
        path.write_text(GCODE, encoding="utf-8")
    history = print_history.records([entry(one, name="one"), entry(two, name="two")])
    out = print_history.remove(history, history[1].id)
    assert [item["name"] for item in out] == ["one"] and kept(copies) == ["one.gcode"]
    assert print_history.remove(out, "nobody") == out, "a print that is not there changes nothing"
    assert print_history.cleared(out) == [] and kept(copies) == []
    assert print_history.cleared([]) == [] and kept(copies) == []


def test_a_copy_is_kept_in_the_history_folder_and_nowhere_else(copies):
    source = copies.parent / "temp.gcode"
    source.write_text(GCODE, encoding="utf-8")
    kept_at = print_history.keep_copy(source, "ab12cd34")
    assert in_folder(copies, kept_at) and os.path.isfile(kept_at)
    assert open(kept_at, encoding="utf-8").read() == GCODE
    assert in_folder(copies, print_history.keep_copy(source, "../outside")), \
        "an id that points out of the folder stays in it"
    assert print_history.keep_copy(copies.parent / "never was.gcode", "x") == ""
    assert print_history.keep_copy("", "x") == "" and print_history.keep_copy(None, "x") == ""
    assert not print_history.is_copy(source) and not print_history.is_copy(copies)
    assert not print_history.is_copy("") and not print_history.is_copy(None)
    print_history.forget(source)
    assert source.is_file(), "Volura takes away nothing of the user's own files"
    print_history.forget(copies / "ab12cd34.gcode")
    assert "ab12cd34.gcode" not in kept(copies) and source.is_file()


def test_one_entry_per_print_written_when_the_preview_closes(win, tmp_path, copies):
    """What `store` writes is what the preview and the history dialog both use."""
    reset(win)
    source = slice_of(tmp_path, "slice")
    made = print_history.store(win.settings,
                               print_history.Entry(name="Box", how="started", grams=20.0,
                                                   minutes=5.0),
                               source=source)
    assert [item["how"] for item in win.settings.print_history] == ["started"]
    assert recorded(win)[0].grams == 20.0
    assert made.gcode and in_folder(copies, made.gcode), "a started print keeps its G-code"
    assert source.is_file() and kept(copies) == [os.path.basename(made.gcode)]
    # a print of which nothing is left but the saying that it ran has nothing to print again
    reset(win)
    before = kept(copies)
    print_history.store(win.settings, print_history.Entry(name="Box", how="marked"),
                        source=source)
    assert recorded(win)[0].gcode == ""
    assert kept(copies) == before, "a print that never went to a printer needs no copy"
    # a window without settings has nowhere to write to, and says nothing about that
    assert print_history.store(None, print_history.Entry(name="Box", how="sent",
                                                        grams=1.0)).gcode == ""
    assert kept(copies) == before


def test_the_settings_remember_the_history_and_survive_junk(win):
    from volura.app import Settings
    made = entry("/prints/bracket.gcode", name="Bracket", how="started", grams=12.0,
                 minutes=3.0, printer="Octo", filaments=[("PLA", "#ff0000")])
    reset(win, made)
    assert Settings().print_history == [print_history.to_dict(made)]
    qsettings().setValue("print_history", "nonsense")
    qsettings().sync()
    assert Settings().print_history == []
    qsettings().remove("print_history")
    qsettings().sync()
    assert Settings().print_history == []


# ------------------------------------------------------------------ what the preview writes

def test_a_preview_that_started_a_print_records_one_started_print(win, tmp_path, copies,
                                                                  monkeypatch):
    from volura import print_preview as pp
    reset(win, printers=[PRINTER])
    sent = Uploads()
    monkeypatch.setattr(pp.printer_link, "upload", sent)
    monkeypatch.setattr(pp.SendDialog, "exec", press)
    dlg = preview(win, tmp_path, RED, grams=60.0, minutes=42.5)
    try:
        dlg._send()
        assert dlg._job.wait(20000), dlg._job.error
        assert sent.calls == [("Octo", "box.gcode", "volura.gcode", True)]
        assert dlg.sent == "Sent volura.gcode to Octo and started printing"
        dlg.done(QDialog.Accepted)
        dlg.done(QDialog.Rejected)              # closing it again adds nothing to the history
        history = recorded(win)
        assert [item.how for item in history] == ["started"], "one print, one entry"
        got = history[0]
        assert (got.name, got.printer) == ("Untitled", "Octo")
        assert (got.grams, got.minutes) == (60.0, 42.5), "the numbers of the slice, not its text"
        assert got.filaments == [["Red PLA", "#ff0000"]]
        assert in_folder(copies, got.gcode) and got.gcode != str(tmp_path / "box.gcode")
        assert open(got.gcode, encoding="utf-8").read() == GCODE
        assert print_history.gcode_of(got) == got.gcode
    finally:
        empty(win)
        dlg.close()


def test_a_print_that_was_saved_and_started_is_recorded_as_started(win, tmp_path, copies,
                                                                   monkeypatch):
    """The strongest thing that happened is what is kept, and the saved file is the G-code."""
    from volura import print_preview as pp
    reset(win, printers=[PRINTER])
    monkeypatch.setattr(pp.printer_link, "upload", Uploads())
    monkeypatch.setattr(pp.SendDialog, "exec", press)
    dlg = preview(win, tmp_path, RED, grams=60.0, minutes=42.5)
    try:
        target = str(tmp_path / "saved.gcode")
        dlg.save_to(target)
        dlg._send()
        assert dlg._job.wait(20000), dlg._job.error
        dlg.done(QDialog.Accepted)
        assert [item.how for item in recorded(win)] == ["started"]
        assert recorded(win)[0].gcode == target
        assert kept(copies) == [], "the file the user saved is already kept somewhere else"
    finally:
        empty(win)
        dlg.close()


def test_a_print_that_only_went_to_the_printer_is_recorded_as_sent(win, tmp_path, copies,
                                                                   monkeypatch):
    """Nothing ran yet, so nothing counts in the totals - but it can go on the printer again."""
    from volura import print_preview as pp
    reset(win, printers=[PRINTER])
    sent = Uploads()
    monkeypatch.setattr(pp.printer_link, "upload", sent)
    monkeypatch.setattr(pp.SendDialog, "exec", hand_over)
    dlg = preview(win, tmp_path, RED, grams=60.0, minutes=42.5)
    try:
        dlg._send()
        assert dlg._job.wait(20000), dlg._job.error
        assert sent.calls == [("Octo", "box.gcode", "volura.gcode", False)]
        dlg.done(QDialog.Accepted)
        assert [item.how for item in recorded(win)] == ["sent"]
        assert recorded(win)[0].printer == "Octo"
        assert in_folder(copies, recorded(win)[0].gcode)
        assert recorded(win)[0].grams == 60.0
        assert print_history.totals(win.settings.print_history)[0] == 0
    finally:
        empty(win)
        dlg.close()


def test_a_send_that_never_reached_the_printer_records_nothing(win, tmp_path, copies,
                                                               monkeypatch):
    """A printer that does not answer has printed nothing, and so was nothing printed."""
    from volura import print_preview as pp
    reset(win, printers=[PRINTER])
    box = Answer()
    monkeypatch.setattr(pp, "QMessageBox", box)
    monkeypatch.setattr(pp.printer_link, "upload", Uploads(fails=True))
    monkeypatch.setattr(pp.SendDialog, "exec", press)
    dlg = preview(win, tmp_path, RED, grams=60.0, minutes=42.5)
    try:
        dlg._send()
        assert dlg._job.wait(20000), dlg._job.error
        assert box.said, "what went wrong is said to the user"
        assert dlg.sent is None and dlg.send_button.isEnabled()
        dlg.done(QDialog.Accepted)
        assert win.settings.print_history == [] and kept(copies) == []
    finally:
        empty(win)
        dlg.close()


def test_a_preview_that_was_only_looked_at_records_nothing(win, tmp_path, copies):
    reset(win)
    dlg = preview(win, tmp_path, RED, grams=60.0, minutes=42.5)
    try:
        dlg.done(QDialog.Rejected)
        assert win.settings.print_history == [] and kept(copies) == []
    finally:
        dlg.close()


def test_a_print_marked_as_printed_is_named_after_its_model(win, tmp_path, copies):
    """A print started by hand from an SD card is a print: it just has no G-code to send again."""
    reset(win)
    was, win.doc.path = win.doc.path, str(tmp_path / "bracket.volura")
    try:
        dlg = preview(win, tmp_path, RED, grams=60.0, minutes=42.5)
        try:
            dlg._mark_printed()
            dlg.done(QDialog.Accepted)
            assert [item.how for item in recorded(win)] == ["marked"]
            got = recorded(win)[0]
            assert got.name == "bracket" and got.printer == ""
            assert got.gcode == "" and got.filaments == [["Red PLA", "#ff0000"]]
            assert kept(copies) == []
            assert print_history.totals(win.settings.print_history) == (1, 60.0, 42.5)
        finally:
            dlg.close()
    finally:
        win.doc.path = was


# ------------------------------------------------------------------ Window > Print History…

def test_the_history_lists_the_prints_with_their_totals(win, tmp_path):
    from volura import print_history_dialog as hd
    there = slice_of(tmp_path, "bracket")
    reset(win, entry(there, name="Bracket", how="started", grams=100.0, minutes=90.0,
                     printer="Octo", when="2026-03-04T15:30:00"),
          entry(tmp_path / "gone.gcode", name="Box", how="saved", grams=20.0, minutes=5.0,
                when="2026-03-05T09:00:00"))
    dim = win.palette().color(QPalette.Disabled, QPalette.Text)
    dlg = hd.PrintHistoryDialog(win)
    try:
        assert dlg.windowTitle() == "Print History" and dlg.table.columnCount() == 6
        assert dlg.table.rowCount() == 2
        assert [dlg.table.item(0, col).text() for col in range(6)] == \
            ["2026-03-04 15:30", "Bracket", "Octo", "Started", "100 g", "1 h 30 min"]
        assert dlg.table.item(1, 3).text() == "Saved"
        assert dlg.table.item(1, 5).text() == "5 min"
        assert dlg.table.item(0, 0).foreground().color() != dim
        assert dlg.table.item(1, 0).foreground().color() == dim, \
            "a print whose G-code is gone stands grey"
        assert "Started 2026-03-04 15:30: Bracket on Octo" in dlg.table.item(0, 1).toolTip()
        assert "The G-code is no longer there" in dlg.table.item(1, 1).toolTip()
        assert dlg.total.text() == "1 print, 0.1 kg filament, 1.5 h printing"
        assert not dlg.b_again.isEnabled() and not dlg.b_show.isEnabled(), "nothing is chosen"
        assert not dlg.b_remove.isEnabled() and dlg.b_clear.isEnabled()
        dlg.table.selectRow(1)
        assert not dlg.b_again.isEnabled(), "the G-code is gone, so there is nothing to send"
        assert "no longer there" in dlg.b_again.toolTip()
        assert not dlg.b_show.isEnabled() and dlg.b_remove.isEnabled()
        dlg.table.selectRow(0)
        assert not dlg.b_again.isEnabled(), "there is no printer under Preferences yet"
        assert "Preferences > Printers" in dlg.b_again.toolTip()
        assert dlg.b_show.isEnabled() and dlg.b_remove.isEnabled()
    finally:
        dlg.close()


def test_print_again_sends_the_stored_gcode_and_records_a_new_print(win, tmp_path, copies,
                                                                   monkeypatch):
    from volura import print_history_dialog as hd, print_preview as pp
    source = slice_of(tmp_path, "bracket")
    reset(win, entry(source, name="Bracket", how="saved", grams=100.0, minutes=90.0),
          printers=[PRINTER])
    sent = Uploads()
    monkeypatch.setattr(pp.printer_link, "upload", sent)
    monkeypatch.setattr(pp.SendDialog, "exec", press)
    dlg = hd.PrintHistoryDialog(win)
    try:
        dlg.table.selectRow(0)
        assert dlg.b_again.isEnabled(), "the G-code is there and a printer is set up"
        dlg._print_again()
        assert dlg._job is not None and dlg._job.wait(20000), dlg._job.error
        assert sent.calls == [("Octo", "bracket.gcode", "Bracket.gcode", True)], \
            "the stored G-code goes to the printer, under the name of the model"
        history = recorded(win)
        assert [item.how for item in history] == ["started", "saved"], "the new print is on top"
        again = history[0]
        assert (again.name, again.printer) == ("Bracket", "Octo")
        assert (again.grams, again.minutes) == (100.0, 90.0)
        assert in_folder(copies, again.gcode) and again.gcode != str(source)
        assert source.is_file(), "the file it was sent from stays where it is"
        assert dlg.table.rowCount() == 2 and dlg.table.currentRow() == 0
        assert dlg.total.text() == "1 print, 0.1 kg filament, 1.5 h printing"
    finally:
        empty(win)
        dlg.close()


def test_print_again_wants_a_gcode_that_is_still_there(win, tmp_path, monkeypatch):
    from volura import print_history_dialog as hd, print_preview as pp
    reset(win, entry(tmp_path / "gone.gcode", name="Box", how="started", grams=10.0),
          printers=[PRINTER])
    sent = Uploads()
    monkeypatch.setattr(pp.printer_link, "upload", sent)
    monkeypatch.setattr(pp.SendDialog, "exec", press)
    dlg = hd.PrintHistoryDialog(win)
    try:
        dlg.table.selectRow(0)
        assert not dlg.b_again.isEnabled(), "there is no G-code left to send"
        dlg._print_again()
        assert dlg._job is None and sent.calls == []
        assert [item.how for item in recorded(win)] == ["started"], "nothing was written"
    finally:
        empty(win)
        dlg.close()


def test_a_cancelled_question_leaves_the_history_as_it_was(win, tmp_path, copies, monkeypatch):
    """Print Again… asks which printer takes it; nobody who answers asks prints nothing."""
    from volura import print_history_dialog as hd, print_preview as pp
    source = slice_of(tmp_path, "bracket")
    reset(win, entry(source, name="Bracket", how="started", grams=10.0), printers=[PRINTER])
    monkeypatch.setattr(pp.SendDialog, "exec", neither)
    dlg = hd.PrintHistoryDialog(win)
    try:
        dlg.table.selectRow(0)
        assert dlg.b_again.isEnabled()
        dlg._print_again()
        assert dlg._job is None
        assert [item.how for item in recorded(win)] == ["started"], "one print, not two"
        assert kept(copies) == [] and dlg.b_again.isEnabled()
    finally:
        empty(win)
        dlg.close()


def test_show_file_opens_the_folder_of_the_gcode(win, tmp_path, monkeypatch):
    from volura import print_history_dialog as hd
    source = slice_of(tmp_path, "bracket")
    reset(win, entry(source, name="Bracket", how="started"),
          entry(tmp_path / "gone.gcode", name="Box", how="started"))
    opened = Opened()
    monkeypatch.setattr(hd, "QDesktopServices", opened)
    dlg = hd.PrintHistoryDialog(win)
    try:
        dlg.table.selectRow(1)
        assert not dlg.b_show.isEnabled(), "a print of which the file is gone has no folder"
        dlg._show_file()
        assert opened.urls == []
        dlg.table.selectRow(0)
        assert dlg.b_show.isEnabled()
        dlg._show_file()
        assert opened.urls == [QUrl.fromLocalFile(str(tmp_path)).toString()]
    finally:
        dlg.close()


def test_the_history_removes_a_print_and_clears_them_all(win, tmp_path, copies, monkeypatch):
    from volura import print_history_dialog as hd
    one = copies / "one.gcode"
    one.write_text(GCODE, encoding="utf-8")
    mine = slice_of(tmp_path, "mine")
    reset(win, entry(one, name="One", how="started", grams=50.0, minutes=20.0),
          entry(mine, name="Mine", how="marked", grams=50.0, minutes=20.0))
    box = Answer(Answer.No)
    monkeypatch.setattr(hd, "QMessageBox", box)
    dlg = hd.PrintHistoryDialog(win)
    try:
        dlg.table.selectRow(0)
        dlg._remove_entry()
        assert box.asked and "One" in box.asked[0][2]
        assert len(win.settings.print_history) == 2, "no said no, so the print stays"
        assert dlg.total.text() == "2 prints, 0.1 kg filament, 0.7 h printing"
        box.answer = Answer.Yes
        dlg._remove_entry()
        assert [item["name"] for item in win.settings.print_history] == ["Mine"]
        assert kept(copies) == [], "the G-code Volura kept itself went with the print"
        assert mine.is_file(), "the file the user saved stays exactly where it is"
        assert dlg.table.rowCount() == 1
        assert dlg.total.text() == "1 print, 0.05 kg filament, 0.3 h printing"
        box.asked.clear()
        dlg._clear_all()
        assert box.asked and win.settings.print_history == [] and kept(copies) == []
        assert dlg.table.rowCount() == 0 and dlg.total.text() == NO_PRINTS
        assert not dlg.b_clear.isEnabled() and not dlg.b_remove.isEnabled()
        dlg._clear_all()
        assert len(box.asked) == 1, "an empty history is not asked about again"
    finally:
        dlg.close()


def test_the_history_of_a_window_without_settings_changes_nothing(win, tmp_path):
    """A dialog that has no window to keep its list has nothing to keep, and says nothing."""
    from volura import print_history_dialog as hd
    source = slice_of(tmp_path, "bracket")
    reset(win, entry(source, name="Bracket", how="started"))
    dlg = hd.PrintHistoryDialog(None)
    try:
        assert dlg.table.rowCount() == 0 and dlg.total.text() == NO_PRINTS
        assert not dlg.b_again.isEnabled() and not dlg.b_clear.isEnabled()
        dlg._keep()
        dlg._remove_entry()
        dlg._clear_all()
    finally:
        dlg.close()
    assert [item.name for item in recorded(win)] == ["Bracket"]


def test_the_window_offers_the_print_history(win, monkeypatch):
    from volura import print_history_dialog as hd
    seen = []

    def show(dlg):
        seen.append(dlg.windowTitle())
        dlg.close()
        return QDialog.Rejected

    monkeypatch.setattr(hd.PrintHistoryDialog, "exec", show)
    assert "Print History" in win.a_print_history.text().replace("&", "")
    win.a_print_history.trigger()
    assert seen == ["Print History"]


# ------------------------------------------------------------------ what is odd about a print


def test_a_moment_that_is_no_moment_says_what_it_was_given():
    assert print_history.when_text("2026-03-04T15:30:00.123456") == "2026-03-04 15:30"
    assert print_history.when_text("middle of last Tuesday") == "middle of last Tuesday"
    odd = print_history.from_dict({"when": "noon-ish", "grams": "many", "minutes": "some"})
    line = print_history.entry_text(odd)
    assert "Untitled" in line and "Saved" in line and "0 g" not in line


def test_a_name_stands_on_one_line_of_the_list():
    loud = print_history.Entry(name="base   plate\nwith\tlid", printer="  Octo  Print  ",
                               filaments=[("red\nPLA", "#ff0000")])
    assert loud.name == "base plate with lid"
    assert loud.printer == "Octo Print" and loud.filaments == [["red PLA", "#ff0000"]]
    assert print_history.Entry(name="  \n \t ").name == "Untitled"
    assert len(print_history.Entry(name="k" * 900).name) == print_history.MAX_TEXT
    assert print_history.Entry(name="0").name == "0", "a name that is a zero is still a name"


def test_a_folder_is_no_gcode_and_is_never_taken_away(copies, tmp_path):
    elsewhere = tmp_path / "somewhere"
    elsewhere.mkdir()
    inside = copies / "aaabbbccc.gcode"
    inside.mkdir()                       # junk in the settings can make this happen
    assert print_history.keep_copy(str(elsewhere), "id1") == "", "a folder holds no G-code"
    assert print_history.gcode_of(entry(elsewhere)) is None
    assert print_history.gcode_of(entry(inside)) is None
    print_history.forget(inside)         # may not raise, and may not take the folder away
    assert inside.is_dir() and elsewhere.is_dir()
    assert kept(copies) == ["aaabbbccc.gcode"]


def test_an_id_that_would_walk_out_of_the_folder_stays_inside_it(copies):
    folder = os.path.abspath(str(copies))
    for evil in ("../../evil", "..\\..\\evil", "/abs/olute", "a b/../c", "", "x" * 400):
        path = print_history.copy_path(evil)
        assert os.path.dirname(path) == folder, f"{evil!r} walks out of the history folder"
        assert print_history.is_copy(path)


def test_a_copy_that_cannot_be_written_keeps_the_print_with_what_it_used(win, tmp_path, copies,
                                                                        monkeypatch):
    source = slice_of(tmp_path, "bracket")

    def refuse(src, dst, *args, **kwargs):
        raise OSError("the disk of this computer is full")

    monkeypatch.setattr(print_history.shutil, "copyfile", refuse)
    assert print_history.keep_copy(str(source), "id1") == ""
    reset(win)
    try:
        made = print_history.store(win.settings,
                                   print_history.Entry(name="Bracket", how="started"),
                                   source=str(source))
        assert made.gcode == str(source), "the print keeps the file it really went with"
        assert [item.name for item in recorded(win)] == ["Bracket"]
        assert kept(copies) == []
    finally:
        empty(win)


def test_a_name_of_every_kind_of_letter_comes_back_from_the_settings(win):
    from volura.app import Settings
    odd = entry("/prints/keutel 2.gcode", name='Keutel «ç» – 中文 😀 "quoted" plate',
                printer="Ünïcode ✓", how="marked", grams=1.25, minutes=0.0,
                filaments=[("Étamine ☺", "#00ff00")])
    long = entry("/prints/x.gcode", name="k" * 250, how="sent")
    reset(win, odd, long)
    try:
        assert Settings().print_history == [print_history.to_dict(odd), print_history.to_dict(long)]
        assert recorded(win)[0].name == odd.name and recorded(win)[0].grams == 1.25
    finally:
        empty(win)

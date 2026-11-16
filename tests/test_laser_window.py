"""The laser controller window: what it shows, and what its buttons send to the board.

`volura/laser_window.py` is the hand on the machine: Laser and Mill > Laser Controller…
opens it, the pad jogs, "Zero here" puts the work origin under the head, the overrides
change the feed and the power while a job runs, and the job comes from the worksheet (or
from a G-code file) and goes out block by block through `volura/laser.py`.

The board here is a fake port that answers what it is told, so the tests see the very
blocks the window sends and can answer them at their leisure. No message box may appear
on the screen: they are caught, because a question on the screen stops a test run.

Run with:  python -m pytest tests/test_laser_window.py
"""
import os
import sys
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from PySide6.QtCore import QCoreApplication  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from volura import grbl, laser  # noqa: E402
from volura.i18n import tr  # noqa: E402
from volura.laser_window import LaserWindow  # noqa: E402
from test_laser import FakePort  # noqa: E402

_app = QApplication.instance() or QApplication(sys.argv[:1])


def rect(w, h, x=0.0, y=0.0):
    return np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=float)


def pump(dlg, what=lambda: True, timeout=4.0):
    """Let the thread read the cable and the window look at it, until `what()` is true."""
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        QCoreApplication.processEvents()
        if what():
            return True
        time.sleep(0.005)
    QCoreApplication.processEvents()
    return bool(what())


def pick(box, value):
    """The index of the choice that carries `value` in a combo box."""
    return [box.itemData(i) for i in range(box.count())].index(value)


class Box:
    """The message boxes of the window, caught instead of shown."""

    def __init__(self):
        self.messages = []

    def _say(self, parent, title, text, *a, **k):
        self.messages.append(text)
        return 0

    information = warning = critical = _say


class SlowPort(FakePort):
    """A board that thinks about every block: it answers only when the test says so.

    A job that would be over between two clicks cannot be paused, so the tests that work
    with a running job hang it on this board.
    """

    def _answer(self, data):
        if data == b"?":
            self.say(self.report)         # the board always knows where it is


@pytest.fixture(autouse=True)
def machine_choices():
    """Every test starts with a board of its own: nothing remembered from the last one."""
    from volura.settings import qsettings
    q = qsettings()
    for key in (laser.PORT_KEY, laser.BAUD_KEY, laser.STEP_KEY, laser.FEED_KEY):
        q.remove(key)
    q.sync()
    yield q


@pytest.fixture()
def boxes(monkeypatch):
    import volura.laser_window as lw
    box = Box()
    monkeypatch.setattr(lw, "QMessageBox", box)
    return box


@pytest.fixture()
def win():
    from volura.app import MainWindow
    w = MainWindow()
    w.doc.dirty = False
    QCoreApplication.processEvents()
    yield w
    w.doc.dirty = False
    w.close()


@pytest.fixture()
def dlg(win):
    d = LaserWindow(win)
    d.show()
    QCoreApplication.processEvents()
    yield d
    d.shutdown()


def connect(dlg, monkeypatch, cls=FakePort):
    """Hang a board of `cls` on the cable of `dlg` and open it.

    The thread has to have taken the first blocks over before a test looks at the cable,
    so this waits until the board was reset, asked and unlocked.
    """
    port = cls()
    monkeypatch.setattr(laser, "ports", lambda: [("COM9", "Test board")])
    monkeypatch.setattr(laser, "open_port", lambda device, baud=None: port)
    dlg.refresh_ports()
    dlg.connect_board()
    assert pump(dlg, lambda: all(block in port.text()
                                 for block in (grbl.RESET, grbl.SETTINGS, grbl.UNLOCK)))
    return port


@pytest.fixture()
def board(dlg, monkeypatch):
    """The window with a board on its cable that answers everything at once."""
    return connect(dlg, monkeypatch)


@pytest.fixture()
def slow(dlg, monkeypatch):
    """The window with a board that thinks about every block, so a job keeps running."""
    return connect(dlg, monkeypatch, SlowPort)


def worksheet(win):
    """A square on the worksheet, with the speed and the passes the laser needs."""
    sheet = win.doc.sheet
    for lay in sheet.layers.values():
        lay.params.update({"speed": 10.0, "power": 50.0, "passes": 1})
    sheet.add_path(rect(20.0, 20.0), closed=True, layer="Cut", src=1)
    return sheet


# ------------------------------------------------------------------ the window

def clean(text):
    return text.replace("&", "").replace("…", "").strip()


def test_the_command_is_in_the_menu_of_the_machine(win):
    menu = None
    for action in win.menuBar().actions():
        if action.menu() is not None and clean(action.menu().title()) == "Laser and Mill":
            menu = action.menu()
    assert menu is not None, "there is no Laser and Mill menu"
    labels = [clean(a.text()) for a in menu.actions() if not a.isSeparator()]
    assert "Laser Controller" in labels, labels


def test_the_command_opens_one_window_that_stays_the_same(win):
    first = win.laser_controller()
    second = win.laser_controller()
    try:
        assert isinstance(first, LaserWindow)
        assert first is second, "the command opened a second controller"
        assert clean(first.windowTitle()) == "Laser Controller"
    finally:
        first.shutdown()
        win._laser_controller = None


def test_the_window_names_every_control_for_a_screen_reader(dlg):
    from test_accessibility import report
    problems = report(dlg)
    assert not problems, problems


def test_a_worksheet_without_a_cable_says_what_is_missing(dlg, monkeypatch):
    monkeypatch.setattr(laser, "has_serial", lambda: True)
    monkeypatch.setattr(laser, "ports", lambda: [])
    dlg.refresh_ports()
    assert "serial" in dlg.port_hint.text().lower()
    assert dlg.connected is False
    assert dlg.b_start.isEnabled() is False


def test_a_computer_without_pyserial_is_told_what_to_install(dlg, monkeypatch):
    monkeypatch.setattr(laser, "has_serial", lambda: False)
    dlg.refresh_ports()
    assert "pyserial" in dlg.port_hint.text()


# ------------------------------------------------------------------ the cable

def test_the_ports_of_the_computer_stand_in_the_list(dlg, monkeypatch):
    monkeypatch.setattr(laser, "ports", lambda: [("COM9", "USB Serial"), ("COM3", "Bluetooth")])
    dlg.refresh_ports()
    assert [dlg.port_box.itemData(i) for i in range(dlg.port_box.count())] == ["COM9", "COM3"]


def test_connecting_opens_the_port_and_asks_the_board_who_it_is(dlg, board):
    assert dlg.connected
    text = board.text()
    assert grbl.RESET in text and grbl.SETTINGS in text and grbl.UNLOCK in text
    assert dlg.b_connect.text() == tr("Disconnect")
    assert dlg.port_box.isEnabled() is False


def test_a_port_that_will_not_open_is_said_and_nothing_is_connected(dlg, monkeypatch, boxes):
    def refuse(device, baud=None):
        raise laser.CableError(f"{device} is busy")
    monkeypatch.setattr(laser, "ports", lambda: [("COM9", "Test board")])
    monkeypatch.setattr(laser, "open_port", refuse)
    dlg.refresh_ports()
    dlg.connect_board()
    assert dlg.connected is False
    assert boxes.messages and "COM9" in boxes.messages[0]


def test_without_a_port_to_choose_nothing_is_opened(dlg, monkeypatch, boxes):
    monkeypatch.setattr(laser, "ports", lambda: [])
    dlg.refresh_ports()
    dlg.connect_board()
    assert dlg.connected is False and boxes.messages


def test_the_port_and_the_speed_are_remembered_for_next_time(dlg, board):
    from volura.settings import qsettings
    q = qsettings()
    assert str(q.value(laser.PORT_KEY)) == "COM9"
    assert int(q.value(laser.BAUD_KEY)) == grbl.DEFAULT_BAUD


def test_the_board_that_answered_stands_on_the_screen(dlg, board):
    board.say("<Idle|MPos:12.000,3.000,0.000|WPos:2.000,1.000,0.000|FS:800,0>")
    assert pump(dlg, lambda: "12.000" in dlg.mpos_label.text())
    assert "2.000" in dlg.wpos_label.text()
    assert dlg.state_label.text() == grbl.state_label("Idle")
    assert "800" in dlg.feed_label.text()


def test_a_cable_that_breaks_is_said_and_the_window_lets_go(dlg, board):
    board.raises = OSError("the USB cable is gone")
    assert pump(dlg, lambda: not dlg.connected)
    assert "USB cable" in dlg.log.toPlainText()


# ------------------------------------------------------------------ the pad and the origin

def test_the_pad_moves_one_step_of_the_size_that_is_chosen(dlg, board):
    dlg.step_box.setCurrentIndex(pick(dlg.step_box, 10.0))
    dlg.feed_box.setCurrentIndex(pick(dlg.feed_box, 3000))
    dlg.pad_buttons["→"].click()
    assert pump(dlg, lambda: b"$J=G21 G91 X10.000 F3000\n" in b"".join(board.written))


def test_the_pad_lowers_the_head_of_the_same_step(dlg, board):
    dlg.step_box.setCurrentIndex(pick(dlg.step_box, 5.0))
    dlg.feed_box.setCurrentIndex(pick(dlg.feed_box, 500))
    dlg.pad_buttons["Z ↓"].click()
    assert pump(dlg, lambda: b"$J=G21 G91 Z-5.000 F500\n" in b"".join(board.written))


def test_the_home_of_the_pad_goes_to_the_origin_of_the_work(dlg, board):
    dlg.b_home_origin.click()
    assert pump(dlg, lambda: b"G90 G0 X0.000 Y0.000\n" in b"".join(board.written))


def test_zero_here_puts_the_work_origin_under_the_head(dlg, board):
    dlg.b_zero_xy.click()
    assert pump(dlg, lambda: b"G10 L20 P1 X0 Y0\n" in b"".join(board.written))


def test_zero_x_leaves_the_other_axes_where_they_are(dlg, board):
    dlg.b_zero_x.click()
    assert pump(dlg, lambda: b"G10 L20 P1 X0\n" in b"".join(board.written))
    assert b"G10 L20 P1 X0 Y0\n" not in b"".join(board.written)


def test_the_board_has_its_own_buttons(dlg, board):
    dlg.b_unlock.click()
    dlg.b_home.click()
    dlg.b_settings.click()
    assert pump(dlg, lambda: all(block in board.text()
                                 for block in ("$X", "$H", "$$")))


def test_the_reset_of_the_window_empties_what_the_board_has_queued(dlg, board):
    dlg.b_reset.click()
    assert pump(dlg, lambda: grbl.RESET in board.text())


def test_check_mode_is_the_same_block_on_and_off(dlg, board):
    dlg.b_check.click()
    assert pump(dlg, lambda: board.text().count("$C") == 1)
    dlg.b_check.click()
    assert pump(dlg, lambda: board.text().count("$C") == 2)


def test_the_overrides_are_the_bytes_of_the_board(dlg, board):
    more = [b for b in dlg.findChildren(type(dlg.b_start)) if b.accessibleName().startswith("More")]
    assert more, "there is no override button to press"
    more[0].click()
    assert pump(dlg, lambda: grbl.override("feed", "up10") in board.written)


# ------------------------------------------------------------------ the job

def test_the_job_of_the_worksheet_is_loaded_as_blocks(dlg, win, monkeypatch):
    worksheet(win)
    monkeypatch.setattr(win, "sheet_export_options",
                        lambda kind: {"layers": ["Cut"], "compensate": False,
                                      "s_max": 1000, "dynamic": True})
    dlg.load_job()
    assert dlg.job[0] == "G21"
    assert any(line.startswith("G1") for line in dlg.job)
    assert "blocks" in dlg.job_label.text()


def test_a_worksheet_that_is_not_for_the_laser_is_a_message_and_no_job(dlg, win, monkeypatch,
                                                                       boxes):
    monkeypatch.setattr(win, "sheet_export_options",
                        lambda kind: {"layers": ["Cut"], "compensate": False,
                                      "s_max": 1000, "dynamic": True})
    dlg.load_job()
    assert not dlg.job and boxes.messages


def test_a_cancelled_export_dialog_leaves_the_job_alone(dlg, win, monkeypatch):
    monkeypatch.setattr(win, "sheet_export_options", lambda kind: None)
    dlg.set_job(["G21"], "old")
    dlg.load_job()
    assert dlg.job == ["G21"]


def test_a_gcode_file_on_disk_can_be_sent(dlg, monkeypatch, tmp_path):
    import volura.laser_window as lw
    path = tmp_path / "job.gcode"
    path.write_text("; my drawing\nG21\nG90\nM5\n", encoding="utf-8")
    monkeypatch.setattr(lw, "QFileDialog",
                        type("Files", (), {"getOpenFileName": staticmethod(
                            lambda *a, **k: (str(path), ""))}))
    dlg.open_job()
    assert dlg.job == ["G21", "G90", "M5"]
    assert "job.gcode" in dlg.job_label.text()


def test_the_job_goes_out_and_the_bar_follows(dlg, board):
    dlg.set_job(["G21", "G90", "M4 S500", "M5"], "test")
    assert dlg.b_start.isEnabled()
    dlg.b_start.click()
    assert pump(dlg, lambda: dlg.session.fraction == 1.0)
    dlg.refresh()
    assert dlg.bar.value() == 100
    assert dlg.b_start.isEnabled()


def test_a_job_that_is_over_can_be_sent_again(dlg, board):
    dlg.set_job(["G21"], "first")
    dlg.b_start.click()
    assert pump(dlg, lambda: dlg.session.fraction == 1.0)
    dlg.set_job(["M5"], "second")
    dlg.b_start.click()
    assert pump(dlg, lambda: "M5\n" in board.text())


def test_pause_and_resume_change_the_button(dlg, slow):
    dlg.set_job(["G21"] * 40, "test")
    dlg.b_start.click()
    assert pump(dlg, lambda: dlg.b_pause.isEnabled())
    dlg.b_pause.click()
    assert dlg.session.paused
    assert dlg.b_pause.text() == tr("Resume")
    dlg.b_pause.click()
    assert not dlg.session.paused
    assert dlg.b_pause.text() == tr("Pause")


def test_stop_ends_the_job_and_the_board_gets_a_reset(dlg, slow):
    dlg.set_job(["G21"] * 40, "test")
    dlg.b_start.click()
    assert pump(dlg, lambda: dlg.b_stop.isEnabled())
    dlg.b_stop.click()
    assert not dlg.session.running
    assert pump(dlg, lambda: grbl.RESET in slow.text())


def test_the_start_button_waits_for_a_cable_and_for_a_job(dlg):
    assert dlg.b_start.isEnabled() is False            # no cable and no job yet
    dlg.set_job(["G21"], "test")
    assert dlg.b_start.isEnabled() is False            # still no cable


# ------------------------------------------------------------------ what it says

def test_the_size_and_the_speed_of_the_pad_come_back(dlg, win):
    dlg.step_box.setCurrentIndex(pick(dlg.step_box, 50.0))
    dlg.feed_box.setCurrentIndex(pick(dlg.feed_box, 6000))
    again = LaserWindow(win)
    try:
        assert float(again.step_box.currentData()) == 50.0
        assert float(again.feed_box.currentData()) == 6000.0
    finally:
        again.shutdown()


def test_what_the_board_said_comes_into_the_log(dlg, board):
    board.say("[MSG:GCODE source pinned]")
    assert pump(dlg, lambda: "pinned" in dlg.log.toPlainText())


def test_a_refused_block_is_written_as_an_error(dlg, board):
    dlg.set_job(["G99 nonsense"], "test")
    dlg.b_start.click()
    assert pump(dlg, lambda: "Error:" in dlg.log.toPlainText())


def test_closing_the_window_hides_it_and_lets_the_job_run(dlg, slow):
    dlg.set_job(["G21"] * 40, "test")
    dlg.b_start.click()
    assert pump(dlg, lambda: dlg.session.running)
    dlg.close()
    assert not dlg.isVisible()
    assert dlg.connected and dlg.session.running
    dlg.shutdown()
    assert not dlg.connected and slow.closed

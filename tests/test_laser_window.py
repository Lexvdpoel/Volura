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
from PySide6.QtCore import QCoreApplication, Qt  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from volura import grbl, laser  # noqa: E402
from volura import laser_image, laser_profiles as profiles  # noqa: E402
from volura.i18n import tr  # noqa: E402
from volura.laser_window import LaserWindow, image_gray  # noqa: E402
from volura.settings import qsettings  # noqa: E402
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


class Buttons:
    """What a question can answer, without the real message box on the screen."""

    Yes = 1
    No = 2


class Box:
    """The message boxes of the window, caught instead of shown."""

    StandardButton = Buttons

    def __init__(self):
        self.messages = []
        self.answer = Buttons.Yes

    def _say(self, parent, title, text, *a, **k):
        self.messages.append(text)
        return 0

    information = warning = critical = _say

    def question(self, parent, title, text, *a, **k):
        self.messages.append(text)
        return self.answer


class AnsweringPort(FakePort):
    """A board that answers `$$` with what it is set to, like a real one does."""

    SETTINGS = ("$30=255.000 (max spindle speed)",
                "$32=0 (laser mode)",
                "$130=300.000 (x max travel)",
                "$131=180.000 (y max travel)")

    def _answer(self, data):
        if data == b"?":
            self.say(self.report)
            return
        for block in data.decode("ascii", "replace").split("\n"):
            block = block.strip()
            if block == "$$":
                for line in self.SETTINGS:
                    self.say(line)
                self.say("ok")
            elif block:
                self.say("ok")


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
    from volura.sheet_dialogs import GCODE_DYNAMIC_KEY, GCODE_S_MAX_KEY
    q = qsettings()
    for key in (laser.PORT_KEY, laser.BAUD_KEY, laser.STEP_KEY, laser.FEED_KEY,
                profiles.MACHINE_KEY, profiles.TRAVEL_X_KEY, profiles.TRAVEL_Y_KEY,
                profiles.CLAMP_KEY, profiles.PRESETS_KEY,
                laser.DOT_KEY, laser.RASTER_FEED_KEY, laser.IMAGE_MODE_KEY,
                laser.IMAGE_POWER_KEY, laser.IMAGE_PASSES_KEY,
                laser.IMAGE_INVERT_KEY, laser.IMAGE_ONEWAY_KEY,
                GCODE_S_MAX_KEY, GCODE_DYNAMIC_KEY):
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
    board.report = "<Idle|MPos:12.000,3.000,0.000|WPos:2.000,1.000,0.000|FS:800,0>"
    assert pump(dlg, lambda: "12.000" in dlg.mpos_label.text()
                and "800" in dlg.feed_label.text())
    assert "2.000" in dlg.wpos_label.text()
    assert dlg.state_label.text() == grbl.state_label("Idle")


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


def mill_sheet(win):
    """A part with a hole on the mill layers of the worksheet."""
    sheet = win.doc.sheet
    for lay in sheet.layers.values():
        lay.params.update({"tool_d": 3.0, "depth": 2.0, "step_depth": 1.0})
    sheet.add_path(rect(40.0, 30.0), closed=True, layer="Mill outside", src=1)
    sheet.add_path(rect(6.0, 6.0, 10.0, 10.0), closed=True, layer="Mill inside", src=2)
    return sheet


def test_the_cnc_panel_makes_a_milling_job_of_the_worksheet(win, monkeypatch, boxes):
    mill_sheet(win)
    monkeypatch.setattr(win, "sheet_export_options",
                        lambda kind: {"layers": ["Mill outside", "Mill inside"],
                                      "compensate": True})
    panel = LaserWindow(win, "cnc")
    try:
        assert not panel.b_load.isHidden(), "a mill takes its job from the worksheet too"
        panel.load_job()
        assert panel.job[0] == "G21"
        assert "M3 S10000" in panel.job, "the spindle of the layer"
        assert any(line.startswith("G1 Z-") for line in panel.job), "Z = 0 is the top"
        assert panel.job[-1] == "G0 X0 Y0"
        assert "blocks" in panel.job_label.text()
    finally:
        panel.shutdown()


def test_a_worksheet_that_is_not_for_the_mill_is_a_message_and_no_job(win, monkeypatch, boxes):
    worksheet(win)
    monkeypatch.setattr(win, "sheet_export_options",
                        lambda kind: {"layers": ["Cut"], "compensate": False})
    panel = LaserWindow(win, "cnc")
    try:
        panel.load_job()
        assert not panel.job and boxes.messages
    finally:
        panel.shutdown()


def test_the_laser_panel_still_burns_instead_of_milling(dlg, win, monkeypatch):
    worksheet(win)
    monkeypatch.setattr(win, "sheet_export_options",
                        lambda kind: {"layers": ["Cut"], "compensate": False,
                                      "s_max": 1000, "dynamic": True})
    dlg.load_job()
    assert "M4" in dlg.job and "M3 S10000" not in dlg.job
    assert not any(line.startswith("G1 Z-") for line in dlg.job)


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


def test_another_tab_hides_the_controller_and_lets_the_job_run(dlg, slow):
    dlg.set_job(["G21"] * 40, "test")
    dlg.b_start.click()
    assert pump(dlg, lambda: dlg.session.running)
    win = dlg.win
    win._laser_controller = dlg
    win.set_workspace("laser")
    assert win.machine_stack.currentWidget() is dlg
    win.set_workspace("design")
    assert win.machine_stack.isHidden()
    assert dlg.connected and dlg.session.running
    win._laser_controller = None
    dlg.shutdown()
    assert not dlg.connected and slow.closed


# ------------------------------------------------------------------ the machine it is
def test_a_preset_puts_its_work_area_on_the_screen(dlg):
    dlg.machine_box.setCurrentIndex(pick(dlg.machine_box, "s9"))
    assert dlg.travel_x.text() == "410" and dlg.travel_y.text() == "420"
    assert "410 × 420 mm" in dlg.machine_hint.text()
    assert "S up to 1000" in dlg.machine_hint.text()


def test_a_preset_moves_the_s_word_of_the_g_code(dlg, machine_choices):
    from volura.sheet_dialogs import GCODE_DYNAMIC_KEY, GCODE_S_MAX_KEY
    machine_choices.setValue(GCODE_S_MAX_KEY, 255)
    machine_choices.setValue(GCODE_DYNAMIC_KEY, False)
    dlg.machine_box.setCurrentIndex(pick(dlg.machine_box, "grbl8"))
    assert int(machine_choices.value(GCODE_S_MAX_KEY)) == 1000
    assert str(machine_choices.value(GCODE_DYNAMIC_KEY)).lower() in ("true", "1")


def test_the_machine_of_the_last_time_comes_back(dlg):
    dlg.machine_box.setCurrentIndex(pick(dlg.machine_box, "lm3"))
    other = LaserWindow(dlg.win)
    other.show()
    QCoreApplication.processEvents()
    try:
        assert other.machine_box.currentData() == "lm3"
        assert other.travel_x.text() == "400" and other.travel_y.text() == "380"
    finally:
        other.shutdown()


def test_the_work_area_someone_measured_stays_put(dlg, machine_choices):
    dlg.machine_box.setCurrentIndex(pick(dlg.machine_box, "lm2pro"))
    dlg.travel_x.setText("395")
    dlg.travel_x.editingFinished.emit()
    assert machine_choices.value(profiles.TRAVEL_X_KEY) == "395"
    assert dlg.travel_size() == (395.0, 400.0)


def test_a_jog_past_the_end_of_the_machine_is_refused(dlg, board):
    dlg.clamp_check.setChecked(True)
    dlg.travel_x.setText("20")
    dlg.travel_y.setText("20")
    dlg.step_box.setCurrentIndex(pick(dlg.step_box, 50.0))
    assert pump(dlg, lambda: dlg.session.mpos is not None)
    dlg.pad_buttons["→"].click()
    QCoreApplication.processEvents()
    assert "past the end" in dlg.log.toPlainText()
    assert b"$J=G21 G91 X50.000" not in b"".join(board.written)


def test_the_same_jog_is_fine_when_staying_on_the_machine_is_off(dlg, board):
    dlg.travel_x.setText("20")
    dlg.travel_y.setText("20")
    dlg.clamp_check.setChecked(False)
    dlg.step_box.setCurrentIndex(pick(dlg.step_box, 50.0))
    assert pump(dlg, lambda: dlg.session.mpos is not None)
    dlg.pad_buttons["→"].click()
    assert pump(dlg, lambda: b"$J=G21 G91 X50.000" in b"".join(board.written))


def test_a_machine_of_which_the_size_is_unknown_jogs_anywhere(dlg, board):
    dlg.travel_x.setText("")
    dlg.travel_y.setText("")
    dlg.step_box.setCurrentIndex(pick(dlg.step_box, 50.0))
    assert pump(dlg, lambda: dlg.session.mpos is not None)
    dlg.pad_buttons["→"].click()
    assert pump(dlg, lambda: b"$J=G21 G91 X50.000" in b"".join(board.written))


def test_the_board_says_what_it_is_and_the_window_believes_it(dlg, monkeypatch):
    connect(dlg, monkeypatch, AnsweringPort)
    assert pump(dlg, lambda: dlg.b_take.isEnabled())
    dlg.b_take.click()
    assert dlg.travel_x.text() == "300" and dlg.travel_y.text() == "180"
    assert dlg.machine_box.currentData() == "grbl32"      # S 255: not a machine of the list
    assert "S up to 255" in dlg.log.toPlainText()
    assert "not in the list" in dlg.log.toPlainText()
    assert "laser mode" in dlg.machine_hint.text()


def test_a_board_that_has_not_answered_cannot_be_asked(dlg, boxes):
    assert not dlg.b_take.isEnabled()
    dlg.take_from_board()                       # the information box, not a crash
    assert "has not answered" in boxes.messages[-1]


def test_the_settings_go_to_the_board_after_asking(dlg, monkeypatch, boxes):
    port = connect(dlg, monkeypatch)
    dlg.machine_box.setCurrentIndex(pick(dlg.machine_box, "s9"))
    dlg.b_apply.click()
    assert "Set the board to this?" in boxes.messages[-1]
    assert pump(dlg, lambda: "$32=1" in port.text())
    text = port.text()
    assert "$30=1000" in text and "$110=" not in text     # no top speed published
    assert "$130=410" in text and "$131=420" in text


def test_the_user_can_say_no(dlg, monkeypatch, boxes):
    boxes.answer = Buttons.No
    port = connect(dlg, monkeypatch)
    dlg.b_apply.click()
    assert "$32=1" not in port.text()


def test_a_board_that_is_cutting_takes_no_new_settings(dlg, slow, boxes):
    dlg.set_job(["G1 X10.000 Y10.000 F1000"] * 8, "test")
    dlg.start_job()
    assert pump(dlg, lambda: dlg.session.running)
    dlg.b_apply.click()
    assert "A job is running" in boxes.messages[-1]
    assert "$32=1" not in slow.text()


def test_the_machine_buttons_wait_for_a_cable(dlg):
    assert not dlg.b_apply.isEnabled()
    assert not dlg.b_take.isEnabled()


# ------------------------------------------------------------------ the presets of the user
def open_presets(dlg):
    """The preset window of `dlg`, on screen."""
    from volura.laser_window import PresetsDialog
    d = PresetsDialog(dlg)
    d.show()
    QCoreApplication.processEvents()
    return d


def row_of(dialog, number):
    """The row of the table that holds the setting with this number."""
    for row in range(dialog.table.rowCount()):
        if dialog.table.item(row, 1).data(Qt.UserRole) == number:
            return row
    return -1


def sending(dialog, number, on):
    """Tick or untick the setting of a row."""
    dialog.table.item(row_of(dialog, number), 0).setCheckState(
        Qt.Checked if on else Qt.Unchecked)


def is_sending(dialog, number) -> bool:
    return dialog.table.item(row_of(dialog, number), 0).checkState() == Qt.Checked


def test_the_preset_window_shows_what_the_board_answered(dlg, monkeypatch):
    connect(dlg, monkeypatch, AnsweringPort)
    assert pump(dlg, lambda: dlg.b_take.isEnabled())
    d = open_presets(dlg)
    try:
        assert row_of(d, 30) >= 0 and row_of(d, 130) >= 0
        assert d.table.item(row_of(d, 30), 2).text() == "255"
    finally:
        d.done(0)


def test_a_preset_keeps_everything_the_board_answered(dlg, monkeypatch):
    connect(dlg, monkeypatch, AnsweringPort)
    assert pump(dlg, lambda: dlg.b_take.isEnabled())
    dlg.take_from_board()          # the work area of the board onto the screen
    d = open_presets(dlg)
    try:
        d.name_edit.setText("Atelier")
        d.b_save.click()
        preset = profiles.find_preset("Atelier")
        assert set(preset["settings"]) == {"30", "32", "130", "131"}
        assert preset["x"] == 300.0 and preset["y"] == 180.0
        assert profiles.user_key("Atelier") in [k for k, _ in profiles.choices()]
    finally:
        d.done(0)


def test_a_preset_without_a_name_is_not_saved(dlg, boxes):
    d = open_presets(dlg)
    try:
        d.b_save.click()
        assert "A name first." in boxes.messages[-1]
        assert profiles.presets() == []
    finally:
        d.done(0)


def test_a_saved_preset_is_a_machine_one_can_choose(dlg, monkeypatch, machine_choices):
    from volura.sheet_dialogs import GCODE_S_MAX_KEY
    connect(dlg, monkeypatch, AnsweringPort)
    assert pump(dlg, lambda: dlg.b_take.isEnabled())
    profiles.add_preset(profiles.new_preset("Atelier", dlg.session.settings, 300, 180))
    dlg.refresh_machines()
    dlg.machine_box.setCurrentIndex(pick(dlg.machine_box, profiles.user_key("Atelier")))
    assert dlg.travel_x.text() == "300" and dlg.travel_y.text() == "180"
    assert int(machine_choices.value(GCODE_S_MAX_KEY)) == 255


def test_the_calibration_of_a_preset_stays_in_the_preset(dlg, monkeypatch, boxes):
    port = connect(dlg, monkeypatch, AnsweringPort)
    assert pump(dlg, lambda: dlg.b_take.isEnabled())
    profiles.add_preset(profiles.new_preset(
        "Atelier", {"30": 255, "100": 250, "130": 300}, 300, 180))
    dlg.refresh_machines()
    d = open_presets(dlg)
    try:
        assert not is_sending(d, 100) and is_sending(d, 30)
        d.b_apply.click()
        assert pump(dlg, lambda: "$30=255" in port.text())
        assert "$100=250" not in port.text()
        sending(d, 100, True)
        d.b_apply.click()
        assert pump(dlg, lambda: "$100=250" in port.text())
    finally:
        d.done(0)


def test_forgetting_a_preset_asks_first(dlg, boxes):
    profiles.add_preset(profiles.new_preset("Atelier", {"30": 255}, 300, 200))
    d = open_presets(dlg)
    try:
        boxes.answer = Buttons.No
        d.b_delete.click()
        assert profiles.find_preset("Atelier") is not None
        boxes.answer = Buttons.Yes
        d.b_delete.click()
        assert profiles.find_preset("Atelier") is None
        assert not d.b_delete.isEnabled()
    finally:
        d.done(0)


def test_presets_go_to_a_file_and_come_back(dlg, monkeypatch, tmp_path):
    from volura import laser_window as lw
    profiles.add_preset(profiles.new_preset("Atelier", {"30": 255}, 300, 200))
    path = str(tmp_path / "presets.json")
    monkeypatch.setattr(lw.QFileDialog, "getSaveFileName",
                        staticmethod(lambda *a, **k: (path, "")))
    d = open_presets(dlg)
    d.export_presets()
    d.done(0)
    profiles.remove_preset("Atelier")
    assert profiles.find_preset("Atelier") is None
    monkeypatch.setattr(lw.QFileDialog, "getOpenFileName",
                        staticmethod(lambda *a, **k: (path, "")))
    d = open_presets(dlg)
    d.import_presets()
    d.done(0)
    assert profiles.find_preset("Atelier")["y"] == 200.0


# ------------------------------------------------------------------ a picture to engrave
def picture(path, width=6, height=4, dark=(0, 0, 4, 4)):
    """A PNG on disk: a dark rectangle on the left, white paper on the right."""
    from PySide6.QtGui import QColor, QImage, QPainter
    image = QImage(width, height, QImage.Format_RGB32)
    image.fill(QColor(255, 255, 255))
    painter = QPainter(image)
    painter.fillRect(dark[0], dark[1], dark[2], dark[3], QColor(0, 0, 0))
    painter.end()
    assert image.save(str(path))
    return str(path)


def set_mode(dlg, mode):
    dlg.mode_box.setCurrentIndex(pick(dlg.mode_box, mode))
    QCoreApplication.processEvents()


def test_the_picture_box_offers_the_modes_of_lightburn(dlg):
    assert [dlg.mode_box.itemData(n) for n in range(dlg.mode_box.count())] == \
        ["gray", "threshold", "halftone", "dither"]
    assert [dlg.style_box.itemData(n) for n in range(dlg.style_box.count())] == \
        list(laser_image.DITHERS)


def test_the_matrix_of_the_box_is_the_one_of_the_mode(dlg):
    set_mode(dlg, "halftone")
    assert [dlg.style_box.itemData(n) for n in range(dlg.style_box.count())] == \
        list(laser_image.HALFTONES)
    set_mode(dlg, "gray")
    assert dlg.style_box.count() == 0 and not dlg.style_box.isEnabled()


def test_a_picture_on_disk_becomes_grey_values(dlg, tmp_path):
    path = picture(tmp_path / "p.png")
    gray = image_gray(path)
    assert gray.shape == (4, 6)
    assert gray[0, 0] == pytest.approx(1.0) and gray[0, 5] == pytest.approx(0.0)  # dark burns


def test_a_file_that_is_not_a_picture_says_so(dlg, tmp_path):
    path = tmp_path / "geen.png"
    path.write_text("this is not a picture")
    with pytest.raises(ValueError):
        image_gray(str(path))


def test_a_picture_is_read_through_the_dialog_and_shows_what_it_costs(dlg, monkeypatch,
                                                                     tmp_path):
    from volura import laser_window as lw
    path = picture(tmp_path / "p.png")
    monkeypatch.setattr(lw.QFileDialog, "getOpenFileName",
                        staticmethod(lambda *a, **k: (path, "")))
    dlg.dot_box.setCurrentIndex(pick(dlg.dot_box, 1.0))
    dlg.width_spin.setValue(6.0)
    dlg.b_image.click()
    assert dlg.image is not None and dlg.image_name == "p.png"
    assert "6 × 4 mm" in dlg.image_hint.text()


def test_without_a_picture_the_window_says_so(dlg, boxes):
    assert dlg.image is None
    dlg.image_job()
    assert "Choose a picture first." in boxes.messages[-1]


def test_engraving_a_picture_makes_a_job_of_the_raster(dlg, tmp_path):
    dlg.image = image_gray(picture(tmp_path / "p.png"))
    dlg.image_name = "p.png"
    set_mode(dlg, "threshold")
    dlg.dot_box.setCurrentIndex(pick(dlg.dot_box, 1.0))
    dlg.width_spin.setValue(6.0)
    dlg.power_spin.setValue(50)
    dlg.b_image_job.click()
    text = "\n".join(dlg.job)
    assert "G21" in text and "M3" in text and "S500" in text
    assert dlg.job_name == "p.png"
    assert "the picture is 6 × 4 mm" in dlg.log.toPlainText()


def test_the_darkest_pixel_of_the_picture_gets_the_chosen_power(dlg, tmp_path):
    dlg.image = np.ones((2, 2))
    dlg.dot_box.setCurrentIndex(pick(dlg.dot_box, 1.0))
    dlg.power_spin.setValue(25)
    dlg.b_image_job.click()
    assert "S250" in "\n".join(dlg.job)          # 25 % of a board that counts to 1000


def test_inverting_the_picture_burns_the_other_half(dlg, tmp_path):
    dlg.image = image_gray(picture(tmp_path / "p.png"))
    set_mode(dlg, "threshold")
    dlg.dot_box.setCurrentIndex(pick(dlg.dot_box, 1.0))
    before = dlg.image_dots()
    dlg.invert_check.setChecked(True)
    assert np.array_equal(before, 1.0 - dlg.image_dots())


def test_the_board_of_the_machine_decides_the_highest_S_word(dlg, monkeypatch, tmp_path):
    from volura.sheet_dialogs import GCODE_S_MAX_KEY
    qsettings().setValue(GCODE_S_MAX_KEY, 255)
    dlg.image = np.ones((2, 2))
    dlg.dot_box.setCurrentIndex(pick(dlg.dot_box, 1.0))
    dlg.power_spin.setValue(100)
    dlg.b_image_job.click()
    assert "S255" in "\n".join(dlg.job)


def test_the_number_of_passes_is_the_number_of_times_over(dlg, tmp_path):
    dlg.image = np.ones((2, 2))
    dlg.dot_box.setCurrentIndex(pick(dlg.dot_box, 1.0))
    dlg.width_spin.setValue(2.0)
    dlg.passes_spin.setValue(3)
    dlg.b_image_job.click()
    burns = [line for line in dlg.job if line.startswith("G1")]
    assert len(burns) == 2 * 3                    # two rows of one stretch, three times


# ------------------------------------------------------------------ the tabs under the viewer

def test_the_tabs_under_the_viewer_are_design_laser_and_cnc(win):
    bar = win.workspace_bar
    keys = [bar.tabs.tabData(i) for i in range(bar.tabs.count())]
    assert keys == ["design", "print", "laser", "cnc"]
    assert win.workspace == "design" and win.machine_stack.isHidden()


def test_the_laser_tab_is_only_2d_and_shows_the_laser(win):
    win.set_mode("3d")
    win.set_workspace("laser")
    try:
        assert win.mode == "2d"
        assert win.mode_button.isHidden(), "Laser has no 3D | 2D switch"
        panel = win.machine_stack.currentWidget()
        assert isinstance(panel, LaserWindow) and panel.kind == "laser"
        win.set_mode("3d")                     # the model is not in Laser: back to Design
        assert win.workspace == "design" and win.mode == "3d"
        assert win.machine_stack.isHidden() and not win.mode_button.isHidden()
    finally:
        win._laser_controller.shutdown()


def test_the_cnc_tab_keeps_both_views_and_a_mill_of_its_own(win):
    win.set_workspace("laser")
    win.set_workspace("cnc")
    try:
        assert not win.mode_button.isHidden()
        win.set_mode("3d")
        assert win.workspace == "cnc" and win.mode == "3d"
        panel = win.machine_stack.currentWidget()
        assert panel.kind == "cnc" and panel is not win._laser_controller
        assert not panel.b_load.isHidden() and not panel.machine_box.isHidden()
        keys = [panel.machine_box.itemData(i) for i in range(panel.machine_box.count())]
        assert "lead1010" in keys and "s9" not in keys and "custom" not in keys
        assert "3018pro" in keys and panel.machine.kind == "cnc"
        assert panel._k("laser/port") == "cnc/port"
        assert "CNC" in panel.windowTitle()
    finally:
        win._laser_controller.shutdown()
        win._cnc_controller.shutdown()


def test_the_laser_controller_command_goes_to_the_laser_tab(win):
    panel = win.laser_controller()
    try:
        assert win.workspace == "laser"
        assert win.workspace_bar.key() == "laser"
        assert win.machine_stack.currentWidget() is panel
    finally:
        panel.shutdown()


def test_the_print_tab_is_only_3d_with_the_printer_above_it(win):
    win.set_mode("2d")
    win.set_workspace("print")
    assert win.mode == "3d" and win.mode_button.isHidden()
    assert win.machine_stack.isHidden()
    bar = win.tb_printing
    assert not bar.isHidden()
    for a in (win.a_slice, win.a_slice_resin, win.a_check, win.a_print_setup):
        assert a in bar.actions()
    win.set_mode("2d")                         # the worksheet is not in Print: to Design
    assert win.workspace == "design" and win.mode == "2d" and bar.isHidden()


def test_the_laser_controller_is_no_button_on_the_side_any_more(win):
    assert win.a_laser_controller not in win.tb_laser.actions()


def _shown_panels(win):
    return {t for t in list(win.tray.panel_titles) + list(win.tray.extra_titles)
            if win.tray.is_shown(t)}


def test_the_tray_only_shows_what_belongs_to_the_tab(win):
    from volura.app import (PANEL_COLORS, PANEL_LAYERS, PANEL_MATERIALS, PANEL_OUTLINER,
                            PANEL_PRINT_CHECK)
    from volura.i18n import tr
    try:
        win.set_workspace("design")
        win.set_mode("3d")
        assert tr(PANEL_PRINT_CHECK) not in _shown_panels(win)
        assert tr(PANEL_MATERIALS) not in _shown_panels(win)     # filaments: only in Print
        assert tr(PANEL_COLORS) in _shown_panels(win)
        win.set_workspace("print")
        assert tr(PANEL_PRINT_CHECK) in _shown_panels(win)
        assert tr(PANEL_MATERIALS) in _shown_panels(win)
        assert tr(PANEL_COLORS) not in _shown_panels(win)
        assert tr(PANEL_OUTLINER) not in _shown_panels(win)
        win.set_workspace("cnc")
        assert tr(PANEL_MATERIALS) not in _shown_panels(win)
        assert tr(PANEL_PRINT_CHECK) not in _shown_panels(win)
        win.set_workspace("laser")
        assert tr(PANEL_LAYERS) in _shown_panels(win)
        assert tr(PANEL_MATERIALS) not in _shown_panels(win)
    finally:
        for n in ("_laser_controller", "_cnc_controller"):
            if getattr(win, n, None) is not None:
                getattr(win, n).shutdown()


def test_the_layers_of_the_laser_are_not_in_the_cnc_tab(win):
    panel = win.layers_panel
    try:
        win.set_workspace("cnc")
        win.set_mode("2d")
        ops = [win.doc.sheet.layers[n].op for row, n in enumerate(panel.names)
               if not panel.table.isRowHidden(row)]
        assert ops and all(op not in ("cut", "engrave", "raster") for op in ops)
        assert win.doc.sheet.layers[win.sheet_view.current_layer].op not in ("cut", "raster")
        win.set_workspace("laser")
        ops = [win.doc.sheet.layers[n].op for row, n in enumerate(panel.names)
               if not panel.table.isRowHidden(row)]
        assert ops and all(op not in ("mill_outside", "mill_inside", "pocket") for op in ops)
        win.set_workspace("design")
        assert not any(panel.table.isRowHidden(r) for r in range(len(panel.names)))
    finally:
        win._laser_controller.shutdown()
        win._cnc_controller.shutdown()


def test_the_left_column_and_the_menu_follow_the_tab(win):
    try:
        win.set_workspace("laser")
        assert win.a_fit_test in win.tb_laser.actions()
        assert win.a_dogbone_corners not in win.tb_laser.actions()
        assert not win.a_dogbone_corners.isVisible()
        win.set_workspace("cnc")
        win.set_mode("2d")
        assert win.a_dogbone_corners in win.tb_laser.actions()
        assert win.a_fit_test not in win.tb_laser.actions()
        assert win.a_job_estimate not in win.tb_laser.actions()
        assert not win.a_fit_test.isVisible()
        win.set_mode("3d")
        assert not win.print_tools_menu.menuAction().isVisible()
        win.set_workspace("design")
        assert win.print_tools_menu.menuAction().isVisible()
        assert win.a_fit_test.isVisible() and win.a_dogbone_corners.isVisible()
    finally:
        win._laser_controller.shutdown()
        win._cnc_controller.shutdown()


def test_a_color_of_the_design_tab_paints_and_becomes_a_filament(win):
    panel = win.colors_panel
    colors = [panel.list.item(i).data(Qt.UserRole) for i in range(panel.list.count())]
    assert len(colors) >= 24 and len({c.lower() for c in colors}) == len(colors)
    new = next(i for i in range(panel.list.count())
               if not any(c.lower() == panel.list.item(i).data(Qt.UserRole).lower()
                          for _n, c in win.doc.materials))
    color = panel.list.item(new).data(Qt.UserRole)
    panel.pick(panel.list.item(new))
    assert win.settings.current_color == color
    assert win.view.tool.name == "paint"
    assert any(c.lower() == color.lower() for _n, c in win.doc.materials)


def test_a_mill_has_a_height_and_a_laser_has_none(win):
    win.set_workspace("cnc")
    try:
        panel = win._cnc_controller
        panel.machine_box.setCurrentIndex(pick(panel.machine_box, "lead1010"))
        assert not panel.travel_z.isHidden() and panel.travel_z.text() == "100"
        assert panel.travel_height() == 100.0
        assert "730 × 810 × 100 mm" in panel.machine_hint.text()
        win.set_workspace("laser")
        assert win._laser_controller.travel_z.isHidden()
        assert win._laser_controller.travel_height() == 0.0
    finally:
        win._laser_controller.shutdown()
        win._cnc_controller.shutdown()

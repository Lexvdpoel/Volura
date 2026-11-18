"""GRBL over the cable: the blocks that go to the board and the lines that come back.

`volura/grbl.py` is the protocol without Qt and without a machine: the blocks a job is
made of (`job_lines`), what a line of the board means (`classify`, `parse_report`), the
blocks a controller window sends (`jog`, `move`, `zero_here`, `beam`, ...) and the
character-counting sender that keeps a job from outrunning the input buffer of the board
(`Sender`). Everything here is plain text, so it can be checked without a laser hanging
on the other end of the wire.

Run with:  python -m pytest tests/test_grbl.py
"""
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest  # noqa: E402

from volura import grbl  # noqa: E402


# ------------------------------------------------------------------ the blocks of a job

def test_comments_and_empty_lines_do_not_go_to_the_board():
    text = "; a comment\n\nG21\n\n; another one\nG90\n"
    assert grbl.job_lines(text) == ["G21", "G90"]


def test_the_blocks_come_back_in_the_order_of_the_job():
    text = "G21\nG90\nM4\nG0 X10.000 Y5.000\nG1 X20.000 F600 S400\nM5\n"
    assert grbl.job_lines(text) == ["G21", "G90", "M4", "G0 X10.000 Y5.000",
                                    "G1 X20.000 F600 S400", "M5"]


def test_an_inline_comment_is_cut_but_the_block_stays():
    assert grbl.job_lines("G1 X10.000 F600 ; the first cut\n") == ["G1 X10.000 F600"]


def test_a_settings_command_is_a_block_like_any_other():
    assert grbl.job_lines("$X\n$110=3000 ; rapid\n$$\n") == ["$X", "$110=3000", "$$"]


def test_windows_line_endings_give_the_same_blocks():
    assert grbl.job_lines("G21\r\nG90\r\n") == ["G21", "G90"]


def test_a_job_of_nothing_is_no_blocks():
    assert grbl.job_lines("; nothing but a comment\n\n") == []


# ------------------------------------------------------------------ what comes back

@pytest.mark.parametrize("line, kind, rest", (
    ("ok", "ok", ""),
    ("  ok  ", "ok", ""),
    ("error:3", "error", "3"),
    ("error:20", "error", "20"),
    ("echo:G21", "echo", "G21"),
    ("[MSG:GCODE source pinned]", "message", "GCODE source pinned"),
    ("Alarm:8", "alarm", "Alarm:8"),
    ("<Idle|MPos:0.000,0.000,0.000>", "status", "<Idle|MPos:0.000,0.000,0.000>"),
    ("something the board makes up", "text", "something the board makes up"),
    ("", "empty", ""),
))
def test_a_line_of_the_board_is_known(line, kind, rest):
    assert grbl.classify(line) == (kind, rest)


def test_a_known_error_number_is_explained():
    assert grbl.error_text("3") == grbl.ERRORS["3"]
    assert grbl.error_text(" 3 ") == grbl.ERRORS["3"]


def test_an_unknown_error_number_still_says_which_number_it_was():
    text = grbl.error_text("99")
    assert "99" in text


def test_a_full_status_report_gives_state_position_and_speeds():
    report = grbl.parse_report("<Idle|MPos:12.000,3.000,0.000|FS:1200,0|Ov:100,50,100>")
    assert report["state"] == "Idle"
    assert report["mpos"] == (12.0, 3.0, 0.0)
    assert report["feed"] == 1200.0
    assert report["speed"] == 0.0
    assert (report["feed_override"], report["rapid_override"],
            report["power_override"]) == (100.0, 50.0, 100.0)


def test_a_running_board_reports_the_block_it_is_at():
    report = grbl.parse_report("<Run:34|MPos:5.000,5.000,0.000|FS:800,800>")
    assert report["state"] == "Run"
    assert report["line"] == 34


def test_check_mode_reports_the_state_behind_its_name():
    """`<Check:Idle|…>` is idle while the board only plans: the state is what follows."""
    assert grbl.parse_report("<Check:Idle|MPos:0.000,0.000,0.000>")["state"] == "Idle"


def test_the_work_position_and_the_offset_come_out_too():
    report = grbl.parse_report("<Run|MPos:20.000,20.000,0.000|WPos:10.000,10.000,0.000"
                               "|WCO:10.000,10.000,0.000|Coor:G54.1|PnX:DEH>")
    assert report["wpos"] == (10.0, 10.0, 0.0)
    assert report["wco"] == (10.0, 10.0, 0.0)
    assert report["coord"] == "G54.1"
    assert report["pinx"] == "DEH"


def test_a_board_that_sends_less_still_gives_a_position():
    """An older GRBL knows no Ov and no Coor; the fields it leaves out are simply absent."""
    report = grbl.parse_report("<Idle|WPos:1.500,2.500,0.000>")
    assert report["state"] == "Idle"
    assert report["wpos"] == (1.5, 2.5, 0.0)
    assert "feed" not in report and "power_override" not in report and "coord" not in report


def test_a_field_the_program_does_not_know_is_kept_rather_than_lost():
    report = grbl.parse_report("<Idle|Bf:0,127|MPos:0.000,0.000,0.000>")
    assert report["other"]["bf"] == "0,127"


@pytest.mark.parametrize("line", ("ok", "error:3", "<Idle|MPos:0.000,0.000",
                                  "<Idle|MPos:1.000,1.000,0.000"))
def test_a_line_that_is_not_a_report_gives_nothing_at_all(line):
    """A half report is cut off in the middle of a position, so it says nothing."""
    assert grbl.parse_report(line) == {}


def test_a_position_the_board_cannot_be_read_from_is_no_position():
    assert grbl.parse_report("<Idle|MPos:unknown>")["mpos"] is None
    assert grbl.parse_report("<Idle|MPos:1.000,abc,0.000>")["mpos"] is None


def test_the_state_of_the_board_is_said_the_way_the_user_reads_it():
    assert grbl.state_label("Run") == grbl.STATE_LABELS["Run"]
    assert grbl.state_label("Idle") == grbl.STATE_LABELS["Idle"]


def test_a_state_nobody_has_seen_before_is_passed_on_as_the_board_says_it():
    assert grbl.state_label("Quantum") == "Quantum"


# ------------------------------------------------------------------ the blocks of a controller

def test_a_jog_is_a_relative_move_in_jog_mode():
    assert grbl.jog(dx=10.0) == "$J=G21 G91 X10.000 F1000"


def test_a_jog_names_only_the_axes_that_move():
    assert grbl.jog(dy=-2.5, dz=1.0, feed=2000) == "$J=G21 G91 Y-2.500 Z1.000 F2000"


def test_a_jog_of_nothing_is_no_block_at_all():
    """A click next to the pad may not send an empty `$J=` the board refuses."""
    assert grbl.jog() == ""
    assert grbl.jog(dx=0.0, dy=0.0, dz=0.0) == ""


def test_a_rapid_move_is_absolute_and_names_only_the_axes_it_is_given():
    assert grbl.move(x=30.0, y=15.0) == "G90 G0 X30.000 Y15.000"
    assert grbl.move(z=5.0) == "G90 G0 Z5.000"
    assert grbl.move() == ""


def test_a_move_with_the_beam_on_is_a_fed_move():
    assert grbl.move(x=30.0, feed=500) == "G90 G0 X30.000 F500"


def test_zeroing_here_moves_the_offset_and_leaves_the_other_axes_alone():
    assert grbl.zero_here() == "G10 L20 P1 X0 Y0"
    assert grbl.zero_here(x=True, y=False) == "G10 L20 P1 X0"
    assert grbl.zero_here(x=False, y=False, z=True) == "G10 L20 P1 Z0"
    assert grbl.zero_here(x=False, y=False) == ""


def test_zeroing_at_a_point_of_the_machine_names_the_offset_itself():
    assert grbl.zero_at(x=10.0, y=20.0) == "G10 L2 P1 X10.000 Y20.000"
    assert grbl.zero_at(x=10.0, y=20.0, z=1.5) == "G10 L2 P1 X10.000 Y20.000 Z1.500"


@pytest.mark.parametrize("power, s", ((100, 1000), (50, 500), (12.5, 125), (0, 0)))
def test_the_power_is_a_percentage_of_what_the_board_allows(power, s):
    assert grbl.beam(power, s_max=1000) == f"M4 S{s}"


def test_the_power_stays_between_silence_and_the_highest_the_board_gives():
    assert grbl.beam(150, s_max=1000) == "M4 S1000"
    assert grbl.beam(-20, s_max=1000) == "M4 S0"


def test_an_old_board_with_a_smaller_maximum_is_scaled_the_same_way():
    assert grbl.beam(25, s_max=255) == "M4 S64"


def test_the_beam_follows_the_feed_with_M4_and_holds_with_M3():
    assert grbl.beam(80, dynamic=True).startswith("M4")
    assert grbl.beam(80, dynamic=False).startswith("M3")
    assert grbl.BEAM_OFF == "M5"


def test_the_board_has_words_for_unlocking_homing_and_checking():
    assert (grbl.UNLOCK, grbl.HOME, grbl.SETTINGS, grbl.CHECK) == ("$X", "$H", "$$", "$C")


def test_the_real_time_commands_are_single_bytes_without_a_newline():
    for cmd in (grbl.STATUS, grbl.FEED_HOLD, grbl.CYCLE_START, grbl.RESET):
        assert len(cmd) == 1


@pytest.mark.parametrize("kind, action, byte", (
    ("feed", "up10", b"\x91"),
    ("feed", "down10", b"\x92"),
    ("feed", "set", b"\x90"),
    ("rapid", "half", b"\x96"),
    ("power", "up10", b"\x9a"),
    ("power", "down10", b"\x9b"),
    ("power", "up1", b"\x9c"),
    ("power", "down1", b"\x9d"),
    ("power", "set", b"\x99"),
))
def test_an_override_is_the_byte_the_board_reads(kind, action, byte):
    assert grbl.override(kind, action) == byte


def test_an_override_that_does_not_exist_is_nothing_to_write():
    assert grbl.override("feed", "up1000") == b""
    assert grbl.override("spindle", "set") == b""


# ------------------------------------------------------------------ sending without losing

def test_the_sender_hands_out_no_more_than_the_buffer_holds():
    lines = [f"G1 X{i}.000 F600" for i in range(10)]        # 14 bytes + newline each
    sender = grbl.Sender(lines, buffer=30)
    block = sender.take()
    assert block.count(b"\n") == 2                          # 2 x 15 = 30 fits, 3 x 15 does not
    assert sender.in_flight == 30
    assert sender.take() == b""                             # the board has to answer first


def test_the_sender_leaves_room_for_what_else_the_board_holds():
    lines = [f"G1 X{i}.000 F600" for i in range(10)]        # 15 bytes each
    sender = grbl.Sender(lines, buffer=30)
    assert sender.take(extra=10).count(b"\n") == 1          # 10 + 15 fits, 10 + 30 not
    assert sender.take(extra=10) == b""
    full = grbl.Sender(lines, buffer=30)
    assert full.take(extra=25) == b""                       # no room at all: wait


def test_a_block_longer_than_the_whole_buffer_still_goes_on_its_own():
    """Nothing may follow a block that is too big anyway, so it goes out alone."""
    long = "G1 " + "X1.000 " * 60
    sender = grbl.Sender([long, "M5"], buffer=20)
    assert sender.take() == (long + "\n").encode()
    assert sender.take() == b""


def test_every_answer_frees_room_for_the_next_block():
    sender = grbl.Sender(["G21", "G90", "M4"], buffer=8)    # 4 + 4 + 3 bytes
    assert sender.take() == b"G21\nG90\n"
    sender.confirm()
    assert sender.take() == b"M4\n"
    assert sender.index == 3


def test_the_blocks_go_out_as_ASCII_with_a_newline_behind_each_of_them():
    sender = grbl.Sender(["G0 X1.000", "M5"])
    assert sender.take() == b"G0 X1.000\nM5\n"


def test_a_character_the_board_cannot_read_is_replaced_rather_than_lost():
    """GRBL speaks ASCII; a stray accent in a comment must not stop the job."""
    sender = grbl.Sender(["G1 X10.000 café"])
    assert sender.take() == b"G1 X10.000 caf?\n"


def test_pause_stops_new_blocks_and_resume_carries_on_where_it_stopped():
    sender = grbl.Sender(["G21", "G90", "M4"], buffer=128)
    assert sender.take() == b"G21\nG90\nM4\n"
    sender.pause()
    sender.confirm()
    sender.confirm()
    sender.confirm()
    assert sender.take() == b""
    assert sender.done
    sender.resume()
    assert sender.take() == b""                             # nothing was left to send


def test_a_paused_job_keeps_what_it_has_not_sent_yet():
    sender = grbl.Sender(["G21", "G90", "M4"], buffer=6)
    assert sender.take() == b"G21\n"
    sender.pause()
    assert sender.take() == b""
    assert sender.index == 1 and not sender.done
    sender.confirm()
    sender.resume()
    assert sender.take() == b"G90\n"


def test_cancel_ends_the_job_without_sending_the_rest():
    sender = grbl.Sender(["G21", "G90", "M4"], buffer=128)
    sender.cancel()
    assert sender.take() == b""
    assert sender.done


def test_a_refused_block_is_remembered_with_its_number():
    sender = grbl.Sender(["G21", "G99 nonsense", "M5"])
    sender.take()
    sender.confirm(ok=True)
    sender.confirm(ok=False)
    sender.error("1")
    assert sender.errors == [(2, "1")]
    assert sender.acked == 2


def test_an_answer_without_anything_waiting_costs_nothing():
    """A board that echoes more than it was told may not run the count into the negative."""
    sender = grbl.Sender(["G21"])
    sender.take()
    sender.confirm()
    sender.confirm()
    assert sender.acked == 1


def test_the_fraction_counts_the_answers_and_not_the_blocks_written():
    sender = grbl.Sender(["G21", "G90", "M4", "M5"], buffer=128)
    assert sender.fraction == 0.0
    sender.take()
    assert sender.fraction == 0.0
    sender.confirm()
    sender.confirm()
    assert sender.fraction == pytest.approx(0.5)
    sender.confirm()
    sender.confirm()
    assert sender.fraction == 1.0 and sender.done


def test_a_job_of_nothing_is_finished_immediately():
    sender = grbl.Sender([])
    assert sender.take() == b""
    assert sender.done and sender.fraction == 1.0


def test_the_buffer_is_never_smaller_than_one_block_of_room():
    sender = grbl.Sender(["G21"], buffer=0)
    assert sender.buffer == 1
    assert sender.take() == b"G21\n"

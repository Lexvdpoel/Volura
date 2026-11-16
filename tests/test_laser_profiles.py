"""The presets of the lasers that are known: what a preset may set, and what it may not.

`volura/laser_profiles.py` is a table of machines and nothing else: no Qt, no cable, no
thread. What matters about a table of machines is that it never asks for something that
hurts: a preset that changes the steps per millimetre moves the whole machine, and a
preset that switches the limits off removes the thing that was there to save it. So those
numbers are tested here, together with the reading of what a board answers to `$$` - the
board is the truth, and the window has to be able to listen to it.

Run with:  python -m pytest tests/test_laser_profiles.py
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from volura import laser_profiles as profiles  # noqa: E402

ANSWER = [
    "Grbl 1.1h ['$' for help]",
    "$G0=G17,G21,G90",
    "$100=250.000",
    "$110=6000.000",
    "$130=400.000 (x max travel)",
    "$131=415.000 (y max travel)",
    "$30=1000.000 (max spindle speed)",
    "$32=1 (laser mode)",
]


def test_there_are_presets_and_the_open_ones_are_marked():
    assert len(profiles.MACHINES) >= 10
    assert sum(1 for m in profiles.MACHINES if m.open_source) >= 4


def test_every_preset_has_a_key_and_a_name_of_its_own():
    keys = [m.key for m in profiles.MACHINES]
    labels = [m.label for m in profiles.MACHINES]
    assert len(set(keys)) == len(keys) and "" not in keys
    assert len(set(labels)) == len(labels)
    assert all(m.label.strip() for m in profiles.MACHINES)


def test_every_preset_is_a_machine_with_a_work_area():
    for machine in profiles.MACHINES:
        assert machine.known, machine.key
        assert machine.x > 0 and machine.y > 0
        assert machine.s_max in (255, 1000), machine.key
        assert machine.jog > 0 and machine.rapid >= machine.jog
        assert machine.baud in (115200, 57600, 74880), machine.key


def test_a_preset_makes_a_laser_of_the_board():
    for machine in profiles.MACHINES:
        lines = machine.settings()
        assert "$32=1" in lines, machine.key
        assert f"$30={machine.s_max}" in lines, machine.key


def test_a_preset_never_touches_the_calibration_or_the_limits():
    for machine in profiles.MACHINES:
        numbers = {int(profiles.SETTING.match(line).group(1)) for line in machine.settings()}
        assert not numbers & set(profiles.NEVER), (machine.key, numbers & set(profiles.NEVER))


def test_the_settings_go_out_in_the_order_of_the_board():
    numbers = [int(profiles.SETTING.match(line).group(1))
               for line in profiles.MACHINES[0].settings()]
    assert numbers == sorted(numbers)


def test_the_work_area_on_the_screen_wins_from_the_brochure():
    lines = profiles.find("s9").settings((123.0, 45.0))
    assert "$130=123" in lines and "$131=45" in lines
    assert "$130=400" not in lines


def test_a_work_area_of_nothing_says_nothing_about_the_travel():
    for travel in ((0, 0), (400, 0), (0, 400)):
        lines = profiles.find("s9").settings(travel)
        assert not any(line.startswith("$13") for line in lines), travel


def test_a_point_outside_the_machine_is_outside():
    s9 = profiles.find("s9")
    assert s9.inside(0, 0) and s9.inside(400, 415)
    assert not s9.inside(-0.5, 0) and not s9.inside(401, 100)
    assert s9.inside(-0.5, 0, margin=1.0)


def test_an_unknown_machine_has_room_for_everything():
    assert not profiles.CUSTOM.known
    assert profiles.CUSTOM.inside(1e9, -1e9)


def test_the_own_machine_is_the_first_thing_you_can_choose():
    assert profiles.choices()[0][0] == profiles.CUSTOM.key
    assert [key for key, _ in profiles.choices()] == \
        [profiles.CUSTOM.key] + [key for key, _ in profiles.names()]


def test_an_unknown_key_is_the_own_machine():
    assert profiles.find("nonsense") is profiles.CUSTOM
    assert profiles.find(None) is profiles.CUSTOM
    assert profiles.find("s9").key == "s9"


def test_what_the_board_answered_becomes_numbers():
    settings = profiles.parse_settings(ANSWER)
    assert settings[130] == 400.0 and settings[131] == 415.0
    assert settings[30] == 1000.0 and settings[32] == 1.0
    assert settings[100] == 250.0


def test_the_greeting_and_the_g_codes_are_not_settings():
    settings = profiles.parse_settings(["ok", "$G54=0.000", "$#", "", "  ", "error:5"])
    assert settings == {}


def test_a_number_that_is_not_a_number_is_left_alone():
    assert profiles.parse_settings(["$30=many", "$31=1.5"]) == {31: 1.5}


def test_the_travel_and_the_s_word_come_from_the_board():
    settings = profiles.parse_settings(ANSWER)
    assert profiles.travel_of(settings) == (400.0, 415.0)
    assert profiles.s_max_of(settings) == 1000
    assert profiles.laser_mode(settings) is True


def test_a_board_that_said_nothing_is_not_a_preset():
    assert profiles.travel_of({}) == (0.0, 0.0)
    assert profiles.s_max_of({}, default=255) == 255
    assert profiles.s_max_of({30: 0}, default=255) == 255
    assert profiles.laser_mode({}) is None


def test_a_board_that_says_the_beam_stays_on_is_believed():
    assert profiles.laser_mode(profiles.parse_settings(["$32=0 (laser mode)"])) is False


def test_a_machine_that_matches_the_board_is_named():
    assert profiles.match(400, 415, 1000).key == "s9"
    assert profiles.match(400, 415).key == "s9"
    assert profiles.match(400, 415, 255) is None
    assert profiles.match(123, 45) is None
    assert profiles.match(0, 0) is None

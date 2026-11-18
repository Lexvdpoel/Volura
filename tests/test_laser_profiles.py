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

import pytest  # noqa: E402

from volura import laser_profiles as profiles  # noqa: E402

@pytest.fixture(autouse=True)
def no_saved_presets():
    """A run of these tests sees no preset another test left behind."""
    from volura.settings import qsettings
    q = qsettings()
    keys = (profiles.PRESETS_KEY, profiles.CNC_PRESETS_KEY)
    was = {key: q.value(key) for key in keys}
    for key in keys:
        q.remove(key)
    q.sync()
    yield
    for key in keys:
        if was[key] is None:
            q.remove(key)
        else:
            q.setValue(key, was[key])
    q.sync()


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


#: the machines of a list whose size is chosen by whoever builds or wires them
GENERIC = {"grbl32", "grbl_mill", "lowrider4", "mpcnc"}
BOTH = (profiles.LASER, profiles.CNC)


def test_there_are_presets_and_the_open_ones_are_marked():
    assert len(profiles.MACHINES) >= 20
    assert sum(1 for m in profiles.MACHINES if m.open_source) >= 6
    assert len(profiles.MILLS) >= 20
    assert sum(1 for m in profiles.MILLS if m.open_source) >= 8


def test_every_preset_has_a_key_and_a_name_of_its_own():
    for kind in BOTH:
        machines = profiles.machines(kind)
        keys = [m.key for m in machines]
        labels = [m.label for m in machines]
        assert len(set(keys)) == len(keys) and "" not in keys
        assert len(set(labels)) == len(labels)
        assert all(m.label.strip() for m in machines)
        assert all(m.kind == kind for m in machines), kind


def test_every_preset_is_a_machine_with_a_work_area():
    for kind in BOTH:
        for machine in profiles.machines(kind):
            assert machine.known or machine.key in GENERIC, machine.key
            assert machine.s_max in (255, 1000), machine.key
            assert machine.jog > 0 and machine.rapid >= 0
            assert machine.rapid == 0 or machine.rapid >= machine.jog, machine.key
            assert machine.baud in (115200, 57600, 74880), machine.key


def test_a_preset_makes_a_laser_of_the_board():
    for machine in profiles.MACHINES:
        lines = machine.settings()
        assert "$32=1" in lines, machine.key
        assert f"$30={machine.s_max}" in lines, machine.key


def test_a_mill_preset_switches_laser_mode_off_and_leaves_the_spindle_alone():
    for machine in profiles.MILLS:
        lines = machine.settings()
        assert "$32=0" in lines, machine.key
        assert not any(line.startswith("$30=") for line in lines), machine.key
        assert not machine.dynamic


def test_a_top_speed_nobody_published_is_not_sent():
    lines = profiles.find("a5pro").settings()
    assert not any(line.startswith(("$110=", "$111=")) for line in lines)
    assert "$110=15000" in profiles.find("lm2pro").settings()


def test_a_preset_never_touches_the_calibration_or_the_limits():
    for machine in profiles.MACHINES + profiles.MILLS:
        numbers = {int(profiles.SETTING.match(line).group(1)) for line in machine.settings()}
        allowed = {132} if machine.kind == profiles.CNC else set()   # the height of a mill
        assert not numbers & (set(profiles.NEVER) - allowed), (machine.key, numbers)


def test_a_mill_sends_its_height_and_a_laser_does_not():
    lead = profiles.find("lead1010", profiles.CNC)
    assert "$132=100" in lead.settings()
    assert "$132=55" in lead.settings((700, 800, 55))
    assert not any(line.startswith("$132") for line in lead.settings((700, 800, 0))
                   if lead.z == 0)
    assert not any(line.startswith("$132") for line in profiles.find("s9").settings())
    assert profiles.height_of({132: 45.0}) == 45.0 and profiles.height_of({}) == 0.0


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
    assert s9.inside(0, 0) and s9.inside(410, 420)
    assert not s9.inside(-0.5, 0) and not s9.inside(411, 100)
    assert s9.inside(-0.5, 0, margin=1.0)


def test_an_unknown_machine_has_room_for_everything():
    assert not profiles.CUSTOM.known
    assert profiles.CUSTOM.inside(1e9, -1e9)


def test_there_is_no_own_machine_to_choose_only_the_list():
    for kind in BOTH:
        keys = [key for key, _ in profiles.choices(kind)]
        assert profiles.CUSTOM.key not in keys
        assert keys == [key for key, _ in profiles.names(kind)]


def test_an_unknown_key_is_an_unknown_board_of_its_kind():
    assert profiles.find("nonsense") is profiles.CUSTOM
    assert profiles.find(None) is profiles.CUSTOM
    assert profiles.find("s9").key == "s9"
    assert profiles.find("s9", profiles.CNC) is profiles.CUSTOM_CNC
    assert profiles.find("3018pro", profiles.CNC).kind == profiles.CNC


def test_a_board_is_matched_in_its_own_list():
    assert profiles.match(730, 810, kind=profiles.CNC).key == "lead1010"
    assert profiles.match(730, 810) is None
    assert profiles.match(0, 0, kind=profiles.CNC) is None


def test_the_saved_presets_of_a_laser_and_a_mill_are_kept_apart():
    profiles.add_preset(profiles.new_preset("Mine", {30: 24000}), profiles.CNC)
    assert profiles.presets() == []
    assert [p["name"] for p in profiles.presets(profiles.CNC)] == ["Mine"]
    assert ("user:Mine", "Mine") in profiles.choices(profiles.CNC)
    assert profiles.find("user:Mine", profiles.CNC).kind == profiles.CNC


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
    assert profiles.match(410, 420, 1000).key == "s9"
    assert profiles.match(410, 420).key == "s9"
    assert profiles.match(410, 420, 255) is None
    assert profiles.match(123, 45) is None
    assert profiles.match(0, 0) is None

def test_the_k40_is_only_there_with_a_grbl_board():
    k40 = profiles.find("k40")
    assert k40.key == "k40"
    assert (k40.x, k40.y) == (300.0, 200.0)
    assert k40.s_max == 1000
    assert "M2" in k40.note and "GRBL" in k40.label


# ------------------------------------------------------------------ presets of the user
@pytest.fixture()
def stored():
    """The presets of this test run, empty to start with."""
    from volura.settings import qsettings
    q = qsettings()
    was = q.value(profiles.PRESETS_KEY)
    q.remove(profiles.PRESETS_KEY)
    q.sync()
    yield q
    if was is None:
        q.remove(profiles.PRESETS_KEY)
    else:
        q.setValue(profiles.PRESETS_KEY, was)
    q.sync()


def board_answers():
    return {"$30=255.000 (max spindle speed)", "$32=1 (laser mode)",
            "$100=250.000 (x step pulse)", "$110=3000.000 (x max rate)",
            "$130=300.000 (x max travel)", "$131=200.000 (y max travel)"}


def test_a_preset_keeps_everything_the_board_answered():
    preset = profiles.new_preset("Atelier", profiles.parse_settings(board_answers()),
                                 300, 200, s_max=255)
    assert set(preset["settings"]) == {"30", "32", "100", "110", "130", "131"}


def test_a_preset_sends_its_calibration_to_nobody():
    preset = profiles.new_preset("Atelier", profiles.parse_settings(board_answers()),
                                 300, 200, s_max=255)
    assert preset["send"] == [30, 32, 110, 130, 131]
    lines = profiles.preset_lines(preset)
    assert "$100=250" not in lines and "$130=300" in lines


def test_a_user_who_ticks_the_calibration_gets_it_sent():
    preset = profiles.new_preset("Atelier", profiles.parse_settings(board_answers()),
                                 300, 200, s_max=255)
    lines = profiles.preset_lines(preset, include=[30, 32, 100, 110, 130, 131])
    assert "$100=250" in lines


def test_a_preset_comes_back_out_of_the_settings(stored):
    assert profiles.presets() == []
    profiles.add_preset(profiles.new_preset("Atelier", {"30": 255}, 300, 200))
    profiles.add_preset(profiles.new_preset("Zolder", {"30": 1000}, 400, 400))
    assert [p["name"] for p in profiles.presets()] == ["Atelier", "Zolder"]
    assert profiles.find_preset("Atelier")["x"] == 300.0
    assert profiles.find_preset("Nooit") is None


def test_saving_a_preset_again_replaces_it_and_keeps_the_place(stored):
    profiles.add_preset(profiles.new_preset("Atelier", {"30": 255}, 300, 200))
    profiles.add_preset(profiles.new_preset("Zolder", {"30": 1000}, 400, 400))
    profiles.add_preset(profiles.new_preset("Atelier", {"30": 255}, 320, 210))
    assert [p["name"] for p in profiles.presets()] == ["Zolder", "Atelier"]
    assert profiles.find_preset("Atelier")["x"] == 320.0


def test_forgetting_a_preset_leaves_the_others(stored):
    profiles.add_preset(profiles.new_preset("Atelier", {"30": 255}, 300, 200))
    profiles.add_preset(profiles.new_preset("Zolder", {"30": 1000}, 400, 400))
    assert [p["name"] for p in profiles.remove_preset("Atelier")] == ["Zolder"]


def test_presets_go_to_a_file_and_come_back(stored):
    profiles.add_preset(profiles.new_preset("Atelier", {"30": 255, "32": 1}, 300, 200))
    text = profiles.presets_to_json(profiles.presets())
    assert profiles.presets_from_json(text)[0]["send"] == [30, 32]


def test_a_file_that_is_not_presets_is_nothing_at_all():
    for text in ("", "   ", "not json at all", "[]", "{}", '[{"x": 1}]', '{"a": 1}'):
        assert profiles.presets_from_json(text) == [], text


def test_a_preset_of_the_user_is_a_machine_of_the_list(stored):
    profiles.add_preset(profiles.new_preset("Atelier", {"30": 255, "130": 300}, 300, 200))
    machine = profiles.find(profiles.user_key("Atelier"))
    assert machine.label == "Atelier"
    assert (machine.x, machine.y) == (300.0, 200.0)
    assert machine.s_max == 255
    assert machine.settings() == ["$30=255", "$130=300"]
    assert (profiles.user_key("Atelier"), "Atelier") in profiles.choices()


def test_a_preset_that_was_forgotten_is_the_own_machine(stored):
    assert profiles.find(profiles.user_key("Nooit")) is profiles.CUSTOM


def test_the_list_comes_first_and_the_saved_ones_behind_it(stored):
    profiles.add_preset(profiles.new_preset("Atelier", {"30": 255}, 300, 200))
    keys = [key for key, _ in profiles.choices()]
    assert keys[0] == profiles.MACHINES[0].key
    assert keys[-1] == profiles.user_key("Atelier")


def test_a_setting_knows_what_it_is():
    assert profiles.setting_name(110).endswith("(X)")
    assert profiles.setting_name(101).endswith("(Y)")
    assert profiles.setting_name(122).endswith("(Z)")
    assert "(Z)" not in profiles.setting_name(32)
    assert profiles.setting_name(32) == profiles.setting_group(32)
    assert "999" in profiles.setting_name(999)


def test_the_numbers_that_belong_to_one_machine_are_marked():
    assert profiles.is_risky(100) and profiles.is_risky(20) and profiles.is_risky(132)
    assert not profiles.is_risky(30) and not profiles.is_risky(32) and not profiles.is_risky(130)

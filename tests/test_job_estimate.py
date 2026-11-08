"""What a laser job takes: the metres it cuts, the pierces and the time.

`job_estimate` in `volura/vector2d.py` stands next to the path helpers and works on the
shared 2D model: per layer that the beam follows the length of its paths, the number of
pierces, the passes, the speed and the time, with the rapid moves between the paths and
the total on top. Raster and mill layers are only named, since their time does not follow
from the layer. The dialog in `volura/app.py` shows nothing but those numbers.

Run with:  python -m pytest tests/test_job_estimate.py
"""
import math
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from PySide6.QtWidgets import QApplication, QLabel  # noqa: E402

from volura import app as app_module  # noqa: E402
from volura.vector2d import Drawing, job_estimate, path_length  # noqa: E402

_app = QApplication.instance() or QApplication(sys.argv[:1])


# ------------------------------------------------------------------ helpers

def rect(w, h, x=0.0, y=0.0):
    return np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=float)


def sheet(**params) -> Drawing:
    """An empty worksheet whose default layers carry `params`."""
    d = Drawing.new_sheet()
    for lay in d.layers.values():
        lay.params.update(params)
    return d


def rows_of(estimate) -> dict:
    return {row["layer"]: row for row in estimate["layers"]}


# ------------------------------------------------------------------ the length of a path

def test_a_closed_path_includes_the_run_back_to_the_start():
    assert path_length([[0, 0], [30, 0], [30, 40]]) == pytest.approx(70.0)
    assert path_length([[0, 0], [30, 0], [30, 40]], closed=True) == pytest.approx(120.0)
    assert path_length([[0, 0]]) == 0.0


# ------------------------------------------------------------------ the estimate

def test_a_rectangle_of_100x50_is_300_mm_and_thirty_seconds():
    d = sheet(passes=2)
    d.add_path(rect(100, 50), closed=True, layer="Cut")
    row = rows_of(job_estimate(d))["Cut"]
    assert row["length"] == pytest.approx(300.0)
    assert row["pierces"] == 1 and row["passes"] == 2
    assert row["speed"] == pytest.approx(20.0)
    assert row["time"] == pytest.approx(300.0 * 2 / 20.0)


def test_an_open_line_of_40_mm_gets_a_row_of_its_own():
    d = sheet()
    d.add_path([[10, 10], [50, 10]], layer="Engrave")
    row = rows_of(job_estimate(d))["Engrave"]
    assert row["length"] == pytest.approx(40.0)
    assert row["pierces"] == 1 and row["time"] == pytest.approx(2.0)


def test_every_path_is_one_pierce_and_a_pierce_costs_time_per_pass():
    d = sheet(passes=3)
    d.add_path(rect(10, 10), closed=True, layer="Cut")
    d.add_path([[20, 0], [20, 10]], layer="Cut")
    row = rows_of(job_estimate(d, pierce_time=1.5))["Cut"]
    assert row["pierces"] == 2
    assert row["time"] == pytest.approx((40.0 + 10.0) * 3 / 20.0 + 2 * 3 * 1.5)


def test_only_the_layers_with_paths_are_listed():
    d = sheet()
    d.add_path(rect(10, 10), closed=True, layer="Cut")
    est = job_estimate(d)
    assert [row["layer"] for row in est["layers"]] == ["Cut"]
    assert est["skipped"] == []


def test_the_head_travels_from_the_origin_to_the_first_path():
    d = sheet()
    d.add_path([[0, 30], [40, 30]], layer="Cut")
    assert job_estimate(d)["travel"] == pytest.approx(30.0)


def test_the_head_travels_between_the_paths_in_the_order_of_the_drawing():
    d = sheet()
    d.add_path(rect(100, 50), closed=True, layer="Cut")        # ends at (0, 0) again
    d.add_path([[10, 10], [50, 10]], layer="Engrave")
    est = job_estimate(d, travel_speed=100.0)
    assert est["travel"] == pytest.approx(math.hypot(10.0, 10.0))
    assert est["travel_time"] == pytest.approx(est["travel"] / 100.0)
    assert est["total"] == pytest.approx(15.0 + 2.0 + est["travel_time"])


def test_a_raster_layer_is_named_but_not_timed():
    d = sheet()
    d.add_path(rect(10, 10), closed=True, layer="Raster")
    est = job_estimate(d)
    assert est["layers"] == [] and est["skipped"] == ["Raster"]
    assert est["travel"] == 0.0 and est["total"] == pytest.approx(0.0)


def test_mill_layers_are_skipped_because_a_mill_has_its_own_export():
    d = sheet()
    d.add_path(rect(10, 10), closed=True, layer="Cut")
    d.add_path(rect(10, 10, 20), closed=True, layer="Mill outside")
    d.add_path(rect(10, 10, 40), closed=True, layer="Pocket")
    est = job_estimate(d)
    assert [row["layer"] for row in est["layers"]] == ["Cut"]
    assert est["skipped"] == ["Mill outside", "Pocket"]


def test_a_hidden_layer_goes_to_no_machine_at_all():
    d = sheet()
    d.add_path(rect(10, 10), closed=True, layer="Cut")
    d.layers["Cut"].visible = False
    est = job_estimate(d)
    assert est["layers"] == [] and est["skipped"] == []


def test_a_speed_of_zero_gives_no_time_instead_of_an_error():
    d = sheet()
    d.layers["Cut"].params["speed"] = 0
    d.add_path(rect(100, 50), closed=True, layer="Cut")
    est = job_estimate(d)
    assert est["layers"][0]["speed"] == 0.0
    assert est["layers"][0]["time"] is None
    assert est["total"] is None, "a total that leaves a whole layer out is no total"


# ------------------------------------------------------------------ the times in the table

@pytest.mark.parametrize("seconds, text", (
        (0, "0:00"), (9.4, "0:09"), (29.6, "0:30"), (65, "1:05"), (3599, "59:59"),
        (3600, "1:00:00"), (3725, "1:02:05"), (7325.0, "2:02:05"), (None, "—")))
def test_times_stand_as_m_ss_and_as_h_mm_ss(seconds, text):
    assert app_module.fmt_time(seconds) == text


# ------------------------------------------------------------------ through the window

@pytest.fixture()
def win(monkeypatch, tmp_path):
    from volura.app import MainWindow
    monkeypatch.setattr(app_module.errors, "log_dir", lambda: str(tmp_path))
    monkeypatch.setattr(app_module, "autosave_path", lambda: str(tmp_path / "autosave.volura"))
    w = MainWindow()
    w.doc.dirty = False
    yield w
    w.doc.dirty = False
    w.close()


@pytest.fixture()
def shown(monkeypatch):
    """The Job Estimate dialogs that are opened, without blocking on the screen."""
    opened = []

    def no_exec(self, *args):
        opened.append(self)
        return 0

    monkeypatch.setattr(app_module.JobEstimateDialog, "exec", no_exec)
    return opened


def cell(dlg, row, col) -> str:
    item = dlg.table.item(row, col)
    return "" if item is None else item.text()


def test_the_action_shows_the_numbers_of_the_worksheet(win, shown):
    win.doc.sheet.add_path(rect(100, 50), closed=True, layer="Cut", src=1)
    win.doc.sheet.layers["Cut"].params["passes"] = 2
    win.a_job_estimate.trigger()
    assert len(shown) == 1
    dlg = shown[0]
    assert [cell(dlg, 0, c) for c in range(6)] == ["Cut", "0.300", "1", "2×", "20", "0:30"]
    assert cell(dlg, 1, 0) == "Travel" and cell(dlg, 1, 5) == "0:00"
    assert cell(dlg, 2, 0) == "Total" and cell(dlg, 2, 5) == "0:30"


def test_the_note_names_the_layers_that_are_not_timed(win, shown):
    win.doc.sheet.add_path(rect(10, 10), closed=True, layer="Raster", src=1)
    win.doc.sheet.add_path(rect(10, 10, 20), closed=True, layer="Cut", src=2)
    win.a_job_estimate.trigger()
    assert shown[0].findChild(QLabel).text() == "Not included: Raster."


def test_an_empty_worksheet_stays_in_the_status_bar(win, shown):
    win.a_job_estimate.trigger()
    assert shown == []
    assert win.hint.text() == "The worksheet is empty."

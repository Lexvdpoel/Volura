"""Make the screenshots of the website and the manual: docs/website/img/<lang>/<name>.png.

Run from the repository root with the Volura virtualenv:

    python tools/screenshots.py            # every picture, in English and Dutch
    python tools/screenshots.py knob       # only the pictures whose name contains "knob"

The pages load the pictures from GitHub (raw.githubusercontent.com, branch main), so
after a push the website and the manual show the new pictures without uploading them
again. Run this after a change that shows in a picture, check the result, and commit
docs/website/img with it.

It needs a real display: the 3D view is drawn with OpenGL, which the "offscreen"
platform of Qt does not draw. The window opens far outside the screen, so you do not
see it, and it uses its own empty settings, so your own preferences stay as they are.
"""
from __future__ import annotations

import os
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

TMP = tempfile.mkdtemp(prefix="volura-shots-")
_ini = os.path.join(TMP, "settings.ini")
with open(_ini, "w", encoding="utf-8") as f:
    f.write("[General]\nshow_welcome=false\nask_simplify_import=false\ncheck_updates=false\n")
os.environ["VOLURA_SETTINGS_FILE"] = _ini

from PySide6.QtCore import QEventLoop, QPoint, QRect, QTimer  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

OUT = os.path.join(ROOT, "docs", "website", "img")
SIZE = (1280, 800)
PART_COLOR = "#3b7dd8"

app = QApplication.instance() or QApplication(sys.argv[:1])


def wait(ms=200):
    loop = QEventLoop()
    QTimer.singleShot(ms, loop.quit)
    loop.exec()


def wait_for(test, timeout=30.0):
    """Let the event loop run until `test()` holds: a part is built in the background."""
    end = time.time() + timeout
    while not test():
        if time.time() > end:
            raise TimeoutError("the window did not get there in time")
        wait(100)
    wait(300)


class Shots:
    def __init__(self, lang):
        from volura import i18n
        i18n.set_language(lang)         # before volura.app is imported
        from volura import app as app_module
        # logs and the autosave go to a folder of their own: the user's autosave stays
        app_module.errors.log_dir = lambda: TMP
        self.app_module = app_module
        self.lang = lang
        self.w = app_module.MainWindow()
        self.w.resize(*SIZE)
        self.w.move(-6000, -6000)          # out of sight, but really drawn
        self.w.show()
        wait(500)
        os.makedirs(os.path.join(OUT, lang), exist_ok=True)

    # ------------------------------------------------------------ helpers
    def fresh(self, mode="3d"):
        w = self.w
        w.doc.dirty = False
        w.file_new()
        # blue parts: the red default would hide the orange-red of the overhang view
        w.settings.current_color = PART_COLOR
        w.set_mode(mode)
        w.doc.dirty = False
        wait(200)

    def build(self, make, options):
        """A part that is built in the background: wait until it is in the model."""
        before = len(self.w.doc.groups)
        make(options)
        wait_for(lambda: len(self.w.doc.groups) > before)

    def view_3d(self, view="iso", closer=0.7):
        """The parts fill the picture: zoom extents leaves room for the whole bed."""
        w = self.w
        w.doc.set_selection([])
        w.hint.setText("")             # the status bar: no message of the last build
        w.view.set_view(view)
        w.view.zoom_extents()
        w.view.cam.distance *= closer
        w.view.commit_camera()
        w.view.update()
        wait(400)

    def save(self, name, pixmap):
        path = os.path.join(OUT, self.lang, name + ".png")
        pixmap.save(path)
        print("  ", os.path.relpath(path, ROOT), flush=True)

    def grab_window(self, name):
        self.save(name, self.w.grab())

    def grab_widget(self, name, widget, margin=0):
        w = self.w
        top_left = widget.mapTo(w, QPoint(0, 0))
        rect = QRect(top_left, widget.size()).adjusted(-margin, -margin, margin, margin)
        self.save(name, w.grab(rect))

    def grab_dialog(self, name, dlg):
        dlg.move(-5000, -5000)
        dlg.show()
        wait(300)
        self.save(name, dlg.grab())
        dlg.close()

    def grab_menu(self, name, menu):
        menu.adjustSize()
        menu.ensurePolished()
        self.save(name, menu.grab())

    # ------------------------------------------------------------ the pictures
    def parts(self):
        """The printable parts, each in a filament colour of its own (also the homepage)."""
        m, w = self.app_module, self.w
        self.fresh()
        box = m.ProjectBoxDialog(w).options()
        box.update(screws="M3", vents=4)
        knob = m.KnobDialog(w).options()
        standoff = m.StandoffDialog(w).options()
        standoff.update(count=4)
        label = m.LabelPlateDialog(w).options()
        label.update(text="Volura")
        for colour, make, options in (
                (PART_COLOR, w.make_project_box_model, box),
                ("#f2c230", w.make_knob_model, knob),
                ("#f08a24", w.make_standoff_model, standoff),
                ("#d23b3b", w.make_label_plate_model, label),
                ("#3fa34d", w.make_cable_clip_model, m.CableClipDialog(w).options())):
            w.settings.current_color = colour
            self.build(make, options)
        w.settings.current_color = PART_COLOR
        self.view_3d()
        self.grab_window("window_3d")
        self.grab_widget("parts", w.view)

    def project_box(self):
        m, w = self.app_module, self.w
        self.fresh()
        box = m.ProjectBoxDialog(w).options()
        box.update(screws="M3", vents=4)
        self.build(w.make_project_box_model, box)
        self.view_3d()
        self.grab_widget("project_box", w.view)
        self.grab_dialog("dialog_project_box", m.ProjectBoxDialog(w))

    def knob(self):
        m, w = self.app_module, self.w
        self.fresh()
        for grip in ("knurl", "lobes", "smooth"):
            o = m.KnobDialog(w).options()
            o.update(grip=grip)
            self.build(w.make_knob_model, o)
        self.view_3d()
        self.grab_widget("knob", w.view)
        self.grab_dialog("dialog_knob", m.KnobDialog(w))

    def hinge_and_text(self):
        """The print-in-place hinge, the battery holder and 3D text."""
        from volura.part_dialogs import BatteryHolderDialog, HingeDialog, Text3DDialog
        w = self.w
        self.fresh()
        battery = BatteryHolderDialog(w).options()
        battery.update(cell="AA", count=3)
        text = Text3DDialog(w).options()
        text.update(text="Volura")
        for colour, make, options in (
                ("#3fa34d", w.make_battery_holder_model, battery),     # the tallest at the back
                (PART_COLOR, w.make_hinge_model, HingeDialog(w).options()),
                ("#d23b3b", w.make_text3d_model, text)):
            w.settings.current_color = colour
            self.build(make, options)
        w.settings.current_color = PART_COLOR
        self.view_3d()
        self.grab_widget("hinge_and_text", w.view)

    def test_prints(self):
        w = self.w
        self.fresh()
        before = len(w.doc.groups)
        w._apply_clearance_test(5.0, 0.0, 0.1, 6)
        wait_for(lambda: len(w.doc.groups) > before)
        before = len(w.doc.groups)
        w._apply_overhang_test(30.0, 10.0, 6)
        wait_for(lambda: len(w.doc.groups) > before)
        self.view_3d()
        self.grab_widget("test_prints", w.view)

    def overhangs(self):
        """Overhang view: a mushroom (column with a wide cap) and a ball, seen from low."""
        import manifold3d as mf
        from volura import ops
        w = self.w
        self.fresh()
        mushroom = (mf.Manifold.cube([10.0, 10.0, 25.0]).translate((15.0, 15.0, 0.0))
                    + mf.Manifold.cube([40.0, 40.0, 6.0]).translate((0.0, 0.0, 25.0)))
        ball = mf.Manifold.sphere(15.0, 64).translate((75.0, 20.0, 15.0))
        w.doc.checkpoint()
        for name, solid in (("Mushroom", mushroom), ("Ball", ball)):
            ops.group_from_manifold(w.doc, solid, PART_COLOR, name)
        w.doc.notify()
        w.a_overhangs.setChecked(True)
        w.toggle_overhangs(True)
        self.view_3d(closer=0.8)
        # the faces that need support are undersides: look at them from just above the bed
        w.view.cam.yaw, w.view.cam.pitch = -60.0, -8.0
        w.view.commit_camera()
        w.view.update()
        wait(400)
        self.grab_widget("overhangs", w.view)
        w.a_overhangs.setChecked(False)
        w.toggle_overhangs(False)

    def worksheet(self):
        m, w = self.app_module, self.w
        self.fresh("2d")
        w.make_box_sheet(m.BoxJointDialog(w).options())
        wait(500)
        w.make_spur_gear_sheet(m.SpurGearDialog(w).options())
        wait(300)
        w.sheet_view.clear_selection()
        w.zoom_extents()
        wait(400)
        self.grab_window("worksheet")

    def vent_pattern(self):
        w = self.w
        self.fresh("2d")
        sheet = w.doc.sheet
        p = sheet.add_path([[0, 0], [120, 0], [120, 80], [0, 80]], closed=True,
                           layer=sheet.default_layer("cut"))
        w.sheet_view.set_selection([p])
        w.apply_vent_pattern({"shape": "hex", "size": 6.0, "web": 2.0, "margin": 4.0})
        w.sheet_view.clear_selection()
        w.zoom_extents()
        wait(400)
        self.grab_widget("vent_pattern", w.sheet_view)

    def menus(self):
        w = self.w
        self.fresh("3d")
        self.grab_menu("menu_3d_printing", w.print_tools_menu)
        w.set_mode("2d")
        wait(200)
        self.grab_menu("menu_laser", w.laser_menu)
        w.set_mode("3d")

    ALL = ("parts", "project_box", "knob", "hinge_and_text", "test_prints", "overhangs", "worksheet",
           "vent_pattern", "menus")

    def close(self):
        self.w.doc.dirty = False
        self.w.close()


def main(argv):
    """Every language in a process of its own: texts are translated when the modules are
    imported, so one process cannot switch from English to Dutch halfway."""
    if argv[:1] == ["--lang"]:
        lang, want = argv[1], argv[2:]
        print(lang, flush=True)
        shots = Shots(lang)
        try:
            for name in Shots.ALL:
                if not want or any(x in name for x in want):
                    getattr(shots, name)()
        finally:
            shots.close()
        return 0
    import subprocess
    for lang in ("en", "nl"):
        code = subprocess.call([sys.executable, os.path.abspath(__file__), "--lang", lang] + argv)
        if code:
            return code
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

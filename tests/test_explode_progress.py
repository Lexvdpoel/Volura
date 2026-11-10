"""Explode Group runs in the background with a progress bar.

Merging a large solid into the loose geometry can take a long time, so `ops.exploded_root`
builds the new loose geometry on a copy and reports how far it is, and the window puts it in
place only when it is done: one undo step, the same faces as a plain `ops.explode`, and a
cancel leaves the model as it was.

Run with:  python -m pytest tests/test_explode_progress.py
"""
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import manifold3d as mf  # noqa: E402
import pytest  # noqa: E402
from PySide6.QtCore import QCoreApplication  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

import volura.app as app_module  # noqa: E402
from volura import background, ops  # noqa: E402

_app = QApplication.instance() or QApplication(sys.argv[:1])


@pytest.fixture()
def win(monkeypatch, tmp_path):
    monkeypatch.setattr(app_module.errors, "log_dir", lambda: str(tmp_path))
    from volura.app import MainWindow
    w = MainWindow()
    w.doc.dirty = False
    QCoreApplication.processEvents()
    yield w
    w.doc.dirty = False
    w.close()


def ball(win, segments=24):
    g = ops.group_from_manifold(win.doc, mf.Manifold.sphere(10.0, segments), "#3b7dd8", "Ball")
    win.doc.set_selection([("g", g.id, g.id)])
    win.doc.notify()
    return g


def test_the_progress_goes_up_to_the_end():
    from volura.model import Document
    doc = Document()
    g = ops.group_from_manifold(doc, mf.Manifold.sphere(10.0, 24), "#3b7dd8", "Ball")
    seen = []
    root = ops.exploded_root(doc, [g], lambda frac, text=None: seen.append(frac))
    assert len(seen) > 10, "the progress is not reported along the way"
    assert seen == sorted(seen) and 0.0 <= seen[0] and seen[-1] <= 1.0
    assert len(doc.root.F) == 0, "the document changed before the end"
    assert len(root.F) == len(g.ents.F)


def test_explode_gives_the_same_faces_as_before_in_one_undo_step(win):
    g = ball(win)
    faces = len(g.ents.F)
    steps = len(win.doc.undo_stack)
    win.explode_group(wait=True)
    assert win.doc.group(g.id) is None, "the group is still there"
    assert len(win.doc.root.F) == faces
    assert all(f.color for f in win.doc.root.F.values()), "the faces lost the group colour"
    assert len(win.doc.undo_stack) == steps + 1
    win.doc.undo()
    assert win.doc.group(g.id) is not None and len(win.doc.root.F) == 0


def test_a_cancelled_explode_leaves_the_model_alone(win, monkeypatch):
    g = ball(win)

    def cancel(*_args, **_kw):
        raise background.BackgroundCancelled()

    monkeypatch.setattr(ops, "exploded_root", lambda doc, groups, report: cancel())
    steps = len(win.doc.undo_stack)
    win.explode_group(wait=True)
    assert win.doc.group(g.id) is not None
    assert len(win.doc.root.F) == 0
    assert len(win.doc.undo_stack) == steps


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))

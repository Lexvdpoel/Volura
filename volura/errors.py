"""Fault tolerance of the interface: one log file and event handlers that never crash.

A Qt override that raises would normally end in an endless series of
"Error calling Python override of QOpenGLWidget::event()" messages, each with a
dialog of its own. `safe_handler` runs the handler, and when it raises: the
traceback goes to stderr and to volura.log, the half finished edit is undone,
the active tool is reset and the status bar says what happened. The application
keeps running.
"""
from __future__ import annotations

import functools
import os
import sys
import tempfile
import traceback
from datetime import datetime

from PySide6.QtCore import QStandardPaths

from . import _glconfig
from .i18n import tr

ERROR_STATUS = tr("Internal error, see volura.log")
LOG_NAME = "volura.log"
GL_ERROR_DRAIN_MAX = 10


# ------------------------------------------------------------------ logging

def log_dir():
    """The folder of the program: the log and the autosave live here."""
    folder = ""
    try:
        folder = QStandardPaths.writableLocation(QStandardPaths.AppDataLocation) or ""
    except Exception:
        folder = ""
    if not folder:
        folder = os.path.join(tempfile.gettempdir(), "Volura")
    try:
        os.makedirs(folder, exist_ok=True)
    except OSError:
        folder = tempfile.gettempdir()
    return folder


def log_path():
    return os.path.join(log_dir(), LOG_NAME)


def format_exception(exc_type, exc, tb):
    if exc_type is None:
        exc_type, exc, tb = sys.exc_info()
    if exc_type is None:
        return ""
    return "".join(traceback.format_exception(exc_type, exc, tb))


def log_exception(exc_type=None, exc=None, tb=None, note=""):
    """The traceback to stderr and to volura.log; logging itself never fails."""
    text = format_exception(exc_type, exc, tb)
    try:
        sys.stderr.write(text)
        sys.stderr.flush()
    except Exception:
        pass
    try:
        stamp = datetime.now().isoformat(timespec="seconds")
        with open(log_path(), "a", encoding="utf-8") as f:
            f.write(f"\n===== {stamp} {note}\n{text}")
    except OSError:
        pass
    return text


# ------------------------------------------------------------------ OpenGL

GL_DEBUG_ENV = _glconfig.GL_DEBUG_ENV
gl_debug = _glconfig.gl_debug


def configure_pyopengl():
    """Switch off the error check that PyOpenGL does after every GL call; returns whether it checks.

    PyOpenGL wraps every call with ``glGetError`` when it decides to check at all - half
    a second per twenty frames of the 3D view. The viewport drains the error queue itself
    (``clear_gl_errors``), so the per call check is only wanted while looking for the call
    that goes wrong: ``VOLURA_GL_DEBUG=1`` keeps it on.

    That decision is taken from the environment when ``OpenGL`` is imported and from
    ``OpenGL.ERROR_CHECKING`` when ``OpenGL.error`` is imported, so ``volura/__init__.py``
    sets the environment (``_glconfig.apply_env``) before any OpenGL import and this sets
    both again for the case in which ``OpenGL`` was already imported by something else.
    Once ``OpenGL._configflags`` exists, nothing can change it any more;
    ``_glconfig.error_checking()`` reports what PyOpenGL really does.
    """
    _glconfig.apply_env()
    return _glconfig.apply_module()


def clear_gl_errors(limit=GL_ERROR_DRAIN_MAX):
    """Empty the error queue of OpenGL; an old error must not come back at glClearColor.

    Returns how many errors were removed; a driver without a context simply raises,
    which is caught here too.
    """
    try:
        configure_pyopengl()
        from OpenGL import GL as gl
    except Exception:
        return 0
    n = 0
    try:
        for _ in range(max(1, int(limit))):
            if gl.glGetError() == gl.GL_NO_ERROR:
                break
            n += 1
    except Exception:
        pass
    return n


# ------------------------------------------------------------------ recovery

def rollback(doc, depth):
    """Undo every edit that started after `depth` checkpoints: what a failed handler left behind."""
    undid = False
    while len(doc.undo_stack) > depth:
        doc.undo()
        undid = True
    if undid:
        del doc.redo_stack[:]
        doc.notify()
    return undid


def recover(view, depth=None):
    """What happens after a failed handler: undo, reset the tool, say it in the status bar."""
    doc = getattr(view, "doc", None)
    if doc is not None and depth is not None:
        try:
            rollback(doc, depth)
        except Exception:
            log_exception(note="rollback failed")
    tool = getattr(view, "tool", None)
    if tool is not None:
        try:
            tool.reset()
        except Exception:
            log_exception(note="resetting the tool failed")
    try:
        view.preview_transform = None
    except Exception:
        pass
    try:
        view.set_status(ERROR_STATUS)
    except Exception:
        log_exception(note="status message failed")


def safe_handler(fn=None, ret=None, boolean=False):
    """Decorator for a Qt event handler: an exception is logged, never shown as a dialog.

    `ret` is what the handler returns when it failed, `boolean` makes sure a handler
    that Qt expects to answer with True or False always does so.
    """
    def decorate(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            doc = getattr(self, "doc", None)
            depth = len(doc.undo_stack) if doc is not None else None
            try:
                out = func(self, *args, **kwargs)
            except Exception:
                log_exception(note=f"error in {getattr(func, '__qualname__', 'event-handler')}")
                recover(self, depth)
                out = ret
            return bool(out) if boolean else out
        return wrapper
    return decorate(fn) if fn is not None else decorate

"""How PyOpenGL is configured, in one place and without importing OpenGL itself.

PyOpenGL decides only once whether it asks GL for an error after every single call:
``OpenGL`` reads the ``PYOPENGL_*`` variables from the environment at its own import, and
``OpenGL._configflags`` copies ``OpenGL.ERROR_CHECKING`` when ``OpenGL.error`` is
imported. After that the decision cannot be taken back, so it has to be made before the
first OpenGL import of the program - which is what ``volura/__init__.py`` does, because
every module of volura runs it first.

That per call ``glGetError`` costs about half a second per twenty frames of the 3D view.
The view drains the error queue itself (``errors.clear_gl_errors``), so the check is only
wanted while looking for the one call that goes wrong: ``VOLURA_GL_DEBUG=1`` asks for the
slow, checking PyOpenGL again.
"""
import os

GL_DEBUG_ENV = "VOLURA_GL_DEBUG"
FLAGS = ("PYOPENGL_ERROR_CHECKING", "PYOPENGL_ERROR_LOGGING")


def gl_debug():
    """True when VOLURA_GL_DEBUG=1 asks for the slow, checking PyOpenGL."""
    return os.environ.get(GL_DEBUG_ENV, "").lower() in ("1", "true")


def apply_env():
    """The PyOpenGL environment variables; returns whether error checking is wanted.

    ``VOLURA_GL_DEBUG`` wins over anything that was already in the environment, so
    turning the check back on never depends on how the process was started; without it a
    choice the user made in ``PYOPENGL_*`` is kept."""
    check = gl_debug()
    for flag in FLAGS:
        if check:
            os.environ[flag] = "1"
        else:
            os.environ.setdefault(flag, "0")
    return check


def apply_module():
    """The same for a process that imported ``OpenGL`` already but not ``OpenGL.error``.

    Returns whether checking is wanted; ``error_checking()`` says what PyOpenGL really
    does with it."""
    check = gl_debug()
    try:
        import OpenGL
    except Exception:
        return check
    OpenGL.ERROR_CHECKING = check
    OpenGL.ERROR_LOGGING = check
    return check


def error_checking():
    """What PyOpenGL really does: the flag ``OpenGL.error`` read at its own import.

    None when PyOpenGL is not installed."""
    try:
        import OpenGL._configflags as flags
    except Exception:
        return None
    return bool(flags.ERROR_CHECKING)

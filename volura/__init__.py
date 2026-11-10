"""Volura - open source 3D modeller for 3D printing."""
from . import _glconfig        # before anything imports OpenGL: see volura/_glconfig.py

__version__ = "0.2.40"

_glconfig.apply_env()

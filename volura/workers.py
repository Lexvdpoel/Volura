"""Spread independent per-group work over all the cores with a process pool.

CPython has the GIL, so threads do not speed up pure Python number crunching;
only separate processes use more than one core. The heavy work in Volura -
triangulating the faces of a group for an export, or building a group's render
triangles - is independent per group, so it splits cleanly over the cores.

The pool is a single :class:`concurrent.futures.ProcessPoolExecutor` that starts
lazily on first use and lives for the life of the app. It is warmed up shortly
after the window appears (see :func:`prewarm`) so the first real job does not
wait for the processes to boot. ``VOLURA_WORKERS`` sets how many workers to use;
``0`` runs everything in the main process, which is what the tests do.

Workers are plain processes that import only the geometry modules a job needs
(no Qt). A job is a top level function plus picklable numpy / list arguments and
the result comes back the same way, so a ``Manifold`` is always rebuilt in the
main process from the arrays a worker returns: a ``Manifold`` itself cannot be
pickled.

The executor and ``multiprocessing`` are imported inside the functions, not at
import time: ``concurrent.futures`` and ``multiprocessing`` together cost about
150 ms, and importing ``volura`` (which reaches this module through ``ops``)
should not pay that until a job actually asks for the pool.
"""
from __future__ import annotations

import os
import sys

# More workers stop paying long before a 48-core machine runs out of cores: on the
# benchmark model (24 groups) 8 workers export as fast as 47, while every worker
# holds about 27 MB (numpy and manifold3d), so 47 of them would cost 1.3 GB.
MAX_DEFAULT_WORKERS = 8

_pool = None
_broken = False
_stopped = False


def max_workers():
    """How many worker processes to use (``0`` = everything in the main process).

    ``VOLURA_WORKERS`` overrides the default of one less than every core, so a
    four-core laptop leaves a core free for the interface, and ``VOLURA_WORKERS=0``
    turns the pool off entirely. The default stops at ``MAX_DEFAULT_WORKERS``."""
    v = os.environ.get("VOLURA_WORKERS")
    if v is not None and v.strip():
        try:
            return max(0, int(v))
        except ValueError:
            pass
    return max(1, min(MAX_DEFAULT_WORKERS, (os.cpu_count() or 1) - 1))


def _in_child():
    """True inside a worker process, so a job never starts a pool of its own."""
    from multiprocessing import current_process
    return current_process().name != "MainProcess"


def _safe_main():
    """Point spawn at an empty module so a worker never re-runs the caller's script.

    A script without an ``if __name__ == "__main__"`` guard (a benchmark, a
    ``python -c`` snippet) would otherwise run again in every worker. A frozen
    build already guards its entry and ``freeze_support`` handles it, so there we
    leave the main module alone."""
    if getattr(sys, "frozen", False):
        return
    main = sys.modules.get("__main__")
    if main is None or getattr(main, "__spec__", "x") is not None:
        return
    try:
        import importlib.util
        spec = importlib.util.find_spec("volura._poolmain")
        if spec is not None:
            main.__spec__ = spec
    except Exception:
        pass


def pool():
    """The shared pool, or ``None`` when it is off, stopped, broken, or we are a worker."""
    global _pool, _broken
    if _broken or _stopped or _in_child() or max_workers() < 1:
        return None
    if _pool is None:
        try:
            from concurrent.futures import ProcessPoolExecutor
            from multiprocessing import get_context
            _safe_main()
            _pool = ProcessPoolExecutor(max_workers=max_workers(), mp_context=get_context("spawn"))
        except Exception:
            _broken = True
            return None
    return _pool


def _noop(_):
    return None


def prewarm():
    """Boot the workers in the background so the first job does not wait for them."""
    p = pool()
    if p is None:
        return
    try:
        for _ in range(max_workers()):
            p.submit(_noop, None)
    except Exception:
        global _broken
        _broken = True


def parallel_map(fn, items, min_items=1, chunksize=1):
    """``list(map(fn, items))`` spread over the pool, results in the same order.

    Runs serially when there is less than ``min_items`` of work, when the pool is
    off or unavailable, or when we are inside a worker. An exception a worker
    raises comes back here unchanged; only a pool that cannot start at all falls
    back to a serial run (and is then left off for the rest of the session)."""
    items = list(items)
    p = pool()
    if p is None or len(items) < max(1, min_items):
        return [fn(it) for it in items]
    from concurrent.futures.process import BrokenProcessPool
    try:
        return list(p.map(fn, items, chunksize=chunksize))
    except BrokenProcessPool:
        global _broken
        _broken = True
        return [fn(it) for it in items]


def shutdown():
    """Stop the pool on the way out of the app; pending jobs are cancelled."""
    global _pool, _stopped
    _stopped = True
    p, _pool = _pool, None
    if p is not None:
        try:
            p.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass


def _reset():
    """Bring the module back to a fresh state (used by the tests)."""
    global _pool, _broken, _stopped
    p, _pool = _pool, None
    _broken = _stopped = False
    if p is not None:
        try:
            p.shutdown(wait=True, cancel_futures=True)
        except Exception:
            pass

"""The process pool: same results as serial, in order, and a clean shutdown.

The pool is off in the rest of the suite (``conftest`` sets ``VOLURA_WORKERS=0``);
these tests turn it on for a moment. A module fixture boots it once and puts the
environment back afterwards, so the pool starts a single time and never leaks
into the other test files.
"""
import os

import numpy as np
import manifold3d as mf
import pytest

from volura import ops, workers
from volura.geom import mesh_arrays
from volura.inference import GeoCache
from volura.model import Document

WORKERS = "4"


# top level so a worker process can import them by name (they must not be closures)
def square(x):
    return x * x


def boom(x):
    if x == 2:
        raise ValueError("boom on two")
    return x


def who_am_i(_):
    return os.getpid()


@pytest.fixture(scope="module")
def pool():
    """Turn the pool on for this module, boot it once, and restore the environment."""
    saved = os.environ.get("VOLURA_WORKERS")
    os.environ["VOLURA_WORKERS"] = WORKERS
    workers._reset()
    workers.prewarm()
    yield
    workers.shutdown()
    workers._reset()
    if saved is None:
        os.environ.pop("VOLURA_WORKERS", None)
    else:
        os.environ["VOLURA_WORKERS"] = saved


def _model(groups=30):
    """A model with enough faces across enough groups for ``ops.parts`` to use the pool."""
    doc = Document()
    colors = ["#f2f2f2", "#d23b3b", "#2f6fd6", "#f08a24"]
    for i in range(groups):
        m = mf.Manifold.cylinder(20, 12, 12, 40) - mf.Manifold.cylinder(20, 6, 6, 40)
        ops.group_from_manifold(doc, m.translate((i * 40.0, 0.0, 0.0)), colors[i % 4], f"part {i}")
    return doc


def _signature(part):
    """The triangles of a part as a sorted, rounded array, so order cannot matter."""
    v, t = mesh_arrays(part.manifold)
    tri = np.round(v[t], 6).reshape(-1, 9)
    return part.name, part.color, np.sort(tri, axis=0)


def test_parallel_map_matches_serial_in_order(pool):
    items = list(range(200))
    assert workers.parallel_map(square, items, min_items=2) == [square(x) for x in items]


def test_parallel_map_actually_uses_other_processes(pool):
    pids = workers.parallel_map(who_am_i, list(range(20)), min_items=2)
    assert os.getpid() not in pids or len(set(pids)) > 1     # ran in the workers
    assert len(pids) == 20


def test_worker_exception_comes_back(pool):
    with pytest.raises(ValueError):
        workers.parallel_map(boom, [1, 2, 3], min_items=2)


def test_export_via_pool_matches_serial(tmp_path, pool):
    doc = _model(30)
    assert ops.PARTS_POOL_MIN_FACES <= sum(len(g.ents.F) for g in doc.groups)   # big enough to spread
    pool_parts = ops.parts(doc)
    assert workers.pool() is not None
    os.environ["VOLURA_WORKERS"] = "0"
    workers._reset()
    try:
        serial_parts = ops.parts(doc)
        assert workers.pool() is None
    finally:
        os.environ["VOLURA_WORKERS"] = WORKERS
        workers._reset()
        workers.prewarm()
    assert len(pool_parts) == len(serial_parts) == 30
    for a, b in zip(pool_parts, serial_parts):
        assert a.problem == b.problem is None
        na, ca, ta = _signature(a)
        nb, cb, tb = _signature(b)
        assert (na, ca) == (nb, cb)
        assert ta.shape == tb.shape and np.array_equal(ta, tb)


def test_zero_workers_runs_without_processes(monkeypatch):
    monkeypatch.setenv("VOLURA_WORKERS", "0")
    workers._reset()
    try:
        assert workers.pool() is None
        assert workers.parallel_map(square, list(range(50)), min_items=2) == [x * x for x in range(50)]
    finally:
        workers._reset()


def test_pool_shuts_down_cleanly(pool):
    workers.parallel_map(square, list(range(10)), min_items=2)   # make sure it is up
    workers.shutdown()
    try:
        assert workers.pool() is None
        assert workers.parallel_map(square, [1, 2, 3], min_items=2) == [1, 4, 9]   # serial now
    finally:
        workers._reset()
        workers.prewarm()


def _same_cache(a, b):
    keys = ("T", "tri_fid", "tri_nrm", "tri_hidden", "p0", "e1", "e2", "A", "B", "corners", "mid", "bbox")
    for k in keys:
        x, y = np.asarray(a[k]), np.asarray(b[k])
        if x.shape != y.shape or not np.allclose(x, y, equal_nan=True):
            return False
    return set(a["face_tris"]) == set(b["face_tris"]) and all(
        np.allclose(a["face_tris"][f], b["face_tris"][f]) for f in b["face_tris"])


def test_render_warm_matches_serial_and_rebuilds_after_change(pool):
    doc = _model(30)
    ents = [g.ents for g in doc.groups]
    cache = GeoCache()
    cache.warm(ents)
    assert len(cache._c) == len(ents)                       # the pool warmed every group
    for e in ents:
        assert _same_cache(cache._c[id(e)], GeoCache._build(e))
    # after a change the cached entry is stale, and get() builds a fresh one
    e = ents[0]
    fid = next(iter(e.F))
    e.hidden_f.add(fid)
    e.touch()
    fresh = cache.get(e)
    assert fresh["tri_hidden"][fresh["tri_fid"] == fid].all()
    assert cache._c[id(e)] is fresh and fresh["version"] == e.version   # the stale warm entry is gone

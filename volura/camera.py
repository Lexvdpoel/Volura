"""Orbit camera with Z up (like SketchUp: red = X, green = Y, blue = Z)."""
from __future__ import annotations

import math

import numpy as np

from .geom import normalize

VIEWS = {
    "iso": (-50.0, 30.0),
    "top": (-90.0, 89.99),
    "bottom": (-90.0, -89.99),
    "front": (-90.0, 0.0),
    "back": (90.0, 0.0),
    "left": (180.0, 0.0),
    "right": (0.0, 0.0),
}

HORIZON_FAR = 1e5


def horizon_line(cam):
    """(y at the left edge, y at the right edge) of the horizon, in widget pixels.

    SketchUp draws the sky above the horizon and the grey ground below it. The
    horizon is where the view ray is horizontal. A value above the top (-) means
    the whole view is ground (looking down), a value below the bottom (+) means
    the whole view is sky (looking up); in parallel projection the view rays are
    all the same direction, so the horizon is always off screen.
    """
    f, r, u = cam.basis()
    w, h = max(1, cam.width), max(1, cam.height)
    if cam.ortho:
        if f[2] < -1e-9:
            return -HORIZON_FAR, -HORIZON_FAR
        if f[2] > 1e-9:
            return HORIZON_FAR, HORIZON_FAR
        return 0.5 * h, 0.5 * h
    t = math.tan(math.radians(cam.fov) / 2)
    if abs(u[2]) < 1e-9:
        return 0.5 * h, 0.5 * h
    a = t * (w / h) * r[2]
    b = t * u[2]
    ys = [(1 + (f[2] + a * nx) / b) * 0.5 * h for nx in (-1.0, 1.0)]
    return ys[0], ys[1]


def horizon_y(cam):
    """Screen y of the horizon in the middle of the view (see horizon_line)."""
    y0, y1 = horizon_line(cam)
    return 0.5 * (y0 + y1)


class Camera:
    def __init__(self):
        self.target = np.array([50.0, 50.0, 0.0])
        self.distance = 400.0
        self.yaw = -60.0
        self.pitch = 30.0
        self.fov = 35.0
        self.ortho = False
        self.width = 1
        self.height = 1
        self._d = None   # cached matrices, see _derived

    # ---------------------------------------------------------- matrices
    def _signature(self):
        """A key for every property the matrices below depend on.

        The camera is changed by assigning to its attributes (orbit, pan, zoom, fit,
        a resize or a direct ``cam.yaw = ...``), never in place, so comparing the
        current values with the cached ones detects any change without hooks."""
        t = self.target
        return (float(t[0]), float(t[1]), float(t[2]), float(self.distance),
                float(self.yaw), float(self.pitch), float(self.fov),
                bool(self.ortho), self.width, self.height)

    def _derived(self):
        """(sig, eye, f, r, u, view, proj, viewproj), rebuilt only when the camera moves.

        ``project`` is called many times per mouse move and every frame; without this
        it rebuilt the basis (a normalize and two cross products) and both matrices
        each time. The returned arrays are shared, so callers must not edit them."""
        sig = self._signature()
        d = self._d
        if d is not None and d[0] == sig:
            return d
        y, p = math.radians(self.yaw), math.radians(self.pitch)
        dirv = np.array([math.cos(p) * math.cos(y), math.cos(p) * math.sin(y), math.sin(p)])
        eye = self.target + dirv * self.distance
        f = normalize(self.target - eye)
        r = np.cross(f, [0.0, 0.0, 1.0])
        if np.linalg.norm(r) < 1e-6:
            r = np.array([-math.sin(y), math.cos(y), 0.0])
            if self.pitch > 0:
                r = -r
        r = normalize(r)
        u = np.cross(r, f)
        V = np.eye(4)
        V[0, :3], V[1, :3], V[2, :3] = r, u, -f
        V[:3, 3] = -V[:3, :3] @ eye
        aspect = self.width / max(1, self.height)
        near, far = self.clip()
        Pr = np.zeros((4, 4))
        if self.ortho:
            hh = self.half_height()
            w = hh * aspect
            far += self.distance
            near = -far
            Pr[0, 0], Pr[1, 1] = 1 / w, 1 / hh
            Pr[2, 2] = -2 / (far - near)
            Pr[2, 3] = -(far + near) / (far - near)
            Pr[3, 3] = 1
        else:
            t = 1 / math.tan(math.radians(self.fov) / 2)
            Pr[0, 0], Pr[1, 1] = t / aspect, t
            Pr[2, 2] = (far + near) / (near - far)
            Pr[2, 3] = 2 * far * near / (near - far)
            Pr[3, 2] = -1
        d = (sig, eye, f, r, u, V, Pr, Pr @ V)
        self._d = d
        return d

    @property
    def eye(self):
        return self._derived()[1]

    def basis(self):
        d = self._derived()
        return d[2], d[3], d[4]

    def view_matrix(self):
        return self._derived()[5]

    def clip(self):
        near = max(self.distance * 0.002, 0.01)
        far = self.distance * 50 + 5000.0
        return near, far

    def half_height(self):
        return self.distance * math.tan(math.radians(self.fov) / 2)

    def proj_matrix(self):
        return self._derived()[6]

    # ---------------------------------------------------------- mapping
    def ray(self, x, y):
        """World ray (origin, direction) through a widget pixel."""
        _, _, f, r, u = self._derived()[:5]
        aspect = self.width / max(1, self.height)
        nx = (2 * x / self.width - 1)
        ny = (1 - 2 * y / self.height)
        h = self.half_height()
        if self.ortho:
            o = self.eye + r * nx * h * aspect + u * ny * h - f * self.distance * 20
            return o, f
        t = math.tan(math.radians(self.fov) / 2)
        d = normalize(f + r * nx * t * aspect + u * ny * t)
        return self.eye.copy(), d

    def project(self, pts):
        """World points (n,3) -> screen (n,2) and a visibility mask."""
        pts = np.asarray(pts, dtype=float).reshape(-1, 3)
        P = self._derived()[7]
        h = np.c_[pts, np.ones(len(pts))] @ P.T
        w = h[:, 3]
        ok = w > 1e-9
        w = np.where(ok, w, 1.0)
        ndc = h[:, :2] / w[:, None]
        sx = (ndc[:, 0] + 1) * 0.5 * self.width
        sy = (1 - ndc[:, 1]) * 0.5 * self.height
        return np.c_[sx, sy], ok

    def pixel_size(self, p):
        """World size of one pixel at point p."""
        if self.ortho:
            return 2 * self.half_height() / max(1, self.height)
        f, _, _ = self.basis()
        depth = max(float(np.dot(np.asarray(p) - self.eye, f)), 1e-3)
        return 2 * depth * math.tan(math.radians(self.fov) / 2) / max(1, self.height)

    # ---------------------------------------------------------- state
    def state(self):
        """The whole camera as a plain tuple, for the view history."""
        return (tuple(float(v) for v in self.target), float(self.distance), float(self.yaw),
                float(self.pitch), float(self.fov), bool(self.ortho))

    def set_state(self, s):
        """Restore a state() tuple."""
        self.target = np.array(s[0], dtype=float)
        self.distance, self.yaw, self.pitch, self.fov = (float(s[1]), float(s[2]), float(s[3]),
                                                         float(s[4]))
        self.ortho = bool(s[5])

    # ---------------------------------------------------------- navigation
    def orbit(self, dx, dy):
        self.yaw -= dx * 0.4
        self.pitch = max(-89.99, min(89.99, self.pitch + dy * 0.4))

    def pan(self, dx, dy, ref=None):
        _, r, u = self.basis()
        s = self.pixel_size(self.target if ref is None else ref)
        self.target = self.target - r * dx * s + u * dy * s

    def zoom(self, factor, about=None):
        if about is not None:
            self.target = about + (self.target - about) * factor
        self.distance = max(0.5, self.distance * factor)

    def set_view(self, name):
        self.yaw, self.pitch = VIEWS[name]

    def fit(self, bbox):
        lo, hi = bbox
        self.target = (lo + hi) / 2
        radius = max(float(np.linalg.norm(hi - lo)) / 2, 5.0)
        aspect = self.width / max(1, self.height)
        half = math.radians(self.fov) / 2
        if aspect < 1:
            half = math.atan(math.tan(half) * aspect)
        self.distance = radius / math.sin(half) * 1.05

"""Eclipses: when a satellite around Earth is in the penumbra or umbra.

The Sun position comes from low-precision Almanac formulas (about 0.01°, equator of date) and the shadow
from a conical model with a spherical Earth of equatorial radius (Montenbruck & Gill 3.4.2).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy.optimize import brentq

from .bodies import Earth
from .core import sun as _sun
from .time import SECONDS_PER_DAY, Epoch

if TYPE_CHECKING:
    from .propagation import Trajectory

ASSUMPTIONS = (
    "Sun position: low-precision Astronomical Almanac formulas (Vallado algorithm 29), about 0.01°",
    "conical shadow of a spherical Earth with the equatorial radius (oblateness ignored)",
    "no atmospheric refraction or absorption at the shadow boundary",
)


@dataclass(frozen=True)
class Eclipse:
    """One passage through Earth's shadow. umbra_start/umbra_end are None for a penumbral-only pass."""

    penumbra_start: Epoch
    umbra_start: Epoch | None
    umbra_end: Epoch | None
    penumbra_end: Epoch
    truncated: bool = False          # the trajectory starts or ends inside the shadow

    @property
    def duration(self) -> float:
        """Time in the penumbra or umbra [s]."""
        return self.penumbra_end - self.penumbra_start

    @property
    def umbra_duration(self) -> float:
        """Time in the umbra [s]."""
        if self.umbra_start is None:
            return 0.0
        return self.umbra_end - self.umbra_start

    def __repr__(self) -> str:
        cut = ", truncated" if self.truncated else ""
        return (f"Eclipse({self.penumbra_start.utc[:19]} UTC, {self.duration / 60:.1f} min, "
                f"umbra {self.umbra_duration / 60:.1f} min{cut})")


@dataclass(frozen=True, eq=False)
class EclipseReport:
    """All eclipses of a trajectory and the sunlit fraction at every sample."""

    eclipses: tuple[Eclipse, ...]
    t: np.ndarray                    # sample times since the trajectory start [s]
    illumination: np.ndarray         # visible fraction of the Sun's disk at each sample (1 = sunlit)
    span: float                      # trajectory duration [s]
    assumptions: tuple[str, ...]

    def __post_init__(self):
        for name in ("t", "illumination"):
            arr = np.array(getattr(self, name), dtype=float)
            arr.flags.writeable = False
            object.__setattr__(self, name, arr)

    def __len__(self) -> int:
        return len(self.eclipses)

    def __iter__(self):
        return iter(self.eclipses)

    def __getitem__(self, i):
        return self.eclipses[i]

    @property
    def shadow_fraction(self) -> float:
        """Fraction of the trajectory's time spent in the penumbra or umbra."""
        return sum(e.duration for e in self.eclipses) / abs(self.span) if self.span else 0.0

    def explain(self):
        """Shadow geometry, eclipse statistics and assumptions."""
        from .explain import explain_eclipses

        return explain_eclipses(self)

    def plot(self, ax=None):
        """Sunlit fraction against time."""
        from .viz import plot_illumination

        return plot_illumination(self, ax)

    def __repr__(self) -> str:
        head = (f"{len(self)} eclipses over {self.span / SECONDS_PER_DAY:.3f} days "
                f"({100 * self.shadow_fraction:.1f}% of the time in shadow)")
        lines = [f"  {e!r}" for e in self.eclipses[:10]]
        if len(self) > 10:
            lines.append(f"  ... {len(self) - 10} more")
        return "\n".join([head, *lines])


def eclipses(tr: Trajectory) -> EclipseReport:
    """Eclipses of a trajectory around Earth. Implementation of `Trajectory.eclipses`."""
    if tr.body != Earth:
        raise ValueError(f"Eclipses need the Sun's position relative to the central body; only Earth "
                         f"is supported, not {tr.body.name}")
    t = np.asarray(tr.t, dtype=float)
    if len(t) < 2 or t[-1] <= t[0]:
        raise ValueError("Eclipse search needs a forward trajectory with at least two samples")
    R = tr.body.radius
    jd0 = tr.epoch.jd

    def sun_at(x):
        return _sun.sun_position(jd0 + x / SECONDS_PER_DAY)

    r_sun = sun_at(t)
    a, b, c = _sun.disk_angles(tr.r, r_sun, R)
    g_pen = c - (a + b)          # < 0 in penumbra or umbra
    g_umb = c - (b - a)          # < 0 in umbra
    lit = np.array([_sun.illumination(r, s, R) for r, s in zip(tr.r, r_sun, strict=True)])

    def boundary(func):
        def f(x):
            r, _ = tr.state_at(x)
            return float(func(r, sun_at(x), R))
        return f

    f_pen, f_umb = boundary(_sun.penumbra_function), boundary(_sun.umbra_function)

    def crossings(g, f):
        """Times where g changes sign: (time, entering) pairs, refined by root finding."""
        out = []
        for k in np.nonzero(np.sign(g[1:]) != np.sign(g[:-1]))[0]:
            if g[k] == 0.0:
                continue
            out.append((brentq(f, t[k], t[k + 1], xtol=1e-3), g[k] > 0.0))
        return out

    pen = crossings(g_pen, f_pen)
    umb = crossings(g_umb, f_umb)
    if g_pen[0] < 0.0:
        pen.insert(0, (t[0], True))
    if g_pen[-1] < 0.0:
        pen.append((t[-1], False))
    if g_umb[0] < 0.0:
        umb.insert(0, (t[0], True))
    if g_umb[-1] < 0.0:
        umb.append((t[-1], False))

    found = []
    start = tr.epoch
    for k in range(0, len(pen) - 1, 2):
        (t_in, entering), (t_out, _) = pen[k], pen[k + 1]
        if not entering:
            continue
        inside = [x for x, _ in umb if t_in <= x <= t_out]
        u_in = start + float(inside[0]) if inside else None
        u_out = start + float(inside[-1]) if len(inside) >= 2 else None
        if u_in is not None and u_out is None:
            u_out = start + float(t_out)
        found.append(Eclipse(start + float(t_in), u_in, u_out, start + float(t_out),
                             truncated=t_in == t[0] or t_out == t[-1]))
    return EclipseReport(tuple(found), t, lit, float(t[-1] - t[0]), ASSUMPTIONS)

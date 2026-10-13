"""Orbital lifetime under atmospheric drag."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import astropy.units as u
import numpy as np

from . import units
from .core import lifetime as _life
from .core import twobody as _tb
from .environment import as_environment
from .time import SECONDS_PER_DAY, Epoch

if TYPE_CHECKING:
    from .orbits import Orbit

# Largest eccentricity for the orbit-averaged (circular) method
AVERAGED_MAX_ECC = 0.02


@dataclass(frozen=True, eq=False)
class Lifetime:
    """Result of a lifetime estimate. reentry is None if the orbit outlived the time limit."""

    initial: Orbit
    method: str
    days: float                      # time until reentry, or the time limit
    reentry: Epoch | None
    reentry_altitude: float          # [m]
    t: np.ndarray                    # time since start [s]
    altitude: np.ndarray             # mean altitude [m] (averaged method) or altitude [m] (numerical)
    assumptions: tuple[str, ...]

    def __post_init__(self):
        for name in ("t", "altitude"):
            arr = np.array(getattr(self, name), dtype=float)
            arr.flags.writeable = False
            object.__setattr__(self, name, arr)

    @property
    def reentered(self) -> bool:
        return self.reentry is not None

    def explain(self):
        """Decay model, formulas, result and assumptions."""
        from .explain import explain_lifetime

        return explain_lifetime(self)

    def plot(self, ax=None):
        """Altitude against time."""
        from .viz import plot_lifetime

        return plot_lifetime(self, ax)

    def __repr__(self) -> str:
        if self.reentered:
            return (f"Lifetime({self.days:.1f} days, reentry {self.reentry.utc[:10]} UTC, "
                    f"method={self.method})")
        return f"Lifetime(more than {self.days:.1f} days, method={self.method})"


def lifetime(orbit: Orbit, *, cd: float = 2.2, area, mass, density=None, method: str = "averaged",
             reentry_altitude=100e3, max_days=3650.0) -> Lifetime:
    """Estimate the orbital lifetime. Implementation of `Orbit.lifetime`.

    method="averaged" integrates the orbit-averaged semi-major axis of a near-circular orbit
    (fast: years in about a second). method="numerical" propagates the full j2+drag model.
    """
    body = orbit.body
    area_si = units.to_si(area, units.AREA)
    mass_si = units.to_si(mass, units.MASS)
    for name, value in (("cd", cd), ("area", area_si), ("mass", mass_si)):
        if not value > 0.0:
            raise ValueError(f"{name} must be positive: {value}")
    if density is not None:
        rho = as_environment(density, units.DENSITY)
    elif body.atmosphere is not None:
        rho = body.atmosphere
    else:
        raise ValueError(f"{body.name} has no atmosphere model; pass density")
    h_stop = units.to_si(reentry_altitude, units.LENGTH)
    t_max = units.to_si(max_days, u.day) * SECONDS_PER_DAY      # plain numbers are days
    if not t_max > 0.0:
        raise ValueError(f"max_days must be positive: {max_days}")
    ballistic = cd * area_si / mass_si
    common = [
        f"atmospheric density: {rho.description}",
        f"constant drag coefficient and area (Cd={cd:g}, A={area_si:g} m², m={mass_si:g} kg)",
        "real lifetimes vary several-fold with solar activity; treat the result as an order of magnitude "
        "for the assumed density",
    ]

    if method == "averaged":
        if orbit.ecc > AVERAGED_MAX_ECC:
            raise ValueError(f"The averaged method is for near-circular orbits (e ≤ {AVERAGED_MAX_ECC}); "
                             f"e = {orbit.ecc:.3f}. Use method='numerical'")
        a0 = _mean_semi_major_axis(orbit)
        correction = _j2_correction(orbit, a0, body.radius + h_stop, ballistic, rho)
        res = _life.decay_history(a0, orbit.inc, body.mu, body.rotation_rate, ballistic, rho,
                                  body.radius + h_stop, t_max, orbit.epoch.tdb_seconds, correction)
        days = res.t[-1] / SECONDS_PER_DAY
        assumptions = (
            "orbit-averaged decay of a circular orbit: da/dt = 2 a_T / n averaged over the argument of latitude",
            "starts from the semi-major axis averaged over one J2 orbit (osculating a swings by kilometers)",
            "decay rate corrected by its ratio to the rate averaged along J2 orbits at several altitudes "
            "(the J2 orbit radius varies with latitude, which changes the mean density)",
            f"reentry when the mean altitude falls below {h_stop / 1e3:g} km",
            *common,
        )
        return Lifetime(orbit, "averaged", days, orbit.epoch + float(res.t[-1]) if res.reached else None,
                        h_stop, res.t, res.a - body.radius, assumptions)

    if method == "numerical":
        model = "j2+drag" if body.J2 else "drag"
        tr = orbit.propagate(t_max, model=model, cd=cd, area=area_si, mass=mass_si, density=density)
        alt = tr.altitude
        below = np.nonzero(alt < h_stop)[0]
        if len(below):
            k = below[0]
            days = tr.t[k] / SECONDS_PER_DAY
            reentry = orbit.epoch + float(tr.t[k])
        else:
            days, reentry = tr.t[-1] / SECONDS_PER_DAY, None
        assumptions = (
            f"full numerical propagation ({tr.info.model}, {tr.info.integrator})",
            f"reentry at the first sample below {h_stop / 1e3:g} km altitude",
            *common,
        )
        return Lifetime(orbit, "numerical", days, reentry, h_stop, tr.t, alt, assumptions)

    raise ValueError(f"Unknown method: {method!r} (averaged, numerical)")


def _j2_correction(orbit: Orbit, a0: float, r_stop: float, ballistic: float, rho):
    """K(a): drag decay rate averaged along a J2 orbit divided by the circular-orbit rate.

    Evaluated on one-orbit J2 propagations at six altitudes between the start and the stop radius,
    then interpolated linearly in a. Returns None when the body has no J2.
    """
    body = orbit.body
    if not body.J2:
        return None
    from .core import forces as _forces
    from .core import propagate as _prop
    from .core import elements as _el

    t0 = orbit.epoch.tdb_seconds
    shift = orbit.a - a0                       # osculating minus mean, for a circular start at the node
    levels = np.linspace(a0, max(r_stop + 20e3, a0 - (a0 - r_stop) * 0.95), 6)
    a_means, factors = [], []
    for a_target in levels:
        r_start = a_target + shift
        r0, v0 = _el.coe_to_rv(r_start, 0.0, orbit.inc, 0.0, 0.0, 0.0, body.mu)
        period = 2.0 * np.pi * np.sqrt(r_start**3 / body.mu)

        def accel(t, r, v):
            return _forces.accel_twobody(r, body.mu) + _forces.accel_j2(r, body.mu, body.radius, body.J2)

        res = _prop.cowell(r0, v0, period, accel, t_eval=np.linspace(0.0, period, 181))
        r, v = res.r[:-1], res.v[:-1]
        a_mean = float(np.mean(_tb.semi_major_axis(r, v, body.mu)))
        true = _life.sampled_decay_rate(r, v, body.mu, body.rotation_rate, ballistic, rho, t0)
        ideal = _life.mean_decay_rate(a_mean, orbit.inc, body.mu, body.rotation_rate, ballistic, rho, t0)
        a_means.append(a_mean)
        factors.append(true / ideal if ideal != 0.0 else 1.0)
    order = np.argsort(a_means)
    xs, ys = np.array(a_means)[order], np.array(factors)[order]
    return lambda a: float(np.interp(a, xs, ys))


def _mean_semi_major_axis(orbit: Orbit) -> float:
    """Semi-major axis averaged over one orbit of J2 motion (the osculating value if J2 = 0)."""
    if not orbit.body.J2:
        return orbit.a
    tr = orbit.propagate(orbit.period, model="j2", n_points=361)
    a = _tb.semi_major_axis(tr.r[:-1], tr.v[:-1], orbit.body.mu)
    return float(np.mean(a))

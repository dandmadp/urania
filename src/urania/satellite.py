"""Satellite operations helpers: relative motion, Doppler shift, coverage footprint, beta angle.

Implementations of `Orbit.relative_to`, `Trajectory.relative_to`, `Trajectory.doppler`, `Orbit.footprint`,
`Orbit.beta_angle` and friends. Formulas live in core.relative, core.coverage and core.sun.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, NamedTuple

import numpy as np

from . import units
from .bodies import Earth
from .core import coverage as _cov
from .core import frames as _fr
from .core import propagate as _prop
from .core import relative as _rel
from .core import sun as _sun

if TYPE_CHECKING:
    from .ground import GroundStation
    from .orbits import Orbit
    from .propagation import Trajectory

C_LIGHT = 299792458.0  # [m/s]


def _ro(x) -> np.ndarray:
    a = np.array(x, dtype=float)
    a.flags.writeable = False
    return a


# ---------------------------------------------------------------- relative motion

@dataclass(frozen=True, eq=False)
class RelativeMotion:
    """Deputy position and velocity in the chief's LVLH frame (x radial, y along-track, z normal) [m, m/s]."""

    chief: str
    t: np.ndarray        # time since the deputy trajectory's start [s]
    r: np.ndarray        # N×3
    v: np.ndarray        # N×3

    def __post_init__(self):
        for name in ("t", "r", "v"):
            object.__setattr__(self, name, _ro(getattr(self, name)))

    @property
    def distance(self) -> np.ndarray:
        """Separation from the chief [m]."""
        return np.linalg.norm(self.r, axis=1)

    def plot(self, ax=None):
        """Along-track (x axis) against radial (y axis) motion, the usual proximity-operations view."""
        from .viz import plot_relative

        return plot_relative(self, ax)

    def __repr__(self) -> str:
        d = self.distance
        return (f"RelativeMotion(relative to {self.chief}, {len(self.t)} samples, "
                f"distance {d.min():,.1f} to {d.max():,.1f} m)")


def _chief_states(chief, epoch0, t) -> tuple[np.ndarray, np.ndarray]:
    from .orbits import Orbit

    t = np.asarray(t, dtype=float)
    if isinstance(chief, Orbit):
        return _prop.kepler_states(chief.r, chief.v, (epoch0 - chief.epoch) + t, chief.body.mu)
    offset = epoch0 - chief.epoch
    tc = offset + t
    if tc.min() < chief.t[0] - 1e-6 or tc.max() > chief.t[-1] + 1e-6:
        raise ValueError("The chief trajectory does not cover the deputy's time span")
    states = [chief.state_at(float(x)) for x in tc]
    return np.array([s[0] for s in states]), np.array([s[1] for s in states])


def relative_state(deputy: Orbit, chief) -> tuple[np.ndarray, np.ndarray]:
    """LVLH position and velocity of an orbit relative to a chief (Orbit, propagated two-body, or Trajectory)."""
    if deputy.body != chief.body:
        raise ValueError(f"Central bodies differ: {deputy.body.name}, {chief.body.name}")
    rc, vc = _chief_states(chief, deputy.epoch, [0.0])
    return _rel.to_lvlh(rc[0], vc[0], deputy.r, deputy.v)


def relative_motion(tr: Trajectory, chief) -> RelativeMotion:
    """Relative motion of a trajectory with respect to a chief at every sample."""
    if tr.body != chief.body:
        raise ValueError(f"Central bodies differ: {tr.body.name}, {chief.body.name}")
    rc, vc = _chief_states(chief, tr.epoch, tr.t)
    pairs = [_rel.to_lvlh(rc[k], vc[k], tr.r[k], tr.v[k]) for k in range(len(tr.t))]
    name = getattr(chief, "name", None) or "chief"
    return RelativeMotion(name, tr.t, np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs]))


# ---------------------------------------------------------------- Doppler

@dataclass(frozen=True, eq=False)
class Doppler:
    """Range, range rate and Doppler shift seen from a ground station at every trajectory sample."""

    station: GroundStation
    frequency: float     # transmitted frequency [Hz]
    t: np.ndarray        # [s]
    range: np.ndarray    # [m]
    range_rate: np.ndarray   # [m/s], positive when receding
    shift: np.ndarray    # received minus transmitted frequency [Hz]
    elevation: np.ndarray    # [rad]

    def __post_init__(self):
        for name in ("t", "range", "range_rate", "shift", "elevation"):
            object.__setattr__(self, name, _ro(getattr(self, name)))

    def plot(self, ax=None, min_elevation=0.0):
        """Doppler shift [kHz] against time while the satellite is above min_elevation."""
        from .viz import plot_doppler

        return plot_doppler(self, ax, units.to_si(min_elevation, units.ANGLE))

    def __repr__(self) -> str:
        up = self.elevation > 0
        span = (f", ±{np.abs(self.shift[up]).max() / 1e3:.2f} kHz while visible" if up.any() else ", never visible")
        return f"Doppler({self.station.name}, {self.frequency / 1e6:.3f} MHz{span})"


def doppler(tr: Trajectory, station: GroundStation, frequency) -> Doppler:
    """Doppler shift of a carrier of the given frequency, received at the station (first order, -f ṙ/c)."""
    from .ground import _EarthRotation, _require_earth

    _require_earth(tr.body)
    f = units.to_si(frequency, units.u.Hz)
    if not f > 0:
        raise ValueError(f"Frequency must be positive: {f}")
    rot = _EarthRotation(tr)
    site_ecef = station.r_ecef
    site = _fr.rotate_z(np.broadcast_to(site_ecef, tr.r.shape), -rot.theta)
    omega = np.array([0.0, 0.0, Earth.rotation_rate])
    site_v = np.cross(omega, site)
    d = tr.r - site
    rng = np.linalg.norm(d, axis=1)
    rr = np.sum(d * (tr.v - site_v), axis=1) / rng
    _, el, _ = _fr.look_angles(_fr.rotate_z(tr.r, rot.theta), site_ecef, station.lat, station.lon)
    return Doppler(station, f, tr.t, rng, rr, -f * rr / C_LIGHT, el)


# ---------------------------------------------------------------- footprint and Sun geometry

class Footprint(NamedTuple):
    """Visible area of a satellite on a spherical body."""

    central_angle: float   # Earth central angle λ from the sub-satellite point to the edge [rad]
    radius: float          # ground distance to the edge, R·λ [m]
    slant_range: float     # distance from the satellite to the edge [m]
    edge_elevation: float  # elevation of the satellite seen from the edge [rad]
    nadir_angle: float     # off-nadir angle of the edge seen from the satellite [rad]
    area: float            # spherical cap area 2πR²(1 - cos λ) [m²]

    def __repr__(self) -> str:
        return (f"Footprint(radius {self.radius / 1e3:,.0f} km, slant range {self.slant_range / 1e3:,.0f} km, "
                f"edge elevation {math.degrees(self.edge_elevation):.1f}°)")


def footprint(orbit: Orbit, min_elevation=0.0, half_angle=None) -> Footprint:
    """Footprint at the orbit's current altitude, for a minimum elevation or a nadir sensor half-angle."""
    R = orbit.body.radius
    h = float(np.linalg.norm(orbit.r)) - R
    if half_angle is None:
        eps = units.to_si(min_elevation, units.ANGLE)
        lam, eta, D = _cov.footprint(h, R, eps)
    else:
        eta = units.to_si(half_angle, units.ANGLE)
        lam, eps, D = _cov.sensor_footprint(h, R, eta)
        eta = min(eta, _cov.angular_radius(h, R))
    return Footprint(lam, R * lam, D, eps, eta, 2 * math.pi * R**2 * (1 - math.cos(lam)))


def _require_earth_sun(body) -> None:
    if body != Earth:
        raise ValueError(f"The Sun position model is geocentric; only Earth orbits are supported, not {body.name}")


def beta_angle(orbit: Orbit) -> float:
    """Beta angle [rad] at the orbit's epoch."""
    _require_earth_sun(orbit.body)
    return float(_sun.beta_angle(orbit.r, orbit.v, _sun.sun_position(orbit.epoch.jd)))


def trajectory_beta_angle(tr: Trajectory) -> np.ndarray:
    """Beta angle [rad] at every sample."""
    _require_earth_sun(tr.body)
    r_sun = _sun.sun_position(tr.epoch.jd + np.asarray(tr.t) / 86400.0)
    return _ro(_sun.beta_angle(tr.r, tr.v, r_sun))


def eclipse_fraction(orbit: Orbit) -> float:
    """Fraction of the orbit in Earth's shadow from the beta angle (circular orbit, cylindrical shadow)."""
    return _sun.eclipse_fraction(beta_angle(orbit), orbit.a, orbit.body.radius)

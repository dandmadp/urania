"""Orbit objects and presets."""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from functools import cached_property

import astropy.units as u
import numpy as np

from . import units
from .bodies import Body, Earth
from .constants import SSO_RAAN_RATE
from .core import elements as _el
from .core import j2 as _j2
from .core import kepler as _kepler
from .core import twobody as _tb
from .time import J2000, Epoch


# name → (display unit, SI unit) for Orbit.qty
def _body(body) -> Body:
    """A Body, or a body name such as "Mars"."""
    if isinstance(body, str):
        from .bodies import get

        return get(body)
    return body


def _has_unit(value) -> bool:
    if isinstance(value, (u.Quantity, str)):
        return True
    return isinstance(value, (list, tuple)) and any(isinstance(x, (u.Quantity, str)) for x in value)


_QTY_UNITS = {
    "r": (u.km, u.m), "v": (u.km / u.s, u.m / u.s),
    "a": (u.km, u.m), "p": (u.km, u.m), "r_periapsis": (u.km, u.m), "r_apoapsis": (u.km, u.m),
    "periapsis_altitude": (u.km, u.m), "apoapsis_altitude": (u.km, u.m),
    "ecc": (u.dimensionless_unscaled, u.dimensionless_unscaled),
    "inc": (u.deg, u.rad), "raan": (u.deg, u.rad), "argp": (u.deg, u.rad), "nu": (u.deg, u.rad),
    "M": (u.deg, u.rad),
    "period": (u.min, u.s), "mean_motion": (u.deg / u.s, u.rad / u.s),
    "energy": (u.km**2 / u.s**2, u.J / u.kg), "h": (u.km**2 / u.s, u.m**2 / u.s),
    "raan_rate": (u.deg / u.day, u.rad / u.s), "argp_rate": (u.deg / u.day, u.rad / u.s),
}


class OrbitQuantities:
    """Orbit properties as astropy Quantities in convenient units (km, deg, min, ...).

    ``orbit.qty.a`` → ``<Quantity 6798.137 km>``; ``print(orbit.qty)`` lists them all.
    Values that are undefined for the orbit (e.g. the period of a hyperbola) are skipped in the list.
    """

    def __init__(self, orbit: Orbit):
        self._orbit = orbit

    def __getattr__(self, name: str):
        if name.startswith("_") or name not in _QTY_UNITS:
            raise AttributeError(f"No quantity {name!r}; available: {', '.join(_QTY_UNITS)}")
        unit, si = _QTY_UNITS[name]
        return (getattr(self._orbit, name) * si).to(unit)

    def __dir__(self):
        return list(_QTY_UNITS)

    def __repr__(self) -> str:
        lines = [repr(self._orbit)]
        for name in _QTY_UNITS:
            try:
                q = getattr(self, name)
            except (ValueError, ZeroDivisionError):
                continue
            if q.isscalar:
                lines.append(f"  {name:<19} {q.value:,.6g} {q.unit}")
            else:
                lines.append(f"  {name:<19} [{', '.join(f'{x:,.6g}' for x in q.value)}] {q.unit}")
        return "\n".join(lines)


def _frozen_vector(x) -> np.ndarray:
    arr = np.array(x, dtype=float)
    if arr.shape != (3,):
        raise ValueError(f"Expected a 3-vector: shape={arr.shape}")
    arr.flags.writeable = False
    return arr


@dataclass(frozen=True, eq=False)
class Orbit:
    """An orbit around a central body, defined by the state vector (r, v) at an epoch.

    Orbital elements and derived quantities are computed from the state vector by core functions.
    All properties return SI numbers.
    """

    body: Body
    r: np.ndarray        # position [m], inertial frame
    v: np.ndarray        # velocity [m/s]
    epoch: Epoch = field(default=J2000)

    def __post_init__(self):
        if isinstance(self.body, str):
            from .bodies import get

            object.__setattr__(self, "body", get(self.body))
        object.__setattr__(self, "epoch", Epoch.parse(self.epoch))
        object.__setattr__(self, "r", _frozen_vector(self.r))
        object.__setattr__(self, "v", _frozen_vector(self.v))

    # ---------------------------------------------------------- construction

    @classmethod
    def from_vectors(cls, body: Body, r, v, epoch: Epoch = J2000) -> Orbit:
        """From a state vector, given as Quantities or SI numbers."""
        r_si = units.to_si(r, units.LENGTH)
        body = _body(body)
        if not _has_unit(r) and 0.0 < np.linalg.norm(r_si) < 0.01 * body.radius:
            warnings.warn(f"The position is a plain number vector of length {np.linalg.norm(r_si):g} m, inside "
                          f"{body.name}. Plain numbers are meters; for kilometers multiply by u.km",
                          units.UnitWarning, stacklevel=units._user_stacklevel())
        return cls(body, r_si, units.to_si(v, units.VELOCITY), epoch)

    @classmethod
    def from_elements(cls, body: Body, *, a=None, p=None, ecc: float = 0.0,
                      inc=0.0, raan=0.0, argp=0.0, nu=0.0,
                      epoch: Epoch = J2000) -> Orbit:
        """From classical orbital elements. Give the size as either a (semi-major axis) or p (semi-latus rectum).

        Hyperbolas use a < 0; parabolas (ecc = 1) can only be given with p.
        Angle conventions for circular and equatorial orbits follow core.elements.
        """
        body = _body(body)
        if (a is None) == (p is None):
            raise ValueError("Specify exactly one of a and p")
        size = a if p is None else p
        if not _has_unit(size) and 0.0 < abs(units.to_si(size, units.LENGTH)) < 0.01 * body.radius:
            warnings.warn(f"{'a' if p is None else 'p'}={units.to_si(size, units.LENGTH):g} is a plain number, "
                          f"so it is read as meters, far smaller than {body.name}'s radius. For kilometers write "
                          f"{units.to_si(size, units.LENGTH):g} * u.km", units.UnitWarning, stacklevel=units._user_stacklevel())
        if ecc < 0.0:
            raise ValueError(f"Eccentricity must be non-negative: ecc={ecc}")
        if p is None:
            a = units.to_si(a, units.LENGTH)
            if ecc == 1.0:
                raise ValueError("Specify a parabolic orbit with p")
            if (a > 0) != (ecc < 1.0):
                raise ValueError(f"a={a} m and ecc={ecc} are inconsistent "
                                 "(ellipses need a > 0, hyperbolas a < 0)")
            p = a * (1.0 - ecc**2)
        else:
            p = units.to_si(p, units.LENGTH)
        r, v = _el.coe_to_rv(
            p, ecc,
            units.to_si(inc, units.ANGLE),
            units.to_si(raan, units.ANGLE),
            units.to_si(argp, units.ANGLE),
            units.to_si(nu, units.ANGLE),
            body.mu,
        )
        return cls(body, r, v, epoch)

    @classmethod
    def circular(cls, body: Body, altitude, *, inc=0.0, raan=0.0, arglat=0.0,
                 epoch: Epoch = J2000) -> Orbit:
        """Circular orbit. altitude is above the equatorial radius.

        arglat is the angle from the ascending node to the position (from the x axis for an
        equatorial orbit).
        """
        body = _body(body)
        a = body.radius + units.altitude(altitude)
        return cls.from_elements(body, a=a, inc=inc, raan=raan, nu=arglat, epoch=epoch)

    @classmethod
    def from_apsides(cls, body: Body, periapsis_altitude, apoapsis_altitude, *, inc=0.0,
                     raan=0.0, argp=0.0, nu=0.0, epoch: Epoch = J2000) -> Orbit:
        """Elliptic orbit from its periapsis and apoapsis altitudes above the equatorial radius.

        Example: a 400 x 600 km orbit is ``Orbit.from_apsides(Earth, 400 * u.km, 600 * u.km)``.
        """
        body = _body(body)
        rp = body.radius + units.altitude(periapsis_altitude, "periapsis_altitude")
        ra = body.radius + units.altitude(apoapsis_altitude, "apoapsis_altitude")
        a, ecc = _tb.apsides_to_ae(rp, ra)
        return cls.from_elements(body, a=a, ecc=ecc, inc=inc, raan=raan, argp=argp, nu=nu,
                                 epoch=epoch)

    # ---------------------------------------------------------- orbital elements

    @property
    def qty(self) -> OrbitQuantities:
        """Properties as Quantities in km, deg, min, ...: ``orbit.qty.a``, ``print(orbit.qty)``."""
        return OrbitQuantities(self)

    def summary(self) -> str:
        """All properties in convenient units, one per line."""
        return repr(self.qty)

    @property
    def altitude(self) -> float:
        """Current altitude above the equatorial radius [m]."""
        return float(np.linalg.norm(self.r)) - self.body.radius

    @cached_property
    def elements(self) -> _el.Elements:
        return _el.rv_to_coe(self.r, self.v, self.body.mu)

    @property
    def p(self) -> float:
        """Semi-latus rectum [m]."""
        return self.elements.p

    @property
    def a(self) -> float:
        """Semi-major axis [m]. inf for a parabola, negative for a hyperbola."""
        return self.elements.a

    @property
    def ecc(self) -> float:
        """Eccentricity."""
        return self.elements.ecc

    @property
    def inc(self) -> float:
        """Inclination [rad]."""
        return self.elements.inc

    @property
    def raan(self) -> float:
        """Right ascension of the ascending node [rad]."""
        return self.elements.raan

    @property
    def argp(self) -> float:
        """Argument of periapsis [rad]."""
        return self.elements.argp

    @property
    def nu(self) -> float:
        """True anomaly [rad]."""
        return self.elements.nu

    @property
    def M(self) -> float:
        """Mean anomaly [rad]."""
        return _kepler.true_to_mean(self.nu, self.ecc)

    # ---------------------------------------------------------- derived quantities

    @property
    def period(self) -> float:
        """Orbital period [s]. Elliptic orbits only."""
        return _tb.period(self.a, self.body.mu)

    @property
    def mean_motion(self) -> float:
        """Mean motion [rad/s]."""
        return _tb.mean_motion(self.a, self.body.mu)

    @property
    def energy(self) -> float:
        """Specific mechanical energy [J/kg]."""
        return _tb.specific_energy(self.a, self.body.mu)

    @property
    def h(self) -> float:
        """Specific angular momentum magnitude [m²/s]."""
        return math.sqrt(self.p * self.body.mu)

    @property
    def r_periapsis(self) -> float:
        """Periapsis radius [m]."""
        return self.p / (1.0 + self.ecc)

    @property
    def r_apoapsis(self) -> float:
        """Apoapsis radius [m]. inf if not elliptic."""
        if self.ecc >= 1.0:
            return math.inf
        return self.p / (1.0 - self.ecc)

    @property
    def periapsis_altitude(self) -> float:
        """Periapsis altitude [m] above the equatorial radius."""
        return self.r_periapsis - self.body.radius

    @property
    def apoapsis_altitude(self) -> float:
        """Apoapsis altitude [m] above the equatorial radius."""
        return self.r_apoapsis - self.body.radius

    @property
    def raan_rate(self) -> float:
        """Mean nodal rate due to J2 [rad/s]."""
        b = self.body
        return _j2.raan_rate(self.a, self.ecc, self.inc, b.mu, b.radius, b.J2)

    @property
    def argp_rate(self) -> float:
        """Mean apsidal rate due to J2 [rad/s]."""
        b = self.body
        return _j2.argp_rate(self.a, self.ecc, self.inc, b.mu, b.radius, b.J2)

    # ---------------------------------------------------------- propagation

    def propagate(self, duration=None, *, days=None, hours=None, minutes=None, model="twobody", **kwargs):
        """Propagate the orbit and return a Trajectory. The orbit itself is unchanged.

        Args:
            duration: propagation time (seconds, a time Quantity or a string such as "90 min"),
                or one of days, hours, minutes (plain numbers in that unit).
            model: "twobody", "j2", "j2+drag" or fidelity 0 to 2
            **kwargs: cd, area, mass, density (constant, function or Environment),
                method, rtol, atol, n_points. See `propagation.propagate`.

        Example: ``ISS.propagate(days=30, model="j2+drag", area=1500, mass=420000).final``
        """
        from .propagation import propagate, resolve_duration

        return propagate(self, resolve_duration(duration, days, hours, minutes), model=model, **kwargs)

    def after(self, duration=None, *, days=None, hours=None, minutes=None, model="twobody", **kwargs) -> Orbit:
        """The orbit after propagating; shorthand for ``propagate(...).final``.

        Raises if the propagation stops early (e.g. on reaching the surface) instead of
        returning the orbit at the stopping point.

        Example: ``ISS.after(days=1, model="j2")``
        """
        from .propagation import propagate, resolve_duration

        duration = resolve_duration(duration, days, hours, minutes)
        tr = propagate(self, duration, model=model, **kwargs)
        if tr.info.terminated:
            raise RuntimeError(f"Propagation stopped early ({tr.info.terminated}) after "
                               f"{tr.t[-1]:.1f} s of {duration:.1f} s")
        return tr.final

    def lifetime(self, *, area, mass, cd: float = 2.2, density=None, method: str = "averaged",
                 reentry_altitude=100e3, max_days=3650.0):
        """Estimate how long the orbit survives atmospheric drag; returns a Lifetime.

        method="averaged" (default) integrates the orbit-averaged decay of a near-circular orbit and
        takes about a second for years; method="numerical" propagates the full j2+drag model.
        max_days (days, or a time Quantity) limits the search.

        Example: ``Orbit.circular(Earth, 400 * u.km, inc=51.6 * u.deg).lifetime(area=0.03, mass=4)``
        """
        from .lifetime import lifetime

        return lifetime(self, cd=cd, area=area, mass=mass, density=density, method=method,
                        reentry_altitude=reentry_altitude, max_days=max_days)

    # ---------------------------------------------------------- maneuvers

    def transfer_to(self, target: Orbit, method: str = "hohmann", *, rb=None,
                    plane_split="optimal"):
        """Compute a transfer to the circular orbit target and return a Transfer.

        Args:
            method: "hohmann" or "bielliptic"
            rb: intermediate apoapsis radius of a bi-elliptic transfer (length)
            plane_split: fraction of the plane change done by the first Hohmann burn
                (0 to 1), or "optimal" (minimum total Δv)

        Example: ``ISS.transfer_to(GEO).total_dv``
        """
        from .maneuvers import transfer

        return transfer(self, target, method, rb=rb, plane_split=plane_split)

    def rendezvous(self, target: Orbit, tof, *, wait=0.0, revolutions: int = 0, branch: str = "left"):
        """Two-burn Lambert rendezvous with the spacecraft on `target` after `tof` (any orbits); a Transfer.

        wait: coast before the first burn. revolutions/branch select multi-revolution Lambert arcs.
        Example: ``chaser.rendezvous(target, 45 * u.min)``
        """
        from .maneuvers import rendezvous

        return rendezvous(self, target, tof, wait=wait, revolutions=revolutions, branch=branch)

    def phasing(self, angle=None, *, target: Orbit | None = None, revolutions: int = 1):
        """Move `angle` ahead in the same circular orbit (negative: behind) with a phasing orbit; a Transfer.

        With target=, catch up with a spacecraft in the same orbit.
        """
        from .maneuvers import phasing

        return phasing(self, angle, target=target, revolutions=revolutions)

    def deorbit(self, perigee_altitude="50 km"):
        """One retrograde burn that lowers periapsis to perigee_altitude (default 50 km); a Transfer."""
        from .maneuvers import deorbit

        return deorbit(self, perigee_altitude)

    # ---------------------------------------------------------- operations

    def relative_to(self, chief):
        """Position and velocity relative to a chief Orbit or Trajectory, in the chief's LVLH frame
        (x radial, y along-track, z orbit normal) [m, m/s]."""
        from .satellite import relative_state

        return relative_state(self, chief)

    def footprint(self, min_elevation=0.0, half_angle=None):
        """Visible ground area at the current altitude, above min_elevation or for a nadir sensor half-angle."""
        from .satellite import footprint

        return footprint(self, min_elevation, half_angle)

    def beta_angle(self) -> float:
        """Angle between the orbit plane and the Sun direction at the epoch [rad] (Earth orbits)."""
        from .satellite import beta_angle

        return beta_angle(self)

    def eclipse_fraction(self) -> float:
        """Fraction of the orbit in Earth's shadow, from the beta angle (circular orbit, cylindrical shadow)."""
        from .satellite import eclipse_fraction

        return eclipse_fraction(self)

    # ---------------------------------------------------------- explanation and plotting

    def explain(self):
        """Show step by step how the orbital elements follow from the state vector."""
        from .explain import explain_orbit

        return explain_orbit(self)

    def plot(self, ax=None):
        """Plot the orbit in 2D in its own plane. Returns the matplotlib Axes."""
        from .viz import plot_orbit

        return plot_orbit(self, ax)

    def __repr__(self) -> str:
        return (f"Orbit({self.body.name}, a={self.a / 1e3:.1f} km, "
                f"ecc={self.ecc:.4f}, inc={math.degrees(self.inc):.2f}°, "
                f"epoch={self.epoch.iso})")


def sun_synchronous(altitude, *, ecc: float = 0.0, raan=0.0, epoch: Epoch = J2000,
                    body: Body = Earth, raan_rate=None) -> Orbit:
    """Sun-synchronous orbit: the inclination makes the J2 nodal rate follow the Sun.

    For Earth the nodal rate is 360° per tropical year; for another planet, 360° per its orbital period
    around the Sun (``body.orbital_period``). Pass raan_rate to override.
    """
    body = _body(body)
    a = body.radius + units.altitude(altitude)
    if raan_rate is None:
        if body == Earth:
            rate = SSO_RAAN_RATE
        elif body.parent is not None and body.parent.parent is None:
            rate = 2.0 * math.pi / body.orbital_period
        else:
            raise ValueError(f"Pass raan_rate for {body.name}: 360° per its year around the Sun, in rad/s")
    else:
        rate = units.to_si(raan_rate, units.ANGULAR_VELOCITY)
    inc = _j2.sso_inclination(a, ecc, body.mu, body.radius, body.J2, rate)
    return Orbit.from_elements(body, a=a, ecc=ecc, inc=inc, raan=raan, epoch=epoch)


# ---------------------------------------------------------------- presets (Earth)
# Representative values. Use a TLE (TLEOrbit) for the actual ISS position.

LEO = Orbit.circular(Earth, 500e3)
ISS = Orbit.circular(Earth, 420e3, inc=math.radians(51.64))
SSO = sun_synchronous(700e3)
GEO = Orbit.circular(Earth, _tb.synchronous_radius(Earth.mu, Earth.rotation_rate) - Earth.radius)

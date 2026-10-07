"""Orbit objects and presets."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import cached_property

import numpy as np

from . import units
from .bodies import Body, Earth
from .constants import SSO_RAAN_RATE
from .core import elements as _el
from .core import j2 as _j2
from .core import kepler as _kepler
from .core import twobody as _tb
from .time import J2000, Epoch


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
        object.__setattr__(self, "r", _frozen_vector(self.r))
        object.__setattr__(self, "v", _frozen_vector(self.v))

    # ---------------------------------------------------------- construction

    @classmethod
    def from_vectors(cls, body: Body, r, v, epoch: Epoch = J2000) -> Orbit:
        """From a state vector, given as Quantities or SI numbers."""
        return cls(body, units.to_si(r, units.LENGTH),
                   units.to_si(v, units.VELOCITY), epoch)

    @classmethod
    def from_elements(cls, body: Body, *, a=None, p=None, ecc: float = 0.0,
                      inc=0.0, raan=0.0, argp=0.0, nu=0.0,
                      epoch: Epoch = J2000) -> Orbit:
        """From classical orbital elements. Give the size as either a (semi-major axis) or p (semi-latus rectum).

        Hyperbolas use a < 0; parabolas (ecc = 1) can only be given with p.
        Angle conventions for circular and equatorial orbits follow core.elements.
        """
        if (a is None) == (p is None):
            raise ValueError("Specify exactly one of a and p")
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
        a = body.radius + units.to_si(altitude, units.LENGTH)
        return cls.from_elements(body, a=a, inc=inc, raan=raan, nu=arglat, epoch=epoch)

    @classmethod
    def from_apsides(cls, body: Body, periapsis_altitude, apoapsis_altitude, *, inc=0.0,
                     raan=0.0, argp=0.0, nu=0.0, epoch: Epoch = J2000) -> Orbit:
        """Elliptic orbit from its periapsis and apoapsis altitudes above the equatorial radius.

        Example: a 400 x 600 km orbit is ``Orbit.from_apsides(Earth, 400 * u.km, 600 * u.km)``.
        """
        rp = body.radius + units.to_si(periapsis_altitude, units.LENGTH)
        ra = body.radius + units.to_si(apoapsis_altitude, units.LENGTH)
        a, ecc = _tb.apsides_to_ae(rp, ra)
        return cls.from_elements(body, a=a, ecc=ecc, inc=inc, raan=raan, argp=argp, nu=nu,
                                 epoch=epoch)

    # ---------------------------------------------------------- orbital elements

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

    def propagate(self, duration=None, *, days=None, model="twobody", **kwargs):
        """Propagate the orbit and return a Trajectory. The orbit itself is unchanged.

        Args:
            duration: propagation time (seconds or a time Quantity). Give this or days.
            days: propagation time [day]. Plain numbers are days; Quantities are converted.
            model: "twobody", "j2", "j2+drag" or fidelity 0 to 2
            **kwargs: cd, area, mass, density (constant, function or Environment),
                method, rtol, atol, n_points. See `propagation.propagate`.

        Example: ``ISS.propagate(days=30, model="j2+drag", area=1500, mass=420000).final``
        """
        from .propagation import propagate, resolve_duration

        return propagate(self, resolve_duration(duration, days), model=model, **kwargs)

    def after(self, duration=None, *, days=None, model="twobody", **kwargs) -> Orbit:
        """The orbit after propagating; shorthand for ``propagate(...).final``.

        Raises if the propagation stops early (e.g. on reaching the surface) instead of
        returning the orbit at the stopping point.

        Example: ``ISS.after(days=1, model="j2")``
        """
        from .propagation import propagate, resolve_duration

        duration = resolve_duration(duration, days)
        tr = propagate(self, duration, model=model, **kwargs)
        if tr.info.terminated:
            raise RuntimeError(f"Propagation stopped early ({tr.info.terminated}) after "
                               f"{tr.t[-1]:.1f} s of {duration:.1f} s")
        return tr.final

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

    For Earth the nodal rate is 360° per tropical year. For another body pass its own rate,
    360° per its orbital period around the Sun (e.g. Mars: ``2*pi / (686.98 * 86400)`` rad/s).
    """
    a = body.radius + units.to_si(altitude, units.LENGTH)
    if raan_rate is None:
        if body != Earth:
            raise ValueError(f"Pass raan_rate for {body.name}: 360° per its orbital period "
                             f"around the Sun, in rad/s (the default is Earth's year)")
        rate = SSO_RAAN_RATE
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

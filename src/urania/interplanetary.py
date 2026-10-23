"""Patched-conic Hohmann transfers between bodies: the BodyTransfer result object.

Implementation of `Body.hohmann_to`. Three cases (Curtis ch. 8):

- siblings: two bodies orbiting the same parent (Earth → Mars, Io → Europa). The Hohmann ellipse runs around
  the parent; the spacecraft escapes the origin on a hyperbola and is captured at the target on another.
- to_moon: a body to one of its moons (Earth → Moon). The Hohmann ellipse starts at the parking orbit.
- to_parent: a moon to its parent (Moon → Earth). The Hohmann ellipse ends at the capture orbit.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING

from scipy.optimize import brentq

from . import units
from .core import interplanetary as _ip
from .core import maneuvers as _man
from .core import planets as _pl
from .core.kepler import _wrap_pi
from .core.twobody import vis_viva
from .time import Epoch

if TYPE_CHECKING:
    from .bodies import Body

DAY = 86400.0


@dataclass(frozen=True, eq=False)
class BodyTransfer:
    """Result of a patched-conic Hohmann transfer between two bodies. All values are SI."""

    kind: str                          # "siblings", "to_moon", "to_parent"
    origin: Body
    target: Body
    central: Body                      # body the Hohmann ellipse goes around
    r1: float                          # radius of the transfer start around central [m]
    r2: float                          # radius of the transfer end around central [m]
    parking_altitude: float            # parking orbit altitude at the origin [m]
    capture_altitude: float | None     # capture periapsis altitude at the target [m] (None: no capture)
    capture_ecc: float
    v_inf_departure: float | None      # hyperbolic excess speed leaving the origin [m/s] (None for to_moon)
    v_inf_arrival: float | None        # hyperbolic excess speed reaching the target [m/s] (None for to_parent)
    dv_departure: float                # Δv of the departure burn [m/s]
    dv_arrival: float                  # Δv of the capture burn [m/s] (0 without capture)
    tof: float                         # transfer time [s]
    phase_angle: float | None          # target lead over origin needed at departure [rad]
    synodic_period: float | None       # [s]
    stay_time: float | None            # shortest wait at the target for a return Hohmann [s] (siblings)
    departure_epoch: Epoch | None      # first departure after `after` (from the ephemeris)
    departure_longitude: float | None  # origin's heliocentric ecliptic longitude at departure [rad]
    assumptions: tuple[str, ...]

    @property
    def total_dv(self) -> float:
        """Departure plus capture Δv [m/s]."""
        return self.dv_departure + self.dv_arrival

    @property
    def c3(self) -> float | None:
        """Characteristic energy v∞² at departure [m²/s²]."""
        return None if self.v_inf_departure is None else self.v_inf_departure**2

    @property
    def a_transfer(self) -> float:
        """Semi-major axis of the Hohmann ellipse [m]."""
        return 0.5 * (self.r1 + self.r2)

    @property
    def arrival_epoch(self) -> Epoch | None:
        """Arrival time (departure_epoch + tof)."""
        return None if self.departure_epoch is None else self.departure_epoch + self.tof

    def summary(self, mass=None, isp=None):
        """Key numbers in km/s, days and degrees, one per line; with mass and isp also the propellant."""
        from ._summary import body_transfer_summary

        return body_transfer_summary(self, mass, isp)

    def propellant(self, mass, isp) -> float:
        """Propellant mass [kg] for total_dv from the rocket equation, given the wet mass and Isp [s]."""
        from .core.propulsion import propellant_mass

        return propellant_mass(self.total_dv, units.to_si(isp, units.TIME), units.to_si(mass, units.MASS))

    def explain(self):
        """Show the patched-conic formulas, v∞, Δv of each burn, phase angle and timing step by step."""
        from .explain import explain_body_transfer

        return explain_body_transfer(self)

    def plot(self, ax=None):
        """Plot the Hohmann ellipse around the central body with both bodies' orbits."""
        from .viz import plot_body_transfer

        return plot_body_transfer(self, ax)

    def __repr__(self) -> str:
        date = "" if self.departure_epoch is None else f", depart {self.departure_epoch.iso[:10]}"
        return (f"BodyTransfer({self.origin.name} → {self.target.name}, Δv={self.total_dv / 1e3:.4f} km/s, "
                f"tof={self.tof / DAY:.1f} d{date})")


def _relation(origin: Body, target: Body) -> tuple[str, Body]:
    if origin == target:
        raise ValueError(f"Origin and target are the same body: {origin.name}")
    if origin.parent is not None and origin.parent == target.parent:
        return "siblings", origin.parent
    if target.parent is not None and target.parent == origin:
        return "to_moon", origin
    if origin.parent is not None and origin.parent == target:
        return "to_parent", target
    raise ValueError(
        f"No direct Hohmann transfer from {origin.name} to {target.name}: they must share a parent, or one "
        f"must orbit the other. Chain transfers instead, e.g. Earth → Saturn, then Saturn → Titan")


def _altitude(value, name: str, body: Body) -> float:
    h = units.altitude(value, name)
    if not math.isfinite(h) or h < 0.0:
        raise ValueError(f"{name} must be a finite altitude ≥ 0: {h} m")
    if body.soi <= body.radius:
        raise ValueError(f"The sphere of influence of {body.name} ({body.soi / 1e3:.1f} km) is smaller than "
                         f"its radius: no orbit around it exists in the patched-conic model")
    if body.radius + h >= body.soi:
        raise ValueError(f"{name} {h / 1e3:.0f} km is outside the sphere of influence of {body.name} "
                         f"({(body.soi - body.radius) / 1e3:.0f} km altitude)")
    return h


def next_departure(origin: Body, target: Body, after: Epoch, phase: float, synodic: float) -> Epoch:
    """First time after `after` at which the target leads the origin by `phase` in ecliptic longitude.

    Longitudes come from the Standish mean elements (core.planets); the search scans 1.2 synodic periods
    and refines the first crossing with Brent's method.
    """
    def f(jd: float) -> float:
        return _wrap_pi(_pl.ecliptic_longitude(target.ephemeris, jd)
                        - _pl.ecliptic_longitude(origin.ephemeris, jd) - phase)

    jd0 = after.jd
    span = 1.2 * synodic / DAY
    step = min(span / 400.0, 2.0)
    jd_a, f_a = jd0, f(jd0)
    if f_a == 0.0:
        return after
    while jd_a < jd0 + span:
        jd_b = jd_a + step
        f_b = f(jd_b)
        # A sign change with a small jump is a crossing; a jump of about 2π is the ±π wrap
        if (f_a < 0.0) != (f_b < 0.0) and abs(f_b - f_a) < math.pi:
            jd = brentq(f, jd_a, jd_b, xtol=1e-9)
            lo, hi = _pl.VALID_JD
            if not lo <= jd <= hi:
                warnings.warn("The departure date is outside 1800-2050, where the Standish elements are "
                              "valid; the date is less accurate", stacklevel=3)
            return Epoch.from_jd(jd)
        jd_a, f_a = jd_b, f_b
    raise RuntimeError(f"No departure found within {span:.0f} days of {after.iso[:10]}")


def hohmann_between(origin: Body, target: Body, parking_altitude, capture_altitude=None, *,
                    capture_ecc: float = 0.0, after: Epoch | None = None) -> BodyTransfer:
    """Patched-conic Hohmann transfer origin → target. Implementation of `Body.hohmann_to`."""
    kind, central = _relation(origin, target)
    mu = central.mu
    h_park = _altitude(parking_altitude, "parking_altitude", origin)
    h_cap = None if capture_altitude is None else _altitude(capture_altitude, "capture_altitude", target)
    capture_ecc = float(capture_ecc)
    if not 0.0 <= capture_ecc < 1.0:
        raise ValueError(f"capture_ecc must be in [0, 1): {capture_ecc}")
    r_park = origin.radius + h_park
    r_cap = None if h_cap is None else target.radius + h_cap

    assumptions = [
        "patched conics: only one body's gravity acts at a time; spheres of influence are points on the "
        "primary's scale",
        f"circular coplanar orbits around {central.name} at the mean semi-major axes",
        "impulsive burns; hyperbolic departures and arrivals burn at periapsis",
        "circular parking orbit at the origin",
    ]
    if kind == "siblings" and origin.retrograde != target.retrograde:
        raise ValueError(f"{origin.name} and {target.name} orbit {central.name} in opposite directions; "
                         f"a Hohmann transfer needs both orbits in the same direction")
    if kind != "siblings":
        moon = target if kind == "to_moon" else origin
        if moon.retrograde:
            assumptions.append(f"{moon.name} orbits retrograde; the spacecraft's orbit around {central.name} "
                               f"is taken in the same (retrograde) direction")
        assumptions.append(f"the sphere of influence of {moon.name} is {moon.soi / moon.semi_major_axis:.0%} of "
                           f"its orbit radius, so the patched-conic split is a rough approximation")

    if kind == "siblings":
        r1, r2 = origin.semi_major_axis, target.semi_major_axis
        v_inf_dep, v_inf_arr, tof = _ip.hohmann_v_inf(r1, r2, mu)
        dv_dep = _ip.departure_dv(v_inf_dep, r_park, origin.mu)
    elif kind == "to_moon":
        r1, r2 = r_park, target.semi_major_axis
        dv1, dv2, tof = _man.hohmann(r1, r2, mu)
        v_inf_dep, v_inf_arr, dv_dep = None, abs(dv2), abs(dv1)
        assumptions.append(f"the transfer starts in the parking orbit around {central.name}; "
                           f"{target.name} is reached at the far end of the ellipse")
    else:
        if r_cap is None:
            raise ValueError(f"A transfer to the parent body {target.name} needs capture_altitude: "
                             f"the Hohmann ellipse ends at the capture orbit")
        r1, r2 = origin.semi_major_axis, r_cap
        dv1, _, tof = _man.hohmann(r1, r2, mu)
        v_inf_dep, v_inf_arr = abs(dv1), None
        dv_dep = _ip.departure_dv(v_inf_dep, r_park, origin.mu)

    if r_cap is None:
        dv_arr = 0.0
        assumptions.append("no capture burn: the spacecraft passes the target with the arrival v∞")
    elif kind == "to_parent":
        dv_arr = abs(vis_viva(r2, 0.5 * (r1 + r2), mu) - math.sqrt(mu * (1.0 + capture_ecc) / r2))
    else:
        dv_arr = _ip.capture_dv(v_inf_arr, r_cap, target.mu, capture_ecc)

    phase = synodic = stay = None
    if kind != "to_parent":
        phase = _ip.phase_angle(r1, r2, mu)
        synodic = _ip.synodic_period(2.0 * math.pi * math.sqrt(r1**3 / mu), 2.0 * math.pi * math.sqrt(r2**3 / mu))
    if kind == "siblings":
        stay = _ip.stay_time(r1, r2, mu)

    departure = longitude = None
    if after is not None:
        after = Epoch.parse(after)
        if kind != "siblings" or origin.ephemeris is None or target.ephemeris is None:
            raise ValueError("A departure date needs two planets with an ephemeris (Mercury to Neptune); "
                             "moon positions are not modelled")
        departure = next_departure(origin, target, after, phase, synodic)
        longitude = _pl.ecliptic_longitude(origin.ephemeris, departure.jd)
        assumptions.append("departure date: first time after the given epoch when the actual phase angle "
                           "(Standish mean elements, ecliptic longitudes) equals the Hohmann phase angle; "
                           "a Lambert solution (porkchop plot) gives the true optimum")

    return BodyTransfer(kind, origin, target, central, r1, r2, h_park, h_cap, capture_ecc,
                        v_inf_dep, v_inf_arr, dv_dep, dv_arr, tof, phase, synodic, stay,
                        departure, longitude, tuple(assumptions))

"""Interplanetary trajectories from Lambert arcs: direct transfers, gravity-assist sequences, porkchop plots.

Planet positions and velocities come from the Standish mean elements (core.planets) in the heliocentric J2000
ecliptic frame. Each leg is a Lambert arc between planet centers (patched conics); each intermediate planet joins
the incoming and outgoing v∞ with a flyby, powered at periapsis when the magnitudes differ.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from . import units
from .core import flyby as _fb
from .core import interplanetary as _ip
from .core import lambert as _lam
from .core import planets as _pl
from .time import Epoch

if TYPE_CHECKING:
    from .bodies import Body

DAY = 86400.0


def _ro(x) -> np.ndarray:
    a = np.array(x, dtype=float)
    a.flags.writeable = False
    return a


def _is_duration(x) -> bool:
    """A time span (seconds, a time Quantity or a string such as "300 day") rather than a date."""
    import astropy.units as u

    if isinstance(x, (int, float)):
        return True
    if isinstance(x, u.Quantity):
        return x.unit.physical_type == "time"
    if isinstance(x, str):
        try:
            return u.Quantity(x).unit.physical_type == "time"
        except (TypeError, ValueError):
            return False
    return False


def _require_planet(body: Body) -> None:
    if body.ephemeris is None:
        raise ValueError(f"{body.name} has no ephemeris: positions are modeled only for the planets "
                         f"(Mercury to Neptune)")


def planet_state(body: Body, epoch: Epoch) -> tuple[np.ndarray, np.ndarray]:
    """Heliocentric position [m] and velocity [m/s] of a planet, J2000 ecliptic (Standish mean elements)."""
    _require_planet(body)
    return _pl.heliocentric_state(body.ephemeris, epoch.jd, body.parent.mu)


@dataclass(frozen=True, eq=False)
class Leg:
    """One heliocentric Lambert arc between two planets. Vectors are J2000 ecliptic, SI."""

    origin: Body
    target: Body
    departure: Epoch
    arrival: Epoch
    r1: np.ndarray
    v1: np.ndarray           # spacecraft velocity leaving origin
    r2: np.ndarray
    v2: np.ndarray           # spacecraft velocity reaching target
    v_origin: np.ndarray     # planet velocities
    v_target: np.ndarray
    revolutions: int

    def __post_init__(self):
        for name in ("r1", "v1", "r2", "v2", "v_origin", "v_target"):
            object.__setattr__(self, name, _ro(getattr(self, name)))

    @property
    def tof(self) -> float:
        return self.arrival - self.departure

    @property
    def v_inf_out(self) -> np.ndarray:
        """Excess velocity leaving the origin [m/s]."""
        return self.v1 - self.v_origin

    @property
    def v_inf_in(self) -> np.ndarray:
        """Excess velocity reaching the target [m/s]."""
        return self.v2 - self.v_target

    @property
    def transfer_angle(self) -> float:
        return _lam.transfer_angle(self.r1, self.r2)

    def __repr__(self) -> str:
        return (f"Leg({self.origin.name} → {self.target.name}, {self.departure.iso[:10]} → "
                f"{self.arrival.iso[:10]}, v∞ {np.linalg.norm(self.v_inf_out) / 1e3:.3f} → "
                f"{np.linalg.norm(self.v_inf_in) / 1e3:.3f} km/s)")


@dataclass(frozen=True, eq=False)
class Flyby:
    """A planetary flyby: v∞ in and out, periapsis, turn angle and (for a powered flyby) the periapsis burn."""

    body: Body
    epoch: Epoch | None
    v_inf_in: np.ndarray
    v_inf_out: np.ndarray
    r_p: float                  # periapsis radius [m]
    dv: float                   # periapsis burn [m/s] (0 for an unpowered flyby)
    v_body: np.ndarray | None   # heliocentric planet velocity [m/s], if known
    min_altitude: float = 0.0   # lowest allowed periapsis altitude [m]

    def __post_init__(self):
        object.__setattr__(self, "v_inf_in", _ro(self.v_inf_in))
        object.__setattr__(self, "v_inf_out", _ro(self.v_inf_out))
        if self.v_body is not None:
            object.__setattr__(self, "v_body", _ro(self.v_body))

    @property
    def altitude(self) -> float:
        """Periapsis altitude [m]."""
        return self.r_p - self.body.radius

    @property
    def turn_angle(self) -> float:
        """Angle between v∞_in and v∞_out [rad]."""
        a, b = self.v_inf_in, self.v_inf_out
        return math.atan2(float(np.linalg.norm(np.cross(a, b))), float(a @ b))

    @property
    def feasible(self) -> bool:
        """Periapsis at or above min_altitude (above the surface by default)."""
        return self.altitude >= self.min_altitude

    @property
    def speed_gain(self) -> float | None:
        """Change of heliocentric speed [m/s] (None without the planet velocity)."""
        if self.v_body is None:
            return None
        return float(np.linalg.norm(self.v_body + self.v_inf_out) - np.linalg.norm(self.v_body + self.v_inf_in))

    def explain(self):
        from .explain import explain_flyby

        return explain_flyby(self)

    def __repr__(self) -> str:
        gain = "" if self.speed_gain is None else f", heliocentric {self.speed_gain / 1e3:+.3f} km/s"
        burn = f", burn {self.dv / 1e3:.3f} km/s" if self.dv > 0 else ""
        return (f"Flyby({self.body.name}, altitude {self.altitude / 1e3:,.0f} km, "
                f"turn {math.degrees(self.turn_angle):.2f}°{burn}{gain})")


@dataclass(frozen=True, eq=False)
class Mission:
    """A sequence of Lambert legs joined by flybys, with departure and capture burns."""

    legs: tuple[Leg, ...]
    flybys: tuple[Flyby, ...]
    parking_altitude: float
    capture_altitude: float | None
    capture_ecc: float
    dv_departure: float
    dv_arrival: float
    assumptions: tuple[str, ...]

    @property
    def sequence(self) -> tuple[Body, ...]:
        return (self.legs[0].origin,) + tuple(leg.target for leg in self.legs)

    @property
    def departure(self) -> Epoch:
        return self.legs[0].departure

    @property
    def arrival(self) -> Epoch:
        return self.legs[-1].arrival

    @property
    def tof(self) -> float:
        return self.arrival - self.departure

    @property
    def v_inf_departure(self) -> float:
        return float(np.linalg.norm(self.legs[0].v_inf_out))

    @property
    def v_inf_arrival(self) -> float:
        return float(np.linalg.norm(self.legs[-1].v_inf_in))

    @property
    def c3(self) -> float:
        """Launch energy v∞² [m²/s²]."""
        return self.v_inf_departure**2

    @property
    def dv_flybys(self) -> float:
        return sum(f.dv for f in self.flybys)

    @property
    def total_dv(self) -> float:
        """Departure + powered flybys + capture [m/s]."""
        return self.dv_departure + self.dv_flybys + self.dv_arrival

    @property
    def feasible(self) -> bool:
        """Every flyby passes above its minimum altitude."""
        return all(f.feasible for f in self.flybys)

    def summary(self, mass=None, isp=None):
        """Key numbers in km/s, days and degrees, one per line; with mass and isp also the propellant."""
        from ._summary import mission_summary

        return mission_summary(self, mass, isp)

    def propellant(self, mass, isp) -> float:
        """Propellant mass [kg] for total_dv (rocket equation), starting from the wet mass."""
        from .core.propulsion import propellant_mass

        return propellant_mass(self.total_dv, units.to_si(isp, units.TIME), units.to_si(mass, units.MASS))

    def explain(self):
        from .explain import explain_mission

        return explain_mission(self)

    def plot(self, ax=None):
        from .viz import plot_mission

        return plot_mission(self, ax)

    def __repr__(self) -> str:
        path = " → ".join(b.name for b in self.sequence)
        flag = "" if self.feasible else ", infeasible flyby"
        return (f"Mission({path}, Δv={self.total_dv / 1e3:.4f} km/s, C3={self.c3 / 1e6:.2f} km²/s², "
                f"tof={self.tof / DAY:.1f} d{flag})")


def _altitude(value, name: str, body: Body) -> float:
    h = units.altitude(value, name)
    if not math.isfinite(h) or h < 0.0:
        raise ValueError(f"{name} must be a finite altitude ≥ 0: {h} m")
    return h


def _leg(origin: Body, target: Body, t1: Epoch, t2: Epoch, revolutions: int, branch: str,
         prograde: bool) -> Leg:
    r1, vp1 = planet_state(origin, t1)
    r2, vp2 = planet_state(target, t2)
    v1, v2 = _lam.lambert(r1, r2, t2 - t1, origin.parent.mu, prograde=prograde, revolutions=revolutions,
                          branch=branch)
    return Leg(origin, target, t1, t2, r1, v1, r2, v2, vp1, vp2, revolutions)


def flyby_mission(sequence, dates, parking_altitude, capture_altitude=None, *, capture_ecc: float = 0.0,
                  min_flyby_altitude=0.0, revolutions=0, branch: str = "left",
                  prograde: bool = True) -> Mission:
    """Planet-to-planet trajectory through gravity assists, e.g. [Earth, Venus, Mars] with three dates.

    Args:
        sequence: planets in visiting order (at least two)
        dates: an Epoch for each planet (departure, flybys, arrival)
        parking_altitude: circular parking orbit altitude at the first planet (length)
        capture_altitude: capture periapsis altitude at the last planet (length), None for no capture
        capture_ecc: capture orbit eccentricity
        min_flyby_altitude: lowest acceptable flyby altitude (length); lower flybys are flagged infeasible
        revolutions: complete heliocentric revolutions per leg (an int for all legs, or one per leg)
        branch: multi-revolution Lambert branch, "left" or "right"
        prograde: heliocentric direction of motion (prograde for every planet mission)
    """
    from .bodies import get

    seq = [get(b) for b in sequence]
    dates = [Epoch.parse(d) for d in dates]
    if len(seq) < 2 or len(seq) != len(dates):
        raise ValueError(f"Need at least two bodies and one date per body ({len(seq)} bodies, {len(dates)} dates)")
    for b in seq:
        _require_planet(b)
    if any(b - a <= 0 for a, b in zip(dates, dates[1:])):
        raise ValueError("dates must increase")
    revs = [int(revolutions)] * (len(seq) - 1) if np.ndim(revolutions) == 0 else [int(n) for n in revolutions]
    if len(revs) != len(seq) - 1:
        raise ValueError(f"revolutions needs one value per leg ({len(seq) - 1})")
    h_park = _altitude(parking_altitude, "parking_altitude", seq[0])
    h_cap = None if capture_altitude is None else _altitude(capture_altitude, "capture_altitude", seq[-1])
    h_min = units.to_si(min_flyby_altitude, units.LENGTH)
    capture_ecc = float(capture_ecc)
    if not 0.0 <= capture_ecc < 1.0:
        raise ValueError(f"capture_ecc must be in [0, 1): {capture_ecc}")

    legs = tuple(_leg(seq[k], seq[k + 1], dates[k], dates[k + 1], revs[k], branch, prograde)
                 for k in range(len(seq) - 1))
    flybys = []
    for prev, nxt in zip(legs, legs[1:]):
        body = prev.target
        v_in, v_out = prev.v_inf_in, nxt.v_inf_out
        vi, vo = float(np.linalg.norm(v_in)), float(np.linalg.norm(v_out))
        delta = math.atan2(float(np.linalg.norm(np.cross(v_in, v_out))), float(v_in @ v_out))
        rp = _fb.periapsis_for_turn(vi, vo, min(delta, math.pi * (1 - 1e-12)), body.mu)
        flybys.append(Flyby(body, prev.arrival, v_in, v_out, rp, _fb.powered_dv(vi, vo, rp, body.mu),
                            prev.v_target, h_min))

    first, last = seq[0], seq[-1]
    dv_dep = _ip.departure_dv(float(np.linalg.norm(legs[0].v_inf_out)), first.radius + h_park, first.mu)
    dv_arr = 0.0
    if h_cap is not None:
        dv_arr = _ip.capture_dv(float(np.linalg.norm(legs[-1].v_inf_in)), last.radius + h_cap, last.mu,
                                capture_ecc)

    assumptions = [
        "patched conics: Lambert arcs between planet centers under the Sun's gravity only",
        "planet states from the Standish mean elements (JPL, 1800 to 2050; arcminute accuracy)",
        "impulsive burns: departure and capture at hyperbola periapsis, flyby burns at flyby periapsis",
        "circular parking orbit at the first planet",
    ]
    if flybys:
        assumptions.append("flybys may be powered: a periapsis burn joins different incoming and outgoing v∞ "
                           "magnitudes (the MGA model); an unpowered flyby needs equal magnitudes")
    if h_cap is None:
        assumptions.append("no capture burn: arrival v∞ only")
    return Mission(legs, tuple(flybys), h_park, h_cap, capture_ecc, dv_dep, dv_arr, tuple(assumptions))


def lambert_transfer(origin: Body, target: Body, departure: Epoch, arrival, parking_altitude,
                     capture_altitude=None, **kwargs) -> Mission:
    """Direct Lambert transfer; arrival is a time (Epoch or date string) or a time of flight ("300 day")."""
    departure = Epoch.parse(departure)
    if isinstance(arrival, str) and re.match(r"\s*\d{4}-\d{2}-\d{2}", arrival) or not _is_duration(arrival):
        arrival = Epoch.parse(arrival)
    else:
        arrival = departure + units.to_si(arrival, units.TIME)
    return flyby_mission([origin, target], [departure, arrival], parking_altitude, capture_altitude, **kwargs)


def flyby(body: Body, v_inf_in, altitude, *, epoch: Epoch | None = None, b_angle=0.0) -> Flyby:
    """Unpowered flyby of `body` with incoming excess velocity v∞_in (heliocentric frame of the planet).

    b_angle rotates the flyby plane about v∞_in: 0 turns v∞ counterclockwise seen from the ecliptic pole
    (passing behind the planet when v∞_in points along its motion), π clockwise.
    """
    v_in = np.asarray(units.to_si(v_inf_in, units.VELOCITY), dtype=float)
    if v_in.shape != (3,) or not np.linalg.norm(v_in) > 0:
        raise ValueError("v_inf_in must be a nonzero 3-vector")
    h = _altitude(altitude, "altitude", body)
    rp = body.radius + h
    v_out = _fb.outgoing_v_inf(v_in, rp, body.mu, units.to_si(b_angle, units.ANGLE))
    v_body = None
    if epoch is not None:
        epoch = Epoch.parse(epoch)
        v_body = planet_state(body, epoch)[1]
    return Flyby(body, epoch, v_in, v_out, rp, 0.0, v_body)


# ---------------------------------------------------------------- porkchop

@dataclass(frozen=True, eq=False)
class Porkchop:
    """Grid of direct Lambert transfers over departure and arrival dates.

    Arrays have shape (len(arrival_jd), len(departure_jd)); NaN where no transfer exists (arrival before
    departure, or the Lambert solver found no solution).
    """

    origin: Body
    target: Body
    departure_jd: np.ndarray
    arrival_jd: np.ndarray
    c3: np.ndarray                 # launch energy [m²/s²]
    v_inf_arrival: np.ndarray      # [m/s]
    total_dv: np.ndarray | None    # departure + capture Δv [m/s] when altitudes are given
    parking_altitude: float | None
    capture_altitude: float | None

    @property
    def tof(self) -> np.ndarray:
        """Time of flight [s] on the grid."""
        return (self.arrival_jd[:, None] - self.departure_jd[None, :]) * DAY

    def best(self, criterion: str = "c3") -> Mission:
        """The grid point that minimizes "c3", "v_inf_arrival", "v_inf_sum" or "total_dv", as a Mission.

        Without a parking altitude the Mission's departure Δv is computed from the surface (altitude 0).
        """
        grids = {"c3": self.c3, "v_inf_arrival": self.v_inf_arrival, "total_dv": self.total_dv,
                 "v_inf_sum": np.sqrt(self.c3) + self.v_inf_arrival}
        if criterion not in grids or grids[criterion] is None:
            from ._hints import did_you_mean

            valid = [k for k, v in grids.items() if v is not None]
            raise ValueError(f"criterion must be one of {valid}." + did_you_mean(criterion, valid))
        grid = grids[criterion]
        if np.all(np.isnan(grid)):
            raise ValueError("No transfer on the grid")
        i, j = np.unravel_index(np.nanargmin(grid), grid.shape)
        return flyby_mission([self.origin, self.target],
                             [Epoch.from_jd(self.departure_jd[j]), Epoch.from_jd(self.arrival_jd[i])],
                             self.parking_altitude or 0.0, self.capture_altitude)

    def plot(self, ax=None, max_c3=None):
        from .viz import plot_porkchop

        return plot_porkchop(self, ax, max_c3)

    def __repr__(self) -> str:
        c3 = np.nanmin(self.c3) / 1e6 if not np.all(np.isnan(self.c3)) else math.nan
        return (f"Porkchop({self.origin.name} → {self.target.name}, {len(self.departure_jd)}×"
                f"{len(self.arrival_jd)} grid, min C3 {c3:.2f} km²/s²)")


def _date_grid(span, step_days: float) -> np.ndarray:
    if isinstance(span, (str, Epoch)):
        raise TypeError("Give each date range as (start, end), e.g. ('2026-08-01', '2027-01-01')")
    try:
        start, stop = span
    except (TypeError, ValueError):
        raise TypeError("Give each date range as (start, end), e.g. ('2026-08-01', '2027-01-01')") from None
    start, stop = Epoch.parse(start), Epoch.parse(stop)
    if stop - start <= 0:
        raise ValueError("Each date range needs start < end")
    n = int(math.floor((stop.jd - start.jd) / step_days + 1e-9)) + 1
    return start.jd + step_days * np.arange(n)


def porkchop(origin: Body, target: Body, departure, arrival, *, step=5 * DAY, parking_altitude=None,
             capture_altitude=None, min_tof=None) -> Porkchop:
    """C3 and arrival v∞ of direct (type I and II) Lambert transfers over a departure × arrival date grid.

    Args:
        departure, arrival: (start Epoch, end Epoch) ranges
        step: grid spacing (duration, default 5 days)
        parking_altitude, capture_altitude: give both to get the total Δv grid (lengths)
        min_tof: skip transfers shorter than this (duration, default 10 days)
    """
    from .bodies import get

    origin, target = get(origin), get(target)
    _require_planet(origin)
    _require_planet(target)
    step_days = units.to_si(step, units.TIME) / DAY
    if not step_days > 0:
        raise ValueError(f"step must be positive: {step}")
    dep, arr = _date_grid(departure, step_days), _date_grid(arrival, step_days)
    min_days = 10.0 if min_tof is None else units.to_si(min_tof, units.TIME) / DAY
    mu = origin.parent.mu
    s1 = [_pl.heliocentric_state(origin.ephemeris, jd, mu) for jd in dep]
    s2 = [_pl.heliocentric_state(target.ephemeris, jd, mu) for jd in arr]
    c3 = np.full((len(arr), len(dep)), np.nan)
    vinf = np.full_like(c3, np.nan)
    for i, (r2, vp2) in enumerate(s2):
        for j, (r1, vp1) in enumerate(s1):
            days = arr[i] - dep[j]
            if days < min_days:
                continue
            try:
                v1, v2 = _lam.lambert(r1, r2, days * DAY, mu)
            except ValueError:
                continue
            c3[i, j] = float(np.sum((v1 - vp1) ** 2))
            vinf[i, j] = float(np.linalg.norm(v2 - vp2))
    total = None
    h_park = h_cap = None
    if parking_altitude is not None:
        h_park = _altitude(parking_altitude, "parking_altitude", origin)
        rp = origin.radius + h_park
        dep_dv = np.sqrt(c3 + 2 * origin.mu / rp) - math.sqrt(origin.mu / rp)
        total = dep_dv
        if capture_altitude is not None:
            h_cap = _altitude(capture_altitude, "capture_altitude", target)
            rc = target.radius + h_cap
            total = dep_dv + np.sqrt(vinf**2 + 2 * target.mu / rc) - math.sqrt(target.mu / rc)
    elif capture_altitude is not None:
        raise ValueError("capture_altitude needs parking_altitude for the total Δv grid")
    return Porkchop(origin, target, _ro(dep), _ro(arr), _ro(c3), _ro(vinf),
                    None if total is None else _ro(total), h_park, h_cap)

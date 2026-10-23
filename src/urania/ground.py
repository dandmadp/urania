"""Ground stations, ground tracks and pass prediction (Earth only).

Positions in urania's inertial frame (z = Earth's spin axis) are rotated to the Earth-fixed frame by the
Greenwich mean sidereal time (IAU 1982) with UT1 ≈ UTC. That approximation shifts longitudes by at
most about 0.4 km at the equator and pass times by well under a second, which suits ground tracks and
station contact planning.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from scipy.optimize import brentq, minimize_scalar

from . import units
from .bodies import Body, Earth
from .core import frames as _fr
from .time import Epoch, utc_jd

if TYPE_CHECKING:
    from .propagation import Trajectory

FRAME_ASSUMPTIONS = (
    "inertial frame: z axis along Earth's spin axis (true equator of date, like TEME)",
    "Earth-fixed frame: rotation by Greenwich mean sidereal time (IAU 1982), UT1 ≈ UTC (|UT1 - UTC| < 0.9 s)",
    "polar motion ignored (about 0.3\", under 10 m at the surface)",
    "geodetic latitude and height on the body's reference ellipsoid",
)


def _require_earth(body: Body) -> None:
    if body != Earth:
        raise ValueError(f"Earth-fixed coordinates need Earth's rotation model; the trajectory is around "
                         f"{body.name}")


def _readonly(x) -> np.ndarray:
    a = np.array(x, dtype=float)
    a.flags.writeable = False
    return a


class _EarthRotation:
    """GMST as a function of time since a trajectory's start, from UTC Julian dates at its samples."""

    def __init__(self, tr: Trajectory):
        self.t = np.asarray(tr.t, dtype=float)
        self.jd = utc_jd(tr.epoch.tdb_seconds + self.t)
        self.theta = _fr.gmst82(self.jd)

    def at(self, t: float) -> float:
        """GMST at time t of an increasing-time trajectory. UTC advances with TDB between samples
        (a leap second inside an interval is the only exception)."""
        i = int(np.clip(np.searchsorted(self.t, t, side="right"), 1, len(self.t))) - 1
        return _fr.gmst82(self.jd[i] + (t - self.t[i]) / 86400.0)


@dataclass(frozen=True)
class GroundStation:
    """A site on Earth: geodetic latitude, longitude [rad] and height above the ellipsoid [m]."""

    name: str
    lat: float
    lon: float
    alt: float = 0.0
    body: Body = field(default=Earth, repr=False)

    def __post_init__(self):
        if not -math.pi / 2 <= self.lat <= math.pi / 2:
            raise ValueError(f"Latitude must be within ±90°: {math.degrees(self.lat):g}°. Plain numbers are "
                             f"radians; write degrees as 36.4 * u.deg or '36.4 deg'")

    @classmethod
    def create(cls, name: str, lat, lon, alt=0.0) -> GroundStation:
        """From values with units, e.g. ``GroundStation.create("Daejeon", 36.37 * u.deg, 127.36 * u.deg)``.

        Plain numbers are SI (rad, m).
        """
        return cls(name, units.to_si(lat, units.ANGLE), units.to_si(lon, units.ANGLE),
                   units.to_si(alt, units.LENGTH))

    @property
    def r_ecef(self) -> np.ndarray:
        """Earth-fixed position [m]."""
        b = self.body
        return _fr.geodetic_to_ecef(self.lat, self.lon, self.alt, b.radius, b.flattening)

    def __repr__(self) -> str:
        return (f"GroundStation({self.name!r}, lat={math.degrees(self.lat):.4f}°, "
                f"lon={math.degrees(self.lon):.4f}°, alt={self.alt:.0f} m)")


# ---------------------------------------------------------------- ground track

@dataclass(frozen=True, eq=False)
class GroundTrack:
    """Sub-satellite points: geodetic latitude, longitude [rad] and height [m] at every sample."""

    epoch: Epoch
    t: np.ndarray
    lat: np.ndarray
    lon: np.ndarray
    alt: np.ndarray
    assumptions: tuple[str, ...]

    def __post_init__(self):
        for name in ("t", "lat", "lon", "alt"):
            object.__setattr__(self, name, _readonly(getattr(self, name)))

    def __len__(self) -> int:
        return len(self.t)

    def plot(self, ax=None, stations=()):
        """Plot latitude against longitude in degrees, optionally with ground stations."""
        from .viz import plot_ground_track

        return plot_ground_track(self, ax, stations)

    def __repr__(self) -> str:
        return (f"GroundTrack({len(self)} points, {self.t[-1] / 86400:.3f} days from "
                f"{self.epoch.utc[:19]} UTC, latitude {math.degrees(self.lat.min()):.1f}° to "
                f"{math.degrees(self.lat.max()):.1f}°)")


def ground_track(tr: Trajectory) -> GroundTrack:
    """Ground track of a trajectory around Earth. Implementation of `Trajectory.ground_track`."""
    _require_earth(tr.body)
    rot = _EarthRotation(tr)
    lat, lon, alt = _fr.geodetic(_fr.rotate_z(tr.r, rot.theta), tr.body.radius, tr.body.flattening)
    return GroundTrack(tr.epoch, tr.t, lat, lon, alt, FRAME_ASSUMPTIONS)


# ---------------------------------------------------------------- passes

@dataclass(frozen=True)
class Pass:
    """One pass over a ground station: acquisition, culmination and loss of signal."""

    aos: Epoch                 # rises above the minimum elevation
    tca: Epoch                 # highest elevation
    los: Epoch                 # sets below the minimum elevation
    max_elevation: float       # [rad]
    aos_azimuth: float         # [rad, from north toward east]
    los_azimuth: float         # [rad]
    min_range: float           # [m], at the highest elevation
    truncated: bool = False    # the trajectory starts or ends during the pass

    @property
    def duration(self) -> float:
        """Time above the minimum elevation [s]."""
        return self.los - self.aos

    def __repr__(self) -> str:
        cut = ", truncated" if self.truncated else ""
        return (f"Pass({self.aos.utc[:19]} UTC, {self.duration / 60:.1f} min, max elevation "
                f"{math.degrees(self.max_elevation):.1f}°, az {math.degrees(self.aos_azimuth):.0f}° → "
                f"{math.degrees(self.los_azimuth):.0f}°{cut})")


@dataclass(frozen=True, eq=False)
class PassReport:
    """All passes of a trajectory over one station."""

    station: GroundStation
    min_elevation: float
    passes: tuple[Pass, ...]
    start: Epoch
    end: Epoch
    assumptions: tuple[str, ...]

    def __len__(self) -> int:
        return len(self.passes)

    def __iter__(self):
        return iter(self.passes)

    def __getitem__(self, i):
        return self.passes[i]

    @property
    def total_contact(self) -> float:
        """Sum of all pass durations [s]."""
        return sum(p.duration for p in self.passes)

    def explain(self):
        """How elevations are computed, the pass list and the assumptions."""
        from .explain import explain_passes

        return explain_passes(self)

    def __repr__(self) -> str:
        head = (f"{len(self)} passes over {self.station.name} above {math.degrees(self.min_elevation):.0f}° "
                f"({self.total_contact / 60:.1f} min total)")
        return "\n".join([head] + [f"  {p!r}" for p in self.passes])


class _LookAngles:
    """Elevation and azimuth from a station at any time of a trajectory (Hermite-interpolated)."""

    def __init__(self, tr: Trajectory, station: GroundStation, rot: _EarthRotation):
        self.tr, self.station, self.rot = tr, station, rot
        self.site = station.r_ecef

    def sampled(self):
        r_ecef = _fr.rotate_z(self.tr.r, self.rot.theta)
        return _fr.look_angles(r_ecef, self.site, self.station.lat, self.station.lon)

    def at(self, t: float):
        r, _ = self.tr.state_at(t)
        r_ecef = _fr.rotate_z(r, self.rot.at(t))
        return _fr.look_angles(r_ecef, self.site, self.station.lat, self.station.lon)


def passes(tr: Trajectory, station: GroundStation, min_elevation=math.radians(10.0)) -> PassReport:
    """Passes of a trajectory over a station. Implementation of `Trajectory.passes`.

    Rise and set times are found between samples by root finding on the elevation, and the
    culmination by a bounded minimization; states between samples come from `Trajectory.state_at`.
    A short, low pass that peaks between two samples is caught by refining sampled local maxima.
    """
    _require_earth(tr.body)
    if station.body != tr.body:
        raise ValueError("The station and the trajectory are on different bodies")
    min_el = units.to_si(min_elevation, units.ANGLE)
    t = np.asarray(tr.t, dtype=float)
    if len(t) < 2 or t[-1] <= t[0]:
        raise ValueError("Pass prediction needs a forward trajectory with at least two samples")
    rot = _EarthRotation(tr)
    look = _LookAngles(tr, station, rot)
    _, el, _ = look.sampled()
    n = len(t)

    def margin(x):
        return look.at(x)[1] - min_el

    def peak(lo, hi):
        res = minimize_scalar(lambda x: -look.at(x)[1], bounds=(lo, hi), method="bounded",
                              options={"xatol": 0.01})
        return float(res.x)

    # Each candidate is (rise bracket, set bracket) in seconds, or None for a trajectory edge.
    candidates = []
    above = el > min_el
    i = 0
    while i < n:
        if above[i]:
            j = i
            while j + 1 < n and above[j + 1]:
                j += 1
            rise = None if i == 0 else (t[i - 1], t[i])
            setb = None if j == n - 1 else (t[j], t[j + 1])
            candidates.append((t[i], rise, setb))
            i = j + 1
        else:
            if 0 < i < n - 1 and el[i] >= el[i - 1] and el[i] >= el[i + 1]                     and el[i] > min_el - math.radians(3.0):
                tm = peak(t[i - 1], t[i + 1])
                if margin(tm) > 0.0:
                    candidates.append((tm, (t[i - 1], tm), (tm, t[i + 1])))
            i += 1
    candidates.sort(key=lambda c: c[0])

    out = []
    for _, rise, setb in candidates:
        aos = t[0] if rise is None else brentq(margin, rise[0], rise[1], xtol=1e-3)
        los = t[-1] if setb is None else brentq(margin, setb[0], setb[1], xtol=1e-3)
        tca = peak(aos, los)
        _, max_el, rng = look.at(tca)
        start = tr.epoch
        out.append(Pass(start + float(aos), start + tca, start + float(los), max_el,
                        look.at(aos)[0], look.at(los)[0], rng, rise is None or setb is None))
    return PassReport(station, min_el, tuple(out), tr.epoch, tr.epoch + float(t[-1]), FRAME_ASSUMPTIONS)

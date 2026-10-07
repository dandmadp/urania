"""TLE (Two-Line Element) orbits: a wrapper around the sgp4 package.

TLE elements are mean elements specific to SGP4 theory and are meaningful only when propagated with SGP4.
So they are kept apart from `Orbit` in a separate object; to hand over to the urania propagator,
extract a state vector at a given epoch explicitly with `to_orbit()`.

- Frame: the SGP4 output frame TEME is used as the inertial frame as is (no precession/nutation).
- Time: TLE epochs are UTC. astropy converts them to TDB (`Epoch`).
- Gravity constants: WGS72, the convention TLEs are generated with.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import cached_property

import numpy as np
from astropy.time import Time
from sgp4.api import SGP4_ERRORS, WGS72, Satrec

from .bodies import Earth
from .propagation import PropagationInfo, Trajectory, resolve_duration
from .time import JD_J2000, SECONDS_PER_DAY, Epoch

KM = 1e3
MU_WGS72 = 398600.8e9   # Earth μ of the TLE convention [m³/s²]

ASSUMPTIONS = (
    "SGP4/SDP4 analytic theory (for TLE mean elements only, sgp4 package)",
    "frame: TEME used as the inertial frame (no precession/nutation transformation)",
    "gravity constants: WGS72 (TLE convention)",
    "TLE epoch is UTC, converted to TDB by astropy",
    "accuracy degrades with TLE age (typically 1 km at epoch, growing by several km per day)",
)


def checksum(line: str) -> int:
    """TLE line checksum: sum of the digits in the first 68 characters plus the number of '-', mod 10."""
    return sum(int(c) if c.isdigit() else (1 if c == "-" else 0) for c in line[:68]) % 10


def _validate(line: str, number: str) -> None:
    if len(line) < 69 or line[0] != number:
        raise ValueError(f"Not a valid TLE line {number}: {line!r}")
    if not line[68].isdigit() or int(line[68]) != checksum(line):
        raise ValueError(f"TLE line {number} checksum mismatch (expected {checksum(line)}, got {line[68]})")


def _epoch_to_utc_jd(seconds) -> tuple[np.ndarray, np.ndarray]:
    """TDB seconds since J2000 (scalar or array) → UTC Julian date (integer part, fraction)."""
    s = np.atleast_1d(np.asarray(seconds, dtype=float))
    utc = Time(np.full(s.shape, JD_J2000), s / SECONDS_PER_DAY, format="jd", scale="tdb").utc
    return utc.jd1, utc.jd2


@dataclass(frozen=True, eq=False)
class TLEOrbit:
    """An Earth orbit defined by a TLE. Propagated with SGP4 only."""

    line1: str
    line2: str
    name: str = ""
    _sat: Satrec = field(init=False, repr=False)

    def __post_init__(self):
        object.__setattr__(self, "line1", self.line1.rstrip())
        object.__setattr__(self, "line2", self.line2.rstrip())
        _validate(self.line1, "1")
        _validate(self.line2, "2")
        if self.line1[2:7] != self.line2[2:7]:
            raise ValueError("The two lines have different satellite numbers")
        object.__setattr__(self, "_sat", Satrec.twoline2rv(self.line1, self.line2, WGS72))

    @classmethod
    def from_text(cls, text: str) -> TLEOrbit:
        """From two-line or three-line (name first) TLE text. Indentation is ignored."""
        lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
        if len(lines) == 2:
            return cls(lines[0], lines[1])
        if len(lines) == 3:
            name = lines[0].strip()
            if name.startswith("0 "):
                name = name[2:]
            return cls(lines[1], lines[2], name=name)
        raise ValueError(f"A TLE must have 2 or 3 lines ({len(lines)} given)")

    # ---------------------------------------------------------- TLE elements (SGP4 mean elements)

    @property
    def satnum(self) -> str:
        """NORAD catalog number."""
        return self.line1[2:7].strip()

    @cached_property
    def epoch(self) -> Epoch:
        """TLE epoch (converted to TDB)."""
        s = self._sat
        return Epoch.from_astropy(Time(s.jdsatepoch, s.jdsatepochF, format="jd", scale="utc"))

    @property
    def inc(self) -> float:
        """Mean inclination [rad]."""
        return self._sat.inclo

    @property
    def raan(self) -> float:
        """Mean right ascension of the ascending node [rad]."""
        return self._sat.nodeo

    @property
    def ecc(self) -> float:
        """Mean eccentricity."""
        return self._sat.ecco

    @property
    def argp(self) -> float:
        """Mean argument of perigee [rad]."""
        return self._sat.argpo

    @property
    def M(self) -> float:
        """Mean anomaly [rad]."""
        return self._sat.mo

    @property
    def mean_motion(self) -> float:
        """Mean motion [rad/s] (Kozai mean)."""
        return self._sat.no_kozai / 60.0

    @property
    def revs_per_day(self) -> float:
        """Revolutions per day."""
        return self.mean_motion * SECONDS_PER_DAY / (2.0 * math.pi)

    @property
    def period(self) -> float:
        """Orbital period from the mean motion [s]."""
        return 2.0 * math.pi / self.mean_motion

    @property
    def bstar(self) -> float:
        """Drag term B* [1/Earth radii]."""
        return self._sat.bstar

    # ---------------------------------------------------------- SGP4 propagation

    def states(self, epochs) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """TEME state vectors at several epochs.

        Args:
            epochs: an Epoch, a sequence of Epochs, or an array of TDB seconds since J2000

        Returns:
            (error, r [m] N×3, v [m/s] N×3). error is the SGP4 error code (0 = OK).
        """
        if isinstance(epochs, Epoch):
            seconds = [epochs.tdb_seconds]
        elif len(epochs) and isinstance(epochs[0], Epoch):
            seconds = [e.tdb_seconds for e in epochs]
        else:
            seconds = epochs
        jd, fr = _epoch_to_utc_jd(seconds)
        err, r, v = self._sat.sgp4_array(jd, fr)
        return err, r * KM, v * KM

    def state_at(self, epoch: Epoch) -> tuple[np.ndarray, np.ndarray]:
        """TEME state vector at one epoch (r [m], v [m/s])."""
        err, r, v = self.states(epoch)
        if err[0]:
            raise RuntimeError(f"SGP4 error {err[0]}: {SGP4_ERRORS[err[0]]}")
        return r[0], v[0]

    def propagate(self, duration=None, *, days=None, start: Epoch | None = None,
                  n_points: int | None = None) -> Trajectory:
        """Propagate with SGP4 and return a Trajectory.

        Args:
            duration, days: propagation time (give one)
            start: start epoch. Defaults to the TLE epoch.
            n_points: number of output points. Defaults to 100 per orbit.
        """
        duration = resolve_duration(duration, days)
        if n_points is not None and n_points < 2:
            raise ValueError(f"n_points must be at least 2 (start and end): {n_points}")
        start = start or self.epoch
        if n_points is None:
            n_points = min(max(math.ceil(abs(duration) / self.period * 100) + 1, 101), 200001)
        t = np.linspace(0.0, duration, n_points)
        err, r, v = self.states(start.tdb_seconds + t)

        terminated = None
        bad = np.nonzero(err)[0]
        if len(bad):
            i = bad[0]
            if i == 0:
                raise RuntimeError(f"SGP4 error {err[0]}: {SGP4_ERRORS[err[0]]}")
            terminated = f"SGP4 error {err[i]} ({SGP4_ERRORS[err[i]]})"
            t, r, v = t[:i], r[:i], v[:i]

        info = PropagationInfo(model="sgp4", integrator="SGP4 (sgp4 package, WGS72)",
                               rtol=None, atol=None, assumptions=ASSUMPTIONS,
                               terminated=terminated)
        return Trajectory(Earth, start, t, r, v, info)

    def to_orbit(self, epoch: Epoch | None = None):
        """Extract the SGP4 state vector as an Orbit for the urania propagator (default: TLE epoch).

        Orbit.propagate() then uses the urania propagator, not SGP4.
        The frame stays TEME; the central body constants are Earth (WGS 84).
        """
        from .orbits import Orbit

        epoch = epoch or self.epoch
        r, v = self.state_at(epoch)
        return Orbit(Earth, r, v, epoch)

    # ---------------------------------------------------------- explanation

    def explain(self):
        """Decode and show the TLE fields."""
        from .explain import explain_tle

        return explain_tle(self)

    def __repr__(self) -> str:
        label = f"{self.name!r}, " if self.name else ""
        return (f"TLEOrbit({label}#{self.satnum}, epoch={self.epoch.iso[:19]} TDB, "
                f"inc={math.degrees(self.inc):.2f}°, {self.revs_per_day:.4f} rev/day)")

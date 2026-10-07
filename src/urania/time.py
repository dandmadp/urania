"""Time representation.

Internal time is a single number: TDB seconds elapsed since J2000 (2000-01-01 12:00:00 TDB).
Conversion to other time scales such as UTC/TAI is out of MVP scope; use astropy Time if needed.

Calendar strings and Julian dates are interpreted as TDB.
The TDB calendar has no leap seconds, so a plain date difference gives exact elapsed seconds.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

from . import units

SECONDS_PER_DAY = 86400.0
JD_J2000 = 2451545.0
_DT_J2000 = datetime(2000, 1, 1, 12, 0, 0)


@dataclass(frozen=True, order=True)
class Epoch:
    """A TDB instant, stored as seconds since J2000."""

    tdb_seconds: float

    # ---------------------------------------------------------- construction

    @classmethod
    def from_iso(cls, text: str) -> Epoch:
        """From an ISO 8601 string (TDB), e.g. "2026-10-07T12:00:00"."""
        dt = datetime.fromisoformat(text)
        if dt.tzinfo is not None:
            raise ValueError("Time zone information is not supported (strings are read as TDB)")
        return cls((dt - _DT_J2000).total_seconds())

    @classmethod
    def from_jd(cls, jd: float) -> Epoch:
        """From a TDB Julian date."""
        return cls((jd - JD_J2000) * SECONDS_PER_DAY)

    @classmethod
    def from_astropy(cls, t) -> Epoch:
        """From an astropy Time. astropy converts other time scales to TDB."""
        tdb = t.tdb
        return cls(((tdb.jd1 - JD_J2000) + tdb.jd2) * SECONDS_PER_DAY)

    # ---------------------------------------------------------- output

    @property
    def jd(self) -> float:
        """TDB Julian date."""
        return JD_J2000 + self.tdb_seconds / SECONDS_PER_DAY

    @property
    def iso(self) -> str:
        """ISO 8601 string (TDB, microsecond resolution)."""
        return (_DT_J2000 + timedelta(seconds=self.tdb_seconds)).isoformat()

    def to_astropy(self):
        """Convert to astropy Time(scale='tdb')."""
        from astropy.time import Time

        return Time(JD_J2000, self.tdb_seconds / SECONDS_PER_DAY,
                    format="jd", scale="tdb")

    # ---------------------------------------------------------- arithmetic

    def __add__(self, dt) -> Epoch:
        """Epoch + interval. The interval is seconds (number) or a time Quantity."""
        return Epoch(self.tdb_seconds + units.to_si(dt, units.TIME))

    def __sub__(self, other):
        """Epoch - Epoch → elapsed seconds (float); Epoch - interval → Epoch."""
        if isinstance(other, Epoch):
            return self.tdb_seconds - other.tdb_seconds
        return Epoch(self.tdb_seconds - units.to_si(other, units.TIME))

    def __repr__(self) -> str:
        return f"Epoch('{self.iso}' TDB)"


J2000 = Epoch(0.0)

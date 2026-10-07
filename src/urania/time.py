"""Time representation.

Internal time is a single number: TDB seconds elapsed since J2000 (2000-01-01 12:00:00 TDB).
urania does not implement time scale conversions itself; UTC input and output go through astropy
(`Epoch.from_utc`, `Epoch.utc`, `Epoch.now`), which handles leap seconds.

`from_iso` and `from_jd` read their input as TDB. The TDB calendar has no leap seconds, so a plain
date difference gives exact elapsed seconds.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

from . import units

SECONDS_PER_DAY = 86400.0
JD_J2000 = 2451545.0
_DT_J2000 = datetime(2000, 1, 1, 12, 0, 0)


def _scalar_seconds(dt) -> float:
    if isinstance(dt, Epoch):
        raise TypeError("Two Epochs cannot be added; subtract them to get the elapsed seconds")
    seconds = units.to_si(dt, units.TIME)
    if not isinstance(seconds, float):
        raise TypeError("An Epoch can only be shifted by a single interval, not an array")
    return seconds


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
            raise ValueError(f"from_iso reads TDB and takes no time zone: {text!r}. "
                             "For a UTC time use Epoch.from_utc()")
        return cls((dt - _DT_J2000).total_seconds())

    @classmethod
    def from_utc(cls, text: str) -> Epoch:
        """From a UTC time string, e.g. "2026-10-08T00:00:00" or "2026-10-08T00:00:00Z".

        astropy converts UTC to TDB, including leap seconds.
        """
        from astropy.time import Time

        text = text.strip()
        if text.endswith(("Z", "z")):
            text = text[:-1]
        return cls.from_astropy(Time(text, scale="utc"))

    @classmethod
    def now(cls) -> Epoch:
        """The current time (from the system clock in UTC)."""
        from astropy.time import Time

        return cls.from_astropy(Time.now())

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

    @property
    def utc(self) -> str:
        """ISO 8601 string in UTC (millisecond resolution), converted by astropy."""
        return self.to_astropy().utc.isot

    def to_astropy(self):
        """Convert to astropy Time(scale='tdb')."""
        from astropy.time import Time

        return Time(JD_J2000, self.tdb_seconds / SECONDS_PER_DAY,
                    format="jd", scale="tdb")

    # ---------------------------------------------------------- arithmetic

    def __add__(self, dt) -> Epoch:
        """Epoch + interval. The interval is seconds (number) or a time Quantity."""
        return Epoch(self.tdb_seconds + _scalar_seconds(dt))

    __radd__ = __add__

    def __sub__(self, other):
        """Epoch - Epoch → elapsed seconds (float); Epoch - interval → Epoch."""
        if isinstance(other, Epoch):
            return self.tdb_seconds - other.tdb_seconds
        return Epoch(self.tdb_seconds - _scalar_seconds(other))

    def __repr__(self) -> str:
        return f"Epoch('{self.iso}' TDB)"


J2000 = Epoch(0.0)

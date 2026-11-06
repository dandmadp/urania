"""Environment injection interface.

The library provides the computational framework; environment data (atmospheric density, ...)
comes from the user. All three forms below are accepted and handled internally as one `Environment`:

- constant: a number or astropy Quantity → `Constant`, which always returns the same value
- function: f(r, t) → `Function`
- `Environment` object: used as is

r is the inertial position [m] (numpy array), t is TDB seconds since J2000.
"""

from __future__ import annotations

import math
import warnings
from abc import ABC, abstractmethod

import astropy.units as u

from . import units
from .core.forces import exponential_density, geodetic_altitude


class Environment(ABC):
    """Environment model giving a value as a function of position and time. Returns an SI number."""

    description: str = ""

    @abstractmethod
    def __call__(self, r, t: float) -> float: ...


class Constant(Environment):
    """Environment that always returns the same value."""

    def __init__(self, value: float, description: str = ""):
        self.value = value
        self.description = description or f"constant {value:g}"

    def __call__(self, r, t: float) -> float:
        return self.value


class Function(Environment):
    """Wraps a user function f(r, t). A returned Quantity is converted to SI."""

    def __init__(self, func, si_unit: u.UnitBase, description: str = ""):
        self.func = func
        self.si_unit = si_unit
        name = getattr(func, "__name__", repr(func))
        self.description = description or f"user function {name}(r, t)"

    def __call__(self, r, t: float) -> float:
        value = self.func(r, t)
        if isinstance(value, u.Quantity):
            return units.to_si(value, self.si_unit)
        return value


def as_environment(value, si_unit: u.UnitBase) -> Environment:
    """Turn a constant, function or Environment into an Environment."""
    if isinstance(value, Environment):
        return value
    if isinstance(value, type):
        raise TypeError(f"Pass an instance, not the class {value.__name__} "
                        f"(e.g. {value.__name__}(...))")
    if callable(value):
        return Function(value, si_unit)
    return Constant(units.to_si(value, si_unit))


# ---------------------------------------------------------------- atmospheric density models

class ExponentialAtmosphere(Environment):
    """Exponential atmosphere of Vallado table 8-4.

    Altitude is the height above the reference ellipsoid (radius, flattening), measured about
    the inertial z axis. With flattening = 0 it is the spherical height |r| - R.
    """

    def __init__(self, radius: float, flattening: float = 0.0):
        self.radius = radius
        self.flattening = flattening
        surface = "ellipsoid" if flattening else "sphere"
        self.description = f"exponential atmosphere (Vallado table 8-4, height above the {surface})"

    def __call__(self, r, t: float) -> float:
        return exponential_density(geodetic_altitude(r, self.radius, self.flattening))


class NRLMSISE00(Environment):
    """NRLMSISE-00 atmosphere (Picone et al. 2002), through the pymsis package (``pip install urania[msis]``).

    Density depends on altitude, latitude, local time and solar and geomagnetic activity: give the 10.7 cm
    solar flux F10.7 (previous day and its 81-day mean, in solar flux units) and the daily Ap index, pick a
    preset ``activity`` ("low", "moderate", "high"), or set ``historical=True`` to use the observed values
    for each date (pymsis downloads them from CelesTrak on first use).

    Positions are inertial (urania's frame); the Earth-fixed position comes from GMST with UT1 ≈ UTC and the
    geodetic latitude, longitude and height from the WGS 84 ellipsoid.
    """

    ACTIVITY = {"low": (70.0, 4.0), "moderate": (150.0, 12.0), "high": (230.0, 40.0)}   # (F10.7, Ap)

    def __init__(self, f107=None, f107a=None, ap=None, *, activity: str = "moderate", historical: bool = False,
                 version: float = 0):
        try:
            from pymsis import msis
        except ImportError as exc:
            raise ImportError("NRLMSISE00 needs the pymsis package: pip install urania[msis]") from exc
        self._calculate = msis.calculate
        self.version = version
        self.historical = historical
        if historical:
            if any(x is not None for x in (f107, f107a, ap)):
                raise ValueError("Give either historical=True or F10.7/Ap values, not both")
            self.f107 = self.f107a = self.ap = None
            weather = "observed F10.7 and Ap (CelesTrak)"
        else:
            if activity not in self.ACTIVITY:
                from ._hints import did_you_mean

                raise ValueError(f"activity must be one of {list(self.ACTIVITY)}." + did_you_mean(activity, self.ACTIVITY))
            base_f107, base_ap = self.ACTIVITY[activity]
            self.f107 = float(base_f107 if f107 is None else f107)
            self.f107a = float(self.f107 if f107a is None else f107a)
            self.ap = float(base_ap if ap is None else ap)
            for name, value in (("f107", self.f107), ("f107a", self.f107a)):
                if not 0.0 < value < 1000.0:
                    raise ValueError(f"{name} is in solar flux units (typically 65 to 300): {value}")
            if not 0.0 <= self.ap <= 400.0:
                raise ValueError(f"ap must be between 0 and 400: {self.ap}")
            weather = f"F10.7 = {self.f107:g}, 81-day mean {self.f107a:g}, Ap = {self.ap:g}"
        name = "NRLMSISE-00" if version == 0 else f"NRLMSIS {version}"
        self.description = f"{name} (pymsis), {weather}"
        self._offset_epoch = None
        self._offset = 0.0

    def _tdb_minus_utc(self, t: float) -> float:
        """TDB - UTC [s], from astropy, refreshed every 30 days of simulated time (it changes by leap seconds)."""
        if self._offset_epoch is None or abs(t - self._offset_epoch) > 30 * 86400.0:
            from .time import Epoch

            e = Epoch(t)
            with warnings.catch_warnings():
                # Future dates lie beyond the leap-second table ("dubious year"); the last offset is used
                warnings.simplefilter("ignore")
                utc = e.to_astropy().utc
            self._offset = (e.jd - (utc.jd1 + utc.jd2)) * 86400.0
            self._offset_epoch = t
        return self._offset

    def geodetic(self, r, t: float) -> tuple[float, float, float]:
        """Geodetic latitude [deg], longitude [deg], height [km] of inertial position r at TDB seconds t."""
        from .core.frames import geodetic_scalar, gmst82

        jd_utc = 2451545.0 + (t - self._tdb_minus_utc(t)) / 86400.0
        theta = gmst82(jd_utc)
        c, s = math.cos(theta), math.sin(theta)
        x, y, z = float(r[0]), float(r[1]), float(r[2])
        lat, lon, h = geodetic_scalar(c * x + s * y, -s * x + c * y, z, 6378137.0, 1 / 298.257223563)
        return math.degrees(lat), math.degrees(lon), h / 1e3

    def many(self, r, t: float):
        """Densities [kg/m³] at N×3 inertial positions at one time, in a single pymsis call."""
        import numpy as np

        from .core.frames import geodetic, gmst82, rotate_z

        r = np.atleast_2d(np.asarray(r, dtype=float))
        offset = self._tdb_minus_utc(t)
        lat, lon, h = geodetic(rotate_z(r, gmst82(2451545.0 + (t - offset) / 86400.0)), 6378137.0,
                               1 / 298.257223563)
        n = len(r)
        date = np.datetime64("2000-01-01T12:00:00", "us") + np.timedelta64(round((t - offset) * 1e6), "us")
        dates = np.full(n, date)
        if self.historical:
            out = self._calculate(dates, np.degrees(lon), np.degrees(lat), h / 1e3, version=self.version)
        else:
            out = self._calculate(dates, np.degrees(lon), np.degrees(lat), h / 1e3, f107s=np.full(n, self.f107),
                                  f107as=np.full(n, self.f107a), aps=np.full((n, 7), self.ap), version=self.version)
        return np.asarray(out)[..., 0].reshape(n)

    def __call__(self, r, t: float) -> float:
        import numpy as np

        lat, lon, h_km = self.geodetic(r, t)
        utc_us = round((t - self._tdb_minus_utc(t)) * 1e6)
        date = np.datetime64("2000-01-01T12:00:00", "us") + np.timedelta64(utc_us, "us")
        if self.historical:
            out = self._calculate(date, lon, lat, h_km, version=self.version)
        else:
            out = self._calculate(date, lon, lat, h_km, f107s=[self.f107], f107as=[self.f107a],
                                  aps=[[self.ap] * 7], version=self.version)
        return float(np.ravel(out)[0])

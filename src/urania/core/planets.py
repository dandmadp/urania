"""Approximate planet positions from mean Keplerian elements. SI units.

E. M. Standish, "Keplerian Elements for Approximate Positions of the Major Planets" (JPL Solar System
Dynamics, https://ssd.jpl.nasa.gov/planets/approx_pos.html), table 1, valid 1800 AD to 2050 AD.
Elements refer to the mean ecliptic and equinox of J2000. Errors are below about 1' for the inner
planets and several arcminutes for the outer planets. "Earth" is the Earth-Moon barycenter.
"""

import math

import numpy as np

from .elements import coe_to_rv
from .kepler import _wrap_2pi, mean_to_true
from .sun import AU

JD_J2000 = 2451545.0
OBLIQUITY_J2000 = math.radians(23.43928)  # obliquity of the ecliptic at J2000 (Standish)

# a [AU], e, I [deg], L [deg], ϖ (longitude of perihelion) [deg], Ω [deg]; then the rates per Julian century
STANDISH = {
    "Mercury": ((0.38709927, 0.20563593, 7.00497902, 252.25032350, 77.45779628, 48.33076593),
                (0.00000037, 0.00001906, -0.00594749, 149472.67411175, 0.16047689, -0.12534081)),
    "Venus": ((0.72333566, 0.00677672, 3.39467605, 181.97909950, 131.60246718, 76.67984255),
              (0.00000390, -0.00004107, -0.00078890, 58517.81538729, 0.00268329, -0.27769418)),
    "Earth": ((1.00000261, 0.01671123, -0.00001531, 100.46457166, 102.93768193, 0.0),
              (0.00000562, -0.00004392, -0.01294668, 35999.37244981, 0.32327364, 0.0)),
    "Mars": ((1.52371034, 0.09339410, 1.84969142, -4.55343205, -23.94362959, 49.55953891),
             (0.00001847, 0.00007882, -0.00813131, 19140.30268499, 0.44441088, -0.29257343)),
    "Jupiter": ((5.20288700, 0.04838624, 1.30439695, 34.39644051, 14.72847983, 100.47390909),
                (-0.00011607, -0.00013253, -0.00183714, 3034.74612775, 0.21252668, 0.20469106)),
    "Saturn": ((9.53667594, 0.05386179, 2.48599187, 49.95424423, 92.59887831, 113.66242448),
               (-0.00125060, -0.00050991, 0.00193609, 1222.49362201, -0.41897216, -0.28867794)),
    "Uranus": ((19.18916464, 0.04725744, 0.77263783, 313.23810451, 170.95427630, 74.01692503),
               (-0.00196176, -0.00004397, -0.00242939, 428.48202785, 0.40805281, 0.04240589)),
    "Neptune": ((30.06992276, 0.00859048, 1.77004347, -55.12002969, 44.96476227, 131.78422574),
                (0.00026291, 0.00005105, 0.00035372, 218.45945325, -0.32241464, -0.00508664)),
}

VALID_JD = (2378496.5, 2469807.5)  # 1800-01-01 to 2050-12-31


def mean_elements(name: str, jd_tdb: float) -> tuple[float, float, float, float, float, float]:
    """Mean elements of a planet at jd_tdb (Standish).

    Returns:
        (a [m], e, inc, raan, argp, M) with angles in rad, referred to the J2000 ecliptic
    """
    if name not in STANDISH:
        raise ValueError(f"No ephemeris for {name!r} (available: {', '.join(STANDISH)})")
    T = (jd_tdb - JD_J2000) / 36525.0
    a, e, inc, L, varpi, raan = (x0 + rate * T for x0, rate in zip(*STANDISH[name], strict=True))
    inc, L, varpi, raan = (math.radians(x) for x in (inc, L, varpi, raan))
    return a * AU, e, inc, _wrap_2pi(raan), _wrap_2pi(varpi - raan), _wrap_2pi(L - varpi)


def heliocentric_state(name: str, jd_tdb: float, mu_sun: float) -> tuple[np.ndarray, np.ndarray]:
    """Heliocentric position [m] and velocity [m/s] in the J2000 ecliptic frame.

    The velocity is the two-body velocity of the osculating ellipse with the given μ of the Sun.
    """
    a, e, inc, raan, argp, M = mean_elements(name, jd_tdb)
    return coe_to_rv(a * (1.0 - e * e), e, inc, raan, argp, mean_to_true(M, e), mu_sun)


def ecliptic_longitude(name: str, jd_tdb: float) -> float:
    """Heliocentric ecliptic longitude [rad] in [0, 2π)."""
    a, e, inc, raan, argp, M = mean_elements(name, jd_tdb)
    nu = mean_to_true(M, e)
    # Longitude of the projected position: atan2 of the in-orbit argument of latitude rotated by Ω
    u = argp + nu
    return _wrap_2pi(raan + math.atan2(math.cos(inc) * math.sin(u), math.cos(u)))


def ecliptic_to_equatorial(vec) -> np.ndarray:
    """Rotate vectors from the J2000 ecliptic to the J2000 equator (about x by the obliquity)."""
    c, s = math.cos(OBLIQUITY_J2000), math.sin(OBLIQUITY_J2000)
    rot = np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])
    return np.asarray(vec, dtype=float) @ rot.T

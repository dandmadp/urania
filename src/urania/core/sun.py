"""Sun position and eclipses by a spherical body. SI units, angles in rad."""

import math

import numpy as np

AU = 149597870700.0          # astronomical unit [m] (IAU 2012)
SUN_RADIUS = 6.957e8         # [m] (IAU 2015 nominal)


def sun_position(jd_tdb):
    """Geocentric Sun position [m], equator and equinox of date (Vallado algorithm 29).

    Low-precision Astronomical Almanac formulas: about 0.01° in direction, good from 1950 to 2050.
    jd_tdb may be a scalar (returns a 3-vector) or an array of length N (returns N×3).
    """
    jd = np.asarray(jd_tdb, dtype=float)
    T = (jd - 2451545.0) / 36525.0
    lam_m = np.radians(np.mod(280.460 + 36000.771 * T, 360.0))        # mean longitude
    M = np.radians(np.mod(357.5291092 + 35999.05034 * T, 360.0))      # mean anomaly
    lam = lam_m + np.radians(1.914666471 * np.sin(M) + 0.019994643 * np.sin(2.0 * M))
    r_au = 1.000140612 - 0.016708617 * np.cos(M) - 0.000139589 * np.cos(2.0 * M)
    eps = np.radians(23.439291 - 0.0130042 * T)                     # obliquity
    out = (r_au * AU)[..., None] * np.stack((np.cos(lam), np.cos(eps) * np.sin(lam),
                                             np.sin(eps) * np.sin(lam)), axis=-1)
    return out


def disk_angles(r_sat, r_sun, R_body: float, R_sun: float = SUN_RADIUS):
    """Apparent radii of the Sun (a) and the body (b) and their separation (c), seen from r_sat [rad].

    Positions are relative to the body's center. Accepts 3-vectors or N×3 arrays.
    """
    r_sat = np.asarray(r_sat, dtype=float)
    to_sun = np.asarray(r_sun, dtype=float) - r_sat
    d_sun = np.linalg.norm(to_sun, axis=-1)
    d_body = np.linalg.norm(r_sat, axis=-1)
    a = np.arcsin(np.minimum(R_sun / d_sun, 1.0))
    b = np.arcsin(np.minimum(R_body / d_body, 1.0))
    cos_c = np.sum(-r_sat * to_sun, axis=-1) / (d_body * d_sun)
    c = np.arccos(np.clip(cos_c, -1.0, 1.0))
    return a, b, c


def illumination(r_sat, r_sun, R_body: float, R_sun: float = SUN_RADIUS) -> float:
    """Visible fraction of the Sun's disk from r_sat (1 = sunlit, 0 = umbra).

    Conical shadow with partial overlap of the two disks (Montenbruck & Gill, Satellite Orbits, 3.4.2).
    Positions are relative to the shadowing body's center.
    """
    a, b, c = (float(x) for x in disk_angles(r_sat, r_sun, R_body, R_sun))
    if c >= a + b:
        return 1.0                                   # disks apart: full Sun
    if c <= b - a:
        return 0.0                                   # Sun fully behind the body: umbra
    if c <= a - b:
        return 1.0 - (b * b) / (a * a)               # body inside the Sun's disk: annular
    x = (c * c + a * a - b * b) / (2.0 * c)
    y = math.sqrt(max(a * a - x * x, 0.0))
    area = (a * a * math.acos(max(-1.0, min(1.0, x / a)))
            + b * b * math.acos(max(-1.0, min(1.0, (c - x) / b))) - c * y)
    return 1.0 - area / (math.pi * a * a)


def penumbra_function(r_sat, r_sun, R_body: float, R_sun: float = SUN_RADIUS) -> float:
    """c - (a + b): negative inside the penumbra or umbra, zero at penumbra entry/exit."""
    a, b, c = disk_angles(r_sat, r_sun, R_body, R_sun)
    return c - (a + b)


def umbra_function(r_sat, r_sun, R_body: float, R_sun: float = SUN_RADIUS) -> float:
    """c - (b - a): negative inside the umbra, zero at umbra entry/exit."""
    a, b, c = disk_angles(r_sat, r_sun, R_body, R_sun)
    return c - (b - a)


def beta_angle(r, v, r_sun) -> np.ndarray:
    """Angle between the orbit plane and the Sun direction [rad], positive on the orbit-normal side.

    β = asin(ĥ · ŝ). Works on single states or N×3 arrays.
    """
    h = np.cross(r, v)
    h = h / np.linalg.norm(h, axis=-1, keepdims=True)
    s = np.asarray(r_sun, dtype=float)
    s = s / np.linalg.norm(s, axis=-1, keepdims=True)
    return np.arcsin(np.clip(np.sum(h * s, axis=-1), -1.0, 1.0))


def eclipse_fraction(beta: float, radius: float, body_radius: float) -> float:
    """Fraction of a circular orbit inside a cylindrical shadow, from the beta angle.

    f = acos(√(r² - R²) / (r cos β)) / π when |β| < asin(R/r), else 0 (S. Rickman, "Introduction to On-Orbit
    Thermal Environments", NASA TFAWS, eclipse fraction).
    """
    if abs(beta) >= math.asin(body_radius / radius):
        return 0.0
    return math.acos(math.sqrt(radius**2 - body_radius**2) / (radius * math.cos(beta))) / math.pi

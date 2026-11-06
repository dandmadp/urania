"""Third-body perturbations and solar radiation pressure around Earth. SI units, inertial frame.

Moon position: Vallado algorithm 31 (low precision, about 0.3° in longitude and 0.2° in latitude).
Sun position: core.sun.sun_position (Vallado algorithm 29).
Accelerations: Montenbruck & Gill, "Satellite Orbits", eqs. 3.37 (third body) and 3.75 (radiation pressure).
"""

import math

import numpy as np

from .sun import AU, illumination

EARTH_RADIUS_VALLADO = 6378137.0     # Vallado's Earth radius for the Moon distance (algorithm 31)
SOLAR_PRESSURE = 4.56e-6             # solar radiation pressure at 1 AU [N/m²] (Montenbruck & Gill eq. 3.70)
MU_SUN = 1.32712440041e20            # DE440 [m³/s²]
MU_MOON = 4.902800118e12             # DE440 [m³/s²]


def moon_position(jd_tdb):
    """Geocentric Moon position [m] in the mean equator of date (Vallado algorithm 31).

    Accepts a scalar or an array of TDB Julian dates; returns shape (3,) or (N, 3).
    """
    jd = np.asarray(jd_tdb, dtype=float)
    T = (jd - 2451545.0) / 36525.0
    d = np.radians
    lon = (218.32 + 481267.8813 * T
           + 6.29 * np.sin(d(134.9 + 477198.85 * T)) - 1.27 * np.sin(d(259.2 - 413335.38 * T))
           + 0.66 * np.sin(d(235.7 + 890534.23 * T)) + 0.21 * np.sin(d(269.9 + 954397.70 * T))
           - 0.19 * np.sin(d(357.5 + 35999.05 * T)) - 0.11 * np.sin(d(186.6 + 966404.05 * T)))
    lat = (5.13 * np.sin(d(93.3 + 483202.03 * T)) + 0.28 * np.sin(d(228.2 + 960400.87 * T))
           - 0.28 * np.sin(d(318.3 + 6003.18 * T)) - 0.17 * np.sin(d(217.6 - 407332.20 * T)))
    parallax = (0.9508 + 0.0518 * np.cos(d(134.9 + 477198.85 * T)) + 0.0095 * np.cos(d(259.2 - 413335.38 * T))
                + 0.0078 * np.cos(d(235.7 + 890534.23 * T)) + 0.0028 * np.cos(d(269.9 + 954397.70 * T)))
    eps = d(23.439291 - 0.0130042 * T)
    lon, lat, parallax = d(lon), d(lat), d(parallax)
    dist = EARTH_RADIUS_VALLADO / np.sin(parallax)
    cl, sl, cb, sb = np.cos(lon), np.sin(lon), np.cos(lat), np.sin(lat)
    r = np.asarray(dist)[..., None] * np.stack([cb * cl,
                         np.cos(eps) * cb * sl - np.sin(eps) * sb,
                         np.sin(eps) * cb * sl + np.cos(eps) * sb], axis=-1)
    return r


def accel_third_body(r: np.ndarray, r_body: np.ndarray, mu_body: float) -> np.ndarray:
    """Acceleration of a satellite at r by a third body at r_body, relative to the central body.

    a = μ₃ ((s - r)/|s - r|³ - s/|s|³) (Montenbruck & Gill eq. 3.37; Vallado eq. 8-34).
    """
    d = r_body - r
    return mu_body * (d / (d @ d) ** 1.5 - r_body / (r_body @ r_body) ** 1.5)


def accel_srp(r: np.ndarray, r_sun: np.ndarray, cr_area_mass: float, body_radius: float) -> np.ndarray:
    """Solar radiation pressure on a sphere ("cannonball"), with Earth's conical shadow.

    a = ν P⊙ C_R (A/m) AU² (r - r⊙)/|r - r⊙|³ (Montenbruck & Gill eq. 3.75), ν = sunlit fraction of the
    solar disk (core.sun.illumination, Montenbruck & Gill section 3.4.2).
    """
    nu = illumination(r, r_sun, body_radius)
    if nu == 0.0:
        return np.zeros(3)
    d = r - r_sun
    return nu * SOLAR_PRESSURE * cr_area_mass * AU**2 * d / (d @ d) ** 1.5


def accel_zonal(r: np.ndarray, mu: float, R: float, J) -> np.ndarray:
    """Zonal harmonics J2..Jn about the inertial z axis; J = (J2, J3, ..., Jn), unnormalized.

    For the potential term -μ Jn Rⁿ Pn(s)/r^(n+1) with s = z/r:
    a_n = μ Jn Rⁿ / r^(n+2) [(n+1) Pn(s) r̂ - Pn'(s)(ẑ - s r̂)] (gradient of Vallado eq. 8-19 with m = 0).
    Legendre polynomials by Bonnet's recursion, derivatives by P'_{n+1} = P'_{n-1} + (2n+1) Pn.
    """
    r_norm = math.sqrt(r @ r)
    s = r[2] / r_norm
    r_hat = r / r_norm
    z_minus = np.array([0.0, 0.0, 1.0]) - s * r_hat
    n_max = len(J) + 1
    P = [1.0, s]
    dP = [0.0, 1.0]
    for n in range(1, n_max):
        P.append(((2 * n + 1) * s * P[n] - n * P[n - 1]) / (n + 1))
        dP.append(dP[n - 1] + (2 * n + 1) * P[n])
    radial, axial = 0.0, 0.0
    ratio = R / r_norm
    scale = ratio
    for n in range(2, n_max + 1):
        scale *= ratio                      # (R/r)^n
        Jn = J[n - 2]
        radial += Jn * scale * (n + 1) * P[n]
        axial += Jn * scale * dP[n]
    k = mu / (r_norm * r_norm)
    return k * (radial * r_hat - axial * z_minus)


def solar_flux_scale(r_sun: np.ndarray) -> float:
    """(AU/|r⊙|)²: the solar flux relative to its value at 1 AU."""
    return AU**2 / float(r_sun @ r_sun)


__all__ = ["moon_position", "accel_third_body", "accel_srp", "accel_zonal", "MU_SUN", "MU_MOON",
           "SOLAR_PRESSURE"]

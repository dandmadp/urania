"""Earth rotation, Earth-fixed coordinates and topocentric look angles. SI units, angles in rad.

urania's inertial frame has its z axis along Earth's spin axis (true equator of date, like TEME).
The Earth-fixed frame (ECEF) follows from a rotation about z by the Greenwich mean sidereal time;
polar motion (about 0.3", under 10 m at the surface) is ignored.
"""

import math

import numpy as np

TWO_PI = 2.0 * math.pi
JD_J2000 = 2451545.0


def gmst82(jd_ut1):
    """Greenwich mean sidereal time [rad] from the UT1 Julian date (IAU 1982, Vallado eq. 3-47).

    θ = 67310.54841 s + (876600 h + 8640184.812866 s) T + 0.093104 s T² - 6.2e-6 s T³,
    T = (JD_UT1 - 2451545) / 36525. Works on scalars and arrays.
    """
    jd = np.asarray(jd_ut1, dtype=float)
    T = (jd - JD_J2000) / 36525.0
    seconds = (67310.54841 + (876600.0 * 3600.0 + 8640184.812866) * T
               + 0.093104 * T**2 - 6.2e-6 * T**3)
    theta = np.mod(seconds % 86400.0 / 240.0 * math.pi / 180.0, TWO_PI)
    theta = np.where(theta >= TWO_PI, 0.0, theta)
    return float(theta) if theta.ndim == 0 else theta


def rotate_z(r, theta):
    """Rotate vectors r (3 or N×3) by -θ about z: inertial → Earth-fixed for θ = GMST.

    θ may be a scalar or an array of length N.
    """
    r = np.asarray(r, dtype=float)
    c, s = np.cos(theta), np.sin(theta)
    x, y, z = r[..., 0], r[..., 1], r[..., 2]
    xr, yr = c * x + s * y, -s * x + c * y
    return np.stack((xr, yr, np.broadcast_to(z, np.shape(xr))), axis=-1)


def geodetic_scalar(x: float, y: float, z: float, R: float, f: float) -> tuple[float, float, float]:
    """Scalar version of `geodetic` with math functions (fast inside integration loops)."""
    p = math.hypot(x, y)
    lon = math.atan2(y, x)
    if f == 0.0:
        return math.atan2(z, p), lon, math.hypot(p, z) - R
    e2 = f * (2.0 - f)
    lat = math.atan2(z, p * (1.0 - e2))
    for _ in range(6):
        sin_lat = math.sin(lat)
        N = R / math.sqrt(1.0 - e2 * sin_lat * sin_lat)
        lat = math.atan2(z + N * e2 * sin_lat, p)
    sin_lat, cos_lat = math.sin(lat), math.cos(lat)
    return lat, lon, p * cos_lat + z * sin_lat - R * math.sqrt(1.0 - e2 * sin_lat * sin_lat)


def geodetic(r, R: float, f: float):
    """Earth-fixed position [m] → geodetic latitude [rad], longitude [rad], height [m].

    Iterates the geodetic latitude (Vallado algorithm 12); the height formula is well conditioned at
    the poles. With f = 0 the latitude is geocentric and the height is |r| - R.
    Longitude is in (-π, π]. Accepts a 3-vector or an N×3 array.
    """
    r = np.asarray(r, dtype=float)
    if r.ndim == 1:
        return geodetic_scalar(float(r[0]), float(r[1]), float(r[2]), R, f)
    x, y, z = r[..., 0], r[..., 1], r[..., 2]
    p = np.hypot(x, y)
    lon = np.arctan2(y, x)
    if f == 0.0:
        lat = np.arctan2(z, p)
        h = np.hypot(p, z) - R
    else:
        e2 = f * (2.0 - f)
        lat = np.arctan2(z, p * (1.0 - e2))
        for _ in range(6):
            sin_lat = np.sin(lat)
            N = R / np.sqrt(1.0 - e2 * sin_lat * sin_lat)
            lat = np.arctan2(z + N * e2 * sin_lat, p)
        sin_lat, cos_lat = np.sin(lat), np.cos(lat)
        h = p * cos_lat + z * sin_lat - R * np.sqrt(1.0 - e2 * sin_lat * sin_lat)
    return lat, lon, h


def geodetic_to_ecef(lat: float, lon: float, h: float, R: float, f: float) -> np.ndarray:
    """Geodetic latitude, longitude [rad] and height [m] → Earth-fixed position [m]."""
    e2 = f * (2.0 - f)
    sin_lat = math.sin(lat)
    N = R / math.sqrt(1.0 - e2 * sin_lat * sin_lat)
    return np.array([(N + h) * math.cos(lat) * math.cos(lon),
                     (N + h) * math.cos(lat) * math.sin(lon),
                     (N * (1.0 - e2) + h) * sin_lat])


def look_angles(r_sat_ecef, r_site_ecef, lat: float, lon: float):
    """Azimuth [rad, from north toward east], elevation [rad] and range [m] seen from a site.

    The line of sight is rotated into the site's east-north-up frame (geodetic latitude and
    longitude of the site; Vallado algorithm 27). Accepts one satellite position or N×3 positions.
    """
    rho = np.asarray(r_sat_ecef, dtype=float) - np.asarray(r_site_ecef, dtype=float)
    sl, cl = math.sin(lat), math.cos(lat)
    so, co = math.sin(lon), math.cos(lon)
    x, y, z = rho[..., 0], rho[..., 1], rho[..., 2]
    east = -so * x + co * y
    north = -sl * co * x - sl * so * y + cl * z
    up = cl * co * x + cl * so * y + sl * z
    rng = np.sqrt(east**2 + north**2 + up**2)
    el = np.arcsin(np.clip(up / rng, -1.0, 1.0))
    az = np.mod(np.arctan2(east, north), TWO_PI)
    az = np.where(az >= TWO_PI, 0.0, az)          # a tiny negative angle can round to 2π
    if np.ndim(az) == 0:
        return float(az), float(el), float(rng)
    return az, el, rng

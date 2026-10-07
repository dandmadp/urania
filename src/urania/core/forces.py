"""Perturbation models: accelerations and atmospheric density. SI units, inertial frame."""

import bisect
import math

import numpy as np


def accel_twobody(r: np.ndarray, mu: float) -> np.ndarray:
    """Point-mass gravity of the central body: a = -μ r / |r|³."""
    r_norm = np.sqrt(r @ r)
    return -mu * r / r_norm**3


def accel_j2(r: np.ndarray, mu: float, R: float, J2: float) -> np.ndarray:
    """J2 oblateness acceleration (Curtis eq. 12.30, Vallado eq. 8-30).

    Assumes the body's spin axis is the inertial z axis.
    """
    x, y, z = r
    r2 = r @ r
    r_norm = np.sqrt(r2)
    factor = -1.5 * J2 * mu * R**2 / r_norm**5
    k = 5.0 * z * z / r2
    return factor * np.array([x * (1.0 - k), y * (1.0 - k), z * (3.0 - k)])


def accel_drag(r: np.ndarray, v: np.ndarray, rho: float, ballistic: float,
               rotation_rate: float) -> np.ndarray:
    """Atmospheric drag acceleration a = -½ ρ (C_D A / m) |v_rel| v_rel (Vallado eq. 8-28).

    Args:
        rho: atmospheric density [kg/m³]
        ballistic: C_D·A/m [m²/kg]
        rotation_rate: body rotation rate [rad/s]. The atmosphere co-rotates rigidly about z.
    """
    v_rel = v - rotation_rate * np.array([-r[1], r[0], 0.0])  # v - ω×r
    return -0.5 * rho * ballistic * np.sqrt(v_rel @ v_rel) * v_rel


# ---------------------------------------------------------------- atmospheric density
# Exponential atmosphere (Vallado 4th ed. table 8-4): ρ = ρ₀ exp(-(h - h₀)/H) per altitude band.
# A static model for mean solar activity; real density varies several-fold with solar activity.
# (base altitude h₀ [km], base density ρ₀ [kg/m³], scale height H [km])
VALLADO_TABLE = (
    (0, 1.225, 7.249),
    (25, 3.899e-2, 6.349),
    (30, 1.774e-2, 6.682),
    (40, 3.972e-3, 7.554),
    (50, 1.057e-3, 8.382),
    (60, 3.206e-4, 7.714),
    (70, 8.770e-5, 6.549),
    (80, 1.905e-5, 5.799),
    (90, 3.396e-6, 5.382),
    (100, 5.297e-7, 5.877),
    (110, 9.661e-8, 7.263),
    (120, 2.438e-8, 9.473),
    (130, 8.484e-9, 12.636),
    (140, 3.845e-9, 16.149),
    (150, 2.070e-9, 22.523),
    (180, 5.464e-10, 29.740),
    (200, 2.789e-10, 37.105),
    (250, 7.248e-11, 45.546),
    (300, 2.418e-11, 53.628),
    (350, 9.518e-12, 53.298),
    (400, 3.725e-12, 58.515),
    (450, 1.585e-12, 60.828),
    (500, 6.967e-13, 63.822),
    (600, 1.454e-13, 71.835),
    (700, 3.614e-14, 88.667),
    (800, 1.170e-14, 124.64),
    (900, 5.245e-15, 181.05),
    (1000, 3.019e-15, 268.00),
)
_BASE_ALTS = [row[0] * 1e3 for row in VALLADO_TABLE]


def exponential_density(altitude: float) -> float:
    """Altitude [m] → atmospheric density [kg/m³]. Above 1000 km the last band is extended."""
    if altitude < 0.0:
        altitude = 0.0
    i = bisect.bisect_right(_BASE_ALTS, altitude) - 1
    h0_km, rho0, H_km = VALLADO_TABLE[i]
    return rho0 * math.exp(-(altitude - h0_km * 1e3) / (H_km * 1e3))

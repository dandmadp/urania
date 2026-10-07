"""Impulsive maneuver formulas. SI units. Curtis ch. 6, Vallado ch. 6.

Transfers between circular orbits. Δv values are signed tangential components (+ is prograde).
"""

import math

import numpy as np
from scipy.optimize import minimize_scalar

from .twobody import vis_viva as _vis_viva


def hohmann(r1: float, r2: float, mu: float) -> tuple[float, float, float]:
    """Hohmann transfer between circular orbits r1 → r2 (Vallado algorithm 36).

    Returns:
        (dv1, dv2, tof): tangential Δv of both burns [m/s] (negative when descending), transfer time [s]
    """
    a_t = 0.5 * (r1 + r2)
    dv1 = _vis_viva(r1, a_t, mu) - math.sqrt(mu / r1)
    dv2 = math.sqrt(mu / r2) - _vis_viva(r2, a_t, mu)
    tof = math.pi * math.sqrt(a_t**3 / mu)
    return dv1, dv2, tof


def bielliptic(r1: float, rb: float, r2: float, mu: float) -> tuple[float, float, float, float]:
    """Bi-elliptic transfer r1 → rb (intermediate apoapsis) → r2 between circular orbits (Vallado algorithm 37).

    Returns:
        (dv1, dv2, dv3, tof): tangential Δv of the three burns [m/s], transfer time [s]
    """
    if rb < max(r1, r2):
        raise ValueError(f"Intermediate apoapsis rb={rb} must be larger than r1 and r2")
    a1 = 0.5 * (r1 + rb)
    a2 = 0.5 * (rb + r2)
    dv1 = _vis_viva(r1, a1, mu) - math.sqrt(mu / r1)
    dv2 = _vis_viva(rb, a2, mu) - _vis_viva(rb, a1, mu)
    dv3 = math.sqrt(mu / r2) - _vis_viva(r2, a2, mu)
    tof = math.pi * (math.sqrt(a1**3 / mu) + math.sqrt(a2**3 / mu))
    return dv1, dv2, dv3, tof


def plane_change_dv(v: float, dtheta: float) -> float:
    """Δv to rotate the orbit plane by dtheta [rad] at constant speed: 2v sin(Δθ/2)."""
    return 2.0 * v * math.sin(0.5 * abs(dtheta))


def combined_dv(v1: float, v2: float, dtheta: float) -> float:
    """Δv for a combined speed change and plane change (law of cosines)."""
    return math.sqrt(max(v1 * v1 + v2 * v2 - 2.0 * v1 * v2 * math.cos(dtheta), 0.0))


def hohmann_plane_change(r1: float, r2: float, dtheta: float, mu: float,
                         split: float) -> tuple[float, float, float]:
    """Hohmann transfer with a plane change.

    Args:
        dtheta: angle between the two orbit planes [rad]
        split: fraction of the plane change done at the first burn (0 to 1); the rest at the second.

    Returns:
        (dv1, dv2, tof): Δv magnitudes of both burns [m/s], transfer time [s]
    """
    a_t = 0.5 * (r1 + r2)
    dv1 = combined_dv(math.sqrt(mu / r1), _vis_viva(r1, a_t, mu), split * dtheta)
    dv2 = combined_dv(_vis_viva(r2, a_t, mu), math.sqrt(mu / r2), (1.0 - split) * dtheta)
    return dv1, dv2, math.pi * math.sqrt(a_t**3 / mu)


def optimal_plane_split(r1: float, r2: float, dtheta: float, mu: float) -> float:
    """Plane change split (0 to 1) that minimizes the total Δv."""
    if dtheta == 0.0:
        return 0.0
    res = minimize_scalar(lambda s: sum(hohmann_plane_change(r1, r2, dtheta, mu, s)[:2]),
                          bounds=(0.0, 1.0), method="bounded", options={"xatol": 1e-10})
    return float(res.x)


def rotate(vec, axis, angle: float) -> np.ndarray:
    """Rotate a vector by angle [rad] about the unit axis (Rodrigues' formula)."""
    vec = np.asarray(vec, dtype=float)
    k = np.asarray(axis, dtype=float)
    c, s = math.cos(angle), math.sin(angle)
    return vec * c + np.cross(k, vec) * s + k * (k @ vec) * (1.0 - c)

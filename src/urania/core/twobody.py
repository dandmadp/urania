"""Basic two-body quantities. All inputs and outputs are SI."""

import math

import numpy as np


def period(a: float, mu: float) -> float:
    """Orbital period [s]. Elliptic orbits (a > 0) only."""
    if a <= 0.0:
        raise ValueError(f"The period is defined only for elliptic orbits (a > 0): a={a}")
    return 2.0 * math.pi * math.sqrt(a**3 / mu)


def mean_motion(a: float, mu: float) -> float:
    """Mean motion n [rad/s]. Uses |a| for hyperbolas (a < 0)."""
    return math.sqrt(mu / abs(a) ** 3)


def specific_energy(a: float, mu: float) -> float:
    """Specific mechanical energy ε = -μ/(2a) [J/kg]. Zero for a parabola (a = inf)."""
    return -mu / (2.0 * a)


def vis_viva(r: float, a: float, mu: float) -> float:
    """Speed [m/s] at radius r on an orbit with semi-major axis a: v = √(μ(2/r - 1/a))."""
    return math.sqrt(mu * (2.0 / r - 1.0 / a))


def circular_velocity(r: float, mu: float) -> float:
    """Circular orbit speed at radius r [m/s]."""
    return math.sqrt(mu / r)


def escape_velocity(r: float, mu: float) -> float:
    """Escape speed at radius r [m/s]."""
    return math.sqrt(2.0 * mu / r)


def synchronous_radius(mu: float, rotation_rate: float) -> float:
    """Radius [m] of the circular orbit whose period equals the body's rotation (GEO for Earth)."""
    return (mu / rotation_rate**2) ** (1.0 / 3.0)


def semi_major_axis(r, v, mu: float):
    """Osculating semi-major axis [m] from a state vector: a = 1 / (2/|r| - |v|²/μ) (vis-viva).

    Returns a length-N array if r and v are N×3 arrays.
    """
    r = np.asarray(r, dtype=float)
    v = np.asarray(v, dtype=float)
    r_norm = np.linalg.norm(r, axis=-1)
    v2 = np.sum(v * v, axis=-1)
    return 1.0 / (2.0 / r_norm - v2 / mu)

"""Patched-conic transfers between bodies. SI units. Curtis ch. 8.

A transfer between two bodies orbiting the same primary is a Hohmann ellipse around that primary; at each end
the spacecraft leaves or enters the body's sphere of influence on a hyperbola whose excess speed v∞ equals the
Hohmann burn there.
"""

import math

from .kepler import _wrap_2pi, _wrap_pi
from .maneuvers import hohmann


def soi_radius(a: float, mu: float, mu_primary: float) -> float:
    """Sphere of influence radius a·(m/M)^(2/5) of a body orbiting at a (Curtis eq. 8.25)."""
    return a * (mu / mu_primary) ** 0.4


def synodic_period(T1: float, T2: float) -> float:
    """Synodic period T1·T2/|T1 - T2| (Curtis eq. 8.10); inf for equal periods."""
    if T1 == T2:
        return math.inf
    return T1 * T2 / abs(T1 - T2)


def hohmann_v_inf(r1: float, r2: float, mu: float) -> tuple[float, float, float]:
    """Hyperbolic excess speeds of a Hohmann transfer between circular orbits r1 → r2 (Curtis eqs. 8.35, 8.36).

    Returns:
        (v_inf_departure, v_inf_arrival, tof): excess speeds [m/s] (magnitudes), transfer time [s]
    """
    dv1, dv2, tof = hohmann(r1, r2, mu)
    return abs(dv1), abs(dv2), tof


def phase_angle(r1: float, r2: float, mu: float) -> float:
    """Angle [rad] by which the target leads the origin at departure for a Hohmann transfer (Curtis eq. 8.13).

    φ0 = π - n2·t12, in [-π, π); negative means the target trails the origin.
    """
    tof = math.pi * math.sqrt((0.5 * (r1 + r2)) ** 3 / mu)
    return _wrap_pi(math.pi - math.sqrt(mu / r2**3) * tof)


def arrival_phase_angle(r1: float, r2: float, mu: float) -> float:
    """Angle [rad] by which the origin leads the target at arrival (Curtis eq. 8.14): n1·t12 - π, in [-π, π)."""
    tof = math.pi * math.sqrt((0.5 * (r1 + r2)) ** 3 / mu)
    return _wrap_pi(math.sqrt(mu / r1**3) * tof - math.pi)


def wait_time(phase_now: float, phase_needed: float, n_origin: float, n_target: float) -> float:
    """Time [s] until the target's lead angle over the origin changes from phase_now to phase_needed.

    The lead angle changes at the rate n_target - n_origin (Curtis eq. 8.12). Returns 0 when already aligned
    and inf when the rates are equal.
    """
    rate = n_target - n_origin
    if rate == 0.0:
        return 0.0 if _wrap_pi(phase_needed - phase_now) == 0.0 else math.inf
    return _wrap_2pi(math.copysign(1.0, rate) * (phase_needed - phase_now)) / abs(rate)


def stay_time(r1: float, r2: float, mu: float) -> float:
    """Shortest wait [s] at the target before a return Hohmann transfer can start (Curtis eq. 8.16).

    At arrival the origin leads the target by n1·t12 - π; the return needs it to lead by π - n1·t12.
    """
    n1, n2 = math.sqrt(mu / r1**3), math.sqrt(mu / r2**3)
    # The return leg runs target → origin, so the roles of the bodies swap
    return wait_time(arrival_phase_angle(r1, r2, mu), phase_angle(r2, r1, mu), n2, n1)


def departure_dv(v_inf: float, r_p: float, mu: float) -> float:
    """Δv from a circular parking orbit of radius r_p onto an escape hyperbola with excess speed v∞.

    sqrt(v∞² + 2μ/r_p) - sqrt(μ/r_p) (Curtis eq. 8.42).
    """
    return math.sqrt(v_inf * v_inf + 2.0 * mu / r_p) - math.sqrt(mu / r_p)


def capture_dv(v_inf: float, r_p: float, mu: float, ecc: float = 0.0) -> float:
    """Δv at periapsis r_p of an arrival hyperbola to enter an orbit of eccentricity ecc (< 1) there.

    sqrt(v∞² + 2μ/r_p) - sqrt(μ(1 + e)/r_p) (Curtis eq. 8.58 with eq. 8.59 for the capture orbit).
    """
    if not 0.0 <= ecc < 1.0:
        raise ValueError(f"Capture orbit eccentricity must be in [0, 1): {ecc}")
    return math.sqrt(v_inf * v_inf + 2.0 * mu / r_p) - math.sqrt(mu * (1.0 + ecc) / r_p)


def hyperbola(v_inf: float, r_p: float, mu: float) -> tuple[float, float]:
    """Eccentricity and asymptote angle β of a hyperbola with excess speed v∞ and periapsis r_p.

    e = 1 + r_p·v∞²/μ (Curtis eq. 8.38), β = acos(1/e) (eq. 8.43): angle between periapsis and the asymptote.
    """
    e = 1.0 + r_p * v_inf * v_inf / mu
    return e, math.acos(1.0 / e)


def optimal_capture_periapsis(v_inf: float, mu: float, ecc: float = 0.0) -> float:
    """Periapsis radius that minimizes the capture Δv into an orbit of eccentricity ecc (Curtis eq. 8.67).

    r_p = 2μ(1 - e) / (v∞²(1 + e)).
    """
    return 2.0 * mu * (1.0 - ecc) / (v_inf * v_inf * (1.0 + ecc))

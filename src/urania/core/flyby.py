"""Planetary flybys (gravity assists) in the patched-conic model. SI units.

Unpowered flyby: Curtis section 8.9. The hyperbolic excess velocity keeps its magnitude and turns by
δ = 2 asin(1/e), e = 1 + r_p v∞²/μ (Curtis eqs. 8.38, 8.54).
Powered flyby (a burn at periapsis joining two different v∞ magnitudes): the periapsis follows from
asin(1/e_in) + asin(1/e_out) = δ (Izzo, "Global optimization and space pruning for spacecraft trajectory design",
2010, the MGA model), and the burn is the speed difference of the two hyperbolas at periapsis.
"""

import math

import numpy as np
from scipy.optimize import brentq

from .maneuvers import rotate


def turn_angle(v_inf: float, r_p: float, mu: float) -> float:
    """Turn angle δ [rad] of the v∞ vector for an unpowered flyby with periapsis r_p (Curtis eq. 8.54)."""
    return 2.0 * math.asin(1.0 / (1.0 + r_p * v_inf * v_inf / mu))


def periapsis_for_turn(v_inf_in: float, v_inf_out: float, delta: float, mu: float) -> float:
    """Periapsis radius [m] that turns v∞_in into the direction of v∞_out by δ (inf for δ = 0).

    Equal magnitudes reduce to the unpowered relation r_p = μ/v∞² (1/sin(δ/2) - 1).
    """
    if not 0.0 <= delta < math.pi:
        raise ValueError(f"Turn angle must be in [0, π): {delta}")
    if delta == 0.0:
        return math.inf

    def residual(log_rp: float) -> float:
        rp = math.exp(log_rp)
        return (math.asin(1.0 / (1.0 + rp * v_inf_in**2 / mu))
                + math.asin(1.0 / (1.0 + rp * v_inf_out**2 / mu)) - delta)

    scale = mu / max(v_inf_in, v_inf_out) ** 2
    lo, hi = math.log(scale * 1e-12), math.log(scale * 1e12)
    if residual(hi) > 0.0:
        return math.inf   # the turn is negligible
    return math.exp(brentq(residual, lo, hi, xtol=1e-13))


def powered_dv(v_inf_in: float, v_inf_out: float, r_p: float, mu: float) -> float:
    """Δv of a tangential burn at periapsis r_p joining hyperbolas with excess speeds v∞_in and v∞_out."""
    if math.isinf(r_p):
        return abs(v_inf_out - v_inf_in)
    return abs(math.sqrt(v_inf_out**2 + 2.0 * mu / r_p) - math.sqrt(v_inf_in**2 + 2.0 * mu / r_p))


def flyby_axis(v_inf_in, b_angle: float, pole=(0.0, 0.0, 1.0)) -> np.ndarray:
    """Unit normal of the flyby plane: the reference pole, made perpendicular to v∞_in, rotated by b_angle
    about v∞_in (B-plane angle). b_angle = 0 turns v∞ counterclockwise seen from the pole, π clockwise."""
    s = np.asarray(v_inf_in, dtype=float)
    s = s / np.linalg.norm(s)
    k = np.asarray(pole, dtype=float)
    k = k - (k @ s) * s
    if np.linalg.norm(k) < 1e-12:
        k = np.cross(s, [1.0, 0.0, 0.0]) if abs(s[0]) < 0.9 else np.cross(s, [0.0, 1.0, 0.0])
    k = k / np.linalg.norm(k)
    return rotate(k, s, b_angle)


def outgoing_v_inf(v_inf_in, r_p: float, mu: float, b_angle: float = 0.0,
                   pole=(0.0, 0.0, 1.0)) -> np.ndarray:
    """Outgoing excess velocity [m/s] of an unpowered flyby: v∞_in rotated by δ about the flyby normal."""
    v_in = np.asarray(v_inf_in, dtype=float)
    delta = turn_angle(float(np.linalg.norm(v_in)), r_p, mu)
    return rotate(v_in, flyby_axis(v_in, b_angle, pole), delta)

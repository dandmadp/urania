"""Lambert's problem: the orbit through two positions in a given time. SI units.

Universal-variable formulation (Curtis algorithm 5.2, Vallado algorithm 58, Bate-Mueller-White 5.3), solved by
bracketing instead of Newton iteration so it cannot diverge. Multi-revolution solutions use the same formula with
z = ψ beyond (2πN)² (Vallado section 7.6): each N ≥ 1 has a time-of-flight minimum and two solutions around it.
"""

import math

import numpy as np
from scipy.optimize import brentq, minimize_scalar

from .propagate import stumpff

TWO_PI = 2.0 * math.pi
_BIG = 1e30


def transfer_angle(r1, r2, prograde: bool = True, normal=(0.0, 0.0, 1.0)) -> float:
    """Transfer angle Δν [rad] in [0, 2π) from r1 to r2.

    Prograde means the motion is counterclockwise seen from `normal` (Curtis eq. 5.26).
    """
    r1, r2 = np.asarray(r1, dtype=float), np.asarray(r2, dtype=float)
    cross = np.cross(r1, r2)
    dnu = math.atan2(float(np.linalg.norm(cross)), float(r1 @ r2))
    ccw = float(cross @ np.asarray(normal, dtype=float)) >= 0.0
    return dnu if ccw == prograde else TWO_PI - dnu


def _functions(r1n: float, r2n: float, A: float, mu: float):
    def y_of(z: float) -> float:
        C, S = stumpff(z)
        return r1n + r2n + A * (z * S - 1.0) / math.sqrt(C)

    def tof_of(z: float) -> float:
        """Time of flight, or -inf where y < 0 (no real orbit; the time tends to 0 at y = 0)."""
        C, S = stumpff(z)
        y = r1n + r2n + A * (z * S - 1.0) / math.sqrt(C)
        if y <= 0.0:
            return -math.inf
        chi = math.sqrt(y / C)
        return (chi**3 * S + A * math.sqrt(y)) / math.sqrt(mu)

    return y_of, tof_of


def lambert(r1, r2, tof: float, mu: float, *, prograde: bool = True, revolutions: int = 0,
            branch: str = "left", normal=(0.0, 0.0, 1.0)) -> tuple[np.ndarray, np.ndarray]:
    """Velocities at r1 and r2 of the conic from r1 to r2 in time tof.

    Args:
        tof: time of flight [s] (> 0)
        prograde: direction of motion seen from `normal` (default +z)
        revolutions: number of complete revolutions before arrival (0 = direct)
        branch: for revolutions ≥ 1, "left" (smaller ψ, lower-energy side) or "right" (larger ψ)

    Returns:
        (v1, v2) [m/s]
    """
    r1, r2 = np.asarray(r1, dtype=float), np.asarray(r2, dtype=float)
    if not tof > 0.0 or not math.isfinite(tof):
        raise ValueError(f"Time of flight must be positive and finite: {tof}")
    if revolutions < 0 or int(revolutions) != revolutions:
        raise ValueError(f"revolutions must be a non-negative integer: {revolutions}")
    if branch not in ("left", "right"):
        raise ValueError(f"branch must be 'left' or 'right': {branch!r}")
    r1n, r2n = float(np.linalg.norm(r1)), float(np.linalg.norm(r2))
    if r1n == 0.0 or r2n == 0.0:
        raise ValueError("Positions must be nonzero")
    dnu = transfer_angle(r1, r2, prograde, normal)
    sin_dnu = math.sin(dnu)
    if abs(sin_dnu) < 1e-10:
        raise ValueError(f"Transfer angle {math.degrees(dnu):.6f}° is 0° or 180°: the transfer plane is "
                         f"undefined (perturb the positions or split the transfer)")
    A = sin_dnu * math.sqrt(r1n * r2n / (1.0 - math.cos(dnu)))   # Curtis eq. 5.35
    y_of, tof_of = _functions(r1n, r2n, A, mu)

    def f(z: float) -> float:
        return tof_of(z) - tof

    N = int(revolutions)
    if N == 0:
        z_hi = TWO_PI**2 * (1.0 - 1e-12)
        if f(z_hi) < 0.0:
            raise ValueError("No direct solution: the time of flight is beyond the elliptic limit")
        z_lo = -1.0
        while f(z_lo) > 0.0:
            z_lo *= 2.0
            if z_lo < -1e5:
                raise ValueError(f"No solution: time of flight {tof} s is too short")
        z = brentq(f, z_lo, z_hi, xtol=1e-14, rtol=4 * np.finfo(float).eps, maxiter=500)
    else:
        lo, hi = (TWO_PI * N) ** 2 * (1.0 + 1e-10), (TWO_PI * (N + 1)) ** 2 * (1.0 - 1e-10)

        def t_pos(z: float) -> float:
            t = tof_of(z)
            return _BIG if t == -math.inf else t

        best = minimize_scalar(t_pos, bounds=(lo, hi), method="bounded", options={"xatol": 1e-12})
        z_min, t_min = float(best.x), float(best.fun)
        if t_min > tof:
            raise ValueError(f"No {N}-revolution solution: the minimum time of flight is {t_min:.1f} s")

        def g(z: float) -> float:
            return t_pos(z) - tof

        if branch == "left":
            z = brentq(g, lo, z_min, xtol=1e-14, maxiter=500)
        else:
            z = brentq(g, z_min, hi, xtol=1e-14, maxiter=500)

    y = y_of(z)
    f_ = 1.0 - y / r1n                      # Lagrange coefficients (Curtis eqs. 5.46)
    g_ = A * math.sqrt(y / mu)
    gdot = 1.0 - y / r2n
    v1 = (r2 - f_ * r1) / g_
    v2 = (gdot * r2 - r1) / g_
    return v1, v2


def min_tof(r1, r2, mu: float, revolutions: int, *, prograde: bool = True,
            normal=(0.0, 0.0, 1.0)) -> float:
    """Shortest time of flight [s] that has a solution with the given number of revolutions (≥ 1)."""
    r1, r2 = np.asarray(r1, dtype=float), np.asarray(r2, dtype=float)
    r1n, r2n = float(np.linalg.norm(r1)), float(np.linalg.norm(r2))
    dnu = transfer_angle(r1, r2, prograde, normal)
    A = math.sin(dnu) * math.sqrt(r1n * r2n / (1.0 - math.cos(dnu)))
    _, tof_of = _functions(r1n, r2n, A, mu)
    N = int(revolutions)
    lo, hi = (TWO_PI * N) ** 2 * (1.0 + 1e-10), (TWO_PI * (N + 1)) ** 2 * (1.0 - 1e-10)
    best = minimize_scalar(lambda z: _BIG if tof_of(z) == -math.inf else tof_of(z), bounds=(lo, hi),
                           method="bounded", options={"xatol": 1e-12})
    return float(best.fun)

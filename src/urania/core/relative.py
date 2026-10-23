"""Relative motion near a chief satellite. SI units. Curtis ch. 7, Vallado section 6.8.

LVLH frame of the chief (Curtis section 7.2): x radial (outward), y along-track (completes the triad),
z along the orbit normal. Clohessy-Wiltshire equations for a circular chief orbit of mean motion n:
x'' - 3n²x - 2n y' = 0,  y'' + 2n x' = 0,  z'' + n²z = 0 (Curtis eqs. 7.36).
"""

import math

import numpy as np


def lvlh_basis(r, v) -> np.ndarray:
    """Rows are the LVLH unit vectors (x̂ radial, ŷ along-track, ẑ normal) of the state (r, v)."""
    r, v = np.asarray(r, dtype=float), np.asarray(v, dtype=float)
    i = r / np.linalg.norm(r)
    h = np.cross(r, v)
    k = h / np.linalg.norm(h)
    return np.array([i, np.cross(k, i), k])


def to_lvlh(r_chief, v_chief, r, v) -> tuple[np.ndarray, np.ndarray]:
    """Position and velocity of (r, v) relative to the chief, in the chief's rotating LVLH frame.

    δv_rel = Q (v - v_c - Ω × (r - r_c)), Ω = h/r_c² along ĥ (Curtis eqs. 7.11-7.14).
    """
    r_chief, v_chief = np.asarray(r_chief, dtype=float), np.asarray(v_chief, dtype=float)
    Q = lvlh_basis(r_chief, v_chief)
    h = np.cross(r_chief, v_chief)
    omega = h / (r_chief @ r_chief)
    dr = np.asarray(r, dtype=float) - r_chief
    dv = np.asarray(v, dtype=float) - v_chief - np.cross(omega, dr)
    return Q @ dr, Q @ dv


def from_lvlh(r_chief, v_chief, rho, rho_dot) -> tuple[np.ndarray, np.ndarray]:
    """Inverse of to_lvlh: inertial position and velocity from an LVLH relative state."""
    r_chief, v_chief = np.asarray(r_chief, dtype=float), np.asarray(v_chief, dtype=float)
    Q = lvlh_basis(r_chief, v_chief)
    omega = np.cross(r_chief, v_chief) / (r_chief @ r_chief)
    dr = Q.T @ np.asarray(rho, dtype=float)
    dv = Q.T @ np.asarray(rho_dot, dtype=float) + np.cross(omega, dr)
    return r_chief + dr, v_chief + dv


def cw_matrices(n: float, t: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Clohessy-Wiltshire state transition blocks Φrr, Φrv, Φvr, Φvv (Curtis eqs. 7.53)."""
    s, c = math.sin(n * t), math.cos(n * t)
    rr = np.array([[4 - 3 * c, 0, 0], [6 * (s - n * t), 1, 0], [0, 0, c]])
    rv = np.array([[s / n, 2 * (1 - c) / n, 0], [2 * (c - 1) / n, (4 * s - 3 * n * t) / n, 0], [0, 0, s / n]])
    vr = np.array([[3 * n * s, 0, 0], [6 * n * (c - 1), 0, 0], [0, 0, -n * s]])
    vv = np.array([[c, 2 * s, 0], [-2 * s, 4 * c - 3, 0], [0, 0, c]])
    return rr, rv, vr, vv


def cw_propagate(rho0, rho_dot0, n: float, t) -> tuple[np.ndarray, np.ndarray]:
    """Relative position and velocity after t (scalar or array) by the CW equations."""
    rho0, rho_dot0 = np.asarray(rho0, dtype=float), np.asarray(rho_dot0, dtype=float)
    ts = np.atleast_1d(np.asarray(t, dtype=float))
    rho, rho_dot = np.empty((len(ts), 3)), np.empty((len(ts), 3))
    for k, tk in enumerate(ts):
        rr, rv, vr, vv = cw_matrices(n, tk)
        rho[k] = rr @ rho0 + rv @ rho_dot0
        rho_dot[k] = vr @ rho0 + vv @ rho_dot0
    if np.ndim(t) == 0:
        return rho[0], rho_dot[0]
    return rho, rho_dot


def cw_rendezvous(rho0, rho_dot0, tof: float, n: float, rho_f=(0.0, 0.0, 0.0)) -> tuple[np.ndarray, np.ndarray]:
    """Two-impulse CW rendezvous: Δv at the start and at arrival to reach rho_f at rest after tof.

    v0⁺ = Φrv⁻¹ (ρ_f - Φrr ρ0), v_f⁻ = Φvr ρ0 + Φvv v0⁺ (Curtis section 7.5, eqs. 7.55-7.58).

    Returns:
        (dv1, dv2) in LVLH [m/s]
    """
    rr, rv, vr, vv = cw_matrices(n, tof)
    rho0 = np.asarray(rho0, dtype=float)
    v0_plus = np.linalg.solve(rv, np.asarray(rho_f, dtype=float) - rr @ rho0)
    vf_minus = vr @ rho0 + vv @ v0_plus
    return v0_plus - np.asarray(rho_dot0, dtype=float), -vf_minus

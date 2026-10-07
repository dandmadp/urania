"""State vector (r, v) ↔ classical orbital elements.

All inputs and outputs are SI (m, m/s, m³/s², rad).

Orbit size is stored as the semi-latus rectum p instead of the semi-major axis a,
because p is finite for every conic, including the parabola (a = ∞).

Angle conventions for singular orbits (Vallado):
- Equatorial (i ≈ 0 or π): the node is undefined, so raan = 0 and the reference axis is x
- Circular (e ≈ 0): periapsis is undefined, so argp = 0 and ν is measured from the node
  (from the x axis if also equatorial)
With these conventions coe_to_rv(rv_to_coe(r, v)) returns the original state even for singular orbits.
"""

import math
from typing import NamedTuple

import numpy as np

from .kepler import PARABOLIC_TOL, _wrap_2pi

# Dimensionless tolerances for treating an orbit as circular / equatorial
CIRCULAR_TOL = 1e-11
EQUATORIAL_TOL = 1e-11


class Elements(NamedTuple):
    """Classical orbital elements (SI, rad)."""

    p: float      # semi-latus rectum [m]
    ecc: float    # eccentricity [-]
    inc: float    # inclination [rad], 0 to π
    raan: float   # right ascension of the ascending node [rad], 0 to 2π
    argp: float   # argument of periapsis [rad], 0 to 2π
    nu: float     # true anomaly [rad]

    @property
    def a(self) -> float:
        """Semi-major axis [m]. inf for a parabola, negative for a hyperbola."""
        if abs(self.ecc - 1.0) < PARABOLIC_TOL:
            return math.inf
        return self.p / (1.0 - self.ecc**2)


def _angle_between(ref: np.ndarray, vec: np.ndarray, normal: np.ndarray) -> float:
    """Counterclockwise angle from ref to vec about normal, in [0, 2π)."""
    y = np.dot(np.cross(ref, vec), normal)
    x = np.dot(ref, vec)
    return _wrap_2pi(math.atan2(y, x))


def rv_to_coe(r, v, mu: float) -> Elements:
    """Position and velocity vectors → classical orbital elements.

    Args:
        r: position vector [m], length 3
        v: velocity vector [m/s], length 3
        mu: gravitational parameter of the central body [m³/s²]
    """
    r = np.asarray(r, dtype=float)
    v = np.asarray(v, dtype=float)
    r_norm = np.linalg.norm(r)
    v_norm = np.linalg.norm(v)
    if r_norm == 0.0:
        raise ValueError("The position vector is zero")

    h_vec = np.cross(r, v)
    h_norm = np.linalg.norm(h_vec)
    if h_norm == 0.0:
        raise ValueError("Angular momentum is zero (rectilinear orbits are not supported)")
    h_hat = h_vec / h_norm

    n_vec = np.cross([0.0, 0.0, 1.0], h_vec)  # ascending node direction
    n_norm = np.linalg.norm(n_vec)

    e_vec = ((v_norm**2 - mu / r_norm) * r - np.dot(r, v) * v) / mu
    ecc = float(np.linalg.norm(e_vec))

    p = h_norm**2 / mu
    inc = math.acos(np.clip(h_vec[2] / h_norm, -1.0, 1.0))

    equatorial = n_norm < EQUATORIAL_TOL * h_norm
    circular = ecc < CIRCULAR_TOL

    n_hat = np.array([1.0, 0.0, 0.0]) if equatorial else n_vec / n_norm
    raan = 0.0 if equatorial else _wrap_2pi(math.atan2(n_hat[1], n_hat[0]))

    if circular:
        argp = 0.0
        nu = _angle_between(n_hat, r, h_hat)
    else:
        argp = _angle_between(n_hat, e_vec, h_hat)
        nu = _angle_between(e_vec, r, h_hat)

    return Elements(p=p, ecc=ecc, inc=inc, raan=raan, argp=argp, nu=nu)


def _rot_perifocal_to_inertial(inc: float, raan: float, argp: float) -> np.ndarray:
    """Rotation matrix from the perifocal frame (PQW) to inertial: R3(Ω)·R1(i)·R3(ω)."""
    cO, sO = math.cos(raan), math.sin(raan)
    ci, si = math.cos(inc), math.sin(inc)
    cw, sw = math.cos(argp), math.sin(argp)
    return np.array([
        [cO * cw - sO * sw * ci, -cO * sw - sO * cw * ci,  sO * si],
        [sO * cw + cO * sw * ci, -sO * sw + cO * cw * ci, -cO * si],
        [sw * si,                 cw * si,                 ci],
    ])


def coe_to_rv(p: float, ecc: float, inc: float, raan: float, argp: float,
              nu: float, mu: float) -> tuple[np.ndarray, np.ndarray]:
    """Classical orbital elements → position and velocity vectors (r [m], v [m/s])."""
    if p <= 0.0:
        raise ValueError(f"Semi-latus rectum must be positive: p={p}")
    denom = 1.0 + ecc * math.cos(nu)
    if denom <= 0.0:
        raise ValueError(f"ν={nu} rad is not reachable on this orbit (e={ecc})")

    cnu, snu = math.cos(nu), math.sin(nu)
    r_pqw = (p / denom) * np.array([cnu, snu, 0.0])
    v_pqw = math.sqrt(mu / p) * np.array([-snu, ecc + cnu, 0.0])

    Q = _rot_perifocal_to_inertial(inc, raan, argp)
    return Q @ r_pqw, Q @ v_pqw


def elements_to_rv(el: Elements, mu: float) -> tuple[np.ndarray, np.ndarray]:
    """Elements tuple → position and velocity vectors."""
    return coe_to_rv(*el, mu=mu)

"""Kepler's equation and anomaly conversions.

All angles are in rad. The eccentricity e selects the conic type:
ellipse 0 <= e < 1, parabola e == 1, hyperbola e > 1.

For parabolas, the "mean anomaly" is the M in Barker's equation M = D + D³/3 (D = tan(ν/2)).
"""

import math

TWO_PI = 2.0 * math.pi

# Eccentricity tolerance for treating an orbit as parabolic
PARABOLIC_TOL = 1e-12

_NEWTON_TOL = 1e-14
_NEWTON_MAXITER = 50


def _wrap_2pi(angle: float) -> float:
    """Normalize an angle to [0, 2π)."""
    wrapped = angle % TWO_PI
    # A tiny negative angle can round to exactly 2π
    return 0.0 if wrapped == TWO_PI else wrapped


def _wrap_pi(angle: float) -> float:
    """Normalize an angle to [-π, π)."""
    return (angle + math.pi) % TWO_PI - math.pi


def _check_ecc(e: float) -> None:
    if e < 0.0:
        raise ValueError(f"Eccentricity must be non-negative: e={e}")


# ---------------------------------------------------------------- ellipse (e < 1)

def eccentric_to_mean(E: float, e: float) -> float:
    """Eccentric anomaly E → mean anomaly M (Kepler's equation M = E - e sin E)."""
    return E - e * math.sin(E)


def mean_to_eccentric(M: float, e: float) -> float:
    """Mean anomaly M → eccentric anomaly E, solving Kepler's equation by Newton's method.

    The result is in [0, 2π).
    """
    _check_ecc(e)
    if e >= 1.0:
        raise ValueError(f"Only elliptic orbits (e < 1) are allowed: e={e}")
    M = _wrap_pi(M)
    # For large e a start near M can diverge, so start from π (Vallado algorithm 2)
    E = M + e if M >= 0.0 else M - e
    if e > 0.8:
        E = math.pi if M >= 0.0 else -math.pi
    for _ in range(_NEWTON_MAXITER):
        dE = (E - e * math.sin(E) - M) / (1.0 - e * math.cos(E))
        E -= dE
        if abs(dE) < _NEWTON_TOL:
            return _wrap_2pi(E)
    raise RuntimeError(f"Kepler's equation did not converge: M={M}, e={e}")


def eccentric_to_true(E: float, e: float) -> float:
    """Eccentric anomaly E → true anomaly ν. The result is in [0, 2π)."""
    nu = 2.0 * math.atan2(math.sqrt(1.0 + e) * math.sin(E / 2.0),
                          math.sqrt(1.0 - e) * math.cos(E / 2.0))
    return _wrap_2pi(nu)


def true_to_eccentric(nu: float, e: float) -> float:
    """True anomaly ν → eccentric anomaly E. The result is in [0, 2π)."""
    E = 2.0 * math.atan2(math.sqrt(1.0 - e) * math.sin(nu / 2.0),
                         math.sqrt(1.0 + e) * math.cos(nu / 2.0))
    return _wrap_2pi(E)


# -------------------------------------------------------------- hyperbola (e > 1)

def hyperbolic_to_mean(H: float, e: float) -> float:
    """Hyperbolic anomaly H → mean anomaly M (M = e sinh H - H)."""
    return e * math.sinh(H) - H


def mean_to_hyperbolic(M: float, e: float) -> float:
    """Mean anomaly M → hyperbolic anomaly H, by Newton's method."""
    if e <= 1.0:
        raise ValueError(f"Only hyperbolic orbits (e > 1) are allowed: e={e}")
    # For large |M|, e sinh H ≈ M, so start from a logarithmic estimate
    H = math.copysign(math.log(2.0 * abs(M) / e + 1.8), M) if M != 0.0 else 0.0
    for _ in range(_NEWTON_MAXITER):
        dH = (e * math.sinh(H) - H - M) / (e * math.cosh(H) - 1.0)
        H -= dH
        if abs(dH) < _NEWTON_TOL * max(1.0, abs(H)):
            return H
    raise RuntimeError(f"Hyperbolic Kepler equation did not converge: M={M}, e={e}")


def hyperbolic_to_true(H: float, e: float) -> float:
    """Hyperbolic anomaly H → true anomaly ν. The result is in (-π, π)."""
    return 2.0 * math.atan(math.sqrt((e + 1.0) / (e - 1.0)) * math.tanh(H / 2.0))


def true_to_hyperbolic(nu: float, e: float) -> float:
    """True anomaly ν → hyperbolic anomaly H.

    ν must lie inside the asymptotes ±arccos(-1/e).
    """
    nu = _wrap_pi(nu)
    nu_inf = math.acos(-1.0 / e)
    if abs(nu) >= nu_inf:
        raise ValueError(f"ν={nu} rad is outside the hyperbola's asymptotes (±{nu_inf} rad)")
    return 2.0 * math.atanh(math.sqrt((e - 1.0) / (e + 1.0)) * math.tan(nu / 2.0))


# ------------------------------------------------------------- parabola (e == 1)

def parabolic_to_mean(D: float) -> float:
    """Parabolic anomaly D = tan(ν/2) → Barker mean anomaly M = D + D³/3."""
    return D + D**3 / 3.0


def mean_to_parabolic(M: float) -> float:
    """Solve Barker's equation D + D³/3 = M in closed form."""
    # Unique real root of D³ + 3D - 3M = 0 (Cardano): D = ∛(w+s) + ∛(w-s).
    # Since (w+s)(w-s) = -1, ∛(w-s) = -1/∛(w+s). Computing w-s directly cancels badly for large |w|.
    w = 1.5 * abs(M)
    c = math.cbrt(w + math.sqrt(w * w + 1.0))
    return math.copysign(c - 1.0 / c, M)


# ------------------------------------------------------------- any conic

def mean_to_true(M: float, e: float) -> float:
    """Mean anomaly M → true anomaly ν for an ellipse, parabola or hyperbola.

    Ellipses return [0, 2π); parabolas and hyperbolas return (-π, π).
    """
    _check_ecc(e)
    if abs(e - 1.0) < PARABOLIC_TOL:
        return 2.0 * math.atan(mean_to_parabolic(M))
    if e < 1.0:
        return eccentric_to_true(mean_to_eccentric(M, e), e)
    return hyperbolic_to_true(mean_to_hyperbolic(M, e), e)


def true_to_mean(nu: float, e: float) -> float:
    """True anomaly ν → mean anomaly M.

    Ellipses return [0, 2π); parabolas and hyperbolas return a signed value.
    """
    _check_ecc(e)
    if abs(e - 1.0) < PARABOLIC_TOL:
        return parabolic_to_mean(math.tan(_wrap_pi(nu) / 2.0))
    if e < 1.0:
        return _wrap_2pi(eccentric_to_mean(true_to_eccentric(nu, e), e))
    return hyperbolic_to_mean(true_to_hyperbolic(nu, e), e)

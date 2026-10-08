"""Orbit propagation: analytic two-body propagation (universal variables) and Cowell numerical integration."""

from typing import Callable, NamedTuple, Sequence

import numpy as np
from scipy.integrate import solve_ivp

from . import kepler as _kepler


def stumpff(psi: float) -> tuple[float, float]:
    """Stumpff functions c2(ψ), c3(ψ) (Vallado algorithm 1), free of cancellation near ψ = 0.

    c2 = (1 - cos√ψ)/ψ, c3 = (√ψ - sin√ψ)/√ψ³ for ψ > 0, with the hyperbolic forms for ψ < 0.
    """
    if psi > 0.0:
        x = np.sqrt(psi)
        return 2.0 * np.sin(0.5 * x) ** 2 / psi, _kepler._x_minus_sin(x) / (x * psi)
    if psi < 0.0:
        y = np.sqrt(-psi)
        return 2.0 * np.sinh(0.5 * y) ** 2 / -psi, _kepler._sinh_minus_x(y) / (-y * psi)
    return 0.5, 1.0 / 6.0


def _universal_guess(r0n: float, sigma0: float, alpha: float, h: float, dt: float, mu: float) -> float:
    """Starting value of the universal variable χ (Vallado algorithm 8)."""
    sqrt_mu = np.sqrt(mu)
    if alpha * r0n > 1e-6:                                   # ellipse
        return sqrt_mu * dt * alpha
    if alpha * r0n < -1e-6:                                  # hyperbola
        a = 1.0 / alpha
        arg = (-2.0 * mu * alpha * dt) / (sigma0 * sqrt_mu + np.sign(dt) * np.sqrt(-mu * a) * (1.0 - r0n * alpha))
        if arg > 0.0:
            return np.sign(dt) * np.sqrt(-a) * np.log(arg)
    p = h * h / mu                                           # parabola (Barker's equation)
    s = 0.5 * np.arctan2(1.0, 3.0 * np.sqrt(mu / p**3) * dt)
    w = np.arctan(np.cbrt(np.tan(s)))
    return np.sqrt(p) * 2.0 / np.tan(2.0 * w)


def kepler_states(r0, v0, dts: Sequence[float], mu: float) -> tuple[np.ndarray, np.ndarray]:
    """Two-body analytic states at several times dts [s], by universal variables.

    Universal variables (Curtis algorithm 3.4, Vallado algorithm 8) treat ellipses, parabolas and
    hyperbolas with one formula and stay accurate for nearly parabolic orbits, where element-based
    Kepler solutions lose digits in 1 - e. The universal Kepler equation is solved by the
    Laguerre-Conway iteration, which converges from any start.

    Returns:
        r [m] (N×3), v [m/s] (N×3)
    """
    r0 = np.asarray(r0, dtype=float)
    v0 = np.asarray(v0, dtype=float)
    r0n = np.sqrt(r0 @ r0)
    if r0n == 0.0:
        raise ValueError("The position vector is zero")
    sqrt_mu = np.sqrt(mu)
    sigma0 = (r0 @ v0) / sqrt_mu
    alpha = 2.0 / r0n - (v0 @ v0) / mu                       # 1/a
    h = np.linalg.norm(np.cross(r0, v0))
    if h == 0.0:
        raise ValueError("Angular momentum is zero (rectilinear orbits are not supported)")
    period = 2.0 * np.pi / np.sqrt(mu * alpha**3) if alpha * r0n > 1e-6 else None

    r_out = np.empty((len(dts), 3))
    v_out = np.empty((len(dts), 3))
    for i, dt in enumerate(dts):
        dt = float(dt)
        if period is not None:
            dt -= round(dt / period) * period                # two-body motion is periodic
        if dt == 0.0:
            r_out[i], v_out[i] = r0, v0
            continue
        chi = _universal_guess(r0n, sigma0, alpha, h, dt, mu)
        for _ in range(100):
            psi = chi * chi * alpha
            c2, c3 = stumpff(psi)
            F = chi**3 * c3 + sigma0 * chi**2 * c2 + r0n * chi * (1.0 - psi * c3) - sqrt_mu * dt
            dF = chi**2 * c2 + sigma0 * chi * (1.0 - psi * c3) + r0n * (1.0 - psi * c2)   # = r
            ddF = sigma0 * (1.0 - psi * c2) + (1.0 - alpha * r0n) * chi * (1.0 - psi * c3)
            # Laguerre-Conway step with n = 5
            disc = abs(16.0 * dF * dF - 20.0 * F * ddF)
            denom = dF + np.copysign(np.sqrt(disc), dF)
            step = 5.0 * F / denom
            chi -= step
            if abs(step) <= 1e-14 * max(1.0, abs(chi)):
                break
        else:
            raise RuntimeError(f"Universal Kepler equation did not converge: dt={dt}")
        psi = chi * chi * alpha
        c2, c3 = stumpff(psi)
        r_norm = chi**2 * c2 + sigma0 * chi * (1.0 - psi * c3) + r0n * (1.0 - psi * c2)
        f = 1.0 - chi**2 / r0n * c2
        g = dt - chi**3 / sqrt_mu * c3
        fdot = sqrt_mu / (r_norm * r0n) * chi * (psi * c3 - 1.0)
        gdot = 1.0 - chi**2 / r_norm * c2
        r_out[i] = f * r0 + g * v0
        v_out[i] = fdot * r0 + gdot * v0
    return r_out, v_out


def kepler_propagate(r0, v0, dt: float, mu: float) -> tuple[np.ndarray, np.ndarray]:
    """Two-body analytic state after dt [s]."""
    r, v = kepler_states(r0, v0, [dt], mu)
    return r[0], v[0]


class CowellResult(NamedTuple):
    t: np.ndarray          # output times [s], length N
    r: np.ndarray          # positions [m], N×3
    v: np.ndarray          # velocities [m/s], N×3
    terminated: bool       # stopped early by an event
    nfev: int              # number of acceleration evaluations


def cowell(r0, v0, duration: float,
           accel: Callable[[float, np.ndarray, np.ndarray], np.ndarray], *,
           t_eval: Sequence[float] | None = None,
           events: Sequence[Callable] = (),
           method: str = "DOP853", rtol: float = 1e-12, atol: float = 1e-8) -> CowellResult:
    """Numerically integrate r'' = accel(t, r, v) (Cowell's method).

    Args:
        duration: integration time [s]. Negative propagates backward.
        accel: acceleration function (t [s since start], r, v) → a [m/s²]
        events: solve_ivp event functions (t, y) → float. A terminal event stops integration
            and the state at the event is appended to the result.
    """
    y0 = np.concatenate((np.asarray(r0, dtype=float), np.asarray(v0, dtype=float)))
    if duration == 0.0:
        n = len(t_eval) if t_eval is not None else 1
        return CowellResult(t=np.zeros(n), r=np.tile(y0[:3], (n, 1)),
                            v=np.tile(y0[3:], (n, 1)), terminated=False, nfev=0)

    def rhs(t, y):
        return np.concatenate((y[3:], accel(t, y[:3], y[3:])))

    sol = solve_ivp(rhs, (0.0, duration), y0, method=method, t_eval=t_eval,
                    events=list(events) or None, rtol=rtol, atol=atol)
    if sol.status == -1:
        raise RuntimeError(f"Integration failed: {sol.message}")

    t, y = sol.t, sol.y
    terminated = sol.status == 1
    if terminated:
        for t_ev, y_ev in zip(sol.t_events, sol.y_events, strict=True):
            if len(t_ev):
                t = np.append(t, t_ev[0])
                y = np.column_stack((y, y_ev[0]))
                break
    return CowellResult(t=t, r=y[:3].T.copy(), v=y[3:].T.copy(),
                        terminated=terminated, nfev=sol.nfev)

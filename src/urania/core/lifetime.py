"""Orbit-averaged decay of a near-circular orbit by atmospheric drag. SI units."""

import math
from typing import Callable, NamedTuple

import numpy as np
from scipy.integrate import solve_ivp


def _densities(density, r: np.ndarray, t: float) -> np.ndarray:
    """ρ at every row of r; one call when the model offers many(r, t) for N×3 positions."""
    many = getattr(density, "many", None)
    if many is not None:
        return np.asarray(many(r, t), dtype=float)
    return np.array([density(ri, t) for ri in r])


def mean_decay_rate(a: float, inc: float, mu: float, rotation_rate: float, ballistic: float,
                    density: Callable[[np.ndarray, float], float], t: float = 0.0,
                    samples: int = 36, raan: float = 0.0) -> float:
    """Orbit-averaged da/dt [m/s] of a circular orbit with semi-major axis a and inclination inc.

    Along the orbit (argument of latitude u), the tangential drag acceleration is
    a_T = -½ ρ B |v_rel| (v_rel · t̂) with v_rel = v - ω × r, and Gauss's equation for a circular orbit
    gives da/dt = 2 a_T / n. The average over u captures the density's latitude dependence
    (ellipsoid height) and the co-rotating atmosphere. Without rotation it reduces to -ρ B √(μ a).

    Args:
        ballistic: C_D·A/m [m²/kg]
        density: ρ(r, t) [kg/m³] at an inertial position r [m]
        raan: right ascension of the ascending node [rad]; matters for models with a local-time
            dependence (the day-side bulge of NRLMSISE-00)
    """
    v = math.sqrt(mu / a)
    n = v / a
    ci, si = math.cos(inc), math.sin(inc)
    co, so = math.cos(raan), math.sin(raan)
    u = 2.0 * math.pi * (np.arange(samples) + 0.5) / samples
    cu, su = np.cos(u), np.sin(u)
    # position and along-track unit vector in the orbit plane, rotated by the node
    px, py, pz = cu, ci * su, si * su
    tx, ty, tz = -su, ci * cu, si * cu
    r = a * np.column_stack([co * px - so * py, so * px + co * py, pz])
    t_hat = np.column_stack([co * tx - so * ty, so * tx + co * ty, tz])
    v_rel = v * t_hat - rotation_rate * np.column_stack([-r[:, 1], r[:, 0], np.zeros(samples)])
    rho = _densities(density, r, t)
    a_t = -0.5 * rho * ballistic * np.linalg.norm(v_rel, axis=1) * np.sum(v_rel * t_hat, axis=1)
    return float(np.mean(2.0 * a_t / n))


def sampled_decay_rate(r, v, mu: float, rotation_rate: float, ballistic: float,
                       density: Callable[[np.ndarray, float], float], t: float = 0.0) -> float:
    """Average da/dt [m/s] over states r, v (N×3) sampled uniformly in time along one orbit.

    Uses the exact osculating relation da/dt = 2 a² (v · a_drag) / μ (from dε/dt = v · a_drag).
    """
    from .forces import accel_drag

    r, v = np.asarray(r, dtype=float), np.asarray(v, dtype=float)
    rho = _densities(density, r, t)
    total = 0.0
    for ri, vi, rho_i in zip(r, v, rho, strict=True):
        a = 1.0 / (2.0 / math.sqrt(ri @ ri) - (vi @ vi) / mu)
        a_d = accel_drag(ri, vi, rho_i, ballistic, rotation_rate)
        total += 2.0 * a * a * (vi @ a_d) / mu
    return total / len(r)


class DecayResult(NamedTuple):
    t: np.ndarray          # time since start [s]
    a: np.ndarray          # mean semi-major axis [m]
    reached: bool          # stop radius reached within the time limit


def decay_history(a0: float, inc: float, mu: float, rotation_rate: float, ballistic: float,
                  density: Callable[[np.ndarray, float], float], r_stop: float, t_max: float,
                  t0: float = 0.0, correction: Callable[[float], float] | None = None,
                  raan0: float = 0.0, raan_rate: Callable[[float], float] | None = None) -> DecayResult:
    """Integrate the orbit-averaged semi-major axis until it reaches r_stop or t_max [s].

    density is called with the absolute time t0 + t, so time-varying environments work.
    correction(a), if given, multiplies the circular-orbit rate (e.g. the ratio to the rate averaged
    along a J2-perturbed orbit, whose radius varies with latitude).
    The node starts at raan0 and drifts at raan_rate(a) (the J2 precession), so the orbit plane keeps its
    true orientation to the Sun for local-time-dependent atmospheres.
    """
    if a0 <= r_stop:
        return DecayResult(np.array([0.0]), np.array([a0]), True)

    def rhs(t, y):
        rate = mean_decay_rate(y[0], inc, mu, rotation_rate, ballistic, density, t0 + t, raan=y[1])
        drift = raan_rate(y[0]) if raan_rate is not None else 0.0
        return [rate * correction(y[0]) if correction is not None else rate, drift]

    def stop(t, y):
        return y[0] - r_stop

    stop.terminal = True
    stop.direction = -1
    sol = solve_ivp(rhs, (0.0, t_max), [a0, raan0], method="LSODA", events=[stop],
                    rtol=1e-8, atol=[1e-3, 1e-6], dense_output=False, max_step=min(t_max / 50, 5 * 86400.0))
    if sol.status == -1:
        raise RuntimeError(f"Decay integration failed: {sol.message}")
    t, a = sol.t, sol.y[0]
    reached = sol.status == 1
    if reached:
        t = np.append(t, sol.t_events[0][0])
        a = np.append(a, sol.y_events[0][0][0])
    return DecayResult(t, a, reached)

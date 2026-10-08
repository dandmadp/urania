"""Orbit propagation: fidelity model selection, results (Trajectory) and metadata.

Fidelity levels (SPEC 2.7):
    0 "twobody"   two-body
    1 "j2"        + J2
    2 "j2+drag"   + atmospheric drag
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING

import astropy.units as u
import numpy as np
from scipy.optimize import brentq

from . import units
from .core import elements as _el
from .core import forces as _forces
from .core import propagate as _prop
from .environment import Environment, as_environment
from .time import SECONDS_PER_DAY

if TYPE_CHECKING:
    from .bodies import Body
    from .orbits import Orbit
    from .time import Epoch

FIDELITY_MODELS = {0: "twobody", 1: "j2", 2: "j2+drag"}
_SUPPORTED_TERMS = {"twobody", "j2", "drag"}
_FUTURE_TERMS = {"moon", "sun", "srp", "harmonics"}

DEFAULT_RTOL = 1e-12
DEFAULT_ATOL = 1e-8


def parse_model(model) -> frozenset[str]:
    """Turn a model spec ("j2+drag" or a fidelity integer) into a set of terms. Two-body is always included."""
    if isinstance(model, bool) or not isinstance(model, (int, str)):
        raise TypeError(f"model must be a string such as 'j2+drag' or a fidelity level 0 to 2: {model!r}")
    if isinstance(model, int):
        if model not in FIDELITY_MODELS:
            raise NotImplementedError(f"Fidelity level {model} is not supported yet (0 to 2)")
        model = FIDELITY_MODELS[model]
    terms = {t.strip().lower() for t in model.split("+")}
    future = terms & _FUTURE_TERMS
    if future:
        raise NotImplementedError(f"{sorted(future)} perturbations will be supported after the MVP")
    unknown = terms - _SUPPORTED_TERMS
    if unknown:
        raise ValueError(f"Unknown model terms: {sorted(unknown)} (available: {sorted(_SUPPORTED_TERMS)})")
    return frozenset(terms | {"twobody"})


def model_name(terms: frozenset[str]) -> str:
    return "+".join(t for t in ("twobody", "j2", "drag") if t in terms)


@dataclass(frozen=True)
class PropagationInfo:
    """Propagation metadata: which model, integrator and assumptions produced the result."""

    model: str
    integrator: str
    rtol: float | None
    atol: float | None
    assumptions: tuple[str, ...]
    terminated: str | None = None   # reason for stopping early (e.g. reentry)
    nfev: int | None = None


def _readonly(a: np.ndarray) -> np.ndarray:
    a = np.array(a, dtype=float)
    a.flags.writeable = False
    return a


@dataclass(frozen=True, eq=False)
class Trajectory:
    """Propagation result: state vectors versus time since the start epoch, plus metadata."""

    body: Body
    epoch: Epoch            # start epoch
    t: np.ndarray           # time since start [s], length N
    r: np.ndarray           # positions [m], N×3
    v: np.ndarray           # velocities [m/s], N×3
    info: PropagationInfo

    def __post_init__(self):
        for name in ("t", "r", "v"):
            object.__setattr__(self, name, _readonly(getattr(self, name)))

    def __len__(self) -> int:
        return len(self.t)

    def orbit_at(self, i: int) -> Orbit:
        """Osculating orbit at sample i."""
        from .orbits import Orbit

        return Orbit(self.body, self.r[i], self.v[i], self.epoch + float(self.t[i]))

    @property
    def final(self) -> Orbit:
        """Orbit at the last sample."""
        return self.orbit_at(-1)

    @property
    def altitude(self) -> np.ndarray:
        """Altitude above a spherical body [m]."""
        return np.linalg.norm(self.r, axis=1) - self.body.radius

    @property
    def epochs(self) -> list[Epoch]:
        """Epoch of every sample."""
        return [self.epoch + float(t) for t in self.t]

    @cached_property
    def elements(self) -> dict[str, np.ndarray]:
        """Osculating classical elements at every sample, as read-only arrays (SI, rad).

        Keys: p, a, ecc, inc, raan, argp, nu. Angles are not unwrapped.
        """
        els = [_el.rv_to_coe(r, v, self.body.mu) for r, v in zip(self.r, self.v, strict=True)]
        out = {name: np.array([getattr(e, name) for e in els])
               for name in ("p", "a", "ecc", "inc", "raan", "argp", "nu")}
        for arr in out.values():
            arr.flags.writeable = False
        return out

    def explain(self):
        """Show the equations of motion, integrator, changes in the result and interpretation caveats."""
        from .explain import explain_trajectory

        return explain_trajectory(self)

    def plot(self, ax=None, kind: str = "orbit"):
        """kind="orbit": path projected on the initial orbit plane; "altitude": altitude versus time."""
        from .viz import plot_trajectory

        return plot_trajectory(self, ax, kind)


def resolve_duration(duration, days) -> float:
    """Convert duration (seconds or time Quantity) or days (number or Quantity) to seconds."""
    if (duration is None) == (days is None):
        raise ValueError("Specify exactly one of duration and days")
    if days is not None:
        seconds = units.to_si(days, u.day) * SECONDS_PER_DAY
    else:
        seconds = units.to_si(duration, units.TIME)
    if not isinstance(seconds, float):
        raise TypeError("The propagation time must be a single value, not an array")
    if not math.isfinite(seconds):
        raise ValueError(f"The propagation time must be finite: {seconds} s")
    return seconds


def _default_n_points(orbit: Orbit, duration: float) -> int:
    """100 points per orbit, at least 101 and at most 200001."""
    if orbit.ecc >= 1.0:
        return 1001
    n = math.ceil(abs(duration) / orbit.period * 100) + 1
    return min(max(n, 101), 200001)


_SURFACE = "reached the body surface"


def _truncate_at_surface(orbit: Orbit, t, r, v):
    """Cut analytic samples at the first descent below the surface.

    Makes the analytic solution behave like the surface event of numerical integration.
    A grazing pass shorter than the sample spacing (100 points per orbit) can be missed.
    """
    R = orbit.body.radius
    alt = np.linalg.norm(r, axis=1) - R
    below = np.nonzero((alt[1:] < 0.0) & (alt[:-1] >= 0.0))[0]
    if len(below) == 0:
        return t, r, v, False
    i = below[0] + 1

    def altitude_at(dt):
        ri, _ = _prop.kepler_propagate(orbit.r, orbit.v, dt, orbit.body.mu)
        return np.linalg.norm(ri) - R

    t_hit = brentq(altitude_at, t[i - 1], t[i], xtol=1e-6)
    r_hit, v_hit = _prop.kepler_propagate(orbit.r, orbit.v, t_hit, orbit.body.mu)
    return (np.append(t[:i], t_hit), np.vstack((r[:i], r_hit)),
            np.vstack((v[:i], v_hit)), True)


def propagate(orbit: Orbit, duration, *, model="twobody", cd: float = 2.2,
              area=None, mass=None, density=None, method: str = "auto",
              rtol: float = DEFAULT_RTOL, atol: float = DEFAULT_ATOL,
              n_points: int | None = None) -> Trajectory:
    """Propagate an orbit for duration. Implementation of `Orbit.propagate`.

    With method="auto", two-body uses the analytic (Kepler) solution; anything else uses DOP853.
    """
    body = orbit.body
    duration = resolve_duration(duration, None)
    terms = parse_model(model)
    use_drag = "drag" in terms
    if not use_drag:
        ignored = [name for name, value in (("area", area), ("mass", mass), ("density", density))
                   if value is not None]
        if ignored:
            warnings.warn(f"{', '.join(ignored)} given but the model {model_name(terms)!r} has no "
                          f"drag term; use model='j2+drag' (or 'drag') to include drag",
                          UserWarning, stacklevel=3)
    if n_points is not None and n_points < 2:
        raise ValueError(f"n_points must be at least 2 (start and end): {n_points}")
    if np.linalg.norm(orbit.r) < body.radius:
        raise ValueError(f"The orbit starts below the surface of {body.name} "
                         f"(|r| = {np.linalg.norm(orbit.r):.0f} m < R = {body.radius:.0f} m)")
    n = n_points or _default_n_points(orbit, duration)
    t_eval = np.linspace(0.0, duration, n)

    assumptions = ["point-mass gravity of the central body", "fixed inertial axes (precession and nutation ignored)"]

    # two-body analytic solution
    if terms == {"twobody"} and method in ("auto", "kepler"):
        r, v = _prop.kepler_states(orbit.r, orbit.v, t_eval, body.mu)
        t, r, v, hit = _truncate_at_surface(orbit, t_eval, r, v)
        info = PropagationInfo(model="twobody", integrator="kepler (analytic)",
                               rtol=None, atol=None, assumptions=tuple(assumptions),
                               terminated=_SURFACE if hit else None)
        return Trajectory(body, orbit.epoch, t, r, v, info)

    if method == "kepler":
        raise ValueError("The analytic method (kepler) is only available for the two-body model")
    integrator = "DOP853" if method == "auto" else method

    mu, R, J2, omega = body.mu, body.radius, body.J2, body.rotation_rate
    use_j2 = "j2" in terms
    if use_j2:
        if J2 == 0.0:
            raise ValueError(f"J2 of {body.name} is zero")
        assumptions.append("J2 oblateness only (higher-order gravity ignored), spin axis = inertial z")

    if use_drag:
        if area is None or mass is None:
            raise ValueError("The drag model needs area and mass")
        area_si = units.to_si(area, units.AREA)
        mass_si = units.to_si(mass, units.MASS)
        for name, value in (("cd", cd), ("area", area_si), ("mass", mass_si)):
            if not value > 0.0:
                raise ValueError(f"{name} must be positive: {value}")
        ballistic = cd * area_si / mass_si
        if density is not None:
            rho: Environment = as_environment(density, units.DENSITY)
        elif body.atmosphere is not None:
            rho = body.atmosphere
        else:
            raise ValueError(f"{body.name} has no atmosphere model; pass density")
        assumptions += [
            f"atmospheric density: {rho.description}",
            "atmosphere co-rotates rigidly with the body",
            f"constant drag coefficient and area (Cd={cd:g}, A={area_si:g} m², m={mass_si:g} kg)",
        ]
        t0 = orbit.epoch.tdb_seconds

    def accel(t, r, v):
        a = _forces.accel_twobody(r, mu)
        if use_j2:
            a = a + _forces.accel_j2(r, mu, R, J2)
        if use_drag:
            a = a + _forces.accel_drag(r, v, rho(r, t0 + t), ballistic, omega)
        return a

    def hit_surface(t, y):
        return np.sqrt(y[:3] @ y[:3]) - R

    hit_surface.terminal = True
    hit_surface.direction = -1

    res = _prop.cowell(orbit.r, orbit.v, duration, accel, t_eval=t_eval,
                       events=[hit_surface], method=integrator, rtol=rtol, atol=atol)
    info = PropagationInfo(
        model=model_name(terms), integrator=f"scipy solve_ivp {integrator}",
        rtol=rtol, atol=atol, assumptions=tuple(assumptions),
        terminated=_SURFACE if res.terminated else None, nfev=res.nfev,
    )
    return Trajectory(body, orbit.epoch, res.t, res.r, res.v, info)

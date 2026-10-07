"""Orbit transfers: the Transfer result object and burn planning.

Computes Hohmann and bi-elliptic transfers and plane changes between circular orbits.
Each burn adds a Δv vector to the actual state vector; burns are linked by the two-body analytic solution.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from . import units
from .core import maneuvers as _man
from .core import propagate as _prop
from .core.elements import _angle_between

if TYPE_CHECKING:
    from .orbits import Orbit
    from .time import Epoch

# Largest eccentricity treated as circular
CIRCULAR_ECC_TOL = 1e-3


def _readonly(x) -> np.ndarray:
    a = np.array(x, dtype=float)
    a.flags.writeable = False
    return a


@dataclass(frozen=True, eq=False)
class Burn:
    """A single impulsive burn."""

    epoch: Epoch
    r: np.ndarray        # burn position [m]
    dv: np.ndarray       # Δv vector [m/s], inertial frame
    description: str

    def __post_init__(self):
        object.__setattr__(self, "r", _readonly(self.r))
        object.__setattr__(self, "dv", _readonly(self.dv))

    @property
    def magnitude(self) -> float:
        """Δv magnitude [m/s]."""
        return float(np.linalg.norm(self.dv))


@dataclass(frozen=True, eq=False)
class Transfer:
    """Transfer result: list of burns, orbits right after each burn, assumptions."""

    kind: str                       # "hohmann", "bielliptic", "plane_change"
    initial: Orbit
    target: Orbit
    burns: tuple[Burn, ...]
    orbits: tuple[Orbit, ...]       # orbit right after each burn (the last one is on the target orbit)
    plane_change: float             # total plane change angle [rad]
    assumptions: tuple[str, ...]

    @property
    def total_dv(self) -> float:
        """Total Δv [m/s]."""
        return sum(b.magnitude for b in self.burns)

    @property
    def coast(self) -> float:
        """Wait before the first burn [s] (coasting to the node)."""
        if not self.burns:
            return 0.0
        return self.burns[0].epoch - self.initial.epoch

    @property
    def tof(self) -> float:
        """Time from the first burn to the last [s]."""
        if not self.burns:
            return 0.0
        return self.burns[-1].epoch - self.burns[0].epoch

    @property
    def final(self) -> Orbit:
        """Orbit after the transfer."""
        return self.orbits[-1] if self.orbits else self.initial

    def explain(self):
        """Show the transfer formulas, intermediate speeds, Δv per burn and transfer time step by step."""
        from .explain import explain_transfer

        return explain_transfer(self)

    def plot(self, ax=None):
        """Plot the transfer in 2D, unfolding each orbit into its own plane."""
        from .viz import plot_transfer

        return plot_transfer(self, ax)

    def __repr__(self) -> str:
        return (f"Transfer({self.kind}, Δv={self.total_dv / 1e3:.4f} km/s, "
                f"tof={self.tof / 3600:.3f} h, {len(self.burns)} burn{'' if len(self.burns) == 1 else 's'})")


def _unit(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x)


def transfer(initial: Orbit, target: Orbit, method: str = "hohmann", *, rb=None,
             plane_split="optimal") -> Transfer:
    """Compute a transfer between circular orbits initial → target. Implementation of `Orbit.transfer_to`.

    Args:
        method: "hohmann" or "bielliptic"
        rb: intermediate apoapsis radius of a bi-elliptic transfer (length)
        plane_split: fraction of the plane change done by the first Hohmann burn (0 to 1) or "optimal".
            A bi-elliptic transfer does the whole plane change at the slow intermediate apoapsis.
    """
    from .orbits import Orbit

    if method not in ("hohmann", "bielliptic"):
        raise ValueError(f"Unknown transfer method: {method!r} (hohmann, bielliptic)")
    body = initial.body
    if target.body != body:
        raise ValueError(f"Central bodies differ: {body.name} → {target.body.name}")
    for name, o in (("initial", initial), ("target", target)):
        if o.ecc > CIRCULAR_ECC_TOL:
            raise ValueError(f"The {name} orbit is not circular (e={o.ecc:.4g}). "
                             f"The MVP supports transfers between circular orbits only")
    mu = body.mu
    r1, r2 = initial.a, target.a
    h1 = _unit(np.cross(initial.r, initial.v))
    h2 = _unit(np.cross(target.r, target.v))
    cross = np.cross(h1, h2)
    dtheta = math.atan2(np.linalg.norm(cross), h1 @ h2)
    coplanar = dtheta < 1e-9

    # First burn: at the current position if coplanar, otherwise at the next node
    if coplanar:
        coast = 0.0
        node_axis = None
    elif np.linalg.norm(cross) < 1e-12:
        # Opposite planes (Δθ ≈ 180°) share every line through the center; burn where we are
        node_axis = _unit(initial.r)
        coast = 0.0
    else:
        node_axis = _unit(cross)
        angles = [_angle_between(initial.r, s * node_axis, h1) for s in (1.0, -1.0)]
        coast = min(angles) / initial.mean_motion

    same_radius = abs(r1 - r2) < 1e-9 * r1
    assumptions = [
        "impulsive burns (Δv applied instantaneously)",
        "two-body motion between burns (perturbations ignored)",
        "initial and target orbits treated as circular",
        "phasing within the target orbit (rendezvous) is not matched",
    ]
    if not coplanar:
        assumptions.append("plane change performed at the line of nodes between the two planes")

    def tilt(fraction: float) -> np.ndarray:
        """Orbit normal h1 rotated about the node axis by fraction·Δθ toward h2."""
        if coplanar or fraction == 0.0:
            return h1
        if fraction == 1.0:
            return h2
        return _man.rotate(h1, node_axis, fraction * dtheta)

    # Burn plan: (next apoapsis radius or None = circularize, orbit normal after the burn, description)
    if same_radius:
        if coplanar:
            return Transfer("none", initial, target, (), (), 0.0, tuple(assumptions))
        kind = "plane_change"
        plan = [(None, h2, "plane change")]
    elif method == "hohmann":
        kind = "hohmann"
        if plane_split == "optimal":
            split = _man.optimal_plane_split(r1, r2, dtheta, mu)
            if not coplanar:
                assumptions.append(f"plane change split minimizes total Δv (first burn {split:.3f})")
        else:
            try:
                split = float(plane_split)
            except (TypeError, ValueError):
                raise ValueError(f"plane_split must be 'optimal' or a number from 0 to 1: "
                                 f"{plane_split!r}") from None
            if not 0.0 <= split <= 1.0:
                raise ValueError(f"plane_split must be between 0 and 1: {split}")
        plan = [(r2, tilt(split), "burn 1: enter transfer orbit"),
                (None, h2, "burn 2: circularize on target orbit")]
        plane_parts = [split * dtheta, (1.0 - split) * dtheta]
    elif method == "bielliptic":
        if rb is None:
            raise ValueError("A bi-elliptic transfer needs the intermediate apoapsis radius rb")
        rb = units.to_si(rb, units.LENGTH)
        if rb < max(r1, r2):
            raise ValueError(f"rb={rb} m must be larger than the initial and target radii")
        kind = "bielliptic"
        plan = [(rb, h1, "burn 1: enter first transfer ellipse"),
                (r2, h2, "burn 2: switch to second transfer ellipse at apoapsis"),
                (None, h2, "burn 3: circularize on target orbit")]
        plane_parts = [0.0, dtheta, 0.0]

    if kind == "plane_change":
        plane_parts = [dtheta]

    # Execute burns: add Δv to the state vector, then propagate analytically to the next burn
    epoch = initial.epoch + coast
    r, v = _prop.kepler_propagate(initial.r, initial.v, coast, mu)
    burns, orbits = [], []
    for (r_next, normal, desc), d_plane in zip(plan, plane_parts):
        r_now = np.linalg.norm(r)
        direction = _unit(np.cross(normal, r))
        if r_next is None:
            speed = math.sqrt(mu / r_now)
        else:
            speed = math.sqrt(mu * (2.0 / r_now - 2.0 / (r_now + r_next)))
        v_new = speed * direction
        if d_plane > 1e-12:
            desc += f" + plane {math.degrees(d_plane):.2f}°"
        burns.append(Burn(epoch, r, v_new - v, desc))
        orbit = Orbit(body, r, v_new, epoch)
        orbits.append(orbit)
        if r_next is not None:
            half = math.pi * math.sqrt((0.5 * (r_now + r_next)) ** 3 / mu)
            r, v = _prop.kepler_propagate(r, v_new, half, mu)
            epoch = epoch + half

    return Transfer(kind, initial, target, tuple(burns), tuple(orbits), dtheta,
                    tuple(assumptions))

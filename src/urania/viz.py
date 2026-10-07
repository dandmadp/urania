"""2D orbit plots (matplotlib).

Every function returns the matplotlib Axes and draws on ax if one is given.
Coordinates are shown in km.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from .core import elements as _el
from .core import propagate as _prop

if TYPE_CHECKING:
    from .maneuvers import Transfer
    from .orbits import Orbit
    from .propagation import Trajectory

KM = 1e3


def _axes(ax):
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))
    return ax


def _unit(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x)


def _orbit_points(orbit: Orbit, n: int = 361) -> np.ndarray:
    """Inertial points (n×3) [m] along one orbit (inside the asymptotes for a hyperbola)."""
    el = orbit.elements
    if el.ecc < 1.0:
        nus = np.linspace(0.0, 2.0 * math.pi, n)
    else:
        nu_max = 0.95 * math.acos(-1.0 / el.ecc) if el.ecc > 1.0 else 0.95 * math.pi
        nus = np.linspace(-nu_max, nu_max, n)
    return np.array([_el.coe_to_rv(el.p, el.ecc, el.inc, el.raan, el.argp, nu, orbit.body.mu)[0]
                     for nu in nus])


def _project(points: np.ndarray, x_hat: np.ndarray, y_hat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return points @ x_hat / KM, points @ y_hat / KM


def _draw_body(ax, body) -> None:
    import matplotlib.patches as mpatches

    ax.add_patch(mpatches.Circle((0, 0), body.radius / KM, color="#4a7ab8", alpha=0.35,
                                 zorder=0, label=body.name))


def _finish(ax, title: str) -> None:
    ax.set_aspect("equal")
    ax.set_xlabel("x [km]")
    ax.set_ylabel("y [km]")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize="small")


def plot_orbit(orbit: Orbit, ax=None):
    """Plot the orbit in its own plane. The x axis points to periapsis (the node for circular orbits)."""
    ax = _axes(ax)
    el = orbit.elements
    r_peri, _ = _el.coe_to_rv(el.p, el.ecc, el.inc, el.raan, el.argp, 0.0, orbit.body.mu)
    x_hat = _unit(r_peri)
    y_hat = np.cross(_unit(np.cross(orbit.r, orbit.v)), x_hat)

    _draw_body(ax, orbit.body)
    x, y = _project(_orbit_points(orbit), x_hat, y_hat)
    ax.plot(x, y, color="C0", label="orbit")
    px, py = _project(orbit.r[None, :], x_hat, y_hat)
    ax.plot(px, py, "o", color="C3", label=f"position ({orbit.epoch.iso[:19]})")
    if el.ecc > 1e-6:
        ax.plot(orbit.r_periapsis / KM, 0, "^", color="C2", label="periapsis")
        if el.ecc < 1.0:
            ax.plot(-orbit.r_apoapsis / KM, 0, "v", color="C1", label="apoapsis")
    _finish(ax, f"Orbit around {orbit.body.name}\na ={orbit.a / KM:,.0f} km, "
                f"e = {el.ecc:.3f}, i = {math.degrees(el.inc):.1f}°")
    return ax


def plot_trajectory(tr: Trajectory, ax=None, kind: str = "orbit"):
    """Plot a propagation result.

    kind="orbit": path projected on the initial orbit plane; the x axis points to the start position.
    kind="altitude": altitude versus time.
    """
    ax = _axes(ax)
    if kind == "altitude":
        days = tr.t / 86400.0
        alt = tr.altitude / KM
        ax.plot(days, alt, color="C0", lw=0.5, alpha=0.6, label="altitude")
        # One-orbit moving average: filters the once-per-orbit oscillation to show the decay trend
        first = tr.orbit_at(0)
        if first.ecc < 1.0 and len(tr) > 1:
            dt = abs(tr.t[1] - tr.t[0])
            window = int(round(first.period / dt)) if dt > 0 else 0
            if 2 <= window < len(tr) // 2:
                mean = np.convolve(alt, np.ones(window) / window, mode="valid")
                ax.plot(days[window // 2: window // 2 + len(mean)], mean, color="C3",
                        lw=1.5, label="orbit-averaged")
        ax.set_xlabel("time [day]")
        ax.set_ylabel("altitude [km]")
        ax.set_title(f"Altitude ({tr.info.model})")
        ax.grid(alpha=0.3)
        ax.legend(loc="upper right", fontsize="small")
        return ax
    if kind != "orbit":
        raise ValueError(f"kind must be 'orbit' or 'altitude': {kind!r}")

    x_hat = _unit(tr.r[0])
    y_hat = np.cross(_unit(np.cross(tr.r[0], tr.v[0])), x_hat)
    _draw_body(ax, tr.body)
    x, y = _project(tr.r, x_hat, y_hat)
    ax.plot(x, y, color="C0", lw=0.6, label=f"trajectory ({tr.info.model})")
    ax.plot(x[0], y[0], "o", color="C2", label="start")
    ax.plot(x[-1], y[-1], "s", color="C3", label="end")
    days = tr.t[-1] / 86400.0
    _finish(ax, f"Trajectory ({days:.2f} days)\nprojected on initial orbit plane")
    return ax


def plot_transfer(t: Transfer, ax=None):
    """Plot a transfer. Each orbit is unfolded into its own plane, so shapes stay true across plane changes.

    The x axis points to the first burn (the node). All burns lie on the node axis, so the unfolded orbits connect.
    """
    ax = _axes(ax)
    body = t.initial.body
    _draw_body(ax, body)

    if not t.burns:
        x_hat = _unit(t.initial.r)
    else:
        x_hat = _unit(t.burns[0].r)

    def frame(orbit):
        return x_hat, np.cross(_unit(np.cross(orbit.r, orbit.v)), x_hat)

    for orbit, label, style in ((t.initial, "initial", dict(color="C0")),
                                (t.target, "target", dict(color="C2"))):
        x, y = _project(_orbit_points(orbit), *frame(orbit))
        ax.plot(x, y, label=label, **style)

    # Transfer orbits: the flown arc solid, the rest dotted
    transfer_orbits = t.orbits[:-1]
    for k, orbit in enumerate(transfer_orbits):
        fx, fy = frame(orbit)
        x, y = _project(_orbit_points(orbit), fx, fy)
        ax.plot(x, y, ":", color="C1", lw=0.8)
        dt = t.burns[k + 1].epoch - t.burns[k].epoch
        r, _ = _prop.kepler_states(orbit.r, orbit.v, np.linspace(0.0, dt, 181), body.mu)
        x, y = _project(r, fx, fy)
        ax.plot(x, y, color="C1", lw=2, label="transfer" if k == 0 else None)

    for k, (burn, orbit) in enumerate(zip(t.burns, t.orbits)):
        bx, by = _project(burn.r[None, :], *frame(orbit))
        ax.plot(bx, by, "*", color="C3", ms=12, label="burn" if k == 0 else None)
        ax.annotate(f"Δv{k + 1} = {burn.magnitude / KM:.3f} km/s", (bx[0], by[0]),
                    textcoords="offset points", xytext=(8, 8), fontsize="small")

    title = f"{t.kind.replace('_', ' ').title()}\nΔv ={t.total_dv / KM:.3f} km/s, " \
            f"tof = {t.tof / 3600:.2f} h"
    if t.plane_change > 1e-9:
        title += f", plane change {math.degrees(t.plane_change):.2f}°"
    _finish(ax, title)
    return ax

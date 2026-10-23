"""2D orbit plots (matplotlib).

Every function returns the matplotlib Axes and draws on ax if one is given.
Coordinates are shown in km.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from . import units
from .core import elements as _el
from .core import propagate as _prop
from .time import Epoch

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
    _finish(ax, f"Orbit around {orbit.body.name}\na = {orbit.a / KM:,.0f} km, "
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

    for k, (burn, orbit) in enumerate(zip(t.burns, t.orbits, strict=True)):
        bx, by = _project(burn.r[None, :], *frame(orbit))
        ax.plot(bx, by, "*", color="C3", ms=12, label="burn" if k == 0 else None)
        ax.annotate(f"Δv{k + 1} = {burn.magnitude / KM:.3f} km/s", (bx[0], by[0]),
                    textcoords="offset points", xytext=(8, 8), fontsize="small")

    title = f"{t.kind.replace('_', ' ').title()}\nΔv = {t.total_dv / KM:.3f} km/s, " \
            f"tof = {t.tof / 3600:.2f} h"
    if t.plane_change > 1e-9:
        title += f", plane change {math.degrees(t.plane_change):.2f}°"
    _finish(ax, title)
    return ax


def _break_wraps(lon_deg: np.ndarray, lat_deg: np.ndarray):
    """Insert NaNs where the longitude wraps around ±180° so lines do not cross the map."""
    jumps = np.nonzero(np.abs(np.diff(lon_deg)) > 180.0)[0] + 1
    return np.insert(lon_deg, jumps, np.nan), np.insert(lat_deg, jumps, np.nan)


def plot_ground_track(gt, ax=None, stations=()):
    """Ground track in degrees (equirectangular), with optional ground stations."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 5))
    lon, lat = _break_wraps(np.degrees(gt.lon), np.degrees(gt.lat))
    ax.plot(lon, lat, color="C0", lw=0.8, label="ground track")
    ax.plot(math.degrees(gt.lon[0]), math.degrees(gt.lat[0]), "o", color="C2", label="start")
    ax.plot(math.degrees(gt.lon[-1]), math.degrees(gt.lat[-1]), "s", color="C3", label="end")
    for k, st in enumerate(stations):
        ax.plot(math.degrees(st.lon), math.degrees(st.lat), "^", color="C1", ms=9,
                label="station" if k == 0 else None)
        ax.annotate(st.name, (math.degrees(st.lon), math.degrees(st.lat)), textcoords="offset points",
                    xytext=(6, 6), fontsize="small")
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.set_xticks(range(-180, 181, 60))
    ax.set_yticks(range(-90, 91, 30))
    ax.set_aspect("equal")
    ax.set_xlabel("longitude [deg]")
    ax.set_ylabel("latitude [deg]")
    ax.set_title(f"Ground track ({gt.t[-1] / 86400:.2f} days from {gt.epoch.utc[:16]} UTC)")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left", fontsize="small")
    return ax


def plot_illumination(report, ax=None):
    """Visible fraction of the Sun against time."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(9, 3))
    hours = report.t / 3600.0
    ax.fill_between(hours, report.illumination, step=None, color="#f2c14e", alpha=0.6, lw=0)
    ax.plot(hours, report.illumination, color="#b07d00", lw=0.8)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("time [h]")
    ax.set_ylabel("sunlit fraction")
    ax.set_title(f"Eclipses: {len(report)} passages, {100 * report.shadow_fraction:.1f}% of the time in shadow")
    ax.grid(alpha=0.3)
    return ax


def plot_lifetime(life, ax=None):
    """Altitude against time until reentry."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))
    days = life.t / 86400.0
    label = "mean altitude" if life.method == "averaged" else "altitude"
    ax.plot(days, life.altitude / KM, color="C0", lw=0.8 if life.method == "numerical" else 1.5, label=label)
    ax.axhline(life.reentry_altitude / KM, color="C3", ls="--", lw=1, label="reentry altitude")
    ax.set_xlabel("time [day]")
    ax.set_ylabel("altitude [km]")
    status = f"{life.days:.1f} days" if life.reentered else f"> {life.days:.1f} days"
    ax.set_title(f"Lifetime ({life.method}): {status}")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize="small")
    return ax


def plot_body_transfer(t, ax=None):
    """The Hohmann ellipse around the central body, both circular orbits, and the bodies at departure.

    With a departure date the x axis is the ecliptic vernal equinox; otherwise the origin departs along +x.
    """
    import matplotlib.patches as mpatches

    ax = _axes(ax)
    c = t.central
    scale, unit = (1.495978707e11, "AU") if c.parent is None else (KM, "km")
    theta0 = t.departure_longitude or 0.0
    ang = np.linspace(0.0, 2.0 * math.pi, 361)
    ax.add_patch(mpatches.Circle((0, 0), c.radius / scale, color="#e0a030" if c.parent is None else "#4a7ab8",
                                 alpha=0.6, zorder=3, label=c.name))
    if t.kind == "siblings":
        circles = ((t.r1, t.origin.name, "C0"), (t.r2, t.target.name, "C2"))
    elif t.kind == "to_moon":
        circles = ((t.r1, "parking orbit", "C0"), (t.r2, t.target.name, "C2"))
    else:
        circles = ((t.r1, t.origin.name, "C0"), (t.r2, "capture orbit", "C2"))
    for r, label, color in circles:
        ax.plot(r * np.cos(ang) / scale, r * np.sin(ang) / scale, color=color, lw=1, label=label)

    # Half ellipse from r1 at angle theta0 to r2 at theta0 + π (focus at the center)
    a, e = t.a_transfer, abs(t.r2 - t.r1) / (t.r1 + t.r2)
    nu = np.linspace(0.0, math.pi, 181)
    if t.r2 >= t.r1:
        r = a * (1 - e * e) / (1 + e * np.cos(nu))
    else:
        r = a * (1 - e * e) / (1 - e * np.cos(nu))
    ax.plot(r * np.cos(nu + theta0) / scale, r * np.sin(nu + theta0) / scale, color="C1", lw=2, label="transfer")

    ax.plot(t.r1 * math.cos(theta0) / scale, t.r1 * math.sin(theta0) / scale, "o", color="C0")
    ax.plot(-t.r2 * math.cos(theta0) / scale, -t.r2 * math.sin(theta0) / scale, "*", color="C3", ms=12,
            label="arrival")
    if t.phase_angle is not None:
        th = theta0 + t.phase_angle
        ax.plot(t.r2 * math.cos(th) / scale, t.r2 * math.sin(th) / scale, "o", color="C2",
                label=f"{t.target.name} at departure")
    ax.set_aspect("equal")
    ax.set_xlabel(f"x [{unit}]")
    ax.set_ylabel(f"y [{unit}]")
    title = f"{t.origin.name} → {t.target.name}: Δv = {t.total_dv / KM:.3f} km/s, tof = {t.tof / 86400:.1f} d"
    if t.departure_epoch is not None:
        title += f"\ndepart {t.departure_epoch.iso[:10]}, arrive {t.arrival_epoch.iso[:10]}"
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize="small")
    return ax


AU = 1.495978707e11


def plot_mission(m, ax=None):
    """Heliocentric view (J2000 ecliptic, AU): planet orbits, Lambert legs and the planets at each event."""
    from .core import planets as _pl

    ax = _axes(ax)
    ax.plot(0, 0, "o", color="#e0a030", ms=10, label="Sun")
    seen = set()
    for body in m.sequence:
        if body.name in seen:
            continue
        seen.add(body.name)
        jds = m.departure.jd + np.linspace(0.0, body.orbital_period / 86400.0, 241)
        pts = np.array([_pl.heliocentric_state(body.ephemeris, jd, body.parent.mu)[0] for jd in jds])
        ax.plot(pts[:, 0] / AU, pts[:, 1] / AU, color="0.6", lw=0.7)
    for k, leg in enumerate(m.legs):
        r, _ = _prop.kepler_states(leg.r1, leg.v1, np.linspace(0.0, leg.tof, 361), leg.origin.parent.mu)
        ax.plot(r[:, 0] / AU, r[:, 1] / AU, color=f"C{k}", lw=2, label=f"{leg.origin.name} → {leg.target.name}")
    events = [(m.legs[0].origin, m.legs[0].r1, m.legs[0].departure)]
    events += [(leg.target, leg.r2, leg.arrival) for leg in m.legs]
    for body, r, epoch in events:
        ax.plot(r[0] / AU, r[1] / AU, "o", color="k", ms=5)
        ax.annotate(f"{body.name}\n{epoch.iso[:10]}", (r[0] / AU, r[1] / AU), textcoords="offset points",
                    xytext=(6, 6), fontsize="small")
    ax.set_aspect("equal")
    ax.set_xlabel("x [AU] (ecliptic J2000)")
    ax.set_ylabel("y [AU]")
    path = " → ".join(b.name for b in m.sequence)
    ax.set_title(f"{path}\nΔv = {m.total_dv / KM:.3f} km/s, C3 = {m.c3 / 1e6:.2f} km²/s², "
                 f"tof = {m.tof / 86400:.0f} d")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize="small")
    return ax


def plot_porkchop(p, ax=None, max_c3=None):
    """C3 contours over departure and arrival dates, with time-of-flight lines and the minimum marked."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))
    x = p.departure_jd - p.departure_jd[0]
    y = p.arrival_jd - p.departure_jd[0]
    c3 = p.c3 / 1e6
    if not np.all(np.isnan(c3)):
        low = np.nanmin(c3)
        top = 4 * low if max_c3 is None else units.to_si(max_c3, units.u.km**2 / units.u.s**2) / 1e6
        cs = ax.contour(x, y, np.where(c3 <= top, c3, np.nan), levels=np.linspace(low, top, 15),
                        cmap="viridis")
        ax.clabel(cs, fmt="%.1f", fontsize="x-small")
        ts = ax.contour(x, y, p.tof / 86400.0, levels=8, colors="0.5", linestyles="--", linewidths=0.6)
        ax.clabel(ts, fmt="%.0f d", fontsize="x-small")
        i, j = np.unravel_index(np.nanargmin(c3), c3.shape)
        ax.plot(x[j], y[i], "*", color="C3", ms=12, label=f"min C3 {c3[i, j]:.2f} km²/s²")
        ax.legend(loc="upper left", fontsize="small")
    start = Epoch.from_jd(p.departure_jd[0]).iso[:10]
    ax.set_xlabel(f"departure [days after {start}]")
    ax.set_ylabel(f"arrival [days after {start}]")
    ax.set_title(f"{p.origin.name} → {p.target.name}: C3 [km²/s²]")
    ax.grid(alpha=0.3)
    return ax


def plot_relative(rel, ax=None):
    """LVLH relative motion: along-track (horizontal) against radial (vertical), the chief at the origin."""
    ax = _axes(ax)
    ax.plot(rel.r[:, 1], rel.r[:, 0], color="C0", label="deputy")
    ax.plot(rel.r[0, 1], rel.r[0, 0], "o", color="C2", label="start")
    ax.plot(rel.r[-1, 1], rel.r[-1, 0], "s", color="C3", label="end")
    ax.plot(0, 0, "+", color="k", ms=12, label=rel.chief)
    ax.set_xlabel("along-track y [m]")
    ax.set_ylabel("radial x [m]")
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_title(f"Relative motion (LVLH of {rel.chief})")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize="small")
    return ax


def plot_doppler(d, ax=None, min_elevation=0.0):
    """Doppler shift [kHz] against time [min], only while the satellite is above min_elevation."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))
    shift = np.where(d.elevation >= min_elevation, d.shift / 1e3, np.nan)
    ax.plot(d.t / 60.0, shift, color="C0")
    ax.axhline(0, color="0.5", lw=0.6)
    ax.set_xlabel("time [min]")
    ax.set_ylabel("Doppler shift [kHz]")
    ax.set_title(f"Doppler at {d.station.name}, {d.frequency / 1e6:.3f} MHz")
    ax.grid(alpha=0.3)
    return ax

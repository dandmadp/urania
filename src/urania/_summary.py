"""summary(): the key numbers of a result in everyday units (km/s, days, kg), one per line."""

from __future__ import annotations

import math

import numpy as np

DAY = 86400.0


class Summary(str):
    """Text that displays as itself in a REPL or Jupyter (no quotes or escapes)."""

    def __repr__(self) -> str:
        return str(self)


def _table(title: str, rows: list[tuple[str, str]]) -> Summary:
    width = max(len(k) for k, _ in rows)
    return Summary("\n".join([title] + [f"  {k:<{width}}  {v}" for k, v in rows]))


def _kms(x: float) -> str:
    return f"{x / 1e3:.4f} km/s"


def _duration(seconds: float) -> str:
    if abs(seconds) >= 2 * DAY:
        return f"{seconds / DAY:.1f} days"
    if abs(seconds) >= 2 * 3600:
        return f"{seconds / 3600:.2f} h"
    return f"{seconds / 60:.1f} min"


def _propellant(rows, total_dv: float, mass, isp) -> None:
    if mass is None and isp is None:
        return
    if mass is None or isp is None:
        raise ValueError("Give both mass and isp for the propellant estimate")
    from . import units
    from .core.propulsion import propellant_mass

    m0, isp_s = units.to_si(mass, units.MASS), units.to_si(isp, units.TIME)
    mp = propellant_mass(total_dv, isp_s, m0)
    rows.append(("propellant", f"{mp:.3f} kg of {m0:g} kg ({100 * mp / m0:.1f}%, Isp {isp_s:g} s)"))


def transfer_summary(t, mass=None, isp=None) -> Summary:
    names = {"hohmann": "Hohmann transfer", "bielliptic": "Bi-elliptic transfer", "plane_change": "Plane change",
             "lambert": "Lambert rendezvous", "phasing": "Phasing maneuver", "deorbit": "Deorbit burn",
             "none": "No transfer"}
    rows = [("total Δv", _kms(t.total_dv))]
    for k, b in enumerate(t.burns, 1):
        what = b.description.split(": ", 1)[-1]
        rows.append((f"burn {k}", f"{_kms(b.magnitude)}  at {b.epoch.iso[:19]} TDB  ({what})"))
    if t.burns:
        rows.append(("transfer time", _duration(t.tof)))
    if t.coast > 0:
        rows.append(("coast before", _duration(t.coast)))
    if t.plane_change > 1e-9:
        rows.append(("plane change", f"{math.degrees(t.plane_change):.3f}°"))
    f = t.final
    rows.append(("final orbit", f"{f.periapsis_altitude / 1e3:,.1f} × {f.apoapsis_altitude / 1e3:,.1f} km, "
                                f"i = {math.degrees(f.inc):.2f}°"))
    _propellant(rows, t.total_dv, mass, isp)
    return _table(names.get(t.kind, t.kind), rows)


def body_transfer_summary(t, mass=None, isp=None) -> Summary:
    rows = []
    if t.v_inf_departure is not None:
        rows += [("v∞ departure", _kms(t.v_inf_departure)), ("C3", f"{t.c3 / 1e6:.3f} km²/s²")]
    if t.v_inf_arrival is not None:
        rows.append(("v∞ arrival", _kms(t.v_inf_arrival)))
    rows += [("Δv departure", f"{_kms(t.dv_departure)}  from a {t.parking_altitude / 1e3:,.0f} km orbit"),
             ("Δv capture", "none (arrival only)" if t.capture_altitude is None
              else f"{_kms(t.dv_arrival)}  into a {t.capture_altitude / 1e3:,.0f} km periapsis"),
             ("total Δv", _kms(t.total_dv)),
             ("flight time", _duration(t.tof))]
    if t.phase_angle is not None:
        rows.append(("phase angle", f"{math.degrees(t.phase_angle):.2f}° (target ahead at departure)"))
        rows.append(("synodic period", _duration(t.synodic_period)))
    if t.stay_time is not None:
        rows.append(("wait for return", _duration(t.stay_time)))
    if t.departure_epoch is not None:
        rows += [("departure", f"{t.departure_epoch.iso[:10]}"), ("arrival", f"{t.arrival_epoch.iso[:10]}")]
    _propellant(rows, t.total_dv, mass, isp)
    return _table(f"Hohmann transfer {t.origin.name} → {t.target.name} (around {t.central.name})", rows)


def mission_summary(m, mass=None, isp=None) -> Summary:
    rows = [("dates", f"{m.departure.iso[:10]} → {m.arrival.iso[:10]} ({m.tof / DAY:.0f} days)"),
            ("C3", f"{m.c3 / 1e6:.3f} km²/s²  (v∞ {_kms(m.v_inf_departure)})"),
            ("Δv departure", f"{_kms(m.dv_departure)}  from a {m.parking_altitude / 1e3:,.0f} km orbit")]
    for f in m.flybys:
        state = "" if f.feasible else "  INFEASIBLE (below the minimum altitude)"
        rows.append((f"flyby {f.body.name}", f"{f.epoch.iso[:10]}, altitude {f.altitude / 1e3:,.0f} km, "
                                             f"burn {_kms(f.dv)}{state}"))
    rows += [("v∞ arrival", _kms(m.v_inf_arrival)),
             ("Δv capture", "none (arrival only)" if m.capture_altitude is None else _kms(m.dv_arrival)),
             ("total Δv", _kms(m.total_dv))]
    _propellant(rows, m.total_dv, mass, isp)
    return _table("Mission " + " → ".join(b.name for b in m.sequence), rows)


CHEATSHEET = """urania cheat sheet  (import urania as ur; from urania import u)
Values take units: 400 * u.km, "400 km", 51.6 * u.deg, "90 min". Plain numbers are SI (m, s, rad).
Dates: "2026-11-01", "2026-11-01 12:30" (TDB) or "2026-11-01T12:30Z" (UTC).

Orbits
  ur.Orbit.circular("Earth", "500 km", inc="97.4 deg")     circular orbit
  ur.Orbit.from_apsides(ur.Earth, "400 km", "600 km")       400 × 600 km ellipse
  ur.sun_synchronous("700 km")                              sun-synchronous orbit
  ur.ISS, ur.LEO, ur.SSO, ur.GEO                            presets
  orbit.summary()   orbit.explain()   orbit.plot()          numbers, formulas, picture

Propagation and operations (Earth)
  tr = orbit.propagate(days=1, model="j2")                  "twobody", "j2", "j2+drag"
  tr.ground_track().plot()                                  ground track
  station = ur.GroundStation.create("Daejeon", "36.37 deg", "127.36 deg")
  tr.passes(station)   tr.eclipses()   tr.doppler(station, "437.5 MHz")
  orbit.lifetime(area="0.03 m2", mass="4 kg")               decay under drag
  orbit.footprint("10 deg")   orbit.beta_angle()   orbit.eclipse_fraction()
  ur.TLEOrbit.from_text(tle_text).propagate(hours=12)       real satellites from TLEs

Maneuvers (every result has .summary(mass=, isp=), .explain(), .plot())
  ur.LEO.transfer_to(ur.GEO)                                Hohmann transfer
  chaser.rendezvous(target, "345 min", revolutions=3)       meet another spacecraft
  orbit.phasing("10 deg")   orbit.deorbit()                 move along the orbit, dispose

Planets and moons
  ur.body("titan")   ur.PLANETS   ur.MOONS                  presets, by name
  ur.Earth.hohmann_to("Mars", "300 km", "300 km", after="2026-01-01")
  ur.Earth.lambert_to("Mars", "2026-11-01", "300 day", "300 km")
  ur.porkchop("Earth", "Mars", ("2026-08-01", "2027-01-01"), ("2027-04-01", "2028-03-01")).plot()
  ur.flyby_mission(["Earth", "Venus", "Mercury"], ["1973-11-03", "1974-02-05", "1974-03-29"], "185 km")

In a script, call matplotlib.pyplot.show() after .plot() to open the window.
"""


def cheatsheet() -> Summary:
    """Common tasks and the one line of code for each."""
    return Summary(CHEATSHEET.rstrip("\n"))

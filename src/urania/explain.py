"""explain(): the formulas used, intermediate values, assumptions and warnings as step-by-step text.

Intermediate values are computed with core functions (formulas live only in core).
If a recomputed value disagrees with the value stored in the result object, a warning is added.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from .core import maneuvers as _man
from .core import twobody as _tb

if TYPE_CHECKING:
    from .maneuvers import Transfer
    from .orbits import Orbit
    from .propagation import Trajectory
    from .tle import TLEOrbit

DAY = 86400.0


# ---------------------------------------------------------------- formatting

def _km(x: float) -> str:
    return f"{x / 1e3:,.3f} km"


def _kms(x: float) -> str:
    return f"{x / 1e3:.4f} km/s"


def _deg(x: float) -> str:
    return f"{math.degrees(x):.4f}°"


def _vec(x, scale: float, unit: str) -> str:
    return "[" + ", ".join(f"{c / scale:,.3f}" for c in x) + f"] {unit}"


class Explanation:
    """Step-by-step explanation. print() it, or view it directly in a REPL or Jupyter."""

    def __init__(self, title: str):
        self.title = title
        self.steps: list[tuple[str, list[str]]] = []
        self.assumptions: list[str] = []
        self.warnings: list[str] = []

    def step(self, heading: str, *lines: str) -> None:
        self.steps.append((heading, list(lines)))

    def __str__(self) -> str:
        out = [self.title, "─" * max(len(self.title), 20)]
        for i, (heading, lines) in enumerate(self.steps, 1):
            out.append(f"{i}. {heading}")
            out.extend(f"   {line}" for line in lines)
        if self.assumptions:
            out.append("Assumptions")
            out.extend(f"   - {a}" for a in self.assumptions)
        if self.warnings:
            out.append("Warnings")
            out.extend(f"   ! {w}" for w in self.warnings)
        return "\n".join(out)

    __repr__ = __str__


# ---------------------------------------------------------------- Orbit

def explain_orbit(o: Orbit) -> Explanation:
    """How the orbital elements follow from the state vector (Curtis algorithm 4.2)."""
    b = o.body
    mu = b.mu
    r_norm, v_norm = np.linalg.norm(o.r), np.linalg.norm(o.v)
    ex = Explanation(f"Orbital elements: orbit around {b.name}, {o.epoch.iso[:19]} TDB")

    ex.step("Input state vector",
            f"r = {_vec(o.r, 1e3, 'km')},  |r| = {_km(r_norm)}",
            f"v = {_vec(o.v, 1e3, 'km/s')},  |v| = {_kms(v_norm)}",
            f"μ = {mu:.6e} m³/s²  ({b.source or 'source not given'})")
    ex.step("Specific angular momentum and semi-latus rectum",
            "h = r × v",
            f"|h| = {o.h / 1e6:,.3f} km²/s",
            f"p = h²/μ = {_km(o.p)}")
    ex.step("Eccentricity",
            "e = ((v² − μ/r) r − (r·v) v) / μ",
            f"|e| = {o.ecc:.6f}")
    ex.step("Energy and semi-major axis (vis-viva)",
            "ε = v²/2 − μ/r,  a = −μ/(2ε)",
            f"ε = {o.energy / 1e6:.4f} km²/s²",
            f"a = {_km(o.a)}" + ("  (negative for a hyperbola)" if o.a < 0 else ""))
    ex.step("Inclination", "i = arccos(h_z / |h|)", f"i = {_deg(o.inc)}")
    ex.step("Right ascension of the ascending node", "n = ẑ × h,  Ω = atan2(n_y, n_x)",
            f"Ω = {_deg(o.raan)}")
    ex.step("Argument of periapsis", "ω = angle from n to e (in the orbit plane, direction of motion)",
            f"ω = {_deg(o.argp)}")
    ex.step("True anomaly", "ν = angle from e to r (in the orbit plane, direction of motion)",
            f"ν = {_deg(o.nu)}")

    derived = [f"periapsis radius r_p = p/(1+e) = {_km(o.r_periapsis)}  "
               f"(altitude {_km(o.periapsis_altitude)})"]
    if o.ecc < 1.0:
        derived.append(f"apoapsis radius r_a = p/(1−e) = {_km(o.r_apoapsis)}  "
                       f"(altitude {_km(o.apoapsis_altitude)})")
        derived.append(f"period T = 2π√(a³/μ) = {o.period:,.1f} s = {o.period / 60:.2f} min")
        if b.J2:
            derived.append(f"J2 nodal rate dΩ/dt = −(3/2) n J2 (R/p)² cos i = "
                           f"{math.degrees(o.raan_rate) * DAY:.4f} °/day")
    ex.step("Derived quantities", *derived)

    if o.ecc < 1e-11:
        ex.assumptions.append("circular orbit: periapsis is undefined, so ω = 0 and ν is measured "
                              "from the node")
    if o.inc < 1e-11 or abs(o.inc - math.pi) < 1e-11:
        ex.assumptions.append("equatorial orbit: the node is undefined, so Ω = 0 and the reference "
                              "axis is inertial x")
    ex.assumptions.append("two-body elements (the osculating orbit at this instant)")
    return ex


# ---------------------------------------------------------------- Trajectory

def explain_trajectory(tr: Trajectory) -> Explanation:
    """Equations of motion, integrator, changes in the result and interpretation caveats."""
    info = tr.info
    terms = info.model.split("+")
    ex = Explanation(f"Orbit propagation: {info.model}, {tr.t[-1] / DAY:.3f} days")

    sgp4 = info.model == "sgp4"
    if sgp4:
        ex.step("Model",
                "SGP4: analytic perturbation theory for TLE mean elements (Hoots & Roehrich 1980)",
                "includes J2, J3, J4 zonal terms and B*-based drag; "
                "SDP4 (Moon, Sun, resonance) for periods of 225 min or more")
    else:
        eq = ["r̈ = " + " + ".join({"twobody": "a_2body", "j2": "a_J2", "drag": "a_drag"}[t]
                                   for t in terms),
              "a_2body = −μ r / |r|³"]
        if "j2" in terms:
            eq.append("a_J2 = −(3/2) J2 μ R² / r⁵ · [x(1 − 5z²/r²), y(1 − 5z²/r²), z(3 − 5z²/r²)]")
        if "drag" in terms:
            eq.append("a_drag = −½ ρ (C_D A/m) |v_rel| v_rel,  v_rel = v − ω × r")
        ex.step("Equations of motion", *eq)

    if sgp4:
        ex.step("Solution", f"{info.integrator}: analytic formulas evaluated at each time",
                f"{len(tr)} output points")
    elif info.rtol is None:
        ex.step("Solution", f"{info.integrator}: Kepler's equation solved at each time",
                f"{len(tr)} output points")
    else:
        ex.step("Numerical integration", info.integrator,
                f"tolerances rtol = {info.rtol:g}, atol = {info.atol:g} (m, m/s)",
                f"{info.nfev:,} acceleration evaluations, {len(tr)} output points")

    first, last = tr.orbit_at(0), tr.final
    lines = []
    for name, f, fmt in (("a", lambda o: o.a, _km), ("e", lambda o: o.ecc, lambda x: f"{x:.6f}"),
                         ("i", lambda o: o.inc, _deg), ("Ω", lambda o: o.raan, _deg)):
        lines.append(f"{name}: {fmt(f(first))} → {fmt(f(last))}")
    alt = tr.altitude
    lines.append(f"altitude: {_km(alt[0])} → {_km(alt[-1])}")
    ex.step("Start → end (osculating elements)", *lines)

    if ("j2" in terms or sgp4) and first.ecc < 1.0:
        a = _tb.semi_major_axis(tr.r, tr.v, tr.body.mu)
        T = first.period
        head = a[np.abs(tr.t - tr.t[0]) <= T]
        tail = a[np.abs(tr.t[-1] - tr.t) <= T]
        if abs(tr.t[-1] - tr.t[0]) >= 2 * T:
            ex.step("Short-period oscillation of the semi-major axis (J2)",
                    f"osculating a range: {_km(a.min())} to {_km(a.max())} "
                    f"(spread {_km(a.max() - a.min())})",
                    f"mean a over the first orbit = {_km(head.mean())}",
                    f"mean a over the last orbit = {_km(tail.mean())}",
                    f"change in mean = {_km(tail.mean() - head.mean())}")
        ex.warnings.append(
            "With J2 the osculating semi-major axis swings strongly within each orbit. "
            "Do not read the start-to-end difference as orbital decay; use the change in the "
            "orbit-averaged value.")

    ex.assumptions.extend(info.assumptions)
    if info.terminated:
        ex.warnings.append(f"{info.terminated}: propagation stopped at {tr.t[-1] / DAY:.4f} days.")
    return ex


# ---------------------------------------------------------------- Transfer

def _normal(o: Orbit) -> np.ndarray:
    h = np.cross(o.r, o.v)
    return h / np.linalg.norm(h)


def _angle(a: np.ndarray, b: np.ndarray) -> float:
    return math.atan2(np.linalg.norm(np.cross(a, b)), a @ b)


def _burn_lines(v_before: float, v_after: float, alpha: float, stored: float,
                ex: Explanation, label: str) -> list[str]:
    """Δv formula and value for one burn, checked against the state-vector result (stored)."""
    if alpha > 1e-12:
        dv = _man.combined_dv(v_before, v_after, alpha)
        lines = [f"combined with plane change α = {_deg(alpha)}",
                 f"Δv = √(v₁² + v₂² − 2 v₁ v₂ cos α) = {_kms(dv)}"]
    else:
        dv = abs(v_after - v_before)
        lines = [f"Δv = |v₂ − v₁| = {_kms(dv)}"]
    if abs(dv - stored) > 1e-6 * max(dv, 1.0):
        ex.warnings.append(f"{label}: formula value {_kms(dv)} differs from the state-vector "
                           f"value {_kms(stored)}.")
    return lines


def explain_transfer(t: Transfer) -> Explanation:
    """Transfer formulas, intermediate speeds, Δv per burn and transfer time."""
    mu = t.initial.body.mu
    r1, r2 = t.initial.a, t.target.a
    names = {"hohmann": "Hohmann transfer", "bielliptic": "Bi-elliptic transfer",
             "plane_change": "Plane change", "none": "No transfer"}
    if t.kind in ("lambert", "phasing", "deorbit"):
        return _explain_other_transfer(t)
    ex = Explanation(f"{names[t.kind]}: r₁ = {_km(r1)} → r₂ = {_km(r2)}")

    start = [f"μ = {mu:.6e} m³/s²",
             f"angle between the orbit planes Δθ = arccos(ĥ₁·ĥ₂) = {_deg(t.plane_change)}"]
    if t.coast > 0:
        start.append(f"coast to the node: {t.coast:,.1f} s ({t.coast / 60:.2f} min)")
    ex.step("Initial conditions", *start)

    if t.kind == "none":
        ex.step("Result", "Already on the same orbit; no burn is needed.")
        ex.assumptions.extend(t.assumptions)
        return ex

    burns = t.burns
    if t.kind == "plane_change":
        v = _tb.circular_velocity(r1, mu)
        dv = _man.plane_change_dv(v, t.plane_change)
        ex.step("Plane change (at the node)",
                f"v = √(μ/r) = {_kms(v)}",
                f"Δv = 2 v sin(Δθ/2) = {_kms(dv)}")
        if abs(dv - burns[0].magnitude) > 1e-6 * dv:
            ex.warnings.append("The formula value differs from the state-vector value.")
    elif t.kind == "hohmann":
        a_t = 0.5 * (r1 + r2)
        alpha1 = _angle(_normal(t.initial), _normal(t.orbits[0]))
        alpha2 = t.plane_change - alpha1
        v_c1, v_p = _tb.circular_velocity(r1, mu), _tb.vis_viva(r1, a_t, mu)
        v_a, v_c2 = _tb.vis_viva(r2, a_t, mu), _tb.circular_velocity(r2, mu)
        ex.step("Transfer ellipse", f"a_t = (r₁ + r₂)/2 = {_km(a_t)}")
        ex.step("Burn 1 (r₁)",
                f"v₁ = √(μ/r₁) = {_kms(v_c1)}  (initial circular orbit)",
                f"v₂ = √(μ(2/r₁ − 1/a_t)) = {_kms(v_p)}  (transfer ellipse)",
                *_burn_lines(v_c1, v_p, alpha1, burns[0].magnitude, ex, "Burn 1"))
        ex.step("Burn 2 (r₂)",
                f"v₁ = √(μ(2/r₂ − 1/a_t)) = {_kms(v_a)}  (transfer ellipse)",
                f"v₂ = √(μ/r₂) = {_kms(v_c2)}  (target circular orbit)",
                *_burn_lines(v_a, v_c2, alpha2, burns[1].magnitude, ex, "Burn 2"))
        ex.step("Transfer time", f"tof = π√(a_t³/μ) = {t.tof:,.1f} s = {t.tof / 3600:.4f} h")
    elif t.kind == "bielliptic":
        rb = np.linalg.norm(burns[1].r)
        a1, a2 = 0.5 * (r1 + rb), 0.5 * (rb + r2)
        ex.step("Two transfer ellipses",
                f"intermediate apoapsis r_b = {_km(rb)}",
                f"a₁ = (r₁ + r_b)/2 = {_km(a1)},  a₂ = (r_b + r₂)/2 = {_km(a2)}")
        v = [(_tb.circular_velocity(r1, mu), _tb.vis_viva(r1, a1, mu), 0.0, "r₁"),
             (_tb.vis_viva(rb, a1, mu), _tb.vis_viva(rb, a2, mu), t.plane_change, "r_b"),
             (_tb.vis_viva(r2, a2, mu), _tb.circular_velocity(r2, mu), 0.0, "r₂")]
        for k, (vb, va, alpha, where) in enumerate(v):
            ex.step(f"Burn {k + 1} ({where})",
                    f"v₁ = {_kms(vb)},  v₂ = {_kms(va)}  (vis-viva √(μ(2/r − 1/a)))",
                    *_burn_lines(vb, va, alpha, burns[k].magnitude, ex, f"Burn {k + 1}"))
        ex.step("Transfer time", f"tof = π(√(a₁³/μ) + √(a₂³/μ)) = {t.tof / 3600:.4f} h")

    ex.step("Total", " + ".join(_kms(b.magnitude) for b in burns) + f" = {_kms(t.total_dv)}")
    ex.assumptions.extend(t.assumptions)
    return ex


# ---------------------------------------------------------------- TLE

def explain_tle(tle: TLEOrbit) -> Explanation:
    """Decoded TLE fields. The values are SGP4 mean elements, not osculating elements."""
    from .tle import MU_WGS72, checksum

    title = f"TLE: {tle.name or 'unnamed'} (NORAD {tle.satnum})"
    ex = Explanation(title)
    ex.step("Raw lines", tle.line1, tle.line2,
            f"checksums: line 1 {checksum(tle.line1)}, line 2 {checksum(tle.line2)} (valid)")
    epoch = tle.epoch
    ex.step("Epoch",
            f"line 1 columns 19–32 (year + day of year, UTC) → {epoch.to_astropy().utc.iso} UTC",
            f"= {epoch.iso} TDB")
    n = tle.mean_motion
    a = (MU_WGS72 / n**2) ** (1.0 / 3.0)
    ex.step("Mean elements (line 2)",
            f"inclination i = {_deg(tle.inc)}",
            f"right ascension of the ascending node Ω = {_deg(tle.raan)}",
            f"eccentricity e = {tle.ecc:.7f}  (written without the decimal point)",
            f"argument of perigee ω = {_deg(tle.argp)}",
            f"mean anomaly M = {_deg(tle.M)}",
            f"mean motion n = {tle.revs_per_day:.8f} rev/day",
            f"→ semi-major axis a ≈ (μ/n²)^(1/3) = {_km(a)}, period {2 * math.pi / n / 60:.2f} min")
    ex.step("Drag term (line 1)", f"B* = {tle.bstar:.5e} (1/Earth radii)",
            "Used only inside SGP4; it is not the physical C_D·A/m.")
    ex.assumptions.extend(("TLE elements are SGP4 mean elements. Do not use them as classical "
                           "elements; extract a state vector with to_orbit().",
                           "a is approximated from the Kozai mean motion and the WGS72 μ"))
    return ex


# ---------------------------------------------------------------- passes, eclipses, lifetime

def explain_passes(report) -> Explanation:
    """How pass predictions are computed and what they found."""
    st = report.station
    ex = Explanation(f"Passes over {st.name}: {len(report)} above {math.degrees(report.min_elevation):.1f}°")
    ex.step("Station",
            f"geodetic latitude {_deg(st.lat)}, longitude {_deg(st.lon)}, height {st.alt:,.0f} m",
            f"Earth-fixed position r_site = {_vec(st.r_ecef, 1e3, 'km')}")
    ex.step("Earth rotation",
            "θ = GMST(UT1), IAU 1982: 67310.54841 s + (876600 h + 8640184.812866 s) T + 0.093104 s T² − 6.2e-6 s T³",
            "r_ECEF = R3(θ) r_inertial")
    ex.step("Look angles (east-north-up at the station)",
            "ρ = r_ECEF − r_site,  elevation = asin(ρ_up / |ρ|),  azimuth = atan2(ρ_east, ρ_north)",
            "rise and set: roots of elevation − minimum; peak: bounded maximization between them")
    lines = [repr(p) for p in report.passes] or ["no pass in this interval"]
    lines.append(f"total contact {report.total_contact / 60:.1f} min from {report.start.utc[:19]} to "
                 f"{report.end.utc[:19]} UTC")
    ex.step("Passes", *lines)
    ex.assumptions.extend(report.assumptions)
    if any(p.truncated for p in report.passes):
        ex.warnings.append("A pass is cut by the start or end of the trajectory; its duration is incomplete.")
    return ex


def explain_eclipses(report) -> Explanation:
    """Shadow geometry and eclipse statistics."""
    ex = Explanation(f"Eclipses: {len(report)} over {report.span / DAY:.3f} days")
    ex.step("Shadow geometry (conical, Montenbruck & Gill 3.4.2)",
            "a = asin(R_sun / |r_sun − r|) (apparent Sun radius), b = asin(R_earth / |r|) (apparent Earth radius)",
            "c = angle between the Earth and Sun centers seen from the satellite",
            "penumbra: c < a + b,  umbra: c < b − a",
            "sunlit fraction = 1 − (overlap area of the two disks) / (π a²)")
    if report.eclipses:
        dur = [e.duration for e in report.eclipses if not e.truncated] or [e.duration for e in report.eclipses]
        umb = [e.umbra_duration for e in report.eclipses if not e.truncated] or [0.0]
        ex.step("Statistics",
                f"shadow per pass: {min(dur) / 60:.2f} to {max(dur) / 60:.2f} min "
                f"(umbra up to {max(umb) / 60:.2f} min)",
                f"fraction of time in shadow: {100 * report.shadow_fraction:.2f}%",
                *[repr(e) for e in report.eclipses[:5]],
                *(["..."] if len(report) > 5 else []))
    else:
        ex.step("Statistics", "no eclipse in this interval (full sunlight)")
    ex.assumptions.extend(report.assumptions)
    return ex


def explain_lifetime(life) -> Explanation:
    """Decay model, formulas and result."""
    o = life.initial
    ex = Explanation(f"Lifetime ({life.method}): "
                     + (f"{life.days:.1f} days" if life.reentered else f"more than {life.days:.1f} days"))
    ex.step("Initial orbit",
            f"periapsis altitude {_km(o.periapsis_altitude)}, apoapsis altitude {_km(o.apoapsis_altitude)}, "
            f"i = {_deg(o.inc)}")
    if life.method == "averaged":
        ex.step("Orbit-averaged decay",
                "a_T = −½ ρ (C_D A/m) |v_rel| (v_rel · t̂),  v_rel = v − ω × r",
                "da/dt = ⟨2 a_T / n⟩ over the argument of latitude (reduces to −ρ B √(μa) without rotation)",
                f"mean altitude at the start {_km(life.altitude[0])} (osculating {_km(o.a - o.body.radius)})")
    else:
        ex.step("Numerical propagation", "full j2+drag equations of motion; see Trajectory.explain()")
    if life.reentered:
        ex.step("Result", f"altitude falls below {_km(life.reentry_altitude)} after {life.days:.2f} days",
                f"reentry {life.reentry.utc[:19]} UTC")
    else:
        ex.step("Result", f"still above {_km(life.reentry_altitude)} after {life.days:.1f} days "
                          f"(increase max_days to search further)")
    ex.assumptions.extend(life.assumptions)
    return ex


# ---------------------------------------------------------------- body-to-body transfers

def explain_body_transfer(t) -> Explanation:
    """Patched-conic Hohmann transfer between bodies (Curtis ch. 8)."""
    from .core import interplanetary as _ip

    c, o, g = t.central, t.origin, t.target
    mu = c.mu
    a_t = t.a_transfer
    ex = Explanation(f"Hohmann transfer {o.name} → {g.name} around {c.name} (patched conics)")
    unit = (lambda x: f"{x / 1.495978707e11:.6f} AU") if c.parent is None else _km
    ex.step(f"Hohmann ellipse around {c.name}",
            f"μ = {mu:.6e} m³/s²",
            f"r₁ = {unit(t.r1)},  r₂ = {unit(t.r2)}",
            f"a_t = (r₁ + r₂)/2 = {unit(a_t)}",
            f"tof = π√(a_t³/μ) = {t.tof / DAY:.2f} days")

    v1, v2 = _tb.circular_velocity(t.r1, mu), _tb.circular_velocity(t.r2, mu)
    vp, va = _tb.vis_viva(t.r1, a_t, mu), _tb.vis_viva(t.r2, a_t, mu)
    r_park = o.radius + t.parking_altitude
    if t.kind == "to_moon":
        ex.step("Departure burn (in the parking orbit)",
                f"v_c = √(μ/r₁) = {_kms(v1)},  v_t = √(μ(2/r₁ − 1/a_t)) = {_kms(vp)}",
                f"Δv = |v_t − v_c| = {_kms(t.dv_departure)}")
    else:
        e, beta = _ip.hyperbola(t.v_inf_departure, r_park, o.mu)
        ex.step(f"Leaving {o.name}",
                f"speed of {o.name} V₁ = √(μ/r₁) = {_kms(v1)},  transfer speed √(μ(2/r₁ − 1/a_t)) = {_kms(vp)}",
                f"v∞ = |difference| = {_kms(t.v_inf_departure)},  C3 = v∞² = {t.c3 / 1e6:.4f} km²/s²",
                f"parking orbit r_p = {_km(r_park)} ({_km(t.parking_altitude)} altitude)",
                f"Δv = √(v∞² + 2μ/r_p) − √(μ/r_p) = {_kms(t.dv_departure)}  (μ of {o.name})",
                f"escape hyperbola e = 1 + r_p v∞²/μ = {e:.5f},  β = arccos(1/e) = {_deg(beta)}")

    if t.kind == "to_parent":
        lines = [f"transfer speed at r₂: √(μ(2/r₂ − 1/a_t)) = {_kms(va)}"]
        v_cap = math.sqrt(mu * (1.0 + t.capture_ecc) / t.r2)
        lines += [f"capture orbit speed at periapsis √(μ(1 + e)/r₂) = {_kms(v_cap)}  (e = {t.capture_ecc:g})",
                  f"Δv = {_kms(t.dv_arrival)}"]
        ex.step(f"Arriving at {g.name}", *lines)
    else:
        lines = [f"speed of {g.name} V₂ = √(μ/r₂) = {_kms(v2)},  transfer speed √(μ(2/r₂ − 1/a_t)) = {_kms(va)}",
                 f"v∞ = |difference| = {_kms(t.v_inf_arrival)}"]
        if t.capture_altitude is None:
            lines.append("no capture burn requested")
        else:
            r_cap = g.radius + t.capture_altitude
            lines += [f"capture periapsis r_p = {_km(r_cap)} ({_km(t.capture_altitude)} altitude), "
                      f"capture orbit e = {t.capture_ecc:g}",
                      f"Δv = √(v∞² + 2μ/r_p) − √(μ(1 + e)/r_p) = {_kms(t.dv_arrival)}  (μ of {g.name})"]
            rp_opt = _ip.optimal_capture_periapsis(t.v_inf_arrival, g.mu, t.capture_ecc)
            lines.append(f"Δv-optimal periapsis 2μ(1 − e)/(v∞²(1 + e)) = {_km(rp_opt)} "
                         f"({_km(rp_opt - g.radius)} altitude)")
        ex.step(f"Arriving at {g.name}", *lines)

    if t.phase_angle is not None:
        lines = [f"n₂ = √(μ/r₂³);  φ₀ = π − n₂·tof = {_deg(t.phase_angle)} "
                 f"({g.name} {'ahead of' if t.phase_angle >= 0 else 'behind'} "
                 f"{o.name if t.kind == 'siblings' else 'the spacecraft'})",
                 f"synodic period T₁T₂/|T₁ − T₂| = {t.synodic_period / DAY:.2f} days (how often the geometry repeats)"]
        if t.stay_time is not None:
            lines.append(f"shortest stay at {g.name} before a Hohmann return: {t.stay_time / DAY:.1f} days")
        ex.step("Phasing", *lines)
    if t.departure_epoch is not None:
        ex.step("Dates (Standish mean elements)",
                f"departure {t.departure_epoch.iso[:16]} TDB "
                f"({o.name} at ecliptic longitude {_deg(t.departure_longitude)})",
                f"arrival {t.arrival_epoch.iso[:16]} TDB")
    ex.step("Total", f"{_kms(t.dv_departure)} + {_kms(t.dv_arrival)} = {_kms(t.total_dv)}")
    ex.assumptions.extend(t.assumptions)
    return ex


def _explain_other_transfer(t: Transfer) -> Explanation:
    """Lambert rendezvous, phasing and deorbit transfers."""
    from .core import lambert as _lam

    mu = t.initial.body.mu
    if t.kind == "lambert":
        b1, b2 = t.burns
        arc = t.orbits[0]
        dnu = _lam.transfer_angle(b1.r, b2.r, True, np.cross(t.initial.r, t.initial.v))
        ex = Explanation(f"Lambert rendezvous: tof = {t.tof / 60:.2f} min")
        if t.coast > 0:
            ex.step("Coast", f"wait {t.coast:,.1f} s before the first burn")
        ex.step("Lambert problem (universal variables, Curtis algorithm 5.2)",
                f"r₁ = {_vec(b1.r, 1e3, 'km')} (chaser at the first burn)",
                f"r₂ = {_vec(b2.r, 1e3, 'km')} (target after tof)",
                f"transfer angle Δν = {_deg(dnu)}",
                "solve tof = (χ³S(z) + A√y(z))/√μ for z;  A = sin Δν √(r₁r₂/(1 − cos Δν))",
                "v₁ = (r₂ − f r₁)/g,  v₂ = (ġ r₂ − r₁)/g with f = 1 − y/r₁, g = A√(y/μ), ġ = 1 − y/r₂")
        ex.step("Transfer arc", f"a = {_km(arc.a)}, e = {arc.ecc:.6f}, periapsis altitude "
                                f"{_km(arc.periapsis_altitude)}")
        ex.step("Burns", f"Δv₁ = |v₁ − v_chaser| = {_kms(b1.magnitude)}",
                f"Δv₂ = |v_target − v₂| = {_kms(b2.magnitude)}")
    elif t.kind == "phasing":
        r = float(np.linalg.norm(t.initial.r))
        ph = t.orbits[0]
        k = round(t.tof / ph.period)
        n = math.sqrt(mu / r**3)
        dtheta = 2 * math.pi * k - n * t.tof
        ex = Explanation(f"Phasing: {math.degrees(dtheta):+.4f}° in {k} revolution(s)")
        ex.step("Phasing orbit (Curtis section 6.6)",
                f"T = 2π/n = {2 * math.pi / n / 60:.3f} min (circular orbit, r = {_km(r)})",
                f"T_ph = (2πk − Δθ)/(n k) = {ph.period / 60:.3f} min",
                f"a_ph = (μ T_ph²/4π²)^(1/3) = {_km(ph.a)};  other apsis {_km(2 * ph.a - r)}")
        v_c, v_ph = _tb.circular_velocity(r, mu), _tb.vis_viva(r, ph.a, mu)
        ex.step("Burns", f"v_c = {_kms(v_c)},  v_ph = √(μ(2/r − 1/a_ph)) = {_kms(v_ph)}",
                f"Δv = 2|v_ph − v_c| = {_kms(2 * abs(v_ph - v_c))} (enter and leave)",
                f"duration k·T_ph = {t.tof / 3600:.4f} h")
        if abs(2 * abs(v_ph - v_c) - t.total_dv) > 1e-6 * max(t.total_dv, 1.0):
            ex.warnings.append("The formula Δv differs from the state-vector value.")
    else:
        b = t.burns[0]
        after = t.final
        ex = Explanation(f"Deorbit: periapsis altitude {_km(after.periapsis_altitude)}")
        r_a, r_p = float(np.linalg.norm(b.r)), after.r_periapsis
        ex.step("Burn at apoapsis",
                f"r_a = {_km(r_a)},  target r_p = {_km(r_p)}",
                f"v after = √(μ(2/r_a − 2/(r_a + r_p))) = {_kms(_tb.vis_viva(r_a, 0.5 * (r_a + r_p), mu))}",
                f"Δv = {_kms(b.magnitude)} (retrograde)",
                f"time from the burn to periapsis: {after.period / 2 / 60:.1f} min")
    ex.step("Total", f"{_kms(t.total_dv)}")
    ex.assumptions.extend(t.assumptions)
    return ex


def explain_flyby(f) -> Explanation:
    """Flyby turn angle, periapsis and burn (Curtis section 8.9)."""
    from .core import flyby as _fb

    mu = f.body.mu
    vi, vo = float(np.linalg.norm(f.v_inf_in)), float(np.linalg.norm(f.v_inf_out))
    ex = Explanation(f"Flyby of {f.body.name}")
    lines = [f"|v∞_in| = {_kms(vi)},  |v∞_out| = {_kms(vo)}",
             f"periapsis r_p = {_km(f.r_p)} (altitude {_km(f.altitude)})",
             f"turn angle δ = {_deg(f.turn_angle)}"]
    if f.dv > 0:
        lines += ["powered flyby: asin(1/e_in) + asin(1/e_out) = δ with e = 1 + r_p v∞²/μ",
                  f"burn at periapsis |√(v∞_out² + 2μ/r_p) − √(v∞_in² + 2μ/r_p)| = {_kms(f.dv)}"]
    else:
        e = 1 + f.r_p * vi * vi / mu
        lines += [f"e = 1 + r_p v∞²/μ = {e:.5f},  δ = 2 asin(1/e) = {_deg(_fb.turn_angle(vi, f.r_p, mu))}"]
    ex.step("Hyperbola", *lines)
    if f.speed_gain is not None:
        v_b = float(np.linalg.norm(f.v_body))
        ex.step("Heliocentric effect",
                f"planet speed {_kms(v_b)}",
                f"spacecraft speed {_kms(np.linalg.norm(f.v_body + f.v_inf_in))} → "
                f"{_kms(np.linalg.norm(f.v_body + f.v_inf_out))} ({f.speed_gain / 1e3:+.4f} km/s)")
    if not f.feasible:
        ex.warnings.append(f"The periapsis altitude {_km(f.altitude)} is below the allowed "
                           f"{_km(f.min_altitude)}: this flyby cannot provide the turn")
    ex.assumptions += ["patched conics: the flyby is instantaneous on the heliocentric scale",
                       "planet treated as a point mass"]
    return ex


def explain_mission(m) -> Explanation:
    """Lambert legs, flybys and burns of an interplanetary mission."""
    path = " → ".join(b.name for b in m.sequence)
    ex = Explanation(f"Mission {path}: {m.departure.iso[:10]} → {m.arrival.iso[:10]} ({m.tof / DAY:.1f} days)")
    first = m.sequence[0]
    r_park = first.radius + m.parking_altitude
    ex.step(f"Departure from {first.name}",
            f"v∞ = |v₁ − V_{first.name}| = {_kms(m.v_inf_departure)},  C3 = {m.c3 / 1e6:.3f} km²/s²",
            f"Δv = √(v∞² + 2μ/r_p) − √(μ/r_p) = {_kms(m.dv_departure)}  (r_p = {_km(r_park)})")
    for k, leg in enumerate(m.legs):
        lines = [f"{leg.departure.iso[:10]} → {leg.arrival.iso[:10]}, tof {leg.tof / DAY:.1f} days, "
                 f"Δν = {_deg(leg.transfer_angle)}" + (f", {leg.revolutions} rev" if leg.revolutions else ""),
                 f"v∞ out {_kms(np.linalg.norm(leg.v_inf_out))}, v∞ in {_kms(np.linalg.norm(leg.v_inf_in))}"]
        ex.step(f"Leg {k + 1}: {leg.origin.name} → {leg.target.name} (Lambert arc around the Sun)", *lines)
        if k < len(m.flybys):
            f = m.flybys[k]
            fx = explain_flyby(f)
            ex.step(f"Flyby of {f.body.name}", *[x for _, lines in fx.steps for x in lines])
            ex.warnings.extend(fx.warnings)
    last = m.sequence[-1]
    if m.capture_altitude is None:
        ex.step(f"Arrival at {last.name}", f"v∞ = {_kms(m.v_inf_arrival)}; no capture burn")
    else:
        ex.step(f"Capture at {last.name}",
                f"v∞ = {_kms(m.v_inf_arrival)}, periapsis altitude {_km(m.capture_altitude)}, "
                f"e = {m.capture_ecc:g}",
                f"Δv = √(v∞² + 2μ/r_p) − √(μ(1 + e)/r_p) = {_kms(m.dv_arrival)}")
    parts = [_kms(m.dv_departure)] + [_kms(f.dv) for f in m.flybys] + [_kms(m.dv_arrival)]
    ex.step("Total", " + ".join(parts) + f" = {_kms(m.total_dv)}")
    ex.assumptions.extend(m.assumptions)
    return ex

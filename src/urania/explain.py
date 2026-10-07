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
        head = a[tr.t <= tr.t[0] + T]
        tail = a[tr.t >= tr.t[-1] - T]
        if tr.t[-1] - tr.t[0] >= 2 * T:
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

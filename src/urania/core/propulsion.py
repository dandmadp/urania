"""Rocket equation. SI units. Curtis section 6.1 (eq. 6.1), Tsiolkovsky.

Δv = Isp g0 ln(m0 / mf), with the standard gravity g0 converting specific impulse in seconds to exhaust velocity.
"""

import math

G0 = 9.80665  # standard gravity [m/s²]


def exhaust_velocity(isp: float) -> float:
    """Effective exhaust velocity Isp·g0 [m/s] for a specific impulse in seconds."""
    return isp * G0


def mass_ratio(dv: float, isp: float) -> float:
    """m0/mf = exp(Δv / (Isp g0))."""
    if not isp > 0.0:
        raise ValueError(f"Specific impulse must be positive: {isp}")
    if dv < 0.0:
        raise ValueError(f"Δv must be ≥ 0: {dv}")
    return math.exp(dv / exhaust_velocity(isp))


def propellant_mass(dv: float, isp: float, m0: float) -> float:
    """Propellant [kg] to give Δv to a vehicle of initial (wet) mass m0: m0 (1 - exp(-Δv/(Isp g0)))."""
    if not m0 > 0.0:
        raise ValueError(f"Mass must be positive: {m0}")
    return m0 * (1.0 - 1.0 / mass_ratio(dv, isp))


def delta_v(isp: float, m0: float, mf: float) -> float:
    """Δv [m/s] from initial and final mass: Isp g0 ln(m0/mf)."""
    if not 0.0 < mf <= m0:
        raise ValueError(f"Need 0 < final mass ≤ initial mass: {mf}, {m0}")
    return exhaust_velocity(isp) * math.log(m0 / mf)

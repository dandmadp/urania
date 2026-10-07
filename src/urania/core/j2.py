"""Secular rates of orbital elements due to J2 (oblateness). Curtis sec. 4.7, Vallado sec. 9.6.

Secular rates are orbit averages, so they are defined only for elliptic orbits (0 <= e < 1, a > 0).
"""

import math


def _mean_motion_and_p(a: float, ecc: float, mu: float) -> tuple[float, float]:
    if a <= 0.0 or not 0.0 <= ecc < 1.0:
        raise ValueError(f"J2 secular rates are defined only for elliptic orbits: a={a}, e={ecc}")
    return math.sqrt(mu / a**3), a * (1.0 - ecc**2)


def raan_rate(a: float, ecc: float, inc: float, mu: float, R: float, J2: float) -> float:
    """Mean rate of the right ascension of the ascending node dΩ/dt [rad/s].

    dΩ/dt = -(3/2) n J2 (R/p)² cos i
    """
    n, p = _mean_motion_and_p(a, ecc, mu)
    return -1.5 * n * J2 * (R / p) ** 2 * math.cos(inc)


def argp_rate(a: float, ecc: float, inc: float, mu: float, R: float, J2: float) -> float:
    """Mean rate of the argument of periapsis dω/dt [rad/s].

    dω/dt = (3/4) n J2 (R/p)² (5cos²i - 1)
    """
    n, p = _mean_motion_and_p(a, ecc, mu)
    return 0.75 * n * J2 * (R / p) ** 2 * (5.0 * math.cos(inc) ** 2 - 1.0)


def sso_inclination(a: float, ecc: float, mu: float, R: float, J2: float,
                    target_rate: float) -> float:
    """Inclination [rad] that gives a nodal rate of target_rate [rad/s] (for sun-synchronous orbits).

    The raan_rate formula solved for cos i.
    """
    n, p = _mean_motion_and_p(a, ecc, mu)
    cos_i = -target_rate / (1.5 * n * J2 * (R / p) ** 2)
    if abs(cos_i) > 1.0:
        raise ValueError(f"The requested nodal rate is not achievable at a={a} m")
    return math.acos(cos_i)

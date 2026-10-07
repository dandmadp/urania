"""J2(편평도)에 의한 궤도 요소의 장기(secular) 변화율. Curtis 4.7절, Vallado 9.6절."""

import math


def raan_rate(a: float, ecc: float, inc: float, mu: float, R: float, J2: float) -> float:
    """승교점 적경의 평균 변화율 dΩ/dt [rad/s].

    dΩ/dt = -(3/2) n J2 (R/p)² cos i
    """
    n = math.sqrt(mu / a**3)
    p = a * (1.0 - ecc**2)
    return -1.5 * n * J2 * (R / p) ** 2 * math.cos(inc)


def argp_rate(a: float, ecc: float, inc: float, mu: float, R: float, J2: float) -> float:
    """근지점 인수의 평균 변화율 dω/dt [rad/s].

    dω/dt = (3/4) n J2 (R/p)² (5cos²i - 1)
    """
    n = math.sqrt(mu / a**3)
    p = a * (1.0 - ecc**2)
    return 0.75 * n * J2 * (R / p) ** 2 * (5.0 * math.cos(inc) ** 2 - 1.0)


def sso_inclination(a: float, ecc: float, mu: float, R: float, J2: float,
                    target_rate: float) -> float:
    """승교점 변화율이 target_rate [rad/s]가 되는 경사각 [rad] (태양동기궤도용).

    raan_rate 식을 cos i 에 대해 푼 것.
    """
    n = math.sqrt(mu / a**3)
    p = a * (1.0 - ecc**2)
    cos_i = -target_rate / (1.5 * n * J2 * (R / p) ** 2)
    if abs(cos_i) > 1.0:
        raise ValueError(f"a={a} m 에서는 요구한 승교점 변화율을 만들 수 없습니다")
    return math.acos(cos_i)

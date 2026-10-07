"""J2(편평도)에 의한 궤도 요소의 장기(secular) 변화율. Curtis 4.7절, Vallado 9.6절.

장기 변화율은 한 바퀴 평균이므로 타원 궤도(0 <= e < 1, a > 0)에서만 정의된다.
"""

import math


def _mean_motion_and_p(a: float, ecc: float, mu: float) -> tuple[float, float]:
    if a <= 0.0 or not 0.0 <= ecc < 1.0:
        raise ValueError(f"J2 장기 변화율은 타원 궤도에서만 정의됩니다: a={a}, e={ecc}")
    return math.sqrt(mu / a**3), a * (1.0 - ecc**2)


def raan_rate(a: float, ecc: float, inc: float, mu: float, R: float, J2: float) -> float:
    """승교점 적경의 평균 변화율 dΩ/dt [rad/s].

    dΩ/dt = -(3/2) n J2 (R/p)² cos i
    """
    n, p = _mean_motion_and_p(a, ecc, mu)
    return -1.5 * n * J2 * (R / p) ** 2 * math.cos(inc)


def argp_rate(a: float, ecc: float, inc: float, mu: float, R: float, J2: float) -> float:
    """근지점 인수의 평균 변화율 dω/dt [rad/s].

    dω/dt = (3/4) n J2 (R/p)² (5cos²i - 1)
    """
    n, p = _mean_motion_and_p(a, ecc, mu)
    return 0.75 * n * J2 * (R / p) ** 2 * (5.0 * math.cos(inc) ** 2 - 1.0)


def sso_inclination(a: float, ecc: float, mu: float, R: float, J2: float,
                    target_rate: float) -> float:
    """승교점 변화율이 target_rate [rad/s]가 되는 경사각 [rad] (태양동기궤도용).

    raan_rate 식을 cos i 에 대해 푼 것.
    """
    n, p = _mean_motion_and_p(a, ecc, mu)
    cos_i = -target_rate / (1.5 * n * J2 * (R / p) ** 2)
    if abs(cos_i) > 1.0:
        raise ValueError(f"a={a} m 에서는 요구한 승교점 변화율을 만들 수 없습니다")
    return math.acos(cos_i)

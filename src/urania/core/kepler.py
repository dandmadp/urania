"""케플러 방정식과 근점이각(anomaly) 변환.

모든 각도는 rad. 이심률 e로 궤도 종류를 구분한다:
타원 0 <= e < 1, 포물선 e == 1, 쌍곡선 e > 1.

포물선의 "평균근점이각"은 Barker 방정식 M = D + D³/3 (D = tan(ν/2))의 M을 뜻한다.
"""

import math

TWO_PI = 2.0 * math.pi

# 포물선으로 간주하는 이심률 허용오차
PARABOLIC_TOL = 1e-12

_NEWTON_TOL = 1e-14
_NEWTON_MAXITER = 50


def _wrap_2pi(angle: float) -> float:
    """각도를 [0, 2π) 범위로 정규화."""
    wrapped = angle % TWO_PI
    # 아주 작은 음수는 부동소수점 반올림으로 정확히 2π가 될 수 있다
    return 0.0 if wrapped == TWO_PI else wrapped


def _wrap_pi(angle: float) -> float:
    """각도를 [-π, π) 범위로 정규화."""
    return (angle + math.pi) % TWO_PI - math.pi


def _check_ecc(e: float) -> None:
    if e < 0.0:
        raise ValueError(f"이심률은 0 이상이어야 합니다: e={e}")


# ---------------------------------------------------------------- 타원 (e < 1)

def eccentric_to_mean(E: float, e: float) -> float:
    """이심근점이각 E → 평균근점이각 M (케플러 방정식 M = E - e sin E)."""
    return E - e * math.sin(E)


def mean_to_eccentric(M: float, e: float) -> float:
    """평균근점이각 M → 이심근점이각 E. 케플러 방정식을 뉴턴법으로 푼다.

    결과는 [0, 2π) 범위.
    """
    _check_ecc(e)
    if e >= 1.0:
        raise ValueError(f"타원 궤도(e < 1)만 가능합니다: e={e}")
    M = _wrap_pi(M)
    # e가 크면 M 근처 초기값이 발산할 수 있어 π 쪽에서 출발 (Vallado 알고리즘 2)
    E = M + e if M >= 0.0 else M - e
    if e > 0.8:
        E = math.pi if M >= 0.0 else -math.pi
    for _ in range(_NEWTON_MAXITER):
        dE = (E - e * math.sin(E) - M) / (1.0 - e * math.cos(E))
        E -= dE
        if abs(dE) < _NEWTON_TOL:
            return _wrap_2pi(E)
    raise RuntimeError(f"케플러 방정식이 수렴하지 않았습니다: M={M}, e={e}")


def eccentric_to_true(E: float, e: float) -> float:
    """이심근점이각 E → 진근점이각 ν. 결과는 [0, 2π)."""
    nu = 2.0 * math.atan2(math.sqrt(1.0 + e) * math.sin(E / 2.0),
                          math.sqrt(1.0 - e) * math.cos(E / 2.0))
    return _wrap_2pi(nu)


def true_to_eccentric(nu: float, e: float) -> float:
    """진근점이각 ν → 이심근점이각 E. 결과는 [0, 2π)."""
    E = 2.0 * math.atan2(math.sqrt(1.0 - e) * math.sin(nu / 2.0),
                         math.sqrt(1.0 + e) * math.cos(nu / 2.0))
    return _wrap_2pi(E)


# -------------------------------------------------------------- 쌍곡선 (e > 1)

def hyperbolic_to_mean(H: float, e: float) -> float:
    """쌍곡선 근점이각 H → 평균근점이각 M (M = e sinh H - H)."""
    return e * math.sinh(H) - H


def mean_to_hyperbolic(M: float, e: float) -> float:
    """평균근점이각 M → 쌍곡선 근점이각 H. 뉴턴법으로 푼다."""
    if e <= 1.0:
        raise ValueError(f"쌍곡선 궤도(e > 1)만 가능합니다: e={e}")
    # 큰 |M|에서는 e sinh H ≈ M 이므로 로그 근사로 출발
    H = math.copysign(math.log(2.0 * abs(M) / e + 1.8), M) if M != 0.0 else 0.0
    for _ in range(_NEWTON_MAXITER):
        dH = (e * math.sinh(H) - H - M) / (e * math.cosh(H) - 1.0)
        H -= dH
        if abs(dH) < _NEWTON_TOL * max(1.0, abs(H)):
            return H
    raise RuntimeError(f"쌍곡선 케플러 방정식이 수렴하지 않았습니다: M={M}, e={e}")


def hyperbolic_to_true(H: float, e: float) -> float:
    """쌍곡선 근점이각 H → 진근점이각 ν. 결과는 (-π, π)."""
    return 2.0 * math.atan(math.sqrt((e + 1.0) / (e - 1.0)) * math.tanh(H / 2.0))


def true_to_hyperbolic(nu: float, e: float) -> float:
    """진근점이각 ν → 쌍곡선 근점이각 H.

    ν는 점근선 각도 ±arccos(-1/e) 안쪽이어야 한다.
    """
    nu = _wrap_pi(nu)
    nu_inf = math.acos(-1.0 / e)
    if abs(nu) >= nu_inf:
        raise ValueError(f"ν={nu} rad 는 쌍곡선 점근선(±{nu_inf} rad) 밖입니다")
    return 2.0 * math.atanh(math.sqrt((e - 1.0) / (e + 1.0)) * math.tan(nu / 2.0))


# ------------------------------------------------------------- 포물선 (e == 1)

def parabolic_to_mean(D: float) -> float:
    """포물선 근점이각 D = tan(ν/2) → Barker 평균근점이각 M = D + D³/3."""
    return D + D**3 / 3.0


def mean_to_parabolic(M: float) -> float:
    """Barker 방정식 D + D³/3 = M 을 닫힌 형태로 푼다."""
    # 3차 방정식 D³ + 3D - 3M = 0 의 유일한 실근 (Cardano): D = ∛(w+s) + ∛(w-s).
    # (w+s)(w-s) = -1 이므로 ∛(w-s) = -1/∛(w+s). |w|가 크면 w-s 직접 계산은 상쇄 오차가 크다.
    w = 1.5 * abs(M)
    c = math.cbrt(w + math.sqrt(w * w + 1.0))
    return math.copysign(c - 1.0 / c, M)


# ------------------------------------------------------------- 종류 자동 판별

def mean_to_true(M: float, e: float) -> float:
    """평균근점이각 M → 진근점이각 ν. 이심률에 따라 타원/포물선/쌍곡선 처리.

    타원은 [0, 2π), 포물선·쌍곡선은 (-π, π) 범위로 돌려준다.
    """
    _check_ecc(e)
    if abs(e - 1.0) < PARABOLIC_TOL:
        return 2.0 * math.atan(mean_to_parabolic(M))
    if e < 1.0:
        return eccentric_to_true(mean_to_eccentric(M, e), e)
    return hyperbolic_to_true(mean_to_hyperbolic(M, e), e)


def true_to_mean(nu: float, e: float) -> float:
    """진근점이각 ν → 평균근점이각 M.

    타원은 [0, 2π), 포물선·쌍곡선은 부호 있는 값을 돌려준다.
    """
    _check_ecc(e)
    if abs(e - 1.0) < PARABOLIC_TOL:
        return parabolic_to_mean(math.tan(_wrap_pi(nu) / 2.0))
    if e < 1.0:
        return _wrap_2pi(eccentric_to_mean(true_to_eccentric(nu, e), e))
    return hyperbolic_to_mean(true_to_hyperbolic(nu, e), e)

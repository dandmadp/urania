"""임펄스 기동 공식. 모든 입력·출력은 SI. Curtis 6장, Vallado 6장.

원 궤도 사이의 전이를 다룬다. Δv는 부호 있는 접선 방향 성분이다 (+는 진행 방향 가속).
"""

import math

import numpy as np
from scipy.optimize import minimize_scalar


def _vis_viva(r: float, a: float, mu: float) -> float:
    """반지름 r에서 장반경 a 궤도의 속도 v = √(μ(2/r - 1/a))."""
    return math.sqrt(mu * (2.0 / r - 1.0 / a))


def hohmann(r1: float, r2: float, mu: float) -> tuple[float, float, float]:
    """원 궤도 r1 → r2 호만 전이 (Vallado 알고리즘 36).

    Returns:
        (dv1, dv2, tof): 두 기동의 접선 Δv [m/s] (내려갈 때는 음수), 전이 시간 [s]
    """
    a_t = 0.5 * (r1 + r2)
    dv1 = _vis_viva(r1, a_t, mu) - math.sqrt(mu / r1)
    dv2 = math.sqrt(mu / r2) - _vis_viva(r2, a_t, mu)
    tof = math.pi * math.sqrt(a_t**3 / mu)
    return dv1, dv2, tof


def bielliptic(r1: float, rb: float, r2: float, mu: float) -> tuple[float, float, float, float]:
    """원 궤도 r1 → rb(중간 원점) → r2 이중타원 전이 (Vallado 알고리즘 37).

    Returns:
        (dv1, dv2, dv3, tof): 세 기동의 접선 Δv [m/s], 전이 시간 [s]
    """
    if rb < max(r1, r2):
        raise ValueError(f"중간 원점 rb={rb}는 r1, r2보다 커야 합니다")
    a1 = 0.5 * (r1 + rb)
    a2 = 0.5 * (rb + r2)
    dv1 = _vis_viva(r1, a1, mu) - math.sqrt(mu / r1)
    dv2 = _vis_viva(rb, a2, mu) - _vis_viva(rb, a1, mu)
    dv3 = math.sqrt(mu / r2) - _vis_viva(r2, a2, mu)
    tof = math.pi * (math.sqrt(a1**3 / mu) + math.sqrt(a2**3 / mu))
    return dv1, dv2, dv3, tof


def plane_change_dv(v: float, dtheta: float) -> float:
    """속도 크기를 유지한 채 궤도면을 dtheta [rad] 돌리는 Δv = 2v sin(Δθ/2)."""
    return 2.0 * v * math.sin(0.5 * abs(dtheta))


def combined_dv(v1: float, v2: float, dtheta: float) -> float:
    """속도 크기 변경과 궤도면 변경을 한 번에 하는 Δv (코사인 법칙)."""
    return math.sqrt(max(v1 * v1 + v2 * v2 - 2.0 * v1 * v2 * math.cos(dtheta), 0.0))


def hohmann_plane_change(r1: float, r2: float, dtheta: float, mu: float,
                         split: float) -> tuple[float, float, float]:
    """궤도면 변경을 포함한 호만 전이.

    Args:
        dtheta: 두 궤도면 사이 각도 [rad]
        split: 첫 기동에서 수행하는 궤도면 변경 비율 (0~1). 나머지는 두 번째 기동에서.

    Returns:
        (dv1, dv2, tof): 두 기동의 Δv 크기 [m/s], 전이 시간 [s]
    """
    a_t = 0.5 * (r1 + r2)
    dv1 = combined_dv(math.sqrt(mu / r1), _vis_viva(r1, a_t, mu), split * dtheta)
    dv2 = combined_dv(_vis_viva(r2, a_t, mu), math.sqrt(mu / r2), (1.0 - split) * dtheta)
    return dv1, dv2, math.pi * math.sqrt(a_t**3 / mu)


def optimal_plane_split(r1: float, r2: float, dtheta: float, mu: float) -> float:
    """총 Δv가 최소가 되는 궤도면 변경 분배 비율 (0~1)."""
    if dtheta == 0.0:
        return 0.0
    res = minimize_scalar(lambda s: sum(hohmann_plane_change(r1, r2, dtheta, mu, s)[:2]),
                          bounds=(0.0, 1.0), method="bounded", options={"xatol": 1e-10})
    return float(res.x)


def rotate(vec, axis, angle: float) -> np.ndarray:
    """벡터를 단위 축 axis 기준으로 angle [rad] 회전 (로드리게스 공식)."""
    vec = np.asarray(vec, dtype=float)
    k = np.asarray(axis, dtype=float)
    c, s = math.cos(angle), math.sin(angle)
    return vec * c + np.cross(k, vec) * s + k * (k @ vec) * (1.0 - c)

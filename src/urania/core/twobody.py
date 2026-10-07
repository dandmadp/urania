"""2체 문제의 기본 물리량. 모든 입력·출력은 SI."""

import math

import numpy as np


def period(a: float, mu: float) -> float:
    """공전 주기 [s]. 타원 궤도(a > 0)만."""
    if a <= 0.0:
        raise ValueError(f"주기는 타원 궤도(a > 0)에서만 정의됩니다: a={a}")
    return 2.0 * math.pi * math.sqrt(a**3 / mu)


def mean_motion(a: float, mu: float) -> float:
    """평균 운동 n [rad/s]. 쌍곡선(a < 0)은 |a|를 쓴다."""
    return math.sqrt(mu / abs(a) ** 3)


def specific_energy(a: float, mu: float) -> float:
    """비역학적 에너지 ε = -μ/(2a) [J/kg]. 포물선(a = inf)이면 0."""
    return -mu / (2.0 * a)


def vis_viva(r: float, a: float, mu: float) -> float:
    """반지름 r에서 장반경 a 궤도의 속도 [m/s]: v = √(μ(2/r - 1/a))."""
    return math.sqrt(mu * (2.0 / r - 1.0 / a))


def circular_velocity(r: float, mu: float) -> float:
    """반지름 r에서의 원 궤도 속도 [m/s]."""
    return math.sqrt(mu / r)


def escape_velocity(r: float, mu: float) -> float:
    """반지름 r에서의 탈출 속도 [m/s]."""
    return math.sqrt(2.0 * mu / r)


def synchronous_radius(mu: float, rotation_rate: float) -> float:
    """천체 자전과 주기가 같은 원 궤도 반지름 [m] (지구면 GEO)."""
    return (mu / rotation_rate**2) ** (1.0 / 3.0)


def semi_major_axis(r, v, mu: float):
    """상태벡터에서 접촉 장반경 [m]: a = 1 / (2/|r| - |v|²/μ) (활력 방정식).

    r, v가 N×3 배열이면 길이 N 배열을 돌려준다.
    """
    r = np.asarray(r, dtype=float)
    v = np.asarray(v, dtype=float)
    r_norm = np.linalg.norm(r, axis=-1)
    v2 = np.sum(v * v, axis=-1)
    return 1.0 / (2.0 / r_norm - v2 / mu)

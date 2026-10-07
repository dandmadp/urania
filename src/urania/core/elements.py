"""상태벡터(r, v) ↔ 고전 궤도 요소 변환.

모든 입력·출력은 SI(m, m/s, m³/s², rad).

궤도 크기는 장반경 a 대신 반통경(semi-latus rectum) p로 저장한다.
p는 포물선(a = ∞)을 포함한 모든 원뿔곡선에서 유한하기 때문이다.

특이 궤도의 각도 규약 (Vallado 방식):
- 적도 궤도 (i ≈ 0 또는 π): 승교점이 정의되지 않으므로 raan = 0, 기준축은 x축
- 원 궤도 (e ≈ 0): 근지점이 정의되지 않으므로 argp = 0, ν는 승교점(적도면이면 x축)부터 잰다
이 규약 덕분에 coe_to_rv(rv_to_coe(r, v))는 특이 궤도에서도 원래 상태벡터를 돌려준다.
"""

import math
from typing import NamedTuple

import numpy as np

from .kepler import PARABOLIC_TOL, _wrap_2pi

# 원 궤도 / 적도 궤도로 간주하는 허용오차 (무차원)
CIRCULAR_TOL = 1e-11
EQUATORIAL_TOL = 1e-11


class Elements(NamedTuple):
    """고전 궤도 요소 (SI, rad)."""

    p: float      # 반통경 [m]
    ecc: float    # 이심률 [-]
    inc: float    # 경사각 [rad], 0 ~ π
    raan: float   # 승교점 적경 [rad], 0 ~ 2π
    argp: float   # 근지점 인수 [rad], 0 ~ 2π
    nu: float     # 진근점이각 [rad]

    @property
    def a(self) -> float:
        """장반경 [m]. 포물선이면 inf, 쌍곡선이면 음수."""
        if abs(self.ecc - 1.0) < PARABOLIC_TOL:
            return math.inf
        return self.p / (1.0 - self.ecc**2)


def _angle_between(ref: np.ndarray, vec: np.ndarray, normal: np.ndarray) -> float:
    """ref에서 vec까지 normal 축 기준 반시계 각도. 결과는 [0, 2π)."""
    y = np.dot(np.cross(ref, vec), normal)
    x = np.dot(ref, vec)
    return _wrap_2pi(math.atan2(y, x))


def rv_to_coe(r, v, mu: float) -> Elements:
    """위치·속도 벡터 → 고전 궤도 요소.

    Args:
        r: 위치 벡터 [m], 길이 3
        v: 속도 벡터 [m/s], 길이 3
        mu: 중심천체 중력상수 [m³/s²]
    """
    r = np.asarray(r, dtype=float)
    v = np.asarray(v, dtype=float)
    r_norm = np.linalg.norm(r)
    v_norm = np.linalg.norm(v)

    h_vec = np.cross(r, v)
    h_norm = np.linalg.norm(h_vec)
    if h_norm == 0.0:
        raise ValueError("각운동량이 0입니다 (직선 궤도는 지원하지 않음)")
    h_hat = h_vec / h_norm

    n_vec = np.cross([0.0, 0.0, 1.0], h_vec)  # 승교점 방향
    n_norm = np.linalg.norm(n_vec)

    e_vec = ((v_norm**2 - mu / r_norm) * r - np.dot(r, v) * v) / mu
    ecc = float(np.linalg.norm(e_vec))

    p = h_norm**2 / mu
    inc = math.acos(np.clip(h_vec[2] / h_norm, -1.0, 1.0))

    equatorial = n_norm < EQUATORIAL_TOL * h_norm
    circular = ecc < CIRCULAR_TOL

    n_hat = np.array([1.0, 0.0, 0.0]) if equatorial else n_vec / n_norm
    raan = 0.0 if equatorial else _wrap_2pi(math.atan2(n_hat[1], n_hat[0]))

    if circular:
        argp = 0.0
        nu = _angle_between(n_hat, r, h_hat)
    else:
        argp = _angle_between(n_hat, e_vec, h_hat)
        nu = _angle_between(e_vec, r, h_hat)

    return Elements(p=p, ecc=ecc, inc=inc, raan=raan, argp=argp, nu=nu)


def _rot_perifocal_to_inertial(inc: float, raan: float, argp: float) -> np.ndarray:
    """근점 좌표계(PQW) → 관성 좌표계 회전 행렬 R3(Ω)·R1(i)·R3(ω)."""
    cO, sO = math.cos(raan), math.sin(raan)
    ci, si = math.cos(inc), math.sin(inc)
    cw, sw = math.cos(argp), math.sin(argp)
    return np.array([
        [cO * cw - sO * sw * ci, -cO * sw - sO * cw * ci,  sO * si],
        [sO * cw + cO * sw * ci, -sO * sw + cO * cw * ci, -cO * si],
        [sw * si,                 cw * si,                 ci],
    ])


def coe_to_rv(p: float, ecc: float, inc: float, raan: float, argp: float,
              nu: float, mu: float) -> tuple[np.ndarray, np.ndarray]:
    """고전 궤도 요소 → 위치·속도 벡터 (r [m], v [m/s])."""
    if p <= 0.0:
        raise ValueError(f"반통경은 양수여야 합니다: p={p}")
    denom = 1.0 + ecc * math.cos(nu)
    if denom <= 0.0:
        raise ValueError(f"ν={nu} rad 는 이 궤도(e={ecc})에서 도달할 수 없는 위치입니다")

    cnu, snu = math.cos(nu), math.sin(nu)
    r_pqw = (p / denom) * np.array([cnu, snu, 0.0])
    v_pqw = math.sqrt(mu / p) * np.array([-snu, ecc + cnu, 0.0])

    Q = _rot_perifocal_to_inertial(inc, raan, argp)
    return Q @ r_pqw, Q @ v_pqw


def elements_to_rv(el: Elements, mu: float) -> tuple[np.ndarray, np.ndarray]:
    """Elements 튜플 → 위치·속도 벡터."""
    return coe_to_rv(*el, mu=mu)

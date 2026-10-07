"""섭동 가속도 모델. 모든 입력·출력은 SI, 관성 좌표계."""

import numpy as np


def accel_twobody(r: np.ndarray, mu: float) -> np.ndarray:
    """중심천체 질점 중력 a = -μ r / |r|³."""
    r_norm = np.sqrt(r @ r)
    return -mu * r / r_norm**3


def accel_j2(r: np.ndarray, mu: float, R: float, J2: float) -> np.ndarray:
    """J2 편평도 섭동 가속도 (Curtis 식 12.30, Vallado 식 8-30).

    천체 자전축이 관성 z축과 같다고 가정한다.
    """
    x, y, z = r
    r2 = r @ r
    r_norm = np.sqrt(r2)
    factor = -1.5 * J2 * mu * R**2 / r_norm**5
    k = 5.0 * z * z / r2
    return factor * np.array([x * (1.0 - k), y * (1.0 - k), z * (3.0 - k)])


def accel_drag(r: np.ndarray, v: np.ndarray, rho: float, ballistic: float,
               rotation_rate: float) -> np.ndarray:
    """대기 항력 가속도 a = -½ ρ (C_D A / m) |v_rel| v_rel (Vallado 식 8-28).

    Args:
        rho: 대기 밀도 [kg/m³]
        ballistic: C_D·A/m [m²/kg]
        rotation_rate: 천체 자전 각속도 [rad/s]. 대기는 z축 기준으로 함께 강체 자전한다고 가정.
    """
    v_rel = v - rotation_rate * np.array([-r[1], r[0], 0.0])  # v - ω×r
    return -0.5 * rho * ballistic * np.sqrt(v_rel @ v_rel) * v_rel

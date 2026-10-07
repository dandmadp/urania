"""대기 밀도 모델 (Environment 구현)."""

import numpy as np

from .core.atmosphere import exponential_density
from .environment import Environment


class ExponentialAtmosphere(Environment):
    """Vallado 표 8-4 지수 대기 모델. 고도는 구형 천체 기준 |r| - R."""

    description = "지수 대기 모델 (Vallado 표 8-4, 구형 지구 고도)"

    def __init__(self, radius: float):
        self.radius = radius

    def __call__(self, r, t: float) -> float:
        return exponential_density(np.sqrt(r @ r) - self.radius)

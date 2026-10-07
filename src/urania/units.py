"""단위 변환 경계.

라이브러리 내부는 SI 숫자(m, kg, s, K, W, rad)만 다룬다.
사용자 입력은 이 모듈을 거쳐 SI 숫자가 되고, 출력은 요청할 때만 단위가 붙는다.

- 입력이 astropy Quantity면 지정한 SI 단위로 변환한다.
- 입력이 맨 숫자(또는 리스트·배열)면 이미 SI라고 간주한다. 각도도 rad로 본다.
"""

import astropy.units as u
import numpy as np

# 물리량별 내부 SI 단위
LENGTH = u.m
AREA = u.m**2
VELOCITY = u.m / u.s
ACCELERATION = u.m / u.s**2
TIME = u.s
MASS = u.kg
ANGLE = u.rad
ANGULAR_VELOCITY = u.rad / u.s
MU = u.m**3 / u.s**2        # 중력상수
DENSITY = u.kg / u.m**3
TEMPERATURE = u.K
POWER = u.W


class UnitError(ValueError):
    """입력 단위가 기대한 물리량과 맞지 않을 때."""


def to_si(value, si_unit: u.UnitBase):
    """입력값을 SI 숫자로 변환한다.

    Args:
        value: astropy Quantity, 숫자, 또는 숫자 시퀀스
        si_unit: 기대하는 물리량의 SI 단위 (예: `LENGTH`)

    Returns:
        스칼라면 float, 아니면 numpy 배열
    """
    if isinstance(value, u.Quantity):
        try:
            # 섭씨 → 켈빈처럼 오프셋이 있는 변환을 위해 온도 등가관계를 허용
            result = value.to_value(si_unit, equivalencies=u.temperature())
        except u.UnitConversionError as exc:
            raise UnitError(
                f"{value.unit}는 {si_unit} 로 변환할 수 없습니다"
            ) from exc
    else:
        result = np.asarray(value, dtype=float)
    return float(result) if np.ndim(result) == 0 else np.asarray(result, dtype=float)


def from_si(value, si_unit: u.UnitBase, unit=None):
    """SI 숫자를 출력용으로 변환한다.

    unit이 None이면 SI 숫자를 그대로 돌려주고, 지정하면 그 단위의 Quantity를 돌려준다.
    """
    if unit is None:
        return value
    return (value * si_unit).to(unit, equivalencies=u.temperature())

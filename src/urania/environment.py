"""환경 주입 인터페이스.

라이브러리는 계산의 틀만 책임지고, 환경 데이터(대기 밀도 등)는 사용자가 넣는다.
다음 세 형태를 모두 받아 내부에서는 `Environment` 하나로 다룬다.

- 상수: 숫자 또는 astropy Quantity → 항상 같은 값을 돌려주는 `Constant`
- 함수: f(r, t) → `Function`
- `Environment` 객체: 그대로 사용

r은 관성 좌표계 위치 [m] (numpy 배열), t는 J2000 기준 TDB 초다.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import astropy.units as u

from . import units


class Environment(ABC):
    """위치·시간에 따라 값을 주는 환경 모델. 반환값은 SI 숫자."""

    description: str = ""

    @abstractmethod
    def __call__(self, r, t: float) -> float: ...


class Constant(Environment):
    """항상 같은 값을 주는 환경."""

    def __init__(self, value: float, description: str = ""):
        self.value = value
        self.description = description or f"상수 {value:g}"

    def __call__(self, r, t: float) -> float:
        return self.value


class Function(Environment):
    """사용자 함수 f(r, t)를 감싼 환경. 함수가 Quantity를 돌려주면 SI로 변환한다."""

    def __init__(self, func, si_unit: u.UnitBase, description: str = ""):
        self.func = func
        self.si_unit = si_unit
        name = getattr(func, "__name__", repr(func))
        self.description = description or f"사용자 함수 {name}(r, t)"

    def __call__(self, r, t: float) -> float:
        value = self.func(r, t)
        if isinstance(value, u.Quantity):
            return units.to_si(value, self.si_unit)
        return value


def as_environment(value, si_unit: u.UnitBase) -> Environment:
    """상수·함수·Environment 중 어느 형태든 Environment로 만든다."""
    if isinstance(value, Environment):
        return value
    if callable(value):
        return Function(value, si_unit)
    return Constant(units.to_si(value, si_unit))

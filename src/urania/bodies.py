"""천체 객체와 프리셋."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from . import units
from .atmosphere import ExponentialAtmosphere
from .constants import G


@dataclass(frozen=True)
class Body:
    """중심천체. 모든 값은 SI.

    μ(GM)를 기본값으로 저장한다. 관측으로 직접 결정되는 값이라 질량보다 훨씬 정확하다.
    """

    name: str
    mu: float                   # 중력상수 GM [m³/s²]
    radius: float               # 적도 반지름 [m]
    J2: float = 0.0             # 2차 띠 조화계수 [-]
    rotation_rate: float = 0.0  # 자전 각속도 [rad/s] (항성일 기준)
    atmosphere: Any = field(default=None, compare=False)  # 대기 밀도 Environment
    source: str = field(default="", compare=False)        # 상수 출처

    @classmethod
    def create(cls, name: str, mu, radius, J2: float = 0.0, rotation_rate=0.0,
               atmosphere=None, source: str = "") -> Body:
        """단위가 붙은 값으로 생성. 맨 숫자는 SI로 간주한다."""
        return cls(
            name=name,
            mu=units.to_si(mu, units.MU),
            radius=units.to_si(radius, units.LENGTH),
            J2=J2,
            rotation_rate=units.to_si(rotation_rate, units.ANGULAR_VELOCITY),
            atmosphere=atmosphere,
            source=source,
        )

    @property
    def mass(self) -> float:
        """질량 [kg] = μ / G."""
        return self.mu / G

    def __repr__(self) -> str:
        return f"Body({self.name!r}, R={self.radius / 1e3:.1f} km)"


Sun = Body(
    name="Sun",
    mu=1.32712440041e20,
    radius=6.957e8,
    rotation_rate=2.8653e-6,  # 캐링턴 자전 주기 25.38일
    source="μ: DE440; R: IAU 2015 B3 공칭값",
)

Earth = Body(
    name="Earth",
    mu=3.986004418e14,
    radius=6378137.0,
    J2=1.08262668e-3,
    rotation_rate=7.292115e-5,
    atmosphere=ExponentialAtmosphere(radius=6378137.0),
    source="μ, R: WGS 84; J2: EGM-08 (Vallado 표 D-1)",
)

Moon = Body(
    name="Moon",
    mu=4.902800118e12,
    radius=1.7374e6,
    J2=2.033e-4,
    rotation_rate=2.6617e-6,  # 항성월 27.3217일
    source="μ: DE440; J2: GRAIL GRGM1200A",
)

Mars = Body(
    name="Mars",
    mu=4.282837e13,
    radius=3.3962e6,
    J2=1.96045e-3,
    rotation_rate=7.088218e-5,  # 항성일 24.6229시간
    source="μ: DE440; R, J2: Mars Fact Sheet (NASA GSFC)",
)

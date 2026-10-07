"""궤도 객체와 프리셋."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import cached_property

import astropy.units as u
import numpy as np

from . import units
from .bodies import Body, Earth
from .constants import DAY, SSO_RAAN_RATE
from .core import elements as _el
from .core import j2 as _j2
from .core import kepler as _kepler
from .core import twobody as _tb
from .time import J2000, Epoch


def _frozen_vector(x) -> np.ndarray:
    arr = np.array(x, dtype=float)
    if arr.shape != (3,):
        raise ValueError(f"3차원 벡터여야 합니다: shape={arr.shape}")
    arr.flags.writeable = False
    return arr


@dataclass(frozen=True, eq=False)
class Orbit:
    """중심천체 주위의 궤도. 특정 시각의 상태벡터(r, v)로 정의된다.

    궤도 요소와 파생 물리량은 상태벡터에서 core 함수로 계산한다.
    모든 속성은 SI 숫자를 돌려준다.
    """

    body: Body
    r: np.ndarray        # 위치 [m], 관성 좌표계
    v: np.ndarray        # 속도 [m/s]
    epoch: Epoch = field(default=J2000)

    def __post_init__(self):
        object.__setattr__(self, "r", _frozen_vector(self.r))
        object.__setattr__(self, "v", _frozen_vector(self.v))

    # ---------------------------------------------------------- 생성

    @classmethod
    def from_vectors(cls, body: Body, r, v, epoch: Epoch = J2000) -> Orbit:
        """상태벡터로 생성. 단위 Quantity 또는 SI 숫자."""
        return cls(body, units.to_si(r, units.LENGTH),
                   units.to_si(v, units.VELOCITY), epoch)

    @classmethod
    def from_elements(cls, body: Body, *, a=None, p=None, ecc: float = 0.0,
                      inc=0.0, raan=0.0, argp=0.0, nu=0.0,
                      epoch: Epoch = J2000) -> Orbit:
        """고전 궤도 요소로 생성. 크기는 a(장반경) 또는 p(반통경) 중 하나로 준다.

        쌍곡선은 a < 0, 포물선(ecc = 1)은 p로만 줄 수 있다.
        원·적도 궤도의 각도 규약은 core.elements 모듈 설명을 따른다.
        """
        if (a is None) == (p is None):
            raise ValueError("a와 p 중 정확히 하나를 지정하세요")
        if p is None:
            a = units.to_si(a, units.LENGTH)
            if ecc == 1.0:
                raise ValueError("포물선 궤도는 p로 지정하세요")
            if (a > 0) != (ecc < 1.0):
                raise ValueError(f"a={a} m 와 ecc={ecc} 가 맞지 않습니다 "
                                 "(타원은 a > 0, 쌍곡선은 a < 0)")
            p = a * (1.0 - ecc**2)
        else:
            p = units.to_si(p, units.LENGTH)
        r, v = _el.coe_to_rv(
            p, ecc,
            units.to_si(inc, units.ANGLE),
            units.to_si(raan, units.ANGLE),
            units.to_si(argp, units.ANGLE),
            units.to_si(nu, units.ANGLE),
            body.mu,
        )
        return cls(body, r, v, epoch)

    @classmethod
    def circular(cls, body: Body, altitude, *, inc=0.0, raan=0.0, arglat=0.0,
                 epoch: Epoch = J2000) -> Orbit:
        """원 궤도. altitude는 적도 반지름 기준 고도, arglat은 승교점부터 잰 위치."""
        a = body.radius + units.to_si(altitude, units.LENGTH)
        return cls.from_elements(body, a=a, inc=inc, raan=raan, nu=arglat, epoch=epoch)

    # ---------------------------------------------------------- 궤도 요소

    @cached_property
    def elements(self) -> _el.Elements:
        return _el.rv_to_coe(self.r, self.v, self.body.mu)

    @property
    def p(self) -> float:
        """반통경 [m]."""
        return self.elements.p

    @property
    def a(self) -> float:
        """장반경 [m]. 포물선이면 inf, 쌍곡선이면 음수."""
        return self.elements.a

    @property
    def ecc(self) -> float:
        return self.elements.ecc

    @property
    def inc(self) -> float:
        """경사각 [rad]."""
        return self.elements.inc

    @property
    def raan(self) -> float:
        """승교점 적경 [rad]."""
        return self.elements.raan

    @property
    def argp(self) -> float:
        """근지점 인수 [rad]."""
        return self.elements.argp

    @property
    def nu(self) -> float:
        """진근점이각 [rad]."""
        return self.elements.nu

    @property
    def M(self) -> float:
        """평균근점이각 [rad]."""
        return _kepler.true_to_mean(self.nu, self.ecc)

    # ---------------------------------------------------------- 파생 물리량

    @property
    def period(self) -> float:
        """공전 주기 [s]. 타원 궤도만."""
        return _tb.period(self.a, self.body.mu)

    @property
    def mean_motion(self) -> float:
        """평균 운동 [rad/s]."""
        return _tb.mean_motion(self.a, self.body.mu)

    @property
    def energy(self) -> float:
        """비역학적 에너지 [J/kg]."""
        return _tb.specific_energy(self.a, self.body.mu)

    @property
    def h(self) -> float:
        """비각운동량 크기 [m²/s]."""
        return math.sqrt(self.p * self.body.mu)

    @property
    def r_periapsis(self) -> float:
        """근점 반지름 [m]."""
        return self.p / (1.0 + self.ecc)

    @property
    def r_apoapsis(self) -> float:
        """원점 반지름 [m]. 타원이 아니면 inf."""
        if self.ecc >= 1.0:
            return math.inf
        return self.p / (1.0 - self.ecc)

    @property
    def periapsis_altitude(self) -> float:
        """근점 고도 [m] (적도 반지름 기준)."""
        return self.r_periapsis - self.body.radius

    @property
    def apoapsis_altitude(self) -> float:
        """원점 고도 [m] (적도 반지름 기준)."""
        return self.r_apoapsis - self.body.radius

    @property
    def raan_rate(self) -> float:
        """J2에 의한 승교점 평균 변화율 [rad/s]."""
        b = self.body
        return _j2.raan_rate(self.a, self.ecc, self.inc, b.mu, b.radius, b.J2)

    @property
    def argp_rate(self) -> float:
        """J2에 의한 근지점 인수 평균 변화율 [rad/s]."""
        b = self.body
        return _j2.argp_rate(self.a, self.ecc, self.inc, b.mu, b.radius, b.J2)

    # ---------------------------------------------------------- 전파

    def propagate(self, duration=None, *, days=None, model="twobody", **kwargs):
        """궤도를 전파해 Trajectory를 돌려준다. 원본은 바뀌지 않는다.

        Args:
            duration: 전파 시간 (초 또는 시간 Quantity). days와 둘 중 하나만.
            days: 전파 시간 [일]. 맨 숫자는 일, Quantity면 변환한다.
            model: "twobody", "j2", "j2+drag" 또는 충실도 0~2
            **kwargs: cd, area, mass, density(상수·함수·Environment),
                method, rtol, atol, n_points. 자세한 내용은 `propagation.propagate`.

        예: ``ISS.propagate(days=30, model="j2+drag", area=1500, mass=420000).final``
        """
        from .propagation import propagate

        if (duration is None) == (days is None):
            raise ValueError("duration과 days 중 정확히 하나를 지정하세요")
        if days is not None:
            duration = units.to_si(days, u.day) * DAY
        return propagate(self, duration, model=model, **kwargs)

    # ---------------------------------------------------------- 기동

    def transfer_to(self, target: Orbit, method: str = "hohmann", *, rb=None,
                    plane_split="optimal"):
        """원 궤도 target으로 가는 전이를 계산해 Transfer를 돌려준다.

        Args:
            method: "hohmann" 또는 "bielliptic"
            rb: 이중타원 전이의 중간 원점 반지름 (길이)
            plane_split: 궤도면이 다를 때 호만 1차 기동이 맡을 궤도면 변경 비율
                (0~1) 또는 "optimal"(총 Δv 최소)

        예: ``ISS.transfer_to(GEO).total_dv``
        """
        from .maneuvers import transfer

        return transfer(self, target, method, rb=rb, plane_split=plane_split)

    def __repr__(self) -> str:
        return (f"Orbit({self.body.name}, a={self.a / 1e3:.1f} km, "
                f"ecc={self.ecc:.4f}, inc={math.degrees(self.inc):.2f}°, "
                f"epoch={self.epoch.iso})")


def sun_synchronous(altitude, *, ecc: float = 0.0, raan=0.0, epoch: Epoch = J2000,
                    body: Body = Earth) -> Orbit:
    """태양동기궤도. 경사각은 J2 승교점 변화율이 1 회귀년에 360°가 되도록 정한다."""
    a = body.radius + units.to_si(altitude, units.LENGTH)
    inc = _j2.sso_inclination(a, ecc, body.mu, body.radius, body.J2, SSO_RAAN_RATE)
    return Orbit.from_elements(body, a=a, ecc=ecc, inc=inc, raan=raan, epoch=epoch)


# ---------------------------------------------------------------- 프리셋 (지구)
# 대표값이다. 실제 ISS 위치가 필요하면 TLE(TLEOrbit)를 쓴다.

LEO = Orbit.circular(Earth, 500e3)
ISS = Orbit.circular(Earth, 420e3, inc=math.radians(51.64))
SSO = sun_synchronous(700e3)
GEO = Orbit.circular(Earth, _tb.synchronous_radius(Earth.mu, Earth.rotation_rate) - Earth.radius)

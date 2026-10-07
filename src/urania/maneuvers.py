"""궤도 전이: Transfer 결과 객체와 기동 계획.

원 궤도 사이의 호만·이중타원 전이와 궤도면 변경을 계산한다.
각 기동은 실제 상태벡터에 Δv 벡터를 더하는 방식으로 만들고, 기동 사이는 2체 해석해로 잇는다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from . import units
from .core import maneuvers as _man
from .core import propagate as _prop
from .core.elements import _angle_between

if TYPE_CHECKING:
    from .orbits import Orbit
    from .time import Epoch

# 원 궤도로 간주하는 이심률 상한
CIRCULAR_ECC_TOL = 1e-3


def _readonly(x) -> np.ndarray:
    a = np.array(x, dtype=float)
    a.flags.writeable = False
    return a


@dataclass(frozen=True, eq=False)
class Burn:
    """임펄스 기동 한 번."""

    epoch: Epoch
    r: np.ndarray        # 기동 위치 [m]
    dv: np.ndarray       # Δv 벡터 [m/s], 관성 좌표계
    description: str

    def __post_init__(self):
        object.__setattr__(self, "r", _readonly(self.r))
        object.__setattr__(self, "dv", _readonly(self.dv))

    @property
    def magnitude(self) -> float:
        """Δv 크기 [m/s]."""
        return float(np.linalg.norm(self.dv))


@dataclass(frozen=True, eq=False)
class Transfer:
    """궤도 전이 결과: 기동 목록, 기동 직후 궤도들, 가정."""

    kind: str                       # "hohmann", "bielliptic", "plane_change"
    initial: Orbit
    target: Orbit
    burns: tuple[Burn, ...]
    orbits: tuple[Orbit, ...]       # 각 기동 직후의 궤도 (마지막은 목표 궤도에 올라선 상태)
    plane_change: float             # 전체 궤도면 변경 각도 [rad]
    assumptions: tuple[str, ...]

    @property
    def total_dv(self) -> float:
        """총 Δv [m/s]."""
        return sum(b.magnitude for b in self.burns)

    @property
    def coast(self) -> float:
        """첫 기동까지 기다리는 시간 [s] (노드 도달 대기)."""
        if not self.burns:
            return 0.0
        return self.burns[0].epoch - self.initial.epoch

    @property
    def tof(self) -> float:
        """첫 기동부터 마지막 기동까지 걸리는 시간 [s]."""
        if not self.burns:
            return 0.0
        return self.burns[-1].epoch - self.burns[0].epoch

    @property
    def final(self) -> Orbit:
        """전이를 마친 궤도."""
        return self.orbits[-1] if self.orbits else self.initial

    def __repr__(self) -> str:
        return (f"Transfer({self.kind}, Δv={self.total_dv / 1e3:.4f} km/s, "
                f"tof={self.tof / 3600:.3f} h, 기동 {len(self.burns)}회)")


def _unit(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x)


def transfer(initial: Orbit, target: Orbit, method: str = "hohmann", *, rb=None,
             plane_split="optimal") -> Transfer:
    """원 궤도 initial → target 전이를 계산한다. `Orbit.transfer_to`의 구현.

    Args:
        method: "hohmann" 또는 "bielliptic"
        rb: 이중타원 전이의 중간 원점 반지름 (길이)
        plane_split: 호만 전이에서 첫 기동이 맡을 궤도면 변경 비율 (0~1) 또는 "optimal".
            이중타원 전이는 속도가 가장 느린 중간 원점에서 궤도면을 모두 바꾼다.
    """
    from .orbits import Orbit

    body = initial.body
    if target.body is not body:
        raise ValueError(f"중심천체가 다릅니다: {body.name} → {target.body.name}")
    for name, o in (("시작", initial), ("목표", target)):
        if o.ecc > CIRCULAR_ECC_TOL:
            raise ValueError(f"{name} 궤도가 원 궤도가 아닙니다 (e={o.ecc:.4g}). "
                             f"MVP는 원 궤도 사이 전이만 지원합니다")
    mu = body.mu
    r1, r2 = initial.a, target.a
    h1 = _unit(np.cross(initial.r, initial.v))
    h2 = _unit(np.cross(target.r, target.v))
    cross = np.cross(h1, h2)
    dtheta = math.atan2(np.linalg.norm(cross), h1 @ h2)
    coplanar = dtheta < 1e-9

    # 첫 기동 위치: 같은 평면이면 지금 위치, 아니면 다음에 만나는 노드
    if coplanar:
        coast = 0.0
        node_axis = None
    else:
        node_axis = _unit(cross)
        angles = [_angle_between(initial.r, s * node_axis, h1) for s in (1.0, -1.0)]
        coast = min(angles) / initial.mean_motion

    same_radius = abs(r1 - r2) < 1e-9 * r1
    assumptions = [
        "임펄스 기동 (Δv가 순간적으로 적용됨)",
        "기동 사이는 2체 궤도 (섭동 무시)",
        "시작·목표 궤도는 원 궤도로 간주",
        "목표 궤도 안의 위상(랑데부)은 맞추지 않음",
    ]
    if not coplanar:
        assumptions.append("궤도면 변경은 두 궤도면의 교선(노드)에서 수행")

    def tilt(fraction: float) -> np.ndarray:
        """h1을 노드 축 기준으로 fraction·Δθ 만큼 h2 쪽으로 돌린 궤도면 법선."""
        if coplanar or fraction == 0.0:
            return h1
        if fraction == 1.0:
            return h2
        return _man.rotate(h1, node_axis, fraction * dtheta)

    # 기동 계획: (다음 원점 반지름 또는 None=원형화, 기동 후 궤도면 법선, 설명)
    if same_radius:
        if coplanar:
            return Transfer("none", initial, target, (), (), 0.0, tuple(assumptions))
        kind = "plane_change"
        plan = [(None, h2, "궤도면 변경")]
    elif method == "hohmann":
        kind = "hohmann"
        if plane_split == "optimal":
            split = _man.optimal_plane_split(r1, r2, dtheta, mu)
            if not coplanar:
                assumptions.append(f"궤도면 변경 분배는 총 Δv 최소화 (1차 {split:.3f})")
        else:
            split = float(plane_split)
            if not 0.0 <= split <= 1.0:
                raise ValueError(f"plane_split은 0~1 이어야 합니다: {split}")
        plan = [(r2, tilt(split), "1차: 전이 궤도 진입"),
                (None, h2, "2차: 목표 궤도 원형화")]
        plane_parts = [split * dtheta, (1.0 - split) * dtheta]
    elif method == "bielliptic":
        if rb is None:
            raise ValueError("이중타원 전이에는 중간 원점 반지름 rb가 필요합니다")
        rb = units.to_si(rb, units.LENGTH)
        if rb < max(r1, r2):
            raise ValueError(f"rb={rb} m 는 시작·목표 궤도 반지름보다 커야 합니다")
        kind = "bielliptic"
        plan = [(rb, h1, "1차: 첫 전이 타원 진입"),
                (r2, h2, "2차: 중간 원점에서 둘째 전이 타원으로"),
                (None, h2, "3차: 목표 궤도 원형화")]
        plane_parts = [0.0, dtheta, 0.0]
    else:
        raise ValueError(f"알 수 없는 전이 방법: {method!r} (hohmann, bielliptic)")

    if kind == "plane_change":
        plane_parts = [dtheta]

    # 기동 실행: 상태벡터에 Δv를 더하고, 다음 기동까지 해석해로 전파
    epoch = initial.epoch + coast
    r, v = _prop.kepler_propagate(initial.r, initial.v, coast, mu)
    burns, orbits = [], []
    for (r_next, normal, desc), d_plane in zip(plan, plane_parts):
        r_now = np.linalg.norm(r)
        direction = _unit(np.cross(normal, r))
        if r_next is None:
            speed = math.sqrt(mu / r_now)
        else:
            speed = math.sqrt(mu * (2.0 / r_now - 2.0 / (r_now + r_next)))
        v_new = speed * direction
        if d_plane > 1e-12:
            desc += f" + 궤도면 {math.degrees(d_plane):.2f}°"
        burns.append(Burn(epoch, r, v_new - v, desc))
        orbit = Orbit(body, r, v_new, epoch)
        orbits.append(orbit)
        if r_next is not None:
            half = math.pi * math.sqrt((0.5 * (r_now + r_next)) ** 3 / mu)
            r, v = _prop.kepler_propagate(r, v_new, half, mu)
            epoch = epoch + half

    return Transfer(kind, initial, target, tuple(burns), tuple(orbits), dtheta,
                    tuple(assumptions))

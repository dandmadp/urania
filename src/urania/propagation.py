"""궤도 전파: 충실도 모델 선택, 결과(Trajectory)와 메타데이터.

충실도 단계 (SPEC 2.7):
    0 "twobody"   2체
    1 "j2"        + J2
    2 "j2+drag"   + 대기 항력
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy.optimize import brentq

from . import units
from .core import forces as _forces
from .core import propagate as _prop
from .environment import Environment, as_environment

if TYPE_CHECKING:
    from .bodies import Body
    from .orbits import Orbit
    from .time import Epoch

FIDELITY_MODELS = {0: "twobody", 1: "j2", 2: "j2+drag"}
_SUPPORTED_TERMS = {"twobody", "j2", "drag"}
_FUTURE_TERMS = {"moon", "sun", "srp", "harmonics"}

DEFAULT_RTOL = 1e-12
DEFAULT_ATOL = 1e-8


def parse_model(model) -> frozenset[str]:
    """모델 지정("j2+drag" 또는 충실도 정수)을 섭동 항 집합으로 바꾼다. 2체는 항상 포함."""
    if isinstance(model, int):
        if model not in FIDELITY_MODELS:
            raise NotImplementedError(f"충실도 {model}단계는 아직 지원하지 않습니다 (0~2)")
        model = FIDELITY_MODELS[model]
    terms = {t.strip().lower() for t in model.split("+")}
    future = terms & _FUTURE_TERMS
    if future:
        raise NotImplementedError(f"{sorted(future)} 섭동은 MVP 이후에 지원합니다")
    unknown = terms - _SUPPORTED_TERMS
    if unknown:
        raise ValueError(f"알 수 없는 모델 항: {sorted(unknown)} (가능: {sorted(_SUPPORTED_TERMS)})")
    return frozenset(terms | {"twobody"})


def model_name(terms: frozenset[str]) -> str:
    return "+".join(t for t in ("twobody", "j2", "drag") if t in terms)


@dataclass(frozen=True)
class PropagationInfo:
    """전파 결과 메타데이터: 어떤 모델·적분기·가정으로 계산했는지."""

    model: str
    integrator: str
    rtol: float | None
    atol: float | None
    assumptions: tuple[str, ...]
    terminated: str | None = None   # 조기 종료 사유 (예: "재진입")
    nfev: int | None = None


def _readonly(a: np.ndarray) -> np.ndarray:
    a = np.array(a, dtype=float)
    a.flags.writeable = False
    return a


@dataclass(frozen=True, eq=False)
class Trajectory:
    """전파 결과. 시작 시각 기준 경과 시간별 상태벡터와 메타데이터를 담는다."""

    body: Body
    epoch: Epoch            # 시작 시각
    t: np.ndarray           # 시작 기준 경과 시간 [s], 길이 N
    r: np.ndarray           # 위치 [m], N×3
    v: np.ndarray           # 속도 [m/s], N×3
    info: PropagationInfo

    def __post_init__(self):
        for name in ("t", "r", "v"):
            object.__setattr__(self, name, _readonly(getattr(self, name)))

    def __len__(self) -> int:
        return len(self.t)

    def orbit_at(self, i: int) -> Orbit:
        """i번째 샘플의 접촉 궤도(osculating orbit)."""
        from .orbits import Orbit

        return Orbit(self.body, self.r[i], self.v[i], self.epoch + float(self.t[i]))

    @property
    def final(self) -> Orbit:
        """마지막 시점의 궤도."""
        return self.orbit_at(-1)

    @property
    def altitude(self) -> np.ndarray:
        """구형 천체 기준 고도 [m]."""
        return np.linalg.norm(self.r, axis=1) - self.body.radius

    def explain(self):
        """운동 방정식, 적분기, 결과 변화, 해석상 주의점을 보여준다."""
        from .explain import explain_trajectory

        return explain_trajectory(self)

    def plot(self, ax=None, kind: str = "orbit"):
        """kind="orbit": 시작 궤도면에 투영한 경로, "altitude": 시간별 고도."""
        from .viz import plot_trajectory

        return plot_trajectory(self, ax, kind)


def _default_n_points(orbit: Orbit, duration: float) -> int:
    """궤도 1바퀴에 100점, 최소 101점, 최대 200001점."""
    if orbit.ecc >= 1.0:
        return 1001
    n = math.ceil(abs(duration) / orbit.period * 100) + 1
    return min(max(n, 101), 200001)


_SURFACE = "천체 표면 도달"


def _truncate_at_surface(orbit: Orbit, t, r, v):
    """해석해 샘플에서 처음 표면 아래로 내려가는 구간을 찾아 그 시점에서 자른다.

    수치 적분의 표면 도달 이벤트와 같은 동작을 해석해에도 맞추기 위함이다.
    샘플 간격(1바퀴 100점)보다 짧게 스치는 관통은 놓칠 수 있다.
    """
    R = orbit.body.radius
    alt = np.linalg.norm(r, axis=1) - R
    below = np.nonzero((alt[1:] < 0.0) & (alt[:-1] >= 0.0))[0]
    if len(below) == 0:
        return t, r, v, False
    i = below[0] + 1

    def altitude_at(dt):
        ri, _ = _prop.kepler_propagate(orbit.r, orbit.v, dt, orbit.body.mu)
        return np.linalg.norm(ri) - R

    t_hit = brentq(altitude_at, t[i - 1], t[i], xtol=1e-6)
    r_hit, v_hit = _prop.kepler_propagate(orbit.r, orbit.v, t_hit, orbit.body.mu)
    return (np.append(t[:i], t_hit), np.vstack((r[:i], r_hit)),
            np.vstack((v[:i], v_hit)), True)


def propagate(orbit: Orbit, duration, *, model="twobody", cd: float = 2.2,
              area=None, mass=None, density=None, method: str = "auto",
              rtol: float = DEFAULT_RTOL, atol: float = DEFAULT_ATOL,
              n_points: int | None = None) -> Trajectory:
    """궤도를 duration 만큼 전파한다. `Orbit.propagate`의 구현.

    method="auto"면 2체는 해석해(케플러), 그 외는 DOP853 수치 적분을 쓴다.
    """
    body = orbit.body
    duration = units.to_si(duration, units.TIME)
    terms = parse_model(model)
    n = n_points or _default_n_points(orbit, duration)
    t_eval = np.linspace(0.0, duration, n)

    assumptions = ["중심천체 질점 중력", "관성 좌표계 축 고정 (세차·장동 무시)"]

    # 2체 해석해
    if terms == {"twobody"} and method in ("auto", "kepler"):
        r, v = _prop.kepler_states(orbit.r, orbit.v, t_eval, body.mu)
        t, r, v, hit = _truncate_at_surface(orbit, t_eval, r, v)
        info = PropagationInfo(model="twobody", integrator="kepler (해석해)",
                               rtol=None, atol=None, assumptions=tuple(assumptions),
                               terminated=_SURFACE if hit else None)
        return Trajectory(body, orbit.epoch, t, r, v, info)

    if method == "kepler":
        raise ValueError("해석해(kepler)는 2체 모델에서만 쓸 수 있습니다")
    integrator = "DOP853" if method == "auto" else method

    mu, R, J2, omega = body.mu, body.radius, body.J2, body.rotation_rate
    use_j2 = "j2" in terms
    if use_j2:
        if J2 == 0.0:
            raise ValueError(f"{body.name}의 J2가 0입니다")
        assumptions.append("J2 편평도만 포함 (고차 중력장 무시), 자전축 = 관성 z축")

    use_drag = "drag" in terms
    if use_drag:
        if area is None or mass is None:
            raise ValueError("항력 모델에는 area(단면적)와 mass(질량)가 필요합니다")
        area_si = units.to_si(area, units.AREA)
        mass_si = units.to_si(mass, units.MASS)
        ballistic = cd * area_si / mass_si
        if density is not None:
            rho: Environment = as_environment(density, units.DENSITY)
        elif body.atmosphere is not None:
            rho = body.atmosphere
        else:
            raise ValueError(f"{body.name}에 대기 모델이 없습니다. density를 지정하세요")
        assumptions += [
            f"대기 밀도: {rho.description}",
            "대기는 천체와 함께 강체 자전",
            f"항력 계수·단면적 일정 (Cd={cd:g}, A={area_si:g} m², m={mass_si:g} kg)",
        ]
        t0 = orbit.epoch.tdb_seconds

    def accel(t, r, v):
        a = _forces.accel_twobody(r, mu)
        if use_j2:
            a = a + _forces.accel_j2(r, mu, R, J2)
        if use_drag:
            a = a + _forces.accel_drag(r, v, rho(r, t0 + t), ballistic, omega)
        return a

    def hit_surface(t, y):
        return np.sqrt(y[:3] @ y[:3]) - R

    hit_surface.terminal = True
    hit_surface.direction = -1

    res = _prop.cowell(orbit.r, orbit.v, duration, accel, t_eval=t_eval,
                       events=[hit_surface], method=integrator, rtol=rtol, atol=atol)
    info = PropagationInfo(
        model=model_name(terms), integrator=f"scipy solve_ivp {integrator}",
        rtol=rtol, atol=atol, assumptions=tuple(assumptions),
        terminated=_SURFACE if res.terminated else None, nfev=res.nfev,
    )
    return Trajectory(body, orbit.epoch, res.t, res.r, res.v, info)

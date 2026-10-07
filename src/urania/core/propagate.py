"""궤도 전파: 해석적 케플러 전파와 Cowell 수치 적분."""

from typing import Callable, NamedTuple, Sequence

import numpy as np
from scipy.integrate import solve_ivp

from . import elements as _el
from . import kepler as _kepler


def kepler_states(r0, v0, dts: Sequence[float], mu: float) -> tuple[np.ndarray, np.ndarray]:
    """2체 해석해로 여러 시각 dts [s]의 상태벡터를 구한다. 타원·포물선·쌍곡선 모두 가능.

    Returns:
        r [m] (N×3), v [m/s] (N×3)
    """
    el = _el.rv_to_coe(r0, v0, mu)
    p, e = el.p, el.ecc
    if abs(e - 1.0) < _kepler.PARABOLIC_TOL:
        # Barker: D + D³/3 = 2√(μ/p³)·(t - t_p)
        n = 2.0 * np.sqrt(mu / p**3)
    else:
        n = np.sqrt(mu / abs(el.a) ** 3)
    M0 = _kepler.true_to_mean(el.nu, e)
    r = np.empty((len(dts), 3))
    v = np.empty((len(dts), 3))
    for i, dt in enumerate(dts):
        nu = _kepler.mean_to_true(M0 + n * dt, e)
        r[i], v[i] = _el.coe_to_rv(p, e, el.inc, el.raan, el.argp, nu, mu)
    return r, v


def kepler_propagate(r0, v0, dt: float, mu: float) -> tuple[np.ndarray, np.ndarray]:
    """2체 해석해로 dt [s] 후의 상태벡터를 구한다."""
    r, v = kepler_states(r0, v0, [dt], mu)
    return r[0], v[0]


class CowellResult(NamedTuple):
    t: np.ndarray          # 출력 시각 [s], 길이 N
    r: np.ndarray          # 위치 [m], N×3
    v: np.ndarray          # 속도 [m/s], N×3
    terminated: bool       # 이벤트로 조기 종료했는지
    nfev: int              # 가속도 함수 호출 횟수


def cowell(r0, v0, duration: float,
           accel: Callable[[float, np.ndarray, np.ndarray], np.ndarray], *,
           t_eval: Sequence[float] | None = None,
           events: Sequence[Callable] = (),
           method: str = "DOP853", rtol: float = 1e-12, atol: float = 1e-8) -> CowellResult:
    """운동 방정식 r'' = accel(t, r, v) 를 수치 적분한다 (Cowell 방법).

    Args:
        duration: 적분 시간 [s]. 음수면 과거로 전파.
        accel: 가속도 함수 (t [s, 시작 기준], r, v) → a [m/s²]
        events: solve_ivp 이벤트 함수 (t, y) → float. terminal 속성이 있으면 조기 종료.
            종료 시 이벤트 시점의 상태를 결과 끝에 덧붙인다.
    """
    y0 = np.concatenate((np.asarray(r0, dtype=float), np.asarray(v0, dtype=float)))
    if duration == 0.0:
        n = len(t_eval) if t_eval is not None else 1
        return CowellResult(t=np.zeros(n), r=np.tile(y0[:3], (n, 1)),
                            v=np.tile(y0[3:], (n, 1)), terminated=False, nfev=0)

    def rhs(t, y):
        return np.concatenate((y[3:], accel(t, y[:3], y[3:])))

    sol = solve_ivp(rhs, (0.0, duration), y0, method=method, t_eval=t_eval,
                    events=list(events) or None, rtol=rtol, atol=atol)
    if sol.status == -1:
        raise RuntimeError(f"적분 실패: {sol.message}")

    t, y = sol.t, sol.y
    terminated = sol.status == 1
    if terminated:
        for t_ev, y_ev in zip(sol.t_events, sol.y_events):
            if len(t_ev):
                t = np.append(t, t_ev[0])
                y = np.column_stack((y, y_ev[0]))
                break
    return CowellResult(t=t, r=y[:3].T.copy(), v=y[3:].T.copy(),
                        terminated=terminated, nfev=sol.nfev)

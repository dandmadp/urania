"""explain(): 사용한 공식, 대입한 중간값, 가정, 경고를 단계별 텍스트로 보여준다.

중간값은 core 함수로 계산한다 (공식 구현은 core 한 곳에만 둔다).
결과 객체에 저장된 값과 다시 계산한 값이 어긋나면 경고로 표시한다.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from .core import maneuvers as _man
from .core import twobody as _tb

if TYPE_CHECKING:
    from .maneuvers import Transfer
    from .orbits import Orbit
    from .propagation import Trajectory

DAY = 86400.0


# ---------------------------------------------------------------- 출력 형식

def _km(x: float) -> str:
    return f"{x / 1e3:,.3f} km"


def _kms(x: float) -> str:
    return f"{x / 1e3:.4f} km/s"


def _deg(x: float) -> str:
    return f"{math.degrees(x):.4f}°"


def _vec(x, scale: float, unit: str) -> str:
    return "[" + ", ".join(f"{c / scale:,.3f}" for c in x) + f"] {unit}"


class Explanation:
    """단계별 설명. print()하거나 REPL·Jupyter에서 그대로 보면 된다."""

    def __init__(self, title: str):
        self.title = title
        self.steps: list[tuple[str, list[str]]] = []
        self.assumptions: list[str] = []
        self.warnings: list[str] = []

    def step(self, heading: str, *lines: str) -> None:
        self.steps.append((heading, list(lines)))

    def __str__(self) -> str:
        out = [self.title, "─" * max(len(self.title), 20)]
        for i, (heading, lines) in enumerate(self.steps, 1):
            out.append(f"{i}. {heading}")
            out.extend(f"   {line}" for line in lines)
        if self.assumptions:
            out.append("가정")
            out.extend(f"   - {a}" for a in self.assumptions)
        if self.warnings:
            out.append("주의")
            out.extend(f"   ! {w}" for w in self.warnings)
        return "\n".join(out)

    __repr__ = __str__


# ---------------------------------------------------------------- Orbit

def explain_orbit(o: Orbit) -> Explanation:
    """상태벡터에서 궤도 요소를 구하는 과정 (Curtis 알고리즘 4.2)."""
    b = o.body
    mu = b.mu
    r_norm, v_norm = np.linalg.norm(o.r), np.linalg.norm(o.v)
    ex = Explanation(f"궤도 요소 계산: {b.name} 주위, {o.epoch.iso[:19]} TDB")

    ex.step("입력 상태벡터",
            f"r = {_vec(o.r, 1e3, 'km')},  |r| = {_km(r_norm)}",
            f"v = {_vec(o.v, 1e3, 'km/s')},  |v| = {_kms(v_norm)}",
            f"μ = {mu:.6e} m³/s²  ({b.source or '출처 미기재'})")
    ex.step("비각운동량과 반통경",
            "h = r × v",
            f"|h| = {o.h / 1e6:,.3f} km²/s",
            f"p = h²/μ = {_km(o.p)}")
    ex.step("이심률",
            "e = ((v² − μ/r) r − (r·v) v) / μ",
            f"|e| = {o.ecc:.6f}")
    ex.step("에너지와 장반경 (활력 방정식)",
            "ε = v²/2 − μ/r,  a = −μ/(2ε)",
            f"ε = {o.energy / 1e6:.4f} km²/s²",
            f"a = {_km(o.a)}" + ("  (쌍곡선이라 음수)" if o.a < 0 else ""))
    ex.step("경사각", "i = arccos(h_z / |h|)", f"i = {_deg(o.inc)}")
    ex.step("승교점 적경", "n = ẑ × h,  Ω = atan2(n_y, n_x)", f"Ω = {_deg(o.raan)}")
    ex.step("근지점 인수", "ω = n에서 e까지의 각 (궤도면 안, 운동 방향)", f"ω = {_deg(o.argp)}")
    ex.step("진근점이각", "ν = e에서 r까지의 각 (궤도면 안, 운동 방향)", f"ν = {_deg(o.nu)}")

    derived = [f"근점 반지름 r_p = p/(1+e) = {_km(o.r_periapsis)}  (고도 {_km(o.periapsis_altitude)})"]
    if o.ecc < 1.0:
        derived.append(f"원점 반지름 r_a = p/(1−e) = {_km(o.r_apoapsis)}  (고도 {_km(o.apoapsis_altitude)})")
        derived.append(f"주기 T = 2π√(a³/μ) = {o.period:,.1f} s = {o.period / 60:.2f} 분")
        if b.J2:
            derived.append(f"J2 승교점 변화 dΩ/dt = −(3/2) n J2 (R/p)² cos i = "
                           f"{math.degrees(o.raan_rate) * DAY:.4f} °/일")
    ex.step("파생 물리량", *derived)

    if o.ecc < 1e-11:
        ex.assumptions.append("원 궤도: 근지점이 정의되지 않아 ω = 0, ν는 승교점부터 잼")
    if o.inc < 1e-11 or abs(o.inc - math.pi) < 1e-11:
        ex.assumptions.append("적도 궤도: 승교점이 정의되지 않아 Ω = 0, 기준축은 관성 x축")
    ex.assumptions.append("2체 궤도 요소 (그 순간의 접촉 궤도, osculating)")
    return ex


# ---------------------------------------------------------------- Trajectory

def explain_trajectory(tr: Trajectory) -> Explanation:
    """전파에 쓴 운동 방정식, 적분기, 결과 변화와 해석상 주의점."""
    info = tr.info
    terms = info.model.split("+")
    ex = Explanation(f"궤도 전파: {info.model}, {tr.t[-1] / DAY:.3f}일")

    eq = ["r̈ = " + " + ".join({"twobody": "a_2체", "j2": "a_J2", "drag": "a_항력"}[t]
                               for t in terms),
          "a_2체 = −μ r / |r|³"]
    if "j2" in terms:
        eq.append("a_J2 = −(3/2) J2 μ R² / r⁵ · [x(1 − 5z²/r²), y(1 − 5z²/r²), z(3 − 5z²/r²)]")
    if "drag" in terms:
        eq.append("a_항력 = −½ ρ (C_D A/m) |v_rel| v_rel,  v_rel = v − ω × r")
    ex.step("운동 방정식", *eq)

    if info.rtol is None:
        ex.step("풀이", f"{info.integrator}: 케플러 방정식을 각 시각에서 직접 풂",
                f"출력 {len(tr)}점")
    else:
        ex.step("수치 적분", info.integrator,
                f"허용오차 rtol = {info.rtol:g}, atol = {info.atol:g} (m, m/s)",
                f"가속도 함수 호출 {info.nfev:,}회, 출력 {len(tr)}점")

    first, last = tr.orbit_at(0), tr.final
    lines = []
    for name, f, fmt in (("a", lambda o: o.a, _km), ("e", lambda o: o.ecc, lambda x: f"{x:.6f}"),
                         ("i", lambda o: o.inc, _deg), ("Ω", lambda o: o.raan, _deg)):
        lines.append(f"{name}: {fmt(f(first))} → {fmt(f(last))}")
    alt = tr.altitude
    lines.append(f"고도: {_km(alt[0])} → {_km(alt[-1])}")
    ex.step("시작 → 끝 (접촉 궤도 요소)", *lines)

    if "j2" in terms and first.ecc < 1.0:
        a = _tb.semi_major_axis(tr.r, tr.v, tr.body.mu)
        T = first.period
        head = a[tr.t <= tr.t[0] + T]
        tail = a[tr.t >= tr.t[-1] - T]
        if tr.t[-1] - tr.t[0] >= 2 * T:
            ex.step("장반경의 단주기 진동 (J2)",
                    f"접촉 장반경 범위: {_km(a.min())} ~ {_km(a.max())} "
                    f"(폭 {_km(a.max() - a.min())})",
                    f"첫 1바퀴 평균 a = {_km(head.mean())}",
                    f"마지막 1바퀴 평균 a = {_km(tail.mean())}",
                    f"평균 변화 = {_km(tail.mean() - head.mean())}")
        ex.warnings.append(
            "J2가 있으면 접촉 장반경이 한 바퀴 안에서 크게 출렁인다. "
            "시작·끝 값의 차이를 궤도 감쇠량으로 읽지 말고, 1바퀴 평균 변화를 보라.")

    ex.assumptions.extend(info.assumptions)
    if info.terminated:
        ex.warnings.append(f"{info.terminated}: {tr.t[-1] / DAY:.4f}일에 전파를 멈췄다.")
    return ex


# ---------------------------------------------------------------- Transfer

def _normal(o: Orbit) -> np.ndarray:
    h = np.cross(o.r, o.v)
    return h / np.linalg.norm(h)


def _angle(a: np.ndarray, b: np.ndarray) -> float:
    return math.atan2(np.linalg.norm(np.cross(a, b)), a @ b)


def _burn_lines(v_before: float, v_after: float, alpha: float, stored: float,
                ex: Explanation, label: str) -> list[str]:
    """한 기동의 Δv 공식과 값. stored(상태벡터로 계산한 값)와 비교한다."""
    if alpha > 1e-12:
        dv = _man.combined_dv(v_before, v_after, alpha)
        lines = [f"궤도면 변경 α = {_deg(alpha)} 동시 수행",
                 f"Δv = √(v₁² + v₂² − 2 v₁ v₂ cos α) = {_kms(dv)}"]
    else:
        dv = abs(v_after - v_before)
        lines = [f"Δv = |v₂ − v₁| = {_kms(dv)}"]
    if abs(dv - stored) > 1e-6 * max(dv, 1.0):
        ex.warnings.append(f"{label}: 공식 값 {_kms(dv)}와 상태벡터 계산 값 {_kms(stored)}이 다르다.")
    return lines


def explain_transfer(t: Transfer) -> Explanation:
    """전이의 공식, 중간 속도, 기동별 Δv, 전이 시간."""
    mu = t.initial.body.mu
    r1, r2 = t.initial.a, t.target.a
    names = {"hohmann": "호만 전이", "bielliptic": "이중타원 전이",
             "plane_change": "궤도면 변경", "none": "전이 없음"}
    ex = Explanation(f"{names[t.kind]}: r₁ = {_km(r1)} → r₂ = {_km(r2)}")

    start = [f"μ = {mu:.6e} m³/s²",
             f"두 궤도면 사이 각 Δθ = arccos(ĥ₁·ĥ₂) = {_deg(t.plane_change)}"]
    if t.coast > 0:
        start.append(f"노드까지 대기 {t.coast:,.1f} s ({t.coast / 60:.2f} 분)")
    ex.step("출발 조건", *start)

    if t.kind == "none":
        ex.step("결과", "이미 같은 궤도라 기동이 필요 없다.")
        ex.assumptions.extend(t.assumptions)
        return ex

    burns = t.burns
    if t.kind == "plane_change":
        v = _tb.circular_velocity(r1, mu)
        dv = _man.plane_change_dv(v, t.plane_change)
        ex.step("궤도면 변경 (노드에서)",
                f"v = √(μ/r) = {_kms(v)}",
                f"Δv = 2 v sin(Δθ/2) = {_kms(dv)}")
        if abs(dv - burns[0].magnitude) > 1e-6 * dv:
            ex.warnings.append("공식 값과 상태벡터 계산 값이 다르다.")
    elif t.kind == "hohmann":
        a_t = 0.5 * (r1 + r2)
        alpha1 = _angle(_normal(t.initial), _normal(t.orbits[0]))
        alpha2 = t.plane_change - alpha1
        v_c1, v_p = _tb.circular_velocity(r1, mu), _tb.vis_viva(r1, a_t, mu)
        v_a, v_c2 = _tb.vis_viva(r2, a_t, mu), _tb.circular_velocity(r2, mu)
        ex.step("전이 타원", f"a_t = (r₁ + r₂)/2 = {_km(a_t)}")
        ex.step("1차 기동 (r₁)",
                f"v₁ = √(μ/r₁) = {_kms(v_c1)}  (출발 원 궤도)",
                f"v₂ = √(μ(2/r₁ − 1/a_t)) = {_kms(v_p)}  (전이 타원)",
                *_burn_lines(v_c1, v_p, alpha1, burns[0].magnitude, ex, "1차 기동"))
        ex.step("2차 기동 (r₂)",
                f"v₁ = √(μ(2/r₂ − 1/a_t)) = {_kms(v_a)}  (전이 타원)",
                f"v₂ = √(μ/r₂) = {_kms(v_c2)}  (목표 원 궤도)",
                *_burn_lines(v_a, v_c2, alpha2, burns[1].magnitude, ex, "2차 기동"))
        ex.step("전이 시간", f"tof = π√(a_t³/μ) = {t.tof:,.1f} s = {t.tof / 3600:.4f} h")
    elif t.kind == "bielliptic":
        rb = np.linalg.norm(burns[1].r)
        a1, a2 = 0.5 * (r1 + rb), 0.5 * (rb + r2)
        ex.step("전이 타원 두 개",
                f"중간 원점 r_b = {_km(rb)}",
                f"a₁ = (r₁ + r_b)/2 = {_km(a1)},  a₂ = (r_b + r₂)/2 = {_km(a2)}")
        v = [(_tb.circular_velocity(r1, mu), _tb.vis_viva(r1, a1, mu), 0.0, "r₁"),
             (_tb.vis_viva(rb, a1, mu), _tb.vis_viva(rb, a2, mu), t.plane_change, "r_b"),
             (_tb.vis_viva(r2, a2, mu), _tb.circular_velocity(r2, mu), 0.0, "r₂")]
        for k, (vb, va, alpha, where) in enumerate(v):
            ex.step(f"{k + 1}차 기동 ({where})",
                    f"v₁ = {_kms(vb)},  v₂ = {_kms(va)}  (활력 방정식 √(μ(2/r − 1/a)))",
                    *_burn_lines(vb, va, alpha, burns[k].magnitude, ex, f"{k + 1}차 기동"))
        ex.step("전이 시간", f"tof = π(√(a₁³/μ) + √(a₂³/μ)) = {t.tof / 3600:.4f} h")

    ex.step("합계", " + ".join(_kms(b.magnitude) for b in burns) + f" = {_kms(t.total_dv)}")
    ex.assumptions.extend(t.assumptions)
    return ex

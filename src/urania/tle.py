"""TLE(Two-Line Element) 궤도: sgp4 패키지 래퍼.

TLE 요소는 SGP4 이론 전용 평균 요소라, SGP4로만 전파해야 의미가 있다.
그래서 `Orbit`과 섞지 않고 별도 객체로 두며, 자체 전파기로 넘기려면 `to_orbit()`으로
특정 시각의 상태벡터를 명시적으로 꺼낸다.

- 좌표계: SGP4 출력인 TEME를 그대로 관성계로 쓴다 (세차·장동 변환 없음).
- 시간: TLE 시각은 UTC다. TDB(`Epoch`) 변환은 astropy가 한다.
- 중력 상수: TLE 생성 규약인 WGS72.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from astropy.time import Time
from sgp4.api import SGP4_ERRORS, WGS72, Satrec

from .bodies import Earth
from .propagation import PropagationInfo, Trajectory, resolve_duration
from .time import JD_J2000, SECONDS_PER_DAY, Epoch

KM = 1e3
MU_WGS72 = 398600.8e9   # TLE 규약의 지구 μ [m³/s²]

ASSUMPTIONS = (
    "SGP4/SDP4 해석 이론 (TLE 평균 요소 전용, sgp4 패키지)",
    "좌표계: TEME를 관성계로 그대로 사용 (세차·장동 미변환)",
    "중력 상수: WGS72 (TLE 생성 규약)",
    "TLE 시각은 UTC, TDB 변환은 astropy",
    "정확도는 TLE 나이에 따라 나빠진다 (보통 epoch에서 1 km, 하루에 수 km 증가)",
)


def checksum(line: str) -> int:
    """TLE 줄의 체크섬: 앞 68글자의 숫자 합 + '-' 개수, mod 10."""
    return sum(int(c) if c.isdigit() else (1 if c == "-" else 0) for c in line[:68]) % 10


def _validate(line: str, number: str) -> None:
    if len(line) < 69 or line[0] != number:
        raise ValueError(f"TLE {number}번째 줄 형식이 아닙니다: {line!r}")
    if not line[68].isdigit() or int(line[68]) != checksum(line):
        raise ValueError(f"TLE {number}번째 줄 체크섬 불일치 (기대 {checksum(line)}, 실제 {line[68]})")


def _epoch_to_utc_jd(seconds) -> tuple[np.ndarray, np.ndarray]:
    """J2000 기준 TDB 초 (스칼라 또는 배열) → UTC 율리우스일 (정수부, 소수부)."""
    s = np.atleast_1d(np.asarray(seconds, dtype=float))
    utc = Time(np.full(s.shape, JD_J2000), s / SECONDS_PER_DAY, format="jd", scale="tdb").utc
    return utc.jd1, utc.jd2


@dataclass(frozen=True, eq=False)
class TLEOrbit:
    """TLE로 정의된 지구 궤도. SGP4로만 전파한다."""

    line1: str
    line2: str
    name: str = ""
    _sat: Satrec = field(init=False, repr=False)

    def __post_init__(self):
        object.__setattr__(self, "line1", self.line1.rstrip())
        object.__setattr__(self, "line2", self.line2.rstrip())
        _validate(self.line1, "1")
        _validate(self.line2, "2")
        if self.line1[2:7] != self.line2[2:7]:
            raise ValueError("두 줄의 위성 번호가 다릅니다")
        object.__setattr__(self, "_sat", Satrec.twoline2rv(self.line1, self.line2, WGS72))

    @classmethod
    def from_text(cls, text: str) -> TLEOrbit:
        """2줄 또는 3줄(첫 줄이 이름) TLE 텍스트로 생성."""
        lines = [ln for ln in text.strip().splitlines() if ln.strip()]
        if len(lines) == 2:
            return cls(lines[0], lines[1])
        if len(lines) == 3:
            name = lines[0].strip()
            if name.startswith("0 "):
                name = name[2:]
            return cls(lines[1], lines[2], name=name)
        raise ValueError(f"TLE는 2줄 또는 3줄이어야 합니다 ({len(lines)}줄)")

    # ---------------------------------------------------------- TLE 요소 (SGP4 평균 요소)

    @property
    def satnum(self) -> str:
        """NORAD 위성 번호."""
        return self.line1[2:7].strip()

    @property
    def epoch(self) -> Epoch:
        """TLE 기준 시각 (TDB로 변환)."""
        s = self._sat
        return Epoch.from_astropy(Time(s.jdsatepoch, s.jdsatepochF, format="jd", scale="utc"))

    @property
    def inc(self) -> float:
        """평균 경사각 [rad]."""
        return self._sat.inclo

    @property
    def raan(self) -> float:
        """평균 승교점 적경 [rad]."""
        return self._sat.nodeo

    @property
    def ecc(self) -> float:
        """평균 이심률."""
        return self._sat.ecco

    @property
    def argp(self) -> float:
        """평균 근지점 인수 [rad]."""
        return self._sat.argpo

    @property
    def M(self) -> float:
        """평균근점이각 [rad]."""
        return self._sat.mo

    @property
    def mean_motion(self) -> float:
        """평균 운동 [rad/s] (Kozai 평균)."""
        return self._sat.no_kozai / 60.0

    @property
    def revs_per_day(self) -> float:
        """하루 공전 횟수."""
        return self.mean_motion * SECONDS_PER_DAY / (2.0 * math.pi)

    @property
    def bstar(self) -> float:
        """항력 항 B* [1/지구반지름]."""
        return self._sat.bstar

    # ---------------------------------------------------------- SGP4 전파

    def states(self, epochs) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """여러 시각의 TEME 상태벡터.

        Args:
            epochs: Epoch, Epoch 시퀀스, 또는 J2000 기준 TDB 초 배열

        Returns:
            (error, r [m] N×3, v [m/s] N×3). error는 SGP4 오류 코드 (0 = 정상).
        """
        if isinstance(epochs, Epoch):
            seconds = [epochs.tdb_seconds]
        elif len(epochs) and isinstance(epochs[0], Epoch):
            seconds = [e.tdb_seconds for e in epochs]
        else:
            seconds = epochs
        jd, fr = _epoch_to_utc_jd(seconds)
        err, r, v = self._sat.sgp4_array(jd, fr)
        return err, r * KM, v * KM

    def state_at(self, epoch: Epoch) -> tuple[np.ndarray, np.ndarray]:
        """한 시각의 TEME 상태벡터 (r [m], v [m/s])."""
        err, r, v = self.states(epoch)
        if err[0]:
            raise RuntimeError(f"SGP4 오류 {err[0]}: {SGP4_ERRORS[err[0]]}")
        return r[0], v[0]

    def propagate(self, duration=None, *, days=None, start: Epoch | None = None,
                  n_points: int | None = None) -> Trajectory:
        """SGP4로 전파해 Trajectory를 돌려준다.

        Args:
            duration, days: 전파 시간 (둘 중 하나)
            start: 시작 시각. 기본값은 TLE epoch.
            n_points: 출력 점 개수. 기본값은 1바퀴 100점.
        """
        duration = resolve_duration(duration, days)
        start = start or self.epoch
        if n_points is None:
            period = 2.0 * math.pi / self.mean_motion
            n_points = min(max(math.ceil(abs(duration) / period * 100) + 1, 101), 200001)
        t = np.linspace(0.0, duration, n_points)
        err, r, v = self.states(start.tdb_seconds + t)

        terminated = None
        bad = np.nonzero(err)[0]
        if len(bad):
            i = bad[0]
            if i == 0:
                raise RuntimeError(f"SGP4 오류 {err[0]}: {SGP4_ERRORS[err[0]]}")
            terminated = f"SGP4 오류 {err[i]} ({SGP4_ERRORS[err[i]]})"
            t, r, v = t[:i], r[:i], v[:i]

        info = PropagationInfo(model="sgp4", integrator="SGP4 (sgp4 패키지, WGS72)",
                               rtol=None, atol=None, assumptions=ASSUMPTIONS,
                               terminated=terminated)
        return Trajectory(Earth, start, t, r, v, info)

    def to_orbit(self, epoch: Epoch | None = None):
        """SGP4 상태벡터를 꺼내 자체 전파기용 Orbit으로 바꾼다 (기본: TLE epoch).

        이후 Orbit.propagate()는 SGP4가 아니라 urania 전파기를 쓴다.
        좌표계는 TEME 그대로이고, 중심천체 상수는 Earth(WGS 84)다.
        """
        from .orbits import Orbit

        epoch = epoch or self.epoch
        r, v = self.state_at(epoch)
        return Orbit(Earth, r, v, epoch)

    # ---------------------------------------------------------- 설명

    def explain(self):
        """TLE 필드를 풀어 보여준다."""
        from .explain import explain_tle

        return explain_tle(self)

    def __repr__(self) -> str:
        label = f"{self.name!r}, " if self.name else ""
        return (f"TLEOrbit({label}#{self.satnum}, epoch={self.epoch.iso[:19]} TDB, "
                f"inc={math.degrees(self.inc):.2f}°, {self.revs_per_day:.4f} rev/day)")

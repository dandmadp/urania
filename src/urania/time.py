"""시간 표현.

내부 시간은 J2000(2000-01-01 12:00:00 TDB)부터 경과한 TDB 초 하나로 고정한다.
UTC/TAI 등 다른 시간 척도 변환은 MVP 범위 밖이며, 필요하면 astropy Time을 거친다.

달력 문자열·율리우스일 입력은 TDB 기준으로 해석한다.
TDB 달력에는 윤초가 없으므로 단순 날짜 차이로 경과 초를 계산해도 정확하다.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

from . import units

SECONDS_PER_DAY = 86400.0
JD_J2000 = 2451545.0
_DT_J2000 = datetime(2000, 1, 1, 12, 0, 0)


@dataclass(frozen=True, order=True)
class Epoch:
    """TDB 시각. 내부 값은 J2000 기준 경과 초."""

    tdb_seconds: float

    # ---------------------------------------------------------- 생성

    @classmethod
    def from_iso(cls, text: str) -> Epoch:
        """ISO 8601 문자열(TDB 기준)로 생성. 예: "2026-10-07T12:00:00"."""
        dt = datetime.fromisoformat(text)
        if dt.tzinfo is not None:
            raise ValueError("시간대 정보는 지원하지 않습니다 (TDB 달력으로 해석)")
        return cls((dt - _DT_J2000).total_seconds())

    @classmethod
    def from_jd(cls, jd: float) -> Epoch:
        """TDB 율리우스일로 생성."""
        return cls((jd - JD_J2000) * SECONDS_PER_DAY)

    @classmethod
    def from_astropy(cls, t) -> Epoch:
        """astropy Time으로 생성. 다른 시간 척도는 astropy가 TDB로 변환한다."""
        tdb = t.tdb
        return cls(((tdb.jd1 - JD_J2000) + tdb.jd2) * SECONDS_PER_DAY)

    # ---------------------------------------------------------- 출력

    @property
    def jd(self) -> float:
        """TDB 율리우스일."""
        return JD_J2000 + self.tdb_seconds / SECONDS_PER_DAY

    @property
    def iso(self) -> str:
        """ISO 8601 문자열 (TDB, 마이크로초 단위)."""
        return (_DT_J2000 + timedelta(seconds=self.tdb_seconds)).isoformat()

    def to_astropy(self):
        """astropy Time(scale='tdb')으로 변환."""
        from astropy.time import Time

        return Time(JD_J2000, self.tdb_seconds / SECONDS_PER_DAY,
                    format="jd", scale="tdb")

    # ---------------------------------------------------------- 연산

    def __add__(self, dt) -> Epoch:
        """시각 + 시간 간격. 간격은 초(숫자) 또는 시간 Quantity."""
        return Epoch(self.tdb_seconds + units.to_si(dt, units.TIME))

    def __sub__(self, other):
        """시각 - 시각 → 경과 초(float), 시각 - 간격 → 시각."""
        if isinstance(other, Epoch):
            return self.tdb_seconds - other.tdb_seconds
        return Epoch(self.tdb_seconds - units.to_si(other, units.TIME))

    def __repr__(self) -> str:
        return f"Epoch('{self.iso}' TDB)"


J2000 = Epoch(0.0)

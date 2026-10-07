import math

import pytest

from urania.core import j2, twobody

MU = 3.986004418e14
R = 6378137.0
J2 = 1.08262668e-3
DAY = 86400.0


def test_period_and_mean_motion():
    a = 7000e3
    T = twobody.period(a, MU)
    assert T == pytest.approx(5828.5, abs=0.5)  # 약 97.1분
    assert twobody.mean_motion(a, MU) * T == pytest.approx(2 * math.pi)


def test_period_rejects_hyperbola():
    with pytest.raises(ValueError):
        twobody.period(-7000e3, MU)


def test_velocities_and_energy():
    r = 6778e3
    vc = twobody.circular_velocity(r, MU)
    assert vc == pytest.approx(7668.6, abs=0.5)
    assert twobody.escape_velocity(r, MU) == pytest.approx(math.sqrt(2) * vc)
    assert twobody.specific_energy(r, MU) == pytest.approx(-vc**2 / 2)
    assert twobody.specific_energy(math.inf, MU) == 0.0


def test_geo_radius():
    """지구 정지궤도 반지름 42164 km."""
    assert twobody.synchronous_radius(MU, 7.292115e-5) / 1e3 == pytest.approx(42164.17, abs=0.05)


def test_iss_raan_regression():
    """420 km, 51.64° 원 궤도의 승교점 후퇴율.

    손계산: n = 1.12638e-3 rad/s, (R/a)² = 0.88025, cos i = 0.62043
    → dΩ/dt = -1.5·n·J2·(R/a)²·cos i = -9.99e-7 rad/s = -4.945°/일
    """
    rate = j2.raan_rate(R + 420e3, 0.0, math.radians(51.64), MU, R, J2)
    assert math.degrees(rate) * DAY == pytest.approx(-4.945, abs=0.005)


def test_critical_inclination_freezes_argp():
    """임계 경사각 63.43°(cos²i = 1/5)에서 dω/dt = 0."""
    inc = math.acos(math.sqrt(0.2))
    assert j2.argp_rate(26600e3, 0.74, inc, MU, R, J2) == pytest.approx(0.0, abs=1e-20)


@pytest.mark.parametrize("alt_km, inc_deg", [(500, 97.40), (700, 98.19), (800, 98.60)])
def test_sso_inclination(alt_km, inc_deg):
    """원형 태양동기궤도 경사각 (Vallado 그림 9-14, Wertz SMAD 표와 대조)."""
    target = 2 * math.pi / (365.24219 * DAY)
    inc = j2.sso_inclination(R + alt_km * 1e3, 0.0, MU, R, J2, target)
    assert math.degrees(inc) == pytest.approx(inc_deg, abs=0.02)
    assert j2.raan_rate(R + alt_km * 1e3, 0.0, inc, MU, R, J2) == pytest.approx(target)


def test_sso_impossible_altitude():
    target = 2 * math.pi / (365.24219 * DAY)
    with pytest.raises(ValueError):
        j2.sso_inclination(R + 10000e3, 0.0, MU, R, J2, target)


@pytest.mark.parametrize("a, ecc", [(-20000e3, 1.5), (7000e3, 1.0)])
def test_j2_rates_reject_non_elliptic(a, ecc):
    """쌍곡선·포물선에서는 원인을 알 수 있는 ValueError (회귀 테스트)."""
    with pytest.raises(ValueError, match="타원"):
        j2.raan_rate(a, ecc, 0.5, MU, R, J2)

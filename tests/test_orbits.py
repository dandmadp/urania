import math

import astropy.units as u
import numpy as np
import pytest

from urania import GEO, ISS, LEO, SSO, Earth, Epoch, Orbit, sun_synchronous

DAY = 86400.0


def test_from_vectors_matches_core():
    """Vallado 예제 2-5 상태벡터를 단위 Quantity로 넣는다."""
    o = Orbit.from_vectors(Earth, [6524.834, 6862.875, 6448.296] * u.km,
                           [4.901327, 5.533756, -1.976341] * u.km / u.s)
    assert o.a / 1e3 == pytest.approx(36127.343, abs=0.05)
    assert o.ecc == pytest.approx(0.832853, abs=1e-6)
    assert math.degrees(o.inc) == pytest.approx(87.870, abs=1e-3)


def test_from_elements_roundtrip():
    o = Orbit.from_elements(Earth, a=8000 * u.km, ecc=0.1, inc=30 * u.deg,
                            raan=40 * u.deg, argp=60 * u.deg, nu=90 * u.deg)
    assert o.a == pytest.approx(8.0e6)
    assert o.ecc == pytest.approx(0.1)
    assert math.degrees(o.inc) == pytest.approx(30)
    assert math.degrees(o.raan) == pytest.approx(40)
    assert math.degrees(o.argp) == pytest.approx(60)
    assert math.degrees(o.nu) == pytest.approx(90)
    assert o.r_periapsis == pytest.approx(7.2e6)
    assert o.r_apoapsis == pytest.approx(8.8e6)


def test_from_elements_hyperbola_and_parabola():
    hyp = Orbit.from_elements(Earth, a=-20000e3, ecc=1.5)
    assert hyp.a == pytest.approx(-20000e3)
    assert hyp.energy > 0
    assert hyp.r_apoapsis == math.inf
    par = Orbit.from_elements(Earth, p=14000e3, ecc=1.0)
    assert par.ecc == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize("kwargs", [
    {},                                  # 크기 없음
    {"a": 7e6, "p": 7e6},                # 둘 다
    {"a": 7e6, "ecc": 1.5},              # 쌍곡선인데 a > 0
    {"a": -7e6, "ecc": 0.5},             # 타원인데 a < 0
    {"a": 7e6, "ecc": 1.0},              # 포물선은 p로만
])
def test_from_elements_invalid(kwargs):
    with pytest.raises(ValueError):
        Orbit.from_elements(Earth, **kwargs)


def test_circular_and_altitudes():
    o = Orbit.circular(Earth, 400 * u.km, inc=51.6 * u.deg, arglat=45 * u.deg)
    assert o.ecc == pytest.approx(0.0, abs=1e-12)
    assert o.periapsis_altitude == pytest.approx(400e3)
    assert o.apoapsis_altitude == pytest.approx(400e3)
    # 원 궤도에서 ν는 승교점부터 잰 각도 (DECISIONS D2)
    assert math.degrees(o.nu) == pytest.approx(45)


def test_immutable_state():
    o = LEO
    with pytest.raises(AttributeError):
        o.r = np.zeros(3)
    with pytest.raises(ValueError):
        o.r[0] = 0.0


def test_input_array_is_copied():
    r = np.array([7000e3, 0.0, 0.0])
    o = Orbit(Earth, r, [0.0, 7546.0, 0.0])
    r[0] = 0.0
    assert o.r[0] == 7000e3


def test_epoch():
    t = Epoch.from_iso("2026-10-07T00:00:00")
    o = Orbit.circular(Earth, 500e3, epoch=t)
    assert o.epoch == t


def test_presets():
    assert LEO.periapsis_altitude == pytest.approx(500e3)
    assert math.degrees(ISS.inc) == pytest.approx(51.64)
    # a = 6798.137 km → T = 2π√(a³/μ) = 5578.2 s (손계산)
    assert ISS.period / 60 == pytest.approx(92.97, abs=0.01)
    # GEO 주기는 항성일
    assert GEO.period == pytest.approx(86164.09, abs=0.1)
    assert GEO.inc == 0.0
    # SSO는 하루 약 0.9856° 동쪽으로 회전
    assert math.degrees(SSO.raan_rate) * DAY == pytest.approx(0.98563, abs=1e-4)


def test_sun_synchronous_eccentric():
    o = sun_synchronous(600 * u.km, ecc=0.01)
    assert math.degrees(o.raan_rate) * DAY == pytest.approx(0.98563, abs=1e-4)

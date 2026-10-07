import math

import numpy as np
import pytest

from urania.core import Elements, coe_to_rv, elements_to_rv, rv_to_coe

KM = 1e3
MU_EARTH_CURTIS = 398600 * KM**3      # Curtis 교재 값
MU_EARTH_VALLADO = 398600.4418 * KM**3  # Vallado 교재 값
deg = math.radians


def test_curtis_example_4_3_rv_to_coe():
    """Curtis, Orbital Mechanics for Engineering Students, 예제 4.3."""
    r = np.array([-6045, -3490, 2500]) * KM
    v = np.array([-3.457, 6.618, 2.533]) * KM
    el = rv_to_coe(r, v, MU_EARTH_CURTIS)

    h = math.sqrt(el.p * MU_EARTH_CURTIS)
    assert h == pytest.approx(58310 * KM**2, rel=1e-4)
    assert el.ecc == pytest.approx(0.1712, abs=1e-4)
    assert math.degrees(el.inc) == pytest.approx(153.2, abs=0.05)
    assert math.degrees(el.raan) == pytest.approx(255.3, abs=0.05)
    assert math.degrees(el.argp) == pytest.approx(20.07, abs=0.01)
    assert math.degrees(el.nu) == pytest.approx(28.45, abs=0.01)
    assert el.a == pytest.approx(8788 * KM, rel=1e-4)


def test_curtis_example_4_7_coe_to_rv():
    """Curtis 예제 4.7: 쌍곡선 궤도 요소 → 상태벡터."""
    h = 80000 * KM**2
    p = h**2 / MU_EARTH_CURTIS
    r, v = coe_to_rv(p, 1.4, deg(30), deg(40), deg(60), deg(30), MU_EARTH_CURTIS)

    # 교재 값은 유효숫자 4자리로 반올림됨
    np.testing.assert_allclose(r / KM, [-4040, 4815, 3629], rtol=5e-4)
    np.testing.assert_allclose(v / KM, [-10.39, -4.772, 1.744], rtol=5e-4)


def test_vallado_example_2_5_rv_to_coe():
    """Vallado, Fundamentals of Astrodynamics and Applications, 예제 2-5."""
    r = np.array([6524.834, 6862.875, 6448.296]) * KM
    v = np.array([4.901327, 5.533756, -1.976341]) * KM
    el = rv_to_coe(r, v, MU_EARTH_VALLADO)

    assert el.p / KM == pytest.approx(11067.790, abs=1e-2)
    assert el.a / KM == pytest.approx(36127.343, abs=1e-2)
    assert el.ecc == pytest.approx(0.832853, abs=1e-6)
    assert math.degrees(el.inc) == pytest.approx(87.870, abs=1e-3)
    assert math.degrees(el.raan) == pytest.approx(227.898, abs=1e-3)
    assert math.degrees(el.argp) == pytest.approx(53.38, abs=1e-2)
    assert math.degrees(el.nu) == pytest.approx(92.335, abs=1e-3)


def _random_elements(rng, ecc):
    p = rng.uniform(6600, 50000) * KM
    nu_max = math.pi if ecc < 1 else 0.95 * math.acos(-1 / ecc)
    return Elements(
        p=p,
        ecc=ecc,
        inc=rng.uniform(0.01, math.pi - 0.01),
        raan=rng.uniform(0, 2 * math.pi),
        argp=rng.uniform(0, 2 * math.pi),
        nu=rng.uniform(-nu_max, nu_max),
    )


@pytest.mark.parametrize("ecc", [0.001, 0.3, 0.9, 1.0, 1.5, 4.0])
def test_roundtrip_rv(ecc):
    """r, v → 요소 → r, v 가 원래 값으로 돌아와야 한다."""
    rng = np.random.default_rng(42)
    for _ in range(50):
        r0, v0 = elements_to_rv(_random_elements(rng, ecc), MU_EARTH_VALLADO)
        r1, v1 = elements_to_rv(rv_to_coe(r0, v0, MU_EARTH_VALLADO), MU_EARTH_VALLADO)
        np.testing.assert_allclose(r1, r0, rtol=0, atol=1e-6 * np.linalg.norm(r0))
        np.testing.assert_allclose(v1, v0, rtol=0, atol=1e-6 * np.linalg.norm(v0))


@pytest.mark.parametrize(
    "inc, ecc",
    [(0.0, 0.0), (math.pi, 0.0), (0.0, 0.2), (math.pi, 0.2), (deg(51.6), 0.0)],
    ids=["equatorial-circular", "retro-equatorial-circular", "equatorial",
         "retro-equatorial", "circular-inclined"],
)
def test_singular_orbits_roundtrip(inc, ecc):
    """원·적도 궤도는 규약(raan=0, argp=0)에 따라 일관되게 왕복해야 한다."""
    p = 7000 * KM
    r0, v0 = coe_to_rv(p, ecc, inc, deg(30) if inc not in (0.0, math.pi) else 0.0,
                       deg(45) if ecc else 0.0, deg(100), MU_EARTH_VALLADO)
    el = rv_to_coe(r0, v0, MU_EARTH_VALLADO)
    if inc in (0.0, math.pi):
        assert el.raan == 0.0
    if ecc == 0.0:
        assert el.argp == 0.0
    r1, v1 = elements_to_rv(el, MU_EARTH_VALLADO)
    np.testing.assert_allclose(r1, r0, atol=1e-6)
    np.testing.assert_allclose(v1, v0, atol=1e-9)


def test_unreachable_true_anomaly_raises():
    # e = 2 쌍곡선의 점근선은 ν = 120°, 그 밖은 도달 불가
    with pytest.raises(ValueError):
        coe_to_rv(7000 * KM, 2.0, 0.1, 0, 0, deg(150), MU_EARTH_VALLADO)


def test_rectilinear_raises():
    with pytest.raises(ValueError):
        rv_to_coe([7000 * KM, 0, 0], [1000, 0, 0], MU_EARTH_VALLADO)

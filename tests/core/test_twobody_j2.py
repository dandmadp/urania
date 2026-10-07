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
    assert T == pytest.approx(5828.5, abs=0.5)  # about 97.1 min
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
    """Earth geostationary radius 42164 km."""
    assert twobody.synchronous_radius(MU, 7.292115e-5) / 1e3 == pytest.approx(42164.17, abs=0.05)


def test_iss_raan_regression():
    """Nodal regression of a 420 km, 51.64° circular orbit.

    Hand calculation: n = 1.12638e-3 rad/s, (R/a)² = 0.88025, cos i = 0.62043
    → dΩ/dt = -1.5·n·J2·(R/a)²·cos i = -9.99e-7 rad/s = -4.945°/day
    """
    rate = j2.raan_rate(R + 420e3, 0.0, math.radians(51.64), MU, R, J2)
    assert math.degrees(rate) * DAY == pytest.approx(-4.945, abs=0.005)


def test_critical_inclination_freezes_argp():
    """dω/dt = 0 at the critical inclination 63.43° (cos²i = 1/5)."""
    inc = math.acos(math.sqrt(0.2))
    assert j2.argp_rate(26600e3, 0.74, inc, MU, R, J2) == pytest.approx(0.0, abs=1e-20)


@pytest.mark.parametrize("alt_km, inc_deg", [(500, 97.40), (700, 98.19), (800, 98.60)])
def test_sso_inclination(alt_km, inc_deg):
    """Circular sun-synchronous inclinations (compare Vallado fig. 9-14, Wertz SMAD tables)."""
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
    """Hyperbolas and parabolas give a clear ValueError (regression test)."""
    with pytest.raises(ValueError, match="elliptic"):
        j2.raan_rate(a, ecc, 0.5, MU, R, J2)


def test_apsides_to_ae():
    a, e = twobody.apsides_to_ae(6778e3, 6978e3)
    assert a == pytest.approx(6878e3)
    assert e == pytest.approx(200e3 / 13756e3)
    assert twobody.apsides_to_ae(7000e3, 7000e3) == (7000e3, 0.0)
    with pytest.raises(ValueError):
        twobody.apsides_to_ae(7000e3, 6000e3)


def test_parabolic_energy_is_exactly_zero():
    assert math.copysign(1.0, twobody.specific_energy(math.inf, MU)) == 1.0

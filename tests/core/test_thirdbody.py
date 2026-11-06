"""Moon position, third-body and radiation-pressure accelerations, zonal harmonics."""

import math

import numpy as np
import pytest
from numpy.polynomial import legendre

from urania.core.forces import accel_j2
from urania.core.sun import AU, sun_position
from urania.core.thirdbody import (MU_MOON, MU_SUN, SOLAR_PRESSURE, accel_srp, accel_third_body, accel_zonal,
                                   moon_position)

MU, R = 3.986004418e14, 6378137.0
J_EGM08 = (1.08262668355e-3, -2.53265648533e-6, -1.61962159137e-6, -2.27296082869e-7, 5.40681239107e-7)


def test_vallado_example_5_3_moon():
    """Vallado Example 5-3: Moon on 1994-04-28 0h UT1 (JD 2449470.5):
    r = (-134240.626, -311571.590, -126693.785) km."""
    r = moon_position(2449470.5) / 1e3
    assert r == pytest.approx([-134240.626, -311571.590, -126693.785], abs=1e-3)


@pytest.mark.parametrize("jd", [2451545.0, 2455000.25, 2460000.75, 2461350.5])
def test_moon_against_astropy(jd):
    """Within the algorithm's stated accuracy (0.3° longitude, 0.2° latitude) of astropy's ephemeris."""
    from astropy.coordinates import get_body
    from astropy.time import Time

    t = Time(jd, format="jd", scale="tdb")
    ref = get_body("moon", t).cartesian.xyz.to("m").value      # GCRS
    mine = moon_position(jd)
    angle = math.degrees(math.acos(ref @ mine / (np.linalg.norm(ref) * np.linalg.norm(mine))))
    # urania's frame is the equator of date; GCRS differs by precession (≈ 0.014°/year from J2000)
    assert angle < 0.4 + 0.014 * abs(jd - 2451545.0) / 365.25
    assert np.linalg.norm(mine) == pytest.approx(np.linalg.norm(ref), rel=0.01)
    assert moon_position(np.array([jd, jd])).shape == (2, 3)


def test_third_body_tidal_limit():
    """For |r| ≪ |s| the third-body acceleration tends to the tidal term μ/s³ (3(ŝ·r)ŝ - r)."""
    s = np.array([3.8e8, 1.0e8, -0.5e8])
    r = np.array([7e6, -2e6, 1e6])
    s_hat = s / np.linalg.norm(s)
    tidal = MU_MOON / np.linalg.norm(s) ** 3 * (3 * (s_hat @ r) * s_hat - r)
    assert accel_third_body(r, s, MU_MOON) == pytest.approx(tidal, rel=0.05)
    assert accel_third_body(np.zeros(3) + 1e-3, s, MU_MOON) == pytest.approx(0, abs=1e-15)


def test_srp_magnitude_direction_and_shadow():
    r_sun = np.array([AU, 0.0, 0.0])
    sunlit = np.array([0.0, 7e6, 0.0])
    a = accel_srp(sunlit, r_sun, 0.02, R)
    d = sunlit - r_sun
    assert np.linalg.norm(a) == pytest.approx(SOLAR_PRESSURE * 0.02 * AU**2 / (d @ d), rel=1e-12)
    assert a @ d > 0                                           # pushes away from the Sun
    assert accel_srp(np.array([-7e6, 0.0, 0.0]), r_sun, 0.02, R) == pytest.approx(0)   # behind Earth


def test_zonal_reduces_to_j2_and_is_a_potential_gradient():
    rng = np.random.default_rng(1)
    for _ in range(20):
        r = rng.normal(size=3) * 8e6
        assert accel_zonal(r, MU, R, J_EGM08[:1]) == pytest.approx(accel_j2(r, MU, R, J_EGM08[0]), rel=1e-12)

    def potential(r):
        rn = np.linalg.norm(r)
        s = r[2] / rn
        return sum(-MU * Jn * R**n / rn ** (n + 1) * legendre.legval(s, [0] * n + [1])
                   for n, Jn in enumerate(J_EGM08, start=2))

    for _ in range(10):
        r = rng.normal(size=3) * 8e6
        h = 0.5
        grad = np.array([(potential(r + h * e) - potential(r - h * e)) / (2 * h) for e in np.eye(3)])
        assert accel_zonal(r, MU, R, J_EGM08) == pytest.approx(grad, rel=1e-6)
    # On the pole the axis terms stay finite
    assert np.all(np.isfinite(accel_zonal(np.array([0.0, 0.0, 7e6]), MU, R, J_EGM08)))


def test_sun_mu_consistent():
    """The Sun's tidal acceleration at GEO is about 1.6e-6 m/s² at most (order-of-magnitude check)."""
    r = np.array([42164e3, 0.0, 0.0])
    a = accel_third_body(r, sun_position(2451545.0), MU_SUN)
    assert 5e-7 < np.linalg.norm(a) < 4e-6

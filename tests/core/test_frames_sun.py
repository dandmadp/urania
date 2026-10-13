import math

import numpy as np
import pytest

from urania.core import frames, lifetime, sun
from urania.core.propagate import hermite, kepler_propagate

R, F = 6378137.0, 1 / 298.257223563
MU = 3.986004418e14


def test_vallado_example_3_5_gmst():
    """Vallado example 3-5: 1992-08-20 12:14:00 UT1 → GMST = 152.578787810°."""
    jd = 2448855.009722222   # 1992-08-20 12:14:00
    assert math.degrees(frames.gmst82(jd)) == pytest.approx(152.578787810, abs=1e-8)


def test_gmst_rate_is_sidereal():
    """One sidereal day (86164.0905 s) brings GMST back to the same angle."""
    jd = 2461320.3
    d = (frames.gmst82(jd + 86164.0905 / 86400) - frames.gmst82(jd) + math.pi) % (2 * math.pi) - math.pi
    assert abs(d) < 1e-6
    arr = frames.gmst82(np.array([jd, jd + 0.5]))
    assert arr.shape == (2,)


def test_vallado_example_5_1_sun():
    """Vallado example 5-1: 2006-04-02 00:00 UT1 → r_sun = [146186178, 28788917, 12481064] km.

    The book evaluates the series with T_UT1; we use the same date. Direction agrees to about 1",
    distance to 1e-6.
    """
    r = sun.sun_position(2453827.5)
    ref = np.array([146186178.0, 28788917.0, 12481064.0]) * 1e3
    cos_angle = r @ ref / (np.linalg.norm(r) * np.linalg.norm(ref))
    assert math.degrees(math.acos(min(1.0, cos_angle))) * 3600 < 2.0
    assert np.linalg.norm(r) == pytest.approx(np.linalg.norm(ref), rel=1e-6)
    assert sun.sun_position(np.array([2453827.5, 2453828.5])).shape == (2, 3)


def test_rotate_z_round_trip():
    r = np.array([[7000e3, 1000e3, 2000e3], [-3000e3, 5000e3, -100e3]])
    theta = np.array([0.3, 2.0])
    back = frames.rotate_z(frames.rotate_z(r, theta), -theta)
    np.testing.assert_allclose(back, r, atol=1e-6)
    # θ = 90°: inertial x axis appears at Earth-fixed -y
    np.testing.assert_allclose(frames.rotate_z([1.0, 0.0, 0.0], math.pi / 2), [0.0, -1.0, 0.0], atol=1e-15)


@pytest.mark.parametrize("lat, lon, h", [(0.3, 2.0, 400e3), (-1.2, -3.0, 0.0), (math.pi / 2, 0.0, 1e6)])
def test_geodetic_round_trip(lat, lon, h):
    r = frames.geodetic_to_ecef(lat, lon, h, R, F)
    la, lo, hh = frames.geodetic(r, R, F)
    assert la == pytest.approx(lat, abs=1e-12) and hh == pytest.approx(h, abs=1e-6)
    if abs(lat) < math.pi / 2:
        assert lo == pytest.approx(lon, abs=1e-12)
    arr = frames.geodetic(np.array([r, r]), R, F)
    np.testing.assert_allclose(arr[2], [h, h], atol=1e-6)


def test_look_angles_geometry():
    site = frames.geodetic_to_ecef(0.6, 2.1, 100.0, R, F)
    up = np.array([math.cos(0.6) * math.cos(2.1), math.cos(0.6) * math.sin(2.1), math.sin(0.6)])
    north = np.array([-math.sin(0.6) * math.cos(2.1), -math.sin(0.6) * math.sin(2.1), math.cos(0.6)])
    east = np.array([-math.sin(2.1), math.cos(2.1), 0.0])
    az, el, rng = frames.look_angles(site + 500e3 * up, site, 0.6, 2.1)
    assert math.degrees(el) == pytest.approx(90.0) and rng == pytest.approx(500e3)
    az, el, _ = frames.look_angles(site + 1e6 * east, site, 0.6, 2.1)
    assert math.degrees(az) == pytest.approx(90.0) and el == pytest.approx(0.0, abs=1e-12)
    az, el, _ = frames.look_angles(site + 1e6 * north + 1e6 * up, site, 0.6, 2.1)
    assert math.degrees(az) == pytest.approx(0.0, abs=1e-9) and math.degrees(el) == pytest.approx(45.0)


def test_illumination_cases():
    """Sunlit on the day side, umbra on the night side, partial in between."""
    r_sun = np.array([1.496e11, 0.0, 0.0])
    assert sun.illumination([7000e3, 0, 0], r_sun, R) == 1.0
    assert sun.illumination([-7000e3, 0, 0], r_sun, R) == 0.0
    # just at the shadow cylinder edge the Sun is partly hidden
    edge = sun.illumination([-7000e3, R, 0], r_sun, R)
    assert 0.0 < edge < 1.0


def test_hermite_accuracy():
    """Midpoint of a 56 s LEO step: interpolated position within 1 m of the analytic one."""
    r0, v0 = np.array([6778e3, 0.0, 0.0]), np.array([0.0, 7668.6, 0.0])
    r1, v1 = kepler_propagate(r0, v0, 56.0, MU)
    rm, vm = kepler_propagate(r0, v0, 28.0, MU)
    ri, vi = hermite(28.0, 0.0, 56.0, r0, v0, r1, v1)
    assert np.linalg.norm(ri - rm) < 1.0
    assert np.linalg.norm(vi - vm) < 0.05
    np.testing.assert_allclose(hermite(56.0, 0.0, 56.0, r0, v0, r1, v1)[0], r1)


def test_mean_decay_rate_without_rotation():
    """Constant density, no rotation: da/dt = -ρ B √(μ a)."""
    rate = lifetime.mean_decay_rate(7000e3, 0.9, MU, 0.0, 0.02, lambda r, t: 1e-12)
    assert rate == pytest.approx(-1e-12 * 0.02 * math.sqrt(MU * 7000e3))


def test_decay_history_stops():
    res = lifetime.decay_history(6778e3, 0.9, MU, 0.0, 0.02, lambda r, t: 1e-11, 6478e3, 1e9)
    assert res.reached and res.a[-1] == pytest.approx(6478e3, abs=1.0)
    # analytic: da/dt = -ρ B √(μ a) integrates to 2(√a0 - √a) = ρ B √μ t
    t_expected = 2 * (math.sqrt(6778e3) - math.sqrt(6478e3)) / (1e-11 * 0.02 * math.sqrt(MU))
    assert res.t[-1] == pytest.approx(t_expected, rel=1e-5)


def test_azimuth_never_360():
    """A satellite due north must give azimuth 0, not 360° (rounding of a tiny negative angle)."""
    site = frames.geodetic_to_ecef(0.0, 0.0, 0.0, R, F)
    for eps in (0.0, -1e-17, -1e-12):
        az, _, _ = frames.look_angles(site + np.array([0.0, eps, 1e6]), site, 0.0, 0.0)
        assert 0.0 <= az < 2 * math.pi


def test_rotate_z_broadcasts():
    """One vector with a length-1 angle array used to fail with a shape error."""
    out = frames.rotate_z([7000e3, 0.0, 1000e3], np.array([0.5]))
    assert out.shape == (1, 3)
    np.testing.assert_allclose(out[0], frames.rotate_z([7000e3, 0.0, 1000e3], 0.5))

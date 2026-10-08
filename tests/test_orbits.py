import math

import astropy.units as u
import numpy as np
import pytest

from urania import GEO, ISS, LEO, SSO, Earth, Epoch, Mars, Orbit, sun_synchronous

DAY = 86400.0


def test_from_vectors_matches_core():
    """Vallado example 2-5 state vector given as Quantities."""
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
    {},                                  # no size
    {"a": 7e6, "p": 7e6},                # both
    {"a": 7e6, "ecc": 1.5},              # hyperbola with a > 0
    {"a": -7e6, "ecc": 0.5},             # ellipse with a < 0
    {"a": 7e6, "ecc": 1.0},              # parabola needs p
])
def test_from_elements_invalid(kwargs):
    with pytest.raises(ValueError):
        Orbit.from_elements(Earth, **kwargs)


def test_circular_and_altitudes():
    o = Orbit.circular(Earth, 400 * u.km, inc=51.6 * u.deg, arglat=45 * u.deg)
    assert o.ecc == pytest.approx(0.0, abs=1e-12)
    assert o.periapsis_altitude == pytest.approx(400e3)
    assert o.apoapsis_altitude == pytest.approx(400e3)
    # On a circular orbit ν is measured from the node (DECISIONS D2)
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
    # a = 6798.137 km → T = 2π√(a³/μ) = 5578.2 s (hand calculation)
    assert ISS.period / 60 == pytest.approx(92.97, abs=0.01)
    # The GEO period is one sidereal day
    assert GEO.period == pytest.approx(86164.09, abs=0.1)
    assert GEO.inc == 0.0
    # An SSO plane turns east by about 0.9856° per day
    assert math.degrees(SSO.raan_rate) * DAY == pytest.approx(0.98563, abs=1e-4)


def test_sun_synchronous_eccentric():
    o = sun_synchronous(600 * u.km, ecc=0.01)
    assert math.degrees(o.raan_rate) * DAY == pytest.approx(0.98563, abs=1e-4)


def test_rejects_negative_eccentricity():
    """ecc=-0.1 used to silently build an e=0.1 orbit (regression test)."""
    with pytest.raises(ValueError, match="non-negative"):
        Orbit.from_elements(Earth, a=7e6, ecc=-0.1)


def test_from_apsides():
    o = Orbit.from_apsides(Earth, 400 * u.km, 600 * u.km, inc=51.6 * u.deg)
    assert o.periapsis_altitude == pytest.approx(400e3)
    assert o.apoapsis_altitude == pytest.approx(600e3)
    assert math.degrees(o.inc) == pytest.approx(51.6)
    with pytest.raises(ValueError):
        Orbit.from_apsides(Earth, 600 * u.km, 400 * u.km)


def test_from_vectors_with_lists_of_quantities():
    o = Orbit.from_vectors(Earth, [6524.834 * u.km, 6862.875 * u.km, 6448.296 * u.km],
                           [4.901327 * u.km / u.s, 5.533756 * u.km / u.s, -1.976341 * u.km / u.s])
    assert o.ecc == pytest.approx(0.832853, abs=1e-6)


def test_after_is_final_orbit():
    a = ISS.after(days=1, model="j2")
    b = ISS.propagate(days=1, model="j2").final
    np.testing.assert_allclose(a.r, b.r, atol=1e-3)
    assert a.epoch == b.epoch


def test_after_raises_when_propagation_stops():
    with pytest.raises(RuntimeError, match="stopped early"):
        Orbit.circular(Earth, 130e3).after(days=5, model="drag", area=1, mass=1)


def test_after_catches_surface_crossing_between_endpoints():
    """A suborbital arc that dips below the surface and comes back up must not be missed."""
    sub = Orbit.from_elements(Earth, a=6000e3, ecc=0.2, nu=math.radians(180))
    with pytest.raises(RuntimeError, match="stopped early"):
        sub.after(sub.period)


def test_sun_synchronous_other_body_needs_rate():
    """Around Mars the Earth year used to be applied silently (regression test)."""
    with pytest.raises(ValueError, match="raan_rate"):
        sun_synchronous(300 * u.km, body=Mars)
    rate = 2 * math.pi / (686.98 * DAY)
    o = sun_synchronous(300 * u.km, body=Mars, raan_rate=rate)
    assert o.raan_rate == pytest.approx(rate)


def test_mean_anomaly_matches_kepler_equation():
    """M = E - e sin E with E from tan(ν/2) = √((1+e)/(1-e)) tan(E/2)."""
    o = Orbit.from_elements(Earth, a=10000e3, ecc=0.3, nu=math.radians(100))
    E = 2 * math.atan(math.sqrt(0.7 / 1.3) * math.tan(math.radians(50)))
    assert o.M == pytest.approx(E - 0.3 * math.sin(E))


def test_argp_rate_formula():
    """dω/dt = (3/4) n J2 (R/p)² (5 cos²i - 1), and it vanishes at the critical inclination."""
    o = Orbit.from_elements(Earth, a=8000e3, ecc=0.1, inc=math.radians(30))
    n = math.sqrt(Earth.mu / 8000e3**3)
    p = 8000e3 * (1 - 0.01)
    expected = 0.75 * n * Earth.J2 * (Earth.radius / p) ** 2 * (5 * math.cos(math.radians(30)) ** 2 - 1)
    assert o.argp_rate == pytest.approx(expected)
    crit = Orbit.from_elements(Earth, a=26600e3, ecc=0.74, inc=math.acos(math.sqrt(0.2)))
    assert crit.argp_rate == pytest.approx(0.0, abs=1e-20)


def test_repr_of_body_and_epoch():
    assert repr(Earth) == "Body('Earth', R=6378.1 km)"
    assert repr(Epoch.from_iso("2026-01-01T00:00:00")) == "Epoch('2026-01-01T00:00:00' TDB)"

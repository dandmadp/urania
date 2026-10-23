"""Patched-conic formulas (Curtis ch. 8) and the Standish planet ephemeris."""

import math

import numpy as np
import pytest

from urania.core import interplanetary as ip
from urania.core import planets as pl
from urania.core.maneuvers import hohmann
from urania.core.sun import sun_position

# Curtis constants (Table A.1): km and km³/s²
MU_SUN = 1.327e11
MU_EARTH, R_EARTH = 398600.0, 6378.0
MU_MARS, R_MARS = 42830.0, 3396.0
R_EO, R_MO = 149.6e6, 227.9e6
DAY = 86400.0


def test_curtis_example_8_1_synodic_period():
    """Curtis Example 8.1: Earth-Mars synodic period 2.135 years (T_Earth 365.256 d, T_Mars 1.8807 y)."""
    T_e, T_m = 365.256, 1.8807 * 365.256
    assert ip.synodic_period(T_e, T_m) / 365.256 == pytest.approx(2.135, abs=5e-4)


def test_curtis_example_8_3_departure():
    """Curtis Example 8.3: Earth → Mars from a 300 km parking orbit: v∞ = 2.943 km/s, Δv = 3.590 km/s."""
    v_inf, _, _ = ip.hohmann_v_inf(R_EO, R_MO, MU_SUN)
    assert v_inf == pytest.approx(2.943, abs=5e-4)
    assert ip.departure_dv(v_inf, R_EARTH + 300.0, MU_EARTH) == pytest.approx(3.590, abs=5e-4)


def test_curtis_example_8_4_capture():
    """Curtis Example 8.4: Mars arrival (v∞ = 2.648 km/s) into the Δv-optimal 7-hour orbit.

    a = 8832 km, rp = 5447 km, Δv = 1.470 km/s.
    """
    _, v_inf, _ = ip.hohmann_v_inf(R_EO, R_MO, MU_SUN)
    assert v_inf == pytest.approx(2.648, abs=5e-4)
    T = 7 * 3600.0
    a = (MU_MARS * (T / (2 * math.pi)) ** 2) ** (1 / 3)
    assert a == pytest.approx(8832, abs=0.5)
    # Continue with the book's rounded intermediate values, as Curtis does
    v_inf, a = 2.648, 8832.0
    e = 2 * MU_MARS / (a * v_inf**2) - 1          # Curtis eq. 8.70
    rp = ip.optimal_capture_periapsis(v_inf, MU_MARS, e)
    assert rp == pytest.approx(a * (1 - e))
    assert rp == pytest.approx(5447, abs=1)
    # 1.4705 from the rounded inputs; the book's last digit carries the rounding of e
    assert ip.capture_dv(v_inf, rp, MU_MARS, e) == pytest.approx(1.470, abs=1e-3)


def test_hohmann_phasing_earth_mars():
    """Curtis Example 8.2-style numbers: tof 259 days, phase 44.3°, wait at Mars 454 days."""
    _, _, tof = ip.hohmann_v_inf(R_EO, R_MO, MU_SUN)
    assert tof / DAY == pytest.approx(258.8, abs=0.1)
    assert math.degrees(ip.phase_angle(R_EO, R_MO, MU_SUN)) == pytest.approx(44.3, abs=0.05)
    assert ip.stay_time(R_EO, R_MO, MU_SUN) / DAY == pytest.approx(454, abs=1)


@pytest.mark.parametrize("r1,r2", [(1.0, 1.524), (1.0, 0.723), (5.2, 9.54), (30.07, 0.387), (1.0, 1.0001)])
def test_phase_angle_property(r1, r2):
    """The target, starting φ0 ahead, reaches the point opposite departure after tof."""
    mu = 1.0
    _, _, tof = hohmann(r1, r2, mu)
    phi0 = ip.phase_angle(r1, r2, mu)
    assert math.cos(phi0 + math.sqrt(mu / r2**3) * tof) == pytest.approx(-1.0, abs=1e-12)
    # At arrival the origin has moved n1·tof from the departure point
    phi_f = ip.arrival_phase_angle(r1, r2, mu)
    assert math.cos(phi_f + math.pi - math.sqrt(mu / r1**3) * tof) == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize("r1,r2", [(1.0, 1.524), (1.0, 0.723), (5.2, 9.54)])
def test_stay_time_property(r1, r2):
    """After the stay the origin leads the target by the return-leg phase angle."""
    mu = 1.0
    n1, n2 = math.sqrt(mu / r1**3), math.sqrt(mu / r2**3)
    stay = ip.stay_time(r1, r2, mu)
    assert 0 <= stay < ip.synodic_period(2 * math.pi / n1, 2 * math.pi / n2)
    lead = ip.arrival_phase_angle(r1, r2, mu) + (n1 - n2) * stay
    assert math.cos(lead - ip.phase_angle(r2, r1, mu)) == pytest.approx(1.0, abs=1e-12)


def test_wait_time():
    assert ip.wait_time(0.3, 0.3, 1.0, 2.0) == 0.0
    assert ip.wait_time(0.0, 1.0, 1.0, 2.0) == pytest.approx(1.0)
    assert ip.wait_time(0.0, 1.0, 2.0, 1.0) == pytest.approx(2 * math.pi - 1.0)
    assert ip.wait_time(0.0, 1.0, 1.0, 1.0) == math.inf
    assert ip.synodic_period(1.0, 1.0) == math.inf


def test_departure_and_capture_properties():
    mu, rp = 3.986e14, 6.678e6
    vc = math.sqrt(mu / rp)
    # v∞ = 0 is a parabolic escape: Δv = (√2 − 1) v_c
    assert ip.departure_dv(0.0, rp, mu) == pytest.approx((math.sqrt(2) - 1) * vc)
    # Circular capture is the departure burn reversed
    assert ip.capture_dv(3000.0, rp, mu) == pytest.approx(ip.departure_dv(3000.0, rp, mu))
    # A more eccentric capture orbit costs less
    assert ip.capture_dv(3000.0, rp, mu, 0.5) < ip.capture_dv(3000.0, rp, mu)
    with pytest.raises(ValueError):
        ip.capture_dv(3000.0, rp, mu, 1.0)
    e, beta = ip.hyperbola(3000.0, rp, mu)
    assert e == pytest.approx(1 + rp * 9e6 / mu)
    assert math.cos(beta) == pytest.approx(1 / e)


@pytest.mark.parametrize("ecc", [0.0, 0.3, 0.8])
def test_optimal_capture_periapsis_minimizes(ecc):
    """Curtis eq. 8.67: the optimum gives Δv = v∞√((1 − e)/2) (eq. 8.69) and neighbors cost more."""
    mu, v_inf = 4.283e13, 2650.0
    rp = ip.optimal_capture_periapsis(v_inf, mu, ecc)
    dv = ip.capture_dv(v_inf, rp, mu, ecc)
    assert dv == pytest.approx(v_inf * math.sqrt((1 - ecc) / 2))
    assert ip.capture_dv(v_inf, 0.9 * rp, mu, ecc) > dv
    assert ip.capture_dv(v_inf, 1.1 * rp, mu, ecc) > dv


def test_soi_radius_curtis_earth():
    """Curtis eq. 8.25 example: Earth's sphere of influence about 925 000 km."""
    assert ip.soi_radius(R_EO, 5.974e24, 1.989e30) == pytest.approx(925e3, rel=2e-3)


# ---------------------------------------------------------------- ephemeris

@pytest.mark.parametrize("days", [-150.0, -30.0, 0.0, 45.0, 170.0])
def test_earth_opposite_the_sun(days):
    """The Standish Earth-Moon barycenter lies opposite Vallado's Sun position (algorithm 29).

    Near J2000 the frames agree to within precession (50"/year) and the barycenter offset (0.002°).
    """
    jd = pl.JD_J2000 + days
    r, _ = pl.heliocentric_state("Earth", jd, 1.32712440041e20)
    r_eq = pl.ecliptic_to_equatorial(r)
    sun = np.asarray(sun_position(jd))
    cos = -(r_eq @ sun) / (np.linalg.norm(r_eq) * np.linalg.norm(sun))
    assert math.degrees(math.acos(min(cos, 1.0))) < 0.02
    assert np.linalg.norm(r_eq) == pytest.approx(np.linalg.norm(sun), rel=2e-4)


@pytest.mark.parametrize("name", list(pl.STANDISH))
def test_ephemeris_consistency(name):
    """Longitude matches the state vector; speed obeys vis-viva; elements stay bounded."""
    mu = 1.32712440041e20
    for jd in (2451545.0, 2461000.5, 2400000.5):
        r, v = pl.heliocentric_state(name, jd, mu)
        assert pl.ecliptic_longitude(name, jd) == pytest.approx(math.atan2(r[1], r[0]) % (2 * math.pi),
                                                                abs=1e-12)
        a, e, *_ = pl.mean_elements(name, jd)
        assert v @ v == pytest.approx(mu * (2 / np.linalg.norm(r) - 1 / a), rel=1e-10)
        assert 0 <= e < 0.25


def test_unknown_ephemeris():
    with pytest.raises(ValueError, match="No ephemeris"):
        pl.mean_elements("Pluto", 2451545.0)

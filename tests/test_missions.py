"""Lambert transfers between planets, gravity-assist missions and porkchop plots."""

import math
import warnings

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import pytest

import urania as ur
from urania.core.propagate import kepler_propagate

E = ur.Epoch.from_iso
DAY = 86400.0


def test_curtis_example_8_8_mars_global_surveyor():
    """Curtis Example 8.8: Earth 1996-11-07 → Mars 1997-09-12: v∞ 3.16 km/s at departure, 2.885 km/s at arrival.

    Curtis uses the 1992 edition of Standish's elements (his Table 8.1), urania the current one, so positions
    differ by up to 0.1% and v∞ by a few m/s.
    """
    m = ur.Earth.lambert_to(ur.Mars, E("1996-11-07T00:00:00"), E("1997-09-12T00:00:00"), 180 * u.km)
    leg = m.legs[0]
    assert leg.r1 / 1e3 == pytest.approx([1.0500e8, 1.0466e8, 988.33], rel=1e-3, abs=500)
    assert leg.r2 / 1e3 == pytest.approx([-2.0833e7, -2.1840e8, -3.6669e6], rel=2e-3, abs=5e5)
    assert m.v_inf_departure / 1e3 == pytest.approx(3.16, abs=0.01)
    assert m.v_inf_arrival / 1e3 == pytest.approx(2.885, abs=0.005)
    assert leg.tof / DAY == pytest.approx(309)


def test_mariner_10_venus_flyby():
    """Mariner 10 (launch 1973-11-03, Venus 1974-02-05, Mercury 1974-03-29): the Venus flyby passed 5768 km
    above the surface and was unpowered. The Lambert legs reproduce a nearly unpowered flyby at that altitude."""
    m = ur.flyby_mission([ur.Earth, ur.Venus, ur.Mercury],
                         [E("1973-11-03T17:45:00"), E("1974-02-05T17:01:00"), E("1974-03-29T20:47:00")],
                         185 * u.km)
    f = m.flybys[0]
    assert f.altitude / 1e3 == pytest.approx(5768, rel=0.02)
    assert f.dv < 50.0                      # m/s: essentially unpowered
    assert f.speed_gain < -4000             # Venus slowed the spacecraft toward Mercury
    assert m.feasible and m.sequence == (ur.Earth, ur.Venus, ur.Mercury)
    assert m.total_dv == pytest.approx(m.dv_departure + f.dv + m.dv_arrival)


def test_legs_connect_planets():
    m = ur.flyby_mission([ur.Earth, ur.Venus, ur.Mercury],
                         [E("1973-11-03T17:45:00"), E("1974-02-05T17:01:00"), E("1974-03-29T20:47:00")],
                         185 * u.km, 200 * u.km)
    for leg in m.legs:
        r, v = kepler_propagate(leg.r1, leg.v1, leg.tof, ur.Sun.mu)
        assert np.linalg.norm(r - leg.r2) < 1.0
        assert np.linalg.norm(v - leg.v2) < 1e-6
    f = m.flybys[0]
    assert f.v_inf_in == pytest.approx(m.legs[0].v_inf_in)
    assert f.v_inf_out == pytest.approx(m.legs[1].v_inf_out)
    assert m.dv_arrival > 0 and m.capture_altitude == 200e3


def test_lambert_optimum_near_hohmann_estimate():
    """The best Lambert C3 around the Hohmann window is close to the Hohmann C3.

    An exact 180° transfer is not the optimum: Mars's 1.85° inclination makes a 180° arc nearly polar.
    """
    h = ur.Earth.hohmann_to(ur.Mars, 300 * u.km, after=E("2026-01-01T00:00:00"))
    d = h.departure_epoch
    p = ur.porkchop(ur.Earth, ur.Mars, (d + -40 * DAY, d + 40 * DAY), (d + 180 * DAY, d + 340 * DAY), step=4 * u.day)
    assert np.nanmin(p.c3) == pytest.approx(h.c3, rel=0.15)


def test_infeasible_flyby_flagged():
    m = ur.flyby_mission([ur.Earth, ur.Venus, ur.Mars],
                         [E("2031-06-01T00:00:00"), E("2031-11-01T00:00:00"), E("2032-06-01T00:00:00")],
                         300 * u.km)
    assert not m.feasible
    assert "infeasible" in repr(m)
    assert "Warnings" in str(m.explain())


def test_unpowered_flyby_object():
    v_in = [10e3, 0, 0]
    f = ur.Jupiter.flyby(v_in * u.m / u.s, 200000 * u.km, epoch=E("2030-01-01T00:00:00"))
    assert np.linalg.norm(f.v_inf_out) == pytest.approx(1e4)
    rp = ur.Jupiter.radius + 2e8
    assert f.turn_angle == pytest.approx(2 * math.asin(1 / (1 + rp * 1e8 / ur.Jupiter.mu)))
    assert f.dv == 0 and f.feasible
    # The pole-side choice flips the turn direction
    g = ur.Jupiter.flyby(v_in, 2e8, b_angle=math.pi)
    assert np.cross(f.v_inf_in, f.v_inf_out)[2] > 0 > np.cross(g.v_inf_in, g.v_inf_out)[2]
    assert g.speed_gain is None
    with pytest.raises(ValueError, match="3-vector"):
        ur.Jupiter.flyby([1.0, 2.0], 1e7)


def test_porkchop_2026_mars():
    p = ur.porkchop(ur.Earth, ur.Mars, (E("2026-08-01T00:00:00"), E("2027-01-01T00:00:00")),
                    (E("2027-04-01T00:00:00"), E("2028-03-01T00:00:00")), step=10 * u.day,
                    parking_altitude=300 * u.km, capture_altitude=300 * u.km)
    assert p.c3.shape == (len(p.arrival_jd), len(p.departure_jd))
    best = p.best()
    assert best.c3 == pytest.approx(np.nanmin(p.c3), rel=1e-9)
    assert 8e6 < best.c3 < 12e6                                 # m²/s²: a typical Mars opportunity
    assert "2026-10" <= best.departure.iso[:7] <= "2026-12"
    i, j = np.unravel_index(np.nanargmin(p.total_dv), p.total_dv.shape)
    assert p.best("total_dv").total_dv == pytest.approx(p.total_dv[i, j], rel=1e-9)
    assert "Porkchop(Earth → Mars" in repr(p)
    with pytest.raises(ValueError, match="criterion"):
        p.best("speed")


def test_porkchop_skips_short_transfers():
    p = ur.porkchop(ur.Earth, ur.Venus, (E("2026-01-01T00:00:00"), E("2026-01-21T00:00:00")),
                    (E("2026-01-01T00:00:00"), E("2026-03-01T00:00:00")), step=10 * u.day)
    assert np.isnan(p.c3[0, 0])
    assert p.total_dv is None
    with pytest.raises(ValueError, match="criterion"):
        p.best("total_dv")


def test_mission_errors():
    d = [E("2026-11-01T00:00:00"), E("2027-08-01T00:00:00")]
    with pytest.raises(ValueError, match="ephemeris"):
        ur.flyby_mission([ur.Earth, ur.Moon], d, 300 * u.km)
    with pytest.raises(ValueError, match="increase"):
        ur.flyby_mission([ur.Earth, ur.Mars], d[::-1], 300 * u.km)
    with pytest.raises(ValueError, match="one date per body"):
        ur.flyby_mission([ur.Earth, ur.Mars], d[:1], 300 * u.km)
    with pytest.raises(TypeError, match="as a time"):
        ur.flyby_mission([ur.Earth, ur.Mars], [20261101, d[1]], 300 * u.km)
    with pytest.raises(ValueError, match="one value per leg"):
        ur.flyby_mission([ur.Earth, ur.Mars], d, 300 * u.km, revolutions=[0, 1])
    with pytest.raises(TypeError, match="start, end"):
        ur.porkchop(ur.Earth, ur.Mars, d[0], d, step=10 * u.day)


def test_propellant():
    m = ur.Earth.lambert_to(ur.Mars, E("1996-11-07T00:00:00"), 309 * u.day, 180 * u.km, 300 * u.km)
    mp = m.propellant(1000 * u.kg, 320 * u.s)
    assert mp == pytest.approx(1000 * (1 - math.exp(-m.total_dv / (320 * 9.80665))))
    h = ur.Earth.hohmann_to(ur.Mars, 300 * u.km, 300 * u.km)
    assert 0 < h.propellant(1000, 320) < 1000


def test_explain_and_plot():
    m = ur.flyby_mission([ur.Earth, ur.Venus, ur.Mercury],
                         [E("1973-11-03T17:45:00"), E("1974-02-05T17:01:00"), E("1974-03-29T20:47:00")],
                         185 * u.km, 300 * u.km)
    text = str(m.explain())
    assert "Flyby of Venus" in text and "Total" in text and "Warnings" not in text
    p = ur.porkchop(ur.Earth, ur.Mars, (E("2026-09-01T00:00:00"), E("2026-12-01T00:00:00")),
                    (E("2027-06-01T00:00:00"), E("2027-12-01T00:00:00")), step=15 * u.day)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        m.plot()
        p.plot()
        str(m.flybys[0].explain())
    plt.close("all")

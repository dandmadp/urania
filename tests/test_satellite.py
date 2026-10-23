"""Satellite operations: rendezvous, phasing, deorbit, relative motion, Doppler, footprint, beta angle."""

import math
import warnings

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import pytest

import urania as ur
from urania.core import relative as rel
from urania.core.propagate import kepler_propagate

INC = 51.64 * u.deg


def test_rendezvous_reaches_target():
    chaser = ur.Orbit.circular(ur.Earth, 400 * u.km, inc=INC)
    target = ur.Orbit.from_apsides(ur.Earth, 410 * u.km, 450 * u.km, inc=52 * u.deg, raan=0.5 * u.deg,
                                   nu=-40 * u.deg)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        t = chaser.rendezvous(target, 100 * u.min, wait=10 * u.min)
    assert t.kind == "lambert" and len(t.burns) == 2
    assert t.coast == pytest.approx(600.0) and t.tof == pytest.approx(6000.0)
    arc = t.orbits[0]
    r, v = kepler_propagate(arc.r, arc.v, t.tof, ur.Earth.mu)
    rt, vt = kepler_propagate(target.r, target.v, t.burns[1].epoch - target.epoch, ur.Earth.mu)
    assert r == pytest.approx(rt, abs=1e-3)
    assert v + t.burns[1].dv == pytest.approx(vt, abs=1e-6)
    assert t.final.r == pytest.approx(rt)
    assert t.plane_change > 0


def test_lambert_rendezvous_matches_cw():
    """At 2 km separation the Lambert rendezvous and the linear CW rendezvous agree to about 1%."""
    target = ur.Orbit.circular(ur.Earth, 400 * u.km, inc=INC)
    n = target.mean_motion
    rho0 = np.array([0.0, -2000.0, 0.0])                   # 2 km behind, at rest in the LVLH frame
    r0, v0 = rel.from_lvlh(target.r, target.v, rho0, np.zeros(3))
    chaser = ur.Orbit(ur.Earth, r0, v0, target.epoch)
    tof = 0.4 * target.period
    t = chaser.rendezvous(target, tof)
    dv1, dv2 = rel.cw_rendezvous(rho0, np.zeros(3), tof, n)
    cw_total = np.linalg.norm(dv1) + np.linalg.norm(dv2)
    assert t.total_dv == pytest.approx(cw_total, rel=0.02)


def test_rendezvous_warns_through_the_earth():
    chaser = ur.Orbit.circular(ur.Earth, 400 * u.km, inc=INC)
    target = ur.Orbit.circular(ur.Earth, 420 * u.km, inc=INC, arglat=30 * u.deg)
    with pytest.warns(UserWarning, match="below the surface"):
        chaser.rendezvous(target, 50 * u.min)


def test_phasing_moves_along_the_orbit():
    """After the phasing maneuver the spacecraft sits Δθ ahead of where it would have been."""
    o = ur.Orbit.circular(ur.Earth, 500 * u.km, inc=INC, raan=20 * u.deg)
    for angle, k in ((15.0, 1), (-40.0, 2), (90.0, 8)):
        t = o.phasing(angle * u.deg, revolutions=k)
        ref, _ = kepler_propagate(o.r, o.v, t.tof, ur.Earth.mu)
        got = t.final.r
        h = np.cross(o.r, o.v)
        lead = math.atan2(np.cross(ref, got) @ h / np.linalg.norm(h), ref @ got)
        assert math.degrees(lead) == pytest.approx(angle, abs=1e-6)
        assert t.final.ecc < 1e-9
        assert len(t.burns) == 2 and t.burns[0].magnitude == pytest.approx(t.burns[1].magnitude)
        assert "Warnings" not in str(t.explain())


def test_phasing_with_target_and_limits():
    o = ur.Orbit.circular(ur.Earth, 500 * u.km, inc=INC)
    target = ur.Orbit.circular(ur.Earth, 500 * u.km, inc=INC, arglat=25 * u.deg)
    t = o.phasing(target=target, revolutions=3)
    rt, _ = kepler_propagate(target.r, target.v, t.final.epoch - target.epoch, ur.Earth.mu)
    assert t.final.r == pytest.approx(rt, abs=1e-3)
    with pytest.raises(ValueError, match="below the surface"):
        o.phasing(170 * u.deg, revolutions=1)
    with pytest.raises(ValueError, match="circular"):
        ur.Orbit.from_apsides(ur.Earth, 300 * u.km, 900 * u.km).phasing(10 * u.deg)
    with pytest.raises(ValueError, match="same circular orbit"):
        o.phasing(target=ur.Orbit.circular(ur.Earth, 520 * u.km, inc=INC))
    with pytest.raises(ValueError, match="not both"):
        o.phasing(10 * u.deg, target=target)


def test_deorbit():
    t = ur.LEO.deorbit(60 * u.km)
    assert t.final.periapsis_altitude == pytest.approx(60e3, abs=1e-3)
    r = np.linalg.norm(ur.LEO.r)
    rp = ur.Earth.radius + 60e3
    expected = math.sqrt(ur.Earth.mu / r) - math.sqrt(ur.Earth.mu * (2 / r - 2 / (r + rp)))
    assert t.total_dv == pytest.approx(expected, rel=1e-9)
    ecc = ur.Orbit.from_apsides(ur.Earth, 400 * u.km, 1200 * u.km, nu=10 * u.deg)
    d = ecc.deorbit()
    assert np.linalg.norm(d.burns[0].r) == pytest.approx(ecc.r_apoapsis, rel=1e-9)
    assert d.final.periapsis_altitude == pytest.approx(50e3, abs=1e-3)
    with pytest.raises(ValueError, match="not below"):
        ur.LEO.deorbit(800 * u.km)
    d0 = ur.LEO.deorbit()
    assert d0.propellant(4 * u.kg, 220 * u.s) == pytest.approx(4 * (1 - math.exp(-d0.total_dv / (220 * 9.80665))))


def test_relative_motion_after_deployment():
    """A CubeSat released at 0.5 m/s drifts as the CW equations predict (to second order)."""
    iss = ur.Orbit.circular(ur.Earth, 410 * u.km, inc=INC)
    r, v = rel.from_lvlh(iss.r, iss.v, np.zeros(3), np.array([-0.5, -0.3, 0.1]))
    cubesat = ur.Orbit(ur.Earth, r, v, iss.epoch)
    rm = cubesat.propagate(iss.period).relative_to(iss)
    cw, _ = rel.cw_propagate(np.zeros(3), np.array([-0.5, -0.3, 0.1]), iss.mean_motion, rm.t)
    err = np.linalg.norm(rm.r - cw, axis=1)
    assert np.all(err <= np.maximum(rm.distance, 1.0) ** 2 / np.linalg.norm(iss.r) + 1e-3)
    assert rm.distance[0] == pytest.approx(0, abs=1e-6)
    rho, rho_dot = cubesat.relative_to(iss)
    assert rho == pytest.approx(0, abs=1e-6) and rho_dot == pytest.approx([-0.5, -0.3, 0.1], abs=1e-9)
    # Chief given as a trajectory gives the same answer
    rm2 = cubesat.propagate(iss.period).relative_to(iss.propagate(iss.period, n_points=2000))
    assert rm2.r == pytest.approx(rm.r, abs=0.05)
    with pytest.raises(ValueError, match="Central bodies"):
        ur.Orbit.circular(ur.Mars, 300 * u.km).relative_to(iss)


def test_doppler_iss():
    iss = ur.TLEOrbit.from_text("""ISS (ZARYA)
1 25544U 98067A   19343.69339541  .00001764  00000-0  38792-4 0  9991
2 25544  51.6439 211.2001 0007417  17.6667  85.6398 15.50103472202482""")
    tr = iss.propagate(days=1, n_points=86401)
    station = ur.GroundStation.create("Daejeon", 36.37 * u.deg, 127.36 * u.deg, 70 * u.m)
    d = tr.doppler(station, 437.5 * u.MHz)
    # Range rate equals the derivative of the range (1 s samples)
    rr_fd = np.gradient(d.range, d.t)
    assert np.max(np.abs(rr_fd - d.range_rate)[1:-1]) < 0.5
    up = d.elevation > 0
    assert 9.5e3 < np.abs(d.shift[up]).max() < 10.2e3          # about f·v/c for 7 km/s
    assert np.all(np.sign(d.shift[up & (np.abs(d.range_rate) > 100)])
                  == -np.sign(d.range_rate[up & (np.abs(d.range_rate) > 100)]))
    # Elevation agrees with the pass report
    rep = tr.passes(station, min_elevation=0.0)
    peak = rep[0]
    k = np.argmin(np.abs(d.t - (peak.tca - tr.epoch)))
    assert math.degrees(d.elevation[k]) == pytest.approx(math.degrees(peak.max_elevation), abs=0.05)
    assert d.range[k] == pytest.approx(peak.min_range, rel=1e-3)
    assert "kHz" in repr(d)


def test_footprint_and_beta():
    o = ur.Orbit.circular(ur.Earth, 500 * u.km, inc=97.4 * u.deg)
    f0, f10 = o.footprint(), o.footprint(10 * u.deg)
    assert f10.radius < f0.radius
    assert f0.central_angle == pytest.approx(math.acos(ur.Earth.radius / (ur.Earth.radius + 500e3)))
    s = o.footprint(half_angle=20 * u.deg)
    assert s.nadir_angle == pytest.approx(math.radians(20)) and s.radius < f10.radius
    assert s.area == pytest.approx(2 * math.pi * ur.Earth.radius**2 * (1 - math.cos(s.central_angle)))


def test_beta_angle_solstice_equatorial():
    """An equatorial orbit at the June solstice has β ≈ the Sun's declination (+23.44°)."""
    epoch = ur.Epoch.from_iso("2026-06-21T08:24:00")
    o = ur.Orbit.circular(ur.Earth, 500 * u.km, epoch=epoch)
    assert math.degrees(o.beta_angle()) == pytest.approx(23.44, abs=0.02)
    tr = o.propagate(o.period)
    assert np.degrees(tr.beta_angle) == pytest.approx(23.44, abs=0.02)


def test_eclipse_fraction_matches_numerical():
    """The cylindrical-shadow fraction agrees with the conical eclipse search to about 1% of the orbit."""
    for o in (ur.ISS, ur.Orbit.circular(ur.Earth, 600 * u.km, inc=97.8 * u.deg, raan=60 * u.deg)):
        tr = o.propagate(o.period, n_points=2000)
        numerical = tr.eclipses().shadow_fraction
        assert o.eclipse_fraction() == pytest.approx(numerical, abs=0.01)


def test_plots_and_explain():
    chaser = ur.Orbit.circular(ur.Earth, 400 * u.km, inc=INC)
    target = ur.Orbit.circular(ur.Earth, 420 * u.km, inc=INC, arglat=-20 * u.deg)
    station = ur.GroundStation.create("Daejeon", 36.37 * u.deg, 127.36 * u.deg)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for t in (chaser.rendezvous(target, 70 * u.min), chaser.phasing(10 * u.deg), chaser.deorbit()):
            assert "Total" in str(t.explain())
            t.plot()
        r, v = rel.from_lvlh(chaser.r, chaser.v, np.zeros(3), np.array([0.0, 0.2, 0.0]))
        ur.Orbit(ur.Earth, r, v, chaser.epoch).propagate(chaser.period).relative_to(chaser).plot()
        chaser.propagate(days=1).doppler(station, 145.8e6).plot(min_elevation=5 * u.deg)
    plt.close("all")

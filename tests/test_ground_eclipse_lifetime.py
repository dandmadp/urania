"""Ground tracks, passes, eclipses and lifetime. Skyfield is the external reference where available."""

import math
from datetime import timezone

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import pytest

from urania import (ISS, LEO, SSO, Earth, Epoch, GroundStation, Mars, Orbit, TLEOrbit)

L1 = "1 25544U 98067A   19343.69339541  .00001764  00000-0  38792-4 0  9991"
L2 = "2 25544  51.6439 211.2001 0007417  17.6667  85.6398 15.50103472202482"
DAEJEON = GroundStation.create("Daejeon", 36.37 * u.deg, 127.36 * u.deg, 70 * u.m)


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


@pytest.fixture(scope="module")
def iss_tle_track():
    return TLEOrbit(L1, L2).propagate(days=3)


def _utc(epoch):
    return epoch.to_astropy().utc.datetime.replace(tzinfo=timezone.utc)


# ---------------------------------------------------------------- ground track

def test_ground_track_matches_skyfield():
    """Sub-satellite points of the ISS TLE agree with Skyfield: latitude and height exactly,
    longitude within UT1 - UTC (about 0.18 s in December 2019 → 0.0008°)."""
    sf = pytest.importorskip("skyfield.api")
    ts = sf.load.timescale(builtin=True)
    sat = sf.EarthSatellite(L1, L2, "ISS", ts)
    tr = TLEOrbit(L1, L2).propagate(days=1, n_points=97)
    gt = tr.ground_track()
    t = ts.from_datetimes([_utc(e) for e in tr.epochs])
    sp = sf.wgs84.subpoint(sat.at(t))
    np.testing.assert_allclose(np.degrees(gt.lat), sp.latitude.degrees, atol=1e-4)
    dlon = (np.degrees(gt.lon) - sp.longitude.degrees + 180) % 360 - 180
    assert np.abs(dlon).max() < 0.002
    np.testing.assert_allclose(gt.alt, sp.elevation.m, atol=1.0)


def test_ground_track_of_equatorial_orbit():
    gt = LEO.propagate(days=0.5).ground_track()
    assert np.abs(gt.lat).max() < 1e-9
    assert np.all(np.abs(gt.alt - 500e3) < 1.0)          # equator: ellipsoid height = |r| - R
    assert "GroundTrack(" in repr(gt)
    ax = gt.plot(stations=[DAEJEON])
    assert ax.get_xlim() == (-180, 180)


def test_ground_track_needs_earth():
    with pytest.raises(ValueError, match="Earth"):
        Orbit.circular(Mars, 300e3).propagate(days=0.1).ground_track()


# ---------------------------------------------------------------- passes

@pytest.mark.parametrize("name, lat, lon, h, count", [
    ("Daejeon", 36.37, 127.36, 70, 14), ("Quito", -0.18, -78.47, 2850, 7),
    ("Hobart", -42.88, 147.33, 10, 18), ("Svalbard", 78.23, 15.39, 500, 0),
])
def test_passes_match_skyfield(iss_tle_track, name, lat, lon, h, count):
    """Same passes as Skyfield's find_events, checked two ways:

    - Skyfield's own elevation at urania's rise and set times is the minimum elevation within 0.005°
      (independent of how precisely each library locates events);
    - event times agree within 2 s and the peak elevation within 0.02°. With Skyfield 1.55 the worst
      differences over these 39 passes were 0.23 s and 0.0085°; Skyfield 1.45 locates events less
      precisely (one set time 1.65 s off, at 9.835° elevation, while urania's time gives 9.9996°).
    """
    st = GroundStation.create(name, lat * u.deg, lon * u.deg, h * u.m)
    report = iss_tle_track.passes(st, min_elevation=10 * u.deg)
    assert len(report) == count
    sf = pytest.importorskip("skyfield.api")
    ts = sf.load.timescale(builtin=True)
    sat = sf.EarthSatellite(L1, L2, "ISS", ts)
    topo = sf.wgs84.latlon(lat, lon, h)
    tr = iss_tle_track
    times, events = sat.find_events(topo, ts.from_datetime(_utc(tr.epoch)),
                                    ts.from_datetime(_utc(tr.epoch + float(tr.t[-1]))), altitude_degrees=10.0)
    sky, cur = [], {}
    for ti, e in zip(times, events):
        cur[int(e)] = ti
        if e == 2 and 0 in cur and 1 in cur:
            sky.append((cur[0], cur[1], ti, (sat - topo).at(cur[1]).altaz()[0].degrees))
            cur = {}
    assert len(sky) == count
    for p, (aos, tca, los, el) in zip(report, sky):
        for edge in (p.aos, p.los):
            sky_el = (sat - topo).at(ts.from_datetime(_utc(edge))).altaz()[0].degrees
            assert abs(sky_el - 10.0) < 0.005
        assert abs((_utc(p.aos) - aos.utc_datetime()).total_seconds()) < 2.0
        assert abs((_utc(p.tca) - tca.utc_datetime()).total_seconds()) < 2.0
        assert abs((_utc(p.los) - los.utc_datetime()).total_seconds()) < 2.0
        assert abs(math.degrees(p.max_elevation) - el) < 0.02


def test_pass_report_details(iss_tle_track):
    report = iss_tle_track.passes(DAEJEON)
    p = report[0]
    assert p.aos < p.tca < p.los
    assert 10 <= math.degrees(p.max_elevation) <= 90
    assert p.duration == pytest.approx(p.los - p.aos)
    assert report.total_contact == pytest.approx(sum(x.duration for x in report))
    assert "passes over Daejeon" in repr(report)
    text = str(report.explain())
    assert "GMST" in text and "elevation = asin" in text and "UT1 ≈ UTC" in text


def test_overhead_pass_reaches_high_elevation():
    """A station under an equatorial orbit sees passes up to nearly 90°."""
    quito_eq = GroundStation.create("Equator", 0 * u.deg, 0 * u.deg)
    report = LEO.propagate(days=1).passes(quito_eq, min_elevation=0 * u.deg)
    assert len(report) > 0
    assert max(math.degrees(p.max_elevation) for p in report) > 80


def test_truncated_pass_at_start():
    """A trajectory that starts while the satellite is visible gives a truncated first pass."""
    tr = TLEOrbit(L1, L2).propagate(days=1)
    first = tr.passes(DAEJEON)[0]
    start = first.tca - 30.0
    report = TLEOrbit(L1, L2).propagate(3600, start=start).passes(DAEJEON)
    assert report[0].truncated and report[0].aos == start


def test_passes_errors():
    with pytest.raises(ValueError, match="forward"):
        ISS.propagate(days=-0.5).passes(DAEJEON)
    with pytest.raises(ValueError):
        GroundStation.create("Bad", 95 * u.deg, 0 * u.deg)


def test_station_repr_and_position():
    assert "Daejeon" in repr(DAEJEON) and "36.3700°" in repr(DAEJEON)
    assert np.linalg.norm(DAEJEON.r_ecef) == pytest.approx(6.37e6, rel=0.01)


# ---------------------------------------------------------------- eclipses

def _cylindrical_shadow_fraction(alt, beta):
    """Fraction of a circular orbit in a cylindrical shadow: acos(√(h² + 2Rh) / ((R+h) cos β)) / π."""
    R = Earth.radius
    x = math.sqrt(alt**2 + 2 * R * alt) / ((R + alt) * math.cos(beta))
    return math.acos(x) / math.pi if x < 1 else 0.0


def test_eclipse_duration_matches_cylindrical_estimate():
    """ISS-like orbit: umbra ≤ cylindrical estimate ≤ penumbra + umbra (conical shadow brackets it)."""
    tr = ISS.propagate(days=1, model="j2")
    report = tr.eclipses()
    e = report[1]
    # β from the orbit normal and the Sun direction at the eclipse time
    from urania.core.sun import sun_position

    r, v = tr.state_at(e.penumbra_start)
    h = np.cross(r, v)
    s = sun_position(e.penumbra_start.jd)
    beta = math.asin(h @ s / (np.linalg.norm(h) * np.linalg.norm(s)))
    cyl = _cylindrical_shadow_fraction(420e3, beta) * ISS.period
    assert e.umbra_duration <= cyl <= e.duration
    assert e.duration - e.umbra_duration < 30.0          # penumbra lasts seconds in LEO
    assert 0.3 < report.shadow_fraction < 0.45
    assert report.illumination.min() == 0.0 and report.illumination.max() == 1.0


def test_dawn_dusk_orbit_has_no_eclipse():
    """A sun-synchronous orbit whose plane faces the Sun (β ≈ 90°) stays in sunlight."""
    from urania.core.sun import sun_position

    epoch = Epoch.from_iso("2026-06-21T00:00:00")
    s = sun_position(epoch.jd)
    raan = math.atan2(s[1], s[0]) - math.pi / 2       # node 90° behind the Sun
    o = Orbit.circular(Earth, 800e3, inc=SSO.inc, raan=raan, epoch=epoch)
    report = o.propagate(days=0.5).eclipses()
    assert len(report) == 0 and report.shadow_fraction == 0.0
    assert "no eclipse" in str(report.explain())


def test_eclipse_report_api():
    report = ISS.propagate(days=0.5).eclipses()
    assert len(report) == len(list(report)) >= 7
    assert report[0].penumbra_start < report[0].umbra_start < report[0].umbra_end < report[0].penumbra_end
    assert "eclipses over" in repr(report) and "Eclipse(" in repr(report[0])
    text = str(report.explain())
    assert "penumbra: c < a + b" in text and "Sun position" in text
    assert report.plot().get_ylabel() == "sunlit fraction"


def test_eclipse_needs_earth():
    with pytest.raises(ValueError, match="Earth"):
        Orbit.circular(Mars, 300e3).propagate(days=0.1).eclipses()


# ---------------------------------------------------------------- lifetime

CUBESAT = {"area": 0.03 * u.m**2, "mass": 4 * u.kg}


def test_lifetime_averaged_matches_full_propagation():
    """350 km 3U CubeSat: orbit-averaged estimate within 3% of the full j2+drag propagation (74.9 days)."""
    o = Orbit.circular(Earth, 350 * u.km, inc=51.6 * u.deg)
    fast = o.lifetime(**CUBESAT)
    assert fast.reentered and fast.days == pytest.approx(74.9, rel=0.03)
    assert fast.reentry - o.epoch == pytest.approx(fast.days * 86400)
    assert fast.altitude[0] < 350e3 - 4e3                   # mean altitude is ~6 km below the osculating one


def test_lifetime_numerical_and_limits():
    o = Orbit.circular(Earth, 250 * u.km, inc=51.6 * u.deg)
    num = o.lifetime(**CUBESAT, method="numerical", max_days=30)
    fast = o.lifetime(**CUBESAT)
    assert num.reentered and num.days == pytest.approx(fast.days, rel=0.05)
    high = Orbit.circular(Earth, 800 * u.km).lifetime(**CUBESAT, max_days=30)
    assert not high.reentered and high.days == pytest.approx(30.0)
    assert "more than" in repr(high)


def test_lifetime_scales_with_density():
    o = Orbit.circular(Earth, 350 * u.km, inc=51.6 * u.deg)
    base = o.lifetime(**CUBESAT)
    dense = o.lifetime(**CUBESAT, density=lambda r, t: 2 * Earth.atmosphere(r, t))
    assert dense.days == pytest.approx(base.days / 2, rel=0.05)


def test_lifetime_errors_and_output():
    ell = Orbit.from_apsides(Earth, 300 * u.km, 2000 * u.km)
    with pytest.raises(ValueError, match="numerical"):
        ell.lifetime(**CUBESAT)
    with pytest.raises(ValueError, match="positive"):
        ISS.lifetime(area=-1, mass=1)
    with pytest.raises(ValueError, match="atmosphere"):
        Orbit.circular(Mars, 300e3).lifetime(**CUBESAT)
    life = Orbit.circular(Earth, 350 * u.km).lifetime(**CUBESAT)
    text = str(life.explain())
    assert "da/dt" in text and "solar activity" in text
    assert life.plot().get_ylabel() == "altitude [km]"


# ---------------------------------------------------------------- usability

def test_concise_reprs():
    tr = ISS.propagate(days=1)
    assert repr(tr).startswith("Trajectory(Earth, twobody, ") and "\n" not in repr(tr)
    burn = ISS.transfer_to(LEO).burns[0]
    assert repr(burn).startswith("Burn(burn 1") and "km/s" in repr(burn)


def test_quantities_and_summary():
    assert ISS.qty.a.unit == u.km and ISS.qty.a.value == pytest.approx(6798.137)
    assert ISS.qty.inc.to_value(u.deg) == pytest.approx(51.64)
    assert ISS.qty.period.unit == u.min
    with pytest.raises(AttributeError):
        ISS.qty.bogus
    text = ISS.summary()
    assert "periapsis_altitude  420 km" in text and "raan_rate" in text
    hyp = Orbit.from_elements(Earth, a=-20000e3, ecc=1.5)
    assert "period" not in hyp.summary()                    # undefined values are skipped


def test_state_at():
    tr = ISS.propagate(days=0.2, model="j2")
    r, v = tr.state_at(tr.t[3])
    np.testing.assert_allclose(r, tr.r[3])
    mid = 0.5 * (tr.t[3] + tr.t[4])
    exact = ISS.propagate(mid, model="j2", n_points=2).final
    r, v = tr.state_at(mid)
    assert np.linalg.norm(r - exact.r) < 1.0
    r2, _ = tr.state_at(tr.epoch + mid)
    np.testing.assert_allclose(r2, r)
    with pytest.raises(ValueError, match="outside"):
        tr.state_at(-10.0)
    back = ISS.propagate(days=-0.1)
    rb, _ = back.state_at(back.t[5])
    np.testing.assert_allclose(rb, back.r[5])

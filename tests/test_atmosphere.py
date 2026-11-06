"""NRLMSISE-00 atmosphere (pymsis)."""

import math

import numpy as np
import pytest

import urania as ur

pymsis = pytest.importorskip("pymsis")


def test_matches_pymsis_at_independent_coordinates():
    """Density equals pymsis evaluated at the geodetic point computed independently by astropy (ITRS)."""
    import astropy.units as u
    from astropy.coordinates import GCRS, ITRS, CartesianRepresentation
    from pymsis import msis

    model = ur.NRLMSISE00(140, 135, 10)
    rng = np.random.default_rng(5)
    for _ in range(6):
        epoch = ur.Epoch.from_jd(2451545 + rng.uniform(0, 9000))
        r = rng.normal(size=3)
        r *= (6378137 + rng.uniform(200e3, 800e3)) / np.linalg.norm(r)
        t = epoch.to_astropy()
        # urania's frame is the true equator of date; TETE is astropy's equivalent
        from astropy.coordinates import TETE

        itrs = TETE(CartesianRepresentation(*r, unit=u.m), obstime=t).transform_to(ITRS(obstime=t))
        loc = itrs.earth_location
        ref = msis.calculate(np.datetime64(t.utc.datetime, "us"), loc.lon.deg, loc.lat.deg, loc.height.to_value(u.km),
                             f107s=[140], f107as=[135], aps=[[10] * 7], version=0)
        assert model(r, epoch.tdb_seconds) == pytest.approx(float(np.ravel(ref)[0]), rel=2e-3)
        lat, lon, h = model.geodetic(r, epoch.tdb_seconds)
        assert lat == pytest.approx(loc.lat.deg, abs=1e-3)
        assert h == pytest.approx(loc.height.to_value(u.km), abs=1e-3)


def test_activity_presets_and_diurnal_bulge():
    epoch = ur.Epoch.parse("2026-11-01T12:00Z")
    r = np.array([6778e3, 0.0, 0.0])
    rho = {a: ur.NRLMSISE00(activity=a)(r, epoch.tdb_seconds) for a in ("low", "moderate", "high")}
    assert rho["low"] < rho["moderate"] < rho["high"]
    assert 5 < rho["high"] / rho["low"] < 40                      # solar-cycle swing at 400 km
    from urania.core.sun import sun_position

    s = sun_position(epoch.jd)
    s = 6778e3 * s / np.linalg.norm(s)
    model = ur.NRLMSISE00()
    assert model(s, epoch.tdb_seconds) > 1.3 * model(-s, epoch.tdb_seconds)    # day side denser
    assert "NRLMSISE-00" in model.description and "F10.7 = 150" in model.description


def test_lifetime_depends_on_solar_activity():
    o = ur.Orbit.circular("Earth", "400 km", inc="51.6 deg", epoch="2026-11-01")
    days = [o.lifetime(area="0.03 m2", mass="4 kg", density=ur.NRLMSISE00(activity=a)).days
            for a in ("low", "moderate", "high")]
    assert days[0] > days[1] > days[2]
    assert days[0] / days[2] > 4


def test_validation_and_other_versions():
    with pytest.raises(ValueError, match="historical"):
        ur.NRLMSISE00(150, historical=True)
    with pytest.raises(ValueError, match="solar flux units"):
        ur.NRLMSISE00(5000)
    with pytest.raises(ValueError, match="ap"):
        ur.NRLMSISE00(150, ap=-1)
    with pytest.raises(ValueError, match="Did you mean 'high'"):
        ur.NRLMSISE00(activity="hihg")
    m21 = ur.NRLMSISE00(version=2.1)
    assert "NRLMSIS 2.1" in m21.description
    assert m21(np.array([6778e3, 0, 0]), 8e8) > 0


def test_missing_pymsis_message(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake(name, *args, **kwargs):
        if name.startswith("pymsis"):
            raise ImportError("no pymsis")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake)
    with pytest.raises(ImportError, match=r"urania\[msis\]"):
        ur.NRLMSISE00()


def test_many_matches_single_calls_and_node_matters():
    model = ur.NRLMSISE00()
    rs = np.random.default_rng(2).normal(size=(12, 3))
    rs = 6778e3 * rs / np.linalg.norm(rs, axis=1)[:, None]
    t = ur.Epoch.parse("2026-11-01").tdb_seconds
    assert model.many(rs, t) == pytest.approx([model(r, t) for r in rs], rel=1e-6)
    # A dawn-dusk sun-synchronous orbit avoids the day-side bulge and lasts longer than a noon-midnight one
    from urania.core.sun import sun_position

    epoch = ur.Epoch.parse("2026-06-21")
    s = sun_position(epoch.jd)
    ra_sun = math.atan2(s[1], s[0])
    noon = ur.sun_synchronous("450 km", raan=ra_sun, epoch=epoch)
    dawn = ur.sun_synchronous("450 km", raan=ra_sun + math.pi / 2, epoch=epoch)
    kw = dict(area="0.03 m2", mass="4 kg", density=model, max_days=2000)
    assert dawn.lifetime(**kw).days > 1.03 * noon.lifetime(**kw).days

"""Frame conversions (TEME ↔ GCRS/ITRS) and CCSDS OEM export."""

import numpy as np
import pytest

import urania as ur
from urania.coordinates import convert, read_oem

R_TEME = np.array([5094.18016210, 6127.64465950, 6380.34453270]) * 1e3
V_TEME = np.array([-4.746131487, 0.785818041, 5.531931288]) * 1e3
EPOCH = ur.Epoch.from_utc("2004-04-06T07:51:28.386009")


def test_vallado_example_3_15_gcrf():
    """Vallado Example 3-15: TEME → GCRF r = (5102.508958, 6123.011401, 6378.136928) km,
    v = (-4.74322016, 0.79053650, 5.53375528) km/s (printed to the mm and mm/s)."""
    r, v = convert(R_TEME, V_TEME, EPOCH, "teme", "gcrs")
    assert r / 1e3 == pytest.approx([5102.508958, 6123.011401, 6378.136928], abs=1e-5)
    assert v / 1e3 == pytest.approx([-4.74322016, 0.79053650, 5.53375528], abs=1e-6)


def test_vallado_example_3_15_itrf():
    """Vallado Example 3-15: TEME → ITRF r = (-1033.4793830, 7901.2952754, 6380.3565958) km. astropy uses its own
    Earth-orientation data (Vallado: xp = -0.140682", yp = 0.333309", ΔUT1 = -0.4399619 s), so allow 1 m."""
    r, v = convert(R_TEME, V_TEME, EPOCH, "teme", "itrs")
    assert np.linalg.norm(r / 1e3 - [-1033.4793830, 7901.2952754, 6380.3565958]) < 1e-3
    assert v / 1e3 == pytest.approx([-3.225636520, -2.872451450, 5.531924446], abs=1e-6)


def test_round_trips_and_orbit_methods():
    o = ur.Orbit.circular("Earth", "500 km", inc="51.6 deg", epoch="2026-11-01T12:00Z")
    for frame in ("gcrs", "itrs", "teme"):
        r, v = o.in_frame(frame)
        back = ur.Orbit.from_frame("Earth", r, v, o.epoch, frame)
        assert back.r == pytest.approx(o.r, abs=1e-5) and back.v == pytest.approx(o.v, abs=1e-7)
    r, v = o.in_frame("teme")
    assert r == pytest.approx(o.r) and v == pytest.approx(o.v)
    tr = o.propagate(hours=1, n_points=5)
    rs, vs = tr.in_frame("gcrs")
    assert rs.shape == (5, 3)
    assert rs[2] == pytest.approx(tr.orbit_at(2).in_frame("gcrs")[0], abs=1e-6)
    with_units = ur.Orbit.from_frame("Earth", [7000, 0, 0] * ur.u.km, [0, 7.5, 0] * ur.u.km / ur.u.s, "2026-11-01")
    assert np.linalg.norm(with_units.r) == pytest.approx(7e6)


def test_itrs_matches_ground_track_geodetic():
    """The ITRS position agrees with urania's own GMST-based geodetic coordinates to UT1 - UTC (< 0.001° lon)."""
    o = ur.Orbit.circular("Earth", "500 km", inc="51.6 deg", arglat="40 deg", epoch="2026-11-01T12:00Z")
    tr = o.propagate(minutes=1, n_points=2)
    track = tr.ground_track()
    import astropy.units as u
    from astropy.coordinates import ITRS, CartesianRepresentation

    r, _ = tr.in_frame("itrs")
    loc = ITRS(CartesianRepresentation(*r[0], unit=u.m), obstime=o.epoch.to_astropy()).earth_location
    assert np.degrees(track.lon[0]) == pytest.approx(loc.lon.deg, abs=1e-3)
    assert np.degrees(track.lat[0]) == pytest.approx(loc.lat.deg, abs=1e-3)


def test_oem_round_trip(tmp_path):
    o = ur.Orbit.circular("Earth", "500 km", inc="97.4 deg", epoch="2026-11-01T12:00Z")
    tr = o.propagate(hours=1, n_points=7)
    path = tr.to_oem(tmp_path / "sat.oem", object_name="CUBESAT-1", object_id="2026-999A")
    text = path.read_text()
    assert "REF_FRAME = GCRF" in text and "OBJECT_NAME = CUBESAT-1" in text
    assert "START_TIME = 2026-11-01T12:00:00.000" in text and "STOP_TIME = 2026-11-01T13:00:00.000" in text
    epochs, r, v, ref = read_oem(path)
    rg, vg = tr.in_frame("gcrs")
    assert ref == "GCRF" and len(epochs) == 7
    assert np.abs(r - rg).max() < 1e-3 and np.abs(v - vg).max() < 1e-6
    assert abs(epochs[-1] - tr.epochs[-1]) < 1e-3
    teme = tr.to_oem(tmp_path / "teme.oem", frame="TEME")
    assert "REF_FRAME = TEME" in teme.read_text()


def test_errors():
    with pytest.raises(ValueError, match="Unknown frame"):
        ur.ISS.in_frame("j2001")
    with pytest.raises(ValueError, match="barycentric"):
        ur.ISS.in_frame("icrs")
    with pytest.raises(ValueError, match="Earth orbits"):
        ur.Orbit.circular("Mars", "300 km").in_frame("gcrs")
    with pytest.raises(ValueError, match="OEM frame"):
        ur.ISS.propagate(minutes=10).to_oem("x.oem", frame="ICRF")

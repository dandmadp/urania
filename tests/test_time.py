import astropy.units as u
import pytest
from astropy.time import Time

from urania.time import J2000, Epoch


def test_j2000_reference():
    assert J2000.jd == 2451545.0
    assert J2000.iso == "2000-01-01T12:00:00"
    assert Epoch.from_iso("2000-01-01T12:00:00") == J2000


def test_iso_roundtrip():
    t = Epoch.from_iso("2026-10-07T03:25:41.123456")
    assert t.iso == "2026-10-07T03:25:41.123456"


def test_jd_roundtrip():
    t = Epoch.from_jd(2461320.75)
    assert t.jd == pytest.approx(2461320.75, abs=1e-9)


def test_tdb_calendar_has_no_leap_seconds():
    # UTC had a leap second on 2016-12-31, but a TDB calendar day is exactly 86400 s
    a = Epoch.from_iso("2016-12-31T00:00:00")
    b = Epoch.from_iso("2017-01-01T00:00:00")
    assert b - a == 86400.0


def test_arithmetic():
    t = Epoch.from_iso("2026-01-01T00:00:00")
    assert (t + 3600).iso == "2026-01-01T01:00:00"
    assert (t + 2 * u.day).iso == "2026-01-03T00:00:00"
    assert (t - 30 * u.min).iso == "2025-12-31T23:30:00"
    assert (t + 90) - t == pytest.approx(90.0)


def test_ordering_and_immutability():
    a = Epoch.from_iso("2026-01-01T00:00:00")
    b = a + 1
    assert a < b
    with pytest.raises(AttributeError):
        a.tdb_seconds = 0.0


def test_astropy_roundtrip():
    t = Epoch.from_iso("2026-10-07T12:34:56.5")
    at = t.to_astropy()
    assert at.scale == "tdb"
    assert Epoch.from_astropy(at).tdb_seconds == pytest.approx(t.tdb_seconds, abs=1e-6)


def test_from_astropy_converts_scale():
    """astropy converts UTC input to TDB (TDB - UTC ≈ 69.184 s in 2026)."""
    utc = Time("2026-01-01T00:00:00", scale="utc")
    t = Epoch.from_astropy(utc)
    expected = Epoch.from_iso("2026-01-01T00:00:00")
    assert t - expected == pytest.approx(69.184, abs=0.01)


def test_timezone_rejected():
    with pytest.raises(ValueError):
        Epoch.from_iso("2026-01-01T00:00:00+09:00")

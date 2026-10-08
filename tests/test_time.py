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


def test_epoch_shift_must_be_scalar():
    import numpy as np

    with pytest.raises(TypeError):
        J2000 + np.array([1.0, 2.0])
    with pytest.raises(TypeError):
        J2000 - [1.0, 2.0] * u.s


def test_from_utc_and_utc_round_trip():
    """2026: TDB - UTC = 69.184 s (37 leap seconds + 32.184 s); a trailing Z is accepted."""
    t = Epoch.from_utc("2026-10-08T00:00:00Z")
    assert t - Epoch.from_iso("2026-10-08T00:00:00") == pytest.approx(69.184, abs=0.002)
    assert Epoch.from_utc("2026-10-08T00:00:00") == t
    assert t.utc == "2026-10-08T00:00:00.000000"
    # microsecond resolution: a round trip through the string keeps the time to 1 µs
    t2 = t + 0.1234567
    assert abs(Epoch.from_utc(t2.utc) - t2) < 1e-6


def test_from_iso_with_time_zone_points_to_from_utc():
    with pytest.raises(ValueError, match="from_utc"):
        Epoch.from_iso("2026-10-08T00:00:00Z")


def test_now_is_recent():
    assert Epoch.now() > Epoch.from_iso("2026-01-01T00:00:00")


def test_number_plus_epoch_and_epoch_plus_epoch():
    assert 3600 + J2000 == J2000 + 3600
    with pytest.raises(TypeError, match="subtract"):
        J2000 + J2000


@pytest.mark.parametrize("text, iso", [
    ("2026-10-07T12:34:56.5", "2026-10-07T12:34:56.500000"),
    ("2026-10-07T12:34:56.123456789", "2026-10-07T12:34:56.123456"),
    ("2026-10-07 01:02:03", "2026-10-07T01:02:03"),
    ("2026-10-07", "2026-10-07T00:00:00"),
])
def test_from_iso_formats(text, iso):
    """Python 3.10's fromisoformat rejects some of these; urania normalizes them (regression test)."""
    assert Epoch.from_iso(text).iso == iso


@pytest.mark.parametrize("text", ["2026-10-08T00:00:00Z", "2026-10-08T00:00:00+09:00",
                                  "2026-10-08T00:00:00-0500"])
def test_from_iso_rejects_any_time_zone(text):
    with pytest.raises(ValueError, match="from_utc"):
        Epoch.from_iso(text)

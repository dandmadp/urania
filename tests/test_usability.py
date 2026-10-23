"""Beginner conveniences: strings for units, dates and bodies; unit-mistake warnings; summaries; hints."""

import math
import re
import warnings
from datetime import datetime, timedelta, timezone

import astropy.units as u
import pytest
from astropy.time import Time

import urania as ur
from urania.units import UnitError, UnitWarning


def test_strings_with_units():
    a = ur.Orbit.circular("Earth", "500 km", inc="97.4 deg", arglat="10 deg")
    b = ur.Orbit.circular(ur.Earth, 500 * u.km, inc=97.4 * u.deg, arglat=10 * u.deg)
    assert a.r == pytest.approx(b.r) and a.v == pytest.approx(b.v)
    assert ur.units.to_si(["7000 km", "0 km", "0 km"], ur.units.LENGTH) == pytest.approx([7e6, 0, 0])
    with pytest.raises(UnitError, match="no unit"):
        ur.units.to_si("400", ur.units.LENGTH)
    with pytest.raises(UnitError, match="Cannot read"):
        ur.units.to_si("four hundred km", ur.units.LENGTH)
    with pytest.raises(UnitError, match="Cannot convert"):
        ur.units.to_si("400 s", ur.units.LENGTH)


def test_dates_as_strings_datetimes_and_astropy():
    tdb = ur.Epoch.from_iso("2026-11-01T12:30:00")
    assert ur.Epoch.parse("2026-11-01 12:30") == tdb
    assert ur.Epoch.parse("2026-11-01") == ur.Epoch.from_iso("2026-11-01T00:00:00")
    utc = ur.Epoch.from_utc("2026-11-01T12:30:00")
    assert ur.Epoch.parse("2026-11-01T12:30Z") == utc
    assert ur.Epoch.parse("2026-11-01 12:30 UTC") == utc
    assert ur.Epoch.parse(datetime(2026, 11, 1, 12, 30, tzinfo=timezone.utc)).tdb_seconds == pytest.approx(
        utc.tdb_seconds, abs=1e-6)
    seoul = timezone(timedelta(hours=9))
    assert ur.Epoch.parse(datetime(2026, 11, 1, 21, 30, tzinfo=seoul)).tdb_seconds == pytest.approx(
        utc.tdb_seconds, abs=1e-6)
    assert ur.Epoch.parse(datetime(2026, 11, 1, 12, 30)) == tdb
    assert ur.Epoch.parse(Time("2026-11-01T12:30:00", scale="tdb")).tdb_seconds == pytest.approx(
        tdb.tdb_seconds, abs=1e-6)
    assert ur.Epoch.parse(tdb) is tdb
    with pytest.raises(TypeError, match="as a time"):
        ur.Epoch.parse(2026)
    o = ur.Orbit.circular(ur.Earth, "500 km", epoch="2026-11-01")
    assert o.epoch == ur.Epoch.from_iso("2026-11-01T00:00:00")


def test_bodies_by_name():
    assert ur.body("mars") is ur.Mars and ur.body(" Titan ") is ur.Titan and ur.body(ur.Io) is ur.Io
    assert ur.BODIES["Neptune"] is ur.Neptune and len(ur.BODIES) == 29
    with pytest.raises(ValueError, match="Did you mean 'Jupiter'"):
        ur.body("Jupyter")
    t = ur.Earth.hohmann_to("Mars", "300 km", "300 km", after="2026-01-01")
    assert t.target is ur.Mars and t.departure_epoch.iso.startswith("2026-12")
    assert ur.Orbit.circular("moon", "100 km").body is ur.Moon


def test_missions_with_strings():
    m = ur.flyby_mission(["Earth", "Venus", "Mercury"],
                         ["1973-11-03 17:45", "1974-02-05 17:01", "1974-03-29 20:47"], "185 km")
    assert m.flybys[0].altitude / 1e3 == pytest.approx(5768, rel=0.02)
    a = ur.Earth.lambert_to("Mars", "2026-11-01", "300 day", "300 km")
    b = ur.Earth.lambert_to("Mars", "2026-11-01", "2027-08-28", "300 km")
    assert a.arrival == b.arrival
    p = ur.porkchop("Earth", "Mars", ("2026-09-01", "2026-12-01"), ("2027-06-01", "2027-12-01"), step="15 day")
    assert p.c3.shape == (13, 7)


def test_durations():
    o = ur.LEO
    assert o.propagate(hours=2).t[-1] == pytest.approx(7200)
    assert o.propagate(minutes=90).t[-1] == pytest.approx(5400)
    assert o.propagate("90 min").t[-1] == pytest.approx(5400)
    assert o.after(minutes=10).epoch - o.epoch == pytest.approx(600)
    with pytest.raises(ValueError, match="once"):
        o.propagate(days=1, hours=2)
    with pytest.raises(ValueError, match="once"):
        o.propagate()


def test_unit_mistake_warnings():
    with pytest.warns(UnitWarning, match="400 m"):
        ur.Orbit.circular(ur.Earth, 400, inc=math.radians(51.6))
    with pytest.warns(UnitWarning, match="radians"):
        ur.Orbit.circular(ur.Earth, 400e3, inc=51.6)
    with pytest.warns(UnitWarning, match="radians"):
        ur.GroundStation.create("Daejeon", 0.63, 127.36)
    with pytest.warns(UnitWarning, match="meters"):
        ur.Orbit.from_elements(ur.Earth, a=7000, ecc=0.01)
    with pytest.warns(UnitWarning, match="meters"):
        ur.Orbit.from_vectors(ur.Earth, [7000, 0, 0], [0, 7.5, 0])
    with pytest.warns(UnitWarning, match="400 m"):
        ur.Earth.hohmann_to(ur.Mars, 400)
    with pytest.raises(ValueError, match="radians"), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ur.GroundStation.create("x", 36.4, 2.2)


def test_no_warnings_for_correct_input():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ur.Orbit.circular(ur.Earth, 400 * u.km, inc=51.6 * u.deg)
        ur.Orbit.circular(ur.Earth, 400e3, inc=0.9)
        ur.Orbit.from_elements(ur.Earth, a=6000e3, ecc=0.2)       # dips below the surface on purpose
        ur.GroundStation.create("Daejeon", 36.37 * u.deg, 127.36 * u.deg, 70)   # 70 m: plain meters are fine
        ur.Orbit.circular(ur.Earth, 0)
        ur.LEO.phasing(5 * u.deg)


def test_warning_points_at_the_callers_line():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ur.Orbit.circular(ur.Earth, 400e3, inc=51.6)
    assert caught and caught[0].filename == __file__


def test_hints_in_errors():
    with pytest.raises(ValueError, match="Did you mean 'drag'"):
        ur.LEO.propagate(days=1, model="j2+darg")
    with pytest.raises(ValueError, match="Did you mean 'hohmann'"):
        ur.LEO.transfer_to(ur.GEO, "homann")


def test_summaries():
    t = ur.LEO.transfer_to(ur.GEO)
    s = t.summary()
    assert s.startswith("Hohmann transfer") and f"{t.total_dv / 1e3:.4f} km/s" in s
    assert repr(s) == str(s)                     # prints cleanly in a REPL
    s = t.summary(mass="500 kg", isp="300 s")
    mp = float(re.search(r"propellant\s+([\d.]+) kg", s).group(1))
    assert mp == pytest.approx(t.propellant(500, 300), abs=1e-3)
    with pytest.raises(ValueError, match="both"):
        t.summary(mass=500)
    h = ur.Earth.hohmann_to("Mars", "300 km", after="2026-01-01").summary()
    assert "none (arrival only)" in h and "2026-12-04" in h and "phase angle" in h
    m = ur.flyby_mission(["Earth", "Venus", "Mercury"],
                         ["1973-11-03 17:45", "1974-02-05 17:01", "1974-03-29 20:47"], "185 km").summary()
    assert "flyby Venus" in m and "INFEASIBLE" not in m
    assert "burn 1" in ur.LEO.deorbit().summary()


def test_altitude_and_sun_synchronous_elsewhere():
    assert ur.ISS.altitude == pytest.approx(420e3)
    mars_sso = ur.sun_synchronous("300 km", body="Mars")
    assert math.degrees(mars_sso.inc) == pytest.approx(92.65, abs=0.05)


def test_cheatsheet_lines_run():
    """Every self-contained line of the cheat sheet runs."""
    text = str(ur.cheatsheet())
    assert "ur.Orbit.circular" in text and repr(ur.cheatsheet()) == text
    calls = [line.strip().split("   ")[0].strip() for line in text.splitlines() if line.startswith("  ur.")]
    assert len(calls) >= 10
    import matplotlib.pyplot as plt

    for code in calls:
        if "tle_text" in code:
            continue
        exec(code, {"ur": ur, "u": u})
    plt.close("all")

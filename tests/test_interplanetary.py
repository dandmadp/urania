"""Planet and moon presets and Body.hohmann_to (BodyTransfer)."""

import math
import warnings

import astropy.units as u
import matplotlib.pyplot as plt
import pytest

import urania as ur
from urania.core import planets as pl

DAY = 86400.0

# JPL SSD planet physical parameters: sidereal orbit period [years of 365.25 days]
JPL_PERIOD_YEARS = {"Mercury": 0.2408467, "Venus": 0.61519726, "Earth": 1.0000174, "Mars": 1.8808476,
                    "Jupiter": 11.862615, "Saturn": 29.447498, "Uranus": 84.016846, "Neptune": 164.79132}

# JPL SSD satellite mean elements: period [days]
JPL_MOON_PERIOD = {"Moon": 27.322, "Phobos": 0.3187, "Deimos": 1.2625, "Io": 1.762732, "Europa": 3.525463,
                   "Ganymede": 7.155588, "Callisto": 16.690440, "Mimas": 0.942422, "Enceladus": 1.370218,
                   "Tethys": 1.887802, "Dione": 2.736916, "Rhea": 4.517503, "Titan": 15.945448,
                   "Iapetus": 79.331002, "Miranda": 1.413479, "Ariel": 2.520379, "Umbriel": 4.144177,
                   "Titania": 8.705869, "Oberon": 13.463237, "Triton": 5.8792}

# Curtis Table 8.1: sphere of influence radii [km]
CURTIS_SOI = {"Mercury": 112e3, "Venus": 616e3, "Earth": 925e3, "Mars": 577e3, "Jupiter": 48.2e6,
              "Saturn": 54.8e6, "Uranus": 51.8e6, "Neptune": 86.6e6, "Moon": 66.2e3}


def _two_body_period(b):
    return 2 * math.pi * math.sqrt(b.semi_major_axis**3 / (b.parent.mu + b.mu))


@pytest.mark.parametrize("planet", ur.PLANETS, ids=lambda b: b.name)
def test_planet_presets(planet):
    """Kepler's third law with the preset μ and semi-major axis reproduces JPL's orbit periods.

    Standish's mean semi-major axes are fitted (they absorb planetary perturbations), so the outer planets
    differ by up to 0.06%.
    """
    assert planet.parent is ur.Sun and planet.ephemeris == planet.name
    assert _two_body_period(planet) / (365.25 * DAY) == pytest.approx(JPL_PERIOD_YEARS[planet.name], rel=1e-3)
    assert planet.source


@pytest.mark.parametrize("moon", ur.MOONS, ids=lambda b: b.name)
def test_moon_presets(moon):
    """Mean semi-major axis and μ give JPL's mean period.

    The parent's J2 speeds up the mean motion of close moons (Europa 0.75%, Mimas 0.5%), which the two-body
    period ignores.
    """
    assert moon.parent in ur.PLANETS and moon.ephemeris is None
    assert _two_body_period(moon) / DAY == pytest.approx(abs(JPL_MOON_PERIOD[moon.name]), rel=1e-2)
    assert moon.radius < moon.semi_major_axis


def test_planet_masses():
    """Masses μ/G agree with JPL's physical parameters [10²⁴ kg]."""
    jpl = {"Mercury": 0.330103, "Venus": 4.86731, "Mars": 0.641691, "Jupiter": 1898.125, "Saturn": 568.317,
           "Uranus": 86.8099, "Neptune": 102.4092}
    for name, m in jpl.items():
        assert getattr(ur, name).mass / 1e24 == pytest.approx(m, rel=2e-4), name


@pytest.mark.parametrize("name", list(CURTIS_SOI))
def test_soi_matches_curtis_table(name):
    assert getattr(ur, name).soi / 1e3 == pytest.approx(CURTIS_SOI[name], rel=6e-3)


def test_soi_and_period_without_parent():
    assert ur.Sun.soi == math.inf
    with pytest.raises(ValueError, match="no parent"):
        ur.Sun.orbital_period


def test_body_parent_validation():
    with pytest.raises(ValueError, match="needs a parent"):
        ur.Body("X", 1.0, 1.0, semi_major_axis=1e9)
    with pytest.raises(ValueError, match="positive semi_major_axis"):
        ur.Body("X", 1.0, 1.0, parent=ur.Sun)
    b = ur.Body.create("Rock", 1 * u.km**3 / u.s**2, 10 * u.km, parent=ur.Sun, semi_major_axis=2.5 * u.AU)
    assert b.semi_major_axis == pytest.approx(2.5 * 149597870700.0)
    assert b == ur.Body("Rock", 1e9, 1e4)   # orbit fields do not take part in equality


def test_earth_mars_curtis():
    """Curtis Examples 8.3 and 8.4 setting with the library's constants: Δv 3.590 km/s from 300 km."""
    t = ur.Earth.hohmann_to(ur.Mars, 300 * u.km, 300 * u.km)
    assert t.kind == "siblings" and t.central is ur.Sun
    assert t.v_inf_departure / 1e3 == pytest.approx(2.94, abs=0.01)
    assert t.v_inf_arrival / 1e3 == pytest.approx(2.65, abs=0.01)
    assert t.dv_departure / 1e3 == pytest.approx(3.590, abs=0.002)
    assert t.tof / DAY == pytest.approx(258.9, abs=0.3)
    assert math.degrees(t.phase_angle) == pytest.approx(44.3, abs=0.1)
    assert t.stay_time / DAY == pytest.approx(454, abs=1)
    assert t.total_dv == pytest.approx(t.dv_departure + t.dv_arrival)
    assert t.c3 == pytest.approx(t.v_inf_departure**2)
    assert t.departure_epoch is None and t.arrival_epoch is None
    assert "Earth → Mars" in repr(t)


def test_no_capture():
    t = ur.Earth.hohmann_to(ur.Jupiter, 300 * u.km)
    assert t.dv_arrival == 0.0 and t.total_dv == t.dv_departure
    assert t.v_inf_arrival > 5000


def test_inward_transfer():
    t = ur.Earth.hohmann_to(ur.Venus, 300 * u.km, 300 * u.km)
    back = ur.Venus.hohmann_to(ur.Earth, 300 * u.km, 300 * u.km)
    assert t.v_inf_departure == pytest.approx(back.v_inf_arrival)
    assert t.tof == pytest.approx(back.tof)
    assert t.phase_angle < 0   # Venus must trail Earth


def test_moon_cases_are_mirror_images():
    out = ur.Earth.hohmann_to(ur.Moon, 300 * u.km, 100 * u.km)
    back = ur.Moon.hohmann_to(ur.Earth, 100 * u.km, 300 * u.km)
    assert out.kind == "to_moon" and back.kind == "to_parent"
    assert out.central is ur.Earth and back.central is ur.Earth
    assert out.dv_departure == pytest.approx(back.dv_arrival)
    assert out.dv_arrival == pytest.approx(back.dv_departure)
    assert out.tof / DAY == pytest.approx(4.98, abs=0.01)
    assert out.v_inf_departure is None and back.v_inf_arrival is None
    assert back.phase_angle is None and back.synodic_period is None
    # Earth → Moon Hohmann departure from LEO is about 3.1 km/s (Curtis section 6.2 order of magnitude)
    assert out.dv_departure / 1e3 == pytest.approx(3.106, abs=0.002)


def test_sibling_moons():
    t = ur.Io.hohmann_to(ur.Europa, 100 * u.km, 100 * u.km)
    assert t.central is ur.Jupiter and t.stay_time is not None
    assert ur.Ganymede.hohmann_to(ur.Callisto, 100 * u.km).tof / DAY < 10


def test_departure_date_earth_mars_2026():
    """The 2026 Earth → Mars opportunity: Hohmann phase alignment in late 2026."""
    start = ur.Epoch.from_iso("2026-01-01T00:00:00")
    t = ur.Earth.hohmann_to(ur.Mars, 300 * u.km, after=start)
    d = t.departure_epoch
    assert "2026-10-01" <= d.iso[:10] <= "2027-01-31"
    lead = pl.ecliptic_longitude("Mars", d.jd) - pl.ecliptic_longitude("Earth", d.jd)
    assert math.cos(lead - t.phase_angle) == pytest.approx(1.0, abs=1e-12)
    assert t.arrival_epoch - d == pytest.approx(t.tof)
    assert t.departure_longitude == pytest.approx(pl.ecliptic_longitude("Earth", d.jd))
    # The next opportunity is about one synodic period later
    t2 = ur.Earth.hohmann_to(ur.Mars, 300 * u.km, after=d + DAY)
    assert (t2.departure_epoch - d) / DAY == pytest.approx(t.synodic_period / DAY, rel=0.1)


@pytest.mark.parametrize("target", [b for b in ur.PLANETS if b is not ur.Earth], ids=lambda b: b.name)
def test_departure_date_every_planet(target):
    start = ur.Epoch.from_iso("2030-01-01T00:00:00")
    t = ur.Earth.hohmann_to(target, 300 * u.km, after=start)
    assert 0 <= t.departure_epoch - start <= 1.2 * t.synodic_period


def test_departure_date_outside_validity_warns():
    with pytest.warns(UserWarning, match="1800-2050"):
        ur.Earth.hohmann_to(ur.Mars, 300 * u.km, after=ur.Epoch.from_iso("2060-01-01T00:00:00"))


def test_errors():
    with pytest.raises(ValueError, match="same body"):
        ur.Earth.hohmann_to(ur.Earth, 300 * u.km)
    with pytest.raises(ValueError, match="Chain transfers"):
        ur.Earth.hohmann_to(ur.Titan, 300 * u.km)
    with pytest.raises(ValueError, match="≥ 0"):
        ur.Earth.hohmann_to(ur.Mars, -1 * u.km)
    with pytest.raises(ValueError, match="sphere of influence"):
        ur.Earth.hohmann_to(ur.Mars, 2e6 * u.km)
    with pytest.raises(ValueError, match="smaller than its radius"):
        ur.Mars.hohmann_to(ur.Phobos, 300 * u.km, 1 * u.km)
    with pytest.raises(ValueError, match="needs capture_altitude"):
        ur.Moon.hohmann_to(ur.Earth, 100 * u.km)
    with pytest.raises(ValueError, match="capture_ecc"):
        ur.Earth.hohmann_to(ur.Mars, 300 * u.km, 300 * u.km, capture_ecc=1.0)
    with pytest.raises(ValueError, match="ephemeris"):
        ur.Io.hohmann_to(ur.Europa, 100 * u.km, after=ur.J2000)
    with pytest.raises(TypeError, match="as a time"):
        ur.Earth.hohmann_to(ur.Mars, 300 * u.km, after=20260101)
    with pytest.raises(ur.units.UnitError):
        ur.Earth.hohmann_to(ur.Mars, 300 * u.s)


def test_phobos_without_capture():
    t = ur.Mars.hohmann_to(ur.Phobos, 300 * u.km)
    assert t.kind == "to_moon" and t.dv_arrival == 0.0


@pytest.mark.parametrize("args", [
    (ur.Earth, ur.Mars, 300 * u.km, 300 * u.km, ur.Epoch.from_iso("2026-01-01T00:00:00")),
    (ur.Earth, ur.Moon, 300 * u.km, 100 * u.km, None),
    (ur.Moon, ur.Earth, 100 * u.km, 300 * u.km, None),
    (ur.Saturn, ur.Titan, 1e5 * u.km, None, None),
])
def test_explain_and_plot(args):
    origin, target, h1, h2, after = args
    t = origin.hohmann_to(target, h1, h2, after=after)
    text = str(t.explain())
    assert f"{origin.name} → {target.name}" in text and "Total" in text
    assert "Warnings" not in text
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ax = t.plot()
    assert ax.get_title()
    plt.close("all")

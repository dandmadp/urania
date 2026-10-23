"""Regression tests for the issues found in the 0.0.3 review."""

import math

import numpy as np
import pytest

import urania as ur
from urania.core.coverage import footprint, sensor_footprint
from urania.core.lambert import lambert

MU = 3.986004418e14


def test_epoch_with_utc_offset():
    utc = ur.Epoch.from_utc("2026-11-01T12:30:00")
    assert ur.Epoch.parse("2026-11-01T21:30+09:00") == utc
    assert ur.Epoch.parse("2026-11-01 21:30+0900") == utc
    assert ur.Epoch.parse("2026-11-01T07:30:00-05:00") == utc
    assert ur.Epoch.parse("2026-11-01") == ur.Epoch.from_iso("2026-11-01T00:00:00")   # "-01" is not an offset


def test_lambert_rejects_ill_conditioned_angles():
    r1 = np.array([7e6, 0, 0])
    for deg in (1e-6, 359.999):
        a = math.radians(deg)
        with pytest.raises(ValueError, match="ill-conditioned"):
            lambert(r1, 8e6 * np.array([math.cos(a), math.sin(a), 0]), 3000, MU)
    for deg in (1e-3, 179.999, 180.001, 359.0):
        a = math.radians(deg)
        lambert(r1, 8e6 * np.array([math.cos(a), math.sin(a), 0]), 3000, MU)
    with pytest.raises(ValueError, match="finite"):
        lambert(r1, [np.nan, 1, 1], 3000, MU)


def test_fast_interplanetary_legs_solve():
    """Very fast legs (strongly hyperbolic arcs) used to fail because the propagator stalled on rounding noise."""
    m = ur.Earth.lambert_to("Neptune", "2030-01-01", "60 day", "300 km")
    assert m.v_inf_departure > 100e3
    m.plot()
    import matplotlib.pyplot as plt

    plt.close("all")


def test_retrograde_moons():
    t = ur.Neptune.hohmann_to("Triton", "10000 km", "100 km")
    assert any("retrograde" in a for a in t.assumptions)
    backward = ur.Body.create("Backward", "100 km3/s2", "100 km", parent=ur.Jupiter, semi_major_axis="500000 km",
                              retrograde=True)
    with pytest.raises(ValueError, match="opposite directions"):
        ur.Io.hohmann_to(backward, "100 km")
    assert ur.Triton.retrograde and not ur.Moon.retrograde


def test_footprint_at_zenith_and_sensor_limits():
    R = 6378137.0
    lam, eta, D = footprint(500e3, R, math.pi / 2)
    assert lam == 0.0 and D == pytest.approx(500e3)
    lam, eps, D = sensor_footprint(500e3, R, 0.0)
    assert lam == 0.0 and D == pytest.approx(500e3)
    for e in np.linspace(0, math.pi / 2, 7):
        lam, eta, D = footprint(500e3, R, e)
        # law of cosines closes the triangle center, ground point, satellite
        assert (R + 500e3) ** 2 == pytest.approx(R**2 + D**2 + 2 * R * D * math.sin(e), rel=1e-12)


def test_deorbit_guards():
    hyperbola = ur.Orbit.from_elements(ur.Earth, a="-20000 km", ecc=1.5)
    with pytest.raises(ValueError, match="elliptic"):
        hyperbola.deorbit()
    with pytest.raises(ValueError, match="below the center"):
        ur.LEO.deorbit("-7000 km")


def test_mission_accepts_numpy_revolutions_and_small_porkchop_plot_error():
    m = ur.flyby_mission(["Earth", "Mars"], ["2030-01-01", "2032-06-01"], "300 km", revolutions=np.int64(1))
    assert m.legs[0].revolutions == 1
    p = ur.porkchop("Earth", "Mars", ("2026-08-01", "2026-08-05"), ("2027-04-01", "2027-04-03"), step="10 day")
    with pytest.raises(ValueError, match="at least 2"):
        p.plot()


def test_bodies_registry_is_read_only():
    with pytest.raises(TypeError):
        ur.BODIES["Pluto"] = ur.Moon

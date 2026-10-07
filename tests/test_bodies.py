import astropy.units as u
import pytest

from urania import Body, Earth, Mars, Moon, Sun


def test_presets_reasonable():
    assert Earth.mass == pytest.approx(5.972e24, rel=1e-3)
    assert Moon.mass == pytest.approx(7.346e22, rel=1e-3)
    assert Mars.mass == pytest.approx(6.417e23, rel=1e-3)
    assert Sun.mass == pytest.approx(1.989e30, rel=1e-3)
    assert Earth.radius == 6378137.0
    assert Earth.source


def test_create_with_units():
    b = Body.create("Test", mu=398600 * u.km**3 / u.s**2, radius=6000 * u.km,
                    rotation_rate=360 * u.deg / u.day)
    assert b.mu == pytest.approx(3.986e14)
    assert b.radius == pytest.approx(6.0e6)
    assert b.rotation_rate == pytest.approx(7.2722e-5, rel=1e-4)


def test_immutable():
    with pytest.raises(AttributeError):
        Earth.mu = 1.0


@pytest.mark.parametrize("kwargs", [
    {"mu": -1.0, "radius": 1e6},
    {"mu": 1e14, "radius": 0.0},
    {"mu": 1e14, "radius": 1e6, "flattening": 1.5},
    {"mu": float("nan"), "radius": 1e6},
])
def test_invalid_body_rejected(kwargs):
    with pytest.raises(ValueError):
        Body.create("X", **kwargs)


def test_flattening_presets():
    assert Earth.flattening == pytest.approx(1 / 298.257223563)
    assert 0 < Mars.flattening < 0.01

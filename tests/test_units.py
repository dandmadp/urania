import math

import astropy.units as u
import numpy as np
import pytest

from urania import units
from urania.units import UnitError, from_si, to_si


def test_quantity_converted_to_si():
    assert to_si(7000 * u.km, units.LENGTH) == pytest.approx(7.0e6)
    assert to_si(7.8 * u.km / u.s, units.VELOCITY) == pytest.approx(7800.0)
    assert to_si(1 * u.day, units.TIME) == pytest.approx(86400.0)
    assert to_si(90 * u.deg, units.ANGLE) == pytest.approx(math.pi / 2)


def test_bare_number_is_si():
    assert to_si(7.0e6, units.LENGTH) == 7.0e6
    assert isinstance(to_si(3, units.LENGTH), float)
    # Plain-number angles are rad
    assert to_si(1.0, units.ANGLE) == 1.0


def test_arrays():
    r = to_si([7000, 0, 0] * u.km, units.LENGTH)
    assert isinstance(r, np.ndarray)
    np.testing.assert_allclose(r, [7.0e6, 0, 0])
    np.testing.assert_allclose(to_si([1, 2, 3], units.LENGTH), [1.0, 2.0, 3.0])


def test_temperature_offset():
    assert to_si(0 * u.deg_C, units.TEMPERATURE) == pytest.approx(273.15)


def test_mu_unit():
    assert to_si(398600.4418 * u.km**3 / u.s**2, units.MU) == pytest.approx(3.986004418e14)


def test_incompatible_unit_raises():
    with pytest.raises(UnitError):
        to_si(5 * u.s, units.LENGTH)
    # UnitError is also a ValueError
    with pytest.raises(ValueError):
        to_si(5 * u.kg, units.VELOCITY)


def test_from_si():
    assert from_si(7.0e6, units.LENGTH) == 7.0e6
    q = from_si(7.0e6, units.LENGTH, u.km)
    assert q.unit == u.km
    assert q.value == pytest.approx(7000.0)
    assert from_si(math.pi, units.ANGLE, u.deg).value == pytest.approx(180.0)


def test_sequence_of_quantities():
    """[7000 km, 0 km, 1 m] used to fail with an unclear astropy error."""
    np.testing.assert_allclose(to_si([7000 * u.km, 0 * u.km, 1 * u.m], units.LENGTH), [7e6, 0.0, 1.0])
    np.testing.assert_allclose(to_si((1 * u.km, 2 * u.km), units.LENGTH), [1e3, 2e3])


def test_sequence_mixing_quantities_and_numbers_rejected():
    with pytest.raises(UnitError, match="mixes"):
        to_si([7000 * u.km, 0, 0], units.LENGTH)
    with pytest.raises(UnitError, match="incompatible"):
        to_si([7000 * u.km, 1 * u.s], units.LENGTH)

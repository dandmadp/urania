import astropy.units as u
import numpy as np
import pytest

from urania import units
from urania.environment import Constant, Environment, Function, as_environment

R = np.array([7000e3, 0, 0])


def test_constant_forms():
    assert as_environment(1e-12, units.DENSITY)(R, 0.0) == 1e-12
    q = as_environment(1e-12 * u.g / u.cm**3, units.DENSITY)
    assert isinstance(q, Constant)
    assert q(R, 0.0) == pytest.approx(1e-9)


def test_function_form():
    env = as_environment(lambda r, t: 2.0 * t, units.DENSITY)
    assert isinstance(env, Function)
    assert env(R, 3.0) == 6.0


def test_function_returning_quantity():
    env = as_environment(lambda r, t: 1.0 * u.g / u.m**3, units.DENSITY)
    assert env(R, 0.0) == pytest.approx(1e-3)


def test_environment_passthrough():
    class Half(Environment):
        description = "test"

        def __call__(self, r, t):
            return 0.5

    e = Half()
    assert as_environment(e, units.DENSITY) is e


def test_wrong_unit_rejected():
    with pytest.raises(ValueError):
        as_environment(3 * u.s, units.DENSITY)


def test_class_instead_of_instance_rejected():
    from urania.environment import ExponentialAtmosphere

    with pytest.raises(TypeError, match="인스턴스"):
        as_environment(ExponentialAtmosphere, units.DENSITY)

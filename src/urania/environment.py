"""Environment injection interface.

The library provides the computational framework; environment data (atmospheric density, ...)
comes from the user. All three forms below are accepted and handled internally as one `Environment`:

- constant: a number or astropy Quantity → `Constant`, which always returns the same value
- function: f(r, t) → `Function`
- `Environment` object: used as is

r is the inertial position [m] (numpy array), t is TDB seconds since J2000.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import astropy.units as u
import numpy as np

from . import units
from .core.forces import exponential_density


class Environment(ABC):
    """Environment model giving a value as a function of position and time. Returns an SI number."""

    description: str = ""

    @abstractmethod
    def __call__(self, r, t: float) -> float: ...


class Constant(Environment):
    """Environment that always returns the same value."""

    def __init__(self, value: float, description: str = ""):
        self.value = value
        self.description = description or f"constant {value:g}"

    def __call__(self, r, t: float) -> float:
        return self.value


class Function(Environment):
    """Wraps a user function f(r, t). A returned Quantity is converted to SI."""

    def __init__(self, func, si_unit: u.UnitBase, description: str = ""):
        self.func = func
        self.si_unit = si_unit
        name = getattr(func, "__name__", repr(func))
        self.description = description or f"user function {name}(r, t)"

    def __call__(self, r, t: float) -> float:
        value = self.func(r, t)
        if isinstance(value, u.Quantity):
            return units.to_si(value, self.si_unit)
        return value


def as_environment(value, si_unit: u.UnitBase) -> Environment:
    """Turn a constant, function or Environment into an Environment."""
    if isinstance(value, Environment):
        return value
    if isinstance(value, type):
        raise TypeError(f"Pass an instance, not the class {value.__name__} "
                        f"(e.g. {value.__name__}(...))")
    if callable(value):
        return Function(value, si_unit)
    return Constant(units.to_si(value, si_unit))


# ---------------------------------------------------------------- atmospheric density models

class ExponentialAtmosphere(Environment):
    """Exponential atmosphere of Vallado table 8-4. Altitude is spherical: |r| - R."""

    description = "exponential atmosphere (Vallado table 8-4, spherical-Earth altitude)"

    def __init__(self, radius: float):
        self.radius = radius

    def __call__(self, r, t: float) -> float:
        return exponential_density(np.sqrt(r @ r) - self.radius)

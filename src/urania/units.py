"""Unit conversion boundary.

Internally the library works only with SI numbers (m, kg, s, K, W, rad).
User input becomes SI numbers through this module; output carries units only on request.

- An astropy Quantity is converted to the given SI unit.
- A plain number (or list/array) is assumed to be SI already. Angles are taken as rad.
"""

import astropy.units as u
import numpy as np

# Internal SI unit for each quantity
LENGTH = u.m
AREA = u.m**2
VELOCITY = u.m / u.s
ACCELERATION = u.m / u.s**2
TIME = u.s
MASS = u.kg
ANGLE = u.rad
ANGULAR_VELOCITY = u.rad / u.s
MU = u.m**3 / u.s**2        # gravitational parameter
DENSITY = u.kg / u.m**3
TEMPERATURE = u.K
POWER = u.W


class UnitError(ValueError):
    """Raised when an input unit does not match the expected quantity."""


def to_si(value, si_unit: u.UnitBase):
    """Convert an input value to an SI number.

    Args:
        value: astropy Quantity, number, or a sequence of numbers or of Quantities
            (e.g. ``[7000 * u.km, 0 * u.km, 0 * u.km]``)
        si_unit: SI unit of the expected quantity (e.g. `LENGTH`)

    Returns:
        float for scalars, otherwise a numpy array
    """
    if isinstance(value, (list, tuple)) and any(isinstance(x, u.Quantity) for x in value):
        if not all(isinstance(x, u.Quantity) for x in value):
            raise UnitError("A sequence mixes Quantities and plain numbers; give every value "
                            "with a unit, or none")
        try:
            value = u.Quantity(value)
        except u.UnitConversionError as exc:
            raise UnitError(f"The values in the sequence have incompatible units: {exc}") from exc
    if isinstance(value, u.Quantity):
        try:
            # Allow temperature equivalencies for offset conversions such as Celsius → Kelvin
            result = value.to_value(si_unit, equivalencies=u.temperature())
        except u.UnitConversionError as exc:
            raise UnitError(
                f"Cannot convert {value.unit} to {si_unit}"
            ) from exc
    else:
        result = np.asarray(value, dtype=float)
    return float(result) if np.ndim(result) == 0 else np.asarray(result, dtype=float)


def from_si(value, si_unit: u.UnitBase, unit=None):
    """Convert an SI number for output.

    Returns the SI number unchanged if unit is None, otherwise a Quantity in that unit.
    """
    if unit is None:
        return value
    return (value * si_unit).to(unit, equivalencies=u.temperature())

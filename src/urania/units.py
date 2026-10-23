"""Unit conversion boundary.

Internally the library works only with SI numbers (m, kg, s, K, W, rad).
User input becomes SI numbers through this module; output carries units only on request.

- An astropy Quantity is converted to the given SI unit.
- A string with a unit, such as "400 km", "51.6 deg" or "90 min", is read as a Quantity.
- A plain number (or list/array) is assumed to be SI already. Angles are taken as rad.

Plain numbers that are almost certainly a unit mistake (an angle above 2π, an orbit altitude below 1 km) raise
a UnitWarning that suggests the intended unit.
"""

import math
import warnings

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


class UnitWarning(UserWarning):
    """A plain number that looks like it was meant in another unit (degrees, kilometers)."""


def _user_stacklevel() -> int:
    """Stack level of the first frame outside the urania package, so warnings point at the user's line."""
    import os
    import sys

    package = os.path.dirname(os.path.abspath(__file__))
    frame, level = sys._getframe(1), 1
    while frame is not None and os.path.abspath(frame.f_code.co_filename).startswith(package):
        frame, level = frame.f_back, level + 1
    return level


def _from_string(text: str) -> u.Quantity:
    try:
        q = u.Quantity(text)
    except (TypeError, ValueError) as exc:
        raise UnitError(f"Cannot read {text!r} as a value with a unit (e.g. '400 km', '51.6 deg')") from exc
    if q.unit == u.dimensionless_unscaled:
        raise UnitError(f"{text!r} has no unit; write it with one, e.g. '{text.strip()} km'")
    return q


def to_si(value, si_unit: u.UnitBase):
    """Convert an input value to an SI number.

    Args:
        value: astropy Quantity, number, or a sequence of numbers or of Quantities
            (e.g. ``[7000 * u.km, 0 * u.km, 0 * u.km]``)
        si_unit: SI unit of the expected quantity (e.g. `LENGTH`)

    Returns:
        float for scalars, otherwise a numpy array
    """
    if isinstance(value, str):
        value = _from_string(value)
    elif isinstance(value, (list, tuple)) and any(isinstance(x, str) for x in value):
        value = [_from_string(x) if isinstance(x, str) else x for x in value]
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
        if si_unit == ANGLE and np.ndim(result) == 0 and abs(float(result)) > 2 * math.pi + 1e-9:
            warnings.warn(f"The angle {float(result):g} is a plain number, so it is read as radians "
                          f"({math.degrees(float(result)):,.0f}°). For degrees write {float(result):g} * u.deg "
                          f"or '{float(result):g} deg'", UnitWarning, stacklevel=_user_stacklevel())
    return float(result) if np.ndim(result) == 0 else np.asarray(result, dtype=float)


def altitude(value, name: str = "altitude") -> float:
    """An altitude in meters. A plain number between 0 and 1000 is almost surely kilometers: warn."""
    h = to_si(value, LENGTH)
    if not isinstance(value, (u.Quantity, str)) and 0.0 < h < 1000.0:
        warnings.warn(f"{name}={h:g} is a plain number, so it is read as {h:g} m. For kilometers write "
                      f"{h:g} * u.km or '{h:g} km'", UnitWarning, stacklevel=_user_stacklevel())
    return h


def from_si(value, si_unit: u.UnitBase, unit=None):
    """Convert an SI number for output.

    Returns the SI number unchanged if unit is None, otherwise a Quantity in that unit.
    """
    if unit is None:
        return value
    return (value * si_unit).to(unit, equivalencies=u.temperature())

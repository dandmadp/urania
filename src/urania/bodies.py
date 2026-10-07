"""Celestial body objects and presets."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from . import units
from .environment import ExponentialAtmosphere
from .constants import G


@dataclass(frozen=True)
class Body:
    """Central body. All values are SI.

    Stores μ (GM) as the primary value: it is measured directly and is far more accurate than the mass.
    """

    name: str
    mu: float                   # gravitational parameter GM [m³/s²]
    radius: float               # equatorial radius [m]
    J2: float = 0.0             # second zonal harmonic [-]
    rotation_rate: float = 0.0  # rotation rate [rad/s] (sidereal)
    flattening: float = 0.0     # reference ellipsoid flattening f = (a - b)/a [-]
    atmosphere: Any = field(default=None, compare=False)  # atmospheric density Environment
    source: str = field(default="", compare=False)        # source of the constants

    def __post_init__(self):
        if not self.mu > 0.0:
            raise ValueError(f"mu must be positive: {self.mu}")
        if not self.radius > 0.0:
            raise ValueError(f"radius must be positive: {self.radius}")
        if not 0.0 <= self.flattening < 1.0:
            raise ValueError(f"flattening must be in [0, 1): {self.flattening}")

    @classmethod
    def create(cls, name: str, mu, radius, J2: float = 0.0, rotation_rate=0.0,
               flattening: float = 0.0, atmosphere=None, source: str = "") -> Body:
        """Create from values with units. Plain numbers are taken as SI."""
        return cls(
            name=name,
            mu=units.to_si(mu, units.MU),
            radius=units.to_si(radius, units.LENGTH),
            J2=J2,
            rotation_rate=units.to_si(rotation_rate, units.ANGULAR_VELOCITY),
            flattening=flattening,
            atmosphere=atmosphere,
            source=source,
        )

    @property
    def mass(self) -> float:
        """Mass [kg] = μ / G."""
        return self.mu / G

    def __repr__(self) -> str:
        return f"Body({self.name!r}, R={self.radius / 1e3:.1f} km)"


Sun = Body(
    name="Sun",
    mu=1.32712440041e20,
    radius=6.957e8,
    rotation_rate=2.8653e-6,  # Carrington rotation period 25.38 days
    source="μ: DE440; R: IAU 2015 B3 nominal",
)

Earth = Body(
    name="Earth",
    mu=3.986004418e14,
    radius=6378137.0,
    J2=1.08262668e-3,
    rotation_rate=7.292115e-5,
    flattening=1 / 298.257223563,
    atmosphere=ExponentialAtmosphere(radius=6378137.0, flattening=1 / 298.257223563),
    source="μ, R, f: WGS 84; J2: EGM-08 (Vallado table D-1)",
)

Moon = Body(
    name="Moon",
    mu=4.902800118e12,
    radius=1.7374e6,
    J2=2.033e-4,
    rotation_rate=2.6617e-6,  # sidereal month 27.3217 days
    flattening=0.0012,
    source="μ: DE440; J2: GRAIL GRGM1200A; f: Moon Fact Sheet (NASA GSFC)",
)

Mars = Body(
    name="Mars",
    mu=4.282837e13,
    radius=3.3962e6,
    J2=1.96045e-3,
    rotation_rate=7.088218e-5,  # sidereal day 24.6229 hours
    flattening=0.00589,
    source="μ: DE440; R, J2, f: Mars Fact Sheet (NASA GSFC)",
)

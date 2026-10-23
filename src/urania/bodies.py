"""Celestial body objects and presets: the Sun, the eight planets and their major moons."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from . import units
from .constants import DAY, G
from .core.interplanetary import soi_radius
from .core.planets import STANDISH
from .core.sun import AU
from .environment import ExponentialAtmosphere


@dataclass(frozen=True)
class Body:
    """Central body. All values are SI.

    Stores μ (GM) as the primary value: it is measured directly and is far more accurate than the mass.
    A body that orbits another (a planet around the Sun, a moon around its planet) records its `parent` and the
    mean semi-major axis of that orbit; planets also name their mean-element ephemeris (`core.planets`).
    A negative rotation rate means retrograde rotation (Venus, Uranus, Triton).
    """

    name: str
    mu: float                   # gravitational parameter GM [m³/s²]
    radius: float               # equatorial radius [m]
    J2: float = 0.0             # second zonal harmonic [-]
    rotation_rate: float = 0.0  # rotation rate [rad/s] (sidereal)
    flattening: float = 0.0     # reference ellipsoid flattening f = (a - b)/a [-]
    atmosphere: Any = field(default=None, compare=False)  # atmospheric density Environment
    source: str = field(default="", compare=False)        # source of the constants
    parent: Body | None = field(default=None, compare=False)    # body it orbits
    semi_major_axis: float = field(default=0.0, compare=False)  # mean semi-major axis around parent [m]
    ephemeris: str | None = field(default=None, compare=False)  # key into core.planets.STANDISH

    def __post_init__(self):
        if not self.mu > 0.0:
            raise ValueError(f"mu must be positive: {self.mu}")
        if not self.radius > 0.0:
            raise ValueError(f"radius must be positive: {self.radius}")
        if not 0.0 <= self.flattening < 1.0:
            raise ValueError(f"flattening must be in [0, 1): {self.flattening}")
        if self.parent is None and self.semi_major_axis != 0.0:
            raise ValueError("semi_major_axis needs a parent body")
        if self.parent is not None and not self.semi_major_axis > 0.0:
            raise ValueError(f"A body with a parent needs a positive semi_major_axis: {self.semi_major_axis}")

    @classmethod
    def create(cls, name: str, mu, radius, J2: float = 0.0, rotation_rate=0.0,
               flattening: float = 0.0, atmosphere=None, source: str = "", parent: Body | None = None,
               semi_major_axis=0.0) -> Body:
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
            parent=parent,
            semi_major_axis=units.to_si(semi_major_axis, units.LENGTH),
        )

    @property
    def mass(self) -> float:
        """Mass [kg] = μ / G."""
        return self.mu / G

    @property
    def soi(self) -> float:
        """Sphere of influence radius [m] around the parent body (inf for a body without a parent)."""
        if self.parent is None:
            return math.inf
        return soi_radius(self.semi_major_axis, self.mu, self.parent.mu)

    @property
    def orbital_period(self) -> float:
        """Period [s] of a circular orbit at semi_major_axis around the parent (two-body, parent μ only)."""
        if self.parent is None:
            raise ValueError(f"{self.name} has no parent body")
        return 2.0 * math.pi * math.sqrt(self.semi_major_axis**3 / self.parent.mu)

    def hohmann_to(self, target: Body, parking_altitude, capture_altitude=None, *,
                   capture_ecc: float = 0.0, after=None):
        """Patched-conic Hohmann transfer to another body (Curtis ch. 8). Returns a `BodyTransfer`.

        Supported pairs: two bodies with the same parent (Earth → Mars, Io → Europa), a body to one of its
        moons (Earth → Moon) and a moon to its parent (Moon → Earth).

        Args:
            parking_altitude: altitude of the circular orbit the spacecraft leaves (length)
            capture_altitude: periapsis altitude of the capture orbit at the target (length);
                None means no capture burn (the arrival v∞ is still reported)
            capture_ecc: eccentricity of the capture orbit (0 = circular)
            after: an `Epoch`; if both bodies have an ephemeris, find the first departure after it
                at which the actual phase angle equals the Hohmann phase angle
        """
        from .interplanetary import hohmann_between

        return hohmann_between(self, target, parking_altitude, capture_altitude,
                               capture_ecc=capture_ecc, after=after)

    def lambert_to(self, target: Body, departure, arrival, parking_altitude, capture_altitude=None, **kwargs):
        """Direct transfer on a Lambert arc between two planets on given dates; returns a `Mission`.

        arrival is an Epoch or a time of flight. Planet positions come from the Standish mean elements.
        Example: ``Earth.lambert_to(Mars, Epoch.from_iso("2026-11-20T00:00:00"), 300 * u.day, 300 * u.km)``
        """
        from .missions import lambert_transfer

        return lambert_transfer(self, target, departure, arrival, parking_altitude, capture_altitude, **kwargs)

    def flyby(self, v_inf_in, altitude, *, epoch=None, b_angle=0.0):
        """Unpowered flyby with incoming excess velocity v∞_in (3-vector, ecliptic frame); returns a `Flyby`.

        With epoch, the planet's velocity gives the change of heliocentric speed.
        """
        from .missions import flyby

        return flyby(self, v_inf_in, altitude, epoch=epoch, b_angle=b_angle)

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
    source="μ, R, f: WGS 84; J2: EGM-08 (Vallado table D-1); orbit: Earth-Moon barycenter (Standish)",
    parent=Sun,
    semi_major_axis=STANDISH["Earth"][0][0] * AU,
    ephemeris="Earth",
)

Moon = Body(
    name="Moon",
    mu=4.902800118e12,
    radius=1.7374e6,
    J2=2.033e-4,
    rotation_rate=2.6617e-6,  # sidereal month 27.3217 days
    flattening=0.0012,
    source="μ: DE440; J2: GRAIL GRGM1200A; f: Moon Fact Sheet (NASA GSFC); orbit: JPL SSD mean elements",
    parent=Earth,
    semi_major_axis=384400e3,
)

_JPL = "JPL SSD planet physical parameters"
_FACT = "NASA GSFC planetary fact sheet"


def _planet(name: str, mu: float, radius_km: float, J2: float, sidereal_day: float, flattening: float,
            source: str) -> Body:
    """A planet orbiting the Sun; semi-major axis from its Standish J2000 mean elements."""
    return Body(name=name, mu=mu, radius=radius_km * 1e3, J2=J2,
                rotation_rate=2.0 * math.pi / (sidereal_day * DAY), flattening=flattening,
                source=source + "; orbit: Standish mean elements",
                parent=Sun, semi_major_axis=STANDISH[name][0][0] * AU, ephemeris=name)


# μ: DE440 (Mercury, Venus, Mars) or the JPL satellite ephemerides (JUP365, SAT441, URA111, NEP097), planet only.
# Equatorial radius and sidereal rotation [days]: JPL SSD. J2 and flattening: NASA GSFC fact sheets (the
# J2 reference radius is not stated there; it is taken as the equatorial radius).
Mercury = _planet("Mercury", 2.2031868551e13, 2440.53, 50.3e-6, 58.6462, 0.0009,
                  f"μ: DE440; R, rotation: {_JPL}; J2, f: {_FACT}")
Venus = _planet("Venus", 3.24858592e14, 6051.8, 4.458e-6, -243.018, 0.0,
                f"μ: DE440; R, rotation: {_JPL}; J2: {_FACT}")
Mars = Body(
    name="Mars",
    mu=4.282837e13,
    radius=3.3962e6,
    J2=1.96045e-3,
    rotation_rate=7.088218e-5,  # sidereal day 24.6229 hours
    flattening=0.00589,
    source="μ: DE440; R, J2, f: Mars Fact Sheet (NASA GSFC); orbit: Standish mean elements",
    parent=Sun,
    semi_major_axis=STANDISH["Mars"][0][0] * AU,
    ephemeris="Mars",
)
Jupiter = _planet("Jupiter", 1.266865319e17, 71492.0, 14736e-6, 0.41354, 0.06487,
                  f"μ: JUP365; R, rotation: {_JPL}; J2, f: {_FACT}")
Saturn = _planet("Saturn", 3.793120623e16, 60268.0, 16298e-6, 0.44401, 0.09796,
                 f"μ: SAT441; R, rotation: {_JPL}; J2, f: {_FACT}")
Uranus = _planet("Uranus", 5.7939513e15, 25559.0, 3343.43e-6, -0.71833, 0.02293,
                 f"μ: URA111; R, rotation: {_JPL}; J2, f: {_FACT}")
Neptune = _planet("Neptune", 6.83509997e15, 24764.0, 3411e-6, 0.67125, 0.01708,
                  f"μ: NEP097; R, rotation: {_JPL}; J2, f: {_FACT}")


def _moon(name: str, parent: Body, gm_km3: float, radius_km: float, a_km: float, period_days: float) -> Body:
    """A moon in synchronous rotation (a negative period means a retrograde orbit).

    μ and mean radius: JPL SSD satellite physical parameters; semi-major axis and period: JPL SSD satellite
    mean elements (planet equatorial frame).
    """
    return Body(name=name, mu=gm_km3 * 1e9, radius=radius_km * 1e3,
                rotation_rate=2.0 * math.pi / (period_days * DAY),
                source="μ, R (mean): JPL SSD satellite physical parameters; orbit: JPL SSD mean elements",
                parent=parent, semi_major_axis=a_km * 1e3)


Phobos = _moon("Phobos", Mars, 0.0007087, 11.08, 9375.0, 0.3187)
Deimos = _moon("Deimos", Mars, 0.0000962, 6.2, 23457.0, 1.2625)

Io = _moon("Io", Jupiter, 5959.91547, 1821.49, 421800.0, 1.762732)
Europa = _moon("Europa", Jupiter, 3202.71210, 1560.80, 671100.0, 3.525463)
Ganymede = _moon("Ganymede", Jupiter, 9887.83275, 2631.20, 1070400.0, 7.155588)
Callisto = _moon("Callisto", Jupiter, 7179.28340, 2410.30, 1882700.0, 16.690440)

Mimas = _moon("Mimas", Saturn, 2.50349, 198.20, 186000.0, 0.942422)
Enceladus = _moon("Enceladus", Saturn, 7.21037, 252.10, 238400.0, 1.370218)
Tethys = _moon("Tethys", Saturn, 41.21353, 531.10, 295000.0, 1.887802)
Dione = _moon("Dione", Saturn, 73.11607, 561.40, 377700.0, 2.736916)
Rhea = _moon("Rhea", Saturn, 153.94175, 763.50, 527200.0, 4.517503)
Titan = _moon("Titan", Saturn, 8978.13710, 2574.76, 1221900.0, 15.945448)
Iapetus = _moon("Iapetus", Saturn, 120.51511, 734.30, 3561700.0, 79.331002)

Miranda = _moon("Miranda", Uranus, 4.3, 235.8, 129846.0, 1.413479)
Ariel = _moon("Ariel", Uranus, 83.5, 578.9, 190929.0, 2.520379)
Umbriel = _moon("Umbriel", Uranus, 85.1, 584.7, 265986.0, 4.144177)
Titania = _moon("Titania", Uranus, 226.9, 788.9, 436298.0, 8.705869)
Oberon = _moon("Oberon", Uranus, 205.3, 761.4, 583511.0, 13.463237)

# Triton orbits retrograde (157° to Neptune's equator)
Triton = _moon("Triton", Neptune, 1428.49546, 1352.60, 354800.0, -5.8792)

PLANETS = (Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, Neptune)
MOONS = (Moon, Phobos, Deimos, Io, Europa, Ganymede, Callisto, Mimas, Enceladus, Tethys, Dione, Rhea, Titan,
         Iapetus, Miranda, Ariel, Umbriel, Titania, Oberon, Triton)

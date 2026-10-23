"""urania — an astrodynamics library that computes, shows and explains.

New here? ``print(urania.cheatsheet())`` lists common tasks with one line of code each.
"""

from astropy import units as u

from ._summary import cheatsheet
from .bodies import (
    BODIES, MOONS, PLANETS, Ariel, Body, Callisto, Deimos, Dione, Earth, Enceladus, Europa, Ganymede, Iapetus, Io,
    Jupiter, Mars, Mercury, Mimas, Miranda, Moon, Neptune, Oberon, Phobos, Rhea, Saturn, Sun, Tethys, Titan,
    Titania, Triton, Umbriel, Uranus, Venus,
)
from .bodies import get as body
from .eclipse import Eclipse, EclipseReport
from .environment import Environment
from .ground import GroundStation, GroundTrack, Pass, PassReport
from .interplanetary import BodyTransfer
from .lifetime import Lifetime
from .missions import Flyby, Leg, Mission, Porkchop, flyby_mission, porkchop
from .satellite import Doppler, Footprint, RelativeMotion
from .maneuvers import Burn, Transfer
from .orbits import GEO, ISS, LEO, SSO, Orbit, sun_synchronous
from .propagation import PropagationInfo, Trajectory
from .time import J2000, Epoch
from .tle import TLEOrbit

__version__ = "0.0.3"

__all__ = [
    "u", "cheatsheet", "body", "BODIES",
    "Body", "Sun", "Mercury", "Venus", "Earth", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune",
    "Moon", "Phobos", "Deimos", "Io", "Europa", "Ganymede", "Callisto",
    "Mimas", "Enceladus", "Tethys", "Dione", "Rhea", "Titan", "Iapetus",
    "Miranda", "Ariel", "Umbriel", "Titania", "Oberon", "Triton", "PLANETS", "MOONS",
    "Orbit", "sun_synchronous", "LEO", "ISS", "SSO", "GEO",
    "Epoch", "J2000",
    "Environment", "Trajectory", "PropagationInfo", "Transfer", "Burn", "BodyTransfer",
    "Mission", "Leg", "Flyby", "Porkchop", "flyby_mission", "porkchop",
    "RelativeMotion", "Doppler", "Footprint",
    "TLEOrbit",
    "GroundStation", "GroundTrack", "Pass", "PassReport", "Eclipse", "EclipseReport", "Lifetime",
]

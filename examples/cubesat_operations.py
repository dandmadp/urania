"""CubeSat operations: ground track, station passes, eclipses and lifetime for the ISS TLE.

    python examples/cubesat_operations.py
"""

from pathlib import Path

import astropy.units as u
import matplotlib.pyplot as plt

import urania as ur

iss = ur.TLEOrbit.from_text("""ISS (ZARYA)
1 25544U 98067A   19343.69339541  .00001764  00000-0  38792-4 0  9991
2 25544  51.6439 211.2001 0007417  17.6667  85.6398 15.50103472202482""")
day = iss.propagate(days=1)

daejeon = ur.GroundStation.create("Daejeon", 36.37 * u.deg, 127.36 * u.deg, 70 * u.m)
passes = day.passes(daejeon, min_elevation=10 * u.deg)
print(passes)
print()

eclipses = day.eclipses()
print(eclipses)
print()

# Lifetime of a 3U CubeSat released at the ISS altitude (mean solar activity)
start = iss.to_orbit()
life = start.lifetime(area=0.03 * u.m**2, mass=4 * u.kg)
print(life)
print(life.explain())

fig, axes = plt.subplots(2, 1, figsize=(10, 9))
day.ground_track().plot(axes[0], stations=[daejeon])
life.plot(axes[1])
fig.tight_layout()
out = Path(__file__).with_name("cubesat_operations.png")
fig.savefig(out, dpi=120, bbox_inches="tight")
print(f"\nSaved figure: {out}")

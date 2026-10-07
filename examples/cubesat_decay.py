"""Orbital decay of a 3U CubeSat: how far does drag pull it down from 350 km?

    python examples/cubesat_decay.py

The exponential atmosphere (Vallado table 8-4) assumes mean solar activity. Real lifetimes vary
several-fold with solar activity, so it is worth trying other densities (see the density example below).
"""

from pathlib import Path

import astropy.units as u
import matplotlib.pyplot as plt

import urania as ur

cubesat = dict(cd=2.2, area=0.03 * u.m**2, mass=4.0 * u.kg)   # 3U, long side facing the flow
start = ur.Orbit.circular(ur.Earth, 350 * u.km, inc=51.6 * u.deg)

tr = start.propagate(days=60, model="j2+drag", **cubesat)
print(tr.explain())

# Environment injection: twice the standard density (rough solar maximum)
dense = start.propagate(days=60, model="j2+drag", **cubesat,
                        density=lambda r, t: 2.0 * ur.Earth.atmosphere(r, t))

fig, ax = plt.subplots(figsize=(8, 4))
tr.plot(ax, kind="altitude")
ax.plot(dense.t / 86400, dense.altitude / 1e3, color="C1", lw=0.5, alpha=0.6,
        label="2x density")
ax.legend()
out = Path(__file__).with_name("cubesat_decay.png")
fig.savefig(out, dpi=120, bbox_inches="tight")
print(f"\nSaved figure: {out}")

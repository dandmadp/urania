"""Hohmann transfers from Earth to every planet: Δv, flight time and the next departure date.

    python examples/interplanetary.py
"""

from pathlib import Path

import astropy.units as u
import matplotlib.pyplot as plt

import urania as ur

start = ur.Epoch.from_iso("2026-01-01T00:00:00")
print(f"{'target':8s} {'v∞ [km/s]':>9s} {'Δv dep':>7s} {'Δv cap':>7s} {'tof [d]':>8s}  departure")
for planet in ur.PLANETS:
    if planet is ur.Earth:
        continue
    t = ur.Earth.hohmann_to(planet, 300 * u.km, 500 * u.km, after=start)
    print(f"{planet.name:8s} {t.v_inf_departure / 1e3:9.3f} {t.dv_departure / 1e3:7.3f} "
          f"{t.dv_arrival / 1e3:7.3f} {t.tof / 86400:8.1f}  {t.departure_epoch.iso[:10]}")

mars = ur.Earth.hohmann_to(ur.Mars, 300 * u.km, 300 * u.km, after=start)
print()
print(mars.explain())

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
mars.plot(ax1)
ur.Jupiter.hohmann_to(ur.Europa, 200000 * u.km, 100 * u.km).plot(ax2)
out = Path(__file__).with_name("interplanetary.png")
plt.savefig(out, dpi=120, bbox_inches="tight")
print(f"\nSaved figure: {out}")

"""Lambert transfers, a porkchop plot, the Mariner 10 gravity assist, and satellite operations.

    python examples/missions_and_operations.py
"""

from pathlib import Path

import astropy.units as u
import matplotlib.pyplot as plt

import urania as ur

E = ur.Epoch.from_iso

# Porkchop plot for the 2026 Mars opportunity
pork = ur.porkchop(ur.Earth, ur.Mars, (E("2026-08-01T00:00:00"), E("2027-01-01T00:00:00")),
                   (E("2027-04-01T00:00:00"), E("2028-03-01T00:00:00")), step=5 * u.day,
                   parking_altitude=300 * u.km, capture_altitude=300 * u.km)
print(pork)
best = pork.best("total_dv")
print("Best total Δv:", best)

# Mariner 10: Earth → Venus flyby → Mercury
mariner = ur.flyby_mission([ur.Earth, ur.Venus, ur.Mercury],
                           [E("1973-11-03T17:45:00"), E("1974-02-05T17:01:00"), E("1974-03-29T20:47:00")],
                           parking_altitude=185 * u.km)
print(mariner)
print(mariner.explain())

# Satellite operations
chaser = ur.Orbit.circular(ur.Earth, 400 * u.km, inc=51.64 * u.deg)
target = ur.Orbit.circular(ur.Earth, 420 * u.km, inc=51.64 * u.deg, arglat=5 * u.deg)
meet = chaser.rendezvous(target, 345 * u.min, revolutions=3, branch="right")   # about 27 m/s
print(meet, f"propellant for a 12 kg CubeSat at Isp 220 s: {meet.propellant(12 * u.kg, 220 * u.s):.3f} kg")
print(chaser.phasing(target=ur.Orbit.circular(ur.Earth, 400 * u.km, inc=51.64 * u.deg, arglat=30 * u.deg),
                     revolutions=4))
print(chaser.deorbit(), chaser.footprint(10 * u.deg))

station = ur.GroundStation.create("Daejeon", 36.37 * u.deg, 127.36 * u.deg, 70 * u.m)
day = chaser.propagate(days=1, n_points=8641)
doppler = day.doppler(station, 437.5 * u.MHz)
print(doppler)

fig, axes = plt.subplots(2, 2, figsize=(13, 11))
pork.plot(axes[0, 0])
mariner.plot(axes[0, 1])
meet.plot(axes[1, 0])
doppler.plot(axes[1, 1], min_elevation=0 * u.deg)
out = Path(__file__).with_name("missions_and_operations.png")
plt.savefig(out, dpi=110, bbox_inches="tight")
print(f"\nSaved figure: {out}")

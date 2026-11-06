# 2. Propagation and perturbations

*Curtis, chapters 4.7 and 10 (orbital perturbations); Vallado, chapters 8–9.*

## Choosing the forces

`Orbit.propagate` returns a `Trajectory`; the model is a "+"-joined list of force terms or a fidelity level:

| level | model | adds |
|---|---|---|
| 0 | `"twobody"` | point-mass gravity (analytic) |
| 1 | `"j2"` | Earth's oblateness |
| 2 | `"j2+drag"` | atmospheric drag |
| 3 | `"j2+drag+sun+moon"` | Sun and Moon gravity |
| 4 | `"zonal+drag+sun+moon+srp"` | zonal harmonics J2–J6, solar radiation pressure |

```python
import urania as ur

start = ur.Orbit.circular("Earth", "500 km", inc="51.6 deg", epoch="2026-11-01")
tr = start.propagate(days=1, model="j2")
print(tr)
print(tr.info.integrator, tr.info.rtol)
print(tr.info.assumptions)
```

## J2: regression of the node (Curtis Section 4.7)

Earth's equatorial bulge turns the orbit plane. Curtis's equation 4.52 gives the mean rate, and the propagated
node follows it:

```python
import math
import numpy as np
import urania as ur

start = ur.Orbit.circular("Earth", "500 km", inc="51.6 deg")
tr = start.propagate(days=10, model="j2", n_points=1001)
drift = np.unwrap(tr.elements["raan"])
measured = (drift[-1] - drift[0]) / tr.t[-1]
print(math.degrees(measured) * 86400, "deg/day")
print(math.degrees(start.raan_rate) * 86400, "deg/day (Curtis eq. 4.52)")
```

That drift is what a sun-synchronous orbit exploits: `ur.sun_synchronous("700 km")` picks the inclination that
turns the plane 360° per year (98.19° at 700 km).

## Drag and the atmosphere

Drag needs the spacecraft's area and mass (and `cd`, 2.2 by default). The default atmosphere is Vallado's
exponential table; NRLMSISE-00 (`pip install urania[msis]`) adds the solar cycle and the day-night bulge:

```python
import urania as ur

cubesat = dict(cd=2.2, area="0.03 m2", mass="4 kg")
start = ur.Orbit.circular("Earth", "400 km", inc="51.6 deg", epoch="2026-11-01")

table = start.propagate(days=3, model="j2+drag", **cubesat)
msis = start.propagate(days=3, model="j2+drag", density=ur.NRLMSISE00(activity="high"), **cubesat)
print(table.final.qty.periapsis_altitude, msis.final.qty.periapsis_altitude)
ax = msis.plot(kind="altitude")
```

You can also inject your own density: a constant, or any function `f(r, t)` of the inertial position and TDB
seconds since J2000 (the environment-injection principle of the design):

```python
import urania as ur

start = ur.Orbit.circular("Earth", "400 km")
kw = dict(days=1, model="drag", area="0.03 m2", mass="4 kg")
start.propagate(density=3e-12, **kw)
start.propagate(density=lambda r, t: 2 * ur.Earth.atmosphere(r, t), **kw)
```

## Sun, Moon and radiation pressure

Above a few thousand kilometers drag vanishes and the Sun and Moon take over. At GEO they pull the inclination by
roughly 0.9° per year; radiation pressure pushes on large, light spacecraft:

```python
import urania as ur

geo = ur.Orbit.circular("Earth", "35786 km", inc="0.05 deg", epoch="2026-11-01")
week = geo.propagate(days=7, model="sun+moon+srp", area="40 m2", mass="2000 kg", cr=1.3)
print(week.final.qty.inc)
print(week.info.assumptions)
```

Each term is validated against NASA GMAT (see [Validation](../validation.md)). The Sun and Moon positions are
analytic (Vallado algorithms 29 and 31), good to about 0.3°.

## What the integrator did

```python
import urania as ur

tr = ur.ISS.propagate(days=1, model="j2+drag", area="1500 m2", mass="420000 kg")
print(tr.explain())
```

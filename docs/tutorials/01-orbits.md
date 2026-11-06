# 1. Orbits and Kepler's equation

*Curtis, chapters 2–4: the two-body problem, orbital elements, and position as a function of time.*

## State vector to orbital elements (Curtis Example 4.3)

A spacecraft has r = (−6045, −3490, 2500) km and v = (−3.457, 6.618, 2.533) km/s. Curtis uses
μ = 398 600 km³/s² for Earth, so we create that Earth to reproduce his numbers exactly:

```python
import urania as ur
from urania import u

earth_curtis = ur.Body.create("Earth (Curtis)", mu=398600 * u.km**3 / u.s**2, radius="6378 km")
orbit = ur.Orbit.from_vectors(earth_curtis, [-6045, -3490, 2500] * u.km, [-3.457, 6.618, 2.533] * u.km / u.s)

q = orbit.qty
print(q.h, q.ecc, q.inc, q.raan, q.argp, q.nu)
# 58310 km²/s, 0.1712, 153.2°, 255.3°, 20.07°, 28.45°  — Curtis's answers
print(q.a, q.period)     # 8788 km, 136.6 min (2.278 h)
```

`orbit.explain()` shows each step of Curtis's algorithm 4.2 with the intermediate vectors:

```python
print(orbit.explain())
```

## Elements to a state vector

```python
import urania as ur
from urania import u

o = ur.Orbit.from_elements("Earth", a="26600 km", ecc=0.74, inc="63.4 deg", raan="40 deg",
                           argp="270 deg", nu="0 deg")          # a Molniya orbit at perigee
print(o.r, o.v)
print(o.qty.periapsis_altitude, o.qty.apoapsis_altitude, o.qty.period)
```

Circular and equatorial orbits have undefined angles; urania follows Vallado's conventions (argument of latitude,
true longitude), documented in [DECISIONS D2](../DECISIONS.md).

## Where will it be? Kepler's equation

`Orbit.after` gives the orbit later. Two-body motion uses the analytic universal-variable solution
(Curtis algorithm 3.4), which works the same for ellipses, parabolas and hyperbolas:

```python
import urania as ur

o = ur.Orbit.from_elements("Earth", a="26600 km", ecc=0.74, inc="63.4 deg", argp="270 deg")
later = o.after(hours=3)
print(later.qty.nu)                       # true anomaly three hours after perigee
print(o.after(o.period).r - o.r)          # back where it started (to rounding)
```

The pure functions behind it work with plain SI numbers, which is how the textbook examples are tested:

```python
import math
from urania.core import kepler

E = kepler.mean_to_eccentric(math.radians(235.4), 0.4)    # Vallado Example 2-1
print(math.degrees(E))                                     # 220.512074767522°
```

## Presets and plots

```python
import urania as ur

for o in (ur.ISS, ur.SSO, ur.GEO):
    print(o, o.qty.period)
ax = ur.SSO.plot()
```

# Getting started

## Your first orbit

```python
import urania as ur

orbit = ur.Orbit.circular("Earth", "500 km", inc="97.4 deg", epoch="2026-11-01")
print(orbit)
print(orbit.summary())          # every property in km, deg, min
```

Values carry their units as strings (`"500 km"`, `"97.4 deg"`, `"90 min"`) or as astropy Quantities
(`500 * ur.u.km`). **Plain numbers are SI**: meters, seconds, radians. If a plain number looks like a unit mistake,
urania warns and points at your line:

```python
import warnings
import urania as ur

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    ur.Orbit.circular("Earth", 500, inc=97.4)     # 500 m and 97.4 rad: almost surely not meant
for w in caught:
    print(w.message)
```

## Compute, show, explain

Every result object has three views:

```python
import urania as ur

t = ur.LEO.transfer_to(ur.GEO)
print(t)                 # one line
print(t.summary())       # the key numbers
print(t.explain())       # formulas, intermediate values, assumptions
ax = t.plot()            # a matplotlib Axes; call matplotlib.pyplot.show() in a script
```

The raw numbers are attributes in SI units, ready for your own calculations:

```python
import urania as ur

t = ur.LEO.transfer_to(ur.GEO)
print(t.total_dv, "m/s", t.tof, "s")
print(t.burns[0].dv)     # Δv vector of the first burn [m/s]
```

## Dates and bodies

```python
import urania as ur

ur.Epoch.parse("2026-11-01")              # TDB, like Epoch.from_iso
ur.Epoch.parse("2026-11-01T12:30Z")       # UTC
ur.Epoch.parse("2026-11-01T21:30+09:00")  # a civil time zone
print(ur.body("titan"), ur.Titan.parent, ur.Titan.soi)
```

## A short tour

```python
import urania as ur

print(ur.cheatsheet())
```

Then continue with the [tutorials](tutorials/01-orbits.md).

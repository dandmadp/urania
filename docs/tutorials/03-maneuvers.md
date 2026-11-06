# 3. Orbital maneuvers

*Curtis, chapter 6; Vallado, chapter 6.*

## Hohmann transfer (Vallado Example 6-1)

From a 191.34411 km circular orbit to 35 781.34857 km: Vallado finds Δv₁ = 2.457038 km/s, Δv₂ = 1.478187 km/s and
5.256713 hours. The core functions take SI numbers:

```python
from urania.core import maneuvers

R, mu = 6378137.0, 398600.4418e9
dv1, dv2, tof = maneuvers.hohmann(R + 191344.11, R + 35781348.57, mu)
print(dv1 / 1e3, dv2 / 1e3, tof / 3600)
```

With objects, the burns happen on the actual state vectors and the result can be checked, plotted and explained:

```python
import urania as ur

low = ur.Orbit.circular("Earth", "191.34411 km")
high = ur.Orbit.circular("Earth", "35781.34857 km")
t = low.transfer_to(high)
print(t.summary())
print(t.explain())
```

## Plane changes

Inclination changes are expensive at low speed and cheap at high speed, so a Hohmann transfer to GEO does most of
the plane change at apogee. urania splits it between the two burns to minimize the total:

```python
import urania as ur

t = ur.ISS.transfer_to(ur.GEO)                 # 51.64° plane change included
print(t.summary())
alone = ur.ISS.transfer_to(ur.GEO, plane_split=0)   # all of it at apogee
print(alone.total_dv - t.total_dv, "m/s saved by splitting")
```

## Bi-elliptic transfers

For a radius ratio above about 15.6, three burns beat two:

```python
import urania as ur

leo = ur.Orbit.circular("Earth", "300 km")
far = ur.Orbit.circular("Earth", "150000 km")
print(leo.transfer_to(far).total_dv, leo.transfer_to(far, "bielliptic", rb="300000 km").total_dv)
```

## Phasing, rendezvous and disposal

```python
import urania as ur

target = ur.Orbit.circular("Earth", "500 km", inc="97.4 deg", arglat="30 deg")
chaser = ur.Orbit.circular("Earth", "500 km", inc="97.4 deg")
catch_up = chaser.phasing(target=target, revolutions=5)     # Curtis section 6.6
print(catch_up.summary())

dispose = chaser.deorbit("50 km")                           # perigee into the atmosphere
print(dispose.summary(mass="4 kg", isp="220 s"))            # Δv and propellant (rocket equation)
```

Lambert rendezvous between arbitrary orbits is in [tutorial 4](04-lambert-rendezvous.md).

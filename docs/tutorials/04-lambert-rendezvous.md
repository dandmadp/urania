# 4. Lambert's problem and rendezvous

*Curtis, chapters 5.3 (Lambert's problem) and 7 (relative motion and rendezvous).*

## Lambert's problem (Curtis Example 5.2)

Which orbit goes from r₁ = (5000, 10000, 2100) km to r₂ = (−14600, 2500, 7000) km in one hour? Curtis finds
v₁ = (−5.9925, 1.9254, 3.2456) km/s and v₂ = (−3.3125, −4.1966, −0.38529) km/s:

```python
import numpy as np
from urania.core.lambert import lambert

v1, v2 = lambert(np.array([5000, 10000, 2100]) * 1e3, np.array([-14600, 2500, 7000]) * 1e3, 3600, 398600e9)
print(v1 / 1e3, v2 / 1e3)
```

The solver uses universal variables with bracketing (it cannot diverge), handles multiple revolutions
(`revolutions=`, `branch="left"/"right"`) and verifies every solution by propagation.

## Rendezvous with another spacecraft

`Orbit.rendezvous` plans the two burns: onto a Lambert arc, then matching the target's velocity. Any pair of
orbits works, eccentric or in different planes. Multi-revolution arcs are usually far cheaper:

```python
import urania as ur

chaser = ur.Orbit.circular("Earth", "400 km", inc="51.64 deg")
target = ur.Orbit.circular("Earth", "420 km", inc="51.64 deg", arglat="5 deg")
quick = chaser.rendezvous(target, "45 min")
patient = chaser.rendezvous(target, "345 min", revolutions=3, branch="right")
print(quick.total_dv, patient.total_dv)
print(patient.explain())
```

## Relative motion: the Clohessy-Wiltshire equations

Near a chief spacecraft, motion is easiest to read in its LVLH frame (x radial, y along-track, z normal).
A CubeSat released from the ISS at 0.5 m/s does not drift straight away: it loops and slides along-track.

```python
import numpy as np
import urania as ur
from urania.core import relative

iss = ur.Orbit.circular("Earth", "410 km", inc="51.64 deg")
r, v = relative.from_lvlh(iss.r, iss.v, np.zeros(3), np.array([-0.5, -0.3, 0.1]))   # release velocity [m/s]
cubesat = ur.Orbit(ur.Earth, r, v, iss.epoch)

motion = cubesat.propagate(iss.period * 3).relative_to(iss)
print(motion)
ax = motion.plot()

# The linear Clohessy-Wiltshire solution (Curtis eq. 7.53) predicts the same to second order
rho, _ = relative.cw_propagate(np.zeros(3), np.array([-0.5, -0.3, 0.1]), iss.mean_motion, motion.t[-1])
print(rho, motion.r[-1])
```

`urania.core.relative.cw_rendezvous` gives the two-impulse CW rendezvous for close-range planning.

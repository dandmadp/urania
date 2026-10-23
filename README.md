# urania

**An astrodynamics library that computes, shows and explains**

urania is a Python library for people who are not professionals but want to compute orbits seriously: CubeSat teams, students, space simulation game players. Every result records the model and assumptions it was computed with, and `.explain()` walks through the formulas and intermediate values step by step.

> Status: MVP in development (v0.0.2, [changelog](https://github.com/dandmadp/urania/blob/main/CHANGELOG.md)). Design: [SPEC.md](https://github.com/dandmadp/urania/blob/main/SPEC.md). Design decisions: [docs/DECISIONS.md](https://github.com/dandmadp/urania/blob/main/docs/DECISIONS.md).

## Installation

Not on PyPI yet, so install from source. Python 3.10 or newer.

```bash
git clone https://github.com/dandmadp/urania.git
cd urania
python -m venv .venv
.venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

Tested on Python 3.10 to 3.13, with both the newest and the oldest allowed dependency versions.

## Quick start

The Δv needed to go from the ISS orbit to geostationary orbit (GEO):

```python
import urania as ur

transfer = ur.ISS.transfer_to(ur.GEO)
print(transfer)                    # Transfer(hohmann, Δv=4.7797 km/s, tof=5.295 h, 2 burns)
print(transfer.explain())          # formulas, intermediate speeds, Δv per burn, assumptions
transfer.plot()                    # 2D plot (matplotlib)
```

Part of the `explain()` output:

```text
3. Burn 1 (r₁)
   v₁ = √(μ/r₁) = 7.6573 km/s  (initial circular orbit)
   v₂ = √(μ(2/r₁ − 1/a_t)) = 10.0492 km/s  (transfer ellipse)
   combined with plane change α = 2.8913°
   Δv = √(v₁² + v₂² − 2 v₁ v₂ cos α) = 2.4325 km/s
```

## Creating orbits

Pass values with units (astropy), or plain numbers, which are taken as SI (m, m/s, rad).

```python
import astropy.units as u
import urania as ur

leo = ur.Orbit.circular(ur.Earth, 500 * u.km, inc=97.4 * u.deg)
ell = ur.Orbit.from_elements(ur.Earth, a=8000 * u.km, ecc=0.1, inc=30 * u.deg,
                             raan=40 * u.deg, argp=60 * u.deg, nu=0 * u.deg)
sv = ur.Orbit.from_vectors(ur.Earth, [6524.834, 6862.875, 6448.296] * u.km,
                           [4.901327, 5.533756, -1.976341] * u.km / u.s)

print(sv.a, sv.ecc, sv.period)     # results are SI numbers: m, -, s
print(sv.qty.a, sv.qty.inc)        # or Quantities in km, deg, min, ...
print(sv.summary())                # every property in convenient units
print(sv.explain())                # how the elements follow from the state vector

# a 400 x 600 km orbit, at a UTC epoch
eo = ur.Orbit.from_apsides(ur.Earth, 400 * u.km, 600 * u.km, inc=51.6 * u.deg,
                           epoch=ur.Epoch.from_utc("2026-10-08T00:00:00Z"))
```

Presets: bodies `Sun`, `Earth`, `Moon`, `Mars`; orbits `LEO` (500 km), `ISS` (420 km, 51.64°), `SSO` (700 km sun-synchronous), `GEO`.

## Propagation

Choose the fidelity: `"twobody"` (0) → `"j2"` (1) → `"j2+drag"` (2). The original object is unchanged; a new `Trajectory` is returned.

```python
import astropy.units as u
import urania as ur

cubesat = dict(cd=2.2, area=0.03 * u.m**2, mass=4.0 * u.kg)
start = ur.Orbit.circular(ur.Earth, 400 * u.km, inc=51.6 * u.deg)

tr = start.propagate(days=3, model="j2+drag", **cubesat)
print(tr.final)                    # orbit after 3 days
print(tr.info.assumptions)         # model and assumptions used
print(tr.elements["raan"][-1])     # osculating elements at every sample, as arrays
tr.plot(kind="altitude")           # altitude with an orbit-averaged line

later = start.after(days=1, model="j2")   # just the orbit: shorthand for propagate(...).final
```

Passing `area`, `mass` or `density` without a drag term in `model` gives a warning, since drag would otherwise be left out silently.

You can inject environment data yourself: a constant, a function `f(r, t)`, or an `Environment` object.

```python
import urania as ur

start = ur.Orbit.circular(ur.Earth, 400e3)
kw = dict(days=1, model="drag", area=0.03, mass=4.0)

start.propagate(density=3e-12, **kw)                                       # constant [kg/m³]
start.propagate(density=lambda r, t: 2 * ur.Earth.atmosphere(r, t), **kw)  # function
```

## Maneuvers

```python
import astropy.units as u
import urania as ur

ur.LEO.transfer_to(ur.GEO)                                   # Hohmann transfer
far = ur.Orbit.circular(ur.Earth, 150000 * u.km)
ur.LEO.transfer_to(far, "bielliptic", rb=300000 * u.km)     # bi-elliptic transfer
ur.ISS.transfer_to(ur.GEO, plane_split=0)                    # whole plane change at apoapsis
```

When the planes differ, burns happen at the line of nodes, and by default a Hohmann transfer splits the plane change optimally between its two burns.

## Planets and moons

Presets cover the Sun, the eight planets and 20 major moons (`ur.PLANETS`, `ur.MOONS`): Moon; Phobos, Deimos;
Io, Europa, Ganymede, Callisto; Mimas, Enceladus, Tethys, Dione, Rhea, Titan, Iapetus; Miranda, Ariel, Umbriel,
Titania, Oberon; Triton. Each knows its parent body, mean orbit radius and sphere of influence (`body.soi`).

`Body.hohmann_to` computes a patched-conic Hohmann transfer between two bodies in one call: escape from a parking
orbit, the Hohmann ellipse around the common parent, and capture at the target.

```python
import astropy.units as u
import urania as ur

t = ur.Earth.hohmann_to(ur.Mars, parking_altitude=300 * u.km, capture_altitude=300 * u.km,
                        after=ur.Epoch.from_iso("2026-01-01T00:00:00"))
print(t)              # BodyTransfer(Earth → Mars, Δv=5.6807 km/s, tof=258.9 d, depart 2026-12-04)
print(t.v_inf_departure, t.c3, t.phase_angle, t.synodic_period, t.stay_time)
print(t.explain())    # v∞, departure and capture Δv, phase angle, dates
t.plot()

ur.Earth.hohmann_to(ur.Moon, 300 * u.km, 100 * u.km)      # planet → its moon
ur.Io.hohmann_to(ur.Europa, 100 * u.km, 100 * u.km)       # moons of the same planet
ur.Earth.hohmann_to(ur.Jupiter, 300 * u.km)               # no capture: arrival v∞ only
```

With `after=`, the departure date is the first time the actual planet positions (JPL's Standish mean elements,
1800 to 2050) reach the Hohmann phase angle. The pure formulas (v∞, departure and capture Δv, phase angle, synodic
period, sphere of influence) are in `urania.core.interplanetary` and work with plain numbers.

### Lambert transfers, porkchop plots and gravity assists

Real launch dates and flight times need Lambert's problem: the arc between the planets' actual positions.

```python
import astropy.units as u
import urania as ur

E = ur.Epoch.from_iso
m = ur.Earth.lambert_to(ur.Mars, E("2026-11-01T00:00:00"), 300 * u.day, 300 * u.km, 300 * u.km)
print(m, m.c3, m.v_inf_arrival)

p = ur.porkchop(ur.Earth, ur.Mars, (E("2026-08-01T00:00:00"), E("2027-01-01T00:00:00")),
                (E("2027-04-01T00:00:00"), E("2028-03-01T00:00:00")), step=10 * u.day)
best = p.best("c3")        # the minimum-C3 transfer as a Mission
p.plot()

# Mariner 10: Earth → Venus flyby → Mercury
mariner = ur.flyby_mission([ur.Earth, ur.Venus, ur.Mercury],
                           [E("1973-11-03T17:45:00"), E("1974-02-05T17:01:00"), E("1974-03-29T20:47:00")],
                           parking_altitude=185 * u.km)
print(mariner.flybys[0])   # altitude 5,739 km (actual 5,768 km), turn 33.8°, heliocentric -4.57 km/s
mariner.plot()
```

Each intermediate planet joins the incoming and outgoing v∞ with a flyby; when the magnitudes differ, a periapsis
burn makes up the difference (a powered flyby), and flybys below `min_flyby_altitude` are flagged infeasible.
`Body.flyby(v_inf_in, altitude)` shows what a single unpowered flyby does. The solver
(`urania.core.lambert.lambert`) handles any conic, both directions and multiple revolutions.

## Rendezvous, phasing and deorbit

```python
import astropy.units as u
import urania as ur

chaser = ur.Orbit.circular(ur.Earth, 400 * u.km, inc=51.64 * u.deg)
target = ur.Orbit.circular(ur.Earth, 420 * u.km, inc=51.64 * u.deg, arglat=5 * u.deg)
# two burns on a Lambert arc (any pair of orbits); 3 revolutions on the way: about 27 m/s
chaser.rendezvous(target, 345 * u.min, revolutions=3, branch="right")
target.phasing(-10 * u.deg, revolutions=3)     # drift 10° back along the same orbit
chaser.deorbit(50 * u.km)                      # lower perigee to 50 km
t = chaser.deorbit()
print(t.propellant(mass=4 * u.kg, isp=220 * u.s))   # rocket equation, kg
```


Ground tracks, station passes, eclipses and lifetime work on any trajectory around Earth (from `Orbit.propagate`
or `TLEOrbit.propagate`).

```python
import astropy.units as u
import urania as ur

iss = ur.TLEOrbit.from_text("""ISS (ZARYA)
1 25544U 98067A   19343.69339541  .00001764  00000-0  38792-4 0  9991
2 25544  51.6439 211.2001 0007417  17.6667  85.6398 15.50103472202482""")
day = iss.propagate(days=1)

station = ur.GroundStation.create("Daejeon", 36.37 * u.deg, 127.36 * u.deg, 70 * u.m)
passes = day.passes(station, min_elevation=10 * u.deg)   # rise, peak and set of every pass
print(passes)
eclipses = day.eclipses()                                 # penumbra and umbra intervals
print(eclipses.shadow_fraction)
track = day.ground_track()                                # geodetic latitude, longitude, height
track.plot(stations=[station])

life = iss.to_orbit().lifetime(area=0.03 * u.m**2, mass=4 * u.kg)   # about a second
print(life, life.explain())

doppler = day.doppler(station, 437.5 * u.MHz)            # range rate and Doppler shift
doppler.plot(min_elevation=10 * u.deg)
orbit = iss.to_orbit()
print(orbit.footprint(10 * u.deg))                        # visible area above 10° elevation
print(orbit.footprint(half_angle=30 * u.deg))             # nadir camera swath
print(orbit.beta_angle(), orbit.eclipse_fraction())       # Sun angle to the orbit plane, shadow fraction

# Relative motion of a deployed CubeSat (LVLH: x radial, y along-track, z normal)
deployed = ur.Orbit(ur.Earth, orbit.r, orbit.v + [0.0, 0.0, 0.5], orbit.epoch)
deployed.propagate(orbit.period * 3).relative_to(orbit).plot()
```

## TLE

```python
import urania as ur

iss = ur.TLEOrbit.from_text("""ISS (ZARYA)
1 25544U 98067A   19343.69339541  .00001764  00000-0  38792-4 0  9991
2 25544  51.6439 211.2001 0007417  17.6667  85.6398 15.50103472202482""")

sgp4 = iss.propagate(days=1)                          # SGP4 only
mine = iss.to_orbit().propagate(days=1, model="j2")   # extract the state, propagate with urania
```

TLE elements are SGP4-specific mean elements, so they are kept separate from `Orbit`. The frame is SGP4's TEME, used as is.

## Accuracy and validation

| What | Reference | Result |
|---|---|---|
| Element conversion | Curtis examples 4.3, 4.7 / Vallado example 2-5 | matches to the textbook's printed precision |
| Kepler's equation and propagation | Vallado examples 2-1, 2-3, 2-4 | within 1e-9 rad / 1 m |
| Hohmann and bi-elliptic transfers | Vallado examples 6-1, 6-2 | Δv within 1 mm/s |
| J2 secular drift | analytic nodal rate | within 1% over 10 days |
| Drag decay | analytic circular decay da/dt = −ρB√(μa) | within 1% |
| SGP4 wrapper | Vallado SGP4 verification set (satellite 00005) | within 1e-6 km |
| urania propagator vs SGP4 | real ISS TLE (2019-12-09), J2 model | at most 2.34 km over 24 hours |
| NASA GMAT R2026a, two-body | 1 day, RK89 | at most 0.009 m |
| NASA GMAT R2026a, J2 | 7 days, 420 km, 51.64° | at most 3.3 m (144 m without the true pole, see below) |
| NASA GMAT R2026a, J2 + drag | 3 days, 400 km, exponential atmosphere | at most 3.3 m (73 m without the true pole) |
| Sidereal time and Sun position | Vallado examples 3-5, 5-1 | GMST to 1e-8°, Sun direction within 2" |
| Ground track | Skyfield, ISS TLE, 1 day | latitude and height identical, longitude within UT1 − UTC (0.0008°) |
| Station passes | Skyfield `find_events`, 4 stations, 3 days, 39 passes | rise/peak/set within 0.23 s, peak elevation within 0.009° |
| Lifetime (averaged method) | full j2+drag propagation, 250 to 350 km | within 2% |
| Interplanetary Hohmann transfers | Curtis examples 8.1, 8.3, 8.4; Table 8.1 (spheres of influence) | printed precision |
| Lambert's problem | Curtis example 5.2, Vallado example 7-5; random round trips (multi-revolution) | printed precision; 1e-6 relative |
| Lambert transfers between planets | Curtis example 8.8 (Mars Global Surveyor) | v∞ within 5 m/s |
| Gravity assist | Mariner 10 Venus flyby (1974) | flyby altitude 5,739 km vs 5,768 km flown, nearly unpowered |
| Relative motion | two-body propagation | Clohessy-Wiltshire within second-order terms; CW and Lambert rendezvous agree within 2% |
| Doppler and eclipses from β | finite differences; conical eclipse search | range rate within 0.5 m/s; shadow fraction within 1% |
| Planet positions (Standish) | JPL DE421 via Skyfield, 1900 to 2050 | within 0.5' (inner planets), 1.3' (Mars), 12' (Saturn) |

SGP4 itself is accurate to about 1 km near epoch, so the SGP4 comparison is a sanity check rather than precise validation. Precise validation comes from the GMAT comparison ([validation/gmat](https://github.com/dandmadp/urania/blob/main/validation/gmat), [tests/test_gmat.py](https://github.com/dandmadp/urania/blob/main/tests/test_gmat.py)).

GMAT evaluates J2 and the atmosphere about Earth's true spin axis, which at the J2000 epoch is 7.69" away from the J2000 z axis. urania uses the frame's z axis as the spin axis. The small figures above compare both in a frame whose z axis is the true pole; the larger figures in parentheses are the direct comparison in the J2000 frame.

## Limitations (MVP)

- Perturbations: J2 and atmospheric drag. Lunar/solar perturbations, solar radiation pressure and higher-order gravity come later.
- Atmosphere: Vallado's exponential model (mean solar activity), with height above the reference ellipsoid. Real density varies several-fold with solar activity.
- Maneuvers: impulsive transfers between circular orbits. Rendezvous (phasing) comes later. Body-to-body transfers
  use patched conics (Hohmann estimates, or Lambert arcs between Standish planet positions with powered or
  unpowered flybys); deep-space maneuvers and trajectory optimization are not included. Moon positions are not
  modeled, so dated transfers exist only between planets.
- Time: TDB internally. UTC goes through astropy: `Epoch.from_utc()`, `epoch.utc`, `Epoch.now()`.
- Frames: the inertial frame's z axis is Earth's spin axis (true equator of date, like TEME); precession and nutation
  are not modeled, so give state vectors in that frame. Earth-fixed coordinates use GMST with UT1 ≈ UTC and no polar
  motion (longitudes within about 0.4 km). TLEs use TEME as is.
- Plots: 2D only.

## Examples

- [examples/quickstart.py](https://github.com/dandmadp/urania/blob/main/examples/quickstart.py): ISS → GEO transfer
- [examples/cubesat_decay.py](https://github.com/dandmadp/urania/blob/main/examples/cubesat_decay.py): 3U CubeSat decay and sensitivity to atmospheric density
- [examples/missions_and_operations.py](https://github.com/dandmadp/urania/blob/main/examples/missions_and_operations.py): porkchop plot, Mariner 10 gravity assist, rendezvous and Doppler
- [examples/interplanetary.py](https://github.com/dandmadp/urania/blob/main/examples/interplanetary.py): Hohmann transfers from Earth to every planet with departure dates
- [examples/tle_vs_propagator.py](https://github.com/dandmadp/urania/blob/main/examples/tle_vs_propagator.py): comparison with SGP4 using a real ISS TLE

## License

MIT

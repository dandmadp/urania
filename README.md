# urania

**An astrodynamics library that computes, shows and explains**

urania is a Python library for people who are not professionals but want to compute orbits seriously: CubeSat teams, students, space simulation game players. Every result records the model and assumptions it was computed with, and `.explain()` walks through the formulas and intermediate values step by step.

> Status: MVP in development (v0.0.1). Design: [SPEC.md](SPEC.md). Design decisions: [docs/DECISIONS.md](docs/DECISIONS.md).

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
print(sv.explain())                # how the elements follow from the state vector
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
tr.plot(kind="altitude")           # altitude with an orbit-averaged line
```

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

SGP4 itself is accurate to about 1 km near epoch, so the SGP4 comparison is a sanity check rather than precise validation. Precise validation comes from the GMAT comparison ([validation/gmat](validation/gmat), [tests/test_gmat.py](tests/test_gmat.py)).

GMAT evaluates J2 and the atmosphere about Earth's true spin axis, which at the J2000 epoch is 7.69" away from the J2000 z axis. urania uses the frame's z axis as the spin axis. The small figures above compare both in a frame whose z axis is the true pole; the larger figures in parentheses are the direct comparison in the J2000 frame.

## Limitations (MVP)

- Perturbations: J2 and atmospheric drag. Lunar/solar perturbations, solar radiation pressure and higher-order gravity come later.
- Atmosphere: Vallado's exponential model (mean solar activity), with height above the reference ellipsoid. Real density varies several-fold with solar activity.
- Maneuvers: impulsive transfers between circular orbits. Rendezvous (phasing) and interplanetary transfers come later.
- Time: TDB internally. UTC input goes through `Epoch.from_astropy()`.
- Frames: fixed inertial axes (precession and nutation ignored). TLEs use TEME as is.
- Plots: 2D only.

## Examples

- [examples/quickstart.py](examples/quickstart.py): ISS → GEO transfer
- [examples/cubesat_decay.py](examples/cubesat_decay.py): 3U CubeSat decay and sensitivity to atmospheric density
- [examples/tle_vs_propagator.py](examples/tle_vs_propagator.py): comparison with SGP4 using a real ISS TLE

## License

MIT

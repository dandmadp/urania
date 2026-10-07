# urania — Astrodynamics Library Design Specification (MVP)

> **An astrodynamics library that computes, shows and explains**
> Package name: **urania** (after Urania, the muse of astronomy). Confirmed free on PyPI as of 2026-10-07. Reserve the name with an empty package when work starts.

## 1. Purpose and target users

- Audience: university clubs (CubeSat teams, etc.), students, space simulation game players, educational content creators, space enthusiasts
- What they share: not professionals, but serious about understanding and computing things in space
- Accuracy target: at least textbook level, good enough for a CubeSat team's practical calculations. Space-agency level (high-order gravity fields, etc.) is not a goal
- Every result must be verifiable, and must record which assumptions and models produced it

## 2. Core design principles

1. **The object API is the official entry point**
   - Documentation, README and tutorial examples are all written with the object API
   - Example: `ISS.orbit.transfer_to(GEO).plot()`
2. **A functional core inside**
   - Actual computation is implemented only in pure functions in `urania.core`
   - Object methods only call core functions (one implementation, in one place)
   - Core functions must also work with plain numbers (SI) → for checking against textbook examples
3. **Immutable objects**
   - All major objects are immutable, e.g. `dataclass(frozen=True)`
   - `orbit.propagate(...)` returns a new object without changing the original
4. **Units: SI internally, conversion only at the boundary**
   - Internal computation uses only m, kg, s, K, W
   - Input: values with units (choose astropy.units or pint and record why) are converted; plain numbers are taken as SI
   - Output: SI by default, converted on request
5. **Time: always TDB seconds internally (seconds elapsed since an epoch)**
   - UTC/TAI conversion comes after the MVP, but time is encapsulated in one type so it can be added later
6. **Environment factors are injected as arguments**
   - The library provides the computational framework; the user supplies environment data (radiation, solar flux, etc.)
   - Three forms are accepted: constant / `f(position, t)` function / `Environment` object
   - Internally a constant is wrapped as "a function that always returns the same value", so there is one form
   - The MVP does not implement environment models themselves, only this injection interface
7. **Selectable fidelity levels**
   - 0: two-body / 1: +J2 / 2: +atmospheric drag (MVP scope)
   - 3: +lunar/solar perturbations / 4: +solar radiation pressure, high-order gravity (after the MVP)
   - Example: `orbit.propagate(days=30, model="j2+drag")`
8. **Result metadata**
   - Result objects keep the model, integrator, tolerances and key assumptions used
9. **Explanation and visualization**
   - `.explain()`: prints the formulas, assumptions and intermediate values step by step
   - `.plot()`: 2D orbit plot (matplotlib)

## 3. MVP scope

### Included
- **Objects**
  - `Body`: mass, radius, μ, J2, hook for an atmosphere model
  - `Orbit`: orbital elements ↔ position/velocity (state vector) conversion
  - `Transfer`: maneuver result (total Δv, duration, list of burns)
  - `TLEOrbit`: TLE-based orbit (see TLE below)
- **Presets**
  - Bodies: Sun, Earth, Moon, Mars (the other planets if time allows)
  - Orbits: representative orbits such as `LEO`, `ISS`, `GEO`, `SSO`
- **Orbit propagation**: fidelity levels 0 to 2
  - Keep drag simple, around an exponential atmosphere model
  - Integration: adaptive, based on `scipy.integrate.solve_ivp` (e.g. DOP853)
- **Maneuvers**: Hohmann transfer, bi-elliptic transfer, plane change
- **TLE support**
  - A thin wrapper around the `sgp4` package
  - TLE orbits are propagated only with SGP4 and never mixed with the in-house propagator (separate `TLEOrbit` object)
  - Provide an explicit path that extracts a state vector and converts it to an `Orbit` when needed
- **Explanation/visualization**: `.explain()`, `.plot()`
- **Unit handling, result metadata, environment injection interface**

### Excluded (next version)
- Environment models such as temperature and radiation
- Lunar/solar perturbations, JPL ephemeris (DE440) integration
- Interplanetary transfers, launch window (porkchop) calculations
- 3D visualization, animation
- UTC/TAI/TDB conversion

## 4. Validation

- Compare with textbook examples (Curtis, Vallado worked solutions)
- Compare with NASA GMAT results (reference data stored as test fixtures)
- Compare SGP4 predictions with the in-house propagator using a real ISS TLE (doubles as an example and a test)
- Goal: be able to state in the README a figure such as "within ○○ of GMAT", with evidence
- Test framework: pytest

## 5. Proposed module structure

```
urania/
  core/          # pure functions: orbit conversions, Kepler's equation, maneuver formulas
  bodies.py      # Body, body presets
  orbits.py      # Orbit, orbit presets
  propagation/   # force models per fidelity level, integrators
  maneuvers.py   # Transfer, maneuver calculations
  tle.py         # TLEOrbit (sgp4 wrapper)
  environment.py # environment injection interface
  units.py       # unit conversion boundary
  time.py        # time representation (internal TDB seconds)
  viz.py         # plot
  explain.py     # explain output
tests/
examples/
```

The structure is a proposal; adjust it with a reason if a better layout exists.

## 6. Development order (recommended)

1. core: state vector ↔ orbital elements, Kepler's equation solver + textbook example tests
2. Unit and time boundary (`units.py`, `time.py`)
3. `Body`, `Orbit` objects and presets
4. Propagator: two-body → J2 → drag
5. Maneuvers (`Transfer`)
6. `.plot()`, `.explain()`
7. TLE wrapper
8. GMAT and TLE comparison tests, README examples

## 7. Working requests

- Make the tests pass at the end of each step before moving on
- Record design decisions (such as the unit library choice) with a short rationale
- Show code changes as code blocks containing only the changed parts

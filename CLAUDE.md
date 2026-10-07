# CLAUDE.md

urania: a Python astrodynamics library that computes, shows and explains. Full specification in [SPEC.md](SPEC.md), design decisions in [docs/DECISIONS.md](docs/DECISIONS.md).

## Commands

Windows. Call the virtual environment's Python directly.

```
.venv/Scripts/python -m pytest -q                         # all tests
.venv/Scripts/python -m pytest tests/core/test_kepler.py  # one file
.venv/Scripts/python -m pip install -e ".[dev]"           # reinstall after dependency changes
```

## Architecture (summary of SPEC.md section 2)

- `src/` layout. The package is `src/urania/`; tests mirror the package under `tests/` (`tests/core/...`).
- **Computation lives only in pure functions in `urania.core`.** Object methods (`Orbit`, `Body`, ...) only call core functions.
- Core functions take and return **SI numbers** (m, kg, s, K, W, rad). Unit conversion happens only at the `units.py` boundary.
- Time is TDB seconds internally.
- Major objects are immutable (`dataclass(frozen=True)` or `NamedTuple`). Return new objects instead of mutating.
- Result objects keep the model, integrator, tolerances and assumptions as metadata.

## Code rules

- Everything is in English: code, docstrings, comments, error messages, `explain()` output, tests and documents (DECISIONS D26).
- Cite formula and algorithm sources (Curtis, Vallado).
- Normalize angles with `core.kepler._wrap_2pi` / `_wrap_pi` (a raw `% 2π` can round to exactly 2π).
- Singular orbit (circular, equatorial) conventions follow DECISIONS D2.

## Test rules

- New computations get both **textbook example tests** (example number in the docstring) and **property tests** (round trips, equation residuals).
- Match tolerances to the textbook's printed precision (significant digits). Do not mistake rounding for a bug, and do not loosen tolerances excessively.
- Earth's μ differs between textbooks: Curtis 398600 km³/s², Vallado 398600.4418 km³/s².

## Workflow

- Follow the step order in SPEC.md section 6; make all tests pass at each step before moving on.
- Record design decisions with a short rationale in `docs/DECISIONS.md`, numbered D1, D2, ...
- When reporting to the user, show only the changed parts as code blocks.
- After each step, commit and push to `origin` (github.com/dandmadp/urania).
- Talk with the user in Korean (the repository itself is English only).

## Progress

- [x] 1. core: state vector ↔ orbital elements, Kepler's equation
- [x] 2. Unit and time boundary (astropy, `Epoch`)
- [x] 3. `Body`, `Orbit` objects and presets
- [x] 4. Propagator: two-body → J2 → drag (`Orbit.propagate` → `Trajectory`)
- [x] 5. Maneuvers (`Orbit.transfer_to` → `Transfer`)
- [x] 6. `.plot()`, `.explain()` (`viz.py`, `explain.py`)
- [x] 7. TLE wrapper (`TLEOrbit`, sgp4)
- [x] 8. GMAT and TLE comparison, README, examples (GMAT R2026a at `D:\gmat-win-R2026a`; rerun with `bin/GmatConsole.exe --run <script>`)

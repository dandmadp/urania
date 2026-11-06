# Contributing

Thank you for helping! Bug reports with a short reproducing script, corrections to formulas or sources, new
validation cases and documentation fixes are all welcome:
[github.com/dandmadp/urania/issues](https://github.com/dandmadp/urania/issues).

## Setting up

```text
git clone https://github.com/dandmadp/urania
cd urania
python -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"      # Windows: .venv\Scripts\python
.venv/bin/python -m pytest -q
```

## How the code is organized

| place | contents |
|---|---|
| `src/urania/core/` | pure functions, SI numbers in and out; all the physics lives here |
| `src/urania/*.py` | immutable objects (`Orbit`, `Trajectory`, `Transfer`, ...) that call the core |
| `src/urania/explain.py`, `viz.py` | `explain()` and `plot()` for every result |
| `tests/core/`, `tests/` | tests mirroring the package |
| `validation/gmat/` | GMAT scripts generating reference data in `tests/fixtures/gmat/` |
| `docs/` | this site (MkDocs); `docs/DECISIONS.md` records design decisions |

## Rules for changes

- **Cite the source** of every formula or algorithm (Curtis, Vallado, Montenbruck & Gill, ...) in the docstring.
- **Test against a textbook example** (its number in the test docstring), with a tolerance matching the printed
  precision, **and** with a property test (a round trip, a conservation law, an equation residual).
- Units: core functions use SI; conversion happens only at the boundary (`urania.units`).
- Objects are immutable; methods return new objects.
- Results carry the model and assumptions they used, and gain `explain()` and, where useful, `plot()`.
- Everything in the repository is in English.
- Record a design decision with a short rationale in `docs/DECISIONS.md`.
- Public API changes update `tests/fixtures/public_api.txt` (see [API stability](stability.md)) and the changelog.

## Documentation

Code blocks in `README.md` and `docs/` are run by the test suite, so examples cannot go stale. Preview the site
with:

```text
pip install mkdocs-material "mkdocstrings[python]"
mkdocs serve
```

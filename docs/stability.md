# API stability

## What is public

The public API is

- every name in `urania.__all__` (what `import urania as ur` exposes), with its documented methods, properties and
  fields;
- the functions of the `urania.core` modules listed in the [API reference](api.md);
- `urania.coordinates`, `urania.units.to_si` and `urania.units.UnitWarning`.

Names starting with an underscore, and modules not listed, are internal and may change at any time.

The whole public surface, with every signature, is recorded in `tests/fixtures/public_api.txt`; a test fails when
it changes, so no change to the API happens by accident.

## Versioning

urania follows semantic versioning with the usual 0.x convention:

| release | may contain |
|---|---|
| patch (0.1.0 → 0.1.1) | bug fixes, documentation, new optional arguments; never a breaking change |
| minor (0.1 → 0.2) | new features; breaking changes only after a deprecation period |
| 1.0 | the API is declared stable; breaking changes then wait for 2.0 |

## Deprecation

A public name that is renamed or removed keeps working for at least one minor release and emits a
`UraniaDeprecationWarning` (a `FutureWarning`, so it is visible by default) that says what to use instead and
when the old name goes away. Every deprecation is listed in the changelog.

## Numbers

Results may change in a minor release when a model becomes more accurate (for example a better ephemeris or
atmosphere); such changes are listed in the changelog with their size. Patch releases change numbers only to fix
bugs.

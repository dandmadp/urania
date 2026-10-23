# Changelog

## Unreleased

### Added
- Mercury, Venus, Jupiter, Saturn, Uranus and Neptune, and 20 major moons (Phobos to Triton), with JPL constants.
  Bodies record their parent, mean orbit radius and sphere of influence (`Body.soi`, `Body.orbital_period`).
- `Body.hohmann_to`: patched-conic Hohmann transfer between planets, between moons of one planet, or between a
  planet and its moon (`BodyTransfer` with v∞, C3, departure and capture Δv, phase angle, synodic period, stay time,
  `explain()` and `plot()`); `after=` finds the next departure date from JPL's approximate planet positions.
- `core.interplanetary` (patched-conic formulas) and `core.planets` (Standish mean-element ephemeris).
- Ground tracks (`Trajectory.ground_track`), station passes (`Trajectory.passes` with `GroundStation`),
  eclipses (`Trajectory.eclipses`) and lifetime estimates (`Orbit.lifetime`), each with `explain()` and `plot()`.
- `Trajectory.state_at(t)`: position and velocity at any time between samples.
- `orbit.qty` (Quantities in km, deg, min, ...) and `orbit.summary()`.
- `core.frames` (GMST, Earth-fixed and geodetic coordinates, look angles), `core.sun` (Sun position, shadow),
  `core.lifetime` (orbit-averaged decay).

### Changed
- The inertial frame is defined as the true equator of date (z = Earth's spin axis).
- `Trajectory` and `Burn` print a one-line summary.

### Fixed
- `core.frames.look_angles` could return an azimuth of 360° instead of 0°.

## 0.0.2 (2026-10-08)

### Fixed
- Nearly parabolic orbits (|1 - e| below about 1e-6) were propagated with errors of hundreds of meters to
  hundreds of kilometers, and Kepler's equation could fail to converge. Analytic two-body propagation now
  uses universal variables.
- Python 3.10: parabolic orbits crashed (`math.cbrt` needs 3.11) and `Epoch.from_iso` rejected fractional
  seconds that were not 3 or 6 digits.
- `sun_synchronous(body=...)` around a body other than Earth applied Earth's year silently; it now requires
  `raan_rate`.
- Negative or zero `cd`, `area` or `mass` were accepted (negative mass accelerated the orbit); now rejected.
- Bodies with a non-positive μ or radius, or a flattening outside [0, 1), are rejected.
- Non-finite or array propagation times are rejected with a clear message.
- `transfer_to` rejected a central body equal to, but not the same object as, the initial orbit's body.
- TLE text indented inside a triple-quoted string was rejected.
- Clearer errors for `plane_split` typos, `Epoch + Epoch`, a zero position vector and time zones in
  `Epoch.from_iso`.
- `Epoch.utc` now has microsecond resolution (milliseconds lost about 4 m of position in low orbit).
- The specific energy of a parabola was -0.0; plot titles missed a space; `Transfer` repr said "1 burns".

### Added
- `Epoch.from_utc`, `Epoch.utc`, `Epoch.now`, and `number + Epoch`.
- `Orbit.from_apsides` (orbit from periapsis and apoapsis altitudes) and `Orbit.after`
  (shorthand for `propagate(...).final`).
- `Trajectory.elements` (osculating elements at every sample) and `Trajectory.epochs`.
- `TLEOrbit.period`.
- Lists of Quantities, such as `[7000 * u.km, 0 * u.km, 0 * u.km]`, are accepted wherever a vector is.
- A warning when `area`, `mass` or `density` are given to a model without a drag term.

### Changed
- Tested on Python 3.10 to 3.13 and with the oldest allowed dependency versions (GitHub Actions).

## 0.0.1 (2026-10-08)

First release: orbits and presets, two-body/J2/drag propagation, Hohmann and bi-elliptic transfers with
plane changes, TLE propagation with SGP4, `explain()` and `plot()`, validated against Curtis and Vallado
examples, NASA GMAT and SGP4.

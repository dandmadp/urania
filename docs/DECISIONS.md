# Design decisions

## D1. Store orbit size as the semi-latus rectum p, not the semi-major axis a (step 1)
- a is infinite for a parabola and negative for a hyperbola, which is awkward for a single type.
- p is positive and finite for every conic. a is provided through the `Elements.a` property.

## D2. Angle conventions for singular orbits (step 1)
- Equatorial (i ≈ 0, π): raan = 0, reference axis is inertial x.
- Circular (e ≈ 0): argp = 0, ν is measured from the node (from x if also equatorial).
- This packs Vallado's true longitude and argument of latitude into the usual six elements without extra fields.
  Tests guarantee that `coe_to_rv(rv_to_coe(r, v))` is the identity even for singular orbits.
- Detection tolerances are dimensionless 1e-11 (`CIRCULAR_TOL`, `EQUATORIAL_TOL`).

## D3. Core functions are scalar and `math`-based (step 1)
- Kepler's equation and anomaly conversions take scalar floats. They are called inside propagation loops,
  so this avoids numpy array overhead and keeps the code close to the textbook formulas.
- If vectorization is needed (e.g. bulk sampling for plots), add separate functions.

## D4. The parabolic mean anomaly is Barker's M = D + D³/3 (step 1)
- So that `mean_to_true`/`true_to_mean` handle all three conic types from the eccentricity alone.

## D5. astropy for units (step 2)
- UTC/TAI/TDB conversion will be needed after the MVP, and `astropy.time` provides it. With pint we would have to implement time scales ourselves.
- It is the de facto standard in astronomy and space, so target users are likely familiar with it.
- Its downsides (install size, Quantity overhead) do not affect computation because units are converted only at the boundary (`units.py`).

## D6. Plain numbers are SI, including angles (step 2)
- `to_si(30, ANGLE)` is 30 rad. Degrees must be given explicitly as `30 * u.deg`.
- Making angles an exception would mean two rules and more mistakes.

## D7. Time is one float64 of TDB seconds since J2000 (step 2)
- Encapsulated in `Epoch(tdb_seconds)`. Over 2000 to 2100 float64 resolves about 1e-7 s, which is under 1 mm
  of position error at LEO speed (7.8 km/s), enough for the MVP accuracy target.
- Calendar and JD input is read as TDB. The TDB calendar has no leap seconds, so plain date differences are exact.
- Other time scales go through `Epoch.from_astropy()`, where astropy converts them. No in-house conversion.

## D8. Body stores μ (GM) instead of mass (step 3)
- μ is determined directly from orbit observations and is hundreds of thousands of times more accurate than G or M alone (G itself is uncertain to about 2e-5).
- Mass is provided as the property `Body.mass = μ/G`. Each preset records the source of its constants in `source`.

## D9. An Orbit is defined by its state vector (r, v, epoch) (step 3)
- The state vector has no singularities for circular, equatorial or parabolic orbits, and matches the propagator's input/output format.
- Orbital elements are computed once with `cached_property`. The object is immutable, so the cache never goes stale.
- r and v are copied into read-only arrays. Value comparison of numpy arrays is ambiguous, so `eq=False` (identity).

## D10. The ISS preset is an Orbit (step 3)
- SPEC's example `ISS.orbit.transfer_to(GEO)` implies a spacecraft object, but spacecraft are not in the MVP.
  The preset list (LEO, ISS, GEO, SSO) classifies them as orbits, so the form is `ISS.transfer_to(GEO)`.
- Presets are representative (ISS: 420 km circular, 51.64°). Use a TLE for the actual position.
- The GEO radius comes from Earth's rotation rate, the SSO inclination from the J2 nodal rate.

## D11. propagate() returns a Trajectory, not an Orbit (step 4)
- SPEC asks both for "return a new object" and "keep result metadata". Rather than mixing metadata into Orbit,
  it returns a `Trajectory` with the time series (t, r, v) and `PropagationInfo` (model, integrator, tolerances, assumptions, stop reason).
- The final orbit is `trajectory.final`, intermediate ones `orbit_at(i)`. The step 6 `.plot()` uses the time series directly.

## D12. Two-body uses the analytic solution by default, everything else DOP853 (step 4)
- `method="auto"`: two-body uses the Kepler solution (exact and fast); J2 and drag use `solve_ivp` DOP853.
  `method="DOP853"` integrates two-body numerically too, which is useful for checking the integrator.
- Default tolerances rtol=1e-12, atol=1e-8. With rtol=1e-10 an eccentric orbit drifted 51 m from the analytic solution over 10 days;
  with 1e-12, 0.5 m. Cost is about 1.7x (30 days of ISS j2+drag takes about 5.5 s). Accuracy was chosen for the GMAT comparison.
- Integration stops on reaching the body's surface and records it in `info.terminated`.

## D13. All environment values become an Environment (step 4)
- Constant → `Constant`, function f(r, t) → `Function`, `Environment` objects as is. `as_environment()` converts.
- t is TDB seconds since J2000 (float). It is called every step, so no Epoch object is created.
- Each Environment has a `description` that goes into the assumptions of the result metadata.
- Earth's default atmosphere is the Vallado table 8-4 exponential model (`Body.atmosphere`). `density=` overrides it.

## D14. A single propagation.py module instead of a package (step 4)
- SPEC suggested a `propagation/` package, but at the current size (about 200 lines) one module is easier to read.
  Force models live in `core/forces.py` and integration in `core/propagate.py`, so propagation.py only assembles them.
  Split it into a package when fidelity levels 3 and 4 are added.
- For the same reason atmospheric density has no separate file: the table and density function are in `core/forces.py`,
  and `ExponentialAtmosphere` is in `environment.py`, since an atmosphere model is an Environment.
- The two-body analytic solution also stops at the surface, like integration does (brentq between samples finds the impact time).

## Note (resolved in step 6, D19): J2 oscillation of the osculating semi-major axis
- With J2 the osculating semi-major axis oscillates strongly with a short period (0 to -12 km at ISS altitude). Drag-only decay over 30 days is 2.65 km,
  but comparing only the endpoints makes it look like 14 km. `Trajectory.explain()` warns and shows the orbit-averaged change; the altitude plot shows a one-orbit moving average.

## D15. Transfers are built from state vectors, not just Δv formulas (step 5)
- `Transfer` holds `Burn`s with position, epoch and Δv vector, plus the orbit right after each burn (`orbits`).
  Separately from the magnitude-only formulas (core/maneuvers.py), it adds Δv to the state vector and links burns with the analytic solution.
  The step 6 plot() uses it directly, and tests cross-check formulas against the vector results.
- MVP limits: initial and target orbits must be circular (e ≤ 1e-3); phasing within the target orbit (rendezvous) is not matched.

## D16. Burns happen at the node when planes differ; Hohmann plane change is split optimally (step 5)
- A plane change is only possible at the line of nodes. If the start is not at a node, wait for the next node
  (`Transfer.coast`). A Hohmann transfer covers half an orbit, so burn 2 happens at the opposite node.
- Default `plane_split="optimal"`: the split of the plane change between the two burns minimizes total Δv
  (ISS→GEO: 2.89° at burn 1, about 40 m/s cheaper than doing it all at apoapsis). A number 0 to 1 can be given instead.
- A bi-elliptic transfer does the whole plane change at the slow intermediate apoapsis.

## D17. pytest uses the importlib import mode (step 5)
- So core and object-layer tests can share file names, such as `tests/core/test_maneuvers.py` and `tests/test_maneuvers.py`,
  `--import-mode=importlib` is enabled (the mode pytest recommends).

## D18. matplotlib is imported only when plotting (step 6)
- urania itself imports faster. Plot labels are in English like the rest of the library (D26).

## D19. explain() returns an Explanation object and cross-checks values with core (step 6)
- No print side effect. `__repr__` is the text, so it reads directly in a REPL or Jupyter, and print() works too.
- Intermediate values (circular speed, transfer ellipse speeds, ...) are computed with core functions (`vis_viva`, `combined_dv`, ...).
  If a Δv recomputed from formulas disagrees with the state-vector Burn, a warning is added (`warnings`).

## D20. Transfer plots unfold each orbit into its own plane (step 6)
- Projecting onto one plane distorts a target orbit in another plane into an ellipse. All burns lie on the node axis,
  so with x along the node and (ĥ × x̂) as each orbit's y axis, the unfolded orbits connect.
- Trajectory plots project onto the initial orbit plane (J2 precession shows in the projection). Altitude plots overlay a one-orbit moving average.

## D21. TLEs are a separate TLEOrbit propagated only with SGP4 (step 7)
- TLE elements are SGP4-specific mean elements; using them as classical elements is wrong. `TLEOrbit` does not inherit from `Orbit`,
  and its `propagate()` uses only SGP4. The result is the same `Trajectory`, so plot and explain work as is.
- The only way to the urania propagator is the explicit `to_orbit(epoch)`, which extracts a state vector.
- The checksum and the satellite numbers of both lines are checked on construction. On an SGP4 error (decay, ...) the output stops
  before it and `info.terminated` records it.

## D22. The TLE frame TEME is used as the inertial frame (step 7)
- TEME → GCRF needs precession and nutation models, outside the MVP scope (no time or frame transformations).
- The urania propagator also assumes "fixed inertial axes, spin axis = z". TEME's z axis is the true spin axis at that time,
  so comparing SGP4 with urania is actually consistent. The assumptions list states that the frame is TEME.

## D23. TLE epochs (UTC) ↔ Epoch (TDB) go through astropy (step 7)
- D7 ruled out in-house time scale conversion. Only the TLE epoch and SGP4 call times go through astropy Time.
- Gravity constants are WGS72, the TLE convention. An Orbit from `to_orbit()` uses the Earth (WGS 84) constants.

## D24. GMAT comparison uses a script generator and fixed fixtures (step 8)
- GMAT output cannot be produced in this repository, so it is never made up. `validation/gmat/make_scripts.py` writes GMAT scripts
  from urania scenarios, and the user commits the results of running them in GMAT (`tests/fixtures/gmat/*.txt`).
  `tests/test_gmat.py` skips if the fixtures are missing. Tolerances were finalized from the GMAT R2026a results (D27).
- Conditions are matched to reduce differences: J2000 epoch (zero precession), a test body with GMAT's EGM96 constants, RK89 at 1e-13.

## D25. README code is executed by the tests (step 8)
- `tests/test_docs.py` runs every ```python block in the README and checks that the Transfer repr written in the README matches.
  It also runs the example scripts (except cubesat_decay.py, which is slow with two 60-day runs).
- The urania-vs-SGP4 figure is the maximum measured at 1-minute steps (2.34 km). Looking at only 4 times almost gave 0.98 km.

## D26. Everything in English
- Code, docstrings, error messages, `explain()` output, tests and documents are all in English, for an international open-source audience.
  One language avoids keeping two versions in sync. This replaces the earlier Korean-docstring rule and the
  "explain() text is Korean" part of D18.

## D27. Atmospheric density uses the height above the reference ellipsoid (step 8, GMAT comparison)
- GMAT's exponential atmosphere uses the same Vallado table but the height above the reference ellipsoid. With a spherical
  height urania differed from GMAT by 28.6 km after 3 days at 400 km and 51.64°: at the same |r| the ellipsoid height is up to
  about 21 km larger at high latitude, so the density is 20 to 25% lower.
- `Body.flattening` was added (Earth WGS 84 1/298.257223563, Moon 0.0012, Mars 0.00589), and `core.forces.geodetic_altitude`
  computes the exact height (iterated geodetic latitude). `ExponentialAtmosphere(radius, flattening)` uses it.
  With the same spin axis as GMAT the difference is now 3.3 m.
- The surface-impact event and `Trajectory.altitude` stay spherical (|r| - R).

## Note: the frame's z axis is taken as the spin axis
- J2, atmospheric rotation and the ellipsoid all use the inertial z axis. GMAT uses the true pole of date, 7.69" from the J2000 z axis
  at the J2000 epoch, which gives a 144 m difference after 7 days in the J2 scenario (3.3 m once both use the same axis).
- Not changed in this step; recorded here because it matters for state vectors given in the J2000 frame at epochs far from J2000.

## D28. Input validation and convenience API (code review)
- Inputs that used to give silently wrong results are now rejected: non-positive `cd`, `area`, `mass`; non-finite or array
  durations; Bodies with non-positive μ or radius or a flattening outside [0, 1); `sun_synchronous` around a body other
  than Earth without its own `raan_rate` (Earth's year was applied silently).
- Drag inputs (`area`, `mass`, `density`) given to a model without a drag term raise a `UserWarning` rather than an error,
  so the same keyword set can be reused across models.
- Central bodies are compared by value, not identity, so a copy of Earth works in `transfer_to`.
- Additions: `Epoch.from_utc`, `Epoch.utc`, `Epoch.now` (UTC through astropy, consistent with D7/D23),
  `Orbit.from_apsides`, `Orbit.after` (shorthand for `propagate(...).final`, raises if the run stopped early),
  `Trajectory.elements` and `Trajectory.epochs`, `TLEOrbit.period`, lists of Quantities in `units.to_si`,
  indented TLE text.

## D29. Analytic two-body propagation uses universal variables (bug hunt)
- Randomized tests found that the element-based Kepler solution was off by hundreds of meters to hundreds of
  kilometers for nearly parabolic orbits (|1 - e| below about 1e-6): the digits of 1 - e, a and the mean motion are lost.
  It also failed to converge (RuntimeError) for some of them.
- `kepler_states` now uses universal variables with Stumpff functions (Curtis algorithm 3.4, Vallado algorithm 8),
  solved by the Laguerre-Conway iteration. One formula covers all conics; elliptic times are reduced modulo the period.
  GMAT and textbook results are unchanged.
- The anomaly functions in `core.kepler` evaluate E - e sin E and e sinh H - H without cancellation and start
  Newton's method from a cubic estimate, so they converge for nearly parabolic orbits. The public `true_to_mean`
  still returns elliptic M in [0, 2π); near periapsis of a nearly parabolic orbit that wrapping limits the precision
  of a round trip through M, which the propagation path avoids.

## D30. Python 3.10 support is tested (bug hunt)
- 0.0.1 declared Python 3.10 but was only tested on 3.11. On 3.10 `math.cbrt` does not exist (parabolic orbits crashed)
  and `datetime.fromisoformat` rejects fractional seconds that are not 3 or 6 digits.
  Fixed with `numpy.cbrt` and a small ISO normalizer.
- GitHub Actions runs the tests on Python 3.10 to 3.13 (Linux), 3.11 on Windows, and 3.10 with the oldest
  dependency versions allowed by pyproject.toml (`uv --resolution lowest-direct`).

## D31. Inertial frame = true equator of date; Earth-fixed via GMST (v0.1)
- urania's inertial frame is now defined explicitly: z along Earth's spin axis (true equator of date, like TEME), which is
  what J2, the co-rotating atmosphere and the ellipsoid already assumed. Precession and nutation are not modeled.
- Earth-fixed coordinates rotate by GMST (IAU 1982, Vallado eq. 3-47) with UT1 ≈ UTC and no polar motion. Against Skyfield
  (full Earth orientation) latitude and height are identical and longitude differs by UT1 − UTC (0.0008° in Dec 2019).
  This avoids IERS downloads and is ample for ground tracks and contact planning.
- `core.frames` holds the pure functions (gmst82, rotate_z, geodetic, geodetic_to_ecef, look_angles). The atmosphere's
  height now uses the same geodetic code (a scalar math path for speed inside the integrator).

## D32. Passes and eclipses: sampled search refined between samples (v0.1)
- Elevation and shadow functions are evaluated at the trajectory samples, sign changes are refined by root finding, and
  pass peaks by bounded minimization. States between samples come from `Trajectory.state_at` (cubic Hermite, well under
  a meter at 100 samples per orbit). Sampled local maxima just below the threshold are refined so short, low passes
  between samples are not missed.
- Eclipses use the conical shadow with partial disk overlap (Montenbruck & Gill 3.4.2) and the low-precision Almanac
  Sun position (Vallado algorithm 29), spherical Earth. Randomized tests compare against 1 s brute-force sampling.
- Skyfield is a dev dependency used only as a validation reference. Its 1.45 release locates events less precisely (one
  set time 1.65 s off), so the test checks Skyfield's own elevation at urania's event times (within 0.005°).

## D33. Lifetime: orbit-averaged decay with a J2 correction (v0.1)
- The averaged method integrates da/dt = ⟨2 a_T / n⟩ over a circular orbit, starting from the semi-major axis averaged
  over one J2 orbit (the osculating value at the node is about 6 km higher at 350 km, 51.6°).
- A circular orbit of constant radius was 12 to 17% off the full propagation at 280 km: the J2 orbit's radius varies
  with latitude, which changes the mean density (scale height ~40 km). The rate is therefore multiplied by K(a), the
  ratio of the rate averaged along one-orbit J2 propagations to the circular rate, evaluated at six altitudes and
  interpolated. Agreement with full j2+drag propagation is now within 2% from 250 to 350 km at all inclinations,
  at about 0.2 s per estimate. Eccentric orbits (e > 0.02) need method="numerical".
- Reentry is when the mean altitude falls below 100 km (the remaining descent takes minutes).

## D34. Usability: concise reprs and Quantities (v0.1)
- `Trajectory` and `Burn` print one line instead of arrays. `orbit.qty.<name>` returns astropy Quantities in km, deg,
  min, deg/day; `orbit.summary()` lists them. Internal values stay SI (D6); this is an output convenience only.

## D35. Planets and moons: JPL constants, parent and orbit radius on Body (v0.1)
- μ: DE440 for Mercury, Venus, Mars; the JPL satellite ephemerides for the giant planets (JUP365, SAT441, URA111,
  NEP097, planet without moons) and for every moon. Radii and rotation: JPL SSD; J2 and flattening: NASA fact sheets
  (their J2 reference radius is not stated; taken as the equatorial radius). A negative rotation rate is retrograde.
- `Body` gains `parent`, `semi_major_axis` (mean, from Standish for planets and JPL mean elements for moons) and
  `ephemeris`. They do not take part in equality, so a body created with the same physical constants still equals
  the preset, as before.
- Moons do not get J2 or flattening (mostly unmeasured or irrelevant for the transfer estimates they serve).

## D36. Body-to-body transfers: patched-conic Hohmann, Standish positions for dates (v0.1)
- `Body.hohmann_to` covers siblings (same parent), planet → its moon and moon → its parent. Other pairs (Earth →
  Titan) raise an error that suggests chaining: a single Hohmann ellipse cannot connect bodies with different parents.
- Circular coplanar orbits at the mean semi-major axes (Curtis ch. 8). This is the standard first estimate; real
  missions solve Lambert's problem (porkchop plots), which is left for later.
- Departure dates: the first time after the given epoch at which the target's heliocentric ecliptic longitude leads
  the origin's by the Hohmann phase angle, using Standish's mean elements (JPL, 1800 to 2050, arcminute accuracy).
  This avoids a DE ephemeris download. Checked once against DE421 (Skyfield), 1900 to 2050: longitude errors
  below 0.5' for the inner planets, 1.3' for Mars, 7' for Jupiter and 12' for Saturn, as Standish states. The search scans 1.2 synodic periods and refines with Brent's method; dates
  outside 1800 to 2050 warn. Moon positions are not modeled, so moon transfers give phasing but no date.
- A body whose sphere of influence is smaller than its radius (Phobos, Deimos) cannot be orbited in this model;
  transfers to it without capture still work.

## D37. Lambert solver: universal variables with bracketing (v0.1)
- Curtis algorithm 5.2 / Vallado algorithm 58 in the universal variable z, solved with Brent's method on a bracket
  instead of Newton's method: the time of flight is monotonic in z for direct transfers, so the root is always
  found. The region where y(z) < 0 (no real orbit) counts as time 0, which keeps the bracket continuous.
- Multi-revolution arcs use the same formula on z in ((2πN)², (2π(N+1))²); the time of flight has one minimum
  there, found by bounded minimization, and the "left"/"right" branches lie on either side.
- Izzo's algorithm is faster, but universal variables match the textbooks the library cites and solve in about
  0.15 ms, enough for porkchop grids of thousands of points. Transfers of exactly 0° or 180° raise an error (the
  plane is undefined).

## D38. Gravity assists: Lambert legs with powered flybys (v0.1)
- A mission is a list of planets and dates. Each leg is a Lambert arc between planet centers (Standish states);
  each flyby joins the incoming and outgoing v∞. If their magnitudes differ, a periapsis burn makes up the
  difference and the periapsis follows from asin(1/e_in) + asin(1/e_out) = δ (the MGA model of Izzo 2010). This
  turns any set of dates into a computable mission instead of requiring an unpowered match.
- Flybys below `min_flyby_altitude` (the surface by default) are flagged infeasible rather than rejected, so
  porkchop-style searches and explain() can show why a date fails.
- Validated with Mariner 10: the Venus flyby comes out nearly unpowered (19 m/s) at 5,739 km altitude against the
  5,768 km flown. Deep-space maneuvers and automatic date optimization are left out.

## D39. Satellite operations: chosen for CubeSat teams (v0.1)
- Rendezvous (Lambert, any orbits), phasing (Curtis 6.6) and deorbit reuse the `Transfer` object, so explain(),
  plot() and the new `propellant()` work unchanged.
- Relative motion uses the chief's LVLH frame (Curtis ch. 7: x radial, y along-track, z normal) with the exact
  rotating-frame transformation; `core.relative` adds the Clohessy-Wiltshire solution and two-impulse rendezvous.
  Deployment and formation questions are the most common relative-motion use for small satellites.
- Doppler uses the first-order shift -f ṙ/c; the neglected second-order terms are about 1e-10 of the carrier,
  far below receiver tuning steps.
- Footprint and beta angle assume a spherical Earth; the eclipse fraction from β uses a cylindrical shadow and
  is within 1% of the orbit of the conical eclipse search. `Trajectory.eclipses()` remains the precise tool.

## D40. Beginner conveniences without giving up SI (v0.1)
- Plain numbers stay SI (D6): changing that would silently change results for existing users. Instead, the unit
  boundary also accepts strings with units ("500 km", parsed by astropy), so beginners need no astropy import.
- The most common silent mistakes (degrees or kilometers passed as plain numbers) now raise a `UnitWarning` with
  the fix, attached to the caller's line. Thresholds are chosen so correct SI input never warns: angles above 2π,
  altitudes between 0 and 1000 m (no orbit is that low), semi-major axes or positions under 1% of the body's
  radius. A warning, not an error, because scripts may compute such values on purpose.
- Date strings are TDB, like `Epoch.from_iso`, so the same string means the same instant everywhere; a trailing
  "Z" or "UTC" selects UTC explicitly (the difference is 69 s). Timezone-aware datetimes are converted from UTC.
- `lambert_to` takes the arrival as a date or a time of flight; a string starting with YYYY-MM-DD is a date,
  anything with a time unit is a duration.
- `summary()` gives the few numbers people look for first, in km/s, days and degrees, plus the propellant from
  the rocket equation when mass and Isp are given; `explain()` stays the full derivation.

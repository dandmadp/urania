# Generating GMAT reference data

How to produce the reference data that compares the urania propagator with NASA GMAT.

1. Install GMAT: https://sourceforge.net/projects/gmat/ (R2022a or newer recommended)
2. Generate the scripts (only needed after changing scenarios):
   ```
   python validation/gmat/make_scripts.py
   ```
3. Open GMAT and run `twobody.script`, `j2.script` and `j2_drag.script` one after another (F5).
   Output is saved to `tests/fixtures/gmat/<scenario>.txt`.
   From the command line: `GMAT.exe --run --exit validation/gmat/j2.script`
4. Compare:
   ```
   pytest tests/test_gmat.py -s
   ```

## Scenarios

| Name | Orbit | Force model | Duration |
|---|---|---|---|
| twobody | a 8000 km, e 0.1, i 30° | point mass | 1 day |
| j2 | circular 420 km, i 51.64° | EGM96 degree 2 order 0 (J2) | 7 days |
| j2_drag | circular 400 km, i 51.64° | J2 + exponential atmosphere (C_D 2.2, A 1 m², m 100 kg) | 3 days |

- Epoch 2000-01-01 12:00:00 TT. Precession is zero there, so GMAT's true spin axis and urania's fixed z axis nearly coincide.
- Gravity constants come from a test body matched to GMAT's EGM96 values (μ 398600.4415 km³/s², R 6378.1363 km).
- Output every hour, EarthMJ2000Eq frame, km and km/s.
- The output path in the scripts is the absolute path on the machine that generated them. On another machine, rerun step 2.

## Results (GMAT R2026a)

| scenario | EarthMJ2000Eq frame | true-pole frame |
|---|---|---|
| twobody, 1 day | 0.009 m | - |
| j2, 7 days | 144 m | 3.30 m |
| j2_drag, 3 days | 73 m | 3.32 m |

## Sources of difference (all identified)

- GMAT's exponential atmosphere table (`data/atmosphere/earth/EarthExponentialAtmosphereData.txt`) is identical
  to urania's Vallado table 8-4. GMAT measures the height above the reference ellipsoid; urania does too since D27
  (with a spherical height the j2_drag difference was 28.6 km). The scripts pin Earth's radius and flattening.
- GMAT evaluates J2 and the atmosphere about the true spin axis (nutation, EOP), 7.69" from the MJ2000Eq z axis at
  the epoch. urania uses the frame's z axis. `tests/test_gmat.py` also compares in a frame whose z axis is the true
  pole, which removes this difference.

## GMAT scripting notes

- The ElapsedSecs stop condition counts from the start of each `Propagate` command, so every loop pass propagates 3600 s.
- A ReportFile must not be named `Report` (it clashes with the `Report` command).
- GMAT writes the header line again on every `Report` command; the loader keeps numeric rows only.
- Command line: `GmatConsole.exe --run <script>` from GMAT's `bin` folder (about 1 to 2 s per scenario).

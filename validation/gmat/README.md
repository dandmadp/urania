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

## Known sources of difference

- Whether GMAT's exponential atmosphere (`Exponential`) matches urania's Vallado table must be confirmed from the results.
  If your GMAT version does not support `AtmosphereModel = Exponential`, the j2_drag script will fail.
- GMAT uses the true Earth orientation (nutation, EOP) for J2 and atmospheric rotation. urania uses a fixed z axis and a constant rotation rate.

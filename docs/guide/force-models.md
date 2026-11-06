# Force models

`Orbit.propagate(model=...)` sums the terms you name. Without perturbations the analytic Kepler solution is used;
otherwise Cowell's method with scipy's DOP853 (rtol 1e-12, atol 1e-8 m by default).

| term | acceleration | parameters | source |
|---|---|---|---|
| `twobody` | −μ r/r³ | – | always included |
| `j2` | Earth's oblateness | `body.J2` | Vallado eq. 8-30 |
| `zonal` | zonal harmonics J2…Jn | `body.zonal` (EGM-08 J2–J6 for Earth) | gradient of Vallado eq. 8-19 |
| `drag` | −½ ρ (C_D A/m) \|v_rel\| v_rel | `cd`, `area`, `mass`, `density` | Vallado eq. 8-28 |
| `sun` | Sun point mass | – | Montenbruck & Gill eq. 3.37; Sun: Vallado alg. 29 |
| `moon` | Moon point mass | – | Montenbruck & Gill eq. 3.37; Moon: Vallado alg. 31 |
| `srp` | radiation pressure on a sphere, conical shadow | `cr`, `area`, `mass` | Montenbruck & Gill eq. 3.75 |

Fidelity levels are shortcuts: 0 = `twobody`, 1 = `j2`, 2 = `j2+drag`, 3 = `j2+drag+sun+moon`,
4 = `zonal+drag+sun+moon+srp`.

## Atmosphere models

| model | depends on | use |
|---|---|---|
| `Earth.atmosphere` (default) | height above the ellipsoid | quick estimates; mean solar activity |
| `NRLMSISE00(f107, f107a, ap)` | height, latitude, local time, solar flux, geomagnetic activity | realistic drag and lifetime |
| `NRLMSISE00(activity="low"/"moderate"/"high")` | presets: F10.7 = 70/150/230, Ap = 4/12/40 | bracketing the solar cycle |
| `NRLMSISE00(historical=True)` | observed F10.7 and Ap of each date (downloaded) | past orbits |
| a number or `f(r, t)` | whatever you supply | your own data |

## Assumptions to keep in mind

- The spin axis is the frame's z axis (TEME); J2 and zonal terms use it.
- Tesseral harmonics, ocean and solid tides, and relativity are not modeled.
- Sun and Moon positions are analytic (about 0.01° and 0.3°).
- Drag and radiation pressure use a constant area (no attitude model).

Every `Trajectory` lists the assumptions actually used in `tr.info.assumptions`, and `tr.explain()` prints them.

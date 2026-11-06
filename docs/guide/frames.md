# Frames and data exchange

## urania's frame

All computations use one Earth-centered inertial frame: **TEME** (true equator, mean equinox of date), whose z
axis is Earth's spin axis. It is the frame of TLEs and SGP4, so TLE states enter without conversion, and Earth's
rotation is a single rotation by Greenwich mean sidereal time (used for ground tracks, passes and drag).

## Converting to standard frames

| frame | name in urania | typical use |
|---|---|---|
| TEME | `"teme"` | TLEs, SGP4, urania itself |
| GCRS | `"gcrs"` | ≈ J2000 / EME2000 / GCRF; GMAT's EarthMJ2000Eq, most tools |
| ITRS | `"itrs"` | Earth-fixed; GNSS, ground positions |

```python
import astropy.units as u
import urania as ur

orbit = ur.Orbit.from_frame("Earth", [7000, 0, 0] * u.km, [0, 6.5, 3.5] * u.km / u.s, "2026-11-01", "gcrs")
print(orbit.in_frame("gcrs"))
print(orbit.in_frame("itrs"))
```

Conversions use astropy (IAU 2006/2000A precession-nutation, IERS Earth orientation). Vallado's Example 3-15 is
reproduced to 3 mm in GCRF.

## CCSDS OEM

`Trajectory.to_oem(path)` writes a CCSDS Orbit Ephemeris Message (KVN) in GCRF (or TEME/ITRF), UTC, km and km/s,
which STK, GMAT, Orekit and most ground-station software read. `urania.coordinates.read_oem` reads one back.

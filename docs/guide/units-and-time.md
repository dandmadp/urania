# Units, dates and time scales

## Units

Internally everything is SI: meters, seconds, kilograms, radians. Inputs may be

- **strings with a unit**: `"400 km"`, `"51.6 deg"`, `"90 min"`, `"0.03 m2"`, `"437.5 MHz"` (astropy's unit
  names, case-sensitive);
- **astropy Quantities**: `400 * ur.u.km` (`ur.u` is `astropy.units`);
- **plain numbers**, taken as SI.

Outputs are SI numbers. For convenient units, use `orbit.qty` (Quantities in km, deg, min), `orbit.summary()`, or
`summary()` on any transfer or mission.

```python
import urania as ur

o = ur.Orbit.circular("Earth", "400 km", inc="51.6 deg")
print(o.a, o.inc)            # SI: 6778137.0 m, 0.9006 rad
print(o.qty.a, o.qty.inc)    # 6778.137 km, 51.6 deg
```

### Unit-mistake warnings

Plain numbers that are almost surely in the wrong unit raise a `urania.units.UnitWarning` that names the fix:
angles above 2π (degrees given as radians), altitudes between 0 and 1000 m (kilometers given as meters), and
semi-major axes or positions under 1% of the body's radius. Correct SI input never triggers them.

## Dates

`Epoch` is an instant, stored as TDB seconds since J2000. Anything that takes a date accepts:

| input | meaning |
|---|---|
| `"2026-11-01"`, `"2026-11-01 12:30"` | TDB (like `Epoch.from_iso`) |
| `"2026-11-01T12:30Z"`, `"2026-11-01 12:30 UTC"` | UTC |
| `"2026-11-01T21:30+09:00"` | a civil time zone |
| `datetime` with a time zone | converted from UTC |
| naive `datetime` | TDB |
| astropy `Time` | any scale, converted to TDB |

```python
import urania as ur

e = ur.Epoch.parse("2026-11-01T21:30+09:00")
print(e, e.utc, e.jd)
print(e + 3600, (e + "90 min") - e)
```

## Why TDB?

TDB is the uniform time of the equations of motion and of planetary ephemerides. UTC has leap seconds, so a day
in UTC is not always 86400 s. TDB − UTC is about 69 s today; urania converts with astropy wherever UTC appears
(TLE epochs, ground stations, reports), and for dates beyond the published leap seconds it keeps the last offset.

## Durations

Time spans are seconds, a time Quantity, or a string: `orbit.propagate("90 min")`. `propagate` and `after` also
take `days=`, `hours=` or `minutes=`.

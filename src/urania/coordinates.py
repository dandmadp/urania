"""Conversions between urania's inertial frame and standard frames, and CCSDS OEM export.

urania's inertial frame has its z axis along Earth's true spin axis and its x axis at the mean equinox, the
definition of TEME (DECISIONS D31). astropy transforms TEME to GCRS (≈ J2000/EME2000, GMAT's EarthMJ2000Eq,
within 23 mas) and to ITRS (Earth-fixed, with precession, nutation, Earth rotation and polar motion from IERS
data when available).
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from . import units
from .time import Epoch, _future_leap_seconds_ok

if TYPE_CHECKING:
    from .bodies import Body
    from .orbits import Orbit
    from .propagation import Trajectory

FRAMES = ("teme", "gcrs", "itrs", "icrs")


def _frame_class(name: str):
    from astropy.coordinates import GCRS, ICRS, ITRS, TEME

    key = name.lower()
    classes = {"teme": TEME, "gcrs": GCRS, "itrs": ITRS, "icrs": ICRS}
    if key not in classes:
        from ._hints import did_you_mean

        raise ValueError(f"Unknown frame {name!r} (available: {', '.join(FRAMES)})." + did_you_mean(key, FRAMES))
    if key == "icrs":
        raise ValueError("ICRS is barycentric; use 'gcrs' for Earth-centered J2000-like coordinates")
    return classes[key]


def _iers_age():
    """Accept Earth-orientation data up to a year old instead of astropy's 30 days.

    astropy otherwise re-downloads the IERS-A file (several MB) in every new process once its bundled copy is a
    month old. A year-old UT1 prediction errs by tens of milliseconds (meters at the surface), below urania's
    own UT1 ≈ UTC approximation elsewhere. A user's own setting is kept when it is already longer, or None.
    """
    from astropy.utils import iers

    age = iers.conf.auto_max_age
    return iers.conf.set_temp("auto_max_age", None if age is None else max(age, 365.0))


def _require_earth(body: Body) -> None:
    from .propagation import _is_earth

    if not _is_earth(body):
        raise ValueError(f"Frame conversions are defined for Earth orbits, not {body.name}")


def convert(r, v, epochs, from_frame: str, to_frame: str) -> tuple[np.ndarray, np.ndarray]:
    """Transform positions [m] and velocities [m/s] (3 or N×3) at epochs (one Epoch or a list) between frames."""
    import astropy.units as u
    from astropy.coordinates import CartesianDifferential, CartesianRepresentation
    from astropy.time import Time

    src, dst = _frame_class(from_frame), _frame_class(to_frame)
    r, v = np.asarray(r, dtype=float), np.asarray(v, dtype=float)
    single = r.ndim == 1
    r2, v2 = np.atleast_2d(r), np.atleast_2d(v)
    epochs = [epochs] if isinstance(epochs, Epoch) else [Epoch.parse(e) for e in epochs]
    if len(epochs) == 1 and len(r2) > 1:
        epochs = epochs * len(r2)
    if len(epochs) != len(r2):
        raise ValueError(f"{len(epochs)} epochs for {len(r2)} states")
    secs = np.array([e.tdb_seconds for e in epochs])
    t = Time(np.full(len(secs), 2451545.0), secs / 86400.0, format="jd", scale="tdb")
    rep = CartesianRepresentation(r2.T * u.m, differentials=CartesianDifferential(v2.T * u.m / u.s))
    with _future_leap_seconds_ok(), _iers_age():
        out = src(rep, obstime=t).transform_to(dst(obstime=t))
    cart = out.cartesian
    r_out = cart.xyz.to_value(u.m).T
    v_out = cart.differentials["s"].d_xyz.to_value(u.m / u.s).T
    return (r_out[0], v_out[0]) if single else (r_out, v_out)


def orbit_in(orbit: Orbit, frame: str) -> tuple[np.ndarray, np.ndarray]:
    """Position and velocity of an orbit in another frame. Implementation of `Orbit.in_frame`."""
    _require_earth(orbit.body)
    return convert(orbit.r, orbit.v, orbit.epoch, "teme", frame)


def trajectory_in(tr: Trajectory, frame: str) -> tuple[np.ndarray, np.ndarray]:
    """Positions and velocities of every trajectory sample in another frame."""
    _require_earth(tr.body)
    return convert(tr.r, tr.v, tr.epochs, "teme", frame)


def orbit_from(body, r, v, epoch, frame: str) -> Orbit:
    """An Orbit from a state given in another frame. Implementation of `Orbit.from_frame`."""
    from .orbits import Orbit, _body

    body = _body(body)
    _require_earth(body)
    epoch = Epoch.parse(epoch)
    r_si = units.to_si(r, units.LENGTH)
    v_si = units.to_si(v, units.VELOCITY)
    r_t, v_t = convert(r_si, v_si, epoch, frame, "teme")
    return Orbit(body, r_t, v_t, epoch)


# ---------------------------------------------------------------- CCSDS OEM

def _utc_stamps(tr: Trajectory) -> list[str]:
    """UTC ISO times of every sample, rounded to the millisecond."""
    from astropy.time import Time

    secs = tr.epoch.tdb_seconds + np.asarray(tr.t, dtype=float)
    with _future_leap_seconds_ok():
        t = Time(np.full(len(secs), 2451545.0), secs / 86400.0, format="jd", scale="tdb").utc
        t.precision = 3
        return list(t.isot)


def write_oem(tr: Trajectory, path, *, frame: str = "GCRS", object_name: str = "SATELLITE",
              object_id: str = "UNKNOWN", originator: str = "urania") -> Path:
    """Write a trajectory as a CCSDS Orbit Ephemeris Message (CCSDS 502.0-B-3, KVN format), in km and km/s.

    frame: "GCRS" (written as REF_FRAME = GCRF, the usual choice), "TEME" or "ITRS" (ITRF).
    Times are UTC. Returns the path.
    """
    from . import __version__

    _require_earth(tr.body)
    key = frame.lower()
    ref = {"gcrs": "GCRF", "teme": "TEME", "itrs": "ITRF"}.get(key)
    if ref is None:
        raise ValueError(f"OEM frame must be GCRS, TEME or ITRS: {frame!r}")
    r, v = (tr.r, tr.v) if key == "teme" else trajectory_in(tr, key)
    stamps = _utc_stamps(tr)
    created = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    lines = [
        "CCSDS_OEM_VERS = 3.0",
        f"CREATION_DATE = {created}",
        f"ORIGINATOR = {originator}",
        "",
        "META_START",
        f"OBJECT_NAME = {object_name}",
        f"OBJECT_ID = {object_id}",
        "CENTER_NAME = EARTH",
        f"REF_FRAME = {ref}",
        "TIME_SYSTEM = UTC",
        f"START_TIME = {stamps[0]}",
        f"STOP_TIME = {stamps[-1]}",
        "META_STOP",
        "",
        f"COMMENT Generated by urania {__version__}; model {tr.info.model}, {tr.info.integrator}",
    ]
    for stamp, ri, vi in zip(stamps, r / 1e3, v / 1e3, strict=True):
        lines.append(f"{stamp} " + " ".join(f"{x:.9f}" for x in ri) + " " + " ".join(f"{x:.12f}" for x in vi))
    path = Path(path)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def read_oem(path) -> tuple[list[Epoch], np.ndarray, np.ndarray, str]:
    """Read the data lines of a single-segment OEM (KVN): (UTC epochs, r [m], v [m/s], REF_FRAME)."""
    epochs, rs, vs, ref = [], [], [], ""
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("REF_FRAME"):
            ref = line.split("=", 1)[1].strip()
        parts = line.split()
        if len(parts) == 7 and parts[0][:4].isdigit() and "T" in parts[0]:
            epochs.append(Epoch.from_utc(parts[0]))
            rs.append([float(x) * 1e3 for x in parts[1:4]])
            vs.append([float(x) * 1e3 for x in parts[4:7]])
    return epochs, np.array(rs), np.array(vs), ref

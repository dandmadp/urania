"""Comparison against NASA GMAT (R2026a).

Reference data comes from running the GMAT scripts made by validation/gmat/make_scripts.py
(tests/fixtures/gmat/<scenario>.txt). Tests are skipped if the files are missing.

Measured results (urania vs GMAT, maximum position difference over the run):

| scenario         | EarthMJ2000Eq frame | true-pole frame |
|------------------|---------------------|-----------------|
| twobody, 1 day   | 0.009 m             | -               |
| j2, 7 days       | 144 m               | 3.30 m          |
| j2_drag, 3 days  | 73 m                | 3.32 m          |

GMAT evaluates J2 and the atmosphere about Earth's true spin axis, which at the J2000 epoch is
tilted 7.69" from the MJ2000Eq z axis (nutation). urania uses the frame's z axis as the spin axis.
Rotating the problem into a frame whose z axis is the true pole removes that difference, and the
remaining few meters show that the force models and integrators agree.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from urania import Orbit

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures" / "gmat"

_spec = importlib.util.spec_from_file_location(
    "gmat_scenarios", ROOT / "validation" / "gmat" / "make_scripts.py")
gmat = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gmat)

MODELS = {"twobody": "twobody", "j2": "j2", "j2_drag": "j2+drag"}

# Maximum allowed position difference [m] in the EarthMJ2000Eq frame (measured value + margin)
LIMITS = {"twobody": 0.05, "j2": 200.0, "j2_drag": 150.0}

# Maximum allowed position difference [m] once both use the true spin axis
TRUE_POLE_LIMITS = {"j2": 5.0, "j2_drag": 5.0}


def load_gmat(name: str) -> np.ndarray:
    path = FIXTURES / f"{name}.txt"
    if not path.exists():
        pytest.skip(f"No GMAT reference data: {path.relative_to(ROOT)} "
                    f"(run validation/gmat/{name}.script in GMAT)")
    # GMAT repeats the header line on every Report command, so keep numeric rows only
    rows = [line.split() for line in path.read_text().splitlines()]
    data = np.array([[float(x) for x in row] for row in rows
                     if row and row[0][0] in "0123456789-."])
    return data  # columns: elapsed s, X, Y, Z [km], VX, VY, VZ [km/s]


def true_pole_rotation() -> np.ndarray:
    """Rotation from MJ2000Eq to a frame whose z axis is Earth's true spin axis at the epoch."""
    import astropy.units as u
    from astropy.coordinates import GCRS, ITRS, CartesianRepresentation
    from astropy.time import Time

    t = Time("2000-01-01T12:00:00", scale="tt")
    pole = ITRS(CartesianRepresentation(0, 0, 1, unit=u.one), obstime=t).transform_to(GCRS(obstime=t))
    p = pole.cartesian.xyz.value
    p /= np.linalg.norm(p)
    e1 = np.cross([0.0, 1.0, 0.0], p)
    e1 /= np.linalg.norm(e1)
    return np.array([e1, np.cross(p, e1), p])


def position_errors(name: str, rotation: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """(elapsed seconds, position difference between urania and GMAT [m]).

    With a rotation, both the initial state and the GMAT positions are rotated first.
    """
    data = load_gmat(name)
    sc = gmat.SCENARIOS[name]
    orbit = sc["orbit"]()
    M = np.eye(3) if rotation is None else rotation
    orbit = Orbit(orbit.body, M @ orbit.r, M @ orbit.v, orbit.epoch)
    t = data[:, 0]
    kwargs = gmat.DRAG_SAT if sc["drag"] else {}
    # Sample urania at every GMAT output time
    tr = orbit.propagate(float(t[-1]), model=MODELS[name], n_points=len(t), **kwargs)
    np.testing.assert_allclose(tr.t, t, atol=1e-6)
    gmat_r = (M @ (data[:, 1:4].T * 1e3)).T
    return t, np.linalg.norm(tr.r - gmat_r, axis=1)


def _report(label: str, t: np.ndarray, errs: np.ndarray) -> None:
    print(f"{label}: max {errs.max():.3f} m (t = {t[errs.argmax()] / 3600:.1f} h), "
          f"end {errs[-1]:.3f} m")


@pytest.mark.parametrize("name", list(LIMITS))
def test_against_gmat(name):
    t, errs = position_errors(name)
    _report(name, t, errs)
    assert errs.max() < LIMITS[name]


@pytest.mark.parametrize("name", list(TRUE_POLE_LIMITS))
def test_against_gmat_true_pole(name):
    """With the same spin axis as GMAT, the J2 and drag models agree to a few meters."""
    t, errs = position_errors(name, true_pole_rotation())
    _report(f"{name} (true pole)", t, errs)
    assert errs.max() < TRUE_POLE_LIMITS[name]


def test_initial_states_match_scripts():
    """Initial states in the generated scripts must match the scenario definitions (catches stale scripts)."""
    for name, sc in gmat.SCENARIOS.items():
        script = ROOT / "validation" / "gmat" / f"{name}.script"
        if not script.exists():
            pytest.skip("No scripts: run python validation/gmat/make_scripts.py")
        values = {}
        for line in script.read_text(encoding="utf-8").splitlines():
            for key in ("X", "Y", "Z", "VX", "VY", "VZ"):
                if line.startswith(f"GMAT Sat.{key} = "):
                    values[key] = float(line.split("=")[1].strip(" ;"))
        o = sc["orbit"]()
        np.testing.assert_allclose([values[k] for k in ("X", "Y", "Z")], o.r / 1e3, atol=1e-9)
        np.testing.assert_allclose([values[k] for k in ("VX", "VY", "VZ")], o.v / 1e3, atol=1e-12)

"""Comparison against NASA GMAT.

Reference data comes from running the GMAT scripts made by validation/gmat/make_scripts.py
(tests/fixtures/gmat/<scenario>.txt). Tests are skipped if the files are missing.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures" / "gmat"

_spec = importlib.util.spec_from_file_location(
    "gmat_scenarios", ROOT / "validation" / "gmat" / "make_scripts.py")
gmat = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gmat)

# Per scenario: (urania model, maximum allowed position error over the whole run [m])
# Provisional: to be finalized after seeing the actual errors against GMAT data.
CASES = {
    "twobody": ("twobody", 1.0),
    "j2": ("j2", 100.0),
    "j2_drag": ("j2+drag", 1000.0),
}


def load_gmat(name: str) -> np.ndarray:
    path = FIXTURES / f"{name}.txt"
    if not path.exists():
        pytest.skip(f"No GMAT reference data: {path.relative_to(ROOT)} "
                    f"(run validation/gmat/{name}.script in GMAT)")
    data = np.loadtxt(path, skiprows=1)
    return data  # columns: elapsed s, X, Y, Z [km], VX, VY, VZ [km/s]


def position_errors(name: str) -> tuple[np.ndarray, np.ndarray]:
    """(elapsed seconds, position difference between urania and GMAT [m])."""
    data = load_gmat(name)
    sc = gmat.SCENARIOS[name]
    model, _ = CASES[name]
    orbit = sc["orbit"]()
    t = data[:, 0]
    kwargs = gmat.DRAG_SAT if sc["drag"] else {}
    # Sample urania at every GMAT output time
    tr = orbit.propagate(float(t[-1]), model=model, method="auto", n_points=len(t), **kwargs)
    np.testing.assert_allclose(tr.t, t, atol=1e-6)
    errs = np.linalg.norm(tr.r - data[:, 1:4] * 1e3, axis=1)
    return t, errs


@pytest.mark.parametrize("name", list(CASES))
def test_against_gmat(name):
    t, errs = position_errors(name)
    _, limit = CASES[name]
    worst = errs.max()
    print(f"{name}: max {worst:.3f} m (t = {t[errs.argmax()] / 3600:.1f} h), end {errs[-1]:.3f} m")
    assert worst < limit


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

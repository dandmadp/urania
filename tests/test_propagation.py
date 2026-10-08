import dataclasses
import math

import astropy.units as u
import numpy as np
import pytest

from urania import ISS, Earth, Environment, Mars, Orbit
from urania.propagation import DEFAULT_RTOL, parse_model

DAY = 86400.0
ISS_DRAG = {"area": 1500, "mass": 420000}


def test_parse_model():
    assert parse_model("twobody") == {"twobody"}
    assert parse_model("j2") == {"twobody", "j2"}
    assert parse_model("J2 + drag") == {"twobody", "j2", "drag"}
    assert parse_model(2) == {"twobody", "j2", "drag"}
    with pytest.raises(NotImplementedError):
        parse_model("j2+moon")
    with pytest.raises(NotImplementedError):
        parse_model(3)
    with pytest.raises(ValueError):
        parse_model("j3")


def test_original_unchanged_and_final_epoch():
    r0 = ISS.r.copy()
    tr = ISS.propagate(days=1)
    np.testing.assert_array_equal(ISS.r, r0)
    assert tr.final.epoch - ISS.epoch == pytest.approx(DAY)
    assert tr.info.integrator.startswith("kepler")


def test_duration_forms():
    a = ISS.propagate(3600)
    b = ISS.propagate(60 * u.min)
    np.testing.assert_allclose(a.final.r, b.final.r)
    with pytest.raises(ValueError):
        ISS.propagate()
    with pytest.raises(ValueError):
        ISS.propagate(3600, days=1)


def test_numeric_twobody_matches_analytic():
    a = ISS.propagate(days=3)
    b = ISS.propagate(days=3, method="DOP853")
    assert np.linalg.norm(a.final.r - b.final.r) < 1.0
    assert b.info.rtol == DEFAULT_RTOL


def test_j2_raan_drift_matches_secular_rate():
    """The node shift from J2 numerical propagation matches the secular rate formula within 1%."""
    days = 10
    tr = ISS.propagate(days=days, model="j2")
    d_raan = (tr.final.raan - ISS.raan + math.pi) % (2 * math.pi) - math.pi
    expected = ISS.raan_rate * days * DAY
    assert d_raan == pytest.approx(expected, rel=0.01)
    assert "J2" in " ".join(tr.info.assumptions)


def test_drag_decay_matches_analytic():
    """Circular orbit decay da/dt = -ρ B √(μa) for a non-rotating body with constant density.

    From Gauss's planetary equation da/dt = 2a²v·a_T/μ with circular-orbit drag a_T = -½ρBv².
    """
    body = dataclasses.replace(Earth, rotation_rate=0.0)
    o = Orbit.circular(body, 400e3)
    rho, cd, area, mass = 3.725e-12, 2.2, 10.0, 100.0
    B = cd * area / mass
    tr = o.propagate(days=1, model="drag", cd=cd, area=area, mass=mass, density=rho)
    expected = -rho * B * math.sqrt(Earth.mu * o.a) * DAY
    assert tr.final.a - o.a == pytest.approx(expected, rel=0.01)


def test_density_injection_forms_agree():
    """Injecting a constant, Quantity, function or Environment object must give the same result."""
    rho = 3e-12

    class ConstRho(Environment):
        description = "test"

        def __call__(self, r, t):
            return rho

    kw = dict(days=0.5, model="j2+drag", **ISS_DRAG)
    finals = [
        ISS.propagate(density=rho, **kw).final.r,
        ISS.propagate(density=rho * u.kg / u.m**3, **kw).final.r,
        ISS.propagate(density=lambda r, t: rho, **kw).final.r,
        ISS.propagate(density=ConstRho(), **kw).final.r,
    ]
    for f in finals[1:]:
        np.testing.assert_allclose(f, finals[0], atol=1e-6)


def test_default_atmosphere_decays_iss():
    tr = ISS.propagate(days=2, model="j2+drag", **ISS_DRAG)
    assert tr.final.a < ISS.a
    assert "exponential atmosphere" in " ".join(tr.info.assumptions)
    assert tr.info.terminated is None


def test_reentry_terminates():
    o = Orbit.circular(Earth, 130e3)
    tr = o.propagate(days=5, model="drag", area=1.0, mass=1.0)
    assert tr.info.terminated
    assert tr.altitude[-1] == pytest.approx(0.0, abs=1e-3)
    assert tr.t[-1] < 5 * DAY


def test_drag_requires_parameters():
    with pytest.raises(ValueError):
        ISS.propagate(days=1, model="j2+drag")
    mars_orbit = Orbit.circular(Mars, 300e3)
    with pytest.raises(ValueError):
        mars_orbit.propagate(days=1, model="drag", **ISS_DRAG)


def test_backward_then_forward():
    back = ISS.propagate(days=-1, model="j2").final
    fwd = back.propagate(days=1, model="j2").final
    assert np.linalg.norm(fwd.r - ISS.r) < 1e-2
    assert fwd.epoch == ISS.epoch


def test_trajectory_immutable():
    tr = ISS.propagate(3600)
    with pytest.raises(ValueError):
        tr.r[0, 0] = 0.0
    assert len(tr) == tr.r.shape[0]


def test_days_accepts_quantity():
    """days=1*u.day must not be multiplied by 86400 again (regression test)."""
    assert ISS.propagate(days=1 * u.day).final.epoch - ISS.epoch == pytest.approx(DAY)
    assert ISS.propagate(days=12 * u.h).final.epoch - ISS.epoch == pytest.approx(DAY / 2)


@pytest.mark.parametrize("model", ["twobody", "j2"])
def test_zero_duration(model):
    tr = ISS.propagate(0, model=model)
    np.testing.assert_allclose(tr.final.r, ISS.r)


@pytest.mark.parametrize("days", [0.1, -0.1])
def test_analytic_and_numeric_both_stop_at_surface(days):
    """Orbit with perigee inside the Earth: the analytic solution stops at the surface like integration does."""
    sub = Orbit.from_elements(Earth, a=6000e3, ecc=0.2, nu=math.radians(180))
    a = sub.propagate(days=days)
    b = sub.propagate(days=days, method="DOP853")
    assert a.info.terminated and b.info.terminated
    assert a.altitude[-1] == pytest.approx(0.0, abs=1e-3)
    assert a.t[-1] == pytest.approx(b.t[-1], abs=1e-3)
    np.testing.assert_allclose(a.r[-1], b.r[-1], atol=1e-2)


def test_rejects_too_few_points():
    """n_points=1 used to return only the start state (regression test)."""
    with pytest.raises(ValueError, match="at least 2"):
        ISS.propagate(3600, n_points=1)


def test_rejects_start_below_surface():
    with pytest.raises(ValueError, match="below the surface"):
        Orbit.circular(Earth, -10e3).propagate(600, model="j2")


@pytest.mark.parametrize("model", [True, 1.0, None])
def test_rejects_non_model_types(model):
    with pytest.raises(TypeError):
        parse_model(model)


@pytest.mark.parametrize("kwargs, names", [
    ({"area": 10, "mass": 100}, "area, mass"),
    ({"model": "j2", "density": 1e-12}, "density"),
])
def test_drag_inputs_without_drag_model_warn(kwargs, names):
    """Drag inputs used to be ignored silently when the model had no drag term."""
    with pytest.warns(UserWarning, match=names):
        ISS.propagate(days=0.1, **kwargs)


@pytest.mark.parametrize("bad", [{"cd": -2.2}, {"area": -1}, {"mass": -1}, {"mass": 0}])
def test_drag_parameters_must_be_positive(bad):
    kw = {"area": 1, "mass": 1, **bad}
    with pytest.raises(ValueError, match="must be positive"):
        ISS.propagate(days=0.1, model="drag", **kw)


@pytest.mark.parametrize("duration", [float("inf"), float("nan")])
def test_rejects_non_finite_duration(duration):
    with pytest.raises(ValueError, match="finite"):
        ISS.propagate(duration)


def test_rejects_array_duration():
    with pytest.raises(TypeError, match="single value"):
        ISS.propagate([60.0, 120.0])


def test_trajectory_elements_and_epochs():
    tr = ISS.propagate(days=1, model="j2", n_points=25)
    el = tr.elements
    assert set(el) == {"p", "a", "ecc", "inc", "raan", "argp", "nu"}
    assert el["a"].shape == (25,)
    assert el["a"][0] == pytest.approx(ISS.a)
    # J2 makes the node regress over the day
    d_raan = (el["raan"][-1] - el["raan"][0] + math.pi) % (2 * math.pi) - math.pi
    assert d_raan < 0
    with pytest.raises(ValueError):
        el["a"][0] = 0.0
    assert tr.epochs[-1] == tr.final.epoch
    assert len(tr.epochs) == len(tr)


def test_hyperbola_default_sampling_and_energy():
    """A hyperbolic escape gets 1001 points by default; two-body integration conserves energy."""
    hyp = Orbit.from_elements(Earth, a=-20000e3, ecc=1.5)
    tr = hyp.propagate(days=1, method="DOP853")
    assert len(tr) == 1001
    energy = [np.linalg.norm(v) ** 2 / 2 - Earth.mu / np.linalg.norm(r) for r, v in zip(tr.r, tr.v)]
    assert max(energy) - min(energy) < 1e-7 * abs(energy[0])


def test_kepler_method_only_for_twobody():
    with pytest.raises(ValueError, match="only available"):
        ISS.propagate(3600, model="j2", method="kepler")


def test_j2_needs_nonzero_j2():
    from urania import Sun

    with pytest.raises(ValueError, match="J2 of Sun is zero"):
        Orbit.circular(Sun, 1e9).propagate(days=1, model="j2")

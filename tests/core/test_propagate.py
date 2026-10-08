import math

import numpy as np
import pytest

from urania.core import forces
from urania.core import elements as el
from urania.core.propagate import cowell, kepler_propagate, kepler_states, stumpff

KM = 1e3
MU = 398600.4418 * KM**3


def test_vallado_example_2_4():
    """Vallado example 2-4: 40-minute Kepler propagation."""
    r0 = np.array([1131.340, -2282.343, 6672.423]) * KM
    v0 = np.array([-5.64305, 4.30333, 2.42879]) * KM
    r, v = kepler_propagate(r0, v0, 40 * 60, MU)
    np.testing.assert_allclose(r / KM, [-4219.7527, 4363.0292, -3958.7666], atol=1e-3)
    np.testing.assert_allclose(v / KM, [3.689866, -1.916735, -6.112511], atol=1e-6)


def test_kepler_full_period_returns():
    r0, v0 = np.array([7000e3, 0, 0]), np.array([0, 8000.0, 1000.0])
    a = 1 / (2 / 7000e3 - (8000**2 + 1000**2) / MU)
    T = 2 * math.pi * math.sqrt(a**3 / MU)
    r, v = kepler_propagate(r0, v0, T, MU)
    np.testing.assert_allclose(r, r0, atol=1e-4)
    np.testing.assert_allclose(v, v0, atol=1e-7)


def _twobody(t, r, v):
    return forces.accel_twobody(r, MU)


@pytest.mark.parametrize("v0", [[0, 7546.0, 0], [0, 9000.0, 1500.0], [0, 11500.0, 0]],
                         ids=["circular", "elliptic", "hyperbolic"])
def test_cowell_matches_kepler(v0):
    """Numerical integration (DOP853) must match the analytic solution."""
    r0 = np.array([7000e3, 0.0, 0.0])
    v0 = np.array(v0)
    duration = 10 * 86400 if np.linalg.norm(v0) < 11000 else 86400
    res = cowell(r0, v0, duration, _twobody)
    r_ref, v_ref = kepler_propagate(r0, v0, duration, MU)
    assert np.linalg.norm(res.r[-1] - r_ref) < 1.0       # 1 m
    assert np.linalg.norm(res.v[-1] - v_ref) < 1e-3      # 1 mm/s
    assert not res.terminated


def test_cowell_backward():
    r0, v0 = np.array([7000e3, 0, 0]), np.array([0, 7546.0, 0])
    res = cowell(r0, v0, -3000.0, _twobody)
    r_ref, _ = kepler_propagate(r0, v0, -3000.0, MU)
    assert np.linalg.norm(res.r[-1] - r_ref) < 1e-3


def test_cowell_terminal_event_appends_state():
    r0, v0 = np.array([7000e3, 0, 0]), np.array([0, 2000.0, 0])  # nearly falling straight down

    def below_6500km(t, y):
        return np.linalg.norm(y[:3]) - 6500e3

    below_6500km.terminal = True
    res = cowell(r0, v0, 10000.0, _twobody, t_eval=np.linspace(0, 10000, 101),
                 events=[below_6500km])
    assert res.terminated
    assert np.linalg.norm(res.r[-1]) == pytest.approx(6500e3, abs=1e-3)
    assert res.t[-1] < 10000.0


def test_kepler_states_matches_single():
    r0, v0 = np.array([7000e3, 0, 0]), np.array([0, 8000.0, 1000.0])
    dts = [0.0, 1000.0, -500.0, 1e5]
    r, v = kepler_states(r0, v0, dts, MU)
    for i, dt in enumerate(dts):
        ri, vi = kepler_propagate(r0, v0, dt, MU)
        np.testing.assert_array_equal(r[i], ri)
        np.testing.assert_array_equal(v[i], vi)


def test_kepler_parabolic_matches_cowell():
    """Parabolic analytic solution (Barker) versus numerical integration."""
    r0 = np.array([7000e3, 0.0, 0.0])
    v0 = np.array([0.0, math.sqrt(2 * MU / 7000e3), 0.0])  # escape speed = parabola
    r_ref, _ = kepler_propagate(r0, v0, 20000.0, MU)
    res = cowell(r0, v0, 20000.0, _twobody)
    assert np.linalg.norm(res.r[-1] - r_ref) < 1e-2


def test_cowell_zero_duration():
    r0, v0 = np.array([7000e3, 0, 0]), np.array([0, 7546.0, 0])
    res = cowell(r0, v0, 0.0, _twobody, t_eval=np.zeros(5))
    assert res.r.shape == (5, 3)
    np.testing.assert_array_equal(res.r[-1], r0)


def test_stumpff_values():
    """c2(0) = 1/2, c3(0) = 1/6; at ψ = π²: c2 = 2/π², c3 = 1/π²; smooth across ψ = 0."""
    assert stumpff(0.0) == (0.5, 1.0 / 6.0)
    c2, c3 = stumpff(math.pi**2)
    assert c2 == pytest.approx(2 / math.pi**2) and c3 == pytest.approx(1 / math.pi**2)
    for psi in (1e-12, -1e-12, 1e-6, -1e-6):
        c2, c3 = stumpff(psi)
        assert c2 == pytest.approx(0.5 - psi / 24, rel=1e-12)
        assert c3 == pytest.approx(1 / 6 - psi / 120, rel=1e-12)
    c2, c3 = stumpff(-4.0)   # y = 2: (cosh 2 - 1)/4, (sinh 2 - 2)/8
    assert c2 == pytest.approx((math.cosh(2) - 1) / 4) and c3 == pytest.approx((math.sinh(2) - 2) / 8)


@pytest.mark.parametrize("e", [1 - 1e-6, 1 - 1e-9, 1 - 1e-11, 1 + 1e-11, 1 + 1e-9, 1 + 1e-6])
def test_nearly_parabolic_propagation(e):
    """Nearly parabolic orbits used to be off by up to hundreds of km (element-based Kepler).

    Universal variables must agree with numerical integration and round-trip forward/backward.
    """
    rp = 7000e3
    r0, v0 = el.coe_to_rv(rp * (1 + e), e, 0.7, 1.1, 0.4, -1.0, MU)
    r1, v1 = kepler_propagate(r0, v0, 3000.0, MU)
    res = cowell(r0, v0, 3000.0, _twobody)
    assert np.linalg.norm(res.r[-1] - r1) < 0.05
    r2, _ = kepler_propagate(r1, v1, -3000.0, MU)
    assert np.linalg.norm(r2 - r0) < 1e-3


def test_many_revolutions_match_integration():
    """Elliptic propagation over 200 revolutions: the period reduction keeps the result exact."""
    r0, v0 = np.array([7000e3, 0.0, 0.0]), np.array([0.0, 8000.0, 1000.0])
    a = 1 / (2 / 7000e3 - (8000**2 + 1000**2) / MU)
    T = 2 * math.pi * math.sqrt(a**3 / MU)
    r, v = kepler_propagate(r0, v0, 200 * T + 1234.5, MU)
    r_ref, v_ref = kepler_propagate(r0, v0, 1234.5, MU)
    assert np.linalg.norm(r - r_ref) < 1e-3

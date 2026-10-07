import math

import numpy as np
import pytest

from urania.core import forces
from urania.core.propagate import cowell, kepler_propagate

KM = 1e3
MU = 398600.4418 * KM**3


def test_vallado_example_2_4():
    """Vallado 예제 2-4: 40분 케플러 전파."""
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
    """수치 적분(DOP853)과 해석해가 일치해야 한다."""
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
    r0, v0 = np.array([7000e3, 0, 0]), np.array([0, 2000.0, 0])  # 거의 낙하

    def below_6500km(t, y):
        return np.linalg.norm(y[:3]) - 6500e3

    below_6500km.terminal = True
    res = cowell(r0, v0, 10000.0, _twobody, t_eval=np.linspace(0, 10000, 101),
                 events=[below_6500km])
    assert res.terminated
    assert np.linalg.norm(res.r[-1]) == pytest.approx(6500e3, abs=1e-3)
    assert res.t[-1] < 10000.0

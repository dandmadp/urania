import math

import numpy as np
import pytest

from urania.core import kepler as k

deg = math.radians


def test_vallado_example_2_1_elliptic():
    """Vallado example 2-1: M = 235.4°, e = 0.4 → E = 220.512074767522°."""
    E = k.mean_to_eccentric(deg(235.4), 0.4)
    assert math.degrees(E) == pytest.approx(220.512074767522, abs=1e-9)


def test_vallado_example_2_3_hyperbolic():
    """Vallado example 2-3: M = 235.4°, e = 2.4 → H = 1.601376144 rad."""
    H = k.mean_to_hyperbolic(deg(235.4), 2.4)
    assert H == pytest.approx(1.601376144, abs=1e-9)


@pytest.mark.parametrize("e", [0.0, 0.01, 0.5, 0.9, 0.99, 0.999999])
def test_kepler_elliptic_residual(e):
    """E - e sin E = M must hold for every M (including high eccentricity)."""
    for M in np.linspace(0, 2 * math.pi, 361, endpoint=False):
        E = k.mean_to_eccentric(M, e)
        assert 0.0 <= E < 2 * math.pi
        assert (k.eccentric_to_mean(E, e) - M + math.pi) % (2 * math.pi) - math.pi \
            == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("e", [1.0001, 1.5, 3.0, 50.0])
def test_kepler_hyperbolic_residual(e):
    for M in [-1e4, -100, -3, -0.1, 0.0, 1e-6, 0.1, 3, 100, 1e4]:
        H = k.mean_to_hyperbolic(M, e)
        assert k.hyperbolic_to_mean(H, e) == pytest.approx(M, rel=1e-12, abs=1e-12)


def test_barker_residual():
    for M in [-1e3, -2.0, -1e-3, 0.0, 1e-3, 2.0, 1e3]:
        D = k.mean_to_parabolic(M)
        assert k.parabolic_to_mean(D) == pytest.approx(M, rel=1e-12, abs=1e-14)


@pytest.mark.parametrize("e", [0.0, 0.3, 0.95])
def test_true_eccentric_roundtrip(e):
    for nu in np.linspace(0, 2 * math.pi, 73, endpoint=False):
        E = k.true_to_eccentric(nu, e)
        assert k.eccentric_to_true(E, e) == pytest.approx(nu, abs=1e-12)


@pytest.mark.parametrize("e", [0.0, 0.5, 0.9, 1.0, 1.2, 5.0])
def test_mean_true_roundtrip(e):
    nu_max = math.pi * 0.99 if e <= 1 else 0.99 * math.acos(-1 / e)
    for nu in np.linspace(-nu_max, nu_max, 51):
        M = k.true_to_mean(nu, e)
        nu_back = k.mean_to_true(M, e)
        diff = (nu_back - nu + math.pi) % (2 * math.pi) - math.pi
        assert diff == pytest.approx(0.0, abs=1e-10)


def test_circular_orbit_anomalies_equal():
    """For e = 0, M = E = ν."""
    for M in [0.0, 1.0, 3.0, 5.0]:
        assert k.mean_to_true(M, 0.0) == pytest.approx(M, abs=1e-14)


def test_invalid_inputs():
    with pytest.raises(ValueError):
        k.mean_to_eccentric(1.0, 1.5)
    with pytest.raises(ValueError):
        k.mean_to_hyperbolic(1.0, 0.5)
    with pytest.raises(ValueError):
        k.mean_to_true(1.0, -0.1)
    with pytest.raises(ValueError):
        k.true_to_hyperbolic(deg(170), 2.0)  # outside the 120° asymptote


@pytest.mark.parametrize("M", [1e6, 1e9, 1e12, -1e9])
def test_barker_large_M_precision(M):
    """No cancellation error in Cardano's formula for large M (regression test)."""
    D = k.mean_to_parabolic(M)
    assert k.parabolic_to_mean(D) == pytest.approx(M, rel=1e-13)

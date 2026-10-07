import math

import numpy as np
import pytest

from urania.core import maneuvers as m

KM = 1e3
MU = 398600.4418 * KM**3
R = 6378.137 * KM


def test_vallado_example_6_1_hohmann():
    """Vallado example 6-1: Hohmann transfer from 191.34411 km to 35781.34857 km altitude."""
    dv1, dv2, tof = m.hohmann(R + 191.34411 * KM, R + 35781.34857 * KM, MU)
    assert dv1 / KM == pytest.approx(2.457038, abs=1e-6)
    assert dv2 / KM == pytest.approx(1.478187, abs=1e-6)
    assert tof / 3600 == pytest.approx(5.256713, abs=1e-6)


def test_vallado_example_6_2_bielliptic():
    """Vallado example 6-2: 191.34411 km → 376310 km altitude, intermediate apoapsis at 503873 km."""
    dv1, dv2, dv3, tof = m.bielliptic(R + 191.34411 * KM, R + 503873 * KM, R + 376310 * KM, MU)
    assert dv1 / KM == pytest.approx(3.156233, abs=1e-6)
    assert dv3 / KM == pytest.approx(-0.070466, abs=1e-6)  # the last burn is retrograde
    assert (abs(dv1) + abs(dv2) + abs(dv3)) / KM == pytest.approx(3.904057, abs=1e-6)
    assert tof / 3600 == pytest.approx(593.92, abs=0.01)


def test_hohmann_descending_is_symmetric():
    up = m.hohmann(7000e3, 42164e3, MU)
    down = m.hohmann(42164e3, 7000e3, MU)
    assert down[0] == pytest.approx(-up[1])
    assert down[1] == pytest.approx(-up[0])
    assert down[2] == pytest.approx(up[2])


def test_bielliptic_degenerates_to_hohmann():
    """With rb = r2 a bi-elliptic transfer equals a Hohmann transfer."""
    r1, r2 = 7000e3, 100000e3
    h = m.hohmann(r1, r2, MU)
    b = m.bielliptic(r1, r2, r2, MU)
    assert abs(b[0]) + abs(b[1]) + abs(b[2]) == pytest.approx(abs(h[0]) + abs(h[1]))


@pytest.mark.parametrize("ratio, bielliptic_better", [(10.0, False), (20.0, True)])
def test_bielliptic_vs_hohmann_threshold(ratio, bielliptic_better):
    """For r2/r1 < 11.94 bi-elliptic is always worse, for > 15.58 always better (Vallado sec. 6.3)."""
    r1 = 7000e3
    r2 = ratio * r1
    h = sum(abs(x) for x in m.hohmann(r1, r2, MU)[:2])
    for rb_factor in (1.2, 2.0, 10.0):
        b = sum(abs(x) for x in m.bielliptic(r1, rb_factor * r2, r2, MU)[:3])
        assert (b < h) == bielliptic_better


def test_bielliptic_rejects_small_rb():
    with pytest.raises(ValueError):
        m.bielliptic(7000e3, 20000e3, 42164e3, MU)


def test_plane_change_dv():
    assert m.plane_change_dv(7500.0, math.radians(60)) == pytest.approx(7500.0)  # 2v sin30°
    assert m.plane_change_dv(7500.0, 0.0) == 0.0


def test_combined_dv():
    assert m.combined_dv(7000.0, 7500.0, 0.0) == pytest.approx(500.0)
    # Equal speeds reduce to a pure plane change
    assert m.combined_dv(3000.0, 3000.0, 0.5) == pytest.approx(m.plane_change_dv(3000.0, 0.5))


def test_hohmann_plane_change_limits():
    r1, r2, di = R + 300e3, 42164e3, math.radians(28)
    dv1, dv2, _ = m.hohmann_plane_change(r1, r2, 0.0, MU, 0.5)
    h = m.hohmann(r1, r2, MU)
    assert dv1 == pytest.approx(h[0]) and dv2 == pytest.approx(h[1])
    # Hand calculation with the whole plane change at apoapsis:
    # a_t = 24421.07 km, v_ta = √(μ(2/r2 - 1/a_t)) = 1.60785 km/s, v_geo = 3.07466 km/s
    # Δv = √(2.58518 + 9.45354 - 2·1.60785·3.07466·cos28°) = √3.30886 = 1.81903 km/s
    _, dv2_all, _ = m.hohmann_plane_change(r1, r2, di, MU, 0.0)
    assert dv2_all / KM == pytest.approx(1.81903, abs=1e-4)


def test_optimal_split_beats_extremes():
    r1, r2, di = R + 300e3, 42164e3, math.radians(28)
    s = m.optimal_plane_split(r1, r2, di, MU)
    total = lambda x: sum(m.hohmann_plane_change(r1, r2, di, MU, x)[:2])
    assert 0.0 < s < 0.2  # most of it at the slow apoapsis
    assert total(s) < total(0.0)
    assert total(s) < total(1.0)
    assert total(s) <= min(total(s - 0.01), total(s + 0.01))


def test_rotate():
    v = m.rotate([1.0, 0.0, 0.0], [0.0, 0.0, 1.0], math.pi / 2)
    np.testing.assert_allclose(v, [0.0, 1.0, 0.0], atol=1e-15)

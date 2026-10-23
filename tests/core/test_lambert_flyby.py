"""Lambert's problem, flybys, relative motion, coverage, beta angle and the rocket equation (core functions)."""

import math

import numpy as np
import pytest

from urania.core import coverage as cov
from urania.core import flyby as fb
from urania.core import propulsion as prop
from urania.core import relative as rel
from urania.core import sun
from urania.core.lambert import lambert, min_tof, transfer_angle
from urania.core.propagate import kepler_propagate, kepler_states

MU_CURTIS = 398600e9
MU_VALLADO = 398600.4418e9


def test_curtis_example_5_2():
    """Curtis Example 5.2: r1 = (5000, 10000, 2100) km, r2 = (-14600, 2500, 7000) km, 1 hour."""
    v1, v2 = lambert(np.array([5000, 10000, 2100]) * 1e3, np.array([-14600, 2500, 7000]) * 1e3, 3600,
                     MU_CURTIS)
    assert v1 / 1e3 == pytest.approx([-5.9925, 1.9254, 3.2456], abs=5e-5)
    assert v2 / 1e3 == pytest.approx([-3.3125, -4.1966, -0.38529], abs=5e-5)


def test_vallado_example_7_5():
    """Vallado Example 7-5: r0 = (15945.34, 0, 0) km, r = (12214.83899, 10249.46731, 0) km, 76 min."""
    v1, v2 = lambert(np.array([15945.34, 0, 0]) * 1e3, np.array([12214.83899, 10249.46731, 0]) * 1e3,
                     76 * 60, MU_VALLADO)
    assert v1 / 1e3 == pytest.approx([2.058913, 2.915965, 0], abs=1e-6)
    assert v2 / 1e3 == pytest.approx([-3.451565, 0.910315, 0], abs=1e-6)


def test_lambert_random_round_trip():
    """Propagating v1 for tof reaches r2 with v2 (direct and multi-revolution, both directions)."""
    rng = np.random.default_rng(7)
    solved = 0
    for k in range(400):
        r1, r2 = rng.normal(size=3) * 7e6, rng.normal(size=3) * 2e7
        tof = rng.uniform(600, 2e5)
        try:
            v1, v2 = lambert(r1, r2, tof, MU_CURTIS, revolutions=int(rng.integers(0, 3)),
                             branch=("left", "right")[k % 2], prograde=bool(k % 3))
        except ValueError:
            continue   # no multi-revolution solution for that time
        solved += 1
        r, v = kepler_propagate(r1, v1, tof, MU_CURTIS)
        assert np.linalg.norm(r - r2) < 1e-6 * np.linalg.norm(r2)
        assert np.linalg.norm(v - v2) < 1e-6 * np.linalg.norm(v2)
    assert solved > 250


def test_multi_revolution_branches():
    r1, r2 = np.array([7000e3, 0, 0]), np.array([0, 8000e3, 0])
    t_min = min_tof(r1, r2, MU_CURTIS, 1)
    tof = 1.3 * t_min
    sols = [lambert(r1, r2, tof, MU_CURTIS, revolutions=1, branch=b) for b in ("left", "right")]
    assert not np.allclose(sols[0][0], sols[1][0])
    for v1, v2 in sols:
        r, _ = kepler_propagate(r1, v1, tof, MU_CURTIS)
        assert r == pytest.approx(r2, abs=1e-2)
    with pytest.raises(ValueError, match="minimum time of flight"):
        lambert(r1, r2, 0.9 * t_min, MU_CURTIS, revolutions=1)


def test_lambert_errors():
    r1 = np.array([7000e3, 0, 0])
    with pytest.raises(ValueError, match="180°"):
        lambert(r1, -r1 * 1.1, 3000, MU_CURTIS)
    with pytest.raises(ValueError, match="positive"):
        lambert(r1, [0, 7e6, 0], -1, MU_CURTIS)
    with pytest.raises(ValueError, match="branch"):
        lambert(r1, [0, 7e6, 0], 3000, MU_CURTIS, revolutions=1, branch="up")


def test_retrograde_is_the_long_way():
    r1, r2 = np.array([7000e3, 0, 0]), np.array([0, 7000e3, 0])
    assert transfer_angle(r1, r2, True) == pytest.approx(math.pi / 2)
    assert transfer_angle(r1, r2, False) == pytest.approx(1.5 * math.pi)
    v1, _ = lambert(r1, r2, 3000, MU_CURTIS, prograde=False)
    assert np.cross(r1, v1)[2] < 0


# ---------------------------------------------------------------- flyby

def test_turn_angle_formula():
    mu, rp, v = 3.2486e14, 6551.8e3, 5000.0
    e = 1 + rp * v * v / mu
    assert fb.turn_angle(v, rp, mu) == pytest.approx(2 * math.asin(1 / e))
    # Unpowered periapsis for that turn
    assert fb.periapsis_for_turn(v, v, fb.turn_angle(v, rp, mu), mu) == pytest.approx(rp, rel=1e-9)
    assert fb.periapsis_for_turn(v, v, 0.0, mu) == math.inf
    assert fb.powered_dv(v, v, rp, mu) == 0.0


def test_outgoing_v_inf_matches_hyperbola_propagation():
    """The asymptotic velocities of a propagated hyperbola equal v∞_in and the rotated v∞_out."""
    mu, rp = 1.26686534e17, 5e8
    v_in = np.array([3000.0, -9000.0, 1500.0])
    for b_angle in (0.0, 1.0, math.pi):
        v_out = fb.outgoing_v_inf(v_in, rp, mu, b_angle)
        assert np.linalg.norm(v_out) == pytest.approx(np.linalg.norm(v_in))
        vi = np.linalg.norm(v_in)
        # Periapsis state: velocity along the bisector direction of v_in and v_out, position toward the planet
        h_hat = fb.flyby_axis(v_in, b_angle)
        vp_dir = (v_in + v_out) / np.linalg.norm(v_in + v_out)
        r_dir = np.cross(vp_dir, h_hat)
        r_p, v_p = rp * r_dir, math.sqrt(vi**2 + 2 * mu / rp) * vp_dir
        assert np.cross(r_p, v_p) @ h_hat > 0
        T = 1e14 / vi   # far enough that 2μ/r and the miss distance are negligible
        (_, _), (va, vb) = kepler_states(r_p, v_p, [-T, T], mu)
        assert va == pytest.approx(v_in, abs=1.0)
        assert vb == pytest.approx(v_out, abs=1.0)


def test_powered_flyby_periapsis_solves_turn():
    mu = 3.2486e14
    vi, vo, delta = 4000.0, 6000.0, math.radians(40)
    rp = fb.periapsis_for_turn(vi, vo, delta, mu)
    total = math.asin(1 / (1 + rp * vi**2 / mu)) + math.asin(1 / (1 + rp * vo**2 / mu))
    assert total == pytest.approx(delta, abs=1e-12)
    assert fb.powered_dv(vi, vo, rp, mu) == pytest.approx(
        math.sqrt(vo**2 + 2 * mu / rp) - math.sqrt(vi**2 + 2 * mu / rp))


# ---------------------------------------------------------------- relative motion

def _chief():
    r = np.array([6778.137e3, 0, 0])
    v = np.array([0, math.sqrt(MU_VALLADO / 6778.137e3) * math.cos(0.9), math.sqrt(MU_VALLADO / 6778.137e3)
                  * math.sin(0.9)])
    return r, v


def test_lvlh_round_trip_and_axes():
    rc, vc = _chief()
    rho, rho_dot = np.array([100.0, -2000.0, 50.0]), np.array([0.1, -0.2, 0.05])
    r, v = rel.from_lvlh(rc, vc, rho, rho_dot)
    back = rel.to_lvlh(rc, vc, r, v)
    assert back[0] == pytest.approx(rho, abs=1e-6)
    assert back[1] == pytest.approx(rho_dot, abs=1e-9)
    Q = rel.lvlh_basis(rc, vc)
    assert Q @ Q.T == pytest.approx(np.eye(3), abs=1e-14)
    assert Q[1] @ vc > 0   # along-track points along the velocity for a circular orbit


def test_cw_matches_nonlinear_motion():
    """For a 1 km offset the CW solution follows the two-body motion to second order in ρ/r over one orbit."""
    rc, vc = _chief()
    n = math.sqrt(MU_VALLADO / np.linalg.norm(rc) ** 3)
    rho0, rho_dot0 = np.array([200.0, -1000.0, 100.0]), np.array([0.05, 0.1, -0.02])
    r0, v0 = rel.from_lvlh(rc, vc, rho0, rho_dot0)
    ts = np.linspace(0, 2 * math.pi / n, 25)
    rcs, vcs = kepler_states(rc, vc, ts, MU_VALLADO)
    rds, vds = kepler_states(r0, v0, ts, MU_VALLADO)
    rho_cw, _ = rel.cw_propagate(rho0, rho_dot0, n, ts)
    for k in range(len(ts)):
        rho_true, _ = rel.to_lvlh(rcs[k], vcs[k], rds[k], vds[k])
        # The neglected terms are second order: |ρ|²/r (5 m at the 6 km reached here)
        assert np.linalg.norm(rho_true - rho_cw[k]) < max(np.linalg.norm(rho_true), 1e3) ** 2 / np.linalg.norm(rc)


def test_cw_composition_and_rendezvous():
    n = 0.0011
    rr, rv, vr, vv = rel.cw_matrices(n, 0.0)
    assert np.block([[rr, rv], [vr, vv]]) == pytest.approx(np.eye(6))
    phi = lambda t: np.block([[m for m in rel.cw_matrices(n, t)[:2]], [m for m in rel.cw_matrices(n, t)[2:]]])
    assert phi(700.0) @ phi(500.0) == pytest.approx(phi(1200.0), abs=1e-9)
    rho0, v0 = np.array([0.0, -1000.0, 0.0]), np.zeros(3)
    dv1, dv2 = rel.cw_rendezvous(rho0, v0, 2000.0, n)
    rho, rho_dot = rel.cw_propagate(rho0, v0 + dv1, n, 2000.0)
    assert rho == pytest.approx(0, abs=1e-6)
    assert rho_dot + dv2 == pytest.approx(0, abs=1e-9)


# ---------------------------------------------------------------- coverage and Sun geometry

@pytest.mark.parametrize("h,eps", [(400e3, 0.0), (400e3, math.radians(10)), (35786e3, math.radians(5)),
                                   (700e3, math.radians(30))])
def test_footprint_geometry(h, eps):
    """A ground point at the footprint edge sees the satellite at exactly the minimum elevation."""
    R = 6378137.0
    lam, eta, D = cov.footprint(h, R, eps)
    sat = np.array([0.0, 0.0, R + h])
    ground = R * np.array([math.sin(lam), 0.0, math.cos(lam)])
    los = sat - ground
    elev = math.asin(los @ ground / (np.linalg.norm(los) * R))
    assert elev == pytest.approx(eps, abs=1e-12)
    assert np.linalg.norm(los) == pytest.approx(D, rel=1e-12)
    assert math.asin(abs(np.cross(sat, -los)[1]) / (np.linalg.norm(los) * (R + h))) == pytest.approx(eta)


def test_footprint_limits():
    R, h = 6378137.0, 500e3
    lam0, _, _ = cov.footprint(h, R, 0.0)
    assert lam0 == pytest.approx(math.acos(R / (R + h)))
    # A sensor wider than the Earth's disk sees to the horizon
    lam, eps, _ = cov.sensor_footprint(h, R, math.radians(80))
    assert lam == pytest.approx(lam0) and eps == pytest.approx(0.0, abs=1e-7)
    lam, eps, _ = cov.sensor_footprint(h, R, math.radians(20))
    assert cov.footprint(h, R, eps)[0] == pytest.approx(lam)


def test_beta_angle_and_eclipse_fraction():
    r, v = np.array([7e6, 0, 0]), np.array([0, 7.5e3, 0])
    assert math.degrees(sun.beta_angle(r, v, [0, 0, 1.5e11])) == pytest.approx(90)
    assert sun.beta_angle(r, v, [1.5e11, 0, 0]) == pytest.approx(0)
    R = 6378137.0
    # β = 0: shadow half-angle asin(R/r) on each side of midnight
    assert sun.eclipse_fraction(0.0, 7e6, R) == pytest.approx(math.asin(R / 7e6) / math.pi)
    assert sun.eclipse_fraction(math.radians(80), 7e6, R) == 0.0


def test_rocket_equation():
    m0, isp = 1000.0, 300.0
    mp = prop.propellant_mass(3000.0, isp, m0)
    assert prop.delta_v(isp, m0, m0 - mp) == pytest.approx(3000.0)
    assert prop.mass_ratio(3000.0, isp) == pytest.approx(math.exp(3000 / (300 * 9.80665)))
    assert prop.propellant_mass(0.0, isp, m0) == 0.0
    with pytest.raises(ValueError):
        prop.propellant_mass(100.0, 0.0, m0)

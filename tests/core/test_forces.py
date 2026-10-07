import numpy as np
import pytest

from urania.core import forces

MU = 3.986004418e14
R = 6378137.0
J2 = 1.08262668e-3


def test_twobody_accel():
    r = np.array([7000e3, 0.0, 0.0])
    a = forces.accel_twobody(r, MU)
    np.testing.assert_allclose(a, [-MU / 7000e3**2, 0, 0])


def _j2_potential(r):
    """U_J2 = -(μ J2 R² / 2r³)(3z²/r² - 1), a = ∇U."""
    rn = np.linalg.norm(r)
    return -MU * J2 * R**2 / (2 * rn**3) * (3 * r[2] ** 2 / rn**2 - 1)


@pytest.mark.parametrize("r", [[7000e3, 0, 0], [4000e3, 3000e3, 5000e3], [0, 0, 7000e3],
                               [-5000e3, 2000e3, -4000e3]])
def test_j2_accel_is_potential_gradient(r):
    """The J2 acceleration must match the numerical gradient of the J2 potential."""
    r = np.array(r, dtype=float)
    h = 1.0
    grad = np.array([(_j2_potential(r + h * e) - _j2_potential(r - h * e)) / (2 * h)
                     for e in np.eye(3)])
    np.testing.assert_allclose(forces.accel_j2(r, MU, R, J2), grad, rtol=1e-6, atol=1e-15)


def test_drag_opposes_relative_velocity():
    r = np.array([6778e3, 0, 0])
    v = np.array([0, 7669.0, 0])
    omega = 7.292115e-5
    a = forces.accel_drag(r, v, rho=3.7e-12, ballistic=0.01, rotation_rate=omega)
    v_rel = v - np.cross([0, 0, omega], r)
    assert np.allclose(np.cross(a, v_rel), 0.0)
    assert a @ v_rel < 0
    assert np.linalg.norm(a) == pytest.approx(0.5 * 3.7e-12 * 0.01 * np.linalg.norm(v_rel) ** 2)


@pytest.mark.parametrize("h_km, rho", [(0, 1.225), (100, 5.297e-7), (400, 3.725e-12),
                                       (1000, 3.019e-15)])
def test_exponential_density_table(h_km, rho):
    """At the base altitudes of Vallado table 8-4 the density equals the tabulated ρ₀."""
    assert forces.exponential_density(h_km * 1e3) == pytest.approx(rho)


def test_exponential_density_within_band():
    # 425 km: 400 km band, H = 58.515 km
    expected = 3.725e-12 * np.exp(-25 / 58.515)
    assert forces.exponential_density(425e3) == pytest.approx(expected)


def test_exponential_density_monotonic_and_edges():
    hs = np.linspace(0, 1500e3, 3001)
    rho = [forces.exponential_density(h) for h in hs]
    assert all(a > b for a, b in zip(rho, rho[1:]))
    assert forces.exponential_density(-100.0) == 1.225


R_WGS84, F_WGS84 = 6378137.0, 1 / 298.257223563


def _point_on_ellipsoid(lat: float, h: float) -> np.ndarray:
    """Inverse of geodetic_altitude: geodetic latitude and height → position (longitude 0)."""
    e2 = F_WGS84 * (2 - F_WGS84)
    N = R_WGS84 / np.sqrt(1 - e2 * np.sin(lat) ** 2)
    return np.array([(N + h) * np.cos(lat), 0.0, (N * (1 - e2) + h) * np.sin(lat)])


@pytest.mark.parametrize("lat_deg", [0.0, 30.0, 51.6, 89.9, 90.0, -45.0])
@pytest.mark.parametrize("h", [0.0, 400e3, 35786e3])
def test_geodetic_altitude_round_trip(lat_deg, h):
    r = _point_on_ellipsoid(np.radians(lat_deg), h)
    assert forces.geodetic_altitude(r, R_WGS84, F_WGS84) == pytest.approx(h, abs=1e-6)


def test_geodetic_altitude_known_values():
    # Polar radius b = a(1 - f) = 6356752.314 km: a point at r = b on the pole has zero height
    assert forces.geodetic_altitude(np.array([0.0, 0.0, 6356752.314245]), R_WGS84, F_WGS84) \
        == pytest.approx(0.0, abs=1e-3)
    # Same |r| is about 21 km higher above the ellipsoid at the pole than at the equator
    r = R_WGS84 + 400e3
    eq = forces.geodetic_altitude(np.array([r, 0.0, 0.0]), R_WGS84, F_WGS84)
    pole = forces.geodetic_altitude(np.array([0.0, 0.0, r]), R_WGS84, F_WGS84)
    assert pole - eq == pytest.approx(R_WGS84 * F_WGS84, rel=1e-6)


def test_geodetic_altitude_sphere():
    r = np.array([4000e3, 3000e3, 5000e3])
    assert forces.geodetic_altitude(r, R_WGS84, 0.0) == pytest.approx(np.linalg.norm(r) - R_WGS84)

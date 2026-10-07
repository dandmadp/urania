import math

import astropy.units as u
import numpy as np
import pytest

from urania import GEO, ISS, LEO, Earth, Mars, Orbit
from urania.core import maneuvers as core_m
from urania.core.propagate import kepler_propagate


def _check_on_target(transfer):
    """After the transfer the orbit must match the target in size, shape and plane."""
    f, t = transfer.final, transfer.target
    assert f.a == pytest.approx(t.a, rel=1e-9)
    assert f.ecc < 1e-9
    h_f = np.cross(f.r, f.v) / np.linalg.norm(np.cross(f.r, f.v))
    h_t = np.cross(t.r, t.v) / np.linalg.norm(np.cross(t.r, t.v))
    np.testing.assert_allclose(h_f, h_t, atol=1e-9)


def test_leo_to_geo_matches_core():
    tr = LEO.transfer_to(GEO)
    dv1, dv2, tof = core_m.hohmann(LEO.a, GEO.a, Earth.mu)
    assert tr.kind == "hohmann"
    assert len(tr.burns) == 2
    assert tr.burns[0].magnitude == pytest.approx(dv1)
    assert tr.burns[1].magnitude == pytest.approx(dv2)
    assert tr.total_dv == pytest.approx(dv1 + dv2)
    assert tr.tof == pytest.approx(tof)
    assert tr.coast == 0.0
    _check_on_target(tr)


def test_transfer_orbit_apsides():
    tr = LEO.transfer_to(GEO)
    t_orbit = tr.orbits[0]
    assert t_orbit.r_periapsis == pytest.approx(LEO.a, rel=1e-12)
    assert t_orbit.r_apoapsis == pytest.approx(GEO.a, rel=1e-12)
    # Propagating tof after burn 1 must reach the burn 2 position
    r, _ = kepler_propagate(t_orbit.r, t_orbit.v, tr.tof, Earth.mu)
    np.testing.assert_allclose(r, tr.burns[1].r, atol=1e-3)


def test_burns_are_tangential_when_coplanar():
    tr = LEO.transfer_to(GEO)
    for b, o in zip(tr.burns, [LEO, tr.orbits[0]]):
        assert np.linalg.norm(np.cross(b.dv, b.r)) / (b.magnitude * np.linalg.norm(b.r)) \
            == pytest.approx(1.0)  # Δv ⟂ r (tangential on a circular orbit)


def test_descending_transfer():
    up = LEO.transfer_to(GEO)
    down = GEO.transfer_to(LEO)
    assert down.total_dv == pytest.approx(up.total_dv)
    _check_on_target(down)


def test_iss_to_geo_with_plane_change():
    tr = ISS.transfer_to(GEO)
    assert math.degrees(tr.plane_change) == pytest.approx(51.64)
    split = core_m.optimal_plane_split(ISS.a, GEO.a, tr.plane_change, Earth.mu)
    dv1, dv2, _ = core_m.hohmann_plane_change(ISS.a, GEO.a, tr.plane_change, Earth.mu, split)
    assert tr.burns[0].magnitude == pytest.approx(dv1, rel=1e-9)
    assert tr.burns[1].magnitude == pytest.approx(dv2, rel=1e-9)
    _check_on_target(tr)
    # The optimal split is cheaper than doing it all at apoapsis
    assert tr.total_dv < ISS.transfer_to(GEO, plane_split=0.0).total_dv


def test_plane_change_waits_for_node():
    """Starting away from the node, the first burn waits until the node."""
    start = Orbit.circular(Earth, 500e3, inc=30 * u.deg, raan=20 * u.deg, arglat=100 * u.deg)
    target = Orbit.circular(Earth, 2000e3, inc=50 * u.deg, raan=80 * u.deg)
    tr = start.transfer_to(target)
    assert 0.0 < tr.coast < start.period / 2
    # Angle between the planes: cos Δθ = cos i1 cos i2 + sin i1 sin i2 cos ΔΩ (spherical trigonometry)
    i1, i2, dO = math.radians(30), math.radians(50), math.radians(60)
    expected = math.acos(math.cos(i1) * math.cos(i2) + math.sin(i1) * math.sin(i2) * math.cos(dO))
    assert tr.plane_change == pytest.approx(expected)
    # The first burn lies in both planes (the node)
    for o in (start, target):
        h = np.cross(o.r, o.v)
        assert abs(h @ tr.burns[0].r) / (np.linalg.norm(h) * np.linalg.norm(tr.burns[0].r)) < 1e-9
    _check_on_target(tr)


def test_pure_plane_change():
    start = Orbit.circular(Earth, 700e3, inc=98 * u.deg)
    target = Orbit.circular(Earth, 700e3, inc=90 * u.deg)
    tr = start.transfer_to(target)
    assert tr.kind == "plane_change"
    assert len(tr.burns) == 1
    v = math.sqrt(Earth.mu / start.a)
    assert tr.total_dv == pytest.approx(core_m.plane_change_dv(v, math.radians(8)), rel=1e-9)
    _check_on_target(tr)


def test_same_orbit_no_burns():
    tr = LEO.transfer_to(LEO)
    assert tr.kind == "none"
    assert tr.total_dv == 0.0
    assert tr.final is LEO


def test_bielliptic_matches_core():
    far = Orbit.circular(Earth, 20 * LEO.a - Earth.radius)
    rb = 40 * LEO.a
    tr = LEO.transfer_to(far, "bielliptic", rb=rb)
    b = core_m.bielliptic(LEO.a, rb, far.a, Earth.mu)
    assert tr.kind == "bielliptic"
    assert len(tr.burns) == 3
    assert tr.total_dv == pytest.approx(sum(abs(x) for x in b[:3]), rel=1e-9)
    assert tr.tof == pytest.approx(b[3])
    assert tr.total_dv < LEO.transfer_to(far).total_dv  # r2/r1 = 20 > 15.58
    _check_on_target(tr)


def test_bielliptic_with_plane_change_at_rb():
    target = Orbit.circular(Earth, 20 * LEO.a - Earth.radius, inc=30 * u.deg)
    tr = LEO.transfer_to(target, "bielliptic", rb=40 * LEO.a * u.m)
    assert "plane" in tr.burns[1].description
    _check_on_target(tr)


@pytest.mark.parametrize("kwargs, err", [
    ({"method": "lambert"}, ValueError),
    ({"method": "bielliptic"}, ValueError),                 # rb missing
    ({"method": "bielliptic", "rb": 10000e3}, ValueError),  # rb < r2
    ({"plane_split": 1.5}, ValueError),
])
def test_invalid_arguments(kwargs, err):
    with pytest.raises(err):
        ISS.transfer_to(GEO, **kwargs)


def test_rejects_elliptic_and_other_body():
    ell = Orbit.from_elements(Earth, a=10000e3, ecc=0.1)
    with pytest.raises(ValueError, match="not circular"):
        ell.transfer_to(GEO)
    with pytest.raises(ValueError, match="Central bodies"):
        Orbit.circular(Mars, 300e3).transfer_to(GEO)


def test_opposite_planes():
    """Prograde → retrograde equatorial (Δθ = 180°): no NaN; the plane flip happens at apoapsis.

    Hand calculation: the transfer apoapsis speed v_a reverses into -v_geo, so Δv2 = v_a + v_geo.
    """
    retro_geo = Orbit.circular(Earth, GEO.a - Earth.radius, inc=math.pi)
    tr = LEO.transfer_to(retro_geo)
    assert math.degrees(tr.plane_change) == pytest.approx(180.0)
    assert np.isfinite(tr.total_dv)
    dv1, dv2, _ = core_m.hohmann(LEO.a, GEO.a, Earth.mu)
    v_geo = math.sqrt(Earth.mu / GEO.a)
    v_a = v_geo - dv2
    assert tr.total_dv == pytest.approx(dv1 + v_a + v_geo, rel=1e-9)
    _check_on_target(tr)


def test_bad_method_rejected_even_for_plane_change():
    """An unknown method used to be ignored when the radii were equal (regression test)."""
    start = Orbit.circular(Earth, 500e3, inc=0.5)
    with pytest.raises(ValueError, match="Unknown transfer method"):
        start.transfer_to(LEO, method="bogus")


def test_repr_singular_burn():
    pc = Orbit.circular(Earth, 700e3, inc=1.7).transfer_to(Orbit.circular(Earth, 700e3, inc=1.5))
    assert repr(pc).endswith("1 burn)")
    assert repr(LEO.transfer_to(GEO)).endswith("2 burns)")

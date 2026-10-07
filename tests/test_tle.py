import math

import numpy as np
import pytest
from astropy.time import Time

from urania import Epoch, Orbit, TLEOrbit
from urania.tle import checksum

ISS_TEXT = """ISS (ZARYA)
1 25544U 98067A   19343.69339541  .00001764  00000-0  38792-4 0  9991
2 25544  51.6439 211.2001 0007417  17.6667  85.6398 15.50103472202482"""

# Vallado SGP4 검증 세트 (SGP4-VER.TLE, 위성 00005)
V5_L1 = "1 00005U 58002B   00179.78495062  .00000023  00000-0  28098-4 0  4753"
V5_L2 = "2 00005  34.2682 348.7242 1859667 331.7664  19.3264 10.82419157413667"


@pytest.fixture
def iss():
    return TLEOrbit.from_text(ISS_TEXT)


def test_vallado_sgp4_verification_00005():
    """Vallado et al. 2006 "Revisiting Spacetrack Report #3", tcppver.out 위성 00005."""
    tle = TLEOrbit(V5_L1, V5_L2)
    tr = tle.propagate(360 * 60, n_points=2)
    np.testing.assert_allclose(tr.r[0] / 1e3, [7022.46529266, -1400.08296755, 0.03995155], atol=1e-6)
    np.testing.assert_allclose(tr.v[0] / 1e3, [1.893841015, 6.405893759, 4.534807250], atol=1e-8)
    np.testing.assert_allclose(tr.r[1] / 1e3, [-7154.03120202, -3783.17682504, -3536.19412294],
                               atol=1e-6)
    np.testing.assert_allclose(tr.v[1] / 1e3, [4.741887409, -4.151817765, -2.093935425], atol=1e-8)


def test_parse_and_fields(iss):
    assert iss.name == "ISS (ZARYA)"
    assert iss.satnum == "25544"
    assert math.degrees(iss.inc) == pytest.approx(51.6439)
    assert math.degrees(iss.raan) == pytest.approx(211.2001)
    assert iss.ecc == pytest.approx(0.0007417)
    assert iss.revs_per_day == pytest.approx(15.50103472)
    assert iss.bstar == pytest.approx(3.8792e-5)


def test_epoch_utc_to_tdb(iss):
    """19343.69339541 = 2019-12-09 16:38:29.363 UTC, TDB - UTC ≈ 69.184 s."""
    utc = Time("2019-12-09T16:38:29.363", scale="utc")
    assert iss.epoch - Epoch.from_iso("2019-12-09T16:38:29.363") == pytest.approx(69.184, abs=0.002)
    assert iss.epoch.to_astropy().utc.jd == pytest.approx(utc.jd, abs=1e-8)


def test_state_at_matches_sgp4_direct(iss):
    """Epoch(TDB) → UTC 변환을 거쳐도 sgp4 직접 호출과 같아야 한다."""
    jd, fr = 2458827, 0.362605
    e, r_ref, v_ref = iss._sat.sgp4(jd, fr)
    epoch = Epoch.from_astropy(Time(jd, fr, format="jd", scale="utc"))
    r, v = iss.state_at(epoch)
    np.testing.assert_allclose(r / 1e3, r_ref, atol=1e-5)
    np.testing.assert_allclose(v / 1e3, v_ref, atol=1e-8)


def test_two_line_text_and_name_prefix():
    lines = ISS_TEXT.splitlines()
    assert TLEOrbit.from_text("\n".join(lines[1:])).name == ""
    assert TLEOrbit.from_text("0 ISS\n" + "\n".join(lines[1:])).name == "ISS"
    with pytest.raises(ValueError):
        TLEOrbit.from_text(lines[1])


def test_checksum_validation():
    lines = ISS_TEXT.splitlines()
    assert checksum(lines[1]) == 1 and checksum(lines[2]) == 2
    bad = lines[1][:20] + ("9" if lines[1][20] != "9" else "8") + lines[1][21:]
    with pytest.raises(ValueError, match="체크섬"):
        TLEOrbit(bad, lines[2])
    with pytest.raises(ValueError):
        TLEOrbit(lines[2], lines[1])          # 줄 순서 바뀜
    with pytest.raises(ValueError, match="위성 번호"):
        TLEOrbit(lines[1], V5_L2)


def test_propagate_trajectory(iss):
    tr = iss.propagate(days=1)
    assert tr.info.model == "sgp4"
    assert any("TEME" in a for a in tr.info.assumptions)
    assert tr.epoch == iss.epoch
    assert len(tr) == math.ceil(iss.revs_per_day * 100) + 1
    assert 400e3 < tr.altitude.min() and tr.altitude.max() < 440e3
    assert tr.info.terminated is None


def test_propagate_from_custom_start(iss):
    start = iss.epoch + 3600
    tr = iss.propagate(600, start=start, n_points=3)
    r, _ = iss.state_at(start + 600)
    np.testing.assert_allclose(tr.r[-1], r)


def test_to_orbit(iss):
    o = iss.to_orbit()
    assert isinstance(o, Orbit)
    assert o.epoch == iss.epoch
    r, v = iss.state_at(iss.epoch)
    np.testing.assert_array_equal(o.r, r)
    # 접촉 경사각은 평균 경사각과 비슷하다
    assert math.degrees(o.inc) == pytest.approx(51.64, abs=0.05)


def test_own_propagator_close_to_sgp4_short_term(iss):
    """to_orbit()으로 넘긴 뒤 J2 전파 1시간: SGP4와 수 km 이내 (8단계에서 정밀 비교)."""
    o = iss.to_orbit()
    mine = o.propagate(3600, model="j2").final.r
    sgp4, _ = iss.state_at(iss.epoch + 3600)
    assert np.linalg.norm(mine - sgp4) < 5e3


def test_decayed_satellite_terminates():
    """평균 운동이 크고 B*가 매우 큰 위성은 SGP4가 오류(재진입)를 내고 거기서 멈춘다."""
    def with_checksum(line):
        return line[:68] + str(checksum(line))

    l1 = with_checksum("1 99999U 00000A   20001.00000000  .00000000  00000-0  50000-0 0  9990")
    l2 = with_checksum("2 99999  51.6000 100.0000 0005000  90.0000 270.0000 16.30000000000010")
    tle = TLEOrbit(l1, l2)
    tr = tle.propagate(days=30)
    assert tr.info.terminated and "SGP4 오류" in tr.info.terminated
    assert tr.t[-1] < 30 * 86400


def test_explain_and_plot(iss):
    text = str(iss.explain())
    assert "NORAD 25544" in text and "B*" in text and "16:38:29" in text
    tr_text = str(iss.propagate(days=1).explain())
    assert "SGP4" in tr_text and "a_J2" not in tr_text
    import matplotlib.pyplot as plt

    ax = iss.propagate(days=0.2).plot()
    assert ax.get_lines()
    plt.close("all")


def test_repr(iss):
    assert "25544" in repr(iss) and "rev/day" in repr(iss)

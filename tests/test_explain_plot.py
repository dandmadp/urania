import math

import astropy.units as u
import matplotlib.pyplot as plt
import pytest

from urania import GEO, ISS, LEO, Earth, Orbit
from urania.explain import Explanation

ISS_DRAG = {"area": 1500, "mass": 420000}


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


# ---------------------------------------------------------------- explain

def test_explanation_rendering():
    ex = Explanation("제목")
    ex.step("단계", "a = 1", "b = 2")
    ex.assumptions.append("가정 하나")
    ex.warnings.append("경고 하나")
    text = str(ex)
    assert text.startswith("제목")
    assert "1. 단계" in text and "   a = 1" in text
    assert "- 가정 하나" in text and "! 경고 하나" in text
    assert repr(ex) == text


def test_orbit_explain_values():
    o = Orbit.from_vectors(Earth, [6524.834, 6862.875, 6448.296] * u.km,
                           [4.901327, 5.533756, -1.976341] * u.km / u.s)
    text = str(o.explain())
    assert "h = r × v" in text
    assert f"{o.ecc:.6f}" in text                    # 0.832853
    assert f"{math.degrees(o.inc):.4f}°" in text     # 87.8691°
    assert "주기 T" in text
    assert "WGS 84" in text                          # μ 출처


def test_orbit_explain_special_conventions():
    text = str(GEO.explain())
    assert "원 궤도" in text and "적도 궤도" in text


def test_transfer_explain_matches_burns():
    """공식으로 다시 계산한 Δv가 상태벡터 결과와 같으면 경고가 없다."""
    for tr in (LEO.transfer_to(GEO), ISS.transfer_to(GEO),
               LEO.transfer_to(Orbit.circular(Earth, 20 * LEO.a - Earth.radius, inc=20 * u.deg),
                               "bielliptic", rb=40 * LEO.a),
               Orbit.circular(Earth, 700e3, inc=98 * u.deg)
               .transfer_to(Orbit.circular(Earth, 700e3, inc=90 * u.deg))):
        ex = tr.explain()
        assert ex.warnings == [], ex.warnings
        text = str(ex)
        assert f"{tr.total_dv / 1e3:.4f} km/s" in text
        assert "임펄스 기동" in text


def test_hohmann_explain_shows_formulas():
    text = str(ISS.transfer_to(GEO).explain())
    assert "a_t = (r₁ + r₂)/2" in text
    assert "cos α" in text                 # 궤도면 변경 결합 공식
    assert "tof = π√(a_t³/μ)" in text


def test_transfer_explain_none():
    assert "기동이 필요 없다" in str(LEO.transfer_to(LEO).explain())


def test_trajectory_explain_twobody():
    ex = ISS.propagate(days=1).explain()
    text = str(ex)
    assert "케플러" in text
    assert "a_J2" not in text
    assert ex.warnings == []


def test_trajectory_explain_j2_warns_osculating():
    """J2가 있으면 접촉 장반경 진동 경고와 1바퀴 평균 변화를 보여준다 (DECISIONS 메모)."""
    ex = ISS.propagate(days=3, model="j2+drag", **ISS_DRAG).explain()
    text = str(ex)
    assert "a_J2" in text and "a_항력" in text
    assert "rtol = 1e-12" in text
    assert "1바퀴 평균" in text
    assert any("출렁" in w for w in ex.warnings)
    assert "지수 대기" in text


def test_trajectory_explain_terminated():
    ex = Orbit.circular(Earth, 130e3).propagate(days=5, model="drag", area=1, mass=1).explain()
    assert any("표면" in w for w in ex.warnings)


# ---------------------------------------------------------------- plot

def test_orbit_plot():
    o = Orbit.from_elements(Earth, a=12000e3, ecc=0.4, inc=0.5, nu=2.0)
    ax = o.plot()
    labels = [line.get_label() for line in ax.get_lines()]
    assert "orbit" in labels and "periapsis" in labels and "apoapsis" in labels
    assert ax.get_aspect() == 1.0


def test_hyperbolic_orbit_plot():
    ax = Orbit.from_elements(Earth, a=-20000e3, ecc=1.5).plot()
    assert "apoapsis" not in [line.get_label() for line in ax.get_lines()]


def test_plot_on_given_axes():
    fig, ax = plt.subplots()
    assert LEO.plot(ax) is ax


def test_transfer_plot_burn_markers():
    tr = LEO.transfer_to(GEO, "bielliptic", rb=80000e3)
    ax = tr.plot()
    burns = [line for line in ax.get_lines() if line.get_marker() == "*"]
    assert len(burns) == 3
    assert len(ax.texts) == 3                # Δv 주석
    assert "Bielliptic" in ax.get_title()


def test_transfer_plot_plane_change_title():
    ax = ISS.transfer_to(GEO).plot()
    assert "plane change 51.64°" in ax.get_title()


def test_trajectory_plots():
    tr = ISS.propagate(days=3, model="j2+drag", **ISS_DRAG)
    ax = tr.plot()
    assert "start" in [line.get_label() for line in ax.get_lines()]
    ax = tr.plot(kind="altitude")
    assert "orbit-averaged" in [line.get_label() for line in ax.get_lines()]
    with pytest.raises(ValueError):
        tr.plot(kind="3d")

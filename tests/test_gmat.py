"""NASA GMAT 대조 테스트.

참조 데이터는 validation/gmat/make_scripts.py로 만든 GMAT 스크립트를 GMAT에서 실행해 얻는다
(tests/fixtures/gmat/<시나리오>.txt). 파일이 없으면 건너뛴다.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures" / "gmat"

_spec = importlib.util.spec_from_file_location(
    "gmat_scenarios", ROOT / "validation" / "gmat" / "make_scripts.py")
gmat = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gmat)

# 시나리오별 (urania 모델, 전파 시간 전체에서 허용하는 최대 위치 오차 [m])
# 잠정값: GMAT 참조 데이터를 받은 뒤 실제 오차를 보고 확정한다.
CASES = {
    "twobody": ("twobody", 1.0),
    "j2": ("j2", 100.0),
    "j2_drag": ("j2+drag", 1000.0),
}


def load_gmat(name: str) -> np.ndarray:
    path = FIXTURES / f"{name}.txt"
    if not path.exists():
        pytest.skip(f"GMAT 참조 데이터 없음: {path.relative_to(ROOT)} "
                    f"(validation/gmat/{name}.script를 GMAT에서 실행)")
    data = np.loadtxt(path, skiprows=1)
    return data  # 열: 경과 초, X, Y, Z [km], VX, VY, VZ [km/s]


def position_errors(name: str) -> tuple[np.ndarray, np.ndarray]:
    """(경과 초, urania와 GMAT의 위치 차이 [m])."""
    data = load_gmat(name)
    sc = gmat.SCENARIOS[name]
    model, _ = CASES[name]
    orbit = sc["orbit"]()
    t = data[:, 0]
    kwargs = gmat.DRAG_SAT if sc["drag"] else {}
    # GMAT 출력 시각마다 urania 결과를 뽑는다
    tr = orbit.propagate(float(t[-1]), model=model, method="auto", n_points=len(t), **kwargs)
    np.testing.assert_allclose(tr.t, t, atol=1e-6)
    errs = np.linalg.norm(tr.r - data[:, 1:4] * 1e3, axis=1)
    return t, errs


@pytest.mark.parametrize("name", list(CASES))
def test_against_gmat(name):
    t, errs = position_errors(name)
    _, limit = CASES[name]
    worst = errs.max()
    print(f"{name}: 최대 {worst:.3f} m (t = {t[errs.argmax()] / 3600:.1f} h), 끝 {errs[-1]:.3f} m")
    assert worst < limit


def test_initial_states_match_scripts():
    """생성된 스크립트의 초기 상태가 현재 시나리오 정의와 같아야 한다 (스크립트 재생성 누락 방지)."""
    for name, sc in gmat.SCENARIOS.items():
        script = ROOT / "validation" / "gmat" / f"{name}.script"
        if not script.exists():
            pytest.skip("스크립트 없음: python validation/gmat/make_scripts.py")
        values = {}
        for line in script.read_text(encoding="utf-8").splitlines():
            for key in ("X", "Y", "Z", "VX", "VY", "VZ"):
                if line.startswith(f"GMAT Sat.{key} = "):
                    values[key] = float(line.split("=")[1].strip(" ;"))
        o = sc["orbit"]()
        np.testing.assert_allclose([values[k] for k in ("X", "Y", "Z")], o.r / 1e3, atol=1e-9)
        np.testing.assert_allclose([values[k] for k in ("VX", "VY", "VZ")], o.v / 1e3, atol=1e-12)

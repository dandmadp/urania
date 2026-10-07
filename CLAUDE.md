# CLAUDE.md

urania: 계산하고, 보여주고, 설명하는 파이썬 우주역학 라이브러리. 전체 명세는 [SPEC.md](SPEC.md), 설계 결정은 [docs/DECISIONS.md](docs/DECISIONS.md).

## 명령어

Windows 환경. 가상환경 파이썬을 직접 호출한다.

```
.venv/Scripts/python -m pytest -q                         # 전체 테스트
.venv/Scripts/python -m pytest tests/core/test_kepler.py  # 파일 단위
.venv/Scripts/python -m pip install -e ".[dev]"           # 의존성 변경 후 재설치
```

## 구조 원칙 (SPEC.md 2절 요약)

- `src/` 레이아웃. 패키지는 `src/urania/`, 테스트는 `tests/`에 패키지 구조를 따라 둔다 (`tests/core/...`).
- **계산은 `urania.core`의 순수 함수에만** 구현한다. 객체(`Orbit`, `Body` 등) 메서드는 core 함수를 호출만 한다.
- core 함수의 입력·출력은 **SI 숫자**(m, kg, s, K, W, rad). 단위 변환은 `units.py` 경계에서만 한다.
- 시간은 내부적으로 TDB 기준 경과 초.
- 주요 객체는 불변 (`dataclass(frozen=True)` 또는 `NamedTuple`). 변경 대신 새 객체를 반환한다.
- 결과 객체는 사용한 모델·적분기·허용오차·가정을 메타데이터로 보관한다.

## 코드 규칙

- docstring과 주석은 한국어. 공식·알고리즘 출처(Curtis, Vallado)를 밝힌다.
- 각도 정규화는 `core.kepler._wrap_2pi` / `_wrap_pi`를 쓴다 (`% 2π`를 직접 쓰면 부동소수점 반올림으로 2π가 나올 수 있음).
- 특이 궤도(원·적도) 규약은 DECISIONS.md D2를 따른다.

## 테스트 규칙

- 새 계산 기능은 **교과서 예제 테스트**(예제 번호를 docstring에 명시)와 **성질 테스트**(왕복 변환, 방정식 잔차)를 함께 작성한다.
- 교과서 값과 비교할 때 허용오차는 교재의 표기 정밀도(유효숫자)에 맞춘다. 반올림 차이를 버그로 오인하거나 허용오차를 과하게 풀지 않는다.
- 지구 μ는 교재마다 다르다: Curtis 398600 km³/s², Vallado 398600.4418 km³/s².

## 작업 방식

- SPEC.md 6절의 단계 순서대로 진행하고, 단계마다 테스트를 모두 통과시킨 뒤 넘어간다.
- 설계 결정은 짧은 근거와 함께 `docs/DECISIONS.md`에 D번호로 추가한다.
- 사용자에게 결과를 보고할 때는 변경된 부분만 코드 블록으로 보여준다.
- 단계가 끝나면 커밋하고 `origin`(github.com/dandmadp/urania)에 push한다.
- 사용자와는 한국어로 소통한다.

## 진행 상황

- [x] 1. core: 상태벡터 ↔ 궤도 요소, 케플러 방정식
- [x] 2. 단위·시간 경계 (astropy, `Epoch`)
- [x] 3. `Body`, `Orbit` 객체와 프리셋
- [x] 4. 전파기: 2체 → J2 → 항력 (`Orbit.propagate` → `Trajectory`)
- [x] 5. 기동 계산 (`Orbit.transfer_to` → `Transfer`)
- [x] 6. `.plot()`, `.explain()` (`viz.py`, `explain.py`)
- [x] 7. TLE 래퍼 (`TLEOrbit`, sgp4)
- [~] 8. TLE 대조·README·예제 완료. GMAT 대조는 틀만 있음: 사용자가 GMAT으로 `validation/gmat/*.script` 실행 → 픽스처 커밋 → 허용오차 확정 → README 수치 기입

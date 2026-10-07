# GMAT 대조 데이터 만들기

urania 전파기를 NASA GMAT과 비교하기 위한 참조 데이터를 만드는 방법입니다.

1. GMAT 설치: https://sourceforge.net/projects/gmat/ (R2022a 이상 권장)
2. 스크립트 생성 (시나리오를 바꿨을 때만):
   ```
   python validation/gmat/make_scripts.py
   ```
3. GMAT을 열고 `twobody.script`, `j2.script`, `j2_drag.script`를 차례로 열어 실행(F5)합니다.
   결과는 `tests/fixtures/gmat/<시나리오>.txt`에 저장됩니다.
   명령줄 실행: `GMAT.exe --run --exit validation/gmat/j2.script`
4. 비교:
   ```
   pytest tests/test_gmat.py -s
   ```

## 시나리오

| 이름 | 궤도 | 힘 모델 | 기간 |
|---|---|---|---|
| twobody | a 8000 km, e 0.1, i 30° | 점질량 | 1일 |
| j2 | 원 궤도 420 km, i 51.64° | EGM96 2차 0차 (J2) | 7일 |
| j2_drag | 원 궤도 400 km, i 51.64° | J2 + 지수 대기 (C_D 2.2, A 1 m², m 100 kg) | 3일 |

- 기준 시각 2000-01-01 12:00:00 TT. 세차가 0이라 GMAT의 실제 자전축과 urania의 고정 z축이 거의 같습니다.
- 중력 상수는 GMAT EGM96 값(μ 398600.4415 km³/s², R 6378.1363 km)에 맞춘 테스트용 천체를 씁니다.
- 출력은 1시간 간격, EarthMJ2000Eq 좌표계, km·km/s.
- 스크립트의 출력 경로는 생성한 컴퓨터의 절대 경로입니다. 다른 컴퓨터에서는 2번을 다시 실행하세요.

## 알려진 차이 요인

- GMAT의 지수 대기 모델(`Exponential`)과 urania의 Vallado 표가 같은지 결과로 확인해야 합니다.
  GMAT 버전에 따라 `AtmosphereModel = Exponential`을 지원하지 않으면 j2_drag 스크립트가 실패합니다.
- GMAT은 J2 계산과 대기 자전에 실제 지구 방향(장동, EOP)을 씁니다. urania는 고정 z축과 일정한 자전 각속도를 씁니다.

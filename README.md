# urania

**계산하고, 보여주고, 설명하는 우주역학 라이브러리**

큐브샛 동아리, 학생, 우주 시뮬레이션 게임 플레이어처럼 전문가는 아니지만 진지하게 궤도를 계산하고 싶은 사람을 위한 파이썬 라이브러리입니다. 모든 결과는 어떤 모델과 가정으로 계산했는지 함께 기록하고, `.explain()`으로 공식과 중간값을 단계별로 보여줍니다.

> 상태: MVP 개발 중 (v0.0.1). 설계는 [SPEC.md](SPEC.md), 설계 결정은 [docs/DECISIONS.md](docs/DECISIONS.md).

## 설치

PyPI 배포 전이라 소스에서 설치합니다. Python 3.10 이상.

```bash
git clone https://github.com/dandmadp/urania.git
cd urania
python -m venv .venv
.venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

## 빠른 시작

ISS 궤도에서 정지궤도(GEO)로 가는 데 필요한 Δv:

```python
import urania as ur

transfer = ur.ISS.transfer_to(ur.GEO)
print(transfer)                    # Transfer(hohmann, Δv=4.7797 km/s, tof=5.295 h, 기동 2회)
print(transfer.explain())          # 공식, 중간 속도, 기동별 Δv, 가정
transfer.plot()                    # 2D 그림 (matplotlib)
```

`explain()` 출력 일부:

```text
3. 1차 기동 (r₁)
   v₁ = √(μ/r₁) = 7.6573 km/s  (출발 원 궤도)
   v₂ = √(μ(2/r₁ − 1/a_t)) = 10.0492 km/s  (전이 타원)
   궤도면 변경 α = 2.8913° 동시 수행
   Δv = √(v₁² + v₂² − 2 v₁ v₂ cos α) = 2.4325 km/s
```

## 궤도 만들기

단위가 붙은 값(astropy)을 넣거나, 맨 숫자를 넣으면 SI 단위(m, m/s, rad)로 간주합니다.

```python
import astropy.units as u
import urania as ur

leo = ur.Orbit.circular(ur.Earth, 500 * u.km, inc=97.4 * u.deg)
ell = ur.Orbit.from_elements(ur.Earth, a=8000 * u.km, ecc=0.1, inc=30 * u.deg,
                             raan=40 * u.deg, argp=60 * u.deg, nu=0 * u.deg)
sv = ur.Orbit.from_vectors(ur.Earth, [6524.834, 6862.875, 6448.296] * u.km,
                           [4.901327, 5.533756, -1.976341] * u.km / u.s)

print(sv.a, sv.ecc, sv.period)     # 결과는 SI 숫자: m, -, s
print(sv.explain())                # 상태벡터에서 궤도 요소를 구하는 과정
```

프리셋: 천체 `Sun`, `Earth`, `Moon`, `Mars` / 궤도 `LEO`(500 km), `ISS`(420 km, 51.64°), `SSO`(700 km 태양동기), `GEO`.

## 궤도 전파

충실도를 고를 수 있습니다: `"twobody"`(0) → `"j2"`(1) → `"j2+drag"`(2). 원래 객체는 바뀌지 않고 결과 `Trajectory`가 새로 만들어집니다.

```python
import astropy.units as u
import urania as ur

cubesat = dict(cd=2.2, area=0.03 * u.m**2, mass=4.0 * u.kg)
start = ur.Orbit.circular(ur.Earth, 400 * u.km, inc=51.6 * u.deg)

tr = start.propagate(days=3, model="j2+drag", **cubesat)
print(tr.final)                    # 3일 뒤 궤도
print(tr.info.assumptions)         # 사용한 모델·가정
tr.plot(kind="altitude")           # 고도 + 1바퀴 평균선
```

환경 데이터는 직접 넣을 수 있습니다. 상수, 함수 `f(r, t)`, `Environment` 객체를 모두 받습니다.

```python
import astropy.units as u
import urania as ur

start = ur.Orbit.circular(ur.Earth, 400 * u.km)
kw = dict(days=1, model="drag", area=0.03, mass=4.0)

start.propagate(density=3e-12, **kw)                                   # 상수 [kg/m³]
start.propagate(density=lambda r, t: 2 * ur.Earth.atmosphere(r, t), **kw)  # 함수
```

## 기동

```python
import astropy.units as u
import urania as ur

ur.LEO.transfer_to(ur.GEO)                                   # 호만 전이
far = ur.Orbit.circular(ur.Earth, 150000 * u.km)
ur.LEO.transfer_to(far, "bielliptic", rb=300000 * u.km)     # 이중타원 전이
ur.ISS.transfer_to(ur.GEO, plane_split=0)                    # 궤도면 변경을 원점에서 모두 수행
```

궤도면이 다르면 두 궤도면이 만나는 노드에서 기동하고, 호만 전이의 궤도면 변경은 기본으로 두 기동에 최적 분배합니다.

## TLE

```python
import urania as ur

iss = ur.TLEOrbit.from_text("""ISS (ZARYA)
1 25544U 98067A   19343.69339541  .00001764  00000-0  38792-4 0  9991
2 25544  51.6439 211.2001 0007417  17.6667  85.6398 15.50103472202482""")

sgp4 = iss.propagate(days=1)                   # SGP4로만 전파
mine = iss.to_orbit().propagate(days=1, model="j2")   # 상태벡터를 꺼내 urania 전파기로
```

TLE 요소는 SGP4 전용 평균 요소라 `Orbit`과 섞지 않습니다. 좌표계는 SGP4의 TEME를 그대로 씁니다.

## 정확도와 검증

| 대상 | 기준 | 결과 |
|---|---|---|
| 궤도 요소 변환 | Curtis 예제 4.3, 4.7 / Vallado 예제 2-5 | 교재 표기 정밀도 내 일치 |
| 케플러 방정식·전파 | Vallado 예제 2-1, 2-3, 2-4 | 1e-9 rad / 1 m 이내 |
| 호만·이중타원 전이 | Vallado 예제 6-1, 6-2 | Δv 1 mm/s 이내 |
| J2 장기 변화 | 승교점 변화율 해석식 | 10일 전파에서 1% 이내 |
| 항력 감쇠 | 원 궤도 감쇠율 해석식 da/dt = −ρB√(μa) | 1% 이내 |
| SGP4 래퍼 | Vallado SGP4 검증 세트 (위성 00005) | 1e-6 km 이내 |
| 자체 전파기 vs SGP4 | 실제 ISS TLE (2019-12-09), J2 모델 | 24시간 동안 최대 2.34 km |
| NASA GMAT | 2체 / J2 / J2+항력 시나리오 | **참조 데이터 생성 대기 중** ([validation/gmat](validation/gmat)) |

참고로 SGP4 자체의 예측 정확도가 epoch 근처에서 약 1 km라, 자체 전파기와 SGP4의 비교는 정밀 검증이 아니라 상식선의 확인입니다. 정밀 검증은 GMAT 대조로 합니다.

## 한계 (MVP)

- 섭동: J2와 대기 항력까지. 달·태양 섭동, 태양 복사압, 고차 중력장은 다음 버전.
- 대기: Vallado 지수 모델(평균 태양활동). 실제 밀도는 태양활동에 따라 몇 배씩 달라집니다.
- 기동: 원 궤도 사이의 임펄스 전이. 랑데부(위상 맞추기), 행성 간 전이는 다음 버전.
- 시간: 내부는 TDB. UTC 입력은 `Epoch.from_astropy()`를 거칩니다.
- 좌표: 관성계 축 고정(세차·장동 무시). TLE는 TEME를 그대로 씁니다.
- 시각화: 2D만.

## 예제

- [examples/quickstart.py](examples/quickstart.py): ISS → GEO 전이
- [examples/cubesat_decay.py](examples/cubesat_decay.py): 3U 큐브샛 궤도 감쇠와 대기 밀도 민감도
- [examples/tle_vs_propagator.py](examples/tle_vs_propagator.py): 실제 ISS TLE로 SGP4와 비교

## 라이선스

MIT

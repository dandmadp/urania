"""3U 큐브샛의 궤도 감쇠: 고도 350 km에서 대기 항력으로 얼마나 내려가나.

    python examples/cubesat_decay.py

지수 대기 모델(Vallado 표 8-4)은 평균 태양활동 기준이다. 실제 수명은 태양활동에 따라
몇 배씩 달라지므로, 밀도를 바꿔 넣어 범위를 보는 것이 좋다 (아래 density 예시).
"""

from pathlib import Path

import astropy.units as u
import matplotlib.pyplot as plt

import urania as ur

cubesat = dict(cd=2.2, area=0.03 * u.m**2, mass=4.0 * u.kg)   # 3U, 긴 면이 진행 방향과 수직
start = ur.Orbit.circular(ur.Earth, 350 * u.km, inc=51.6 * u.deg)

tr = start.propagate(days=60, model="j2+drag", **cubesat)
print(tr.explain())

# 환경 주입: 대기를 표준의 2배로 가정한 경우 (태양활동 극대기 근사)
dense = start.propagate(days=60, model="j2+drag", **cubesat,
                        density=lambda r, t: 2.0 * ur.Earth.atmosphere(r, t))

fig, ax = plt.subplots(figsize=(8, 4))
tr.plot(ax, kind="altitude")
ax.plot(dense.t / 86400, dense.altitude / 1e3, color="C1", lw=0.5, alpha=0.6,
        label="2x density")
ax.legend()
out = Path(__file__).with_name("cubesat_decay.png")
fig.savefig(out, dpi=120, bbox_inches="tight")
print(f"\n그림 저장: {out}")

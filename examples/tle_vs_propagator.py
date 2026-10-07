"""실제 ISS TLE로 SGP4 예측과 urania 전파기를 비교한다.

    python examples/tle_vs_propagator.py

TLE epoch의 SGP4 상태벡터를 to_orbit()으로 꺼내 urania로 전파하고, 같은 시각의 SGP4 결과와 비교한다.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import urania as ur

iss = ur.TLEOrbit.from_text("""ISS (ZARYA)
1 25544U 98067A   19343.69339541  .00001764  00000-0  38792-4 0  9991
2 25544  51.6439 211.2001 0007417  17.6667  85.6398 15.50103472202482""")
print(iss.explain())

n = 24 * 60 + 1                     # 1분 간격, 24시간
sgp4 = iss.propagate(days=1, n_points=n)
start = iss.to_orbit()

fig, ax = plt.subplots(figsize=(8, 4))
print("\n24시간 동안 SGP4와의 위치 차이")
for model in ("twobody", "j2"):
    mine = start.propagate(days=1, model=model, n_points=n)
    err = np.linalg.norm(mine.r - sgp4.r, axis=1) / 1e3
    print(f"  {model:8s} 최대 {err.max():8.2f} km, 24시간 끝 {err[-1]:8.2f} km")
    ax.semilogy(mine.t / 3600, np.maximum(err, 1e-3), label=model)

ax.set_xlabel("time since TLE epoch [h]")
ax.set_ylabel("position difference vs SGP4 [km]")
ax.set_title("ISS: urania propagator vs SGP4")
ax.grid(alpha=0.3, which="both")
ax.legend()
out = Path(__file__).with_name("tle_vs_propagator.png")
fig.savefig(out, dpi=120, bbox_inches="tight")
print(f"\n그림 저장: {out}")

"""빠른 시작: ISS 궤도에서 정지궤도로 가는 데 필요한 Δv.

    python examples/quickstart.py
"""

from pathlib import Path

import matplotlib.pyplot as plt

import urania as ur

transfer = ur.ISS.transfer_to(ur.GEO)

print(transfer)
print(f"총 Δv: {transfer.total_dv / 1e3:.3f} km/s")
print()
print(transfer.explain())

transfer.plot()
out = Path(__file__).with_name("quickstart.png")
plt.savefig(out, dpi=120, bbox_inches="tight")
print(f"\n그림 저장: {out}")

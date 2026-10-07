"""Quick start: the Δv needed to go from the ISS orbit to geostationary orbit.

    python examples/quickstart.py
"""

from pathlib import Path

import matplotlib.pyplot as plt

import urania as ur

transfer = ur.ISS.transfer_to(ur.GEO)

print(transfer)
print(f"Total Δv: {transfer.total_dv / 1e3:.3f} km/s")
print()
print(transfer.explain())

transfer.plot()
out = Path(__file__).with_name("quickstart.png")
plt.savefig(out, dpi=120, bbox_inches="tight")
print(f"\nSaved figure: {out}")

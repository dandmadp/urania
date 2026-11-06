"""Regenerate the figures of the documentation site (docs/images/*.png).

    python docs/make_figures.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import urania as ur

OUT = Path(__file__).resolve().parent / "images"


def save(ax, name: str) -> None:
    OUT.mkdir(exist_ok=True)
    ax.figure.savefig(OUT / name, dpi=110, bbox_inches="tight")
    plt.close("all")
    print(f"Wrote {OUT / name}")


def main() -> None:
    save(ur.ISS.transfer_to(ur.GEO).plot(), "transfer.png")

    p = ur.porkchop("Earth", "Mars", ("2026-08-01", "2027-01-01"), ("2027-04-01", "2028-03-01"), step="4 day")
    save(p.plot(), "porkchop.png")

    iss = ur.TLEOrbit.from_text("""ISS (ZARYA)
1 25544U 98067A   19343.69339541  .00001764  00000-0  38792-4 0  9991
2 25544  51.6439 211.2001 0007417  17.6667  85.6398 15.50103472202482""")
    station = ur.GroundStation.create("Daejeon", "36.37 deg", "127.36 deg")
    save(iss.propagate(days=1).ground_track().plot(stations=[station]), "ground_track.png")

    orbit = ur.Orbit.circular("Earth", "400 km", inc="51.6 deg", epoch="2026-11-01")
    fig, ax = plt.subplots(figsize=(8, 4))
    for activity, color in (("low", "C0"), ("moderate", "C1"), ("high", "C3")):
        life = orbit.lifetime(area="0.03 m2", mass="4 kg", density=ur.NRLMSISE00(activity=activity))
        ax.plot(life.t / 86400, life.altitude / 1e3, color=color, label=f"{activity} solar activity: {life.days:.0f} days")
    ax.axhline(100, color="0.5", ls="--", lw=1)
    ax.set_xlabel("time [day]")
    ax.set_ylabel("mean altitude [km]")
    ax.set_title("3U CubeSat (4 kg, 0.03 m²) from 400 km, NRLMSISE-00")
    ax.grid(alpha=0.3)
    ax.legend()
    save(ax, "lifetime.png")


if __name__ == "__main__":
    main()

"""
Figures for the H_eff ML-MCTDH Trotter-error estimate.

Reads results/small_direct_benchmark/mlmctdh_heff/convergence.csv (written by
analyze_heff_heidelberg.py) and writes to figures/small_direct_benchmark/.

Two estimators are compared with the exact yardstick E_eff_exact and the true
Trotter error E_trot:

  E_eff_mlmctdh        = max_t max_i | n_i^{H_eff,ML-MCTDH} - n_i^{exact} |
                         -> contaminated once the signal drops near the
                            ML-MCTDH representation-error floor.
  E_eff_mlmctdh_diff   = max_t max_i | n_i^{H_eff,ML-MCTDH} - n_i^{H,ML-MCTDH} |
                         -> the same-tree difference; the ML-MCTDH error is
                            common mode and cancels.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "results" / "small_direct_benchmark" / "mlmctdh_heff" / "convergence.csv"
OUT = ROOT / "figures" / "small_direct_benchmark"
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA).sort_values("dt_au")
dt = df["dt_au"].to_numpy()
floor = float(df["delta_mlmctdh"].iloc[0])


def slope(x, y):
    m = y > 0
    return float(np.polyfit(np.log(x[m]), np.log(y[m]), 1)[0])


# ---------------------------------------------------------------- scaling ----
fig, ax = plt.subplots(figsize=(8.5, 6))

ax.loglog(dt, df["E_eff_exact"], "-", color="k", lw=2,
          label=rf"$E_{{\rm eff}}$ exact ($H_{{\rm eff}}$ prop.), $p={slope(dt, df['E_eff_exact'].to_numpy()):.2f}$")
ax.loglog(dt, df["E_trot"], "--", color="0.45", lw=1.5,
          label=rf"$E_{{\rm trot}}$ (true Trotter error), $p={slope(dt, df['E_trot'].to_numpy()):.2f}$")
ax.loglog(dt, df["E_eff_mlmctdh_diff"], "s", ms=8, color="C0",
          label=r"$\hat E_{\rm ML\text{-}MCTDH}$  (same-tree difference)")
ax.loglog(dt, df["E_eff_mlmctdh"], "o", ms=7, color="C3",
          label=r"$\hat E_{\rm ML\text{-}MCTDH}$  (vs exact)")
ax.axhline(floor, color="C3", ls=":", lw=1.2)
ax.text(dt.max(), floor * 1.15, r"ML-MCTDH floor $\delta_{\rm ML\text{-}MCTDH}$",
        ha="right", va="bottom", fontsize=8, color="C3")

ax.set_xlabel(r"Trotter step $\Delta t$ (a.u.)")
ax.set_ylabel("maximum molecular-population error")
ax.set_title("ML-MCTDH estimate of the Trotter error vs the exact estimate")
ax.grid(True, which="both", alpha=0.25)
ax.legend(fontsize=9)
fig.tight_layout()
fig.savefig(OUT / "heff_mlmctdh_scaling.png", dpi=300)
plt.close(fig)


# ------------------------------------------------------------------- gap ----
fig, ax = plt.subplots(figsize=(8.5, 6))
gap = np.abs(df["E_eff_mlmctdh_diff"].to_numpy() - df["E_eff_exact"].to_numpy())
ax.loglog(dt, gap, "s-", ms=7, color="C0",
          label=r"$|\hat E_{\rm ML\text{-}MCTDH}^{\rm diff} - E_{\rm eff}^{\rm exact}|$")
ax.loglog(dt, df["E_trot"], "--", color="0.45", lw=1.5, label=r"$E_{\rm trot}$ (the signal)")
ax.axhline(floor, color="C3", ls=":", lw=1.2, label=r"ML-MCTDH floor")
ax.set_xlabel(r"Trotter step $\Delta t$ (a.u.)")
ax.set_ylabel("error")
ax.set_title(r"Common-mode cancellation: the same-tree difference recovers $E_{\rm eff}$")
ax.grid(True, which="both", alpha=0.25)
ax.legend(fontsize=9)
fig.tight_layout()
fig.savefig(OUT / "heff_mlmctdh_gap.png", dpi=300)
plt.close(fig)

print("Saved:")
print(" ", OUT / "heff_mlmctdh_scaling.png")
print(" ", OUT / "heff_mlmctdh_gap.png")
print()
print(df[["dt_au", "E_eff_mlmctdh_diff", "E_eff_exact", "E_eff_mlmctdh", "E_trot"]]
      .to_string(index=False))
print(f"\nmax |E_diff - E_eff_exact| over sweep : {gap.max():.2e}")
print(f"ML-MCTDH floor (delta)               : {floor:.2e}")
